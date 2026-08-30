#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""
RISC-V AIA Advanced Platform-Level Interrupt Controller (APLIC).

APLIC 将传统有线中断 (level/edge-triggered) 转换为 MSI 消息，
经 ``msi-parent`` (IMSIC) 投递到目标 hart。本实现面向 **MSI 投递模式**
(``msimode``), 与 QEMU ``riscv_aplic.c`` 及 Linux ``irq-riscv-aplic-msi.c``
逐寄存器对齐。

寄存器布局 (每寄存器 32-bit, 4 字节对齐, 参考 AIA 规范 §4 与
vendor/qemu-10.2.0/hw/intc/riscv_aplic.c):

    +0x0000            domaincfg
    +0x0004 + (i-1)*4  sourcecfg[i]   — SM 触发类型 (0=inactive, 4=edge_rise,
                                        5=edge_fall, 6=level_high, 7=level_low)
    +0x1bc0            mmsicfgaddr / +0x1bc4 mmsicfgaddrH
    +0x1bc8            smsicfgaddr / +0x1bcc smsicfgaddrH
    +0x1c00            setip          — 置 pending 位图
    +0x1cdc            setipnum       — 单源置 pending
    +0x1d00            clrip          — 清 pending 位图
    +0x1ddc            clripnum
    +0x1e00            setie          — 置 enable 位图
    +0x1edc            setienum
    +0x1f00            clrie          — 清 enable 位图
    +0x1fdc            clrienum
    +0x2000            setipnum_le    — 单源置 pending (小端, 供 EOI retrigger)
    +0x3000            genmsi
    +0x3004 + (i-1)*4  target[i]      — (hart_idx << 18) | (guest_idx << 12) | eiid
    +0x4000            IDC 结构 (MSI 模式不使用)

关键语义 (与 QEMU 对齐):

- **sourcecfg[i]** 仅编码 SM 触发类型 (``SM_LEVEL_HIGH`` 等), **不含** 目标 hart
  或 EIID — 这些信息在 **target[i]** 寄存器中。
- **target[i]** = ``(hart_idx << 18) | (guest_idx << 12) | eiid``: 指定该源
  触发时向 ``hart_idx`` 的 IMSIC 投递 ``eiid`` 外部中断身份。
- **MSI 投递**: 当 pending 且 enabled 且 ``domaincfg.IE`` 时, 清除 pending,
  调用 ``IMSIC.set_ip_number(hart_idx, 'S', eiid)`` 置 S-file eip。本 APLIC 的
  ``msi-parent`` 为 S-mode IMSIC 节点 (见 dtb.py), 故投递到 S-file。
- **电平触发**: 外设拉高电平 → 置 pending → 投递; guest EOI 后写 ``setipnum``
  retrigger (``aplic_msi_irq_eoi``) 以在电平仍高时重新投递。
"""

from __future__ import annotations

import threading

from pyremu.interrupt.imsic import IMSIC
from pyremu.memory.bus import Device
from pyremu.utils.mask import mask32

# 最大中断源数量 (source 0 保留, 有效源 1..N)
_MAX_SOURCES = 64  # pyremu 规模 — 可扩展
_BITFIELD_WORDS = (_MAX_SOURCES + 31) // 32  # 2 words for 64 sources

# ---- 寄存器偏移 (AIA 规范 / QEMU) ----
_DOMAINCFG = 0x0000
_SOURCECFG_BASE = 0x0004
_MMSICFGADDR = 0x1BC0
_MMSICFGADDRH = 0x1BC4
_SMSICFGADDR = 0x1BC8
_SMSICFGADDRH = 0x1BCC
_SETIP_BASE = 0x1C00
_SETIPNUM = 0x1CDC
_CLRIP_BASE = 0x1D00
_CLRIPNUM = 0x1DDC
_SETIE_BASE = 0x1E00
_SETIENUM = 0x1EDC
_CLRIE_BASE = 0x1F00
_CLRIENUM = 0x1FDC
_SETIPNUM_LE = 0x2000
_GENMSI = 0x3000
_TARGET_BASE = 0x3004
_IDC_BASE = 0x4000
_REGION_SIZE = 0x6000  # 总 MMIO 覆盖 (含 IDC 区域)

# ---- domaincfg 字段 ----
_DOMAINCFG_IE = 1 << 8  # interrupt enable
_DOMAINCFG_DM = 1 << 2

# ---- sourcecfg SM 触发类型 (bits[9:0]) ----
_SM_INACTIVE = 0x0
_SM_EDGE_RISE = 0x4
_SM_EDGE_FALL = 0x5
_SM_LEVEL_HIGH = 0x6
_SM_LEVEL_LOW = 0x7
_SM_MASK = 0x7

# ---- target 字段 ----
_TARGET_HART_IDX_SHIFT = 18
_TARGET_HART_IDX_MASK = 0x3FFF
_TARGET_GUEST_IDX_SHIFT = 12
_TARGET_GUEST_IDX_MASK = 0x3F
_TARGET_EIID_MASK = 0x7FF

# 每源状态位
_STATE_INPUT = 1 << 8   # 输入电平 (0=低, 1=高)
_STATE_ENABLED = 1 << 1
_STATE_PENDING = 1 << 0


class APLIC(Device):
    """RISC-V AIA APLIC — 有线 → MSI 桥 (MSI 投递模式)。

    外设调用 ``set_irq(source_num, level)`` 时, 依据 ``sourcecfg`` 的 SM 触发
    类型更新 pending, 并在 pending & enabled & domaincfg.IE 时经 ``target``
    路由投递 MSI 到目标 hart 的 IMSIC S-file ``eip[eiid]``。
    """

    def __init__(
        self,
        imsic: IMSIC,
        base_addr: int = 0x0C00_0000,
        num_sources: int = _MAX_SOURCES,
    ) -> None:
        self._imsic: IMSIC = imsic
        self.base_addr = base_addr
        self.num_sources = num_sources
        self.size = _REGION_SIZE

        # sourcecfg[i]: SM 触发类型 (bits[9:0]).
        self._sourcecfg: list[int] = [0] * (num_sources + 1)
        # target[i]: (hart_idx << 18) | (guest_idx << 12) | eiid.
        self._target: list[int] = [0] * (num_sources + 1)
        # state[i]: input(bit8) | enabled(bit1) | pending(bit0).
        self._state: list[int] = [0] * (num_sources + 1)

        self._domaincfg: int = _DOMAINCFG_IE  # IE=1 (默认使能)

        # 并发安全: 外设线程经 set_irq 写 state, 主线程经 read/write 读/写
        # sourcecfg/target/state — 共享状态访问需互斥. 锁顺序 APLIC -> IMSIC
        # (set_irq -> _deliver_from_source -> imsic.set_ip_number), 不得反向.
        self._lock = threading.Lock()

    # ================================================================
    #  外设编程接口
    # ================================================================

    def set_irq(self, source_num: int, level: bool = True) -> None:
        """外设调用入口: 拉高/拉低指定 wired 中断源输入电平。

        等价 QEMU ``riscv_aplic_request(irq, level)``: 依据 SM 触发类型与电平
        变化决定是否置 pending, 并尝试 MSI 投递。

        - level=True  → 输入电平拉高 (设备断言中断线)
        - level=False → 输入电平拉低 (设备撤除中断线)
        """
        with self._lock:
            if not (0 < source_num <= self.num_sources):
                return

            sm = self._sourcecfg[source_num] & _SM_MASK
            if sm == _SM_INACTIVE:
                return

            old_state = self._state[source_num]
            old_input = (old_state & _STATE_INPUT) != 0

            # 更新输入电平
            if level:
                self._state[source_num] |= _STATE_INPUT
            else:
                self._state[source_num] &= ~_STATE_INPUT

            # 依据触发类型决定是否置 pending
            update = False
            if sm == _SM_LEVEL_HIGH and level and not (old_state & _STATE_PENDING):
                self._state[source_num] |= _STATE_PENDING
                update = True
            elif sm == _SM_LEVEL_LOW and not level and not (old_state & _STATE_PENDING):
                self._state[source_num] |= _STATE_PENDING
                update = True
            elif (
                sm == _SM_EDGE_RISE
                and level and not old_input
                and not (old_state & _STATE_PENDING)
            ):
                self._state[source_num] |= _STATE_PENDING
                update = True
            elif (
                sm == _SM_EDGE_FALL
                and not level and old_input
                and not (old_state & _STATE_PENDING)
            ):
                self._state[source_num] |= _STATE_PENDING
                update = True

            if update:
                self._deliver_from_source(source_num)

    # ================================================================
    #  Device 接口 (MMIO)
    # ================================================================

    def read(self, offset: int, size: int) -> bytes:
        with self._lock:
            val = self._mmio_read(offset)
            return val.to_bytes(size, "little", signed=False)

    def write(self, offset: int, data: bytes) -> None:
        with self._lock:
            val = int.from_bytes(data, "little", signed=False)
            self._mmio_write(offset, mask32(val))

    # ================================================================
    #  MMIO 内部方法
    # ================================================================

    def _mmio_read(self, offset: int) -> int:
        if offset == _DOMAINCFG:
            return self._domaincfg

        if _SOURCECFG_BASE <= offset < _SOURCECFG_BASE + self.num_sources * 4:
            idx = (offset - _SOURCECFG_BASE) // 4 + 1
            if 1 <= idx <= self.num_sources:
                return self._sourcecfg[idx]
            return 0

        for base in (_SETIP_BASE, _CLRIP_BASE):
            if base <= offset < base + _BITFIELD_WORDS * 4:
                return self._read_word(offset, base, _STATE_PENDING)

        if _SETIPNUM <= offset < _SETIPNUM + 4 or _CLRIPNUM <= offset < _CLRIPNUM + 4:
            return 0

        for base in (_SETIE_BASE, _CLRIE_BASE):
            if base <= offset < base + _BITFIELD_WORDS * 4:
                return self._read_word(offset, base, _STATE_ENABLED)

        if _SETIENUM <= offset < _SETIENUM + 4 or _CLRIENUM <= offset < _CLRIENUM + 4:
            return 0

        if _SETIPNUM_LE <= offset < _SETIPNUM_LE + 4:
            return 0

        if _TARGET_BASE <= offset < _TARGET_BASE + self.num_sources * 4:
            idx = (offset - _TARGET_BASE) // 4 + 1
            if 1 <= idx <= self.num_sources:
                return self._target[idx]
            return 0

        # mmsicfgaddr / smsicfgaddr / genmsi / IDC: 读零 (MSI 模式无硬件回报)
        return 0

    def _mmio_write(self, offset: int, val: int) -> None:
        if offset == _DOMAINCFG:
            # 保留 IE (bit 8) 与 DM (bit 2) 位, 丢弃保留位.
            # Linux irq-riscv-aplic-main.c 写入 IE|DM 后回读校验,
            # 若丢弃 DM 会误报 "unable to write 0x104 in domaincfg".
            self._domaincfg = val & (_DOMAINCFG_IE | _DOMAINCFG_DM)
            if self._domaincfg & _DOMAINCFG_IE:
                # IE 0→1: 投递所有已 pending 且 enabled 的源
                for src in range(1, self.num_sources + 1):
                    self._deliver_from_source(src)
            return

        if _SOURCECFG_BASE <= offset < _SOURCECFG_BASE + self.num_sources * 4:
            idx = (offset - _SOURCECFG_BASE) // 4 + 1
            if 1 <= idx <= self.num_sources:
                # 仅保留 SM 触发类型 (bits[2:0]); 无子 APLIC 时清除 D (bit 10).
                self._sourcecfg[idx] = val & _SM_MASK
                if (val & _SM_MASK) == _SM_INACTIVE:
                    # 配成 INACTIVE 时清 pending 与 enabled (与 QEMU 一致)
                    self._state[idx] &= ~(_STATE_PENDING | _STATE_ENABLED)
                elif self._input_active(idx):
                    # 配成活动类型且输入已有效: 重新置 pending (与 QEMU 一致)
                    self._state[idx] |= _STATE_PENDING
            return

        if _SETIP_BASE <= offset < _SETIP_BASE + _BITFIELD_WORDS * 4:
            self._write_word(offset, _SETIP_BASE, val, set_bit=True, bit=_STATE_PENDING)
            return

        if _SETIPNUM <= offset < _SETIPNUM + 4:
            if 0 < val <= self.num_sources:
                if self._set_pending(val, True):
                    self._deliver_from_source(val)
            return

        if _CLRIP_BASE <= offset < _CLRIP_BASE + _BITFIELD_WORDS * 4:
            self._write_word(offset, _CLRIP_BASE, val, set_bit=False, bit=_STATE_PENDING)
            return

        if _CLRIPNUM <= offset < _CLRIPNUM + 4:
            if 0 < val <= self.num_sources:
                self._set_pending(val, False)
            return

        if _SETIE_BASE <= offset < _SETIE_BASE + _BITFIELD_WORDS * 4:
            self._write_word(offset, _SETIE_BASE, val, set_bit=True, bit=_STATE_ENABLED)
            return

        if _SETIENUM <= offset < _SETIENUM + 4:
            if 0 < val <= self.num_sources:
                self._state[val] |= _STATE_ENABLED
                self._deliver_from_source(val)
            return

        if _CLRIE_BASE <= offset < _CLRIE_BASE + _BITFIELD_WORDS * 4:
            self._write_word(offset, _CLRIE_BASE, val, set_bit=False, bit=_STATE_ENABLED)
            return

        if _CLRIENUM <= offset < _CLRIENUM + 4:
            if 0 < val <= self.num_sources:
                self._state[val] &= ~_STATE_ENABLED
            return

        if _SETIPNUM_LE <= offset < _SETIPNUM_LE + 4:
            # 小端 setipnum — EOI retrigger (AIA §4.9.2): 仅源仍有效时置 pending
            if 0 < val <= self.num_sources:
                if self._set_pending(val, True):
                    self._deliver_from_source(val)
            return

        if _TARGET_BASE <= offset < _TARGET_BASE + self.num_sources * 4:
            idx = (offset - _TARGET_BASE) // 4 + 1
            if 1 <= idx <= self.num_sources:
                self._target[idx] = val
            return

        # genmsi / mmsicfgaddr / smsicfgaddr / IDC: 接受写入, 无副作用.

    # ================================================================
    #  位图辅助
    # ================================================================

    def _read_word(self, offset: int, base: int, bit: int) -> int:
        word = (offset - base) // 4
        result = 0
        for i in range(32):
            src = word * 32 + i
            if 0 < src <= self.num_sources and (self._state[src] & bit):
                result |= 1 << i
        return result

    def _write_word(self, offset: int, base: int, val: int, *, set_bit: bool, bit: int) -> None:
        word = (offset - base) // 4
        for i in range(32):
            if not (val & (1 << i)):
                continue
            src = word * 32 + i
            if not (0 < src <= self.num_sources):
                continue
            if bit == _STATE_PENDING:
                # pending 位图走电平感知写入, 避免电平触发源在输入已撤除时
                # 被 setip 位图错误地重新置 pending (AIA §4.9.2)
                if self._set_pending(src, set_bit):
                    self._deliver_from_source(src)
                continue
            if set_bit:
                self._state[src] |= bit
            else:
                self._state[src] &= ~bit
            # 置 enable 或置 pending 都可能使源变为可投递
            self._deliver_from_source(src)

    # ================================================================
    #  pending 置位 / 有效输入电平 (对齐 QEMU riscv_aplic.c)
    # ================================================================

    def _input_active(self, src: int) -> bool:
        """源 ``src`` 的输入是否处于有效电平 (已应用低有效极性)。

        对齐 QEMU ``riscv_aplic_irq_rectified_val``: LEVEL_LOW / EDGE_FALL
        源取输入电平反相, 其余同相。
        """
        sm = self._sourcecfg[src] & _SM_MASK
        if sm == _SM_INACTIVE:
            return False
        raw = (self._state[src] & _STATE_INPUT) != 0
        return raw ^ (sm in (_SM_LEVEL_LOW, _SM_EDGE_FALL))

    def _set_pending(self, src: int, pending: bool) -> bool:
        """置/清源 ``src`` 的 pending 位, 电平触发源需输入仍有效。

        对齐 QEMU ``riscv_aplic_set_pending``。电平触发 (LEVEL_HIGH/LEVEL_LOW)
        源在 MSI 模式下, ``setipnum``/``setipnum_le`` 仅在输入仍处于有效电平时
        才重新置 pending (AIA 规范 §4.9.2); 否则每次 EOI retrigger 都会无条件
        重投, 形成中断风暴。

        返回 pending 位是否被实际置起 (供调用方决定是否投递)。
        """
        sm = self._sourcecfg[src] & _SM_MASK
        if sm == _SM_INACTIVE:
            return False

        if sm in (_SM_LEVEL_HIGH, _SM_LEVEL_LOW):
            if not pending:
                self._state[src] &= ~_STATE_PENDING
                return False
            if not self._input_active(src):
                return False
            self._state[src] |= _STATE_PENDING
            return True

        # 边沿触发: 直接置/清 pending
        if pending:
            self._state[src] |= _STATE_PENDING
            return True
        self._state[src] &= ~_STATE_PENDING
        return False

    # ================================================================
    #  MSI 投递
    # ================================================================

    def _deliver_from_source(self, src: int) -> None:
        """若源 pending & enabled 且 domaincfg.IE, 投递 MSI 到 target[src].

        与 QEMU ``riscv_aplic_msi_irq_update`` 对齐: 投递时清除 pending.
        """
        if (self._domaincfg & _DOMAINCFG_IE) == 0:
            return
        state = self._state[src]
        if (state & (_STATE_PENDING | _STATE_ENABLED)) != (_STATE_PENDING | _STATE_ENABLED):
            return

        # 清除 pending 后投递 (MSI 模式)
        self._state[src] &= ~_STATE_PENDING

        target = self._target[src]
        hart_idx = (target >> _TARGET_HART_IDX_SHIFT) & _TARGET_HART_IDX_MASK
        eiid = target & _TARGET_EIID_MASK
        if eiid == 0:
            return
        self._imsic.set_ip_number(hart_idx, 'S', eiid)
