#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""
RISC-V AIA Incoming MSI Controller (IMSIC).

每个 hart 有 M-mode 和 S-mode 两个 interrupt file，分别管理 2048 个
外部 + 软件中断 MSI 身份 (identity 1-2047)。

QEMU 兼容内存布局 (两个独立地址范围, 每 hart stride 0x1000):

    M-file 范围 (base = 0x2400_0000):
        base + hart_id * 0x1000 + 0x0000  M-file seteipnum (写即设 eip[val])
        base + hart_id * 0x1000 + 0x0008  M-file clreipnum (写即清 eip[val])

    S-file 范围 (base = 0x2800_0000):
        base + hart_id * 0x1000 + 0x0000  S-file seteipnum
        base + hart_id * 0x1000 + 0x0008  S-file clreipnum

Linux 内核通过 ``local->msi_pa = base + hart * 0x1000`` 计算 MSI 目标地址
(IMSIC_MMIO_PAGE_SHIFT=12), 因此每 hart stride 必须为 0x1000 而非 0x2000。

与 CLINT 集成: IMSIC 不实现 InterruptController ABC，而是与 PLIC 相同的角色 —
在 check_pending_interrupts 中将 get_pending_mip() 返回的 MEIP/SEIP 位合并
到 mip 寄存器。CLINT 保留处理软件中断 (MSIP via msip MMIO) 和定时器 (MTIP)。

用法:
    imsic = IMSIC(num_harts=4, m_base_addr=0x2400_0000, s_base_addr=0x2800_0000)
    imsic.set_ip_number(hart_id=0, priv='M', eip_num=10)
    mip_bits = imsic.get_pending_mip(hart_id=0)  # -> MEIP | SEIP
"""

from __future__ import annotations

from pyremu.memory.bus import Device
from pyremu.utils.mask import mask32

# 每 hart MSI doorbell 页大小 (AIA spec: IMSIC_MMIO_PAGE_SZ = 4 KiB)
_PAGE_STRIDE = 0x1000

# MMIO 寄存器 (per interrupt file page)
_EIPNUM_SET_OFF = 0x0000  # seteipnum — 写入 eip_num 即置 pending
_EIPNUM_CLR_OFF = 0x0008  # clreipnum — 写入 eip_num 即清 pending

# CSR select 索引 (RISC-V AIA 规范 v1.0, 与 Linux/OpenSBI 一致)
_SEL_EIDELIVERY = 0x70
_SEL_EITHRESHOLD = 0x72
_SEL_EIP_BASE = 0x80   # eip0 .. eip63
_SEL_EIP_END = 0xBF
# Per-bit eip/eie CSR indirect access (AI 规范, siselect=0x58 等)
# 写入 minor IID 到 sireg 时 set/clear 对应 eip/eie 位.
_SEL_SETEIPNUM = 0x58
_SEL_CLREIPNUM = 0x59
_SEL_SETEIENUM = 0x5A
_SEL_CLREIENUM = 0x5B

_SEL_EIE_BASE = 0xC0   # eie0 .. eie63
_SEL_EIE_END = 0xFF

# 中断文件总数 (per hart: M + S)
_FILES_PER_HART = 2

# 优先级模型: 中断号即优先级 (简化, 与 RISC-V AIA default map 兼容)
# 返回的 topei 值 = (IID << 16) | priority
_MAX_INTERRUPT_IDS = 2048
_EIP_WORDS = 64  # 2048 bits / 32 bits per word

# IMSIC minor identity 用作 OpenSBI 的 IPI doorbell 值 (AIA spec §3.2.1)
# 这些是 MINOR identity (外部中断编号), 不是 MAJOR identity (中断 cause)。
# ``IMSIC_IPI_ID = 1`` 是 OpenSBI 的约定: IPI 即向目标 hart 的 M-file 写
# ``seteipnum = 1`` 的普通 MSI, 作为外部中断 (MEIP) 投递, 经 MTOPEI 发现。
# IID=3 仅为对称保留, OpenSBI 实际不使用 (始终用 minor identity 1)。
_IID_S_IPI = 1  # OpenSBI IPI minor identity (M-file)
_IID_M_IPI = 3  # legacy M-file IPI minor identity
_IID_EXT_MIN = 6  # 外部中断起始 identity


class _ImsicFile:
    """单个 IMSIC interrupt file (M-mode 或 S-mode).

    eip/eie 为 u32 列表 (索引 = interrupt number // 32, bit = interrupt number % 32).
    """

    __slots__ = ("eip", "eie", "eidelivery", "eithreshold", "_any_ext")

    def __init__(self) -> None:
        self.eip: list[int] = [0] * _EIP_WORDS
        self.eie: list[int] = [0] * _EIP_WORDS
        self.eidelivery: int = 0
        self.eithreshold: int = 0
        self._any_ext: bool = False  # cached: any external (>=6) eip bit set

    def _update_any_ext(self) -> None:
        """Recompute _any_ext after eip modification.

        All IMSIC interrupts (software IID=1,3 and external IID>=6)
        contribute to _any_ext.  IPIs route through MEIP/SEIP per AIA spec.
        """
        for i in range(_EIP_WORDS):
            if self.eip[i]:
                self._any_ext = True
                return
        self._any_ext = False

    def set_pending(self, eip_num: int) -> None:
        """置 pending 位.  All IIDs (including software IPIs) contribute to _any_ext."""
        if 0 < eip_num < _MAX_INTERRUPT_IDS:
            word = eip_num >> 5
            bit = eip_num & 31
            self.eip[word] |= 1 << bit
            self._any_ext = True

    def clear_pending(self, eip_num: int) -> None:
        """清 pending 位."""
        if 0 < eip_num < _MAX_INTERRUPT_IDS:
            word = eip_num >> 5
            bit = eip_num & 31
            self.eip[word] &= ~(1 << bit)
            self._update_any_ext()

    def _bit_is_set(self, eip_num: int) -> bool:
        """检查指定 identity 的 pending & enabled.

        Software interrupts (IID=1,3): eip only — eie 不参与.
        External interrupts (IID>=6): eip & eie.
        """
        if not self.eidelivery:
            return False
        word = eip_num >> 5
        bit = eip_num & 31
        is_sw = eip_num in (_IID_M_IPI, _IID_S_IPI)
        if is_sw:
            return (self.eip[word] & (1 << bit)) != 0
        return (self.eip[word] & self.eie[word] & (1 << bit)) != 0

    def has_ext_pending(self) -> bool:
        """任意外部中断 (identity >= 6) pending & enabled.

        Fast-path: ``_any_ext`` flag avoids O(64) scan when no external
        eip bits are set — the common case during boot.
        """
        if not self.eidelivery or not self._any_ext:
            return False
        for i in range(_EIP_WORDS):
            pending = self.eip[i] & self.eie[i]
            if i == 0:
                pending &= ~((1 << _IID_M_IPI) | (1 << _IID_S_IPI))
            if pending:
                return True
        return False

    def has_pending(self) -> bool:
        """任一 pending 且 eidelivery=1.

        Software interrupts (IID=1,3) only need eip — no eie requirement.
        External interrupts (IID>=6) need both eip & eie.
        """
        if not self.eidelivery:
            return False
        sw_mask = (1 << _IID_M_IPI) | (1 << _IID_S_IPI)
        ext_mask = ~sw_mask & 0xFFFFFFFF
        for i in range(_EIP_WORDS):
            if i == 0:
                pending = (self.eip[i] & sw_mask) | (self.eip[i] & self.eie[i] & ext_mask)
            else:
                pending = self.eip[i] & self.eie[i]
            if pending:
                return True
        return False

    def peek_topei(self) -> int:
        """返回 top interrupt identity + priority, 不修改 pending 位.

        mtopi (0xFB0) 使用此方法 — 只读, 不 claim.
        返回 (IID << 16) | priority, 或 0 表示无中断.

        Software interrupts (IID=1,3): eip only — eie 不参与 (eie 仅管控外部中断).
        External interrupts (IID>=6): eip & eie.
        """
        # IPI fast-path: software interrupts (IID=1,3) are always "enabled"
        # when pending — they do NOT require eie NOR eidelivery.
        # eidelivery only gates external interrupts (IID >= 6).
        # Checking IPIs first ensures pending IPIs are visible even before
        # the kernel sets eidelivery, matching the Rust imsic_topei_peek.
        sw_mask = (1 << _IID_M_IPI) | (1 << _IID_S_IPI)
        ipi_pending = (self.eip[0] & sw_mask) != 0
        if ipi_pending:
            bit = (self.eip[0] & sw_mask).bit_length() - 1
            iid = bit
            prio = iid & 0xFF
            return (iid << 16) | prio
        if not self.eidelivery:
            return 0
        ext_mask = mask32(~sw_mask)
        if not self._any_ext:
            return 0
        for i in range(_EIP_WORDS - 1, -1, -1):
            pending_enabled = self.eip[i] & self.eie[i]
            if i == 0:
                pending_enabled = (self.eip[i] & sw_mask) | (pending_enabled & ext_mask)

            if pending_enabled == 0:
                continue
            bit = pending_enabled.bit_length() - 1
            iid = (i << 5) + bit
            prio = iid & 0xFF
            if prio > self.eithreshold:
                return (iid << 16) | prio
        return 0

    def read_topei(self) -> int:
        """返回 top interrupt identity + priority 并 claim.

        mtopei (0x35C) / stopei (0x15C) 使用此方法 — 读并清除 pending.
        返回 (IID << 16) | priority, 或 0 表示无中断.

        Software interrupts (IID=1,3): eip only — eie 不参与.
        External interrupts (IID>=6): eip & eie.
        """
        # IPI fast-path: software interrupts (IID=1,3) are always "enabled"
        # when pending — they do NOT require eie NOR eidelivery (matching
        # both peek_topei above and Rust's imsic_topei_peek).
        sw_mask = (1 << _IID_M_IPI) | (1 << _IID_S_IPI)
        ipi_pending = (self.eip[0] & sw_mask) != 0
        if ipi_pending:
            bit = (self.eip[0] & sw_mask).bit_length() - 1
            iid = bit
            self.eip[0] &= ~(1 << bit)
            self._update_any_ext()
            prio = iid & 0xFF
            return (iid << 16) | prio
        if not self.eidelivery:
            return 0
        ext_mask = mask32(~sw_mask)
        if not self._any_ext:
            return 0
        # 从高优先级 (高 IID) 向低扫描
        for i in range(_EIP_WORDS - 1, -1, -1):
            pending_enabled = self.eip[i] & self.eie[i]
            if i == 0:
                pending_enabled = (
                    (self.eip[i] & sw_mask)
                    | (pending_enabled & ext_mask)
                )
            if pending_enabled == 0:
                continue
            # 找到最高置位 (bit_length - 1)
            bit = pending_enabled.bit_length() - 1
            iid = (i << 5) + bit
            # claim: 清除 pending 位
            self.eip[i] &= ~(1 << bit)
            self._update_any_ext()
            # 优先级 = IID 的低 8 位 (简化, 符合 AIA identity-based priority)
            prio = iid & 0xFF
            if prio > self.eithreshold:
                return (iid << 16) | prio
        return 0

    def csr_read(self, select: int) -> int:
        """根据 miselect/siselect 索引读取寄存器."""
        if select == _SEL_EIDELIVERY:
            return self.eidelivery
        if select == _SEL_EITHRESHOLD:
            return self.eithreshold
        if _SEL_EIP_BASE <= select <= _SEL_EIP_END:
            idx = select - _SEL_EIP_BASE
            lo = self.eip[idx]
            hi = self.eip[idx + 1] if idx + 1 < _EIP_WORDS else 0
            return lo | (hi << 32)
        if _SEL_EIE_BASE <= select <= _SEL_EIE_END:
            idx = select - _SEL_EIE_BASE
            lo = self.eie[idx]
            hi = self.eie[idx + 1] if idx + 1 < _EIP_WORDS else 0
            return lo | (hi << 32)
        return 0

    def csr_write(self, select: int, val: int) -> None:
        """根据 miselect/siselect 索引写入寄存器."""
        if select == _SEL_EIDELIVERY:
            self.eidelivery = val & 1
            return
        if select == _SEL_EITHRESHOLD:
            self.eithreshold = val & 0x3FF
            return
        if _SEL_EIP_BASE <= select <= _SEL_EIP_END:
            idx = select - _SEL_EIP_BASE
            self.eip[idx] = mask32(val)
            if idx + 1 < _EIP_WORDS:
                self.eip[idx + 1] = mask32(val >> 32)
            self._update_any_ext()
            return
        # Per-bit eip/eie manipulation: write minor IID → set/clear bit.
        if select == _SEL_SETEIPNUM:
            eip_num = val & 0x7FF
            if eip_num < _MAX_INTERRUPT_IDS:
                word = eip_num >> 5
                self.eip[word] |= 1 << (eip_num & 31)
                self._update_any_ext()
            return
        if select == _SEL_CLREIPNUM:
            eip_num = val & 0x7FF
            if eip_num < _MAX_INTERRUPT_IDS:
                word = eip_num >> 5
                self.eip[word] &= ~(1 << (eip_num & 31))
                self._update_any_ext()
            return
        if select == _SEL_SETEIENUM:
            eip_num = val & 0x7FF
            if eip_num < _MAX_INTERRUPT_IDS:
                word = eip_num >> 5
                self.eie[word] |= 1 << (eip_num & 31)
            return
        if select == _SEL_CLREIENUM:
            eip_num = val & 0x7FF
            if eip_num < _MAX_INTERRUPT_IDS:
                word = eip_num >> 5
                self.eie[word] &= ~(1 << (eip_num & 31))
            return
        if _SEL_EIE_BASE <= select <= _SEL_EIE_END:
            idx = select - _SEL_EIE_BASE
            self.eie[idx] = mask32(val)
            if idx + 1 < _EIP_WORDS:
                self.eie[idx + 1] = mask32(val >> 32)


class IMSIC(Device):
    """RISC-V AIA Incoming MSI Controller — 每 hart 两个 interrupt file.

    所有 IMSIC 中断 (软件 IPI IID=1,3 和外部 IID>=6) 通过外部中断线
    (MEIP/SEIP) 投递，符合 AIA 规范。get_pending_mip() 将 eip 映射到
    mip.MEIP/mip.SEIP，M-mode handler 通过 MTOPEI/STOPEI 读取 IID。

    QEMU 兼容布局: M-file 和 S-file 使用独立地址范围, 每 hart stride 0x1000.
    """

    def __init__(
        self,
        num_harts: int,
        m_base_addr: int = 0x2400_0000,
        s_base_addr: int = 0x2800_0000,
        ipi_target = None,  # callable(hart_id) — CLINT.send_ipi 或兼容接口
    ) -> None:
        self.num_harts = num_harts
        self.m_base_addr = m_base_addr
        self.s_base_addr = s_base_addr

        # IPI 路由目标 (CLINT)
        self._ipi_target = ipi_target

        # 每 hart 的 M/S interrupt file: _files[hart_id] = (m_file, s_file)
        self._files: list[tuple[_ImsicFile, _ImsicFile]] = [
            (_ImsicFile(), _ImsicFile()) for _ in range(num_harts)
        ]
        # 总 MMIO 覆盖范围: 连续 M+S, 每 hart stride 0x1000.
        # QEMU 兼容单一 reg 条目: <base, 2*num_harts*0x1000>.
        self.size = 2 * num_harts * _PAGE_STRIDE
        self.base_addr = m_base_addr  # 保留旧接口兼容

    # ----------------------------------------------------------
    #  内部辅助
    # ----------------------------------------------------------

    def _resolve(self, offset: int) -> tuple[int, _ImsicFile, int] | None:
        """根据 MMIO 偏移解析 (hart_id, file, offset_within_file).

        QEMU-compatible contiguous layout:
          - M-files: offset [0,          num_harts * 0x1000)
          - S-files: offset [N*0x1000, 2*num_harts * 0x1000)
        每范围内 stride = 0x1000 (IMSIC_MMIO_PAGE_SZ).
        Returns None 若不在有效范围内.
        """
        m_size = self.num_harts * _PAGE_STRIDE

        if 0 <= offset < m_size:
            # M-file 范围
            hart_id = offset // _PAGE_STRIDE
            off = offset % _PAGE_STRIDE
            return (hart_id, self._files[hart_id][0], off)
        if m_size <= offset < 2 * m_size:
            # S-file 范围
            rel = offset - m_size
            hart_id = rel // _PAGE_STRIDE
            off = rel % _PAGE_STRIDE
            return (hart_id, self._files[hart_id][1], off)
        return None

    def _file_for(
        self,
        hart_id: int,
        priv: str,
    ) -> _ImsicFile | None:
        """返回指定 hart + privilege 的 interrupt file."""
        if not (0 <= hart_id < self.num_harts):
            return None
        mf, sf = self._files[hart_id]
        if priv == 'M':
            return mf
        if priv == 'S':
            return sf
        return None

    # ----------------------------------------------------------
    #  Device 接口 (MMIO)
    # ----------------------------------------------------------

    def read(self, offset: int, size: int) -> bytes:
        """IMSIC MMIO 读 — seteipnum/clreipnum 为只写, 返回 0."""
        return b"\x00" * size

    def write(self, offset: int, data: bytes) -> None:
        """IMSIC MMIO 写 — seteipnum / clreipnum.

        所有 IMSIC 中断 (软件 IPI IID=1,3 和外部 IID>=6) 通过
        MEIP/SEIP 外部中断线投递，符合 AIA 规范。get_pending_mip()
        将 eip 映射到 mip.MEIP/mip.SEIP，check_pending_interrupts
        检测到后投递 MEI/SEI trap。M-mode handler 通过 CSR_MTOPEI
        读取中断 identity 并 dispatch。

        **直接映射, 无跨文件路由**: ``seteipnum = N`` 在被寻址的 interrupt
        file 中置 ``eip[N]`` 并驱动该文件的 external 中断线 (M-file → MEIP,
        S-file → SEIP)。写入的值是 MINOR identity (外部中断编号), 不是 MAJOR
        identity (中断 cause)。OpenSBI 通过向目标 hart 的 M-file 写 minor
        identity 1 (``IMSIC_IPI_ID``) 发送 IPI, 接收方 M-mode 经 ``MTOPEI``
        读取并 dispatch 到 ``sbi_ipi_process``; 若将 M-file 的 IID=1 路由到
        S-file 会丢失 M-mode IPI, 导致次级 hart 在 SMP 启动时无法唤醒。

        *offset* 是相对于 device.base_addr 的偏移 (Bus 约定).
        """
        resolved = self._resolve(offset)
        if resolved is None:
            return
        _hart_id, file, file_off = resolved
        val = int.from_bytes(data, "little", signed=False)
        val32 = mask32(val)

        if file_off == _EIPNUM_SET_OFF:
            file.set_pending(val32)
        elif file_off == _EIPNUM_CLR_OFF:
            file.clear_pending(val32)

    # ----------------------------------------------------------
    #  外部中断查询 (供 check_pending_interrupts 使用)
    # ----------------------------------------------------------

    def get_pending_mip(self, hart_id: int) -> int:
        """返回该 hart 的中断 pending bits.

        所有 IMSIC 中断 (软件 IPI IID=1,3 和外部 IID>=6) 通过外部中断线
        (MEIP/SEIP) 投递，符合 AIA 规范。 M-mode trap handler 从 MTOPEI/STOPEI
        读取中断 identity，并通过 MTOPEI/STOPEI 写操作 claim。

        Uses peek_topei which correctly handles IPI IID=1,3 even when
        eidelivery=0 (IPI fast-path bypasses eidelivery).  External interrupts
        (IID>=6) require eidelivery=1 — when eidelivery=0 they are managed
        by the legacy PLIC path or Rust ext_irq drain.
        """
        if not (0 <= hart_id < self.num_harts):
            return 0
        mip = 0

        m_val = self.peek_topei(hart_id, 'M')
        if m_val != 0:
            mip |= 1 << 11  # MEIP

        s_val = self.peek_topei(hart_id, 'S')
        if s_val != 0:
            mip |= 1 << 9   # SEIP

        return mip

    def is_delivery_active(self, hart_id: int) -> bool:
        """Return True if IMSIC delivery is enabled for this hart.

        When eidelivery=0 for both M and S files, external interrupts are
        managed by the legacy PLIC path and IMSIC should not gate MEIP/SEIP.
        """
        mf = self._file_for(hart_id, 'M')
        sf = self._file_for(hart_id, 'S')
        return (mf is not None and mf.eidelivery != 0) or \
               (sf is not None and sf.eidelivery != 0)

    # ----------------------------------------------------------
    #  MSI 注入 (供 Phase 2 APLIC 使用)
    # ----------------------------------------------------------

    def set_ip_number(self, hart_id: int, priv: str, eip_num: int) -> None:
        """向指定 hart 的指定 privilege IMSIC file 注入 MSI.

        所有 IMSIC 中断 (软件 IPI 和外部) 通过 MEIP/SEIP 投递。
        get_pending_mip() 将 eip 映射到 mip 位，check_pending_interrupts
        检测并投递 MEI/SEI trap。M-mode handler 通过 MTOPEI 读取 IID。
        """
        file = self._file_for(hart_id, priv)
        if file is not None:
            file.set_pending(eip_num)

    def clear_ip_number(self, hart_id: int, priv: str, eip_num: int) -> None:
        """清除指定 hart 的指定 privilege IMSIC file 的 pending 位.

        供 APLIC set_irq(source, False) 在设备撤除 level-triggered
        中断时调用, 将 IMSIC eip 同步清除 (guest 已通过 stopei claim,
        但设备端仍需通知 IMSIC 中断已不存在).
        """
        file = self._file_for(hart_id, priv)
        if file is not None:
            file.clear_pending(eip_num)

    # ----------------------------------------------------------
    #  mtopi / stopi IID claim — 完整覆盖全部 6 个主要 IID
    # ----------------------------------------------------------

    def claim_mtopi_iid(self, hart_id: int, iid: int) -> int:
        """M-mode mtopi claim: 认领 IID 并返回需清除的 mip 位掩码.

        覆盖全部 M 模式 IID:
          IID=0: no-op
          IID=3 (MSIP):  清 IMSIC M-file eip[3] + mip.MSIP
          IID=7 (MTIP):  清 mip.MTIP (bump mtimecmp 由调用方负责)
          IID=11 (MEIP): 清 IMSIC M-file 最高 eip + mip.MEIP
          其他外部 IID:  清 IMSIC M-file 指定 eip + mip.MEIP
        """
        if iid == 0:
            return 0
        mf = self._file_for(hart_id, 'M')
        if iid == 7 or mf is None:
            return 1 << iid
        if iid == 11:
            # MEI major identity: claim TOP M-file eip (kernel read MTOPEI
            # first for minor IID).  clear_pending(11) would target eip[11]
            # which is NOT the correct pending bit.
            mf.read_topei()
            return 1 << 11
        mf.clear_pending(iid)
        return 1 << iid

    def claim_stopi_iid(self, hart_id: int, iid: int) -> int:
        """S-mode stopi claim: 认领 IID 并返回需清除的 mip 位掩码.

        覆盖全部 S 模式 IID (AIA 规范):
          IID=0: no-op
          IID=1 (SSIP):  清 IMSIC S-file eip[1] + mip.SSIP
          IID=5 (STIP):  清 mip.STIP (bump stimecmp 由调用方负责)
          IID=9 (SEIP):  清 IMSIC S-file 最高 eip (major→top claim)
          其他外部 IID:  清 IMSIC S-file 指定 eip + mip.SEIP
        """
        if iid == 0:
            return 0
        sf = self._file_for(hart_id, 'S')
        if iid == 5 or sf is None:
            return 1 << iid
        if iid == 9:
            # SEI major identity: claim TOP S-file eip (kernel read STOPEI
            # first for minor IID).  clear_pending(9) would target eip[9]
            # which is NOT the correct pending bit.
            sf.read_topei()
            return 1 << 9
        sf.clear_pending(iid)
        return 1 << iid

    # ----------------------------------------------------------
    #  CSR 接口 (供 hart.py write_csr / read_csr 调用)
    # ----------------------------------------------------------

    def csr_read(self, hart_id: int, priv: str, select: int) -> int:
        """根据 miselect/siselect 读 IMSIC 寄存器."""
        file = self._file_for(hart_id, priv)
        if file is None:
            return 0
        return file.csr_read(select)

    def csr_write(self, hart_id: int, priv: str, select: int, val: int) -> None:
        """根据 miselect/siselect 写 IMSIC 寄存器."""
        file = self._file_for(hart_id, priv)
        if file is not None:
            file.csr_write(select, val)

    # ----------------------------------------------------------
    #  topei (mtopei / stopei) 读
    # ----------------------------------------------------------

    def peek_topei(self, hart_id: int, priv: str) -> int:
        """返回 (IID << 16) | priority, 不修改 pending 位 (mtopi 用)."""
        file = self._file_for(hart_id, priv)
        if file is None:
            return 0
        return file.peek_topei()

    def read_topei(self, hart_id: int, priv: str) -> int:
        """返回 (IID << 16) | priority, 同时 claim (清 pending)."""
        file = self._file_for(hart_id, priv)
        if file is None:
            return 0
        return file.read_topei()
