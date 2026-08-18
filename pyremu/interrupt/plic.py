#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""
RISC-V Platform-Level Interrupt Controller (PLIC).

实现 PLIC 规范 v1.0.0 的核心功能:
- 中断源优先级 (可编程)
- 中断挂起 / 清除 (pending)
- 按上下文 (hart * privilege) 的使能位
- 优先级阈值 + Claim/Complete 机制

内存布局 (SiFive 兼容, context stride = 0x1000):

  0x000000 - 0x000FFC   Source priorities (4 bytes each, source 1..N)
  0x001000 - 0x00107C   Pending bits (32 words * 32 bits)
  0x002000 - 0x00207C   Enable bits context 0 (32 words)
  0x002080 - 0x0020FC   Enable bits context 1
  ...                   (每个 context 0x80 字节)
  0x200000              Context 0 threshold (u32)
  0x200004              Context 0 claim/complete (u32)
  0x201000              Context 1 threshold (u32)
  ...

与 CLINT 集成: PLIC 不直接实现 InterruptController;
而是通过 hart._plic 引用, 在 check_pending_interrupts 中
将 PLIC 返回的 MEIP/SEIP 位合并到 mip 寄存器.

用法:
    plic = PLIC(base_addr=0x0C00_0000, num_sources=128, num_contexts=4)
    plic.set_irq(source_id=10, pending=True)   # 硬件置起中断
    plic.get_pending_mip(hart_id=0)            # -> MEIP 或 SEIP bit
"""

from __future__ import annotations

from pyremu.memory.bus import Device

# 诊断日志开关 (PYREMU_DIAG_VERBOSE=1): 打印 set_irq / claim / complete,
# 用于确认外设完成中断是否被正确投递并被 hart claim/complete。


# PLIC 常量 — 地址空间分区
PLIC_PRIORITY_BASE = 0x000000
PLIC_PRIORITY_STRIDE = 4
PLIC_PENDING_BASE = 0x001000
PLIC_ENABLE_BASE = 0x002000
PLIC_ENABLE_STRIDE = 0x80  # 每 context 128 bytes = 32 u32 words
PLIC_CONTEXT_BASE = 0x200000
PLIC_CONTEXT_STRIDE = 0x1000  # 每 context 4 KiB
PLIC_CONTEXT_THRESHOLD_OFF = 0x000
PLIC_CONTEXT_CLAIM_OFF = 0x004

# 默认 PLIC 基址 (SiFive FU540: 0x0C00_0000)
PLIC_DEFAULT_BASE = 0x0C00_0000


class PLIC(Device):
    """RISC-V Platform-Level Interrupt Controller (PLIC).

    实现 MMIO 寄存器接口供 Guest 访问,
    同时提供 set_irq / get_pending_mip 供硬件模型集成.
    """

    def __init__(
        self,
        base_addr: int = PLIC_DEFAULT_BASE,
        num_sources: int = 128,
        num_contexts: int = 1,
    ) -> None:
        self.base_addr = base_addr
        self._num_sources = num_sources  # source 1..N (source 0 保留)
        self._num_contexts = num_contexts

        # 寄存器状态
        # _priority[i]: source i 的优先级 (0=禁用, 1..7 有效)
        self._priority: list[int] = [0] * (num_sources + 1)

        # _pending[i]: source i 是否挂起
        self._pending: list[bool] = [False] * (num_sources + 1)

        # _level[i]: source i 的电平状态 (设备侧最近一次 set_irq 的值)。
        # PLIC gateway 语义 (对照 QEMU sifive_plic): claim 清除 pending,
        # complete 时若电平仍为高则重新置位 pending — 否则电平中断在
        # "claim 后无任何设备寄存器访问" 的窗口内会永久丢失 (如 UART TX
        # watermark: ISR 每次仅发 FIFO 深度个字符, TXDATA 写由 Rust inline
        # 处理不经 Python, complete 后无人再拉 set_irq).
        self._level: list[bool] = [False] * (num_sources + 1)

        # _enable[c][i]: context c 启用 source i
        self._enable: list[list[bool]] = [
            [False] * (num_sources + 1) for _ in range(num_contexts)
        ]

        # _threshold[c]: context c 的优先级阈值 (低于此的 source 不通知)
        self._threshold: list[int] = [0] * num_contexts

        # _claimed[c]: context c 当前正在服务的中断源 (0 = 无)
        self._claimed: list[int] = [0] * num_contexts

        # 计算设备区域大小 — 到最后一个 context + 4 KiB
        last_context_end = (
            PLIC_CONTEXT_BASE
            + (num_contexts - 1) * PLIC_CONTEXT_STRIDE
            + 0x1000
        )
        self.size = last_context_end

    # ---- 硬件集成 API (供模拟器其他组件调用) ----

    def set_irq(self, source: int, pending: bool) -> None:
        """由设备模型调用: 设置中断源电平与挂起状态."""
        if 0 < source <= self._num_sources:
            self._level[source] = pending
            self._pending[source] = pending

    def get_pending_mip(self, hart_id: int) -> int:
        """返回该 hart 的待处理外部中断 mip 位.

        标准双 context 布局: context 2*hart_id 为 M 模式, 2*hart_id+1 为 S 模式。
        分别检查两个 context 是否有优先级 > 阈值且使能且挂起的中断源:
        - M-context 命中 -> MEIP (bit 11)
        - S-context 命中 -> SEIP (bit 9)
        两者可同时置位。
        """
        m_ctx = 2 * hart_id
        s_ctx = m_ctx + 1
        mip = 0
        if m_ctx < self._num_contexts and self._find_highest(m_ctx) > 0:
            mip |= 1 << 11  # MEIP (M-context)
        if s_ctx < self._num_contexts and self._find_highest(s_ctx) > 0:
            mip |= 1 << 9  # SEIP (S-context)
        return mip

    # ---- 中断仲裁 ----

    def _find_highest(self, context: int) -> int:
        """返回 context 中优先级最高且符合条件的中断源编号, 若无则返回 0.

        条件: pending[i] AND enable[context][i] AND priority[i] > threshold[context]
        有多个时取最高优先级; 同优先级取最小 source id (RISC-V spec §7.3).
        """
        threshold = self._threshold[context]
        best_source = 0
        best_prio = 0

        for i in range(1, self._num_sources + 1):
            if not self._pending[i] or not self._enable[context][i]:
                continue
            # 该源已在该 context 上被 claim 但尚未 complete — 跳过
            if self._claimed[context] == i:
                continue
            pri = self._priority[i]
            if pri <= threshold:
                continue
            if pri > best_prio:
                best_prio = pri
                best_source = i
            # 同优先级取最小 ID (由于 id 递增遍历, 自然满足)

        return best_source

    # ---- Device 接口 ----

    def read(self, offset: int, size: int) -> bytes:
        if size not in (2, 4):
            return b"\x00" * size
        val = self._mmio_read(offset)
        return val.to_bytes(size, "little")

    def write(self, offset: int, data: bytes) -> None:
        size = len(data)
        if size not in (2, 4):
            return
        val = int.from_bytes(data, "little")
        self._mmio_write(offset, val)

    def _mmio_read(self, offset: int) -> int:
        # Source priorities (0x000000 - 0x000FFC)
        if offset < PLIC_PENDING_BASE:
            src = offset // PLIC_PRIORITY_STRIDE
            if 1 <= src <= self._num_sources:
                return self._priority[src] & 0x7  # 仅 bits [2:0] 有效
            return 0

        # Pending bits (0x001000 - 0x00107C)
        if offset < PLIC_ENABLE_BASE:
            word_idx = (offset - PLIC_PENDING_BASE) // 4
            return self._read_pending_word(word_idx)

        # Enable bits (0x002000 - ...)
        if offset < PLIC_CONTEXT_BASE:
            rel = offset - PLIC_ENABLE_BASE
            context = rel // PLIC_ENABLE_STRIDE
            word_idx = (rel % PLIC_ENABLE_STRIDE) // 4
            return self._read_enable_word(context, word_idx)

        # Context registers (0x200000 - ...)
        rel = offset - PLIC_CONTEXT_BASE
        context = rel // PLIC_CONTEXT_STRIDE
        ctx_off = rel % PLIC_CONTEXT_STRIDE

        if context >= self._num_contexts:
            return 0

        if ctx_off == PLIC_CONTEXT_THRESHOLD_OFF:
            return self._threshold[context] & 0x7
        if ctx_off == PLIC_CONTEXT_CLAIM_OFF:
            return self._do_claim(context)

        return 0

    def _mmio_write(self, offset: int, val: int) -> None:
        # Source priorities
        if offset < PLIC_PENDING_BASE:
            src = offset // PLIC_PRIORITY_STRIDE
            if 1 <= src <= self._num_sources:
                self._priority[src] = val & 0x7
            return

        # Pending bits — read-only (pending 由硬件 set_irq 控制)
        if offset < PLIC_ENABLE_BASE:
            return

        # Enable bits
        if offset < PLIC_CONTEXT_BASE:
            rel = offset - PLIC_ENABLE_BASE
            context = rel // PLIC_ENABLE_STRIDE
            word_idx = (rel % PLIC_ENABLE_STRIDE) // 4
            self._write_enable_word(context, word_idx, val)
            return

        # Context registers
        rel = offset - PLIC_CONTEXT_BASE
        context = rel // PLIC_CONTEXT_STRIDE
        ctx_off = rel % PLIC_CONTEXT_STRIDE

        if context >= self._num_contexts:
            return

        if ctx_off == PLIC_CONTEXT_THRESHOLD_OFF:
            self._threshold[context] = val & 0x7
        elif ctx_off == PLIC_CONTEXT_CLAIM_OFF:
            self._do_complete(context, val)

    # ---- Claim / Complete ----

    def _do_claim(self, context: int) -> int:
        """Claim: 返回最高优先级待处理源, 清除 pending 位.

        RISC-V PLIC 规范: claim 仅清除 pending, 不修改 level.
        level 由设备侧 set_irq 独占控制 — 设备降电平后 complete
        不会重挂 pending, 设备保持高电平时 complete 立即重挂.
        """
        src = self._find_highest(context)
        if src > 0:
            self._pending[src] = False
            self._claimed[context] = src
        return src

    def _do_complete(self, context: int, src: int) -> None:
        """Complete: 标记中断处理完成, 允许再次触发.

        若设备电平仍为高 (claim 后设备未拉低 set_irq),
        complete 时重新置位 pending, 使中断再次投递
        """
        if 0 < src <= self._num_sources and self._claimed[context] == src:
            self._claimed[context] = 0
            if self._level[src]:
                self._pending[src] = True

    # ---- 位数组辅助 ----

    def _read_pending_word(self, word_idx: int) -> int:
        result = 0
        word_idx <<= 5
        for i in range(32):
            src = word_idx + i
            if src <= self._num_sources and self._pending[src]:
                result |= 1 << i
        return result

    def _read_enable_word(self, context: int, word_idx: int) -> int:
        if context >= self._num_contexts:
            return 0
        result = 0
        word_idx <<= 5
        for i in range(32):
            src = word_idx + i
            if src <= self._num_sources and self._enable[context][src]:
                result |= 1 << i
        return result

    def _write_enable_word(self, context: int, word_idx: int, val: int) -> None:
        if context >= self._num_contexts:
            return
        word_idx <<= 5
        for i in range(32):
            src = word_idx + i
            if src < 1 or src > self._num_sources:
                continue
            self._enable[context][src] = (val >> i) & 1 != 0
