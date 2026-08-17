#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026
# Created at 2026/06/09 星期二

"""
SiFive 兼容的 CLINT (Core Local Interruptor) 设备。

CLINT 提供:
- 核间中断 (IPI): 通过内存映射的 msip 寄存器, hart A 写 hart B 的 msip -> hart B 收到
                    Machine Software Interrupt
- 定时器中断:     每个 hart 有独立的 mtimecmp, 当 mtime >= mtimecmp 时触发
                   Machine Timer Interrupt
- 全局计数器:     mtime, 单调递增

寄存器布局 (SiFive 标准, 基址 0x0200_0000):
    MSIP_BASE     = 0x0000  (每个 hart 4 字节, bit 0 = software interrupt pending)
    MTIMECMP_BASE = 0x4000  (每个 hart 8 字节)
    time_base_val    = 0xBFF8  (8 字节, 全局共享)

同时实现 InterruptController 和 Device 接口, 以便 Bus 和 Emulator 使用。
"""

from collections.abc import Callable

from pyremu.interrupt.controller import INT_SOURCE_MIP_MASK, InterruptController, IntSource
from pyremu.memory.bus import Device
from pyremu.utils.mask import mask64

# CLINT 标准基址 (SiFive)
CLINT_BASE = 0x0200_0000
CLINT_SIZE = 0xC000  # 48 KiB

# 中断优先级 -> mip 位掩码 (按优先级从高到低排列).
# 预计算为 (mask, IntSource) 元组, 避免每条指令在 check_interrupt 中
# 遍历 IntSource Enum 并做 dict 查找 (profile 显示 6M Enum.__hash__/s).
_INT_PRIORITY: list[tuple[int, IntSource]] = [
    (1 << 11, IntSource.MEI),  # MEIP
    (1 << 3,  IntSource.MSI),  # MSIP
    (1 << 7,  IntSource.MTI),  # MTIP
    (1 << 9,  IntSource.SEI),  # SEIP
    (1 << 1,  IntSource.SSI),  # SSIP
    (1 << 5,  IntSource.STI),  # STIP
]

# 寄存器偏移
MSIP_OFFSET = 0x0000
MTIMECMP_OFFSET = 0x4000
MTIME_OFFSET = 0xBFF8


class CLINT(InterruptController, Device):
    """SiFive 兼容 CLINT 设备.

    同时实现:
    - InterruptController: Emulator 用于中断轮询和 IPI
    - Device:              Bus 用于内存映射读写
    """

    def __init__(self, num_harts: int, base_addr: int = CLINT_BASE) -> None:
        self._num_harts = num_harts
        self.base_addr = base_addr
        self.size = CLINT_SIZE

        # 每个 hart 的 MSIP (软件中断挂起): bit 0 有效
        self._msip = [0] * num_harts

        # 每个 hart 的 MTIMECMP (定时器比较值): 64-bit, 上电默认 0 (未配置)
        self._mtimecmp = [0] * num_harts

        # 全局 MTIME (单调计数器): 64-bit
        self._mtime = 0

        # 中断状态变化回调列表 — 每个 hart 注册自己的 notify_int_state_change
        self._int_state_cbs: list[Callable[[], None]] = []

    # ----------------------------------------------------------
    #  InterruptController 接口
    # ----------------------------------------------------------

    def check_interrupt(self, hart_id: int) -> tuple[bool, int, IntSource | None]:
        """检查 hart 的中断挂起状态.

        返回 (has_pending, mip_value, highest_source).
        注意: 这里只返回 mip 值, mie 的检查由 Hart 负责.
        """
        mip = 0

        # MSIP: Machine Software Interrupt
        if self._msip[hart_id] & 1:
            mip |= INT_SOURCE_MIP_MASK[IntSource.MSI]

        # MTIP: Machine Timer Interrupt
        if self._mtime >= self._mtimecmp[hart_id] and self._mtimecmp[hart_id] > 0:
            mip |= INT_SOURCE_MIP_MASK[IntSource.MTI]

        # 按优先级找最高优先级的待处理中断.
        # 用预计算元组替代 Enum 迭代 + dict 查找, 避免每条指令:
        #   - Enum.__iter__ 调用 (1M/s)
        #   - Enum.__hash__ 调用 (6M/s)
        #   - dict.__getitem__ 开销
        highest = None
        for mask, src in _INT_PRIORITY:
            if mip & mask:
                highest = src
                break

        has_pending = mip != 0 and highest is not None
        return has_pending, mip, highest

    def set_int_state_change_callback(self, cb: Callable[[], None]) -> None:
        """注册中断状态变化回调 (每个 hart 在绑定 interrupt_ctrl 时调用).

        支持多次注册 — CLINT 状态变化时所有回调均被调用.
        """
        self._int_state_cbs.append(cb)

    def _notify_state_change(self) -> None:
        """通知所有 hart 中断状态可能已改变."""
        for cb in self._int_state_cbs:
            cb()

    def send_ipi(self, target_hart_id: int) -> None:
        """向目标 hart 发送 IPI (置位 MSIP)."""
        if 0 <= target_hart_id < self._num_harts:
            self._msip[target_hart_id] |= 1
            self._notify_state_change()

    def clear_ipi(self, hart_id: int) -> None:
        """清除 hart 的软件中断挂起位."""
        if 0 <= hart_id < self._num_harts:
            self._msip[hart_id] &= ~1
            self._notify_state_change()

    def get_mtime(self) -> int:
        """返回当前全局时钟值."""
        return self._mtime

    def get_next_timer_wakeup(self, hart_id: int) -> int:
        """返回 hart 的下一次定时器唤醒时间 (mtime 值); 0 表示无定时器使能.

        供中断缓存快速路径: 若 mtime < next_wakeup, 可跳过全量中断检查.
        """
        if not (0 <= hart_id < self._num_harts):
            return 0
        cmp = self._mtimecmp[hart_id]
        return cmp if cmp > 0 else 0

    def get_mtimecmp(self, hart_id: int) -> int:
        """返回指定 hart 的 mtimecmp 值 (裸值, 不做 >0 过滤)."""
        if not (0 <= hart_id < self._num_harts):
            return 0
        return self._mtimecmp[hart_id]

    def set_mtimecmp(self, hart_id: int, val: int) -> None:
        """设置指定 hart 的定时器比较值."""
        if 0 <= hart_id < self._num_harts:
            self._mtimecmp[hart_id] = mask64(val)
            self._notify_state_change()

    def tick(self, cycles: int = 1) -> None:
        """推进全局时钟."""
        self._mtime = mask64(self._mtime + cycles)

    # ----------------------------------------------------------
    #  Device 接口 (内存映射寄存器访问)
    # ----------------------------------------------------------

    def read(self, offset: int, size: int) -> bytes:
        """读取 CLINT 内存映射寄存器.

        offset 是相对于 CLINT base_addr 的偏移.
        """
        # MSIP 区域: offset 0x0000..0x3FFC
        if offset < MTIMECMP_OFFSET:
            hart_id = offset // 4
            if 0 <= hart_id < self._num_harts:
                val = self._msip[hart_id] & 1
                return val.to_bytes(size, "little", signed=False)
            return b"\x00" * size

        # MTIMECMP 区域: offset 0x4000..0xBFF7
        if offset < MTIME_OFFSET:
            local_off = offset - MTIMECMP_OFFSET
            hart_id = local_off // 8
            if 0 <= hart_id < self._num_harts:
                return self._mtimecmp[hart_id].to_bytes(size, "little", signed=False)
            return b"\x00" * size

        # MTIME 寄存器: offset 0xBFF8
        if MTIME_OFFSET <= offset < MTIME_OFFSET + 8:
            return self._mtime.to_bytes(size, "little", signed=False)

        return b"\x00" * size

    def write(self, offset: int, data: bytes) -> None:
        """写入 CLINT 内存映射寄存器.

        offset 是相对于 CLINT base_addr 的偏移.
        """
        val = int.from_bytes(data, "little", signed=False)

        # MSIP 区域
        if offset < MTIMECMP_OFFSET:
            hart_id = offset // 4
            if 0 <= hart_id < self._num_harts:
                self._msip[hart_id] = val & 1
                self._notify_state_change()
            return

        # MTIMECMP 区域
        if offset < MTIME_OFFSET:
            local_off = offset - MTIMECMP_OFFSET
            hart_id = local_off // 8
            if 0 <= hart_id < self._num_harts:
                self._mtimecmp[hart_id] = mask64(val)
                self._notify_state_change()
            return

        # MTIME 寄存器
        if MTIME_OFFSET <= offset < MTIME_OFFSET + 8:
            self._mtime = mask64(val)

    # ----------------------------------------------------------
    #  属性
    # ----------------------------------------------------------

    @property
    def num_harts(self) -> int:
        return self._num_harts
