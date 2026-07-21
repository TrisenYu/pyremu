#!/usr/bin/env python3
"""RISC-V 兼容看门狗虚拟设备.

MMIO 寄存器布局 (每个 hart 独立, stride=0x10):
  Offset 0x00: WDOG_CTRL    — 控制寄存器 (bit 0: enable, bit 1: reset on kick)
  Offset 0x04: WDOG_TIMEOUT — 超时周期 (单位: 批次)
  Offset 0x08: WDOG_COUNT   — 当前计数值 (只读)
  Offset 0x0C: WDOG_KICK    — 踢狗寄存器 (写任意值重置计数)

当任一 hart 的计数器归零时, 设备自动对全体 WFI hart 注入 MSIP=1,
利用现有中断路径打破可能的跨核死锁.  同时也拉高中断线供 PLIC/APLIC 使用
(暂未实现 — 当前仅直接操作 CLINT MSIP).

DTB 绑定:
  compatible = "pyremu,hart-watchdog-1.0"
  reg = <base size>
  pyremu,num-harts = <N>
"""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING

from pyremu.memory.bus import Device

if TYPE_CHECKING:
    from pyremu.emulator import Emulator

# 寄存器偏移 & 位定义
_WDOG_CTRL = 0x00
_WDOG_TIMEOUT = 0x04
_WDOG_COUNT = 0x08
_WDOG_KICK = 0x0C
_WDOG_STRIDE = 0x10  # 每个 hart 寄存器块间隔

_CTRL_ENABLE = 1 << 0


class HartWatchdog(Device):
    """多 hart 看门狗 — 每个 hart 独立的倒计时 + 全局 MSIP 注入."""

    def __init__(
        self,
        emu: Emulator,
        base: int = 0x1000_4000,
        num_harts: int = 2,
        timeout: int = 500,  # 默认 500 批次超时
    ) -> None:
        self.base_addr = base
        self._num_harts = num_harts
        self._timeout = timeout
        self._emu = emu

        # 每 hart: [ctrl, timeout_val, count, _reserved]
        self._ctrl: list[int] = [0] * num_harts
        self._timeout_val: list[int] = [timeout] * num_harts
        self._count: list[int] = [timeout] * num_harts

        # 全部 hart 启用, count = timeout
        for i in range(num_harts):
            self._ctrl[i] = _CTRL_ENABLE
            self._count[i] = timeout

    # ---- Device interface ----

    @property
    def size(self) -> int:
        return self._num_harts * _WDOG_STRIDE

    def read(self, offset: int, size: int) -> bytes:
        hart = offset // _WDOG_STRIDE
        reg = offset % _WDOG_STRIDE
        if hart >= self._num_harts:
            return b"\x00" * size

        if reg == _WDOG_CTRL:
            return struct.pack("<I", self._ctrl[hart])
        if reg == _WDOG_TIMEOUT:
            return struct.pack("<I", self._timeout_val[hart])
        if reg == _WDOG_COUNT:
            return struct.pack("<I", self._count[hart])
        if reg == _WDOG_KICK:
            return b"\x00" * size
        return b"\x00" * size

    def write(self, offset: int, data: bytes) -> None:
        hart = offset // _WDOG_STRIDE
        reg = offset % _WDOG_STRIDE
        if hart >= self._num_harts:
            return

        val = int.from_bytes(data, "little", signed=False)
        if reg == _WDOG_CTRL:
            self._ctrl[hart] = val
        elif reg == _WDOG_TIMEOUT:
            self._timeout_val[hart] = val
        elif reg == _WDOG_KICK:
            self._count[hart] = self._timeout_val[hart]
        # WDOG_COUNT is read-only; writes ignored

    # ---- Emulator integration ----

    def kick_all(self) -> None:
        """所有 hart 重置计数器 — 模拟器每批次调用."""
        for i in range(self._num_harts):
            if self._ctrl[i] & _CTRL_ENABLE:
                self._count[i] = self._timeout_val[i]

    def kick(self, hart_id: int) -> None:
        """单个 hart 重置计数器."""
        if hart_id < self._num_harts and self._ctrl[hart_id] & _CTRL_ENABLE:
            self._count[hart_id] = self._timeout_val[hart_id]

    def tick(self) -> bool:
        """推进所有 hart 的计数器; 若任一归零则触发恢复操作.

        Returns:
            True 若执行了恢复操作.
        """
        fired = False
        for i in range(self._num_harts):
            if self._ctrl[i] & _CTRL_ENABLE == 0:
                continue
            if self._count[i] > 0:
                self._count[i] -= 1
            if self._count[i] == 0:
                # 计数器归零 — 触发恢复
                self._do_recover()
                self._count[i] = self._timeout_val[i]
                fired = True
        return fired

    def _do_recover(self) -> None:
        """恢复操作: 对全体 WFI hart 注入 MSIP.

        利用现有的 CLINT MSIP -> WFI 唤醒 -> M-mode trap -> sbi_ipi_process
        路径, 让空闲 hart 重新处理可能排队的 IPI 事件.
        不访问固件特定地址 — 仅使用标准 CLINT 接口.
        """
        clint = self._emu.clint
        for h in self._emu.harts:
            if h._waiting and not h._halted:
                hid = h.id
                if hid < len(clint._msip):
                    clint._msip[hid] = 1
