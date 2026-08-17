#!/usr/bin/env python3
"""通用 Hart 停滞检测与恢复 — 模拟器内置看门狗.

无需固件适配, 利用 emulator 自身的单轮加速执行轮转作为"时钟".
当某个 hart 在 M-mode 中停滞 (PC 不再推进) 且另一个 hart 在 WFI 空闲时,
自动重新触发 MSIP 给空闲 hart, 打破可能的跨核死锁.

不依赖任何固件特定地址 — 只使用 CLINT MSIP 寄存器 (标准硬件接口).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pyremu.core.hart import RiscvMode

if TYPE_CHECKING:
    from pyremu.emulator import Emulator


class HartStallWatchdog:
    """单轮加速执行级 hart 停滞检测器 — 在 ``Emulator._speedup_for_cmd_step`` 中每单轮加速执行调用."""

    def __init__(self, emu: Emulator, *, threshold: int = 5) -> None:
        self._emu = emu
        self._threshold = threshold
        # per-hart: (last_pc, stuck_count)
        self._track: dict[int, tuple[int, int]] = {}

    def check(self) -> bool:
        """每单轮加速执行返回后调用; 若执行了恢复操作则返回 True."""
        emu = self._emu
        harts = emu.harts
        clint = emu.clint

        if len(harts) < 2:
            return False

        # 检测: 一个 hart 在 S-mode WFI + 另一个在 M-mode 活躍
        wfi_hart: int | None = None
        mmode_hart: int | None = None

        for h in harts:
            if h._halted:
                continue
            if h.mode == RiscvMode.S and h._waiting:
                wfi_hart = h.id
            elif h.mode == RiscvMode.M and not h._waiting:
                mmode_hart = h.id

        if wfi_hart is None or mmode_hart is None:
            self._track.clear()
            return False

        # 跟踪 M-mode hart 的 PC 是否停滞
        mm_pc = harts[mmode_hart].pc
        prev_pc, stuck = self._track.get(mmode_hart, (mm_pc, 0))

        if mm_pc == prev_pc:
            stuck += 1
        else:
            stuck = 0

        self._track[mmode_hart] = (mm_pc, stuck)

        if stuck < self._threshold:
            return False

        # 恢复: 给 WFI hart 重新设置 MSIP, 打破可能的跨核死锁
        clint._msip[wfi_hart] = 1
        self._track.clear()
        return True
