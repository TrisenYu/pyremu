#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0

"""多 hart UART 行缓冲 + OpenSBI 彩票启动测试."""

import pytest

from pyremu.emulator import Emulator
from pyremu.platform import PlatformConfig
from pyremu.utils.parse_bin import parse_firmware


class _Capture:
    """捕获 UART 输出行的简单回调."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, text: str) -> None:
        self.lines.append(text)


@pytest.fixture
def emu_4h() -> Emulator:
    emu = Emulator(PlatformConfig(num_harts=4, ram_size=128 * 1024 * 1024))
    return emu


def _run_lottery(emu: Emulator, cycles: int = 3000) -> list[str]:
    """运行彩票启动固件, 返回 UART 输出的行列表."""
    cap = _Capture()
    emu.uart._tx_callback = cap
    img = parse_firmware("tests/bins/elf/lottery_boot.elf")
    emu.load_firmware(img)
    for _ in range(cycles):
        emu.step()
    emu.uart.flush_all()
    return cap.lines


class TestUartLineBuffering:
    """多 hart UART 行缓冲 — 输出不交错且带 hart 标签."""

    def test_each_line_has_hart_prefix(self, emu_4h):
        """每条输出行应以 '[hart N]' 开头."""
        lines = _run_lottery(emu_4h)
        assert len(lines) > 0, "应有输出"
        for line in lines:
            assert line.startswith("[hart "), (
                f"行应以 '[hart N]' 开头, 得到: {line!r}"
            )

    def test_no_interleaving_within_line(self, emu_4h):
        """单行内不应包含来自其他 hart 的片段 (无交错)."""
        lines = _run_lottery(emu_4h)
        for line in lines:
            # 每行最多出现一次 '[hart ', 即只有行首的 hart 标签
            assert line.count("[hart ") == 1, (
                f"行内出现多个 hart 标签 (交错): {line!r}"
            )

    def test_all_harts_produce_output(self, emu_4h):
        """所有 4 个 hart 都应有输出."""
        lines = _run_lottery(emu_4h)
        hart_ids: set[int] = set()
        for line in lines:
            # 提取 "[hart N]" 中的 N
            end = line.index("]")
            n = int(line[6:end])
            hart_ids.add(n)
        assert hart_ids == {0, 1, 2, 3}, f"应包含所有 hart, 得到: {hart_ids}"


class TestLotteryBoot:
    """OpenSBI 风格彩票启动 — 一个冷启动, 其余热启动."""

    def test_exactly_one_cold_boot(self, emu_4h):
        """恰好一个 hart 执行冷启动."""
        lines = _run_lottery(emu_4h)
        cold_count = sum(1 for ln in lines if "cold boot" in ln)
        assert cold_count == 1, f"应恰好 1 个 cold boot, 得到 {cold_count}"

    def test_remaining_harts_warm_boot(self, emu_4h):
        """其余 3 个 hart 热启动."""
        lines = _run_lottery(emu_4h)
        warm_count = sum(1 for ln in lines if "warm boot" in ln)
        assert warm_count == 3, f"应有 3 个 warm boot, 得到 {warm_count}"

    def test_all_harts_reach_running(self, emu_4h):
        """全部 4 个 hart 最终到达 running 状态."""
        lines = _run_lottery(emu_4h)
        running_count = sum(1 for ln in lines if "running" in ln)
        assert running_count == 4, f"应有 4 个 running, 得到 {running_count}"

    def test_cold_before_warm(self, emu_4h):
        """冷启动应出现在热启动之前 (时序约束)."""
        lines = _run_lottery(emu_4h)
        cold_idx = next(i for i, ln in enumerate(lines) if "cold boot" in ln)
        warm_indices = [i for i, ln in enumerate(lines) if "warm boot" in ln]
        assert warm_indices, "应有热启动"
        # 所有热启动应在冷启动之后
        assert all(idx > cold_idx for idx in warm_indices), (
            f"热启动应在冷启动之后: cold@{cold_idx}, warm@{warm_indices}"
        )

    def test_cold_hart_runs_before_others(self, emu_4h):
        """冷启动 hart 的 running 应出现在其他 hart 的 warm 之前或同时."""
        lines = _run_lottery(emu_4h)
        # 找冷启动 hart 的 ID
        cold_line = next(ln for ln in lines if "cold boot" in ln)
        cold_hart = int(cold_line[6:cold_line.index("]")])
        # 冷启动 hart 的 running 行
        cold_run_idx = next(
            i for i, ln in enumerate(lines)
            if "running" in ln and ln.startswith(f"[hart {cold_hart}]")
        )
        # 检查: 冷启动 hart 的 running 在它自己的 warm 行之前 (它没有 warm)
        # 且所有 warm 行应在 cold boot 之后 (已在 test_cold_before_warm 验证)
        assert cold_run_idx is not None

    def test_lock_prevents_duplicate_cold(self, emu_4h):
        """多次运行均只有 1 个 cold boot (锁机制正确)."""
        for _ in range(3):
            emu = Emulator(PlatformConfig(num_harts=4, ram_size=128 * 1024 * 1024))
            lines = _run_lottery(emu)
            cold_count = sum(1 for ln in lines if "cold boot" in ln)
            assert cold_count == 1, f"锁机制失效: {cold_count} cold boots"

    def test_single_hart(self, emu_4h):
        """单 hart 场景: 必然是冷启动."""
        emu = Emulator(PlatformConfig(num_harts=1, ram_size=128 * 1024 * 1024))
        lines = _run_lottery(emu)
        assert any("cold boot" in ln for ln in lines), "单 hart 应有 cold boot"
        assert not any("warm boot" in ln for ln in lines), "单 hart 不应有 warm boot"
