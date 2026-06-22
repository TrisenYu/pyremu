#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0

"""多 hart UART 行缓冲 + OpenSBI 彩票启动测试."""

import functools

import pytest

from pyremu.emulator import Emulator
from pyremu.platform import PlatformConfig
from pyremu.utils.parse_bin import FirmwareImage, parse_firmware


# ------------------------------------------------------------
#  模块级固件缓存 — 避免每个测试重复解析 ELF
# ------------------------------------------------------------

@functools.cache
def _cached_firmware(path: str) -> FirmwareImage | None:
    return parse_firmware(path)


_LOTTERY_ELF = "tests/bins/elf/lottery_boot.elf"
# 固件在 1000 周期内即可完成彩票启动并输出全部关键行
_LOTTERY_CYCLES = 1000


# ------------------------------------------------------------
#  输出捕获
# ------------------------------------------------------------

class _Capture:
    """捕获 UART 输出行的简单回调."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, text: str) -> None:
        self.lines.append(text)


# ------------------------------------------------------------
#  辅助
# ------------------------------------------------------------

def _run_lottery(emu: Emulator, cycles: int = _LOTTERY_CYCLES) -> list[str]:
    """运行彩票启动固件, 返回 UART 输出的行列表."""
    assert emu.uart is not None
    cap = _Capture()
    emu.uart._tx_callback = cap
    img = _cached_firmware(_LOTTERY_ELF)
    emu.load_firmware(img)
    for _ in range(cycles):
        emu.step()
    emu.uart.flush_all()
    return cap.lines


def _make_emu(num_harts: int = 4) -> Emulator:
    """创建轻量多 hart 模拟器."""
    return Emulator(PlatformConfig(num_harts=num_harts, ram_size=8 * 1024 * 1024))


# ------------------------------------------------------------
#  Fixtures
# ------------------------------------------------------------

@pytest.fixture(scope="class")
def emu_4h() -> Emulator:
    """4 hart 模拟器 — 类级别复用."""
    return _make_emu(4)


@pytest.fixture(scope="class")
def lottery_lines_4h(emu_4h: Emulator) -> list[str]:
    """运行一次彩票启动并缓存全部输出行 (类级别)."""
    return _run_lottery(emu_4h)


# ------------------------------------------------------------
#  TestUartLineBuffering
# ------------------------------------------------------------

class TestUartLineBuffering:
    """多 hart UART 行缓冲 — 输出不交错且带 hart 标签."""

    def test_each_line_has_hart_prefix(self, lottery_lines_4h: list[str]):
        """每条输出行应以 '[hart N]' 开头."""
        assert len(lottery_lines_4h) > 0, "应有输出"
        for line in lottery_lines_4h:
            assert line.startswith("[hart "), (
                f"行应以 '[hart N]' 开头, 得到: {line!r}"
            )

    def test_no_interleaving_within_line(self, lottery_lines_4h: list[str]):
        """单行内不应包含来自其他 hart 的片段 (无交错)."""
        for line in lottery_lines_4h:
            assert line.count("[hart ") == 1, (
                f"行内出现多个 hart 标签 (交错): {line!r}"
            )

    def test_all_harts_produce_output(self, lottery_lines_4h: list[str]):
        """所有 4 个 hart 都应有输出."""
        hart_ids: set[int] = set()
        for line in lottery_lines_4h:
            end = line.index("]")
            n = int(line[6:end])
            hart_ids.add(n)
        assert hart_ids == {0, 1, 2, 3}, f"应包含所有 hart, 得到: {hart_ids}"


# ------------------------------------------------------------
#  TestLotteryBoot
# ------------------------------------------------------------

class TestLotteryBoot:
    """OpenSBI 风格彩票启动 — 一个冷启动, 其余热启动."""

    def test_exactly_one_cold_boot(self, lottery_lines_4h: list[str]):
        """恰好一个 hart 执行冷启动."""
        cold_count = sum(1 for ln in lottery_lines_4h if "cold boot" in ln)
        assert cold_count == 1, f"应恰好 1 个 cold boot, 得到 {cold_count}"

    def test_remaining_harts_warm_boot(self, lottery_lines_4h: list[str]):
        """其余 3 个 hart 热启动."""
        warm_count = sum(1 for ln in lottery_lines_4h if "warm boot" in ln)
        assert warm_count == 3, f"应有 3 个 warm boot, 得到 {warm_count}"

    def test_all_harts_reach_running(self, lottery_lines_4h: list[str]):
        """全部 4 个 hart 最终到达 running 状态."""
        running_count = sum(1 for ln in lottery_lines_4h if "running" in ln)
        assert running_count == 4, f"应有 4 个 running, 得到 {running_count}"

    def test_cold_before_warm(self, lottery_lines_4h: list[str]):
        """冷启动应出现在热启动之前 (时序约束)."""
        cold_idx = next(i for i, ln in enumerate(lottery_lines_4h) if "cold boot" in ln)
        warm_indices = [i for i, ln in enumerate(lottery_lines_4h) if "warm boot" in ln]
        assert warm_indices, "应有热启动"
        assert all(idx > cold_idx for idx in warm_indices), (
            f"热启动应在冷启动之后: cold@{cold_idx}, warm@{warm_indices}"
        )

    def test_cold_hart_runs_before_others(self, lottery_lines_4h: list[str]):
        """冷启动 hart 的 running 应出现在其他 hart 的 warm 之前或同时."""
        cold_line = next(ln for ln in lottery_lines_4h if "cold boot" in ln)
        cold_hart = int(cold_line[6:cold_line.index("]")])
        cold_run_idx = next(
            i for i, ln in enumerate(lottery_lines_4h)
            if "running" in ln and ln.startswith(f"[hart {cold_hart}]")
        )
        assert cold_run_idx is not None

    def test_lock_prevents_duplicate_cold(self):
        """多次运行均只有 1 个 cold boot (锁机制正确)."""
        for _ in range(3):
            emu = _make_emu(4)
            lines = _run_lottery(emu, cycles=_LOTTERY_CYCLES)
            cold_count = sum(1 for ln in lines if "cold boot" in ln)
            assert cold_count == 1, f"锁机制失效: {cold_count} cold boots"

    def test_single_hart(self):
        """单 hart 场景: 必然是冷启动."""
        emu = _make_emu(1)
        lines = _run_lottery(emu, cycles=_LOTTERY_CYCLES)
        assert any("cold boot" in ln for ln in lines), "单 hart 应有 cold boot"
        assert not any("warm boot" in ln for ln in lines), "单 hart 不应有 warm boot"
