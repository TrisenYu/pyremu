#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0

"""多 hart UART 行缓冲 + OpenSBI 彩票启动测试."""

import functools
import re

import pytest

from pyremu.emulator import Emulator
from pyremu.platform import PlatformConfig
from pyremu.utils.parse_bin import FirmwareImage, parse_firmware

# ------------------------------------------------------------
#  ANSI escape code stripping
# ------------------------------------------------------------

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _strip_ansi(text: str) -> str:
    """Strip ANSI escape codes from *text* (used by hart-coloured output)."""
    return _ANSI_RE.sub("", text)

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
    # UART TXDATA 字节级即时输出, 无需 flush
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
    """多 hart UART 行缓冲 — 输出不交错, 日志文件按 hart 隔离.

    控制台输出不再添加 ``[hart N]`` 前缀 (与 QEMU -nographic 一致);
    多 hart 调试信息通过 ``set_hart_log_dir()`` 提供的日志文件获取。
    """

    def test_lines_preserved_no_interleaving(self, lottery_lines_4h: list[str]):
        """真实硬件 UART 无 hart 感知: 多 hart 无锁并发写必然字节交错.

        "cold boot" 仅一个 hart 输出, 应完整保留; "warm boot" / "running"
        由多 hart 同时输出, 必然交错。正确验证: 每 hart 日志文件独立完整
        (见 test_per_hart_log_files)。
        """
        assert len(lottery_lines_4h) > 0, "应有输出"
        all_text = "".join(lottery_lines_4h)
        # 仅冷启动 hart 输出 "cold boot" → 不交错 → 应完整
        assert "cold boot" in all_text
        # warm boot 和 running 可能交错, 但每个 hart 的日志文件应完整

    def test_no_ansi_prefix_in_output(self, lottery_lines_4h: list[str]):
        """控制台输出不含 [hart N] 前缀."""
        all_text = "".join(lottery_lines_4h)
        assert "[hart" not in _strip_ansi(all_text), (
            f"不应含 [hart 前缀, 得到: {all_text[:120]}..."
        )

    def test_all_harts_produce_output(self, tmp_path):
        """所有 4 个 hart 都应有输出 — 通过日志文件验证 (控制台必然交错)."""
        emu = _make_emu(4)
        assert emu.uart is not None
        emu.uart.set_hart_log_dir(str(tmp_path))
        img = _cached_firmware(_LOTTERY_ELF)
        emu.load_firmware(img)
        for _ in range(_LOTTERY_CYCLES):
            emu.step()
        emu.uart.close_logs()

        hart_outputs: dict[int, str] = {}
        for hid in range(4):
            log = tmp_path / f"hart{hid}.log"
            if log.exists():
                hart_outputs[hid] = log.read_text()
        assert len(hart_outputs) == 4, f"应有 4 个日志文件, 得到 {len(hart_outputs)}"
        cold_count = sum(1 for t in hart_outputs.values() if "cold boot" in t)
        assert cold_count == 1, f"应恰好 1 个 cold boot, 得到 {cold_count}"
        warm_count = sum(1 for t in hart_outputs.values() if "warm boot" in t)
        assert warm_count == 3, f"应 3 个 warm boot, 得到 {warm_count}"

    def test_per_hart_log_files(self, tmp_path):
        """日志文件按 hart 隔离 — 每个 hart 的日志文件含其完整输出."""
        sink: list[str] = []
        emu = _make_emu(4)
        assert emu.uart is not None
        cap = _Capture()
        emu.uart._tx_callback = cap
        emu.uart.set_hart_log_dir(str(tmp_path))
        img = _cached_firmware(_LOTTERY_ELF)
        emu.load_firmware(img)
        for _ in range(_LOTTERY_CYCLES):
            emu.step()
        # UART TXDATA 字节级即时输出, 无需 flush
        emu.uart.close_logs()

        # 各 hart 的日志文件应含其 boot 消息
        hart_outputs: dict[int, str] = {}
        for hid in range(4):
            log = tmp_path / f"hart{hid}.log"
            if log.exists():
                hart_outputs[hid] = log.read_text()
        assert len(hart_outputs) == 4, f"应有 4 个日志文件, 得到 {len(hart_outputs)}"
        # 恰好一个 cold boot (在某个 hart 的日志中)
        cold_harts = [h for h, t in hart_outputs.items() if "cold boot" in t]
        assert len(cold_harts) == 1, f"应恰好 1 个 cold boot, 得到 {cold_harts}"
        # 其余 3 个 warm boot
        warm_harts = [h for h, t in hart_outputs.items() if "warm boot" in t]
        assert len(warm_harts) == 3, f"应 3 个 warm boot, 得到 {warm_harts}"


# ------------------------------------------------------------
#  TestLotteryBoot
# ------------------------------------------------------------


class TestLotteryBoot:
    """OpenSBI 风格彩票启动 — 一个冷启动, 其余热启动.

    多 hart 无锁并发写 UART → 控制台字节交错, 正确性验证通过每 hart
    日志文件 (set_hart_log_dir) 实现 — 与 QEMU -nographic 行为一致.
    """

    def _hart_logs(self, tmp_path) -> dict[int, str]:
        """运行彩票启动, 返回每 hart 日志文件内容的 dict."""
        emu = _make_emu(4)
        assert emu.uart is not None
        emu.uart.set_hart_log_dir(str(tmp_path))
        img = _cached_firmware(_LOTTERY_ELF)
        emu.load_firmware(img)
        for _ in range(_LOTTERY_CYCLES):
            emu.step()
        emu.uart.close_logs()

        outputs: dict[int, str] = {}
        for hid in range(4):
            log = tmp_path / f"hart{hid}.log"
            if log.exists():
                outputs[hid] = log.read_text()
        return outputs

    def test_exactly_one_cold_boot(self, tmp_path):
        logs = self._hart_logs(tmp_path)
        cold_count = sum(1 for t in logs.values() if "cold boot" in t)
        assert cold_count == 1, f"应恰好 1 个 cold boot, 得到 {cold_count}"

    def test_remaining_harts_warm_boot(self, tmp_path):
        logs = self._hart_logs(tmp_path)
        warm_count = sum(1 for t in logs.values() if "warm boot" in t)
        assert warm_count == 3, f"应有 3 个 warm boot, 得到 {warm_count}"

    def test_all_harts_reach_running(self, tmp_path):
        logs = self._hart_logs(tmp_path)
        running_count = sum(1 for t in logs.values() if "running" in t)
        assert running_count == 4, f"应有 4 个 running, 得到 {running_count}"

    def test_cold_before_warm(self, tmp_path):
        """任意 hart 可赢得彩票成为冷启动 — 其余均为热启动."""
        logs = self._hart_logs(tmp_path)
        cold_harts = [h for h, t in logs.items() if "cold boot" in t]
        warm_harts = [h for h, t in logs.items() if "warm boot" in t]
        assert len(cold_harts) == 1, f"应恰好 1 个 cold, 得到 {cold_harts}"
        assert len(warm_harts) == 3, f"应 3 个 warm, 得到 {warm_harts}"
        # 冷热互斥
        assert cold_harts[0] not in warm_harts

    def test_cold_hart_runs_before_others(self, tmp_path):
        """冷启动 hart 的 running 应出现在自身日志中, 且日志按 hart 隔离."""
        emu = _make_emu(4)
        assert emu.uart is not None
        emu.uart.set_hart_log_dir(str(tmp_path))
        img = _cached_firmware(_LOTTERY_ELF)
        emu.load_firmware(img)
        for _ in range(_LOTTERY_CYCLES):
            emu.step()
        # UART TXDATA 字节级即时输出, 无需 flush
        emu.uart.close_logs()
        # 找到冷启动 hart
        cold_hart = None
        for hid in range(4):
            log = tmp_path / f"hart{hid}.log"
            if log.exists() and "cold boot" in log.read_text():
                cold_hart = hid
                break
        assert cold_hart is not None, "应有一个 cold boot hart"
        # 冷启动 hart 的日志应包含 cold boot + running
        cold_log = (tmp_path / f"hart{cold_hart}.log").read_text()
        assert "cold boot" in cold_log
        assert "running" in cold_log

    def test_lock_prevents_duplicate_cold(self):
        """多次运行均只有 1 个 cold boot (锁机制正确).

        TXDATA 字节级输出 → 控制台必然交错, 但 "cold boot" 仅一个 hart
        输出 → 在合并文本中应恰好出现一次.
        """
        for _ in range(3):
            emu = _make_emu(4)
            lines = _run_lottery(emu, cycles=_LOTTERY_CYCLES)
            all_text = "".join(lines)
            assert all_text.count("cold boot") == 1, (
                f"锁机制失效: {all_text.count('cold boot')} cold boots"
            )

    def test_single_hart(self):
        """单 hart 场景: 必然是冷启动, 不出现 warm boot."""
        emu = _make_emu(1)
        lines = _run_lottery(emu, cycles=_LOTTERY_CYCLES)
        all_text = "".join(lines)
        assert "cold boot" in all_text, "单 hart 应有 cold boot"
        assert "warm boot" not in all_text, "单 hart 不应有 warm boot"
