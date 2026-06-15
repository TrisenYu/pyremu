#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""短时间片抢占测试 — 验证抢占下进程交替执行与正确性."""

from pathlib import Path

import pytest

from pyremu.emulator import Emulator
from pyremu.platform import PlatformConfig
from pyremu.utils.parse_bin import parse_firmware
from tests.loader import MultiProgramLoader

BIN_DIR = Path(__file__).resolve().parent / "bins" / "elf"
KERNEL_ELF = BIN_DIR / "kernel.elf"


def _make_emu(*procs: str):
    if not KERNEL_ELF.exists():
        pytest.skip(f"{KERNEL_ELF} 不存在, 先运行 make -C tests/src-env build-multi")

    kernel = parse_firmware(str(KERNEL_ELF))
    emu = Emulator(PlatformConfig.qemu_virt())
    emu.load_firmware(kernel)
    loader = MultiProgramLoader(emu, kernel)
    for p in procs:
        loader.add_process(p)
    return emu, loader


class TestPreemptInterleave:
    """A+B 交替输出: 验证抢占下进程轮流执行."""

    def test_interleaved(self):
        _, loader = _make_emu("u_prog_a_entry", "u_prog_b_entry")
        out = loader.run(cycles=3_000_000)
        seq = "".join(c for c in out if c in "AB")
        transitions = sum(1 for i in range(len(seq) - 1) if seq[i] != seq[i + 1])
        assert transitions > 1, f"未交错, transitions={transitions}"

    def test_both_produce_output(self):
        _, loader = _make_emu("u_prog_a_entry", "u_prog_b_entry")
        out = loader.run(cycles=5_000_000)
        assert out.count("A") >= 50, f"A count too low: {out.count('A')}"
        assert out.count("B") >= 50, f"B count too low: {out.count('B')}"


class TestPreemptCorrectness:
    """单独运行 fib/nqueen — 验证抢占下计算正确."""

    @pytest.mark.parametrize("n,expected", [(2, 1), (5, 5), (7, 13)])
    def test_fib_alone(self, n, expected):
        """fib 单独运行 (无其他进程抢占)."""
        _, loader = _make_emu("u_fib_entry")
        out = loader.run(cycles=300_000, uart_input=f"{n}\n".encode())
        assert f"fib({n})={expected}" in out

    def test_nqueen_alone(self):
        """nqueen 单独运行."""
        _, loader = _make_emu("u_nqueen_entry")
        out = loader.run(cycles=500_000)
        for n, v in [(1, 1), (2, 0), (3, 0), (4, 2)]:
            assert f"nq({n})={v}" in out, f"nq({n})={v} 未找到"

    def test_fib_nqueen_together(self):
        """fib + nqueen 同时运行 — 基本正确性."""
        _, loader = _make_emu("u_nqueen_entry", "u_fib_entry")
        out = loader.run(cycles=500_000, uart_input=b"7\n")
        assert "fib(7)=13" in out
        # nqueen 输出可能因交错而难以精确匹配, 但 exit 应出现
        assert "exit: code=0" in out
