#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多程序调度演示: fib + n-queen 在 S 模式调度下交替执行.

内核 (kernel.s) 提供 round-robin 调度器和 CLINT 定时器中断.
两个 U 模式程序通过 ECALL 进行 I/O.

演示:
  - 两进程交替输出 (timer 抢占)
  - S 模式捕获页错误终止进程
  - Python Loader API 简化多程序加载

编译:
    make -C tests/src-env build-multi
运行:
    uv run python examples/demo_multi.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pyremu.emulator import Emulator
from tests.loader import MultiProgramLoader
from pyremu.platform import PlatformConfig
from pyremu.utils.parse_bin import parse_firmware


def main() -> None:
    emu = Emulator(PlatformConfig.qemu_virt())
    kernel = parse_firmware("tests/bins/elf/kernel.elf")
    emu.load_firmware(kernel)

    loader = MultiProgramLoader(emu, kernel)
    loader.add_process("u_nqueen_entry")  # pid 0 — n-queen first
    loader.add_process("u_fib_entry")     # pid 1 — fib second

    print("=" * 55)
    print("  Multi-Program Scheduler Demo")
    print("  fib + n-queen under S-mode round-robin")
    print("=" * 55)

    # Preload fib with input "5\n"
    uart_input = b"5\n"
    output = loader.run(cycles=200000, uart_input=uart_input)

    print(output)

    # Verify both programs produced output
    checks = [
        ("fib(5)=5", "fib produced correct result"),
        ("nq(1)=1", "n-queen n=1 correct"),
        ("nq(2)=0", "n-queen n=2 correct"),
        ("nq(3)=0", "n-queen n=3 correct"),
        ("nq(4)=2", "n-queen n=4 correct"),
    ]
    for pattern, desc in checks:
        status = "✓" if pattern in output else "✗"
        print(f"  {status} {desc}: '{pattern}'")


if __name__ == "__main__":
    main()
