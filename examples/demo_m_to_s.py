#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""示例: M 模式切换 S 模式 — 模拟 OpenSBI → OS 移交.

预编译固件 m_mode_to_s_mode.elf 流程:
  1. M 模式: 初始化 UART, 配置 PMP, 设置 mstatus.MPP=S, mret 移交
  2. S 模式: 读取 mstatus 值, 经 UART 输出 "mstatus=0x..." 和 "Hello World!"
  3. WFI 进入低功耗等待

编译固件:
    make -C tests/src-env build-m2s
运行:
    uv run python examples/demo_m_to_s.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pyremu.emulator import Emulator
from pyremu.platform import PlatformConfig
from pyremu.utils.parse_bin import parse_firmware


def main() -> None:
    emu = Emulator(PlatformConfig.qemu_virt())
    fw = parse_firmware("tests/bins/elf/m_mode_to_s_mode.elf")
    emu.load_firmware(fw)

    h = emu.harts[0]

    # ---- 执行 M 模式初始化, 直到检测到 mret 切换 ----
    print("=== M 模式初始化阶段 ===")
    for i in range(50):
        if h._halted:
            break
        mode_before = h.mode.name
        emu.step()
        if h.mode.name != mode_before:
            print(f"  执行 {i + 1} 条指令后 PC = {h.pc:#018x}")
            break

    # ---- 已切换到 S 模式, 打印 UART 输出 ----
    print(f"\n=== MRET 切换: {mode_before} → {h.mode.name} ===")
    print(f"  mstatus = {h.mstatus_val:#018x}")

    # ---- 继续执行 S 模式代码, 等待 UART 输出 ----
    print("\n=== S 模式执行: UART 输出 ===  (UART TX → stdout)")
    emu.run(2000)

    print("\n  最终状态:")
    print(f"  Mode = {h.mode.name}")
    print(f"  Waiting (WFI) = {h._waiting}")
    print(f"  Cycles = {emu.cycle}")
    print(f"  Total Instructions = {emu.total_instructions}")


if __name__ == "__main__":
    main()
