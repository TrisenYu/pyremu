#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""示例: 以编程方式使用 Debugger 进行非交互式指令级调试.

验证:
    uv run python examples/demo_debugger.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pyremu.debugger import Debugger
from pyremu.emulator import Emulator
from pyremu.platform import PlatformConfig
from pyremu.utils.parse_bin import parse_firmware

# 1. 创建模拟器并加载固件
emu = Emulator(PlatformConfig.qemu_virt())
fw = parse_firmware("tests/bins/elf/m_mode_to_s_mode.elf")
emu.load_firmware(fw)

# 2. 创建调试器 (默认 hart=0)
dbg = Debugger(emu, image=fw)

# 3. 打印入口点反汇编
print("=== 入口点反汇编 ===")
# 直接调用 cmd_disasm — 参数类型与 REPL 中一致
dbg.cmd_disasm(hex(emu.harts[0].pc), "64")
print()

# 4. 单步执行并观察模式切换
print("=== 逐条执行, 观察 M→S 切换 ===")
h = dbg.hart
# 逐步执行直到检测到模式切换 (mret)
# 使用 emu.step 而非 dbg.cmd_step, 避免每条都打印 PC + 反汇编
print("--- 执行 M 模式初始化 ... ---")
for _ in range(50):
    if h._halted or h._waiting:
        break
    mode_before = h.mode.name
    emu.step()
    if h.mode.name != mode_before:
        print(f">>> 特权级切换: {mode_before} → {h.mode.name} <<<")
        break

# 用调试器确认切换后的状态
print("\n--- S 模式入口 ---")
dbg.cmd_pc()
print("(继续执行 8 条 S 模式指令, 其中 UART 输出通过 stdout 可见)")
for _ in range(8):
    emu.step()

print()

# 5. 检查寄存器和栈内存
print("=== Hart 0 寄存器 ===")
dbg.cmd_regs()

print("\n=== mstatus 值 ===")
dbg.cmd_mstatus()

print("\n=== 模式 ===")
dbg.cmd_mode()
