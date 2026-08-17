#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""示例: 纯 Python API 使用 Emulator 运行 RISC-V 固件.

验证:
    uv run python examples/demo_emulator.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pyremu.emulator import Emulator
from pyremu.platform import PlatformConfig
from pyremu.configs_aux import elf_dir, path_join
from pyremu.utils.parse_bin import parse_firmware

# 1. 创建模拟器实例 — 默认 qemu_virt 平台
emu = Emulator(PlatformConfig.qemu_virt())

# 2. 解析并加载 ELF 固件
fw = parse_firmware(path_join(elf_dir(), "m_mode_to_s_mode.elf"))
emu.load_firmware(fw)

# 3. 执行指定周期数
emu.run(2000)

# 4. 检查 hart 状态
h = emu.harts[0]
print(f"PC        = {h.pc:#018x}")
print(f"Mode      = {h.mode.name}")
print(f"mstatus   = {h.mstatus_val:#018x}")
print(f"mepc      = {h.mepc_val:#018x}")
print(f"mcause    = {h.mcause_val:#018x}")
print(f"Waiting   = {h._waiting}")
print(f"Halted    = {h._halted}")
print(f"Cycles    = {emu.cycle}")
print(f"Total Instrs = {emu.total_instructions}")

# 5. 读取并打印物理内存 (十六进制 dump)
print("\n--- 内存 dump (0x80002000, 64 字节) ---")
print(emu.mem_hexdump(0x80002000, 64))

# 6. 导出全部 GPR
print("\n--- Hart 0 GPRs ---")
regs = emu.dump_hart_regs(0)
for name, val in regs["gprs"].items():
    print(f"  {name} = {val}")
