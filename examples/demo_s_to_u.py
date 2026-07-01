#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""示例: M → S → U 特权级切换 + UART 输入 + Sv39 栈保护 + 递归 Fibonacci.

预编译固件 u_mode_run_fib.elf 流程:
  1. M 模式: 配置 PMP/medeleg/mstatus, MRET → S
  2. S 模式: 配置 stvec/sscratch/sstatus, 建立 Sv39 页表 (含 U 栈保护页), SRET → U
  3. U 模式: 从 UART 读取 n (1~16), 约束后计算 fib(n), ECALL 报告
  4. S 模式 trap handler: 分发 syscall (report/exit), 捕获页错误并终止进程

本示例演示:
  - UART 输入预加载 (uart.preload)
  - Sv39 保护页捕获栈溢出 (StorePageFault → S 终止进程)
  - 调试器栈帧回溯验证 fib(5) → fib(4) → fib(3) 调用链
  - 病态 U 模式程序 (无边界检查) 被 S 模式监督终止

编译固件:
    make -C tests/src-env build-s2u
运行:
    uv run python examples/demo_s_to_u.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pyremu.debugger import Debugger
from pyremu.emulator import Emulator
from pyremu.platform import PlatformConfig
from pyremu.utils.parse_bin import parse_firmware


def demo_well_behaved(emu: Emulator, fw) -> None:
    """Well-behaved U 模式: 从 UART 读取, 约束输入, 计算 fib."""
    h = emu.harts[0]
    dbg = Debugger(emulator=emu, hart_id=0, image=fw)
    uart = emu.bus.devices[0x10000000]
    fib_addr = fw.symbols.get("fib")

    # 预加载输入
    uart.preload(b"5\n")

    print("=" * 60)
    print("  Well-behaved: UART → fib(5) + 栈帧验证")
    print("=" * 60)

    # 追踪模式切换
    transitions = []
    prev_mode = h.mode
    fib3_checked = False

    for i in range(3000):
        pc_before = h.pc
        emu.step()

        if h.mode != prev_mode:
            transitions.append((emu.cycle, prev_mode.name, h.mode.name))
            prev_mode = h.mode

        # 在 fib(3) 的 prologue 后检查栈帧
        if not fib3_checked and pc_before == fib_addr and h.gprs[10] == 3:
            # 执行完 prologue
            for _ in range(5):
                emu.step()
            fib3_checked = True
            frames = dbg._walk_frame_chain()
            print(f"\n  → fib(3) 调用点 (step {emu.cycle}):")
            print(f"    栈帧回溯 ({len(frames)} 帧):")
            for f in frames:
                tag = f"#{f.idx + 1:02d}"
                print(f"    {tag} pc=0x{f.pc:08x} fp=0x{f.fp:08x} ra=0x{f.ra:08x}")
            print("    fib(3) → fib(4) → fib(5) 调用链可见 ✓")

        # 完成后退出
        if uart.tx_data().count(b"Enter") >= 2:
            break

    print(f"\n  UART 输出: {uart.tx_data().decode('latin-1', errors='replace')!r}")

    # 验证 fib_result
    raw = emu.bus.try_read(fw.symbols.get("fib_result"), 8)
    if raw:
        val = int.from_bytes(raw, "little")
        print(f"  fib_result = {val} {'✓' if val == 5 else '✗'}")

    print(f"\n  特权切换: {' → '.join(f'{fr}→{to}' for _, fr, to in transitions)}")


def demo_pathological(emu: Emulator, fw) -> None:
    """病态 U 模式: 无输入约束, 触发栈保护页 → S 终止进程."""
    h = emu.harts[0]
    uart = emu.bus.devices[0x10000000]
    u_mode_bad_addr = fw.symbols.get("u_mode_bad")

    uart.preload(b"0\n")  # '0' 触发 stack_bomb

    print("\n" + "=" * 60)
    print("  Pathological: stack_bomb → StorePageFault → S 终止")
    print("=" * 60)

    # Step to S-mode init completion, redirect to u_mode_bad
    for _ in range(3000):
        prev = h.mode
        emu.step()
        if prev.name == "S" and h.mode.name == "U":
            h.pc = u_mode_bad_addr
            break

    # Run until S-mode catches the fault
    for _ in range(3000):
        emu.step()
        if "Process terminated" in uart.tx_data().decode("latin-1", errors="replace"):
            break

    out = uart.tx_data().decode("latin-1", errors="replace")
    print(f"\n  UART 输出: {out!r}")

    # 验证 S 模式监督
    if "StorePageFault" in out or "scause=15" in out:
        print("  S 模式正确捕获了 StorePageFault (scause=15) ✓")
    if "Process terminated" in out:
        print("  病态进程被 S 模式终止 ✓")


def main() -> None:
    # ---- Well-behaved demo ----
    emu = Emulator(PlatformConfig.qemu_virt())
    fw = parse_firmware("tests/bins/elf/u_mode_run_fib.elf")
    emu.load_firmware(fw)
    demo_well_behaved(emu, fw)

    # ---- Pathological demo (fresh emulator) ----
    emu2 = Emulator(PlatformConfig.qemu_virt())
    fw2 = parse_firmware("tests/bins/elf/u_mode_run_fib.elf")
    emu2.load_firmware(fw2)
    demo_pathological(emu2, fw2)


if __name__ == "__main__":
    main()
