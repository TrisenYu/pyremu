#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""多程序加载器 — 将多个 U 模式程序注入内核进程表.

内核 (kernel.s) 在 BSS 中定义了 process_table (PCB 数组) 和
num_processes. 本模块通过 FirmwareImage 符号表解析这些地址,
在固件加载后填充 PCB, 使内核调度器可以启动多个用户进程.

PCB 布局 (每进程 40 字节, 与 kernel.s 中 PCB_*_OFF 常量一致):
    offset 0:  state        (4B, uint32) — 0=EMPTY, 1=READY, 2=RUNNING
    offset 4:  entry_pc     (8B, uint64)
    offset 12: stack_top    (8B, uint64)
    offset 20: saved_sepc   (8B, uint64)
    offset 28: saved_sp     (8B, uint64)
    offset 36: exit_code    (4B, uint32)

用法:
    from pyremu.emulator import Emulator
    from pyremu.platform import PlatformConfig
    from pyremu.utils.parse_bin import parse_firmware
    from pyremu.loader import MultiProgramLoader
    from pyremu.configs_aux import test_elf_dir

    emu = Emulator(PlatformConfig.qemu_virt())
    kernel = parse_firmware(test_elf_dir("kernel.elf"))
    emu.load_firmware(kernel)

    loader = MultiProgramLoader(emu, kernel)
    loader.add_process("u_fib_entry")
    loader.add_process("u_nqueen_entry")
    loader.run()
"""

from __future__ import annotations

import struct

from pyremu.emulator import Emulator
from pyremu.peripheral.uart import UART
from pyremu.utils.parse_bin import FirmwareImage

# PCB 字段偏移 (与 kernel.s 保持一致 — 含对齐 padding)
PCB_STATE_OFF = 0
PCB_ENTRY_OFF = 8  # +4B padding for 8B-alignment
PCB_STACK_OFF = 16
PCB_SEPC_OFF = 24
PCB_SP_OFF = 32
PCB_EXIT_OFF = 40
PCB_GPR_OFF = 48  # GPR save area (x1-x31)
PCB_SIZE = 296  # 48 + 31*8 (GPR save area)

# 进程状态
PS_EMPTY = 0
PS_READY = 1
PS_RUNNING = 2

# 每进程 U 栈大小 (页数, 含 1 保护页)
STACK_PAGES = 2  # 1 栈页 + 1 保护页
PAGE_SIZE = 4096
STACK_BASE_U = 0x80100000


class MultiProgramLoader:
    """多程序加载器 — 填充内核进程表并运行调度器."""

    def __init__(
        self,
        emulator: Emulator,
        kernel_image: FirmwareImage | None,
    ) -> None:
        if kernel_image is None:
            raise ValueError("载入了空的内核文件")
        self._emu = emulator
        self._image = kernel_image
        self._hart = emulator.harts[0]
        self._uart: UART = emulator.bus.devices[0x10000000]  # type: ignore[assignment]
        self._next_pid = 0

        # 解析内核符号
        syms = kernel_image.symbols
        proc_table = syms.get("process_table")
        num_procs_addr = syms.get("num_processes")

        if proc_table is None:
            raise KeyError("kernel 未导出 process_table 符号")
        if num_procs_addr is None:
            raise KeyError("kernel 未导出 num_processes 符号")
        self._proc_table: int = proc_table
        self._num_procs_addr: int = num_procs_addr

    # ---- public ----

    def add_process(
        self,
        entry_symbol: str,
    ) -> int:
        """从内核符号表查找 *entry_symbol* 地址, 分配栈, 写入 PCB.

        Returns:
            分配的 pid (0..MAX_PROCS-1).
        """
        pid = self._next_pid
        self._next_pid += 1

        # 查找入口地址
        entry_pc = self._image.symbols.get(entry_symbol)
        if entry_pc is None:
            raise KeyError(f"未找到符号 '{entry_symbol}' — 确保用户程序已链接到内核 ELF")

        # 分配 U 栈: STACK_BASE_U + pid * 2 * PAGE_SIZE 为栈页基址.
        #   栈页上方 = 保护页 (未映射), 栈页 = 实际可用页.
        #   栈顶 = 栈页顶部 = STACK_BASE_U + (pid * 2 + 1) * PAGE_SIZE
        stack_top = STACK_BASE_U + (pid * 2 + 1) * PAGE_SIZE

        # 写 PCB
        pcb_addr = self._proc_table + pid * PCB_SIZE
        self._write_pcb(pcb_addr, entry_pc, stack_top)

        self._emu.bus.write(
            self._num_procs_addr,
            struct.pack("<I", self._next_pid),
        )

        return pid

    def set_timeslice(self, cycles: int = 5000) -> None:
        """设置调度时间片 (cycle 数). 必须在 load_firmware 之后调用."""
        pass  # 时间片在 kernel.s 中硬编码为 TIMESLICE

    def run(
        self,
        uart_input: bytes | None = None,
    ) -> str:
        """运行模拟直至内核停机 (semihosting SYS_EXIT / 全部 hart halted),
        返回 UART 输出.

        若提供 *uart_input*, 预加载到 UART RX buffer.
        """
        if uart_input:
            self._uart.preload(uart_input)

        self._emu.run(timeout=0)

        return self._uart.tx_data().decode("latin-1", errors="replace")

    # ---- internal ----

    def _write_pcb(
        self,
        base: int,
        entry_pc: int,
        stack_top: int,
    ) -> None:
        """向物理地址 *base* 写入一个 PCB 条目."""
        bus = self._emu.bus

        # state = READY
        bus.write(base + PCB_STATE_OFF, struct.pack("<I", PS_READY))
        # entry_pc
        bus.write(base + PCB_ENTRY_OFF, struct.pack("<Q", entry_pc))
        # stack_top
        bus.write(base + PCB_STACK_OFF, struct.pack("<Q", stack_top))
        # saved_sepc = 0 (首次启动用 entry_pc)
        bus.write(base + PCB_SEPC_OFF, struct.pack("<Q", 0))
        # saved_sp = 0 (首次启动用 stack_top)
        bus.write(base + PCB_SP_OFF, struct.pack("<Q", 0))
        # exit_code = 0
        bus.write(base + PCB_EXIT_OFF, struct.pack("<I", 0))
        # GPR area: x2/sp = stack_top (fresh process needs valid sp)
        bus.write(base + PCB_GPR_OFF + 1 * 8, struct.pack("<Q", stack_top))
