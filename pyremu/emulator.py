#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026
# Created at 2026/06/06 星期六 17:00:47
# Last modified at 2026/06/09 星期二

"""
多核 RISC-V 模拟器顶层 — 管理 Bus、CLINT、多个 Hart 的执行循环。

上电后 hart 从可配置的复位向量 (reset vector) 开始执行。
裸金属固件应被直接加载到复位向量对应的物理地址。

Usage:
    emu = Emulator(num_harts=4)
    emu.load_code(addr=0x80000000, code=my_firmware)
    emu.step()        # 所有 hart 各执行一条指令
    emu.run(1000)     # 执行 1000 个周期
"""

from pyremu.core.decoder import Hart
from pyremu.core.trap import TrapType
from pyremu.interrupt.clint import CLINT
from pyremu.memory.bus import Bus
from pyremu.memory.l2cache import L2Cache


class Emulator:
    """多核 RISC-V 模拟器.

    管理 N 个 hart, 共享总线和中断控制器。
    提供 step / run 执行循环及状态检查辅助方法。
    """

    def __init__(
        self,
        num_harts: int = 1,
        ram_size: int = 128 * 1024 * 1024,
        reset_vector: int = 0x8000_0000,
        l2_enabled: bool = False,
        l2_size: int = 256 * 1024,
    ) -> None:
        self._num_harts = num_harts
        self._reset_vector = reset_vector
        self._cycle = 0
        self._total_instrs = 0

        # L2 缓存 (可选)
        l2 = L2Cache(size=l2_size) if l2_enabled else None

        # 共享总线
        self.bus = Bus(ram_size=ram_size, l2_cache=l2)

        # CLINT 设备
        self.clint = CLINT(num_harts=num_harts)
        self.bus.add_device(self.clint.base_addr, self.clint)

        # 创建 harts, 注入后端
        self.harts: list[Hart] = []
        for i in range(num_harts):
            h = Hart(id=i)
            h.pc = reset_vector
            h.set_memory_backend(self.bus.read, self.bus.write)
            h.bus = self.bus
            h.interrupt_ctrl = self.clint
            self.harts.append(h)

    # ----------------------------------------------------------
    #  代码加载
    # ----------------------------------------------------------

    def load_code(self, addr: int, code: bytes) -> None:
        """将机器码写入物理 RAM 的指定地址.

        裸金属程序应加载到复位向量对应的地址 (默认 0x8000_0000).
        """
        self.bus.write(addr, code)

    # ----------------------------------------------------------
    #  执行
    # ----------------------------------------------------------

    def step(self) -> int:
        """所有 hart 各执行一条指令 (round-robin).

        每条指令执行后检查中断, 每个周期推进 CLINT 时钟.

        Returns:
            本轮执行的指令数.
        """
        executed = 0
        for hart in self.harts:
            # 取指
            instr_bytes = self.bus.read(hart.pc, 4)
            instr = int.from_bytes(instr_bytes, "little", signed=False)

            try:
                advance = hart.exec_instr(instr)
            except NotImplementedError:
                hart._take_trap(TrapType.IllInstr, tval=instr, is_interrupt=False)
                advance = 0

            if advance != 0:
                hart.pc = (hart.pc + advance) & 0xFFFF_FFFF_FFFF_FFFF

            # 指令边界 — 检查中断
            hart.check_pending_interrupts()
            executed += 1

        self._cycle += 1
        self._total_instrs += executed
        self.clint.tick(1)
        return executed

    def run(self, max_cycles: int) -> int:
        """执行 *max_cycles* 个周期.

        Returns:
            实际执行的周期数.
        """
        for _ in range(max_cycles):
            self.step()
        return self._cycle

    # ----------------------------------------------------------
    #  状态检查辅助 (调试用)
    # ----------------------------------------------------------

    def dump_hart_regs(self, hart_id: int = 0) -> dict:
        """导出指定 hart 的关键寄存器状态."""
        h = self.harts[hart_id]
        return {
            "hart_id": h.id,
            "pc": h.pc,
            "mode": h.mode.name,
            "mstatus": hex(h.mstatus_val),
            "mepc": hex(h.mepc_val),
            "mcause": hex(h.mcause_val),
            "mtval": hex(h.mtval_val),
            "mie": h.mie,
            "gprs": {f"x{i}": hex(h.gprs[i].val) for i in range(32)},
        }

    def dump_memory(self, addr: int, size: int) -> bytes:
        """读取物理内存的 *size* 字节."""
        return self.bus.read(addr, size)

    def mem_hexdump(self, addr: int, size: int) -> str:
        """返回物理内存的十六进制 dump 字符串."""
        data = self.bus.read(addr, size)
        lines = []
        for offset in range(0, len(data), 16):
            chunk = data[offset : offset + 16]
            hex_part = " ".join(f"{b:02x}" for b in chunk)
            ascii_part = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
            lines.append(f"{addr + offset:016x}  {hex_part:<48s}  |{ascii_part}|")
        return "\n".join(lines)

    # ----------------------------------------------------------
    #  属性
    # ----------------------------------------------------------

    @property
    def num_harts(self) -> int:
        return self._num_harts

    @property
    def cycle(self) -> int:
        return self._cycle

    @property
    def total_instructions(self) -> int:
        return self._total_instrs

    @property
    def reset_vector(self) -> int:
        return self._reset_vector
