#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""MemoryOps — 虚拟/物理内存读写、地址解析、反汇编.

依赖 DebuggerBase + SymLookupOps.
"""

import re

from rich.table import Table

from pyremu._native import decode_fields
from pyremu.core.hart import RiscvMode
from pyremu.core.mem_check_aux import translate_addr
from pyremu.core.registers import csr_addr_from_name, gpr_idx_from_name
from pyremu.core.trap_def import trap_cause_name
from pyremu.debug.attrs import SharedAttrs
from pyremu.debug.utils import (
    check_rv64_addr,
    hex_addr,
    MAX_INSTR_COUNT,
    read_code_bytes,
)
from pyremu.memory.mmu import sv39_walk
from pyremu.utils.disassem import disasm
from pyremu.utils.mask import mask64
from pyremu.utils.str_aux import dye_hexdump
from pyremu.utils.wrapper import seize_val_err

# 反汇编着色常量
_GPR_ALIASES = frozenset(
    {
        "zero",
        "ra",
        "sp",
        "gp",
        "tp",
        "t0",
        "t1",
        "t2",
        "t3",
        "t4",
        "t5",
        "t6",
        "s0",
        "s1",
        "s2",
        "s3",
        "s4",
        "s5",
        "s6",
        "s7",
        "s8",
        "s9",
        "s10",
        "s11",
        "a0",
        "a1",
        "a2",
        "a3",
        "a4",
        "a5",
        "a6",
        "a7",
        "fp",
    }
)
_BRANCH_JUMP_MNEMONICS = frozenset(
    {
        "beq",
        "bne",
        "blt",
        "bge",
        "bltu",
        "bgeu",
        "jal",
        "jalr",
        "j",
        "jr",
        "ret",
        # Compressed jump/branch mnemonics
        "c.j",
        "c.jr",
        "c.jalr",
        "c.beqz",
        "c.bnez",
    }
)
_LOAD_STORE_MNEMONICS = frozenset(
    {
        "ld",
        "lw",
        "lh",
        "lb",
        "lwu",
        "lhu",
        "lbu",
        "sd",
        "sw",
        "sh",
        "sb",
        # Compressed load/store mnemonics
        "c.ld",
        "c.ldsp",
        "c.lw",
        "c.lwsp",
        "c.sd",
        "c.sdsp",
        "c.sw",
        "c.swsp",
    }
)
_PRIVILEGED_MNEMONICS = frozenset({"ecall", "ebreak", "mret", "sret", "wfi", "c.ebreak"})
_FENCE_AMO_MNEMONICS = frozenset(
    {
        "fence",
        "fence.i",
        "sfence.vma",
        "lr.w",
        "lr.d",
        "sc.w",
        "sc.d",
        "amoswap.w",
        "amoswap.d",
        "amoadd.w",
        "amoadd.d",
        "amoxor.w",
        "amoxor.d",
        "amoand.w",
        "amoand.d",
        "amoor.w",
        "amoor.d",
        "amomin.w",
        "amomin.d",
        "amomax.w",
        "amomax.d",
        "amominu.w",
        "amominu.d",
        "amomaxu.w",
        "amomaxu.d",
    }
)

# 助记符类别与对应的颜色名, 按该顺序判定
_MNEMONIC_COLORS = (
    (_BRANCH_JUMP_MNEMONICS, "green"),
    (_FENCE_AMO_MNEMONICS, "yellow"),
    (_PRIVILEGED_MNEMONICS, "red"),
    (_LOAD_STORE_MNEMONICS, "cyan"),
)

# 32 位指令的 opcode 到控制流分类
_OPCODE_FLOW_KINDS = {
    0x6F: "jal",
    0x67: "jalr",
    0x63: "branch",
    0x0F: "fence",
    0x2F: "fence",
}

# 反汇编在此类控制流之后画分隔线并停止累计步数
_TERMINATOR_KINDS = frozenset(
    {
        "jal",
        "jalr",
        "ecall",
        "ebreak",
        "mret",
        "sret",
        "wfi",
        "branch",
    }
)

# SYSTEM 指令的 funct12 到控制流分类
_SYSTEM_FLOW_KINDS = {
    0x000: "ecall",
    0x001: "ebreak",
    0x302: "mret",
    0x102: "sret",
    0x105: "wfi",
}


def _mnemonic_color(mnemonic: str) -> str | None:
    """返回反汇编助记符对应的颜色名; 无对应颜色时返回 None."""
    for group, color in _MNEMONIC_COLORS:
        if mnemonic in group:
            return color
    return None


class MemoryOps(SharedAttrs):
    """内存访问和反汇编."""

    # ----------------------------------------------------------
    #  内存读取
    # ----------------------------------------------------------

    def _try_read_va(self, va: int, size: int) -> bytes | None:
        """从虚拟地址读取内存, 自动完成 VA 到 PA 的翻译."""
        if size <= 0 or size > 4096:
            return None
        hart = self.hart
        if hart.mmu_mode == 0 or hart.mode == RiscvMode.M:
            return self._emu.bus.try_read(va, size)
        result = bytearray()
        remain = size
        cur = va
        while remain > 0:
            page_off = cur & 0xFFF
            chunk = min(remain, 0x1000 - page_off)
            ok, pa = translate_addr(hart, cur, is_hart_access=False)
            if not ok:
                return None
            data = self._emu.bus.try_read(pa, chunk)
            if data is None:
                return None
            result.extend(data)
            remain -= chunk
            cur += chunk
        return bytes(result)

    def _try_read_code_va(self, va: int, size: int) -> bytes | None:
        """按取指语义读取指令字节.

        取指只判定页面的可执行位; 数据读语义还要求 R=1, 只在 mstatus.MXR
        置位时才由 X=1 顶替. 管理器把自身 .text 映射为只可执行, 故 pc 与栈
        回溯读取指令字节时不能用数据读语义.
        """
        return read_code_bytes(self._emu.bus, self.hart, va, size)

    def _try_read_pa(self, pa: int, size: int) -> bytes | None:
        """从物理地址读取内存."""
        if size <= 0 or size > 4096:
            return None
        return self._emu.bus.try_read(pa, size)

    def _try_read_va_forced(self, va: int, size: int) -> tuple[int, bytes] | None:
        """尝试 VA 读取并返回翻译后的 PA."""
        if size <= 0 or size > 4096:
            return None
        hart = self.hart
        if hart.mmu_mode == 0 or hart.mode == RiscvMode.M:
            data = self._emu.bus.try_read(va, size)
            return (va, data) if data is not None else None
        ok, pa = translate_addr(hart, va, is_hart_access=False)
        if not ok:
            return None
        data = self._emu.bus.try_read(pa, size)
        return (pa, data) if data is not None else None

    # ----------------------------------------------------------
    #  地址解析
    # ----------------------------------------------------------

    def _find_gpr(self, name: str) -> int | None:
        """按名称查找 GPR 索引 (x12, t0, a0, sp 等)."""
        return gpr_idx_from_name(name)

    def _resolve_addr(self, arg: str) -> int | None:
        """将字符串解析为地址: pc / 寄存器名 / 符号名 / 数值."""
        if arg.lower() == "pc":
            return self.hart.pc
        idx = self._find_gpr(arg)
        if idx is not None:
            return mask64(self.hart.gprs[idx])
        v = self._read_csr_by_name(arg)
        if v != 0 or csr_addr_from_name(arg) is not None:
            return v

        name_addr = self._resolve_symbol_operand(arg)
        if name_addr is not None:
            return name_addr
        return self._parse_addr_literal(arg)

    def _resolve_symbol_operand(self, name: str) -> int | None:
        """按固件符号、已加载模块、外部符号表的顺序解析符号名."""
        if self._image and self._image.symbols:
            sym_va = self._image.symbols.get(name)
            if sym_va is not None:
                return sym_va + self._load_offset
        module_pa = self._module_runtime_addr_of_name(name)
        if module_pa is not None:
            return module_pa
        if self._sym_symbols:
            sym_va = self._sym_symbols.get(name)
            if sym_va is not None:
                return self._resolve_sym_addr(sym_va)
        return None

    @staticmethod
    def _parse_addr_literal(text: str) -> int | None:
        """解析数值字面量; 超出 RV64 范围或非整数时返回 None."""
        try:
            v = int(text, 0)
        except ValueError:
            return None
        if v < 0 or v >= (1 << 64):
            return None
        return v

    # ----------------------------------------------------------
    #  反汇编辅助
    # ----------------------------------------------------------

    def _fetch_and_disasm(self, pc: int) -> tuple[str, str] | None:
        """从 pc 取指并反汇编一条指令, 返回 (hex_str, asm_str).

        MMU 启用时 PC 为虚拟地址, 需先做 VA 到 PA 的翻译再读取.
        读取用取指语义 (只判定页面的可执行位), 只可执行的页面在数据读语义下
        必定被拒, pc 将显示为不可读取.
        """
        raw = self._try_read_code_va(pc, 4)
        if raw is None:
            return None
        instr = int.from_bytes(raw.ljust(4, b"\x00"), "little", signed=False)
        if decode_fields(instr).is_compressed:
            return f"{instr & 0xFFFF:04x}", disasm(instr, pc)
        return f"{instr:08x}", disasm(instr, pc)

    @staticmethod
    def _colorize_asm(asm_text: str) -> str:
        """为反汇编文本添加 Rich 颜色标记."""
        if not asm_text or asm_text.startswith("[") or asm_text == "(unknown)":
            return asm_text
        parts = asm_text.split(maxsplit=1)
        mnemonic = parts[0]
        operands = parts[1] if len(parts) > 1 else ""
        color = _mnemonic_color(mnemonic)
        if not operands:
            return asm_text if color is None else f"[{color}]{mnemonic}[/]"
        # 十六进制与纯数字立即数显示为品红
        operands = re.sub(r"\b(0x[0-9a-fA-F]+|\d+)\b", r"[magenta]\1[/]", operands)
        if color is None:
            return f"{mnemonic} {operands}"
        return f"[{color}]{mnemonic}[/] {operands}"

    @staticmethod
    def _ctrl_flow_kind(instr: int) -> str:
        """32-bit 指令的控制流分类."""
        opcode = instr & 0x7F
        if opcode == 0x73:
            return _SYSTEM_FLOW_KINDS.get((instr >> 20) & 0xFFF, "csr")
        return _OPCODE_FLOW_KINDS.get(opcode, "normal")

    @staticmethod
    def _ctrl_flow_kind_compressed(instr16: int) -> str:
        """16-bit 压缩指令的控制流分类."""
        op = instr16 & 0x3
        funct3 = (instr16 >> 13) & 0x7
        if op == 0x1 and funct3 in (0x1, 0x5):
            return "jal"
        if op == 0x2 and funct3 == 0x4:
            # C2 funct3=100: C.JR/C.JALR (rs2=0) vs C.MV/C.ADD (rs2≠0)
            rs2 = (instr16 >> 2) & 0x1F
            if rs2 == 0:
                return "jalr"
        elif op == 0x2 and funct3 == 0x6:
            # C.JALR (rs2=0 only)
            rs2 = (instr16 >> 2) & 0x1F
            if rs2 == 0:
                return "jalr"
        if op == 0x1 and funct3 in (0x6, 0x7):
            return "branch"
        if op == 0x2 and funct3 == 0x0:
            return "ebreak"
        return "normal"

    @staticmethod
    def _trap_cause_name_static(mcause_val: int) -> str:
        return trap_cause_name(mcause_val)

    def _warn_pc_if_suspect(self, h, v: int) -> None:
        """PC 值可疑时输出警告 (不可达 / trap / 越界)."""
        if v & 1:
            self._warn(f"PC={hex_addr(v)} 地址未对齐, 可能导致取指失败")
        if not check_rv64_addr(v):
            self._warn(f"PC={hex_addr(v)} 超出 RV64 地址范围")

    # ----------------------------------------------------------
    #  PC 读/设
    # ----------------------------------------------------------

    def cmd_pc(self, addr: str | None = None) -> None:
        """读/设 PC: pc (读取并反汇编), pc <addr> (设置)."""
        h = self.hart
        if addr is not None:
            v = self._resolve_addr(addr)
            if v is None:
                self._err(f"无法解析地址: {addr}")
                return
            if not check_rv64_addr(v):
                self._err(f"PC {v:#x} 超出 RV64 范围")
                return
            self._warn_pc_if_suspect(h, v)
            old, h.pc = h.pc, v
            self._snapshot = None
            self._mem_changes = []
            self._console.print(
                f"PC: [yellow]{hex_addr(old)}[/] -> [green]{hex_addr(h.pc)}[/]"
            )
        pc = h.pc
        result = self._fetch_and_disasm(pc)
        raw_hex, asm = ("(无法读取)", "(无法解码)")
        if result is not None:
            raw_hex, asm = result
        self._console.print(
            f"PC   = [bold yellow]{hex_addr(pc)}[/]\n"
            f"Mode = [bold cyan]{h.mode.name}[/]"
            + ("  [dim]WFI[/]" if h._waiting else "")
            + f"\nRaw  = [bright_black]{raw_hex}[/]"
            + f"\n[bold green]      {asm}[/]"
        )
        self._print_trap_context(h)

    def _print_trap_context(self, h) -> None:
        """输出 M/S 两侧的 trap 上下文; 同一 mcause 只输出一次."""
        mcause = h.mcause_val
        if mcause == 0:
            self._trap_displayed_mcause = None
            return
        if mcause == self._trap_displayed_mcause:
            return
        self._trap_displayed_mcause = mcause

        is_intr = (mcause >> 63) & 1
        _mstatus = h.mstatus_val
        _mcause_name = trap_cause_name(mcause)
        _scause = h.csrs["scause"].val
        _sstatus = h.csrs["sstatus"].val

        if h.mode in (RiscvMode.M, RiscvMode.D):
            m_style, s_style = "bold cyan", "dim"
        elif h.mode == RiscvMode.S:
            m_style, s_style = "dim", "bold cyan"
        else:
            m_style, s_style = "cyan", "cyan"

        tbl = Table(
            title=f"Hart {self._hart_id}  Trap 上下文 {'中断' if is_intr else '异常'}",
            border_style="red",
            show_header=True,
        )
        tbl.add_column("M-mode", style=m_style, justify="left")
        tbl.add_column("S-mode", style=s_style, justify="left")
        tbl.add_row(
            f"mcause  = {hex_addr(mcause)}",
            f"scause  = {hex_addr(_scause)}",
        )
        tbl.add_row(
            f"mepc    = {hex_addr(h.mepc_val)}",
            f"sepc    = {hex_addr(h.csrs['sepc'].val)}",
        )
        tbl.add_row(
            f"mtval   = {hex_addr(h.mtval_val)}",
            f"stval   = {hex_addr(h.csrs['stval'].val)}",
        )
        tbl.add_row(
            f"mstatus = {hex_addr(_mstatus)}",
            f"sstatus = {hex_addr(_sstatus)}",
        )
        self._console.print(tbl)

        m_anno = (
            f"[cyan]{_mcause_name}[/]  MIE={(_mstatus >> 3) & 1} MPP={(_mstatus >> 11) & 3}"
        )
        s_anno = ""
        if _scause != 0:
            _scause_name = trap_cause_name(_scause)
            s_anno = (
                f"[cyan]{_scause_name}[/]  SIE={(_sstatus >> 1) & 1} SPP={(_sstatus >> 8) & 1}"
            )
        elif _sstatus != 0:
            s_anno = f"SIE={(_sstatus >> 1) & 1} SPP={(_sstatus >> 8) & 1}"
        self._console.print(f"  {m_anno}    {s_anno}")

    # ----------------------------------------------------------
    #  反汇编命令
    # ----------------------------------------------------------

    def _count_instrs_between(self, start: int, end: int) -> int:
        """计算从 [start, end) 的指令条数."""
        bus = self._emu.bus
        # 引导只读存储器同样可以存放指令, 从该窗口起始的区间参与计数.
        if not (bus.is_ram_addr(start) or bus.is_rom_addr(start)):
            return MAX_INSTR_COUNT
        count = 0
        addr = start
        while addr < end:
            if count >= MAX_INSTR_COUNT:
                return count
            raw = bus.try_read(addr, 4)
            if raw is None:
                break
            instr = int.from_bytes(raw, "little", signed=False)
            addr += 2 if decode_fields(instr).is_compressed else 4
            count += 1
        return count

    def cmd_disasm(self, addr_str: str, inst_count_str: str = "16") -> None:
        """反汇编指定内存区域: disasm <addr> [count]."""
        addr = self._resolve_addr(addr_str)
        if addr is None and self._image and self._image.symbols:
            sym_addr = self._image.symbols.get(addr_str)
            if sym_addr is not None:
                addr = sym_addr + self._load_offset
        inst_count = int(inst_count_str, 0)

        if addr is None:
            self._err(f"无法解析地址: {addr_str}")
            return
        if inst_count <= 0 or inst_count > 512:
            self._err("指令数需在 1-512 之间")
            return
        if not check_rv64_addr(addr):
            self._err(f"地址 {addr:#x} 超出 RV64 范围")
            return
        if addr & 1:
            orig = addr
            addr &= ~1
            self._warn(f"addr 至少应为 2-字节对齐, 已从 0x{orig:016x} 对齐到 0x{addr:016x}")
        raw = self._try_read_pa(addr, inst_count * 4)
        if raw is None:
            self._err(f"无法读取物理地址: [yellow]{hex_addr(addr)}[/]")
            return

        is_continue = self._disasm_ref_pc is not None and addr == self._disasm_next_addr
        instrs, offset = self._decode_pa_range(addr, raw, inst_count)
        self._disasm_next_addr = addr + offset
        if not instrs:
            self._console.print("[dim](空)[/]")
            return

        self._update_disasm_anchor(addr, instrs, is_continue)
        lines = self._disasm_lines(instrs)
        self._console.print(
            f"[bold]反汇编[/] [yellow]{hex_addr(addr)}[/]"
            f"  {inst_count} 条指令\n" + "\n".join(lines)
        )

    def _decode_pa_range(
        self,
        addr: int,
        raw: bytes,
        inst_count: int,
    ) -> tuple[list[tuple[int, str, str, str]], int]:
        """按物理地址逐条解码, 返回指令列表与已解码的字节数."""
        instrs: list[tuple[int, str, str, str]] = []
        offset = 0
        max_offset = len(raw)
        while offset < max_offset and len(instrs) < inst_count:
            pc_addr = addr + offset
            remaining = max_offset - offset
            chunk = raw[offset : offset + min(4, remaining)]
            instr = int.from_bytes(chunk.ljust(4, b"\x00"), "little", signed=False)
            is_compressed = decode_fields(instr).is_compressed
            inst_size = 2 if is_compressed else 4
            if remaining < inst_size:
                extra = self._emu.bus.try_read(pc_addr, inst_size)
                if extra is None:
                    hex_s = " ".join(f"{b:02x}" for b in raw[offset:])
                    instrs.append((pc_addr, hex_s, "[dim](截断)[/]", "normal"))
                    break
                raw = raw[:offset] + extra + raw[offset + len(extra) :]
            asm = disasm(instr, pc_addr)
            raw_hex = f"{instr & 0xFFFF:04x}" if is_compressed else f"{instr:08x}"
            ctrl = (
                self._ctrl_flow_kind_compressed(instr & 0xFFFF)
                if is_compressed
                else self._ctrl_flow_kind(instr)
            )
            instrs.append((pc_addr, raw_hex, asm, ctrl))
            offset += inst_size
        return instrs, offset

    def _update_disasm_anchor(
        self,
        addr: int,
        instrs: list[tuple[int, str, str, str]],
        is_continue: bool,
    ) -> None:
        """本次反汇编接续上一次时直接返回. 否则以当前 PC 为参照 PC, 计算首条指令
        到参照 PC 之间的指令条数并写入 `_disasm_base_step`; 条数达到上限时置
        `_disasm_past_terminator`.
        """
        if is_continue:
            return
        self._disasm_ref_pc = self.hart.pc
        self._disasm_past_terminator = False
        ref_idx: int | None = None
        for i, (pc_addr, _, _, _) in enumerate(instrs):
            if pc_addr == self._disasm_ref_pc:
                ref_idx = i
                break
        if ref_idx is not None:
            self._disasm_base_step = -ref_idx
            return
        if self._disasm_ref_pc is None:
            self._disasm_base_step = 0
            return
        if addr > self._disasm_ref_pc:
            step_cnt = self._count_instrs_between(self._disasm_ref_pc, addr)
            self._disasm_base_step = step_cnt
        else:
            step_cnt = self._count_instrs_between(addr, self._disasm_ref_pc)
            self._disasm_base_step = -step_cnt
        if step_cnt >= MAX_INSTR_COUNT:
            self._disasm_past_terminator = True

    def _scope_label(self, pc_addr: int) -> str:
        """按符号名与段名给出作用域标注; 都解析不到时返回空串."""
        sym_name = self._resolve_any_symbol_name(pc_addr)
        seg = self._find_segment(pc_addr - self._load_offset)
        seg_name = seg.name if seg and seg.name else ""
        if seg_name and sym_name:
            return f"<[dim]{seg_name}[/]:[yellow]{sym_name}[/]>"
        if sym_name:
            return f"<[yellow]{sym_name}[/]>"
        return ""

    @staticmethod
    def _disasm_prefix(step: int, at_ref: bool, past_term: bool) -> str:
        """给出反汇编行首的步数标注."""
        if at_ref:
            return "pc -> "
        if step > 0 and not past_term:
            return f"+{step:<5d}"
        return "      "

    def _disasm_lines(self, instrs: list[tuple[int, str, str, str]]) -> list[str]:
        """渲染反汇编行, 并更新下一次续接时使用的步数值与终止标记."""
        base = self._disasm_base_step
        past_term = self._disasm_past_terminator
        next_base = base
        lines: list[str] = []
        last_scope: str | None = None
        for i, (pc_addr, raw_hex, asm, ctrl) in enumerate(instrs):
            step = base + i
            scope = self._scope_label(pc_addr)
            if scope and scope != last_scope:
                last_scope = scope
                lines.append(f"  {scope}")
            prefix = self._disasm_prefix(step, pc_addr == self._disasm_ref_pc, past_term)
            lines.append(
                f"{prefix}    [blue]{hex_addr(pc_addr)}[/]    "
                f"[bright_black]{raw_hex:>8s}[/]    {self._colorize_asm(asm)}"
            )
            if ctrl in _TERMINATOR_KINDS:
                lines.append(f"  {'─' * 80}")
                if not past_term and step >= 0:
                    past_term = True
            if not past_term:
                next_base = step + 1
        self._disasm_base_step = next_base
        self._disasm_past_terminator = past_term
        return lines

    def cmd_vdisasm(self, addr_str: str, inst_count_str: str = "16") -> None:
        """虚拟地址反汇编: vdisasm <vaddr> [count]."""
        prepared = self._prepare_vdisasm(addr_str, inst_count_str)
        if prepared is None:
            return
        va, first_pa, raw, inst_count = prepared

        instrs, offset = self._decode_va_range(va, first_pa, raw, inst_count)
        self._vdisasm_next_addr = va + offset
        if not instrs:
            self._console.print("[dim](空)[/]")
            return

        lines = self._vdisasm_lines(instrs)
        self._console.print(
            f"[bold]vdisasm[/] VA [yellow]{hex_addr(va)}[/]"
            f" -> PA [yellow]{hex_addr(first_pa)}[/]"
            f"  {inst_count} 条指令\n" + "\n".join(lines)
        )

    def _resolve_vdisasm_va(self, addr_str: str) -> int | None:
        """把 vdisasm 的参数解析成可遍历页表的虚拟地址."""
        va = self._resolve_addr(addr_str)
        if va is None and self._image and self._image.symbols:
            sym_addr = self._image.symbols.get(addr_str)
            if sym_addr is not None:
                va = sym_addr + self._load_offset
        if va is None:
            self._err(f"无法解析地址: {addr_str}")
            return None
        if not check_rv64_addr(va):
            self._err(f"地址 {va:#x} 超出 RV64 范围")
            return None
        if self.hart.mmu_mode == 0:
            self._err("satp 未使能 (Bare 模式), vdisasm 无可用翻译; 请用 disasm")
            return None
        # _resolve_sym_addr 在 Sv39 模式下返回 PA (经 translate_addr),
        # 但 vdisasm 需要 VA 才能做页表遍历. 通过 _pa_to_va 反查回 VA.
        if self._pa_to_va:
            maybe_va = self._pa_to_va.get(va)
            if maybe_va is not None:
                va = maybe_va
        if va & 1:
            orig = va
            va &= ~1
            self._warn(f"addr 至少应为 2-字节对齐, 已从 0x{orig:016x} 对齐到 0x{va:016x}")
        return va

    def _prepare_vdisasm(
        self,
        addr_str: str,
        inst_count_str: str,
    ) -> tuple[int, int, bytes, int] | None:
        """完成地址解析、页表遍历与首次读取; 任一步失败时报告原因并返回 None."""
        inst_count = int(inst_count_str, 0)
        if inst_count <= 0 or inst_count > 512:
            self._err("指令数需在 1-512 之间")
            return None
        va = self._resolve_vdisasm_va(addr_str)
        if va is None:
            return None
        if self.hart._mem_read_phy is None:
            self._err("内存后端未挂载")
            return None
        root_ppn = self.hart.satp_val & ((1 << 44) - 1)
        ok, ppn, _perm, page_size = sv39_walk(root_ppn, va, self.hart._mem_read_phy)
        if not ok:
            self._err(
                f"VA [yellow]{hex_addr(va)}[/] 页表翻译失败 (缺页);"
                f" satp root PPN=0x{root_ppn:x}"
            )
            return None
        # sv39_walk 返回 PPN (非 PA); 用页大小计算完整物理地址.
        # 注: 不走 _try_read_va_forced — 后者在 M-mode 下会将 VA 当作
        # PA (Bare 翻译) 返回, 对 S-mode 内核 VA 会读到错误物理地址.
        page_mask = page_size - 1
        first_pa = mask64(((ppn << 12) | (va & page_mask)))
        raw = self._emu.bus.try_read(first_pa, inst_count * 4)
        if raw is None:
            self._err(f"VA [yellow]{hex_addr(va)}[/] 翻译后物理读取失败")
            return None
        return va, first_pa, raw, inst_count

    def _decode_va_range(
        self,
        va: int,
        first_pa: int,
        raw: bytes,
        inst_count: int,
    ) -> tuple[list[tuple[int, str, str, str, int]], int]:
        """按虚拟地址逐条解码, 返回指令列表与已解码的字节数."""
        instrs: list[tuple[int, str, str, str, int]] = []
        offset = 0
        max_offset = len(raw)
        while offset < max_offset and len(instrs) < inst_count:
            pc_va = va + offset
            pc_pa = first_pa + offset
            remaining = max_offset - offset
            chunk = raw[offset : offset + min(4, remaining)]
            instr = int.from_bytes(chunk.ljust(4, b"\x00"), "little", signed=False)
            is_compressed = decode_fields(instr).is_compressed
            inst_size = 2 if is_compressed else 4
            if remaining < inst_size:
                extra = self._emu.bus.try_read(pc_pa, inst_size)
                if extra is None:
                    hex_s = " ".join(f"{b:02x}" for b in raw[offset:])
                    instrs.append((pc_va, hex_s, "[dim](截断)[/]", "normal", pc_pa))
                    break
                raw = raw[:offset] + extra + raw[offset + len(extra) :]
            asm = disasm(instr, pc_va)
            raw_hex = f"{instr & 0xFFFF:04x}" if is_compressed else f"{instr:08x}"
            ctrl = (
                self._ctrl_flow_kind_compressed(instr & 0xFFFF)
                if is_compressed
                else self._ctrl_flow_kind(instr)
            )
            instrs.append((pc_va, raw_hex, asm, ctrl, pc_pa))
            offset += inst_size
        return instrs, offset

    def _vdisasm_lines(
        self,
        instrs: list[tuple[int, str, str, str, int]],
    ) -> list[str]:
        """渲染虚拟地址反汇编行, 表头之后逐条给出 VA、PA、原字节与助记符."""
        ref_pc = self.hart.pc
        last_scope: str | None = None
        lines: list[str] = [
            # pc ->的长度与4空格间距
            f"{' ' * (5 + 4)}VA{' ' * (16 + 4)}PA{' ' * (16 + 8 + 1)}raw    asm"
        ]
        for pc_va, raw_hex, asm, ctrl, pc_pa in instrs:
            scope = self._scope_label(pc_va)
            if scope and scope != last_scope:
                last_scope = scope
                lines.append(f"{scope}")
            prefix_fixed = "pc ->" if pc_va == ref_pc else " " * 5
            lines.append(
                f"{prefix_fixed}    [dim]{hex_addr(pc_va)}[/]    "
                f"[bright_black]{hex_addr(pc_pa)}[/]    "
                f"[bright_black]{raw_hex:>8s}[/]    {self._colorize_asm(asm)}"
            )
            if ctrl in _TERMINATOR_KINDS:
                lines.append(f"  {'─' * 80}")
        return lines

    # ----------------------------------------------------------
    #  内存显示命令
    # ----------------------------------------------------------

    def _show_va_mapping_header(self, va: int, size: int) -> None:
        """显示 VA 到 PA 的翻译及各页的 PTE 权限位."""
        h = self.hart
        page_cnt = ((va & 0xFFF) + size + 0xFFF) >> 12
        lines: list[str] = []
        for pi in range(page_cnt):
            page_va = (va & ~0xFFF) + (pi << 12)
            vpn = page_va >> 12
            hit, ppn, perm = h.dtlb.lookup(vpn)
            if not hit and h._mem_read_phy is None:
                lines.append(f"  [red]VA 0x{page_va:016x}  内存后端未挂载[/]")
                continue
            if not hit and h._mem_read_phy is not None:
                root_ppn = h.satp_val & ((1 << 44) - 1)
                ok, ppn, perm, _ = sv39_walk(root_ppn, page_va, h._mem_read_phy)
                if not ok:
                    lines.append(f"  [red]VA 0x{page_va:016x}  缺页 (无法翻译)[/]")
                    continue
            pa = ppn << 12
            r = "r" if perm & 1 else "-"
            w = "w" if perm & 2 else "-"
            x = "x" if perm & 4 else "-"
            u = "u" if perm & 8 else "s"
            lines.append(f"  VA 0x{page_va:016x} -> PA 0x{pa:016x}  [{r}{w}{x}{u}]")
        self._console.print("\n".join(lines))

    @seize_val_err("addr 和 size 需为整数 (支持 0x 前缀)")
    def cmd_mem(self, addr_str: str, size_str: str = "64") -> None:
        """物理内存 hexdump: mem <addr> [size]."""
        addr = self._resolve_addr(addr_str)
        if addr is None:
            self._err(f"无法解析地址: {addr_str}")
            return
        size = int(size_str, 0)
        if size <= 0 or size > 4096:
            self._err("size 需在 1-4096 之间")
            return
        if not check_rv64_addr(addr):
            self._err(f"地址 {addr:#x} 超出 RV64 范围")
            return
        data = self._try_read_pa(addr, size)
        if data is None:
            self._err("无法读取物理地址")
            return
        self._console.print(dye_hexdump(data, addr))

    def cmd_vmem(self, addr_str: str, size_str: str = "64") -> None:
        """虚拟内存 hexdump: vmem <vaddr> [size]."""
        va = self._resolve_addr(addr_str)
        if va is None:
            self._err(f"无法解析地址: {addr_str}")
            return
        size = int(size_str, 0)
        if size <= 0 or size > 4096:
            self._err("size 需在 1-4096 之间")
            return
        if not check_rv64_addr(va):
            self._err(f"地址 {va:#x} 超出 RV64 范围")
            return
        if self.hart.mmu_mode == 0:
            self._err("satp 未使能 (Bare 模式), vmem 无可用翻译; 请用 mem")
            return
        # _resolve_sym_addr 在 Sv39 模式下返回 PA (经 translate_addr),
        # 但 vmem 需要 VA 才能做页表遍历. 通过 _pa_to_va 反查回 VA.
        if va is not None and self._pa_to_va:
            maybe_va = self._pa_to_va.get(va)
            if maybe_va is not None:
                va = maybe_va
        result = self._walk_and_read_va(va, size)
        if result is None:
            return
        first_pa, data = result
        self._show_va_mapping_header(va, size)
        self._console.print(f"[dim]VA {hex_addr(va)} -> PA {hex_addr(first_pa)}[/]")
        self._console.print(dye_hexdump(data, va))

    def _walk_and_read_va(self, va: int, size: int) -> tuple[int, bytes] | None:
        """遍历页表并按虚拟地址读出 size 字节; 失败时报告原因并返回 None."""
        if self.hart._mem_read_phy is None:
            self._err("内存后端未挂载")
            return None
        root_ppn = self.hart.satp_val & ((1 << 44) - 1)
        ok, _, _, _ = sv39_walk(root_ppn, va, self.hart._mem_read_phy)
        if not ok:
            self._err(
                f"VA [yellow]{hex_addr(va)}[/] 页表翻译失败 (缺页);"
                f" satp root PPN=0x{root_ppn:x}"
            )
            return None
        result = self._try_read_va_forced(va, size)
        if result is None:
            self._err(f"VA [yellow]{hex_addr(va)}[/] 翻译后物理读取失败")
            return None
        return result
