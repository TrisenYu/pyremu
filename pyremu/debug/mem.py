#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""MemoryMixin — 虚拟/物理内存读写、地址解析、反汇编.

依赖 DebuggerBase + SymbolMixin.
"""

import re

from rich.table import Table

from pyremu._native import decode_fields
from pyremu.core.hart import RiscvMode
from pyremu.core.mem_check_aux import translate_addr
from pyremu.core.registers import csr_addr_from_name, gpr_idx_from_name
from pyremu.core.trap_def import trap_cause_name
from pyremu.debug._attrs import SharedMixinAttrs
from pyremu.debug.utils import MAX_INSTR_COUNT, check_rv64_addr, hex_addr
from pyremu.memory.mmu import sv39_walk
from pyremu.utils.disassem import disasm
from pyremu.utils.wrapper import seize_val_err
from pyremu.utils.mask import mask64

# 反汇编着色常量
_GPR_ALIASES = frozenset({
    "zero", "ra", "sp", "gp", "tp",
    "t0", "t1", "t2", "t3", "t4", "t5", "t6",
    "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11",
    "a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7",
    "fp",
})
_BRANCH_JUMP_MNEMONICS = frozenset({
    "beq", "bne", "blt", "bge", "bltu", "bgeu",
    "jal", "jalr", "j", "jr", "ret",
    # Compressed jump/branch mnemonics
    "c.j", "c.jr", "c.jalr", "c.beqz", "c.bnez",
})
_LOAD_STORE_MNEMONICS = frozenset({
    "ld", "lw", "lh", "lb", "lwu", "lhu", "lbu",
    "sd", "sw", "sh", "sb",
    # Compressed load/store mnemonics
    "c.ld", "c.ldsp", "c.lw", "c.lwsp",
    "c.sd", "c.sdsp", "c.sw", "c.swsp",
})
_FENCE_AMO_MNEMONICS = frozenset({
    "fence", "fence.i", "sfence.vma",
    "lr.w", "lr.d", "sc.w", "sc.d",
    "amoswap.w", "amoswap.d", "amoadd.w", "amoadd.d",
    "amoxor.w", "amoxor.d", "amoand.w", "amoand.d",
    "amoor.w", "amoor.d", "amomin.w", "amomin.d",
    "amomax.w", "amomax.d", "amominu.w", "amominu.d",
    "amomaxu.w", "amomaxu.d",
})

class MemoryMixin(SharedMixinAttrs):
    """内存访问和反汇编."""

    # ----------------------------------------------------------
    #  内存读取
    # ----------------------------------------------------------

    def _try_read_va(self, va: int, size: int) -> bytes | None:
        """从虚拟地址读取内存 (自动 VA->PA 翻译)."""
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
            ok, pa = translate_addr(hart, cur)
            if not ok:
                return None
            data = self._emu.bus.try_read(pa, chunk)
            if data is None:
                return None
            result.extend(data)
            remain -= chunk
            cur += chunk
        return bytes(result)

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
        ok, pa = translate_addr(hart, va)
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
        if self._image and self._image.symbols:
            sym_va = self._image.symbols.get(arg)
            if sym_va is not None:
                return sym_va + self._load_offset
        if self._sym_symbols:
            sym_va = self._sym_symbols.get(arg)
            if sym_va is not None:
                return self._resolve_sym_addr(sym_va)
        try:
            v = int(arg, 0)
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

        MMU 启用时 PC 为虚拟地址, 需经 VA->PA 翻译后读取.
        """
        raw = self._try_read_va(pc, 4)
        if raw is None:
            return None
        instr = int.from_bytes(raw.ljust(4, b"\x00"), "little", signed=False)
        if decode_fields(instr).is_compressed:
            return f"{instr & 0xFFFF:04x}", disasm(instr, pc)
        return f"{instr:08x}", disasm(instr, pc)

    @staticmethod
    def _colorize_asm(asm_text: str) -> str:
        """为反汇编文本添加 Rich 颜色标记 (助记符 + 立即数)."""
        if not asm_text or asm_text.startswith("[") or asm_text == "(unknown)":
            return asm_text
        parts = asm_text.split(maxsplit=1)
        mnemonic = parts[0]
        rest = parts[1] if len(parts) > 1 else ""
        # 立即数着色: 十六进制 / 纯数字 -> 品红
        if rest:
            rest = re.sub(
                r'\b(0x[0-9a-fA-F]+|\d+)\b',
                r'[magenta]\1[/]',
                rest,
            )
        if mnemonic in _BRANCH_JUMP_MNEMONICS:
            return f"[green]{mnemonic}[/] {rest}".rstrip()
        if mnemonic in _FENCE_AMO_MNEMONICS:
            return f"[yellow]{mnemonic}[/] {rest}".rstrip()
        if mnemonic in {"ecall", "ebreak", "mret", "sret", "wfi", "c.ebreak"}:
            return f"[red]{mnemonic}[/] {rest}".rstrip()
        if mnemonic in _LOAD_STORE_MNEMONICS:
            return f"[cyan]{mnemonic}[/] {rest}".rstrip()
        if rest:
            return f"{mnemonic} {rest}".rstrip()
        return asm_text

    @staticmethod
    def _ctrl_flow_kind(instr: int) -> str:
        """32-bit 指令的控制流分类."""
        opcode = instr & 0x7F
        if opcode == 0x6F:
            return "jal"
        if opcode == 0x67:
            return "jalr"
        if opcode == 0x63:
            return "branch"
        if opcode == 0x73:
            funct12 = (instr >> 20) & 0xFFF
            if funct12 == 0x000:
                return "ecall"
            if funct12 == 0x001:
                return "ebreak"
            if funct12 == 0x302:
                return "mret"
            if funct12 == 0x102:
                return "sret"
            if funct12 == 0x105:
                return "wfi"
            return "csr"
        if opcode in (0x0F, 0x2F):
            return "fence"
        return "normal"

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
            stale_note = ""
        elif h.mode == RiscvMode.S:
            m_style, s_style = "dim", "bold cyan"
            stale_note = (
                "" if _scause != 0
                else " [dim](M-mode 陈旧)[/]"
            )
        else:
            m_style, s_style = "cyan", "cyan"
            stale_note = " [dim](陈旧)[/]"

        tbl = Table(
            title=f"Hart {self._hart_id}  Trap 上下文"
                  f" ({'中断' if is_intr else '异常'}){stale_note}",
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
            f"[cyan]{_mcause_name}[/]  "
            f"MIE={(_mstatus >> 3) & 1} MPP={(_mstatus >> 11) & 3}"
        )
        s_anno = ""
        if _scause != 0:
            _scause_name = trap_cause_name(_scause)
            s_anno = (
                f"[cyan]{_scause_name}[/]  "
                f"SIE={(_sstatus >> 1) & 1} SPP={(_sstatus >> 8) & 1}"
            )
        elif _sstatus != 0:
            s_anno = (
                f"SIE={(_sstatus >> 1) & 1} "
                f"SPP={(_sstatus >> 8) & 1}"
            )
        self._console.print(f"  {m_anno}    {s_anno}")

    # ----------------------------------------------------------
    #  反汇编命令
    # ----------------------------------------------------------

    def _count_instrs_between(self, start: int, end: int) -> int:
        """计算从 *start* (含) 到 *end* (不含) 之间的指令条数."""
        bus = self._emu.bus
        if not bus.is_ram_addr(start):
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
            self._warn(
                f"addr 至少应为 2-字节对齐, 已从 0x{orig:016x} 对齐到 0x{addr:016x}"
            )
        raw = self._try_read_pa(addr, inst_count * 4)
        if raw is None:
            self._err(f"无法读取物理地址: [yellow]{hex_addr(addr)}[/]")
            return

        is_continue = (
            self._disasm_ref_pc is not None
            and addr == self._disasm_next_addr
        )
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
                extra = self._emu.bus.try_read(addr + offset, inst_size)
                if extra is None:
                    hex_s = " ".join(f"{b:02x}" for b in raw[offset:])
                    instrs.append((pc_addr, hex_s, "[dim](截断)[/]", "normal"))
                    break
                raw = raw[:offset] + extra + raw[offset + len(extra):]
                remaining = inst_size
            asm = disasm(instr, pc_addr)
            raw_hex = f"{instr & 0xFFFF:04x}" if is_compressed else f"{instr:08x}"
            ctrl = (
                self._ctrl_flow_kind_compressed(instr & 0xFFFF)
                if is_compressed
                else self._ctrl_flow_kind(instr)
            )
            instrs.append((pc_addr, raw_hex, asm, ctrl))
            offset += inst_size
        self._disasm_next_addr = addr + offset
        if not instrs:
            self._console.print("[dim](空)[/]")
            return

        if not is_continue:
            self._disasm_ref_pc = self.hart.pc
            self._disasm_past_terminator = False
            ref_idx: int | None = None
            for i, (pc_addr, _, _, _) in enumerate(instrs):
                if pc_addr == self._disasm_ref_pc:
                    ref_idx = i
                    break
            if ref_idx is not None:
                self._disasm_base_step = -ref_idx
            elif self._disasm_ref_pc is None:
                self._disasm_base_step = 0
            elif addr > self._disasm_ref_pc:
                step_cnt = self._count_instrs_between(self._disasm_ref_pc, addr)
                self._disasm_base_step = step_cnt
                if step_cnt >= MAX_INSTR_COUNT:
                    self._disasm_past_terminator = True
            else:
                step_cnt = self._count_instrs_between(addr, self._disasm_ref_pc)
                self._disasm_base_step = -step_cnt
                if step_cnt >= MAX_INSTR_COUNT:
                    self._disasm_past_terminator = True

        ref_pc = self._disasm_ref_pc
        base = self._disasm_base_step
        past_term = self._disasm_past_terminator
        next_base = base
        lines: list[str] = []
        last_scope: str | None = None
        for i, (pc_addr, raw_hex, asm, ctrl) in enumerate(instrs):
            step = base + i
            at_ref = pc_addr == ref_pc
            if at_ref:
                prefix = "pc -> "
            elif step > 0 and not past_term:
                prefix = f"+{step:<5d}"
            else:
                prefix = "      "
            link_addr = pc_addr - self._load_offset
            sym_name = self._resolve_any_symbol_name(pc_addr)
            seg = self._find_segment(link_addr)
            seg_name = seg.name if seg and seg.name else ""
            scope = (
                f"<[dim]{seg_name}[/]:[yellow]{sym_name}[/]>"
                if (seg_name and sym_name)
                else f"<[yellow]{sym_name}[/]>" if sym_name
                else ""
            )
            if scope and scope != last_scope:
                last_scope = scope
                lines.append(f"  {scope}")
            asm_colored = self._colorize_asm(asm)
            lines.append(
                f"{prefix}    [blue]{hex_addr(pc_addr)}[/]    "
                f"[bright_black]{raw_hex:>8s}[/]    {asm_colored}"
            )
            if ctrl in (
                    "jal", "jalr", "ecall", "ebreak",
                    "mret", "sret", "wfi", "branch",
                ):
                lines.append(f"  {'─' * 80}")
                if not past_term and step >= 0:
                    past_term = True
            if not past_term:
                next_base = step + 1
        self._disasm_base_step = next_base
        self._disasm_past_terminator = past_term
        self._console.print(
            f"[bold]反汇编[/] [yellow]{hex_addr(addr)}[/]"
            f"  {inst_count} 条指令\n" + "\n".join(lines)
        )

    def cmd_vdisasm(self, addr_str: str, inst_count_str: str = "16") -> None:
        """虚拟地址反汇编: vdisasm <vaddr> [count]."""
        va = self._resolve_addr(addr_str)
        if va is None and self._image and self._image.symbols:
            sym_addr = self._image.symbols.get(addr_str)
            if sym_addr is not None:
                va = sym_addr + self._load_offset
        inst_count = int(inst_count_str, 0)
        if va is None:
            self._err(f"无法解析地址: {addr_str}")
            return
        if inst_count <= 0 or inst_count > 512:
            self._err("指令数需在 1-512 之间")
            return
        if not check_rv64_addr(va):
            self._err(f"地址 {va:#x} 超出 RV64 范围")
            return
        if self.hart.mmu_mode == 0:
            self._err("satp 未使能 (Bare 模式), vdisasm 无可用翻译; 请用 disasm")
            return
        # _resolve_sym_addr 在 Sv39 模式下返回 PA (经 translate_addr),
        # 但 vdisasm 需要 VA 才能做页表遍历. 通过 _pa_to_va 反查回 VA.
        if va is not None and self._pa_to_va:
            maybe_va = self._pa_to_va.get(va)
            if maybe_va is not None:
                va = maybe_va
        if va & 1:
            orig = va
            va &= ~1
            self._warn(
                f"addr 至少应为 2-字节对齐, 已从 0x{orig:016x} 对齐到 0x{va:016x}"
            )
        if self.hart._mem_read_phy is None:
            self._err("内存后端未挂载")
            return
        root_ppn = self.hart.satp_val & ((1 << 44) - 1)
        ok, ppn, _perm, page_size = sv39_walk(root_ppn, va, self.hart._mem_read_phy)
        if not ok:
            self._err(
                f"VA [yellow]{hex_addr(va)}[/] 页表翻译失败 (缺页);"
                f" satp root PPN=0x{root_ppn:x}"
            )
            return
        # sv39_walk 返回 PPN (非 PA); 用页大小计算完整物理地址.
        # 注: 不走 _try_read_va_forced — 后者在 M-mode 下会将 VA 当作
        # PA (Bare 翻译) 返回, 对 S-mode 内核 VA 会读到错误物理地址.
        page_mask = page_size - 1
        first_pa = mask64(((ppn << 12) | (va & page_mask)))
        raw = self._emu.bus.try_read(first_pa, inst_count * 4)
        if raw is None:
            self._err(f"VA [yellow]{hex_addr(va)}[/] 翻译后物理读取失败")
            return

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
                if extra is not None:
                    raw = raw[:offset] + extra + raw[offset + len(extra):]
                    remaining = inst_size
                else:
                    hex_s = " ".join(f"{b:02x}" for b in raw[offset:])
                    instrs.append((pc_va, hex_s, "[dim](截断)[/]", "normal", pc_pa))
                    break
            asm = disasm(instr, pc_va)
            raw_hex = f"{instr & 0xFFFF:04x}" if is_compressed else f"{instr:08x}"
            ctrl = (
                self._ctrl_flow_kind_compressed(instr & 0xFFFF)
                if is_compressed
                else self._ctrl_flow_kind(instr)
            )
            instrs.append((pc_va, raw_hex, asm, ctrl, pc_pa))
            offset += inst_size
        self._vdisasm_next_addr = va + offset
        if not instrs:
            self._console.print("[dim](空)[/]")
            return

        ref_pc = self.hart.pc
        last_scope: str | None = None
        lines: list[str] = [
            # pc ->的长度与4空格间距
            f"{' '*(5+4)}VA{' '*(16+4)}PA{' '*(16+8+1)}raw    asm"
        ]
        for pc_va, raw_hex, asm, ctrl, pc_pa in instrs:
            link_addr = pc_va - self._load_offset
            sym_name = self._resolve_any_symbol_name(pc_va)
            seg = self._find_segment(link_addr)
            seg_name = seg.name if seg and seg.name else ""
            scope = (
                f"<[dim]{seg_name}[/]:[yellow]{sym_name}[/]>"
                if (seg_name and sym_name)
                else f"<[yellow]{sym_name}[/]>" if sym_name
                else ""
            )
            if scope and scope != last_scope:
                last_scope = scope
                lines.append(f"{scope}")
            va_str = hex_addr(pc_va)
            pa_str = hex_addr(pc_pa)
            at_ref = pc_va == ref_pc
            prefix_fixed = "pc ->" if at_ref else " "*5
            asm_colored = self._colorize_asm(asm)
            lines.append(
                f"{prefix_fixed}    [dim]{va_str}[/]    "
                f"[bright_black]{pa_str}[/]    "
                f"[bright_black]{raw_hex:>8s}[/]    {asm_colored}"
            )
            if ctrl in (
                "jal", "jalr", "ecall", "ebreak", "mret", "sret", "wfi", "branch",
            ):
                lines.append(f"  {'─' * 80}")
        self._console.print(
            f"[bold]vdisasm[/] VA [yellow]{hex_addr(va)}[/]"
            f" -> PA [yellow]{hex_addr(first_pa)}[/]"
            f"  {inst_count} 条指令\n" + "\n".join(lines)
        )

    # ----------------------------------------------------------
    #  内存显示命令
    # ----------------------------------------------------------

    def _show_va_mapping_header(self, va: int, size: int) -> None:
        """显示 VA->PA 翻译及各页的 PTE 权限位."""
        h = self.hart
        page_cnt = ((va & 0xFFF) + size + 0xFFF) >> 12
        lines: list[str] = []
        for pi in range(page_cnt):
            page_va = (va & ~0xFFF) + (pi << 12)
            vpn = page_va >> 12
            hit, ppn, perm = h.dtlb.lookup(vpn)
            if not hit and h._mem_read_phy is None:
                lines.append(
                    f"  [red]VA 0x{page_va:016x}  内存后端未挂载[/]"
                )
                continue
            if not hit and h._mem_read_phy is not None:
                root_ppn = h.satp_val & ((1 << 44) - 1)
                ok, ppn, perm, _ = sv39_walk(root_ppn, page_va, h._mem_read_phy)
                if not ok:
                    lines.append(
                        f"  [red]VA 0x{page_va:016x}  缺页 (无法翻译)[/]"
                    )
                    continue
            pa = ppn << 12
            r = "r" if perm & 1 else "-"
            w = "w" if perm & 2 else "-"
            x = "x" if perm & 4 else "-"
            u = "u" if perm & 8 else "s"
            lines.append(
                f"  VA 0x{page_va:016x} -> PA 0x{pa:016x}  [{r}{w}{x}{u}]"
            )
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
        self._console.print(self._emu._fmt_hexdump(addr, data))

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
        if self.hart._mem_read_phy is None:
            self._err("内存后端未挂载")
            return
        root_ppn = self.hart.satp_val & ((1 << 44) - 1)
        ok, _, _, _ = sv39_walk(root_ppn, va, self.hart._mem_read_phy)
        if not ok:
            self._err(
                f"VA [yellow]{hex_addr(va)}[/] 页表翻译失败 (缺页);"
                f" satp root PPN=0x{root_ppn:x}"
            )
            return
        result = self._try_read_va_forced(va, size)
        if result is None:
            self._err(f"VA [yellow]{hex_addr(va)}[/] 翻译后物理读取失败")
            return
        first_pa, data = result
        self._show_va_mapping_header(va, size)
        self._console.print(
            f"[dim]VA {hex_addr(va)} -> PA {hex_addr(first_pa)}[/]"
        )
        self._console.print(self._emu._fmt_hexdump(va, data))
