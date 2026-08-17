#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""BreakpointMixin — 断点设置/命中检查/条件评估/子命令分发.

依赖 DebuggerBase + MemoryMixin + SymbolMixin.
"""

from rich.table import Table

from pyremu._native import decode_fields
from pyremu.core.hart import RiscvMode
from pyremu.core.mem_check_aux import translate_addr
from pyremu.core.registers import csr_addr_from_name, gpr_idx_from_name
from pyremu.debug._attrs import SharedMixinAttrs
from pyremu.debug.types import Breakpoint
from pyremu.debug.utils import _INSTR_BP_NAMES, _KNOWN_OPCODES, check_rv64_addr, hex_addr
from pyremu.utils.wrapper import seize_val_err


class BreakpointMixin(SharedMixinAttrs):
    """断点管理."""

    # ----------------------------------------------------------
    #  断点缓存 (性能 — 避免每单轮加速执行 Python 侧冗余检查)
    # ----------------------------------------------------------

    def _refresh_bp_cache(self) -> None:
        """更新断点性能缓存标志; 所有修改 _breakpoints 的路径均需调用."""
        self._has_non_addr_bps = any(
            bp.kind != "addr" or bp.cond_type  # cond breakpoint on addr still needs Python eval
            for bp in self._breakpoints
        )

    # ----------------------------------------------------------
    #  条件断点预过滤
    # ----------------------------------------------------------

    def _instr_may_affect_cond(self, instr: int, bp: "Breakpoint") -> bool:
        """快速预过滤: 指令是否可能改变条件断点关注的寄存器/CSR.

        返回 False 时可安全跳过条件评估, 大幅降低 cond 断点的性能开销:
        - CSR 条件: opcode=0x73 且 CSR 地址匹配, 或上一条指令写了目标 CSR
        - GPR 条件: 仅 rd 字段匹配目标寄存器且 rd≠x0 时返回 True
        """
        if bp.cond_type == "csr":
            target = csr_addr_from_name(bp.cond_reg)
            if target is None:
                return True
            f = decode_fields(instr)
            if f.opcode == 0x73 and f.func12 == target:
                return True
            if self._prev_instr_csr_addr == target:
                return True
            return False
        elif bp.cond_type == "reg":
            rd = decode_fields(instr).rd
            if rd == 0:
                return False
            target = gpr_idx_from_name(bp.cond_reg)
            return target is not None and rd == target
        return True

    # ----------------------------------------------------------
    #  条件评估
    # ----------------------------------------------------------

    def _eval_bp_condition(self, hart, bp: "Breakpoint") -> bool:
        """评估断点的附加条件; 无条件时返回 True."""
        if not bp.cond_type:
            return True
        if bp.cond_type == "reg":
            actual = hart.read_gpr_by_name(bp.cond_reg)
        elif bp.cond_type == "csr":
            actual = hart.read_csr_by_name(bp.cond_reg)
        else:
            return True
        c = bp.cond_val
        if bp.cond_op == "==":
            return actual == c
        elif bp.cond_op == "!=":
            return actual != c
        elif bp.cond_op == "<":
            return actual < c
        elif bp.cond_op == ">":
            return actual > c
        return actual == c

    # ----------------------------------------------------------
    #  命中报告
    # ----------------------------------------------------------

    def _report_bp_hit(self, bp: "Breakpoint", hart, pc: int) -> None:
        """打印断点命中信息并暂停."""
        asm_info = self._fetch_and_disasm(pc)
        asm_line = f"\n    {asm_info[1]}" if asm_info else ""
        cond_hint = ""
        if bp.cond_type:
            actual_val = (
                self._read_gpr_by_name(bp.cond_reg)
                if bp.cond_type == "reg"
                else self._read_csr_by_name(bp.cond_reg)
            )
            cond_hint = f" [{bp.cond_reg}=0x{actual_val:x}]"
        self._console.print(
            f"  [bold yellow]* 断点命中[/]  {bp.desc}{cond_hint}  "
            f"@ [cyan]Hart {hart.id}[/]  {hex_addr(pc)}{asm_line}"
        )
        if self._bp_mode == "async":
            self._hart_paused.add(hart.id)
        else:
            self._paused = True

    # ----------------------------------------------------------
    #  地址匹配
    # ----------------------------------------------------------

    @staticmethod
    def _bp_match_pc(hart, pc: int, bp_value: int) -> bool:
        """断点地址匹配: 先尝试直接 VA 比较, 再尝试 PA 翻译比较."""
        if pc == bp_value:
            return True
        if hart.mmu_mode != 0 and hart.mode != RiscvMode.M:
            ok, pa = translate_addr(hart, pc)
            if ok and pa == bp_value:
                return True
        return False

    # ----------------------------------------------------------
    #  断点检查 (执行前)
    # ----------------------------------------------------------

    def _check_breakpoints(self, hart, pc: int, instr: int) -> bool:
        """若当前指令/地址命中任何断点, 暂停并返回 True."""
        if not self._breakpoints:
            return False

        for bp in self._breakpoints:
            hit = False
            if bp.kind == "cond":
                hit = self._instr_may_affect_cond(instr, bp)
            elif bp.kind == "addr":
                hit = self._bp_match_pc(hart, pc, bp.value)
            elif bp.kind == "instr":
                f = decode_fields(instr)
                hit = (
                    f.opcode == 0x73
                    and f.func3 == 0
                    and f.func12 == bp.value
                )
            elif bp.kind == "opcode":
                f = decode_fields(instr)
                if f.is_compressed:
                    continue
                hit = f.opcode == bp.value

            if not hit or not self._eval_bp_condition(hart, bp):
                continue

            bp_key = (bp.kind, pc) if bp.kind == "addr" else (bp.kind, id(bp))
            if bp_key in self._bp_hit_this_run:
                continue

            self._bp_hit_this_run.add(bp_key)
            self._report_bp_hit(bp, hart, pc)
            return True

        self._prev_instr_csr_addr = -1
        f = decode_fields(instr)
        if f.opcode == 0x73:
            self._prev_instr_csr_addr = f.func12
        return False

    # ----------------------------------------------------------
    #  符号断点设置
    # ----------------------------------------------------------

    def _add_bp_and_report(self, addr: int, name: str) -> None:
        """Append an address breakpoint and print the confirmation line."""
        bp = Breakpoint(kind="addr", value=addr, desc=f"{name} ({hex_addr(addr)})")
        self._breakpoints.append(bp)
        self._refresh_bp_cache()
        asm_info = self._fetch_and_disasm(addr)
        asm_str = f"  {asm_info[1]}" if asm_info else ""
        self._console.print(
            f"  [green]断点 {len(self._breakpoints)}[/]  "
            f"[yellow]{name}[/] {hex_addr(addr)}{asm_str}"
        )

    def _try_set_symbol_bp(self, name: str) -> bool:
        """尝试按符号名设置地址断点.  成功返回 True, 未匹配返回 False."""
        # 1) 固件符号 (OpenSBI / 测试固件)
        if self._image is not None and self._image.symbols:
            sym_addr = self._image.symbols.get(name)
            if sym_addr is not None:
                addr = sym_addr + self._load_offset
                if not check_rv64_addr(addr):
                    return True
                self._add_bp_and_report(addr, name)
                return True

        # 2) 外部调试符号 (Linux vmlinux 等)
        if self._sym_symbols is None or not self._sym_symbols:
            return False
        sym_va = self._sym_symbols.get(name)
        if sym_va is None:
            return False
        runtime_addr = self._resolve_sym_addr(sym_va)
        if runtime_addr is None:
            self._err(f"无法解析符号运行时地址: {name}")
            return True
        if not check_rv64_addr(runtime_addr):
            return True
        self._add_bp_and_report(runtime_addr, name)
        return True

    # ----------------------------------------------------------
    #  bp 命令
    # ----------------------------------------------------------

    def cmd_bp_set(self, rest: str) -> None:
        """设置断点: bp <addr> | bp <symbol> | bp <type>.

        支持条件: bp <addr> if reg <name> <op> <val>
                  bp <addr> if csr <name> <op> <val>
        """
        cond_type, cond_reg, cond_op, cond_val = "", "", "==", 0
        parts = rest.split(None, 2)
        if len(parts) >= 3 and parts[1] == "if":
            rest = parts[0]
            cond_parts = parts[2].split(None, 3)
            if len(cond_parts) >= 3 and cond_parts[0] in ("reg", "csr"):
                cond_type = cond_parts[0]
                cond_reg = cond_parts[1]
                cond_op = cond_parts[2] if len(cond_parts) >= 3 else "=="
                cond_val = int(cond_parts[3], 0) if len(cond_parts) >= 4 else 0

        # 1) 符号名
        if self._try_set_symbol_bp(rest):
            if not cond_type:
                return
            self._breakpoints[-1].cond_type = cond_type
            self._breakpoints[-1].cond_reg = cond_reg
            self._breakpoints[-1].cond_op = cond_op
            self._breakpoints[-1].cond_val = cond_val
            self._breakpoints[-1].desc += (
                f" if {cond_type} {cond_reg}{cond_op}0x{cond_val:x}"
            )
            self._console.print(
                f"    条件: {cond_type} {cond_reg} {cond_op} 0x{cond_val:x}"
            )
            return

        # 2) 地址 (pc / 寄存器名 / 十六进制)
        addr = self._resolve_addr(rest)
        if addr is not None:
            if not check_rv64_addr(addr):
                return
            desc = hex_addr(addr)
            label = "断点"
            if cond_type:
                desc += f" if {cond_type} {cond_reg}{cond_op}0x{cond_val:x}"
                label = "条件断点"
            bp = Breakpoint(
                kind="addr",
                value=addr, desc=desc,
                cond_type=cond_type,
                cond_reg=cond_reg,
                cond_op=cond_op,
                cond_val=cond_val,
            )
            self._breakpoints.append(bp)
            self._refresh_bp_cache()
            asm_info = self._fetch_and_disasm(addr)
            asm_str = f"  {asm_info[1]}" if asm_info else ""
            cond_str = (
                f" [bold yellow]if[/] {cond_type} {cond_reg} "
                f"{cond_op} [yellow]0x{cond_val:x}[/]"
                if cond_type
                else ""
            )
            self._console.print(
                f"  [green]{label} {len(self._breakpoints)}[/]{cond_str}  "
                f"[yellow]{hex_addr(addr)}[/]"
                f"{asm_str}"
            )
            return

        # 3) 命名指令
        name = rest.lower()
        if name not in _INSTR_BP_NAMES:
            self._err(f"无法识别的断点参数: {rest}")
            return
        bp = Breakpoint(kind="instr", value=_INSTR_BP_NAMES[name], desc=name)
        self._breakpoints.append(bp)
        self._refresh_bp_cache()
        self._console.print(
            f"  [green]断点 {len(self._breakpoints)}[/]  [yellow]{name}[/]"
        )

    @seize_val_err("opcode 需为十六进制整数 (支持 0x 前缀)")
    def cmd_bp_opcode(self, arg: str) -> None:
        """设置 opcode 断点: bp opcode <hex>."""
        raw = int(arg, 0)
        val = raw & 0x7F
        if raw < 0 or raw > 0x7F:
            self._err(f"opcode 0x{raw:x} 超出 7-bit 范围 [0, 127]")
            return
        if val not in _KNOWN_OPCODES:
            self._err(f"opcode 0x{val:02x} 不是已知的 RV64 opcode, 拒绝设置断点")
            return
        bp = Breakpoint(kind="opcode", value=val, desc=f"opcode 0x{val:02x}")
        self._breakpoints.append(bp)
        self._refresh_bp_cache()
        self._console.print(
            f"  [green]断点 {len(self._breakpoints)}[/]  [yellow]opcode 0x{val:02x}[/]"
        )

    def cmd_bp_list(self) -> None:
        """列出所有断点."""
        if not self._breakpoints:
            self._console.print("  [dim](无断点)[/]")
            return
        tbl = Table(title="断点列表", border_style="blue")
        tbl.add_column("#", style="cyan", justify="right")
        tbl.add_column("类型", style="yellow")
        tbl.add_column("值", style="green")
        for i, bp in enumerate(self._breakpoints, 1):
            tbl.add_row(str(i), bp.kind, bp.desc)
        self._console.print(tbl)

    @seize_val_err("断点编号需为整数")
    def cmd_bp_delete(self, idx_str: str) -> None:
        """删除断点: bp delete <n> (1-based)."""
        idx = int(idx_str, 0) - 1
        if idx < 0 or idx >= len(self._breakpoints):
            self._err(
                f"断点编号超出范围: {idx_str} (当前 {len(self._breakpoints)} 个)"
            )
            return
        removed = self._breakpoints.pop(idx)
        self._refresh_bp_cache()
        self._console.print(f"  [dim]已删除断点 [yellow]{removed.desc}[/][/]")

    def cmd_bp_clear(self) -> None:
        """清除全部断点."""
        count = len(self._breakpoints)
        self._breakpoints.clear()
        self._refresh_bp_cache()
        self._console.print(f"  [dim]已清除 {count} 个断点[/]")

    def cmd_bp_mode(self, arg: str | None = None) -> None:
        """查看或设置断点模式: bp mode [sync|async]."""
        if arg is None:
            self._console.print(f"  断点模式: [yellow]{self._bp_mode}[/]")
            return
        mode = arg.lower()
        if mode not in ("sync", "async"):
            self._err(f"无效模式: {arg} (应为 sync 或 async)")
            return
        self._bp_mode = mode
        self._hart_paused.clear()
        self._console.print(f"  断点模式 -> [yellow]{mode}[/]")

    def _dispatch_bp(self, rest: list[str]) -> None:
        """bp 命令子分发: 将无歧义的子命令名路由到对应 handler."""
        if not rest:
            self.cmd_bp_list()
            return
        sub = rest[0].lower()
        if sub == "mode":
            self.cmd_bp_mode(rest[1] if len(rest) > 1 else None)
        elif sub == "delete":
            if len(rest) > 1:
                self.cmd_bp_delete(rest[1])
                return
            self._warn("用法: bp delete <编号>")
        elif sub == "clear":
            self.cmd_bp_clear()
        elif sub == "if":
            if not (len(rest) >= 4 and rest[1] in ("reg", "csr")):
                self._warn("用法: bp if reg/csr <name> <op> <val>")
                return
            ct, cr, co = rest[1], rest[2], rest[3] if len(rest) > 3 else "=="
            cv = int(rest[4], 0) if len(rest) > 4 else 0
            desc = f"if {ct} {cr}{co}0x{cv:x}"
            bp = Breakpoint(
                kind="cond", value=0, desc=desc,
                cond_type=ct, cond_reg=cr, cond_op=co, cond_val=cv,
            )
            self._breakpoints.append(bp)
            self._refresh_bp_cache()
            self._console.print(
                f"  [green]条件断点 {len(self._breakpoints)}[/]  "
                f"[yellow]{desc}[/]"
            )
        elif sub == "opcode":
            if len(rest) < 2:
                self._warn("用法: bp opcode <hex>")
                return
            self.cmd_bp_opcode(rest[1])
        else:
            self.cmd_bp_set(" ".join(rest))
