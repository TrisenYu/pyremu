#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""StatusMixin — Hart 状态、CSR Bitfield 显示.

依赖 DebuggerBase.
"""

from rich.table import Table

from pyremu.core.trap_def import trap_cause_name
from pyremu.debug._attrs import SharedMixinAttrs
from pyremu.debug.utils import (
    EXC_NAMES,
    fmt_instr_count,
    hex_addr,
    ip_bits,
    IRQ_NAMES,
)


class StatusMixin(SharedMixinAttrs):
    """状态和 CSR bitfield 命令."""

    # ----------------------------------------------------------
    #  模式
    # ----------------------------------------------------------

    def cmd_mode(self) -> None:
        h = self.hart
        self._console.print(f"Mode = [bold cyan]{h.mode.name}[/] ({h.mode.value})")

    # ----------------------------------------------------------
    #  mstatus
    # ----------------------------------------------------------

    def cmd_mstatus(self) -> None:
        h = self.hart
        v = h.mstatus_val
        fields = [
            ("MIE", (v >> 3) & 1),
            ("MPIE", (v >> 7) & 1),
            ("MPP", (v >> 11) & 0b11),
            ("SIE", (v >> 1) & 1),
            ("SPIE", (v >> 5) & 1),
            ("SPP", (v >> 8) & 1),
            ("MPRV", (v >> 17) & 1),
            ("SUM", (v >> 18) & 1),
            ("MXR", (v >> 19) & 1),
            ("TVM", (v >> 20) & 1),
            ("TW", (v >> 21) & 1),
            ("TSR", (v >> 22) & 1),
            ("FS", (v >> 13) & 0b11),
            ("SD", (v >> 63) & 1),
        ]
        tbl = Table(title="mstatus 内部字段情况", border_style="magenta")
        tbl.add_column("Field", style="cyan")
        tbl.add_column("Value", style="green")
        tbl.add_row("Hex", hex_addr(v))
        tbl.add_section()
        for fname, fval in fields:
            tbl.add_row(fname, str(fval))
        self._console.print(tbl)

    # ----------------------------------------------------------
    #  mcause / scause
    # ----------------------------------------------------------

    def cmd_mcause(self) -> None:
        self._show_cause("mcause", self.hart.mcause_val)

    def cmd_scause(self) -> None:
        self._show_cause("scause", self.hart.scause_val)

    def _show_cause(self, name: str, v: int) -> None:
        is_irq = (v >> 63) & 1
        code = v & 0x7FFF_FFFF_FFFF_FFFF
        cause_name = trap_cause_name(v)

        tbl = Table(title=f"{name} 内部字段", border_style="magenta")
        tbl.add_column("Field", style="cyan")
        tbl.add_column("Value", style="green")
        tbl.add_column("说明", style="dim")
        tbl.add_row("Hex", hex_addr(v), "")
        tbl.add_section()
        tbl.add_row("Interrupt", str(is_irq), "中断" if is_irq else "异常")
        tbl.add_row("Code", str(code), cause_name)
        self._console.print(tbl)

    # ----------------------------------------------------------
    #  mtvec / stvec
    # ----------------------------------------------------------

    def cmd_mtvec(self) -> None:
        self._show_tvec("mtvec", self.hart.mtvec_val)

    def cmd_stvec(self) -> None:
        self._show_tvec("stvec", self.hart.stvec_val)

    def _show_tvec(self, name: str, v: int) -> None:
        mode = v & 0b11
        base = v & ~0b11
        mode_names = {0: "Direct (直接)", 1: "Vectored (向量)"}

        tbl = Table(title=f"{name} 内部字段", border_style="magenta")
        tbl.add_column("Field", style="cyan")
        tbl.add_column("Value", style="green")
        tbl.add_column("说明", style="dim")
        tbl.add_row("Hex", hex_addr(v), "")
        tbl.add_section()
        tbl.add_row("BASE", hex_addr(base), f"陷态向量基址 (0x{base:016X})")
        tbl.add_row("MODE", str(mode), mode_names.get(mode, f"保留({mode})"))
        self._console.print(tbl)

    # ----------------------------------------------------------
    #  mip / mie / sip / sie
    # ----------------------------------------------------------

    def cmd_mip(self) -> None:
        self._show_ip("mip", self.hart.csrs["mip"].val)

    def cmd_mie(self) -> None:
        self._show_ip("mie", self.hart.csrs["mie"].val)

    def cmd_sip(self) -> None:
        self._show_ip("sip", self.hart.csrs["sip"].val)

    def cmd_sie(self) -> None:
        self._show_ip("sie", self.hart.csrs["sie"].val)

    def _show_ip(self, name: str, v: int) -> None:
        tbl = Table(title=f"{name} 内部字段", border_style="magenta")
        tbl.add_column("Field", style="cyan")
        tbl.add_column("Bit", style="yellow", justify="center")
        tbl.add_column("Value", style="green", justify="center")
        tbl.add_column("说明", style="dim")
        tbl.add_row("Hex", "", hex_addr(v), "")
        tbl.add_section()
        for bit_name, bit, desc in ip_bits():
            val = (v >> bit) & 1
            tbl.add_row(bit_name, str(bit), str(val), desc)
        self._console.print(tbl)

    # ----------------------------------------------------------
    #  medeleg / mideleg
    # ----------------------------------------------------------

    def cmd_medeleg(self) -> None:
        self._show_deleg(
            "medeleg", self.hart.csrs["medeleg"].val, EXC_NAMES
        )

    def cmd_mideleg(self) -> None:
        self._show_deleg(
            "mideleg", self.hart.csrs["mideleg"].val, IRQ_NAMES
        )

    def _show_deleg(self, name: str, v: int, names: dict[int, str]) -> None:
        tbl = Table(
            title=f"{name} 内部字段 (置位 = 委派到 S 模式)",
            border_style="magenta",
        )
        tbl.add_column("Field", style="cyan")
        tbl.add_column("Bit", style="yellow", justify="center")
        tbl.add_column("Val", style="green", justify="center")
        tbl.add_column("说明", style="dim")
        tbl.add_row("Hex", "", hex_addr(v), "")
        tbl.add_section()
        for bit in sorted(names.keys()):
            val = (v >> bit) & 1
            tbl.add_row(f"bit{bit:02d}", str(bit), str(val), names[bit])
        self._console.print(tbl)

    # ----------------------------------------------------------
    #  status (概览 / 详情)
    # ----------------------------------------------------------

    def cmd_status(self, hart_id_str: str | None = None) -> None:
        """显示 hart 状态. 无参数=概览, 带 hart ID=详情."""
        if hart_id_str is None:
            self._show_hart_overview()
            return
        try:
            hid = int(hart_id_str, 0)
        except ValueError:
            self._err(f"无效 hart ID: {hart_id_str}")
            return
        if hid < 0 or hid >= self._emu.num_harts:
            self._err(f"hart ID 超出范围: 0-{self._emu.num_harts - 1}")
            return
        self._show_hart_detail(hid)

    def _show_hart_overview(self) -> None:
        """终端对齐的全部 hart 概览表."""
        harts = self._emu.harts
        total = len(harts)
        active_hart = self._hart_id

        tbl = Table(
            title=f"Harts ({total} total)  [dim]* = 当前[/]",
            border_style="blue",
        )
        tbl.add_column("", style="cyan", width=1)
        tbl.add_column("#", style="cyan", justify="right")
        tbl.add_column("PC", style="green")
        tbl.add_column("Mode", style="yellow", width=5)
        tbl.add_column("State", width=8)
        tbl.add_column("Instr", justify="right")
        for h in harts:
            mark = "*" if h.id == active_hart else " "
            if h._halted:
                state = "[red]halted[/]"
            elif h._waiting:
                state = "[dim]pending[/]" # waiting 也行
            else:
                state = "[green]running[/]"
            # Per-hart instruction count — WFI-waiting harts don't
            # execute, so their count stays unchanged while running
            # harts advance.
            instr_fmt = fmt_instr_count(h._total_instrs)
            tbl.add_row(
                mark, str(h.id), hex_addr(h.pc),
                h.mode.name, state, instr_fmt,
            )
        self._console.print(tbl)

    def _show_hart_detail(self, hart_id: int) -> None:
        h = self._emu.harts[hart_id]
        halted_note = " [red]已暂停 — 不可恢复陷态[/]" if h._halted else ""
        instr_fmt = fmt_instr_count(h._total_instrs)
        tbl = Table(
            title=f"Hart {hart_id}  指令计数: {instr_fmt}{halted_note}",
            border_style="blue",
        )
        tbl.add_column("Field", style="cyan")
        tbl.add_column("Value", style="green")
        tbl.add_row("PC", hex_addr(h.pc))
        tbl.add_row("Mode", h.mode.name)
        tbl.add_row("mstatus", hex_addr(h.mstatus_val))
        tbl.add_row("mie / mpie / mpp", f"{h.mie}, {h.mpie}, {h.mpp.name}")
        tbl.add_row("mepc", hex_addr(h.mepc_val))
        tbl.add_row("mcause", hex_addr(h.mcause_val))
        tbl.add_row("mtval", hex_addr(h.mtval_val))
        tbl.add_row("mtvec", hex_addr(h.mtvec_val))
        tbl.add_row("satp", hex_addr(h.satp_val))
        tbl.add_row(
            "reservation",
            f"valid={h.reservation_valid}, addr=0x{h.reservation_addr:x}",
        )
        self._console.print(tbl)
