#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""RegisterMixin — GPR/CSR 读写命令.

依赖 DebuggerBase.
"""

from rich.table import Table

from pyremu.core.registers import gpr_alias, gpr_name
from pyremu.core.trap_def import trap_cause_name
from pyremu.debug._attrs import SharedMixinAttrs
from pyremu.debug.utils import hex_addr
from pyremu.utils.wrapper import seize_val_err


class RegisterMixin(SharedMixinAttrs):
    """GPR 和 CSR 读写."""

    # ----------------------------------------------------------
    #  GPR 查找
    # ----------------------------------------------------------

    @seize_val_err("无效的gpr索引值")
    def _find_gpr(self, name: str) -> int | None:
        """按 xN 或 ABI 名查找 GPR 索引 (0-31)."""
        if name.startswith("x"):
            idx = int(name[1:])
            if 0 <= idx <= 31:
                return idx
        # 按 ABI 名查找
        for i in range(32):
            if gpr_alias(i).lower() == name.lower():
                return i
        return None

    # ----------------------------------------------------------
    #  GPR 命令
    # ----------------------------------------------------------

    def cmd_regs(self) -> None:
        """显示全部 GPR (两列布局)."""
        h = self.hart
        tbl = Table(title=f"Hart {self._hart_id}  GPRs", border_style="blue")
        tbl.add_column("Reg", style="cyan", no_wrap=True)
        tbl.add_column("Value", style="green")
        tbl.add_column("Reg", style="cyan", no_wrap=True)
        tbl.add_column("Value", style="green")
        for i in range(16):
            lo_name, lo_alias = gpr_name(i), gpr_alias(i)
            hi_name, hi_alias = gpr_name(i + 16), gpr_alias(i + 16)
            tbl.add_row(
                f"{lo_name} ({lo_alias})", hex_addr(h.gprs[i]),
                f"{hi_name} ({hi_alias})", hex_addr(h.gprs[i + 16]),
            )
        self._console.print(tbl)

    def cmd_reg(self, raw_args: str) -> None:
        """读取 GPR: reg <name>  或  reg <n1>, <n2>, ... (逗号分隔多寄存器)."""
        h = self.hart
        wanted = [n.strip() for n in raw_args.split(",") if n.strip()]
        if not wanted:
            return

        idxs: list[int] = []
        for n in wanted:
            idx = self._find_gpr(n)
            if idx is None:
                self._err(f"未知寄存器: {n}")
                return
            idxs.append(idx)
        max_w = max(len(gpr_name(idx)) for idx in idxs)
        max_a = max(len(gpr_alias(idx)) for idx in idxs)
        lines = [
            f"  [cyan]{gpr_name(idx):<{max_w}}[/] "
            f"([dim]{gpr_alias(idx):<{max_a}}[/]) = "
            f"[green]{hex_addr(h.gprs[idx])}[/]" for idx in idxs
        ]
        self._console.print("\n".join(lines))

    @seize_val_err("无效值")
    def cmd_set(self, name: str, value: str) -> None:
        """写入 GPR: set <name> <value>."""
        h = self.hart
        idx = self._find_gpr(name)
        if idx is None:
            self._err(f"未知寄存器: {name}")
            return
        v = int(value, 0) & 0xFFFF_FFFF_FFFF_FFFF
        old = h.gprs[idx]
        h.gprs[idx] = v
        self._console.print(
            f"{gpr_name(idx)} ([cyan]{gpr_alias(idx)}[/]): "
            f"[yellow]{hex_addr(old)}[/] -> [green]{hex_addr(v)}[/]"
        )

    # ----------------------------------------------------------
    #  CSR 命令
    # ----------------------------------------------------------

    def cmd_csr(self, raw_args: str) -> None:
        """读取 CSR.

        csr <name>                  — 单个
        csr <name1>, <name2>, ...   — 多个 (逗号分隔), 对齐显示
        csr list                    — 列出所有可用 CSR 名称
        """
        h = self.hart

        if raw_args == "list":
            names = sorted(h.csrs.keys())
            self._console.print(f"可用 CSR: [dim]{', '.join(names)}[/]")
            return

        wanted = [n.strip() for n in raw_args.split(",") if n.strip()]
        if not wanted:
            return

        for n in wanted:
            if n not in h.csrs:
                self._err(f"未知 CSR: {n} (用 'csr list' 查看可用列表)")
                return

        max_w = max(len(n) for n in wanted)

        lines: list[str] = []
        for n in wanted:
            csr = h.csrs[n]
            line = (
                f"  [cyan]{n:<{max_w}}[/] = [green]{hex_addr(csr.val)}[/]"
                f" (dec: {csr.val})"
            )
            if n in ("mstatus", "mstatush"):
                line += f"  [dim]模式 [bold]{h.mode.name}[/][/]"
            elif n in ("mcause", "scause"):
                if csr.val != 0:
                    line += f"  [dim]{trap_cause_name(csr.val)}[/]"
                else:
                    line += "  [dim](无)[/]"
            lines.append(line)
        self._console.print("\n".join(lines))

    @seize_val_err("无效值")
    def cmd_csrw(self, name: str, value: str) -> None:
        """写入 CSR: csrw <name> <value>."""
        h = self.hart
        if name not in h.csrs:
            self._err(f"未知 CSR: {name} (用 'csr list' 查看可用列表)")
            return
        v = int(value, 0) & 0xFFFF_FFFF_FFFF_FFFF
        csr = h.csrs[name]
        old, csr.val = csr.val, v
        self._console.print(
            f"[cyan]{name}[/]: [yellow]{hex_addr(old)}[/] -> [green]{hex_addr(v)}[/]"
        )
