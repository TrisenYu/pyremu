#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""SymCmdOps — sym 命令的分发、帮助文本与固件符号输出."""

from __future__ import annotations

from rich.table import Table
from rich.text import Text

from pyremu import configs_gen
from pyremu.debug.attrs import SharedAttrs
from pyremu.debug.sym import cmd_files, cmd_modules
from pyremu.debug.types import ImageKind
from pyremu.debug.utils import group_order

# 按前缀过滤的结果超过该条目数时改为列出下一级子组
SYMBOL_SUBGROUP_THRESHOLD = 50

SYM_HELP_ROWS = (  # 帮助文本集中在此处: 每行给出子命令用法与一句说明
    ("chdir <绝对路径>", "记录候选目录; 省略参数时显示当前目录"),
    ("pwd", "显示所有符号表共用的候选目录"),
    ("ls", "列出候选目录中的普通文件并编号"),
    ("load <#n|文件名> [pa:<addr>|va:<addr>]", "加载符号文件, 校验内容后登记为模块"),
    ("detail [地址|pc|符号名]", "列出已加载模块, 或列出覆盖某地址的模块"),
    ("rm <序号|文件名>", "移除符号模块"),
    ("info", "显示外部符号文件信息"),
    ("list [filt]", "按前缀过滤固件符号"),
    ("help", "显示本帮助"),
)


class SymCmdOps(SharedAttrs):
    """sym 命令的分发与固件符号输出."""

    def cmd_sym(self, args: str) -> None:
        """分发 sym 子命令."""
        parts = args.split(maxsplit=1)
        sub = parts[0].lower() if parts and parts[0] else ""
        rest = parts[1] if len(parts) > 1 else ""

        if sub == "chdir":
            cmd_files.sym_chdir(self, rest)
        elif sub == "pwd":
            cmd_files.sym_pwd(self)
        elif sub == "ls":
            cmd_files.sym_ls(self)
        elif sub == "load":
            cmd_files.sym_load(self, rest)
        elif sub == "detail":
            cmd_modules.sym_detail(self, rest)
        elif sub == "rm":
            cmd_modules.sym_rm(self, rest)
        elif sub == "info":
            cmd_modules.sym_info(self)
        elif sub == "list":
            self.cmd_symbols(rest)
        elif sub == "help":
            print_sym_help(self)
        else:
            self.cmd_symbols(args)

    # ----------------------------------------------------------
    #  固件符号
    # ----------------------------------------------------------

    def cmd_symbols(self, filter_str: str = "") -> None:
        """列出固件符号表."""
        if self._image is None:
            self._warn("无可用的符号表 (非 ELF 文件)")
            return

        syms = self._image.symbols
        if not syms:
            self._console.print("[dim](符号表为空)[/]")
            return

        if not filter_str:
            self._print_symbol_groups(syms)
            return

        entries = self._filter_symbol_entries(syms, filter_str)
        if not entries:
            self._console.print(f"[dim]无以前缀 '{filter_str}' 开头的符号[/]")
            return

        if len(entries) > SYMBOL_SUBGROUP_THRESHOLD:
            self._print_symbol_subgroups(entries, filter_str)
            return
        self._print_symbol_rows(entries, filter_str)

    def _print_symbol_groups(self, syms: dict[str, int]) -> None:
        """按首字符统计固件符号."""
        groups: dict[str, int] = {}
        for name in syms:
            key = name[0] if name else "?"
            groups[key] = groups.get(key, 0) + 1
        sorted_keys = sorted(groups.keys(), key=group_order)

        tbl = Table(
            title=f"符号分组 ({len(syms)} 项, {len(sorted_keys)} 组)",
            border_style="blue",
            expand=True,
        )
        tbl.add_column("Prefix", style="cyan")
        tbl.add_column("Count", style="yellow", justify="right")
        for key in sorted_keys:
            tbl.add_row(f"{key}...", str(groups[key]))
        self._console.print(tbl)
        self._console.print("[dim]输入 sym <prefix> 查看具体符号 (如 sym a, sym f, sym _)[/]")

    @staticmethod
    def _filter_symbol_entries(
        syms: dict[str, int],
        filter_str: str,
    ) -> list[tuple[str, int]]:
        """按前缀筛选固件符号, 按地址升序返回 (名称, 地址)."""
        lowered = filter_str.lower()
        entries = [
            (name, addr) for name, addr in syms.items() if name.lower().startswith(lowered)
        ]
        entries.sort(key=lambda item: item[1])
        return entries

    def _print_symbol_subgroups(
        self,
        entries: list[tuple[str, int]],
        filter_str: str,
    ) -> None:
        """条目过多时按前缀的下一级字符分组."""
        prefix_len = len(filter_str)
        subgroups: dict[str, int] = {}
        for name, _addr in entries:
            next_key = name if len(name) <= prefix_len else name[: prefix_len + 1]
            subgroups[next_key] = subgroups.get(next_key, 0) + 1

        def _calc_order(x: str) -> tuple[int, str]:
            if not x:
                return (5, x)
            last = x[-1].lower() if len(x) > prefix_len else x.lower()
            return group_order(last)

        sorted_keys = sorted(subgroups.keys(), key=_calc_order)
        tbl = Table(
            title=f"符号: '{filter_str}' ({len(entries)} 项, {len(sorted_keys)} 子组)",
            border_style="blue",
            expand=True,
        )
        tbl.add_column("Prefix", style="cyan")
        tbl.add_column("Count", style="yellow", justify="right")
        for key in sorted_keys:
            if not key:
                continue
            tbl.add_row(f"{filter_str}[bold]{key[prefix_len:]}[/]...", str(subgroups[key]))
        self._console.print(tbl)

        second = sorted_keys[1] if len(sorted_keys) > 1 else sorted_keys[0]
        self._console.print(
            f"[dim]输入 sym <prefix> 继续深入 (如 sym {sorted_keys[0]}, sym {second})[/]"
        )

    def _print_symbol_rows(
        self,
        entries: list[tuple[str, int]],
        filter_str: str,
    ) -> None:
        """逐个列出固件符号的地址与名称."""
        tbl = Table(
            title=f"符号: '{filter_str}' ({len(entries)} 项)",
            border_style="blue",
            expand=True,
        )
        tbl.add_column("Address", style="yellow")
        tbl.add_column("Name", style="cyan")
        for name, addr in entries:
            tbl.add_row(f"0x{addr:016x}", name)
        self._console.print(tbl)


def print_sym_help(dbg) -> None:
    """输出符号子命令总表与地址前缀的用法."""
    dbg._console.print(Text("用法: sym <子命令> [参数]"))
    dbg._console.print()
    tbl = Table(box=None, show_header=False, pad_edge=False, expand=False)
    tbl.add_column(style="cyan", no_wrap=True)
    tbl.add_column()
    for usage, desc in SYM_HELP_ROWS:
        tbl.add_row(Text(f"  {usage}"), Text(desc))
    dbg._console.print(tbl)
    dbg._console.print()
    dbg._console.print(
        Text(
            "地址前缀: pa:<addr> 为物理地址, va:<addr> 为虚拟地址;"
            " 地址写作十六进制, 省略地址时按链接地址部署",
            style="dim",
        )
    )
    dbg._console.print(Text(_kernel_addr_hint(dbg), style="dim"))


def _kernel_addr_hint(dbg) -> str:
    """按当前配置给出内核 Image 的预载地址."""
    kernel_image = dbg._bins_images.get(ImageKind.KERNEL)
    if kernel_image is None:
        kernel_pa = dbg._emu.bus.ram_base + configs_gen.FW_RESERVED_MEM_SIZE
    else:
        kernel_pa = kernel_image.start
    return f"推荐地址: 内核 vmlinux 用 pa:0x{kernel_pa:x} 加载"
