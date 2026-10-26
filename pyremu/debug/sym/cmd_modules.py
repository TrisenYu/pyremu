#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""sym detail / sym rm / sym info."""

from __future__ import annotations

from rich.markup import escape
from rich.table import Table
from rich.text import Text

from pyremu.debug.exception import UsageError
from pyremu.debug.sym import cmd_files, model, registry
from pyremu.debug.utils import fmt_display_name


def sym_detail(dbg, rest: str) -> None:
    """不带参数时列出已加载模块; 带地址时列出覆盖该地址的模块."""
    registry.refresh_modules(dbg)
    text = rest.strip()
    if not text:
        _print_module_table(dbg)
        return

    addr = _resolve_detail_addr(dbg, text)
    if addr is None:
        return
    _print_cover_table(dbg, addr)


def sym_rm(dbg, rest: str) -> None:
    """按序号或文件名移除模块."""
    try:
        selector = cmd_files.parse_name_arg(rest)
    except UsageError as exc:
        dbg._err(escape(str(exc)))
        return
    if not selector:
        dbg._err("用法: sym rm <序号|文件名>")
        return

    mod = _pick_module(dbg, selector)
    if mod is None:
        return
    registry.remove_module(dbg, mod)
    dbg._console.print(
        Text.assemble(("sym", "bold"), " 已移除 ", (fmt_display_name(mod.file_name), "green"))
    )


def sym_info(dbg) -> None:
    """显示外部符号文件信息."""
    if not dbg._sym_symbols:
        dbg._console.print("[dim]未加载外部调试符号[/]")
        if dbg._sym_mods:
            dbg._console.print(f"  符号模块: {len(dbg._sym_mods)} 个; 执行 sym detail 查看")
        return

    entry_va = min(dbg._sym_symbols.values())
    shown_path = escape(fmt_display_name(str(dbg._sym_path)))
    text = (
        f"[bold]sym info[/]\n"
        f"  文件: [green]{shown_path}[/]\n"
        f"  符号数: {len(dbg._sym_symbols):,}\n"
        f"  入口 VA: 0x{entry_va:x}\n"
        f"  load_offset: 0x{dbg._sym_load_offset:x}"
    )
    if dbg._sym_mods:
        text += f"\n  符号模块: {len(dbg._sym_mods)} 个; 执行 sym detail 查看"
    dbg._console.print(text)


def _resolve_detail_addr(dbg, text: str) -> int | None:
    """把 pc、寄存器名、符号名或数值解析为运行时地址."""
    addr = dbg._resolve_addr(text)
    if addr is None:
        dbg._err(f"无法解析地址或符号名: {text}")
    return addr


def _pick_module(dbg, selector: str) -> model.SymMod | None:
    """按序号或文件名选中模块; 文件名对应多个模块时要求改用序号."""
    mods = dbg._sym_mods
    if not mods:
        dbg._err("没有已加载的符号模块; 执行 sym load 加载")
        return None

    if selector.isdigit():
        number = int(selector, 10)
        if 1 <= number <= len(mods):
            return mods[number - 1]
        dbg._err(f"模块序号超出范围: {selector}; 当前共 {len(mods)} 个模块")
        return None

    matches = [mod for mod in mods if mod.file_name == selector]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        dbg._err(f"没有文件名为 {selector} 的模块")
        return None
    numbers = ", ".join(str(mods.index(mod) + 1) for mod in matches)
    dbg._err(f"文件名 {selector} 对应多个模块, 请使用序号: {numbers}")
    return None


def _base_text(mod: model.SymMod) -> str:
    """模块的物理基址与虚拟基址."""
    parts: list[str] = []
    if mod.pa_base is not None:
        parts.append(f"pa 0x{mod.pa_base:x}")
    if mod.va_base is not None:
        parts.append(f"va 0x{mod.va_base:x}")
    return " / ".join(parts) if parts else "-"


def _print_module_table(dbg) -> None:
    """列出全部已加载模块."""
    if not dbg._sym_mods:
        dbg._console.print("[dim]没有已加载的符号模块; 执行 sym load 加载[/]")
        return

    tbl = Table(title=f"符号模块 ({len(dbg._sym_mods)} 项)", border_style="blue")
    tbl.add_column("#", style="yellow", justify="right")
    tbl.add_column("状态", justify="center")
    tbl.add_column("文件名", style="cyan")
    tbl.add_column("基址")
    tbl.add_column("符号数", justify="right")
    tbl.add_column("覆盖范围")
    tbl.add_column("路径", overflow="fold")
    for number, mod in enumerate(dbg._sym_mods, 1):
        tbl.add_row(
            str(number),
            mod.status_name,
            Text(fmt_display_name(mod.file_name)),
            _base_text(mod),
            f"{mod.sym_count:,}",
            f"[0x{mod.cover_start:x}, 0x{mod.cover_end:x})",
            Text(fmt_display_name(mod.path)),
        )
    dbg._console.print(tbl)
    dbg._console.print("[dim]sym rm <序号|文件名> 移除模块[/]")


def _print_cover_table(dbg, addr: int) -> None:
    """列出覆盖 *addr* 的全部模块与各自的解析结果."""
    name = dbg._resolve_any_symbol_name(addr)
    shown = name if name else "(无)"
    dbg._console.print(
        Text.assemble("地址 ", (f"0x{addr:x}", "yellow"), " 的符号: ", (shown, "green"))
    )

    selected = model.select_module(registry.lookup_modules(dbg), addr)
    rows = _cover_rows(dbg, addr, selected)
    if not rows:
        dbg._console.print("[dim]没有符号模块覆盖该地址[/]")
        return

    tbl = Table(title="覆盖该地址的模块", border_style="blue")
    tbl.add_column("#", style="yellow", justify="right")
    tbl.add_column("状态", justify="center")
    tbl.add_column("文件名", style="cyan")
    tbl.add_column("空间", justify="center")
    tbl.add_column("覆盖范围")
    tbl.add_column("符号名")
    tbl.add_column("选中", justify="center")
    for number, mod, space, link, symbol, is_selected in rows:
        tbl.add_row(
            str(number),
            mod.status_name,
            Text(fmt_display_name(mod.file_name)),
            space,
            f"[0x{mod.cover_start:x}, 0x{mod.cover_end:x}) -> 0x{link:x}",
            Text(symbol) if symbol else "(无)",
            "是" if is_selected else "否",
        )
    dbg._console.print(tbl)


def _cover_rows(
    dbg,
    addr: int,
    selected: tuple[model.SymMod, str, str] | None,
) -> list[tuple[int, model.SymMod, str, int, str | None, bool]]:
    """收集覆盖 *addr* 的模块、地址空间、链接地址与选中状态."""
    rows: list[tuple[int, model.SymMod, str, int, str | None, bool]] = []
    for number, mod in enumerate(dbg._sym_mods, 1):
        for space in model.module_spaces(mod):
            link = model.link_addr_of(mod, addr, space)
            if link is None:
                continue
            symbol = mod.index.resolve_addr(link) if mod.index is not None else None
            is_selected = selected is not None and selected[0] is mod and selected[1] == space
            rows.append((number, mod, space, link, symbol, is_selected))
    return rows
