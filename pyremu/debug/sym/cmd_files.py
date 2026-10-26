#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""sym chdir / sym ls / sym load."""

from __future__ import annotations

from pathlib import Path
import shlex

from rich.markup import escape
from rich.table import Table
from rich.text import Text

from pyremu.debug.exception import UsageError
from pyremu.debug.sym import loader, model, registry
from pyremu.debug.utils import fmt_display_name, parse_hex_addr
from pyremu.utils.file_ops import (
    file_size_bytes,
    is_file_executable,
    is_file_readable,
    is_file_writable,
    iter_regular_files,
)

LOAD_USAGE = "用法: sym load <#n|文件名> [pa:<addr>|va:<addr>]"
CHDIR_USAGE = "用法: sym chdir <绝对路径>; 省略参数时显示当前候选目录"
ABSOLUTE_HINT = "候选目录须写成绝对路径, 以 / 开头"
ADDR_USAGE = (
    "符号文件所作用的内存区域起始位置写作 pa:<addr> 或 va:<addr>,"
    " 其中 <addr> 为十六进制数, 如 pa:0x80400000"
)
QUOTE_HINT = "文件名含空白时用引号包围"
CANDIDATE_HINT = "先执行 sym chdir <目录> 与 sym ls"


def split_quoted(rest: str) -> list[str]:
    """按空白切分参数; 引号包围的字符串作为单个参数, 引号内支持转义."""
    try:
        return shlex.split(rest)
    except ValueError:
        raise UsageError(f"引号不配对: {rest}")


def parse_name_arg(rest: str) -> str:
    """解析只接受一个名字的参数.

    加引号时按引号内容解析; 未加引号时按原文作为目录或文件名, 允许其中的空白.
    """
    tokens = split_quoted(rest)
    if not tokens:
        return ""
    if len(tokens) == 1:
        return tokens[0]
    return rest.strip()


def sym_chdir(dbg, rest: str) -> None:
    """记录候选目录; 省略参数时显示当前目录."""
    try:
        target = parse_name_arg(rest)
    except UsageError as exc:
        dbg._err(escape(str(exc)))
        return

    if not target:
        if dbg._sym_mod_dir:
            dbg._console.print(
                Text.assemble("候选目录: ", (fmt_display_name(dbg._sym_mod_dir), "green"))
            )
        else:
            dbg._console.print(Text("未设置候选目录", style="dim"))
        return

    if not Path(target).is_absolute():
        dbg._err(escape(f"{ABSOLUTE_HINT}: {fmt_display_name(target)}; {CHDIR_USAGE}"))
        return

    dbg._sym_mod_dir = target
    _clear_candidates(dbg)
    dbg._console.print(
        Text.assemble(
            "候选目录设为 ",
            (fmt_display_name(target), "green"),
            "; 执行 sym ls 列出其中的文件",
        )
    )


def sym_pwd(dbg) -> None:
    """显示所有符号表共用的候选目录."""
    if not dbg._sym_mod_dir:
        dbg._console.print(Text("未设置候选目录", style="dim"))
        return
    dbg._console.print(Text(fmt_display_name(dbg._sym_mod_dir)))


def sym_ls(dbg) -> None:
    """列出候选目录中的普通文件并缓存清单."""
    dir_path = dbg._sym_mod_dir
    if not dir_path:
        dbg._err(f"未设置候选目录, {CANDIDATE_HINT}")
        return

    try:
        paths = iter_regular_files(dir_path)
    except OSError as exc:
        _clear_candidates(dbg)
        dbg._err(escape(f"无法列出目录 {fmt_display_name(dir_path)}: {exc.strerror or exc}"))
        return

    items = [_make_item(dbg, number, path) for number, path in enumerate(paths, 1)]
    dbg._sym_avail_items = items
    _print_candidates(dbg, items, dir_path)


def sym_load(dbg, rest: str) -> None:
    """从候选清单选中文件, 部署并校验后登记为模块."""
    try:
        selector, addr_text = _split_load_args(rest)
        space, base = _parse_addr_spec(addr_text)
        item = _pick_candidate(dbg, selector)
    except UsageError as exc:
        dbg._err(escape(str(exc)))
        return

    mod = loader.load_symbol_module(dbg, item.path, space, base)
    if mod is None:
        return

    registry.register_module(dbg, mod)
    if mod.pa_base is not None and mod.va_base is not None:
        dbg._pa_to_va[mod.pa_base] = mod.va_base
    seq = dbg._sym_mods.index(mod) + 1
    dbg._console.print(f"[dim]模块序号 {seq}; 执行 sym detail 查看[/]")


def _split_load_args(rest: str) -> tuple[str, str]:
    """拆出选择项与地址说明; 选择项为空时报告用法."""
    tokens = split_quoted(rest)
    if not tokens:
        raise UsageError(LOAD_USAGE)
    if len(tokens) > 2:
        raise UsageError(f"{QUOTE_HINT}, 或改用编号; {LOAD_USAGE}")
    if len(tokens) > 1 and not tokens[1].strip():
        raise UsageError(ADDR_USAGE)
    addr_text = tokens[1] if len(tokens) > 1 else ""
    return tokens[0], addr_text


def _pick_candidate(dbg, selector: str) -> model.SymAvailItem:
    """按编号或文件名从候选清单中选中一项."""
    if "/" in selector or "\\" in selector:
        raise UsageError(f"sym load 只接受候选清单中的编号或文件名; {CANDIDATE_HINT}")
    if dbg._sym_mod_dir is None:
        raise UsageError(f"未设置候选目录, {CANDIDATE_HINT}")
    if not dbg._sym_avail_items:
        raise UsageError("候选清单为空; 执行 sym ls 刷新清单")

    item = _find_candidate(dbg._sym_avail_items, selector)
    _check_candidate(item)
    return item


def _find_candidate(items: list[model.SymAvailItem], selector: str) -> model.SymAvailItem:
    """按 #n 或文件名查找候选项."""
    if selector.startswith("#"):
        return _candidate_by_number(items, selector)
    for item in items:
        if item.name == selector:
            return item
    raise UsageError(f"候选清单中没有文件 {selector}; 执行 sym ls 刷新清单")


def _candidate_by_number(
    items: list[model.SymAvailItem],
    selector: str,
) -> model.SymAvailItem:
    try:
        number = int(selector[1:], 10)
    except ValueError:
        raise UsageError(f"编号不是整数: {selector}")
    for item in items:
        if item.number == number:
            return item
    raise UsageError(f"候选清单中没有编号 {selector}; 有效编号为 1 至 {len(items)}")


def _check_candidate(item: model.SymAvailItem) -> None:
    """确认选中项可读且仍然存在."""
    if not Path(item.path).exists():
        raise UsageError(f"文件不存在: {item.path}; 执行 sym ls 刷新清单")
    if not item.is_readable or not is_file_readable(item.path):
        raise UsageError(f"文件不可读: {item.path}")


def _parse_addr_spec(addr_text: str) -> tuple[str | None, int | None]:
    """解析 pa:<addr> 或 va:<addr>; 省略时返回 (None, None)."""
    text = addr_text.strip()
    if not text:
        return None, None

    prefix, sep, value = text.partition(":")
    prefix = prefix.lower()
    if not sep or prefix not in (model.SPACE_PA, model.SPACE_VA):
        raise UsageError(ADDR_USAGE)
    return prefix, parse_hex_addr(value)


def _make_item(dbg, number: int, path: Path) -> model.SymAvailItem:
    """为候选目录中的一项建立清单条目."""
    try:
        size = file_size_bytes(path)
    except OSError:
        size = 0
    path_text = str(path)
    return model.SymAvailItem(
        number=number,
        name=path.name,
        path=path_text,
        size=size,
        is_readable=is_file_readable(path),
        is_writable=is_file_writable(path),
        is_executable=is_file_executable(path),
        is_loaded=registry.find_module_by_path(dbg, path_text) is not None,
    )


def _clear_candidates(dbg) -> None:
    dbg._sym_avail_items = []


def _print_candidates(dbg, items: list[model.SymAvailItem], dir_path: str) -> None:
    """输出候选清单."""
    shown_dir = fmt_display_name(dir_path)
    if not items:
        dbg._console.print(Text(f"目录 {shown_dir} 中没有普通文件", style="dim"))
        return

    dbg._console.print(Text(f"候选目录 {shown_dir} 中有 {len(items)} 个普通文件"))
    tbl = Table(border_style="blue")
    tbl.add_column("#", style="yellow", justify="center", width=len(str(len(items))))
    tbl.add_column("文件名", style="cyan", justify="center", no_wrap=True)
    tbl.add_column("大小", justify="center")
    tbl.add_column("rwx-load", justify="center")
    for item in items:
        tbl.add_row(
            Text(str(item.number), justify="right"),
            Text(fmt_display_name(item.name), justify="left"),
            Text(f"{item.size:,} B", justify="right"),
            Text(_perm_text(item)),
        )
    dbg._console.print(tbl)
    legend = escape("rwx-load: r 可读, w 可写, x 可执行, y 已加载, n 未加载")
    dbg._console.print(f"[dim]{legend}[/]")
    hint = escape(
        "sym load <#n|文件名> [pa:<addr>|va:<addr>] 加载候选文件; 文件名含空白时用引号包围"
    )
    dbg._console.print(f"[dim]{hint}[/]")


def _perm_text(item: model.SymAvailItem) -> str:
    """渲染权限列: 前三位为 r/w/x, 末位为加载状态 y 或 n."""
    readable = "r" if item.is_readable else "-"
    writable = "w" if item.is_writable else "-"
    executable = "x" if item.is_executable else "-"
    loaded = "y" if item.is_loaded else "n"
    return readable + writable + executable + loaded
