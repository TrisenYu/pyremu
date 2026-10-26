#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""符号模块的数据结构与不依赖调试器状态的选取计算."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pyremu.utils.mask import mask64

if TYPE_CHECKING:
    from pyremu.debug.sym.elf_idx import ElfSymbolIndex


SPACE_PA = "pa"
SPACE_VA = "va"


@dataclass
class SymAvailItem:
    """候选目录中的一行, 即 sym ls 的一项."""

    number: int
    name: str
    path: str
    size: int
    is_readable: bool
    is_loaded: bool
    is_writable: bool = False
    is_executable: bool = False


@dataclass
class SymMod:
    """一份已加载的符号文件."""

    file_name: str
    path: str
    pa_base: int | None
    va_base: int | None
    cover_start: int
    cover_end: int
    sym_count: int
    is_verified: bool
    load_order: int
    verify_epoch: int
    index: "ElfSymbolIndex | None" = None
    is_valid: bool = True
    has_reported_invalid: bool = False
    anchor_link: int = 0
    sample_offset: int = 0
    sample_len: int = 0
    sample_data: bytes = field(default=b"", repr=False)

    @property
    def span(self) -> int:
        return self.cover_end - self.cover_start

    @property
    def status_name(self) -> str:
        if not self.is_verified:
            return "未校验"
        return "有效" if self.is_valid else "失效"


def module_spaces(mod: SymMod) -> tuple[str, ...]:
    """模块已知基址对应的地址空间."""
    spaces: list[str] = []
    if mod.pa_base is not None:
        spaces.append(SPACE_PA)
    if mod.va_base is not None:
        spaces.append(SPACE_VA)
    return tuple(spaces)


def link_addr_of(mod: SymMod, addr: int, space: str) -> int | None:
    """把 *space* 空间中的运行时地址换算为链接地址; 不在覆盖范围内时返回 None."""
    base = mod.pa_base if space == SPACE_PA else mod.va_base
    if base is None:
        return None
    link = addr - base + mod.cover_start
    if mod.cover_start <= link < mod.cover_end:
        return link
    return None


def _selection_key(mod: SymMod, has_pa: bool) -> tuple[int, int, int]:
    """覆盖范围最广者优先, 相同取先加载者, 仍相同取物理区间.

    选取时比较元组的大小, 故物理区间取 1, 虚拟区间取 0.
    """
    return (mod.span, -mod.load_order, 1 if has_pa else 0)


def select_module(mods: list[SymMod], addr: int) -> tuple[SymMod, str, str] | None:
    """选取覆盖 *addr* 且能解析出符号名的模块.

    Returns:
        (模块, 地址空间, 符号名); 没有模块能解析时返回 None.
    """
    best: tuple[SymMod, str, str] | None = None
    best_key: tuple[int, int, int] | None = None
    for mod in mods:
        if mod.index is None:
            continue
        for space in module_spaces(mod):
            link = link_addr_of(mod, addr, space)
            if link is None:
                continue
            name = mod.index.resolve_addr(link)
            if not name:
                continue
            key = _selection_key(mod, space == SPACE_PA)
            if best_key is None or key > best_key:
                best_key = key
                best = (mod, space, name)
    return best


def name_to_runtime_addr(
    mods: list[SymMod],
    name: str,
) -> tuple[int | None, int | None] | None:
    """按名称查找运行时地址.

    Returns:
        (物理地址, 虚拟地址); 未知的一侧为 None; 没有模块含该名称时返回 None.
    """
    best: tuple[int | None, int | None] | None = None
    best_key: tuple[int, int, int] | None = None
    for mod in mods:
        if mod.index is None:
            continue
        link = mod.index.lookup_name(name)
        if link is None:
            continue
        offset = link - mod.cover_start
        pa = mask64(mod.pa_base + offset) if mod.pa_base is not None else None
        va = mask64(mod.va_base + offset) if mod.va_base is not None else None
        key = _selection_key(mod, pa is not None)
        if best_key is None or key > best_key:
            best_key = key
            best = (pa, va)
    return best
