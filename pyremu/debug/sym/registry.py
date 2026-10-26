#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""已加载符号模块的清单: 登记、替换、移除与有效状态刷新."""

from __future__ import annotations

import os

from pyremu.debug.sym import loader, model


def assign_load_order(dbg, mod: "model.SymMod") -> None:
    """按加载次序编号."""
    mod.load_order = dbg._sym_next_order
    dbg._sym_next_order += 1


def register_module(dbg, mod: "model.SymMod") -> None:
    """登记模块; 同一路径的原有模块被替换."""
    previous = find_module_by_path(dbg, mod.path)
    if previous is not None:
        _drop_module(dbg, previous, is_clear_legacy=True)
    dbg._sym_mods.append(mod)


def find_module_by_path(dbg, path: str) -> "model.SymMod | None":
    target = os.path.abspath(path)
    for mod in dbg._sym_mods:
        if os.path.abspath(mod.path) == target:
            return mod
    return None


def remove_module(dbg, mod: "model.SymMod") -> None:
    """移除模块; 移除 load_kernel_symbols 登记的模块时同时清空主符号表."""
    _drop_module(dbg, mod, is_clear_legacy=True)


def remove_legacy_modules(dbg) -> None:
    """移除全部由 load_kernel_symbols 登记的模块并清空主符号表."""
    for mod in list(dbg._sym_mods):
        if not mod.is_verified:
            _drop_module(dbg, mod, is_clear_legacy=True)


def _drop_module(
    dbg,
    mod: "model.SymMod",
    *,
    is_clear_legacy: bool,
) -> None:
    if mod in dbg._sym_mods:
        dbg._sym_mods.remove(mod)
    if mod.index is not None:
        mod.index.close()
    if is_clear_legacy and not mod.is_verified:
        _clear_legacy_tables(dbg)


def _clear_legacy_tables(dbg) -> None:
    dbg._sym_symbols = {}
    dbg._sym_symbols_pa.clear()
    dbg._sym_ranges = []
    dbg._sym_ranges_pa.clear()
    dbg._sym_load_offset = 0
    dbg._sym_path = None
    dbg._pa_to_va.clear()


def refresh_modules(dbg) -> None:
    """指令计数变化后重新校验各模块, 首次失效时报告一次."""
    epoch = dbg._instr_count
    for mod in list(dbg._sym_mods):
        if not mod.is_verified or not mod.is_valid:
            continue
        if mod.verify_epoch == epoch:
            continue
        mod.verify_epoch = epoch
        if loader.verify_module(dbg, mod):
            continue
        mod.is_valid = False
        if not mod.has_reported_invalid:
            mod.has_reported_invalid = True
            dbg._warn(
                f"符号模块 {mod.file_name} 的校验样本与内存不一致, 已标记失效;"
                f" 重新执行 sym load 可恢复"
            )


def lookup_modules(dbg) -> list["model.SymMod"]:
    """刷新后返回参与查找的模块."""
    refresh_modules(dbg)
    return [mod for mod in dbg._sym_mods if mod.is_valid]
