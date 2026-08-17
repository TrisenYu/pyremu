# -*- coding: utf-8 -*-
"""配置辅助 — 读取环境变量或回退到 configs_gen 默认值.

Usage:
    from pyremu.configs_aux import cfg_str, cfg_bool

    diag_path = cfg_str("PYREMU_DIAG_LOG")          # str, env 优先, 否则 configs_gen
    verbose = cfg_bool("PYREMU_DIAG_VERBOSE")        # bool: ""/"0" -> False
"""

from __future__ import annotations

from functools import partial
import os
from pathlib import Path

from pyremu import configs_gen
from pyremu.utils.file_ops import path_join


def cfg_str(name: str) -> str:
    """读取环境变量 *name*, 未设置时回退到 configs_gen 中的同名默认值."""
    val = os.environ.get(name, "")
    if val != "":
        return val
    return str(getattr(configs_gen, name, ""))


def cfg_bool(name: str) -> bool:
    """布尔语义: env 为空 / "0" -> False; 其他 -> True.
    未设置时回退到 configs_gen 默认值."""
    val = os.environ.get(name, "")
    if val != "":
        return val != "0"
    default = str(getattr(configs_gen, name, "0"))
    return default != "0"


def cfg_is_set(name: str) -> bool:
    """环境变量 *name* 是否被显式设置 (非 configs_gen 默认)."""
    return name in os.environ

# ---- 构建产物路径 ----
# partial 以位置参数预绑定 path_join 的 dir_path: 无参调用返回目录
# Path, 传 *name* 返回目录内文件 Path (path_join 的 name 有默认值).

firmware_dir = partial(
    path_join,
    getattr(configs_gen, "FIRM_DIR", "build/firm-bin")
)

elf_dir = partial(
    path_join,
    getattr(configs_gen, "ELF_DIR", "build/elf")
)

test_elf_dir = partial(
    path_join,
    Path(__file__).resolve().parent.parent / "tests" / "build" / "elf",
)
