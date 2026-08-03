# -*- coding: utf-8 -*-
"""配置辅助 — 读取环境变量或回退到 configs_gen 默认值.

Usage:
    from pyremu.configs_aux import cfg_str, cfg_bool

    diag_path = cfg_str("PYREMU_DIAG_LOG")          # str, env 优先, 否则 configs_gen
    verbose = cfg_bool("PYREMU_DIAG_VERBOSE")        # bool: ""/"0" -> False
"""

from __future__ import annotations

import os

from pyremu import configs_gen


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
