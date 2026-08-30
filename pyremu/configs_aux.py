# -*- coding: utf-8 -*-
"""配置辅助 — 读取环境变量或回退到 configs_gen 默认值.

Usage:
    from pyremu.configs_aux import cfg_str, cfg_bool

    diag_path = cfg_str("PYREMU_DIAG_LOG")          # str, env 优先, 否则 configs_gen
    verbose = cfg_bool("PYREMU_DIAG_VERBOSE")        # bool: ""/"0" -> False
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
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


@contextmanager
def force_config(name: str, value: object) -> Iterator[None]:
    """临时覆盖 configs_gen 中的配置项, 退出上下文时恢复原值.

    供测试项在局部强制切换编译期开关 (如 PYREMU_AIA), 不改动全局默认值.
    恢复在 finally 中执行, 测试中途抛异常也会还原, 避免污染后续用例.

    用法::

        with force_config("PYREMU_AIA", 0):
            emu = Emulator(num_harts=2)
    """
    old = getattr(configs_gen, name)
    setattr(configs_gen, name, value)
    try:
        yield
    finally:
        setattr(configs_gen, name, old)

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

examed_elf_dir = partial(
    path_join,
    str((Path(__file__).resolve().parent.parent / "tests" / "build" / "elf").absolute()),
)
