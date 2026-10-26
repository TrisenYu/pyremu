#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Created at 2026/06/05 星期五 13:16:26
# Last modified at 2026/06/06 星期六 16:48:21
from __future__ import annotations

import os
from os import PathLike
from pathlib import Path

from pyremu.utils.wrapper import seize_err_if_any


def reloc_path(src: PathLike, *intermidate: str | PathLike) -> str:
    ret = Path(src)
    for mid in intermidate:
        ret = ret.joinpath(mid)
    return str(ret)


def is_file(path: PathLike | str) -> bool:
    return Path(path).is_file()


def is_path_existed(path: str) -> bool:
    return Path(path).exists()


@seize_err_if_any()
def get_abs_filename_arr_from_dir(dir_path: str | PathLike) -> list[Path]:
    return [f for f in Path(dir_path).iterdir() if f.is_file()]


def iter_regular_files(dir_path: str | PathLike) -> list[Path]:
    """非递归列出目录下的普通文件, 按名称升序.

    符号链接按其目标判定, 指向普通文件的列入. 目录不存在、不是目录、无读取权限时,
    异常向上传递.
    """
    entries = [p for p in Path(dir_path).iterdir() if p.is_file()]
    entries.sort(key=lambda p: p.name)
    return entries


def is_file_readable(path: str | PathLike) -> bool:
    """判定文件当前是否可读."""
    return os.access(path, os.R_OK)


def is_file_writable(path: str | PathLike) -> bool:
    """判定文件当前是否可写."""
    return os.access(path, os.W_OK)


def is_file_executable(path: str | PathLike) -> bool:
    """判定文件当前是否可执行."""
    return os.access(path, os.X_OK)


def file_size_bytes(path: str | PathLike) -> int:
    """返回文件的字节数."""
    return Path(path).stat().st_size


def path_join(dir_path: str | PathLike, name: str | None = None) -> Path:
    """拼接目录与文件名; *name* 为 None 时返回目录本身.

    返回 ``Path`` 而非 str — configs_aux 中的 partial 预绑定 ``dir_path``
    后即成为目录/文件路径工厂, 调用方需要 ``.exists()`` 等 pathlib API.
    """
    base = Path(dir_path)
    return base / name if name else base
