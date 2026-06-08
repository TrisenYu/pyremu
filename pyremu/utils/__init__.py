#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0

"""工具函数: 文件操作, 错误处理, ELF 解析."""

from pyremu.utils.file_ops import (
    get_abs_filename_arr_from_dir,
    is_file,
    is_path_existed,
    reloc_path,
)
from pyremu.utils.parse_bin import parse_bin
from pyremu.utils.wrapper import die_if_err, seize_err_if_any

__all__ = [
    "die_if_err",
    "get_abs_filename_arr_from_dir",
    "is_file",
    "is_path_existed",
    "parse_bin",
    "reloc_path",
    "seize_err_if_any",
]
