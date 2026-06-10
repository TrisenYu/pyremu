#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0

"""工具函数: 文件操作, 错误处理, 固件解析, 反汇编."""

from pyremu.utils.disassem import disasm
from pyremu.utils.file_ops import (
    get_abs_filename_arr_from_dir,
    is_file,
    is_path_existed,
    reloc_path,
)
from pyremu.utils.parse_bin import (
    FirmwareImage,
    FirmwareSegment,
    detect_format,
    parse_firmware,
)
from pyremu.utils.wrapper import die_if_err, seize_err_if_any

__all__ = [
    "FirmwareImage",
    "FirmwareSegment",
    "detect_format",
    "die_if_err",
    "disasm",
    "get_abs_filename_arr_from_dir",
    "is_file",
    "is_path_existed",
    "parse_firmware",
    "reloc_path",
    "seize_err_if_any",
]
