#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT

"""工具函数: 文件操作, 错误处理, 固件解析, 反汇编, 格式化."""

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
from pyremu.utils.str_aux import fmt_addr, fmt_hexdump
from pyremu.utils.wrapper import (
    die_if_err,
    print_exc_on_err,
    seize_err_if_any,
    seize_val_err,
    silent_on_err,
)
from pyremu.utils.disassem import disasm

__all__ = [
    "FirmwareImage",
    "FirmwareSegment",
    "detect_format",
    "die_if_err",
    "disasm",
    "fmt_addr",
    "fmt_hexdump",
    "get_abs_filename_arr_from_dir",
    "is_file",
    "is_path_existed",
    "parse_firmware",
    "print_exc_on_err",
    "reloc_path",
    "seize_err_if_any",
    "seize_val_err",
    "silent_on_err",
]
