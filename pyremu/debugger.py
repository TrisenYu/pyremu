#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0

"""RISC-V 交互式调试器 (rvdb) — 兼容性薄封装.

全部实现已迁移至 pyremu.debug 包, 本文件仅向后兼容地 re-export 公开符号.
"""

from pyremu.debug import (
    Debugger,
    HartSnapshot,
    MemoryChange,
    MemWriteTracker,
    StackFrame,
)
from pyremu.debug.cli import debugger
from pyremu.debug.utils import MAX_INSTR_COUNT

__all__ = [
    "Debugger",
    "HartSnapshot",
    "MemoryChange",
    "MemWriteTracker",
    "StackFrame",
    "MAX_INSTR_COUNT",
    "debugger",
]

if __name__ == "__main__":
    debugger()
