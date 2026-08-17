#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0

"""RISC-V 交互式调试器 (rvdb)
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
