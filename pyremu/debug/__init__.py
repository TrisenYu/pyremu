#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""RISC-V 交互式调试器 (rvdb) — 模块化实现.

通过操作类的多重继承组合为完整的 Debugger 类, 与 pyremu.debugger 等价.
"""

from pyremu.core.trap_def import trap_cause_name
from pyremu.debug.base import DebuggerBase
from pyremu.debug.bins import BinsInfoOps
from pyremu.debug.breakpoint import BreakpointOps
from pyremu.debug.disk import DiskViewOps
from pyremu.debug.dispatch import DispatchOps
from pyremu.debug.exec import ExecutionOps
from pyremu.debug.mem import MemoryOps
from pyremu.debug.mmu_view import MmuViewOps
from pyremu.debug.reg import RegisterOps
from pyremu.debug.status import StatusOps
from pyremu.debug.stk_frame import StackWalkOps
from pyremu.debug.sym import SymCmdOps, SymLookupOps
from pyremu.debug.tlb_cache import TlbCacheOps
from pyremu.debug.types import (
    HartSnapshot,
    MemoryChange,
    MemWriteTracker,
    StackFrame,
)
from pyremu.debug.utils import (
    fmt_instr_count,
    fmt_size,
    hex_addr,
    ip_bits,
)


class Debugger(
    DispatchOps,
    StatusOps,
    StackWalkOps,
    SymLookupOps,
    SymCmdOps,
    BinsInfoOps,
    DiskViewOps,
    MmuViewOps,
    TlbCacheOps,
    MemoryOps,
    RegisterOps,
    BreakpointOps,
    ExecutionOps,
    DebuggerBase,
):
    """RISC-V 交互式调试器 — 所有功能域通过操作类组合."""

    # Backward-compatible static helpers — 原始 debugger.py 暴露为类方法.
    _hex = staticmethod(lambda v: hex_addr(v, styled=False))
    _fmt_size = staticmethod(fmt_size)
    _fmt_instr_count = staticmethod(fmt_instr_count)
    _hexdump_bytes = staticmethod(TlbCacheOps._hexdump_bytes)
    _ip_bits = staticmethod(ip_bits)
    _trap_cause_name = staticmethod(trap_cause_name)
    _tlb_page_size = staticmethod(TlbCacheOps._tlb_page_size)
    _colorize_asm = staticmethod(MemoryOps._colorize_asm)
    _ctrl_flow_kind = staticmethod(MemoryOps._ctrl_flow_kind)
    _ctrl_flow_kind_compressed = staticmethod(MemoryOps._ctrl_flow_kind_compressed)
    _resolve_symbol = staticmethod(SymLookupOps._resolve_symbol)


__all__ = [
    "Debugger",
    "HartSnapshot",
    "MemoryChange",
    "MemWriteTracker",
    "StackFrame",
]
