#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""RISC-V 交互式调试器 (rvdb) — 模块化实现.

通过 mixin 多重继承组合为完整的 Debugger 类, 与 pyremu.debugger 等价.
"""

from pyremu.debug.base import DebuggerBase
from pyremu.debug.breakpoint import BreakpointMixin
from pyremu.debug.dispatch import DispatchMixin
from pyremu.debug.exec import ExecutionMixin
from pyremu.debug.mem import MemoryMixin
from pyremu.debug.mmu_view import MmuViewMixin
from pyremu.debug.reg import RegisterMixin
from pyremu.debug.status import StatusMixin
from pyremu.debug.stk_frame import StackWalkMixin
from pyremu.debug.sym import SymbolMixin
from pyremu.debug.tlb_cache import TlbCacheMixin
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
    trap_cause_name,
)


class Debugger(
    DispatchMixin,
    StatusMixin,
    StackWalkMixin,
    SymbolMixin,
    MmuViewMixin,
    TlbCacheMixin,
    MemoryMixin,
    RegisterMixin,
    BreakpointMixin,
    ExecutionMixin,
    DebuggerBase,
):
    """RISC-V 交互式调试器 — 所有功能域通过 mixin 组合."""

    # Backward-compatible static helpers — 原始 debugger.py 暴露为类方法.
    _hex = staticmethod(lambda v: hex_addr(v, styled=False))
    _fmt_size = staticmethod(fmt_size)
    _fmt_instr_count = staticmethod(fmt_instr_count)
    _hexdump_bytes = staticmethod(TlbCacheMixin._hexdump_bytes)
    _ip_bits = staticmethod(ip_bits)
    _trap_cause_name = staticmethod(trap_cause_name)
    _tlb_page_size = staticmethod(TlbCacheMixin._tlb_page_size)
    _colorize_asm = staticmethod(MemoryMixin._colorize_asm)
    _ctrl_flow_kind = staticmethod(MemoryMixin._ctrl_flow_kind)
    _ctrl_flow_kind_compressed = staticmethod(MemoryMixin._ctrl_flow_kind_compressed)
    _resolve_symbol = staticmethod(SymbolMixin._resolve_symbol)


__all__ = [
    "Debugger",
    "HartSnapshot",
    "MemoryChange",
    "MemWriteTracker",
    "StackFrame",
]
