#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0

"""共享属性声明 — 供 Pylance 理解 mixin 之间的跨类引用.

各 mixin 继承此类以获得:
1. 属性类型标注 — 跨 mixin 共享的所有实例属性
2. ``__getattr__`` — 告诉静态分析器任何未在本类中定义的属性访问都是合法的
   (由其他 mixin 在最终 ``Debugger`` 类中提供)

``__getattr__`` 定义了所有跨 mixin 方法调用的"无限制访问"语义,
Pylance 不再报告 "属性未知" 错误.  无需为每个方法单独写存根.
``if TYPE_CHECKING:`` 确保 ``__getattr__`` 只在静态分析时存在,
运行时不会影响 ``Debugger`` 的 MRO 方法解析.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import WordCompleter
    from prompt_toolkit.history import FileHistory
    from rich.console import Console

    from pyremu.debug.types import (
        Breakpoint,
        HartSnapshot,
        MemoryChange,
        StackFrame,
    )
    from pyremu.emulator import Emulator
    from pyremu.utils.parse_bin import FirmwareImage


class SharedMixinAttrs:
    """声明所有跨 mixin 共享的属性; 跨 mixin 方法通过 __getattr__ 放行."""

    # -- DebuggerBase 初始化 --
    # (hart 为 @property, 不在类属性中声明 — __getattr__ 可解析)
    _emu: Emulator
    _hart_id: int
    _image: FirmwareImage | None
    _load_offset: int
    _preload_path: str | None
    _kernel_path: str | None
    _kernel_addr: int

    _sym_symbols: dict[str, int] | None
    _sym_symbols_pa: dict[str, int]
    _sym_ranges: list[tuple[int, int, str]]
    _sym_ranges_pa: list[tuple[int, int, str]]
    _sym_load_offset: int
    _sym_path: str | None
    _pa_to_va: dict[int, int]

    _running: bool
    _sigint_count: int
    _paused: bool
    _terminated: bool

    _snapshot: HartSnapshot | None
    _mem_changes: list[MemoryChange]
    _instr_count: int
    _fdt_addr: int | None

    _console: Console

    _trap_displayed_mcause: int | None
    _last_command: str | None

    _disasm_next_addr: int | None
    _vdisasm_next_addr: int | None
    _disasm_ref_pc: int | None
    _disasm_base_step: int
    _disasm_past_terminator: bool

    _breakpoints: list[Breakpoint]
    _bp_mode: str
    _hart_paused: set[int]
    _bp_hit_this_run: set[tuple[str, int]]
    _prev_instr_csr_addr: int

    _watch_ranges: list[tuple[int, int]]
    _watch_hit: tuple[int, int, int, bytes] | None
    _watch_installed: bool

    _stack_frames: list[StackFrame]
    _current_frame_idx: int

    _history: FileHistory
    _completer: WordCompleter
    _session: PromptSession

    _MODE_COLORS: dict[str, str]

    _show_diag: bool

    # UART stdin 转发 (终端 raw mode 管理)
    _stdin_forward: bool
    _stdin_fd: int
    _saved_term_attrs: Any

    # ----------------------------------------------------------
    #  __getattr__ — 仅在 TYPE_CHECKING 时存在
    #
    #  告诉 Pylance: 通过 ``self`` 访问任何在 ``SharedMixinAttrs``
    #  中未定义的属性/方法都是合法的 (由其他 mixin 在最终
    #  ``Debugger`` 类的 MRO 中提供).
    #
    #  返回 ``Any`` 是为了避免方法签名验证带来的级联错误 —
    #  跨 mixin 方法的参数和返回类型由各 mixin 自己的实现决定.
    # ----------------------------------------------------------

    if TYPE_CHECKING:

        def __getattr__(self, name: str) -> Any: ...
