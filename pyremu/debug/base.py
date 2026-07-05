#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""DebuggerBase — 调试器核心: 共享状态初始化、hart 访问、输出、信号处理."""

from __future__ import annotations

import io
import select
import signal
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyremu.core.decoder import Hart

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import FileHistory
from prompt_toolkit.styles import Style
from rich.console import Console

from pyremu.core.registers import (
    register_csr,
    register_fpr,
    register_gpr,
)
from pyremu.core.trap import trap_cause_name
from pyremu.debug._attrs import SharedMixinAttrs
from pyremu.debug.types import (
    Breakpoint,
    HartSnapshot,
    MemoryChange,
    StackFrame,
)
from pyremu.debug.utils import (
    hex_addr,
    trim_history,
)
from pyremu.emulator import Emulator
from pyremu.utils.parse_bin import FirmwareImage


def _yield_cpu(interval: float = 0.001) -> None:
    """通过 ``select`` 在 stdin 上等待 *interval* 秒以让出 CPU."""
    try:
        select.select([sys.stdin], [], [], interval)
    except (io.UnsupportedOperation, TypeError, OSError):
        time.sleep(interval)


class DebuggerBase(SharedMixinAttrs):
    """调试器核心基类 — 初始化所有共享状态, 提供输出和信号处理.

    各功能域作为独立 mixin 通过多重继承组合; 本类必须在 MRO 最后 (最右侧)
    以确保 ``__init__`` 优先初始化所有属性.
    """

    _DEFAULT_TIMEOUT = 3600.0
    _MODE_COLORS = {"M": "red", "S": "cyan", "U": "green", "H": "yellow", "D": "magenta"}

    def __init__(
        self,
        emulator: Emulator,
        hart_id: int = 0,
        image: FirmwareImage | None = None,
    ) -> None:
        self._emu: Emulator = emulator
        self._hart_id: int = hart_id
        self._image: FirmwareImage | None = image
        self._load_offset: int = 0
        self._preload_path: str | None = None
        self._kernel_path: str | None = None
        self._kernel_addr: int = 0

        # 外部调试符号
        self._sym_symbols: dict[str, int] = {}
        self._sym_symbols_pa: dict[str, int] = {}
        self._sym_ranges: list[tuple[int, int, str]] = []
        self._sym_ranges_pa: list[tuple[int, int, str]] = []
        self._sym_load_offset: int = 0
        self._sym_path: str | None = None

        self._running: bool = False
        self._sigint_count: int = 0
        self._paused: bool = False
        self._terminated: bool = False

        self._snapshot: HartSnapshot | None = None
        self._mem_changes: list[MemoryChange] = []
        self._instr_count: int = 0
        self._fdt_addr: int | None = None

        # Rich console
        self._console: Console = Console(highlight=True)

        self._trap_displayed_mcause: int | None = None
        self._last_command: str | None = None

        # disasm 状态
        self._disasm_next_addr: int | None = None
        self._vdisasm_next_addr: int | None = None
        self._disasm_ref_pc: int | None = None
        self._disasm_base_step: int = 0
        self._disasm_past_terminator: bool = False

        # 断点
        self._breakpoints: list[Breakpoint] = []
        self._bp_mode: str = "sync"
        self._hart_paused: set[int] = set()
        self._bp_hit_this_run: set[tuple[str, int]] = set()
        self._prev_instr_csr_addr: int = -1

        # 写监控
        self._watch_ranges: list[tuple[int, int]] = []
        self._watch_hit: tuple[int, int, int, bytes] | None = None
        self._watch_installed: bool = False

        # 栈帧
        self._stack_frames: list[StackFrame] = []
        self._current_frame_idx: int = 0

        # prompt_toolkit REPL
        self._history: FileHistory = FileHistory(str(Path.home() / ".pyremu_history"))
        trim_history(max_entries=10000)
        self._completer: WordCompleter = self._build_completer()
        self._session: PromptSession[str] = PromptSession(
            history=self._history,
            completer=self._completer,
            style=Style.from_dict({"prompt": "#00aa00 bold", "": "#cccccc"}),
        )

    # ----------------------------------------------------------
    #  hart 访问
    # ----------------------------------------------------------

    @property
    def hart(self) -> "Hart":
        return self._emu.harts[self._hart_id]

    # ----------------------------------------------------------
    #  输出
    # ----------------------------------------------------------

    def _warn(self, msg: str) -> None:
        self._console.print(f"[yellow]警告:[/] {msg}")

    def _err(self, msg: str) -> None:
        self._console.print(f"[red bold]错误:[/] {msg}")

    # ----------------------------------------------------------
    #  信号处理
    # ----------------------------------------------------------

    def _sigint_repl(self, _signum: int, _frame) -> None:
        raise KeyboardInterrupt

    def _sigint_run(self, _signum: int, _frame) -> None:
        self._sigint_count += 1
        payload = "\n[yellow]暂停请求 — 当前指令完成后回到 REPL[/]"
        if self._sigint_count != 1:
            payload = "\n[red bold]强制终止模拟循环[/]"
            self._terminated = True
        self._console.print(payload)
        self._paused = True

    def _enter_repl_mode(self) -> None:
        signal.signal(signal.SIGINT, self._sigint_repl)

    def _enter_run_mode(self) -> None:
        signal.signal(signal.SIGINT, self._sigint_run)

    # ----------------------------------------------------------
    #  Tab 补全
    # ----------------------------------------------------------

    def _build_completer(self) -> WordCompleter:
        words: list[str] = [
            "s", "step", "c", "continue", "r", "run",
            "undo", "rollback", "restart", "b", "bp",
            "regs", "gpr", "reg", "set", "w", "csr", "csrw",
            "pc", "mode", "mstatus", "tlb", "tlbflush",
            "cache", "satp", "pt", "mem", "vmem",
            "status", "info", "symbols", "sym", "disasm", "vdisasm",
            "stack", "show", "bt", "frame", "f",
            "hart", "h", "help", "?", "q", "quit", "exit",
        ]
        for r in register_gpr():
            words.append(r.name)
            if r.alias:
                words.append(r.alias)
        for r in register_fpr():
            words.append(r.name)
            if r.alias:
                words.append(r.alias)
        words.extend(register_csr().keys())
        words.append("list")
        return WordCompleter(words, ignore_case=True, sentence=True)

    # ----------------------------------------------------------
    #  陷态上下文
    # ----------------------------------------------------------

    def _show_trap_context(self, h) -> None:
        """hart 进入不可恢复陷态时的上下文摘要."""
        cause = h.mcause_val
        name = trap_cause_name(cause)
        is_int = (cause >> 63) & 1
        tag = "中断" if is_int else "异常"
        self._warn(f"Hart {h.id} 进入不可恢复陷态, 已暂停")
        self._console.print(
            f"  [red bold]{tag}[/] {name}  "
            f"mcause=0x{cause:x}  mepc={hex_addr(h.mepc_val)}  "
            f"mtval={hex_addr(h.mtval_val)}"
        )

    # ----------------------------------------------------------
    #  REPL 主循环 (依赖 _dispatch — 由 DispatchMixin 提供)
    # ----------------------------------------------------------

    def repl(self) -> None:
        """主交互循环."""
        self._enter_repl_mode()
        banner = (
            f"[bold]pyremu rvdb[/] — "
            f"{self._emu.num_harts} hart(s), "
            f"RAM {hex_addr(self._emu.bus._ram_base)}–"
            f"{hex_addr(self._emu.bus._ram_end)}, "
            f"固件: {hex_addr(self.hart.pc)}"
        )
        if self._fdt_addr:
            banner += f", FDT@0x{self._fdt_addr:x}"
        self._console.print(f"\n{banner}")
        self._console.print("输入 'help' 查看可用命令, '(enter)' 重复上一条.\n")

        self._running = False
        while True:
            try:
                prompt = (f"[bold red]rvdb:{self._hart_id}[/]"
                          f" ([cyan]{self.hart.mode.name}[/]) > ")
                user_input = self._session.prompt(prompt).strip()
            except (KeyboardInterrupt, EOFError):
                self._console.print("\n[dim]goodbye[/]")
                break

            if user_input == "":
                user_input = self._last_command or "status"
            self._last_command = user_input
            self._history.append_string(user_input)

            parts = user_input.split()
            if not parts:
                continue

            try:
                if not self._dispatch(parts):
                    break
            except KeyboardInterrupt:
                self._console.print("\n[dim]REPL 中断[/]")
            except Exception:
                self._console.print_exception()
