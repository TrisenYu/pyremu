#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""DebuggerBase — 调试器核心: 共享状态初始化、hart 访问、输出、信号处理."""

from __future__ import annotations

import os
import select
import signal
import sys
import termios
import tty
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
from pyremu.core.trap_def import trap_cause_name
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
        self._pa_to_va: dict[int, int] = {}

        self._running: bool = False
        self._sigint_count: int = 0
        self._paused: bool = False
        self._terminated: bool = False

        # Pending stdin bytes — UART RX FIFO is only 8 bytes, so excess
        # bytes from os.read() are buffered here and fed on subsequent
        # _feed_uart_stdin calls.  Without this, fast typing / paste in
        # native batch mode silently drops bytes beyond the 8th.
        self._stdin_pending: bytes = b""

        # Diagnostic counters visible only when PYREMU_DIAG_VERBOSE=1
        self._show_diag: bool = os.environ.get("PYREMU_DIAG_VERBOSE") == "1"

        # UART stdin 转发 — 终端 raw 模式管理
        self._stdin_forward: bool = True
        self._stdin_fd: int = sys.stdin.fileno()
        self._saved_term_attrs: Any = None


        self._snapshot: HartSnapshot | None = None
        self._mem_changes: list[MemoryChange] = []
        self._instr_count: int = 0
        self._consecutive_stalls: int = 0
        self._fdt_addr: int | None = None

        # Rich console
        self._console: Console = Console(highlight=False)

        # 覆盖 UART TX 回调: 每次写入后刷新 stdout, 确保不以 \\n 结尾的
        # 部分行 (如 shell 提示符 "# "、字符回显) 立即显示, 而非滞留在
        # Python 的行缓冲中直到下一换行或 Ctrl+C。
        if self._emu.uart is not None:
            self._emu.uart._tx_callback = self._uart_tx

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
        self._has_non_addr_bps: bool = False  # 缓存: 是否有 Rust 不检查的 BP

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
        # 通知 Rust batch engine 尽快退出 (否则须等整批指令执行完)
        try:
            self._emu.request_native_stop()
        except Exception:
            pass

    def _enter_repl_mode(self) -> None:
        """恢复到 REPL 的终端设置 (prompt_toolkit 自行管理 raw 模式)."""
        signal.signal(signal.SIGINT, self._sigint_repl)
        # 停止 TerminalIO 后台线程 (Rust termio 线程会自行恢复终端属性)
        if self._emu._termio is not None:
            self._emu._termio.stop()
        self._restore_term()
        # 清除 idle 轮询回调 (REPL 模式下无需转发 stdin)
        self._emu._idle_poll_cb = None

    def _enter_run_mode(self) -> None:
        """切换到运行模式的终端设置: SIGINT -> 暂停, stdin -> cbreak.

        TX 与 RX 均走纯 Python 路径 (对照 TX 通路的 _uart_tx 回调写法):
        - RX: ``_feed_uart_stdin()`` 每批次后以 select+os.read 非阻塞读 stdin,
          经 ``uart.preload()`` 直接注入 UART RX FIFO。
        - TX: ``_native_flush_uart()`` → ``drain_tx_logs()`` 每批次后从 TX 环形
          缓冲归入 UART 行缓冲, ``_flush_uart_if_present()`` 即时刷出部分行
          (提示符 ``# ``、字符回显等) 到 stdout。
        - WFI 空闲: ``_idle_poll`` 继续转发 stdin + 刷新部分行。
        """
        signal.signal(signal.SIGINT, self._sigint_run)

        if not self._stdin_forward:
            return

        # 丢弃固件启动期间用户在键盘上误敲入的字符
        if self._emu.uart is not None:
            self._emu.uart.clear_rx()

        # 纯 Python 路径: stdin 转发 + 部分行刷新
        self._emu._idle_poll_cb = self._idle_poll

        try:
            if not os.isatty(self._stdin_fd):
                return
            self._saved_term_attrs = termios.tcgetattr(self._stdin_fd)
            tty.setcbreak(self._stdin_fd)
            # prompt_toolkit 的 raw 模式 (在进入 run 循环前) 已
            # 清除了 ICRNL; tty.setcbreak() 只修改 lflag (ICANON,
            # ECHO), 不会恢复 iflag。需显式置位 ICRNL, 否则 Enter
            # 键的 \r 不会被宿主内核转换为 \n, 客机 dash 将其视
            # 为非终止空白符, 用户需按两次回车才能触发命令执行。
            attrs = list(termios.tcgetattr(self._stdin_fd))
            if not (attrs[0] & termios.ICRNL):
                attrs[0] |= termios.ICRNL
                termios.tcsetattr(self._stdin_fd, termios.TCSANOW, attrs)
        except (OSError, termios.error):
            pass

    def _restore_term(self) -> None:
        """恢复 _enter_run_mode 保存的终端属性."""
        try:
            if self._saved_term_attrs is not None:
                termios.tcsetattr(
                    self._stdin_fd, termios.TCSANOW, self._saved_term_attrs
                )
                self._saved_term_attrs = None
        except (OSError, termios.error):
            pass

    # ----------------------------------------------------------
    #  UART TX — 即时刷新 stdout
    # ----------------------------------------------------------

    @staticmethod
    def _uart_tx(text: str) -> None:
        """UART TX 回调: 写入 stdout 并立即刷新.

        ``sys.stdout.write`` 对 TTY 使用行缓冲: 不含 ``\\n`` 的文本
        (如 shell 提示符 ``# ``、内核回显的字符) 会滞留在缓冲区直到
        下一换行或缓冲区满。此回调在每次写入后显式 flush, 确保所有
        UART 输出即时可见。
        """
        sys.stdout.write(text)
        sys.stdout.flush()

    # ----------------------------------------------------------
    #  UART stdin 转发
    # ----------------------------------------------------------

    def _feed_uart_stdin(self) -> bool:
        """非阻塞读取 stdin 并转发到 UART RX buffer.

        每次批次前调用; 有输入时同时唤醒全部 WFI hart,
        确保内核立即处理新到达的终端输入.

        若 Rust ``TermIO`` 后台线程在运行, 则从其 RX 环形缓冲排空
        (termio 线程已通过 ``fd_chr_read_poll`` 从 stdin fd 读取),
        避免两方竞争同一 fd 导致互相抢走字节 → 客机收不到输入.
        """
        if self._emu._termio is not None and self._emu._termio.native_active:
            had_input = self._emu._termio.drain_rx()
            if had_input:
                self._emu._wake_event.set()
            self._flush_uart_if_present()
            return had_input

        # 纯 Python 回退 (TermIO 未启动或已停止时)
        uart = self._emu.uart
        if uart is None:
            return False
        had_input = False
        # 1. 优先注入上次未排入 FIFO 的残留字节
        if self._stdin_pending:
            try:
                accepted = uart.preload(self._stdin_pending)
            except (ValueError, OSError):
                return had_input
            if accepted > 0:
                had_input = True
                self._stdin_pending = self._stdin_pending[accepted:]
            if self._stdin_pending:
                # FIFO 仍满 — 残留字节保留, 等下次调用
                self._emu._wake_event.set()
                return True

        # 2. 读取新 stdin 数据 (仅当无残留时, 避免积压)
        try:
            ready, _, _ = select.select([self._stdin_fd], [], [], 0)
        except OSError:
            return had_input
        if not ready:
            return had_input
        try:
            data = os.read(self._stdin_fd, 4096)
        except OSError:
            return had_input
        if not data:
            return had_input

        # 3. 注入 FIFO; 多余字节暂存
        try:
            accepted = uart.preload(data)
        except (ValueError, OSError):
            return had_input
        if accepted > 0:
            had_input = True
            if accepted < len(data):
                self._stdin_pending = data[accepted:]
        if had_input:
            self._emu._wake_event.set()
        return had_input

    def _flush_uart_if_present(self) -> None:
        """刷新 UART 行缓冲中不以 \\n 结尾的部分行 (如 shell 提示符)."""
        if self._emu.uart is not None:
            self._emu.uart.flush_all()

    def _idle_poll(self) -> bool:
        """WFI 空闲轮询回调 — 转发 stdin + 刷新 UART 部分行缓冲.

        由 ``_wfi_sleep_if_idle`` 在睡眠期间以 ~50ms 间隔调用。
        stdin 转发使客机在 WFI 等待期间能立即收到终端输入;
        flush_all 确保不以 \\n 结尾的残余行 (如 shell 提示符 "# ") 及时显示。

        Returns:
            True 若有 stdin 数据被 preload (中断 WFI 睡眠).
        """
        had_input = self._feed_uart_stdin()
        self._flush_uart_if_present()
        return had_input

    def _idle_poll_termio(self) -> bool:
        """WFI 空闲轮询回调 (Rust termio 线程模式).

        termio 线程已接管 stdin 读取和 TX→stdout 排空; 此处排空 RX 环形
        缓冲到 UART 模型, 并刷新行缓冲确保部分行及时进入 hart 日志文件。

        Returns:
            True 仅当有 stdin 数据被 preload (中断 WFI 睡眠, 立即同步
            PLIC 并重启 batch)。恒返回 True 会使 WFI 轮询循环每次立即
            退出 → 热自旋 100% CPU。
        """
        had_input = False
        if self._emu._termio is not None:
            had_input = self._emu._termio.drain_rx()
        self._flush_uart_if_present()
        return had_input

    def _idle_poll_flush(self) -> bool:
        """WFI 空闲轮询回调 (TerminalIO Python 回退线程模式).

        stdin 由 TerminalIO 的回退轮询线程直接 preload (它会 set 唤醒事件),
        此处不重复读取 stdin — 双读者会产生乱序; 仅刷新 UART 部分行缓冲。

        Returns:
            False — 无自身输入信号, 唤醒依赖回退线程的 wake_event.
        """
        self._flush_uart_if_present()
        return False

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
