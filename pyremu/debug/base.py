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
import threading
import time
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
        self._sym_symbols: dict[str, int] | None = {}
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

        # Stdin daemon thread — 独立于处理器执行循环, 持续将 stdin 转发到
        # UART RX FIFO. 对照 QEMU chardev fd_chr_read_poll 模型:
        #   stdin -> os.read -> uart.preload() -> _update_plic_irq() -> PLIC 中断
        #   -> _wake_event.set() -> 主线程 WFI 睡眠唤醒 -> try_wfi_wakeup()
        # 处理器感知不到 daemon 的存在 — 它只看到 PLIC 中断信号.
        self._stdin_daemon_running: bool = False
        self._stdin_daemon_thread: threading.Thread | None = None

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
        self._paused = True
        self._console.print("\n暂停请求 — 当前指令完成后回到 REPL")
        try:
            self._emu.request_native_stop()
        except Exception:
            pass

    def _enter_repl_mode(self) -> None:
        """恢复到 REPL 的终端设置 (prompt_toolkit 自行管理 raw 模式)."""
        signal.signal(signal.SIGINT, self._sigint_repl)
        self._stop_stdin_daemon()
        if self._emu._termio is not None:
            self._emu._termio.stop()
        self._restore_term()
        self._emu._idle_poll_cb = None

    def _enter_run_mode(self) -> None:
        """切换到运行模式: SIGINT -> 暂停, cbreak stdin, Rust libc::write TX.

        ── TX: QEMU fd_chr_write 模型, 与批次零耦合 ──
        固件写 TXDATA -> Rust inline handler -> libc::write(1, &byte, 1).
        每字节即时输出, _console_echo=False 抑制 Python 双重输出.

        ── RX: daemon 线程 select(stdin) -> uart.preload() -> PLIC 中断 ──
        stdin daemon 独立于处理器执行循环持续读取, 与 WFI 零耦合.
        处理器仅看到 PLIC 外部中断信号 -> try_wfi_wakeup() 自然唤醒.
        """
        signal.signal(signal.SIGINT, self._sigint_run)

        if not self._stdin_forward:
            return

        if self._emu.uart is not None:
            self._emu.uart.clear_rx()

        self._emu._idle_poll_cb = self._idle_poll

        # 终端设为 cbreak (字符即时可读, 不经行缓冲)
        if os.isatty(self._stdin_fd):
            self._saved_term_attrs = termios.tcgetattr(self._stdin_fd)
            tty.setcbreak(self._stdin_fd)
            # prompt_toolkit 的 raw 模式清除了 ICRNL; setcbreak 只修改 lflag
            # (ICANON, ECHO), 不会恢复 iflag. 需显式置位, 否则 Enter 键的 \\r
            # 不会被宿主内核转换为 \\n, 客机 dash 将其视为非终止空白符.
            attrs = list(termios.tcgetattr(self._stdin_fd))
            if not (attrs[0] & termios.ICRNL):
                attrs[0] |= termios.ICRNL
                termios.tcsetattr(self._stdin_fd, termios.TCSANOW, attrs)

        self._start_stdin_daemon()

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
    #  UART TX — REPL 模式回调 (运行模式由 Rust libc::write 直写 stdout)
    # ----------------------------------------------------------

    @staticmethod
    def _uart_tx(text: str) -> None:
        """UART TX 回调: 写入 stdout 并立即刷新."""
        sys.stdout.write(text)
        sys.stdout.flush()

    # ----------------------------------------------------------
    #  Stdin daemon 线程 — 独立于处理器执行循环, 对照 QEMU fd_chr_read_poll
    # ----------------------------------------------------------

    def _stdin_daemon_loop(self) -> None:
        """后台 daemon: select(stdin) -> os.read -> uart.preload() -> PLIC 中断.

        与处理器 WFI 完全解耦: daemon 仅负责把 stdin 字节注入 UART RX FIFO,
        UART 内部的 ``_update_plic_irq()`` 自动置位 PLIC 中断, ``_wake_event``
        唤醒主线程的 WFI 睡眠。处理器看到的是标准的 PLIC 外部中断信号。
        """
        uart = self._emu.uart
        if uart is None:
            return
        stdin_fd = self._stdin_fd
        pending: bytes = b""
        while self._stdin_daemon_running:
            # 有 pending 时用短超时 select 继续读新 stdin, 避免卡在重试循环中丢弃新输入
            timeout = 0.005 if pending else 0.02
            try:
                ready, _, _ = select.select([stdin_fd], [], [], timeout)
            except (OSError, ValueError):
                break
            if not self._stdin_daemon_running:
                break
            # 先读新 stdin 数据, 追加到 pending
            if ready:
                try:
                    data = os.read(stdin_fd, 4096)
                except OSError:
                    break
                if data:
                    pending += data
            # 再尝试 preload pending
            if pending:
                try:
                    n = uart.preload(pending)
                except Exception:
                    n = 0
                if n > 0:
                    pending = pending[n:]
                    self._emu._wake_event.set()

    def _start_stdin_daemon(self) -> None:
        """启动 stdin daemon 线程 (在 cbreak 终端设置之后调用)."""
        if self._stdin_daemon_running:
            return
        if self._emu.uart is None:
            return
        self._stdin_daemon_running = True
        self._stdin_daemon_thread = threading.Thread(
            target=self._stdin_daemon_loop, daemon=True,
        )
        self._stdin_daemon_thread.start()

    def _stop_stdin_daemon(self) -> None:
        """停止 stdin daemon 线程 (在恢复终端之前调用)."""
        self._stdin_daemon_running = False
        if self._stdin_daemon_thread is not None:
            self._emu._wake_event.set()  # 唤醒 select 使其检查 running 标志
            self._stdin_daemon_thread.join(timeout=1.0)
            self._stdin_daemon_thread = None

    # ----------------------------------------------------------
    #  UART stdin 转发 (主线程 select+os.read, 每批次边界执行)
    # ----------------------------------------------------------

    def _feed_uart_stdin(self) -> bool:
        """非阻塞读取 stdin 并转发到 UART RX FIFO.

        daemon 线程已在后台 select(stdin)->os.read->uart.preload,
        此处仍保留原有 select+os.read 路径 — daemon 作为加速补充而非替代.
        """
        # daemon 活跃时也走到这里: 原有 select+os.read 路径不受影响
        # (daemon 读走后 select 返回空即 no-op)

        if self._emu._termio is not None and self._emu._termio.native_active:
            had_input = self._emu._termio.drain_rx()
            if had_input:
                self._emu._wake_event.set()
            self._flush_uart_if_present()
            return had_input

        uart = self._emu.uart
        if uart is None:
            return False
        had_input = False
        if self._stdin_pending:
            try:
                accepted = uart.preload(self._stdin_pending)
            except (ValueError, OSError):
                return had_input
            if accepted > 0:
                had_input = True
                self._stdin_pending = self._stdin_pending[accepted:]
            if self._stdin_pending:
                self._emu._wake_event.set()
                return True

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
        """UART TXDATA 写入已自动即时输出, 无需手动刷新."""

    def _idle_poll(self) -> bool:
        """WFI 空闲轮询回调 — 转发 stdin."""
        had_input = self._feed_uart_stdin()
        return had_input

    def _idle_poll_termio(self) -> bool:
        """WFI 空闲轮询回调 (Rust termio 线程模式).

        termio 线程已接管 stdin 读取和 TX->stdout 排空; 此处排空 RX 环形
        缓冲到 UART 模型, 并刷新行缓冲确保部分行及时进入 hart 日志文件。

        Returns:
            True 仅当有 stdin 数据被 preload (中断 WFI 睡眠, 立即同步
            PLIC 并重启 batch)。恒返回 True 会使 WFI 轮询循环每次立即
            退出 ->热自旋 100% CPU。
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
