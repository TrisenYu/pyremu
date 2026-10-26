#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

"""DebuggerBase — 调试器核心: 共享状态初始化、hart 访问、输出、信号处理."""

from __future__ import annotations

import os
from pathlib import Path
import select
import signal
import sys
import termios
import threading
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from pyremu.core.decoder import Hart
    from pyremu.debug.sym.model import SymAvailItem, SymMod

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
from pyremu.debug.attrs import SharedAttrs
from pyremu.debug.types import (
    Breakpoint,
    HartSnapshot,
    ImageKind,
    LoadedImage,
    MemoryChange,
    StackFrame,
)
from pyremu.debug.utils import (
    hex_addr,
    trim_history,
)
from pyremu.emulator import Emulator
from pyremu.utils.parse_bin import FirmwareImage


def init_sym_mod_dir() -> str | None:
    """符号候选目录的初始值: 进程的当前工作目录; 取不到时取当前用户的主目录.

    工作目录已被删除时 ``os.getcwd`` 抛出 ``OSError``, 此时改用 ``HOME`` 环境变量;
    该变量也没有时返回 ``None``, 即未设置候选目录.
    """
    try:
        return os.getcwd()
    except OSError:
        return os.environ.get("HOME") or None


class DebuggerBase(SharedAttrs):
    """调试器核心基类 — 初始化所有共享状态, 提供输出和信号处理.

    各功能域作为独立的操作类通过多重继承组合; 本类必须在 MRO 最后 (最右侧)
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
        # 命令行装入的映像: 角色 -> 路径、物理起始地址与字节数
        self._bins_images: dict[ImageKind, LoadedImage] = {}

        # 外部调试符号
        self._init_symbol_state()

        self._running: bool = False
        self._sigint_count: int = 0
        self._terminated: bool = False

        self._stdin_pending: bytes = b""
        self._stdin_daemon_running: bool = False
        self._stdin_daemon_thread: threading.Thread | None = None
        # UART stdin 转发 — 终端 raw 模式管理
        self._stdin_forward: bool = True
        self._stdin_fd: int = sys.stdin.fileno()
        self._saved_term_attrs: Any = None

        self._snapshot: HartSnapshot | None = None
        self._mem_changes: list[MemoryChange] = []
        self._instr_count: int = 0
        self._fdt_addr: int | None = None

        # Rich console
        self._console: Console = Console(highlight=False)

        # 覆盖 UART TX 回调: 每次写入后刷新 stdout, 确保不以 \\n 结尾的
        # 部分行, 如 shell 提示符 "# " 与字符回显, 立即显示, 而非留在
        # Python 的行缓冲中直到下一换行或 Ctrl+C。
        if self._emu.uart:
            self._emu.uart._tx_callback = self._uart_tx

        self._trap_displayed_mcause: int | None = None
        self._last_command: str | None = None
        self._command_rest = ""

        self._init_disasm_state()
        self._init_breakpoint_state()

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
            style=Style.from_dict(
                {
                    "prompt": "#00aa00 bold",
                    "clock": "#6c6c6c",
                    "": "#cccccc",
                }
            ),
        )

    def _init_symbol_state(self) -> None:
        """外部调试符号、段范围与地址映射的初始状态."""
        self._sym_symbols: dict[str, int] | None = {}
        self._sym_symbols_pa: dict[str, int] = {}
        self._sym_ranges: list[tuple[int, int, str]] = []
        self._sym_ranges_pa: list[tuple[int, int, str]] = []
        self._sym_load_offset: int = 0
        self._sym_path: str | None = None
        self._pa_to_va: dict[int, int] = {}

        self._sym_avail_items: list[SymAvailItem] = []
        self._sym_mod_dir: str | None = init_sym_mod_dir()
        self._sym_mods: list[SymMod] = []
        self._sym_next_order: int = 0

    def _init_disasm_state(self) -> None:
        self._disasm_next_addr: int | None = None
        self._vdisasm_next_addr: int | None = None
        self._disasm_ref_pc: int | None = None
        self._disasm_base_step: int = 0
        self._disasm_past_terminator: bool = False

    def _init_breakpoint_state(self) -> None:
        """断点表的初始状态."""
        self._breakpoints: list[Breakpoint] = []
        self._bp_mode: str = "sync"
        self._hart_paused: set[int] = set()
        self._bp_hit_this_run: set[tuple[str, int]] = set()
        self._prev_instr_csr_addr: int = -1
        self._has_non_addr_bps: bool = False  # 缓存: 是否有 Rust 不检查的 BP

    @property
    def hart(self) -> "Hart":
        return self._emu.harts[self._hart_id]

    def _warn(self, msg: str) -> None:
        self._console.print(f"[yellow]警告:[/] {msg}")

    def _err(self, msg: str) -> None:
        self._console.print(f"[red bold]错误:[/] {msg}")

    def _sigint_repl(self, _signum: int, _frame) -> None:
        raise KeyboardInterrupt

    def _sigint_run(self, _signum: int, _frame) -> None:
        self._console.print("\n暂停请求 — 当前指令完成后回到 REPL")
        try:
            self._emu.notify_processor()
        except Exception:
            pass

    def _enter_repl_mode(self) -> None:
        """恢复到 REPL 的终端设置 (prompt_toolkit 自行管理 raw 模式)."""
        signal.signal(signal.SIGINT, self._sigint_repl)
        self._stop_stdin_daemon()
        if self._emu._termio:
            self._emu._termio.stop()
        self._restore_term()
        self._emu._stdin_forward_callback = None

    def _enter_run_mode(self) -> None:
        """切换到运行模式: Ctrl+Q 暂停, cbreak stdin 下 ISIG 关闭, Ctrl+C 透传.

        ── TX: 固件写 TXDATA, 随后经 Rust inline handler 调用 libc::write(1, &byte, 1).
        每字节即时输出, _console_echo=False 抑制 Python重复输出.

        ── RX: stdin daemon 独立于处理器执行循环持续读取.
        处理器仅看到 PLIC 外部中断信号, 由 try_wfi_wakeup() 唤醒.
        """
        # 终端 ISIG 关闭后键盘 Ctrl+C (0x03) 作为普通字节透传给受调试程序, 不经信号
        # 路径; Ctrl+Q 停止由 termio 的 notify_emu_stop 直连置位 stop_flag, 同样
        # 不经信号. 此 SIGINT handler 仅覆盖 stdin 未转发的场景 (终端仍生成 SIGINT).
        signal.signal(signal.SIGINT, self._sigint_run)

        if not self._stdin_forward:
            return

        if self._emu.uart:
            self._emu.uart.clear_rx()

        self._emu._stdin_forward_callback = self._idle_poll

        # 完整 raw 模式, 对照 termio/src/lib.rs 的 raw 设置:
        # - 关闭 ECHO/ICANON/IEXTEN/ISIG, 字符原样转发
        # - CS8: 8-bit 数据不丢高位, 对 backspace 0x7F 等关键
        # - ICRNL 保留: 把 \r 映射到 \n, 行终止兼容受调试程序控制台
        # - 关闭 IXON: Ctrl+Q/Ctrl+S 原样转发
        if os.isatty(self._stdin_fd):
            self._saved_term_attrs = termios.tcgetattr(self._stdin_fd)
            attrs = termios.tcgetattr(self._stdin_fd)
            # iflag: 清除输入转换; ICRNL 保留 (受调试程序控制台可能未初始化 \r->\n)
            attrs[0] = (
                attrs[0]
                & ~(
                    termios.IGNBRK
                    | termios.BRKINT
                    | termios.PARMRK
                    | termios.ISTRIP
                    | termios.INLCR
                    | termios.IGNCR
                    | termios.IXON
                )
            ) | termios.ICRNL
            attrs[3] &= ~(
                termios.ECHO | termios.ECHONL | termios.ICANON | termios.IEXTEN | termios.ISIG
            )
            # cflag: 8-bit 字符
            attrs[2] = (attrs[2] & ~termios.CSIZE) | termios.CS8
            attrs[2] &= ~termios.PARENB
            attrs[6][termios.VMIN] = 1
            attrs[6][termios.VTIME] = 0
            termios.tcsetattr(self._stdin_fd, termios.TCSANOW, attrs)

        # 启动 native termio 线程接管 stdin (事件驱动, 零轮询). 启动失败
        # (libtermio.so 缺失 / stdin 非 TTY) 时回退到 Python select daemon.
        if self._emu._termio:
            self._emu._termio.start()
        # TermIO daemon 已接管 stdin (native 模式), Python daemon 不需要再读.
        if self._emu._termio is None or not self._emu._termio.native_active:
            self._start_stdin_daemon()

    def _restore_term(self) -> None:
        """恢复 _enter_run_mode 保存的终端属性."""
        try:
            if self._saved_term_attrs:
                termios.tcsetattr(self._stdin_fd, termios.TCSANOW, self._saved_term_attrs)
                self._saved_term_attrs = None
        except OSError, termios.error:
            pass

    @staticmethod
    def _uart_tx(text: str) -> None:
        """UART TX 回调: 写入 stdout 并立即刷新."""
        sys.stdout.write(text)
        sys.stdout.flush()

    def _stdin_daemon_loop(self) -> None:
        """后台 daemon: 仅 Python 回退时使用 (termio 不可用).
        select(stdin) -> os.read -> uart.preload() -> PLIC 中断.
        Ctrl+Q (0x11) 中断执行返回调试器 REPL.
        """
        uart = self._emu.uart
        if uart is None:
            return
        stdin_fd = self._stdin_fd
        stop_flag = self._emu._native_stop_flag

        while self._stdin_daemon_running:
            try:
                ready, _, _ = select.select([stdin_fd], [], [], 0.02)
            except OSError, ValueError:
                break
            if not self._stdin_daemon_running:
                break
            if not ready:
                continue
            try:
                data = os.read(stdin_fd, 4096)
            except OSError:
                break
            if not data:
                break
            # Ctrl+Q -> 停止模拟, 返回 REPL
            i = data.find(b"\x11")
            if i >= 0:
                stop_flag.value = 1
                self._emu._wake_event.set()
                self._stdin_daemon_running = False
                data = data[:i]  # 仅交付 Ctrl+Q 之前的部分
            if data:
                try:
                    uart.preload(data)
                except Exception:
                    pass
            self._emu._wake_event.set()

    def _start_stdin_daemon(self) -> None:
        """启动 stdin daemon 线程 (在 cbreak 终端设置之后调用)."""
        if self._stdin_daemon_running:
            return
        if self._emu.uart is None:
            return
        self._stdin_daemon_running = True
        self._stdin_daemon_thread = threading.Thread(
            target=self._stdin_daemon_loop,
            daemon=True,
        )
        self._stdin_daemon_thread.start()

    def _stop_stdin_daemon(self) -> None:
        """停止 stdin daemon 线程 (在恢复终端之前调用)."""
        self._stdin_daemon_running = False
        if self._stdin_daemon_thread:
            self._emu._wake_event.set()  # 唤醒 select 使其检查 running 标志
            self._stdin_daemon_thread.join(timeout=1.0)
            self._stdin_daemon_thread = None

    def _feed_uart_stdin(self) -> bool:
        """非阻塞读取 stdin 并转发到 UART RX FIFO.

        daemon 线程已在后台 select(stdin)->os.read->uart.preload,
        此处仍保留原有 select+os.read 路径 — daemon 作为加速补充而非替代.
        """
        # 守护线程活跃时也走到这里: 原有 select+os.read 路径不受影响
        # 守护线程读出后 select 返回空, 此时不进行任何操作

        if self._emu._termio and self._emu._termio.native_active:
            # 从 RX 环形缓冲区读出并写入 UART FIFO, 内部处理 _rx_notify 的 TOCTOU 清零.
            had_input = self._emu._termio.drain_rx_feed_uart()
            self._flush_uart_if_present()
            return had_input

        uart = self._emu.uart
        if uart is None:
            return False

        # daemon 线程已在后台持续 select(stdin) -> os.read -> uart.preload;
        # 主线程不应再从同一 fd 读取, 否则产生竞争且此路径不设 ext_irq.
        # 仅处理 daemon 未完全消费的 _stdin_pending 残余.
        if self._stdin_daemon_running:
            return self._feed_pending_daemon(uart)

        had_input = self._feed_pending_uart(uart)
        if self._stdin_pending and not had_input:
            # 接收缓冲区拒绝待转发数据: 数据留在 _stdin_pending, 本轮不再读取 stdin
            return False
        if had_input and self._stdin_pending:
            return True
        return self._read_stdin(uart, had_input)

    def _feed_pending_daemon(self, uart) -> bool:
        """守护线程运行中: 只把 _stdin_pending 残余写入 UART."""
        if not self._stdin_pending:
            return False
        try:
            accepted = uart.preload(self._stdin_pending)
        except ValueError, OSError:
            return False
        if accepted > 0:
            self._emu._native_ext_irq.pending = 1
            self._stdin_pending = self._stdin_pending[accepted:]
        if self._stdin_pending:
            self._emu._wake_event.set()
        return accepted > 0

    def _feed_pending_uart(self, uart) -> bool:
        """把 _stdin_pending 写入 UART; 返回是否写入了字节."""
        if not self._stdin_pending:
            return False
        try:
            accepted = uart.preload(self._stdin_pending)
        except ValueError, OSError:
            return False
        if accepted <= 0:
            return False
        self._stdin_pending = self._stdin_pending[accepted:]
        if self._stdin_pending:
            self._emu._wake_event.set()
        return True

    def _read_stdin(self, uart, had_input: bool) -> bool:
        """非阻塞读一次 stdin 并写入 UART, had_input 为已写入字节的标记."""
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
        except ValueError, OSError:
            return had_input
        if accepted > 0:
            had_input = True
            self._emu._native_ext_irq.pending = 1
            if accepted < len(data):
                self._stdin_pending = data[accepted:]
        if had_input:
            self._emu._native_ext_irq.pending = 1
            self._emu._wake_event.set()
        return had_input

    def _flush_uart_if_present(self) -> None:
        """UART TXDATA 写入已自动即时输出, 无需手动刷新."""

    def _idle_poll(self) -> bool:
        """WFI 空闲轮询回调 — 转发 stdin."""
        had_input = self._feed_uart_stdin()
        return had_input

    def _idle_poll_termio(self) -> bool:
        """WFI 空闲轮询回调, 用于 Rust termio 线程模式.

        termio 线程已接管 stdin 读取和 TX 到 stdout 的写入; 此处把 RX 环形
        缓冲区的字节读出并写入 UART 模型, 并刷新行缓冲确保部分行及时进入 hart 日志文件。

        Returns:
            返回 True 仅当有 stdin 数据被 preload, 此时中断 WFI 睡眠, 立即同步
            PLIC 并重新使用动态链接库加速。恒返回 True 会使 WFI 轮询循环每次立即
            退出, 导致热自旋 100% CPU。
        """
        had_input = False
        if self._emu._termio:
            had_input = self._emu._termio.drain_rx()
            # _rx_notify 由 idle poll 在确认环形缓冲区已全部读出后清零
            termio = self._emu._termio
            if termio._rx_wr.value == termio._rx_rd.value:
                termio._rx_notify.value = 0
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

    def _build_completer(self) -> WordCompleter:
        words: list[str] = [
            "s",
            "step",
            "c",
            "continue",
            "r",
            "run",
            "undo",
            "rollback",
            "restart",
            "b",
            "bp",
            "regs",
            "gpr",
            "reg",
            "set",
            "w",
            "csr",
            "csrw",
            "pc",
            "mode",
            "mstatus",
            "tlb",
            "tlbflush",
            "cache",
            "satp",
            "pt",
            "mem",
            "vmem",
            "status",
            "info",
            "symbols",
            "sym",
            "bins-info",
            "disk-info",
            "chdir",
            "ls",
            "load",
            "detail",
            "rm",
            "disasm",
            "vdisasm",
            "stack",
            "show",
            "bt",
            "frame",
            "f",
            "hart",
            "h",
            "help",
            "?",
            "q",
            "quit",
            "exit",
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

    def _show_trap_context(self, h) -> None:
        """hart trap后的上下文摘要."""
        cause = h.mcause_val
        name = trap_cause_name(cause)
        is_int = (cause >> 63) & 1
        tag = "中断" if is_int else "异常"
        self._console.print(
            f"[red bold]Hart {h.id} {tag}[/] {name}  "
            f"mcause=0x{cause:x}  mepc={hex_addr(h.mepc_val)}  "
            f"mtval={hex_addr(h.mtval_val)}"
        )

    def repl(self) -> None:
        """主交互循环."""
        self._enter_repl_mode()
        banner = (
            f"[bold]pyremu rvdb[/] — "
            f"{self._emu.num_harts} hart(s), "
            f"RAM {hex_addr(self._emu.bus._ram_base)}-"
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
                prompt = (
                    f"[bold red]rvdb:{self._hart_id}[/]"
                    + f" ([cyan]{self.hart.mode.name}[/]) > "
                )
                user_input = self._session.prompt(prompt).strip()
            except KeyboardInterrupt, EOFError:
                self._console.print("\n[dim]goodbye[/]")
                break

            if user_input == "":
                user_input = self._last_command or "status"
            self._last_command = user_input
            self._history.append_string(user_input)

            parts = user_input.split()
            if not parts:
                continue

            # 命令名之后的原文: 保留内部空白, 供需要含空白参数的子命令解析
            rest = user_input[len(parts[0]) :].lstrip()
            try:
                if not self._dispatch(parts, rest):
                    break
            except KeyboardInterrupt:
                self._console.print("\n[dim]REPL 中断[/]")
            except Exception:
                self._console.print_exception()
        # REPL 退出: 释放 Emulator 全部 I/O 资源 (termio 管道 fd + UART 日志).
        # _enter_repl_mode 已 stop termio; close() 幂等, 额外关闭日志文件.
        self._emu.close()
