#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-LICENSE-IDENTIFIER: GPL2.0
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
        self._terminated: bool = False

        # Pending stdin bytes — UART RX FIFO is only 8 bytes, so excess
        # bytes from os.read() are buffered here and fed on subsequent
        # _feed_uart_stdin calls.  Without this, fast typing / paste in
        # rust lib will silently drop bytes beyond the 8th.
        self._stdin_pending: bytes = b""

        # Stdin daemon thread — 独立于处理器执行循环, 持续将 stdin 转发到
        # UART RX FIFO. 对照 QEMU chardev fd_chr_read_poll 模型:
        #   stdin -> os.read -> uart.preload() -> _update_plic_irq() -> PLIC 中断
        #   -> _wake_event.set() -> 主线程 WFI 睡眠唤醒 -> try_wfi_wakeup()
        # 处理器感知不到 daemon 的存在 — 它只看到 PLIC 中断信号.
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
        self._console.print("\n暂停请求 — 当前指令完成后回到 REPL")
        try:
            self._emu.notify_processor()
        except Exception:
            pass

    def _enter_repl_mode(self) -> None:
        """恢复到 REPL 的终端设置 (prompt_toolkit 自行管理 raw 模式)."""
        signal.signal(signal.SIGINT, self._sigint_repl)
        self._stop_stdin_daemon()
        if self._emu._termio is not None:
            self._emu._termio.stop()
        self._restore_term()
        self._emu._stdin_forward_callback = None

    def _enter_run_mode(self) -> None:
        """切换到运行模式: Ctrl+Q 暂停, cbreak stdin (ISIG 关, Ctrl+C 透传).

        ── TX: QEMU fd_chr_write 模型, 与单轮加速执行零耦合 ──
        固件写 TXDATA -> Rust inline handler -> libc::write(1, &byte, 1).
        每字节即时输出, _console_echo=False 抑制 Python 双重输出.

        ── RX: daemon 线程 select(stdin) -> uart.preload() -> PLIC 中断 ──
        stdin daemon 独立于处理器执行循环持续读取, 与 WFI 零耦合.
        处理器仅看到 PLIC 外部中断信号 -> try_wfi_wakeup() 自然唤醒.
        """
        # 终端 ISIG 关闭后键盘 Ctrl+C (0x03) 作为普通字节透传给客机, 不经信号
        # 路径; Ctrl+Q 停止由 termio 的 notify_emu_stop 直连置位 stop_flag, 同样
        # 不经信号. 此 SIGINT handler 仅覆盖 stdin 未转发的场景 (终端仍生成 SIGINT).
        signal.signal(signal.SIGINT, self._sigint_run)

        if not self._stdin_forward:
            return

        if self._emu.uart is not None:
            self._emu.uart.clear_rx()

        self._emu._stdin_forward_callback = self._idle_poll

        # 完整 raw 模式 (对照 termio/src/lib.rs raw 设置):
        # - 关 ECHO/ICANON/IEXTEN/ISIG: 所有字符原样透传
        # - CS8: 8-bit 数据, 不丢高位 (对 backspace 0x7F 等关键)
        # - ICRNL 保留: \r->\n, 行终止兼容客机控制台
        # - IXON 关: Ctrl+Q/Ctrl+S 透传
        if os.isatty(self._stdin_fd):
            self._saved_term_attrs = termios.tcgetattr(self._stdin_fd)
            attrs = termios.tcgetattr(self._stdin_fd)
            # iflag: 清除输入转换; ICRNL 保留 (客机控制台可能未初始化 \r->\n)
            attrs[0] = (attrs[0] & ~(
                termios.IGNBRK | termios.BRKINT | termios.PARMRK
                | termios.ISTRIP | termios.INLCR | termios.IGNCR
                | termios.IXON
            )) | termios.ICRNL
            # lflag: 清除行编辑/回显/信号生成
            attrs[3] &= ~(
                termios.ECHO | termios.ECHONL | termios.ICANON
                | termios.IEXTEN | termios.ISIG
            )
            # cflag: 8-bit 字符
            attrs[2] = (attrs[2] & ~termios.CSIZE) | termios.CS8
            # cflag: 关校验
            attrs[2] &= ~termios.PARENB
            # cc: 每字节即时可读
            attrs[6][termios.VMIN] = 1
            attrs[6][termios.VTIME] = 0
            termios.tcsetattr(self._stdin_fd, termios.TCSANOW, attrs)

        # 启动 native termio 线程接管 stdin (事件驱动, 零轮询). 启动失败
        # (libtermio.so 缺失 / stdin 非 TTY) 时回退到 Python select daemon.
        if self._emu._termio is not None:
            self._emu._termio.start()
        # TermIO daemon 已接管 stdin (native 模式), Python daemon 不需要再读.
        if self._emu._termio is None or not self._emu._termio.native_active:
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
            except (OSError, ValueError):
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
            i = data.find(b'\x11')
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
    #  UART stdin 转发 (主线程 select+os.read, 每单轮加速执行边界执行)
    # ----------------------------------------------------------

    def _feed_uart_stdin(self) -> bool:
        """非阻塞读取 stdin 并转发到 UART RX FIFO.

        daemon 线程已在后台 select(stdin)->os.read->uart.preload,
        此处仍保留原有 select+os.read 路径 — daemon 作为加速补充而非替代.
        """
        # daemon 活跃时也走到这里: 原有 select+os.read 路径不受影响
        # (daemon 读走后 select 返回空即 no-op)

        if self._emu._termio is not None and self._emu._termio.native_active:
            # 排空 RX ring -> UART FIFO, 内部处理 _rx_notify 的 TOCTOU 清零.
            had_input = self._emu._termio.drain_rx_feed_uart()
            self._flush_uart_if_present()
            return had_input

        uart = self._emu.uart
        if uart is None:
            return False
        had_input = False

        # daemon 线程已在后台持续 select(stdin) -> os.read -> uart.preload;
        # 主线程不应再从同一 fd 读取, 否则产生竞争且此路径不设 ext_irq.
        # 仅处理 daemon 未完全消费的 _stdin_pending 残余.
        if self._stdin_daemon_running:
            if not self._stdin_pending:
                return had_input
            try:
                accepted = uart.preload(self._stdin_pending)
            except (ValueError, OSError):
                return had_input
            if accepted > 0:
                had_input = True
                self._emu._native_ext_irq.pending = 1
                self._stdin_pending = self._stdin_pending[accepted:]
            if self._stdin_pending:
                self._emu._wake_event.set()
            return had_input

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
        """WFI 空闲轮询回调 (Rust termio 线程模式).

        termio 线程已接管 stdin 读取和 TX->stdout 排空; 此处排空 RX 环形
        缓冲到 UART 模型, 并刷新行缓冲确保部分行及时进入 hart 日志文件。

        Returns:
            True 仅当有 stdin 数据被 preload (中断 WFI 睡眠, 立即同步
            PLIC 并重新使用动态链接库加速)。恒返回 True 会使 WFI 轮询循环每次立即
            退出 ->热自旋 100% CPU。
        """
        had_input = False
        if self._emu._termio is not None:
            had_input = self._emu._termio.drain_rx()
            # _rx_notify 由 idle poll 在确认 ring buffer 排空后清零
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
                prompt = f"[bold red]rvdb:{self._hart_id}[/]" + \
                         f" ([cyan]{self.hart.mode.name}[/]) > "
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
        # REPL 退出: 释放 Emulator 全部 I/O 资源 (termio 管道 fd + UART 日志).
        # _enter_repl_mode 已 stop termio; close() 幂等, 额外关闭日志文件.
        self._emu.close()
