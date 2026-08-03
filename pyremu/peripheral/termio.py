#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Terminal I/O 后台线程 — QEMU chardev-stdio 模型 (Rust libtermio.so 实现).

独立于 CPU 模拟批次循环, 以 Rust 后台线程持续转发 stdin->RX 环形缓冲和
TX 环形缓冲->stdout。对照 QEMU ``vendor/qemu-10.2.0/chardev/char-stdio.c``:

- **单一 owner**: 线程运行期间, Rust 线程是唯一的控制台写者 (对照
  ``fd_chr_write`` 是唯一 TX 出口)。UART 行缓冲仅归档 hart 日志文件,
  控制台回显经 ``UART.set_console_echo(False)`` 关闭, 消除双写 stdout。
- **单写者索引**: ``tx_drain`` 由 Rust 线程独占; Python 日志消费者维护
  独立的 ``_tx_log_rd``; ``rx_wr`` 由 Rust 写, ``rx_rd`` 由 Python 写
  (发布消费进度, Rust 据此容量控制 — 缓冲满时停读 stdin)。
- **终端属性**: Rust 侧保存/设置/恢复 termios + fcntl 阻塞标志 (对照
  term_init/term_exit), Ctrl+Z 挂起恢复经 SIGCONT handler 重设 raw。

libtermio.so 不可用时降级为 Python ``select`` 轮询线程 (不设终端属性,
cbreak 由 Debugger 负责)。

Usage:
    term = TerminalIO(uart, wake_event, stdin_fd, stdout_fd)
    native = term.start()  # True: Rust 线程接管终端; False: Python 回退
    term.drain_rx()        # RX 环形缓冲 ->UART RX buffer (每批次前调用)
    term.drain_tx_logs()   # TX 环形缓冲 ->UART 行缓冲/日志 (每批次后调用)
    term.stop()            # 停止线程 + 恢复终端 + 排空残留
"""

from __future__ import annotations

import atexit
import ctypes
import fcntl
import os
import select
import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pyremu.peripheral.uart import UART

from pyremu._native import (
    termio_attach,
    termio_available,
    termio_is_running,
    termio_stop,
    TermIoHandle,
)


class TerminalIO:
    """终端 I/O 管理器 — 封装 Rust 后台线程的启动/停止/环形缓冲排空.

    设计为 UART 外设的可选组件, 由 Emulator 在初始化时创建并注入。
    Debugger 在运行/REPL 模式切换时通过 Emulator 引用调用 start/stop。
    """

    # 环形缓冲容量必须为 2 的幂: 索引按 u32 单调环绕, 槽位 = 索引 % 容量,
    # 容量需整除 2^32, 否则索引跨 u32 环绕时槽位错位。
    RX_CAP = 0x10000  # 64 KiB — 容纳批量粘贴/快速输入突发
    TX_CAP = 0x40000  # 256 KiB 字节容量 (条目容量 = TX_CAP // 2)

    def __init__(
        self,
        uart: UART,
        wake_event: threading.Event,
        stdin_fd: int = 0,
        stdout_fd: int = 1,
        ext_irq: Any = None,
    ) -> None:
        self._uart = uart
        self._wake_event = wake_event
        self._stdin_fd = stdin_fd
        self._stdout_fd = stdout_fd
        self._ext_irq = ext_irq  # FfiExtIrqCtx or None

        # TX 环形缓冲 — CPU hart 线程写入 (客机 TXDATA, 条目 = [hart_id, byte]),
        # Rust termio 线程经 tx_drain 排空到 stdout; Python 经 _tx_log_rd 归档日志。
        self._tx_buf = (ctypes.c_uint8 * self.TX_CAP)()
        self._tx_wr = ctypes.c_uint32(0)     # 写索引 (CPU 引擎独占, 单调 u32)
        self._tx_drain = ctypes.c_uint32(0)  # 排空索引 (Rust termio 线程独占)
        self._tx_log_rd: int = 0             # 日志消费者读索引 (Python 独占)

        # RX 环形缓冲 — Rust termio 线程写入 (stdin 读取), Python drain_rx 排空。
        self._rx_buf = (ctypes.c_uint8 * self.RX_CAP)()
        self._rx_wr = ctypes.c_uint32(0)  # 写索引 (Rust termio 线程独占)
        self._rx_rd = ctypes.c_uint32(0)  # 读索引 (Python 独占; Rust 读之容量控制)

        # 共享停止标志 — Python 置 1 ->Rust 线程退出
        self._stop_flag = ctypes.c_uint8(0)
        self._pause_flag = ctypes.c_uint8(0)  # 调试器暂停/恢复
        self._rx_notify = ctypes.c_uint8(0)   # TermIO daemon 写 ring buffer 后置 1

        # 运行状态
        self._native_active = False           # Rust 线程是否接管了终端
        self._atexit_registered = False

        # Python 轮询回退 (libtermio.so 不可用时启用)
        self._fallback_thread: threading.Thread | None = None
        self._fallback_running = False

        # TX 事件驱动排空: Rust 每写一个 TX 字节到 ring buffer 后
        # 写 1 字节到 _tx_notify_w -> daemon 线程被 select 唤醒 ->
        # drain_tx_logs() -> UART 回调 -> TermProxy -> stdout.
        self._tx_notify_r, self._tx_notify_w = os.pipe()
        # 写端非阻塞: 管道满时 Rust 不卡死, 通知丢失可接受
        # (daemon 会在下次 drain 时追上).
        fl_w = fcntl.fcntl(self._tx_notify_w, fcntl.F_GETFL)
        fcntl.fcntl(self._tx_notify_w, fcntl.F_SETFL, fl_w | os.O_NONBLOCK)
        # 读端阻塞: daemon 线程 select 等待
        self._tx_drain_lock = threading.Lock()
        self._tx_drain_running = False

        # RX 事件驱动通知: Rust termio 线程写 ring buffer 后向 _rx_notify_w
        # 写 1 字节 -> daemon 被 select 唤醒 -> drain_rx() -> UART FIFO.
        # 消除轮询, 延迟从 500μs 降到内核调度延迟 (~10μs).
        self._rx_notify_r, self._rx_notify_w = os.pipe()
        fl_rx = fcntl.fcntl(self._rx_notify_w, fcntl.F_GETFL)
        fcntl.fcntl(self._rx_notify_w, fcntl.F_SETFL, fl_rx | os.O_NONBLOCK)
        self._tx_drain_thread: threading.Thread | None = None

        # TX 归档 daemon — 异步将 ring buffer 写入 hart 日志, 与批次循环解耦
        self._tx_archive_running = False
        self._tx_archive_thread: threading.Thread | None = None

        # RX daemon — 独立线程持续将 ring buffer -> UART FIFO, 与指令执行完全解耦
        self._rx_daemon_running = False
        self._rx_daemon_thread: threading.Thread | None = None

    @property
    def tx_notify_w(self) -> int:
        """TX 通知管道写端 fd, 传给 Rust (UartInfo.tx_notify_fd).
        Rust 每写一个 TX 字节后写 1 字节到此 fd 唤醒 Python daemon.
        """
        return self._tx_notify_w

    # ----------------------------------------------------------
    #  公开接口
    # ----------------------------------------------------------

    @property
    def tx_buf(self):
        """TX 环形缓冲区 (供 Emulator._native_marshal_uart 使用)."""
        return self._tx_buf

    @property
    def tx_wr(self):
        """TX 写索引 ctypes 对象 (供 Emulator._native_marshal_uart 使用)."""
        return self._tx_wr

    @property
    def tx_drain(self):
        """TX 排空索引 ctypes 对象 (Rust termio 线程独占写, Python 只读)."""
        return self._tx_drain

    @property
    def tx_log_rd(self) -> int:
        """日志消费者的独立读索引 (测试用)."""
        return self._tx_log_rd

    @property
    def native_active(self) -> bool:
        """Rust termio 线程是否正在运行并接管终端."""
        return self._native_active

    def start(self) -> bool:
        """启动后台 I/O 线程.

        Returns:
            True — Rust native 线程接管终端 (raw 模式 + stdout 回显均由
            Rust 侧负责); False — 回退到 Python 轮询线程 (终端 cbreak 与
            控制台回显由调用方/UART 行缓冲负责)。
        """
        if self._is_running():
            return self._native_active

        self._stop_flag.value = 0
        self._pause_flag.value = 0
        if termio_available() and termio_attach(self._build_handle()) == 0:
            self._native_active = True
            # 单一 owner: Rust 线程独占 stdout, Python 不重复输出
            self._uart.set_console_echo(False)
            if not self._atexit_registered:
                # 进程异常退出时恢复终端 (对照 QEMU atexit(term_exit))
                atexit.register(self.stop)
                self._atexit_registered = True
            self.start_tx_archive_thread()
            self.start_rx_daemon()
            return True

        # native 启动失败 (库缺失 / stdin 非 TTY) — 回退 Python 轮询
        self._start_fallback()
        return False

    def stop(self) -> None:
        """停止后台 I/O 线程, 恢复终端属性, 排空残留数据.

        顺序敏感: Rust 线程退出前会最终排空 TX->stdout (tx_drain 追平 tx_wr),
        故残留条目须先以"回显关闭"状态归档日志 (避免重复输出), 最后才把
        控制台回显交还给 Python 行缓冲。
        """
        # 先停 RX daemon — 停止接受新的 stdin 数据
        self.stop_rx_daemon()
        # 再停归档 daemon — 线程退出后最终排空由 stop_tx_archive_thread 完成
        self.stop_tx_archive_thread()

        if termio_is_running():
            self._stop_flag.value = 1
            self._wake_event.set()
            termio_stop()
        elif self._fallback_running:
            self._fallback_running = False
            if self._fallback_thread is not None:
                self._fallback_thread.join(timeout=1.0)
                self._fallback_thread = None

        # 排空 RX 与 TX 日志中尚未处理的数据
        self.drain_rx()
        self.drain_tx_logs()
        if self._native_active:
            self._native_active = False
            self._uart.set_console_echo(True)

    def pause(self) -> None:
        """暂停 I/O 线程 (调试器断点)."""
        if self._native_active:
            self._pause_flag.value = 1

    def resume(self) -> None:
        """恢复 I/O 线程."""
        self._pause_flag.value = 0

    def drain_rx(self) -> bool:
        """将 Rust 线程写入的 RX 环形缓冲数据转移到 Python UART 模型.

        对照 QEMU ``fd_chr_read_poll ->qemu_chr_fe_can_read ->chr_read`` 流控:
        - 先查 ``uart.can_rx()`` (FIFO 有空位才接收)
        - 逐字节 preload, 每字节检查 can_rx; FIFO 满即停止
        - 未被消费的字节保留在 ring buffer 中 (rd 不推进),
          且因 rx_rd 未更新, Rust 端 ``free = cap - (wr - rd)`` 变小
          ->termio 线程自然容量控制停止读 stdin (对照 QEMU fd_chr_read_poll)

        Returns:
            True 若有新数据被 preload (可触发 WFI 唤醒).
        """
        if self._uart is None:
            return False
        if not self._uart.can_rx():
            return False
        wr = self._rx_wr.value
        rd = self._rx_rd.value
        if wr == rd:
            return False
        pending = (wr - rd) & 0xFFFF_FFFF
        cap = self.RX_CAP
        had_input = False
        drained = 0
        # 只消费 UART FIFO 能接受的数量; 剩余留在 ring buffer 等下次 drain
        for _ in range(pending):
            if not self._uart.can_rx():
                break
            b = self._rx_buf[rd % cap]
            rd = (rd + 1) & 0xFFFF_FFFF
            self._uart.preload(bytes([b]))
            drained += 1
            had_input = True
        self._rx_rd.value = rd  # 发布消费进度 (Rust 容量控制依据)
        # _rx_notify 不在此处清零 — RX daemon 抢先 drain 后 Rust batch
        # engine 仍需看到通知以触发快速批次退出 (hart_sched.rs:1362).
        # 清零由 idle poll 路径在确认 ring buffer 为空后负责.
        if had_input and self._ext_irq is not None:
            self._ext_irq.pending = 1  # 通知 CPU 引擎内联投递 SEIP/MEIP
        self._wake_event.set()
        return had_input

    def drain_tx_logs(self) -> None:
        """把 CPU 引擎写入 TX 环形缓冲的 (hart, byte) 条目归入 UART 行缓冲.

        本消费者维护独立读索引 ``_tx_log_rd``; 控制台是否回显由
        ``UART._console_echo`` 决定。

        线程安全: 持有 ``_tx_drain_lock``, 允许 TX drain daemon 线程与
        主线程 (``_native_flush_uart``) 并发调用。
        """
        with self._tx_drain_lock:
            self._drain_tx_logs_locked()

    def drain_tx_logs_archive_only(self) -> None:
        """排空 TX ring buffer, 仅归档 hart 日志, 不触发 _tx_callback->stdout.

        用于 Rust libc::write 已即时输出到 stdout 的场景; 此处只保证
        ring buffer 不溢出 + hart 日志完整, 不重复输出到控制台。
        """
        if self._uart is None:
            return
        with self._tx_drain_lock:
            ecap = self.TX_CAP // 2
            wr = self._tx_wr.value
            pending = (wr - self._tx_log_rd) & 0xFFFF_FFFF
            if pending == 0:
                return
            if pending > ecap:
                self._tx_log_rd = (wr - ecap) & 0xFFFF_FFFF
                pending = ecap
            rd = self._tx_log_rd
            uart = self._uart
            for _ in range(pending):
                idx = rd % ecap
                hid = self._tx_buf[2 * idx]
                byte = self._tx_buf[2 * idx + 1]
                log_f = uart._hart_log_file(hid)
                if log_f is not None:
                    log_f.write(chr(byte))
                rd = (rd + 1) & 0xFFFF_FFFF
            self._tx_log_rd = rd

    def _drain_tx_logs_locked(self) -> None:
        """drain_tx_logs 的锁内实现."""
        if self._uart is None:
            return
        ecap = self.TX_CAP // 2
        wr = self._tx_wr.value
        pending = (wr - self._tx_log_rd) & 0xFFFF_FFFF
        if pending == 0:
            return
        if pending > ecap:
            self._tx_log_rd = (wr - ecap) & 0xFFFF_FFFF
            pending = ecap
        rd = self._tx_log_rd
        uart = self._uart
        for _ in range(pending):
            idx = rd % ecap
            hid = self._tx_buf[2 * idx]
            byte = self._tx_buf[2 * idx + 1]
            # Hart 日志文件: 直接从 ring buffer 写入, 不经 UART 行缓冲
            log_f = uart._hart_log_file(hid)
            if log_f is not None:
                log_f.write(chr(byte))
            # UART MMIO 写 — 触发 _tx_callback -> 控制台即时输出
            uart.write(0, bytes([byte]))
            rd = (rd + 1) & 0xFFFF_FFFF
        self._tx_log_rd = rd

    # ----------------------------------------------------------
    #  TX 归档 daemon 线程 — 异步将 ring buffer 写入 hart 日志文件,
    # 不依赖批次边界, 与指令执行循环完全解耦.
    # ----------------------------------------------------------

    def start_tx_archive_thread(self) -> None:
        """启动 TX 归档 daemon: 事件驱动地将 ring buffer 写入 hart 日志."""
        if self._tx_archive_running:
            return
        self._tx_archive_running = True
        self._tx_archive_thread = threading.Thread(
            target=self._tx_archive_loop, daemon=True,
        )
        self._tx_archive_thread.start()

    def stop_tx_archive_thread(self) -> None:
        """停止 TX 归档 daemon, 最终排空残留."""
        self._tx_archive_running = False
        if self._tx_archive_thread is not None:
            self._tx_archive_thread.join(timeout=1.0)
            self._tx_archive_thread = None
        self.drain_tx_logs_archive_only()

    def _tx_archive_loop(self) -> None:
        """TX 归档 daemon 入口: select 等待 Rust notify -> 归档 hart 日志.
        与批次循环完全异步, 仅在 Rust 写入 ring buffer 后触发."""
        notify_r = self._tx_notify_r
        while self._tx_archive_running:
            try:
                ready, _, _ = select.select([notify_r], [], [], 0.2)
            except (ValueError, OSError):
                break
            if not self._tx_archive_running:
                break
            # 排空通知管道
            try:
                while True:
                    os.read(notify_r, 256)
            except BlockingIOError:
                pass
            except OSError:
                break
            self.drain_tx_logs_archive_only()

    # ----------------------------------------------------------
    #  TX 即时排空 daemon 线程
    # ----------------------------------------------------------

    def start_tx_drain_thread(self) -> None:
        """启动 TX drain daemon 线程: 每 ~1ms 检查 ring buffer,
        有新数据则经 UART 行缓冲 -> _tx_callback -> TermProxy writer -> stdout.
        """
        if self._tx_drain_running:
            return
        self._tx_drain_running = True
        self._tx_drain_thread = threading.Thread(
            target=self._tx_drain_loop, daemon=True,
        )
        self._tx_drain_thread.start()

    def stop_tx_drain_thread(self) -> None:
        """停止 TX drain daemon 线程, 并最终排空残留."""
        self._tx_drain_running = False
        if self._tx_drain_thread is not None:
            self._tx_drain_thread.join(timeout=1.0)
            self._tx_drain_thread = None
        # 最终排空: 线程停止后可能还有残留
        self.drain_tx_logs()

    def _tx_drain_loop(self) -> None:
        """TX drain daemon 线程入口: 阻塞等待 Rust 通知 -> 排空 ring buffer.

        Rust 每写一个 TX 字节后写 1 字节到 _tx_notify_w -> select 唤醒 ->
        drain_tx_logs() 走完整 UART 回调链 -> TermProxy writer -> stdout.
        零轮询, 完全事件驱动.
        """
        notify_r = self._tx_notify_r
        while self._tx_drain_running:
            try:
                ready, _, _ = select.select([notify_r], [], [])
            except (ValueError, OSError):
                break
            if not self._tx_drain_running:
                break
            # 排空通知管道中的全部字节 (可能积压了多次通知)
            try:
                while True:
                    os.read(notify_r, 256)
            except BlockingIOError:
                pass
            except OSError:
                break
            # 排空 ring buffer -> UART 回调 -> TermProxy -> stdout
            self.drain_tx_logs()

    # ----------------------------------------------------------
    #  RX daemon — 独立线程, 与指令执行完全解耦
    # ----------------------------------------------------------

    def start_rx_daemon(self) -> None:
        """启动 RX daemon 线程: 持续将 ring buffer -> UART FIFO.

        Rust termio 线程写 stdin 到 ring buffer 后置 _rx_notify=1,
        daemon 检测到后调用 drain_rx() 搬运到 UART FIFO 并设置中断.
        指令执行循环完全不需要参与数据搬运.
        """
        if self._rx_daemon_running or self._stdin_fd < 0:
            # dummy fd (测试环境), 无法 poll
            return
        self._rx_daemon_running = True
        self._rx_daemon_thread = threading.Thread(
            target=self._rx_daemon_loop, daemon=True,
        )
        self._rx_daemon_thread.start()

    def stop_rx_daemon(self) -> None:
        """停止 RX daemon 线程, 最终排空残留."""
        self._rx_daemon_running = False
        if self._rx_daemon_thread is not None:
            self._rx_daemon_thread.join(timeout=1.0)
            self._rx_daemon_thread = None
        self.drain_rx()

    def _rx_daemon_loop(self) -> None:
        """RX daemon 入口: 事件驱动, select 阻塞等待 Rust termio 通知管道.

        Rust termio 线程写 ring buffer 后向 _rx_notify_w 写 1 字节 -> select
        立即返回 -> drain_rx() 搬运到 UART FIFO。完全零轮询, 延迟仅受内核调度
        影响 (~10 μs 量级), 远优于此前 500 μs 轮询。
        """
        notify_r = self._rx_notify_r
        while self._rx_daemon_running:
            try:
                ready, _, _ = select.select([notify_r], [], [], 0.5)
            except (ValueError, OSError):
                break
            if not self._rx_daemon_running:
                break
            # 排空通知管道中积压的字节
            try:
                while True:
                    os.read(notify_r, 256)
            except BlockingIOError:
                pass
            except OSError:
                break
            # 搬运 ring buffer -> UART FIFO (可能已积压多个字节)
            self.drain_rx()

    # ----------------------------------------------------------
    #  Rust native 路径
    # ----------------------------------------------------------

    def _build_handle(self) -> TermIoHandle:
        """构建传给 ``terminal_io_start`` 的 FFI 结构体.

        指针字段指向本实例的 ctypes 缓冲区 (实例存活期内有效);
        Rust 侧在调用返回前复制全部字段, 结构体本身无需长期存活。
        """
        h = TermIoHandle()
        h.stdin_fd = self._stdin_fd
        h.stdout_fd = self._stdout_fd
        h.rx_buf = ctypes.cast(self._rx_buf, ctypes.c_void_p).value or 0
        h.rx_cap = self.RX_CAP  # Rust termio 线程读取 stdin -> ring buffer
        h.rx_wr = ctypes.addressof(self._rx_wr)
        h.rx_rd = ctypes.addressof(self._rx_rd)
        h.tx_buf = ctypes.cast(self._tx_buf, ctypes.c_void_p).value or 0
        h.tx_cap = self.TX_CAP // 2  # 条目容量 (非字节容量)
        h.tx_wr = ctypes.addressof(self._tx_wr)
        h.tx_drain = ctypes.addressof(self._tx_drain)
        h.stop_flag = ctypes.addressof(self._stop_flag)
        h.pause_flag = ctypes.addressof(self._pause_flag)
        h.rx_notify = ctypes.addressof(self._rx_notify)
        h.rx_notify_fd = self._rx_notify_w
        return h

    # ----------------------------------------------------------
    #  Python 回退路径 (libtermio.so 不可用时)
    # ----------------------------------------------------------

    def _preload_stdin_chunk(self) -> bool:
        """从 stdin 读取一块数据并 preload 到 UART RX.

        Returns:
            True 若成功 preload 了数据, False 表示无数据或出错 (需退出轮询).
        """
        try:
            data = os.read(self._stdin_fd, 4096)
        except (OSError, ValueError):
            return False
        if not data:
            return False
        self._uart.preload(data)
        self._wake_event.set()
        return True

    def _fallback_poll_loop(self) -> None:
        """Python 回退轮询循环 — daemon thread 入口."""

        stdin_fd = self._stdin_fd
        while self._fallback_running:
            try:
                ready, _, _ = select.select([stdin_fd], [], [], 0.02)
            except (OSError, ValueError):
                break
            if not self._fallback_running:
                break
            if not ready or self._uart is None:
                continue
            self._preload_stdin_chunk()

    def _start_fallback(self) -> None:
        """启动 Python daemon thread 作为回退轮询.

        在 libtermio.so 不可用或 native 启动失败时使用。
        以 20ms 间隔轮询 stdin, 不设置 cbreak 模式 (由 Debugger 负责终端设置)。
        """
        self._fallback_running = True
        self._fallback_thread = threading.Thread(
            target=self._fallback_poll_loop, daemon=True,
        )
        self._fallback_thread.start()

    def _is_running(self) -> bool:
        """检查后台线程是否已运行."""
        if termio_is_running():
            return True
        if self._fallback_running and self._fallback_thread is not None:
            return self._fallback_thread.is_alive()
        return False
