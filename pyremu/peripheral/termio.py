#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Terminal I/O 后台线程 — QEMU chardev-stdio 模型 (Rust libtermio.so 实现).

独立于 CPU 模拟单轮加速执行循环, 以 Rust 后台线程持续转发 stdin->RX 环形缓冲和
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
    term.drain_rx()        # RX 环形缓冲 ->UART RX buffer (每单轮加速执行前调用)
    term.drain_tx_logs()   # TX 环形缓冲 ->UART 行缓冲/日志 (每单轮加速执行后调用)
    term.stop()            # 停止线程 + 恢复终端 + 排空残留
"""

from __future__ import annotations

import atexit
import ctypes
import fcntl
import os
import select
import threading
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from pyremu.peripheral.uart import UART

from pyremu._native import (
    termio_attach,
    termio_available,
    termio_is_running,
    termio_stop,
    TermIoHandle,
)
from pyremu.utils.mask import mask32


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
        on_irq: Callable[[int, bool], None] | None = None,
    ) -> None:
        self._uart = uart
        self._wake_event = wake_event
        self._stdin_fd = stdin_fd
        self._stdout_fd = stdout_fd
        self._ext_irq = ext_irq  # FfiExtIrqCtx or None
        # 设备中断注入回调 (``Emulator.raise_device_irq``): 同步 PLIC 挂起
        # 状态 + write-through 持久数组, 使 batch 内 Rust 内联仲裁能立即看到
        # 新到达的 UART 数据。None 时退化为仅置 ext_irq 通知位 (独立使用场景)。
        self._on_irq = on_irq

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
        # drain_rx 由 RX daemon 线程与主线程 (_feed_uart_stdin) 并发调用,
        # 串行化读-改-写 _rx_rd 以免两个消费者读到同一 rd 值重复 preload 同批字节.
        self._rx_drain_lock = threading.Lock()

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

        # 三对管道: tx_notify (Rust 写 TX 字节后唤醒归档 daemon), rx_notify
        # (Rust 写 stdin 到 ring buffer 后唤醒 RX daemon), rx_drain (RX daemon
        # 消费后唤醒 Rust 投递剩余字节). 创建细节见 _create_pipes().
        self._create_pipes()
        self._tx_drain_lock = threading.Lock()
        self._tx_drain_running = False
        self._tx_drain_thread: threading.Thread | None = None

        # TX 归档 daemon — 异步将 ring buffer 写入 hart 日志, 与单轮加速执行循环解耦
        self._tx_archive_running = False
        self._tx_archive_thread: threading.Thread | None = None

        # RX daemon — 独立线程持续将 ring buffer -> UART FIFO, 与指令执行完全解耦
        self._rx_daemon_running = False
        self._rx_daemon_thread: threading.Thread | None = None

    @property
    def tx_notify_w(self) -> int:
        """TX 通知管道写端 fd, 传给 Rust (UartInfo.tx_notify_fd).

        注意: Rust CPU 引擎实际从不写此 fd (state.rs 中恒为 -1), 通知仅保留
        接口形态; daemon 靠 poll 超时无条件排空兜底 (见 _tx_archive_loop).
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

        # stop() 已把管道 fd 置 -1: 调试器 REPL↔运行模式往返时, 第二次 start()
        # 必须先重建管道, 否则 daemon 线程在 poll.register(-1) 抛未捕获异常.
        if self._tx_notify_r < 0:
            self._create_pipes()

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
            # 唤醒阻塞在 poll(-1) 的 Rust 线程, 使其看到 stop_flag 退出 —
            # 修复 join 挂起 (线程仅在 drain 管道或 stdin 可读时从 poll 返回).
            self._notify_rx_drained()
            termio_stop()
        elif self._fallback_running:
            self._fallback_running = False
            if self._fallback_thread is not None:
                self._fallback_thread.join(timeout=1.0)
                self._fallback_thread = None

        # 排空 RX 与 TX 日志中尚未处理的数据
        self.drain_rx()
        if self._native_active:
            # native 模式: CPU 引擎内联 try_write_fd 已即时输出 stdout, ring
            # 仅作 hart 日志归档 — 这里只归档不回显, 否则 REPL 返回时会把这
            # 一段 ring 积压重放到控制台 (与键盘输入唤醒重放同一缺陷).
            self.drain_tx_logs_archive_only()
        else:
            self.drain_tx_logs()
        if self._native_active:
            self._native_active = False
            self._uart.set_console_echo(True)
        # 显式关闭全部管道 fd — 否则每实例泄漏 6 个 fd (3 pipe × 2 端),
        # 长进程/测试套件中积累到 3k+ fd, 触发 select() fd>=1024 上限.
        self._close_pipes()

    def _close_pipes(self) -> None:
        """关闭 __init__ 中创建的全部管道 fd (幂等, 可重复调用).

        关闭后对应属性置 -1: __del__ 兜底再次调用时不再重复 close.
        Rust 线程持有的 fd 副本 (rx_drain_r 等) 在 termio_stop() 已先行终止,
        关闭 Python 侧 fd 不会影响已退出线程.
        """
        for name in (
            "_tx_notify_r", "_tx_notify_w",
            "_rx_notify_r", "_rx_notify_w",
            "_rx_drain_r", "_rx_drain_w",
        ):
            fd = getattr(self, name, -1)
            if fd < 0:
                continue
            try:
                os.close(fd)
            except OSError:
                pass
            setattr(self, name, -1)

    def _create_pipes(self) -> None:
        """创建三对管道 (tx_notify / rx_notify / rx_drain), 写端设为非阻塞.

        ``stop()`` 关闭管道时会把 fd 属性置为 -1。调试器在 REPL 与运行模式之间
        往返切换, 会反复调用 ``start()`` 和 ``stop()``; 第二次 ``start()``
        必须先重建管道, 否则 daemon 线程拿到 -1, 在 ``poll.register(-1)`` 处
        抛未捕获的 ValueError。
        """
        self._tx_notify_r, self._tx_notify_w = os.pipe()
        # 写端非阻塞: 管道满时 Rust 不卡死, 通知丢失可接受
        # (daemon 会在下次 drain 时追上).
        fl_w = fcntl.fcntl(self._tx_notify_w, fcntl.F_GETFL)
        fcntl.fcntl(self._tx_notify_w, fcntl.F_SETFL, fl_w | os.O_NONBLOCK)
        # 读端阻塞: daemon 线程 select/poll 等待

        self._rx_notify_r, self._rx_notify_w = os.pipe()
        fl_rx = fcntl.fcntl(self._rx_notify_w, fcntl.F_GETFL)
        fcntl.fcntl(self._rx_notify_w, fcntl.F_SETFL, fl_rx | os.O_NONBLOCK)

        self._rx_drain_r, self._rx_drain_w = os.pipe()
        fl_dr = fcntl.fcntl(self._rx_drain_w, fcntl.F_GETFL)
        fcntl.fcntl(self._rx_drain_w, fcntl.F_SETFL, fl_dr | os.O_NONBLOCK)

    def __del__(self) -> None:
        """对象回收兜底: 未显式 stop() 的实例也释放 fd (防测试/长进程泄漏)."""
        try:
            self._close_pipes()
        except Exception:
            pass

    def pause(self) -> None:
        """暂停 I/O 线程 (调试器断点)."""
        if self._native_active:
            self._pause_flag.value = 1

    def resume(self) -> None:
        """恢复 I/O 线程."""
        self._pause_flag.value = 0

    def _notify_rx_drained(self) -> None:
        """向管道写端写 1 字节, 唤醒 Rust termio 线程.

        用于两种场景: (1) drain_rx 消费了 ring buffer 后, 让 Rust
        投递 recv 中剩余字节; (2) stop 时唤醒阻塞在 poll(-1) 的线程使其看到
        stop_flag 退出, 修复 join 挂起。写端非阻塞, 满则通知丢失可接受。
        """
        try:
            os.write(self._rx_drain_w, b"\x01")
        except OSError:
            pass

    def drain_rx(self) -> bool:
        """将 Rust 线程写入的 RX 环形缓冲数据转移到 Python UART 模型.
        - 先查 ``uart.can_rx()`` (FIFO 有空位才接收)
        - 逐字节 preload, 每字节检查 can_rx; FIFO 满即停止
        - 未被消费的字节保留在 ring buffer 中 (rd 不推进),
          且因 rx_rd 未更新, Rust 端 ``free = cap - (wr - rd)`` 变小
          ->termio 线程自然容量控制停止读 stdin (对照 QEMU fd_chr_read_poll)

        Returns:
            True 若有新数据被 preload (可触发 WFI 唤醒).
        """
        with self._rx_drain_lock:
            return self._drain_rx_locked()

    def _drain_rx_locked(self) -> bool:
        """drain_rx 的锁内实现 (调用方须持有 _rx_drain_lock)."""
        if self._uart is None:
            return False
        if not self._uart.can_rx():
            return False
        wr = self._rx_wr.value
        rd = self._rx_rd.value
        if wr == rd:
            return False
        pending = mask32(wr - rd)
        cap = self.RX_CAP
        had_input = False
        drained = 0
        # 只消费 UART FIFO 能接受的数量; 剩余留在 ring buffer 等下次 drain
        for _ in range(pending):
            if not self._uart.can_rx():
                break
            b = self._rx_buf[rd % cap]
            rd = mask32(rd + 1)
            self._uart.preload(bytes([b]))
            drained += 1
            had_input = True
        self._rx_rd.value = rd  # 发布消费进度 (Rust 容量控制依据)
        if drained > 0:
            self._notify_rx_drained()  # 唤醒 Rust 投递 recv 剩余字节 (反压解除)
        # _rx_notify 不在此处清零 — RX daemon 抢先 drain 后
        # 调用加速执行所用的动态链接库仍需看到通知，以触发快速退出
        # 清零由 idle poll 路径在确认 ring buffer 为空后负责.
        if had_input:
            if self._on_irq is not None and self._uart._irq > 0:
                # 走 Emulator.raise_device_irq: 按当前 UART 电平同步 PLIC 挂起
                # + write-through 持久数组 + 置 ext_irq 通知位。若仅置
                # _ext_irq.pending 而不写持久数组, batch 内 Rust 的
                # plic_recompute_mip 会用 batch 起始的旧数组 (源挂起=0) 覆盖
                # SEIP, 中断永远投递不出去 -> UART 输入冻结 (见 CHANGELOG).
                self._on_irq(self._uart._irq, self._uart.irq_asserted())
            elif self._ext_irq is not None:
                self._ext_irq.pending = 1  # 通知 CPU 引擎内联投递 SEIP/MEIP
            # 仅在有实际输入时唤醒 WFI 空闲睡眠 — 空 drain (ring 空) 不得
            # set: 否则 daemon 每 0.5s 周期 drain 一次, _wake_event 被无限
            # 重新置位, _wfi_sleep_if_idle 的 wait 永不阻塞 -> 全 hart WFI
            # 空闲退化为 100% CPU 忙转 (宿主卡顿 / 输入无响应).
            self._wake_event.set()
        return had_input

    def drain_rx_feed_uart(self) -> bool:
        """排空 RX ring buffer -> UART FIFO, 并安全清零 _rx_notify 通知位.

        在 ``drain_rx()`` 的基础上补充 ``_rx_notify`` 的清零: 该位由 Rust
        termio 线程在写入 ring buffer 后置 1, 需在确认 ring buffer 为空后
        才清零, 清零与判空之间 Rust 可能写入新数据 (竞态), 必须二次校验
        恢复, 否则 CPU 引擎的快速退出信号丢失.

        供调试器空闲轮询路径调用 (原逻辑散落在 debug/base._feed_uart_stdin
        中直接操作私有字段, 见"避免隐式依赖"设计原则). 普通 ``drain_rx()``
        不清零通知位 — RX daemon 抢先排空后 CPU 引擎仍需看到通知以触发
        快速单轮加速执行退出.

        Returns:
            True 若有新数据被 preload (可触发 WFI 唤醒).
        """
        had_input = self.drain_rx()
        if self._rx_wr.value == self._rx_rd.value:
            self._rx_notify.value = 0
            if self._rx_wr.value != self._rx_rd.value:
                self._rx_notify.value = 1
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
        """排空 TX ring buffer, 归档 hart 日志并回填 UART 行缓冲.

        用于 Rust libc::write 已即时输出到 stdout 的场景; 此处只保证
        ring buffer 不溢出 + hart 日志完整 + ``tx_data()`` 可读, 不重复
        输出到控制台 (``record_tx_byte`` 仅追加内存缓冲, 不触发回调).
        """
        if self._uart is None:
            return
        with self._tx_drain_lock:
            ecap = self.TX_CAP // 2
            wr = self._tx_wr.value
            pending = mask32(wr - self._tx_log_rd)
            if pending == 0:
                return
            if pending > ecap:
                self._tx_log_rd = mask32(wr - ecap)
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
                # 回填 UART 行缓冲 (tx_data 调试/测试用) — 不触发 stdout
                uart.record_tx_byte(byte)
                rd = mask32(rd + 1)
            self._tx_log_rd = rd

    def _drain_tx_logs_locked(self) -> None:
        """drain_tx_logs 的锁内实现."""
        if self._uart is None:
            return
        ecap = self.TX_CAP // 2
        wr = self._tx_wr.value
        pending = mask32(wr - self._tx_log_rd)
        if pending == 0:
            return
        if pending > ecap:
            self._tx_log_rd = mask32(wr - ecap)
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
            rd = mask32(rd + 1)
        self._tx_log_rd = rd

    # ----------------------------------------------------------
    #  TX 归档 daemon 线程 — 异步将 ring buffer 写入 hart 日志文件,
    # 不依赖单轮加速执行边界, 与指令执行循环完全解耦.
    # ----------------------------------------------------------

    def start_tx_archive_thread(self) -> None:
        """启动 TX 归档 daemon: 事件驱动地将 ring buffer 写入 hart 日志."""
        if self._tx_archive_running:
            return
        if self._tx_notify_r < 0:
            # 管道已被 stop() 关闭且未重建: 线程启动即抛 ValueError, 不启动.
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

    def _drain_notify_pipe(self, notify_r: int) -> bool:
        """非阻塞排空通知管道中的积压字节.

        读端默认阻塞, 直接 ``os.read`` 会因管道为空挂起 daemon, 故先切非阻塞.
        返回 ``False`` 表示管道已失效 (读到 EOF 或操作出错), 调用方应停止
        poll 该 fd; 返回 ``True`` 表示本次排空正常完成, 后续可能还有通知.
        """
        try:
            os.set_blocking(notify_r, False)
            try:
                while True:
                    if os.read(notify_r, 256) == b"":
                        return False  # 写端已关闭 (EOF)
            except BlockingIOError:
                pass
            finally:
                os.set_blocking(notify_r, True)
        except OSError:
            return False
        return True

    def _tx_archive_loop(self) -> None:
        """TX 归档 daemon 入口: 轮询 + 事件驱动将 ring buffer 写入 hart 日志.

        与单轮加速执行循环完全异步。Rust 引擎实际从不写 ``tx_notify_fd``
        (state.rs 中字段恒为 -1), 通知管道永远不会可读 —— 纯事件驱动会导致
        daemon 在 poll 超时后直接落入阻塞 ``os.read`` 而永久挂起, ring buffer
        永不归档 (hart 日志近乎为空)。故每次 poll 超时 (0.2s 轮询间隔) 后
        无条件 drain 一次, 通知仅作即时唤醒加速; drain 前先按非阻塞方式清空
        管道, 防止空管道阻塞 daemon 线程。

        等待通知用 ``select.poll`` 而非 ``select.select``: select() 只能 watch
        fd < FD_SETSIZE (1024), 对高位 fd 抛 ``ValueError: filedescriptor out of
        range``, 捕获即 break 会使 daemon 静默死亡。长进程/测试套件中 fd 会持续
        增长 (每实例泄漏 4 个管道 fd), 通知 fd 超过 1024 后 select 版必死;
        poll() 无此上限, 可 watch 任意 fd。
        """
        notify_r = self._tx_notify_r
        if notify_r < 0:
            # 管道已被 stop() 关闭: 无可注册 fd, 线程直接退出 (下一次 start()
            # 会先重建管道再启动线程, 见 start() 顶部的守卫).
            return
        poll = select.poll()
        poll.register(notify_r, select.POLLIN)
        while self._tx_archive_running:
            try:
                ready = poll.poll(200)  # 0.2s 轮询周期 (毫秒)
            except (ValueError, OSError):
                break
            if not self._tx_archive_running:
                break
            if ready:
                # 通知仅作即时唤醒; 归档不依赖它 (Rust 从不写通知管道),
                # 管道失效 (EOF/出错) 的返回值得忽略.
                self._drain_notify_pipe(notify_r)
            # 无条件轮询归档 — Rust 不写通知管道, 必须靠 0.2s 周期追上.
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
        if self._tx_notify_r < 0:
            # 管道已被 stop() 关闭且未重建: 线程启动即抛 ValueError, 不启动.
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

        Rust 每写一个 TX 字节后写 1 字节到 _tx_notify_w -> poll 唤醒 ->
        drain_tx_logs() 走完整 UART 回调链 -> TermProxy writer -> stdout.
        零轮询, 完全事件驱动. 等待通知用 poll (见 _tx_archive_loop 注释:
        select() 无法 watch fd >= 1024, 高位 fd 下会静默死亡).
        """
        notify_r = self._tx_notify_r
        if notify_r < 0:
            return  # 管道已被 stop() 关闭, 无法 poll (见 _tx_archive_loop 注释)
        poll = select.poll()
        poll.register(notify_r, select.POLLIN)
        while self._tx_drain_running:
            try:
                # 阻塞等待通知, 无 fd 上限. 返回值不查: timeout=-1 时 poll
                # 只会因注册 fd 就绪/POLLHUP 返回, 不会返回空列表; 就绪后
                # 无论事件类型一律走下方排空.
                poll.poll(-1)
            except (ValueError, OSError):
                break
            if not self._tx_drain_running:
                break
            if not self._drain_notify_pipe(notify_r):
                break  # 写端已关闭 (EOF): 退出线程, 避免无限忙转
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
        if self._rx_daemon_running or self._stdin_fd < 0 or self._rx_notify_r < 0:
            # dummy fd (测试环境) 或管道已被 stop() 关闭且未重建: 无法 poll
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

    def _rx_daemon_tick(self, notify_r: int, poll: select.poll, timeout_ms: int = 500) -> bool:
        """单次 RX daemon 迭代: 等待通知 (≤ timeout_ms), 无论是否就绪都排空 ring.

        事件驱动 + 超时兜底双重保障:
        - 通知到达 -> poll 立即返回 -> 排空 ring buffer 到 UART FIFO.
        - 通知字节可能丢失: Rust termio 线程的写端非阻塞, 突发输入时管道满
          写 EAGAIN 通知被吞; 或 daemon 恰好排空管道后新数据在轮询间隔内到达
          (通知写发生在 poll 返回之前, 字节残留管道中但 poll 已不再可读).
          只靠通知唤醒会让输入滞留 ring buffer — 快速输入回显卡死 (用户报告).
          poll 超时后无条件排空一次兜底, 输入最迟 timeout_ms 内送达 UART FIFO.
        - ring 空时 drain_rx 是廉价 no-op (wr==rd 即返回), 轮询开销可忽略.

        Args:
            notify_r: 通知管道读端 fd.
            poll: 已注册 notify_r 的 select.poll 实例.
            timeout_ms: 轮询超时 (毫秒), 测试可注入小值避免拖慢用例.

        Returns:
            False — 通知管道已失效 (EOF/出错), 调用方应停止 poll 该 fd.
        """
        try:
            ready = poll.poll(timeout_ms)
        except (ValueError, OSError):
            return False
        if ready:
            if not self._drain_notify_pipe(notify_r):
                return False  # 管道已失效 (EOF/出错): 停止 poll 该 fd
        # 无论有无通知都排空 — 见 docstring 的超时兜底设计 (lost-notify 自愈).
        self.drain_rx()
        return True

    def _rx_daemon_loop(self) -> None:
        """RX daemon 入口: 事件驱动 + 超时兜底, poll 等待通知后无条件排空.

        Rust termio 线程写 ring buffer 后向 _rx_notify_w 写 1 字节 -> poll
        立即返回 -> drain_rx() 搬运到 UART FIFO。突发输入时通知字节可能丢失
        (写端非阻塞 + 管道满 EAGAIN), 故每次 poll 超时后也排空一次, 保证输入
        最迟 0.5s 内送达 (与 _tx_archive_loop 的 0.2s 轮询兜底同一模式).
        等待通知用 poll (见 _tx_archive_loop 注释: select() 无法 watch fd >= 1024).
        """
        notify_r = self._rx_notify_r
        if notify_r < 0:
            return  # 管道已被 stop() 关闭, 无法 poll (见 _tx_archive_loop 注释)
        poll = select.poll()
        poll.register(notify_r, select.POLLIN)
        while self._rx_daemon_running:
            if not self._rx_daemon_tick(notify_r, poll):
                break

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
        h.rx_drain_fd = self._rx_drain_r
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
        if stdin_fd < 0:
            return  # stdin 已关闭, 无可轮询的 fd (与各 daemon 线程一致)
        poll = select.poll()
        poll.register(stdin_fd, select.POLLIN)
        while self._fallback_running:
            try:
                # 20ms 超时轮询 stdin; poll 无 fd>=1024 上限 (见 _tx_archive_loop).
                ready = poll.poll(20)
            except (ValueError, OSError):
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
