//! Terminal I/O background thread — QEMU chardev-stdio 模型的 Rust 移植.
//!
//! 对照 `vendor/qemu-10.2.0/chardev/char-stdio.c` / `char-fd.c`:
//!
//! | QEMU                                | 本实现                                      |
//! |-------------------------------------|---------------------------------------------|
//! | `qemu_chr_open_stdio` (term_init)   | `terminal_io_start`: 保存 termios + fcntl   |
//! | `qemu_chr_set_echo_stdio(false)`    | raw-ish 模式: 关 ECHO/ICANON, 保留 OPOST/ISIG |
//! | `term_exit` (atexit + finalize)     | 线程退出路径恢复 termios + 两个 fd 的阻塞标志  |
//! | `term_stdio_handler` (SIGCONT)      | `sigcont_handler`: Ctrl+Z 恢复后重设 raw     |
//! | `fd_chr_read_poll` ->`can_write`    | RX 环形缓冲容量控制: 满时不读 stdin (留内核缓冲)  |
//! | `fd_chr_write` ->非阻塞 `write(1)`   | TX drain: 块写 stdout, EAGAIN 时 POLLOUT 等待 |
//! | glib 事件循环 (fd 驱动)              | 5ms `poll` 轮询 (跨 cdylib 无共享 eventfd)   |
//!
//! # 单一 owner 原则 (QEMU chardev 的核心不变量)
//!
//! - **stdout**: 线程运行期间, 本线程是唯一控制台写者。Python 侧 UART 行缓冲
//!   仅归档 hart 日志文件, 不再重复回显 (`UART.set_console_echo(False)`)。
//! - **tx_drain**: 本线程独占写; Python 日志消费者维护自己的独立读索引。
//! - **rx_wr**: 本线程独占写; **rx_rd**: Python 独占写 (drain 后发布消费进度)。
//!
//! 线程退出前执行最终 TX drain, 保证 `tx_drain == tx_wr` — Python 可据此
//! 判定所有已产生的输出均已写入 stdout。
//!
//! # 索引环绕约定
//!
//! 所有索引单调递增 (u32 wrapping), 槽位 = 索引 % 容量。容量必须为 2 的幂
//! (需整除 2^32, 否则索引跨 u32 环绕时槽位错位)。

use std::os::unix::io::RawFd;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicPtr, AtomicU32, AtomicU8, Ordering};
use std::sync::Mutex;

// ============================================================
//  FFI context — C-repr struct passed from Python ctypes
// ============================================================

/// Shared state for the terminal I/O thread, allocated by Python and passed
/// once at startup.  All pointers remain valid for the lifetime of the thread.
#[repr(C)]
pub struct TermIoHandle {
    /// File descriptor for stdin (usually 0).
    pub stdin_fd: RawFd,
    /// File descriptor for stdout (usually 1).
    pub stdout_fd: RawFd,
    /// Shared RX ring buffer — I/O thread writes, Python reads on guest RXDATA.
    pub rx_buf: *mut u8,
    /// Capacity of ``rx_buf`` in bytes (power of two).
    pub rx_cap: u32,
    /// RX write index — I/O thread exclusive writer (monotonic u32).
    pub rx_wr: *mut AtomicU32,
    /// RX read index — Python exclusive writer.  容量控制依据:
    /// 剩余空间 = rx_cap - (rx_wr - rx_rd); 为 0 时线程停止读 stdin,
    /// 数据留在内核 tty 缓冲 (对照 QEMU fd_chr_read_poll 流控).
    pub rx_rd: *mut AtomicU32,
    /// Shared TX ring buffer — CPU hart threads write (guest TXDATA).
    /// Layout: entry e occupies bytes [2e] = hart_id, [2e+1] = byte value.
    pub tx_buf: *mut u8,
    /// Entry capacity of ``tx_buf`` (= byte capacity / 2, power of two).
    pub tx_cap: u32,
    /// TX write index (entry count) — CPU engine exclusive writer (monotonic u32).
    pub tx_wr: *mut AtomicU32,
    /// TX drain index — I/O thread exclusive writer.  Python 不写此索引
    /// (日志消费者持有独立读索引), 消除旧实现的双写者竞争.
    pub tx_drain: *mut AtomicU32,
    /// Stop flag — Python sets to 1 to request thread exit.
    pub stop_flag: *mut AtomicU8,
    /// Pause flag — Python sets to 1 on debugger break, 0 to resume.
    pub pause_flag: *mut AtomicU8,
}

// Safety: 所有指针由 Python (ctypes 数组) 持有并在线程生命周期内保持有效;
// 各索引遵循上述单写者约定。
unsafe impl Send for TermIoHandle {}

/// 启动时保存的终端状态, 由线程在退出路径恢复 (对照 char-stdio.c term_exit:
/// 恢复 termios 及 *两个* fd 的阻塞标志 — 遗留 O_NONBLOCK 会破坏同一 tty 上
/// 的后续程序, 见 QEMU commit 6807403).
struct SavedTerm {
    oldtty: libc::termios,
    old_fl0: libc::c_int,
    old_fl1: libc::c_int,
}

// ============================================================
//  Global state — thread handle + SIGCONT re-apply support
// ============================================================

static THREAD_HANDLE: Mutex<Option<std::thread::JoinHandle<()>>> = Mutex::new(None);

/// stop_flag 指针副本 — terminal_io_stop 自行置位, 不依赖 Python 先设置.
static STOP_PTR: AtomicPtr<AtomicU8> = AtomicPtr::new(std::ptr::null_mut());

/// SIGCONT handler 所需状态 (对照 char-stdio.c 的 static oldtty/stdio_echo_state):
/// Ctrl+Z 挂起期间 shell 会把终端复位为 cooked 模式, 恢复运行后需重设 raw。
static TERM_ACTIVE: AtomicBool = AtomicBool::new(false);
static RAW_TTY_PTR: AtomicPtr<libc::termios> = AtomicPtr::new(std::ptr::null_mut());
static SIG_STDIN_FD: AtomicI32 = AtomicI32::new(0);

/// 原 SIGCONT 处置, stop 时恢复。libc::sigaction 含裸指针故手动断言 Send。
struct SigActionCell(libc::sigaction);
unsafe impl Send for SigActionCell {}
static OLD_SIGCONT: Mutex<Option<SigActionCell>> = Mutex::new(None);

extern "C" fn sigcont_handler(_sig: libc::c_int) {
    // 仅调用 tcsetattr (POSIX async-signal-safe); QEMU term_stdio_handler 同款。
    if !TERM_ACTIVE.load(Ordering::Acquire) {
        return;
    }
    let raw = RAW_TTY_PTR.load(Ordering::Acquire);
    if raw.is_null() {
        return;
    }
    let fd = SIG_STDIN_FD.load(Ordering::Acquire);
    unsafe {
        libc::tcsetattr(fd, libc::TCSANOW, raw);
    }
}

fn install_sigcont(stdin_fd: RawFd, raw: libc::termios) {
    // termios 槽为进程级单例 (一次性泄漏, 不随 start/stop 累积)。
    let mut p = RAW_TTY_PTR.load(Ordering::Acquire);
    if p.is_null() {
        p = Box::into_raw(Box::new(raw));
        RAW_TTY_PTR.store(p, Ordering::Release);
    } else {
        unsafe { *p = raw };
    }
    SIG_STDIN_FD.store(stdin_fd, Ordering::Release);
    TERM_ACTIVE.store(true, Ordering::Release);
    unsafe {
        let mut act: libc::sigaction = std::mem::zeroed();
        let f: extern "C" fn(libc::c_int) = sigcont_handler;
        act.sa_sigaction = f as usize;
        act.sa_flags = libc::SA_RESTART;
        let mut old: libc::sigaction = std::mem::zeroed();
        if libc::sigaction(libc::SIGCONT, &act, &mut old) == 0 {
            *OLD_SIGCONT.lock().unwrap() = Some(SigActionCell(old));
        }
    }
}

fn uninstall_sigcont() {
    TERM_ACTIVE.store(false, Ordering::Release);
    if let Some(SigActionCell(old)) = OLD_SIGCONT.lock().unwrap().take() {
        unsafe {
            libc::sigaction(libc::SIGCONT, &old, std::ptr::null_mut());
        }
    }
}

// ============================================================
//  I/O helpers
// ============================================================

/// 非阻塞 write 循环: EINTR 重试; EAGAIN 时 POLLOUT 等待可写后续传
/// (对照 sifive_uart_xmit 的 G_IO_OUT watch 续传模式)。
/// stop 请求时放弃剩余输出, 避免终端停滞导致 join 卡死。
fn write_all(fd: RawFd, buf: &[u8], stop: &AtomicU8) {
    let mut off = 0usize;
    while off < buf.len() {
        let n = unsafe {
            libc::write(
                fd,
                buf.as_ptr().add(off) as *const libc::c_void,
                buf.len() - off,
            )
        };
        if n >= 0 {
            off += n as usize;
            continue;
        }
        let err = unsafe { *libc::__errno_location() };
        if err == libc::EINTR {
            continue;
        }
        if err == libc::EAGAIN || err == libc::EWOULDBLOCK {
            if stop.load(Ordering::Acquire) != 0 {
                return;
            }
            let mut pfd = libc::pollfd {
                fd,
                events: libc::POLLOUT,
                revents: 0,
            };
            unsafe { libc::poll(&mut pfd, 1, 100) };
            continue;
        }
        return; // 其他错误: best-effort, 不让 I/O 线程崩溃
    }
}

/// 从 TX 环形缓冲收集待排空条目的字节到 chunk (跳过 hart_id 字节)。
/// 返回 (新 drain 索引, 收集的字节数)。纯函数, 便于单元测试环绕语义。
fn gather_tx_chunk(
    buf: &[u8],
    ecap: u32,
    mut drain: u32,
    wr: u32,
    chunk: &mut [u8],
) -> (u32, usize) {
    let mut n = 0usize;
    while drain != wr && n < chunk.len() {
        let e = (drain % ecap) as usize;
        chunk[n] = buf[2 * e + 1];
        n += 1;
        drain = drain.wrapping_add(1);
    }
    (drain, n)
}

/// 排空 TX 环形缓冲到 stdout。本线程是 tx_drain 的唯一写者。
fn drain_tx(h: &TermIoHandle, chunk: &mut [u8]) {
    let ecap = h.tx_cap;
    if ecap == 0 || h.tx_buf.is_null() {
        return;
    }
    let tx_wr = unsafe { &*h.tx_wr };
    let tx_drain = unsafe { &*h.tx_drain };
    let stop = unsafe { &*h.stop_flag };
    let buf = unsafe { std::slice::from_raw_parts(h.tx_buf, (ecap as usize) * 2) };
    loop {
        // Acquire 配对 CPU 引擎发布 tx_wr 的 Release: 索引可见 ⇒ 槽位数据可见
        let wr = tx_wr.load(Ordering::Acquire);
        let drain = tx_drain.load(Ordering::Relaxed); // 单写者: 本线程
        if drain == wr {
            return;
        }
        let (new_drain, n) = gather_tx_chunk(buf, ecap, drain, wr, chunk);
        write_all(h.stdout_fd, &chunk[..n], stop);
        tx_drain.store(new_drain, Ordering::Release);
    }
}

/// Read available data from stdin into RX ring buffer.  Returns false on EOF.
fn read_stdin_to_rx(h: &TermIoHandle, free: usize, buf: &mut [u8]) -> (bool, usize) {
    let n = unsafe { libc::read(h.stdin_fd, buf.as_mut_ptr() as *mut libc::c_void, free.min(buf.len())) };
    if n > 0 {
        let mut wr = unsafe { &*h.rx_wr }.load(Ordering::Relaxed);
        let rx = unsafe { std::slice::from_raw_parts_mut(h.rx_buf, h.rx_cap as usize) };
        for &b in &buf[..n as usize] {
            rx[(wr % h.rx_cap) as usize] = b;
            wr = wr.wrapping_add(1);
        }
        unsafe { &*h.rx_wr }.store(wr, Ordering::Release);
        return (true, n as usize);
    }
    (n != 0, 0) // n==0 -> EOF; n<0 -> EAGAIN, ignore
}

// ============================================================
//  Thread entry point
// ============================================================

fn termio_thread(h: TermIoHandle, saved: SavedTerm) {
    // QEMU 事件循环由 fd 驱动 (零延迟); 跨 cdylib 无共享 eventfd 可 poll,
    const POLL_MS: libc::c_int = 5;

    let stop = unsafe { &*h.stop_flag };
    let rx_wr = unsafe { &*h.rx_wr };
    let rx_rd = unsafe { &*h.rx_rd };
    let rx_cap = h.rx_cap;
    let mut stdin_buf = [0u8; 1024];
    let mut chunk = [0u8; 4096];
    let mut stdin_eof = false;

    while stop.load(Ordering::Acquire) == 0 {
        // ---- RX 容量控制 (对照 fd_chr_read_poll ->qemu_chr_be_can_write):
        // 环形缓冲满时不监听 POLLIN, 数据自然滞留在内核 tty 缓冲。
        let used = rx_wr
            .load(Ordering::Relaxed)
            .wrapping_sub(rx_rd.load(Ordering::Acquire));
        let free = rx_cap.saturating_sub(used) as usize;

        let mut pfd = libc::pollfd {
            fd: h.stdin_fd,
            events: libc::POLLIN,
            revents: 0,
        };
        let nfds: libc::nfds_t = if !stdin_eof && free > 0 && rx_cap > 0 { 1 } else { 0 };
        // nfds == 0 时 poll 退化为纯睡眠 (对照 QEMU 摘除 fd watch 后的空转)
        let ret = unsafe { libc::poll(&mut pfd, nfds, POLL_MS) };
        if ret < 0 {
            let err = unsafe { *libc::__errno_location() };
            if err == libc::EINTR {
                continue;
            }
            break;
        }

        // ---- read stdin -> RX ring buffer (参照 fd_chr_read) ----
        if nfds == 1 && ret > 0 && (pfd.revents & libc::POLLIN) != 0 {
            let (ok, _) = read_stdin_to_rx(&h, free, &mut stdin_buf);
            stdin_eof = !ok;
        } else if nfds == 1 && ret > 0 {
            stdin_eof = true;
        }

        // ---- TX 环形缓冲 ->stdout ----
        drain_tx(&h, &mut chunk);
    }

    // 退出前最终排空: 保证 tx_drain == tx_wr, 所有已产生输出均已写入 stdout。
    drain_tx(&h, &mut chunk);

    // ---- term_exit (对照 char-stdio.c): 恢复 termios + 两个 fd 的阻塞标志 ----
    unsafe {
        libc::tcsetattr(h.stdin_fd, libc::TCSANOW, &saved.oldtty);
        libc::fcntl(h.stdin_fd, libc::F_SETFL, saved.old_fl0);
        libc::fcntl(h.stdout_fd, libc::F_SETFL, saved.old_fl1);
    }
}

/// Simple I/O thread — no terminal setup (Python cbreak manages it).
/// Supports pause via ``pause_flag`` for debugger break/resume.
fn termio_thread_simple(h: TermIoHandle) {
    const POLL_MS: libc::c_int = 5;
    let stop = unsafe { &*h.stop_flag };
    let pause: *const AtomicU8 = h.pause_flag;
    let rx_wr = unsafe { &*h.rx_wr };
    let rx_rd = unsafe { &*h.rx_rd };
    let rx_cap = h.rx_cap;
    let mut stdin_buf = [0u8; 1024];
    let mut chunk = [0u8; 4096];
    let mut stdin_eof = false;

    while stop.load(Ordering::Acquire) == 0 {
        // ---- debugger pause: sleep, skip I/O ----
        if !pause.is_null() && unsafe { &*pause }.load(Ordering::Acquire) != 0 {
            std::thread::sleep(std::time::Duration::from_millis(POLL_MS as u64));
            continue;
        }
        // ---- RX backpressure ----
        let used = rx_wr.load(Ordering::Relaxed).wrapping_sub(rx_rd.load(Ordering::Acquire));
        let free = rx_cap.saturating_sub(used) as usize;
        let mut pfd = libc::pollfd { fd: h.stdin_fd, events: libc::POLLIN, revents: 0 };
        let nfds: libc::nfds_t = if !stdin_eof && free > 0 && rx_cap > 0 { 1 } else { 0 };
        let ret = unsafe { libc::poll(&mut pfd, nfds, POLL_MS) };
        if ret < 0 {
            if unsafe { *libc::__errno_location() } == libc::EINTR { continue; }
            break;
        }
        // ---- read stdin -> RX ring buffer (参照 fd_chr_read) ----
        if nfds == 1 && ret > 0 && (pfd.revents & libc::POLLIN) != 0 {
            let (ok, _) = read_stdin_to_rx(&h, free, &mut stdin_buf);
            stdin_eof = !ok;
        } else if nfds == 1 && ret > 0 {
            stdin_eof = true;
        }
        // ---- TX ring buffer -> stdout ----
        drain_tx(&h, &mut chunk);
    }
    // final drain
    drain_tx(&h, &mut chunk);
}

// ============================================================
//  FFI entry points
// ============================================================

/// Start the terminal I/O background thread.
///
/// 对照 `qemu_chr_open_stdio`: 保存 termios/fcntl ->stdin/stdout 非阻塞 ->
/// raw-ish 模式 (关 ECHO/ICANON/IEXTEN + ICRNL/IXON, 保留 OPOST 与 ISIG) ->
/// 安装 SIGCONT handler ->启动 I/O 线程。
///
/// Returns 0 on success, -1 if already running or stdin is not a TTY.
/// Caller must ensure `handle` points to a valid `TermIoHandle` whose
/// buffers stay alive for the lifetime of the thread.
#[no_mangle]
pub extern "C" fn terminal_io_start(handle: *const TermIoHandle) -> i32 {
    if handle.is_null() {
        return -1;
    }
    {
        let guard = THREAD_HANDLE.lock().unwrap();
        if guard.is_some() {
            return -1; // 已运行 (对照 QEMU stdio_in_use 单例守卫)
        }
    }

    let h = unsafe { &*handle };
    // 复制所有字段 — handle 位于 Python 栈上, FFI 调用返回后即失效。
    let hc = TermIoHandle {
        stdin_fd: h.stdin_fd,
        stdout_fd: h.stdout_fd,
        rx_buf: h.rx_buf,
        rx_cap: h.rx_cap,
        rx_wr: h.rx_wr,
        rx_rd: h.rx_rd,
        tx_buf: h.tx_buf,
        tx_cap: h.tx_cap,
        tx_wr: h.tx_wr,
        tx_drain: h.tx_drain,
        stop_flag: h.stop_flag,
        pause_flag: h.pause_flag,
    };

    // ---- term_init ----
    let mut oldtty: libc::termios = unsafe { std::mem::zeroed() };
    if unsafe { libc::tcgetattr(hc.stdin_fd, &mut oldtty) } != 0 {
        return -1; // 非 TTY (管道/重定向) — Python 侧回退
    }
    let old_fl0 = unsafe { libc::fcntl(hc.stdin_fd, libc::F_GETFL) };
    let old_fl1 = unsafe { libc::fcntl(hc.stdout_fd, libc::F_GETFL) };
    if old_fl0 < 0 || old_fl1 < 0 {
        return -1;
    }
    unsafe {
        libc::fcntl(hc.stdin_fd, libc::F_SETFL, old_fl0 | libc::O_NONBLOCK);
        libc::fcntl(hc.stdout_fd, libc::F_SETFL, old_fl1 | libc::O_NONBLOCK);
    }

    // raw-ish 模式, 始终从 oldtty 副本重新计算 (幂等, 无漂移):
    // 逐位对照 qemu_chr_set_echo_stdio(echo=false)。
    let mut raw = oldtty;
    // QEMU 清除 ICRNL (依赖客机 tty 层做 \r->\n 转换), 但在 pyremu
    // 客机串口控制台可能未正确初始化 ICRNL (如 dash 作为 PID 1 运行且无
    // devtmpfs 时), 导致 Enter 键的 \r 不被识别为行终止符 — 用户需按两次
    // 回车才能触发命令执行。保留 ICRNL 由宿主机内核完成转换, 消除此问题。
    raw.c_iflag &= !(libc::IGNBRK
        | libc::BRKINT
        | libc::PARMRK
        | libc::ISTRIP
        | libc::INLCR
        | libc::IGNCR
        // ICRNL 保留: 宿主机将 \r->\n, 客机收到 \n 即可行终止
        | libc::IXON);
    raw.c_oflag |= libc::OPOST; // 保留输出后处理: '\n' ->CRLF
    // 保留 ISIG (对照 QEMU stdio 默认 signal=on): Ctrl+C ->SIGINT ->调试器暂停回 REPL
    raw.c_lflag &= !(libc::ECHO | libc::ECHONL | libc::ICANON | libc::IEXTEN);
    raw.c_cflag &= !(libc::CSIZE | libc::PARENB);
    raw.c_cflag |= libc::CS8;
    raw.c_cc[libc::VMIN] = 1; // 每个按键即时可读
    raw.c_cc[libc::VTIME] = 0;
    if unsafe { libc::tcsetattr(hc.stdin_fd, libc::TCSANOW, &raw) } != 0 {
        unsafe {
            libc::fcntl(hc.stdin_fd, libc::F_SETFL, old_fl0);
            libc::fcntl(hc.stdout_fd, libc::F_SETFL, old_fl1);
        }
        return -1;
    }

    // Ctrl+Z 挂起恢复后重设 raw (对照 term_stdio_handler)
    install_sigcont(hc.stdin_fd, raw);
    STOP_PTR.store(hc.stop_flag, Ordering::Release);

    let saved = SavedTerm {
        oldtty,
        old_fl0,
        old_fl1,
    };
    let jh = std::thread::spawn(move || {
        termio_thread(hc, saved);
    });
    *THREAD_HANDLE.lock().unwrap() = Some(jh);
    0
}

/// Attach I/O thread to already-configured fds — no termios changes.
/// Terminal mode (cbreak) is managed by Python; this just reads/writes.
#[no_mangle]
pub extern "C" fn terminal_io_attach(handle: *const TermIoHandle) -> i32 {
    if handle.is_null() { return -1; }
    {
        let guard = THREAD_HANDLE.lock().unwrap();
        if guard.is_some() { return -1; }
    }
    let h = unsafe { &*handle };
    let hc = TermIoHandle {
        stdin_fd: h.stdin_fd, stdout_fd: h.stdout_fd,
        rx_buf: h.rx_buf, rx_cap: h.rx_cap, rx_wr: h.rx_wr, rx_rd: h.rx_rd,
        tx_buf: h.tx_buf, tx_cap: h.tx_cap, tx_wr: h.tx_wr, tx_drain: h.tx_drain,
        stop_flag: h.stop_flag, pause_flag: h.pause_flag,
    };
    STOP_PTR.store(hc.stop_flag, Ordering::Release);
    let jh = std::thread::spawn(move || { termio_thread_simple(hc); });
    *THREAD_HANDLE.lock().unwrap() = Some(jh);
    0
}

/// Stop the I/O thread and wait for it to exit.
///
/// 自行置位 stop flag (不依赖 Python 先设置), join 线程 (线程在退出路径
/// 完成最终 TX drain + 终端恢复), 然后恢复原 SIGCONT 处置。
#[no_mangle]
pub extern "C" fn terminal_io_stop() {
    let stop = STOP_PTR.load(Ordering::Acquire);
    if !stop.is_null() {
        unsafe { &*stop }.store(1, Ordering::Release);
    }
    let jh = THREAD_HANDLE.lock().unwrap().take();
    if let Some(handle) = jh {
        let _ = handle.join();
    }
    uninstall_sigcont();
}

/// Return 1 if the I/O thread is currently running, 0 otherwise.
#[no_mangle]
pub extern "C" fn terminal_io_is_running() -> i32 {
    if THREAD_HANDLE.lock().unwrap().is_some() {
        1
    } else {
        0
    }
}

// ============================================================
//  Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use core::mem::{offset_of, size_of};

    /// FFI 布局锁定: 与 Python ctypes TermIoHandle._fields_ 逐字节一致
    /// (pyremu/_native/__init__.py)。修改任一侧必须同步另一侧。
    #[test]
    fn test_handle_layout_locked() {
        assert_eq!(size_of::<TermIoHandle>(), 88);
        assert_eq!(offset_of!(TermIoHandle, stdin_fd), 0);
        assert_eq!(offset_of!(TermIoHandle, stdout_fd), 4);
        assert_eq!(offset_of!(TermIoHandle, rx_buf), 8);
        assert_eq!(offset_of!(TermIoHandle, rx_cap), 16);
        assert_eq!(offset_of!(TermIoHandle, rx_wr), 24);
        assert_eq!(offset_of!(TermIoHandle, rx_rd), 32);
        assert_eq!(offset_of!(TermIoHandle, tx_buf), 40);
        assert_eq!(offset_of!(TermIoHandle, tx_cap), 48);
        assert_eq!(offset_of!(TermIoHandle, tx_wr), 56);
        assert_eq!(offset_of!(TermIoHandle, tx_drain), 64);
        assert_eq!(offset_of!(TermIoHandle, stop_flag), 72);
        assert_eq!(offset_of!(TermIoHandle, pause_flag), 80);
    }

    #[test]
    fn test_gather_tx_chunk_basic() {
        // 4 个条目: (hid, byte) 对
        let ecap = 8u32;
        let mut buf = vec![0u8; (ecap as usize) * 2];
        for (i, ch) in b"abcd".iter().enumerate() {
            buf[2 * i] = 0; // hid
            buf[2 * i + 1] = *ch;
        }
        let mut chunk = [0u8; 16];
        let (d, n) = gather_tx_chunk(&buf, ecap, 0, 4, &mut chunk);
        assert_eq!(d, 4);
        assert_eq!(&chunk[..n], b"abcd");
    }

    #[test]
    fn test_gather_tx_chunk_ring_wrap() {
        // 索引跨越 ecap 边界: 条目 6,7,8,9 ->槽位 6,7,0,1
        let ecap = 8u32;
        let mut buf = vec![0u8; (ecap as usize) * 2];
        for (i, ch) in [(6usize, b'w'), (7, b'x'), (0, b'y'), (1, b'z')] {
            buf[2 * i + 1] = ch;
        }
        let mut chunk = [0u8; 16];
        let (d, n) = gather_tx_chunk(&buf, ecap, 6, 10, &mut chunk);
        assert_eq!(d, 10);
        assert_eq!(&chunk[..n], b"wxyz");
    }

    #[test]
    fn test_gather_tx_chunk_u32_wraparound() {
        // 索引跨越 u32 环绕: drain=0xFFFF_FFFE, wr=2 ->4 个条目
        // ecap 为 2 的幂 ⇒ 0xFFFF_FFFE % 8 = 6, 槽位连续 6,7,0,1
        let ecap = 8u32;
        let mut buf = vec![0u8; (ecap as usize) * 2];
        for (i, ch) in [(6usize, b'p'), (7, b'q'), (0, b'r'), (1, b's')] {
            buf[2 * i + 1] = ch;
        }
        let mut chunk = [0u8; 16];
        let (d, n) = gather_tx_chunk(&buf, ecap, 0xFFFF_FFFE, 2, &mut chunk);
        assert_eq!(d, 2);
        assert_eq!(&chunk[..n], b"pqrs");
    }

    #[test]
    fn test_gather_tx_chunk_respects_chunk_len() {
        let ecap = 8u32;
        let mut buf = vec![0u8; (ecap as usize) * 2];
        for i in 0..6 {
            buf[2 * i + 1] = b'0' + i as u8;
        }
        let mut chunk = [0u8; 4]; // 小于待排条目数
        let (d, n) = gather_tx_chunk(&buf, ecap, 0, 6, &mut chunk);
        assert_eq!(d, 4);
        assert_eq!(n, 4);
        assert_eq!(&chunk[..n], b"0123");
        // 续传剩余
        let (d2, n2) = gather_tx_chunk(&buf, ecap, d, 6, &mut chunk);
        assert_eq!(d2, 6);
        assert_eq!(&chunk[..n2], b"45");
    }
}
