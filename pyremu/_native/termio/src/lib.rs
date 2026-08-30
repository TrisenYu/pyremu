//! Terminal I/O background thread
//!
//! 两种模式:
//! - **start 模式** (`terminal_io_start` -> `termio_thread`): 本线程是唯一
//!   控制台写者, 持续 drain_tx 到 stdout。当前 Python 侧未启用此模式。
//! - **attach 模式** (`terminal_io_attach` -> `termio_thread_simple`): 本线程
//!   只负责 stdin -> RX ring; TX 到 stdout 由 CPU 引擎内联 `try_write_fd`
//!   即时输出 (no_stdout=0)。线程不得再写 stdout, 否则键盘输入唤醒 poll 时
//!   会把整个 TX ring 积压重放 (重复输出) 且 write_all 阻塞拖住输入读取。
//!
//! - **tx_drain**: start 模式下本线程独占写; Python 日志消费者维护自己的
//!   独立读索引 (_tx_log_rd), 不依赖 tx_drain。
//! - **rx_wr**: 本线程独占写; **rx_rd**: Python 独占写 (drain 后发布消费进度)。
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
	/// RX notification — termio 线程写环形缓冲后置 1,
	/// CPU 引擎每指令边界检查以触发内联外部中断投递.
	pub rx_notify: *mut AtomicU8,
	/// RX notify pipe write-end fd — termio 线程写 ring buffer 后
	/// 写 1 字节到此 fd 唤醒 Python RX daemon (select 事件驱动, 零轮询).
	pub rx_notify_fd: RawFd,
	/// RX drain pipe read-end fd — Python 消费 ring buffer (rx_rd 推进) 后
	/// 写 1 字节到其写端, 唤醒本线程投递 recv 中剩余可转发字节 (反压解除:
	/// 缓冲区满时等待被消耗). 本线程 poll 此 fd, 写端由 Python 持有.
	pub rx_drain_fd: RawFd,
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

/// CPU 引擎停止标志指针 — Ctrl+Q 时由 termio 线程直接置位, 绕过
/// SIGINT -> Python handler -> notify_processor 往返 (主线程阻塞于
/// run_parallel 时信号处理器不执行, 直连置位保证看门狗 5ms 内感知停止).
static EMU_STOP_PTR: AtomicPtr<AtomicU8> = AtomicPtr::new(std::ptr::null_mut());

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
	// 仅调用 tcsetattr (POSIX async-signal-safe);
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

// ============================================================
//  stdin 解析器: Ctrl+Q 停机检测 + 转义序列重组
//
//  对照 vendor/b0gus/terminal/line_editor_aux.go 的 byteSeqToRunes
//  维护方式: 所有输入先保留在接收缓冲 (不丢弃、不节流), 解析器每次
//  有界前瞻 (至多 6 字节窗口) 逐字节判定归属; 无法判定完整性的尾部
//  (如半截 ESC 序列) 保留在 recv 中, 等效 byteSeqToRunes 的 tmpKeep,
//  作为后续解析的开头。
// ============================================================

/// 直连置位 CPU 停止标志 — 与 rx_notify 同模式 (共享内存, 零 Python 往返).
/// 主线程阻塞于 run_parallel 时 SIGINT 处理器不执行, 此直连路径保证 Rust
/// 引擎看门狗 (5ms) 与逐指令检查仍能即时感知停止请求. 独立成函数以便单测.
#[inline]
fn notify_emu_stop() {
	let stop = EMU_STOP_PTR.load(Ordering::Acquire);
	if !stop.is_null() {
		unsafe { (*stop).store(1, Ordering::Release) };
	}
}

/// 判定 ESC 开头序列的完整性 (有界 6 字节窗口, 对照 byteSeqToRunes):
/// Some(n) — 序列判定完整, 共 n 字节; None — 当前数据不足以判定 (需等更多).
fn esc_seq_len(buf: &[u8]) -> Option<usize> {
	debug_assert!(!buf.is_empty() && buf[0] == 0x1b);
	if buf.len() == 1 {
		return None; // 孤立 ESC — 等下一字节区分 Alt 键 / 序列开头
	}
	let b1 = buf[1];
	if b1 != b'[' && b1 != b'O' {
		return Some(2); // Alt+key: ESC X
	}
	if b1 == b'O' {
		return Some(3); // SS3: ESC O X (3 字节)
	}
	// CSI: ESC [ ... final (0x40..=0x7e), 6 字节窗口内找 final 字节.
	for (idx, &b) in buf[2..].iter().take(4).enumerate() {
		if (0x40..=0x7e).contains(&b) {
			return Some(2 + idx + 1);
		}
	}
	// 窗口耗尽仍无 final: 若已积累超过 6 字节, 视为无法重组的长序列 —
	// 透传 ESC 单字节 (不阻塞, 剩余部分按普通字节继续转发, 客机终端自行重组).
	if buf.len() > 6 {
		return Some(1);
	}
	None // 不足 6 字节且无 final — 保留等待补全
}

/// 扫描 recv 前端: 消费连续 0x11 运行 (停机), 识别完整 ESC 序列, 在首个
/// 不完整 ESC 尾部停下。返回 (可投递字节数, 是否请求停机)。
///
///   - 可投递字节数 = recv 前端可安全搬移到 ring 的字节数 (不含已消费的
///     0x11 运行, 不含不完整的 ESC 尾)。0x11 运行被 drain 移除, 因此其
///     之后的字节随 drain 前移, 仍保留在 recv 中待下轮解析.
///   - 停机时 0x11 运行已从 recv 移除; 运行之前的可投递前缀长度即为返回值,
///     与运行之后的保留字节互不丢失.
fn scan_rx(recv: &mut Vec<u8>) -> (usize, bool) {
	let mut i = 0usize;
	while i < recv.len() {
		let b = recv[i];
		if b == 0x11 {
			// 消费整个连续 0x11 运行 — 多个连续 Ctrl+Q 视作同一次停机请求.
			let mut j = i;
			while j < recv.len() && recv[j] == 0x11 {
				j += 1;
			}
			recv.drain(i..j);
			return (i, true); // 停机动作 (notify_emu_stop) 由调用方执行
		}
		if b == 0x1b {
			match esc_seq_len(&recv[i..]) {
				Some(n) if n <= recv.len() - i => {
					i += n;
				}
				_ => {
					return (i, false); // 不完整 ESC 尾 — 保留等待补全 (等效 tmpKeep)
				}
			}
		} else {
			i += 1;
		}
	}
	(i, false)
}

/// 排空一个非阻塞 fd 的可读字节, 消除 poll 的 spurious 连续唤醒。忽略错误。
fn drain_fd(fd: RawFd) {
	if fd < 0 {
		return;
	}
	let mut buf = [0u8; 256];
	loop {
		let n = unsafe { libc::read(fd, buf.as_mut_ptr() as *mut libc::c_void, buf.len()) };
		if n > 0 {
			continue;
		}
		break;
	}
}

// ============================================================
//  stdin -> RX ring buffer (QEMU fd_chr_read model)
// ============================================================

/// 将 stdin 当前可读的所有字节追加到 recv (fd 为 O_NONBLOCK, 读到 EAGAIN)。
/// 不丢弃、不节流: 即使 ring 已满也照常读入, 交由 scan_rx/deliver_rx 分阶段
/// 处理, Ctrl+Q 因而在任何 ring 占用下都能被即时检测 (对照 byteSeqToRunes 的
/// 先读入再解析模型)。
/// Returns false on EOF.
fn read_stdin_into(h: &TermIoHandle, recv: &mut Vec<u8>) -> bool {
	let mut tmp = [0u8; 1024];
	loop {
		let n = unsafe { libc::read(h.stdin_fd, tmp.as_mut_ptr() as *mut libc::c_void, tmp.len()) };
		if n > 0 {
			recv.extend_from_slice(&tmp[..n as usize]);
			continue;
		}
		if n == 0 {
			return false;
		}
		// n < 0
		let err = unsafe { *libc::__errno_location() };
		if err == libc::EAGAIN || err == libc::EWOULDBLOCK {
			break;
		}
		break; // 其他错误 (EINTR 等): 停止本轮
	}
	true
}

/// 将 recv 前端 scan_rx 判定的可投递前缀搬移到 RX ring (受 free 容量约束,
/// 永不丢弃 — 装不下的保留在 recv 待下轮)。返回本批实际写入 ring 的字节数。
fn deliver_rx(h: &TermIoHandle, recv: &mut Vec<u8>, deliverable: usize) -> usize {
	if deliverable == 0 || h.rx_cap == 0 || h.rx_buf.is_null() {
		return 0;
	}
	let rx_wr = unsafe { &*h.rx_wr };
	let rx_rd = unsafe { &*h.rx_rd };
	let rx = unsafe { std::slice::from_raw_parts_mut(h.rx_buf, h.rx_cap as usize) };
	let mut wr = rx_wr.load(Ordering::Relaxed);
	let mut delivered = 0usize;
	while delivered < deliverable {
		let used = wr.wrapping_sub(rx_rd.load(Ordering::Acquire));
		let free_now = h.rx_cap.saturating_sub(used) as usize;
		if free_now == 0 {
			break;
		}
		rx[(wr % h.rx_cap) as usize] = recv[delivered];
		wr = wr.wrapping_add(1);
		delivered += 1;
	}
	if delivered > 0 {
		rx_wr.store(wr, Ordering::Release);
		unsafe { &*h.rx_notify }.store(1, Ordering::Release);
		// 事件驱动唤醒 Python RX daemon — 写 1 字节到通知管道.
		// fd 由 Python 侧 os.pipe() 创建并设为非阻塞, 管道满时
		// write 返回 EAGAIN (通知丢失可接受, daemon 靠 drain_rx 追上).
		if h.rx_notify_fd >= 0 {
			let _ =
				unsafe { libc::write(h.rx_notify_fd, &1u8 as *const u8 as *const libc::c_void, 1) };
		}
		recv.drain(0..delivered);
	}
	delivered
}

// ============================================================
//  Thread entry point
// ============================================================

fn termio_thread(h: TermIoHandle, saved: SavedTerm) {
	let stop = unsafe { &*h.stop_flag };
	let drain_rd = h.rx_drain_fd;
	let mut recv: Vec<u8> = Vec::with_capacity(4096);
	let mut chunk = [0u8; 4096];
	let mut stdin_eof = false;

	while stop.load(Ordering::Acquire) == 0 {
		let mut pfds = [
			libc::pollfd {
				fd: drain_rd,
				events: libc::POLLIN,
				revents: 0,
			},
			libc::pollfd {
				fd: h.stdin_fd,
				events: libc::POLLIN,
				revents: 0,
			},
		];
		// 事件驱动阻塞 (对照 qemu_cond_wait): drain 管道(缓冲区被消耗/停止) +
		// stdin(等待被写入). 不再以 free>0 门控 stdin — 输入由 recv 吸收,
		// Ctrl+Q 因而在 ring 满时仍可被即时检测.
		let nfds: libc::nfds_t = if !stdin_eof && h.rx_cap > 0 { 2 } else { 1 };
		let ret = unsafe { libc::poll(pfds.as_mut_ptr(), nfds, -1) };
		if ret < 0 {
			if unsafe { *libc::__errno_location() } == libc::EINTR {
				continue;
			}
			break;
		}
		// 消费阻塞条件信号 (drain 通知或 stop 通知); stop_flag 于下轮循环顶检查.
		drain_fd(drain_rd);
		if nfds == 2 && (pfds[1].revents & libc::POLLIN) != 0 {
			stdin_eof = !read_stdin_into(&h, &mut recv);
		} else if nfds == 2 && (pfds[1].revents & (libc::POLLHUP | libc::POLLERR)) != 0 {
			stdin_eof = true;
		}
		let (deliverable, stop_req) = scan_rx(&mut recv);
		if stop_req {
			notify_emu_stop();
		}
		let _ = deliver_rx(&h, &mut recv, deliverable);
		drain_tx(&h, &mut chunk);
	}
	// 退出前 best-effort 投递剩余可转发字节 (ring 有空间时; 满 ring / 不完整
	// ESC 尾场景下数据随线程退出丢弃, 与 QEMU chardev 拆除行为一致).
	let (deliverable, _) = scan_rx(&mut recv);
	let _ = deliver_rx(&h, &mut recv, deliverable);
	drain_tx(&h, &mut chunk);
	unsafe {
		libc::tcsetattr(h.stdin_fd, libc::TCSANOW, &saved.oldtty);
		libc::fcntl(h.stdin_fd, libc::F_SETFL, saved.old_fl0);
		libc::fcntl(h.stdout_fd, libc::F_SETFL, saved.old_fl1);
	}
}

/// Simple I/O thread — no terminal setup (Python cbreak manages it).
/// Supports pause via ``pause_flag`` for debugger break/resume.
///
/// Single-fd event-driven poll: blocks on stdin until data arrives.
/// TX output to stdout is handled inline by the speedup execution engine's
/// UART TXDATA handler (libc::write), so this thread only handles
/// stdin -> RX ring buffer forwarding.
fn termio_thread_simple(h: TermIoHandle, old_stdin_fl: libc::c_int) {
	let stop = unsafe { &*h.stop_flag };
	let pause: *const AtomicU8 = h.pause_flag;
	let drain_rd = h.rx_drain_fd;
	let mut recv: Vec<u8> = Vec::with_capacity(4096);
	let mut stdin_eof = false;

	while stop.load(Ordering::Acquire) == 0 {
		if !pause.is_null() && unsafe { &*pause }.load(Ordering::Acquire) != 0 {
			std::thread::sleep(std::time::Duration::from_millis(5));
			continue;
		}
		let mut pfds = [
			libc::pollfd {
				fd: drain_rd,
				events: libc::POLLIN,
				revents: 0,
			},
			libc::pollfd {
				fd: h.stdin_fd,
				events: libc::POLLIN,
				revents: 0,
			},
		];
		let nfds: libc::nfds_t = if !stdin_eof && h.rx_cap > 0 { 2 } else { 1 };
		let ret = unsafe { libc::poll(pfds.as_mut_ptr(), nfds, -1) };
		if ret < 0 {
			if unsafe { *libc::__errno_location() } == libc::EINTR {
				continue;
			}
			break;
		}
		drain_fd(drain_rd);
		if nfds == 2 && (pfds[1].revents & libc::POLLIN) != 0 {
			stdin_eof = !read_stdin_into(&h, &mut recv);
		} else if nfds == 2 && (pfds[1].revents & (libc::POLLHUP | libc::POLLERR)) != 0 {
			stdin_eof = true;
		}
		let (deliverable, stop_req) = scan_rx(&mut recv);
		if stop_req {
			notify_emu_stop();
		}
		let _ = deliver_rx(&h, &mut recv, deliverable);
		// 本线程只负责 stdin -> RX ring (attach 模式).
		// stdout 由 CPU 引擎内联 try_write_fd 即时输出 (no_stdout=0), 这里
		// 不得再调 drain_tx 写 stdout — 否则键盘输入唤醒 poll 时会把整个 TX
		// ring 积压重放一遍 (重复输出), 且 write_all 在 stdout 阻塞时会拖住
		// 本线程, 键盘输入不再送达客机 (无输入响应).
	}
	let (deliverable, _) = scan_rx(&mut recv);
	let _ = deliver_rx(&h, &mut recv, deliverable);
	// terminal_io_attach 曾置 stdin O_NONBLOCK; 线程退出时恢复原阻塞标志,
	// 否则 O_NONBLOCK 泄漏到后续 REPL / prompt_toolkit 的阻塞读, 造成键盘输入丢失.
	if old_stdin_fl >= 0 {
		unsafe { libc::fcntl(h.stdin_fd, libc::F_SETFL, old_stdin_fl) };
	}
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
	/*
	   ---- 显式作用域: 锁在此块结束时立即释放 ----
	   MutexGuard 在 '}' 处 drop, 避免锁被持有到后续 FFI 操作 (unsafe 解引用
	   handle 指针) 期间。若后续代码 panic, 锁仍可被其他路径获取；若省略外层
	   大括号, guard 存活到整个 if 分支结束才析构 —— 后续代码均持锁运行。
	*/
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
		rx_notify: h.rx_notify,
		rx_notify_fd: h.rx_notify_fd,
		rx_drain_fd: h.rx_drain_fd,
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
		// drain 管道读端非阻塞: drain_fd 循环读到 EAGAIN 即停.
		if hc.rx_drain_fd >= 0 {
			let df = libc::fcntl(hc.rx_drain_fd, libc::F_GETFL);
			if df >= 0 {
				libc::fcntl(hc.rx_drain_fd, libc::F_SETFL, df | libc::O_NONBLOCK);
			}
		}
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
							 // 关闭 ISIG: Ctrl+C (0x03) 作为普通字节透传给客机.
							 // Ctrl+Q (0x11) 在 termio 线程中拦截并直连置位停止标志暂停.
	raw.c_lflag &= !(libc::ECHO | libc::ECHONL | libc::ICANON | libc::IEXTEN | libc::ISIG);
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
///
#[no_mangle]
pub extern "C" fn terminal_io_attach(handle: *const TermIoHandle) -> i32 {
	if handle.is_null() {
		return -1;
	}
	{
		let guard = THREAD_HANDLE.lock().unwrap();
		if guard.is_some() {
			return -1;
		}
	}
	let h = unsafe { &*handle };
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
		rx_notify: h.rx_notify,
		rx_notify_fd: h.rx_notify_fd,
		rx_drain_fd: h.rx_drain_fd,
	};
	// QEMU fd_chr_read model: stdin non-blocking so read() returns all
	// currently buffered bytes (escape sequences arrive atomically).
	let old_fl = unsafe { libc::fcntl(hc.stdin_fd, libc::F_GETFL) };
	if old_fl >= 0 {
		unsafe { libc::fcntl(hc.stdin_fd, libc::F_SETFL, old_fl | libc::O_NONBLOCK) };
	}
	// drain 管道读端非阻塞: drain_fd 循环读到 EAGAIN 即停.
	if hc.rx_drain_fd >= 0 {
		let df = unsafe { libc::fcntl(hc.rx_drain_fd, libc::F_GETFL) };
		if df >= 0 {
			unsafe { libc::fcntl(hc.rx_drain_fd, libc::F_SETFL, df | libc::O_NONBLOCK) };
		}
	}
	STOP_PTR.store(hc.stop_flag, Ordering::Release);
	let jh = std::thread::spawn(move || {
		termio_thread_simple(hc, old_fl);
	});
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

/// 注入 CPU 引擎停止标志指针, 供 termio 线程 Ctrl+Q 直连置位.
///
/// Python 侧在启动 termio 前调用, 传入 ``ctypes.addressof(_native_stop_flag)``.
/// 指针由 Python (ctypes 对象) 持有, 进程存活期内稳定有效.
#[no_mangle]
pub extern "C" fn terminal_io_set_emu_stop_flag(ptr: *mut AtomicU8) {
	EMU_STOP_PTR.store(ptr, Ordering::Release);
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
		assert_eq!(size_of::<TermIoHandle>(), 104);
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
		assert_eq!(offset_of!(TermIoHandle, rx_notify), 88);
		assert_eq!(offset_of!(TermIoHandle, rx_notify_fd), 96);
		assert_eq!(offset_of!(TermIoHandle, rx_drain_fd), 100);
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

	/// 回归: Ctrl+Q 单次触发必须直连置位 CPU 停止标志, 而非仅依赖 SIGINT.
	///
	/// 修复前 ``intercept_ctrl_q`` 只发 SIGINT, 依赖 Python 主线程信号处理器
	/// 间接置位停止标志; 主线程阻塞于 run_parallel 时处理器不执行 -> Ctrl+Q
	/// 失灵. 修复后 ``notify_emu_stop`` 直接写共享标志, 与 rx_notify 同模式.
	#[test]
	fn test_notify_emu_stop_sets_flag() {
		let flag = AtomicU8::new(0);
		let flag_ptr: *mut AtomicU8 = &flag as *const AtomicU8 as *mut AtomicU8;
		EMU_STOP_PTR.store(flag_ptr, Ordering::Release);

		notify_emu_stop();
		assert_eq!(flag.load(Ordering::Acquire), 1, "Ctrl+Q 应直连置位停止标志");

		EMU_STOP_PTR.store(std::ptr::null_mut(), Ordering::Release);
	}

	/// 回归: 未注入停止标志指针时 ``notify_emu_stop`` 不得崩溃 (空指针保护).
	#[test]
	fn test_notify_emu_stop_null_ptr_noop() {
		EMU_STOP_PTR.store(std::ptr::null_mut(), Ordering::Release);
		notify_emu_stop(); // 不应 panic
	}

	/// esc_seq_len: 有界 6 字节窗口内的 ESC 序列完整性判定.
	#[test]
	fn test_esc_seq_len_variants() {
		assert_eq!(esc_seq_len(&[0x1b]), None); // 孤立 ESC — 等下一字节
		assert_eq!(esc_seq_len(&[0x1b, b'x']), Some(2)); // Alt+key
		assert_eq!(esc_seq_len(&[0x1b, b'O', b'A']), Some(3)); // SS3
		assert_eq!(esc_seq_len(&[0x1b, b'[', b'A']), Some(3)); // CSI 上箭头
		assert_eq!(esc_seq_len(&[0x1b, b'[']), None); // CSI 无 final
		assert_eq!(esc_seq_len(&[0x1b, b'[', b'1']), None); // 参数未完
		assert_eq!(esc_seq_len(&[0x1b, b'[', b'1', b';', b'2', b'H']), Some(6));
		// 窗口耗尽 (7 字节) 仍无 final — 透传 ESC 单字节, 不阻塞.
		assert_eq!(
			esc_seq_len(&[0x1b, b'[', b'1', b'2', b'3', b'4', b'5']),
			Some(1)
		);
	}

	/// scan_rx: 连续 Ctrl+Q 运行视作同一次停机, 仅消费 0x11, 前后字节不丢失.
	#[test]
	fn test_scan_rx_ctrl_q_run_consumes_once() {
		let mut recv = vec![b'a', 0x11, 0x11, b'b'];
		let (deliverable, stop) = scan_rx(&mut recv);
		assert!(stop, "连续 Ctrl+Q 应触发停机");
		assert_eq!(deliverable, 1, "0x11 运行前的可投递前缀长度");
		assert_eq!(&recv, b"ab", "0x11 运行被消费, 前后字节保留");
	}

	/// scan_rx: 无停机字节时全部可投递, 且不修改 recv.
	#[test]
	fn test_scan_rx_no_ctrl_q() {
		let mut recv = b"hello".to_vec();
		let (deliverable, stop) = scan_rx(&mut recv);
		assert!(!stop);
		assert_eq!(deliverable, 5);
		assert_eq!(&recv, b"hello");
	}

	/// scan_rx: 完整 ESC 序列作为一个整体可投递.
	#[test]
	fn test_scan_rx_complete_esc_sequence() {
		let mut recv = vec![b'a', 0x1b, b'[', b'A', b'b'];
		let (deliverable, stop) = scan_rx(&mut recv);
		assert!(!stop);
		assert_eq!(deliverable, 5);
		assert_eq!(&recv, b"a\x1b[Ab");
	}

	/// scan_rx: 不完整 ESC 尾保留 (等效 tmpKeep), 作为后续解析开头.
	#[test]
	fn test_scan_rx_incomplete_esc_holdback() {
		let mut recv = vec![b'a', 0x1b];
		let (deliverable, stop) = scan_rx(&mut recv);
		assert!(!stop);
		assert_eq!(deliverable, 1, "仅 'a' 可投递, 孤立 ESC 保留");
		assert_eq!(&recv, b"a\x1b");
	}

	/// deliver_rx: 受 free 容量约束, 装不下的保留在 recv (永不丢弃).
	#[test]
	fn test_deliver_rx_respects_free_space() {
		let rx_cap = 4u32;
		let mut rx_buf = vec![0u8; rx_cap as usize];
		let rx_wr = AtomicU32::new(0);
		let rx_rd = AtomicU32::new(0);
		let rx_notify = AtomicU8::new(0);
		let h = TermIoHandle {
			stdin_fd: -1,
			stdout_fd: -1,
			rx_buf: rx_buf.as_mut_ptr(),
			rx_cap,
			rx_wr: &rx_wr as *const AtomicU32 as *mut AtomicU32,
			rx_rd: &rx_rd as *const AtomicU32 as *mut AtomicU32,
			tx_buf: std::ptr::null_mut(),
			tx_cap: 0,
			tx_wr: std::ptr::null_mut(),
			tx_drain: std::ptr::null_mut(),
			stop_flag: std::ptr::null_mut(),
			pause_flag: std::ptr::null_mut(),
			rx_notify: &rx_notify as *const AtomicU8 as *mut AtomicU8,
			rx_notify_fd: -1,
			rx_drain_fd: -1,
		};
		let mut recv = b"abcde".to_vec();
		let delivered = deliver_rx(&h, &mut recv, 5);
		assert_eq!(delivered, 4, "cap=4 只装下前 4 个");
		assert_eq!(&rx_buf, b"abcd");
		assert_eq!(&recv, b"e", "剩余字节保留在 recv");
		assert_eq!(rx_wr.load(Ordering::Acquire), 4);
		assert_eq!(rx_notify.load(Ordering::Acquire), 1, "写入后置 rx_notify");
	}

	/// attach 线程 (termio_thread_simple): 键盘输入唤醒 poll 时不得把 TX ring
	/// 积压重放到 stdout. CPU 引擎已内联 try_write_fd 即时输出 stdout
	/// (no_stdout=0), 线程再 drain_tx 会重复输出 (启动日志重放), 且 write_all
	/// 在 stdout 阻塞时拖住线程, 键盘输入不再送达客机 (无输入响应). 回归:
	/// emu-linux-sh 停于 zsh init / 反复输出同一段串口日志.
	#[test]
	fn test_attach_thread_does_not_reemit_tx_ring_to_stdout() {
		let mut fds = [0i32; 6];
		assert_eq!(unsafe { libc::pipe(fds.as_mut_ptr()) }, 0);
		let (stdout_r, stdout_w) = (fds[0], fds[1]);
		assert_eq!(unsafe { libc::pipe(fds.as_mut_ptr()) }, 0);
		let (stdin_r, stdin_w) = (fds[0], fds[1]);
		assert_eq!(unsafe { libc::pipe(fds.as_mut_ptr()) }, 0);
		let (drain_r, drain_w) = (fds[0], fds[1]);
		// terminal_io_attach 会把这些 fd 设为 O_NONBLOCK (stdin/drain 读端 +
		// stdout 读端供本测试探测).
		for fd in [stdin_r, drain_r, stdout_r] {
			let fl = unsafe { libc::fcntl(fd, libc::F_GETFL) };
			unsafe { libc::fcntl(fd, libc::F_SETFL, fl | libc::O_NONBLOCK) };
		}

		let ecap = 8u32;
		let mut tx_buf = vec![0u8; (ecap as usize) * 2];
		let mut rx_buf = vec![0u8; 8usize];
		let tx_wr = AtomicU32::new(0);
		let tx_drain = AtomicU32::new(0);
		let rx_wr = AtomicU32::new(0);
		let rx_rd = AtomicU32::new(0);
		let rx_notify = AtomicU8::new(0);
		let stop = AtomicU8::new(0);
		let pause = AtomicU8::new(0);

		// CPU 引擎已写 2 个 TX 条目到 ring (stdout 已由 try_write_fd 即时输出).
		tx_buf[0] = 0;
		tx_buf[1] = b'X';
		tx_buf[2] = 1;
		tx_buf[3] = b'Y';
		tx_wr.store(2, Ordering::Release);

		let h = TermIoHandle {
			stdin_fd: stdin_r,
			stdout_fd: stdout_w,
			rx_buf: rx_buf.as_mut_ptr(),
			rx_cap: 8,
			rx_wr: &rx_wr as *const AtomicU32 as *mut AtomicU32,
			rx_rd: &rx_rd as *const AtomicU32 as *mut AtomicU32,
			tx_buf: tx_buf.as_mut_ptr(),
			tx_cap: ecap,
			tx_wr: &tx_wr as *const AtomicU32 as *mut AtomicU32,
			tx_drain: &tx_drain as *const AtomicU32 as *mut AtomicU32,
			stop_flag: &stop as *const AtomicU8 as *mut AtomicU8,
			pause_flag: &pause as *const AtomicU8 as *mut AtomicU8,
			rx_notify: &rx_notify as *const AtomicU8 as *mut AtomicU8,
			rx_notify_fd: -1,
			rx_drain_fd: drain_r,
		};

		let handle = std::thread::spawn(move || termio_thread_simple(h, -1));

		// 用户输入一个字符 -> poll 唤醒 -> scan_rx/deliver_rx -> (修复后) 不写 stdout.
		assert_eq!(
			unsafe { libc::write(stdin_w, b"z".as_ptr() as *const libc::c_void, 1) },
			1
		);
		// 等线程完成一轮处理: rx_wr 推进即已执行完整迭代 (包括旧代码会重放的
		// drain_tx 位置), 此后的 stdout 断言与线程执行顺序无关.
		let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
		while rx_wr.load(Ordering::Acquire) == 0 {
			if std::time::Instant::now() > deadline {
				panic!("attach 线程未在期限内处理 stdin 输入");
			}
			std::thread::sleep(std::time::Duration::from_millis(1));
		}
		// RX 仍正常: 'z' 已送达 RX ring.
		assert_eq!(rx_wr.load(Ordering::Acquire), 1);
		assert_eq!(rx_buf[0], b'z');

		// stdout 不得收到任何 TX ring 字节 — 引擎已输出, 线程不得重放.
		let mut probe = [0u8; 16];
		let n = unsafe {
			libc::read(
				stdout_r,
				probe.as_mut_ptr() as *mut libc::c_void,
				probe.len(),
			)
		};
		assert!(
			n <= 0,
			"attach 线程把 TX ring 积压重放到了 stdout: {:?}",
			&probe[..n as usize]
		);

		// 停线程: 置 stop 后写入唤醒阻塞中的 poll.
		stop.store(1, Ordering::Release);
		unsafe { libc::write(stdin_w, b"q".as_ptr() as *const libc::c_void, 1) };
		handle.join().unwrap();

		for fd in [stdout_r, stdout_w, stdin_r, stdin_w, drain_r, drain_w] {
			unsafe { libc::close(fd) };
		}
	}
}
