//! 终端输入与输出的后台线程.
use std::os::unix::io::RawFd;
use std::sync::atomic::{AtomicBool, AtomicI32, AtomicPtr, AtomicU32, AtomicU8, Ordering};
use std::sync::Mutex;

// ============================================================
//  FFI 上下文: 由 Python ctypes 传入的 C 布局结构体
// ============================================================

/// 终端输入输出线程的共享状态, 由 Python 分配并在启动时传入一次.
/// 所有指针在线程生命周期内保持有效.
#[repr(C)]
pub struct TermIoHandle {
	/// 标准输入的文件描述符, 通常为 0.
	pub stdin_fd: RawFd,
	/// 标准输出的文件描述符, 通常为 1.
	pub stdout_fd: RawFd,
	/// 共享的接收环形缓冲区: 本线程写入, 受调试程序读取 RXDATA 时由 Python 读出.
	pub rx_buf: *mut u8,
	/// ``rx_buf`` 的容量, 以字节计, 取值为 2 的幂.
	pub rx_cap: u32,
	/// 接收写索引, 由本线程独占写入, 单调递增的 u32.
	pub rx_wr: *mut AtomicU32,
	pub rx_rd: *mut AtomicU32,
	/// 共享的发送环形缓冲区, 由各 hart 的 CPU 线程写入受调试程序的 TXDATA.
	/// 布局: 第 e 个条目占据字节 [2e] 与 [2e+1], 依次存放 hart_id 与字节值.
	pub tx_buf: *mut u8,
	/// ``tx_buf`` 的条目容量, 为字节容量除以 2, 取值为 2 的幂.
	pub tx_cap: u32,
	/// 发送写索引, 以条目计, 由 CPU 引擎独占写入, 单调递增的 u32.
	pub tx_wr: *mut AtomicU32,
	/// 由本线程独占写入. Python 只读该索引, 日志消费者另持读索引.
	pub tx_drain: *mut AtomicU32,
	/// 停止标志, Python 置 1 请求本线程退出.
	pub stop_flag: *mut AtomicU8,
	/// 暂停标志, 调试器中断时 Python 置 1, 恢复运行时置 0.
	pub pause_flag: *mut AtomicU8,
	/// 接收通知标志: 本线程写入接收环形缓冲区后设置 1,
	/// CPU 引擎在每个指令边界检查该标志, 以对应投递外部中断.
	pub rx_notify: *mut AtomicU8,
	/// 接收通知管道的写端文件描述符: 本线程写入环形缓冲区后向该端写 1 字节,
	/// 唤醒 Python 侧的接收守护线程, 该线程由 select 事件驱动, 不做轮询.
	pub rx_notify_fd: RawFd,
	/// 接收管道的读端文件描述符. Python 消费环形缓冲区并推进 rx_rd 后
	/// 向管道写端写 1 字节, 唤醒本线程投递接收缓冲区中剩余的可转发字节.
	/// 环形缓冲区满时本线程等待其被消费, 反压由此解除. 写端由 Python 持有.
	pub rx_drain_fd: RawFd,
}

// 安全前提: 所有指针由 Python 的 ctypes 数组持有, 在线程生命周期内保持有效;
// 各位置遵循上述单个写者的假设.
unsafe impl Send for TermIoHandle {}

/// 启动时保存的终端状态, 线程退出时恢复.
struct SavedTerm {
	oldtty: libc::termios,
	old_fl0: libc::c_int,
	old_fl1: libc::c_int,
}

// ============================================================
//  全局线程描述符与 SIGCONT 处理所需的辅助状态
// ============================================================

static THREAD_HANDLE: Mutex<Option<std::thread::JoinHandle<()>>> = Mutex::new(None);

/// 停止标志的指针副本, ``terminal_io_stop`` 据此自行置位, 不依赖 Python 先设置.
static STOP_PTR: AtomicPtr<AtomicU8> = AtomicPtr::new(std::ptr::null_mut());

/// CPU 引擎停止标志的指针, Ctrl+Q 时由本线程直接置位, 不经过 SIGINT 到 Python
/// 处理函数再到 notify_processor 的往返: 主线程阻塞于 run_parallel 时信号处理
/// 函数不执行, 直接置位保证看门狗在 5 ms 内看到停止请求.
static EMU_STOP_PTR: AtomicPtr<AtomicU8> = AtomicPtr::new(std::ptr::null_mut());

/// SIGCONT 处理函数所需的状态: Ctrl+Z 挂起期间 shell 会把终端复位为 cooked
/// 模式, 恢复运行后需要重新设置为 raw 模式.
static TERM_ACTIVE: AtomicBool = AtomicBool::new(false);
static RAW_TTY_PTR: AtomicPtr<libc::termios> = AtomicPtr::new(std::ptr::null_mut());
static SIG_STDIN_FD: AtomicI32 = AtomicI32::new(0);

/// 原有的 SIGCONT 处置, 停止时恢复. ``libc::sigaction`` 含裸指针, 故手动实现 Send.
struct SigActionCell(libc::sigaction);
unsafe impl Send for SigActionCell {}
static OLD_SIGCONT: Mutex<Option<SigActionCell>> = Mutex::new(None);

extern "C" fn sigcont_handler(_sig: libc::c_int) {
	// 只调用异步信号安全的 tcsetattr.
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
	// termios 首次分配后不再释放, 不随启动与停止累积.
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
//  输入输出辅助函数
// ============================================================

/// EINTR 时重试, EAGAIN 时等待 POLLOUT 可写后继续写入.
/// 收到停止请求时放弃剩余输出, 避免终端停滞使线程无法结束.
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
		return; // 其他错误: 尽力而为, 不让输入输出线程崩溃
	}
}

/// 从发送环形缓冲区把待排空条目的字节收集到 chunk, 跳过 hart_id 字节.
/// 返回新的排空索引与收集到的字节数. 该函数为纯函数, 便于测试索引跨越容量边界的情形.
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

/// 通过输出到 stdout 清空发送环形缓冲区. 本线程是 ``tx_drain`` 的唯一写者.
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
		// 此处的 Acquire 与 CPU 引擎设置 tx_wr 时的 Release 配对, 读到新索引即可读到对应槽位的数据.
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
//  stdin 输入解析器: Ctrl+Q 停机检测与转义序列重组
//
//  所有输入先保留在接收缓冲区; 解析器每次在至多 6 字节的窗口内逐字节判定具体的输入
//  当无法判定时, 例如半个 ESC 序列, 保留在接收缓冲区中作为下一轮解析的开头
// ============================================================

/// 直接设置 CPU 引擎的停止标志, 而不经过 Python.
/// 主线程阻塞于 run_parallel 时信号处理函数不执行, 该路径使引擎的看门狗与
/// 逐指令检查仍能即时看到停止请求. 独立成函数以便单元测试.
#[inline]
fn notify_emu_stop() {
	let stop = EMU_STOP_PTR.load(Ordering::Acquire);
	if !stop.is_null() {
		unsafe { (*stop).store(1, Ordering::Release) };
	}
}

/// 判定以 ESC 开头的序列是否完整, 窗口至多 6 字节.
/// 返回 ``Some(n)`` 表示序列完整, 共 n 字节; 返回 ``None`` 表示数据不足以判定.
fn esc_seq_len(buf: &[u8]) -> Option<usize> {
	debug_assert!(!buf.is_empty() && buf[0] == 0x1b);
	if buf.len() == 1 {
		return None; // 等下一字节区分 Alt 键与 ESC 序列开头
	}
	let b1 = buf[1];
	if b1 != b'[' && b1 != b'O' {
		return Some(2); // Alt 组合键, 形如 ESC X
	}
	if b1 == b'O' {
		return Some(3); // SS3: ESC O X, 共 3 字节
	}
	// CSI: ESC [ ... final, final 位于 0x40..=0x7e, 在 6 字节窗口内查找.
	for (idx, &b) in buf[2..].iter().take(4).enumerate() {
		if (0x40..=0x7e).contains(&b) {
			return Some(2 + idx + 1);
		}
	}
	// 已积累超过 6 字节时视为无法重组的长序列: 只把 ESC 作为完整序列,
	// 其余字节按普通字节继续转发, 由受调试程序的终端自行重组.
	if buf.len() > 6 {
		return Some(1);
	}
	None // 不足 6 字节且无 final, 保留等待补全
}

/// 扫描接收缓冲区前端: 消费连续的 0x11 运行即停机请求, 识别完整的 ESC 序列,
/// 在首个不完整的 ESC 序列处停下. 返回可投递字节数与是否请求停机.
/// - 可投递字节数为前端可安全写入接收环形缓冲区的字节数, 不含已消费的 0x11 运行
///   与不完整的 ESC 序列; 运行被移除后其后的字节前移, 留在接收缓冲区等待下轮解析.
/// - 停机时运行之前的可投递前缀长度即为返回值, 运行之后的字节仍然保留.
fn scan_rx(recv: &mut Vec<u8>) -> (usize, bool) {
	let mut i = 0usize;
	while i < recv.len() {
		let b = recv[i];
		if b == 0x11 {
			// 消费整个连续的 0x11 运行, 多个连续 Ctrl+Q 视作同一次停机请求.
			let mut j = i;
			while j < recv.len() && recv[j] == 0x11 {
				j += 1;
			}
			recv.drain(i..j);
			return (i, true); // 停机由调用方调用 notify_emu_stop 执行
		}
		if b == 0x1b {
			match esc_seq_len(&recv[i..]) {
				Some(n) if n <= recv.len() - i => {
					i += n;
				}
				_ => {
					return (i, false); // ESC 序列不完整, 保留等待补全
				}
			}
		} else {
			i += 1;
		}
	}
	(i, false)
}

/// 读出一个非阻塞 fd 中的全部可读字节, 消除 poll 的虚假重复唤醒. 读取错误被忽略.
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
//  标准输入到接收环形缓冲区
// ============================================================

/// 把标准输入当前可读的所有字节追加到接收缓冲区, 该 fd 为非阻塞, 读到 EAGAIN 为止.
/// 接收环形缓冲区已满时仍照常读入, 由 scan_rx 与 deliver_rx
/// 分阶段处理, Ctrl+Q 因而在任意占用下都能被即时检出. 读到文件结束符时返回 false.
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

/// 把 scan_rx 判定的可投递前缀从接收缓冲区写入接收环形缓冲区, 受空闲容量限制,
/// 装不下的部分留在接收缓冲区等待下一轮, 不丢弃. 返回本次实际写入的字节数.
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
		// 向通知管道写 1 字节, 事件驱动地唤醒 Python 侧的接收守护线程.
		// 该 fd 由 Python 用 os.pipe() 创建并设为非阻塞; 管道满时写操作返回
		// EAGAIN, 丢失通知可以接受, 守护线程仍会经 drain_rx 追上.
		if h.rx_notify_fd >= 0 {
			let _ =
				unsafe { libc::write(h.rx_notify_fd, &1u8 as *const u8 as *const libc::c_void, 1) };
		}
		recv.drain(0..delivered);
	}
	delivered
}

// ============================================================
//  线程入口
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
		let nfds: libc::nfds_t = if !stdin_eof && h.rx_cap > 0 { 2 } else { 1 };
		let ret = unsafe { libc::poll(pfds.as_mut_ptr(), nfds, -1) };
		if ret < 0 {
			if unsafe { *libc::__errno_location() } == libc::EINTR {
				continue;
			}
			break;
		}
		// 读出阻塞条件信号, 即清空通知并停止; stop_flag 在下一轮循环开头检查.
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
	// 退出循环前仍尝试把接收缓冲区中的可投递字节写入接收环形缓冲区,
	// 不完整的 ESC 多字节序列保留在接收缓冲区, 不作转发.
	let (deliverable, _) = scan_rx(&mut recv);
	let _ = deliver_rx(&h, &mut recv, deliverable);
	drain_tx(&h, &mut chunk);
	unsafe {
		libc::tcsetattr(h.stdin_fd, libc::TCSANOW, &saved.oldtty);
		libc::fcntl(h.stdin_fd, libc::F_SETFL, saved.old_fl0);
		libc::fcntl(h.stdout_fd, libc::F_SETFL, saved.old_fl1);
	}
}

/// 简化版输入输出线程, 不设置终端属性.
/// 通过 ``pause_flag`` 支持调试器暂停与恢复.
///
/// 只用 poll 监听文件描述符, 在标准输入上阻塞等待数据. 发送方向由加速执行引擎的
/// UART TXDATA 处理函数直接写 stdout, 本线程只负责把标准输入转发进接收环形缓冲区.
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
		// 本线程只负责把标准输入转发到接收环形缓冲区. ``no_stdout`` 为 0 时
		// stdout 已由 CPU 引擎就地经 try_write_fd 输出; 此处不得再调用 drain_tx,
		// 否则键盘输入唤醒后会重复输出发送环形缓冲区中的内容, 且键盘输入不再
		// 投递给受调试程序.
	}
	let (deliverable, _) = scan_rx(&mut recv);
	let _ = deliver_rx(&h, &mut recv, deliverable);
	// terminal_io_attach 把标准输入设为 O_NONBLOCK, 线程退出时恢复原有的标志位,
	// 否则该标志会留给后续 REPL 与 prompt_toolkit 的阻塞读, 导致键盘输入丢失.
	if old_stdin_fl >= 0 {
		unsafe { libc::fcntl(h.stdin_fd, libc::F_SETFL, old_stdin_fl) };
	}
}

// ============================================================
//  FFI 入口
// ============================================================

/// 启动终端输入输出后台线程.
///
/// 返回 0 表示成功, 返回 -1 表示线程已在运行或标准输入不是 TTY.
/// 调用方必须保证 ``handle`` 指向有效的 ``TermIoHandle``, 其缓冲区在线程生命周期内
/// 保持有效.
#[no_mangle]
pub extern "C" fn terminal_io_start(handle: *const TermIoHandle) -> i32 {
	if handle.is_null() {
		return -1;
	}
	/*
	   显式作用域使锁在此块结束时立即释放: MutexGuard 在右大括号处析构,
	   后续 FFI 操作期间不再持锁. 若省略该作用域, guard 会存活到外层分支结束,
	   后续代码都将持锁运行, 且 panic 时其他路径无法取得该锁.
	*/
	{
		let guard = THREAD_HANDLE.lock().unwrap();
		if guard.is_some() {
			return -1;
		}
	}

	let h = unsafe { &*handle };
	// 逐字段复制: handle 位于 Python 栈上, FFI 调用返回后即失效.
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

	// ---- 终端初始化 ----
	let mut oldtty: libc::termios = unsafe { std::mem::zeroed() };
	if unsafe { libc::tcgetattr(hc.stdin_fd, &mut oldtty) } != 0 {
		return -1; // 标准输入不是 TTY, 例如管道或重定向, 由 Python 侧另行处理
	}
	let old_fl0 = unsafe { libc::fcntl(hc.stdin_fd, libc::F_GETFL) };
	let old_fl1 = unsafe { libc::fcntl(hc.stdout_fd, libc::F_GETFL) };
	if old_fl0 < 0 || old_fl1 < 0 {
		return -1;
	}
	unsafe {
		libc::fcntl(hc.stdin_fd, libc::F_SETFL, old_fl0 | libc::O_NONBLOCK);
		libc::fcntl(hc.stdout_fd, libc::F_SETFL, old_fl1 | libc::O_NONBLOCK);
		// 把管道的读端设为非阻塞: drain_fd 循环读到 EAGAIN 即停止.
		if hc.rx_drain_fd >= 0 {
			let df = libc::fcntl(hc.rx_drain_fd, libc::F_GETFL);
			if df >= 0 {
				libc::fcntl(hc.rx_drain_fd, libc::F_SETFL, df | libc::O_NONBLOCK);
			}
		}
	}

	let mut raw = oldtty;
	raw.c_iflag &= !(libc::IGNBRK
        | libc::BRKINT
        | libc::PARMRK
        | libc::ISTRIP
        | libc::INLCR
        | libc::IGNCR
        // 保留 ICRNL: 宿主终端把 \r 转换为 \n, 受调试程序收到 \n 即可结束一行
        | libc::IXON);
	raw.c_oflag |= libc::OPOST; // 保留输出后处理, 把 '\n' 转换为 CRLF
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

	// Ctrl+Z 挂起恢复后重新设置 raw 模式
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

/// 把输入输出线程连接到已配置好的文件描述符上, 不修改 termios 属性.
/// 终端模式由 Python 以 cbreak 管理, 本函数只做读写.
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
	let old_fl = unsafe { libc::fcntl(hc.stdin_fd, libc::F_GETFL) };
	if old_fl >= 0 {
		unsafe { libc::fcntl(hc.stdin_fd, libc::F_SETFL, old_fl | libc::O_NONBLOCK) };
	}
	// 把排空管道的读端设为非阻塞: drain_fd 循环读到 EAGAIN 即停止.
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

/// 停止输入输出线程并等待其退出.
///
/// 自行置位停止标志, 不依赖 Python 先设置; 随后等待线程结束, 线程退出前
/// 完成发送方向的最后排空与终端恢复; 最后恢复原有的 SIGCONT 处置.
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

/// 注入 CPU 引擎停止标志的指针, 供本线程在 Ctrl+Q 时直接置位.
///
/// Python 在启动本线程前调用, 传入 ``ctypes.addressof(_native_stop_flag)``.
/// 指针由 Python 的 ctypes 对象持有, 在进程存活期内保持有效.
#[no_mangle]
pub extern "C" fn terminal_io_set_emu_stop_flag(ptr: *mut AtomicU8) {
	EMU_STOP_PTR.store(ptr, Ordering::Release);
}

/// 输入输出线程正在运行时返回 1, 否则返回 0.
#[no_mangle]
pub extern "C" fn terminal_io_is_running() -> i32 {
	if THREAD_HANDLE.lock().unwrap().is_some() {
		1
	} else {
		0
	}
}

// ============================================================
//  测试
// ============================================================

#[cfg(test)]
mod tests {
	use super::*;
	use core::mem::{offset_of, size_of};

	/// 锁定 FFI 布局: 与 Python ctypes 的 TermIoHandle._fields_ 逐字节一致,
	/// 定义位于 pyremu/_native/__init__.py. 修改任一侧时必须同步另一侧.
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
		// 4 个条目, 每个条目含 hid 与 byte
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
		// 索引跨越条目容量边界: 条目 6、7、8、9 对应槽位 6、7、0、1
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
		let mut chunk = [0u8; 4]; // 小于待排空的条目数
		let (d, n) = gather_tx_chunk(&buf, ecap, 0, 6, &mut chunk);
		assert_eq!(d, 4);
		assert_eq!(n, 4);
		assert_eq!(&chunk[..n], b"0123");
		// 继续收集剩余的条目
		let (d2, n2) = gather_tx_chunk(&buf, ecap, d, 6, &mut chunk);
		assert_eq!(d2, 6);
		assert_eq!(&chunk[..n2], b"45");
	}

	/// 单次 Ctrl+Q 必须直接置位 CPU 引擎的停止标志, 不能只依赖 SIGINT.
	/// ``notify_emu_stop`` 直接写共享标志, 与 rx_notify 的处理方式相同;
	/// 主线程阻塞于 run_parallel 时 Python 侧的信号处理函数不执行.
	#[test]
	fn test_notify_emu_stop_sets_flag() {
		let flag = AtomicU8::new(0);
		let flag_ptr: *mut AtomicU8 = &flag as *const AtomicU8 as *mut AtomicU8;
		EMU_STOP_PTR.store(flag_ptr, Ordering::Release);

		notify_emu_stop();
		assert_eq!(flag.load(Ordering::Acquire), 1, "Ctrl+Q 应直连置位停止标志");

		EMU_STOP_PTR.store(std::ptr::null_mut(), Ordering::Release);
	}

	/// 未注入停止标志指针时 ``notify_emu_stop`` 不执行任何操作, 不崩溃.
	#[test]
	fn test_notify_emu_stop_null_ptr_noop() {
		EMU_STOP_PTR.store(std::ptr::null_mut(), Ordering::Release);
		notify_emu_stop(); // 不应发生 panic
	}

	/// 覆盖 ``esc_seq_len`` 在 6 字节窗口内对 ESC 序列完整性的判定.
	#[test]
	fn test_esc_seq_len_variants() {
		assert_eq!(esc_seq_len(&[0x1b]), None); // 孤立 ESC — 等下一字节
		assert_eq!(esc_seq_len(&[0x1b, b'x']), Some(2)); // Alt+key
		assert_eq!(esc_seq_len(&[0x1b, b'O', b'A']), Some(3)); // SS3
		assert_eq!(esc_seq_len(&[0x1b, b'[', b'A']), Some(3)); // CSI 上箭头
		assert_eq!(esc_seq_len(&[0x1b, b'[']), None); // CSI 无 final
		assert_eq!(esc_seq_len(&[0x1b, b'[', b'1']), None); // 参数未完
		assert_eq!(esc_seq_len(&[0x1b, b'[', b'1', b';', b'2', b'H']), Some(6));
		// 已积累 7 字节仍无 final 时只把 ESC 作为完整序列, 其余字节照常转发.
		assert_eq!(
			esc_seq_len(&[0x1b, b'[', b'1', b'2', b'3', b'4', b'5']),
			Some(1)
		);
	}

	/// 覆盖 ``scan_rx``: 连续的 Ctrl+Q 视作同一次停机, 只消费 0x11, 前后字节不丢失.
	#[test]
	fn test_scan_rx_ctrl_q_run_consumes_once() {
		let mut recv = vec![b'a', 0x11, 0x11, b'b'];
		let (deliverable, stop) = scan_rx(&mut recv);
		assert!(stop, "连续 Ctrl+Q 应触发停机");
		assert_eq!(deliverable, 1, "0x11 运行前的可投递前缀长度");
		assert_eq!(&recv, b"ab", "0x11 运行被消费, 前后字节保留");
	}

	/// 无停机字节时全部可投递, 且不修改接收缓冲区.
	#[test]
	fn test_scan_rx_no_ctrl_q() {
		let mut recv = b"hello".to_vec();
		let (deliverable, stop) = scan_rx(&mut recv);
		assert!(!stop);
		assert_eq!(deliverable, 5);
		assert_eq!(&recv, b"hello");
	}

	/// 完整的 ESC 序列作为一个整体可投递.
	#[test]
	fn test_scan_rx_complete_esc_sequence() {
		let mut recv = vec![b'a', 0x1b, b'[', b'A', b'b'];
		let (deliverable, stop) = scan_rx(&mut recv);
		assert!(!stop);
		assert_eq!(deliverable, 5);
		assert_eq!(&recv, b"a\x1b[Ab");
	}

	/// 不完整的 ESC 序列保留在接收缓冲区, 作为下一轮解析的开头.
	#[test]
	fn test_scan_rx_incomplete_esc_holdback() {
		let mut recv = vec![b'a', 0x1b];
		let (deliverable, stop) = scan_rx(&mut recv);
		assert!(!stop);
		assert_eq!(deliverable, 1, "仅 'a' 可投递, 孤立 ESC 保留");
		assert_eq!(&recv, b"a\x1b");
	}

	/// 受空闲容量约束, 装不下的字节保留在接收缓冲区, 不丢弃.
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

	/// ``no_stdout`` 为 0 时 CPU 引擎已就地经 try_write_fd 输出,
	/// 线程再调用 drain_tx 会重复输出启动日志; 且 write_all 在
	/// stdout 阻塞时会同样阻塞线程, 导致键盘输入无法得到响应.
	#[test]
	fn test_attach_thread_does_not_reemit_tx_ring_to_stdout() {
		let mut fds = [0i32; 6];
		assert_eq!(unsafe { libc::pipe(fds.as_mut_ptr()) }, 0);
		let (stdout_r, stdout_w) = (fds[0], fds[1]);
		assert_eq!(unsafe { libc::pipe(fds.as_mut_ptr()) }, 0);
		let (stdin_r, stdin_w) = (fds[0], fds[1]);
		assert_eq!(unsafe { libc::pipe(fds.as_mut_ptr()) }, 0);
		let (drain_r, drain_w) = (fds[0], fds[1]);
		// terminal_io_attach 会把这些 fd 设为 O_NONBLOCK, 其中 stdout 读端供本测试探测.
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

		// CPU 引擎已向发送环形缓冲区写入 2 个条目, stdout 已由 try_write_fd 输出.
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

		// 用户输入一个字符, poll 被唤醒, 经 scan_rx 与 deliver_rx 处理后不写 stdout.
		assert_eq!(
			unsafe { libc::write(stdin_w, b"z".as_ptr() as *const libc::c_void, 1) },
			1
		);
		// 等待线程完成一轮处理: rx_wr 推进即说明整轮迭代已执行,
		// 之后的 stdout 断言与线程的执行顺序无关.
		let deadline = std::time::Instant::now() + std::time::Duration::from_secs(2);
		while rx_wr.load(Ordering::Acquire) == 0 {
			if std::time::Instant::now() > deadline {
				panic!("attach 线程未在期限内处理 stdin 输入");
			}
			std::thread::sleep(std::time::Duration::from_millis(1));
		}
		// 接收路径仍正常: 'z' 已进入接收环形缓冲区.
		assert_eq!(rx_wr.load(Ordering::Acquire), 1);
		assert_eq!(rx_buf[0], b'z');

		// stdout 不应收到发送环形缓冲区中的任何字节.
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

		// 设置 stop 标志后写一个字节, 唤醒阻塞中的 poll.
		stop.store(1, Ordering::Release);
		unsafe { libc::write(stdin_w, b"q".as_ptr() as *const libc::c_void, 1) };
		handle.join().unwrap();

		for fd in [stdout_r, stdout_w, stdin_r, stdin_w, drain_r, drain_w] {
			unsafe { libc::close(fd) };
		}
	}
}
