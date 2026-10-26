use crate::ffi::FfiUartCtx;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

/// 向 fd 非阻塞写入一个字节。先以超时 0 的 poll 检查 POLLOUT, 可写时才 write.
///
/// 此处只做即时终端回显, 该字节已先写入发送环形缓冲区并进入日志. 对端不可写时
/// 丢弃该字节, 保证 CPU 引擎线程不在输出上阻塞.
fn try_write_fd(fd: i32, byte: u8) -> usize {
	let mut pfd = libc::pollfd {
		fd,
		events: libc::POLLOUT,
		revents: 0,
	};
	let rc = unsafe { libc::poll(&mut pfd as *mut _, 1, 0) };
	if rc > 0 && (pfd.revents & libc::POLLOUT) != 0 {
		unsafe { libc::write(fd, &byte as *const u8 as *const libc::c_void, 1) as usize }
	} else {
		0
	}
}

/// 就地读取 UART 寄存器的当前值, 只处理 0x00、0x08、0x10、0x14 四个偏移,
/// 其余偏移返回 ``None``.
fn handle_uart_read(offset: u64, uart: &FfiUartCtx) -> Option<u64> {
	match offset {
		0 => Some(0), // TXDATA: bit31 为 FIFO 满标志, 恒为 0
		0x08 => Some(uart.txctrl as u64),
		0x10 => Some(uart.ie as u64),
		0x14 => {
			// TXWM: TX FIFO 恒空, 占用字节数恒不大于 txcnt, 故 txcnt > 0 时该位为 1.
			// 受调试程序关闭 TX 中断后 txcnt 为 0,
			// 硬编码为 1 时会与 Python 的 _ip_value 矛盾, 并产生无法清除的虚假 TX 中断.
			let txcnt = (uart.txctrl >> 16) & 0x7;
			let txwm = if txcnt > 0 { 1u32 } else { 0u32 };
			let rxcnt = (uart.rxctrl >> 16) & 0x7;
			let rxwm = if uart.rx_fifo_len > rxcnt {
				1u32 << 1
			} else {
				0u32
			};
			Some((txwm | rxwm) as u64)
		}
		_ => None,
	}
}

/// 向用于发送的环形缓冲区写入一条 ``[hid, byte]`` 条目.
///
/// 每条目占 2 字节, 条目容量为 ``tx_cap / 2``. ``tx_wr`` 以条目为单位单调递增,
/// 以条目容量取模得到下标. ``no_stdout`` 为 0 时随即向 stdout 回显该字节.
/// 返回 false 表示无法写入: 指针为空或条目容量为零.
pub(crate) fn uart_tx_push(uart: &FfiUartCtx, hid: u8, byte: u8) -> bool {
	if uart.tx_buf.is_null() || uart.tx_wr.is_null() {
		return false;
	}
	let ecap = uart.tx_cap / 2;
	if ecap == 0 {
		return false;
	}
	let wr_atomic = unsafe { &*(uart.tx_wr as *const AtomicU32) };
	while UART_TX_LOCK
		.compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
		.is_err()
	{
		std::hint::spin_loop();
	}
	let w = wr_atomic.load(Ordering::Relaxed);
	let e = (w % ecap) as usize;
	unsafe {
		*uart.tx_buf.add(2 * e) = hid;
		*uart.tx_buf.add(2 * e + 1) = byte;
	}
	wr_atomic.store(w.wrapping_add(1), Ordering::Release);
	UART_TX_LOCK.store(false, Ordering::Release);
	// no_stdout 非零时由 Python 的 _tx_callback 负责输出,
	// 由 drain_tx_logs 经 _flush_hart 调用 _tx_callback 写出.
	if uart.no_stdout == 0 {
		// 非阻塞写 stdout, 对端不可写时丢弃, 不阻塞 CPU 引擎线程.
		try_write_fd(libc::STDOUT_FILENO, byte);
	}
	true
}

pub(crate) fn try_handle_uart_concurrent(
	pa: u64,
	is_write: bool,
	write_data: u64,
	hid: u8,
	uart: &FfiUartCtx,
) -> Option<u64> {
	if uart.base == 0 {
		return None;
	}
	let offset = pa.wrapping_sub(uart.base);
	if offset >= 0x100 {
		return None;
	}
	if !is_write {
		return handle_uart_read(offset, uart);
	}

	// TXDATA: 写入发送环形缓冲区, 需要时同时直接写 stdout
	if !(offset == 0 && uart.tx_buf as usize != 0) {
		return None;
	}
	// 其余偏移的写入已在上一处返回 None, 由 Python 侧处理并同步 PLIC.
	if uart.tx_cap < 2 {
		return Some(0);
	}
	uart_tx_push(uart, hid, write_data as u8);
	return Some(0);
}

pub(crate) static UART_TX_LOCK: AtomicBool = AtomicBool::new(false);

#[cfg(test)]
mod tests {
	use super::try_write_fd;
	use super::uart_tx_push;
	use crate::ffi::FfiUartCtx;

	/// 构造串口上下文, 只设置发送环形缓冲区.
	/// ``no_stdout = 1`` 使写入不进入 stdout.
	fn ctx_with_ring(tx_buf: *mut u8, tx_cap: u32, tx_wr: *mut u32) -> FfiUartCtx {
		FfiUartCtx {
			base: 0x1000_0000,
			tx_buf,
			tx_cap,
			tx_wr,
			ie: 0,
			txctrl: 0,
			rxctrl: 0,
			rx_fifo_len: 0,
			tx_notify_fd: -1,
			no_stdout: 1,
			rx_notify: std::ptr::null_mut(),
		}
	}

	/// 覆盖环形缓冲区的条目布局: 每条目为 ``[hid, byte]`` 两字节, 条目容量为
	/// ``tx_cap / 2``, 写索引按条目递增并以条目容量取模.
	#[test]
	fn uart_tx_push_writes_hid_byte_entries() {
		let mut buf = [0u8; 8]; // 4 条目的容量
		let mut wr: u32 = 0;
		let ctx = ctx_with_ring(buf.as_mut_ptr(), 8, &mut wr as *mut u32);

		for (hid, byte) in [(1u8, b'a'), (2, b'b'), (3, b'c'), (4, b'd'), (5, b'e')] {
			assert!(uart_tx_push(&ctx, hid, byte));
		}

		// 第 5 条以条目容量 4 取模, 写入第 1 条所在的槽位.
		assert_eq!(wr, 5);
		assert_eq!(buf, [5, b'e', 2, b'b', 3, b'c', 4, b'd']);
	}

	/// 环形缓冲区无法写入时返回 false, 调用方随后转入通用的 ecall 处理路径.
	#[test]
	fn uart_tx_push_rejects_unattached_ring() {
		let mut wr: u32 = 0;
		let mut buf = [0u8; 8];

		let no_buf = ctx_with_ring(std::ptr::null_mut(), 8, &mut wr as *mut u32);
		assert!(!uart_tx_push(&no_buf, 0, b'x'));

		let no_wr = ctx_with_ring(buf.as_mut_ptr(), 8, std::ptr::null_mut());
		assert!(!uart_tx_push(&no_wr, 0, b'x'));

		// 字节容量不足一条目.
		let small = ctx_with_ring(buf.as_mut_ptr(), 1, &mut wr as *mut u32);
		assert!(!uart_tx_push(&small, 0, b'x'));
		assert_eq!(wr, 0);
	}

	/// 覆盖宿主 stdout 对端不可写时的行为: 管道已满时 ``try_write_fd`` 返回 0
	/// 丢弃字节而不阻塞, 对端读取后恢复可写时正常写出.
	#[test]
	fn try_write_fd_drops_when_pipe_full_and_recovers() {
		let mut fds = [0i32; 2];
		assert_eq!(unsafe { libc::pipe(fds.as_mut_ptr()) }, 0, "pipe() 失败");
		let (r, w) = (fds[0], fds[1]);
		// 写端设非阻塞, 便于循环填满管道.
		let flags = unsafe { libc::fcntl(w, libc::F_GETFL) };
		unsafe {
			libc::fcntl(w, libc::F_SETFL, flags | libc::O_NONBLOCK);
		}
		// 填满管道直到 EAGAIN: Linux 管道容量为 64 KiB, 且对端未读.
		let mut data = [0u8; 4096];
		loop {
			let n = unsafe { libc::write(w, data.as_mut_ptr() as *const libc::c_void, data.len()) };
			if n >= 0 {
				continue;
			}
			let err = std::io::Error::last_os_error();
			assert_eq!(
				err.kind(),
				std::io::ErrorKind::WouldBlock,
				"管道应被填满 (EAGAIN), 实际: {err}"
			);
			break;
		}
		// 管道已满: poll 报告 POLLOUT 不可写, 丢弃字节并返回 0, 不阻塞.
		assert_eq!(try_write_fd(w, b'x'), 0, "满管道必须丢弃而非阻塞");

		// 对端读取后恢复可写: 字节应成功写出.
		let mut buf = [0u8; 4096];
		let n = unsafe { libc::read(r, buf.as_mut_ptr() as *mut libc::c_void, buf.len()) };
		assert!(n > 0, "对端读取应成功");
		assert_eq!(try_write_fd(w, b'y'), 1, "管道恢复可写后应写出字节");

		unsafe {
			libc::close(r);
			libc::close(w);
		}
	}
}
