use crate::state::FfiUartCtx;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

/// 非阻塞写单字节到 fd: 先 poll(POLLOUT, timeout=0) 探测可写性, 可写才 write.
///
/// 按 FIFO 恒空 (TXDATA 读 FULL=0, 即时排空) 持续写 TXDATA, 不等待宿主
/// 消费; 若宿主 stdout 写阻塞, 则 MMIO 写相应会阻塞，致使整个 batch 停顿
/// 大量输出后 uart 停在同一位置、键盘无响应)。字节已先写入 TX ring buffer并为日志所记录
/// 此处仅即时控制台回显 — 对端不可写时丢弃该字节, 保证 CPU 引擎
/// 线程永不阻塞在输出上。
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

/// Inline read of UART shadow registers (参照 QEMU sifive_uart_read).
fn handle_uart_read(offset: u64, uart: &FfiUartCtx) -> Option<u64> {
	match offset {
		0 => Some(0), // TXDATA: bit31 = FIFO full; always 0
		0x08 => Some(uart.txctrl as u64),
		0x10 => Some(uart.ie as u64),
		0x14 => {
			// TXWM: TX FIFO 恒空 -> 若 txcnt > 0 则水位条件恒满足.
			// 必须检查 txcnt, 不可硬编码为 1 — 否则客机驱动关 TX 中断
			// (txcnt=0) 时 Rust 侧仍返回 TXWM=1, 与 Python _ip_value 矛盾,
			// 否则生成无法清除的虚假 TX 中断而引发 PLIC 中断风暴.
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

	// TXDATA: ring buffer (log archive) + direct stdout (参照 QEMU fd_chr_write).
	if offset == 0 && uart.tx_buf as usize != 0 {
		let ecap = uart.tx_cap / 2;
		if ecap <= 0 {
			return Some(0);
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
		let byte = write_data as u8;
		unsafe {
			*uart.tx_buf.add(2 * e) = hid;
			*uart.tx_buf.add(2 * e + 1) = byte;
		}
		wr_atomic.store(w.wrapping_add(1), Ordering::Release);
		UART_TX_LOCK.store(false, Ordering::Release);
		// Python TX (_tx_callback) 负责 stdout 时 (no_stdout=1):
		// 仅写 ring buffer, 由 drain_tx_logs->_flush_hart->_tx_callback 输出.
		if uart.no_stdout == 0 {
			// 非阻塞写 stdout — 对端不可写时丢弃, 绝不阻塞 CPU 引擎线程
			// (见 try_write_fd 注释).
			try_write_fd(libc::STDOUT_FILENO, byte);
		}
		return Some(0);
	}
	// IE, TXCTRL, RXCTRL, IP, DIV -> Python for PLIC updates.
	return None;
}

pub(crate) static UART_TX_LOCK: AtomicBool = AtomicBool::new(false);

#[cfg(test)]
mod tests {
	use super::try_write_fd;

	/// 回归: 宿主 stdout 对端不可写 (管道满 / 对端暂停) 时, try_write_fd 必须
	/// 返回 0 丢弃字节而非阻塞 — 否则 CPU 引擎线程被卡死在 libc::write 上,
	/// 整个 batch 无法完成 (大量输出后 uart 停在同一位置, 用户报告).
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
		// 填满管道直到 EAGAIN (Linux 管道容量 64 KiB, 对端未读).
		let mut data = [0u8; 4096];
		loop {
			let n = unsafe {
				libc::write(w, data.as_mut_ptr() as *const libc::c_void, data.len())
			};
			if n < 0 {
				let err = std::io::Error::last_os_error();
				assert_eq!(
					err.kind(),
					std::io::ErrorKind::WouldBlock,
					"管道应被填满 (EAGAIN), 实际: {err}"
				);
				break;
			}
		}
		// 管道已满: poll(POLLOUT) 报告不可写 -> 丢弃字节 (返回 0), 且不阻塞.
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
