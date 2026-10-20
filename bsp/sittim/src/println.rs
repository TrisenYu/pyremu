//! 基于 SBI DBCN Console Write 的格式化输出。
//!
//! 每条消息先在栈上格式化到固定缓冲区, 然后一次 ecall (SBI DBCN WRITE)
//! 整条发出。M 模式的 `sbi_nputs` 持有 `console_out_lock`, 保证多 hart /
//! 多飞地并发时不发生字符级交织。

use core::fmt::{ self, Write };

/// 单条消息的格式化缓冲区字节数。长度须容得下最长的一条消息, 即陷态现场的记录,
/// 它逐项列出陷态栈帧内的全部通用寄存器, 见 crate::trap 的 report_trap_frame。
const BUF_SIZE: usize = 1024;

pub struct FmtBuf {
	buf: [u8; BUF_SIZE],
	pos: usize,
}

impl FmtBuf {
	pub fn new() -> Self {
		Self {
			buf: [0u8; BUF_SIZE],
			pos: 0,
		}
	}

	pub fn as_written(&self) -> &[u8] {
		&self.buf[..self.pos]
	}
}

impl Write for FmtBuf {
	fn write_str(&mut self, s: &str) -> fmt::Result {
		let bytes = s.as_bytes();
		let n = bytes.len().min(self.buf.len() - self.pos);
		self.buf[self.pos..self.pos + n].copy_from_slice(&bytes[..n]);
		self.pos += n;
		Ok(())
	}
}

/// 格式化输出到控制台。整条消息通过一次 SBI ecall 发出。
///
/// ```ignore
/// println!("hello {}\n", 42);
/// ```
#[macro_export]
macro_rules! println {
	($($arg:tt)*) => {
		{
        let mut __buf = $crate::println::FmtBuf::new();
        let _ = core::fmt::Write::write_fmt(
            &mut __buf,
            format_args!($($arg)*),
        );
        crate::ecall_aux::sbi_console_write(__buf.as_written());
		}
	};
}

/// 格式化输出到控制台, 不含换行符, 调用方自行加 `\n`。
#[macro_export]
macro_rules! print {
	($($arg:tt)*) => {
		{
        let mut __buf = $crate::println::FmtBuf::new();
        let _ = core::fmt::Write::write_fmt(
            &mut __buf,
            format_args!($($arg)*),
        );
        crate::ecall_aux::sbi_console_write(__buf.as_written());
		}
	};
}
