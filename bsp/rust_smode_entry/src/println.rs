//! 基于 SBI DBCN Console Write 的格式化输出。
//!
//! 每条消息先在栈上格式化到固定缓冲区, 然后一次 ecall (SBI DBCN WRITE)
//! 整条发出。M 模式的 `sbi_nputs` 持有 `console_out_lock`, 保证多 hart /
//! 多飞地并发时不发生字符级交织。

use core::fmt::{self, Write};

use crate::ecall_aux;

/// 栈上格式化缓冲区。
const BUF_SIZE: usize = 512;

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
    ($($arg:tt)*) => {{
        let mut __buf = $crate::println::FmtBuf::new();
        let _ = core::fmt::Write::write_fmt(
            &mut __buf,
            format_args!($($arg)*),
        );
        $crate::ecall_aux::sbi_console_write(__buf.as_written());
    }};
}

/// 格式化输出到控制台（不含换行符，调用方自行加 `\n`）。
#[macro_export]
macro_rules! print {
    ($($arg:tt)*) => {{
        let mut __buf = $crate::println::FmtBuf::new();
        let _ = core::fmt::Write::write_fmt(
            &mut __buf,
            format_args!($($arg)*),
        );
        $crate::ecall_aux::sbi_console_write(__buf.as_written());
    }};
}

/// 输出以 `\0` 结尾的字符串（逐字符，调试用）。
#[allow(unused)]
pub fn puts(s: &str) {
	for &byte in s.as_bytes() {
		if byte == 0 {
			break;
		}
		ecall_aux::sbi_putchar(byte);
	}
}

/// 将 u64 按十六进制输出（逐字符，调试用）。
#[allow(unused)]
pub fn putx(val: u64) {
	for shift in (0..16).rev() {
		let nibble = ((val >> (shift * 4)) & 0xF) as u8;
		ecall_aux::sbi_putchar(if nibble < 10 {
			b'0' + nibble
		} else {
			b'a' + nibble - 10
		});
	}
}
