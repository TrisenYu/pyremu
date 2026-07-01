//! 基于 SBI ecall 的格式化输出。通过 `core::fmt::Write` 实现，不引入完整 printf。
//!
//! 输出由 M-mode OpenSBI 代理，无需持有 UART 锁。

use core::fmt::{self, Write};

use crate::uart;

/// 标准输出句柄。
pub struct Stdout;

impl Write for Stdout {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        for &byte in s.as_bytes() {
            uart::uart_putc(byte);
        }
        Ok(())
    }
}

/// 格式化输出到控制台。
///
/// ```ignore
/// println!("hello {}\n", 42);
/// ```
#[macro_export]
macro_rules! println {
    ($($arg:tt)*) => {
        let _ = core::fmt::Write::write_fmt(
            &mut $crate::println::Stdout,
            format_args!($($arg)*),
        );
    };
}

/// 格式化输出到控制台（不含换行符，调用方自行加 `\n`）。
#[macro_export]
macro_rules! print {
    ($($arg:tt)*) => {{
        let _ = core::fmt::Write::write_fmt(
            &mut $crate::println::Stdout,
            format_args!($($arg)*),
        );
    }};
}

/// 输出以 `\0` 结尾的字符串。
#[allow(unused)]
pub fn puts(s: &str) {
    for &byte in s.as_bytes() {
        if byte == 0 {
            break;
        }
        uart::uart_putc(byte);
    }
}

/// 将 u64 按十六进制输出（无 0x 前缀）。
#[allow(unused)]
pub fn putx(val: u64) {
    for shift in (0..16).rev() {
        let nibble = ((val >> (shift * 4)) & 0xF) as u8;
        uart::uart_putc(if nibble < 10 {
            b'0' + nibble
        } else {
            b'a' + nibble - 10
        });
    }
}
