//! 基于 SBI ecall 的控制台输出。
//!
//! 全部 I/O 通过 M-mode OpenSBI 代理，飞地自身不直接访问 UART MMIO，
//! 避免 PMP 权限问题。多 hart 串行化由 M-mode 保证。

use crate::ecall_aux;

// /// 使能发送器（空操作 — SBI ecall 无需硬件初始化）。
// pub fn uart_init() {
//     // ecall 路径无需初始化 UART 硬件
// }

/// 发送一个字节。通过 legacy SBI putchar 阻塞输出。
#[inline]
pub fn uart_putc(c: u8) {
	ecall_aux::sbi_putchar(c);
}

/// 非阻塞接收一个字节。SBI legacy 不支持非阻塞 RX。
/// 待 M-mode 提供对应的 ecall 接口后再实现。
#[allow(unused)]
pub fn uart_getc() -> Option<u8> {
	todo!("SBI ecall based UART RX is not yet implemented")
}
