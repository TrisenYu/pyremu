//! futex 内核实现 (Linux futex(2): FUTEX_WAIT / FUTEX_WAKE / FUTEX_REQUEUE)。
//!
//! 线程阻塞/唤醒与调度解耦在 thread.rs: 本模块只做 futex 语义
//! (操作码解码、字值比较、EAGAIN 判定), 需要阻塞或唤醒时调用
//! thread::block_current / thread::wake_blocked / thread::requeue_blocked。
//!
//! musl 各同步原语落到的实际操作 (见 vendor/musl):
//! - 互斥锁与自旋锁的 __wait: FUTEX_WAIT / FUTEX_WAKE, 可带 FUTEX_PRIVATE;
//! - pthread_cond 私有条件变量: 等待者阻塞在自身栈上 node.barrier 字的
//!   FUTEX_WAIT, signal 对其 a_swap 清零后 FUTEX_WAKE;
//! - pthread_cond 移交 (relock 手递手): 唤醒者持锁状态下对下一位等待者
//!   FUTEX_REQUEUE 迁移到互斥锁字 (nr_wake = 0, nr_requeue = 1);
//! - pthread_join: 对 detach_state 字的 FUTEX_WAIT。
//! 均未使用 WAIT_BITSET / CLOCK 位; 操作码取低 7 位即可覆盖。

#![allow(dead_code)]

use core::ptr;

use crate::syscall::concurrency::thread;
use crate::syscall::{EAGAIN, EINVAL};

// futex 操作码 (linux/futex.h)。高位为标志 (FUTEX_PRIVATE = 128 等),
// 与操作 OR 在一起; 取低 7 位即得操作本身。
const FUTEX_WAIT: u64 = 0;
const FUTEX_WAKE: u64 = 1;
const FUTEX_REQUEUE: u64 = 3;
const FUTEX_OP_MASK: u64 = 0x7f;

/// 读取用户地址处的 32 位字 (futex 等待值比较用; S 模式以 SUM = 1 直读)。
fn read_user_u32(addr: u64) -> u32 {
	unsafe { ptr::read_volatile(addr as *const u32) }
}

/// futex 系统调用入口。参数布局 (Linux rv64):
/// futex(uaddr = a0, op = a1, val = a2, timeout = a3, uaddr2 = a4)。
/// REQUEUE 的第 4 参数实为 nr_requeue (复用 timeout 槽); 超时语义本运行时
/// 未实现, 等待均按无限阻塞处理 (t-test1 未使用带超时的条件等待)。
pub fn futex(uaddr: u64, op: u64, val: u64, arg3: u64, uaddr2: u64) -> u64 {
	match op & FUTEX_OP_MASK {
		FUTEX_WAIT => futex_wait(uaddr, val),
		FUTEX_WAKE => thread::wake_blocked(uaddr, val),
		FUTEX_REQUEUE => thread::requeue_blocked(uaddr, val, arg3, uaddr2),
		_ => EINVAL,
	}
}

/// FUTEX_WAIT: 若 uaddr 处字值仍等于 val 则阻塞当前线程; 否则立即 EAGAIN。
fn futex_wait(uaddr: u64, val: u64) -> u64 {
	// 与 Linux 一致: 非 4 字节对齐的地址返回 EINVAL。
	if uaddr & 0x3 != 0 {
		return EINVAL;
	}
	if read_user_u32(uaddr) != val as u32 {
		return EAGAIN;
	}
	// 阻塞当前线程并让出调度; 返回后由 trap.rs 切换到其它可运行线程。
	thread::block_current(uaddr);
	0
}
