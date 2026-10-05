//! ppoll 系统调用。
//!
//! 就绪状态按描述绑定的对象求出: 控制台按打开标志的访问模式, 文件系统节点恒可读可写,
//! 套接字按协议栈的可接收与可发送状态。等待由两处并列的唤醒来源结束, 描述符就绪由网络
//! 模块按 [`POLL_WAIT_ADDR`] 唤醒, 时限到达由等待定时器设施按线程标识唤醒。

use core::ptr;

use crate::ext_mod::vfs_ops::{mode_readable, mode_writable};

use super::consts::POLL_WAIT_ADDR;
use super::super::concurrency::{thread, timer};
use super::super::fdtable::{FDS, FdTarget, MAX_FD, Tables, open_file_of};
use super::super::net;
use super::super::types::Timespec;
use super::super::{EFAULT, EINVAL};

/// 求描述符 *fd* 在 *events* 上的就绪状态。控制台 fd 按打开标志的访问模式给出可读与
/// 可写方向, 文件 fd 恒可读可写, 套接字 fd 按协议栈的可接收与可发送状态给出; 负 fd
/// 返回 0, 空闲槽位与越界 fd 返回 POLLNVAL。
fn fd_revents(tables: &Tables, fd: i64, events: u16) -> u16 {
	const POLLIN: u16 = 0x0001;
	const POLLOUT: u16 = 0x0004;
	const POLLNVAL: u16 = 0x0020;

	let Ok(fd) = u64::try_from(fd) else {
		return 0;
	};
	let Some((_, open_file)) = open_file_of(tables, fd) else {
		return POLLNVAL;
	};
	match open_file.target {
		FdTarget::Console => {
			let mut console_revents: u16 = 0;
			if mode_readable(open_file.flags) {
				console_revents |= events & POLLIN;
			}
			if mode_writable(open_file.flags) {
				console_revents |= events & POLLOUT;
			}
			console_revents
		}
		// 文件 fd: 恒可读可写
		FdTarget::Vfs(..) => events & (POLLIN | POLLOUT),
		FdTarget::Socket(index) => {
			let mut socket_revents: u16 = 0;
			if net::can_recv(index) {
				socket_revents |= events & POLLIN;
			}
			if net::can_send(index) {
				socket_revents |= events & POLLOUT;
			}
			socket_revents
		}
	}
}

/// 扫描一遍描述符数组, 把各描述符的就绪状态写入 revents, 返回就绪的描述符个数。
///
/// 每次扫描各自取一次 fd 表锁, 扫描之间不持锁: 等待期间须允许其它线程改动 fd 表。
fn scan_fds(fds: u64, nfds: u64) -> u64 {
	let tables = FDS.lock();
	let mut ready: u64 = 0;
	for i in 0..nfds {
		let slot = fds.wrapping_add(i.wrapping_mul(8));
		let fd = (unsafe { (slot as *const i32).read_volatile() }) as i64;
		let events = unsafe { (slot.wrapping_add(4) as *const u16).read_volatile() };
		let revents = fd_revents(&tables, fd, events);
		unsafe {
			(slot.wrapping_add(6) as *mut u16).write_volatile(revents);
		}
		if revents != 0 {
			ready += 1;
		}
	}
	ready
}

/// 读出 ppoll 的时限, 折算为纳秒时长。*timeout* 取空表示不限时, 返回 Ok(None)。
///
/// tv_sec 为负或 tv_nsec 不在 [0, 1_000_000_000) 内时返回 Err(EINVAL)。
fn read_poll_timeout(timeout: u64) -> Result<Option<u64>, u64> {
	if timeout == 0 {
		return Ok(None);
	}
	let ts = unsafe { ptr::read_unaligned(timeout as *const Timespec) };
	if ts.tv_sec < 0 || ts.tv_nsec < 0 || ts.tv_nsec >= 1_000_000_000 {
		return Err(EINVAL);
	}
	Ok(Some(
		(ts.tv_sec as u64).saturating_mul(1_000_000_000).saturating_add(ts.tv_nsec as u64),
	))
}

/// ppoll(73): 逐个求出描述符就绪状态并写入 revents, 返回就绪的描述符个数; 无描述符
/// 就绪时等待至多 *timeout* 指定的时长。
///
/// 时限实现与 nanosleep 同: 时限交给等待定时器设施 (见 concurrency/timer.rs), 由定时器
/// 中断在截止时刻唤醒, 故等待 5 ms 的调用在 5 ms 后返回, 其精度不受时间片周期限制。
/// 描述符就绪与时限到达是两个并列的唤醒来源, 前者由网络模块按 [`POLL_WAIT_ADDR`]
/// 唤醒, 后者由中断按线程标识唤醒。
///
/// *nfds* 超过 `RLIMIT_NOFILE` 时返回 EINVAL, 本层以 [`MAX_FD`] 为上界; *nfds* 为 0 且
/// 有时限时睡眠该时长后返回 0, 不限时则一直等待; *timeout* 非空而 *fds* 取空返回 EFAULT。
/// 载荷传入的第 4 个参数 (信号掩码) 不经本层: 派发只传前三个参数 (见 syscall/mod.rs)。
pub fn ppoll_handler(fds: u64, nfds: u64, timeout: u64) -> u64 {
	if nfds > (MAX_FD as u64) {
		return EINVAL;
	}
	if fds == 0 && nfds != 0 {
		return EFAULT;
	}
	let limit = match read_poll_timeout(timeout) {
		Ok(v) => v,
		Err(e) => {
			return e;
		}
	};

	let me = thread::current_thread_id();
	// 被中断后重新执行本调用时沿用表中已有的截止时刻, 不重新计一段完整的时长。
	if let Some(nsec) = limit {
		if !timer::is_armed(me) {
			timer::arm_after(me, nsec);
		}
	}
	loop {
		let ready = scan_fds(fds, nfds);
		if ready != 0 {
			timer::reset(me);
			return ready;
		}
		if timer::is_expired(me) {
			// 时限已到: 收起定时器并按时限到期返回 0。
			timer::reset(me);
			return 0;
		}
		thread::block_current(POLL_WAIT_ADDR);
		// 有切换目标时退回 ecall 之前重新执行本次调用, 使等待期间让出的线程在别处
		// 运行; 无切换目标说明是被唤醒而来不及登记切换, 重新扫描一次。
		if thread::switch_pending() {
			thread::request_restart();
			return 0;
		}
	}
}
