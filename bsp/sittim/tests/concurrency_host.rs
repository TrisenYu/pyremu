//! 用户线程运行时与等待定时器设施在主机目标上的单元测试入口。
//!
//! 本层覆盖两件事: 定时等待的截止时刻由谁持有, 以及定时到期的唤醒如何进入被阻塞的系统
//! 调用。二者在飞地内分处 `src/syscall/concurrency/` 的两个文件, 用例一并取入, 覆盖的是
//! 两者真实相遇的那一层。
//!
//! 本层对平台的依赖集中在三处, 由本文件给出替身: 时钟源 [`csr`]、内核栈分配 [`mem`], 以及
//! 让出宿主与终止飞地两条 ecall [`ecall_aux`]。三处替身只记录调用与给出可设置的返回值,
//! 不改变本层的判定。陷态栈帧取入真实源文件 [`trap_frame`], 与 [`thread`] 相同。
//!
//! 不在覆盖范围内的一项: 飞地内 [`thread::block_current`] 在没有其它可运行线程时经
//! [`ecall_aux::enclave_call_suspend`] 让出宿主, 该让出在主机侧没有对应行为, 替身只记录
//! 调用。故走到该让出的用例以「不产生该调用」为断言, 其余用例在阻塞之前先经
//! [`thread::clone_thread`] 登记一个可运行线程, 使阻塞按「登记切换目标并返回」这条路径
//! 结束, 与飞地内存在其它可运行线程时完全一致。

#![allow(dead_code)]

use core::sync::atomic::Ordering;

#[path = "../src/constants.rs"]
mod constants;

#[path = "../src/syscall/errno.rs"]
mod errno;

#[path = "../src/syscall/types.rs"]
pub mod types;

#[path = "../src/syscall/concurrency/timer.rs"]
pub mod timer;

#[path = "../src/syscall/concurrency/thread.rs"]
pub mod thread;

#[path = "../src/trap_frame.rs"]
mod trap_frame;

/// 时钟源与 S 模式 CSR 的替身。本层只用到时钟源与 sepc/sstatus 四项, 前者可被用例设定,
/// 后者只作记录。
mod csr {
	use core::sync::atomic::{AtomicU64, Ordering};
	use std::sync::{Mutex, MutexGuard, OnceLock};

	/// 时钟源读数, 由用例设定。
	static TIME: AtomicU64 = AtomicU64::new(0);
	/// 每次读时钟源时按该增量推进时钟源, 0 表示不推进。用例以它模拟等待期间流逝的时间。
	static TIME_STEP: AtomicU64 = AtomicU64::new(0);
	/// 记录的 sepc 与 sstatus, 由用例读出。
	static SEPC: AtomicU64 = AtomicU64::new(0);
	static SSTATUS: AtomicU64 = AtomicU64::new(0);

	/// 时钟源锁。定时器表与时钟源读数都是进程内的全局量, 用例取该锁串行执行。
	fn clock_mutex() -> &'static Mutex<()> {
		static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
		LOCK.get_or_init(|| Mutex::new(()))
	}

	/// 取时钟源锁。用例在持有该锁期间重置定时器表并设定时钟源读数。
	pub fn clock_lock() -> MutexGuard<'static, ()> {
		let mutex = clock_mutex();
		let guard = mutex
			.lock()
			.unwrap_or_else(|poisoned| poisoned.into_inner());
		mutex.clear_poison();
		guard
	}

	/// 读时钟源计数。飞地内为 CSR 0xC01 的读取。
	///
	/// 已设定推进量时每次读取都推进时钟源, 使不返回的等待循环也能看到时间流逝。
	pub fn read_time() -> u64 {
		let step = TIME_STEP.load(Ordering::SeqCst);
		if step == 0 {
			return TIME.load(Ordering::SeqCst);
		}
		TIME.fetch_add(step, Ordering::SeqCst) + step
	}

	/// 设定时钟源计数, 供用例推进时间。
	pub fn set_time(v: u64) {
		TIME.store(v, Ordering::SeqCst);
	}

	/// 设定每次读时钟源时推进的计数, 0 表示不推进。
	pub fn set_time_step(step: u64) {
		TIME_STEP.store(step, Ordering::SeqCst);
	}

	pub fn read_sepc() -> u64 {
		SEPC.load(Ordering::SeqCst)
	}

	pub fn write_sepc(v: u64) {
		SEPC.store(v, Ordering::SeqCst);
	}

	pub fn read_sstatus() -> u64 {
		SSTATUS.load(Ordering::SeqCst)
	}

	pub fn write_sstatus(v: u64) {
		SSTATUS.store(v, Ordering::SeqCst);
	}
}

/// 内核栈分配的替身: 自一张进程内的静态数组切分, 使线程运行时的帧写入落在真实内存内。
mod mem {
	use core::sync::atomic::{AtomicUsize, Ordering};

	/// 可切分的内核栈数。本入口同时存在的线程不超过一个主线程与一个对端线程。
	const NUM_KSTACKS: usize = 4;
	/// 每个内核栈的字节数, 与 [`super::thread::THREAD_KSTACK_SIZE`] 相同。
	const KSTACK_SIZE: usize = 0x4000;

	#[repr(align(16))]
	struct KStacks([[u8; KSTACK_SIZE]; NUM_KSTACKS]);

	static mut KSTACKS: KStacks = KStacks([[0; KSTACK_SIZE]; NUM_KSTACKS]);
	/// 已切分的内核栈数。
	static TAKEN: AtomicUsize = AtomicUsize::new(0);

	/// 把已切分的内核栈数复位, 使下一个用例从第一张内核栈开始。
	pub fn reset_kstacks() {
		TAKEN.store(0, Ordering::SeqCst);
	}

	/// 返回一张内核栈的栈顶地址; 已切分完时返回 0, 与飞地内的失败返回值相同。
	pub fn try_alloc_thread_kstack(_kstack_size: u64) -> u64 {
		let index = TAKEN.fetch_add(1, Ordering::SeqCst);
		if index >= NUM_KSTACKS {
			return 0;
		}
		unsafe {
			let base = core::ptr::addr_of_mut!(KSTACKS.0) as *mut [u8; KSTACK_SIZE];
			(*base.add(index)).as_mut_ptr().add(KSTACK_SIZE) as u64
		}
	}
}

/// 让出宿主与终止飞地两条 ecall 的替身。前者只记录调用, 后者在主机侧终止进程, 与飞地内
/// 不返回的行为一致。
mod ecall_aux {
	use core::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

	/// 自愿让出宿主的次数。
	pub static SUSPEND_COUNT: AtomicUsize = AtomicUsize::new(0);
	/// 让出宿主期间按该地址唤醒一位等待者, 0 表示不唤醒。
	static WAKE_ON_SUSPEND_ADDR: AtomicU64 = AtomicU64::new(0);

	/// 设定让出宿主期间唤醒的等待地址, 0 表示不唤醒。
	pub fn set_wake_on_suspend(wait_addr: u64) {
		WAKE_ON_SUSPEND_ADDR.store(wait_addr, Ordering::SeqCst);
	}

	pub fn enclave_call_suspend(_short_msg: u64) -> (u64, u64, u64) {
		// 唤醒在本次让出期间投递, 故一次阻塞只让出一次; 再次让出说明调用方没有返回。
		assert_eq!(SUSPEND_COUNT.fetch_add(1, Ordering::SeqCst), 0);
		let wait_addr = WAKE_ON_SUSPEND_ADDR.load(Ordering::SeqCst);
		if wait_addr != 0 {
			crate::thread::wake_blocked(wait_addr, 1);
		}
		(0, 0, 0)
	}

	pub fn enclave_call_exit(code: u64) -> ! {
		panic!("enclave_call_exit({})", code);
	}
}

/// 系统调用层。运行时的 `syscall` 模块在本入口内按同一路径引用, 故线程运行时的
/// `crate::syscall::{...}` 与 `crate::syscall::concurrency::{...}` 都能解析。
pub mod syscall {
	pub use crate::errno::*;
	pub use crate::types;

	pub mod concurrency {
		pub use crate::thread;
		pub use crate::timer;

		/// 进程层替身。本入口不覆盖进程式 clone 与进程退出, 故只给出最小判定与空实现。
		pub mod proc {
			use crate::trap_frame::TrapGprs;

			/// 进程式 clone 的标志位: 未请求共享地址空间。
			const CLONE_VM: u64 = 0x0000_0100;

			pub fn is_process_clone(flags: u64) -> bool {
				(flags & CLONE_VM) == 0
			}

			pub fn clone_process(_gprs: &TrapGprs, _flags: u64) -> u64 {
				0
			}

			pub fn exit_process(_slot: usize, _code: u64) {}

			pub fn activate(_slot: usize) {}
		}

		/// futex 替身。本入口不覆盖 futex 语义, 故一律报告无等待者。
		pub mod futex {
			pub fn futex(_uaddr: u64, _op: u64, _val: u64, _arg3: u64, _uaddr2: u64) -> u64 {
				0
			}
		}
	}
}

mod tests {
	use super::*;
	use crate::syscall::{EFAULT, EINVAL};
	const CLONE_VM: u64 = 0x0000_0100;
	const CLONE_THREAD: u64 = 0x0001_0000;

	fn five_ms_ticks() -> u64 {
		timer::nsec_to_ticks(5_000_000)
	}

	fn world() -> std::sync::MutexGuard<'static, ()> {
		let guard = csr::clock_lock();
		csr::set_time(0);
		csr::set_time_step(0);
		timer::tests::reset_table();
		ecall_aux::SUSPEND_COUNT.store(0, Ordering::SeqCst);
		ecall_aux::set_wake_on_suspend(0);
		mem::reset_kstacks();
		thread::tests::reset_scheduler(mem::try_alloc_thread_kstack(thread::THREAD_KSTACK_SIZE));
		guard
	}

	/// 在当前线程之外登记一个可运行线程, 使本线程的阻塞按「登记切换目标并返回」结束。
	fn spawn_peer() {
		let gprs = trap_frame::TrapGprs {
			regs: [0; 32],
			sp_before_trap: 0,
		};
		thread::clone_thread(&gprs, CLONE_VM | CLONE_THREAD, 0, 0, 0, 0);
		thread::tests::clear_switch();
	}

	fn timespec(sec: i64, nsec: i64) -> types::Timespec {
		types::Timespec {
			tv_sec: sec,
			tv_nsec: nsec,
		}
	}

	const WAIT_ADDR: u64 = 0x1000;

	#[test]
	fn test_block_current_returns_when_woken_with_no_other_runnable_thread() {
		let _guard = world();
		ecall_aux::set_wake_on_suspend(WAIT_ADDR);
		thread::block_current(WAIT_ADDR);

		assert_eq!(ecall_aux::SUSPEND_COUNT.load(Ordering::SeqCst), 1);
		assert!(!thread::switch_pending());
	}

	/// 本线程持有尚未到期的截止时刻且本飞地没有其它可运行线程时, 阻塞留在本 hart 上等到
	/// 该时刻, 不交还宿主。
	///
	/// 让出宿主会使该时刻失效: 飞地的 stimecmp 随上下文保存, 到期不再触发定时器中断,
	/// 发起等待的系统调用于是没有任何唤醒来源。故截止时刻本身即结束本次等待的依据。
	#[test]
	fn test_block_current_waits_out_its_cutoff_tick_without_yielding_to_host() {
		let _guard = world();
		let me = thread::current_thread_id();
		csr::set_time(1_000);
		timer::set_wait_timer(me, 5_000_000);
		// 等待期间时间流逝: 每次读时钟源推进一个计数单位, 直至截止时刻。
		csr::set_time_step(1);

		thread::block_current(WAIT_ADDR);

		assert_eq!(ecall_aux::SUSPEND_COUNT.load(Ordering::SeqCst), 0);
		assert!(!thread::tests::is_blocked(me));
		assert!(csr::read_time() >= 1_000 + five_ms_ticks());
	}

	/// 时限指针取空的 nanosleep 返回 EFAULT。
	#[test]
	fn test_nanosleep_rejects_a_null_request() {
		let _guard = world();
		assert_eq!(thread::nanosleep_handler(0, 0), EFAULT);
		assert!(!timer::has_wait_timer(thread::current_thread_id()));
	}

	/// 时限字段越界的 nanosleep 返回 EINVAL: tv_nsec 不在 [0, 1_000_000_000) 内, 或
	/// tv_sec 为负。
	#[test]
	fn test_nanosleep_rejects_an_invalid_timespec() {
		let _guard = world();
		let too_many_nsec = timespec(0, 1_000_000_000);
		let negative_nsec = timespec(0, -1);
		let negative_sec = timespec(-1, 0);
		for req in [too_many_nsec, negative_nsec, negative_sec] {
			assert_eq!(
				thread::nanosleep_handler(&req as *const types::Timespec as u64, 0),
				EINVAL
			);
		}
		assert!(!timer::has_wait_timer(thread::current_thread_id()));
	}

	/// 时长为 0 的 nanosleep 立即返回 0, 不登记定时器也不进入阻塞。
	#[test]
	fn test_nanosleep_with_a_zero_duration_returns_immediately() {
		let _guard = world();
		let none = timespec(0, 0);
		assert_eq!(
			thread::nanosleep_handler(&none as *const types::Timespec as u64, 0),
			0
		);
		assert!(!timer::has_wait_timer(thread::current_thread_id()));
		assert!(!thread::switch_pending());
	}

	/// nanosleep 按传入时长设定截止时刻: 5 ms 的睡眠登记 5 ms 折合的计数增量。
	///
	/// 该用例判定截止时刻是否取整到时间片边界: 取整时登记的增量是 TICKS_PER_SCHED_TICK 的
	/// 倍数, 而不是 5 ms 折合的增量。
	#[test]
	fn test_nanosleep_sets_its_cutoff_tick_from_the_requested_duration() {
		let _guard = world();
		let me = thread::current_thread_id();
		let req = timespec(0, 5_000_000);
		csr::set_time(1_000);

		spawn_peer();
		assert_eq!(
			thread::nanosleep_handler(&req as *const types::Timespec as u64, 0),
			0
		);
		assert_eq!(timer::next_deadline(), Some(1_000 + five_ms_ticks()));
		// 睡眠期间让出的线程由本调用退回 ecall 之前运行, 故本次调用请求重启。
		assert!(thread::take_restart());
		assert!(thread::switch_pending());
		// 阻塞不产生让出宿主的调用: 有可运行线程时只登记切换目标。
		assert_eq!(ecall_aux::SUSPEND_COUNT.load(Ordering::SeqCst), 0);
		assert!(timer::has_wait_timer(me));
	}

	/// 被中断后重新执行 nanosleep 时沿用表中已有的截止时刻, 不重新计一段完整的时长;
	/// 到达该时刻后返回 0 并收起条目。
	#[test]
	fn test_nanosleep_keeps_its_cutoff_tick_across_restarts() {
		let _guard = world();
		let me = thread::current_thread_id();
		let req = timespec(0, 5_000_000);
		let req_addr = &req as *const types::Timespec as u64;
		csr::set_time(1_000);
		spawn_peer();

		assert_eq!(thread::nanosleep_handler(req_addr, 0), 0);
		assert_eq!(timer::next_deadline(), Some(1_000 + five_ms_ticks()));

		// 重新执行本次调用: 截止时刻未到, 按同一截止时刻再次登记切换。
		thread::tests::clear_switch();
		csr::set_time(1_000 + five_ms_ticks() - 1);
		assert_eq!(thread::nanosleep_handler(req_addr, 0), 0);
		assert!(thread::take_restart());
		assert_eq!(timer::next_deadline(), Some(1_000 + five_ms_ticks()));

		// 到达截止时刻: 返回 0 且不再登记切换, 条目收起。
		thread::tests::clear_switch();
		csr::set_time(1_000 + five_ms_ticks());
		assert_eq!(thread::nanosleep_handler(req_addr, 0), 0);
		assert!(!thread::switch_pending());
		assert!(!thread::take_restart());
		assert!(!timer::has_wait_timer(me));
	}

	/// 定时器中断的唤醒路径: 到期条目取走后置为已到期, 按线程标识唤醒其所有者; 被唤醒的
	/// 线程重新执行 nanosleep 即取回到期结果并收起条目。
	///
	/// 与按等待地址匹配的唤醒并列, 该路径不经等待地址。
	#[test]
	fn test_timer_interrupt_wakes_the_sleeping_thread() {
		let _guard = world();
		let me = thread::current_thread_id();
		let req = timespec(0, 5_000_000);
		let req_addr = &req as *const types::Timespec as u64;
		csr::set_time(1_000);
		spawn_peer();

		assert_eq!(thread::nanosleep_handler(req_addr, 0), 0);
		// 取走首次进入时登记的重启请求, 使末尾的断言只反映最后一次进入。
		assert!(thread::take_restart());
		let cutoff_tick = 1_000 + five_ms_ticks();

		// 截止时刻之前的定时器中断不改动条目, 也不唤醒任何线程。
		assert_eq!(timer::elapse_due(cutoff_tick - 1), None);

		// 到达截止时刻: 中断取走条目并唤醒其所有者。
		assert_eq!(timer::elapse_due(cutoff_tick), Some(me));
		assert!(timer::is_expired(me));
		assert!(thread::wake_thread(me));
		// 已唤醒的线程不再是阻塞状态, 重复唤醒不改动状态。
		assert!(!thread::wake_thread(me));

		// 重新执行本次调用: 取回到期结果, 返回 0 并收起条目。
		thread::tests::clear_switch();
		assert_eq!(thread::nanosleep_handler(req_addr, 0), 0);
		assert!(!thread::take_restart());
		assert!(!timer::has_wait_timer(me));
	}
}
