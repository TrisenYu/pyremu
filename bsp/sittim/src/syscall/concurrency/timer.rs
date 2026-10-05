//! 等待定时器设施: 按到期时刻登记的等待定时器表, 与 Linux 的高精度定时器 (hrtimer)
//! 及定时器队列同型。
//!
//! 平台只有一条可编程定时器 (Sstc 的 stimecmp), 故表中最靠前的尚未到期时刻即该定时器的
//! 下一次设定值 (见 [next_deadline]), 与内核把时钟事件设备设到定时器队列头部的到期时刻
//! 同型。定时等待因此不由周期检查完成: 中断在截止时刻本身触发, trap.rs 的中断处理函数
//! 唤醒该时刻到期的线程, 故等待 5 ms 的调用在 5 ms 后唤醒, 其精度不受时间片周期
//! (config.mk 的 TIMER_INTERVAL) 限制。
//!
//! 表中条目按线程标识索引: 每个线程至多发起一个定时等待, 故一个线程占用一个条目。线程
//! 控制块不保存截止时刻, 截止时刻只在本表内。
//!
//! 一个定时器到期后不立即移除条目: 到期是中断处理函数在中断上下文里记下的结果, 而发起
//! 等待的系统调用要在自己的上下文里取回该结果并结束等待。故到期后条目由 [elapse_due]
//! 改为已到期并保留, 直到所属系统调用经 [reset] 复位为空闲; 系统调用被唤醒后重新执行时按
//! [is_expired] 取回该结果, 并按同一截止时刻继续等待, 不重新计一段完整的时长。

use crate::constants::TIMER_FREQ;
use crate::csr;

use super::thread;

/// 定时器表容量。每个线程至多发起一个定时等待, 故与线程表容量相同, 线程标识即本表下标。
const NUM_TIMERS: usize = thread::NUM_THREADS;

/// 定时器状态。
#[derive(Clone, Copy, PartialEq, Eq)]
enum TimerState {
	/// 槽位空闲。
	Free,
	/// 已登记, 截止时刻未到。
	Armed,
	/// 截止时刻已到, 所属系统调用尚未取回该结果。
	Elapsed,
}

/// 等待定时器。
#[derive(Clone, Copy)]
struct Timer {
	state: TimerState,
	/// 到期时刻, 单位与 `csr::read_time` 的返回值相同。
	deadline: u64,
}

impl Timer {
	const fn free() -> Self {
		Self { state: TimerState::Free, deadline: 0 }
	}
}

/// 等待定时器表。表位于飞地 BSS, 跨 SUSPEND 与 RESUME 存活。
static mut TIMERS: [Timer; NUM_TIMERS] = [Timer::free(); NUM_TIMERS];

/// 把纳秒时长折算为时钟源计数的增量; 不足一个计数单位的时长按一个计数单位计。
///
/// 一个计数单位的时长为 `1_000_000_000 / TIMER_FREQ` 纳秒, 在 config.mk 的取值下为 100 ns。
pub fn nsec_to_ticks(nsec: u64) -> u64 {
	nsec.div_ceil(1_000_000_000 / TIMER_FREQ)
}

/// 登记线程 *thread_id* 的等待定时器, 自当前时刻起等待 *nsec* 纳秒。
///
/// 该线程已有条目时按新的截止时刻改写。等待时长为 0 时截止时刻即当前时刻, 该情形由
/// [is_expired] 判定为已到时, 故不进入阻塞。
pub fn arm_after(thread_id: usize, nsec: u64) {
	unsafe {
		TIMERS[thread_id].state = TimerState::Armed;
		TIMERS[thread_id].deadline = csr::read_time().saturating_add(nsec_to_ticks(nsec));
	}
}

/// 把线程 *thread_id* 的条目复位为空闲, 等待结束时由所属系统调用调用。
pub fn reset(thread_id: usize) {
	unsafe {
		TIMERS[thread_id] = Timer::free();
	}
}

/// 线程 *thread_id* 是否已登记等待定时器。
///
/// 发起等待的系统调用据此判定本次进入是否需要按参数登记定时器: 已登记时沿用表中的
/// 截止时刻, 不重新计一段完整的时长。
pub fn is_armed(thread_id: usize) -> bool {
	unsafe { TIMERS[thread_id].state != TimerState::Free }
}

/// 线程 *thread_id* 的等待是否已到时。
///
/// 两种情形都算到时: 定时器中断已把条目改为已到期, 或截止时刻已过而中断尚未送达 ——
/// 后者在等待时长短于一个计数单位时出现, 如等待时长为 0。
pub fn is_expired(thread_id: usize) -> bool {
	unsafe {
		match TIMERS[thread_id].state {
			TimerState::Free => false,
			TimerState::Armed => csr::read_time() >= TIMERS[thread_id].deadline,
			TimerState::Elapsed => true,
		}
	}
}

/// 表中最早的尚未到期时刻; 表中无此类定时器时返回 None。
///
/// 定时器中断据此设定 stimecmp: 取该时刻与下一个时间片边界中的较早者。
pub fn next_deadline() -> Option<u64> {
	unsafe {
		let mut earliest: Option<u64> = None;
		for i in 0..NUM_TIMERS {
			if TIMERS[i].state != TimerState::Armed {
				continue;
			}
			let deadline = TIMERS[i].deadline;
			earliest = match earliest {
				Some(e) if e <= deadline => Some(e),
				_ => Some(deadline),
			};
		}
		earliest
	}
}

/// 把一个截止时刻不晚于 *now* 的条目由已登记改为已到期, 返回其所属线程的标识;
/// 无此类条目时返回 None。
///
/// 供定时器中断处理函数循环调用: 每取得一个即唤醒其所属线程。条目保留在表内, 由所属
/// 系统调用取回结果时经 [reset] 复位。
pub fn elapse_due(now: u64) -> Option<usize> {
	unsafe {
		for i in 0..NUM_TIMERS {
			if TIMERS[i].state == TimerState::Armed && now >= TIMERS[i].deadline {
				TIMERS[i].state = TimerState::Elapsed;
				return Some(i);
			}
		}
		None
	}
}

#[cfg(test)]
pub mod tests {
	use super::*;

	/// 清空定时器表。表与时钟源读数都由用例共用, 故调用方须先取时钟源锁
	/// (crate::csr::clock_lock)。
	pub fn reset_table() {
		unsafe {
			TIMERS = [Timer::free(); NUM_TIMERS];
		}
	}

	/// 取时钟源锁, 把时钟源读数设到 *now*, 并清空定时器表。
	fn world(now: u64) -> std::sync::MutexGuard<'static, ()> {
		let guard = crate::csr::clock_lock();
		crate::csr::set_time(now);
		reset_table();
		guard
	}

	/// 等待时长的折算: 不足一个计数单位的时长按一个计数单位计。
	#[test]
	fn test_nsec_to_ticks_rounds_up() {
		let _guard = world(0);
		assert_eq!(nsec_to_ticks(0), 0);
		assert_eq!(nsec_to_ticks(1), 1);
		assert_eq!(nsec_to_ticks(99), 1);
		assert_eq!(nsec_to_ticks(100), 1);
		assert_eq!(nsec_to_ticks(101), 2);
		// 5 ms 折算为 50_000 个计数单位, 即时间片周期 (1 ms) 的 5 倍。
		assert_eq!(nsec_to_ticks(5_000_000), 50_000);
	}

	/// 登记的定时器在截止时刻之前不到时, 到达截止时刻时到时。
	#[test]
	fn test_timer_expires_at_its_deadline() {
		let _guard = world(1_000);
		arm_after(3, 5_000_000);
		assert!(is_armed(3));
		assert!(!is_expired(3));

		// 到期时刻为 1_000 + 50_000。
		crate::csr::set_time(51_000 - 1);
		assert!(!is_expired(3));
		// 定时器中断尚未送达, 但截止时刻已到, 仍判为到时。
		crate::csr::set_time(51_000);
		assert!(is_expired(3));
	}

	/// 复位后条目回到空闲, 等待不再判为到时。
	#[test]
	fn test_reset_clears_the_timer() {
		let _guard = world(0);
		arm_after(1, 0);
		assert!(is_armed(1));
		assert!(is_expired(1));
		reset(1);
		assert!(!is_armed(1));
		assert!(!is_expired(1));
	}

	/// 未被登记的线程恒不判为到时。
	#[test]
	fn test_unarmed_thread_never_expires() {
		let _guard = world(1_000_000);
		assert!(!is_expired(7));
	}

	/// 表中最早的尚未到期时刻由已登记且未到期的条目给出; 已被中断处理函数取走的条目
	/// 不再参与。
	#[test]
	fn test_next_deadline_is_the_earliest_of_the_armed_timers() {
		let _guard = world(0);
		assert_eq!(next_deadline(), None);
		arm_after(2, 9_000);
		arm_after(5, 100);
		arm_after(9, 5_000);
		// 100 ns 折算为 1 个计数单位。
		assert_eq!(next_deadline(), Some(1));

		assert_eq!(elapse_due(1), Some(5));
		assert_eq!(next_deadline(), Some(5_000 / 100));
		assert_eq!(elapse_due(1), None);
	}

	/// 中断处理函数按到期时刻逐个取走条目, 每次返回所属线程的标识。
	#[test]
	fn test_elapse_due_returns_the_owning_threads() {
		let _guard = world(0);
		// 100 ns 与 200 ns 分别折算为 1 个与 2 个计数单位。
		arm_after(4, 100);
		arm_after(6, 200);

		// 只有一个条目到期时只取走该条目。
		assert_eq!(elapse_due(1), Some(4));
		assert_eq!(elapse_due(1), None);

		// 两个条目都已到期时逐个取走。
		assert_eq!(elapse_due(2), Some(6));
		assert_eq!(elapse_due(2), None);

		// 取走的条目仍保留在表内, 直到所属系统调用撤销它。
		assert!(is_armed(4) && is_expired(4));
		assert!(is_armed(6) && is_expired(6));
	}

	/// 同一线程重复登记按新的截止时刻改写, 不占用第二个条目。
	#[test]
	fn test_rearming_replaces_the_deadline() {
		let _guard = world(0);
		arm_after(8, 1_000);
		arm_after(8, 10_000);
		assert!(is_armed(8));
		assert_eq!(next_deadline(), Some(10_000 / 100));
		assert_eq!(elapse_due(10), None);
		assert_eq!(elapse_due(100), Some(8));
	}
}
