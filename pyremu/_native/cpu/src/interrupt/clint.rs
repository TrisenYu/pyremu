use crate::concurrent::{timer_deadline_own, ConcurrentClintCtx, ModuleState};
use crate::state::HartState;
use std::cell::Cell;
use std::sync::atomic::{AtomicU8, Ordering};
// ============================================================
// CLINT 内联处理函数
// ============================================================
//
// 作用于加速执行引擎的 ClintCtx, 即非并发路径, 由 handlers.rs 重新导出。

use crate::handlers::MemAccess;

/// mtimecmp 写入处理
///
/// - ``val == 0``: 禁用定时器, 清除 MTIP 并把 deadline 置 0。
/// - ``val <= now_mtime``, 即已到期: 立即置位 MTIP 并把 deadline 置 0。
/// - ``val > now_mtime``, 清除 MTIP, deadline 取写入时刻本 hart 的
///   指令计数加上剩余 tick 换算出的指令数。
///
/// 清除只发生在写入未来值或 0 时, ``sync_mtip`` 的活跃分支不清除。
pub(crate) fn write_mtimecmp(state: &mut HartState, now_mtime: u64, val: u64, timebase_hz: u64) {
	if val == 0 {
		state.mip.fetch_and(!(1 << 7), Ordering::AcqRel);
		state.mtip_deadline = 0;
	} else if val <= now_mtime {
		state.mtip_deadline = 0;
		state.mip.fetch_or(1 << 7, Ordering::AcqRel);
	} else {
		state.mtip_deadline = timer_deadline_own(state.total_instrs, now_mtime, val, timebase_hz);
		state.mip.fetch_and(!(1 << 7), Ordering::AcqRel);
	}
}

/// stimecmp 写入处理, 与 ``write_mtimecmp`` 对称。
/// stimecmp 仅由本 hart 写入, 故只取 ``&mut HartState``。
pub(crate) fn write_stimecmp(state: &mut HartState, now_mtime: u64, val: u64, timebase_hz: u64) {
	if val == 0 {
		state.mip.fetch_and(!(1 << 5), Ordering::AcqRel);
		state.stip_deadline = 0;
	} else if val <= now_mtime {
		state.stip_deadline = 0;
		state.mip.fetch_or(1 << 5, Ordering::AcqRel);
	} else {
		state.stip_deadline = timer_deadline_own(state.total_instrs, now_mtime, val, timebase_hz);
		state.mip.fetch_and(!(1 << 5), Ordering::AcqRel);
	}
}

/// 写入 mtimecmp (CLINT MMIO) 后, 为目标 hart 的 MTIP 设定 deadline。
///
/// 目标通常为本 hart,也支持跨 hart 写入: 经 ``hart_states`` 数组取得目标 HartState,
/// 以原始指针加原子对齐 u64 读写, 与 clint_write_msip_concurrent 的跨 hart mip 访问相同。
/// ``hart_states`` 未初始化时跳过设定, MTIP 退化为共享 mtime 比较。
#[inline]
pub(crate) fn write_mtimecmp_for_target(clint: &ConcurrentClintCtx, target: usize, val: u64) {
	let now = unsafe { &*clint.mtime }.load(Ordering::Acquire);
	let hart_states = clint.hart_states.get();
	if !hart_states.is_null() && target < clint.num_harts as usize {
		let ts = unsafe { &mut *(hart_states.add(target) as *mut HartState) };
		write_mtimecmp(ts, now, val, clint.timebase_hz);
	}
}

/// 设置 ``mip``; 已置位时不改变任何值。
///
/// 位已置位时 ``fetch_or`` 不改变任何值, 故跳过与执行等价
#[inline]
fn set_mip_bit(state: &mut HartState, bit: u64) {
	if state.mip.load(Ordering::Acquire) & bit == 0 {
		state.mip.fetch_or(bit, Ordering::AcqRel);
	}
}

/// 清除 ``mip`` 的一位; 该位已清零时不改变任何值。
#[inline]
fn clear_mip_bit(state: &mut HartState, bit: u64) {
	if state.mip.load(Ordering::Acquire) & bit != 0 {
		state.mip.fetch_and(!bit, Ordering::AcqRel);
	}
}

/// 按共享 mtime 与比较值 ``cmp`` 的大小关系设置或清除定时器中断位 ``bit``。
///
/// ``cmp`` 为 0, 即比较值未设定时目标恒为清除, 该分支不读取共享 mtime; mtime 由
/// 全部 hart 以取最大值写入, 是唯一被并发写入的共享行。
#[inline]
fn sync_timer_bit_shared(state: &mut HartState, clint: &ConcurrentClintCtx, bit: u64, cmp: u64) {
	if cmp == 0 {
		clear_mip_bit(state, bit);
		return;
	}
	if unsafe { &*clint.mtime }.load(Ordering::Acquire) >= cmp {
		set_mip_bit(state, bit);
	} else {
		clear_mip_bit(state, bit);
	}
}

pub(crate) fn sync_mtip(state: &mut HartState, clint: &ConcurrentClintCtx) {
	let hart_id = state.mhartid as usize;
	let cmp = unsafe { &*clint.mtimecmp.add(hart_id) }.load(Ordering::Acquire);
	let sstc_cmp = state.stimecmp;

	// ============================================================
	//  MTIP (bit 7): CLINT mtimecmp
	//  ============================================================
	if state.waiting != 0 {
		// 等待中的 hart 按共享 mtime 比较。其他 hart 的进度推进共享 mtime 后本 hart
		// 随之醒来; 受调试程序的 rdtime 读取共享 mtime, 定时器到期即在共享时间中越过
		// mtimecmp。
		sync_timer_bit_shared(state, clint, 1 << 7, cmp);
	} else if state.mtip_deadline != 0 && state.total_instrs >= state.mtip_deadline {
		// 活跃且已设定 deadline: 只有本 hart 的指令计数越过写入时固定的 deadline 才
		// 置位, 其他 hart 推进共享 mtime 不影响此判定。清除只由 ``write_mtimecmp``
		// 在写入未来值或 0 时负责。
		set_mip_bit(state, 1 << 7);
	} else if state.mtip_deadline == 0 {
		// 活跃且 deadline 为 0: 写入未经过 ``write_mtimecmp``, 例如 Python 单步执行
		// 路径写入、测试直接写字段或写入即到期; 此时按共享 mtime 比较。已设定 deadline
		// 的定时器不会进入此分支。
		sync_timer_bit_shared(state, clint, 1 << 7, cmp);
	}

	// ============================================================
	//  STIP (bit 5): SSTC stimecmp
	//  ============================================================
	if state.waiting != 0 {
		// 按共享 mtime 比较, 用于在定时器到期时唤醒本 hart。
		sync_timer_bit_shared(state, clint, 1 << 5, sstc_cmp);
	} else if state.stip_deadline != 0 && state.total_instrs >= state.stip_deadline {
		// 活跃且已设定 deadline，只有本 hart 的指令计数越过 deadline 才置位, 清除只由 ``write_stimecmp`` 负责。
		set_mip_bit(state, 1 << 5);
	} else if state.stip_deadline == 0 {
		// 活跃且 deadline 为 0: Python 单步执行路径或写入即到期, 此时按共享 mtime 比较。
		sync_timer_bit_shared(state, clint, 1 << 5, sstc_cmp);
	}
}

/// 由跨线程的 ``msip_pending`` 原子通道和 CLINT 电平位同步 ``mip.MSIP``。全部模式下
/// 都读取 ``msip_pending``, 非零时把读到的位并入 ``mip``; 旧有 PLIC 模式, 即
/// ``present == 0`` 或 ``eidelivery == 0`` 时, 另采样 CLINT 电平位, 电平为 1 即置位
/// ``mip.MSIP``, 此处从不清除。AIA 模式下本次未读到边沿、CLINT 电平位为 0 且 IMSIC
/// 没有挂起的 IPI 时, 清除残留的 ``mip.MSIP``, 否则 MEI 先于 MSI 投递而
/// ``deliver_trap`` 只清 MEIP, 残留位使 MTOPI 循环反复分发同一 IPI。
#[inline]
pub(crate) fn sync_msip(state: &mut HartState, clint: &ConcurrentClintCtx) {
	let hid = state.mhartid as usize;
	let in_aia_active = state.imsic_m.present != 0 && state.imsic_m.eidelivery != 0;

	// 读取跨线程 MSIP 通道
	let pending_ptr = clint.msip_pending_slot(hid);
	let mut had_edge = false;
	let mut _cross = 0;
	if !pending_ptr.is_null() {
		let slot = unsafe { &*pending_ptr };
		// 无事件时为 0。先普通读取, 为 0 则跳过交换, 交换在槽位为 0 时不改变
		// 任何值。读取与交换之间到达的设置留到下一次调用读取, 延迟一条指令。
		if slot.load(Ordering::Acquire) != 0 {
			_cross = slot.swap(0, Ordering::Acquire);
			if _cross != 0 {
				state.mip.fetch_or(_cross, Ordering::AcqRel);
				had_edge = true;
			}
		}
	}

	// AIA 模式: 边沿触发, 清除残留位
	//
	// AIA 模式下 IMSIC 的 IPI 经 MEIP 与 SEIP 投递, 不经过 MSIP, ``msip_pending``
	// 边沿通道不会被 IPI 操作写入。此处清除旧有操作或直接写 CLINT 留下的
	// ``mip.MSIP``; IMSIC eip 中仍有挂起的 IPI 时不清除, 此时 MEIP 优先级更高,
	// 先于 MSIP 投递。
	if in_aia_active {
		if had_edge || hid >= clint.num_harts as usize {
			return;
		}
		let raw = unsafe { &*clint.msip.add(hid) }.load(Ordering::Acquire);
		let no_imsic_ipi = state.imsic_m.present == 0 || state.imsic_m.eidelivery == 0 || {
			let eip0 = state.imsic_m.eip[0].load(Ordering::Acquire);
			(eip0 & ((1u32 << 3) | (1u32 << 1))) == 0
		};
		if (raw & 1) == 0 && no_imsic_ipi {
			clear_mip_bit(state, 1 << 3);
		}
		return;
	}

	// 旧有 PLIC 模式: CLINT MSIP 为电平触发, 同步到 mip.MSIP
	if hid >= clint.num_harts as usize {
		return;
	}
	// 旧有 PLIC 模式下 mip.MSIP 直接跟随 CLINT 电平位。此处只读, 不清除电平位:
	let raw = unsafe { &*clint.msip.add(hid) }.load(Ordering::Acquire);
	let level_set = (raw & 1) != 0;

	if level_set {
		state.mip.fetch_or(1 << 3, Ordering::AcqRel);
	}
}

// ============================================================

pub(crate) fn clint_write_msip_concurrent(clint: &ConcurrentClintCtx, target: usize, val: u8) {
	if target >= clint.num_harts as usize {
		return;
	}
	let p = unsafe { &*clint.msip.add(target) };
	if val & 1 != 0 {
		// 设置 CLINT 电平位, 并经 ``msip_pending`` 通道原子通知目标 hart。
		// 这样避免对 ``(*states).mip`` 做非原子读改写, 进而与目标 hart 的
		// ``sync_mtip``、``sync_msip`` 形成并发访问。
		p.fetch_or(1, Ordering::Release);
		let pending = clint.msip_pending_slot(target);
		if !pending.is_null() {
			unsafe { &*pending }.fetch_or(1 << 3, Ordering::Release);
		}
		// 唤醒目标 hart 的线程, 它可能正在 ``wfi_spin`` 中经 ``park_timeout`` 睡眠。
		// 在 ``park()`` 之前调用 ``unpark()`` 是安全的, 下一次 ``park()`` 立即返回。
		let threads = clint.hart_threads.get();
		if !threads.is_null() && target < clint.num_harts as usize {
			unsafe { &*threads.add(target) }.unpark();
		}
	} else {
		// 清除 CLINT 电平位, 并清除目标 hart 的 ``mip.MSIP``。
		//
		// ``deliver_trap_mmode`` 只为 MSI (cause 3) 清除 ``mip.MSIP``, 不为 MEI 清除。
		// AIA 模式下同一 IPI 事件中优先级更高的 MEI 先投递, ``mip.MSIP`` 保持置位;
		// ``sbi_ipi_raw_clear`` 在此写入 0 之后必须一并清除该位, 否则
		// ``sbi_trap_aia_irq()`` 的 MTOPI 循环不会终止。
		p.fetch_and(0xFE, Ordering::Release);
		// 发送方已设置 ``msip_pending``, 下一次 ``sync_msip`` 会将其读出并重新置位 ``mip.MSIP``。
		let hart_states = clint.hart_states.get();
		if !hart_states.is_null() {
			// 共享引用, mip 为 AtomicU64, 可安全并发访问。
			let ts = unsafe { &*(hart_states.add(target) as *const HartState) };
			ts.mip.fetch_and(!(1 << 3), Ordering::AcqRel);
		}
	}
}

/// 内联处理一次 CLINT MMIO 访问的并发安全版本。
pub(crate) fn try_handle_clint_concurrent(
	pa: u64,
	is_write: bool,
	write_data: u64,
	_state: &mut HartState,
	clint: &ConcurrentClintCtx,
	_module: &ModuleState,
) -> Option<u64> {
	if clint.base == 0 {
		return None;
	}
	let offset = pa.wrapping_sub(clint.base);

	if offset < 0x4000 {
		// MSIP 区域
		let target = (offset / 4) as usize;
		if target >= clint.num_harts as usize {
			return Some(0);
		}
		if !is_write {
			let val = unsafe { &*clint.msip.add(target) }.load(Ordering::Relaxed) as u64 & 1;
			return Some(val);
		}
		clint_write_msip_concurrent(clint, target, (write_data & 1) as u8);
		return Some(0);
	} else if offset < 0xBFF8 {
		// MTIMECMP 区域
		let target = ((offset - 0x4000) / 8) as usize;
		if target >= clint.num_harts as usize {
			return Some(0);
		}
		if is_write {
			unsafe { &*clint.mtimecmp.add(target) }.store(write_data, Ordering::Release);
			// 写入后为目标 hart 的 MTIP 设定 deadline。
			write_mtimecmp_for_target(clint, target, write_data);
			Some(0)
		} else {
			Some(unsafe { &*clint.mtimecmp.add(target) }.load(Ordering::Relaxed))
		}
	} else if offset < 0xC000 {
		// MTIME 区域
		if is_write {
			None // 应交由 Python 侧处理
		} else {
			Some(unsafe { &*clint.mtime }.load(Ordering::Relaxed))
		}
	} else {
		None
	}
}

pub struct ClintCtx {
	pub base: u64,
	pub mtime: *mut u64,
	pub mtimecmp: *mut u64,
	pub msip: *mut u8,
	pub states: *mut HartState,
	pub num_harts: u32,
	/// mtime 时钟源频率, 单位为 Hz; 为 0 表示未提供时钟源,
	/// 此时不推进时钟。用于 mtimecmp 写入后把 MTIP deadline 换算到指令计数空间。
	pub timebase_hz: u64,
	/// 本 hart 向另一个 hart 写 MSIP=1 时置位。分发循环检查该标志并提前让出当前 hart
	/// 的时间片, 使目标 hart 在同一轮内响应 IPI。
	pub yield_for_ipi: Cell<bool>,
	/// 最近一次跨 hart 发送 MSIP 的 hart 标识。该 hart 获得较短的时间片, 使接收方在
	/// 发送方恢复自旋等待之前完成 IPI 触发的工作, 包括 TLB 刷新与同步计数递减。
	pub ipi_sender_hart: Cell<u8>,
	/// 每次发送新的 MSIP 时重置为初始计数。
	pub ipi_sender_rounds: Cell<u8>,
	/// 指向并发 MSIP 待处理数组的引用,
	/// 在压缩指令路径写入 IMSIC seteipnum 时, 用于唤醒目标 hart。
	pub msip_pending: Cell<*const std::sync::atomic::AtomicU64>,
	/// 指向并发 hart 线程句柄的引用, 用于从 WFI 的 ``park_timeout`` 唤醒时调用
	/// ``unpark()``。由并发加速执行引擎中的 ``exec_compressed_concurrent`` 填充。
	pub hart_threads: Cell<*const std::thread::Thread>,
	/// 指向并发每 hart HartState 数组的引用, 用于跨 hart 修改 IMSIC eip。由并发加速
	/// 执行引擎中的 ``exec_compressed_concurrent`` 填充。
	pub hart_states: Cell<*const crate::state::HartState>,
}

/// 为目标 hart 写 MSIP 并更新其 mip, 在目标为另一个 hart 时设置跨 hart 让出标志。
#[inline]
pub(crate) fn clint_write_msip(clint: &ClintCtx, target: usize, current: usize, val: u8) {
	if target >= clint.num_harts as usize {
		return;
	}
	if clint.states.is_null() {
		unsafe {
			let atomic_msip = clint.msip as *const AtomicU8;
			(*atomic_msip.add(target)).store(val, Ordering::Release);
		}
		return;
	}
	unsafe {
		*clint.msip.add(target) = val;
	}
	let ts = unsafe { &mut *clint.states.add(target) };
	if val == 0 {
		ts.mip.fetch_and(!(1 << 3), Ordering::AcqRel);
		return;
	}
	ts.mip.fetch_or(1 << 3, Ordering::AcqRel);
	if target != current {
		clint.yield_for_ipi.set(true);
		clint.ipi_sender_hart.set(current as u8);
		clint.ipi_sender_rounds.set(16);
	}
}

/// 在加速执行引擎中处理 CLINT MMIO 访问。
pub(crate) fn try_handle_clint(
	access: &MemAccess,
	state: &HartState,
	clint: &ClintCtx,
) -> Option<u64> {
	let pa = access.pa;
	let is_write = access.is_write;
	let write_data = access.write_data;

	if clint.base == 0 {
		return None;
	}
	let offset = pa.wrapping_sub(clint.base);

	if offset < 0x4000 {
		let hart_id = (offset / 4) as usize;
		if hart_id >= clint.num_harts as usize {
			return Some(0);
		}
		if is_write == 0 {
			let val = unsafe { *clint.msip.add(hart_id) } as u64 & 1;
			return Some(val);
		}
		clint_write_msip(
			clint,
			hart_id,
			state.mhartid as usize,
			(write_data & 1) as u8,
		);
		return Some(0);
	} else if offset < 0xBFF8 {
		let hart_id = usize::try_from((offset - 0x4000) / 8).unwrap_or(usize::MAX);
		if hart_id >= clint.num_harts as usize {
			return Some(0);
		}
		if is_write == 0 {
			return Some(unsafe { *clint.mtimecmp.add(hart_id) });
		}
		unsafe {
			*clint.mtimecmp.add(hart_id) = write_data;
		}
		if hart_id < clint.num_harts as usize && !clint.states.is_null() {
			let t = unsafe { &mut *clint.states.add(hart_id) };
			// 写入后为目标 hart 的 MTIP 设定 deadline。
			write_mtimecmp(t, unsafe { *clint.mtime }, write_data, clint.timebase_hz);
		}
		return Some(0);
	} else if offset < 0xC000 && is_write == 0 {
		return Some(unsafe { *clint.mtime });
	}
	None
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::concurrent::MSIP_PENDING_STRIDE;
	use crate::state::HartState;
	use std::sync::atomic::{AtomicU64, AtomicU8};

	/// 相邻 hart 用于暂存 MSIP 待处理请求的内存区域相距 64 字节。
	///
	/// 设置方 ``clint_write_msip_concurrent`` 与读取方 ``sync_msip`` 必须取到同一个
	/// 内存区域; 不一致时 IPI 会写到错误的位置, 导致目标 hart 永远收不到软件中断。
	#[test]
	fn msip_pending_slots_are_distinct_and_addressed_consistently() {
		const NUM_HARTS: usize = 3;
		let slots: Vec<AtomicU64> = (0..NUM_HARTS * MSIP_PENDING_STRIDE)
			.map(|_| AtomicU64::new(0))
			.collect();
		let mtime = AtomicU64::new(0);
		let mtimecmp: Vec<AtomicU64> = (0..NUM_HARTS).map(|_| AtomicU64::new(0)).collect();
		let msip: Vec<AtomicU8> = (0..NUM_HARTS).map(|_| AtomicU8::new(0)).collect();

		let clint = ConcurrentClintCtx {
			base: 0x2000000,
			mtime: &mtime as *const AtomicU64,
			mtimecmp: mtimecmp.as_ptr(),
			msip: msip.as_ptr(),
			num_harts: NUM_HARTS as u32,
			msip_pending: Cell::new(slots.as_ptr()),
			hart_threads: Cell::new(std::ptr::null()),
			hart_states: Cell::new(std::ptr::null()),
			timebase_hz: 0,
		};

		// 相距 64 字节且互不相同。
		let mut seen = Vec::new();
		for hid in 0..NUM_HARTS {
			let addr = clint.msip_pending_slot(hid) as usize;
			let base = slots.as_ptr() as usize;
			assert_eq!(
				addr - base,
				hid * MSIP_PENDING_STRIDE * std::mem::size_of::<AtomicU64>(),
				"hart {hid} 的槽位偏移必须为 hart 索引乘跨步的字节数"
			);
			assert!(
				!seen.contains(&addr),
				"hart {hid} 的槽位与已出现的槽位重复: {addr:#x}"
			);
			seen.push(addr);
		}

		// 向 hart 2 发送 IPI 后, 只有 hart 2 的 mip.MSIP 应被置位。
		clint_write_msip_concurrent(&clint, 2, 1);
		for hid in 0..NUM_HARTS {
			let mut state: HartState = unsafe { std::mem::zeroed() };
			state.mhartid = hid as u64;
			sync_msip(&mut state, &clint);
			let msip_set = state.mip.load(Ordering::Acquire) & (1 << 3);
			if hid == 2 {
				assert_eq!(msip_set, 1 << 3, "hart 2 必须读到发往自己的 IPI 边沿");
			} else {
				assert_eq!(msip_set, 0, "hart {hid} 不得读到发往 hart 2 的 IPI 边沿");
			}
		}
	}

	#[test]
	fn sync_msip_sets_mip_on_level_high() {
		// CLINT MSIP 电平为 1 时, sync_msip 必须置位 mip.MSIP。该位为电平触发,
		// sync_msip 不自动清除电平位, 只有 M 模式处理函数向 CLINT MSIP 写 0 才清除,
		// 与真实硬件一致。msip_pending 原子通道另行提供边沿触发的投递。
		let mut state: HartState = unsafe { std::mem::zeroed() };
		state.mhartid = 0;
		state.mie = 1 << 3;

		let msip_byte = std::sync::atomic::AtomicU8::new(1);
		let mtime = std::sync::atomic::AtomicU64::new(0);
		let mtimecmp = std::sync::atomic::AtomicU64::new(0);

		let clint = ConcurrentClintCtx {
			base: 0x2000000,
			mtime: &mtime as *const AtomicU64,
			mtimecmp: &mtimecmp as *const AtomicU64,
			msip: &msip_byte as *const AtomicU8,
			num_harts: 1,
			msip_pending: Cell::new(std::ptr::null()),
			hart_threads: Cell::new(std::ptr::null()),
			hart_states: Cell::new(std::ptr::null()),
			timebase_hz: 0, // 此处单元测试项不推进时钟源
		};

		assert_eq!(msip_byte.load(Ordering::Relaxed) & 1, 1);
		assert_eq!(state.mip.load(Ordering::Acquire) & (1 << 3), 0);

		sync_msip(&mut state, &clint);

		assert_eq!(
			state.mip.load(Ordering::Acquire) & (1 << 3),
			1 << 3,
			"sync_msip must set mip.MSIP when level=1"
		);
		assert_eq!(
			msip_byte.load(Ordering::Relaxed) & 1,
			1,
			"sync_msip must NOT auto-clear CLINT level bit (level-triggered)"
		);
	}

	#[test]
	fn sync_msip_preserves_mip_when_level_zero() {
		// CLINT 电平为 0 时 sync_msip 不得清除 mip.MSIP, 清除由
		// ``deliver_trap_mmode`` 与 ``deliver_trap_smode`` 负责
		// 若在此清除, WFI 唤醒路径会丢失该中断: ``wfi_sync_and_check`` 调用
		// ``sync_msip`` 时读到电平 0 并清除 ``mip.MSIP``, 随后的
		// ``check_and_deliver_interrupt_concurrent`` 不分发陷态处理逻辑而形成死锁。
		let mut state: HartState = unsafe { std::mem::zeroed() };
		state.mhartid = 0;
		state.mip.store(1 << 3, Ordering::Release);

		let msip_byte = std::sync::atomic::AtomicU8::new(0);
		let mtime = std::sync::atomic::AtomicU64::new(0);
		let mtimecmp = std::sync::atomic::AtomicU64::new(0);

		let clint = ConcurrentClintCtx {
			base: 0x2000000,
			mtime: &mtime as *const AtomicU64,
			mtimecmp: &mtimecmp as *const AtomicU64,
			msip: &msip_byte as *const AtomicU8,
			num_harts: 1,
			msip_pending: Cell::new(std::ptr::null()),
			hart_threads: Cell::new(std::ptr::null()),
			hart_states: Cell::new(std::ptr::null()),
			timebase_hz: 0, // 单元测试不推进时钟源
		};

		sync_msip(&mut state, &clint);

		assert_eq!(
			state.mip.load(Ordering::Acquire) & (1 << 3),
			1 << 3,
			"sync_msip must NOT clear mip.MSIP when CLINT level=0 (trap delivery handles clearing)"
		);
	}

	/// ``mtime`` 等于 ``stimecmp`` 时 STIP 置位, 判定用 ``>=``.
	/// 若改为严格的 ``>``, mtime 被推进到恰好等于 stimecmp 时 STIP 不会置位, 定时发起
	/// 的定时中断无法唤醒 hart.
	#[test]
	fn sync_mtip_sets_stip_at_deadline_boundary() {
		let mut state: HartState = unsafe { std::mem::zeroed() };
		state.mhartid = 0;
		state.stimecmp = 100;

		let mtime = AtomicU64::new(100); // mtime 等于 stimecmp, 恰在截止时刻
		let mtimecmp = AtomicU64::new(0); // MTIP 必须保持清除
		let msip = AtomicU8::new(0);

		let clint = ConcurrentClintCtx {
			base: 0x2000000,
			mtime: &mtime as *const AtomicU64,
			mtimecmp: &mtimecmp as *const AtomicU64,
			msip: &msip as *const AtomicU8,
			num_harts: 1,
			msip_pending: Cell::new(std::ptr::null()),
			hart_threads: Cell::new(std::ptr::null()),
			hart_states: Cell::new(std::ptr::null()),
			timebase_hz: 0,
		};

		sync_mtip(&mut state, &clint);

		assert_eq!(
			state.mip.load(Ordering::Acquire) & (1 << 5),
			1 << 5,
			"sync_mtip must set STIP when mtime == stimecmp (>= comparison)"
		);
		assert_eq!(
			state.mip.load(Ordering::Acquire) & (1 << 7),
			0,
			"MTIP must stay clear when mtimecmp is unarmed (0)"
		);
	}

	/// 共享 mtime 被其他 hart 推进时, 不得提前触发刚重新设置的定时器.
	///
	/// ``write_stimecmp`` 在写入时刻把截止时刻换算到本 hart 的指令计数, 只有本 hart
	/// 执行到该计数才设置 STIP. 若按逐条指令的共享比较 ``cur_mtime >= stimecmp`` 判定,
	/// 其它 hart 经 fetch_max 推进的 mtime 越过刚写入的未来 stimecmp 后, 本 hart 的
	/// ``sync_mtip`` 立即设置 STIP, 导致本 hart 自 mret 返回后立即再次返回中断处理逻辑.
	#[test]
	fn stimecmp_deadline_decouples_from_cross_hart_mtime_inflation() {
		let mut state: HartState = unsafe { std::mem::zeroed() };
		state.mhartid = 0;
		state.total_instrs = 1000;
		state.mie = 1 << 5;

		let mtime = AtomicU64::new(5000);
		let mtimecmp = AtomicU64::new(0);
		let msip = AtomicU8::new(0);

		let clint = ConcurrentClintCtx {
			base: 0x2000000,
			mtime: &mtime as *const AtomicU64,
			mtimecmp: &mtimecmp as *const AtomicU64,
			msip: &msip as *const AtomicU8,
			num_harts: 1,
			msip_pending: Cell::new(std::ptr::null()),
			hart_threads: Cell::new(std::ptr::null()),
			hart_states: Cell::new(std::ptr::null()),
			timebase_hz: 10_000_000, // 10 MHz 时，5 条指令一个 tick
		};

		state.stimecmp = 5100;
		write_stimecmp(&mut state, 5000, 5100, 10_000_000);
		assert_eq!(
			state.stip_deadline,
			1000 + 500,
			"deadline 固定到写入时刻 own 指令计数空间"
		);
		assert_eq!(
			state.mip.load(Ordering::Acquire) & (1 << 5),
			0,
			"未来值写入必须清 STIP"
		);

		mtime.store(5150, Ordering::Release);
		sync_mtip(&mut state, &clint);
		assert_eq!(
			state.mip.load(Ordering::Acquire) & (1 << 5),
			0,
			"共享 mtime 数值被异常推高不得提前触发刚重新设置的 STIP"
		);

		// 本 hart 再执行 500 条指令到达截止时刻, 此时才设置.
		state.total_instrs = 1500;
		sync_mtip(&mut state, &clint);
		assert_eq!(
			state.mip.load(Ordering::Acquire) & (1 << 5),
			1 << 5,
			"own 指令计数越过 deadline 才置位 STIP"
		);
	}

	/// hart A 跨 hart 写 hart B 的 mtimecmp 时, 同样按目标
	/// hart B 的指令计数设定截止时刻, 不因共享 mtime 被 A 推进而提前触发.
	#[test]
	fn mtimecmp_cross_hart_write_arms_target_deadline_not_shared_compare() {
		let mut b: HartState = unsafe { std::mem::zeroed() };
		b.mhartid = 1;
		b.total_instrs = 2000;
		b.mie = 1 << 7;

		let a: HartState = unsafe { std::mem::zeroed() };
		let mut states = [a, b];

		let mtime = AtomicU64::new(7000);
		// mtimecmp 按 hart 标识索引: sync_mtip 为 hart B 读取下标 1 的一项, 与真实
		// CLINT mtimecmp 数组每个 hart 一项的布局一致.
		let mtimecmp = [AtomicU64::new(0), AtomicU64::new(0)];
		let msip = AtomicU8::new(0);

		let clint = ConcurrentClintCtx {
			base: 0x2000000,
			mtime: &mtime as *const AtomicU64,
			mtimecmp: mtimecmp.as_ptr(),
			msip: &msip as *const AtomicU8,
			num_harts: 2,
			msip_pending: Cell::new(std::ptr::null()),
			hart_threads: Cell::new(std::ptr::null()),
			hart_states: Cell::new(states.as_mut_ptr() as *const HartState),
			timebase_hz: 10_000_000,
		};

		// hart A 写 hart B 的 mtimecmp=7200: 距当前共享 mtime=7000 还有 200 ticks, 换算为
		// 1000 条指令. 与真实 CLINT MMIO 写路径一致: 先存 mtimecmp 值, 该值是共享比较的
		// 读取源, 再设定截止时刻. 旧实现读到 mtimecmp=7200 后, 在 mtime=7300 时会提前置位.
		unsafe { &*clint.mtimecmp.add(1) }.store(7200, Ordering::Release);
		write_mtimecmp_for_target(&clint, 1, 7200);
		let b = &mut states[1];
		assert_eq!(
			b.mtip_deadline,
			2000 + 1000,
			"跨 hart 写入按目标 own 指令计数设定 deadline"
		);
		assert_eq!(
			b.mip.load(Ordering::Acquire) & (1 << 7),
			0,
			"未来值写入必须清 MTIP"
		);

		// A 经 fetch_max 把共享 mtime 推进到 7300, 越过 mtimecmp=7200, 即旧实现的触发
		// 条件; B 未推进指令.
		mtime.store(7300, Ordering::Release);
		sync_mtip(b, &clint);
		assert_eq!(
			b.mip.load(Ordering::Acquire) & (1 << 7),
			0,
			"共享 mtime 数值被异常推高不得提前触发跨 hart 刚重新设置的 MTIP"
		);

		// B 再执行 1000 条指令到达截止时刻, 此时才置位.
		b.total_instrs = 3000;
		sync_mtip(b, &clint);
		assert_eq!(
			b.mip.load(Ordering::Acquire) & (1 << 7),
			1 << 7,
			"own 指令计数越过 deadline 才置位 MTIP"
		);
	}

	/// 构造单 hart 的 CLINT 上下文, 共享 mtime 与 mtimecmp 由调用方持有.
	fn clint_with(mtime: &AtomicU64, mtimecmp: &AtomicU64) -> (ConcurrentClintCtx, AtomicU8) {
		let msip = AtomicU8::new(0);
		let clint = ConcurrentClintCtx {
			base: 0x2000000,
			mtime: mtime as *const AtomicU64,
			mtimecmp: mtimecmp as *const AtomicU64,
			msip: &msip as *const AtomicU8,
			num_harts: 1,
			msip_pending: Cell::new(std::ptr::null()),
			hart_threads: Cell::new(std::ptr::null()),
			hart_states: Cell::new(std::ptr::null()),
			timebase_hz: 0,
		};
		(clint, msip)
	}

	/// MTIP 与 STIP 已置位后, 比较值变为不成立时必须在同一次 ``sync_mtip`` 中清除.
	///
	/// ``sync_mtip`` 的置位与清除均先读取该位, 位已处于目标状态时跳过加锁的读改写.
	/// 该跳过只用于不改变任何值的读改写: 若一次本应生效的清除被跳过, MTIP 与 STIP
	/// 保持置位, 受调试程序从定时器处理函数返回后立即再次陷入, 每条指令陷入一次而
	/// 没有进展.
	///
	/// 覆盖两条清除路径: 比较值已设定但未到期; 比较值未设定, 即取值为 0, 该分支不
	/// 读取共享 mtime 而直接清除.
	#[test]
	fn sync_mtip_clears_already_set_timer_bits_when_condition_clears() {
		let mut state: HartState = unsafe { std::mem::zeroed() };
		state.mhartid = 0;
		state.mip.fetch_or((1 << 7) | (1 << 5), Ordering::AcqRel);

		let mtime = AtomicU64::new(100);
		let mtimecmp = AtomicU64::new(200);
		let (clint, _msip) = clint_with(&mtime, &mtimecmp);

		// 比较值已设定为 200, 共享 mtime 为 100 未越过比较值: 两位必须清除.
		state.stimecmp = 200;
		state.waiting = 0;
		sync_mtip(&mut state, &clint);
		assert_eq!(
			state.mip.load(Ordering::Acquire) & ((1 << 7) | (1 << 5)),
			0,
			"比较值未到期时必须清除已置位的 MTIP/STIP"
		);

		// 越过比较值: 两位必须置位.
		mtime.store(200, Ordering::Release);
		sync_mtip(&mut state, &clint);
		assert_eq!(
			state.mip.load(Ordering::Acquire) & ((1 << 7) | (1 << 5)),
			(1 << 7) | (1 << 5),
			"越过比较值时必须置位 MTIP/STIP"
		);

		// 比较值未设定, 即取值为 0: 两位必须清除, 且该路径不读取共享 mtime.
		state.stimecmp = 0;
		mtimecmp.store(0, Ordering::Release);
		sync_mtip(&mut state, &clint);
		assert_eq!(
			state.mip.load(Ordering::Acquire) & ((1 << 7) | (1 << 5)),
			0,
			"比较值未设定时必须清除已置位的 MTIP/STIP"
		);
	}
}
