pub(crate) mod clint;
pub(crate) mod imsic;
pub(crate) mod wfi;

use crate::concurrent::ConcurrentClintCtx;
use crate::ffi::FfiExtIrqCtx;
use crate::state::{riscv_mode, HartState};
use std::sync::atomic::Ordering;

use crate::trap::{deliver_trap, deliver_trap_mmode, mcause_val};
// ============================================================
//  mip 与 mie 的位掩码, 供 check_pending_interrupts 使用
// ============================================================

const MIE_MEIE: u64 = 1 << 11; // M 模式外部中断
const MIE_SEIE: u64 = 1 << 9; // S 模式外部中断
const MIE_MTIE: u64 = 1 << 7; // M 模式定时器中断
const MIE_STIE: u64 = 1 << 5; // S 模式定时器中断
const MIE_MSIE: u64 = 1 << 3; // M 模式软件中断
const MIE_SSIE: u64 = 1 << 1; // S 模式软件中断

pub(crate) use imsic::{
	decode_imsic_addr, imsic_clear_ipi_on_trap, imsic_reg_read, imsic_reg_write,
	imsic_topei_claim_iid, imsic_topei_peek, sync_imsic, sync_imsic_one,
	try_handle_imsic_concurrent, try_handle_imsic_read_concurrent, IID_M_IPI, IID_S_IPI,
};

pub(crate) fn check_pending_interrupts(state: &HartState) -> Option<(u64, bool)> {
	let pending = state.mip.load(Ordering::Acquire) & state.mie;
	if pending == 0 {
		return None;
	}

	// 优先级顺序: MEI、MSI、MTI、SEI、SSI、STI。
	//
	// mideleg 对应位为 1 的中断委派到 S 模式: 仅当 S 级全局中断已打开时投递,
	// 否则保持挂起. 未委派的中断在 M 模式投递. M 模式下委派中断一律保持挂起.
	let checks: [(u64, u64); 6] = [
		(MIE_MEIE, 11),
		(MIE_MSIE, 3),
		(MIE_MTIE, 7),
		(MIE_SEIE, 9),
		(MIE_SSIE, 1),
		(MIE_STIE, 5),
	];

	// M 模式下 mstatus.MIE 为 0 时, 全局中断关闭, 不再判定其余中断.
	if state.mode == riscv_mode::M && state.mstatus & (1 << 3) == 0 {
		return None;
	}

	// S 级全局中断使能: U 模式下 S 级中断可直接投递, S 模式下要求 mstatus.SIE 为 1.
	let s_mode_global = state.mode < riscv_mode::S
		|| (state.mode == riscv_mode::S && state.mstatus & (1 << 1) != 0);

	for (mask, cause) in &checks {
		if pending & mask == 0 {
			continue;
		}
		let delegated = state.mideleg & mask != 0;
		// 委派中断在 S 级全局中断关闭时保持挂起, 既不在 M 模式投递, 也不在 S 模式投递.
		// M 模式下 S 级全局中断恒为关闭, 委派中断因此在该模式保持挂起.
		if delegated && !s_mode_global {
			continue;
		}
		// 委派中断投递到 S 模式, 未委派的中断投递到 M 模式
		return Some((*cause, !delegated));
	}
	None
}

/// 计算 AIA 路径的 MTOPI 值.
///
/// 返回 ``(value, 0)``, value 为 ``(IID << 16) | priority`` 或 0. 优先级依次为 MEI、
/// MSI、MTI, 对应 IID 11、3、7. 外部中断取自 IMSIC M 文件的 ``imsic_topei_peek``,
/// 不使用 ``mip.MEIP``, 该位要到下一条指令边界执行 ``sync_imsic`` 才更新. 返回
/// IID 3 时清除 ``mip.MSIP``, 使下一次读取 MTOPI 得不到同一 IPI.
pub(crate) fn compute_mtopi(state: &mut HartState, _mtime: u64) -> (u64, u8) {
	let mip_mie = state.mip.load(Ordering::Acquire) & state.mie;

	// 1. IMSIC M 文件有挂起的中断时返回 IID=11, 即 MEI; 此处不区分 IPI 与外部中断,
	//    具体原因码在读取 MTOPEI 时给出.
	let (raw, _) = imsic_topei_peek(&state.imsic_m);
	if raw != 0 {
		// IPI 与外部中断在此都返回 IID=11, 即 MEI.
		let prio = raw & 0xFF;
		return ((11u64 << 16) | prio, 0);
	}
	// 2. MSIP 对应 IID=3, 即 IRQ_M_SOFT.
	if (mip_mie & (1 << 3)) != 0 {
		let val = 3u64 << 16 | 1;
		// 读取 MTOPI 返回 IID=3 即认为该 IPI 已得到响应, 随即清除 mip.MSIP;
		// 不清除时下一次读取 MTOPI 仍返回同一 IPI.
		state.mip.fetch_and(!(1 << 3), Ordering::AcqRel);
		return (val, 0);
	}
	// 3. MTIP 对应 IID=7, 即 IRQ_M_TIMER.
	if (mip_mie & (1 << 7)) != 0 {
		let val = 7u64 << 16 | 1;
		return (val, 0);
	}
	(0, 0)
}

/// 计算 AIA 路径的 STOPI 值.
///
/// 返回 ``(value, 0)``, value 为 ``(IID << 16) | priority`` 或 0. 优先级依次为 SEI、
/// SSI、STI, 对应 IID 9、1、5. 外部中断取自 IMSIC S 文件的 ``imsic_topei_peek``.
pub(crate) fn compute_stopi(state: &mut HartState, mtime: u64) -> (u64, u8) {
	let mip_mie = state.mip.load(Ordering::Acquire) & state.mie;

	// 1. IMSIC S 文件有挂起的中断时返回 IID=9, 即 SEI; 此处不区分 IPI 与外部中断,
	//    具体原因码在读取 STOPEI 时给出.
	let (raw, _) = imsic_topei_peek(&state.imsic_s);
	if raw != 0 {
		// IPI 与外部中断在此都返回 IID=9, 即 SEI.
		let prio = raw & 0xFF;
		return ((9u64 << 16) | prio, 0);
	}
	// 2. SSIP 对应 IID=1, 即 IRQ_S_SOFT.
	if (mip_mie & (1 << 1)) != 0 {
		let val = 1u64 << 16 | 1;
		// 读取 STOPI 返回 IID=1 即认为该 IPI 已得到响应, 随即清除 mip.SSIP;
		// 不清除时下一次读取 STOPI 仍返回同一 IPI. 该位也可由 sip 写入置位.
		state.mip.fetch_and(!(1 << 1), Ordering::AcqRel);
		return (val, 0);
	}
	// 3. STIP 对应 IID=5, 即 IRQ_S_TIMER.
	//
	// 1. stip_deadline 非 0 时按本 hart 的指令计数判定， 其他 hart
	//    推进共享 mtime 越过刚写入的 stimecmp 后, mip.STIP 重新置位.
	// 2. stip_deadline 为 0 时按共享 mtime 比较.
	// 3. 陷态处理函数清除 mip.STIP, 并在 stimecmp 不晚于当前 mtime 时把它上调为
	//    当前 mtime 加 4, 因此再次读取 STOPI 得到 0.
	let sti_pending = if state.stip_deadline != 0 {
		state.total_instrs >= state.stip_deadline
	} else {
		state.stimecmp > 0 && mtime >= state.stimecmp
	};
	if (state.mie & (1 << 5)) != 0 && sti_pending {
		let val = 5u64 << 16 | 1;
		return (val, 0);
	}
	(0, 0)
}

pub(crate) fn check_and_deliver_interrupt_concurrent(
	state: &mut HartState,
	clint: &ConcurrentClintCtx,
) -> bool {
	if let Some((cause, is_m_mode)) = check_pending_interrupts(state) {
		let code = mcause_val(cause, true);
		if is_m_mode {
			deliver_trap_mmode(state, code, 0);
		} else {
			deliver_trap(state, code, 0);
		}
		/*
			MSIP 是 CLINT 输出的电平中断. 投递陷态时只清 mip.MSIP, CLINT 的电平位保持为 1,
			而该位是 sync_msip 在每条指令边界重新采样的来源: 电平仍为 1 时 mip.MSIP 重新置位,
			同一 IPI 被再次投递, 接收方为同一次中断反复执行处理函数, 依赖它的同步流程无法
			完成. 只清电平位: 并发到达的 IPI 由边沿计数记录, 在下一个指令边界重新置位 mip.MSIP.
		*/
		if cause == 3 {
			let hid = state.mhartid as usize;
			if hid < clint.num_harts as usize {
				// 只清 bit 0 的电平位, 保留 bits 7:1 的边沿计数; ``store(0)`` 会
				// 一并清除边沿计数, 故用 ``fetch_and(0xFE)``.
				unsafe { &*clint.msip.add(hid) }.fetch_and(0xFE, Ordering::Release);
			}
		}
		true
	} else {
		false
	}
}

/// 按共享外部中断标志 ``ext_irq.pending`` 更新 ``mip.MEIP`` 与 ``mip.SEIP``, 返回该标志是否置位.
///
/// 线路由 IMSIC 文件负责时, 即 ``present`` 不为 0 且 ``eidelivery`` 不为 0, 对应 mip
/// 位由文件的 eip 派生: ``eip_ext_any`` 非零时调用 ``sync_imsic_one``, 为零时保持
/// mip 不变, 清除由 ``step_interrupts`` 执行. 线路不由 IMSIC 负责时在此直接置位.
#[inline]
pub(crate) fn sync_ext_irq_mip(state: &mut HartState, ext_irq: *mut FfiExtIrqCtx) -> bool {
	if ext_irq.is_null() || unsafe { (*ext_irq).pending == 0 } {
		return false;
	}
	let mfile_owns = state.imsic_m.present != 0 && state.imsic_m.eidelivery != 0;
	let sfile_owns = state.imsic_s.present != 0 && state.imsic_s.eidelivery != 0;
	if sfile_owns {
		if state.imsic_s.eip_ext_any.load(Ordering::Relaxed) != 0 {
			sync_imsic_one(state, false);
		}
	} else {
		state.mip.fetch_or(1 << 9, Ordering::AcqRel); // SEIP
	}
	if mfile_owns {
		if state.imsic_m.eip_ext_any.load(Ordering::Relaxed) != 0 {
			sync_imsic_one(state, true);
		}
	} else {
		state.mip.fetch_or(1 << 11, Ordering::AcqRel); // MEIP
	}
	true
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::concurrent::ConcurrentClintCtx;
	use crate::ffi::FfiExtIrqCtx;
	use crate::state::{riscv_mode, HartState};
	use std::cell::Cell;
	use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};

	fn clint_with_msip(level: u8) -> (ConcurrentClintCtx, *const AtomicU8) {
		let msip = Box::new(AtomicU8::new(level));
		let msip_ptr: *const AtomicU8 = &*msip;
		let mtime = Box::new(AtomicU64::new(0));
		let mtimecmp = Box::new(AtomicU64::new(0));
		let clint = ConcurrentClintCtx {
			base: 0x2000000,
			mtime: &*mtime as *const AtomicU64,
			mtimecmp: &*mtimecmp as *const AtomicU64,
			msip: msip_ptr,
			timebase_hz: 0,
			num_harts: 1,
			msip_pending: Cell::new(std::ptr::null()),
			hart_threads: Cell::new(std::ptr::null()),
			hart_states: Cell::new(std::ptr::null()),
		};
		// Box 在此作用域结束时释放存储, 而 ``ConcurrentClintCtx`` 只保存原始指针.
		// ``std::mem::forget`` 放弃所有权, 存储不再被释放, 指针在测试期间保持有效.
		std::mem::forget(msip);
		std::mem::forget(mtime);
		std::mem::forget(mtimecmp);
		(clint, msip_ptr)
	}

	/// MSIP 投递后必须清除 CLINT 电平位, 不能只清 ``mip.MSIP``.
	/// 电平位保持为 1 时, ``sync_msip`` 在下一个指令边界重新置位该中断, 同一 IPI 被
	/// 重复投递. 接收方为同一次中断反复执行处理函数, 依赖它的同步流程无法完成.
	#[test]
	fn msip_delivery_clears_clint_level_bit() {
		let mut state: HartState = unsafe { std::mem::zeroed() };
		state.mhartid = 0;
		state.mode = riscv_mode::M;
		state.mstatus = 1 << 3; // MIE 置位
		state.mie = 1 << 3; // MSIE 置位
		state.mtvec = 0x80000000;
		state.pc = 0x1000;
		state.mip.store(1 << 3, Ordering::Release); // MSIP 挂起

		let (clint, msip_ptr) = clint_with_msip(1);
		assert_eq!(
			unsafe { &*msip_ptr }.load(Ordering::Acquire) & 1,
			1,
			"前置: CLINT 电平位应为 1"
		);

		let delivered = check_and_deliver_interrupt_concurrent(&mut state, &clint);

		assert!(delivered, "MSIP 应被投递");
		assert_eq!(
			unsafe { &*msip_ptr }.load(Ordering::Acquire) & 1,
			0,
			"MSIP 投递后 CLINT 电平位必须清零 (否则 sync_msip 重新断言)"
		);
		assert_eq!(
			state.mip.load(Ordering::Acquire) & (1 << 3),
			0,
			"MSIP 投递后 mip.MSIP 必须清零"
		);
		assert_eq!(state.mode, riscv_mode::M, "应保持 M 模式");
		assert_eq!(state.pc, 0x80000000, "应跳转 mtvec");
	}

	/// 投递 MSIP 后清除电平位时必须保留 bits 7:1 的边沿计数.
	/// 该字节由 Python 侧的 ``_native_marshal_clint`` 编码为 ``level | (edge << 1)``,
	/// 边沿计数用于在加速执行边界之间检测 MSIP 变化. ``store(0)`` 会一并清除
	/// 边沿计数, 因此改用 ``fetch_and(0xFE)``, 与 ``clint_write_msip_concurrent``
	/// 写 0 时的处理一致.
	#[test]
	fn msip_delivery_preserves_edge_counter() {
		let mut state: HartState = unsafe { std::mem::zeroed() };
		state.mhartid = 0;
		state.mode = riscv_mode::M;
		state.mstatus = 1 << 3; // MIE 置位
		state.mie = 1 << 3; // MSIE 置位
		state.mtvec = 0x80000000;
		state.pc = 0x1000;
		state.mip.store(1 << 3, Ordering::Release); // MSIP 挂起

		// 字节编码为 ``level | (edge << 1)``, 取 level 为 1、edge 为 3 时得到 0b111.
		let (clint, msip_ptr) = clint_with_msip(0b111);
		assert_eq!(
			unsafe { &*msip_ptr }.load(Ordering::Acquire),
			0b111,
			"前置: 字节应含 level=1 + edge=3"
		);

		let delivered = check_and_deliver_interrupt_concurrent(&mut state, &clint);

		assert!(delivered, "MSIP 应被投递");
		let raw = unsafe { &*msip_ptr }.load(Ordering::Acquire);
		assert_eq!(raw & 1, 0, "投递后电平位 (bit 0) 必须清零");
		assert_eq!(
			raw >> 1,
			3,
			"投递后 edge counter (bits 7:1) 必须保留, 否则 Python 侧跨加速边界丢失 MSIP 跳变检测"
		);
	}

	/// mideleg 的 bit 7 置位时, MTI 在 S 模式投递, 不投递到 M 模式: hart 处于 S 模式
	/// 时, 陷态使用 ``stvec`` 与 ``scause``, 现场保存到 ``sepc``, 原因码为 7.
	/// 若改投 M 模式, 本应在 S 模式完成的中断处理被跳过, mret 返回后 MTIP 仍然挂起,
	/// 同一中断立即再次投递.
	#[test]
	fn mti_delegated_to_s_when_mideleg_set() {
		let mut state: HartState = unsafe { std::mem::zeroed() };
		state.mhartid = 0;
		state.mode = riscv_mode::S;
		state.mstatus = 1 << 1; // SIE 置位
		state.mie = 1 << 7; // MTIE 置位
		state.mideleg = 1 << 7; // 委派 MTI, 在 S 模式投递
		state.mtvec = 0x80000000;
		state.stvec = 0x80004000;
		state.pc = 0x1000;
		state.mip.store(1 << 7, Ordering::Release); // MTIP 挂起

		let (clint, _) = clint_with_msip(0);

		let delivered = check_and_deliver_interrupt_concurrent(&mut state, &clint);

		assert!(delivered, "MTI 应被投递");
		assert_eq!(
			state.mode,
			riscv_mode::S,
			"委派中断应投递到 S 模式而非 M 模式 (修复前硬编码投递 M -> 死循环)"
		);
		assert_eq!(state.pc, 0x80004000, "应跳转 stvec (S 模式 handler)");
		assert_eq!(state.scause, mcause_val(7, true), "scause 应为 STI");
		assert_eq!(state.sepc, 0x1000, "sepc 应保存中断前 PC");
	}

	/// mideleg 的 bit 7 置位的 MTI 在 M 模式保持挂起, 不投递到 M 模式;
	/// hart 进入 S 或 U 模式且 S 级全局中断使能后才投递.
	#[test]
	fn delegated_mti_stays_pending_in_mmode() {
		let mut state: HartState = unsafe { std::mem::zeroed() };
		state.mhartid = 0;
		state.mode = riscv_mode::M;
		state.mstatus = 1 << 3; // MIE 置位
		state.mie = 1 << 7; // MTIE 置位
		state.mideleg = 1 << 7; // 委派 MTI, 在 S 模式投递
		state.mtvec = 0x80000000;
		state.pc = 0x1000;
		state.mip.store(1 << 7, Ordering::Release); // MTIP 挂起

		let (clint, _) = clint_with_msip(0);

		let delivered = check_and_deliver_interrupt_concurrent(&mut state, &clint);

		assert!(!delivered, "委派中断在 M 模式应保持挂起, 等 hart 降到 S/U");
		assert_eq!(state.pc, 0x1000, "不应跳转 (未投递)");
		assert_eq!(state.mode, riscv_mode::M, "应保持 M 模式");
	}

	/// ``ext_irq`` 的处理只为不由 IMSIC 文件负责的线路就地置位 MEIP 与 SEIP.
	/// ``present`` 为 0 或 ``eidelivery`` 为 0 时线路不由 IMSIC 负责, 必须置位;
	/// ``eidelivery`` 不为 0 时不得置位, 清除由 ``step_interrupts`` 执行.
	#[test]
	fn ext_irq_drain_respects_imsic_ownership() {
		let mut ext = FfiExtIrqCtx {
			pending: 1,
			sources: 0,
			max_priority: 0,
			_pad: [0; 2],
		};
		let ext_ptr: *mut FfiExtIrqCtx = &mut ext;

		// 线路不由 IMSIC 负责: 在此就地置位 MEIP 与 SEIP.
		let mut legacy: HartState = unsafe { std::mem::zeroed() };
		legacy.imsic_m.present = 0;
		legacy.imsic_s.present = 0;
		legacy.mip.store(0, Ordering::Release);
		assert!(
			sync_ext_irq_mip(&mut legacy, ext_ptr),
			"legacy 模式 ext_irq 应被消费"
		);
		assert_ne!(
			legacy.mip.load(Ordering::Acquire) & (1 << 11),
			0,
			"legacy 模式 ext_irq 必须内联置位 MEIP"
		);
		assert_ne!(
			legacy.mip.load(Ordering::Acquire) & (1 << 9),
			0,
			"legacy 模式 ext_irq 必须内联置位 SEIP"
		);

		// AIA 模式且 eidelivery 为 1 时, MEIP 与 SEIP 由 IMSIC 文件的 eip 派生,
		// 此处不得置位.
		let mut aia: HartState = unsafe { std::mem::zeroed() };
		aia.imsic_m.present = 1;
		aia.imsic_m.eidelivery = 1;
		aia.imsic_s.present = 1;
		aia.imsic_s.eidelivery = 1;
		aia.mip.store(0, Ordering::Release);
		assert!(
			sync_ext_irq_mip(&mut aia, ext_ptr),
			"AIA 模式 ext_irq 应被消费"
		);
		assert_eq!(
			aia.mip.load(Ordering::Acquire) & ((1 << 11) | (1 << 9)),
			0,
			"eidelivery=1 时 ext_irq 不得内联置位 MEIP/SEIP (IMSIC 独占该线路)"
		);

		// IMSIC 存在但 eidelivery 为 0 时, 线路不由 IMSIC 负责, 在此就地置位 SEIP.
		let mut aia_off: HartState = unsafe { std::mem::zeroed() };
		aia_off.imsic_m.present = 1;
		aia_off.imsic_m.eidelivery = 0;
		aia_off.imsic_s.present = 1;
		aia_off.imsic_s.eidelivery = 0;
		aia_off.mip.store(0, Ordering::Release);
		assert!(
			sync_ext_irq_mip(&mut aia_off, ext_ptr),
			"eidelivery=0 时 ext_irq 应被消费"
		);
		assert_ne!(
			aia_off.mip.load(Ordering::Acquire) & (1 << 9),
			0,
			"eidelivery=0 时 ext_irq 应内联置位 SEIP (回退 legacy 线路)"
		);
	}

	/// ``eidelivery`` 不为 0 时, 外部中断线路由 IMSIC 文件负责. 设备线程在单轮加速
	/// 执行中注入的 eip 必须在本轮内就地置位 SEIP; 否则引擎读到的 eip 只是本轮开始
	/// 时 Python 侧传入的快照, 此后注入的 eip 不会反映到 mip, hart 从 WFI 醒来后
	/// 读不到中断并回到空闲循环, 直到本轮结束、全部 hart 重新进入等待并在轮边界执行
	/// ``_native_sync_plic_mip`` 之后才投递.
	#[test]
	fn ext_irq_drain_delivers_live_imsic_eip_inline() {
		const EXT_IID: u32 = 10;
		let mut ext = FfiExtIrqCtx {
			pending: 1,
			sources: 0,
			max_priority: 0,
			_pad: [0; 2],
		};
		let ext_ptr: *mut FfiExtIrqCtx = &mut ext;

		let mut state: HartState = unsafe { std::mem::zeroed() };
		state.imsic_s.present = 1;
		state.imsic_s.eidelivery = 1;
		state.mip.store(0, Ordering::Release);

		// 设备线程刚注入一条外部 MSI: eip 与 eie 的对应位同时置位, eie 此前由受调试程序使能.
		state.imsic_s.eip[0].store(1 << EXT_IID, Ordering::Release);
		state.imsic_s.eie[0].store(1 << EXT_IID, Ordering::Release);
		state.imsic_s.eip_ext_any.store(1, Ordering::Release);

		assert!(
			sync_ext_irq_mip(&mut state, ext_ptr),
			"AIA 模式 ext_irq 应被消费"
		);
		assert_ne!(
			state.mip.load(Ordering::Acquire) & (1 << 9),
			0,
			"线路由 IMSIC S 文件负责且 eip 非空时必须置位 SEIP, \
			 否则外部中断要到本轮加速执行结束才投递"
		);

		// 受调试程序读取 ``stopei`` 响应该中断后, IMSIC 已无挂起的中断.
		imsic_topei_claim_iid(&mut state.imsic_s, EXT_IID);
		assert_eq!(
			state.imsic_s.eip_ext_any.load(Ordering::Acquire),
			0,
			"claim 之后 eip_ext_any 缓存必须归零"
		);

		// eip_ext_any 归零后, step_interrupts 在每条指令上的同步据以清除 SEIP.
		sync_imsic(&mut state);
		assert_eq!(
			state.mip.load(Ordering::Acquire) & (1 << 9),
			0,
			"claim 之后 SEIP 必须清除, 否则该 hart 反复进入 SEI 陷态处理而无中断可响应"
		);
	}

	/// ``eip`` 全部为 0 时, ``ext_irq`` 的处理不得置位 ``eidelivery`` 不为 0 的线路对应的
	/// mip 位. ``eip_ext_any`` 为 0 时跳过 ``sync_imsic_one``, 该位保持不变, 清除由
	/// ``step_interrupts`` 执行.
	#[test]
	fn ext_irq_drain_leaves_mip_alone_when_imsic_eip_empty() {
		let mut ext = FfiExtIrqCtx {
			pending: 1,
			sources: 0,
			max_priority: 0,
			_pad: [0; 2],
		};
		let ext_ptr: *mut FfiExtIrqCtx = &mut ext;

		let mut state: HartState = unsafe { std::mem::zeroed() };
		state.imsic_s.present = 1;
		state.imsic_s.eidelivery = 1;
		// 模拟其他路径已置位 SEIP 且受调试程序尚未响应该中断, 例如 WFI 唤醒路径.
		state.mip.store(1 << 9, Ordering::Release);

		assert!(sync_ext_irq_mip(&mut state, ext_ptr));
		assert_ne!(
			state.mip.load(Ordering::Acquire) & (1 << 9),
			0,
			"eip 为空时 ext_irq 排空不得清除 SEIP (清除由 step_interrupts 负责)"
		);
	}
}
