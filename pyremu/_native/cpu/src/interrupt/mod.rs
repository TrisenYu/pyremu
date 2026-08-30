pub(crate) mod clint;
pub(crate) mod imsic;
pub(crate) mod wfi;

use crate::concurrent::{ConcurrentClintCtx, FfiExtIrqCtx};
use crate::state::{riscv_mode, HartState};
use std::sync::atomic::Ordering;

use crate::trap::{deliver_trap, deliver_trap_mmode, mcause_val};
// ============================================================
//  mip / mie bit masks — used by check_pending_interrupts
// ============================================================

const MIE_MEIE: u64 = 1 << 11; // M-mode external
const MIE_SEIE: u64 = 1 << 9; // S-mode external
const MIE_MTIE: u64 = 1 << 7; // M-mode timer
const MIE_STIE: u64 = 1 << 5; // S-mode timer
const MIE_MSIE: u64 = 1 << 3; // M-mode software
const MIE_SSIE: u64 = 1 << 1; // S-mode software

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

	// Priority order: MEI, MSI, MTI, SEI, SSI, STI.
	//
	// 委派完全遵循 mideleg (与 Python check_pending_interrupts 一致):
	// 任一中断只要 mideleg 对应位置位, 就按 S 级中断处理 —
	//    S/U 模式 + S 级全局使能 → 投递到 S 模式
	//    M 模式 → 保持挂起 (委派中断永不投递到 M 模式, 等 hart 降到 S/U)
	// mideleg 位清零的中断 (如 OpenSBI 复位默认 0) 一律投递到 M 模式.
	//
	// 早期实现将 MEI/MSI/MTI 硬编码为不可委派 ("M-mode first" 模型,
	// 给 OpenSBI 的 M 模式定时器处理 + 软件注入 STI 用). 但这不符合
	// RISC-V 规范 §3.1.9 — mideleg[7] 置位时 MTI 必须能直接委派为 STI
	// 投递到 S 模式. 硬编码导致裸核内核 (设 mideleg=0x80 期待 MTI→STI)
	// 在 native 模式下 MTI 永远进 M 模式: m_trap_handler skip+4 → mret →
	// MTIP 仍悬置 → 死循环. 改为按 mideleg 动态判定.
	let checks: [(u64, u64); 6] = [
		(MIE_MEIE, 11),
		(MIE_MSIE, 3),
		(MIE_MTIE, 7),
		(MIE_SEIE, 9),
		(MIE_SSIE, 1),
		(MIE_STIE, 5),
	];

	// M 模式 + mstatus.MIE=0 -> 全局关中断, 不响应任何中断
	if state.mode == riscv_mode::M && state.mstatus & (1 << 3) == 0 {
		return None;
	}

	// S 级全局中断使能: U 模式恒真 (较低特权级恒可被打断), S 模式要求 SIE=1
	let s_mode_global = state.mode < riscv_mode::S
		|| (state.mode == riscv_mode::S && state.mstatus & (1 << 1) != 0);

	for (mask, cause) in &checks {
		if pending & mask == 0 {
			continue;
		}
		let delegated = state.mideleg & mask != 0;
		// 委派中断在 S 级全局关闭时保持挂起 (不投递到 M, 也不投递到 S);
		// 这同时覆盖 M 模式 (s_mode_global=False) — 委派中断在 M 模式保持挂起.
		if delegated && !s_mode_global {
			continue;
		}
		// is_m_mode = !delegated: 委派中断投递到 S 模式, 否则投递到 M 模式
		return Some((*cause, !delegated));
	}
	None
}

/// Compute the Machine Top Interrupt (mtopi) value for the AIA path.
///
/// Returns ``(value, 0)`` where *value* is ``(IID << 16) | priority`` or 0.
///
/// Priority order: MEI(11) > MSI(3) > MTI(7).
/// Only M-mode IIDs are reported because ``sbi_trap_aia_irq()`` only
/// dispatches IRQ_M_EXT(11), IRQ_M_SOFT(3), IRQ_M_TIMER(7).
/// S-level interrupts (SEI/SSI/STI) must be delegated via mideleg and
/// handled in S-mode — they are not reported here.
///
/// External interrupts (MEIP) come exclusively from the IMSIC M-file
/// via ``imsic_topei_peek``.  We do NOT consult ``mip.MEIP`` directly
/// because after ``csr_swap(MTOPEI)`` claims the eip bit, ``mip.MEIP``
/// is stale until the next ``sync_imsic()`` instruction-boundary call.
///
/// When returning IID=3 (MSIP), ``mip.MSIP`` is cleared here.  In AIA
/// mode with ``imsic_ipi_device`` as the primary IPI device,
/// ``sbi_ipi_raw_clear(false)`` is a no-op (no ipi_clear callback).
/// Clearing MSIP here prevents the MTOPI loop in ``sbi_trap_aia_irq()``
/// from re-dispatching the same IPI forever.
pub(crate) fn compute_mtopi(state: &mut HartState, _mtime: u64) -> (u64, u8) {
	let mip_mie = state.mip.load(Ordering::Acquire) & state.mie;

	// 1. IMSIC M-file: ALL interrupts (IPI + external) → IID=11 (MEI major
	//    identity).  The IMSIC IPI is delivered via MEIP exactly like an
	//    external interrupt — its minor identity (1) is NOT a major identity.
	//    Per QEMU's riscv_imsic_update, every pending IMSIC interrupt raises
	//    the MEIP line, and read_mtopi returns the mip bit number (11=MEI).
	//    The minor identity is revealed only via MTOPEI.
	let (raw, _) = imsic_topei_peek(&state.imsic_m);
	if raw != 0 {
		// External interrupt (IPI minor 1 included): return IID=11 (MEI).
		let prio = raw & 0xFF;
		return ((11u64 << 16) | prio, 0);
	}
	// 2. MSIP → IID=3 (IRQ_M_SOFT)
	if (mip_mie & (1 << 3)) != 0 {
		let val = 3u64 << 16 | 1;
		// Clear MSIP now — sbi_ipi_raw_clear(false) is a no-op in AIA
		// mode (imsic_ipi_device has no ipi_clear callback).
		state.mip.fetch_and(!(1 << 3), Ordering::AcqRel);
		return (val, 0);
	}
	// 3. MTIP → IID=7 (IRQ_M_TIMER)
	if (mip_mie & (1 << 7)) != 0 {
		let val = 7u64 << 16 | 1;
		return (val, 0);
	}
	(0, 0)
}

/// Compute the Supervisor Top Interrupt (stopi) value for the AIA path.
///
/// Returns ``(value, 0)`` where *value* is ``(IID << 16) | priority`` or 0.
///
/// Priority order: SEI(9) > SSI(1) > STI(5).
/// Only S-mode IIDs are reported.  The kernel reads ``stopi`` (0xDB0) from
/// S-mode in ``riscv_intc_aia_irq()`` to dispatch all pending S-level
/// interrupts without trapping to M-mode.
///
/// External interrupts (SEIP) come exclusively from the IMSIC S-file
/// via ``imsic_topei_peek``.
pub(crate) fn compute_stopi(state: &mut HartState, mtime: u64) -> (u64, u8) {
	let mip_mie = state.mip.load(Ordering::Acquire) & state.mie;

	// 1. IMSIC S-file: ALL interrupts (IPI + external) → IID=9 (SEI major
	//    identity).  The IMSIC IPI is delivered via SEIP exactly like an
	//    external interrupt — its minor identity (1) is NOT a major identity.
	//    Per QEMU's riscv_imsic_update, every pending IMSIC interrupt raises
	//    the SEIP line, and read_stopi returns the mip bit number (9=SEI).
	//    The minor identity is revealed only via STOPEI.
	let (raw, _) = imsic_topei_peek(&state.imsic_s);
	if raw != 0 {
		// External interrupt (IPI minor 1 included): return IID=9 (SEI).
		let prio = raw & 0xFF;
		return ((9u64 << 16) | prio, 0);
	}
	// 2. SSIP → IID=1 (IRQ_S_SOFT)
	if (mip_mie & (1 << 1)) != 0 {
		let val = 1u64 << 16 | 1;
		// Clear SSIP now — mirrors compute_mtopi's MSIP clearing.
		// In AIA mode SSIP comes from the IMSIC S-file (eip[1] → STOPEI claim),
		// but when falling through to this legacy path the bit must be cleared
		// to prevent re-delivery.
		state.mip.fetch_and(!(1 << 1), Ordering::AcqRel);
		return (val, 0);
	}
	// 3. STIP → IID=5 (IRQ_S_TIMER)
	//
	// 与 sync_mtip 活跃分支一致的判定: 已设定 deadline (write_stimecmp 写入
	// 未来值) 时按本 hart 指令计数空间判定, 与共享 mtime 跨 batch 膨胀解耦 —
	// 否则其他活跃 hart 的进度会把刚 claim 后 bump 到 mtime+4 的 stimecmp
	// 重新推成"已到期", stopi 立即再次报告 STI → 活锁。
	// 未设定 deadline (deadline==0, Python 步骤路径/写入即到期) 时退化为
	// 共享比较 (旧行为)。claim 路径 (csrw stopi IID=5) bump stimecmp 后经
	// write_stimecmp 设定 deadline, 故 stopi 在 claim 后的重读回到 0。
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
		// MSIP (cause 3) is a level-triggered CLINT interrupt.  ``deliver_trap_mmode``
		// clears ``mip.MSIP`` (the CSR bit) but NOT the CLINT level bit, which is
		// the hardware source that ``sync_msip`` re-samples at every instruction
		// boundary.  If the level bit stays set, the same IPI is re-asserted and
		// re-delivered, corrupting the OpenSBI ``tlb_sync`` counter (the receiver's
		// handler decrements it once per delivery) and leaving the sender spinning
		// forever in ``sbi_fifo_dequeue``'s ``spin_lock``.  This mirrors the Python
		// ``_trap_deliver_mmode`` which calls ``ctrl.clear_ipi()``.
		//
		// Clearing only the level bit is safe: a concurrent IPI already in flight
		// is captured by the ``msip_pending`` edge channel, which ``sync_msip``
		// drains on the next boundary and re-asserts ``mip.MSIP``.
		if cause == 3 {
			let hid = state.mhartid as usize;
			if hid < clint.num_harts as usize {
				// 仅清电平位 (bit 0), 保留 edge counter (bits 7:1). 该字节由 Python
				// 侧 ``_native_marshal_clint`` 编码为 ``level | (edge << 1)``, edge
				// counter 用于跨加速执行边界检测 MSIP 变化; ``store(0)`` 会连同
				// edge counter 一起清零, 与 ``clint_write_msip_concurrent`` 的
				// write-0 语义 (``fetch_and(0xFE)``) 不一致.
				unsafe { &*clint.msip.add(hid) }.fetch_and(0xFE, Ordering::Release);
			}
		}
		true
	} else {
		false
	}
}

/// Drain the shared external-interrupt latch (``ext_irq.pending``) into the
/// hart's ``mip.MEIP``/``mip.SEIP`` bits.
///
/// Only lines NOT owned by an IMSIC interrupt file (``eidelivery == 1``) are
/// driven here — when eidelivery=1 the IMSIC file owns that external line and
/// ``sync_imsic`` (or the boundary marshal) drives it from eip & eie.  Setting
/// it here too would create a fetch_or / topei_peek / fetch_and ping-pong
/// across step_interrupts (~22× instruction-throughput regression in AIA mode).
///
/// Returns ``true`` when ``ext_irq.pending`` was asserted (callers may use it
/// as a wake condition).
#[inline]
pub(crate) fn sync_ext_irq_mip(state: &mut HartState, ext_irq: *mut FfiExtIrqCtx) -> bool {
	if ext_irq.is_null() || unsafe { (*ext_irq).pending == 0 } {
		return false;
	}
	let mfile_owns = state.imsic_m.present != 0 && state.imsic_m.eidelivery != 0;
	let sfile_owns = state.imsic_s.present != 0 && state.imsic_s.eidelivery != 0;
	if !sfile_owns {
		state.mip.fetch_or(1 << 9, Ordering::AcqRel); // SEIP
	}
	if !mfile_owns {
		state.mip.fetch_or(1 << 11, Ordering::AcqRel); // MEIP
	}
	true
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::concurrent::{ConcurrentClintCtx, FfiExtIrqCtx};
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
		// Leak the boxes so the pointers stay valid for the test's lifetime.
		std::mem::forget(msip);
		std::mem::forget(mtime);
		std::mem::forget(mtimecmp);
		(clint, msip_ptr)
	}

	/// Regression: delivering an MSIP (cause 3) must clear the CLINT level bit,
	/// not just ``mip.MSIP``.  Otherwise ``sync_msip`` re-asserts the interrupt
	/// at the next boundary and the OpenSBI ``tlb_sync`` protocol livelocks
	/// (the sender spins in ``sbi_fifo_dequeue``'s ``spin_lock`` forever).
	#[test]
	fn msip_delivery_clears_clint_level_bit() {
		let mut state: HartState = unsafe { std::mem::zeroed() };
		state.mhartid = 0;
		state.mode = riscv_mode::M;
		state.mstatus = 1 << 3; // MIE = 1
		state.mie = 1 << 3; // MSIE = 1
		state.mtvec = 0x80000000;
		state.pc = 0x1000;
		state.mip.store(1 << 3, Ordering::Release); // MSIP pending

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

	/// Regression: MSIP 投递后清电平位必须保留 edge counter (bits 7:1).
	///
	/// 该字节由 Python 侧 ``_native_marshal_clint`` 编码为 ``level | (edge << 1)``,
	/// edge counter 用于跨加速执行边界检测 MSIP 变化. ``store(0)`` 会连同 edge
	/// counter 一起清零, 破坏 Python 侧的跳变检测; 必须用 ``fetch_and(0xFE)``
	/// 仅清电平位, 与 ``clint_write_msip_concurrent`` 的 write-0 语义一致.
	#[test]
	fn msip_delivery_preserves_edge_counter() {
		let mut state: HartState = unsafe { std::mem::zeroed() };
		state.mhartid = 0;
		state.mode = riscv_mode::M;
		state.mstatus = 1 << 3; // MIE = 1
		state.mie = 1 << 3; // MSIE = 1
		state.mtvec = 0x80000000;
		state.pc = 0x1000;
		state.mip.store(1 << 3, Ordering::Release); // MSIP pending

		// 字节编码 ``level | (edge << 1)``: level=1, edge=3 -> 0b111.
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

	/// Regression: MTI (cause 7) 在 mideleg[7]=1 时必须按 S 级中断 (STI) 投递到
	/// S 模式, 而不是硬编码投递到 M 模式. 早期 "M-mode first" 模型将
	/// MEI/MSI/MTI 硬编码为不可委派: 裸核内核设 mideleg=0x80 期待 MTI→STI,
	/// 但 native 模式 MTI 永远进 M 模式 → m_trap_handler skip+4 → mret →
	/// MTIP 仍悬置 → 死循环. 修复后 check_pending_interrupts 按 mideleg 动态判定
	/// (RISC-V 规范 §3.1.9).
	#[test]
	fn mti_delegated_to_s_when_mideleg_set() {
		let mut state: HartState = unsafe { std::mem::zeroed() };
		state.mhartid = 0;
		state.mode = riscv_mode::S;
		state.mstatus = 1 << 1; // SIE = 1
		state.mie = 1 << 7; // MTIE = 1
		state.mideleg = 1 << 7; // delegate MTI -> STI
		state.mtvec = 0x80000000;
		state.stvec = 0x80004000;
		state.pc = 0x1000;
		state.mip.store(1 << 7, Ordering::Release); // MTIP pending

		let (clint, _) = clint_with_msip(0);

		let delivered = check_and_deliver_interrupt_concurrent(&mut state, &clint);

		assert!(delivered, "MTI 应被投递");
		assert_eq!(
			state.mode,
			riscv_mode::S,
			"委派中断应投递到 S 模式而非 M 模式 (修复前硬编码投递 M → 死循环)"
		);
		assert_eq!(state.pc, 0x80004000, "应跳转 stvec (S 模式 handler)");
		assert_eq!(state.scause, mcause_val(7, true), "scause 应为 STI");
		assert_eq!(state.sepc, 0x1000, "sepc 应保存中断前 PC");
	}

	/// Regression: 委派中断 (mideleg[7]=1) 在 M 模式保持挂起, 不投递到 M —
	/// 委派中断仅在 hart 降到 S/U 模式且 S 级全局使能时才投递 (与 Python
	/// check_pending_interrupts 的 s_mode_global 语义一致: 委派中断永不进 M).
	#[test]
	fn delegated_mti_stays_pending_in_mmode() {
		let mut state: HartState = unsafe { std::mem::zeroed() };
		state.mhartid = 0;
		state.mode = riscv_mode::M;
		state.mstatus = 1 << 3; // MIE = 1
		state.mie = 1 << 7; // MTIE = 1
		state.mideleg = 1 << 7; // delegate MTI -> STI
		state.mtvec = 0x80000000;
		state.pc = 0x1000;
		state.mip.store(1 << 7, Ordering::Release); // MTIP pending

		let (clint, _) = clint_with_msip(0);

		let delivered = check_and_deliver_interrupt_concurrent(&mut state, &clint);

		assert!(!delivered, "委派中断在 M 模式应保持挂起, 等 hart 降到 S/U");
		assert_eq!(state.pc, 0x1000, "不应跳转 (未投递)");
		assert_eq!(state.mode, riscv_mode::M, "应保持 M 模式");
	}

	/// Regression: ext_irq drain 仅对未被 IMSIC 占用的线路内联置位 MEIP/SEIP.
	///
	/// 双重否定笔误 (``!present != 0``) 使门控变成 ``eidelivery != 0`` — 恰好与
	/// 意图相反: legacy 模式 (present=0) 永不内联置位 (设备中断延迟到 batch 边界),
	/// AIA 模式 (eidelivery=1) 无条件置位 (fetch_or / topei_peek / fetch_and
	/// ping-pong, ~22× 指令吞吐回归). 此测试锁定三个方向的正确语义.
	#[test]
	fn ext_irq_drain_respects_imsic_ownership() {
		let mut ext = FfiExtIrqCtx {
			pending: 1,
			sources: 0,
			max_priority: 0,
			_pad: [0; 2],
		};
		let ext_ptr: *mut FfiExtIrqCtx = &mut ext;

		// legacy: 无 IMSIC -> 线路未被占用 -> 必须内联置位 MEIP/SEIP.
		let mut legacy: HartState = unsafe { std::mem::zeroed() };
		legacy.imsic_m.present = 0;
		legacy.imsic_s.present = 0;
		legacy.mip.store(0, Ordering::Release);
		assert!(sync_ext_irq_mip(&mut legacy, ext_ptr), "legacy 模式 ext_irq 应被消费");
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

		// AIA + eidelivery=1: IMSIC 独占两条线路 -> 此处不得置位 (避免 ping-pong).
		let mut aia: HartState = unsafe { std::mem::zeroed() };
		aia.imsic_m.present = 1;
		aia.imsic_m.eidelivery = 1;
		aia.imsic_s.present = 1;
		aia.imsic_s.eidelivery = 1;
		aia.mip.store(0, Ordering::Release);
		assert!(sync_ext_irq_mip(&mut aia, ext_ptr), "AIA 模式 ext_irq 应被消费");
		assert_eq!(
			aia.mip.load(Ordering::Acquire) & ((1 << 11) | (1 << 9)),
			0,
			"eidelivery=1 时 ext_irq 不得内联置位 MEIP/SEIP (IMSIC 独占该线路)"
		);

		// AIA present 但 eidelivery=0: IMSIC 在场却未投递 -> 线路回退 legacy 排水.
		let mut aia_off: HartState = unsafe { std::mem::zeroed() };
		aia_off.imsic_m.present = 1;
		aia_off.imsic_m.eidelivery = 0;
		aia_off.imsic_s.present = 1;
		aia_off.imsic_s.eidelivery = 0;
		aia_off.mip.store(0, Ordering::Release);
		assert!(sync_ext_irq_mip(&mut aia_off, ext_ptr), "eidelivery=0 时 ext_irq 应被消费");
		assert_ne!(
			aia_off.mip.load(Ordering::Acquire) & (1 << 9),
			0,
			"eidelivery=0 时 ext_irq 应内联置位 SEIP (回退 legacy 线路)"
		);
	}
}
