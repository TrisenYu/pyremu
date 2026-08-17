pub(crate) mod clint;
pub(crate) mod imsic;
pub(crate) mod wfi;

use crate::concurrent::ConcurrentClintCtx;
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
	// Machine-level interrupts (MEI, MSI, MTI) are NOT delegatable by
	// convention — they always trap to M-mode first.  The M-mode handler
	// may then inject a supervisor-level interrupt (SEI, SSI, STI) which
	// IS delegated.  Marking them delegatable here would cause them to be
	// silently skipped when S-mode has interrupts disabled (SIE=0),
	// leading to lost IPIs and TLB-shootdown deadlocks.
	let checks: [(u64, u64, u64); 6] = [
		(MIE_MEIE, 11, 0),
		(MIE_MSIE, 3, 0),
		(MIE_MTIE, 7, 0),
		(MIE_SEIE, 9, 1),
		(MIE_SSIE, 1, 1),
		(MIE_STIE, 5, 1),
	];

	for (mask, cause, delegatable) in &checks {
		if pending & mask == 0 {
			continue;
		}
		if *delegatable == 0 || (state.mideleg & mask) == 0 {
			// delegated = (state.mideleg & mask) != 0
			// 所以这里是 !delegated
			if state.mode < riscv_mode::M || state.mstatus & (1 << 3) != 0 {
				return Some((*cause, true));
			}
			return None;
		}
		if state.mode < riscv_mode::M {
			if state.mstatus & (1 << 1) == 0 {
				continue;
			}
			return Some((*cause, false));
		}
		// M-mode with delegated interrupt:
		// The M-mode handler cannot discover delegated (S-level)
		// interrupts via mtopi in AIA mode — compute_mtopi only
		// checks the M-file, but S-file IPIs (IID=1) create SEIP
		// and are invisible to mtopi.  Delivering a delegated
		// interrupt to M-mode creates an infinite
		// SEI→mtopi=0→MRET→SEI loop that burns ~22x more
		// instructions and starves other harts.
		//
		// Skip: the interrupt is held pending until the hart
		// drops to S-mode, where it will be delivered correctly.
		//
		// M-mode IPIs (IID=3 → MEIP) are NOT delegatable
		// (delegatable=0 in the checks array), so this does NOT
		// break sbi_hart_start wake-up of secondary harts.
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
	#[cfg(feature = "diagnostic")]
	{
		// 限速 stopi 观测: 仅在返回非零 (应触发 IMSIC 分派) 或 SEIP 挂起但
		// 返回 0 (丢失信号) 时记录, 每条最多 40 次. 用于定位 AIA 启动挂死中
		// stopi→stopei 链路的断裂点, 不产生海量日志.
		static N: AtomicU32 = AtomicU32::new(0);
		let seip = (state.mip.load(Ordering::Acquire) & (1 << 9)) != 0;
		if (raw != 0 || seip) && N.fetch_add(1, Ordering::Relaxed) < 40 {
			diag::log_line(&format!(
				"STOPI h{} ret={:#x} raw={:#x} mip={:#x} mie={:#x} mode={} s_eid={} s_eip0={:#x}",
				state.mhartid,
				if raw != 0 { (9u64 << 16) | (raw & 0xFF) } else { 0 },
				raw,
				state.mip.load(Ordering::Acquire),
				state.mie,
				state.mode,
				state.imsic_s.eidelivery,
				state.imsic_s.eip[0].load(Ordering::Acquire),
			));
		}
	}
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
	// Re-evaluate STIP from LIVE mtime + stimecmp (matching the Python
	// _read_stopi path).  Using the cached mip&mie from sync_mtip is NOT
	// sufficient because stimecmp may have been bumped mid-instruction by a
	// prior csrw stopi claim: mtime is frozen within a speedup execution,
	// but stimecmp changes via CSR writes.
	// sync_mtip cleared STIP after the bump, but if the kernel's timer ISR does NOT
	// update stimecmp (e.g. the kernel falls back to sbi_set_timer->mtimecmp
	// instead of the SSTC stimecmp CSR),
	// stimecmp stays just ahead of the frozen mtime and STIP is
	// correctly clear for the rest of an execution.  Re-evaluating live here
	// gives the same answer as sync_mtip — correctness comes from the
	// claim handler bump, not from ignoring the cached mip.
	if (state.mie & (1 << 5)) != 0 && state.stimecmp > 0 && mtime >= state.stimecmp {
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

#[cfg(test)]
mod tests {
	use super::*;
	use crate::concurrent::ConcurrentClintCtx;
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
}
