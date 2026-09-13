//! Trap delivery with delegation support execution engine.
//!
//! Implements medeleg / mideleg-based delegation to S-mode, matching the
//! Python ``deliver_trap`` in ``trap_handler.py``.

use crate::concurrent::{ConcurrentClintCtx, ModuleState};
use crate::hart_sched::read_gpr;
use crate::interrupt::clint::write_mtimecmp;
use crate::interrupt::imsic_clear_ipi_on_trap;
use crate::ffi::FfiDevCtx;
use crate::state::{riscv_mode, HartState};
use std::sync::atomic::Ordering;
// ============================================================
//  Trap cause codes (mcause / scause compatible)
// ============================================================

/// RISC-V exception codes (``mcause[62:0]`` when bit 63 = 0).
#[allow(dead_code)]
pub mod exc_code {
	pub const INSTR_ACCESS_FAULT: u64 = 1;
	pub const ILL_INSTR: u64 = 2;
	pub const BREAKPOINT: u64 = 3;
	pub const LD_MISALIGNED: u64 = 4;
	pub const LD_ACCESS_FAULT: u64 = 5;
	pub const ST_MISALIGNED: u64 = 6;
	pub const ST_ACCESS_FAULT: u64 = 7;
	pub const ECALL_UMODE: u64 = 8;
	pub const ECALL_SMODE: u64 = 9;
	pub const ECALL_MMODE: u64 = 11;
	pub const INSTR_PAGE_FAULT: u64 = 12;
	pub const LD_PAGE_FAULT: u64 = 13;
	pub const ST_PAGE_FAULT: u64 = 15;
}

/// Build an ``mcause``-compatible value: bit 63 = interrupt flag, lower bits = code.
#[inline]
pub fn mcause_val(code: u64, is_interrupt: bool) -> u64 {
	if is_interrupt {
		code | (1u64 << 63)
	} else {
		code
	}
}

// ============================================================
//  mstatus bit constants
// ============================================================

const MSTATUS_SIE: u64 = 1 << 1;
const MSTATUS_MIE: u64 = 1 << 3;
const MSTATUS_SPIE: u64 = 1 << 5;
const MSTATUS_MPIE: u64 = 1 << 7;
const MSTATUS_SPP: u64 = 1 << 8;
const MSTATUS_MPP: u64 = 0b11 << 11;

/// Map privilege mode to MPP field encoding.
#[inline]
fn mode_to_mpp(mode: u8) -> u64 {
	match mode {
		riscv_mode::U => 0,
		riscv_mode::S => 1,
		riscv_mode::M => 3,
		_ => 0,
	}
}

/// Map privilege mode to SPP field encoding (U=0, S=1).
#[inline]
fn mode_to_spp(mode: u8) -> u64 {
	match mode {
		riscv_mode::U => 0,
		riscv_mode::S => 1,
		_ => 0,
	}
}

// ============================================================
//  Trap delivery — delegation-aware entry point
// ============================================================

/// Deliver a trap with ``medeleg`` / ``mideleg`` delegation support.
///
/// Checks the delegation registers and routes the trap to S-mode when
/// the exception/interrupt is delegated and the hart is not already in
/// M-mode.  This matches the Python ``deliver_trap`` in ``trap_handler.py``.
///
/// Returns ``0`` (caller should NOT advance PC — PC already redirected).
pub fn deliver_trap(state: &mut HartState, code: u64, tval: u64) -> u64 {
	let is_interrupt = (code >> 63) != 0;
	let exc_code = code & 0x7FFF_FFFF_FFFF_FFFF;

	// M-mode traps are never delegated.
	let delegate = if state.mode != riscv_mode::M {
		if is_interrupt {
			state.mideleg & (1 << exc_code) != 0
		} else {
			state.medeleg & (1 << exc_code) != 0
		}
	} else {
		false
	};

	if delegate {
		deliver_trap_smode(state, code, tval)
	} else {
		deliver_trap_mmode(state, code, tval)
	}
}

// ============================================================
//  S-mode trap delivery
// ============================================================

/// Deliver a trap to S-mode, saving PC / CSRs and redirecting to ``stvec``.
///
/// Called by ``deliver_trap`` when delegation conditions are met.
/// Matches the Python ``_trap_deliver_smode`` in ``trap_handler.py``.
pub fn deliver_trap_smode(state: &mut HartState, code: u64, tval: u64) -> u64 {
	// Clear LR/SC reservation on any trap
	state.reservation_valid = 0;
	state.reservation_addr = 0;

	// Wake from WFI
	state.waiting = 0;

	// Save PC -> sepc
	state.sepc = state.pc;

	// Save cause
	state.scause = code;

	// Save tval
	state.stval = tval;

	// mstatus: SPIE <- SIE, SIE <- 0, SPP <- current mode
	let sie_present = (state.mstatus & MSTATUS_SIE) != 0;
	let spp_bits = mode_to_spp(state.mode) << 8;
	if sie_present {
		state.mstatus |= MSTATUS_SPIE;
	} else {
		state.mstatus &= !MSTATUS_SPIE;
	}
	state.mstatus &= !MSTATUS_SIE;
	state.mstatus = (state.mstatus & !MSTATUS_SPP) | spp_bits;

	// Switch to S-mode
	state.mode = riscv_mode::S;

	// Jump to stvec
	let stvec = state.stvec;
	let tvec_mode = stvec & 0x3;
	let tvec_base = stvec & !0x3;
	let exc_code = code & 0x7FFF_FFFF_FFFF_FFFFu64;
	let is_interrupt = (code >> 63) != 0;

	if tvec_mode == 0 || !is_interrupt {
		// Direct mode or exception: jump to base
		state.pc = tvec_base;
	} else {
		// Vectored mode + interrupt: base + 4 * exc_code
		state.pc = tvec_base + 4 * exc_code;
	}

	// MSIP/SSIP auto-clear: always, including AIA mode.
	// sbi_ipi_raw_clear(false) is a no-op in AIA mode, so clearing here
	// is the only mechanism to prevent infinite re-delivery.
	// Also clear IMSIC eip to prevent stale IID on MTOPI/STOPI read.
	if !is_interrupt {
		return 0;
	}
	// In AIA mode, IPIs route through MEIP/SEIP (cause 11/9), not
	// MSIP/SSIP (cause 3/1).  When they DO arrive as cause 3/1
	// (legacy path), defer eip clearing to MTOPEI/STOPEI claim so
	// the handler sees the interrupt identity.  In legacy mode,
	// clear MSIP per RISC-V spec §3.1.15.
	let in_aia_s = state.imsic_s.present != 0 && state.imsic_s.eidelivery != 0;
	if exc_code == 3 {
		if in_aia_s {
			// AIA mode: defer MSIP/eip clearing
		} else {
			state.mip.fetch_and(!(1u64 << 3), Ordering::AcqRel); // MSIP
			imsic_clear_ipi_on_trap(state, 3);
		}
	} else if exc_code == 1 {
		if in_aia_s {
			// AIA mode: defer SSIP/eip clearing
		} else {
			state.mip.fetch_and(!(1u64 << 1), Ordering::AcqRel); // SSIP
			imsic_clear_ipi_on_trap(state, 1);
		}
	}
	0 // advance = 0 — PC was redirected
}

// ============================================================
//  M-mode trap delivery (non-delegated fallback)
// ============================================================

/// Deliver a trap to M-mode, saving PC / CSRs and redirecting to ``mtvec``.
///
/// Called by ``deliver_trap`` when the trap is not delegated or the hart is
/// already in M-mode.  Matches the Python ``_trap_deliver_mmode``.
pub fn deliver_trap_mmode(state: &mut HartState, code: u64, tval: u64) -> u64 {
	// Clear LR/SC reservation on any trap
	state.reservation_valid = 0;
	state.reservation_addr = 0;

	// RISC-V spec §3.1.15: non-delegated machine-level interrupts are
	// auto-cleared by hardware on trap entry.  This prevents the
	// interrupt from re-firing immediately after MRET when the CLINT
	// MSIP register is still asserted (level-triggered source).
	//
	// Always clear MSIP, including in AIA mode.  The previous AIA-mode
	// exception assumed that sbi_ipi_raw_clear() -> write 0 to CLINT MSIP
	// would clear it, but sbi_ipi_raw_clear(false) is a no-op in AIA mode
	// (imsic_ipi_device has no ipi_clear callback).  OpenSBI does NOT use
	// the MTOPI dispatch loop for MSIP (cause 3) either — it directly
	// calls sbi_ipi_process().  Without clearing here, MSIP stays set
	// forever -> infinite re-delivery loop on every instruction boundary.
	let is_interrupt = (code >> 63) != 0;
	let exc_code = code & 0x7FFF_FFFF_FFFF_FFFFu64;

	// In AIA mode (eidelivery==1), OpenSBI dispatches via MTOPI
	// (sbi_trap_aia_irq) which reads compute_mtopi -> clears MSIP there.
	// If we clear MSIP and IMSIC eip here, MTOPI sees nothing -> IPI is
	// lost -> RCU stall and SMP boot timeout.
	//
	// In legacy mode (eidelivery==0), OpenSBI dispatches via mcause and
	// clears the CLINT level bit via sbi_ipi_raw_clear.  MSIP must be
	// cleared here for the same reason as the RISC-V spec §3.1.15.
	let in_aia = state.imsic_m.present != 0 && state.imsic_m.eidelivery != 0;
	if is_interrupt && exc_code == 3 {
		if in_aia {
			// AIA mode: defer MSIP/eip clearing to compute_mtopi /
			// MTOPEI claim.  This ensures sbi_trap_aia_irq sees the
			// interrupt identity via MTOPI and dispatches correctly.
		} else {
			// Legacy mode: MSIP auto-clear (RISC-V spec §3.1.15).
			state.mip.fetch_and(!(1u64 << 3), Ordering::AcqRel);
			imsic_clear_ipi_on_trap(state, 3);
		}
	}

	// Wake from WFI
	state.waiting = 0;

	// Save PC -> mepc
	state.mepc = state.pc;

	// Save cause
	state.mcause = code;

	// Save tval
	state.mtval = tval;

	// mstatus: MPIE <- MIE, MIE <- 0, MPP <- current mode
	let mie_present = (state.mstatus & MSTATUS_MIE) != 0;
	let mpp_bits = mode_to_mpp(state.mode) << 11;
	if mie_present {
		state.mstatus |= MSTATUS_MPIE;
	} else {
		state.mstatus &= !MSTATUS_MPIE;
	}
	state.mstatus &= !MSTATUS_MIE;
	state.mstatus = (state.mstatus & !MSTATUS_MPP) | mpp_bits;

	// RISC-V spec: MPRV is cleared on trap entry to M-mode so the
	// handler can safely access its own stack/data without going
	// through the MMU translation of the previous privilege mode.
	// Without this, nested interrupts during OpenSBI's MPRV=1
	// window (sbi_get_insn) corrupt M-mode stack state because
	// loads/stores use S-mode page tables for M-mode stack VAs.
	state.mstatus &= !(1u64 << 17); // clear MPRV

	// Switch to M-mode
	state.mode = riscv_mode::M;

	// Jump to mtvec
	let mtvec = state.mtvec;
	let tvec_mode = mtvec & 0x3;
	let tvec_base = mtvec & !0x3;
	let exc_code = code & 0x7FFF_FFFF_FFFF_FFFFu64;
	let is_interrupt = (code >> 63) != 0;

	if tvec_mode == 0 || !is_interrupt {
		// Direct mode or exception: jump to base
		state.pc = tvec_base;
	} else {
		// Vectored mode + interrupt: base + 4 * exc_code
		state.pc = tvec_base + 4 * exc_code;
	}

	0 // advance = 0 — PC was redirected
}

/// Convenience: deliver IllInstr trap for an invalid encoding.
/// Uses ``deliver_trap`` (delegation-aware).
#[inline]
pub fn deliver_illegal_instruction(state: &mut HartState, tval: u64) -> u64 {
	deliver_trap(state, mcause_val(exc_code::ILL_INSTR, false), tval)
}

// ============================================================
//  Privileged instructions — extracted from concurrent.rs
// ============================================================

pub(crate) fn priv_ecall_concurrent(
	state: &mut HartState,
	_instr: u32,
	clint: &ConcurrentClintCtx,
	_module: &ModuleState,
	dev: &dyn FfiDevCtx,
) -> u64 {
	let a7 = read_gpr(state, 17);
	let a6 = read_gpr(state, 16);

	// 设备快速路径 (如 UART sbi_putchar) — 直接内联处理, 避免 MMIO batch 退出.
	// 每个字符一次 batch 退出 (~1-5 ms marshal/unmarshal) 是飞地输出密集时
	// 数百倍减速的根因之一. 新设备只需实现 FfiDevCtx trait, 无需修改此处分发逻辑.
	if let Some(advance) = dev.try_ecall(state, a7, a6) {
		return advance;
	}

	if a7 == 0x54494D45 && a6 == 0 {
		// SBI_TIME set_timer — programs the M-mode mtimecmp comparator
		// so that MTIP fires at the requested absolute time.  The S-mode
		// stimecmp (SSTC) is an INDEPENDENT device; do NOT overwrite it
		// here — the kernel manages stimecmp directly via the CSR when
		// SSTC is advertised in the ISA string.
		let stime_val = read_gpr(state, 10);
		let hid = state.mhartid as usize;
		if hid < clint.num_harts as usize {
			unsafe { &*clint.mtimecmp.add(hid) }.store(stime_val, Ordering::Release);
			// 与 CLINT MMIO 写路径 (write_mtimecmp_for_target) 对齐: 设定一次性
			// mtip_deadline。否则 mtip_deadline 保持 0, sync_mtip 落入共享 mtime
			// 比较分支, 被其他活跃 hart 经 fetch_max 推高的共享 mtime 过早触发 ->
			// mret 后立即再次 trap 的活锁 (飞地走 sbi_set_timer, 而非 SSTC stimecmp,
			// 是唯一命中此坏路径的调用方)。
			let now = unsafe { &*clint.mtime }.load(Ordering::Acquire);
			write_mtimecmp(state, now, stime_val, clint.timebase_hz);
		}
		state.gprs[10] = 0;
		return 4;
	}

	// NOTE: there is intentionally NO fast path for SBI_IPI (a7=0x735049).
	// Writing only CLINT MSIP is insufficient — the firmware's
	// ``sbi_ipi_send_many()`` also stores the IPI event type
	// (``ipi_smode_event``) in the target hart's ``ipi_data->ipi_type``
	// so that ``sbi_ipi_process()`` can dispatch to
	// ``sbi_ipi_process_smode()`` which sets SSIP.  Without that step
	// Linux's S-mode TLB-shootdown handler never runs -> deadlock.
	//
	// Instead we deliver the ECALL trap inline; the M-mode handler
	// (running within the same acceleration) calls ``sbi_ipi_send_many()``
	// which sets ``ipi_type`` on the target *and* writes MSIP via
	// inline CLINT MMIO.  The target hart's thread sees the atomic
	// MSIP store and processes the IPI, all within one acceleration.

	// Generic ECALL — deliver trap inline (stay in acceleration).
	let ecall_cause = match state.mode {
		riscv_mode::U => mcause_val(exc_code::ECALL_UMODE, false),
		riscv_mode::S => mcause_val(exc_code::ECALL_SMODE, false),
		_ => mcause_val(exc_code::ECALL_MMODE, false),
	};
	deliver_trap(state, ecall_cause, 0);
	// deliver_trap sets PC -> mtvec/stvec and mode -> M/S.
	// Return 0 so the dispatch loop continues execution from the
	// trap handler entry point, all within the same acceleration.
	0
}

pub(crate) fn priv_mret_concurrent(state: &mut HartState, instr: u32) -> u64 {
	// MRET is only legal in M-mode.  Matches Python ``trap_mret``
	// lines 341-342 in trap_handler.py.
	if state.mode != riscv_mode::M {
		deliver_illegal_instruction(state, instr as u64);
		return 0;
	}
	let mpp = (state.mstatus >> 11) & 0x3;
	let mpie = (state.mstatus >> 7) & 1;
	state.mode = match mpp {
		0 => riscv_mode::U,
		1 => riscv_mode::S,
		3 => riscv_mode::M,
		_ => riscv_mode::M,
	};
	state.mstatus &= !(1 << 3);
	if mpie != 0 {
		state.mstatus |= 1 << 3;
	}
	state.mstatus |= 1 << 7;
	state.mstatus &= !(0b11 << 11);
	state.pc = state.mepc;
	state.waiting = 0;
	0
}

pub(crate) fn priv_sret_concurrent(
	state: &mut HartState,
	instr: u32,
	_ctx: &crate::translate::WalkCtx,
) -> u64 {
	if state.mode < riscv_mode::S {
		deliver_illegal_instruction(state, instr as u64);
		return 0;
	}
	let spp = (state.mstatus >> 8) & 1;
	let spie = (state.mstatus >> 5) & 1;
	state.mode = if spp == 0 {
		riscv_mode::U
	} else {
		riscv_mode::S
	};
	state.mstatus &= !(1 << 1);
	if spie != 0 {
		state.mstatus |= 1 << 1;
	}
	state.mstatus |= 1 << 5;
	state.mstatus &= !(1 << 8);
	state.pc = state.sepc;
	state.waiting = 0;
	0
}

pub(crate) fn priv_wfi_concurrent(state: &mut HartState, instr: u32) -> u64 {
	let tw = (state.mstatus >> 21) & 1;
	if tw != 0 && state.mode != riscv_mode::M {
		deliver_illegal_instruction(state, instr as u64);
		return 0;
	}
	if state.wfi_woken != 0 {
		state.wfi_woken = 0;
		return 4;
	}
	let pending = state.mip.load(Ordering::Acquire) & state.mie;
	if pending != 0 {
		return 4;
	}
	state.waiting = 1;
	4
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::ffi::FfiUartCtx;
	use std::cell::Cell;
	use std::sync::atomic::{AtomicU64, AtomicU8};

	fn default_state() -> HartState {
		let mut s: HartState = unsafe { std::mem::zeroed() };
		s.mode = riscv_mode::M;
		s.mtvec = 0x80000000;
		s.stvec = 0x80004000;
		s.pc = 0x1000;
		s
	}

	// ---- delegation tests ----

	#[test]
	fn trap_delegates_to_smode_when_medeleg_set() {
		let mut s = default_state();
		s.mode = riscv_mode::S;
		s.medeleg = 1 << exc_code::ILL_INSTR; // delegate ill-instr
		deliver_trap(&mut s, mcause_val(exc_code::ILL_INSTR, false), 0xDEAD);
		assert_eq!(
			s.mode,
			riscv_mode::S,
			"should stay in S-mode when delegated"
		);
		assert_eq!(s.sepc, 0x1000, "sepc should save faulting PC");
		assert_eq!(s.scause, mcause_val(exc_code::ILL_INSTR, false));
		assert_eq!(s.stval, 0xDEAD);
		assert_eq!(s.pc, 0x80004000, "should jump to stvec base");
	}

	#[test]
	fn trap_not_delegated_from_mmode() {
		let mut s = default_state();
		s.mode = riscv_mode::M;
		s.medeleg = 1 << exc_code::ILL_INSTR; // delegation ignored in M-mode
		deliver_trap(&mut s, mcause_val(exc_code::ILL_INSTR, false), 0xBEEF);
		assert_eq!(s.mode, riscv_mode::M, "M-mode traps never delegate");
		assert_eq!(s.mepc, 0x1000);
		assert_eq!(s.mcause, mcause_val(exc_code::ILL_INSTR, false));
		assert_eq!(s.pc, 0x80000000, "should jump to mtvec base");
	}

	#[test]
	fn trap_not_delegated_when_medeleg_bit_clear() {
		let mut s = default_state();
		s.mode = riscv_mode::S;
		s.medeleg = 0; // nothing delegated
		deliver_trap(&mut s, mcause_val(exc_code::ILL_INSTR, false), 0xBEEF);
		assert_eq!(s.mode, riscv_mode::M, "should switch to M-mode");
		assert_eq!(s.mepc, 0x1000);
	}

	#[test]
	fn interrupt_delegates_to_smode_via_mideleg() {
		let mut s = default_state();
		s.mode = riscv_mode::S;
		s.mideleg = 1 << 7; // delegate MTI (timer interrupt)
		deliver_trap(&mut s, mcause_val(7, true), 0);
		assert_eq!(s.mode, riscv_mode::S, "delegated interrupt stays in S-mode");
		assert_eq!(s.sepc, 0x1000);
		assert_eq!(s.scause, mcause_val(7, true));
	}

	// ---- S-mode delivery tests ----

	#[test]
	fn smode_trap_saves_sie_to_spie() {
		let mut s = default_state();
		s.mode = riscv_mode::S;
		s.mstatus = MSTATUS_SIE; // SIE=1
		s.medeleg = 1 << exc_code::ILL_INSTR;
		deliver_trap(&mut s, mcause_val(exc_code::ILL_INSTR, false), 0);
		assert_eq!(s.mstatus & MSTATUS_SIE, 0, "SIE should be cleared");
		assert_ne!(s.mstatus & MSTATUS_SPIE, 0, "SPIE should be set");
	}

	#[test]
	fn smode_trap_sets_spp() {
		let mut s = default_state();
		s.mode = riscv_mode::S;
		s.medeleg = 1 << exc_code::ILL_INSTR;
		deliver_trap(&mut s, mcause_val(exc_code::ILL_INSTR, false), 0);
		let spp = (s.mstatus & MSTATUS_SPP) >> 8;
		assert_eq!(spp, 1, "SPP should be 1 (was S-mode)");
	}

	#[test]
	fn smode_trap_from_umode_sets_spp_0() {
		let mut s = default_state();
		s.mode = riscv_mode::U;
		s.medeleg = 1 << exc_code::ILL_INSTR;
		deliver_trap(&mut s, mcause_val(exc_code::ILL_INSTR, false), 0);
		let spp = (s.mstatus & MSTATUS_SPP) >> 8;
		assert_eq!(spp, 0, "SPP should be 0 (was U-mode)");
	}

	// ---- existing M-mode delivery tests (updated for deliver_trap) ----

	#[test]
	fn deliver_ill_instr_saves_mepc() {
		let mut s = default_state();
		s.pc = 0xBEEF;
		deliver_illegal_instruction(&mut s, 0xDEAD_BEEF);
		assert_eq!(s.mepc, 0xBEEF);
		assert_eq!(s.mcause, mcause_val(exc_code::ILL_INSTR, false));
		assert_eq!(s.mtval, 0xDEAD_BEEF);
	}

	#[test]
	fn deliver_trap_clears_mie() {
		let mut s = default_state();
		s.mstatus = MSTATUS_MIE; // MIE=1
		deliver_illegal_instruction(&mut s, 0);
		assert_eq!(s.mstatus & MSTATUS_MIE, 0, "MIE should be cleared");
		assert_eq!(
			s.mstatus & MSTATUS_MPIE,
			MSTATUS_MPIE,
			"MPIE should be set because MIE was 1"
		);
	}

	#[test]
	fn deliver_trap_sets_mpie_correctly_when_mie_was_0() {
		let mut s = default_state();
		s.mstatus = 0; // MIE=0
		deliver_illegal_instruction(&mut s, 0);
		assert_eq!(
			s.mstatus & MSTATUS_MPIE,
			0,
			"MPIE should be 0 when MIE was 0"
		);
	}

	#[test]
	fn deliver_trap_switches_to_mmode() {
		let mut s = default_state();
		s.mode = riscv_mode::S;
		deliver_illegal_instruction(&mut s, 0);
		assert_eq!(s.mode, riscv_mode::M);
	}

	#[test]
	fn deliver_trap_saves_mpp() {
		let mut s = default_state();
		s.mode = riscv_mode::S;
		deliver_illegal_instruction(&mut s, 0);
		let mpp = (s.mstatus & MSTATUS_MPP) >> 11;
		assert_eq!(mpp, mode_to_mpp(riscv_mode::S));
	}

	#[test]
	fn deliver_trap_direct_mtvec() {
		let mut s = default_state();
		s.mtvec = 0x80004000; // mode = 0 (direct)
		deliver_illegal_instruction(&mut s, 0);
		assert_eq!(s.pc, 0x80004000); // direct -> base
	}

	#[test]
	fn deliver_trap_vectored_exception() {
		let mut s = default_state();
		s.mtvec = 0x80004000 | 1; // vectored mode
		deliver_illegal_instruction(&mut s, 0);
		// Exception -> base, not base + 4*code
		assert_eq!(s.pc, 0x80004000);
	}

	#[test]
	fn deliver_trap_clears_reservation() {
		let mut s = default_state();
		s.reservation_valid = 1;
		s.reservation_addr = 0x8000_0000;
		deliver_illegal_instruction(&mut s, 0);
		assert_eq!(s.reservation_valid, 0);
	}

	#[test]
	fn mcause_encoding() {
		assert_eq!(mcause_val(2, false), 2);
		assert_eq!(mcause_val(7, true), (1u64 << 63) | 7);
	}

	// ============================================================
	//  Regression: MSIP auto-clear on trap entry (AIA mode)
	// ============================================================
	//
	//  In AIA mode sbi_ipi_raw_clear(false) is a no-op (imsic_ipi_device
	//  has no ipi_clear callback), so deliver_trap_mmode / deliver_trap
	//  MUST clear MSIP/SSIP unconditionally.  The old code had an
	//  AIA-mode exception that skipped the clear -> MSIP stayed set
	//  forever -> infinite re-delivery loop.

	#[test]
	fn msip_cleared_in_deliver_trap_mmode() {
		// deliver_trap_mmode always clears MSIP for cause=3 interrupts
		let mut s = default_state();
		s.mip.store(1 << 3, Ordering::Release); // MSIP set
		deliver_trap_mmode(&mut s, mcause_val(3, true), 0);
		assert_eq!(
			s.mip.load(Ordering::Acquire) & (1 << 3),
			0,
			"MSIP must be cleared by deliver_trap_mmode"
		);
	}

	#[test]
	fn msip_not_cleared_for_non_interrupt_in_deliver_trap_mmode() {
		// deliver_trap_mmode only clears MSIP for interrupts, not exceptions
		let mut s = default_state();
		s.mip.store(1 << 3, Ordering::Release); // MSIP set
		deliver_trap_mmode(&mut s, mcause_val(exc_code::ILL_INSTR, false), 0);
		assert_eq!(
			s.mip.load(Ordering::Acquire) & (1 << 3),
			1 << 3,
			"MSIP must NOT be cleared for exceptions"
		);
	}

	#[test]
	fn ssip_cleared_in_deliver_trap_for_delegated_interrupt() {
		// SSIP (cause=1) is delegatable — verify deliver_trap clears it
		let mut s = default_state();
		s.mode = riscv_mode::S;
		s.mip.store(1 << 1, Ordering::Release); // SSIP set
		s.mideleg = 1 << 1; // delegate SSIP
		s.mstatus |= 1 << 1; // SIE = 1
		deliver_trap(&mut s, mcause_val(1, true), 0);
		assert_eq!(
			s.mip.load(Ordering::Acquire) & (1 << 1),
			0,
			"SSIP must be cleared by deliver_trap for delegated interrupt"
		);
	}

	#[test]
	fn msip_cleared_for_non_mmode_interrupt_in_deliver_trap() {
		// When an interrupt fires while not in M-mode and is NOT delegated,
		// deliver_trap switches to deliver_trap_mmode.  Verify MSIP is
		// cleared on the S->M transition path.
		let mut s = default_state();
		s.mode = riscv_mode::S;
		s.mip.store(1 << 3, Ordering::Release); // MSIP set — not delegatable
		s.mstatus |= 1 << 1; // SIE = 1
					   // MSIP is non-delegatable, so it goes through deliver_trap->M-mode
					   // path, which calls _trap_deliver_mmode internally.
		deliver_trap(&mut s, mcause_val(3, true), 0);
		assert_eq!(
			s.mip.load(Ordering::Acquire) & (1 << 3),
			0,
			"MSIP must be cleared in deliver_trap non-delegated M-interrupt path"
		);
	}

	// ============================================================
	//  Regression: sbi_set_timer fast path must set mtip_deadline
	// ============================================================
	//
	//  旧实现只写 CLINT mtimecmp, 不设 mtip_deadline -> sync_mtip 落入
	//  共享 mtime 比较分支, 被其他活跃 hart 经 fetch_max 推高的共享 mtime
	//  过早触发 -> mret 后立即再 trap 的活锁。飞地走 sbi_set_timer (而非
	//  SSTC stimecmp), 是唯一命中此坏路径的调用方。

	#[test]
	fn sbi_set_timer_fast_path_sets_mtip_deadline() {
		let mut state: HartState = unsafe { std::mem::zeroed() };
		state.mhartid = 0;
		state.total_instrs = 1000;
		// a0 = 目标绝对时间 (未来值), a6 = 0, a7 = SBI_TIME (0x54494D45).
		state.gprs[10] = 11000;
		state.gprs[16] = 0;
		state.gprs[17] = 0x5449_4D45;

		let mtime = AtomicU64::new(1000);
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
			timebase_hz: 10_000_000,
		};
		let module = ModuleState::new(1, 1, 0, 0, vec![0].into_boxed_slice());
		let uart: FfiUartCtx = unsafe { std::mem::zeroed() };

		let advance = priv_ecall_concurrent(&mut state, 0, &clint, &module, &uart);

		assert_eq!(advance, 4, "sbi_set_timer 应返回 advance=4");
		assert_eq!(state.gprs[10], 0, "a0 应清零为 0");
		assert_eq!(mtimecmp.load(Ordering::Acquire), 11000, "mtimecmp 应被写入");
		assert_ne!(
			state.mtip_deadline, 0,
			"sbi_set_timer 必须设定 mtip_deadline (回归: 曾为 0 -> 共享 mtime 活锁)"
		);
		assert!(
			state.mtip_deadline > state.total_instrs,
			"deadline 必须落在未来, 而非立即到期"
		);
		assert_eq!(
			state.mip.load(Ordering::Acquire) & (1 << 7),
			0,
			"未来定时器不应立即置位 MTIP"
		);
	}
}
