use crate::concurrent::{ConcurrentClintCtx, ModuleState};
use crate::diag;
use crate::state::HartState;
use std::cell::Cell;
use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
// ============================================================
// CLINT inline handlers
// ============================================================
//
// Moved from handlers.rs.  These operate on the speedup execution engine
// ClintCtx (non-concurrent path) and are re-exported by handlers.rs for
// backward compatibility.

use crate::handlers::MemAccess;

pub(crate) fn sync_mtip(state: &mut HartState, clint: &ConcurrentClintCtx) {
	let hart_id = state.mhartid as usize;
	let cur_mtime = unsafe { &*clint.mtime }.load(Ordering::Acquire);
	let cmp = unsafe { &*clint.mtimecmp.add(hart_id) }.load(Ordering::Acquire);
	let sstc_cmp = state.stimecmp;

	// MTIP (bit 7): CLINT mtimecmp — used by OpenSBI for SBI_SET_TIMER
	// and its own internal timer needs.  This is INDEPENDENT of stimecmp
	// (matching QEMU / real hardware): writing stimecmp via SSTC CSR does
	// NOT affect mtimecmp, and writing mtimecmp via ACLINT MMIO does NOT
	// affect stimecmp.  Each comparator independently drives its own
	// interrupt bit (MTIP from mtimecmp, STIP from stimecmp).
	if cmp > 0 && cur_mtime >= cmp {
		state.mip.fetch_or(1 << 7, Ordering::AcqRel);
	} else {
		state.mip.fetch_and(!(1 << 7), Ordering::AcqRel);
	}

	// STIP (bit 5): SSTC stimecmp.  MUST only depend on state.stimecmp
	// (NOT clint.mtimecmp) because clint.mtimecmp is frozen at marshal
	// time.  If we included the frozen value, STIP would remain set
	// after the kernel writes a new stimecmp in the middle of current execution,
	// causing stopi to never return 0 and hence an infinite loop
	// in riscv_intc_aia_irq().
	//
	// Comparison is ">=" (NOT ">"), matching the RISC-V SSTC spec
	// (STIP asserts when mtime >= stimecmp) and the MTIP branch above.
	// The comparator fires on the same edge as the deadline, exactly as
	// Python's ``_eval_stip`` does.  This is a universal timer semantic —
	// it must not depend on the AIA ``stopi`` claim path (which re-arms
	// stimecmp to mtime+4 itself and is therefore independent of this
	// comparison).  A strict ">" here misses the deadline when the WFI
	// fast-forward lands mtime exactly on stimecmp, leaving the hart
	// parked with a pending-but-invisible timer interrupt.
	let st_pending = sstc_cmp > 0 && cur_mtime >= sstc_cmp;
	if st_pending {
		state.mip.fetch_or(1 << 5, Ordering::AcqRel);
	} else {
		state.mip.fetch_and(!(1 << 5), Ordering::AcqRel);
	}
}

/// Synchronise ``mip.MSIP`` from the cross-thread ``msip_pending`` atomic
/// channel and the CLINT level bit.
///
/// **All modes**: the ``msip_pending`` channel is always drained and
/// ``mip.MSIP`` is set from it.  This is the primary IPI signalling path.
///
/// **Legacy mode** (``present == 0 || eidelivery == 0``):
/// Additionally samples the CLINT level bit (level-triggered, matches real
/// SiFive CLINT / ACLINT MSWI hardware).  ``mip.MSIP`` is never cleared here —
/// ``deliver_trap`` for MSI (cause 3) clears MSIP when the trap is taken,
/// and the guest clears the CLINT level bit via ``sbi_ipi_raw_clear`` so
/// future calls see level=0 and stop re-setting MSIP.
///
/// **AIA mode with SMAIA** (``present != 0 && eidelivery != 0``):
/// The ``msip_pending`` edge is the authoritative IPI signal.  Additionally,
/// **stale MSIP is cleared** when both the edge channel and the CLINT level
/// bit are quiesced.  This is necessary because:
///
/// 1. ``try_handle_imsic_concurrent`` sets both IMSIC eip AND CLINT MSIP
///    (``msip_pending`` edge) for IPI identities (1/3).
/// 2. ``sync_imsic`` maps the IMSIC eip → MEIP.
/// 3. MEI (cause 11) has priority over MSI (cause 3), so MEI fires first.
/// 4. ``deliver_trap`` for MEI clears MEIP but NOT MSIP — MSIP stays set.
/// 5. ``sbi_trap_aia_irq()`` dispatches via MTOPI: MEI → MTOPEI claim →
///    ``sbi_ipi_process()`` (clears CLINT level).  MTOPI then sees MSIP
///    still set → IID=3 → ``sbi_ipi_process()`` again (``ipi_type`` is 0,
///    no-op).  MSIP stays set → **infinite MTOPI loop**.
///
/// Clearing stale MSIP when both channels are quiesced breaks the loop.
/// The clear is gated on ``!had_edge`` (no new edge in this call) AND
/// CLINT level == 0 (guest cleared it).  A new IPI arriving concurrently
/// sets the CLINT level bit before ``msip_pending``, so the level check
/// sees 1 and skips the clear; the edge is consumed next call.
#[inline]
pub(crate) fn sync_msip(state: &mut HartState, clint: &ConcurrentClintCtx) {
	let hid = state.mhartid as usize;
	let in_aia_active = state.imsic_m.present != 0 && state.imsic_m.eidelivery != 0;

	{
		static SEEN: AtomicU64 = AtomicU64::new(0);
		let mask: u64 = 1 << hid;
		if hid < 64 && (SEEN.load(Ordering::Relaxed) & mask) == 0 {
			SEEN.fetch_or(mask, Ordering::Relaxed);
		}
	}

	// ----- drain cross-thread MSIP channel (always) -----
	let pending_ptr = clint.msip_pending.get();
	let mut had_edge = false;
	let mut _cross = 0;
	if !pending_ptr.is_null() {
		_cross = unsafe { &*pending_ptr.add(hid) }.swap(0, Ordering::Acquire);
		if _cross != 0 {
			state.mip.fetch_or(_cross, Ordering::AcqRel);
			had_edge = true;
		}
	}

	// ----- AIA mode: edge-triggered + stale-clear -----
	//
	// In AIA mode, IMSIC IPIs route through MEIP/SEIP (not MSIP), so
	// the msip_pending edge channel is never written by IPI operations.
	// Stale-clear removes any lingering MSIP from legacy operations or
	// direct CLINT writes.  The IMSIC eip check in condition (b) is a
	// final fallback — when an IPI is still pending in IMSIC eip, MSIP
	// is not cleared (harmless since MEIP will fire first at higher
	// priority).
	if in_aia_active {
		if had_edge || hid >= clint.num_harts as usize {
			return;
		}
		/* !had_edge || hid < clint.num_hars as usize: */
		let raw = unsafe { &*clint.msip.add(hid) }.load(Ordering::Acquire);
		let no_imsic_ipi = state.imsic_m.present == 0 || state.imsic_m.eidelivery == 0 || {
			let eip0 = state.imsic_m.eip[0].load(Ordering::Acquire);
			(eip0 & ((1u32 << 3) | (1u32 << 1))) == 0
		};
		if (raw & 1) == 0 && no_imsic_ipi {
			let _msip_before = (state.mip.load(Ordering::Acquire) >> 3) & 1;
			state.mip.fetch_and(!(1 << 3), Ordering::AcqRel);
		}
		return;
	}

	// ----- legacy: level-triggered CLINT MSIP → mip.MSIP -----
	if hid >= clint.num_harts as usize {
		return;
	}
	// Level-triggered (legacy mode): mip.MSIP directly follows the CLINT
	// level bit.  Read-only — do NOT clear the level bit here.  The
	// level persists until the guest's M-mode handler writes 0 to CLINT
	// MSIP (sbi_ipi_raw_clear), which triggers the actual clear via
	// try_handle_clint_concurrent.
	let raw = unsafe { &*clint.msip.add(hid) }.load(Ordering::Acquire);
	let level_set = (raw & 1) != 0;

	if level_set {
		state.mip.fetch_or(1 << 3, Ordering::AcqRel);
		state.diag.msip_last_seen = (state.diag.msip_last_seen & !1) | (1u64);
	}
}

// ============================================================

pub(crate) fn clint_write_msip_concurrent(clint: &ConcurrentClintCtx, target: usize, val: u8) {
	if target >= clint.num_harts as usize {
		return;
	}
	// Log only non-zero writes (IPI send, not clear).
	if val & 1 != 0 {
		diag::log_line(&format!("CLINT_MSIP target=h{} set=1", target));
	}
	let p = unsafe { &*clint.msip.add(target) };
	if val & 1 != 0 {
		// Write-1: set the CLINT level bit AND atomically signal the
		// target hart via the msip_pending channel.  This avoids the
		// non-atomic RMW on ``(*states).mip`` which raced with the
		// target's sync_mtip/sync_msip operations on the same u64.
		p.fetch_or(1, Ordering::Release);
		let pending = clint.msip_pending.get();
		if !pending.is_null() {
			unsafe { &*pending.add(target) }.fetch_or(1 << 3, Ordering::Release);
		}
		// Unpark the target hart's thread — it may be sleeping in
		// wfi_spin on park_timeout.  unpark() is safe to call before
		// park(); the next park() returns immediately.
		let threads = clint.hart_threads.get();
		if !threads.is_null() && target < clint.num_harts as usize {
			unsafe { &*threads.add(target) }.unpark();
		}
	} else {
		// Write-0: clear CLINT level bit.  Also clear mip.MSIP on
		// the target
		//
		// deliver_trap_mmode clears mip.MSIP for MSI (cause 3) but
		// NOT for MEI (cause 11).  In AIA mode, MEI fires first
		// (higher priority) from the same IPI event, so mip.MSIP
		// stays set.  After sbi_ipi_raw_clear writes 0 here, the
		// stale MSIP must be cleared or the MTOPI loop in
		// sbi_trap_aia_irq() never terminates.
		p.fetch_and(0xFE, Ordering::Release);
		// A new MSIP arriving concurrently is safe: the sender set
		// msip_pending (edge channel), and the next sync_msip call
		// will drain it and re-assert mip.MSIP.
		let hart_states = clint.hart_states.get();
		if !hart_states.is_null() {
			// Shared reference — mip is AtomicU64, safe for concurrent access.
			let ts = unsafe { &*(hart_states.add(target) as *const HartState) };
			ts.mip.fetch_and(!(1 << 3), Ordering::AcqRel);
		}
	}
}

/// Try to handle a CLINT MMIO access inline, concurrent-safe version.
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
		// MSIP region
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
		// MTIMECMP region
		let target = ((offset - 0x4000) / 8) as usize;
		if target >= clint.num_harts as usize {
			return Some(0);
		}
		if is_write {
			unsafe { &*clint.mtimecmp.add(target) }.store(write_data, Ordering::Release);
			Some(0)
		} else {
			Some(unsafe { &*clint.mtimecmp.add(target) }.load(Ordering::Relaxed))
		}
	} else if offset < 0xC000 {
		// MTIME region
		if is_write {
			None // rare — fall through to Python
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
	/// Set when a hart writes MSIP=1 to a *different* hart.  The dispatch
	/// loop checks this flag and yields the current hart's slice early so
	/// the target hart can respond to the IPI within the same round.
	pub yield_for_ipi: Cell<bool>,
	/// Hart ID of the most recent cross-hart MSIP sender; this hart receives
	/// short slices so the receiver can complete IPI-triggered work (TLB
	/// flush, sync counter decrement) before the sender's spin-wait resumes.
	pub ipi_sender_hart: Cell<u8>,
	/// Remaining rounds of short slices for *ipi_sender_hart*.  Decremented
	/// once per round-robin round until zero, then the sender resumes full
	/// slices.  Reset to a fresh count each time a new MSIP is sent.
	pub ipi_sender_rounds: Cell<u8>,
	/// Back-reference to the concurrent MSIP pending array (edge-triggered
	/// cross-thread notification).  Used by ``try_handle_imsic_serial`` to
	/// wake the target hart's WFI thread when IMSIC seteipnum is written
	/// from the compressed-instruction execution path.
	pub msip_pending: Cell<*const std::sync::atomic::AtomicU64>,
	/// Back-reference to the concurrent hart thread handles for ``unpark()``
	/// wake-up from WFI park_timeout.  Populated by ``exec_compressed_concurrent``
	/// when running inside the concurrent speedup execution engine.
	pub hart_threads: Cell<*const std::thread::Thread>,
	/// Back-reference to the concurrent per-hart HartState array for
	/// cross-hart IMSIC eip manipulation.  Populated by
	/// ``exec_compressed_concurrent`` when running inside the concurrent
	/// speedup execution engine.
	pub hart_states: Cell<*const crate::state::HartState>,
}

/// Write MSIP for a target hart, updating its MIP and setting the
/// cross-hart yield flag when the target is a different hart.
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
		ts.diag.clint_msip_clr = ts.diag.clint_msip_clr.wrapping_add(1);
		return;
	}
	ts.mip.fetch_or(1 << 3, Ordering::AcqRel);
	ts.diag.clint_msip_set = ts.diag.clint_msip_set.wrapping_add(1);
	diag::log_line(&format!("CLINT_MSIP h{}->h{} set=1", current, target,));
	if target != current {
		clint.yield_for_ipi.set(true);
		clint.ipi_sender_hart.set(current as u8);
		clint.ipi_sender_rounds.set(16);
	}
}

/// Try to handle a CLINT MMIO access inline inside the speedup execution engine.
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
			t.diag.clint_mtc_wr = t.diag.clint_mtc_wr.wrapping_add(1);
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
	use crate::state::HartState;
	use std::sync::atomic::{AtomicU64, AtomicU8};

	#[test]
	fn sync_msip_sets_mip_on_level_high() {
		// When CLINT MSIP level=1, sync_msip must set mip.MSIP.
		// Level-triggered design: the level bit is NOT auto-cleared by
		// sync_msip — only the M-mode handler's write-0 to CLINT MSIP
		// clears it (matching real hardware).  The msip_pending atomic
		// channel provides independent edge-triggered delivery.
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
			timebase_hz: 0, // 单元测试禁用 clock-source 推进
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
		// mip.MSIP must NOT be cleared by sync_msip when CLINT level=0.
		// Clearing is the responsibility of deliver_trap_mmode /
		// deliver_trap_smode (RISC-V spec §3.1.15).  If sync_msip cleared
		// mip.MSIP here, the WFI wake path would lose the interrupt:
		//   wfi_sync_and_check -> sync_msip (sees level=0, clears mip.MSIP)
		//   -> check_and_deliver_interrupt_concurrent -> no trap -> deadlock.
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
			timebase_hz: 0, // 单元测试禁用 clock-source 推进
		};

		sync_msip(&mut state, &clint);

		assert_eq!(
			state.mip.load(Ordering::Acquire) & (1 << 3),
			1 << 3,
			"sync_msip must NOT clear mip.MSIP when CLINT level=0 (trap delivery handles clearing)"
		);
	}

	/// Regression: STIP must assert when ``mtime == stimecmp`` (``>=``), matching
	/// the RISC-V SSTC spec and the MTIP branch.  A strict ``>`` misses the
	/// deadline when ``wfi_check_all_idle`` fast-forwards mtime exactly onto
	/// stimecmp, leaving the hart parked with a pending-but-invisible timer.
	#[test]
	fn sync_mtip_sets_stip_at_deadline_boundary() {
		let mut state: HartState = unsafe { std::mem::zeroed() };
		state.mhartid = 0;
		state.stimecmp = 100;

		let mtime = AtomicU64::new(100); // mtime == stimecmp (精确截止)
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
}
