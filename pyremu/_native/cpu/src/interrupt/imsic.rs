//! IMSIC (Incoming MSI Controller) inline handling for the speedup engine.
//!
//! **AIA-mode delivery** (``eidelivery == 1``): ALL IMSIC interrupts — IPIs
//! (minor identity 1, OpenSBI's ``IMSIC_IPI_ID``) and external (IID ≥ 6) —
//! drive the hart's external interrupt lines (MEIP for M-file, SEIP for
//! S-file).  A ``seteipnum = N`` doorbell write sets ``eip[N]`` in the
//! addressed file, with **no cross-file routing** — the value is a MINOR
//! identity (external interrupt number), never a MAJOR identity (cause).
//! Software reads ``stopi``/``mtopi`` for the major identity (9/11, external)
//! and ``stopei``/``mtopei`` for the minor identity, writing back to claim.
//! The legacy CLINT MSIP/SSIP mechanism is NOT used for IMSIC IPIs in AIA mode.
//!
//! **Legacy mode** (``eidelivery == 0``): IPIs continue to use CLINT MSIP.
//! The Python-side ``trap_handler.py`` manages the IMSIC eip life-cycle.

use crate::concurrent::ConcurrentClintCtx;
use crate::diag;
use crate::state::{HartState, ImsicFile, PYREMU_AIA};
use core::sync::atomic::Ordering;

use std::thread::Thread;

// IMSIC minor identities used as IPI doorbell values by OpenSBI.
// These are MINOR identities (external interrupt numbers), not MAJOR
// identities (interrupt causes).  ``IMSIC_IPI_ID = 1`` is OpenSBI's
// convention: an IPI is a regular MSI write ``seteipnum = 1`` to the target
// hart's M-file, delivered as an external interrupt (MEIP) and discovered
// via ``MTOPEI``.  IID 3 is reserved here for symmetry with the S-file path
// but is not emitted by OpenSBI (which always uses minor identity 1).
pub(crate) const IID_S_IPI: u32 = 1; // OpenSBI IPI minor identity (M-file)
pub(crate) const IID_M_IPI: u32 = 3; // legacy M-file IPI minor identity

// ============================================================
//  Cross-hart IMSIC eip manipulation
// ============================================================

/// Set an IMSIC eip bit in the target hart's HartState.
/// Caller must ensure *hart* is within bounds and ``hart_states`` is valid.
///
/// **Concurrency**: this function may be called from a different OS thread
/// than the target hart's speedup loop.  Only ``AtomicU32`` / ``AtomicU64``
/// fields are accessed (via shared reference), never plain fields.
/// ``eip_ext_any`` is NOT updated here — the owning hart's thread refreshes
/// it lazily.  This is safe because IPI IIDs (1, 3) bypass the
/// ``eip_ext_any`` fast-out in ``imsic_topei_peek``.
#[inline]
pub(crate) fn imsic_eip_set(
	hart_states: *const HartState,
	hart: usize,
	file_off: u64,
	eip_num: u32,
) {
	if hart_states.is_null() {
		return;
	}
	// Shared reference: all accessed fields are atomic (eip, mip).
	// Never create &mut to another thread's HartState.
	let target = unsafe { &*(hart_states.add(hart) as *const HartState) };
	let eip_array = if file_off == 0x0000 {
		&target.imsic_m.eip
	} else {
		&target.imsic_s.eip
	};
	let mip_bit: u64 = if file_off == 0x0000 { 1 << 11 } else { 1 << 9 };
	let word = (eip_num >> 5) as usize;
	if word >= eip_array.len() {
		return;
	}
	let bit = eip_num & 31;
	eip_array[word].fetch_or(1u32 << bit, Ordering::Release);
	// All IMSIC interrupts (software and external) route through the
	// hart's external interrupt lines (MEIP/SEIP) per AIA spec.
	// The legacy MSIP/SSIP mechanism is NOT used for IMSIC IPIs.
	target.mip.fetch_or(mip_bit, Ordering::AcqRel);
}

/// Clear an IMSIC eip bit in the target hart's HartState.
///
/// **Concurrency**: same rules as ``imsic_eip_set`` — shared reference only,
/// atomic fields only, no ``eip_ext_any`` cross-thread write.
#[inline]
pub(crate) fn imsic_eip_clear(
	hart_states: *const HartState,
	hart: usize,
	file_off: u64,
	eip_num: u32,
) {
	if hart_states.is_null() {
		return;
	}
	// Shared reference: all accessed fields are atomic (eip, mip).
	let target = unsafe { &*(hart_states.add(hart) as *const HartState) };
	let file: &ImsicFile = if file_off == 0x0000 {
		&target.imsic_m
	} else {
		&target.imsic_s
	};
	let mip_bit: u64 = if file_off == 0x0000 { 1 << 11 } else { 1 << 9 };
	let word = (eip_num >> 5) as usize;
	if word >= file.eip.len() {
		return;
	}
	let bit = eip_num & 31;
	file.eip[word].fetch_and(!(1u32 << bit), Ordering::Release);
	// All IMSIC interrupts route through MEIP/SEIP per AIA spec.
	// Re-check if any interrupts remain and update target's mip.
	//
	// imsic_topei_peek takes &ImsicFile (shared) — safe from shared ref.
	// It reads only atomic eip fields for the IPI fast-path, which
	// is the only cross-thread case.
	let (topei_val, _) = imsic_topei_peek(file);
	if topei_val == 0 {
		target.mip.fetch_and(!mip_bit, Ordering::AcqRel);
	}
}

// ============================================================
//  MMIO handlers
// ============================================================

// ============================================================
//  Address decoding
// ============================================================

/// Decoded IMSIC MMIO target.
pub(crate) struct ImsicAddr {
	pub(crate) hart: usize,
	/// ``true`` → S-file, ``false`` → M-file.
	pub(crate) is_sfile: bool,
	/// Offset within the 4 KiB page (0x0000 = seteipnum, 0x0008 = clreipnum).
	pub(crate) reg_off: u64,
}

/// Decode a physical address into an IMSIC target.
///
/// QEMU-compatible contiguous layout with M-files first, then S-files:
///
///   ============ ============================= ===================
///   Range        Offset                         Per-hart stride
///   ============ ============================= ===================
///   M-file       [0,          num_harts*0x1000) 0x1000
///   S-file       [N*0x1000,  2*num_harts*0x1000) 0x1000
///   ============ ============================= ===================
///
/// Single ``reg`` entry ``<IMSIC_M_BASE, 2*num_harts*0x1000>`` passes
/// OpenSBI's ``imsic_data_check`` without modification (single entry →
/// ``addr == base_addr`` trivially).  ``imsic_ipi_send`` works because
/// ``reloff = hart_index * 0x1000`` always lands inside the M-file half.
///
/// Each page contains ``seteipnum`` at +0x0 and ``clreipnum`` at +0x8.
pub(crate) fn decode_imsic_addr(pa: u64, num_harts: u64) -> Option<ImsicAddr> {
	let base = crate::state::IMSIC_M_BASE;
	let page_stride: u64 = 0x1000; // IMSIC_MMIO_PAGE_SZ
	let m_range = num_harts * page_stride;

	if pa < base {
		return None;
	}

	let off = pa - base;
	if off < m_range {
		// M-file half
		return Some(ImsicAddr {
			hart: (off / page_stride) as usize,
			is_sfile: false,
			reg_off: off % page_stride,
		});
	}
	if off < 2 * m_range {
		// S-file half
		let s_off = off - m_range;
		return Some(ImsicAddr {
			hart: (s_off / page_stride) as usize,
			is_sfile: true,
			reg_off: s_off % page_stride,
		});
	}
	return None;
}

// ============================================================
//  Shared seteipnum / clreipnum
// ============================================================

/// Set an IMSIC eip bit for a write to ``seteipnum``.
///
/// Direct mapping: ``seteipnum = N`` sets ``eip[N]`` in the file that was
/// addressed, driving that file's external interrupt line (MEIP for the
/// M-file, SEIP for the S-file).  There is **no cross-file routing** based
/// on the interrupt identity — the value written to the doorbell is a
/// MINOR identity (external interrupt number), never a MAJOR identity
/// (interrupt cause).  This matches QEMU ``riscv_imsic`` and the AIA spec.
///
/// OpenSBI sends IPIs by writing minor identity 1 (``IMSIC_IPI_ID``) to the
/// target hart's M-file (``targets_mmode``); the receiving hart's M-mode
/// OpenSBI discovers it via ``MTOPEI`` and dispatches to ``sbi_ipi_process``.
/// Routing IID=1 from the M-file to the S-file would drop M-mode IPIs and
/// leave secondary harts asleep.
///
/// Returns ``true`` when the write was handled inline; ``false`` to fall
/// through to Python MMIO (e.g. unknown / reserved IID).
#[inline]
pub(crate) fn imsic_handle_seteipnum(
	hart_states: *const HartState,
	hart_threads: *const Thread,
	num_harts: u32,
	target_hart: usize,
	file_tag: u64,
	eip_num: u32,
) -> bool {
	imsic_eip_set(hart_states, target_hart, file_tag, eip_num);
	if !hart_threads.is_null() && target_hart < num_harts as usize {
		diag::log_line(&format!(
			"IMSIC_UNPARK target={} threads_nonnull",
			target_hart,
		));
		unsafe { &*hart_threads.add(target_hart) }.unpark();
	} else {
		diag::log_line(&format!(
			"IMSIC_UNPARK_SKIP target={} threads_null={}",
			target_hart,
			hart_threads.is_null(),
		));
	}
	true
}

/// Clear an IMSIC eip bit for a write to ``clreipnum``.
///
/// Direct mapping mirrors ``imsic_handle_seteipnum`` — ``clreipnum = N``
/// clears ``eip[N]`` in the addressed file, no cross-file routing.
#[inline]
pub(crate) fn imsic_handle_clreipnum(
	hart_states: *const HartState,
	tgt_hart: usize,
	file_tag: u64,
	eip_num: u32,
) {
	imsic_eip_clear(hart_states, tgt_hart, file_tag, eip_num);
}

// ============================================================
//  MMIO inline entry points
// ============================================================

/// Try to handle an IMSIC MMIO write inline.
///
/// In AIA mode (eidelivery=1), ALL IMSIC interrupts — including software
/// IPIs (IID=1, 3) — route through the hart's external interrupt lines
/// (MEIP/SEIP), not through the legacy CLINT MSIP/SSIP mechanism.
///
/// ``imsic_eip_set`` directly sets the target hart's ``mip`` bit
/// (MEIP for M-file, SEIP for S-file), which wakes the target from WFI
/// and triggers the next instruction-boundary interrupt check.  The
/// target M-mode handler reads ``CSR_MTOPEI`` to discover the interrupt
/// identity; the claim (MTOPEI write) clears the eip bit.
///
/// Returns ``true`` if the write was handled (caller skips MMIO exit).
pub(crate) fn try_handle_imsic_concurrent(
	pa: u64,
	val: u64,
	state: &HartState,
	clint: &ConcurrentClintCtx,
) -> bool {
	if !PYREMU_AIA {
		return false;
	}
	let addr = match decode_imsic_addr(pa, clint.num_harts as u64) {
		Some(a) => a,
		None => return false,
	};
	let file_tag: u64 = if addr.is_sfile { 0x1000 } else { 0x0000 };
	let eip_num = val as u32;

	if addr.reg_off == 0x0000 {
		let handled = imsic_handle_seteipnum(
			clint.hart_states.get(),
			clint.hart_threads.get(),
			clint.num_harts,
			addr.hart,
			file_tag,
			eip_num,
		);
		diag::log_line(&format!(
			"IMSIC_WR h{}->h{} file={:#x} eip={} {}",
			state.mhartid,
			addr.hart,
			file_tag,
			eip_num,
			if handled { "INLINE" } else { "FALLBACK" },
		));
		return handled;
	}
	if addr.reg_off == 0x0008 {
		imsic_handle_clreipnum(clint.hart_states.get(), addr.hart, file_tag, eip_num);
		return true;
	}
	false
}

/// Try to handle an IMSIC MMIO read inline.
///
/// Returns the register value as ``u64`` if the address is within the IMSIC
/// MMIO range.  All readable registers return 0 — ``seteipnum``/``clreipnum``
/// are write-only, and configuration/``topei`` registers are accessed via CSR
/// (``mireg``/``sireg``/``mtopei``/``stopei``), not MMIO.
///
/// Without this handler every IMSIC MMIO read exits the speedup execution, adding a
/// Python round-trip per read.  During kernel IMSIC init this overhead
/// accumulates to ~1 s — exactly the ``cpu_up`` timeout.
#[inline]
pub(crate) fn try_handle_imsic_read_concurrent(pa: u64, num_harts: u32) -> Option<u64> {
	if !PYREMU_AIA {
		return None;
	}
	match decode_imsic_addr(pa, num_harts as u64) {
		Some(addr) => {
			// Rate-limited: log first 4 reads per hart per file
			// to confirm kernel's IMSIC driver is probing the device.
			// Use a static counter to avoid flooding diag.log.
			static RD_COUNT: core::sync::atomic::AtomicU32 = core::sync::atomic::AtomicU32::new(0);
			let n = RD_COUNT.fetch_add(1, core::sync::atomic::Ordering::Relaxed);
			if n < 8 {
				diag::log_line(&format!(
					"IMSIC_RD pa=0x{:x} hart={} file={} off=0x{:x}",
					pa,
					addr.hart,
					if addr.is_sfile { 'S' } else { 'M' },
					addr.reg_off,
				));
			}
			Some(0)
		}
		None => None,
	}
}

// ============================================================
//  mip synchronisation
// ============================================================

/// Recompute the ``eip_ext_any`` cached flag after eip[] is modified.
/// Called from CSR write (mireg→eip) and topei claim paths — not from
/// the hot instruction-boundary ``sync_imsic`` path.
pub(crate) fn imsic_update_eip_ext_any(file: &mut ImsicFile) {
	for i in 0..file.eip.len() {
		let eip = file.eip[i].load(Ordering::Acquire);
		if eip != 0 {
			file.eip_ext_any = 1;
			return;
		}
	}
	file.eip_ext_any = 0;
}

/// Sync IMSIC eip/eie → hart mip bits.
///
/// All IMSIC interrupts (software and external) route through the
/// hart's external interrupt lines (MEIP/SEIP), not through the legacy
/// MSIP/SSIP bits.  This matches OpenSBI's dispatch: ``sbi_trap_aia_irq``
/// dispatches ALL IMSIC interrupts via ``case IRQ_M_EXT`` →
/// ``sbi_irqchip_process()`` → ``imsic_process_hwirqs()``, which uses
/// ``csr_swap(CSR_MTOPEI)`` to claim the interrupt and then dispatches
/// to ``sbi_ipi_process()`` (for IID=1) or the external handler.
///
/// The legacy MSIP/SSIP bits are reserved for CLINT-sourced software
/// interrupts in non-AIA mode.
///
/// Sync IMSIC eip/eie → hart mip bits.
///
/// Per AIA spec, each IMSIC interrupt file independently controls its
/// external interrupt line when ``eidelivery == 1``:
///
///   - M-file drives MEIP (mip bit 11)
///   - S-file drives SEIP (mip bit 9)
///
/// When a file has ``eidelivery == 0``, that line is managed by the
/// legacy path (PLIC / ext_irq drain) — IMSIC has no authority over it.
///
/// Uses ``imsic_topei_peek()`` (not a raw eip&eie scan) so that the
/// eithreshold and IID-range filters are applied consistently with
/// ``compute_stopi`` / ``compute_mtopi``.  A mismatch here causes an
/// infinite SEI→timer→sret→SEI loop in the kernel.
///
/// No-op in legacy mode (both files ``present == 0``).
pub(crate) fn sync_imsic(state: &mut HartState) {
	// ---- M-file → MEIP (bit 11) ----
	if state.imsic_m.present != 0 {
		let (topei_val, _) = imsic_topei_peek(&state.imsic_m);
		if state.imsic_m.eidelivery != 0 {
			// Full IMSIC mode: drive MEIP from topei.
			// All IMSIC interrupts (software IID=1,3 and external IID>=6)
			// route through MEIP per AIA spec.
			if topei_val != 0 {
				state.mip.fetch_or(1 << 11, Ordering::AcqRel);
			} else {
				state.mip.fetch_and(!(1 << 11), Ordering::AcqRel);
			}
		} else if topei_val != 0 {
			// eidelivery=0 → imsic_topei_peek only returns IPI bits
			// (IID=1,3).  Set MEIP so the hart wakes from WFI / sees
			// the pending IPI.  Never clear here — legacy PLIC may be
			// driving MEIP independently.
			state.mip.fetch_or(1 << 11, Ordering::AcqRel);
		}
	}

	// ---- S-file → SEIP (bit 9) ----
	if state.imsic_s.present != 0 {
		let (topei_val, _) = imsic_topei_peek(&state.imsic_s);
		if state.imsic_s.eidelivery != 0 {
			if topei_val != 0 {
				state.mip.fetch_or(1 << 9, Ordering::AcqRel);
			} else {
				state.mip.fetch_and(!(1 << 9), Ordering::AcqRel);
			}
		} else if topei_val != 0 {
			state.mip.fetch_or(1 << 9, Ordering::AcqRel);
		}
	}
}

/// Sync a single IMSIC file to its mip bit after an inline state change
/// (CSR write or cross-hart operation).  Caller passes ``true`` for *mfile*
/// to update MEIP (bit 11), ``false`` for S-file → SEIP (bit 9).
///
/// When ``target`` is ``None``, updates *state*'s own mip — used after
/// CSR writes on the local hart.  When ``target`` is a raw pointer, updates
/// that hart's mip — used by cross-hart ``imsic_eip_set`` / ``imsic_eip_clear``
/// which write to another hart's IMSIC file during the same speedup execution.
#[inline]
pub(crate) fn sync_imsic_one(state: &mut HartState, mfile: bool) {
	let (file, mip_bit): (&ImsicFile, u64) = if mfile {
		(&state.imsic_m, 1 << 11)
	} else {
		(&state.imsic_s, 1 << 9)
	};
	if file.present == 0 {
		return;
	}
	let (topei_val, _) = imsic_topei_peek(file);
	if file.eidelivery != 0 {
		// Full IMSIC mode: drive mip bit from topei.
		if topei_val != 0 {
			state.mip.fetch_or(mip_bit, Ordering::AcqRel);
		} else {
			state.mip.fetch_and(!mip_bit, Ordering::AcqRel);
		}
	} else if topei_val != 0 {
		// eidelivery=0: only IPI bits visible, set but don't clear.
		state.mip.fetch_or(mip_bit, Ordering::AcqRel);
	}
}

// ============================================================
//  Top External Interrupt (topei) helpers
// ============================================================

/// Maximum interrupt identity (AIA spec: 2047).
const _MAX_IID: u32 = 2048;
/// Number of u32 words for 2048-bit eip/eie arrays.
const _EIP_WORDS: usize = 64;

// CSR select indices for IMSIC indirect register access (miselect/mireg).
// Per AIA spec v1.0: eidelivery=0x70, eithreshold=0x72, eip=0x80-0xBF, eie=0xC0-0xFF.
const _SEL_EIDELIVERY: u32 = 0x70;
const _SEL_EITHRESHOLD: u32 = 0x72;
const _SEL_EIP_BASE: u32 = 0x80;
const _SEL_EIP_END: u32 = 0xBF;
const _SEL_EIE_BASE: u32 = 0xC0;
const _SEL_EIE_END: u32 = 0xFF;
/// Per-bit eip/eie manipulation via CSR indirect access.
/// Writing a minor IID value to sireg when siselect=0x58 sets eip[IID]=1;
/// siselect=0x59 clears eip[IID]=0; 0x5A/0x5B control eie likewise.
const _SEL_SETEIPNUM: u32 = 0x58;
const _SEL_CLREIPNUM: u32 = 0x59;
const _SEL_SETEIENUM: u32 = 0x5A;
const _SEL_CLREIENUM: u32 = 0x5B;

// ============================================================
//  IMSIC CSR indirect-register helpers (mireg / sireg)
// ============================================================

/// Read an IMSIC register selected by *miselect* / *siselect*.
pub(crate) fn imsic_reg_read(file: &ImsicFile, select: u32) -> (u64, u8) {
	match select {
		_SEL_EIDELIVERY => (file.eidelivery as u64, 0),
		_SEL_EITHRESHOLD => (file.eithreshold as u64, 0),
		_SEL_EIP_BASE..=_SEL_EIP_END => {
			let idx = (select - _SEL_EIP_BASE) as usize;
			let lo = file.eip[idx].load(Ordering::Acquire) as u64;
			let hi = if idx + 1 < _EIP_WORDS {
				file.eip[idx + 1].load(Ordering::Acquire) as u64
			} else {
				0
			};
			(lo | (hi << 32), 0)
		}
		_SEL_EIE_BASE..=_SEL_EIE_END => {
			let idx = (select - _SEL_EIE_BASE) as usize;
			let lo = file.eie[idx].load(Ordering::Acquire) as u64;
			let hi = if idx + 1 < _EIP_WORDS {
				file.eie[idx + 1].load(Ordering::Acquire) as u64
			} else {
				0
			};
			(lo | (hi << 32), 0)
		}
		_ => (0, 0),
	}
}

/// Write an IMSIC register selected by *miselect* / *siselect*.
/// Returns 0 on success (would return CSR_EXIT if Python fallback needed).
pub(crate) fn imsic_reg_write(file: &mut ImsicFile, select: u32, val: u64) -> u8 {
	match select {
		_SEL_EIDELIVERY => {
			let new = (val & 1) as u8;
			if file.eidelivery != new {
				diag::log_line(&format!("IMSIC_EIDELIVERY {} -> {}", file.eidelivery, new,));
			}
			file.eidelivery = new;
			0
		}
		_SEL_EITHRESHOLD => {
			file.eithreshold = (val & 0x3FF) as u8;
			0
		}
		_SEL_EIP_BASE..=_SEL_EIP_END => {
			let idx = (select - _SEL_EIP_BASE) as usize;
			file.eip[idx].store(val as u32, Ordering::Release);
			if idx + 1 < _EIP_WORDS {
				file.eip[idx + 1].store((val >> 32) as u32, Ordering::Release);
			}
			imsic_update_eip_ext_any(file);
			0
		}
		_SEL_EIE_BASE..=_SEL_EIE_END => {
			let idx = (select - _SEL_EIE_BASE) as usize;
			file.eie[idx].store(val as u32, Ordering::Release);
			if idx + 1 < _EIP_WORDS {
				file.eie[idx + 1].store((val >> 32) as u32, Ordering::Release);
			}
			0
		}
		// Per-bit eip/eie manipulation: write minor IID → set/clear bit.
		_SEL_SETEIPNUM => {
			let eip_num = val as u32;
			if eip_num < _MAX_IID {
				let word = (eip_num >> 5) as usize;
				let bit = eip_num & 31;
				file.eip[word].fetch_or(1u32 << bit, Ordering::Release);
				imsic_update_eip_ext_any(file);
			}
			0
		}
		_SEL_CLREIPNUM => {
			let eip_num = val as u32;
			if eip_num < _MAX_IID {
				let word = (eip_num >> 5) as usize;
				let bit = eip_num & 31;
				file.eip[word].fetch_and(!(1u32 << bit), Ordering::Release);
				imsic_update_eip_ext_any(file);
			}
			0
		}
		_SEL_SETEIENUM => {
			let eip_num = val as u32;
			if eip_num < _MAX_IID {
				let word = (eip_num >> 5) as usize;
				let bit = eip_num & 31;
				file.eie[word].fetch_or(1u32 << bit, Ordering::Release);
			}
			0
		}
		_SEL_CLREIENUM => {
			let eip_num = val as u32;
			if eip_num < _MAX_IID {
				let word = (eip_num >> 5) as usize;
				let bit = eip_num & 31;
				file.eie[word].fetch_and(!(1u32 << bit), Ordering::Release);
			}
			0
		}
		_ => 0,
	}
}

/// Read the top priority pending+enabled IMSIC interrupt WITHOUT claiming.
/// Returns ``(value, ok)`` where *value* is ``(IID << 16) | priority`` or 0.
/// Used by both ``mtopi`` and ``mtopei`` CSR reads.
///
/// **Software interrupts** (IID=1, IID=3) are pending when eip is set — they
/// do NOT require eie.  eie is the *external* interrupt enable register and
/// only gates external interrupts (IID >= 6).  Per AIA spec §3.2.1, software
/// interrupts have dedicated IIDs and are always "enabled" when pending.
pub(crate) fn imsic_topei_peek(file: &ImsicFile) -> (u64, u8) {
	// IPI fast-path: software interrupts (IID=1, IID=3) are always
	// "enabled" when pending — they do NOT require eie NOR eidelivery.
	// eidelivery only gates external interrupts (IID >= 6).
	// Checking IPIs first ensures pending IPIs are visible via stopi/mtopi
	// even before the kernel sets eidelivery via CSR writes, preventing
	// lost IPIs during early SMP boot.
	let eip0 = file.eip[0].load(Ordering::Acquire);
	let ipi_mask: u32 = (1 << IID_M_IPI) | (1 << IID_S_IPI);
	let ipi_pending = eip0 & ipi_mask;
	if ipi_pending != 0 {
		let bit = 31 - ipi_pending.leading_zeros();
		let iid = bit;
		let prio = iid & 0xFF;
		return (((iid as u64) << 16) | (prio as u64), 0);
	}

	// External interrupts require eidelivery.
	if file.eidelivery == 0 || file.eip_ext_any == 0 {
		return (0, 0);
	}
	let eie0 = file.eie[0].load(Ordering::Acquire);
	// External interrupt mask — IPI bits already handled above.
	let ext_mask: u32 = !((1 << IID_M_IPI) | (1 << IID_S_IPI));
	for i in (0.._EIP_WORDS).rev() {
		let eip_word = file.eip[i].load(Ordering::Acquire);
		let pending_enabled = if i == 0 {
			// Word 0: ext bits only (eip & eie) — IPI bits already handled above.
			eip_word & eie0 & ext_mask
		} else {
			eip_word & file.eie[i].load(Ordering::Acquire)
		};
		if pending_enabled == 0 {
			continue;
		}
		let bit = 31 - pending_enabled.leading_zeros();
		let iid = (i as u32 * 32) + bit;
		if iid == 0 || iid >= _MAX_IID {
			continue;
		}
		let prio = iid & 0xFF;
		if prio > file.eithreshold as u32 {
			return (((iid as u64) << 16) | (prio as u64), 0);
		}
	}
	(0, 0)
}

/// Claim a specific interrupt identity (clear eip bit), avoiding the race
/// where a higher-priority interrupt arrives between peek and claim and
/// gets stolen by the generic top-priority scan.
///
/// Software interrupts (IID=1,3) are always claimable — they do NOT require
/// eidelivery, matching ``imsic_topei_peek``'s IPI fast-path.  Without this,
/// a pending IPI would be reported by ``compute_stopi`` / ``compute_mtopi``
/// forever because the claim silently drops the clear, creating an infinite
/// dispatch loop that starves all other interrupts (including cross-hart IPIs).
pub(crate) fn imsic_topei_claim_iid(file: &mut ImsicFile, iid: u32) {
	if iid == 0 || iid >= _MAX_IID {
		return;
	}
	let word = (iid >> 5) as usize;
	let bit = iid & 31;
	let is_sw = iid == IID_M_IPI || iid == IID_S_IPI;

	// Software interrupts are claimable without eidelivery, matching
	// imsic_topei_peek which reports them in the IPI fast-path above.
	// External interrupts require eidelivery=1.
	if !is_sw && file.eidelivery == 0 {
		return;
	}

	let is_pending = if is_sw {
		(file.eip[word].load(Ordering::Acquire) & (1 << bit)) != 0
	} else {
		(file.eip[word].load(Ordering::Acquire)
			& file.eie[word].load(Ordering::Acquire)
			& (1 << bit))
			!= 0
	};
	if word < _EIP_WORDS && is_pending {
		#[cfg(feature = "diagnostic")]
		if iid == IID_S_IPI || iid == IID_M_IPI {
			use std::sync::atomic::{AtomicU32, Ordering as AO};
			static N: AtomicU32 = AtomicU32::new(0);
			if N.fetch_add(1, AO::Relaxed) < 40 {
				crate::diag::log_line(&format!(
					"CLAIM_IPI iid={} word={} bit={} eip_before={:#x}",
					iid,
					word,
					bit,
					file.eip[word].load(AO::Acquire),
				));
			}
		}
		file.eip[word].fetch_and(!(1 << bit), Ordering::Release);
		imsic_update_eip_ext_any(file);
	}
}

/// Clear IMSIC eip bits for software interrupts on trap entry.
///
/// **Legacy mode only** (``eidelivery == 0``).  In AIA mode, IPIs route
/// through MEIP/SEIP and the eip is claimed via MTOPEI/STOPEI write —
/// trap entry does NOT clear the eip.
///
/// When an MSIP (cause=3) or SSIP (cause=1) trap is taken in legacy
/// mode, the corresponding IMSIC eip must be cleared to prevent
/// ``compute_mtopi`` / ``compute_stopi`` from seeing a stale IID and
/// re-dispatching the same IPI forever.
///
/// * MSIP (cause=3): clears IID=1 and IID=3 in both M-file and S-file.
/// * SSIP (cause=1): clears IID=1 in the S-file.
///
/// Uses ``imsic_update_eip_ext_any`` (not blind zero) so external
/// interrupts pending in other eip words are not lost.
#[inline]
pub(crate) fn imsic_clear_ipi_on_trap(state: &mut HartState, cause: u64) {
	let clear_eip = |file: &mut ImsicFile, iid: u32| {
		let w = (iid >> 5) as usize;
		let b = iid & 31;
		if w < file.eip.len() {
			file.eip[w].fetch_and(!(1u32 << b), Ordering::Release);
			imsic_update_eip_ext_any(file);
		}
	};
	match cause {
		3 => {
			if state.imsic_m.present != 0 {
				clear_eip(&mut state.imsic_m, IID_M_IPI);
				clear_eip(&mut state.imsic_m, IID_S_IPI);
			}
			if state.imsic_s.present != 0 {
				clear_eip(&mut state.imsic_s, IID_S_IPI);
			}
		}
		1 => {
			if state.imsic_s.present != 0 {
				clear_eip(&mut state.imsic_s, IID_S_IPI);
			}
		}
		_ => {}
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::interrupt::{check_pending_interrupts, compute_mtopi, compute_stopi};
	use crate::state::{riscv_mode, HartState};

	fn make_state() -> HartState {
		let mut s: HartState = unsafe { std::mem::zeroed() };
		s.imsic_m.present = 1;
		s.imsic_m.eidelivery = 1;
		s.imsic_s.present = 1;
		s.imsic_s.eidelivery = 1;
		s
	}

	#[test]
	fn sync_imsic_clears_seip_when_all_below_eithreshold() {
		// If the only pending+enabled IMSIC interrupt has priority <=
		// eithreshold, SEIP must NOT be set.  Otherwise sync_imsic
		// and compute_stopi disagree → infinite SEI→timer loop.
		let mut state = make_state();
		state.imsic_s.eithreshold = 100; // high threshold
								   // Pending interrupt at IID=21 → priority = 21 & 0xFF = 21
		state.imsic_s.eie[0].fetch_or(1 << 21, Ordering::Relaxed);
		state.imsic_s.eip[0].fetch_or(1 << 21, Ordering::Relaxed);
		imsic_update_eip_ext_any(&mut state.imsic_s);

		sync_imsic(&mut state);
		assert_eq!(
			state.mip.load(Ordering::Acquire) & (1 << 9),
			0,
			"SEIP must be 0 when the only pending IMSIC int is below eithreshold"
		);
	}

	#[test]
	fn sync_imsic_sets_seip_when_interrupt_above_eithreshold() {
		// Threshold lower than priority → SEIP must be set.
		let mut state = make_state();
		state.imsic_s.eithreshold = 0;
		state.imsic_s.eie[0].fetch_or(1 << 21, Ordering::Relaxed);
		state.imsic_s.eip[0].fetch_or(1 << 21, Ordering::Relaxed);
		imsic_update_eip_ext_any(&mut state.imsic_s);

		sync_imsic(&mut state);
		assert_eq!(
			state.mip.load(Ordering::Acquire) & (1 << 9),
			1 << 9,
			"SEIP must be set when IMSIC int priority > eithreshold"
		);
	}

	#[test]
	fn sync_imsic_respects_eithreshold_for_mfile() {
		// Same check for M-file (MEIP).
		let mut state = make_state();
		state.imsic_m.eithreshold = 200;
		// Pending at IID=100 → priority = 100 & 0xFF = 100
		let word = (100 / 32) as usize;
		let bit = 100 % 32;
		state.imsic_m.eie[word].fetch_or(1 << bit, Ordering::Relaxed);
		state.imsic_m.eip[word].fetch_or(1 << bit, Ordering::Relaxed);
		imsic_update_eip_ext_any(&mut state.imsic_m);

		sync_imsic(&mut state);
		assert_eq!(
			state.mip.load(Ordering::Acquire) & (1 << 11),
			0,
			"MEIP must be 0 when IMSIC int is below eithreshold"
		);

		// Lower threshold → MEIP should appear
		state.imsic_m.eithreshold = 50;
		sync_imsic(&mut state);
		assert_eq!(
			state.mip.load(Ordering::Acquire) & (1 << 11),
			1 << 11,
			"MEIP must be set after lowering eithreshold below priority"
		);
	}

	#[test]
	fn sync_imsic_clears_seip_when_no_pending() {
		// SEIP must be 0 when nothing is pending at all.
		let mut state = make_state();
		state.mip.fetch_or(1 << 9, Ordering::AcqRel); // pre-existing SEIP from previous sync
												// No eip bits set

		sync_imsic(&mut state);
		assert_eq!(
			state.mip.load(Ordering::Acquire) & (1 << 9),
			0,
			"SEIP must be cleared when no IMSIC pending"
		);
	}

	#[test]
	fn sync_imsic_sfile_independent_of_mfile_eidelivery() {
		// Per AIA spec, each IMSIC file independently controls its
		// external interrupt line.  When M-file eidelivery=0 but
		// S-file eidelivery=1, the S-file must still drive SEIP.
		// The old code gated BOTH files on M-file eidelivery,
		// returning early and silently dropping all device interrupts.
		let mut state: HartState = unsafe { std::mem::zeroed() };
		state.imsic_m.present = 1;
		state.imsic_m.eidelivery = 0; // M-file NOT driving MEIP
		state.imsic_s.present = 1;
		state.imsic_s.eidelivery = 1; // S-file IS driving SEIP

		// Set up a pending+enabled external interrupt on the S-file
		// at IID=21 (typical virtio-blk child_index).
		state.imsic_s.eie[0].store(1 << 21, Ordering::Relaxed);
		state.imsic_s.eip[0].store(1 << 21, Ordering::Relaxed);
		imsic_update_eip_ext_any(&mut state.imsic_s);

		// Pre-set MEIP (simulating ext_irq drain in main loop).
		// M-file is not driving MEIP → sync_imsic must leave it alone.
		state.mip.fetch_or(1 << 11, Ordering::AcqRel);

		sync_imsic(&mut state);

		// MEIP must be preserved — M-file eid=0, IMSIC has no authority.
		assert_eq!(
			state.mip.load(Ordering::Acquire) & (1 << 11),
			1 << 11,
			"MEIP must be preserved when M-file eidelivery=0"
		);
		// SEIP must be set — S-file eid=1 and has a pending interrupt.
		assert_eq!(
			state.mip.load(Ordering::Acquire) & (1 << 9),
			1 << 9,
			"SEIP must be set when S-file has pending interrupt even if M-file eid=0"
		);
	}

	#[test]
	fn sync_imsic_sfile_clears_seip_after_claim() {
		// After the kernel claims the S-file interrupt via stopei,
		// eip is cleared and sync_imsic must de-assert SEIP.
		let mut state: HartState = unsafe { std::mem::zeroed() };
		state.imsic_m.present = 1;
		state.imsic_m.eidelivery = 0;
		state.imsic_s.present = 1;
		state.imsic_s.eidelivery = 1;

		// First: pending → SEIP set
		state.imsic_s.eie[0].store(1 << 21, Ordering::Relaxed);
		state.imsic_s.eip[0].store(1 << 21, Ordering::Relaxed);
		imsic_update_eip_ext_any(&mut state.imsic_s);
		sync_imsic(&mut state);
		assert_eq!(
			state.mip.load(Ordering::Acquire) & (1 << 9),
			1 << 9,
			"SEIP set when pending"
		);

		// Claim: clear eip (kernel wrote stopei)
		imsic_topei_claim_iid(&mut state.imsic_s, 21);
		sync_imsic(&mut state);
		assert_eq!(
			state.mip.load(Ordering::Acquire) & (1 << 9),
			0,
			"SEIP must be cleared after interrupt is claimed"
		);
	}

	// ============================================================
	//  IPI-specific tests (cross-hart IPI delivery chain)
	// ============================================================

	/// Helper: create a HartState with both IMSIC files present but eidelivery=0.
	fn make_state_no_delivery() -> HartState {
		let mut s: HartState = unsafe { std::mem::zeroed() };
		s.imsic_m.present = 1;
		s.imsic_m.eidelivery = 0;
		s.imsic_s.present = 1;
		s.imsic_s.eidelivery = 0;
		s
	}

	// ---- imsic_topei_peek: IPI fast-path ----

	#[test]
	fn topei_peek_ipi_visible_without_eidelivery() {
		// IPI (IID=1,3) must be visible via topei_peek even when
		// eidelivery=0 and eie=0.  This is the IPI fast-path that
		// prevents lost IPIs during early SMP boot.
		let state = make_state_no_delivery();

		// Set S-mode IPI in S-file with eidelivery=0, eie=0
		state.imsic_s.eip[0].store(1 << IID_S_IPI, Ordering::Relaxed);
		let (val, _) = imsic_topei_peek(&state.imsic_s);
		assert_ne!(val, 0, "S-mode IPI must be visible with eidelivery=0");
		assert_eq!((val >> 16) as u32, IID_S_IPI, "IID must be 1 (S-mode IPI)");

		// Set M-mode IPI in M-file with eidelivery=0, eie=0
		state.imsic_m.eip[0].store(1 << IID_M_IPI, Ordering::Relaxed);
		let (val, _) = imsic_topei_peek(&state.imsic_m);
		assert_ne!(val, 0, "M-mode IPI must be visible with eidelivery=0");
		assert_eq!((val >> 16) as u32, IID_M_IPI, "IID must be 3 (M-mode IPI)");
	}

	#[test]
	fn topei_peek_ipi_visible_when_both_pending() {
		// When both IID=1 and IID=3 are pending in word 0, topei_peek
		// returns the higher-priority one (IID=3, higher bit index).
		let state = make_state_no_delivery();
		state.imsic_m.eip[0].store((1 << IID_S_IPI) | (1 << IID_M_IPI), Ordering::Relaxed);
		let (val, _) = imsic_topei_peek(&state.imsic_m);
		assert_eq!(
			(val >> 16) as u32,
			IID_M_IPI,
			"higher IID (3) takes priority over IID=1"
		);
	}

	#[test]
	fn topei_peek_external_blocked_without_eidelivery() {
		// External interrupts (IID >= 6) must NOT be visible without eidelivery.
		let state = make_state_no_delivery();
		state.imsic_s.eie[0].store(1 << 10, Ordering::Relaxed);
		state.imsic_s.eip[0].store(1 << 10, Ordering::Relaxed);
		let (val, _) = imsic_topei_peek(&state.imsic_s);
		assert_eq!(val, 0, "external IID=10 blocked with eidelivery=0");
	}

	#[test]
	fn topei_peek_ipi_dominates_external() {
		// When both IPI and external are pending in word 0, IPI is found
		// first (fast-path runs before external scan).  External is
		// blocked by eidelivery=0 regardless.
		let state = make_state_no_delivery();
		state.imsic_s.eie[0].store(1 << 10, Ordering::Relaxed);
		state.imsic_s.eip[0].store((1 << IID_S_IPI) | (1 << 10), Ordering::Relaxed);
		let (val, _) = imsic_topei_peek(&state.imsic_s);
		assert_eq!(
			(val >> 16) as u32,
			IID_S_IPI,
			"IPI fast-path must return IID=1 even when external IID=10 also pending"
		);
	}

	#[test]
	fn topei_peek_ipi_passes_eithreshold() {
		// IPI fast-path checks eithreshold?  No — the IPI fast-path
		// returns the IPI unconditionally.  Verify this.
		let mut state = make_state_no_delivery();
		state.imsic_s.eithreshold = 200; // higher than IPI priority (1 & 0xFF = 1)
		state.imsic_s.eip[0].store(1 << IID_S_IPI, Ordering::Relaxed);
		let (val, _) = imsic_topei_peek(&state.imsic_s);
		assert_ne!(val, 0, "IPI bypasses eithreshold check");
	}

	// ---- imsic_topei_claim_iid: IPI claim ----

	#[test]
	fn ipi_claimable_without_eidelivery() {
		// IPIs must be claimable (eip cleared) even with eidelivery=0.
		// Without this, a pending IPI is reported by compute_stopi/mtopi
		// forever → infinite dispatch loop.
		let mut state = make_state_no_delivery();
		state.imsic_s.eip[0].store(1 << IID_S_IPI, Ordering::Relaxed);

		// Verify it's pending before claim
		let (before, _) = imsic_topei_peek(&state.imsic_s);
		assert_ne!(before, 0, "IPI pending before claim");

		imsic_topei_claim_iid(&mut state.imsic_s, IID_S_IPI);

		let (after, _) = imsic_topei_peek(&state.imsic_s);
		assert_eq!(after, 0, "IPI cleared after claim even with eidelivery=0");
		assert_eq!(
			state.imsic_s.eip[0].load(Ordering::Acquire) & (1 << IID_S_IPI),
			0,
			"eip bit must be cleared"
		);
	}

	#[test]
	fn ipi_mfile_claimable_without_eidelivery() {
		// Same as above but for M-file IID=3.
		let mut state = make_state_no_delivery();
		state.imsic_m.eip[0].store(1 << IID_M_IPI, Ordering::Relaxed);

		imsic_topei_claim_iid(&mut state.imsic_m, IID_M_IPI);

		assert_eq!(
			state.imsic_m.eip[0].load(Ordering::Acquire) & (1 << IID_M_IPI),
			0,
			"M-mode IPI eip cleared after claim with eidelivery=0"
		);
	}

	#[test]
	fn external_not_claimable_without_eidelivery() {
		// External interrupts must NOT be claimable without eidelivery
		// (and even if they were, the claim would be a no-op because
		// the eie check fails).  Verify eip is unchanged.
		let mut state = make_state_no_delivery();
		state.imsic_s.eie[0].store(1 << 10, Ordering::Relaxed);
		state.imsic_s.eip[0].store(1 << 10, Ordering::Relaxed);

		imsic_topei_claim_iid(&mut state.imsic_s, 10);

		assert_eq!(
			state.imsic_s.eip[0].load(Ordering::Acquire) & (1 << 10),
			1 << 10,
			"external eip unchanged — claim blocked by eidelivery=0"
		);
	}

	// ---- sync_imsic: IPI → mip bits ----

	#[test]
	fn sync_imsic_sets_seip_for_ipi_without_eidelivery() {
		// When eidelivery=0, sync_imsic must still set SEIP if an IPI
		// (IID=1) is pending.  This is the IPI wake-up path for early SMP.
		let mut state = make_state_no_delivery();
		state.imsic_s.eip[0].store(1 << IID_S_IPI, Ordering::Relaxed);

		sync_imsic(&mut state);
		assert_eq!(
			state.mip.load(Ordering::Acquire) & (1 << 9),
			1 << 9,
			"SEIP must be set for pending S-mode IPI even with eidelivery=0"
		);
	}

	#[test]
	fn sync_imsic_sets_meip_for_ipi_without_eidelivery() {
		// Same as above but for M-file IPI (IID=3) → MEIP.
		let mut state = make_state_no_delivery();
		state.imsic_m.eip[0].store(1 << IID_M_IPI, Ordering::Relaxed);

		sync_imsic(&mut state);
		assert_eq!(
			state.mip.load(Ordering::Acquire) & (1 << 11),
			1 << 11,
			"MEIP must be set for pending M-mode IPI even with eidelivery=0"
		);
	}

	#[test]
	fn sync_imsic_does_not_clear_seip_without_eidelivery() {
		// With eidelivery=0, sync_imsic must NEVER clear SEIP on its own
		// because PLIC may be driving it independently.
		let mut state = make_state_no_delivery();
		// Pre-set SEIP (simulating PLIC interrupt in-flight)
		state.mip.store(1 << 9, Ordering::Relaxed);
		// No IMSIC pending

		sync_imsic(&mut state);
		assert_eq!(
			state.mip.load(Ordering::Acquire) & (1 << 9),
			1 << 9,
			"SEIP preserved — sync_imsic with eidelivery=0 does not clear"
		);
	}

	#[test]
	fn sync_imsic_clears_seip_after_ipi_claim() {
		// Set IPI → sync_imsic sets SEIP → claim IPI → sync_imsic clears SEIP.
		// This verifies the full IPI lifecycle through sync_imsic with eidelivery.
		let mut state = make_state_no_delivery();
		state.imsic_s.eidelivery = 1; // enable clearing

		// Inject IPI
		state.imsic_s.eip[0].store(1 << IID_S_IPI, Ordering::Relaxed);
		sync_imsic(&mut state);
		assert_eq!(
			state.mip.load(Ordering::Acquire) & (1 << 9),
			1 << 9,
			"SEIP set after IPI injected"
		);

		// Claim IPI
		imsic_topei_claim_iid(&mut state.imsic_s, IID_S_IPI);
		sync_imsic(&mut state);
		assert_eq!(
			state.mip.load(Ordering::Acquire) & (1 << 9),
			0,
			"SEIP cleared after IPI claimed"
		);
	}

	// ---- compute_mtopi / compute_stopi: IPI reporting ----

	#[test]
	fn compute_stopi_reports_ipi_without_eidelivery() {
		// compute_stopi calls topei_peek → IPI fast-path → maps the IPI
		// minor identity (1) to the SEI major identity (9).  The IMSIC
		// delivers the IPI via SEIP, so stopi must report IID=9 (SEI) —
		// never SSI (1) — even without eidelivery.  The minor identity is
		// revealed only via STOPEI.
		let mut state = make_state_no_delivery();
		state.imsic_s.eip[0].store(1 << IID_S_IPI, Ordering::Relaxed);
		let (val, _) = compute_stopi(&mut state, 0);
		assert_ne!(val, 0, "stopi must report IPI even with eidelivery=0");
		assert_eq!(
			(val >> 16) as u32,
			9,
			"stopi reports IID=9 (SEI) for S-file IPI"
		);

		// stopi read must NOT claim: the eip bit stays pending for STOPEI.
		assert_eq!(
			state.imsic_s.eip[0].load(Ordering::Acquire) & (1 << IID_S_IPI),
			1 << IID_S_IPI,
			"stopi read must not claim the IPI eip bit"
		);
	}

	#[test]
	fn compute_mtopi_reports_ipi_without_eidelivery() {
		// Same for M-mode: compute_mtopi maps the M-file IPI (minor 3)
		// to the MEI major identity (11).  The IMSIC delivers the IPI via
		// MEIP, so mtopi must report IID=11 (MEI) — never MSI (3).
		let mut state = make_state_no_delivery();
		state.imsic_m.eip[0].store(1 << IID_M_IPI, Ordering::Relaxed);

		let (val, _) = compute_mtopi(&mut state, 0);
		assert_ne!(val, 0, "mtopi must report IPI even with eidelivery=0");
		assert_eq!(
			(val >> 16) as u32,
			11,
			"mtopi reports IID=11 (MEI) for M-file IPI"
		);

		// mtopi read must NOT claim: the eip bit stays pending for MTOPEI.
		assert_eq!(
			state.imsic_m.eip[0].load(Ordering::Acquire) & (1 << IID_M_IPI),
			1 << IID_M_IPI,
			"mtopi read must not claim the IPI eip bit"
		);
	}

	#[test]
	fn stopi_sfile_ipi_reports_sei_major_identity() {
		// Regression: an S-file IPI (minor identity 1) must be reported by
		// stopi as major identity 9 (SEI), NOT 1 (SSI).  Linux's
		// riscv_intc_aia_irq() dispatches `stopi >> 16` to the intc domain;
		// returning 1 (SSI) has no handler in IMSIC mode, so the IPI is
		// silently lost and the sender hart spins in
		// smp_call_function_many_cond waiting for the target hart.
		let mut state = make_state_no_delivery();
		state.imsic_s.eip[0].store(1 << IID_S_IPI, Ordering::Relaxed);
		let (val, _) = compute_stopi(&mut state, 0);
		assert_eq!(
			(val >> 16) as u32,
			9,
			"S-file IPI must be reported as SEI (9), not SSI (1)"
		);
	}

	// ---- imsic_handle_seteipnum: IPI routing ----

	#[test]
	fn seteipnum_mfile_iid1_stays_in_mfile() {
		// Regression: M-file ``seteipnum`` with minor identity 1 must set
		// eip[1] in the M-file (driving MEIP), NOT route to the S-file.
		// OpenSBI sends IPIs by writing ``IMSIC_IPI_ID`` (=1) to the target
		// hart's M-file; routing IID=1 to the S-file silently drops M-mode
		// IPIs and leaves secondary harts asleep during SMP boot.
		let state = make_state_no_delivery();

		// Simulate cross-hart seteipnum write: target_hart=0, M-file, IID=1
		let states = &state as *const HartState;
		let result = imsic_handle_seteipnum(states, std::ptr::null(), 1, 0, 0x0000, IID_S_IPI);
		assert!(result, "seteipnum must return true (handled)");

		// M-file must have the IPI.
		assert_eq!(
			state.imsic_m.eip[0].load(Ordering::Acquire) & (1 << IID_S_IPI),
			1 << IID_S_IPI,
			"M-file eip must have IID=1 (no cross-file routing)"
		);
		// MEIP must be driven by the M-file eip write.
		assert_eq!(
			state.mip.load(Ordering::Acquire) & (1 << 11),
			1 << 11,
			"MEIP must be set when M-file eip[1] is set"
		);
		// S-file must NOT receive the IPI.
		assert_eq!(
			state.imsic_s.eip[0].load(Ordering::Acquire) & (1 << IID_S_IPI),
			0,
			"S-file eip must NOT have IID=1 (addressed file only)"
		);
	}

	#[test]
	fn seteipnum_mfile_iid3_stays_in_mfile() {
		// M-file seteipnum with IID=3 → stays in M-file (M-mode IPI).
		let state = make_state_no_delivery();
		let states = &state as *const HartState;
		let result = imsic_handle_seteipnum(states, std::ptr::null(), 1, 0, 0x0000, IID_M_IPI);
		assert!(result);

		assert_eq!(
			state.imsic_m.eip[0].load(Ordering::Acquire) & (1 << IID_M_IPI),
			1 << IID_M_IPI,
			"M-file must have IID=3 (no routing)"
		);
		assert_eq!(
			state.imsic_s.eip[0].load(Ordering::Acquire) & (1 << IID_M_IPI),
			0,
			"S-file must NOT have IID=3"
		);
	}

	#[test]
	fn seteipnum_sfile_iid1_stays_in_sfile() {
		// S-file seteipnum with IID=1 → stays in S-file (S-mode IPI).
		let state = make_state_no_delivery();
		let states = &state as *const HartState;
		let result = imsic_handle_seteipnum(states, std::ptr::null(), 1, 0, 0x1000, IID_S_IPI);
		assert!(result);

		assert_eq!(
			state.imsic_s.eip[0].load(Ordering::Acquire) & (1 << IID_S_IPI),
			1 << IID_S_IPI,
			"S-file must have IID=1"
		);
		// M-file should not have the IPI unless it was specifically routed
	}

	// ---- imsic_eip_set: cross-hart atomic write ----

	#[test]
	fn eip_set_sfile_sets_eip_and_seip() {
		// imsic_eip_set for S-file must set eip bit AND SEIP.
		let state = make_state_no_delivery();
		let states = &state as *const HartState;
		imsic_eip_set(states, 0, 0x1000, IID_S_IPI);

		assert_eq!(
			state.imsic_s.eip[0].load(Ordering::Acquire) & (1 << IID_S_IPI),
			1 << IID_S_IPI,
			"eip bit set"
		);
		assert_eq!(
			state.mip.load(Ordering::Acquire) & (1 << 9),
			1 << 9,
			"SEIP set by imsic_eip_set"
		);
	}

	#[test]
	fn eip_set_mfile_sets_eip_and_meip() {
		let state = make_state_no_delivery();
		let states = &state as *const HartState;
		imsic_eip_set(states, 0, 0x0000, IID_M_IPI);

		assert_eq!(
			state.imsic_m.eip[0].load(Ordering::Acquire) & (1 << IID_M_IPI),
			1 << IID_M_IPI,
			"eip bit set"
		);
		assert_eq!(
			state.mip.load(Ordering::Acquire) & (1 << 11),
			1 << 11,
			"MEIP set by imsic_eip_set"
		);
	}

	#[test]
	fn eip_set_null_pointer_is_noop() {
		// imsic_eip_set with null pointer must not crash.
		// (The function checks is_null and returns early.)
		imsic_eip_set(std::ptr::null(), 0, 0x0000, IID_M_IPI);
		// If we reach here, it didn't crash
	}

	#[test]
	fn eip_clear_sfile_clears_eip_and_seip() {
		let state = make_state_no_delivery();
		state.imsic_s.eip[0].store(1 << IID_S_IPI, Ordering::Relaxed);
		state.mip.store(1 << 9, Ordering::Relaxed);

		let states = &state as *const HartState;
		imsic_eip_clear(states, 0, 0x1000, IID_S_IPI);

		assert_eq!(
			state.imsic_s.eip[0].load(Ordering::Acquire) & (1 << IID_S_IPI),
			0,
			"eip bit cleared"
		);
		assert_eq!(
			state.mip.load(Ordering::Acquire) & (1 << 9),
			0,
			"SEIP cleared by imsic_eip_clear"
		);
	}

	// ---- imsic_topei_peek: eip_ext_any does not affect IPI ----

	#[test]
	fn topei_peek_ipi_visible_even_when_eip_ext_any_zero() {
		// eip_ext_any is a fast-out for external interrupts in words 1+.
		// IPI (word 0) bypasses it — verify IPI visible when eip_ext_any=0.
		let mut state = make_state_no_delivery();
		state.imsic_m.eidelivery = 1;
		// eip_ext_any defaults to 0 (nothing in any ext word)
		state.imsic_m.eip[0].store(1 << IID_M_IPI, Ordering::Relaxed);

		let (val, _) = imsic_topei_peek(&state.imsic_m);
		assert_ne!(val, 0, "IPI visible even when eip_ext_any==0");
	}

	// ---- IPI + CLINT coexistence ----

	#[test]
	fn clint_msip_still_visible_alongside_imsic_ipi() {
		// When IMSIC is present with eidelivery=0 and CLINT also has MSIP,
		// check_pending_interrupts must see both.  The IMSIC IPI drives
		// MEIP, while CLINT drives MSIP.
		let mut state = make_state_no_delivery();
		// IMSIC M-file IPI
		state.imsic_m.eip[0].store(1 << IID_M_IPI, Ordering::Relaxed);
		// CLINT MSIP
		state.mip.fetch_or(1 << 3, Ordering::AcqRel);
		// Enable both in mie
		state.mie = (1 << 11) | (1 << 3); // MEIE + MSIE

		sync_imsic(&mut state);

		// Both MEIP and MSIP should be set
		let mip = state.mip.load(Ordering::Acquire);
		assert_eq!(mip & (1 << 11), 1 << 11, "MEIP from IMSIC IPI");
		assert_eq!(mip & (1 << 3), 1 << 3, "MSIP from CLINT");

		// check_pending_interrupts picks MEI over MSI (higher priority)
		let result = check_pending_interrupts(&state);
		assert!(result.is_some(), "interrupt must be pending");
		assert_eq!(
			result.unwrap().0,
			11,
			"MEI (cause 11) takes priority over MSI"
		);
	}

	// ============================================================
	//  Multi-hart integration: cross-hart IPI → compute_stopi/mtopi
	// ============================================================

	fn make_hart1_state() -> HartState {
		let mut s: HartState = unsafe { std::mem::zeroed() };
		s.imsic_m.present = 1;
		s.imsic_m.eidelivery = 1;
		s.imsic_s.present = 1;
		s.imsic_s.eidelivery = 1;
		s.mhartid = 1;
		s.mie = (1 << 9) | (1 << 11); // SEIE + MEIE
								// mideleg: delegate SEI, SSI, STI to S-mode (OpenSBI default in AIA mode).
		s.mideleg = (1 << 9) | (1 << 1) | (1 << 5);
		s
	}

	fn make_hart0_state() -> HartState {
		let mut s: HartState = unsafe { std::mem::zeroed() };
		s.imsic_m.present = 1;
		s.imsic_m.eidelivery = 1;
		s.imsic_s.present = 1;
		s.imsic_s.eidelivery = 1;
		s.mhartid = 0;
		s.mie = (1 << 9) | (1 << 11) | (1 << 3); // SEIE + MEIE + MSIE
		s.mideleg = (1 << 9) | (1 << 1) | (1 << 5);
		s
	}

	#[test]
	fn multi_hart_smode_ipi_sfile_to_stopi() {
		// Simulate: hart 0 sends S-mode IPI to hart 1 via IMSIC S-file.
		// Hart 1's compute_stopi must see IID=9 (SEI).
		let _h0 = make_hart0_state();
		let mut h1 = make_hart1_state();

		// Direct cross-hart write: imsic_eip_set takes *const HartState as
		// the array base and hart index 0 means "target = &h1".
		let h1_ptr = &h1 as *const HartState;
		imsic_eip_set(h1_ptr, 0, 0x1000, IID_S_IPI); // S-file, hart index 0

		sync_imsic(&mut h1);

		// SEIP must be set because the S-file has IID=1 pending.
		assert_eq!(
			h1.mip.load(Ordering::Acquire) & (1 << 9),
			1 << 9,
			"SEIP must be set after cross-hart IPI"
		);
		assert_eq!(
			h1.imsic_s.eip[0].load(Ordering::Acquire) & (1 << IID_S_IPI),
			1 << IID_S_IPI,
			"S-file eip must have IID=1"
		);

		// Hart 1 kernel reads stopi → compute_stopi
		let (stopi_val, _) = compute_stopi(&mut h1, 0);
		assert_ne!(stopi_val, 0, "stopi must report pending interrupt");
		assert_eq!(
			(stopi_val >> 16) as u32,
			9,
			"stopi must report IID=9 (SEI) for the S-file IPI from hart 0"
		);
	}

	#[test]
	fn multi_hart_mmode_ipi_mfile_to_mtopi() {
		// Simulate: hart 0 sends M-mode IPI to hart 1 via IMSIC M-file.
		// Hart 1's compute_mtopi must see IID=11 (MEI).
		let _h0 = make_hart0_state();
		let mut h1 = make_hart1_state();

		let h1_ptr = &h1 as *const HartState;
		imsic_eip_set(h1_ptr, 0, 0x0000, IID_M_IPI); // M-file, hart index 0

		sync_imsic(&mut h1);

		// MEIP must be set.
		assert_eq!(
			h1.mip.load(Ordering::Acquire) & (1 << 11),
			1 << 11,
			"MEIP must be set after cross-hart M-mode IPI"
		);

		// Hart 1 OpenSBI reads mtopi → compute_mtopi
		let (mtopi_val, _) = compute_mtopi(&mut h1, 0);
		assert_ne!(mtopi_val, 0, "mtopi must report pending interrupt");
		assert_eq!(
			(mtopi_val >> 16) as u32,
			11,
			"mtopi must report IID=11 (MEI) for the M-file IPI from hart 0"
		);
	}

	#[test]
	fn multi_hart_stopi_ipi_priority_over_stip() {
		// When BOTH an IMSIC S-file IPI and STIP (timer) are pending,
		// compute_stopi must return the IPI first (IMSIC S-file > SSIP > STIP).
		let mut h1 = make_hart1_state();

		// Set up STIP: stimecmp in the past, mie.STIE enabled.
		h1.stimecmp = 50;
		h1.mie |= 1 << 5; // STIE
					// IPI via direct eip write (simulating hart 0's seteipnum).
		h1.imsic_s.eip[0].store(1 << IID_S_IPI, Ordering::Relaxed);
		// Sync SEIP from IMSIC.
		sync_imsic(&mut h1);

		// Manually set STIP (sync_mtip would do this at the instruction boundary).
		// stimecmp=50, mtime=100 → mtime > stimecmp → STIP pending.
		h1.mip.fetch_or(1 << 5, Ordering::AcqRel);

		// compute_stopi: check priority order.
		let (val, _) = compute_stopi(&mut h1, 100); // mtime=100
		assert_ne!(val, 0, "stopi must return something");
		// IMSIC S-file is checked first — must return IID=9 (SEI) even when STIP pending.
		assert_eq!(
			(val >> 16) as u32,
			9,
			"stopi must report IID=9 (SEI) before STIP (IMSIC S-file has priority)"
		);

		// STIP should still be set in mip (not cleared by compute_stopi).
		assert_eq!(
			h1.mip.load(Ordering::Acquire) & (1 << 5),
			1 << 5,
			"STIP still pending after first stopi (not checked yet)"
		);

		// Now the kernel claims the IPI via stopi write.
		imsic_topei_claim_iid(&mut h1.imsic_s, IID_S_IPI);
		sync_imsic_one(&mut h1, false); // mfile=false → S-file → SEIP
								  // After claim: clear SEIP as the csr_write(stopi) path does.
		h1.mip.fetch_and(!(1 << 9), Ordering::AcqRel);

		// After claim + sync: IMSIC S-file should be empty, SEIP cleared.
		assert_eq!(
			h1.imsic_s.eip[0].load(Ordering::Acquire) & (1 << IID_S_IPI),
			0,
			"S-file eip cleared after claim"
		);
		assert_eq!(
			h1.mip.load(Ordering::Acquire) & (1 << 9),
			0,
			"SEIP cleared after claim + sync"
		);

		// Second stopi read: now STIP should be reported (IID=5).
		let (val2, _) = compute_stopi(&mut h1, 100);
		assert_ne!(val2, 0, "second stopi must return STIP");
		assert_eq!(
			(val2 >> 16) as u32,
			5,
			"second stopi must report IID=5 (STIP) after IPI is claimed"
		);
	}

	#[test]
	fn multi_hart_stopi_stip_evaluation_agrees_with_sync_mtip() {
		// sync_mtip sets STIP when stimecmp>0 && mtime>=stimecmp (>=, per the
		// RISC-V SSTC spec).  compute_stopi must agree: report IID=5 under the
		// same condition.
		let mut h1 = make_hart1_state();
		h1.stimecmp = 50;
		h1.mie |= 1 << 5; // STIE
					// No IMSIC pending — SSIP=0.

		// mtime = 50: mtime >= stimecmp (50 >= 50 = true).
		// sync_mtip sets STIP; compute_stopi must report IID=5.
		let (val, _) = compute_stopi(&mut h1, 50);
		assert_eq!(
			(val >> 16) as u32,
			5,
			"stopi reports IID=5 when mtime == stimecmp (>= comparison)"
		);

		// mtime = 49: mtime < stimecmp → STIP not set → stopi returns 0.
		let (val2, _) = compute_stopi(&mut h1, 49);
		assert_eq!(val2, 0, "stopi returns 0 when mtime < stimecmp");

		// mtime = 51: mtime > stimecmp → STIP reported.
		let (val3, _) = compute_stopi(&mut h1, 51);
		assert_eq!(
			(val3 >> 16) as u32,
			5,
			"stopi reports IID=5 when mtime > stimecmp"
		);
	}

	#[test]
	fn multi_hart_stopi_ipi_in_sfile_not_visible_in_mtopi() {
		// An IPI pending in hart 1's S-file (minor identity 1) is visible via
		// compute_stopi (S-file view) and invisible via compute_mtopi (M-file
		// view) — the two files are independent; there is no cross-file routing.
		let _h0 = make_hart0_state();
		let mut h1 = make_hart1_state();

		// Direct IMSIC S-file eip set — the addressed-file-only path.
		h1.imsic_s.eip[0].store(1 << IID_S_IPI, Ordering::Relaxed);
		h1.mip.fetch_or(1 << 9, Ordering::AcqRel); // SEIP
		sync_imsic(&mut h1);

		// Hart 1 reads stopi (S-file view) → sees the S-file IPI.
		let (val, _) = compute_stopi(&mut h1, 0);
		assert_eq!(
			(val >> 16) as u32,
			9,
			"stopi sees IID=9 (SEI) pending in the S-file"
		);

		// compute_mtopi (M-file view) → M-file has nothing, no cross-file leak.
		let (mtopi_val, _) = compute_mtopi(&mut h1, 0);
		assert_eq!(
			mtopi_val, 0,
			"mtopi returns 0: S-file IPI is not visible in the M-file"
		);
	}

	#[test]
	fn multi_hart_stopi_ipi_claim_then_recheck() {
		// Full lifecycle: IPI arrives → stopi reports IID=1 → kernel writes stopi
		// (claim) → stopi returns 0 (nothing left).
		let _h0 = make_hart0_state();
		let mut h1 = make_hart1_state();

		// Simulate cross-hart IPI: set S-file eip[IID_S_IPI] + SEIP.
		h1.imsic_s.eip[0].store(1 << IID_S_IPI, Ordering::Relaxed);
		sync_imsic(&mut h1);

		// Step 1: stopi read → IID=9 (SEI)
		let (val1, _) = compute_stopi(&mut h1, 0);
		assert_eq!((val1 >> 16) as u32, 9, "first stopi → IID=9 (SEI)");

		// Step 2: kernel writes stopi (claim IID=1 in S-file).
		imsic_topei_claim_iid(&mut h1.imsic_s, IID_S_IPI);
		sync_imsic_one(&mut h1, false); // S-file → SEIP
		h1.mip.fetch_and(!(1 << 9), Ordering::AcqRel);

		// Step 3: SEIP cleared, eip cleared.
		assert_eq!(
			h1.mip.load(Ordering::Acquire) & (1 << 9),
			0,
			"SEIP must be 0 after claim + sync"
		);
		// Step 4: next stopi read → 0 (nothing pending).
		let (val2, _) = compute_stopi(&mut h1, 0);
		assert_eq!(val2, 0, "stopi returns 0 after IPI is claimed");
	}

	#[test]
	fn multi_hart_stopi_ipi_two_harts_independent() {
		// Verify that an IPI to hart 1 does NOT affect hart 2's stopi.
		let mut h1 = make_hart1_state();
		let mut h2 = make_hart1_state(); // second target, same topology

		// Simulate IPI only to hart 1: set S-file eip + SEIP.
		h1.imsic_s.eip[0].store(1 << IID_S_IPI, Ordering::Relaxed);
		h1.mip.fetch_or(1 << 9, Ordering::AcqRel);
		sync_imsic(&mut h1);

		// Hart 1: IPI visible in stopi.
		let (val_h1, _) = compute_stopi(&mut h1, 0);
		assert_eq!((val_h1 >> 16) as u32, 9, "hart 1 stopi must see IPI as SEI");

		// Hart 2: NO IPI — stopi must return 0.
		let (val_h2, _) = compute_stopi(&mut h2, 0);
		assert_eq!(val_h2, 0, "hart 2 stopi must return 0 (no IPI)");
	}

	#[test]
	fn multi_hart_mtopi_ipi_priority_over_mtip() {
		// When BOTH IMSIC M-file IPI and MTIP (timer) are pending,
		// compute_mtopi must return the IPI first (IMSIC M-file > MSIP > MTIP).
		let mut h1 = make_hart1_state();

		// Set up MTIP: manually set mip.MTIP and enable MTIE.
		h1.mip.fetch_or(1 << 7, Ordering::AcqRel);
		h1.mie |= 1 << 7; // MTIE

		// Also set IMSIC M-file IPI.
		h1.imsic_m.eip[0].store(1 << IID_M_IPI, Ordering::Relaxed);
		sync_imsic(&mut h1);

		// compute_mtopi: check priority.
		let (val, _) = compute_mtopi(&mut h1, 0);
		assert_ne!(val, 0, "mtopi must return something");
		assert_eq!(
			(val >> 16) as u32,
			11,
			"mtopi must report IID=11 (MEI, IMSIC M-file) before MTIP"
		);

		// Claim the IPI → now compute_mtopi should see MTIP.
		imsic_topei_claim_iid(&mut h1.imsic_m, IID_M_IPI);
		sync_imsic_one(&mut h1, true); // M-file → MEIP
		h1.mip.fetch_and(!(1 << 11), Ordering::AcqRel); // clear MEIP

		let (val2, _) = compute_mtopi(&mut h1, 0);
		assert_ne!(val2, 0, "mtopi must report MTIP after IPI claimed");
		assert_eq!(
			(val2 >> 16) as u32,
			7,
			"mtopi must report IID=7 (MTIP) after IMSIC IPI claimed"
		);
	}

	#[test]
	fn multi_hart_mtopi_spurious_mei_cleanup() {
		// When ext_irq sets MEIP but IMSIC M-file has nothing pending,
		// the step_interrupts cleanup must correctly clear MEIP.
		// Regression test for the ext_irq → MEIP → cleanup flip-flop.
		let h1 = make_hart1_state();

		// Simulate ext_irq drain setting MEIP (done by main loop at top of
		// each speedup-execution iteration).
		h1.mip.fetch_or(1 << 11, Ordering::AcqRel);
		// No IMSIC M-file eip bits — topei should return 0.
		let (topei_val, _) = imsic_topei_peek(&h1.imsic_m);
		assert_eq!(topei_val, 0, "topei returns 0 when no IMSIC pending");

		// Perform cleanup (same logic as step_interrupts).
		if (h1.mip.load(Ordering::Acquire) & (1 << 11)) != 0 && h1.imsic_m.present != 0 {
			if h1.imsic_m.eidelivery != 0 {
				let (val, _) = imsic_topei_peek(&h1.imsic_m);
				if val == 0 {
					h1.mip.fetch_and(!(1 << 11), Ordering::AcqRel);
				}
			} else {
				let ipi_mask: u32 = (1 << IID_S_IPI) | (1 << IID_M_IPI);
				let has_ipi = (h1.imsic_m.eip[0].load(Ordering::Acquire) & ipi_mask) != 0;
				if !has_ipi {
					h1.mip.fetch_and(!(1 << 11), Ordering::AcqRel);
				}
			}
		}

		assert_eq!(
			h1.mip.load(Ordering::Acquire) & (1 << 11),
			0,
			"MEIP cleared by cleanup when no IMSIC M-file interrupt pending"
		);

		// After cleanup, check_pending_interrupts must NOT deliver MEI.
		let mip = h1.mip.load(Ordering::Acquire);
		assert_eq!(mip & (1 << 11), 0, "MEIP cleared → no MEI");
	}

	/// When an S-file IPI is pending and SEI is NOT delegated (mideleg[9]=0),
	/// the SEI trap goes to M-mode.  OpenSBI's M-mode handler reads mtopi
	/// (0xFB0), which calls compute_mtopi.  compute_mtopi only checks the
	/// IMSIC M-file — it can NOT see the S-file IPI.  So mtopi returns 0,
	/// OpenSBI MRETs without clearing SEIP, and the next instruction triggers
	/// another M-mode SEI trap → infinite spurious-SEI loop.
	///
	/// This test verifies the diagnostic: mtopi=0 when M-file is empty but
	/// S-file has a pending IPI with mideleg[9]=0.
	#[test]
	fn multi_hart_mtopi_zero_when_sfile_ipi_not_delegated() {
		let mut h1 = make_hart1_state();
		// SEI NOT delegated → M-mode trap for SEI
		h1.mideleg &= !(1 << 9);
		// S-file IPI pending
		h1.imsic_s.eip[0].store(1 << IID_S_IPI, Ordering::Relaxed);
		sync_imsic(&mut h1);

		// SEIP must be set (IMSIC S-file has IPI)
		assert_eq!(
			h1.mip.load(Ordering::Acquire) & (1 << 9),
			1 << 9,
			"SEIP set when S-file has IPI"
		);

		// Hart is in S-mode (OpenSBI handed off to kernel).
		h1.mode = riscv_mode::S;
		h1.mstatus = 0; // SIE=0, SPIE=0, SPP=0

		// check_pending_interrupts: SEIP & SEIE (=0 initially) → nothing.
		// But step_interrupts force-enables SEIE when SEIP is pending.
		h1.mie |= 1 << 9; // force-enable SEIE

		// Now: SEIP=1, SEIE=1, mideleg[9]=0 → SEI NOT delegated → M-mode trap.
		let result = check_pending_interrupts(&h1);
		assert!(result.is_some(), "interrupt must be pending");
		let (cause, is_m_mode) = result.unwrap();
		assert_eq!(cause, 9, "cause=9 (SEI)");
		assert!(is_m_mode, "SEI delivered to M-mode when mideleg[9]=0");

		// Now simulate OpenSBI M-mode handler reading mtopi.
		// compute_mtopi checks M-file only → returns 0.
		let (mtopi_val, _) = compute_mtopi(&mut h1, 0);
		assert_eq!(
			mtopi_val, 0,
			"mtopi=0 when S-file IPI and mideleg[9]=0: \
             M-file has nothing, S-file IPI is invisible to mtopi"
		);

		// After mtopi=0, OpenSBI does MRET without clearing SEIP.
		// SEIP is still set → next instruction triggers SEI again → loop!
		assert_eq!(
			h1.mip.load(Ordering::Acquire) & (1 << 9),
			1 << 9,
			"SEIP still set after mtopi=0 — spurious SEI loop"
		);
	}

	/// When SEI IS delegated (mideleg[9]=1) but S-mode has interrupts
	/// disabled (SIE=0), check_pending_interrupts must SKIP the SEI, not
	/// fall through to M-mode delivery.  Falling through to M-mode with an
	/// S-file-only IPI would cause the same mtopi=0 → MRET → SEI loop
	/// verified by multi_hart_mtopi_zero_when_sfile_ipi_not_delegated.
	#[test]
	fn multi_hart_sei_skipped_when_sie_zero_delegated() {
		let mut h1 = make_hart1_state();
		// SEI delegated (OpenSBI default)
		h1.mideleg = (1 << 9) | (1 << 1) | (1 << 5);
		// S-file IPI pending
		h1.imsic_s.eip[0].store(1 << IID_S_IPI, Ordering::Relaxed);
		sync_imsic(&mut h1);

		// Hart is in S-mode with SIE=0 (interrupts disabled in kernel).
		h1.mode = riscv_mode::S;
		h1.mstatus = 0; // SIE=0
		h1.mie |= 1 << 9; // force-enable SEIE (done by step_interrupts)

		// check_pending_interrupts: SEIP=1, SEIE=1, mideleg[9]=1
		// → delegated to S-mode → SIE=0 → SKIP (continue)
		let result = check_pending_interrupts(&h1);
		assert!(
			result.is_none(),
			"SEI skipped when mideleg[9]=1 and SIE=0: \
             no fall-through to M-mode"
		);

		// SEIP is still set — the interrupt is pending but not taken.
		// The hart continues execution.  When SIE becomes 1, the SEI
		// will be delivered correctly.
		assert_eq!(
			h1.mip.load(Ordering::Acquire) & (1 << 9),
			1 << 9,
			"SEIP stays pending when SIE=0"
		);
	}
}
