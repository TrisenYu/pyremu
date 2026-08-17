//! Sv39 page-table walk and TLB management for the speedup execution engine.
//!
//! Replaces the Python ``sv39_walk`` + ``TLB.lookup/insert/flush`` with
//! Rust-native implementations that operate directly on RAM and inline
//! ``TlbEntry`` arrays in ``HartState``.
//!
//! Implements hardware A/D bit setting: the A (Accessed) bit is set on any
//! PTE that is read during a walk; the D (Dirty) bit is set on the leaf PTE
//! when the access is a write.  This matches RISC-V privileged spec §4.3.2.
//!
//! # SUM/MXR support (RISC-V Privileged Spec §4.1.12)
//!
//! - **SUM** (bit 18): when set, S-mode may access pages marked U=1.
//!   Without SUM, S-mode access to user pages faults.
//! - **MXR** (bit 19): when set, executable-only pages (X=1, R=0) are
//!   readable by load instructions.  Does not affect stores or instruction
//!   fetches.

use core::cell::Cell;
use core::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use crate::mmu::{pte_parse, sv39_decompose_va};
use crate::state::{riscv_mode, HartState, TlbEntry, TLB_ENTRIES};

// ============================================================
//  Page-table walk context
// ============================================================

/// Shared context for memory access, TLB management, and LR/SC reservation.
pub struct WalkCtx {
	pub ram: *mut u8,
	pub ram_size: u64,
	pub ram_base: u64,
	pub shadow_base: u64,
	pub shadow_size: u64,
	/// Global TLB generation counter.  Hart checks this each instruction
	/// boundary; on mismatch, it marks its own TLB entries dirty (not flush),
	/// forcing re-walk on next access.
	pub tlb_gen: *const AtomicU64,
	/// Clock algorithm: eviction hand for itlb / dtlb.
	/// Cell provides interior mutability; handlers take ``&WalkCtx``.
	pub itlb_hand: Cell<u8>,
	pub dtlb_hand: Cell<u8>,
	/// Per-hart LR reservation slots (indexed by hart_id).
	/// LR sets `lr_reserved[hid] = pa`; any store clears all slots.
	/// Allocated in ``ModuleState``, pointer passed through FFI.
	pub lr_reserved: *mut AtomicU64,
	pub num_harts: u32,
}

/// Result of address translation.
#[derive(Debug)]
pub struct TranslateResult {
	pub pa: u64,
	pub perm: u8,
	pub level: u8,
}

/// Trap codes emitted by translation failures.
#[derive(Debug, PartialEq)]
pub enum TranslateFault {
	PageFault(u64), // cause code (12=Instr, 13=Load, 15=Store)
	AccessFault,
}

// ============================================================
//  LR/SC reservation helpers
// ============================================================

/// Set this hart's LR reservation.  pa=0 means "no reservation".
#[inline]
pub(crate) fn lr_set(ctx: &WalkCtx, hid: u8, pa: u64) {
	if ctx.lr_reserved.is_null() {
		return;
	}
	unsafe { &*ctx.lr_reserved.add(hid as usize) }.store(pa, Ordering::Release);
}

/// Check whether this hart still holds a reservation on *pa*.
#[inline]
pub(crate) fn lr_check(ctx: &WalkCtx, hid: u8, pa: u64) -> bool {
	if ctx.lr_reserved.is_null() {
		return false;
	}
	pa != 0 && unsafe { &*ctx.lr_reserved.add(hid as usize) }.load(Ordering::Acquire) == pa
}

/// Clear ALL harts' reservations.  Called on every store / AMO write.
#[inline]
pub(crate) fn lr_clear_all(ctx: &WalkCtx) {
	if ctx.lr_reserved.is_null() {
		return;
	}
	for i in 0..ctx.num_harts as usize {
		unsafe { &*ctx.lr_reserved.add(i) }.store(0, Ordering::Release);
	}
}

// ============================================================
//  RAM read/write (with atomic Acquire/Release for multi-hart)
// ============================================================

/// Read *size* bytes from physical address *pa* in RAM. Little-endian.
/// Aligned 4- and 8-byte reads use atomic Acquire.
#[inline]
pub(crate) fn ram_read(ctx: &WalkCtx, pa: u64, size: u8) -> u64 {
	let off = crate::mem::ram_offset_inline(
		pa,
		size as u32,
		ctx.ram_base,
		ctx.ram_size,
		ctx.shadow_base,
		ctx.shadow_size,
	);
	if off.is_none() {
		return 0;
	}
	let ptr = unsafe { ctx.ram.add(off.unwrap() as usize) };
	match size {
		1 => unsafe { *ptr as u64 },
		2 => {
			let b0 = unsafe { *ptr } as u64;
			let b1 = unsafe { *ptr.add(1) } as u64;
			b0 | (b1 << 8)
		}
		4 if (pa & 3) == 0 => {
			let a = unsafe { &*(ptr as *const AtomicU32) };
			a.load(Ordering::Acquire) as u64
		}
		4 => {
			let b0 = unsafe { *ptr } as u64;
			let b1 = unsafe { *ptr.add(1) } as u64;
			let b2 = unsafe { *ptr.add(2) } as u64;
			let b3 = unsafe { *ptr.add(3) } as u64;
			b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
		}
		8 if (pa & 7) == 0 => {
			let a = unsafe { &*(ptr as *const AtomicU64) };
			a.load(Ordering::Acquire)
		}
		8 => {
			let b = |i: usize| unsafe { *ptr.add(i) } as u64;
			b(0) | (b(1) << 8)
				| (b(2) << 16)
				| (b(3) << 24)
				| (b(4) << 32)
				| (b(5) << 40)
				| (b(6) << 48)
				| (b(7) << 56)
		}
		_ => 0,
	}
}

/// Write *size* bytes of *val* to physical address *pa* in RAM. Little-endian.
/// Aligned 4- and 8-byte writes use atomic Release.  Clears all LR reservations
/// before the store becomes visible (ABA fix).
#[inline]
pub(crate) fn ram_write(ctx: &WalkCtx, pa: u64, val: u64, size: u8) -> bool {
	let off = crate::mem::ram_offset_inline(
		pa,
		size as u32,
		ctx.ram_base,
		ctx.ram_size,
		ctx.shadow_base,
		ctx.shadow_size,
	);
	if off.is_none() {
		return false;
	}
	lr_clear_all(ctx);
	std::sync::atomic::fence(Ordering::SeqCst);
	let ptr = unsafe { ctx.ram.add(off.unwrap() as usize) };
	match size {
		1 => unsafe {
			*ptr = val as u8;
		},
		2 => unsafe {
			*ptr = val as u8;
			*ptr.add(1) = (val >> 8) as u8;
		},
		4 if (pa & 3) == 0 => {
			let a = unsafe { &*(ptr as *const AtomicU32) };
			a.store(val as u32, Ordering::Release);
		}
		4 => unsafe {
			*ptr = val as u8;
			for i in 1..4 {
				*ptr.add(i) = (val >> (i << 3)) as u8;
			}
		},
		8 if (pa & 7) == 0 => {
			let a = unsafe { &*(ptr as *const AtomicU64) };
			a.store(val, Ordering::Release);
		}
		8 => unsafe {
			*ptr = val as u8;
			for i in 1..8 {
				*ptr.add(i) = (val >> (i << 3)) as u8;
			}
		},
		_ => return false,
	}
	true
}

// ============================================================
//  RAM access for page walks
// ============================================================

/// Compute RAM offset; returns None if address is out of range.
#[inline]
fn ram_offset(ctx: &WalkCtx, pa: u64, size: u64) -> Option<usize> {
	let end = pa + size;
	if pa >= ctx.ram_base && end <= ctx.ram_base + ctx.ram_size {
		return Some((pa - ctx.ram_base) as usize);
	}
	if ctx.shadow_size > 0 {
		let sh_end = ctx.shadow_base + ctx.shadow_size;
		if pa >= ctx.shadow_base && end <= sh_end {
			return Some((pa - ctx.shadow_base) as usize);
		}
	}
	None
}

/// Read a 64-bit raw PTE value from physical memory.
///
/// Uses ``AtomicU64::load(Acquire)`` to synchronise with ``Release`` stores
/// from other hart threads (regular store instructions emit ``Release`` on
/// aligned 8-byte writes via ``ram_write_raw``).  Without this ordering the
/// page-table walker may observe a stale PTE that was overwritten by another
/// hart, populate the TLB with a wrong PA, and cause loads/stores to silently
/// hit the wrong physical page — producing garbage register values that can
/// manifest as near-NULL SIGSEGVs in userspace (the dynamic linker loading a
/// pointer from a corrupted page, then dereferencing it).
#[inline]
fn read_pte(ctx: &WalkCtx, pa: u64) -> u64 {
	let off = match ram_offset(ctx, pa, 8) {
		Some(o) => o,
		None => return 0,
	};
	// PTE addresses are always 8-byte aligned in Sv39 (root_ppn << 12 + vpn * 8).
	let ptr = unsafe { ctx.ram.add(off) };
	let a = unsafe { &*(ptr as *const AtomicU64) };
	let raw = a.load(Ordering::Acquire);
	raw
}

/// Write a 64-bit raw PTE value to physical memory (A/D bit update).
///
/// Uses ``AtomicU64::store(Release)`` to pair with the ``Acquire`` load in
/// CAS-based PTE write-back for A/D bit updates.
///
/// Returns ``true`` if the CAS succeeded (PTE was unchanged since *expected*
/// was read).  On failure another hart modified the PTE concurrently — the
/// caller must re-read and re-validate to avoid overwriting a concurrent
/// unmapping/remapping with stale A/D bits.
#[inline]
fn write_pte_cas(ctx: &WalkCtx, pa: u64, expected: u64, new_val: u64) -> bool {
	if new_val == expected {
		return true;
	}
	let off = match ram_offset(ctx, pa, 8) {
		Some(o) => o,
		None => return false,
	};
	let ptr = unsafe { ctx.ram.add(off) };
	let a = unsafe { &*(ptr as *const AtomicU64) };
	a.compare_exchange(expected, new_val, Ordering::AcqRel, Ordering::Acquire)
		.is_ok()
}

// ============================================================
//  PTE flag constants
// ============================================================

#[allow(dead_code)]
pub(crate) const PTE_V: u64 = 1 << 0;
pub(crate) const PTE_R: u64 = 1 << 1;
pub(crate) const PTE_W: u64 = 1 << 2;
pub(crate) const PTE_X: u64 = 1 << 3;
pub(crate) const PTE_U: u64 = 1 << 4;
const PTE_A: u64 = 1 << 6;
const PTE_D: u64 = 1 << 7;

const SATP_MODE_SV39: u64 = 8;

// ============================================================
//  Sv39 3-level page-table walk — extracted helpers
// ============================================================

/// Read and validate the L2 (root) PTE; set A bit.
/// Returns the raw value and parsed PTE on success.
fn walk_l2(ctx: &WalkCtx, root_ppn: u64, vpn2: u64) -> Option<(u64, crate::mmu::PteFields)> {
	let addr = (root_ppn << 12).wrapping_add(vpn2 * 8);
	let raw = read_pte(ctx, addr);
	let pte = pte_parse(raw);
	if pte.v == 0 || pte.is_leaf == 0 && pte.is_ptr == 0 {
		return None;
	}
	if raw & PTE_A == 0 {
		if !write_pte_cas(ctx, addr, raw, raw | PTE_A) {
			let raw2 = read_pte(ctx, addr);
			let pte2 = pte_parse(raw2);
			if pte2.v == 0 || pte2.is_leaf == 0 && pte2.is_ptr == 0 {
				return None;
			}
			return Some((raw2, pte2));
		}
	}
	Some((raw, pte))
}

/// From a valid L2 pointer, read L1; return `(raw, pte)` or `None`.
/// Sets A bit atomically on the L1 PTE.
fn walk_l1(ctx: &WalkCtx, l2_ppn: u64, vpn1: u64) -> Option<(u64, crate::mmu::PteFields)> {
	let addr = (l2_ppn << 12).wrapping_add(vpn1 * 8);
	let raw = read_pte(ctx, addr);
	let pte = pte_parse(raw);
	if pte.v == 0 || pte.is_leaf == 0 && pte.is_ptr == 0 {
		return None;
	}
	if raw & PTE_A == 0 {
		if !write_pte_cas(ctx, addr, raw, raw | PTE_A) {
			let raw2 = read_pte(ctx, addr);
			let pte2 = pte_parse(raw2);
			if pte2.v == 0 || pte2.is_leaf == 0 && pte2.is_ptr == 0 {
				return None;
			}
			return Some((raw2, pte2));
		}
	}
	Some((raw, pte))
}

/// Handle an L1 leaf (2 MiB superpage): check permissions, set A+D, assemble PA.
fn walk_l1_leaf(
	ctx: &WalkCtx,
	l1_addr: u64,
	l1_raw: u64,
	l1_ppn: u64,
	perm: u8,
	vpn0: u64,
	offset: u64,
	is_write: bool,
) -> Option<TranslateResult> {
	let need_perm = if is_write { PTE_R | PTE_W } else { PTE_R };
	if perm as u64 & need_perm != need_perm {
		return None;
	}
	let want_a = l1_raw & PTE_A == 0;
	let want_d = is_write && l1_raw & PTE_D == 0;
	if want_a || want_d {
		let mut new_raw = l1_raw;
		if want_a {
			new_raw |= PTE_A;
		}
		if want_d {
			new_raw |= PTE_D;
		}
		if !write_pte_cas(ctx, l1_addr, l1_raw, new_raw) {
			// CAS failed: another hart modified this PTE concurrently.
			// Re-read and re-validate to avoid using a stale mapping.
			let raw2 = read_pte(ctx, l1_addr);
			let pte2 = pte_parse(raw2);
			if pte2.v == 0 || pte2.is_leaf == 0 {
				return None;
			}
			if pte2.perm as u64 & need_perm != need_perm {
				return None;
			}
			let ppn2 = (pte2.ppn & 0xFFFF_FFFF_FFFF_FE00u64) | vpn0;
			let pa2 = (ppn2 << 12) | offset;
			return Some(TranslateResult {
				pa: pa2 & 0xFFFF_FFFF_FFFF_FFFF,
				perm: pte2.perm,
				level: 1,
			});
		}
	}

	// PPN[43:9] from PTE, PPN[8:0] from VA vpn0
	let ppn = (l1_ppn & 0xFFFF_FFFF_FFFF_FE00u64) | vpn0;
	let pa = (ppn << 12) | offset;
	Some(TranslateResult {
		pa: pa & 0xFFFF_FFFF_FFFF_FFFF,
		perm,
		level: 1,
	})
}

/// From a valid L1 pointer, read L0; return a 4 KiB leaf result or `None`.
/// Sets A (always) and D (on write) bits on the leaf PTE.
fn walk_l0(
	ctx: &WalkCtx,
	l1_ppn: u64,
	vpn0: u64,
	offset: u64,
	is_write: bool,
) -> Option<TranslateResult> {
	let addr = (l1_ppn << 12).wrapping_add(vpn0 * 8);
	let raw = read_pte(ctx, addr);
	let pte = pte_parse(raw);
	if pte.v == 0 || pte.is_leaf == 0 {
		return None;
	}
	let need_perm = if is_write { PTE_R | PTE_W } else { PTE_R };
	if pte.perm as u64 & need_perm != need_perm {
		return None;
	}
	let want_a = raw & PTE_A == 0;
	let want_d = is_write && raw & PTE_D == 0;
	if want_a || want_d {
		let mut new_raw = raw;
		if want_a {
			new_raw |= PTE_A;
		}
		if want_d {
			new_raw |= PTE_D;
		}
		if !write_pte_cas(ctx, addr, raw, new_raw) {
			let raw2 = read_pte(ctx, addr);
			let pte2 = pte_parse(raw2);
			if pte2.v == 0 || pte2.is_leaf == 0 {
				return None;
			}
			if pte2.perm as u64 & need_perm != need_perm {
				return None;
			}
			let pa = (pte2.ppn << 12) | offset;
			return Some(TranslateResult {
				pa: pa & 0xFFFF_FFFF_FFFF_FFFF,
				perm: pte2.perm,
				level: 0,
			});
		}
	}

	let pa = (pte.ppn << 12) | offset;
	Some(TranslateResult {
		pa: pa & 0xFFFF_FFFF_FFFF_FFFF,
		perm: pte.perm,
		level: 0,
	})
}

// ============================================================
//  Diagnostic helpers — PTE consistency verification
// ============================================================

/// VPNs whose page walks are traced in detail (diagnostic builds only).
/// Set ``PYREMU_TRACE_VPN`` env var (hex, no 0x prefix) at runtime to override.
#[cfg(feature = "diagnostic")]
#[allow(dead_code)]
#[allow(unused)]
fn is_trace_vpn(vpn: u64) -> bool {
	static TRACE_VPN: AtomicU64 = AtomicU64::new(u64::MAX);
	let mut target = TRACE_VPN.load(Ordering::Relaxed);
	if target == u64::MAX {
		target = std::env::var("PYREMU_TRACE_VPN")
			.ok()
			.and_then(|s| u64::from_str_radix(&s, 16).ok())
			.unwrap_or(0);
		TRACE_VPN.store(target, Ordering::Relaxed);
	}
	target != 0 && vpn == target
}

// ============================================================
//  Sv39 page-table walk — entry point
// ============================================================

/// Perform a full Sv39 page-table walk with hardware A/D bit updates.
///
/// Returns ``Some(TranslateResult)`` on success (4 KiB or 2 MiB leaf),
/// or ``None`` if the walk encounters an invalid/non-resident PTE.
pub fn sv39_walk(ctx: &WalkCtx, satp: u64, va: u64, is_write: bool) -> Option<TranslateResult> {
	let mode = satp >> 60;
	if mode != SATP_MODE_SV39 {
		return None;
	}
	let root_ppn = satp & 0xF_FFFF_FFFF;
	let vpn = sv39_decompose_va(va);

	let (_l2_raw, l2_pte) = walk_l2(ctx, root_ppn, vpn.vpn2)?;

	if l2_pte.is_ptr == 0 {
		return None;
	}

	let (l1_raw, l1_pte) = walk_l1(ctx, l2_pte.ppn, vpn.vpn1)?;

	if l1_pte.is_leaf != 0 {
		let l1_addr = (l2_pte.ppn << 12).wrapping_add(vpn.vpn1 * 8);
		let r = walk_l1_leaf(
			ctx,
			l1_addr,
			l1_raw,
			l1_pte.ppn,
			l1_pte.perm,
			vpn.vpn0,
			vpn.offset,
			is_write,
		)?;
		return Some(r);
	}

	let r = walk_l0(ctx, l1_pte.ppn, vpn.vpn0, vpn.offset, is_write)?;
	Some(r)
}

// ============================================================
//  TLB operations
// ============================================================

#[inline]
const fn tlb_sz() -> usize {
	TLB_ENTRIES
}

// ============================================================
//  Clock algorithm (second-chance LRU approximation)
// ============================================================

/// Find a victim for eviction using the clock algorithm.
/// Prefers invalid entries; otherwise scans for accessed==0.
#[inline]
fn clock_victim(tlb: &mut [TlbEntry], hand: &mut u8) -> usize {
	let sz = tlb_sz();
	// First pass: prefer invalid entries
	for _ in 0..sz {
		let idx = *hand as usize;
		if tlb[idx].valid == 0 {
			*hand = ((idx + 1) % sz) as u8;
			return idx;
		}
		*hand = ((idx + 1) % sz) as u8;
	}
	// Second pass: clock algorithm — find unaccessed entry
	for _ in 0..sz {
		let idx = *hand as usize;
		if tlb[idx].accessed == 0 {
			*hand = ((idx + 1) % sz) as u8;
			return idx;
		}
		// Give a second chance: clear accessed, advance hand
		let e = unsafe { &mut *tlb.as_mut_ptr().add(idx) };
		e.accessed = 0;
		*hand = ((idx + 1) % sz) as u8;
	}
	// All entries accessed — evict current hand position
	let idx = *hand as usize;
	*hand = ((idx + 1) % sz) as u8;
	idx
}

// ============================================================
//  TLB operations
// ============================================================

/// Result of a TLB lookup.
pub enum TlbResult {
	/// Clean entry found — ready to use.
	Hit(usize),
	/// Dirty entry found at this index — needs re-walk, but caller should
	/// re-insert the new translation at the same index to avoid wasting a slot.
	DirtyReuse(usize),
	/// No matching entry.
	Miss,
}

/// Find a TLB entry matching *vpn*.  Returns:
/// - ``Hit(idx)`` for a clean match (accessed bit already set).
/// - ``DirtyReuse(idx)`` for a stale match — caller must re-walk and insert
///   at this index to avoid leaking the slot.
/// - ``Miss`` if no entry matches.
#[inline]
pub fn tlb_lookup(tlb: &mut [TlbEntry], vpn: u64, asid: u16) -> TlbResult {
	for i in 0..tlb_sz() {
		if tlb[i].valid != 0 && tlb[i].vpn == vpn {
			// ASID-tagged: different address space -> miss.
			// asid=0 (Bare mode) matches any entry (no tag).
			if asid != 0 && tlb[i].asid != 0 && tlb[i].asid != asid {
				continue;
			}
			if tlb[i].dirty != 0 {
				return TlbResult::DirtyReuse(i);
			}
			tlb[i].accessed = 1;
			return TlbResult::Hit(i);
		}
	}
	TlbResult::Miss
}

/// Insert or update a translation at *prefer_idx* (if valid) or via clock
/// eviction.  ``prefer_idx`` should be the index from a prior ``DirtyReuse``.
#[inline]
pub fn tlb_insert(
	tlb: &mut [TlbEntry],
	hand: &mut u8,
	prefer_idx: Option<usize>,
	vpn: u64,
	ppn: u64,
	perm: u8,
	level: u8,
	mdid: u8,
	asid: u16,
) {
	let idx = match prefer_idx {
		Some(i) if i < tlb_sz() => i,
		_ => clock_victim(tlb, hand),
	};
	tlb[idx].vpn = vpn;
	tlb[idx].ppn = ppn;
	tlb[idx].perm = perm;
	tlb[idx].level = level;
	tlb[idx].valid = 1;
	tlb[idx].mdid = mdid;
	tlb[idx].dirty = 0;
	tlb[idx].accessed = 1;
	tlb[idx].asid = asid;
	tlb[idx].tlb_epoch = 0;
}

/// Mark all valid TLB entries as dirty (called when tlb_gen changes).
/// The next lookup will re-walk and refresh the PPN.
#[inline]
pub fn tlb_mark_all_dirty(tlb: &mut [TlbEntry]) {
	for e in tlb.iter_mut() {
		if e.valid != 0 {
			e.dirty = 1;
		}
	}
}

/// Flush the entire TLB (invalidate all entries).
#[inline]
pub fn tlb_flush_all(tlb: &mut [TlbEntry]) {
	for e in tlb.iter_mut() {
		e.valid = 0;
		e.dirty = 0;
		e.accessed = 0;
	}
}

/// Flush a specific VPN from the TLB.
#[inline]
pub fn tlb_flush_vpn(tlb: &mut [TlbEntry], vpn: u64) {
	for e in tlb.iter_mut() {
		if e.valid != 0 && e.vpn == vpn {
			e.valid = 0;
		}
	}
}

// ============================================================
//  translate_va helpers — effective mode + permission check
// ============================================================

/// Compute the effective privilege level for memory translation.
///
/// In M-mode, `MPRV=1` overrides the effective mode to `MPP` (RISC-V
/// Privileged Spec §4.1.12).  If MPRV=0 or MPP=M, returns `None`
/// (caller should use Bare translation).
///
/// Non-M modes use their actual privilege level.
/// Effective privilege mode for MMU translation.
///
/// Returns `None` when the MMU should be bypassed (Bare translation):
/// - M-mode without MPRV, or MPRV pointing back to M-mode
/// - D-mode (debug mode) — always bypasses MMU, same as M-mode
fn effective_mode(state: &HartState) -> Option<u8> {
	// D-mode bypasses MMU just like M-mode (matches Python RiscvMode.D).
	if state.mode == riscv_mode::D {
		return None;
	}
	if state.mode == riscv_mode::M {
		let mprv = (state.mstatus >> 17) & 1;
		let mpp = (state.mstatus >> 11) & 3;
		if mprv == 0 || mpp == riscv_mode::M as u64 {
			return None; // Bare translation
		}
		return Some(mpp as u8);
	}
	Some(state.mode)
}

/// Check PTE permissions with SUM / MXR / U-bit awareness.
///
/// Returns `true` if the access is allowed.
fn check_pte_perm(perm: u8, eff_mode: u8, mstatus: u64, is_write: bool, is_execute: bool) -> bool {
	let perm = perm as u64;
	let pte_u = (perm & PTE_U) != 0;
	let pte_r = (perm & PTE_R) != 0;
	let pte_w = (perm & PTE_W) != 0;
	let pte_x = (perm & PTE_X) != 0;

	// U-bit check
	if eff_mode == riscv_mode::U {
		if !pte_u {
			return false;
		}
	} else if eff_mode == riscv_mode::S {
		// SUM=1 allows S-mode to access user (U=1) pages
		if pte_u && (mstatus & (1 << 18)) == 0 {
			return false;
		}
	}

	if is_execute {
		return pte_x;
	}
	if is_write {
		return pte_r && pte_w;
	}
	// Read: MXR=1 allows executable-only pages to be readable
	if !pte_r {
		if (mstatus & (1 << 19)) != 0 && pte_x {
			return true;
		}
		return false;
	}
	true
}

/// Reconstruct a physical address from a TLB entry and virtual address.
#[inline]
fn reconstruct_tlb_pa(entry: &TlbEntry, va: u64) -> u64 {
	let offset = va & 0xFFF;
	let pa = if entry.level == 1 {
		// 2 MiB superpage: PPN[8:0] from VA vpn0
		let ppn = (entry.ppn & 0xFFFF_FFFF_FFFF_FE00) | ((va >> 12) & 0x1FF);
		(ppn << 12) | offset
	} else {
		(entry.ppn << 12) | offset
	};
	pa & 0xFFFF_FFFF_FFFF_FFFF
}

/// Return the appropriate page-fault cause code for the access type.
#[inline]
fn fault_code(is_execute: bool, is_write: bool) -> u64 {
	if is_execute {
		12
	} else if is_write {
		15
	} else {
		13
	}
}

// ============================================================
//  TLB probe with SFENCE.VMA TOCTOU protection
// ============================================================

/// Result of probing the TLB with generation-counter TOCTOU protection.
enum TlbProbeResult {
	/// Clean TLB hit — return this translation immediately.
	Hit(TranslateResult),
	/// TLB hit but permission check failed — return this fault.
	Fault(TranslateFault),
	/// Fall through to page walk, optionally reusing this index.
	Walk(Option<usize>),
}

/// Probe the TLB for *vpn*, with SFENCE.VMA TOCTOU protection when
/// ``tlb_gen`` is available (multi-hart concurrent path).  Falls back to a
/// plain lookup when ``tlb_gen`` is ``None``
#[inline]
fn tlb_probe(
	tlb: &mut [TlbEntry],
	vpn: u64,
	va: u64,
	asid: u16,
	tlb_gen: Option<&AtomicU64>,
	eff_mode: u8,
	mstatus: u64,
	is_write: bool,
	is_execute: bool,
) -> TlbProbeResult {
	let gen_before = tlb_gen.map(|g| g.load(Ordering::Acquire));
	match tlb_lookup(tlb, vpn, asid) {
		TlbResult::Hit(idx) => {
			if let Some(g) = tlb_gen {
				if gen_before != Some(g.load(Ordering::Acquire)) {
					tlb[idx].valid = 0; // stale — invalidate, walk
					return TlbProbeResult::Walk(None);
				}
			}
			let e = &tlb[idx];
			if !check_pte_perm(e.perm, eff_mode, mstatus, is_write, is_execute) {
				return TlbProbeResult::Fault(TranslateFault::PageFault(fault_code(
					is_execute, is_write,
				)));
			}
			let pa = reconstruct_tlb_pa(e, va);
			TlbProbeResult::Hit(TranslateResult {
				pa,
				perm: e.perm,
				level: e.level,
			})
		}
		TlbResult::DirtyReuse(idx) => TlbProbeResult::Walk(Some(idx)),
		TlbResult::Miss => TlbProbeResult::Walk(None),
	}
}

// ============================================================
//  Main translate helper
// ============================================================

/// Translate a virtual address to a physical address.
///
/// Returns ``Ok(TranslateResult)`` with PA, permissions, and level,
/// or ``Err(TranslateFault)`` on failure.
pub fn translate_va(
	state: &mut HartState,
	ctx: &WalkCtx,
	va: u64,
	is_write: bool,
	is_execute: bool,
) -> Result<TranslateResult, TranslateFault> {
	// ---- Bare mode / M-mode bypass ----
	if state.mmu_mode != 8 {
		return Ok(TranslateResult {
			pa: va & 0xFFFF_FFFF_FFFF_FFFF,
			perm: 0xF,
			level: 0,
		});
	}

	let eff_mode = match effective_mode(state) {
		Some(m) => m,
		None => {
			return Ok(TranslateResult {
				pa: va & 0xFFFF_FFFF_FFFF_FFFF,
				perm: 0xF,
				level: 0,
			});
		}
	};

	let vpn = va >> 12;
	let (tlb, hand) = if is_execute {
		(&mut state.itlb, &ctx.itlb_hand)
	} else {
		(&mut state.dtlb, &ctx.dtlb_hand)
	};

	// ---- TLB bypass (PYREMU_NO_TLB=1) ----
	// Force page walk on every translation, identical to NO_L2 for L2 cache.
	static NO_TLB: AtomicU64 = AtomicU64::new(u64::MAX);
	let no_tlb = {
		let v = NO_TLB.load(Ordering::Relaxed);
		if v == u64::MAX {
			let val: u64 = std::env::var("PYREMU_NO_TLB")
				.ok()
				.and_then(|s| s.parse().ok())
				.unwrap_or(0);
			NO_TLB.store(val, Ordering::Relaxed);
			val
		} else {
			v
		}
	};

	// ---- TLB lookup ----
	let asid = ((state.satp >> 44) & 0xFFFF) as u16;

	// Record gen before TLB probe so we can detect SFENCE.VMA broadcasts
	// from other harts that happen DURING the page-table walk.  Without this
	// re-check, a concurrent SFENCE.VMA after ``tlb_probe`` but before
	// ``tlb_insert`` would leave a stale entry in the TLB — the next lookup
	// would hit and return the now-invalid PPN.
	let gen_before = if ctx.tlb_gen.is_null() {
		None
	} else {
		Some(unsafe { &*ctx.tlb_gen }.load(Ordering::Acquire))
	};

	let reuse_idx = if no_tlb != 0 {
		None
	} else {
		match tlb_probe(
			tlb,
			vpn,
			va,
			asid,
			if ctx.tlb_gen.is_null() {
				None
			} else {
				Some(unsafe { &*ctx.tlb_gen })
			},
			eff_mode,
			state.mstatus,
			is_write,
			is_execute,
		) {
			TlbProbeResult::Hit(result) => return Ok(result),
			TlbProbeResult::Fault(fault) => return Err(fault),
			TlbProbeResult::Walk(reuse) => reuse,
		}
	};

	// ---- Page-table walk ----
	let result = sv39_walk(ctx, state.satp, va, is_write)
		.ok_or_else(|| TranslateFault::PageFault(fault_code(is_execute, is_write)))?;

	if !check_pte_perm(result.perm, eff_mode, state.mstatus, is_write, is_execute) {
		return Err(TranslateFault::PageFault(fault_code(is_execute, is_write)));
	}

	// ---- Re-check gen after walk, before TLB insert ----
	// If another hart executed SFENCE.VMA during our walk, the PTE we just
	// read may be stale.  Skip the TLB insert — the stale entry won't be
	// cached, and the next lookup will re-walk with the current PTE.
	// This closes the TOCTOU window between tlb_probe and tlb_insert.
	let gen_valid = match gen_before {
		None => true,
		Some(gb) => {
			let ga = unsafe { &*ctx.tlb_gen }.load(Ordering::Acquire);
			gb == ga
		}
	};

	// ---- Insert into TLB (only if gen is still valid) ----
	if gen_valid {
		let tlb_mut = if is_execute {
			&mut state.itlb
		} else {
			&mut state.dtlb
		};
		let mut h = hand.get();
		tlb_insert(
			tlb_mut,
			&mut h,
			reuse_idx,
			vpn,
			result.ppn_for_tlb(vpn),
			result.perm,
			result.level,
			state.mdid,
			asid,
		);
		hand.set(h);
	}

	Ok(result)
}

impl TranslateResult {
	/// Reconstruct the PPN value to store in TLB.
	fn ppn_for_tlb(&self, _vpn: u64) -> u64 {
		if self.level == 1 {
			(self.pa >> 12) & 0xFFFF_FFFF_FFFF_FE00
		} else {
			self.pa >> 12
		}
	}
}

// ============================================================
//  Tests
// ============================================================

#[cfg(test)]
mod tests {
	use super::*;
	use crate::state::TlbEntry;

	/// Build a minimal page-table in a Vec and return (vec, WalkCtx).
	/// Places the L2 table at offset 0x0, L1 at 0x1000, L0 at 0x2000.
	fn setup_4k_page_table(ram: &mut [u8], va: u64, pa: u64, perm: u64) -> u64 {
		let vpn = sv39_decompose_va(va);
		let satp = 8u64 << 60; // Sv39, root PPN = 0

		// L2: pointer to L1 at PPN=1 (addr 0x1000)
		let l2_off = (vpn.vpn2 * 8) as usize;
		let l2_val = (1u64 << 10) | PTE_V;
		ram[l2_off..l2_off + 8].copy_from_slice(&l2_val.to_le_bytes());

		// L1: pointer to L0 at PPN=2 (addr 0x2000)
		let l1_off = 0x1000usize + (vpn.vpn1 * 8) as usize;
		let l1_val = (2u64 << 10) | PTE_V;
		ram[l1_off..l1_off + 8].copy_from_slice(&l1_val.to_le_bytes());

		// L0: leaf PTE mapping va->pa
		let l0_off = 0x2000usize + (vpn.vpn0 * 8) as usize;
		let l0_ppn = pa >> 12;
		let l0_val = (l0_ppn << 10) | perm | PTE_V;
		ram[l0_off..l0_off + 8].copy_from_slice(&l0_val.to_le_bytes());

		satp
	}

	#[test]
	fn tlb_lookup_miss() {
		let mut tlb: [TlbEntry; TLB_ENTRIES] = [TlbEntry::empty(); TLB_ENTRIES];
		assert!(matches!(tlb_lookup(&mut tlb, 0x100, 0), TlbResult::Miss));
	}

	#[test]
	fn tlb_insert_and_lookup() {
		let mut tlb: [TlbEntry; TLB_ENTRIES] = [TlbEntry::empty(); TLB_ENTRIES];
		let mut hand: u8 = 0;
		tlb_insert(&mut tlb, &mut hand, None, 0x1000, 0x80000, 0xF, 0, 0, 0);
		let idx = match tlb_lookup(&mut tlb, 0x1000, 0) {
			TlbResult::Hit(i) => i,
			_ => panic!("expected clean hit"),
		};
		let e = &tlb[idx];
		assert_eq!(e.ppn, 0x80000);
		assert_eq!(e.perm, 0xF);
		assert_eq!(e.level, 0);
		assert_eq!(e.valid, 1);
		assert_eq!(e.dirty, 0);
		// Mark dirty -> returns DirtyReuse, not Miss
		tlb[idx].dirty = 1;
		let reuse = match tlb_lookup(&mut tlb, 0x1000, 0) {
			TlbResult::DirtyReuse(i) => i,
			other => panic!(
				"expected DirtyReuse, got {:?}",
				match other {
					TlbResult::Hit(_) => "Hit",
					TlbResult::Miss => "Miss",
					TlbResult::DirtyReuse(_) => unreachable!(),
				}
			),
		};
		assert_eq!(reuse, idx);
		// Re-insert at the dirty slot
		tlb_insert(
			&mut tlb,
			&mut hand,
			Some(reuse),
			0x1000,
			0x90000,
			0xF,
			0,
			0,
			0,
		);
		assert!(matches!(tlb_lookup(&mut tlb, 0x1000, 0), TlbResult::Hit(_)));
		assert_eq!(tlb[reuse].ppn, 0x90000);
	}

	#[test]
	fn test_tlb_flush_all() {
		let mut tlb: [TlbEntry; TLB_ENTRIES] = [TlbEntry::empty(); TLB_ENTRIES];
		let mut hand: u8 = 0;
		tlb_insert(&mut tlb, &mut hand, None, 0x1000, 0x80000, 0xF, 0, 0, 0);
		crate::translate::tlb_flush_all(&mut tlb);
		assert!(matches!(tlb_lookup(&mut tlb, 0x1000, 0), TlbResult::Miss));
	}

	#[test]
	fn test_tlb_flush_vpn() {
		let mut tlb: [TlbEntry; TLB_ENTRIES] = [TlbEntry::empty(); TLB_ENTRIES];
		let mut hand: u8 = 0;
		tlb_insert(&mut tlb, &mut hand, None, 0x1000, 0x80000, 0xF, 0, 0, 0);
		tlb_insert(&mut tlb, &mut hand, None, 0x2000, 0x90000, 0xF, 0, 0, 0);
		crate::translate::tlb_flush_vpn(&mut tlb, 0x1000);
		assert!(matches!(tlb_lookup(&mut tlb, 0x1000, 0), TlbResult::Miss));
		assert!(matches!(tlb_lookup(&mut tlb, 0x2000, 0), TlbResult::Hit(_)));
	}

	#[test]
	fn tlb_clock_evicts() {
		let mut tlb: [TlbEntry; TLB_ENTRIES] = [TlbEntry::empty(); TLB_ENTRIES];
		let mut hand: u8 = 0;
		for i in 0..TLB_ENTRIES {
			tlb_insert(
				&mut tlb,
				&mut hand,
				None,
				i as u64,
				i as u64 * 0x1000,
				0xF,
				0,
				0,
				0,
			);
		}
		assert!(matches!(tlb_lookup(&mut tlb, 0, 0), TlbResult::Hit(_)));
		tlb_insert(
			&mut tlb,
			&mut hand,
			None,
			TLB_ENTRIES as u64,
			0xA000,
			0xF,
			0,
			0,
			0,
		);
		let valid_count = tlb.iter().filter(|e| e.valid != 0).count();
		assert_eq!(valid_count, TLB_ENTRIES);
	}

	#[test]
	fn tlb_dirty_reuse() {
		let mut tlb: [TlbEntry; TLB_ENTRIES] = [TlbEntry::empty(); TLB_ENTRIES];
		let mut hand: u8 = 0;
		// Fill all entries clean
		for i in 0..TLB_ENTRIES {
			tlb_insert(
				&mut tlb,
				&mut hand,
				None,
				i as u64,
				i as u64 * 0x1000,
				0xF,
				0,
				0,
				0,
			);
		}
		// Mark entry 5 dirty
		tlb[5].dirty = 1;
		// Re-insert at dirty slot — should reuse slot 5, not evict another
		tlb_insert(&mut tlb, &mut hand, Some(5), 0x1000, 0x90000, 0xF, 0, 0, 0);
		assert_eq!(tlb[5].ppn, 0x90000);
		assert_eq!(tlb[5].dirty, 0);
		let valid_count = tlb.iter().filter(|e| e.valid != 0).count();
		assert_eq!(valid_count, TLB_ENTRIES); // no extra entry leaked
	}

	#[test]
	fn sv39_walk_sets_a_bit_on_leaf() {
		let mut ram = vec![0u8; 0x3000];
		let satp = setup_4k_page_table(&mut ram, 0x1000, 0x8000_0000, PTE_R | PTE_W | PTE_X);
		let ctx = WalkCtx {
			ram: ram.as_mut_ptr(),
			ram_size: ram.len() as u64,
			ram_base: 0,
			shadow_base: 0,
			shadow_size: 0,
			tlb_gen: std::ptr::null(),
			itlb_hand: Cell::new(0),
			dtlb_hand: Cell::new(0),
			lr_reserved: std::ptr::null_mut(),
			num_harts: 1,
		};

		let result = sv39_walk(&ctx, satp, 0x1000, false);
		assert!(result.is_some());
		let l0_off = 0x2000usize + (sv39_decompose_va(0x1000).vpn0 * 8) as usize;
		let l0_val = u64::from_le_bytes(ram[l0_off..l0_off + 8].try_into().unwrap());
		assert_ne!(l0_val & PTE_A, 0, "A bit should be set on leaf PTE");
	}

	#[test]
	fn sv39_walk_sets_d_bit_on_write() {
		let mut ram = vec![0u8; 0x3000];
		let satp = setup_4k_page_table(&mut ram, 0x2000, 0x8000_1000, PTE_R | PTE_W);
		let ctx = WalkCtx {
			ram: ram.as_mut_ptr(),
			ram_size: ram.len() as u64,
			ram_base: 0,
			shadow_base: 0,
			shadow_size: 0,
			tlb_gen: std::ptr::null(),
			itlb_hand: Cell::new(0),
			dtlb_hand: Cell::new(0),
			lr_reserved: std::ptr::null_mut(),
			num_harts: 1,
		};

		let result = sv39_walk(&ctx, satp, 0x2000, true);
		assert!(result.is_some());
		let l0_off = 0x2000usize + (sv39_decompose_va(0x2000).vpn0 * 8) as usize;
		let l0_val = u64::from_le_bytes(ram[l0_off..l0_off + 8].try_into().unwrap());
		assert_ne!(
			l0_val & PTE_D,
			0,
			"D bit should be set on write to leaf PTE"
		);
	}

	#[test]
	fn sv39_walk_sets_a_bit_on_intermediate_levels() {
		let mut ram = vec![0u8; 0x3000];
		let satp = setup_4k_page_table(&mut ram, 0x3000, 0x8000_2000, PTE_R | PTE_W);
		let ctx = WalkCtx {
			ram: ram.as_mut_ptr(),
			ram_size: ram.len() as u64,
			ram_base: 0,
			shadow_base: 0,
			shadow_size: 0,
			tlb_gen: std::ptr::null(),
			itlb_hand: Cell::new(0),
			dtlb_hand: Cell::new(0),
			lr_reserved: std::ptr::null_mut(),
			num_harts: 1,
		};

		let _result = sv39_walk(&ctx, satp, 0x3000, false);
		let l2_off = (sv39_decompose_va(0x3000).vpn2 * 8) as usize;
		let l2_val = u64::from_le_bytes(ram[l2_off..l2_off + 8].try_into().unwrap());
		assert_ne!(l2_val & PTE_A, 0, "A bit should be set on L2 PTE");

		let l1_off = 0x1000usize + (sv39_decompose_va(0x3000).vpn1 * 8) as usize;
		let l1_val = u64::from_le_bytes(ram[l1_off..l1_off + 8].try_into().unwrap());
		assert_ne!(l1_val & PTE_A, 0, "A bit should be set on L1 PTE");
	}

	#[test]
	fn bare_mode_translate_is_identity() {}

	#[test]
	fn mmode_bypasses_sv39_even_when_satp_is_sv39() {
		let mut state: HartState = unsafe { std::mem::zeroed() };
		state.mode = riscv_mode::M;
		state.mmu_mode = 8;
		state.satp = 8u64 << 60;
		state.mstatus = 0;

		let mut ram = vec![0u8; 0x1000];
		let ctx = WalkCtx {
			ram: ram.as_mut_ptr(),
			ram_size: ram.len() as u64,
			ram_base: 0,
			shadow_base: 0,
			shadow_size: 0,
			tlb_gen: std::ptr::null(),
			itlb_hand: Cell::new(0),
			dtlb_hand: Cell::new(0),
			lr_reserved: std::ptr::null_mut(),
			num_harts: 1,
		};

		let result = translate_va(&mut state, &ctx, 0x8000_1000, false, false);
		assert!(
			result.is_ok(),
			"M-mode with MPRV=0 must use Bare translation"
		);
		let tr = result.unwrap();
		assert_eq!(tr.pa, 0x8000_1000);
		assert_eq!(tr.level, 0);
	}

	#[test]
	fn mmode_mprv1_with_mpp_s_uses_sv39() {
		let mut state: HartState = unsafe { std::mem::zeroed() };
		state.mode = riscv_mode::M;
		state.mmu_mode = 8;
		state.satp = 8u64 << 60;
		state.mstatus = (1u64 << 17) | (1u64 << 11);

		let mut ram = vec![0u8; 0x1000];
		let ctx = WalkCtx {
			ram: ram.as_mut_ptr(),
			ram_size: ram.len() as u64,
			ram_base: 0,
			shadow_base: 0,
			shadow_size: 0,
			tlb_gen: std::ptr::null(),
			itlb_hand: Cell::new(0),
			dtlb_hand: Cell::new(0),
			lr_reserved: std::ptr::null_mut(),
			num_harts: 1,
		};

		let result = translate_va(&mut state, &ctx, 0x8000_1000, false, false);
		assert!(
			result.is_err(),
			"MPRV=1+MPP=S must attempt Sv39 translation"
		);
	}

	/// D-mode (debug mode) must bypass Sv39 translation just like M-mode.
	/// Regression: effective_mode() previously returned `Some(D)` which
	/// caused page-walk attempts in debug mode.
	#[test]
	fn dmode_bypasses_sv39_translation() {
		let mut state: HartState = unsafe { std::mem::zeroed() };
		state.mode = riscv_mode::D;
		state.mmu_mode = 8;
		state.satp = 8u64 << 60; // Sv39, root PPN = 0

		let mut ram = vec![0u8; 0x1000];
		let ctx = WalkCtx {
			ram: ram.as_mut_ptr(),
			ram_size: ram.len() as u64,
			ram_base: 0,
			shadow_base: 0,
			shadow_size: 0,
			tlb_gen: std::ptr::null(),
			itlb_hand: Cell::new(0),
			dtlb_hand: Cell::new(0),
			lr_reserved: std::ptr::null_mut(),
			num_harts: 1,
		};

		let result = translate_va(&mut state, &ctx, 0x8000_1000, false, false);
		assert!(
			result.is_ok(),
			"D-mode must use Bare translation (bypass Sv39)"
		);
		let tr = result.unwrap();
		assert_eq!(tr.pa, 0x8000_1000, "D-mode VA must equal PA");
	}

	/// D-mode instruction fetch must also bypass MMU translation.
	#[test]
	fn dmode_fetch_bypasses_sv39() {
		let mut state: HartState = unsafe { std::mem::zeroed() };
		state.mode = riscv_mode::D;
		state.mmu_mode = 8;
		state.satp = 8u64 << 60;

		let mut ram = vec![0u8; 0x1000];
		let ctx = WalkCtx {
			ram: ram.as_mut_ptr(),
			ram_size: ram.len() as u64,
			ram_base: 0,
			shadow_base: 0,
			shadow_size: 0,
			tlb_gen: std::ptr::null(),
			itlb_hand: Cell::new(0),
			dtlb_hand: Cell::new(0),
			lr_reserved: std::ptr::null_mut(),
			num_harts: 1,
		};

		let pa = crate::hart_sched::translate_fetch_pc_concurrent(&mut state, &ctx, 0x8000_2000);
		assert_eq!(
			pa,
			Some(0x8000_2000),
			"D-mode fetch must bypass Sv39 (VA==PA)"
		);
	}
}
