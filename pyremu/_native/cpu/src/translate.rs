//! Sv39 page-table walk and TLB management for the batch execution engine.
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

use core::sync::atomic::{AtomicU64, Ordering};
use crate::mmu::{pte_parse, sv39_decompose_va};
use crate::state::{riscv_mode, HartState, TlbEntry};

// ============================================================
//  Page-table walk context
// ============================================================

/// Parameters needed for memory access during page walks.
/// ``ram`` is ``*mut u8`` so that A/D bit updates can be written back.
pub struct WalkCtx {
    pub ram: *mut u8,
    pub ram_size: u64,
    pub ram_base: u64,
    pub shadow_base: u64,
    pub shadow_size: u64,
    /// Pointer to the global TLB generation counter in ``ModuleState``.
    /// When non-null, ``translate_va`` reads the current generation directly
    /// instead of relying on the cached value in ``HartState._mmu_mode_pad``.
    /// This closes the window where another hart increments the generation
    /// between the instruction-boundary gen check and the actual load/store
    /// that uses the TLB.
    pub tlb_gen: *const AtomicU64,
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
    a.load(Ordering::Acquire)
}

/// Write a 64-bit raw PTE value to physical memory (A/D bit update).
///
/// Uses ``AtomicU64::store(Release)`` to pair with the ``Acquire`` load in
/// ``read_pte`` so that A/D-bit updates made by one hart's walker are visible
/// to another hart's walker.  Skipping the write when the value is unchanged
/// is handled by the caller (``write_pte_if_changed``).
#[inline]
fn write_pte_raw(ctx: &WalkCtx, pa: u64, val: u64) {
    let off = match ram_offset(ctx, pa, 8) {
        Some(o) => o,
        None => return,
    };
    let ptr = unsafe { ctx.ram.add(off) };
    let a = unsafe { &*(ptr as *mut AtomicU64) };
    a.store(val, Ordering::Release);
}

/// Write-back only if *val* differs from the current PTE (spare writes).
#[inline]
fn write_pte_if_changed(ctx: &WalkCtx, pa: u64, old: u64, new: u64) {
    if new != old {
        write_pte_raw(ctx, pa, new);
    }
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
        write_pte_raw(ctx, addr, raw | PTE_A);
    }
    Some((raw, pte))
}

/// From a valid L2 pointer, read L1; return `(raw, pte)` or `None`.
/// Sets A bit on the L1 PTE.
fn walk_l1(ctx: &WalkCtx, l2_ppn: u64, vpn1: u64) -> Option<(u64, crate::mmu::PteFields)> {
    let addr = (l2_ppn << 12).wrapping_add(vpn1 * 8);
    let raw = read_pte(ctx, addr);
    let pte = pte_parse(raw);
    if pte.v == 0 || pte.is_leaf == 0 && pte.is_ptr == 0 {
        return None;
    }
    if raw & PTE_A == 0 {
        write_pte_raw(ctx, addr, raw | PTE_A);
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
    let mut new_raw = l1_raw | PTE_A;
    if is_write {
        new_raw |= PTE_D;
    }
    write_pte_if_changed(ctx, l1_addr, l1_raw, new_raw);

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
    let mut new_raw = raw | PTE_A;
    if is_write {
        new_raw |= PTE_D;
    }
    write_pte_if_changed(ctx, addr, raw, new_raw);

    let pa = (pte.ppn << 12) | offset;
    Some(TranslateResult {
        pa: pa & 0xFFFF_FFFF_FFFF_FFFF,
        perm: pte.perm,
        level: 0,
    })
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
        return None; // 1 GiB leaf not supported in Sv39
    }

    let (l1_raw, l1_pte) = walk_l1(ctx, l2_pte.ppn, vpn.vpn1)?;

    if l1_pte.is_leaf != 0 {
        return walk_l1_leaf(
            ctx,
            (l2_pte.ppn << 12).wrapping_add(vpn.vpn1 * 8),
            l1_raw,
            l1_pte.ppn,
            l1_pte.perm,
            vpn.vpn0,
            vpn.offset,
            is_write,
        );
    }

    walk_l0(ctx, l1_pte.ppn, vpn.vpn0, vpn.offset, is_write)
}

// ============================================================
//  TLB operations
// ============================================================

/// TLB epoch layout inside ``HartState._mmu_mode_pad``:
///
///   Bits [4:0]   = dtlb FIFO insert index (0–31)
///   Bits [12:8]  = itlb FIFO insert index (0–31)
///   Bits [44:13] = tlb_epoch — low 32 bits of the global ``tlb_gen``
///                  counter stored by ``hart_worker`` after each flush
///
/// The epoch is *not* synchronised with the global gen atomically —
/// it is only written after the TLB is flushed, so it is always
/// monotonically non-decreasing for the local hart.  A TLB entry
/// inserted under epoch *E* is valid iff the current epoch still
/// equals *E*, i.e. no SFENCE.VMA has been observed since the entry
/// was populated.
const TLB_EPOCH_SHIFT: u64 = 13;
const TLB_EPOCH_MASK: u64 = 0xFFFF_FFFF; // 32 bits

#[inline]
fn get_tlb_epoch(state: &HartState) -> u32 {
    ((state._mmu_mode_pad >> TLB_EPOCH_SHIFT) & TLB_EPOCH_MASK) as u32
}

/// Advance the local TLB epoch to *epoch*, forcing all previously
/// cached entries to be treated as stale on the next lookup.
#[inline]
pub fn set_tlb_epoch(state: &mut HartState, epoch: u32) {
    let fifo_bits = state._mmu_mode_pad & 0x1FFF;
    state._mmu_mode_pad = fifo_bits | ((epoch as u64) << TLB_EPOCH_SHIFT);
}

/// Get the current TLB epoch from the global generation counter
/// (low 32 bits).  Must be called AFTER the Acquire load of
/// ``ModuleState.tlb_gen`` so the happens-before edge is established.
#[inline]
pub fn tlb_epoch_from_gen(gen: u64) -> u32 {
    (gen & TLB_EPOCH_MASK) as u32
}

#[inline]
const fn tlb_sz() -> usize {
    32
}

/// Find a TLB entry matching *vpn* AND *epoch*.  Returns index or None.
///
/// Entries populated before the most recent SFENCE.VMA (i.e. whose
/// ``tlb_epoch`` differs from the requested *epoch*) are silently
/// skipped — the caller will fall through to a page walk, which reads
/// the current PTE from RAM.
#[inline]
pub fn tlb_lookup(tlb: &[TlbEntry; 32], vpn: u64, epoch: u32) -> Option<usize> {
    for i in 0..tlb_sz() {
        if tlb[i].valid != 0 && tlb[i].vpn == vpn && tlb[i].tlb_epoch == epoch {
            return Some(i);
        }
    }
    None
}

/// Insert a translation into the TLB (FIFO replacement).
#[inline]
pub fn tlb_insert(
    tlb: &mut [TlbEntry; 32],
    ins_idx: &mut usize,
    vpn: u64,
    ppn: u64,
    perm: u8,
    level: u8,
    mdid: u8,
    epoch: u32,
) {
    let idx = *ins_idx;
    tlb[idx].vpn = vpn;
    tlb[idx].ppn = ppn;
    tlb[idx].perm = perm;
    tlb[idx].level = level;
    tlb[idx].valid = 1;
    tlb[idx].mdid = mdid;
    tlb[idx].tlb_epoch = epoch;
    *ins_idx = (idx + 1) % tlb_sz();
}

/// Flush the entire TLB (SFENCE.VMA).
#[inline]
pub fn tlb_flush_all(tlb: &mut [TlbEntry; 32]) {
    for e in tlb.iter_mut() {
        e.valid = 0;
    }
}

/// Flush a specific VPN from the TLB.
#[inline]
pub fn tlb_flush_vpn(tlb: &mut [TlbEntry; 32], vpn: u64) {
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
    let tlb = if is_execute { &state.itlb } else { &state.dtlb };
    // Use the *current* global generation as the epoch, not the cached
    // copy in HartState.  The cached copy is only refreshed at instruction
    // boundaries, but SFENCE.VMA on another hart can happen between the
    // boundary check and this lookup.  Reading the global counter directly
    // closes that window — a stale TLB entry inserted at a previous
    // generation will have a mismatched epoch and be treated as a miss.
    let epoch = if !ctx.tlb_gen.is_null() {
        tlb_epoch_from_gen(unsafe { (*ctx.tlb_gen).load(Ordering::Acquire) })
    } else {
        get_tlb_epoch(state)
    };

    // ---- TLB lookup ----
    if let Some(idx) = tlb_lookup(tlb, vpn, epoch) {
        let e = &tlb[idx];
        if !check_pte_perm(e.perm, eff_mode, state.mstatus, is_write, is_execute) {
            return Err(TranslateFault::PageFault(fault_code(is_execute, is_write)));
        }
        return Ok(TranslateResult {
            pa: reconstruct_tlb_pa(e, va),
            perm: e.perm,
            level: e.level,
        });
    }

    // ---- Page-table walk ----
    let result = sv39_walk(ctx, state.satp, va, is_write)
        .ok_or_else(|| TranslateFault::PageFault(fault_code(is_execute, is_write)))?;

    if !check_pte_perm(result.perm, eff_mode, state.mstatus, is_write, is_execute) {
        return Err(TranslateFault::PageFault(fault_code(is_execute, is_write)));
    }

    // ---- Insert into TLB ----
    let tlb_mut = if is_execute {
        &mut state.itlb
    } else {
        &mut state.dtlb
    };
    // Extract the FIFO insert index from _mmu_mode_pad.
    // Bits [4:0]   = dtlb insert index
    // Bits [12:8]  = itlb insert index
    let shift = if is_execute { 8 } else { 0 };
    let mut ins_idx = ((state._mmu_mode_pad >> shift) & 0x1F) as usize;
    tlb_insert(
        tlb_mut,
        &mut ins_idx,
        vpn,
        result.ppn_for_tlb(vpn),
        result.perm,
        result.level,
        state.mdid,
        epoch,
    );
    // Store the updated index back so the FIFO advances across calls.
    let mask: u64 = !(0x1F << shift);
    state._mmu_mode_pad = (state._mmu_mode_pad & mask) | ((ins_idx as u64 & 0x1F) << shift);

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

    const TEST_EPOCH: u32 = 42;

    #[test]
    fn tlb_lookup_miss() {
        let tlb: [TlbEntry; 32] = [TlbEntry::empty(); 32];
        assert_eq!(tlb_lookup(&tlb, 0x100, TEST_EPOCH), None);
    }

    #[test]
    fn tlb_insert_and_lookup() {
        let mut tlb: [TlbEntry; 32] = [TlbEntry::empty(); 32];
        let mut ins_idx: usize = 0;
        tlb_insert(&mut tlb, &mut ins_idx, 0x1000, 0x80000, 0xF, 0, 0, TEST_EPOCH);
        let hit = tlb_lookup(&tlb, 0x1000, TEST_EPOCH);
        assert!(hit.is_some());
        let e = &tlb[hit.unwrap()];
        assert_eq!(e.ppn, 0x80000);
        assert_eq!(e.perm, 0xF);
        assert_eq!(e.level, 0);
        assert_eq!(e.valid, 1);
        // Stale epoch -> miss
        assert_eq!(tlb_lookup(&tlb, 0x1000, TEST_EPOCH + 1), None);
    }

    #[test]
    fn test_tlb_flush_all() {
        let mut tlb: [TlbEntry; 32] = [TlbEntry::empty(); 32];
        let mut ins_idx: usize = 0;
        tlb_insert(&mut tlb, &mut ins_idx, 0x1000, 0x80000, 0xF, 0, 0, TEST_EPOCH);
        crate::translate::tlb_flush_all(&mut tlb);
        assert_eq!(tlb_lookup(&tlb, 0x1000, TEST_EPOCH), None);
    }

    #[test]
    fn test_tlb_flush_vpn() {
        let mut tlb: [TlbEntry; 32] = [TlbEntry::empty(); 32];
        let mut ins_idx: usize = 0;
        tlb_insert(&mut tlb, &mut ins_idx, 0x1000, 0x80000, 0xF, 0, 0, TEST_EPOCH);
        tlb_insert(&mut tlb, &mut ins_idx, 0x2000, 0x90000, 0xF, 0, 0, TEST_EPOCH);
        crate::translate::tlb_flush_vpn(&mut tlb, 0x1000);
        assert_eq!(tlb_lookup(&tlb, 0x1000, TEST_EPOCH), None);
        assert!(tlb_lookup(&tlb, 0x2000, TEST_EPOCH).is_some());
    }

    #[test]
    fn tlb_fifo_wraps() {
        let mut tlb: [TlbEntry; 32] = [TlbEntry::empty(); 32];
        let mut ins_idx: usize = 0;
        for i in 0..33 {
            tlb_insert(
                &mut tlb,
                &mut ins_idx,
                i as u64,
                i as u64 * 0x1000,
                0xF,
                0,
                0,
                TEST_EPOCH,
            );
        }
        assert_eq!(tlb_lookup(&tlb, 0, TEST_EPOCH), None, "entry 0 should be evicted");
        assert!(tlb_lookup(&tlb, 32, TEST_EPOCH).is_some(), "entry 32 should be present");
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
        };

        let result = translate_va(&mut state, &ctx, 0x8000_1000, false, false);
        assert!(result.is_ok(), "D-mode must use Bare translation (bypass Sv39)");
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
        };

        let pa = crate::hart_sched::translate_fetch_pc_concurrent(
            &mut state, &ctx, 0x8000_2000,
        );
        assert_eq!(pa, Some(0x8000_2000), "D-mode fetch must bypass Sv39 (VA==PA)");
    }
}
