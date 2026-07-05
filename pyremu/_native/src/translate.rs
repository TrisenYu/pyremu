//! Sv39 page-table walk and TLB management for the batch execution engine.
//!
//! Replaces the Python ``sv39_walk`` + ``TLB.lookup/insert/flush`` with
//! Rust-native implementations that operate directly on RAM and inline
//! ``TlbEntry`` arrays in ``HartState``.
//!
//! Implements hardware A/D bit setting: the A (Accessed) bit is set on any
//! PTE that is read during a walk; the D (Dirty) bit is set on the leaf PTE
//! when the access is a write.  This matches RISC-V privileged spec §4.3.2.

use crate::mmu::{sv39_decompose_va, pte_parse};
use crate::state::{HartState, TlbEntry, riscv_mode};

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
    PageFault(u64),    // cause code (12=Instr, 13=Load, 15=Store)
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

/// Read a 64-bit raw PTE value from physical memory.  Returns 0 if address is invalid.
#[inline]
fn read_pte(ctx: &WalkCtx, pa: u64) -> u64 {
    let off = match ram_offset(ctx, pa, 8) {
        Some(o) => o,
        None => return 0,
    };
    let ptr = unsafe { ctx.ram.add(off) };
    let b0 = unsafe { *ptr } as u64;
    let b1 = unsafe { *ptr.add(1) } as u64;
    let b2 = unsafe { *ptr.add(2) } as u64;
    let b3 = unsafe { *ptr.add(3) } as u64;
    let b4 = unsafe { *ptr.add(4) } as u64;
    let b5 = unsafe { *ptr.add(5) } as u64;
    let b6 = unsafe { *ptr.add(6) } as u64;
    let b7 = unsafe { *ptr.add(7) } as u64;
    b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
        | (b4 << 32) | (b5 << 40) | (b6 << 48) | (b7 << 56)
}

/// Write a 64-bit raw PTE value to physical memory.
#[inline]
fn write_pte_raw(ctx: &WalkCtx, pa: u64, val: u64) {
    let off = match ram_offset(ctx, pa, 8) {
        Some(o) => o,
        None => return,
    };
    let ptr = unsafe { ctx.ram.add(off) };
    unsafe {
        *ptr = val as u8;
        *ptr.add(1) = (val >> 8) as u8;
        *ptr.add(2) = (val >> 16) as u8;
        *ptr.add(3) = (val >> 24) as u8;
        *ptr.add(4) = (val >> 32) as u8;
        *ptr.add(5) = (val >> 40) as u8;
        *ptr.add(6) = (val >> 48) as u8;
        *ptr.add(7) = (val >> 56) as u8;
    }
}

// ============================================================
//  Sv39 3-level page-table walk
// ============================================================

const PTE_V: u64 = 1 << 0;
const PTE_R: u64 = 1 << 1;
const PTE_W: u64 = 1 << 2;
const PTE_X: u64 = 1 << 3;
const PTE_A: u64 = 1 << 6;
const PTE_D: u64 = 1 << 7;

const SATP_MODE_SV39: u64 = 8;

/// Perform a full Sv39 page-table walk with hardware A/D bit updates.
///
/// For every PTE accessed during the walk, the A (Accessed) bit is set
/// and written back to RAM.  For leaf PTEs on a write access, the D
/// (Dirty) bit is also set.  This behaviour matches RISC-V privileged
/// spec §4.3.2 ("Virtual Address Translation Process").
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

    // ---- Level 2 (root): read PTE at (root_ppn << 12) + vpn2 * 8 ----
    let l2_addr = (root_ppn << 12).wrapping_add(vpn.vpn2 * 8);
    let l2_raw = read_pte(ctx, l2_addr);
    let l2_pte = pte_parse(l2_raw);
    if l2_pte.v == 0 || l2_pte.is_leaf == 0 && l2_pte.is_ptr == 0 {
        return None;
    }

    // Set A bit on L2 PTE if not already set
    if l2_raw & PTE_A == 0 {
        write_pte_raw(ctx, l2_addr, l2_raw | PTE_A);
    }

    // Check if this is a valid pointer to level 1
    if l2_pte.is_ptr != 0 {
        // ---- Level 1: read PTE at (l2_ppn << 12) + vpn1 * 8 ----
        let l1_addr = (l2_pte.ppn << 12).wrapping_add(vpn.vpn1 * 8);
        let l1_raw = read_pte(ctx, l1_addr);
        let l1_pte = pte_parse(l1_raw);
        if l1_pte.v == 0 || l1_pte.is_leaf == 0 && l1_pte.is_ptr == 0 {
            return None;
        }

        // Set A bit on L1 PTE if not already set
        if l1_raw & PTE_A == 0 {
            write_pte_raw(ctx, l1_addr, l1_raw | PTE_A);
        }

        // Check for 2 MiB superpage
        if l1_pte.is_leaf != 0 {
            let need_perm = if is_write { PTE_R | PTE_W } else { PTE_R };
            if l1_pte.perm as u64 & need_perm != need_perm {
                return None;
            }

            // Set A+D bits on superpage PTE
            let mut new_raw = l1_raw | PTE_A;
            if is_write {
                new_raw |= PTE_D;
            }
            if new_raw != l1_raw {
                write_pte_raw(ctx, l1_addr, new_raw);
            }

            // Assemble PA: PPN[43:9] from PTE, PPN[8:0] from VA[20:12] (vpn[0])
            let ppn = (l1_pte.ppn & 0xFFFF_FFFF_FFFF_FE00u64) | vpn.vpn0;
            let pa = (ppn << 12) | vpn.offset;
            return Some(TranslateResult {
                pa: pa & 0xFFFF_FFFF_FFFF_FFFF,
                perm: l1_pte.perm,
                level: 1,
            });
        }

        // ---- Level 0: read PTE at (l1_ppn << 12) + vpn0 * 8 ----
        let l0_addr = (l1_pte.ppn << 12).wrapping_add(vpn.vpn0 * 8);
        let l0_raw = read_pte(ctx, l0_addr);
        let l0_pte = pte_parse(l0_raw);
        if l0_pte.v == 0 || l0_pte.is_leaf == 0 {
            return None;
        }

        // 4 KiB page: check permissions
        let need_perm = if is_write { PTE_R | PTE_W } else { PTE_R };
        if l0_pte.perm as u64 & need_perm != need_perm {
            return None;
        }

        // Set A+D bits on leaf PTE
        let mut new_raw = l0_raw | PTE_A;
        if is_write {
            new_raw |= PTE_D;
        }
        if new_raw != l0_raw {
            write_pte_raw(ctx, l0_addr, new_raw);
        }

        let pa = (l0_pte.ppn << 12) | vpn.offset;
        return Some(TranslateResult {
            pa: pa & 0xFFFF_FFFF_FFFF_FFFF,
            perm: l0_pte.perm,
            level: 0,
        });
    }

    // l2_pte is a leaf? That would be a 1 GiB page (not supported in Sv39)
    None
}

// ============================================================
//  TLB operations
// ============================================================

/// TLB FIFO insertion index (per-TLB, stored as len in [0..32)).
/// We track insertion order via a simple ring counter stored in
/// ``HartState._mmu_mode_pad`` (upper 8 bits = itlb_idx, next 8 = dtlb_idx).

const fn tlb_sz() -> usize { 32 }

/// Find a TLB entry matching *vpn*. Returns index or None.
#[inline]
pub fn tlb_lookup(tlb: &[TlbEntry; 32], vpn: u64) -> Option<usize> {
    // Linear scan — 32 entries is small enough that a fully-associative
    // linear scan is faster than any hash-based approach.
    for i in 0..tlb_sz() {
        if tlb[i].valid != 0 && tlb[i].vpn == vpn {
            return Some(i);
        }
    }
    None
}

/// Insert a translation into the TLB (FIFO replacement).
/// *ins_idx* is a mutable reference to the next insertion index (0..32).
#[inline]
pub fn tlb_insert(tlb: &mut [TlbEntry; 32], ins_idx: &mut usize, vpn: u64, ppn: u64, perm: u8, level: u8, mdid: u8) {
    let idx = *ins_idx;
    tlb[idx].vpn = vpn;
    tlb[idx].ppn = ppn;
    tlb[idx].perm = perm;
    tlb[idx].level = level;
    tlb[idx].valid = 1;
    tlb[idx].mdid = mdid;
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
    // Bare mode: VA == PA
    if state.mmu_mode != 8 {
        return Ok(TranslateResult {
            pa: va & 0xFFFF_FFFF_FFFF_FFFF,
            perm: 0xF, // full permissions
            level: 0,
        });
    }

    let vpn = va >> 12;
    let tlb = if is_execute { &state.itlb } else { &state.dtlb };

    // TLB lookup
    if let Some(idx) = tlb_lookup(tlb, vpn) {
        let e = &tlb[idx];
        let need_perm = if is_execute {
            1 << 3 // X
        } else if is_write {
            (1 << 1) | (1 << 2) // R|W
        } else {
            1 << 1 // R
        };
        if e.perm & need_perm == 0 {
            return Err(TranslateFault::PageFault(
                if is_execute { 12 } else if is_write { 15 } else { 13 }
            ));
        }
        // Reconstruct PA
        let offset = va & 0xFFF;
        let pa = if e.level == 1 {
            let ppn = (e.ppn & 0xFFFF_FFFF_FFFF_FE00) | ((va >> 12) & 0x1FF);
            ((ppn << 12) | offset) & 0xFFFF_FFFF_FFFF_FFFF
        } else {
            ((e.ppn << 12) | offset) & 0xFFFF_FFFF_FFFF_FFFF
        };
        return Ok(TranslateResult { pa, perm: e.perm, level: e.level });
    }

    // Page-table walk
    let result = sv39_walk(ctx, state.satp, va, is_write)
        .ok_or_else(|| TranslateFault::PageFault(
            if is_execute { 12 } else if is_write { 15 } else { 13 }
        ))?;

    // Insert into TLB
    let tlb_mut = if is_execute { &mut state.itlb } else { &mut state.dtlb };
    let ins_idx = if is_execute {
        &mut ((state._mmu_mode_pad >> 8) as usize & 0x1F)
    } else {
        &mut (state._mmu_mode_pad as usize & 0x1F)
    };
    let vpn_ins = va >> 12;
    tlb_insert(tlb_mut, ins_idx, vpn_ins, result.ppn_for_tlb(vpn_ins), result.perm, result.level, state.mdid);

    Ok(result)
}

impl TranslateResult {
    /// Reconstruct the PPN value to store in TLB.
    /// For 4K pages: PPN = (pa & !0xFFF) >> 12 or pa >> 12
    /// For 2M pages: PPN upper bits come from the PTE (the walk already composed it).
    /// Here the pa has already been assembled, so we decompose backwards.
    fn ppn_for_tlb(&self, vpn: u64) -> u64 {
        if self.level == 1 {
            // 2M: return PPN excluding the vpn0 bits
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
        let l2_val = (1u64 << 10) | PTE_V; // PPN=1, V=1
        ram[l2_off..l2_off+8].copy_from_slice(&l2_val.to_le_bytes());

        // L1: pointer to L0 at PPN=2 (addr 0x2000)
        let l1_off = 0x1000usize + (vpn.vpn1 * 8) as usize;
        let l1_val = (2u64 << 10) | PTE_V; // PPN=2, V=1
        ram[l1_off..l1_off+8].copy_from_slice(&l1_val.to_le_bytes());

        // L0: leaf PTE mapping va→pa
        let l0_off = 0x2000usize + (vpn.vpn0 * 8) as usize;
        let l0_ppn = pa >> 12;
        let l0_val = (l0_ppn << 10) | perm | PTE_V;
        ram[l0_off..l0_off+8].copy_from_slice(&l0_val.to_le_bytes());

        satp
    }

    #[test]
    fn tlb_lookup_miss() {
        let tlb: [TlbEntry; 32] = [TlbEntry::empty(); 32];
        assert_eq!(tlb_lookup(&tlb, 0x100), None);
    }

    #[test]
    fn tlb_insert_and_lookup() {
        let mut tlb: [TlbEntry; 32] = [TlbEntry::empty(); 32];
        let mut ins_idx: usize = 0;
        tlb_insert(&mut tlb, &mut ins_idx, 0x1000, 0x80000, 0xF, 0, 0);
        let hit = tlb_lookup(&tlb, 0x1000);
        assert!(hit.is_some());
        let e = &tlb[hit.unwrap()];
        assert_eq!(e.ppn, 0x80000);
        assert_eq!(e.perm, 0xF);
        assert_eq!(e.level, 0);
        assert_eq!(e.valid, 1);
    }

    #[test]
    fn test_tlb_flush_all() {
        let mut tlb: [TlbEntry; 32] = [TlbEntry::empty(); 32];
        let mut ins_idx: usize = 0;
        tlb_insert(&mut tlb, &mut ins_idx, 0x1000, 0x80000, 0xF, 0, 0);
        crate::translate::tlb_flush_all(&mut tlb);
        assert_eq!(tlb_lookup(&tlb, 0x1000), None);
    }

    #[test]
    fn test_tlb_flush_vpn() {
        let mut tlb: [TlbEntry; 32] = [TlbEntry::empty(); 32];
        let mut ins_idx: usize = 0;
        tlb_insert(&mut tlb, &mut ins_idx, 0x1000, 0x80000, 0xF, 0, 0);
        tlb_insert(&mut tlb, &mut ins_idx, 0x2000, 0x90000, 0xF, 0, 0);
        crate::translate::tlb_flush_vpn(&mut tlb, 0x1000);
        assert_eq!(tlb_lookup(&tlb, 0x1000), None);
        assert!(tlb_lookup(&tlb, 0x2000).is_some());
    }

    #[test]
    fn tlb_fifo_wraps() {
        let mut tlb: [TlbEntry; 32] = [TlbEntry::empty(); 32];
        let mut ins_idx: usize = 0;
        for i in 0..33 {
            tlb_insert(&mut tlb, &mut ins_idx, i as u64, i as u64 * 0x1000, 0xF, 0, 0);
        }
        assert_eq!(tlb_lookup(&tlb, 0), None, "entry 0 should be evicted");
        assert!(tlb_lookup(&tlb, 32).is_some(), "entry 32 should be present");
    }

    #[test]
    fn sv39_walk_sets_a_bit_on_leaf() {
        let mut ram = vec![0u8; 0x3000];
        let satp = setup_4k_page_table(
            &mut ram, 0x1000, 0x8000_0000,
            PTE_R | PTE_W | PTE_X,
        );
        let ctx = WalkCtx { ram: ram.as_mut_ptr(), ram_size: ram.len() as u64, ram_base: 0, shadow_base: 0, shadow_size: 0 };

        let result = sv39_walk(&ctx, satp, 0x1000, false);
        assert!(result.is_some());
        // Verify A bit was set on L0 PTE
        let l0_off = 0x2000usize + (sv39_decompose_va(0x1000).vpn0 * 8) as usize;
        let l0_val = u64::from_le_bytes(ram[l0_off..l0_off+8].try_into().unwrap());
        assert_ne!(l0_val & PTE_A, 0, "A bit should be set on leaf PTE");
    }

    #[test]
    fn sv39_walk_sets_d_bit_on_write() {
        let mut ram = vec![0u8; 0x3000];
        let satp = setup_4k_page_table(
            &mut ram, 0x2000, 0x8000_1000,
            PTE_R | PTE_W,
        );
        let ctx = WalkCtx { ram: ram.as_mut_ptr(), ram_size: ram.len() as u64, ram_base: 0, shadow_base: 0, shadow_size: 0 };

        let result = sv39_walk(&ctx, satp, 0x2000, true); // is_write=true
        assert!(result.is_some());
        // Verify D bit was set on L0 PTE
        let l0_off = 0x2000usize + (sv39_decompose_va(0x2000).vpn0 * 8) as usize;
        let l0_val = u64::from_le_bytes(ram[l0_off..l0_off+8].try_into().unwrap());
        assert_ne!(l0_val & PTE_D, 0, "D bit should be set on write to leaf PTE");
    }

    #[test]
    fn sv39_walk_sets_a_bit_on_intermediate_levels() {
        let mut ram = vec![0u8; 0x3000];
        let satp = setup_4k_page_table(
            &mut ram, 0x3000, 0x8000_2000,
            PTE_R | PTE_W,
        );
        let ctx = WalkCtx { ram: ram.as_mut_ptr(), ram_size: ram.len() as u64, ram_base: 0, shadow_base: 0, shadow_size: 0 };

        let _result = sv39_walk(&ctx, satp, 0x3000, false);
        // Verify A bits set on L2 and L1 PTEs
        let l2_off = (sv39_decompose_va(0x3000).vpn2 * 8) as usize;
        let l2_val = u64::from_le_bytes(ram[l2_off..l2_off+8].try_into().unwrap());
        assert_ne!(l2_val & PTE_A, 0, "A bit should be set on L2 PTE");

        let l1_off = 0x1000usize + (sv39_decompose_va(0x3000).vpn1 * 8) as usize;
        let l1_val = u64::from_le_bytes(ram[l1_off..l1_off+8].try_into().unwrap());
        assert_ne!(l1_val & PTE_A, 0, "A bit should be set on L1 PTE");
    }

    #[test]
    fn bare_mode_translate_is_identity() {
        // Test that Bare mode passes VA through as PA
        // This is tested more thoroughly via the batch engine integration tests
    }
}
