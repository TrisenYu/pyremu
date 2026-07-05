//! `#[repr(C)]` state structures that cross the Python FFI boundary.

use core::fmt;

// ============================================================
//  TLB entry
// ============================================================

/// A single TLB entry — 24 bytes, 8-byte aligned.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct TlbEntry {
    pub vpn: u64,
    pub ppn: u64,
    pub perm: u8,
    pub level: u8,
    pub valid: u8,
    pub mdid: u8,
    pub _pad: u32,
}

impl TlbEntry {
    pub const fn empty() -> Self {
        TlbEntry { vpn: 0, ppn: 0, perm: 0, level: 0, valid: 0, mdid: 0, _pad: 0 }
    }
}

impl fmt::Debug for TlbEntry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TlbEntry(vpn={:#x} ppn={:#x} perm={} level={} v={})",
            self.vpn, self.ppn, self.perm, self.level, self.valid)
    }
}

// ============================================================
//  Per-hart state
// ============================================================

#[repr(C)]
pub struct HartState {
    // ---- GPRs ----
    pub gprs: [u64; 32],

    // ---- Key CSRs (u64) ----
    pub mstatus: u64,
    pub mtvec: u64,
    pub stvec: u64,
    pub mepc: u64,
    pub sepc: u64,
    pub mcause: u64,
    pub scause: u64,
    pub mtval: u64,
    pub stval: u64,
    pub satp: u64,
    pub mie: u64,
    pub mip: u64,
    pub medeleg: u64,
    pub mideleg: u64,

    // ---- PC ----
    pub pc: u64,

    // ---- Reservation (LR/SC) ----
    pub reservation_addr: u64,

    // ---- single-byte fields ----
    pub reservation_valid: u8,
    pub mode: u8,
    pub mmu_mode: u8,
    pub waiting: u8,
    pub halted: u8,
    pub consecutive_traps: u8,
    pub mdid: u8,
    pub pmpsplit: u8,
    pub _pad: [u8; 6],

    // ---- Phase B: TLB entries ----
    pub itlb: [TlbEntry; 32],
    pub dtlb: [TlbEntry; 32],

    // ---- Phase C: Additional CSRs ----
    pub mscratch: u64,
    pub sscratch: u64,
    pub mhartid: u64,
    pub mcounteren: u64,
    pub scounteren: u64,

    // ---- Phase E: interrupt cache ----
    pub _mmu_mode_pad: u64,
}

// ============================================================
//  Batch result
// ============================================================

#[repr(C)]
pub struct BatchResult {
    pub total_instrs: u64,
    pub exit_reason: u8,
    pub exit_hart_id: u8,
    pub exit_pc: u64,
    pub exit_instr: u32,
    pub trap_cause: u32,
    pub trap_tval: u64,
    pub trap_is_interrupt: u8,
    pub trap_delegated: u8,
    pub _pad: [u8; 6],
}

// ============================================================
//  Constants
// ============================================================

#[allow(dead_code)]
pub mod exit_reason {
    pub const NORMAL: u8 = 0;
    pub const TRAP: u8 = 1;
    pub const MMIO: u8 = 2;
    pub const SYS_EXIT: u8 = 3;
    pub const EBREAK: u8 = 4;
    pub const WFI_WAIT: u8 = 5;
    pub const ERROR: u8 = 6;
}

#[allow(dead_code)]
pub mod riscv_mode {
    pub const U: u8 = 0;
    pub const S: u8 = 1;
    pub const H: u8 = 2;
    pub const M: u8 = 3;
    pub const D: u8 = 8;
}

// ============================================================
//  Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, size_of};

    #[test]
    fn tlb_entry_size() {
        assert_eq!(size_of::<TlbEntry>(), 24);
        assert_eq!(align_of::<TlbEntry>(), 8);
    }

    #[test]
    fn hart_state_alignment() {
        assert_eq!(align_of::<HartState>(), 8);
    }

    #[test]
    fn hart_state_size_reasonable() {
        let sz = size_of::<HartState>();
        assert!(sz < 4096, "HartState size {} should be < 4096", sz);
    }

    #[test]
    fn batch_result_alignment() {
        assert_eq!(align_of::<BatchResult>(), 8);
    }

    #[test]
    fn hart_state_defaults() {
        let hs = HartState {
            gprs: [0u64; 32],
            mstatus: 0, mtvec: 0, stvec: 0,
            mepc: 0, sepc: 0, mcause: 0, scause: 0,
            mtval: 0, stval: 0, satp: 0,
            mie: 0, mip: 0, medeleg: 0, mideleg: 0,
            pc: 0,
            reservation_addr: 0, reservation_valid: 0,
            mode: riscv_mode::M, mmu_mode: 0,
            waiting: 0, halted: 0, consecutive_traps: 0,
            mdid: 0, pmpsplit: 0,
            _pad: [0; 6],
            itlb: [TlbEntry::empty(); 32],
            dtlb: [TlbEntry::empty(); 32],
            mscratch: 0, sscratch: 0,
            mhartid: 0, mcounteren: 0, scounteren: 0,
            _mmu_mode_pad: 0,
        };
        assert_eq!(hs.gprs[0], 0);
        assert_eq!(hs.mode, riscv_mode::M);
    }
}
