//! CSR read/write for the batch execution engine (Phase C).
//!
//! Handles inline read/write for CSRs whose values are stored directly in
//! ``HartState``.  Unknown or write-sensitive CSRs trigger an exit to Python
//! so the full ``registers.py`` machinery can process them.

use crate::state::{HartState, BatchResult, exit_reason, riscv_mode};
use crate::trap::{deliver_illegal_instruction, exc_code, mcause_val};

// ============================================================
//  CSR address constants
// ============================================================

// Machine-level
pub const MSTATUS: u16    = 0x300;
pub const MISA: u16       = 0x301;
pub const MEDELEG: u16    = 0x302;
pub const MIDELEG: u16    = 0x303;
pub const MIE: u16        = 0x304;
pub const MTVEC: u16      = 0x305;
pub const MCOUNTEREN: u16 = 0x306;
pub const MSCRATCH: u16   = 0x340;
pub const MEPC: u16       = 0x341;
pub const MCAUSE: u16     = 0x342;
pub const MTVAL: u16      = 0x343;
pub const MIP: u16        = 0x344;

// Machine info
pub const MVENDORID: u16  = 0xF11;
pub const MARCHID: u16    = 0xF12;
pub const MIMPID: u16     = 0xF13;
pub const MHARTID: u16    = 0xF14;

// Supervisor-level
pub const SSTATUS: u16    = 0x100;
pub const STVEC: u16      = 0x105;
pub const SCOUNTEREN: u16 = 0x106;
pub const SSCRATCH: u16   = 0x140;
pub const SEPC: u16       = 0x141;
pub const SCAUSE: u16     = 0x142;
pub const STVAL: u16      = 0x143;
pub const SATP: u16       = 0x180;

// ============================================================
//  mstatus / sstatus field masks
// ============================================================

const MSTATUS_SIE: u64  = 1 << 1;
const MSTATUS_MIE: u64  = 1 << 3;
const MSTATUS_SPIE: u64 = 1 << 5;
const MSTATUS_MPIE: u64 = 1 << 7;
const MSTATUS_SPP: u64  = 1 << 8;
const MSTATUS_MPP: u64  = 0b11 << 11;
const MSTATUS_FS: u64   = 0b11 << 13;
const MSTATUS_XS: u64   = 0b11 << 15;
const MSTATUS_UXL: u64  = 0b11 << 32;
const MSTATUS_SXL: u64  = 0b11 << 34;
const MSTATUS_SD: u64   = 1 << 63;

const SSTATUS_READ_MASK: u64 =
    MSTATUS_SIE | MSTATUS_SPIE | MSTATUS_SPP
    | MSTATUS_FS | MSTATUS_XS | MSTATUS_UXL | MSTATUS_SD;
const SSTATUS_WRITE_MASK: u64 = MSTATUS_SIE | MSTATUS_SPIE | MSTATUS_SPP;

// ============================================================
//  Public API
// ============================================================

/// Result: 0 = success (handler returns advance 4), 1 = IllInstr trap,
/// 2 = exit to Python (unknown CSR or write requiring Python side-effects).
pub const CSR_OK: u8 = 0;
pub const CSR_ILL: u8 = 1;
pub const CSR_EXIT: u8 = 2;

/// Read a CSR value. Returns ``(value, status)``.
/// status == CSR_OK: value is valid, caller writes to rd.
/// status == CSR_ILL: deliver IllInstr.
/// status == CSR_EXIT: exit to Python.
pub fn csr_read(
    state: &HartState,
    addr: u16,
    hart_id: u8,
    mtime: u64,
) -> (u64, u8) {
    // Check privilege
    let priv_req = (addr >> 8) as u8 & 0x3; // bits [9:8]
    if !csr_priv_ok(state.mode, priv_req) {
        return (0, CSR_ILL);
    }

    // Counter enable checks for S-mode reading M-mode counters
    if priv_req == 0 && state.mode == riscv_mode::S {
        match addr {
            0xC00 | 0xC80 => {
                if state.mcounteren & 1 == 0 { return (0, CSR_ILL); }
            }
            0xC01 | 0xC81 => {
                if state.mcounteren & 2 == 0 { return (0, CSR_ILL); }
            }
            0xC02 | 0xC82 => {
                if state.mcounteren & 4 == 0 { return (0, CSR_ILL); }
            }
            _ => {}
        }
    }
    if priv_req == 1 && state.mode == riscv_mode::U {
        match addr {
            0xC00 | 0xC80 => {
                if state.scounteren & 1 == 0 { return (0, CSR_ILL); }
            }
            0xC01 | 0xC81 => {
                if state.scounteren & 2 == 0 { return (0, CSR_ILL); }
            }
            0xC02 | 0xC82 => {
                if state.scounteren & 4 == 0 { return (0, CSR_ILL); }
            }
            _ => {}
        }
    }

    match addr {
        // Machine-level CSRs in HartState
        MSTATUS => (state.mstatus, CSR_OK),
        MEDELEG => (state.medeleg, CSR_OK),
        MIDELEG => (state.mideleg, CSR_OK),
        MIE     => (state.mie, CSR_OK),
        MTVEC   => (state.mtvec, CSR_OK),
        MSCRATCH => (state.mscratch, CSR_OK),
        MEPC    => (state.mepc, CSR_OK),
        MCAUSE  => (state.mcause, CSR_OK),
        MTVAL   => (state.mtval, CSR_OK),
        MIP     => (state.mip, CSR_OK),
        MCOUNTEREN => (state.mcounteren, CSR_OK),

        // Supervisor-level CSRs in HartState
        SSTATUS => (state.mstatus & SSTATUS_READ_MASK, CSR_OK),
        STVEC   => (state.stvec, CSR_OK),
        SSCRATCH => (state.sscratch, CSR_OK),
        SEPC    => (state.sepc, CSR_OK),
        SCAUSE  => (state.scause, CSR_OK),
        STVAL   => (state.stval, CSR_OK),
        SATP    => (state.satp, CSR_OK),
        SCOUNTEREN => (state.scounteren, CSR_OK),

        // Machine info registers
        MVENDORID => (0, CSR_OK),
        MARCHID   => (0, CSR_OK),
        MIMPID    => (0, CSR_OK),
        MHARTID   => (state.mhartid as u64, CSR_OK),

        // MISA — report RV64IMAC
        MISA => (0x800000000014112Du64, CSR_OK),

        // time (from CLINT mtime)
        0xC01 => (mtime, CSR_OK),
        // timeh
        0xC81 => (mtime >> 32, CSR_OK),

        // cycle/instret — we don't track these exactly in batch mode,
        // but reading them should not trap. Return 0.
        0xC00 | 0xC02 | 0xC80 | 0xC82 => (0, CSR_OK),

        // Everything else: exit to Python
        _ => (0, CSR_EXIT),
    }
}

/// Write a CSR value. Returns ``(status)``.
/// status == CSR_OK: write succeeded.
/// status == CSR_ILL: deliver IllInstr.
/// status == CSR_EXIT: exit to Python.
pub fn csr_write(
    state: &mut HartState,
    addr: u16,
    val: u64,
) -> u8 {
    let priv_req = (addr >> 8) as u8 & 0x3;
    if !csr_priv_ok(state.mode, priv_req) {
        return CSR_ILL;
    }

    match addr {
        MSTATUS => {
            // SD is read-only
            let sd = state.mstatus & MSTATUS_SD;
            let wr = val & !MSTATUS_SD;
            state.mstatus = wr | sd;
            CSR_OK
        }
        MEDELEG => { state.medeleg = val; CSR_OK }
        MIDELEG => { state.mideleg = val; CSR_OK }
        MIE     => { state.mie = val; CSR_OK }
        MTVEC   => { state.mtvec = val; CSR_OK }
        MSCRATCH => { state.mscratch = val; CSR_OK }
        MEPC    => { state.mepc = val; CSR_OK }
        MCAUSE  => { state.mcause = val; CSR_OK }
        MTVAL   => { state.mtval = val; CSR_OK }
        MIP     => { state.mip = val; CSR_OK }
        MCOUNTEREN => { state.mcounteren = val; CSR_OK }

        SSTATUS => {
            let mask = SSTATUS_WRITE_MASK;
            state.mstatus = (state.mstatus & !mask) | (val & mask);
            CSR_OK
        }
        STVEC   => { state.stvec = val; CSR_OK }
        SSCRATCH => { state.sscratch = val; CSR_OK }
        SEPC    => { state.sepc = val; CSR_OK }
        SCAUSE  => { state.scause = val; CSR_OK }
        STVAL   => { state.stval = val; CSR_OK }
        SCOUNTEREN => { state.scounteren = val; CSR_OK }
        SATP    => {
            state.satp = val;
            // Side-effect: update mmu_mode
            let new_mode = (val >> 60) as u8;
            state.mmu_mode = new_mode;
            // Flush TLBs on satp write (SFENCE.VMA semantic)
            tlb_flush_inline(state);
            CSR_OK
        }

        // Everything else (including PMP CSRs, stimecmp, etc.) → exit to Python
        _ => CSR_EXIT,
    }
}

// ============================================================
//  Helpers
// ============================================================

/// Flush both TLBs without importing translate module.
#[inline]
fn tlb_flush_inline(state: &mut HartState) {
    for e in state.itlb.iter_mut() { e.valid = 0; }
    for e in state.dtlb.iter_mut() { e.valid = 0; }
}

/// Check if *mode* is privileged enough to access a CSR at *priv_req*.
#[inline]
fn csr_priv_ok(mode: u8, priv_req: u8) -> bool {
    match priv_req {
        0 => true,                     // user-readable
        1 => mode >= riscv_mode::S,    // supervisor
        2 => mode >= riscv_mode::H,    // hypervisor (unused)
        3 => mode >= riscv_mode::M,    // machine
        _ => false,
    }
}

/// Handle a CSR instruction (CSRRW/CSRRS/CSRRC/CSRRWI/CSRRSI/CSRRCI).
///
/// Returns PC advance (4 on success, 0 on trap).
/// Sets result.exit_reason to SYS_EXIT if the CSR needs Python handling.
pub fn handle_csr(
    state: &mut HartState,
    rd: u8,
    rs1: u8,
    csr_addr: u16,
    funct3: u8,
    instr: u32,
    result: &mut BatchResult,
    hart_id: u8,
    mtime: u64,
) -> u64 {
    // Read old CSR value
    let (old_val, status) = csr_read(state, csr_addr, hart_id, mtime);
    if status == CSR_ILL {
        deliver_illegal_instruction(state, instr as u64, result);
        return 0;
    }
    if status == CSR_EXIT {
        result.exit_reason = exit_reason::SYS_EXIT;
        result.exit_instr = instr;
        return EXIT_SENTINEL;
    }

    // Compute new value based on funct3
    let rs1_val = if funct3 >= 4 {
        // CSRRWI / CSRRSI / CSRRCI: use zero-extended rs1 (5-bit unsigned)
        rs1 as u64
    } else {
        read_gpr(state, rs1)
    };

    let new_val: u64 = match funct3 & 0x3 {
        1 => rs1_val,                           // CSRRW / CSRRWI
        2 => old_val | rs1_val,                 // CSRRS / CSRRSI
        3 => old_val & !rs1_val,                // CSRRC / CSRRCI
        _ => {
            deliver_illegal_instruction(state, instr as u64, result);
            return 0;
        }
    };

    // Write back to CSR
    if rd != 0 {
        let wstatus = csr_write(state, csr_addr, new_val);
        if wstatus == CSR_ILL {
            deliver_illegal_instruction(state, instr as u64, result);
            return 0;
        }
        if wstatus == CSR_EXIT {
            result.exit_reason = exit_reason::SYS_EXIT;
            result.exit_instr = instr;
            return EXIT_SENTINEL;
        }
    }

    // Write old value to rd
    write_gpr(state, rd, old_val);
    4
}

// ============================================================
//  GPR read/write (local copies for csr.rs independence)
// ============================================================

#[inline]
fn read_gpr(state: &HartState, rs: u8) -> u64 {
    if rs == 0 { 0 } else { state.gprs[rs as usize] }
}

#[inline]
fn write_gpr(state: &mut HartState, rd: u8, val: u64) {
    if rd != 0 {
        state.gprs[rd as usize] = val;
    }
}

/// Sentinel value: return this from a handler to signal exit-to-Python.
pub const EXIT_SENTINEL: u64 = u64::MAX;

// ============================================================
//  Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn test_state() -> HartState {
        let mut s: HartState = unsafe { std::mem::zeroed() };
        s.mode = riscv_mode::M;
        s
    }

    #[test]
    fn read_mstatus() {
        let mut s = test_state();
        s.mstatus = 0x1880; // MPP=S, MPIE=1
        let (v, st) = csr_read(&s, MSTATUS, 0, 0);
        assert_eq!(st, CSR_OK);
        assert_eq!(v, 0x1880);
    }

    #[test]
    fn read_sstatus_view() {
        let mut s = test_state();
        s.mstatus = MSTATUS_SIE | MSTATUS_SPIE | MSTATUS_SPP;
        let (v, st) = csr_read(&s, SSTATUS, 0, 0);
        assert_eq!(st, CSR_OK);
        assert_eq!(v & SSTATUS_READ_MASK, MSTATUS_SIE | MSTATUS_SPIE | MSTATUS_SPP);
    }

    #[test]
    fn write_satp_flushes_tlb() {
        let mut s = test_state();
        s.mmu_mode = 8;
        // Pre-fill TLB
        s.itlb[0].valid = 1;
        s.dtlb[0].valid = 1;
        // Write satp
        let st = csr_write(&mut s, SATP, 0x8000000000000000u64); // Sv39, PPN=0
        assert_eq!(st, CSR_OK);
        assert_eq!(s.mmu_mode, 8);
        // TLBs should be flushed
        assert_eq!(s.itlb[0].valid, 0);
        assert_eq!(s.dtlb[0].valid, 0);
    }

    #[test]
    fn write_sstatus_updates_mstatus_fields() {
        let mut s = test_state();
        // Write SIE bit through sstatus
        let st = csr_write(&mut s, SSTATUS, MSTATUS_SIE);
        assert_eq!(st, CSR_OK);
        assert_eq!(s.mstatus & MSTATUS_SIE, MSTATUS_SIE);
    }

    #[test]
    fn write_unknown_csr_triggers_exit() {
        let mut s = test_state();
        let st = csr_write(&mut s, 0xB00, 0); // mcycle
        assert_eq!(st, CSR_EXIT);
    }

    #[test]
    fn read_time() {
        let s = test_state();
        let (v, st) = csr_read(&s, 0xC01, 0, 123456789);
        assert_eq!(st, CSR_OK);
        assert_eq!(v, 123456789);
    }

    #[test]
    fn read_timeh() {
        let s = test_state();
        let (v, st) = csr_read(&s, 0xC81, 0, 0x123456789AB);
        assert_eq!(st, CSR_OK);
        // 0x123456789AB >> 32 → upper bits = 0x123 = 291
        assert_eq!(v, 0x123);
    }

    #[test]
    fn csr_priv_check() {
        assert!(csr_priv_ok(riscv_mode::M, 3));
        assert!(csr_priv_ok(riscv_mode::M, 1));
        assert!(!csr_priv_ok(riscv_mode::S, 3));
        assert!(csr_priv_ok(riscv_mode::S, 1));
        assert!(csr_priv_ok(riscv_mode::U, 0));
        assert!(!csr_priv_ok(riscv_mode::U, 1));
    }
}
