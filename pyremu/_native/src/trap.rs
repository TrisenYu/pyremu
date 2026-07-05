//! Trap delivery with delegation support for the batch execution engine.
//!
//! Implements medeleg / mideleg-based delegation to S-mode, matching the
//! Python ``deliver_trap`` in ``trap_handler.py``.

use crate::state::{riscv_mode, BatchResult, HartState};

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
    if is_interrupt { code | (1u64 << 63) } else { code }
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
pub fn deliver_trap(
    state: &mut HartState,
    code: u64,
    tval: u64,
    result: &mut BatchResult,
) -> u64 {
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
        deliver_trap_smode(state, code, tval, result)
    } else {
        deliver_trap_mmode(state, code, tval, result)
    }
}

// ============================================================
//  S-mode trap delivery
// ============================================================

/// Deliver a trap to S-mode, saving PC / CSRs and redirecting to ``stvec``.
///
/// Called by ``deliver_trap`` when delegation conditions are met.
/// Matches the Python ``_trap_deliver_smode`` in ``trap_handler.py``.
pub fn deliver_trap_smode(
    state: &mut HartState,
    code: u64,
    tval: u64,
    _result: &mut BatchResult,
) -> u64 {
    // Clear LR/SC reservation on any trap
    state.reservation_valid = 0;
    state.reservation_addr = 0;

    // Wake from WFI
    state.waiting = 0;

    // Count consecutive traps
    state.consecutive_traps = state.consecutive_traps.saturating_add(1);

    // Save PC → sepc
    state.sepc = state.pc;

    // Save cause
    state.scause = code;

    // Save tval
    state.stval = tval;

    // mstatus: SPIE ← SIE, SIE ← 0, SPP ← current mode
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

    0 // advance = 0 — PC was redirected
}

// ============================================================
//  M-mode trap delivery (non-delegated fallback)
// ============================================================

/// Deliver a trap to M-mode, saving PC / CSRs and redirecting to ``mtvec``.
///
/// Called by ``deliver_trap`` when the trap is not delegated or the hart is
/// already in M-mode.  Matches the Python ``_trap_deliver_mmode``.
pub fn deliver_trap_mmode(
    state: &mut HartState,
    code: u64,
    tval: u64,
    _result: &mut BatchResult,
) -> u64 {
    // Clear LR/SC reservation on any trap
    state.reservation_valid = 0;
    state.reservation_addr = 0;

    // Wake from WFI
    state.waiting = 0;

    // Count consecutive traps
    state.consecutive_traps = state.consecutive_traps.saturating_add(1);

    // Save PC → mepc
    state.mepc = state.pc;

    // Save cause
    state.mcause = code;

    // Save tval
    state.mtval = tval;

    // mstatus: MPIE ← MIE, MIE ← 0, MPP ← current mode
    let mie_present = (state.mstatus & MSTATUS_MIE) != 0;
    let mpp_bits = mode_to_mpp(state.mode) << 11;
    if mie_present {
        state.mstatus |= MSTATUS_MPIE;
    } else {
        state.mstatus &= !MSTATUS_MPIE;
    }
    state.mstatus &= !MSTATUS_MIE;
    state.mstatus = (state.mstatus & !MSTATUS_MPP) | mpp_bits;

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
pub fn deliver_illegal_instruction(
    state: &mut HartState,
    tval: u64,
    result: &mut BatchResult,
) -> u64 {
    deliver_trap(state, mcause_val(exc_code::ILL_INSTR, false), tval, result)
}

// ============================================================
//  Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn default_state() -> HartState {
        let mut s: HartState = unsafe { std::mem::zeroed() };
        s.mode = riscv_mode::M;
        s.mtvec = 0x80000000;
        s.stvec = 0x80004000;
        s.pc = 0x1000;
        s
    }

    fn default_result() -> BatchResult {
        unsafe { std::mem::zeroed() }
    }

    // ---- delegation tests ----

    #[test]
    fn trap_delegates_to_smode_when_medeleg_set() {
        let mut s = default_state();
        s.mode = riscv_mode::S;
        s.medeleg = 1 << exc_code::ILL_INSTR; // delegate ill-instr
        let mut r = default_result();
        deliver_trap(&mut s, mcause_val(exc_code::ILL_INSTR, false), 0xDEAD, &mut r);
        assert_eq!(s.mode, riscv_mode::S, "should stay in S-mode when delegated");
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
        let mut r = default_result();
        deliver_trap(&mut s, mcause_val(exc_code::ILL_INSTR, false), 0xBEEF, &mut r);
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
        let mut r = default_result();
        deliver_trap(&mut s, mcause_val(exc_code::ILL_INSTR, false), 0xBEEF, &mut r);
        assert_eq!(s.mode, riscv_mode::M, "should switch to M-mode");
        assert_eq!(s.mepc, 0x1000);
    }

    #[test]
    fn interrupt_delegates_to_smode_via_mideleg() {
        let mut s = default_state();
        s.mode = riscv_mode::S;
        s.mideleg = 1 << 7; // delegate MTI (timer interrupt)
        let mut r = default_result();
        deliver_trap(&mut s, mcause_val(7, true), 0, &mut r);
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
        let mut r = default_result();
        deliver_trap(&mut s, mcause_val(exc_code::ILL_INSTR, false), 0, &mut r);
        assert_eq!(s.mstatus & MSTATUS_SIE, 0, "SIE should be cleared");
        assert_ne!(s.mstatus & MSTATUS_SPIE, 0, "SPIE should be set");
    }

    #[test]
    fn smode_trap_sets_spp() {
        let mut s = default_state();
        s.mode = riscv_mode::S;
        s.medeleg = 1 << exc_code::ILL_INSTR;
        let mut r = default_result();
        deliver_trap(&mut s, mcause_val(exc_code::ILL_INSTR, false), 0, &mut r);
        let spp = (s.mstatus & MSTATUS_SPP) >> 8;
        assert_eq!(spp, 1, "SPP should be 1 (was S-mode)");
    }

    #[test]
    fn smode_trap_from_umode_sets_spp_0() {
        let mut s = default_state();
        s.mode = riscv_mode::U;
        s.medeleg = 1 << exc_code::ILL_INSTR;
        let mut r = default_result();
        deliver_trap(&mut s, mcause_val(exc_code::ILL_INSTR, false), 0, &mut r);
        let spp = (s.mstatus & MSTATUS_SPP) >> 8;
        assert_eq!(spp, 0, "SPP should be 0 (was U-mode)");
    }

    // ---- existing M-mode delivery tests (updated for deliver_trap) ----

    #[test]
    fn deliver_ill_instr_saves_mepc() {
        let mut s = default_state();
        s.pc = 0xBEEF;
        let mut r = default_result();
        deliver_illegal_instruction(&mut s, 0xDEAD_BEEF, &mut r);
        assert_eq!(s.mepc, 0xBEEF);
        assert_eq!(s.mcause, mcause_val(exc_code::ILL_INSTR, false));
        assert_eq!(s.mtval, 0xDEAD_BEEF);
    }

    #[test]
    fn deliver_trap_clears_mie() {
        let mut s = default_state();
        s.mstatus = MSTATUS_MIE; // MIE=1
        let mut r = default_result();
        deliver_illegal_instruction(&mut s, 0, &mut r);
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
        let mut r = default_result();
        deliver_illegal_instruction(&mut s, 0, &mut r);
        assert_eq!(s.mstatus & MSTATUS_MPIE, 0, "MPIE should be 0 when MIE was 0");
    }

    #[test]
    fn deliver_trap_switches_to_mmode() {
        let mut s = default_state();
        s.mode = riscv_mode::S;
        let mut r = default_result();
        deliver_illegal_instruction(&mut s, 0, &mut r);
        assert_eq!(s.mode, riscv_mode::M);
    }

    #[test]
    fn deliver_trap_saves_mpp() {
        let mut s = default_state();
        s.mode = riscv_mode::S;
        let mut r = default_result();
        deliver_illegal_instruction(&mut s, 0, &mut r);
        let mpp = (s.mstatus & MSTATUS_MPP) >> 11;
        assert_eq!(mpp, mode_to_mpp(riscv_mode::S));
    }

    #[test]
    fn deliver_trap_direct_mtvec() {
        let mut s = default_state();
        s.mtvec = 0x80004000; // mode = 0 (direct)
        let mut r = default_result();
        deliver_illegal_instruction(&mut s, 0, &mut r);
        assert_eq!(s.pc, 0x80004000); // direct → base
    }

    #[test]
    fn deliver_trap_vectored_exception() {
        let mut s = default_state();
        s.mtvec = 0x80004000 | 1; // vectored mode
        let mut r = default_result();
        deliver_illegal_instruction(&mut s, 0, &mut r);
        // Exception → base, not base + 4*code
        assert_eq!(s.pc, 0x80004000);
    }

    #[test]
    fn deliver_trap_clears_reservation() {
        let mut s = default_state();
        s.reservation_valid = 1;
        s.reservation_addr = 0x8000_0000;
        let mut r = default_result();
        deliver_illegal_instruction(&mut s, 0, &mut r);
        assert_eq!(s.reservation_valid, 0);
    }

    #[test]
    fn consecutive_traps_increment() {
        let mut s = default_state();
        let mut r = default_result();
        assert_eq!(s.consecutive_traps, 0);
        deliver_illegal_instruction(&mut s, 0, &mut r);
        assert_eq!(s.consecutive_traps, 1);
        deliver_illegal_instruction(&mut s, 0, &mut r);
        assert_eq!(s.consecutive_traps, 2);
    }

    #[test]
    fn mcause_encoding() {
        assert_eq!(mcause_val(2, false), 2);
        assert_eq!(mcause_val(7, true), (1u64 << 63) | 7);
    }
}
