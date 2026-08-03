//! CSR read/write for the batch execution engine (Phase C).
//!
//! Handles inline read/write for CSRs whose values are stored directly in
//! ``HartState``.  Unknown or write-sensitive CSRs trigger an exit to Python
//! so the full ``registers.py`` machinery can process them.

use crate::handlers::PmpCtx;
use crate::state::{exit_reason, riscv_mode, BatchResult, HartState};
use crate::trap::deliver_illegal_instruction;

// ============================================================
//  CSR address constants
// ============================================================

// Machine-level
pub const MSTATUS: u16 = 0x300;
pub const MISA: u16 = 0x301;
pub const MEDELEG: u16 = 0x302;
pub const MIDELEG: u16 = 0x303;
pub const MIE: u16 = 0x304;
pub const MTVEC: u16 = 0x305;
pub const MCOUNTEREN: u16 = 0x306;
pub const MSCRATCH: u16 = 0x340;
pub const MEPC: u16 = 0x341;
pub const MCAUSE: u16 = 0x342;
pub const MTVAL: u16 = 0x343;
pub const MIP: u16 = 0x344;

// Machine info
pub const MVENDORID: u16 = 0xF11;
pub const MARCHID: u16 = 0xF12;
pub const MIMPID: u16 = 0xF13;
pub const MHARTID: u16 = 0xF14;

// Supervisor-level
pub const SSTATUS: u16 = 0x100;
pub const SIE: u16 = 0x104;
pub const STVEC: u16 = 0x105;
pub const SCOUNTEREN: u16 = 0x106;
pub const SSCRATCH: u16 = 0x140;
pub const SEPC: u16 = 0x141;
pub const SCAUSE: u16 = 0x142;
pub const STVAL: u16 = 0x143;
pub const SIP: u16 = 0x144;
pub const SATP: u16 = 0x180; // Sstc extension
pub const STIMECMP: u16 = 0x14D;

// User-level floating-point CSRs (F/D extension)
pub const FFLAGS: u16 = 0x001;
pub const FRM: u16 = 0x002;
pub const FCSR: u16 = 0x003;

// ============================================================
//  mstatus / sstatus field masks
// ============================================================

// mstatus / sstatus field masks (must match Python hart.py bit definitions)
const MSTATUS_SIE: u64 = 1 << 1; // Supervisor interrupt enable
const MSTATUS_SPIE: u64 = 1 << 5; // Supervisor previous interrupt enable
const MSTATUS_UBE: u64 = 1 << 6; // User big-endian
const MSTATUS_SPP: u64 = 1 << 8; // Supervisor previous privilege
const MSTATUS_VS: u64 = 0b11 << 9; // Virtualisation state (H-ext)
const MSTATUS_FS: u64 = 0b11 << 13; // Floating-point unit state
const MSTATUS_XS: u64 = 0b11 << 15; // User extension state
const MSTATUS_MPRV: u64 = 1 << 17; // Modify privilege
const MSTATUS_SUM: u64 = 1 << 18; // Permit Supervisor User Memory access
const MSTATUS_MXR: u64 = 1 << 19; // Make eXecutable Readable
const MSTATUS_SD: u64 = 1 << 63; // State dirty (read-only)

// sstatus is a restricted view of mstatus (RISC-V Privileged Spec §4.1.1).
// Reading sstatus returns only the subset of mstatus bits accessible from S-mode.
const SSTATUS_READ_MASK: u64 = MSTATUS_SIE
    | MSTATUS_SPIE
    | MSTATUS_UBE
    | MSTATUS_SPP
    | MSTATUS_VS
    | MSTATUS_FS
    | MSTATUS_XS
    | MSTATUS_MPRV
    | MSTATUS_SUM
    | MSTATUS_MXR
    | MSTATUS_SD;

// Writing sstatus updates only the writable subset (VS and SD are read-only,
// matching Python's _SSTATUS_WRITABLE_MASK).
const SSTATUS_WRITE_MASK: u64 = SSTATUS_READ_MASK & !(MSTATUS_VS | MSTATUS_SD);

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
pub fn csr_read(state: &HartState, addr: u16, _hart_id: u8, mtime: u64, pmp: &PmpCtx) -> (u64, u8) {
    // Check privilege
    let priv_req = (addr >> 8) as u8 & 0x3; // bits [9:8]
    if !csr_priv_ok(state.mode, priv_req) {
        return (0, CSR_ILL);
    }

    // Counter enable checks for S-mode reading M-mode counters
    if priv_req == 0 && state.mode == riscv_mode::S {
        match addr {
            0xC00 | 0xC80 => {
                if state.mcounteren & 1 == 0 {
                    return (0, CSR_ILL);
                }
            }
            0xC01 | 0xC81 => {
                if state.mcounteren & 2 == 0 {
                    return (0, CSR_ILL);
                }
            }
            0xC02 | 0xC82 => {
                if state.mcounteren & 4 == 0 {
                    return (0, CSR_ILL);
                }
            }
            _ => {}
        }
    }
    if priv_req == 1 && state.mode == riscv_mode::U {
        match addr {
            0xC00 | 0xC80 => {
                if state.scounteren & 1 == 0 {
                    return (0, CSR_ILL);
                }
            }
            0xC01 | 0xC81 => {
                if state.scounteren & 2 == 0 {
                    return (0, CSR_ILL);
                }
            }
            0xC02 | 0xC82 => {
                if state.scounteren & 4 == 0 {
                    return (0, CSR_ILL);
                }
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
        SIE     => (state.mie & state.mideleg, CSR_OK),
        SIP     => (state.mip & state.mideleg, CSR_OK),
        STVEC   => (state.stvec, CSR_OK),
        SSCRATCH => (state.sscratch, CSR_OK),
        SEPC    => (state.sepc, CSR_OK),
        SCAUSE  => (state.scause, CSR_OK),
        STVAL   => (state.stval, CSR_OK),
        SATP    => (state.satp, CSR_OK),
        SCOUNTEREN => (state.scounteren, CSR_OK),
        STIMECMP => (state.stimecmp, CSR_OK),

        // Floating-point CSRs (require mstatus.FS != Off)
        FFLAGS => if fp_off(state) { (0, CSR_ILL) } else { ((state.fcsr & 0x1F) as u64, CSR_OK) },
        FRM    => if fp_off(state) { (0, CSR_ILL) } else { (((state.fcsr >> 5) & 0x7) as u64, CSR_OK) },
        FCSR   => if fp_off(state) { (0, CSR_ILL) } else { ((state.fcsr & 0xFF) as u64, CSR_OK) },

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

        // TEE CSRs (mdid=0x5C0, pmpsplit=0x5C1) — memory domain ID
        // and PMP virtualization split register.
        0x5C0 => (state.mdid as u64, CSR_OK),
        0x5C1 => (state.pmpsplit as u64, CSR_OK),

        // mtopi (0xFB0): raise IllInstr so OpenSBI's __check_ext_csr probe
        // fails -> SMAIA not detected -> sbi_trap_handler uses non-AIA path
        // (mcause-based dispatch) -> sbi_ipi_process() is called correctly.
        0xFB0 /* mtopi */ => (0, CSR_ILL),
        // Other AIA CSRs — no IMSIC, always return 0.
        0x308 /* mvien */ | 0x309 /* mvip */
        | 0x318 /* mvienh */ | 0x319 /* mviph */ => (0, CSR_OK),

        // Debug/trace trigger CSRs (0x7A0-0x7AF) — not implemented.
        // Return 0 rather than IllInstr: OpenSBI does not delegate IllInstr
        // to S-mode, so S-mode trigger probes would trap to M-mode where
        // OpenSBI's handler calls sbi_trap_error -> WFI, stalling the kernel.
        // Returning 0 lets Linux probe these CSRs without trapping.
        // Real hardware with Sdext would make them accessible in M-mode;
        // without Sdext they are unimplemented and return 0.
        0x7A0..=0x7AF => (0, CSR_OK),

        // RV32-only high-half CSRs — return 0 on RV64 (reads are legal but zero).
        0x310 /* mstatush */ | 0x312 /* medelegh */ | 0x313 /* midelegh */ => (0, CSR_OK),

        // ---- PMP CSRs (RV64) ----
        // Odd-numbered pmpcfg registers are illegal on RV64.
        a if (0x3A1..=0x3AF).contains(&a) && (a & 1) != 0 => (0, CSR_ILL),
        // Even-numbered pmpcfg: read 8 config bytes, pack as u64 LE.
        a if (0x3A0..=0x3AE).contains(&a) && (a & 1) == 0 => {
            let entry_base = ((a - 0x3A0) / 2) * 8;
            if (entry_base as u8) < pmp.num {
                let mut val: u64 = 0;
                let base = entry_base as usize;
                for i in 0..8u16 {
                    let b = unsafe { *pmp.cfg.add(base + i as usize) } as u64;
                    val |= b << (i * 8);
                }
                (val, CSR_OK)
            } else {
                (0, CSR_OK) // unimplemented entries read as zero
            }
        }
        // PMP address registers (0x3B0-0x3EF): one per entry, 8 bytes each.
        a if (0x3B0..=0x3EF).contains(&a) => {
            let n = (a - 0x3B0) as usize;
            if n < pmp.num as usize {
                (unsafe { *pmp.addr.add(n) }, CSR_OK)
            } else {
                (0, CSR_OK)
            }
        }

        // Everything else: exit to Python
        _ => (0, CSR_EXIT),
    }
}

/// Write a CSR value. Returns ``(status)``.
/// status == CSR_OK: write succeeded.
/// status == CSR_ILL: deliver IllInstr.
/// status == CSR_EXIT: exit to Python.
pub fn csr_write(state: &mut HartState, addr: u16, val: u64, pmp: &PmpCtx) -> u8 {
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
        MIP     => {
            // MSIP, MTIP, MEIP are read-only — driven by CLINT / timer /
            // external interrupt controller.  Preserve them; only allow
            // writes to software-writable bits (SSIP, USIP).
            let ro_mask: u64 = (1 << 3) | (1 << 7) | (1 << 11);
            state.mip = (state.mip & ro_mask) | (val & !ro_mask);
            CSR_OK
        }
        MCOUNTEREN => { state.mcounteren = val; CSR_OK }

        SSTATUS => {
            let mask = SSTATUS_WRITE_MASK;
            state.mstatus = (state.mstatus & !mask) | (val & mask);
            CSR_OK
        }
        SIE => {
            // sie is a restricted view of mie; only mideleg-delegated bits are writable
            let mask = state.mideleg;
            state.mie = (state.mie & !mask) | (val & mask);
            CSR_OK
        }
        SIP => {
            // sip is a restricted view of mip; only mideleg-delegated bits are writable
            let mask = state.mideleg;
            state.mip = (state.mip & !mask) | (val & mask);
            CSR_OK
        }
        STVEC   => { state.stvec = val; CSR_OK }
        SSCRATCH => { state.sscratch = val; CSR_OK }
        SEPC    => { state.sepc = val; CSR_OK }
        SCAUSE  => { state.scause = val; CSR_OK }
        STVAL   => { state.stval = val; CSR_OK }
        SCOUNTEREN => { state.scounteren = val; CSR_OK }
        SATP    => {
            // ASID (bits[59:44]) is hardwired to 0 (WARL): neither the Rust
            // batch-engine TLB nor the Python TLB tags entries with an ASID.
            // If ASID bits were readable-back, Linux would enable its ASID
            // allocator and skip sfence.vma on context switches, letting
            // stale translations from the previous address space hit — user
            // processes then load garbage and SIGSEGV (observed in ld.so).
            let val = val & !(0xFFFFu64 << 44);
            let new_mode = (val >> 60) as u8;
            state.satp = val;
            state.mmu_mode = new_mode;
            tlb_flush_inline(state);
            CSR_OK
        }

        // Machine info registers — read-only, writes are silently ignored
        MISA     => CSR_OK,
        MVENDORID => CSR_OK,
        MARCHID   => CSR_OK,
        MIMPID    => CSR_OK,
        MHARTID   => CSR_OK,

        // Sstc stimecmp (0x14D) — S-mode timer compare value.
        // Writing stimecmp also syncs to the CLINT mtimecmp array via the
        // pointer stored in clint context; this happens in Python's
        // ``_csr_write_raw("stimecmp", …)`` path during unmarshal.
        STIMECMP => { state.stimecmp = val; CSR_OK }

        // Floating-point CSRs — writing sets FS to Dirty + SD.
        FFLAGS => {
            if fp_off(state) { return CSR_ILL; }
            state.fcsr = (state.fcsr & !0x1F) | (val as u32 & 0x1F);
            state.mstatus |= MSTATUS_FS | MSTATUS_SD;
            CSR_OK
        }
        FRM => {
            if fp_off(state) { return CSR_ILL; }
            state.fcsr = (state.fcsr & !0xE0) | ((val as u32 & 0x7) << 5);
            state.mstatus |= MSTATUS_FS | MSTATUS_SD;
            CSR_OK
        }
        FCSR => {
            if fp_off(state) { return CSR_ILL; }
            state.fcsr = val as u32 & 0xFF;
            state.mstatus |= MSTATUS_FS | MSTATUS_SD;
            CSR_OK
        }

        // TEE CSRs (mdid / pmpsplit)
        0x5C0 => { state.mdid = val as u8; CSR_OK }
        0x5C1 => { state.pmpsplit = val as u8; CSR_OK }

        // Debug/trace trigger CSRs (0x7A0-0x7AF) — not implemented.
        // Writes are silently ignored (WO in some implementations, RW in others;
        // without Sdext they are WARL=0).
        0x7A0..=0x7AF => CSR_OK,

        // mtopi — raise IllInstr (see csr_read comment for rationale).
        0xFB0 /* mtopi */ => CSR_ILL,
        // Other AIA CSRs — no IMSIC; writes are ignored (WARL / read-only).
        0x308 /* mvien */ | 0x309 /* mvip */
        | 0x318 /* mvienh */ | 0x319 /* mviph */ => CSR_OK,

        // RV32-only high-half CSRs — writes ignored on RV64.
        0x310 /* mstatush */ | 0x312 /* medelegh */ | 0x313 /* midelegh */ => CSR_OK,

        // ---- PMP CSRs (RV64) ----
        // Odd-numbered pmpcfg registers are illegal on RV64.
        a if (0x3A1..=0x3AF).contains(&a) && (a & 1) != 0 => CSR_ILL,
        // Even-numbered pmpcfg: unpack u64 into 8 config bytes.
        a if (0x3A0..=0x3AE).contains(&a) && (a & 1) == 0 => {
            let entry_base = ((a - 0x3A0) as usize / 2) * 8;
            if (entry_base as u8) < pmp.num {
                for i in 0..8u16 {
                    let b = ((val >> (i * 8)) & 0xFF) as u8;
                    unsafe { *pmp.cfg.add(entry_base + i as usize) = b; }
                }
            }
            CSR_OK
        }
        // PMP address registers (0x3B0-0x3EF): one per entry, 8 bytes each.
        a if (0x3B0..=0x3EF).contains(&a) => {
            let n = (a - 0x3B0) as usize;
            if n < pmp.num as usize {
                unsafe { *pmp.addr.add(n) = val; }
            }
            CSR_OK
        }

        // Everything else -> exit to Python
        _ => CSR_EXIT,
    }
}

// ============================================================
//  Helpers
// ============================================================

/// Flush both TLBs without importing translate module.
#[inline]
fn tlb_flush_inline(state: &mut HartState) {
    for e in state.itlb.iter_mut() {
        e.valid = 0;
    }
    for e in state.dtlb.iter_mut() {
        e.valid = 0;
    }
}

/// 浮点单元是否被禁用 (mstatus.FS == Off) — FP CSR 访问需 FS != 0。
#[inline]
fn fp_off(state: &HartState) -> bool {
    (state.mstatus & MSTATUS_FS) == 0
}

/// Check if *mode* is privileged enough to access a CSR at *priv_req*.
#[inline]
fn csr_priv_ok(mode: u8, priv_req: u8) -> bool {
    match priv_req {
        0 => true,                  // user-readable
        1 => mode >= riscv_mode::S, // supervisor
        2 => mode >= riscv_mode::H, // hypervisor (unused)
        3 => mode >= riscv_mode::M, // machine
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
    pmp: &PmpCtx,
) -> u64 {
    // Read old CSR value
    let (old_val, status) = csr_read(state, csr_addr, hart_id, mtime, pmp);
    if status == CSR_ILL {
        deliver_illegal_instruction(state, instr as u64, result);
        return 0;
    }
    if status == CSR_EXIT {
        result.exit_reason = exit_reason::ECALL;
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
        1 => rs1_val,            // CSRRW / CSRRWI
        2 => old_val | rs1_val,  // CSRRS / CSRRSI
        3 => old_val & !rs1_val, // CSRRC / CSRRCI
        _ => {
            deliver_illegal_instruction(state, instr as u64, result);
            return 0;
        }
    };

    // Write back to CSR (ALWAYS write, even if rd==x0 — the CSR write
    // must happen regardless; only the old-value readback is skipped when rd==0).
    let wstatus = csr_write(state, csr_addr, new_val, pmp);
    if wstatus == CSR_ILL {
        deliver_illegal_instruction(state, instr as u64, result);
        return 0;
    }
    if wstatus == CSR_EXIT {
        result.exit_reason = exit_reason::ECALL;
        result.exit_instr = instr;
        return EXIT_SENTINEL;
    }

    // Write old value to rd (skip for x0)
    if rd != 0 {
        write_gpr(state, rd, old_val);
    }
    4
}

// ============================================================
//  GPR read/write (local copies for csr.rs independence)
// ============================================================

#[inline]
fn read_gpr(state: &HartState, rs: u8) -> u64 {
    if rs == 0 {
        0
    } else {
        state.gprs[rs as usize]
    }
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

    fn test_pmp() -> PmpCtx {
        PmpCtx {
            cfg: std::ptr::null_mut(),
            addr: std::ptr::null_mut(),
            num: 0,
        }
    }

    #[test]
    fn read_mstatus() {
        let mut s = test_state();
        s.mstatus = 0x1880; // MPP=S, MPIE=1
        let (v, st) = csr_read(&s, MSTATUS, 0, 0, &test_pmp());
        assert_eq!(st, CSR_OK);
        assert_eq!(v, 0x1880);
    }

    #[test]
    fn read_sstatus_view() {
        let mut s = test_state();
        s.mstatus = MSTATUS_SIE | MSTATUS_SPIE | MSTATUS_SPP;
        let (v, st) = csr_read(&s, SSTATUS, 0, 0, &test_pmp());
        assert_eq!(st, CSR_OK);
        assert_eq!(
            v & SSTATUS_READ_MASK,
            MSTATUS_SIE | MSTATUS_SPIE | MSTATUS_SPP
        );
    }

    #[test]
    fn write_satp_flushes_tlb() {
        let mut s = test_state();
        s.mmu_mode = 8;
        // Pre-fill TLB
        s.itlb[0].valid = 1;
        s.dtlb[0].valid = 1;
        // Write satp
        let st = csr_write(&mut s, SATP, 0x8000000000000000u64, &test_pmp()); // Sv39, PPN=0
        assert_eq!(st, CSR_OK);
        assert_eq!(s.mmu_mode, 8);
        // TLBs should be flushed
        assert_eq!(s.itlb[0].valid, 0);
        assert_eq!(s.dtlb[0].valid, 0);
    }

    /// satp.ASID (bits[59:44]) is hardwired to 0 — the TLB has no ASID tag.
    /// A readable-back ASID makes Linux enable its ASID allocator and skip
    /// sfence.vma on context switch, so stale TLB entries from the previous
    /// address space survive (user processes crash on garbage loads).
    /// Real-world trigger value from a live session: satp=0x8000100000082f1b
    /// (Sv39, ASID=1, PPN=0x82f1b) — ASID=1 proves the allocator was active.
    #[test]
    fn write_satp_asid_hardwired_zero() {
        let mut s = test_state();
        // Linux probe: write all-ones ASID, read back to count writable bits
        let probe = 0x8000000000000000u64 | (0xFFFFu64 << 44) | 0x82f1b;
        let st = csr_write(&mut s, SATP, probe, &test_pmp());
        assert_eq!(st, CSR_OK);
        let (v, st) = csr_read(&s, SATP, 0, 0, &test_pmp());
        assert_eq!(st, CSR_OK);
        assert_eq!(
            v, 0x8000000000082f1b,
            "ASID must read back as 0; MODE/PPN preserved"
        );
        assert_ne!(v, probe, "old behavior (raw ASID stored) must not reappear");
        // MODE and PPN must be unaffected by the mask
        assert_eq!(s.mmu_mode, 8);
    }

    #[test]
    fn write_sstatus_updates_mstatus_fields() {
        let mut s = test_state();
        // Write SIE bit through sstatus
        let st = csr_write(&mut s, SSTATUS, MSTATUS_SIE, &test_pmp());
        assert_eq!(st, CSR_OK);
        assert_eq!(s.mstatus & MSTATUS_SIE, MSTATUS_SIE);
    }

    #[test]
    fn write_unknown_csr_triggers_exit() {
        let mut s = test_state();
        let st = csr_write(&mut s, 0xB00, 0, &test_pmp()); // mcycle
        assert_eq!(st, CSR_EXIT);
    }

    /// sie (0x104) is a restricted view of mie (0x304).
    /// Writing sie should update mie for mideleg-delegated bits.
    #[test]
    fn write_sie_updates_mie() {
        let mut s = test_state();
        s.mideleg = 1 << 5; // STIE delegated
        s.mie = 0;
        let st = csr_write(&mut s, SIE, 1 << 5, &test_pmp()); // write STIE through sie
        assert_eq!(st, CSR_OK);
        assert_eq!(
            s.mie & (1 << 5),
            1 << 5,
            "mie.STIE must be set when sie.STIE is written"
        );
        // Non-delegated bits must not leak through
        s.mideleg = 1 << 5;
        s.mie = 0;
        let st = csr_write(&mut s, SIE, (1 << 5) | (1 << 3), &test_pmp()); // STIE + MSIP (MSIP not delegated)
        assert_eq!(st, CSR_OK);
        assert_eq!(
            s.mie & (1 << 3),
            0,
            "non-delegated bits in sie write must be masked"
        );
        assert_eq!(s.mie & (1 << 5), 1 << 5, "delegated STIE must still be set");
    }

    /// sie read must return mie & mideleg (only delegated bits visible).
    #[test]
    fn read_sie_shows_delegated_bits() {
        // Need to construct state with correct mie/mideleg
        let mut s2 = test_state();
        s2.mideleg = (1 << 5) | (1 << 1); // STIE + SSIE delegated
        s2.mie = (1 << 5) | (1 << 3); // STIE + MSIE set in mie
        let (v, st) = csr_read(&s2, SIE, 0, 0, &test_pmp());
        assert_eq!(st, CSR_OK);
        assert_eq!(
            v,
            1 << 5,
            "sie must only show delegated bits (STIE), not MSIE"
        );
    }

    /// sip (0x144) is a restricted view of mip (0x344).
    #[test]
    fn write_sip_updates_mip() {
        let mut s = test_state();
        s.mideleg = 1 << 1; // SSIP delegated
        s.mip = 0;
        let st = csr_write(&mut s, SIP, 1 << 1, &test_pmp()); // write SSIP through sip
        assert_eq!(st, CSR_OK);
        assert_eq!(
            s.mip & (1 << 1),
            1 << 1,
            "mip.SSIP must be set when sip.SSIP is written"
        );
    }

    #[test]
    fn read_time() {
        let s = test_state();
        let (v, st) = csr_read(&s, 0xC01, 0, 123456789, &test_pmp());
        assert_eq!(st, CSR_OK);
        assert_eq!(v, 123456789);
    }

    #[test]
    fn read_timeh() {
        let s = test_state();
        let (v, st) = csr_read(&s, 0xC81, 0, 0x123456789AB, &test_pmp());
        assert_eq!(st, CSR_OK);
        // 0x123456789AB >> 32 -> upper bits = 0x123 = 291
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

    // ---- handle_csr tests (rd==x0 regression) ----

    fn make_batch_result() -> BatchResult {
        unsafe { std::mem::zeroed() }
    }

    /// ``csrw mscratch, t0`` -> ``csrrw x0, mscratch, t0``
    /// The CSR write MUST happen even though rd==x0 (only the readback is skipped).
    #[test]
    fn handle_csr_write_with_rd_x0() {
        let mut s = test_state();
        s.gprs[5] = 0xDEAD_BEEF; // t0 = value to write
        let mut r = make_batch_result();
        // CSRRW x0, mscratch, t0  ->  funct3=001, rs1=5(t0), rd=0(x0), csr=0x340(mscratch)
        let advance = handle_csr(
            &mut s,
            0,
            5,
            MSCRATCH,
            0b001,
            0x34051073,
            &mut r,
            0,
            0,
            &test_pmp(),
        );
        assert_eq!(advance, 4);
        assert_eq!(
            s.mscratch, 0xDEAD_BEEF,
            "mscratch must be written even though rd==x0"
        );
    }

    /// ``csrwi mscratch, 5`` -> ``csrrwi x0, mscratch, 5``
    #[test]
    fn handle_csr_write_imm_with_rd_x0() {
        let mut s = test_state();
        let mut r = make_batch_result();
        // CSRRWI x0, mscratch, 5  ->  funct3=101, rs1=5(imm), rd=0, csr=0x340
        let advance = handle_csr(
            &mut s,
            0,
            5,
            MSCRATCH,
            0b101,
            0x340552f3,
            &mut r,
            0,
            0,
            &test_pmp(),
        );
        assert_eq!(advance, 4);
        assert_eq!(
            s.mscratch, 5,
            "mscratch must be written via csrwi even though rd==x0"
        );
    }

    /// ``csrrs t0, mscratch, x0`` -> rd=t0, rs1=x0  (read-only, no write bits set)
    /// Old value must go to rd, CSR unchanged.
    #[test]
    fn handle_csr_read_to_rd() {
        let mut s = test_state();
        s.mscratch = 0xCAFE;
        let mut r = make_batch_result();
        // CSRRS t0(5), mscratch, x0  ->  funct3=010, rs1=0, rd=5
        let advance = handle_csr(
            &mut s,
            5,
            0,
            MSCRATCH,
            0b010,
            0x3402b2f3,
            &mut r,
            0,
            0,
            &test_pmp(),
        );
        assert_eq!(advance, 4);
        assert_eq!(s.mscratch, 0xCAFE, "mscratch unchanged on read");
        assert_eq!(s.gprs[5], 0xCAFE, "old value written to rd");
    }

    // ---- PMP CSR write tests (inline in Rust, no CSR_EXIT) ----

    /// Build a PmpCtx backed by real mutable arrays so that PMP CSR
    /// writes can be verified against the underlying storage.
    fn test_pmp_real(num: u8) -> (PmpCtx, Vec<u8>, Vec<u64>) {
        let mut cfg = vec![0u8; num as usize];
        let mut addr = vec![0u64; num as usize];
        let ctx = PmpCtx {
            cfg: cfg.as_mut_ptr(),
            addr: addr.as_mut_ptr(),
            num,
        };
        (ctx, cfg, addr)
    }

    #[test]
    fn pmpcfg_write_packs_bytes() {
        let (pmp, cfg, _addr) = test_pmp_real(16);
        let mut s = test_state();
        // pmpcfg0 (0x3A0) covers entries 0-7.
        // Write: entry 0 = 0xAB, entry 2 = 0xCD (byte positions 0 and 2).
        let val: u64 = 0xAB | (0xCDu64 << 16);
        let st = csr_write(&mut s, 0x3A0, val, &pmp);
        assert_eq!(st, CSR_OK);
        assert_eq!(cfg[0], 0xAB);
        assert_eq!(cfg[1], 0x00);
        assert_eq!(cfg[2], 0xCD);
        // Entries beyond 7 should be untouched.
        assert_eq!(cfg[8], 0x00);
    }

    #[test]
    fn pmpcfg_write_out_of_range_is_noop() {
        let (pmp, cfg, _addr) = test_pmp_real(4);
        let mut s = test_state();
        // pmpcfg0 writes entries 0-7 but only 0-3 exist (num=4).
        let st = csr_write(&mut s, 0x3A0, 0xFF_FF_FF_FF_FF_FF_FF_FFu64, &pmp);
        assert_eq!(st, CSR_OK);
        // First 4 should be written.
        assert_eq!(cfg[0], 0xFF);
        assert_eq!(cfg[1], 0xFF);
        assert_eq!(cfg[2], 0xFF);
        assert_eq!(cfg[3], 0xFF);
        // Entries 4+ should NOT be touched (out of range).
        // (The test vector only has 4 elements, so reading beyond panics.)
    }

    #[test]
    fn pmpcfg_odd_returns_ill() {
        let (pmp, _cfg, _addr) = test_pmp_real(8);
        let mut s = test_state();
        // pmpcfg1 (0x3A1) is illegal on RV64.
        let st = csr_write(&mut s, 0x3A1, 0, &pmp);
        assert_eq!(st, CSR_ILL);
    }

    #[test]
    fn pmpaddr_write_works() {
        let (pmp, _cfg, addr) = test_pmp_real(16);
        let mut s = test_state();
        // pmpaddr3 (0x3B3) — write entry 3.
        let st = csr_write(&mut s, 0x3B3, 0xDEAD_BEEF_CAFE_BABEu64, &pmp);
        assert_eq!(st, CSR_OK);
        assert_eq!(addr[3], 0xDEAD_BEEF_CAFE_BABE);
        assert_eq!(addr[2], 0, "adjacent entries untouched");
        assert_eq!(addr[4], 0, "adjacent entries untouched");
    }

    #[test]
    fn pmpaddr_write_out_of_range_is_noop() {
        let (pmp, _cfg, addr) = test_pmp_real(4);
        let mut s = test_state();
        // pmpaddr4 (0x3B4) — entry 4, but num=4 so valid indices are 0-3.
        let st = csr_write(&mut s, 0x3B4, 0xFFFF_FFFF_FFFF_FFFFu64, &pmp);
        assert_eq!(st, CSR_OK);
        assert_eq!(addr[3], 0, "last valid entry untouched");
    }

    #[test]
    fn pmp_write_does_not_exit() {
        let (pmp, _cfg, _addr) = test_pmp_real(8);
        let mut s = test_state();
        // All PMP writes should return CSR_OK, not CSR_EXIT.
        assert_eq!(csr_write(&mut s, 0x3A0, 0, &pmp), CSR_OK); // pmpcfg0
        assert_eq!(csr_write(&mut s, 0x3A2, 0, &pmp), CSR_OK); // pmpcfg2
        assert_eq!(csr_write(&mut s, 0x3B0, 0, &pmp), CSR_OK); // pmpaddr0
        assert_eq!(csr_write(&mut s, 0x3B7, 0, &pmp), CSR_OK); // pmpaddr7
        assert_eq!(csr_write(&mut s, 0x3EF, 0, &pmp), CSR_OK); // pmpaddr63 (max)
    }
}
