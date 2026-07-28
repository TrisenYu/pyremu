//! Diagnostic counter helpers for the concurrent execution engine.
//!
//! Every function body is gated behind ``#[cfg(feature = "diagnostic")]``.
//! When the feature is disabled, each function compiles to a no-op — the
//! call sites stay clean (no ``#[cfg]`` at every caller) and the compiler
//! eliminates the dead stores.
//!
//! Parameter names use ``_`` prefix so they don't warn when unused.

use crate::state::HartDiag;

#[cfg(feature = "diagnostic")]
use std::sync::atomic::{AtomicU8, Ordering};

// ============================================================
//  Shared: log file helper
// ============================================================

/// Resolve the diagnostic log path from ``PYREMU_DIAG_LOG`` env var.
#[cfg(feature = "diagnostic")]
fn diag_log_path() -> String {
    std::env::var("PYREMU_DIAG_LOG")
        .unwrap_or_else(|_| "/tmp/pyremu_diag.log".to_string())
}

/// Append a pre-formatted line to the diagnostic log file.
/// All diagnostic output (sret trace, UART MMIO, store fault, etc.)
/// goes through this single sink so that nothing leaks to the
/// guest terminal.
#[inline(always)]
#[allow(dead_code)]
pub fn log_line(_line: &str) {
    #[cfg(feature = "diagnostic")]
    {
        use std::io::Write;
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&diag_log_path())
        {
            let _ = writeln!(f, "{}", _line);
        }
    }
}

// ============================================================
//  WFI spin-loop diagnostics
// ============================================================

/// Track that a direct-CLINT MSIP edge was detected in the WFI loop.
#[inline(always)]
#[allow(dead_code)]
pub fn wfi_msip_edge(_diag: &mut HartDiag) {
    #[cfg(feature = "diagnostic")]
    {
        _diag.clint_mtc_wr = _diag.clint_mtc_wr.wrapping_add(1);
    }
}

/// Track the reason for exiting the WFI spin loop.
#[inline(always)]
pub fn wfi_wake_reason(_diag: &mut HartDiag, _pending: u64, _msip_edge: bool) {
    #[cfg(feature = "diagnostic")]
    {
        if _msip_edge || ((_pending >> 3) & 1) != 0 {
            _diag.wfi_wake_msip = _diag.wfi_wake_msip.wrapping_add(1);
        } else if ((_pending >> 7) & 1) != 0 {
            _diag.wfi_wake_mtip = _diag.wfi_wake_mtip.wrapping_add(1);
        } else {
            _diag.wfi_wake_other = _diag.wfi_wake_other.wrapping_add(1);
        }
    }
}

// ============================================================
//  hart_worker diagnostics
// ============================================================

/// Track that MSIE was forced-set because MSIP was pending but masked.
#[inline(always)]
pub fn msie_forced(_diag: &mut HartDiag) {
    #[cfg(feature = "diagnostic")]
    {
        _diag.msip_masked_by_msie = _diag.msip_masked_by_msie.wrapping_add(1);
    }
}

/// Combined post-WFI diagnostic: MSIP-pending-no-trap snapshot +
/// WFI-wake-without-trap counter.  Called once per hart_worker
/// iteration after ``check_and_deliver_interrupt``.
#[inline(always)]
#[allow(dead_code)]
#[allow(unused)]
pub fn msip_post_wfi(
    _diag: &mut HartDiag,
    _mip: u64,
    _mie: u64,
    _mode: u8,
    _mhartid: u64,
    // Raw pointers to CLINT msip[] array + count (for snapshot).
    _msip_ptr: *const u8,
    _num_harts: u32,
    _msip_was_pending: bool,
    _woke_by_msip_edge: bool,
) {
    #[cfg(feature = "diagnostic")]
    {
        // 早先曾在此短路计数只保留第一帧快照，后发现需记录每次挂起才能
        // 精确定位 MSIP 投递丢失的窗口，故改为无条件累计。
        if !_msip_was_pending {
            if _woke_by_msip_edge {
                _diag.wfi_wake_no_msip_trap = _diag.wfi_wake_no_msip_trap.wrapping_add(1);
            }
            return;
        }
        if _diag.msip_pending_no_trap == 0 {
            _diag.nt_mip_snapshot = _mip;
            _diag.nt_mie_snapshot = _mie;
            _diag.nt_mode = _mode;
            _diag.nt_pending = _mip & _mie;
            let hid = _mhartid as usize;
            if hid < _num_harts as usize {
                let msip_atomics = _msip_ptr as *const AtomicU8;
                _diag.nt_clint_raw = unsafe { &*msip_atomics.add(hid) }.load(Ordering::Relaxed);
            }
        }
        _diag.msip_pending_no_trap = _diag.msip_pending_no_trap.wrapping_add(1);
    }
}

// ============================================================
//  Trap delivery diagnostics
// ============================================================

/// Track an MSIP trap delivery (from ``deliver_trap``).
#[inline(always)]
#[allow(unused)]
pub fn trap_msip(_diag: &mut HartDiag, _delegated: bool) {
    #[cfg(feature = "diagnostic")]
    {
        _diag.trap_msip_total = _diag.trap_msip_total.wrapping_add(1);
        if _delegated {
            _diag.trap_msip_delegated = _diag.trap_msip_delegated.wrapping_add(1);
        }
    }
}

// ============================================================
//  CLINT / CSR write diagnostics
// ============================================================

/// Track a direct CLINT MSIP MMIO write.
#[inline(always)]
#[allow(dead_code)]
#[allow(unused)]
pub fn clint_msip_write(_diag: &mut HartDiag, _target: u64, _self_hartid: u64, _val_one: bool) {
    #[cfg(feature = "diagnostic")]
    {
        if !_val_one {
            _diag.clint_msip_wr0 = _diag.clint_msip_wr0.wrapping_add(1);
            return;
        }
        _diag.clint_msip_wr1 = _diag.clint_msip_wr1.wrapping_add(1);
        if _target == _self_hartid {
            _diag.clint_wr1_self = _diag.clint_wr1_self.wrapping_add(1);
        } else {
            _diag.clint_wr1_remote = _diag.clint_wr1_remote.wrapping_add(1);
        }
    }
}

/// Track sret -> U-mode transitions: log key register state on first N transitions.
/// Gated by ``PYREMU_TRACE_SRET=N`` env var (runtime, requires "diagnostic" feature).
/// Set N=0 or omit to suppress; N>0 logs N transitions then stops.
/// Output goes to the diag log file (see ``log_line``).
#[inline(always)]
#[allow(dead_code)]
pub fn sret_to_umode(_state: &crate::state::HartState, _ctx: &crate::translate::WalkCtx) {
    #[cfg(feature = "diagnostic")]
    {
        use std::sync::atomic::{AtomicU32, Ordering};
        static REMAINING: AtomicU32 = AtomicU32::new(u32::MAX);
        let v = REMAINING.load(Ordering::Relaxed);
        let remaining = if v == u32::MAX {
            let init: u32 = std::env::var("PYREMU_TRACE_SRET")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            REMAINING.store(init, Ordering::Relaxed);
            init
        } else {
            v
        };
        if remaining == 0 {
            return;
        }
        REMAINING.store(remaining - 1, Ordering::Relaxed);
        let sp = _state.gprs[2];
        // Read first 8 u64 from user stack to verify argc/argv/envp/auxv setup.
        let stack = read_user_stack_from_sp(_state, _ctx);
        let msg = format!(
            "[sret->U] pc={:#018x} a0(argc)={:#018x} a1(argv)={:#018x} \
             a2(envp)={:#018x} sp={:#018x} ra={:#018x} \
             satp={:#018x} sepc={:#018x}",
            _state.sepc, _state.gprs[10], _state.gprs[11],
            _state.gprs[12], sp, _state.gprs[1],
            _state.satp, _state.sepc,
        );
        log_line(&msg);
        for (i, chunk) in stack.chunks(4).enumerate() {
            let base = i * 4;
            log_line(&format!(
                "         +0x{:02x}={:#018x} +0x{:02x}={:#018x} +0x{:02x}={:#018x} +0x{:02x}={:#018x}",
                base * 8, chunk[0], (base + 1) * 8, chunk[1],
                (base + 2) * 8, chunk[2], (base + 3) * 8, chunk[3],
            ));
        }
    }
}

/// Read 64 u64 words from the user stack at sp, translating through Sv39 if
/// enabled. Returns [0u64; 64] on translation failure.
#[cfg(feature = "diagnostic")]
fn read_user_stack_from_sp(
    _state: &crate::state::HartState,
    _ctx: &crate::translate::WalkCtx,
) -> [u64; 64] {
    let mut out = [0u64; 64];
    let sp = _state.gprs[2];
    for i in 0..64u64 {
        let va = sp.wrapping_add(i * 8);
        let pa = if _state.mmu_mode == 8 {
            // Sv39: translate VA->PA (read-only, non-execute page walk)
            match crate::translate::sv39_walk(_ctx, _state.satp, va, false) {
                Some(t) => t.pa,
                None => continue,
            }
        } else {
            va // Bare mode: VA=PA
        };
        if pa >= _ctx.ram_base && pa + 8 <= _ctx.ram_base + _ctx.ram_size {
            let off = (pa - _ctx.ram_base) as usize;
            let mut bytes = [0u8; 8];
            for j in 0..8 {
                bytes[j] = unsafe { *_ctx.ram.add(off + j) };
            }
            out[i as usize] = u64::from_le_bytes(bytes);
        }
    }
    out
}

// ============================================================
//  Store page-fault diagnostics
// ============================================================

/// Dump register state and page-table context on a low-address store page fault.
/// Intended to help trace wild-pointer stores (e.g. a corrupted base register).
/// All output goes to the diag log file (see ``log_line``).
#[inline(always)]
#[allow(dead_code)]
pub fn store_pagefault_diag(
    _state: &mut crate::state::HartState,
    _ctx: &crate::translate::WalkCtx,
    _va: u64,
    _instr: u32,
    _base: u64,
    _src: u64,
    _rs1: u8,
    _rs2: u8,
    _store_imm: u64,
) {
    #[cfg(feature = "diagnostic")]
    {
        use crate::translate::translate_va;
        log_line(&format!(
            "[diag-stfault] hart={} va=0x{:x} pc=0x{:x} instr=0x{:08x} \
             base=x{}={:#018x} src=x{}={:#018x} imm={} sp={:#018x}",
            _state.mhartid, _va, _state.pc, _instr,
            _rs1, _base, _rs2, _src, _store_imm as i16, _state.gprs[2],
        ));
        log_line(&format!(
            "[diag-stfault] t0={:#018x} t1={:#018x} t2={:#018x} \
             t3={:#018x} t4={:#018x} t5={:#018x} t6={:#018x}",
            _state.gprs[5], _state.gprs[6], _state.gprs[7],
            _state.gprs[28], _state.gprs[29], _state.gprs[30], _state.gprs[31],
        ));
        let pc_page = _state.pc & !0xFFF;
        let pc_off = _state.pc & 0xFFF;
        // Diagnostic TLB probe — we want ALL entries, including "expired"
        // ones (epoch mismatch), so iterate manually without an epoch filter.
        let mut itlb_hit_idx: Option<usize> = None;
        for i in 0..32 {
            if _state.itlb[i].valid != 0 && _state.itlb[i].vpn == pc_page >> 12 {
                itlb_hit_idx = Some(i);
                break;
            }
        }
        if let Some(idx) = itlb_hit_idx {
            let e = &_state.itlb[idx];
            let instr_pa = (e.ppn << 12) | pc_off;
            log_line(&format!(
                "[diag-stfault] itlb-hit vpn=0x{:x} ppn=0x{:x} pa=0x{:x}",
                pc_page >> 12, e.ppn, instr_pa,
            ));
        }
        if let Ok(t) = translate_va(_state, _ctx, _state.pc, false, true) {
            let instr_pa = t.pa;
            log_line(&format!(
                "[diag-stfault] translate-va pc->pa=0x{:x} level={} perm=0x{:x}",
                instr_pa, t.level, t.perm,
            ));
            if instr_pa >= _ctx.ram_base && instr_pa + 4 <= _ctx.ram_base + _ctx.ram_size {
                let off = (instr_pa - _ctx.ram_base) as usize;
                let b = unsafe {
                    [
                        *_ctx.ram.add(off),
                        *_ctx.ram.add(off + 1),
                        *_ctx.ram.add(off + 2),
                        *_ctx.ram.add(off + 3),
                    ]
                };
                let raw = u32::from_le_bytes(b);
                log_line(&format!(
                    "[diag-stfault] ram@pa=0x{:x} raw_bytes={:02x}{:02x}{:02x}{:02x} instr=0x{:08x}",
                    instr_pa, b[0], b[1], b[2], b[3], raw,
                ));
            }
        }
    }
}

/// Track the PC where MSIE was explicitly cleared (CSR write).
#[inline(always)]
pub fn msie_cleared_at(_diag: &mut HartDiag, _pc: u64) {
    #[cfg(feature = "diagnostic")]
    {
        _diag.msie_cleared_at_pc = _pc;
    }
}

// ============================================================
//  ld-linux.so trap trace
// ============================================================

/// Log trap details when trap PC is in the ld-linux.so range.
/// Helps trace why the dynamic linker fails to read AT_BASE correctly.
#[inline(always)]
#[allow(dead_code)]
pub fn ld_linux_trap(
    _state: &crate::state::HartState,
    _code: u64,
    _tval: u64,
) {
    #[cfg(feature = "diagnostic")]
    {
        // ld-linux.so range: base=0x3ff7fdc000, size=0x20000
        let pc = _state.pc;
        let is_int = (_code >> 63) != 0;
        let exc = _code & 0x7FFF_FFFF_FFFF_FFFF;
        let lo: u64 = 0x3ff7fdc000;
        let hi: u64 = 0x3ff7ffc000;
        if pc < lo || pc >= hi {
            return;
        }
        log_line(&format!(
            "[ld-trap] pc={:#018x} ({:+}) sepc={:#018x} cause={} {} tval={:#018x} \
             a0={:#018x} a1={:#018x} a5={:#018x} sp={:#018x} ra={:#018x}",
            pc,
            pc as i64 - lo as i64,
            _state.sepc,
            if is_int { "IRQ" } else { "EXC" },
            exc,
            _tval,
            _state.gprs[10], _state.gprs[11],
            _state.gprs[15], // a5 / t0
            _state.gprs[2], _state.gprs[1],
        ));
    }
}

// ============================================================
//  Suspicious-small-address access diagnostic
// ============================================================

/// Log when a load/store instruction uses a virtual address < 0x10000.
/// This is the actual crash signature: a small value (like 7=DT_RELA or
/// 0x320) has been loaded into a GPR earlier and is now being used as a
/// memory pointer. Logging at the point of the access captures the exact
/// instruction, address, and base register state at the crash moment.
#[inline(always)]
#[allow(dead_code)]
pub fn diag_small_addr_access(
    _state: &crate::state::HartState,
    _va: u64,
    _base: u64,
    _rs1: u8,
    _instr: u32,
    _is_store: bool,
    _size: u8,
) {
    #[cfg(feature = "diagnostic")]
    {
        if _va >= 0x10000 {
            return;
        }
        const GPR_NAMES: [&str; 32] = [
            "zero","ra","sp","gp","tp","t0","t1","t2",
            "s0","s1","a0","a1","a2","a3","a4","a5",
            "a6","a7","s2","s3","s4","s5","s6","s7",
            "s8","s9","s10","s11","t3","t4","t5","t6",
        ];
        let rn = if (_rs1 as usize) < 32 { GPR_NAMES[_rs1 as usize] } else { "?" };
        let op = if _is_store { "st" } else { "ld" };
        log_line(&format!(
            "[small-addr] pc={:#018x} {} va={:#x} size={} base={}(x{})={:#x} instr={:#010x} \
             a0={:#x} a1={:#x} a5={:#x} sp={:#x} ra={:#x} \
             satp={:#x} mode={}",
            _state.pc, op, _va, _size, rn, _rs1, _base, _instr,
            _state.gprs[10], _state.gprs[11],
            _state.gprs[15],
            _state.gprs[2], _state.gprs[1],
            _state.satp, _state.mode,
        ));
    }
}

// ============================================================
//  Suspicious-small-value store diagnostic
// ============================================================

/// Log when a store instruction writes a small value (1..0xFF) to memory.
/// This catches the moment a tag/offset value (like DT_RELA=7) is stored,
/// which later gets loaded and used as a pointer -> page fault.
#[inline(always)]
#[allow(dead_code)]
pub fn diag_small_store(
    _state: &crate::state::HartState,
    _va: u64,
    _val: u64,
    _size: u8,
    _rs2: u8,
    _instr: u32,
) {
    #[cfg(feature = "diagnostic")]
    {
        if _val == 0 || _val >= 0x100 {
            return;
        }
        const GPR_NAMES: [&str; 32] = [
            "zero","ra","sp","gp","tp","t0","t1","t2",
            "s0","s1","a0","a1","a2","a3","a4","a5",
            "a6","a7","s2","s3","s4","s5","s6","s7",
            "s8","s9","s10","s11","t3","t4","t5","t6",
        ];
        let rn = if (_rs2 as usize) < 32 { GPR_NAMES[_rs2 as usize] } else { "?" };
        log_line(&format!(
            "[small-st] pc={:#018x} st va={:#x} size={} src={}(x{})={:#x} instr={:#010x} \
             a0={:#x} a1={:#x} a5={:#x} sp={:#x} ra={:#x} \
             satp={:#x} mode={}",
            _state.pc, _va, _size, rn, _rs2, _val, _instr,
            _state.gprs[10], _state.gprs[11],
            _state.gprs[15],
            _state.gprs[2], _state.gprs[1],
            _state.satp, _state.mode,
        ));
    }
}

// ============================================================
//  Instruction frequency counters (diagnostic builds only)
// ============================================================

use crate::decode::{CompressedFields, DecodedFields};

#[cfg(feature = "diagnostic")]
use std::sync::atomic::AtomicU64;

/// Mnemonic table — index -> human-readable instruction name.
#[cfg(feature = "diagnostic")]
static INSTR_NAMES: &[&str] = &[
    // 0-7: LOAD (funct3)
    "LB","LH","LW","LD","LBU","LHU","LWU","LD?",
    // 8-15: STORE (funct3)
    "SB","SH","SW","SD","ST?","ST?","ST?","ST?",
    // 16-23: OP (funct7=0)
    "ADD","SLL","SLT","SLTU","XOR","SRL","OR","AND",
    // 24-31: OP (funct7=0x20 or M)
    "SUB","SRA","MUL","MULH","MULHSU","MULHU","DIV","DIVU",
    // 32-33: OP M cont
    "REM","REMU",
    // 34-41: OP-IMM (funct3)
    "ADDI","SLLI","SLTI","SLTIU","XORI","SRLI","ORI","ANDI",
    // 42: SRAI
    "SRAI",
    // 43-46: OP-IMM-32
    "ADDIW","SLLIW","SRLIW","SRAIW",
    // 47-54: OP-32
    "ADDW","SUBW","SLLW","SRLW","SRAW","MULW","DIVW","DIVUW",
    // 55-56
    "REMW","REMUW",
    // 57-60
    "LUI","AUIPC","JAL","JALR",
    // 61-66: BRANCH
    "BEQ","BNE","BLT","BGE","BLTU","BGEU",
    // 67-69: FENCE
    "FENCE","FENCE_I","SFENCE_VMA",
    // 70-77: SYSTEM (func3 0)
    "ECALL","EBREAK","MRET","SRET","WFI","CSRRW","CSRRS","CSRRC",
    // 78-81: SYSTEM cont
    "CSRRWI","CSRRSI","CSRRCI","SYSTEM?",
    // 82-93: AMO
    "LR","SC","AMOSWAP","AMOADD","AMOXOR","AMOAND","AMOOR",
    "AMOMIN","AMOMAX","AMOMINU","AMOMAXU","AMO?",
    // 94-97: FP
    "FLW","FLD","FSW","FSD",
    // 98-99: FP compute
    "FP-OP","FP-FMA",
    // 100-127: COMPRESSED
    "C.ADDI4SPN","C.LW","C.LD","C.SW","C.SD","C.FLD","C.FSD","C0?",
    "C.ADDI","C.JAL","C.LI","C.LUI","C.ARITH","C.J","C.BEQZ","C.BNEZ",
    "C.SLLI","C.LWSP","C.LDSP","C.JR/MV","C.JALR/ADD","C.SWSP","C.SDSP","C2?",
    "C.FLDSP","C.FSDSP","C.FLWSP","C.FSWSP","C.EBREAK","C.ILLEGAL",
    // 128-129
    "ILLEGAL","UNKNOWN",
];

#[cfg(feature = "diagnostic")]
const N_COUNTERS: usize = INSTR_NAMES.len();

#[cfg(feature = "diagnostic")]
static ICOUNT: [AtomicU64; N_COUNTERS] = {
    const Z: AtomicU64 = AtomicU64::new(0);
    [Z; N_COUNTERS]
};

#[cfg(feature = "diagnostic")]
fn iclass_32(f: &DecodedFields) -> usize {
    match f.opcode {
        0b00000_11 => f.func3.min(7) as usize,
        0b01000_11 => 8 + f.func3.min(7) as usize,
        0b01100_11 => {
            if f.func7 & 0x20 == 0 { 16 + f.func3.min(7) as usize }
            else if f.func7 == 0x20 { if f.func3==0 {24} else if f.func3==5 {25} else {129} }
            else if f.func7 == 0x01 { 26 + f.func3.min(7) as usize }
            else { 129 }
        }
        0b00100_11 => {
            if f.func3 == 5 && f.func7 == 0x20 { 42 }
            else if f.func3 <= 7 { 34 + f.func3 as usize }
            else { 129 }
        }
        0b00110_11 => match f.func3 {
            0 => 43, 1 => 44,
            5 if f.func7==0 => 45, 5 if f.func7==0x20 => 46,
            _ => 129,
        }
        0b01110_11 => {
            if f.func7 == 0x00 { match f.func3 { 0=>47,1=>49,5=>50, _=>129 } }
            else if f.func7 == 0x20 { match f.func3 { 0=>48,5=>51, _=>129 } }
            else if f.func7 == 0x01 { match f.func3 { 0=>52,4=>53,5=>54,6=>55,7=>56, _=>129 } }
            else { 129 }
        }
        0b01101_11 => 57, 0b00101_11 => 58,
        0b11011_11 => 59, 0b11001_11 => 60,
        0b11000_11 => 61 + f.func3.min(5) as usize,
        0b00011_11 => match f.func3 { 0 => 67, 1 => 68, _ => 69, }
        0b11100_11 => match f.func3 {
            0b000 => match f.func12 { 0x000=>70, 0x001=>71, 0x302=>72, 0x102=>73, 0x105=>74, _=>81 }
            0b001 => 75, 0b010 => 76, 0b011 => 77,
            0b101 => 78, 0b110 => 79, 0b111 => 80,
            _ => 81,
        }
        0b01011_11 => match (f.func7 >> 2) & 0x1F {
            0b00010=>82, 0b00011=>83, 0b00001=>84, 0b00000=>85,
            0b00100=>86, 0b01100=>87, 0b01000=>88, 0b10000=>89,
            0b10100=>90, 0b11000=>91, 0b11100=>92, _=>93,
        }
        0b00001_11 => 94 + f.func3.min(1) as usize,
        0b01001_11 => 96 + f.func3.min(1) as usize,
        0b10100_11 => 98,
        0b10000_11..=0b10011_11 => 99,
        _ => 128,
    }
}

#[cfg(feature = "diagnostic")]
fn iclass_c(c: &CompressedFields) -> usize {
    match c.quadrant {
        0 => match c.funct3 {
            0=>100, 2=>101, 3=>102, 6=>103, 7=>104, 1=>105, 5=>106, _=>107,
        }
        1 => match c.funct3 {
            0=>108, 1=>109, 2=>110, 3=>111, 4=>112, 5=>113, 6=>114, 7=>115, _=>127,
        }
        2 => match c.funct3 {
            0=>116, 2=>117, 3=>118,
            4 => if c.bit12==0 {119} else {120},
            6=>121, 7=>122, 1=>123, 5=>125, _=>126,
        }
        _ => 127,
    }
}

#[inline]
#[allow(unused)]
pub fn icount_32(_f: &DecodedFields, _instr: u32) {
    #[cfg(feature = "diagnostic")]
    {
        let i = iclass_32(_f);
        if i < N_COUNTERS { ICOUNT[i].fetch_add(1, Ordering::Relaxed); }
        let _ = _instr;
    }
}

#[inline]
#[allow(unused)]
pub fn icount_c(_c: &CompressedFields) {
    #[cfg(feature = "diagnostic")]
    {
        let i = iclass_c(_c);
        if i < N_COUNTERS { ICOUNT[i].fetch_add(1, Ordering::Relaxed); }
    }
}

/// Dump non-zero instruction counters to the diagnostic log, then reset all.
pub fn icount_dump() {
    #[cfg(feature = "diagnostic")]
    {
        let mut v: Vec<(usize, u64)> = (0..N_COUNTERS)
            .map(|i| (i, ICOUNT[i].swap(0, Ordering::Relaxed)))
            .filter(|(_, c)| *c > 0)
            .collect();
        v.sort_by(|a, b| b.1.cmp(&a.1));
        for (idx, cnt) in &v {
            let name = INSTR_NAMES.get(*idx).unwrap_or(&"?");
            log_line(&format!("[icount] {} {}", name, cnt));
        }
    }
}
