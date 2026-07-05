//! Batch instruction execution loop — full RISC-V dispatch (Phases A–E).
//!
//! ``run_batch`` is the single FFI entry point that replaces the Python
//! fetch-decode-execute loop.  It runs up to *max_instrs* instructions
//! across all non-halted harts, handling all RV64IMAC instructions inline.

use crate::decode::decode_fields;
use crate::handlers::{
    handle_load, handle_store, handle_system, handle_amo, handle_compressed,
    EXIT_SENTINEL,
};
use crate::op_dispatcher::{
    handle_alu, handle_auipc, handle_br, handle_fence, handle_jal, handle_jalr, handle_lui,
    handle_op_imm, handle_op_imm32, handle_op32,
};
use crate::state::{exit_reason, BatchResult, HartState, riscv_mode};
use crate::trap::{deliver_trap, deliver_illegal_instruction, exc_code, mcause_val};
use crate::translate::{WalkCtx, translate_va, TranslateFault};

// ============================================================
//  Instruction fetch
// ============================================================

#[inline]
fn fetch_instr(ram: *const u8, ram_size: u64, ram_base: u64, pa: u64) -> Option<u32> {
    let offset = pa.wrapping_sub(ram_base);
    if offset > ram_size.saturating_sub(4) {
        return None;
    }
    let ptr = unsafe { ram.add(offset as usize) };
    let b0 = unsafe { *ptr } as u32;
    let b1 = unsafe { *ptr.add(1) } as u32;
    let b2 = unsafe { *ptr.add(2) } as u32;
    let b3 = unsafe { *ptr.add(3) } as u32;
    Some(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24))
}

// ============================================================
//  Interrupt checking (Phase E)
// ============================================================

/// Interrupt priority order (RISC-V Privileged Spec §3.1.9):
/// MEI > MSI > MTI > SEI > SSI > STI
const MIE_MEIE: u64 = 1 << 11; // M-mode external
const MIE_MSIE: u64 = 1 << 3;  // M-mode software
const MIE_MTIE: u64 = 1 << 7;  // M-mode timer
const MIE_SEIE: u64 = 1 << 9;  // S-mode external
const MIE_SSIE: u64 = 1 << 1;  // S-mode software
const MIE_STIE: u64 = 1 << 5;  // S-mode timer

/// Check if any interrupt is pending and enabled. Returns (cause_code, is_m_mode_interrupt)
/// or None.  Respects delegation via ``mideleg``.
fn check_pending_interrupts(state: &HartState) -> Option<(u64, bool)> {
    let pending = state.mip & state.mie;
    if pending == 0 { return None; }

    // Priority-ordered check (each returns immediately on first hit)
    let checks: [(u64, u64, u64); 6] = [
        (MIE_MEIE, 11, 0), // MEI
        (MIE_MSIE, 3, 0),  // MSI
        (MIE_MTIE, 7, 0),  // MTI
        (MIE_SEIE, 9, 1),  // SEI (delegatable)
        (MIE_SSIE, 1, 1),  // SSI (delegatable)
        (MIE_STIE, 5, 1),  // STI (delegatable)
    ];

    for (mask, cause, delegatable) in &checks {
        if pending & mask == 0 { continue; }

        // --- delegatable interrupt (SEI / SSI / STI) ---
        if *delegatable != 0 {
            let delegated = state.mideleg & mask != 0;
            if delegated && state.mode < riscv_mode::M {
                // S-mode global interrupt enable
                if state.mstatus & (1 << 1) == 0 {
                    continue; // SIE=0 → skip, try next priority
                }
                return Some((*cause, false)); // deliver to S-mode
            }
            // not delegated, or currently in M-mode → handle as M-mode
        }

        // --- M-mode interrupt ---
        // In S/U mode, M-level interrupts always preempt
        if state.mode < riscv_mode::M {
            return Some((*cause, true));
        }
        // In M-mode: need MIE=1
        if state.mstatus & (1 << 3) != 0 {
            return Some((*cause, true));
        }
        // MIE=0 in M-mode → no interrupts taken
        return None;
    }
    None
}

/// Check for interrupts at instruction boundary.
/// Returns true if an interrupt was delivered (caller should stop execution on this hart).
fn check_and_deliver_interrupt(
    state: &mut HartState,
    result: &mut BatchResult,
) -> bool {
    if let Some((cause, is_m_mode)) = check_pending_interrupts(state) {
        let code = mcause_val(cause, true);
        deliver_trap(state, code, 0, result);
        result.exit_reason = exit_reason::TRAP;
        result.trap_cause = code as u32;
        result.trap_is_interrupt = 1;
        result.trap_delegated = if is_m_mode { 0 } else { 1 };
        result.trap_tval = 0;
        true
    } else {
        false
    }
}

// ============================================================
//  Full dispatch table (Phases A-D)
// ============================================================

/// Dispatch a 32-bit instruction. Returns advance (0/4) or EXIT_SENTINEL.
fn dispatch(
    state: &mut HartState,
    f: &crate::decode::DecodedFields,
    instr: u32,
    result: &mut BatchResult,
    ctx: &WalkCtx,
    hart_id: u8,
    mtime: u64,
    pmp_cfg: *const u8,
    pmp_addr: *const u64,
    pmp_num: u8,
    dev_bases: *const u64,
    dev_ends: *const u64,
    num_devices: u8,
) -> u64 {
    match f.opcode {
        0b01100_11 => handle_alu(state, f, instr, result),
        0b00100_11 => handle_op_imm(state, f, instr, result),
        0b00110_11 => handle_op_imm32(state, f, instr, result),
        0b01110_11 => handle_op32(state, f, instr, result),
        0b01101_11 => handle_lui(state, f),
        0b00101_11 => handle_auipc(state, f),
        0b11011_11 => handle_jal(state, f),
        0b11001_11 => handle_jalr(state, f),
        0b11000_11 => handle_br(state, f, instr, result),
        0b00011_11 => handle_fence(state, f, instr, result),

        // Phase B: Loads / Stores
        0b00000_11 => handle_load(
            state, f, instr, result, ctx,
            pmp_cfg, pmp_addr, pmp_num,
            dev_bases, dev_ends, num_devices,
        ),
        0b01000_11 => handle_store(
            state, f, instr, result, ctx,
            pmp_cfg, pmp_addr, pmp_num,
            dev_bases, dev_ends, num_devices,
        ),

        // Phase C: System
        0b11100_11 => handle_system(
            state, f, instr, result, ctx, hart_id, mtime,
            pmp_cfg, pmp_addr, pmp_num,
            dev_bases, dev_ends, num_devices,
        ),

        // Phase D: AMO
        0b01011_11 => handle_amo(
            state, f, instr, result, ctx,
            pmp_cfg, pmp_addr, pmp_num,
            dev_bases, dev_ends, num_devices,
        ),

        _ => EXIT_SENTINEL,
    }
}

/// Translate PC through MMU for instruction fetch.
/// Returns ``Some(pa)`` on success, ``None`` if a trap was delivered.
#[inline]
fn translate_fetch_pc(
    state: &mut HartState,
    ctx: &WalkCtx,
    pc: u64,
    result: &mut BatchResult,
    hart_id: u8,
) -> Option<u64> {
    if state.mmu_mode == 0 {
        return Some(pc); // Bare mode: VA == PA
    }
    match translate_va(state, ctx, pc, false, true) {
        Ok(t) => Some(t.pa),
        Err(e) => {
            let cause = match e {
                TranslateFault::PageFault(c) => c,
                TranslateFault::AccessFault => exc_code::INSTR_ACCESS_FAULT,
            };
            deliver_trap(state, mcause_val(cause, false), pc, result);
            result.exit_reason = exit_reason::TRAP;
            result.exit_hart_id = hart_id;
            result.exit_pc = pc;
            result.exit_instr = 0;
            None
        }
    }
}

/// Fetch an instruction word from physical RAM.
/// Returns ``Some(instr)`` on success, ``None`` if the address is out of range.
#[inline]
fn fetch_instr_safe(
    state: &mut HartState,
    ram: *const u8,
    ram_size: u64,
    ram_base: u64,
    pa: u64,
    pc_before: u64,
    result: &mut BatchResult,
    hart_id: u8,
) -> Option<u32> {
    match fetch_instr(ram, ram_size, ram_base, pa) {
        Some(w) => Some(w),
        None => {
            deliver_trap(
                state,
                mcause_val(exc_code::INSTR_ACCESS_FAULT, false),
                pc_before,
                result,
            );
            result.exit_reason = exit_reason::TRAP;
            result.exit_hart_id = hart_id;
            result.exit_pc = pc_before;
            result.exit_instr = 0;
            None
        }
    }
}

const TRAP_LOOP_THRESHOLD: u8 = 3;

fn run_hart_slice(
    state: &mut HartState,
    hart_id: u8,
    ram: *const u8,
    ram_size: u64,
    ram_base: u64,
    shadow_base: u64,
    shadow_size: u64,
    max_instrs: u64,
    result: &mut BatchResult,
    mtime: u64,
    pmp_cfg: *const u8,
    pmp_addr: *const u64,
    pmp_num: u8,
    dev_bases: *const u64,
    dev_ends: *const u64,
    num_devices: u8,
) -> u64 {
    let mut count: u64 = 0;
    let ctx = WalkCtx { ram: ram as *mut u8, ram_size, ram_base, shadow_base, shadow_size };

    while count < max_instrs {
        // ---- Halted check ----
        if state.halted != 0 { return count; }

        // ---- WFI waiting ----
        if state.waiting != 0 {
            // Check if an interrupt can wake us
            if check_pending_interrupts(state).is_some() {
                state.waiting = 0;
                // Interrupt will be delivered below
            } else {
                result.exit_reason = exit_reason::WFI_WAIT;
                result.exit_hart_id = hart_id;
                result.exit_pc = state.pc;
                result.exit_instr = 0;
                return count;
            }
        }

        // ---- Interrupt check at instruction boundary ----
        if check_and_deliver_interrupt(state, result) {
            result.exit_hart_id = hart_id;
            result.exit_pc = state.pc;
            return count;
        }

        // ---- Instruction fetch with MMU translation ----
        let pc_before = state.pc;
        let fetch_pa = match translate_fetch_pc(state, &ctx, pc_before, result, hart_id) {
            Some(pa) => pa,
            None => return count,
        };
        let instr_word = match fetch_instr_safe(
            state, ram, ram_size, ram_base, fetch_pa, pc_before, result, hart_id,
        ) {
            Some(w) => w,
            None => return count,
        };

        // ---- Decode ----
        let f = decode_fields(instr_word);

        // ---- Compressed instruction ----
        if f.is_compressed != 0 {
            let half = (instr_word & 0xFFFF) as u16;
            let advance = handle_compressed(
                state, half, instr_word, result, &ctx,
                pmp_cfg, pmp_addr, pmp_num,
                dev_bases, dev_ends, num_devices,
            );
            if advance == EXIT_SENTINEL {
                result.exit_reason = exit_reason::SYS_EXIT;
                result.exit_hart_id = hart_id;
                result.exit_pc = pc_before;
                result.exit_instr = instr_word;
                return count;
            }
            if advance == 0 && state.pc == pc_before {
                // Trap or jump; PC already updated by handler
                count += 1;
                check_consecutive_trap(state, result, hart_id, &mut count);
                continue;
            }
            if advance != 0 && state.pc == pc_before {
                state.pc = state.pc.wrapping_add(advance);
                state.consecutive_traps = 0;
            }
            count += 1;
            continue;
        }

        // ---- Dispatch ----
        let advance = dispatch(
            state, &f, instr_word, result, &ctx, hart_id, mtime,
            pmp_cfg, pmp_addr, pmp_num,
            dev_bases, dev_ends, num_devices,
        );

        if advance == EXIT_SENTINEL {
            result.exit_reason = exit_reason::SYS_EXIT;
            result.exit_hart_id = hart_id;
            result.exit_pc = pc_before;
            result.exit_instr = instr_word;
            return count;
        }

        // ---- PC update ----
        if advance != 0 && state.pc == pc_before {
            state.pc = state.pc.wrapping_add(advance);
            state.consecutive_traps = 0;
        } else if advance == 0 {
            // PC was modified by handler (jump/trap), increment count
            // but check for trap loop
        }

        count += 1;

        // ---- Consecutive trap check ----
        if state.consecutive_traps >= TRAP_LOOP_THRESHOLD {
            state.halted = 1;
            result.exit_reason = exit_reason::TRAP;
            result.exit_hart_id = hart_id;
            result.exit_pc = state.pc;
            result.exit_instr = 0;
            return count;
        }
    }

    count
}

#[inline]
fn check_consecutive_trap(
    state: &mut HartState,
    result: &mut BatchResult,
    hart_id: u8,
    count: &mut u64,
) {
    // Count handled above in run_hart_slice main loop
    if state.consecutive_traps >= TRAP_LOOP_THRESHOLD {
        state.halted = 1;
        result.exit_reason = exit_reason::TRAP;
        result.exit_hart_id = hart_id;
        result.exit_pc = state.pc;
        result.exit_instr = 0;
    }
}

// ============================================================
//  Public FFI entry point
// ============================================================

#[no_mangle]
pub unsafe extern "C" fn run_batch(
    states: *mut HartState,
    num_harts: u32,
    ram: *mut u8,
    ram_size: u64,
    ram_base: u64,
    shadow_base: u64,
    shadow_size: u64,
    max_instrs: u64,
    result: *mut BatchResult,
    // PMP state
    pmp_cfg: *const u8,
    pmp_addr: *const u64,
    pmp_num: u8,
    pmpsplit_val: u8,
    // CLINT state
    mtime: u64,
    mtimecmp: *const u64,
    msip: *const u8,
    // Device MMIO ranges
    dev_bases: *const u64,
    dev_ends: *const u64,
    num_devices: u8,
) {
    // Zero-initialise the result
    unsafe {
        (*result).total_instrs = 0;
        (*result).exit_reason = exit_reason::NORMAL;
        (*result).exit_hart_id = 0;
        (*result).exit_pc = 0;
        (*result).exit_instr = 0;
        (*result).trap_cause = 0;
        (*result).trap_tval = 0;
        (*result).trap_is_interrupt = 0;
        (*result).trap_delegated = 0;
    }

    let mut grand_total: u64 = 0;

    for hid in 0..num_harts {
        let state = unsafe { &mut *states.add(hid as usize) };

        // Build per-hart CLINT mip bits based on mtimecmp and msip
        let hart_mip = build_mip(state, hid as u8, mtime, mtimecmp, msip);
        state.mip = hart_mip;

        if state.halted != 0 { continue; }

        let executed = run_hart_slice(
            state, hid as u8,
            ram as *const u8, ram_size, ram_base,
            shadow_base, shadow_size,
            max_instrs.saturating_sub(grand_total),
            unsafe { &mut *result },
            mtime,
            pmp_cfg, pmp_addr, pmp_num,
            dev_bases, dev_ends, num_devices,
        );
        grand_total += executed;

        unsafe { (*result).total_instrs = grand_total; }

        if unsafe { (*result).exit_reason } != exit_reason::NORMAL {
            return;
        }
        if grand_total >= max_instrs {
            return;
        }
    }
}

/// Build the mip CSR value for a hart based on CLINT state.
fn build_mip(
    state: &HartState,
    hart_id: u8,
    mtime: u64,
    mtimecmp: *const u64,
    msip: *const u8,
) -> u64 {
    let mut mip = state.mip;

    // MTIP: timer interrupt
    let cmp = unsafe { *mtimecmp.add(hart_id as usize) };
    if mtime >= cmp {
        mip |= 1 << 7; // MTIP
    } else {
        mip &= !(1 << 7);
    }

    // MSIP: software interrupt
    let sip = unsafe { *msip.add(hart_id as usize) };
    if sip != 0 {
        mip |= 1 << 3; // MSIP
    } else {
        mip &= !(1 << 3);
    }

    // STIP: S-mode timer (if mtime >= stimecmp, which is stored in mtimecmp)
    // Already handled by MTIP check above (mtimecmp is used for both)
    if mtime >= cmp {
        mip |= 1 << 5; // STIP
    } else {
        mip &= !(1 << 5);
    }

    mip
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::riscv_mode;
    use std::mem;

    fn make_state(pc: u64) -> HartState {
        let mut s: HartState = unsafe { std::mem::zeroed() };
        s.mode = riscv_mode::M;
        s.mtvec = 0x8000_0100;
        s.pc = pc;
        s
    }

    fn write_u32_le(buf: &mut [u8], offset: usize, val: u32) {
        buf[offset] = val as u8;
        buf[offset + 1] = (val >> 8) as u8;
        buf[offset + 2] = (val >> 16) as u8;
        buf[offset + 3] = (val >> 24) as u8;
    }

    /// Build default zero-length CLINT / PMP / device arrays and call ``run_batch``.
    unsafe fn run_batch_defaults(
        states: *mut HartState, num: u32,
        ram: *mut u8, ram_sz: u64, ram_base: u64,
        shadow_base: u64, shadow_size: u64,
        max_instrs: u64, result: *mut BatchResult,
    ) {
        let dev_bases: [u64; 0] = [];
        let dev_ends: [u64; 0] = [];
        let pmp_cfg: [u8; 0] = [];
        let pmp_addr: [u64; 0] = [];
        let mtimecmp: [u64; 1] = [u64::MAX];
        let msip: [u8; 1] = [0];
        run_batch(
            states, num, ram, ram_sz, ram_base,
            shadow_base, shadow_size, max_instrs, result,
            pmp_cfg.as_ptr(), pmp_addr.as_ptr(), 0, 0,
            0, mtimecmp.as_ptr(), msip.as_ptr(),
            dev_bases.as_ptr(), dev_ends.as_ptr(), 0,
        );
    }

    #[test]
    fn run_batch_addi_sequence() {
        let mut ram = vec![0u8; 64];
        write_u32_le(&mut ram, 0, 0x00A0_0293);
        write_u32_le(&mut ram, 4, 0x0052_8293);
        write_u32_le(&mut ram, 8, 0xFFD2_8293);
        write_u32_le(&mut ram, 12, 0x0FF0_000F);

        let mut state = make_state(0);
        let mut result: BatchResult = unsafe { mem::zeroed() };
        unsafe {
            run_batch_defaults(
                &mut state as *mut HartState, 1,
                ram.as_mut_ptr(), ram.len() as u64, 0,
                0, 0, 3,
                &mut result as *mut BatchResult,
            );
        }

        assert_eq!(result.exit_reason, exit_reason::NORMAL);
        assert_eq!(result.total_instrs, 3);
        assert_eq!(state.gprs[5], 12);
        assert_eq!(state.pc, 12);
    }

    #[test]
    fn run_batch_branch_loop() {
        let mut ram = vec![0u8; 64];
        write_u32_le(&mut ram, 0, 0x0030_0293);
        write_u32_le(&mut ram, 4, 0xFFF2_8293);
        write_u32_le(&mut ram, 8, 0xFE02_9EE3);
        write_u32_le(&mut ram, 12, 0x0010_0313);

        let mut state = make_state(0);
        let mut result: BatchResult = unsafe { mem::zeroed() };
        unsafe {
            run_batch_defaults(
                &mut state as *mut HartState, 1,
                ram.as_mut_ptr(), ram.len() as u64, 0,
                0, 0, 8,
                &mut result as *mut BatchResult,
            );
        }

        assert_eq!(result.exit_reason, exit_reason::NORMAL);
        assert_eq!(result.total_instrs, 8);
        assert_eq!(state.gprs[5], 0);
        assert_eq!(state.gprs[6], 1);
        assert_eq!(state.pc, 16);
    }

    #[test]
    fn run_batch_max_instrs_limit() {
        let mut ram = vec![0u8; 128];
        for i in 0..10 {
            write_u32_le(&mut ram, (i * 4) as usize, 0x0012_8293);
        }

        let mut state = make_state(0);
        let mut result: BatchResult = unsafe { mem::zeroed() };
        unsafe {
            run_batch_defaults(
                &mut state as *mut HartState, 1,
                ram.as_mut_ptr(), ram.len() as u64, 0,
                0, 0, 5,
                &mut result as *mut BatchResult,
            );
        }

        assert_eq!(result.exit_reason, exit_reason::NORMAL);
        assert_eq!(result.total_instrs, 5);
        assert_eq!(state.gprs[5], 5);
    }

    #[test]
    fn run_batch_load_store() {
        let mut ram = vec![0u8; 256];
        ram[0x40] = 0x42;

        write_u32_le(&mut ram, 0, 0x00050283); // lb x5, 0(x10)
        write_u32_le(&mut ram, 4, 0x00550223); // sb x5, 4(x10)
        write_u32_le(&mut ram, 8, 0x0FF0_000F); // FENCE nop

        let mut state = make_state(0);
        state.gprs[10] = 0x40;

        let mut result: BatchResult = unsafe { mem::zeroed() };
        unsafe {
            run_batch_defaults(
                &mut state as *mut HartState, 1,
                ram.as_mut_ptr(), ram.len() as u64, 0,
                0, 0, 3,
                &mut result as *mut BatchResult,
            );
        }

        assert_eq!(result.exit_reason, exit_reason::NORMAL);
        assert_eq!(state.gprs[5], 0x42);
        assert_eq!(ram[0x44], 0x42);
    }

    #[test]
    fn fetch_out_of_bounds_traps() {
        let ram = vec![0u8; 64];
        let mut state = make_state(ram.len() as u64 + 100);
        state.mtvec = 0x8000_0100;
        let mut result: BatchResult = unsafe { mem::zeroed() };
        unsafe {
            run_batch_defaults(
                &mut state as *mut HartState, 1,
                ram.as_ptr() as *mut u8, ram.len() as u64, 0,
                0, 0, 10,
                &mut result as *mut BatchResult,
            );
        }

        assert_eq!(result.exit_reason, exit_reason::TRAP);
    }

    #[test]
    fn run_batch_exits_on_ecall() {
        let mut ram = vec![0u8; 64];
        write_u32_le(&mut ram, 0, 0x0000_0073); // ECALL

        let mut state = make_state(0);
        let mut result: BatchResult = unsafe { mem::zeroed() };
        unsafe {
            run_batch_defaults(
                &mut state as *mut HartState, 1,
                ram.as_mut_ptr(), ram.len() as u64, 0,
                0, 0, 10,
                &mut result as *mut BatchResult,
            );
        }

        assert_eq!(result.exit_reason, exit_reason::SYS_EXIT);
        assert_eq!(result.exit_instr, 0x0000_0073);
    }

    #[test]
    fn run_batch_mret() {
        let mut ram = vec![0u8; 64];
        write_u32_le(&mut ram, 0, 0x3020_0073); // MRET

        let mut state = make_state(0);
        state.mode = riscv_mode::S;
        state.mstatus = (1 << 7) | (1 << 11) | (1 << 3); // MPIE | MPP=S | MIE
        state.mepc = 0x8000_1000;

        let mut result: BatchResult = unsafe { mem::zeroed() };
        unsafe {
            run_batch_defaults(
                &mut state as *mut HartState, 1,
                ram.as_mut_ptr(), ram.len() as u64, 0,
                0, 0, 1,
                &mut result as *mut BatchResult,
            );
        }

        assert_eq!(result.total_instrs, 1);
        assert_eq!(state.mode, riscv_mode::S);
        assert_eq!(state.pc, 0x8000_1000);
    }

    #[test]
    fn run_batch_compressed_addi() {
        let mut ram = vec![0u8; 64];
        write_u32_le(&mut ram, 0, 0x0000_0515); // c.addi x10, 5
        write_u32_le(&mut ram, 4, 0x0FF0_000F); // FENCE

        let mut state = make_state(0);
        let mut result: BatchResult = unsafe { mem::zeroed() };
        unsafe {
            run_batch_defaults(
                &mut state as *mut HartState, 1,
                ram.as_mut_ptr(), ram.len() as u64, 0,
                0, 0, 2,
                &mut result as *mut BatchResult,
            );
        }

        assert_eq!(result.total_instrs, 2);
        assert_eq!(state.gprs[10], 5);
    }

    #[test]
    fn trap_delegates_to_smode() {
        // Verify delegation: in S-mode with medeleg bit set,
        // deliver_trap redirects to stvec, not mtvec.
        let mut state = make_state(0);
        state.mode = riscv_mode::S;
        state.mtvec = 0x8000_0100;
        state.stvec = 0x8000_0400;
        state.medeleg = 1 << exc_code::ILL_INSTR;
        state.mstatus = 1 << 1; // SIE=1
        state.pc = 0x1000;

        let mut result: BatchResult = unsafe { mem::zeroed() };
        deliver_trap(
            &mut state,
            mcause_val(exc_code::ILL_INSTR, false),
            0xDEAD,
            &mut result,
        );

        assert_eq!(state.mode, riscv_mode::S, "delegated trap stays in S-mode");
        assert_eq!(state.pc, 0x8000_0400, "PC jumps to stvec, not mtvec");
        assert_eq!(state.sepc, 0x1000, "sepc saves faulting PC");
        assert_eq!(state.scause & 0x7FFF_FFFF_FFFF_FFFF, exc_code::ILL_INSTR);
        assert_eq!(state.stval, 0xDEAD);
    }
}
