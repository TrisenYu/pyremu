//! Instruction handlers for Phases B–D: Load/Store, System, AMO, Compressed.
//!
//! Each handler receives ``&mut HartState``, decoded fields, and memory access
//! context; it returns the PC advance (0, 2, or 4) or ``EXIT_SENTINEL`` to
//! signal that Python must take over.

use crate::csr;
use crate::decode::{DecodedFields, CompressedFields, decode_compressed};
use crate::state::{HartState, BatchResult, exit_reason, riscv_mode, TlbEntry};
use crate::trap::{deliver_trap, deliver_illegal_instruction, exc_code, mcause_val};
use crate::translate::{WalkCtx, translate_va, TranslateFault};

// Re-export from csr.rs
pub use crate::csr::EXIT_SENTINEL;

// ============================================================
//  GPR read/write helpers
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

/// Sign-extend 32-bit -> 64-bit.
#[inline]
fn sext32(val: u64) -> u64 {
    ((val as i32) as i64) as u64
}

/// Sign-extend from *bits* to unsigned 64-bit.
#[inline]
fn sext(val: u64, bits: u32) -> u64 {
    let half = 1u64 << (bits - 1);
    (val & (half - 1)).wrapping_sub(val & half) & u64::MAX
}

// ============================================================
//  Memory read/write helpers
// ============================================================

/// Read *size* bytes from physical address *pa* in RAM. Little-endian.
#[inline]
fn ram_read(ctx: &WalkCtx, pa: u64, size: u8) -> u64 {
    let off = super::mem::ram_offset_inline(pa, size as u32, ctx.ram_base, ctx.ram_size, ctx.shadow_base, ctx.shadow_size);
    if off.is_none() { return 0; }
    let ptr = ctx.ram as *mut u8; let ptr = unsafe { ptr.add(off.unwrap() as usize) };
    match size {
        1 => unsafe { *ptr as u64 },
        2 => {
            let b0 = unsafe { *ptr } as u64;
            let b1 = unsafe { *ptr.add(1) } as u64;
            b0 | (b1 << 8)
        }
        4 => {
            let b0 = unsafe { *ptr } as u64;
            let b1 = unsafe { *ptr.add(1) } as u64;
            let b2 = unsafe { *ptr.add(2) } as u64;
            let b3 = unsafe { *ptr.add(3) } as u64;
            b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
        }
        8 => {
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
        _ => 0,
    }
}

/// Write *size* bytes of *val* to physical address *pa* in RAM. Little-endian.
#[inline]
fn ram_write(ctx: &WalkCtx, pa: u64, val: u64, size: u8) -> bool {
    let off = super::mem::ram_offset_inline(pa, size as u32, ctx.ram_base, ctx.ram_size, ctx.shadow_base, ctx.shadow_size);
    if off.is_none() { return false; }
    let ptr = ctx.ram as *mut u8; let ptr = unsafe { ptr.add(off.unwrap() as usize) };
    match size {
        1 => unsafe { *ptr = val as u8; }
        2 => {
            unsafe { *ptr = val as u8; }
            unsafe { *ptr.add(1) = (val >> 8) as u8; }
        }
        4 => {
            unsafe { *ptr = val as u8; }
            unsafe { *ptr.add(1) = (val >> 8) as u8; }
            unsafe { *ptr.add(2) = (val >> 16) as u8; }
            unsafe { *ptr.add(3) = (val >> 24) as u8; }
        }
        8 => {
            unsafe { *ptr = val as u8; }
            unsafe { *ptr.add(1) = (val >> 8) as u8; }
            unsafe { *ptr.add(2) = (val >> 16) as u8; }
            unsafe { *ptr.add(3) = (val >> 24) as u8; }
            unsafe { *ptr.add(4) = (val >> 32) as u8; }
            unsafe { *ptr.add(5) = (val >> 40) as u8; }
            unsafe { *ptr.add(6) = (val >> 48) as u8; }
            unsafe { *ptr.add(7) = (val >> 56) as u8; }
        }
        _ => return false,
    }
    true
}

/// Check if an address falls within a device MMIO range.
/// Returns true if the address should be handled by Python.
#[inline]
fn is_device_addr(
    pa: u64,
    dev_bases: *const u64,
    dev_ends: *const u64,
    num_devices: u8,
) -> bool {
    for i in 0..num_devices as usize {
        let base = unsafe { *dev_bases.add(i) };
        let end = unsafe { *dev_ends.add(i) };
        if pa >= base && pa < end {
            return true;
        }
    }
    false
}

// ============================================================
//  PMP check stub
// ============================================================

/// Check PMP for a physical address access.
/// Returns true if access is allowed.
#[inline]
fn pmp_ok(
    state: &HartState,
    pa: u64,
    size: u32,
    is_write: bool,
    is_execute: bool,
    pmp_cfg: *const u8,
    pmp_addr: *const u64,
    pmp_num: u8,
) -> bool {
    // If no PMP entries or M-mode with MPRV=0, always allow
    if pmp_num == 0 {
        return state.mode == riscv_mode::M;
    }
    // Quick check: M-mode with MPRV=0 bypasses PMP
    if state.mode == riscv_mode::M {
        let mprv = (state.mstatus >> 17) & 1;
        if mprv == 0 {
            return true;
        }
    }
    // Delegate to the full PMP check function
    crate::pmp::pmp_check(
        pmp_cfg, pmp_addr, pmp_num,
        pa, size,
        if is_write { 1 } else { 0 },
        if is_execute { 1 } else { 0 },
        state.mode, state.mstatus, state.pmpsplit, state.mdid,
    ) != 0
}

// ============================================================
//  Load handlers
// ============================================================

pub fn handle_load(
    state: &mut HartState,
    f: &DecodedFields,
    instr: u32,
    result: &mut BatchResult,
    ctx: &WalkCtx,
    pmp_cfg: *const u8,
    pmp_addr: *const u64,
    pmp_num: u8,
    dev_bases: *const u64,
    dev_ends: *const u64,
    num_devices: u8,
) -> u64 {
    let base = read_gpr(state, f.rs1);
    let va = base.wrapping_add(f.imm12_se);
    let (size, signed) = match f.func3 {
        0b000 => (1, true),   // LB
        0b001 => (2, true),   // LH
        0b010 => (4, true),   // LW
        0b011 => (8, false),  // LD
        0b100 => (1, false),  // LBU
        0b101 => (2, false),  // LHU
        0b110 => (4, false),  // LWU
        _ => {
            deliver_illegal_instruction(state, instr as u64, result);
            return 0;
        }
    };

    // Alignment check: misaligned loads are not supported
    if va & (size as u64 - 1) != 0 {
        deliver_trap(state, mcause_val(
            if size == 1 { exc_code::LD_MISALIGNED } else { exc_code::LD_MISALIGNED },
            false), va, result);
        return 0;
    }

    // Translate VA → PA
    let tr = match translate_va(state, ctx, va, false, false) {
        Ok(t) => t,
        Err(TranslateFault::PageFault(cause)) => {
            deliver_trap(state, mcause_val(cause, false), va, result);
            return 0;
        }
        Err(TranslateFault::AccessFault) => {
            deliver_trap(state, mcause_val(exc_code::LD_ACCESS_FAULT, false), va, result);
            return 0;
        }
    };

    // PMP check
    if !pmp_ok(state, tr.pa, size as u32, false, false, pmp_cfg, pmp_addr, pmp_num) {
        deliver_trap(state, mcause_val(exc_code::LD_ACCESS_FAULT, false), va, result);
        return 0;
    }

    // MMIO check
    if is_device_addr(tr.pa, dev_bases, dev_ends, num_devices) {
        result.exit_reason = exit_reason::SYS_EXIT;
        result.exit_instr = instr;
        return EXIT_SENTINEL;
    }

    // Read from RAM
    let val = ram_read(ctx, tr.pa, size);

    // Sign-extend if needed
    let result_val = if signed {
        match size {
            1 => sext(val, 8),
            2 => sext(val, 16),
            4 => sext(val, 32),
            _ => val,
        }
    } else {
        // Zero-extend for unsigned loads (already zero-extended by ram_read)
        val
    };

    write_gpr(state, f.rd, result_val);
    4
}

// ============================================================
//  Store handlers
// ============================================================

pub fn handle_store(
    state: &mut HartState,
    f: &DecodedFields,
    instr: u32,
    result: &mut BatchResult,
    ctx: &WalkCtx,
    pmp_cfg: *const u8,
    pmp_addr: *const u64,
    pmp_num: u8,
    dev_bases: *const u64,
    dev_ends: *const u64,
    num_devices: u8,
) -> u64 {
    let base = read_gpr(state, f.rs1);
    let va = base.wrapping_add(f.imm_s);
    let size: u8 = match f.func3 {
        0b000 => 1,  // SB
        0b001 => 2,  // SH
        0b010 => 4,  // SW
        0b011 => 8,  // SD
        _ => {
            deliver_illegal_instruction(state, instr as u64, result);
            return 0;
        }
    };

    // Alignment check
    if va & (size as u64 - 1) != 0 {
        deliver_trap(state, mcause_val(
            exc_code::ST_MISALIGNED, false), va, result);
        return 0;
    }

    // Translate VA → PA
    let tr = match translate_va(state, ctx, va, true, false) {
        Ok(t) => t,
        Err(TranslateFault::PageFault(cause)) => {
            deliver_trap(state, mcause_val(cause, false), va, result);
            return 0;
        }
        Err(TranslateFault::AccessFault) => {
            deliver_trap(state, mcause_val(exc_code::ST_ACCESS_FAULT, false), va, result);
            return 0;
        }
    };

    // PMP check
    if !pmp_ok(state, tr.pa, size as u32, true, false, pmp_cfg, pmp_addr, pmp_num) {
        deliver_trap(state, mcause_val(exc_code::ST_ACCESS_FAULT, false), va, result);
        return 0;
    }

    // MMIO check
    if is_device_addr(tr.pa, dev_bases, dev_ends, num_devices) {
        result.exit_reason = exit_reason::SYS_EXIT;
        result.exit_instr = instr;
        return EXIT_SENTINEL;
    }

    let val = read_gpr(state, f.rs2);
    ram_write(ctx, tr.pa, val, size);

    4
}

// ============================================================
//  System instruction handler
// ============================================================

pub fn handle_system(
    state: &mut HartState,
    f: &DecodedFields,
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
    match f.func3 {
        0b000 => {
            // ECALL / EBREAK / MRET / SRET / WFI / SFENCE.VMA
            match f.func12 {
                0 => {
                    // ECALL
                    result.exit_reason = exit_reason::SYS_EXIT;
                    result.exit_instr = instr;
                    return EXIT_SENTINEL;
                }
                1 => {
                    // EBREAK
                    result.exit_reason = exit_reason::EBREAK;
                    result.exit_instr = instr;
                    return EXIT_SENTINEL;
                }
                0x302 => {
                    // MRET
                    let mpp = (state.mstatus >> 11) & 0x3;
                    let mpie = (state.mstatus >> 7) & 1;
                    // Restore mode
                    state.mode = match mpp {
                        0 => riscv_mode::U,
                        1 => riscv_mode::S,
                        3 => riscv_mode::M,
                        _ => riscv_mode::M,
                    };
                    // MIE ← MPIE, MPIE ← 1
                    state.mstatus &= !(1 << 3); // clear MIE
                    if mpie != 0 { state.mstatus |= 1 << 3; } // MIE = MPIE
                    state.mstatus |= 1 << 7; // MPIE = 1
                    // MPP ← U (0)
                    state.mstatus &= !(0b11 << 11);
                    // PC ← mepc
                    state.pc = state.mepc;
                    state.waiting = 0;
                    return 0;
                }
                0x102 => {
                    // SRET
                    // Check privilege
                    if state.mode < riscv_mode::S {
                        deliver_illegal_instruction(state, instr as u64, result);
                        return 0;
                    }
                    let spp = (state.mstatus >> 8) & 1;
                    let spie = (state.mstatus >> 5) & 1;
                    // Restore mode
                    state.mode = if spp == 0 { riscv_mode::U } else { riscv_mode::S };
                    // SIE ← SPIE, SPIE ← 1
                    state.mstatus &= !(1 << 1); // clear SIE
                    if spie != 0 { state.mstatus |= 1 << 1; } // SIE = SPIE
                    state.mstatus |= 1 << 5; // SPIE = 1
                    // SPP ← U (0)
                    state.mstatus &= !(1 << 8);
                    // PC ← sepc
                    state.pc = state.sepc;
                    state.waiting = 0;
                    return 0;
                }
                0x105 => {
                    // WFI
                    let tw = (state.mstatus >> 21) & 1;
                    if tw != 0 && state.mode != riscv_mode::M {
                        // TW=1 in S/U mode → illegal instruction
                        deliver_illegal_instruction(state, instr as u64, result);
                        return 0;
                    }
                    // Check if any pending+enabled interrupt
                    let pending = state.mip & state.mie;
                    if pending != 0 {
                        // Interrupt pending → NOP
                        return 4;
                    }
                    state.waiting = 1;
                    result.exit_reason = exit_reason::WFI_WAIT;
                    result.exit_pc = state.pc + 4; // PC after WFI
                    return EXIT_SENTINEL;
                }
                0x120 => {
                    // SFENCE.VMA
                    // Flush all TLBs (both itlb and dtlb)
                    for e in state.itlb.iter_mut() { e.valid = 0; }
                    for e in state.dtlb.iter_mut() { e.valid = 0; }
                    return 4;
                }
                _ => {
                    deliver_illegal_instruction(state, instr as u64, result);
                    return 0;
                }
            }
        }
        0b001 | 0b010 | 0b011 | 0b101 | 0b110 | 0b111 => {
            // CSR instructions: funct3 = CSRRW(001), CSRRS(010), CSRRC(011),
            //                       CSRRWI(101), CSRRSI(110), CSRRCI(111)
            csr::handle_csr(state, f.rd, f.rs1, f.func12, f.func3, instr, result, hart_id, mtime)
        }
        _ => {
            deliver_illegal_instruction(state, instr as u64, result);
            0
        }
    }
}

// All trap codes now imported directly from crate::trap::exc_code

// ============================================================
//  AMO handler (Phase D)
// ============================================================

pub fn handle_amo(
    state: &mut HartState,
    f: &DecodedFields,
    instr: u32,
    result: &mut BatchResult,
    ctx: &WalkCtx,
    pmp_cfg: *const u8,
    pmp_addr: *const u64,
    pmp_num: u8,
    dev_bases: *const u64,
    dev_ends: *const u64,
    num_devices: u8,
) -> u64 {
    let _aq = (instr >> 26) & 1;
    let _rl = (instr >> 25) & 1;
    let funct5 = f.func7 >> 2; // bits [31:27]
    let width: u8 = match f.func3 {
        0b010 => 4,  // .W
        0b011 => 8,  // .D
        _ => {
            deliver_illegal_instruction(state, instr as u64, result);
            return 0;
        }
    };

    let base = read_gpr(state, f.rs1);
    let va = base; // AMO: rs1 is the address, rs2 is the operand

    // Alignment check
    if va & (width as u64 - 1) != 0 {
        deliver_trap(state, mcause_val(exc_code::LD_MISALIGNED, false), va, result);
        return 0;
    }

    // Translate VA → PA
    let tr = match translate_va(state, ctx, va, true, false) {
        Ok(t) => t,
        Err(TranslateFault::PageFault(cause)) => {
            deliver_trap(state, mcause_val(cause, false), va, result);
            return 0;
        }
        Err(_) => {
            deliver_trap(state, mcause_val(exc_code::LD_ACCESS_FAULT, false), va, result);
            return 0;
        }
    };

    // PMP check
    if !pmp_ok(state, tr.pa, width as u32, true, false, pmp_cfg, pmp_addr, pmp_num) {
        deliver_trap(state, mcause_val(exc_code::LD_ACCESS_FAULT, false), va, result);
        return 0;
    }

    // MMIO check - AMO to MMIO exits to Python
    if is_device_addr(tr.pa, dev_bases, dev_ends, num_devices) {
        result.exit_reason = exit_reason::SYS_EXIT;
        result.exit_instr = instr;
        return EXIT_SENTINEL;
    }

    let rd = f.rd;
    let rs2_val = read_gpr(state, f.rs2);

    match funct5 {
        0b00010 => {
            // LR.W / LR.D
            let loaded = ram_read(ctx, tr.pa, width);
            state.reservation_valid = 1;
            state.reservation_addr = tr.pa;
            write_gpr(state, rd, if width == 4 { sext32(loaded) } else { loaded });
            4
        }
        0b00011 => {
            // SC.W / SC.D
            if state.reservation_valid == 0 || state.reservation_addr != tr.pa {
                write_gpr(state, rd, 1);
            } else {
                ram_write(ctx, tr.pa, rs2_val, width);
                state.reservation_valid = 0;
                write_gpr(state, rd, 0);
            }
            4
        }
        0b00001 | 0b00000 | 0b00100 | 0b01100 | 0b01000 | 0b10000
        | 0b10100 | 0b11000 | 0b11100 => {
            // AMOSWAP/AMOADD/AMOXOR/AMOAND/AMOOR/AMOMIN/AMOMAX/AMOMINU/AMOMAXU
            let loaded = ram_read(ctx, tr.pa, width);
            let signed_ld = if width == 4 { sext32(loaded) as i64 } else { loaded as i64 };
            let signed_rs2 = if width == 4 { sext32(rs2_val) as i64 } else { rs2_val as i64 };

            let result_val: u64 = match funct5 {
                0b00001 => rs2_val,                               // AMOSWAP
                0b00000 => loaded.wrapping_add(rs2_val),         // AMOADD
                0b00100 => loaded ^ rs2_val,                     // AMOXOR
                0b01100 => loaded & rs2_val,                     // AMOAND
                0b01000 => loaded | rs2_val,                     // AMOOR
                0b10000 => (signed_ld.min(signed_rs2)) as u64,   // AMOMIN
                0b10100 => (signed_ld.max(signed_rs2)) as u64,   // AMOMAX
                0b11000 => (loaded.min(rs2_val)),                 // AMOMINU
                0b11100 => (loaded.max(rs2_val)),                 // AMOMAXU
                _ => {
                    deliver_illegal_instruction(state, instr as u64, result);
                    return 0;
                }
            };

            ram_write(ctx, tr.pa, result_val, width);
            write_gpr(state, rd, if width == 4 { sext32(loaded) } else { loaded });
            4
        }
        _ => {
            deliver_illegal_instruction(state, instr as u64, result);
            0
        }
    }
}

// ============================================================
//  Compressed instruction handlers (Phase D)
// ============================================================

pub fn handle_compressed(
    state: &mut HartState,
    half: u16,
    instr_word: u32,
    result: &mut BatchResult,
    ctx: &WalkCtx,
    pmp_cfg: *const u8,
    pmp_addr: *const u64,
    pmp_num: u8,
    dev_bases: *const u64,
    dev_ends: *const u64,
    num_devices: u8,
) -> u64 {
    let cf = decode_compressed(half);

    match cf.quadrant {
        0 => handle_c0(state, &cf, half, instr_word, result, ctx, pmp_cfg, pmp_addr, pmp_num, dev_bases, dev_ends, num_devices),
        1 => handle_c1(state, &cf, half, instr_word, result, ctx),
        2 => handle_c2(state, &cf, half, instr_word, result, ctx, pmp_cfg, pmp_addr, pmp_num, dev_bases, dev_ends, num_devices),
        _ => {
            deliver_illegal_instruction(state, half as u64, result);
            0
        }
    }
}

/// C0: C.ADDI4SPN, C.LW, C.LD, C.SW, C.SD
fn handle_c0(
    state: &mut HartState,
    cf: &CompressedFields,
    _half: u16,
    instr_word: u32,
    result: &mut BatchResult,
    ctx: &WalkCtx,
    pmp_cfg: *const u8,
    pmp_addr: *const u64,
    pmp_num: u8,
    dev_bases: *const u64,
    dev_ends: *const u64,
    num_devices: u8,
) -> u64 {
    match cf.funct3 {
        0b000 => {
            // C.ADDI4SPN: rd' = sp + nzuimm
            if cf.imm == 0 {
                deliver_illegal_instruction(state, instr_word as u64, result);
                return 0;
            }
            let sp = state.gprs[2]; // x2 = sp
            write_gpr(state, cf.rd, sp.wrapping_add(cf.imm));
            2
        }
        0b010 => {
            // C.LW: rd' = mem[rs1' + uimm]
            let addr = read_gpr(state, cf.rs1p).wrapping_add(cf.imm);
            let val = load_mem_compressed(state, ctx, addr, 4, false, instr_word, result,
                pmp_cfg, pmp_addr, pmp_num, dev_bases, dev_ends, num_devices);
            if val == EXIT_SENTINEL { return EXIT_SENTINEL; }
            write_gpr(state, cf.rd, sext32(val));
            2
        }
        0b011 => {
            // C.LD: rd' = mem[rs1' + uimm]
            let addr = read_gpr(state, cf.rs1p).wrapping_add(cf.imm);
            let val = load_mem_compressed(state, ctx, addr, 8, false, instr_word, result,
                pmp_cfg, pmp_addr, pmp_num, dev_bases, dev_ends, num_devices);
            if val == EXIT_SENTINEL { return EXIT_SENTINEL; }
            write_gpr(state, cf.rd, val);
            2
        }
        0b110 => {
            // C.SW: mem[rs1' + uimm] = rs2'
            let addr = read_gpr(state, cf.rs1p).wrapping_add(cf.imm);
            let rs2_val = read_gpr(state, cf.rdp) & 0xFFFF_FFFF;
            store_mem_compressed(state, ctx, addr, rs2_val, 4, instr_word, result,
                pmp_cfg, pmp_addr, pmp_num, dev_bases, dev_ends, num_devices);
            2
        }
        0b111 => {
            // C.SD: mem[rs1' + uimm] = rs2'
            let addr = read_gpr(state, cf.rs1p).wrapping_add(cf.imm);
            let rs2_val = read_gpr(state, cf.rdp);
            store_mem_compressed(state, ctx, addr, rs2_val, 8, instr_word, result,
                pmp_cfg, pmp_addr, pmp_num, dev_bases, dev_ends, num_devices);
            2
        }
        _ => {
            deliver_illegal_instruction(state, instr_word as u64, result);
            0
        }
    }
}

/// C1: C.ADDI, C.ADDIW, C.LI, C.LUI, C.ADDI16SP, C.SRLI/C.SRAI/C.ANDI,
///      C.SUB/C.XOR/C.OR/C.AND, C.SUBW/C.ADDW, C.J, C.BEQZ/C.BNEZ
fn handle_c1(
    state: &mut HartState,
    cf: &CompressedFields,
    _half: u16,
    instr_word: u32,
    result: &mut BatchResult,
    _ctx: &WalkCtx,
) -> u64 {
    match cf.funct3 {
        0b000..=0b010 => {
            // C.ADDI / C.ADDIW / C.LI
            let imm = cf.imm;
            if cf.funct3 == 0b000 {
                // C.ADDI: rd += imm
                let v = read_gpr(state, cf.rd).wrapping_add(imm);
                write_gpr(state, cf.rd, v);
            } else if cf.funct3 == 0b001 {
                // C.ADDIW: rd = sext32(rd + imm)
                let v = (read_gpr(state, cf.rd).wrapping_add(imm)) & 0xFFFF_FFFF;
                write_gpr(state, cf.rd, sext32(v));
            } else {
                // C.LI: rd = imm
                write_gpr(state, cf.rd, imm);
            }
            2
        }
        0b011 => {
            if cf.rd == 2 {
                // C.ADDI16SP: sp += nzimm
                if cf.imm == 0 {
                    return EXIT_SENTINEL; // illegal
                }
                let sp = state.gprs[2];
                state.gprs[2] = sp.wrapping_add(cf.imm);
            } else {
                // C.LUI: rd = nzimm << 12
                if cf.imm == 0 {
                    return EXIT_SENTINEL;
                }
                write_gpr(state, cf.rd, (cf.imm << 12) & 0xFFFF_FFFF_FFFF_FFFF);
            }
            2
        }
        0b100 => {
            // C1 ALU: SRLI / SRAI / ANDI / SUB/XOR/OR/AND / SUBW/ADDW
            let rd_rs1 = cf.rs1p; // creg-mapped rd/rs1
            let v1 = read_gpr(state, rd_rs1);
            let sf = cf.sf;
            let r: u64 = match sf {
                0b00 => {
                    let shamt = ((cf.bit12 as u64) << 5) | (cf.rs2 as u64 & 0x1F);
                    v1 >> shamt
                }
                0b01 => {
                    let shamt = ((cf.bit12 as u64) << 5) | (cf.rs2 as u64 & 0x1F);
                    ((v1 as i64) >> shamt) as u64
                }
                0b10 => {
                    let imm = sext(((cf.bit12 as u64) << 5) | (cf.rs2 as u64 & 0x1F), 6);
                    v1 & imm
                }
                0b11 => {
                    let rs2 = cf.rdp; // creg-mapped rs2
                    let v2 = read_gpr(state, rs2);
                    if cf.bit12 == 0 {
                        match cf.bit65 {
                            0b00 => v1.wrapping_sub(v2),
                            0b01 => v1 ^ v2,
                            0b10 => v1 | v2,
                            _    => v1 & v2,
                        }
                    } else {
                        match cf.bit65 {
                            0b00 => sext32((v1.wrapping_sub(v2)) & 0xFFFF_FFFF),
                            0b01 => sext32((v1.wrapping_add(v2)) & 0xFFFF_FFFF),
                            _ => {
                                deliver_illegal_instruction(state, instr_word as u64, result);
                                return 0;
                            }
                        }
                    }
                }
                _ => {
                    deliver_illegal_instruction(state, instr_word as u64, result);
                    return 0;
                }
            };
            write_gpr(state, rd_rs1, r);
            2
        }
        0b101 => {
            // C.J: pc += offset
            state.pc = state.pc.wrapping_add(cf.imm);
            0
        }
        0b110 | 0b111 => {
            // C.BEQZ / C.BNEZ
            let rs1 = cf.rs1p; // creg-mapped from bits[9:7]
            let v = read_gpr(state, rs1);
            let taken = if cf.funct3 == 0b110 { v == 0 } else { v != 0 };
            if taken {
                state.pc = state.pc.wrapping_add(cf.imm);
                0
            } else {
                2
            }
        }
        _ => {
            deliver_illegal_instruction(state, instr_word as u64, result);
            0
        }
    }
}

/// C2: C.SLLI, C.LWSP, C.LDSP, C.JR/C.MV/C.EBREAK/C.JALR/C.ADD, C.SWSP, C.SDSP
fn handle_c2(
    state: &mut HartState,
    cf: &CompressedFields,
    _half: u16,
    instr_word: u32,
    result: &mut BatchResult,
    ctx: &WalkCtx,
    pmp_cfg: *const u8,
    pmp_addr: *const u64,
    pmp_num: u8,
    dev_bases: *const u64,
    dev_ends: *const u64,
    num_devices: u8,
) -> u64 {
    match cf.funct3 {
        0b000 => {
            // C.SLLI: rd = rd << shamt
            let v = read_gpr(state, cf.rd) << cf.imm;
            write_gpr(state, cf.rd, v);
            2
        }
        0b010 => {
            // C.LWSP: rd = mem[sp + uimm]
            let addr = state.gprs[2].wrapping_add(cf.imm);
            let val = load_mem_compressed(state, ctx, addr, 4, false, instr_word, result,
                pmp_cfg, pmp_addr, pmp_num, dev_bases, dev_ends, num_devices);
            if val == EXIT_SENTINEL { return EXIT_SENTINEL; }
            write_gpr(state, cf.rd, sext32(val));
            2
        }
        0b011 => {
            // C.LDSP: rd = mem[sp + uimm]
            let addr = state.gprs[2].wrapping_add(cf.imm);
            let val = load_mem_compressed(state, ctx, addr, 8, false, instr_word, result,
                pmp_cfg, pmp_addr, pmp_num, dev_bases, dev_ends, num_devices);
            if val == EXIT_SENTINEL { return EXIT_SENTINEL; }
            write_gpr(state, cf.rd, val);
            2
        }
        0b100 => {
            let rd_rs1 = cf.rd;
            let rs2 = cf.rs2;
            if rd_rs1 == 0 && rs2 == 0 {
                // C.EBREAK
                result.exit_reason = exit_reason::EBREAK;
                result.exit_instr = instr_word;
                return EXIT_SENTINEL;
            } else if rs2 == 0 {
                let target = read_gpr(state, rd_rs1);
                if cf.bit12 != 0 {
                    // C.JALR: ra = pc + 2
                    state.gprs[1] = state.pc.wrapping_add(2);
                }
                state.pc = target & !1;
                0
            } else if cf.bit12 != 0 {
                // C.ADD: rd += rs2
                let result_val = read_gpr(state, rd_rs1).wrapping_add(read_gpr(state, rs2));
                write_gpr(state, rd_rs1, result_val);
                2
            } else {
                // C.MV: rd = rs2
                let v = read_gpr(state, rs2);
                write_gpr(state, rd_rs1, v);
                2
            }
        }
        0b110 => {
            // C.SWSP: mem[sp + uimm] = rs2
            let addr = state.gprs[2].wrapping_add(cf.imm2);
            let val = read_gpr(state, cf.rs2) & 0xFFFF_FFFF;
            store_mem_compressed(state, ctx, addr, val, 4, instr_word, result,
                pmp_cfg, pmp_addr, pmp_num, dev_bases, dev_ends, num_devices);
            2
        }
        0b111 => {
            // C.SDSP: mem[sp + uimm] = rs2
            let addr = state.gprs[2].wrapping_add(cf.imm2);
            let val = read_gpr(state, cf.rs2);
            store_mem_compressed(state, ctx, addr, val, 8, instr_word, result,
                pmp_cfg, pmp_addr, pmp_num, dev_bases, dev_ends, num_devices);
            2
        }
        _ => {
            deliver_illegal_instruction(state, instr_word as u64, result);
            0
        }
    }
}

// ============================================================
//  Compressed load/store helpers
// ============================================================

#[inline]
fn load_mem_compressed(
    state: &mut HartState,
    ctx: &WalkCtx,
    va: u64,
    size: u8,
    _signed: bool,
    instr_word: u32,
    result: &mut BatchResult,
    pmp_cfg: *const u8,
    pmp_addr: *const u64,
    pmp_num: u8,
    dev_bases: *const u64,
    dev_ends: *const u64,
    num_devices: u8,
) -> u64 {
    let tr = match translate_va(state, ctx, va, false, false) {
        Ok(t) => t,
        Err(TranslateFault::PageFault(cause)) => {
            deliver_trap(state, mcause_val(cause, false), va, result);
            return 0;
        }
        Err(_) => {
            deliver_trap(state, mcause_val(exc_code::LD_ACCESS_FAULT, false), va, result);
            return 0;
        }
    };

    if !pmp_ok(state, tr.pa, size as u32, false, false, pmp_cfg, pmp_addr, pmp_num) {
        deliver_trap(state, mcause_val(exc_code::LD_ACCESS_FAULT, false), va, result);
        return 0;
    }

    if is_device_addr(tr.pa, dev_bases, dev_ends, num_devices) {
        result.exit_reason = exit_reason::SYS_EXIT;
        result.exit_instr = instr_word;
        return EXIT_SENTINEL;
    }

    ram_read(ctx, tr.pa, size)
}

#[inline]
fn store_mem_compressed(
    state: &mut HartState,
    ctx: &WalkCtx,
    va: u64,
    val: u64,
    size: u8,
    instr_word: u32,
    result: &mut BatchResult,
    pmp_cfg: *const u8,
    pmp_addr: *const u64,
    pmp_num: u8,
    dev_bases: *const u64,
    dev_ends: *const u64,
    num_devices: u8,
) -> u64 {
    let tr = match translate_va(state, ctx, va, true, false) {
        Ok(t) => t,
        Err(TranslateFault::PageFault(cause)) => {
            deliver_trap(state, mcause_val(cause, false), va, result);
            return 0;
        }
        Err(_) => {
            deliver_trap(state, mcause_val(exc_code::ST_ACCESS_FAULT, false), va, result);
            return 0;
        }
    };

    if !pmp_ok(state, tr.pa, size as u32, true, false, pmp_cfg, pmp_addr, pmp_num) {
        deliver_trap(state, mcause_val(exc_code::ST_ACCESS_FAULT, false), va, result);
        return 0;
    }

    if is_device_addr(tr.pa, dev_bases, dev_ends, num_devices) {
        result.exit_reason = exit_reason::SYS_EXIT;
        result.exit_instr = instr_word;
        return EXIT_SENTINEL;
    }

    ram_write(ctx, tr.pa, val, size);
    0 // caller returns 2
}

// ============================================================
//  Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sext32_sign_extends() {
        assert_eq!(sext32(0x7FFFFFFF), 0x7FFFFFFF);
        assert_eq!(sext32(0x80000000), 0xFFFF_FFFF_8000_0000u64);
        assert_eq!(sext32(0xFFFFFFFF), 0xFFFF_FFFF_FFFF_FFFFu64);
    }

    #[test]
    fn sext_various_widths() {
        assert_eq!(sext(0xFF, 8), 0xFFFF_FFFF_FFFF_FFFFu64);
        assert_eq!(sext(0x7F, 8), 0x7F);
        assert_eq!(sext(0x8000, 16), 0xFFFF_FFFF_FFFF_8000u64);
        assert_eq!(sext(0x7FFF, 16), 0x7FFF);
    }
}
