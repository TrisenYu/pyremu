//! Pure-compute instruction handlers for Phase A.
//!
//! Each handler receives ``&mut HartState`` and ``&DecodedFields``,
//! executes the instruction, and returns the PC advance (0 or 4).
//!
//! Handlers that encounter an invalid encoding call ``deliver_illegal_instruction``
//! and return 0 (PC already redirected to ``mtvec``).

use crate::alu::{exec_alu_op, exec_op_imm, exec_op32, exec_op_imm32};
use crate::decode::DecodedFields;
use crate::state::BatchResult;
use crate::state::HartState;
use crate::trap::deliver_illegal_instruction;

// ============================================================
//  Helpers
// ============================================================

/// Write ``val`` to GPR ``rd``, skipping x0 (hardwired to 0).
#[inline]
fn write_gpr(state: &mut HartState, rd: u8, val: u64) {
    if rd != 0 {
        state.gprs[rd as usize] = val;
    }
}

/// Read GPR ``rs`` (x0 always returns 0).
#[inline]
fn read_gpr(state: &HartState, rs: u8) -> u64 {
    if rs == 0 { 0 } else { state.gprs[rs as usize] }
}

/// Check if the ALU result has a trap flag set; if so, deliver IllInstr.
#[inline]
fn check_alu_trap(
    state: &mut HartState,
    trap: u8,
    tval: u64,
    result: &mut BatchResult,
) -> bool {
    if trap != 0 {
        deliver_illegal_instruction(state, tval, result);
        true
    } else {
        false
    }
}

// ============================================================
//  R-type ALU: ADD / SUB / SLL / SLT / SLTU / XOR / SRL / SRA
//              OR / AND / MUL / MULH / MULHSU / MULHU /
//              DIV / DIVU / REM / REMU
// ============================================================

pub fn handle_alu(
    state: &mut HartState,
    f: &DecodedFields,
    instr: u32,
    result: &mut BatchResult,
) -> u64 {
    let v1 = read_gpr(state, f.rs1);
    let v2 = read_gpr(state, f.rs2);
    let r = exec_alu_op(f.func3, f.func7, v1, v2);
    if check_alu_trap(state, r.trap, instr as u64, result) {
        return 0;
    }
    write_gpr(state, f.rd, r.value);
    4
}

// ============================================================
//  I-type ALU: ADDI / SLLI / SLTI / SLTIU / XORI /
//              SRLI / SRAI / ORI / ANDI
// ============================================================

pub fn handle_op_imm(
    state: &mut HartState,
    f: &DecodedFields,
    instr: u32,
    result: &mut BatchResult,
) -> u64 {
    let v1 = read_gpr(state, f.rs1);
    let r = exec_op_imm(f.func3, f.func7, v1, f.imm12_se);
    if check_alu_trap(state, r.trap, instr as u64, result) {
        return 0;
    }
    write_gpr(state, f.rd, r.value);
    4
}

// ============================================================
//  RV64 32-bit word ALU: ADDW / SUBW / SLLW / SRLW / SRAW /
//                        MULW / DIVW / DIVUW / REMW / REMUW
// ============================================================

pub fn handle_op32(
    state: &mut HartState,
    f: &DecodedFields,
    instr: u32,
    result: &mut BatchResult,
) -> u64 {
    let v1 = read_gpr(state, f.rs1);
    let v2 = read_gpr(state, f.rs2);
    let r = exec_op32(f.func3, f.func7, v1, v2);
    if check_alu_trap(state, r.trap, instr as u64, result) {
        return 0;
    }
    write_gpr(state, f.rd, r.value);
    4
}

// ============================================================
//  RV64 32-bit immediate ALU: ADDIW / SLLIW / SRLIW / SRAIW
// ============================================================

pub fn handle_op_imm32(
    state: &mut HartState,
    f: &DecodedFields,
    instr: u32,
    result: &mut BatchResult,
) -> u64 {
    let v1 = read_gpr(state, f.rs1);
    let r = exec_op_imm32(f.func3, f.func7, v1, f.imm12_se);
    if check_alu_trap(state, r.trap, instr as u64, result) {
        return 0;
    }
    write_gpr(state, f.rd, r.value);
    4
}

// ============================================================
//  LUI: rd = imm20 << 12  (sign-extended from 32 bits)
// ============================================================

pub fn handle_lui(state: &mut HartState, f: &DecodedFields) -> u64 {
    // imm20_raw is bits[31:12] zero-extended to u64.
    // Shift left 12, then sign-extend from bit 31.
    let imm = (f.imm20_raw << 12) as i32 as i64 as u64;
    write_gpr(state, f.rd, imm);
    4
}

// ============================================================
//  AUIPC: rd = pc + (imm20 << 12)  (sign-extended from 32 bits)
// ============================================================

pub fn handle_auipc(state: &mut HartState, f: &DecodedFields) -> u64 {
    let imm = (f.imm20_raw << 12) as i32 as i64 as u64;
    write_gpr(state, f.rd, state.pc.wrapping_add(imm));
    4
}

// ============================================================
//  JAL: rd = pc + 4;  pc += imm_j
// ============================================================

pub fn handle_jal(state: &mut HartState, f: &DecodedFields) -> u64 {
    let next_pc = state.pc.wrapping_add(4);
    write_gpr(state, f.rd, next_pc);
    state.pc = state.pc.wrapping_add(f.imm_j);
    0
}

// ============================================================
//  JALR: rd = pc + 4;  pc = (rs1 + imm) & ~1
// ============================================================

pub fn handle_jalr(state: &mut HartState, f: &DecodedFields) -> u64 {
    let next_pc = state.pc.wrapping_add(4);
    let target = read_gpr(state, f.rs1).wrapping_add(f.imm12_se) & !1u64;
    write_gpr(state, f.rd, next_pc);
    state.pc = target;
    0
}

// ============================================================
//  Branch: BEQ / BNE / BLT / BGE / BLTU / BGEU
// ============================================================

pub fn handle_br(
    state: &mut HartState,
    f: &DecodedFields,
    instr: u32,
    result: &mut BatchResult,
) -> u64 {
    let v1 = read_gpr(state, f.rs1);
    let v2 = read_gpr(state, f.rs2);

    let taken = match f.func3 {
        0b000 => v1 == v2,                                             // BEQ
        0b001 => v1 != v2,                                             // BNE
        0b100 => (v1 as i64) < (v2 as i64),                            // BLT
        0b101 => (v1 as i64) >= (v2 as i64),                           // BGE
        0b110 => v1 < v2,                                              // BLTU
        0b111 => v1 >= v2,                                             // BGEU
        _ => {
            deliver_illegal_instruction(state, instr as u64, result);
            return 0;
        }
    };

    if taken {
        state.pc = state.pc.wrapping_add(f.imm_b);
        0
    } else {
        4
    }
}

// ============================================================
//  FENCE / FENCE.I: no-ops in single-hart in-order emulator
// ============================================================

pub fn handle_fence(_state: &mut HartState, f: &DecodedFields, instr: u32, result: &mut BatchResult) -> u64 {
    // func3 == 0b000: FENCE  (no-op)
    // func3 == 0b001: FENCE.I (no-op)
    if f.func3 > 1 {
        deliver_illegal_instruction(_state, instr as u64, result);
        return 0;
    }
    4
}

// ============================================================
//  Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode::decode_fields;

    fn new_state() -> HartState {
        let mut s: HartState = unsafe { std::mem::zeroed() };
        s.mode = crate::state::riscv_mode::M;
        s.mtvec = 0;
        s.pc = 0x8000_0000;
        s
    }

    fn new_result() -> BatchResult {
        unsafe { std::mem::zeroed() }
    }

    // -- LUI --

    #[test]
    fn lui_small() {
        let mut s = new_state();
        // lui x5, 0x42  ->  instr = 0x42 << 12 | 5 << 7 | 0x37
        let instr: u32 = (0x42u32 << 12) | (5u32 << 7) | 0x37;
        let f = decode_fields(instr);
        let mut r = new_result();
        let adv = handle_lui(&mut s, &f);
        assert_eq!(adv, 4);
        assert_eq!(s.gprs[5], 0x0004_2000);
        assert_eq!(s.gprs[0], 0); // x0 untouched
    }

    #[test]
    fn lui_negative() {
        let mut s = new_state();
        // lui x10, 0xFFFFF  ->  should give 0xFFFF_FFFF_FFFF_F000
        let instr: u32 = (0xFFFFFu32 << 12) | (10u32 << 7) | 0x37;
        let f = decode_fields(instr);
        let mut r = new_result();
        let adv = handle_lui(&mut s, &f);
        assert_eq!(adv, 4);
        assert_eq!(s.gprs[10], 0xFFFF_FFFF_FFFF_F000);
    }

    #[test]
    fn lui_x0_discards() {
        let mut s = new_state();
        let instr: u32 = (0x123u32 << 12) | (0u32 << 7) | 0x37; // rd=x0
        let f = decode_fields(instr);
        let mut r = new_result();
        let adv = handle_lui(&mut s, &f);
        assert_eq!(adv, 4);
        assert_eq!(s.gprs[0], 0);
    }

    // -- AUIPC --

    #[test]
    fn auipc_pc_relative() {
        let mut s = new_state();
        s.pc = 0x8000_1000;
        let instr: u32 = (0x10u32 << 12) | (5u32 << 7) | 0x17; // auipc x5, 0x10
        let f = decode_fields(instr);
        let mut r = new_result();
        let adv = handle_auipc(&mut s, &f);
        assert_eq!(adv, 4);
        assert_eq!(s.gprs[5], 0x8000_1000 + 0x0001_0000);
    }

    // -- JAL --

    #[test]
    fn jal_forward() {
        let mut s = new_state();
        s.pc = 0x8000_0000;
        // jal x1, +8  -> offset = 8
        // imm_j: bit layout → we'll use decode_fields
        let instr: u32 = 0x0080_00EF; // jal ra, +8 (pre-encoded)
        let f = decode_fields(instr);
        let mut r = new_result();
        let adv = handle_jal(&mut s, &f);
        assert_eq!(adv, 0);
        assert_eq!(s.gprs[1], 0x8000_0004);
        assert_eq!(s.pc, 0x8000_0008);
    }

    #[test]
    fn jal_backward() {
        let mut s = new_state();
        s.pc = 0x8000_0100;
        // jal x0, -4  →  x0 isn't written, PC goes back
        let instr: u32 = 0xFFDF_F06F; // JAL x0, -4 (imm_j = -4 as 21-bit signed)
        let f = decode_fields(instr);
        let mut r = new_result();
        let adv = handle_jal(&mut s, &f);
        assert_eq!(adv, 0);
        assert_eq!(s.gprs[0], 0); // x0 hardwired
        assert_eq!(s.pc, 0x8000_0100u64.wrapping_add(0xFFFF_FFFF_FFFF_FFFC));
    }

    // -- JALR --

    #[test]
    fn jalr_basic() {
        let mut s = new_state();
        s.pc = 0x8000_0000;
        s.gprs[10] = 0x8000_1000;
        // jalr x1, x10, 0
        let instr: u32 = (10u32 << 15) | (1u32 << 7) | 0x67; // func3=000
        let f = decode_fields(instr);
        let mut r = new_result();
        let adv = handle_jalr(&mut s, &f);
        assert_eq!(adv, 0);
        assert_eq!(s.gprs[1], 0x8000_0004);
        assert_eq!(s.pc, 0x8000_1000);
    }

    #[test]
    fn jalr_lsb_clear() {
        let mut s = new_state();
        s.pc = 0x8000_0000;
        s.gprs[10] = 0x8000_1003; // odd address
        let instr: u32 = (10u32 << 15) | 0x67;
        let f = decode_fields(instr);
        let mut r = new_result();
        let adv = handle_jalr(&mut s, &f);
        assert_eq!(s.pc, 0x8000_1002); // LSB cleared
    }

    // -- Branches --

    #[test]
    fn beq_taken() {
        let mut s = new_state();
        s.gprs[1] = 42;
        s.gprs[2] = 42;
        // beq x1, x2, +16  → func3=000, offset=16
        let instr: u32 = (0u32 << 31) | (2u32 << 20) | (1u32 << 15) | (0u32 << 12)
            | (0b000_1000u32 << 7) | 0x63; // imm_b=16, bits [8:7]=01
        // Let's use a pre-computed encoding
        let pre: u32 = (1u32 << 12) << 19    // bit12=1: offset bit 12 set
                      | (1u32 << 8)           // offset bit 4
                      | (0u32 << 7)           // offset bit 11
                      | (2u32 << 20)          // rs2 = x2
                      | (1u32 << 15)          // rs1 = x1
                      | (0u32 << 12)          // func3 = 000
                      | 0x63;                 // opcode = BR
        // Hmm, this is error-prone. Let me just test with a known encoding.
        let instr: u32 = 0x0020_8863; // beq x1, x2, +16
        let f = decode_fields(instr);
        let mut r = new_result();
        let adv = handle_br(&mut s, &f, instr, &mut r);
        assert_eq!(adv, 0, "branch should be taken");
        assert_eq!(s.pc, 0x8000_0000u64.wrapping_add(f.imm_b));
    }

    #[test]
    fn beq_not_taken() {
        let mut s = new_state();
        s.gprs[1] = 42;
        s.gprs[2] = 99;
        let instr: u32 = 0x0020_8863; // beq x1, x2, +16
        let f = decode_fields(instr);
        let mut r = new_result();
        let adv = handle_br(&mut s, &f, instr, &mut r);
        assert_eq!(adv, 4, "branch should NOT be taken");
        assert_eq!(s.pc, 0x8000_0000); // pc unchanged
    }

    #[test]
    fn blt_signed() {
        let mut s = new_state();
        // -1 < 1 is true
        s.gprs[1] = 0xFFFF_FFFF_FFFF_FFFFu64; // -1
        s.gprs[2] = 1;
        let instr: u32 = 0x0021_C463; // blt x1, x2, +8  (approx)
        // Actually let me compute: blt is func3=100
        let instr: u32 = (1u32 << 12) << 19   // bit12=1
                      | (1u32 << 8)
                      | (2u32 << 20)
                      | (1u32 << 15)
                      | (4u32 << 12)           // func3 = 0b100 = BLT
                      | 0x63;
        let f = decode_fields(instr);
        let mut r = new_result();
        let adv = handle_br(&mut s, &f, instr, &mut r);
        assert_eq!(adv, 0, "blt -1 < 1 should be taken");
    }

    #[test]
    fn bgeu_unsigned() {
        let mut s = new_state();
        // 5 >= 3 unsigned
        s.gprs[1] = 5;
        s.gprs[2] = 3;
        let instr: u32 = (1u32 << 12) << 19
                      | (1u32 << 8)
                      | (2u32 << 20)
                      | (1u32 << 15)
                      | (7u32 << 12)           // func3 = 0b111 = BGEU
                      | 0x63;
        let f = decode_fields(instr);
        let mut r = new_result();
        let adv = handle_br(&mut s, &f, instr, &mut r);
        assert_eq!(adv, 0, "bgeu 5 >= 3 should be taken");
    }

    // -- ALU (Rust native, integration test) --

    #[test]
    fn addi_via_handler() {
        let mut s = new_state();
        s.gprs[10] = 100;
        // addi x5, x10, 42
        let instr: u32 = (42u32 & 0xFFF) << 20 | (10u32 << 15) | (0u32 << 12) | (5u32 << 7) | 0x13;
        let f = decode_fields(instr);
        let mut r = new_result();
        let adv = handle_op_imm(&mut s, &f, instr, &mut r);
        assert_eq!(adv, 4);
        assert_eq!(s.gprs[5], 142);
    }

    #[test]
    fn sub_via_handler() {
        let mut s = new_state();
        s.gprs[10] = 100;
        s.gprs[11] = 30;
        // sub x5, x10, x11  → func7=0x20, func3=000
        let instr: u32 = (0x20u32 << 25) | (11u32 << 20) | (10u32 << 15) | (0u32 << 12) | (5u32 << 7) | 0x33;
        let f = decode_fields(instr);
        let mut r = new_result();
        let adv = handle_alu(&mut s, &f, instr, &mut r);
        assert_eq!(adv, 4);
        assert_eq!(s.gprs[5], 70);
    }

    #[test]
    fn x0_source_is_zero() {
        let mut s = new_state();
        // addi x5, x0, 42  → x0 is always 0
        let instr: u32 = (42u32 & 0xFFF) << 20 | (0u32 << 15) | (0u32 << 12) | (5u32 << 7) | 0x13;
        let f = decode_fields(instr);
        let mut r = new_result();
        let adv = handle_op_imm(&mut s, &f, instr, &mut r);
        assert_eq!(s.gprs[5], 42);
    }

    #[test]
    fn illegal_alu_encoding_traps() {
        let mut s = new_state();
        s.gprs[10] = 1;
        s.gprs[11] = 2;
        // func3=001, func7=0b001_0011 (invalid combination for SLL/MULH)
        let instr: u32 = (0x13u32 << 25) | (11u32 << 20) | (10u32 << 15) | (1u32 << 12) | (5u32 << 7) | 0x33;
        let f = decode_fields(instr);
        let mut r = new_result();
        let adv = handle_alu(&mut s, &f, instr, &mut r);
        assert_eq!(adv, 0, "trap → advance should be 0");
        assert_eq!(s.mcause, crate::trap::mcause_val(crate::trap::exc_code::ILL_INSTR, false));
    }
}
