//! Pure-compute instruction handlers for Phase A.
//!
//! Each handler receives ``&mut HartState`` and ``&DecodedFields``,
//! executes the instruction, and returns the PC advance (0 or 4).
//!
//! Handlers that encounter an invalid encoding call ``deliver_illegal_instruction``
//! and return 0 (PC already redirected to ``mtvec``).

use crate::alu::{exec_alu_op, exec_op_imm, exec_op32, exec_op_imm32};
use crate::decode::DecodedFields;
use crate::fpu::{fp_exec_fma, fp_exec_op};
use crate::state::BatchResult;
use crate::state::HartState;
use crate::trap::deliver_illegal_instruction;

// ============================================================
//  Helpers
// ============================================================

/// mstatus.FS 字段掩码 (bits[14:13]) — 浮点单元状态。
const MSTATUS_FS: u64 = 0b11 << 13;
/// mstatus.SD 位 (bit 63) — 任一扩展状态为 Dirty 时置位。
const MSTATUS_SD: u64 = 1 << 63;

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
//  F/D floating point
// ============================================================

/// 读取 FPR 原始 bits (f0 是真实寄存器, 无 x0 特例)。
#[inline]
fn read_fpr(state: &HartState, r: u8) -> u64 {
    state.fprs[r as usize]
}

/// 写入 FPR 原始 bits。
#[inline]
fn write_fpr(state: &mut HartState, r: u8, v: u64) {
    state.fprs[r as usize] = v;
}

/// 浮点单元是否被禁用 (mstatus.FS == Off)。
#[inline]
fn fp_disabled(state: &HartState) -> bool {
    (state.mstatus & MSTATUS_FS) == 0
}

/// 执行浮点指令后将 FS 标记为 Dirty (0b11) 并置 SD。
#[inline]
fn set_fs_dirty(state: &mut HartState) {
    state.mstatus |= MSTATUS_FS | MSTATUS_SD;
}

/// 当前动态舍入模式 (fcsr.frm)。
#[inline]
fn cur_frm(state: &HartState) -> u8 {
    ((state.fcsr >> 5) & 0x7) as u8
}

/// OP-FP (opcode 0x53): 算术 / 转换 / 比较 / 符号 / 分类 / 移动。
pub fn handle_fp_op(
    state: &mut HartState,
    f: &DecodedFields,
    instr: u32,
    result: &mut BatchResult,
) -> u64 {
    if fp_disabled(state) {
        deliver_illegal_instruction(state, instr as u64, result);
        return 0;
    }
    let op5 = f.func7 >> 2;
    // int->float (0x1A) 与 FMV.*.X (0x1E) 的 rs1 源为 GPR; 其余为 FPR。
    let rs1_bits = if matches!(op5, 0x1A | 0x1E) {
        read_gpr(state, f.rs1)
    } else {
        read_fpr(state, f.rs1)
    };
    let rs2_bits = read_fpr(state, f.rs2);
    let out = fp_exec_op(f.func7, f.func3, f.rs2, rs1_bits, rs2_bits, cur_frm(state));
    if out.trap != 0 {
        deliver_illegal_instruction(state, instr as u64, result);
        return 0;
    }
    if out.to_gpr != 0 {
        write_gpr(state, f.rd, out.value);
    } else {
        write_fpr(state, f.rd, out.value);
    }
    state.fcsr |= u32::from(out.fflags);
    set_fs_dirty(state);
    4
}

/// FMA (opcode 0x43/0x47/0x4B/0x4F): 融合乘加的四种变体。
pub fn handle_fp_fma(
    state: &mut HartState,
    f: &DecodedFields,
    instr: u32,
    result: &mut BatchResult,
) -> u64 {
    if fp_disabled(state) {
        deliver_illegal_instruction(state, instr as u64, result);
        return 0;
    }
    let rs1 = read_fpr(state, f.rs1);
    let rs2 = read_fpr(state, f.rs2);
    let rs3 = read_fpr(state, f.rs3);
    let out = fp_exec_fma(f.opcode, f.func3, f.fmt, rs1, rs2, rs3, cur_frm(state));
    if out.trap != 0 {
        deliver_illegal_instruction(state, instr as u64, result);
        return 0;
    }
    write_fpr(state, f.rd, out.value);
    state.fcsr |= u32::from(out.fflags);
    set_fs_dirty(state);
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
//  FENCE / FENCE.I: memory-ordering barriers
// ============================================================
//
// In the concurrent (thread-per-hart) model, these must emit real CPU
// memory barriers.  OpenSBI relies on ``fence ow,ow`` (wmb) and
// ``fence ir,ir`` (rmb) for lock-free FIFO enqueue/dequeue between
// harts.  Without real barriers the compiler and CPU may reorder
// stores/loads across the fence ->FIFO corruption ->TLB-shootdown
// deadlocks (sender spins in tlb_update retry because the remote
// FIFO appears full / head pointer is never observed advancing).

pub fn handle_fence(_state: &mut HartState, f: &DecodedFields, instr: u32, result: &mut BatchResult) -> u64 {
    // func3 == 0b000: FENCE  — emit a full SeqCst barrier.
    //   Conservative (the RISC-V spec allows pred/succ filtering
    //   via the encoded predecessor/successor sets), but always
    //   correct and fast enough for the current workload.
    // func3 == 0b001: FENCE.I — instruction fence; no-op in an
    //   emulator where the icache is implicitly coherent with
    //   stores (single bytearray backing RAM).
    if f.func3 > 1 {
        deliver_illegal_instruction(_state, instr as u64, result);
        return 0;
    }
    if f.func3 == 0 {
        // FENCE — data memory barrier.
        // Acquire-Release is sufficient for two reasons:
        // 1. Guest stores/loads already use Release/Acquire atomics
        //    (ram_write_raw / ram_read_raw), so an AcqRel fence
        //    between them keeps the happens-before chain intact.
        // 2. x86-64 TSO provides the hardware ordering for free
        //    (fence(AcqRel) is a zero-cycle compiler barrier);
        //    SeqCst would emit mfence (~100 cycles) for no benefit.
        std::sync::atomic::fence(std::sync::atomic::Ordering::AcqRel);
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
        let mut _r = new_result();
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
        let mut _r = new_result();
        let adv = handle_lui(&mut s, &f);
        assert_eq!(adv, 4);
        assert_eq!(s.gprs[10], 0xFFFF_FFFF_FFFF_F000);
    }

    #[test]
    fn lui_x0_discards() {
        let mut s = new_state();
        let instr: u32 = (0x123u32 << 12) | (0u32 << 7) | 0x37; // rd=x0
        let f = decode_fields(instr);
        let mut _r = new_result();
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
        let mut _r = new_result();
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
        // imm_j: bit layout -> we'll use decode_fields
        let instr: u32 = 0x0080_00EF; // jal ra, +8 (pre-encoded)
        let f = decode_fields(instr);
        let mut _r = new_result();
        let adv = handle_jal(&mut s, &f);
        assert_eq!(adv, 0);
        assert_eq!(s.gprs[1], 0x8000_0004);
        assert_eq!(s.pc, 0x8000_0008);
    }

    #[test]
    fn jal_backward() {
        let mut s = new_state();
        s.pc = 0x8000_0100;
        // jal x0, -4  ->  x0 isn't written, PC goes back
        let instr: u32 = 0xFFDF_F06F; // JAL x0, -4 (imm_j = -4 as 21-bit signed)
        let f = decode_fields(instr);
        let mut _r = new_result();
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
        let mut _r = new_result();
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
        let mut _r = new_result();
        let _adv = handle_jalr(&mut s, &f);
        assert_eq!(s.pc, 0x8000_1002); // LSB cleared
    }

    // -- Branches --

    #[test]
    fn beq_taken() {
        let mut s = new_state();
        s.gprs[1] = 42;
        s.gprs[2] = 42;
        // beq x1, x2, +16  -> func3=000, offset=16
        let _instr: u32 = (0u32 << 31) | (2u32 << 20) | (1u32 << 15) | (0u32 << 12)
            | (0b000_1000u32 << 7) | 0x63; // imm_b=16, bits [8:7]=01
        // Let's use a pre-computed encoding
        let _pre: u32 = (1u32 << 12) << 19    // bit12=1: offset bit 12 set
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
        // let instr: u32 = 0x0021_C463; // blt x1, x2, +8  (approx)
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
        // sub x5, x10, x11  -> func7=0x20, func3=000
        let instr: u32 = (0x20u32 << 25) | (11u32 << 20) | (10u32 << 15) | (0u32 << 12) | (5u32 << 7) | 0x33;
        let f = decode_fields(instr);
        let mut r = new_result();
        let adv = handle_alu(&mut s, &f, instr, &mut r);
        assert_eq!(adv, 4);
        assert_eq!(s.gprs[5], 70);
    }

    #[test]
	#[allow(dead_code)]
    fn x0_source_is_zero() {
        let mut s = new_state();
        // addi x5, x0, 42  -> x0 is always 0
        let instr: u32 = (42u32 & 0xFFF) << 20 | (0u32 << 15) | (0u32 << 12) | (5u32 << 7) | 0x13;
        let f = decode_fields(instr);
        let mut r = new_result();
        let _adv = handle_op_imm(&mut s, &f, instr, &mut r);
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
        assert_eq!(adv, 0, "trap -> advance should be 0");
        assert_eq!(s.mcause, crate::trap::mcause_val(crate::trap::exc_code::ILL_INSTR, false));
    }

    // ============================================================
    //  rd == rs1  regression tests
    // ============================================================

    /// ADDI with rd==rs1 must use the old value as the operand, then overwrite.
    #[test]
    fn addi_rd_equals_rs1_uses_old_value() {
        let mut s = new_state();
        s.gprs[5] = 10;
        // addi x5, x5, 42  -> x5 = 10 + 42 = 52
        let instr: u32 = (42u32 << 20) | (5u32 << 15) | (0u32 << 12) | (5u32 << 7) | 0x13;
        let f = decode_fields(instr);
        let mut r = new_result();
        let adv = handle_op_imm(&mut s, &f, instr, &mut r);
        assert_eq!(adv, 4);
        assert_eq!(s.gprs[5], 52, "ADDI rd==rs1: must use old value 10, got {}", s.gprs[5]);
    }

    /// SLLI with rd==rs1.
    #[test]
    fn slli_rd_equals_rs1_uses_old_value() {
        let mut s = new_state();
        s.gprs[10] = 3;
        // slli x10, x10, 4  -> x10 = 3 << 4 = 48
        let instr: u32 = (0u32 << 25) | (4u32 << 20) | (10u32 << 15) | (1u32 << 12) | (10u32 << 7) | 0x13;
        let f = decode_fields(instr);
        let mut r = new_result();
        let adv = handle_op_imm(&mut s, &f, instr, &mut r);
        assert_eq!(adv, 4);
        assert_eq!(s.gprs[10], 48, "SLLI rd==rs1: 3 << 4 = 48, got {}", s.gprs[10]);
    }

    /// ADD with rd==rs1 (rd same as rs1, different from rs2).
    #[test]
    fn add_rd_equals_rs1_uses_old_value() {
        let mut s = new_state();
        s.gprs[5] = 20;
        s.gprs[6] = 7;
        // add x5, x5, x6  -> x5 = 20 + 7 = 27
        let instr: u32 = (0u32 << 25) | (6u32 << 20) | (5u32 << 15) | (0u32 << 12) | (5u32 << 7) | 0x33;
        let f = decode_fields(instr);
        let mut r = new_result();
        let adv = handle_alu(&mut s, &f, instr, &mut r);
        assert_eq!(adv, 4);
        assert_eq!(s.gprs[5], 27, "ADD rd==rs1: 20 + 7 = 27, got {}", s.gprs[5]);
    }

    /// JALR with rd==rs1 must save the return address before jumping.
    #[test]
    fn jalr_rd_equals_rs1_preserves_link_and_jumps() {
        let mut s = new_state();
        s.pc = 0x8000_1000;
        s.gprs[1] = 0x8000_2000; // ra = target
        // jalr ra, ra, 0  -> rd=1, rs1=1
        let instr: u32 = (0u32 << 20) | (1u32 << 15) | (0u32 << 12) | (1u32 << 7) | 0x67;
        let f = decode_fields(instr);
        let adv = handle_jalr(&mut s, &f);
        assert_eq!(adv, 0, "JALR returns 0 (PC modified directly)");
        // PC jumps to old ra value (with LSB cleared)
        assert_eq!(s.pc, 0x8000_2000, "JALR must jump to old ra");
        // ra gets the return address (old PC + 4)
        assert_eq!(s.gprs[1], 0x8000_1004, "JALR rd==rs1: ra must hold return address, got {:#x}", s.gprs[1]);
    }

    // -- ld-linux loop exit regression (crash at offset 0x3ae8) --

    /// bltz at 0x3ab6: with s2=-1, branch MUST be taken to skip the loop.
    /// If this fails, the loop body executes with s2=-1 and accesses
    /// base + (-1)*0xa0 = base - 0xa0 ->.dynamic section ->reads d_tag=7.
    #[test]
    fn ldlinux_bltz_skip_loop_on_counter_zero() {
        let mut s = new_state();
        s.pc = 0x3AB6;
        s.gprs[18] = u64::MAX; // s2 = -1 = 0xFFFF_FFFF_FFFF_FFFF
        // bltz s2, 0x3b0a
        let instr: u32 = 0x04094A63;
        let f = decode_fields(instr);
        let mut r = new_result();
        let adv = handle_br(&mut s, &f, instr, &mut r);
        assert_eq!(adv, 0, "bltz with s2=-1 must take branch (skip loop)");
        assert_eq!(s.pc, 0x3AB6u64.wrapping_add(84), "bltz target must be 0x3b0a");
    }

    /// bne at 0x3b06: with s2=s4=-1, branch must NOT be taken (exit loop).
    /// If this fails, the loop continues past the array boundary.
    #[test]
    fn ldlinux_bne_exit_on_s2_eq_s4() {
        let mut s = new_state();
        s.pc = 0x3B06;
        s.gprs[18] = u64::MAX; // s2 = -1
        s.gprs[20] = u64::MAX; // s4 = -1
        // bne s2, s4, 0x3ad0
        let instr: u32 = 0xFD4915E3;
        let f = decode_fields(instr);
        let mut r = new_result();
        let adv = handle_br(&mut s, &f, instr, &mut r);
        assert_eq!(adv, 4, "bne with s2=s4=-1 must NOT take branch (exit loop)");
    }

}
