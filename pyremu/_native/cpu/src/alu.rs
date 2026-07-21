//! RISC-V ALU pure-compute operations.
//!
//! All functions return an `Alu64` struct with a `.value` field (the result)
//! and a `.trap` field (0 = ok, 1 = invalid encoding -> IllInstr).

// ============================================================
//  Return type
// ============================================================

/// ALU result with trap flag.  `trap == 1` means the encoding is illegal
/// and the caller should deliver an Illegal Instruction trap.
#[repr(C)]
pub struct Alu64 {
    pub value: u64,
    pub trap: u8,
}

impl Alu64 {
    #[inline(always)]
    fn ok(value: u64) -> Self {
        Alu64 { value, trap: 0 }
    }
    #[inline(always)]
    fn ill() -> Self {
        Alu64 { value: 0, trap: 1 }
    }
}

// ============================================================
//  RV64 helpers
// ============================================================

/// RISC-V R-type ALU: funct3 selects the operation, funct7 selects the variant.
///
/// | funct3 | funct7=0 | funct7=1 | funct7=0x20 |
/// |--------|----------|----------|-------------|
/// | 000    | ADD      | MUL      | SUB         |
/// | 001    | SLL      | MULH     |             |
/// | 010    | SLT      | MULHSU   |             |
/// | 011    | SLTU     | MULHU    |             |
/// | 100    | XOR      | DIV      |             |
/// | 101    | SRL      | DIVU     | SRA         |
/// | 110    | OR       | REM      |             |
/// | 111    | AND      | REMU     |             |
#[no_mangle]
pub extern "C" fn exec_alu_op(funct3: u8, funct7: u8, v1: u64, v2: u64) -> Alu64 {
    match funct3 {
        0b000 => match funct7 {
            0 => Alu64::ok(v1.wrapping_add(v2)),
            1 => Alu64::ok(v1.wrapping_mul(v2)),
            0x20 => Alu64::ok(v1.wrapping_sub(v2)),
            _ => Alu64::ill(),
        },
        0b001 => match funct7 {
            0 => Alu64::ok(v1 << (v2 & 0x3F)),
            1 => {
                let s1 = v1 as i64 as i128;
                let s2 = v2 as i64 as i128;
                Alu64::ok(((s1 * s2) >> 64) as u64)
            }
            _ => Alu64::ill(),
        },
        0b010 => match funct7 {
            0 => Alu64::ok(if (v1 as i64) < (v2 as i64) { 1 } else { 0 }),
            1 => {
                let s1 = v1 as i64 as i128;
                let u2 = v2 as i128;
                Alu64::ok(((s1 * u2) >> 64) as u64)
            }
            _ => Alu64::ill(),
        },
        0b011 => match funct7 {
            0 => Alu64::ok(if v1 < v2 { 1 } else { 0 }),
            1 => {
                let u1 = v1 as u128;
                let u2 = v2 as u128;
                Alu64::ok(((u1 * u2) >> 64) as u64)
            }
            _ => Alu64::ill(),
        },
        0b100 => match funct7 {
            0 => Alu64::ok(v1 ^ v2),
            1 => {
                let s1 = v1 as i64;
                let s2 = v2 as i64;
                let val = if s2 == 0 { u64::MAX }
                    else if s1 == i64::MIN && s2 == -1 { i64::MIN as u64 }
                    else { (s1 / s2) as u64 };
                Alu64::ok(val)
            }
            _ => Alu64::ill(),
        },
        0b101 => match funct7 {
            0 => Alu64::ok(v1 >> (v2 & 0x3F)),
            1 => Alu64::ok(if v2 == 0 { u64::MAX } else { v1 / v2 }),
            0x20 => Alu64::ok(((v1 as i64) >> (v2 & 0x3F)) as u64),
            _ => Alu64::ill(),
        },
        0b110 => match funct7 {
            0 => Alu64::ok(v1 | v2),
            1 => {
                let s1 = v1 as i64;
                let s2 = v2 as i64;
                let val = if s2 == 0 { v1 }
                    else if s1 == i64::MIN && s2 == -1 { 0 }
                    else { (s1 % s2) as u64 };
                Alu64::ok(val)
            }
            _ => Alu64::ill(),
        },
        0b111 => match funct7 {
            0 => Alu64::ok(v1 & v2),
            1 => Alu64::ok(if v2 == 0 { v1 } else { v1 % v2 }),
            _ => Alu64::ill(),
        },
        _ => Alu64::ill(),
    }
}

// ============================================================
//  I-type ALU (op-imm)
// ============================================================

/// RISC-V I-type immediate ALU: funct3 selects the operation.
///
/// `imm` is the already sign-extended 12-bit immediate.
/// `funct7` provides the funct6 field (bits[31:26]) for SRLI/SRAI distinction.
#[no_mangle]
pub extern "C" fn exec_op_imm(funct3: u8, funct7: u8, v1: u64, imm: u64) -> Alu64 {
    let shamt = imm & 0x3F;
    let funct6 = funct7 >> 1;

    match funct3 {
        0b000 => Alu64::ok(v1.wrapping_add(imm)),
        0b001 => {
            if funct6 != 0 {
                Alu64::ill()
            } else {
                Alu64::ok(v1 << shamt)
            }
        }
        0b010 => Alu64::ok(if (v1 as i64) < (imm as i64) { 1 } else { 0 }),
        0b011 => Alu64::ok(if v1 < imm { 1 } else { 0 }),
        0b100 => Alu64::ok(v1 ^ imm),
        0b101 => match funct6 {
            0 => Alu64::ok(v1 >> shamt),
            0x10 => Alu64::ok(((v1 as i64) >> shamt) as u64),
            _ => Alu64::ill(),
        },
        0b110 => Alu64::ok(v1 | imm),
        0b111 => Alu64::ok(v1 & imm),
        _ => Alu64::ill(),
    }
}

// ============================================================
//  RV64 32-bit word ops (op-32)
// ============================================================

/// RISC-V RV64 32-bit word ALU: operates on lower 32 bits, sign-extends result.
///
/// | funct3 | funct7=0 | funct7=1 | funct7=0x20 |
/// |--------|----------|----------|-------------|
/// | 000    | ADDW     | MULW     | SUBW        |
/// | 001    | SLLW     |          |             |
/// | 100    |          | DIVW     |             |
/// | 101    | SRLW     | DIVUW    | SRAW        |
/// | 110    |          | REMW     |             |
/// | 111    |          | REMUW    |             |
#[no_mangle]
pub extern "C" fn exec_op32(funct3: u8, funct7: u8, v1: u64, v2: u64) -> Alu64 {
    let w1 = (v1 & 0xFFFF_FFFF) as i32;
    let w2 = (v2 & 0xFFFF_FFFF) as i32;

    let result_i32: i32 = match funct3 {
        0b000 => match funct7 {
            0 => w1.wrapping_add(w2),
            1 => w1.wrapping_mul(w2),
            0x20 => w1.wrapping_sub(w2),
            _ => return Alu64::ill(),
        },
        0b001 => {
            if funct7 != 0 { return Alu64::ill(); }
            w1 << (w2 & 0x1F)
        }
        0b100 => {
            if funct7 != 1 { return Alu64::ill(); }
            if w2 == 0 { -1i32 }
            else if w1 == i32::MIN && w2 == -1 { i32::MIN }
            else { w1 / w2 }
        }
        0b101 => match funct7 {
            0 => ((w1 as u32) >> (w2 as u32 & 0x1F)) as i32,
            1 => {
                let u1 = w1 as u32;
                let u2 = w2 as u32;
                if u2 == 0 { u32::MAX as i32 } else { (u1 / u2) as i32 }
            }
            0x20 => w1 >> (w2 & 0x1F),
            _ => return Alu64::ill(),
        },
        0b110 => {
            if funct7 != 1 { return Alu64::ill(); }
            if w2 == 0 { w1 }
            else if w1 == i32::MIN && w2 == -1 { 0 }
            else { w1 % w2 }
        }
        0b111 => {
            if funct7 != 1 { return Alu64::ill(); }
            let u1 = w1 as u32;
            let u2 = w2 as u32;
            if u2 == 0 { u1 as i32 } else { (u1 % u2) as i32 }
        }
        _ => return Alu64::ill(),
    };

    Alu64::ok((result_i32 as i64) as u64)
}

// ============================================================
//  RV64 32-bit word immediate ops (op-imm-32)
// ============================================================

/// RISC-V RV64 32-bit immediate ALU: operates on lower 32 bits, sign-extends.
///
/// | funct3 | operation |
/// |--------|-----------|
/// | 000    | ADDIW     |
/// | 001    | SLLIW     |
/// | 101    | SRLIW / SRAIW |
#[no_mangle]
pub extern "C" fn exec_op_imm32(funct3: u8, funct7: u8, v1: u64, imm: u64) -> Alu64 {
    let w1 = (v1 & 0xFFFF_FFFF) as i32;
    let shamt = imm as u32 & 0x1F;

    let result_i32: i32 = match funct3 {
        0b000 => w1.wrapping_add(imm as i32),
        0b001 => {
            if funct7 != 0 { return Alu64::ill(); }
            w1 << shamt
        }
        0b101 => {
            let funct6 = funct7 >> 1;
            match funct6 {
                0 => ((w1 as u32) >> shamt) as i32,
                0x10 => w1 >> shamt,
                _ => return Alu64::ill(),
            }
        }
        _ => return Alu64::ill(),
    };

    Alu64::ok((result_i32 as i64) as u64)
}

// ============================================================
//  Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn v(r: Alu64) -> u64 { assert_eq!(r.trap, 0); r.value }
    fn trap(r: Alu64) -> bool { r.trap != 0 }

    #[test] fn test_add()           { assert_eq!(v(exec_alu_op(0, 0, 10, 5)), 15); }
    #[test] fn test_sub()           { assert_eq!(v(exec_alu_op(0, 0x20, 10, 5)), 5); }
    #[test] fn test_sll()           { assert_eq!(v(exec_alu_op(1, 0, 1, 3)), 8); }
    #[test] fn test_xor()           { assert_eq!(v(exec_alu_op(4, 0, 0xFF, 0x0F)), 0xF0); }
    #[test] fn test_or()            { assert_eq!(v(exec_alu_op(6, 0, 0xF0, 0x0F)), 0xFF); }
    #[test] fn test_and()           { assert_eq!(v(exec_alu_op(7, 0, 0xFF, 0x0F)), 0x0F); }
    #[test] fn test_slt_true()      { assert_eq!(v(exec_alu_op(2, 0, 5, 10)), 1); }
    #[test] fn test_slt_false()     { assert_eq!(v(exec_alu_op(2, 0, 10, 5)), 0); }
    #[test] fn test_sltu_true()     { assert_eq!(v(exec_alu_op(3, 0, 5, 10)), 1); }
    #[test] fn test_srl()           { assert_eq!(v(exec_alu_op(5, 0, 0x100, 4)), 0x10); }
    #[test] fn test_sra_negative() {
        assert_eq!(v(exec_alu_op(5, 0x20, 0xFFFF_FFFF_FFFF_FF00u64, 4)),
                   0xFFFF_FFFF_FFFF_FFF0u64);
    }
    #[test] fn test_div()           { assert_eq!(v(exec_alu_op(4, 1, 20, 6)), 3); }
    #[test] fn test_div_by_zero()   { assert_eq!(v(exec_alu_op(4, 1, 42, 0)), u64::MAX); }
    #[test] fn test_div_overflow() {
        assert_eq!(v(exec_alu_op(4, 1, i64::MIN as u64, (-1i64) as u64)), i64::MIN as u64);
    }
    #[test] fn test_divu()          { assert_eq!(v(exec_alu_op(5, 1, 20, 6)), 3); }
    #[test] fn test_rem()           { assert_eq!(v(exec_alu_op(6, 1, 20, 6)), 2); }
    #[test] fn test_remu()          { assert_eq!(v(exec_alu_op(7, 1, 20, 6)), 2); }
    #[test] fn test_mul()           { assert_eq!(v(exec_alu_op(0, 1, 6, 7)), 42); }
    #[test] fn test_alu_ill_funct7(){ assert!(trap(exec_alu_op(0, 0x0F, 10, 5))); }
    #[test] fn test_alu_ill_funct3(){ assert!(trap(exec_alu_op(0xFF, 0, 10, 5))); }

    #[test] fn test_addi()          { assert_eq!(v(exec_op_imm(0, 0, 10, 5)), 15); }
    #[test] fn test_addi_negative() { assert_eq!(v(exec_op_imm(0, 0, 10, (-2i64) as u64)), 8); }
    #[test] fn test_slti()          { assert_eq!(v(exec_op_imm(2, 0, 5, 10)), 1); }
    #[test] fn test_xori()          { assert_eq!(v(exec_op_imm(4, 0, 0xFF, 0x0F)), 0xF0); }
    #[test] fn test_slli_bad_f6()   { assert!(trap(exec_op_imm(1, 0x7F, 10, 3))); }
    #[test] fn test_srli_bad_f6() {
        // funct3=101 (SRLI/SRAI group): valid funct6 = 0 or 0x10.
        // funct7=0x3E -> funct6=0x1F (invalid)
        assert!(trap(exec_op_imm(5, 0x3E, 0x100, 2)));
    }

    #[test] fn test_addw()          { assert_eq!(v(exec_op32(0, 0, 10, 5)), 15); }
    #[test] fn test_addw_sign_extend() {
        assert_eq!(v(exec_op32(0, 0, 0x7FFFFFFF, 1)), 0xFFFF_FFFF_8000_0000u64);
    }
    #[test] fn test_subw()          { assert_eq!(v(exec_op32(0, 0x20, 10, 5)), 5); }
    #[test] fn test_op32_ill()      { assert!(trap(exec_op32(0b001, 1, 10, 3))); }

    #[test] fn test_slliw()         { assert_eq!(v(exec_op_imm32(1, 0, 0x100, 2)), 0x400); }
    #[test] fn test_sraiw() {
        // SRAIW: funct3=101, funct7=0100000=0x20 -> funct6=0x10
        assert_eq!(v(exec_op_imm32(5, 0x20, 0xFFFF_FFFF_8000_0000u64, 1)),
                   0xFFFF_FFFF_C000_0000u64);
    }
    #[test] fn test_op_imm32_ill()  { assert!(trap(exec_op_imm32(1, 1, 0x100, 2))); }
}
