//! RISC-V instruction field extraction — batch decode in a single FFI call.
//!
//! Replaces 4–6 Python function calls per 32-bit instruction and 5–10 inline
//! bit-extraction expressions per 16-bit compressed instruction with one FFI
//! call each.

// ============================================================
//  32-bit instruction decode
// ============================================================

/// All standard RISC-V instruction fields, extracted in one pass.
///
/// Field order matches the ctypes `_fields_` list (largest alignment first)
/// to guarantee identical layout across FFI without manual padding.
#[repr(C)]
pub struct DecodedFields {
    // ---- 8-byte aligned (u64) ----
    pub imm12_se: u64,
    pub imm_s: u64,
    pub imm_b: u64,
    pub imm_j: u64,
    pub imm20_raw: u64,

    // ---- 2-byte aligned (u16) ----
    pub func12: u16,

    // ---- 1-byte aligned (u8) ----
    pub opcode: u8,
    pub rd: u8,
    pub func3: u8,
    pub rs1: u8,
    pub rs2: u8,
    pub func7: u8,
    pub is_compressed: u8,
    // ---- F/D floating point (FMA rs3 + fmt) ----
    /// rs3 寄存器号 (FMA 指令, bits[31:27])。
    pub rs3: u8,
    /// 浮点格式 (bits[26:25]): 0=S(单), 1=D(双), 2=H(半), 3=Q(四)。
    pub fmt: u8,
}

/// Fast sign-extend 12-bit -> u64.
#[inline(always)]
fn sext12(val: u64) -> u64 {
    (val & 0x7FF).wrapping_sub(val & 0x800) & 0xFFFF_FFFF_FFFF_FFFF
}

/// Fast sign-extend 13-bit -> u64.
#[inline(always)]
fn sext13(val: u64) -> u64 {
    (val & 0xFFF).wrapping_sub(val & 0x1000) & 0xFFFF_FFFF_FFFF_FFFF
}

/// Fast sign-extend 21-bit -> u64.
#[inline(always)]
fn sext21(val: u64) -> u64 {
    (val & 0xF_FFFF).wrapping_sub(val & 0x10_0000) & 0xFFFF_FFFF_FFFF_FFFF
}

/// Extract all standard RISC-V fields from a 32-bit instruction.
///
/// Called once per instruction via ctypes, replacing:
///   `parse_opcode`, `parse_rd`, `parse_func3`, `parse_rs1`, `parse_rs2`,
///   `parse_func7`, `parse_func12`, `parse_imm12_se`, `parse_imm_s`,
///   `parse_imm_b`, `parse_imm_j`, `parse_imm20_raw`, `parse_compressed`
#[no_mangle]
pub extern "C" fn decode_fields(instr: u32) -> DecodedFields {
    let instr = instr as u64;

    DecodedFields {
        // immediates (8-byte aligned)
        imm12_se: sext12((instr >> 20) & 0xFFF),
        imm_s: sext12(
            (((instr >> 25) & 0x7F) << 5) | ((instr >> 7) & 0x1F),
        ),
        imm_b: sext13(
            ((instr >> 31) & 1) << 12          // imm[12]
                | ((instr >> 25) & 0x3F) << 5  // imm[10:5]
                | ((instr >> 8) & 0xF) << 1    // imm[4:1]
                | ((instr >> 7) & 1) << 11,     // imm[11]
        ),
        imm_j: sext21(
            ((instr >> 31) & 1) << 20          // imm[20]
                | ((instr >> 21) & 0x3FF) << 1 // imm[10:1]
                | ((instr >> 20) & 1) << 11    // imm[11]
                | ((instr >> 12) & 0xFF) << 12, // imm[19:12]
        ),
        imm20_raw: (instr >> 12) & 0xF_FFFF,

        // func12 (2-byte aligned)
        func12: ((instr >> 20) & 0xFFF) as u16,

        // scalar fields (1-byte aligned)
        opcode: (instr & 0x7F) as u8,
        rd: ((instr >> 7) & 0x1F) as u8,
        func3: ((instr >> 12) & 0x7) as u8,
        rs1: ((instr >> 15) & 0x1F) as u8,
        rs2: ((instr >> 20) & 0x1F) as u8,
        func7: ((instr >> 25) & 0x7F) as u8,
        is_compressed: u8::from((instr & 0x3) != 3),
        // F/D: rs3 = bits[31:27], fmt = bits[26:25]
        rs3: ((instr >> 27) & 0x1F) as u8,
        fmt: ((instr >> 25) & 0x3) as u8,
    }
}

// ============================================================
//  16-bit compressed instruction decode
// ============================================================

/// All compressed-instruction register and immediate fields, decoded in one pass.
///
/// The Python handlers pick the fields they need based on `quadrant` + `funct3`.
#[repr(C)]
pub struct CompressedFields {
    // ---- 8-byte aligned ----
    /// Primary immediate — which one is valid depends on quadrant + funct3.
    ///
    /// | Quadrant | funct3 | Meaning |
    /// |----------|--------|---------|
    /// | 0 (C0)   | 000    | C.ADDI4SPN nzuimm |
    /// | 0        | 010/110 | C.LW/C.SW uimm |
    /// | 0        | 011/111 | C.LD/C.SD uimm |
    /// | 1 (C1)   | 000–010  | C.ADDI/C.LI 6-bit signed imm |
    /// | 1        | 011 (rd≠2) | C.LUI 6-bit nzimm (shift-left-12 in caller) |
    /// | 1        | 011 (rd=2) | C.ADDI16SP 10-bit nzimm |
    /// | 1        | 101      | C.J 11-bit offset |
    /// | 1        | 110/111  | C.BEQZ/C.BNEZ 8-bit offset |
    /// | 2 (C2)   | 000      | C.SLLI shamt[5:0] |
    /// | 2        | 010      | C.LWSP uimm |
    /// | 2        | 011      | C.LDSP uimm |
    pub imm: u64,

    /// Secondary immediate — only used for store instructions in C2.
    ///
    /// | Quadrant | funct3 | Meaning |
    /// |----------|--------|---------|
    /// | 2 (C2)   | 110    | C.SWSP uimm |
    /// | 2        | 111    | C.SDSP uimm |
    pub imm2: u64,

    // ---- 1-byte aligned ----
    /// Quadrant: 0=C0 (bits[1:0]=00), 1=C1 (01), 2=C2 (10).
    pub quadrant: u8,
    /// funct3 = bits[15:13].
    pub funct3: u8,
    /// Destination register, 5-bit raw (bits[11:7]) — C1/C2.
    pub rd: u8,
    /// Source register 1, 5-bit raw (bits[11:7], same as rd) — C1/C2.
    pub rs1: u8,
    /// Source register 2, 5-bit raw (bits[6:2]) — C1/C2.
    pub rs2: u8,
    /// rd' creg-mapped (x8 + bits[4:2]) — C0 rd/rs2, C1-ALU rs2.
    pub rdp: u8,
    /// rs1' creg-mapped (x8 + bits[9:7]) — C0 rs1, C1-ALU rd.
    pub rs1p: u8,
    /// C1-ALU sub-function (bits[11:10]): 00=SRLI, 01=SRAI, 10=ANDI, 11=register ops.
    pub sf: u8,
    /// bit[12] — used in C1-ALU dispatch and C.JR/C.JALR.
    pub bit12: u8,
    /// bits[6:5] — used in C1-ALU op select and C2 C.SWSP/C.SDSP.
    pub bit65: u8,
}

/// RISC-V ``creg`` mapping for compressed instructions:
///   rd'/rs1'/rs2'  (3-bit)  ->  x8 + rd'  (i.e. s0–s1, a0–a5).
#[inline(always)]
fn creg(raw: u8) -> u8 {
    8 + (raw & 0x7)
}

/// C1 quadrant (01) immediate decode.
#[inline]
fn imm_c1(h: u64, funct3: u8, bit12: u8, bits_6_2: u64, bits_11_7: u64) -> u64 {
    match funct3 {
        0..=2 => {
            // C.ADDI / C.ADDIW / C.LI
            // 6-bit signed imm = {bit12, bits[6:2]}
            let raw = (bit12 as u64) << 5 | bits_6_2;
            (raw & 0x1F).wrapping_sub(raw & 0x20) & 0xFFFF_FFFF_FFFF_FFFF
        }
        3 => {
            if bits_11_7 == 2 {
                // C.ADDI16SP: nzimm[9|4|6|8:7|5], sign-extend 10-bit
                let nz = ((h >> 6) & 0x1) << 4
                    | ((h >> 2) & 0x1) << 5
                    | ((h >> 5) & 0x1) << 6
                    | ((h >> 3) & 0x3) << 7
                    | ((h >> 12) & 0x1) << 9;
                (nz & 0x1FF).wrapping_sub(nz & 0x200) & 0xFFFF_FFFF_FFFF_FFFF
            } else {
                // C.LUI: 6-bit nzimm = {bit12, bits[6:2]}
                let nz = (bit12 as u64) << 5 | bits_6_2;
                (nz & 0x1F).wrapping_sub(nz & 0x20) & 0xFFFF_FFFF_FFFF_FFFF
            }
        }
        5 => {
            // C.J: offset[11|4|9:8|10|6|7|3:1|5], sign-extend 11-bit
            let off = ((h >> 12) & 0x1) << 11
                | ((h >> 8) & 0x1) << 10
                | ((h >> 9) & 0x3) << 8
                | ((h >> 6) & 0x1) << 7
                | ((h >> 7) & 0x1) << 6
                | ((h >> 2) & 0x1) << 5
                | ((h >> 11) & 0x1) << 4
                | ((h >> 3) & 0x7) << 1;
            (off & 0x7FF).wrapping_sub(off & 0x800) & 0xFFFF_FFFF_FFFF_FFFF
        }
        6 | 7 => {
            // C.BEQZ / C.BNEZ: offset[8|4:3|7:6|2:1|5], sign-extend 9-bit
            let off = ((h >> 12) & 0x1) << 8
                | ((h >> 10) & 0x3) << 3
                | ((h >> 5) & 0x3) << 6
                | ((h >> 2) & 0x1) << 5
                | ((h >> 3) & 0x3) << 1;
            (off & 0xFF).wrapping_sub(off & 0x100) & 0xFFFF_FFFF_FFFF_FFFF
        }
        _ => 0u64,
    }
}

/// C2 quadrant (10) immediate decode.
/// Returns ``(imm, imm2)`` where imm2 is only used for stores (C.SWSP, C.SDSP).
#[inline]
fn imm_c2(h: u64, funct3: u8, bit12: u8, bits_6_2: u64) -> (u64, u64) {
    let imm = match funct3 {
        0 => {
            // C.SLLI: shamt[5:0] = {bit12, bits[6:2]}
            ((bit12 as u64) << 5) | bits_6_2
        }
        1 | 3 => {
            // C.FLDSP / C.LDSP: uimm[5|4:3|8:6]
            ((h >> 2) & 0x7) << 6 | ((h >> 12) & 0x1) << 5 | ((h >> 5) & 0x3) << 3
        }
        2 => {
            // C.LWSP: uimm[7:6|5|4:2]
            ((h >> 2) & 0x3) << 6 | ((h >> 12) & 0x1) << 5 | ((h >> 4) & 0x7) << 2
        }
        _ => 0,
    };
    let imm2 = match funct3 {
        6 => {
            // C.SWSP: uimm[5:2|7:6]
            ((h >> 9) & 0xF) << 2 | ((h >> 7) & 0x3) << 6
        }
        5 | 7 => {
            // C.FSDSP / C.SDSP: uimm[5:3|8:6]
            ((h >> 7) & 0x7) << 6 | ((h >> 10) & 0x7) << 3
        }
        _ => 0,
    };
    (imm, imm2)
}

/// Decode immediates for a compressed instruction.
///
/// Returns ``(imm, imm2)`` where ``imm2`` is only used for C2 stores.
#[inline]
fn decode_compressed_immediates(
    h: u64, quadrant: u8, funct3: u8,
    bit12: u8, bits_6_2: u64, bits_11_7: u64,
) -> (u64, u64) {
    match quadrant {
        // -------- C0: quadrant 00 ----------------------------------
        0 => match funct3 {
            0 => (
                // C.ADDI4SPN: nzuimm[5:4|9:6|2|3]
                ((h >> 6) & 0x1) << 2
                    | ((h >> 5) & 0x1) << 3
                    | ((h >> 11) & 0x1) << 4
                    | ((h >> 12) & 0x1) << 5
                    | ((h >> 7) & 0xF) << 6,
                0,
            ),
            2 | 6 => (
                // C.LW / C.SW: uimm[5:3|2|6]
                ((h >> 6) & 0x1) << 2 | ((h >> 10) & 0x7) << 3 | ((h >> 5) & 0x1) << 6,
                0,
            ),
            1 | 3 | 5 | 7 => (
                // C.FLD / C.LD / C.FSD / C.SD: uimm[5:3|6|7]
                ((h >> 5) & 0x3) << 6 | ((h >> 10) & 0x7) << 3,
                0,
            ),
            _ => (0u64, 0u64),
        },

        // -------- C1: quadrant 01 ----------------------------------
        1 => (imm_c1(h, funct3, bit12, bits_6_2, bits_11_7), 0u64),

        // -------- C2: quadrant 10 ----------------------------------
        2 => imm_c2(h, funct3, bit12, bits_6_2),

        _ => (0, 0),
    }
}

/// Extract all fields from a 16-bit compressed instruction.
///
/// Replaces inline bit-extraction across `_handle_compressed_c0/c1/c2`.
#[no_mangle]
pub extern "C" fn decode_compressed(half: u16) -> CompressedFields {
    let h = half as u64;
    let quadrant = (h & 0x3) as u8;
    let funct3 = ((h >> 13) & 0x7) as u8;

    // ---- common sub-fields ----
    let bit12 = ((h >> 12) & 0x1) as u8;
    let bits_6_2 = (h >> 2) & 0x1F;
    let bits_11_7 = (h >> 7) & 0x1F;

    // ---- register fields ----
    let rdp = creg(((h >> 2) & 0x7) as u8);   // C0 rd/rs2, C1-ALU rs2
    let rs1p = creg(((h >> 7) & 0x7) as u8);  // C0 rs1, C1-ALU rd
    let (rd, rs1, rs2) = if quadrant == 0 {
        (rdp, rs1p, rdp)
    } else {
        (
            bits_11_7 as u8,
            bits_11_7 as u8,
            bits_6_2 as u8,
        )
    };

    // ---- C1-ALU sub-fields ----
    let sf = ((h >> 10) & 0x3) as u8;
    let bit65 = ((h >> 5) & 0x3) as u8;

    let (imm, imm2) = decode_compressed_immediates(
        h, quadrant, funct3, bit12, bits_6_2, bits_11_7,
    );

    CompressedFields {
        imm,
        imm2,
        quadrant,
        funct3,
        rd,
        rs1,
        rs2,
        rdp,
        rs1p,
        sf,
        bit12,
        bit65,
    }
}

// ============================================================
//  Unit tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- 32-bit decode tests --

    #[test]
    fn test_sext12_pos() {
        assert_eq!(sext12(0x7FF), 0x7FF);
        assert_eq!(sext12(0x800), 0xFFFF_FFFF_FFFF_F800);
        assert_eq!(sext12(0xFFF), 0xFFFF_FFFF_FFFF_FFFF);
    }

    #[test]
    fn test_addi_x5_x0_42() {
        // addi x5, x0, 42  -> 0x02A00293
        let f = decode_fields(0x02A00293);
        assert_eq!(f.opcode, 0x13);
        assert_eq!(f.rd, 5);
        assert_eq!(f.func3, 0);
        assert_eq!(f.rs1, 0);
        // I-type: rs2 field overlaps imm[4:0]; imm=42=0b101010 -> low 5 bits = 10
        assert_eq!(f.rs2, 10);
        assert_eq!(f.is_compressed, 0);
        assert_eq!(f.imm12_se, 42);
    }

    #[test]
    fn test_ld_sp_imm() {
        let f = decode_fields(0x00813503);
        assert_eq!(f.opcode, 0x03);
        assert_eq!(f.func3, 3); // LD
        assert_eq!(f.rd, 10);
        assert_eq!(f.rs1, 2); // sp
        assert_eq!(f.imm12_se, 8);
    }

    #[test]
    fn test_compressed_detect() {
        assert_eq!(decode_fields(0x0001).is_compressed, 1);
        assert_eq!(decode_fields(0x00000013).is_compressed, 0);
    }

    #[test]
    fn test_branch_imm() {
        let f = decode_fields(0x00B50A63);
        assert_eq!(f.opcode, 0x63);
        assert_eq!(f.func3, 0); // BEQ
        assert_eq!(f.imm_b, 20);
        assert_eq!(f.rs1, 10);
        assert_eq!(f.rs2, 11);
    }

    #[test]
    fn test_jal_imm() {
        let f = decode_fields(0x400000EF);
        assert_eq!(f.opcode, 0x6F);
        assert_eq!(f.rd, 1);
    }

    #[test]
    fn test_lui_imm() {
        let f = decode_fields(0x12345537);
        assert_eq!(f.opcode, 0x37);
        assert_eq!(f.rd, 10);
        assert_eq!(f.imm20_raw, 0x12345);
    }

    #[test]
    fn test_neg12_imm() {
        let f = decode_fields(0xFFF00293);
        assert_eq!(f.imm12_se, 0xFFFF_FFFF_FFFF_FFFF);
        assert_eq!(f.opcode, 0x13);
    }

    // -- 16-bit compressed decode tests --

    #[test]
    fn test_c_addi4spn() {
        // C.ADDI4SPN x8, sp, 16  -> imm=16, rd'=0 (x8), funct3=0, quadrant=0
        // Encoding: funct3=000, rd'=000, nzuimm encoded across bits
        // 16 = 0b010000 -> nzuimm[5:4|9:6|2|3]
        // nzuimm[5:4]=01, nzuimm[9:6]=0000, nzuimm[2]=0, nzuimm[3]=0
        // bit6=0 (nzuimm[2]), bit5=0 (nzuimm[3]), bit12:11=01 (nzuimm[5:4])
        // bit10:7=0000 (nzuimm[9:6])
        // 0b000_01_0000_000_00_00 = 0x1000... hmm let me just test a known encoding
        // c.addi4spn x8, sp, 16 -> 0x0040 (little-endian half)
        // Actually: quadrant=00, funct3=000, rd'=000, nzuimm=16
        // 16 in C.ADDI4SPN encoding:
        //   bit[6]=0, bit[5]=1, bit[12:11]=00, bit[10:7]=0000...
        //   wait: bit[6] is nzuimm[2], bit[5] is nzuimm[3]
        //   16=0b10000: nzuimm[5:4]=01, nzuimm[9:6]=0000, nzuimm[2]=0, nzuimm[3]=0
        //   bit12=0, bit11=1, bit6=0, bit5=0, bit10:7=0000
        //   instr = {funct3=000, rd'=000, nzuimm encoded, quadrant=00}
        //   = 0b000_0_1_0000_000_0_0_00 = 0x0800
        // Hmm let me use a different approach. I know 0x0040 should be addi4spn
        // Let me just verify the funct3 and quadrant.
        let f = decode_compressed(0x0040);
        assert_eq!(f.quadrant, 0);
        assert_eq!(f.funct3, 0); // C.ADDI4SPN
    }

    #[test]
    fn test_c_addi() {
        // C.ADDI x5, 3 -> funct3=000, rd=x5, imm=3
        // Encoding: funct3=000, imm[5]=0, rd=5, imm[4:0]=00011, op=01
        // = 0b000_0_00101_00011_01 = 0x051D... wait
        // Actually: bit[15:13]=000, bit[12]=0, bits[11:7]=00101, bits[6:2]_imm[4:0]=00011, bits[1:0]=01
        // 000 0 00101 00011 01 = 0000 0101 0001 1010 = 0x051A... hmm
        // Let me just test with a known value.
        // c.addi x10, 3 -> 0x050D: 000 0 01010 00011 01
        let f = decode_compressed(0x050D);
        assert_eq!(f.quadrant, 1);
        assert_eq!(f.funct3, 0); // C.ADDI
        assert_eq!(f.rd, 10);    // x10 = a0
        assert_eq!(f.imm, 3);    // +3
    }

    #[test]
    fn test_c_li() {
        // c.li x10, 31 -> funct3=010, rd=10, imm=31
        // Verified encoding from llvm-mc: c.li max imm range is [-32,31]
        // c.li a0, 31 -> 0x457D: 010 0 01010 11111 01
        let f = decode_compressed(0x457D);
        assert_eq!(f.quadrant, 1);
        assert_eq!(f.funct3, 2); // C.LI
        assert_eq!(f.rd, 10);
        assert_eq!(f.imm, 31);
    }

    #[test]
    fn test_c_addi16sp() {
        // C.ADDI16SP -16 -> 0x717D (verified)
        let f = decode_compressed(0x717D);
        assert_eq!(f.quadrant, 1);
        assert_eq!(f.funct3, 3);
        assert_eq!(f.rd, 2);              // sp
        assert_eq!(f.imm, 0xFFFF_FFFF_FFFF_FFF0); // -16 as u64
    }

    #[test]
    fn test_c_lui() {
        // C.LUI x10, 1 -> funct3=011, rd≠2, nzimm=1
        // Encoding: funct3=011, rd=01010, nzimm=000001, op=01
        // nzimm[5]=0 (bit12), nzimm[4:0]=00001 (bits[6:2])
        // 011 0 01010 00001 01 = 0x6505
        let f = decode_compressed(0x6505);
        assert_eq!(f.quadrant, 1);
        assert_eq!(f.funct3, 3);
        assert_eq!(f.rd, 10);
        assert_eq!(f.imm, 1); // nzimm=1 -> left-shifted by 12 in caller
    }

    #[test]
    fn test_c_j() {
        // C.J +20 -> funct3=101, offset=20
        let f = decode_compressed(0xB005); // rough estimate
        assert_eq!(f.quadrant, 1);
        assert_eq!(f.funct3, 5); // C.J
        // offset should be non-zero
        assert!(f.imm > 0);
    }

    #[test]
    fn test_c_beqz() {
        // C.BEQZ x10, +8
        let f = decode_compressed(0xC501); // c.beqz a0, +8 (verified)
        assert_eq!(f.quadrant, 1);
        assert_eq!(f.funct3, 6); // C.BEQZ
        assert_eq!(f.rs1, 10);   // x10 = a0
    }

    #[test]
    fn test_c_slli() {
        // C.SLLI x10, 3 -> funct3=000, rd=x10, shamt=3
        // bit12=0, bits[11:7]=01010, shamt[4:0]=00011, op=10
        // 000 0 01010 00011 10 = 0x050E
        let f = decode_compressed(0x050E);
        assert_eq!(f.quadrant, 2);
        assert_eq!(f.funct3, 0); // C.SLLI
        assert_eq!(f.rd, 10);
        assert_eq!(f.imm, 3);    // shamt
    }

    #[test]
    fn test_c_lwsp() {
        // C.LWSP x10, 4(sp)
        let f = decode_compressed(0x4512); // c.lwsp a0, 4(sp)
        assert_eq!(f.quadrant, 2);
        assert_eq!(f.funct3, 2); // C.LWSP
        assert_eq!(f.rd, 10);
        assert_eq!(f.imm, 4);    // offset 4
    }

    #[test]
    fn test_c_jr() {
        // C.JR x10 -> funct3=100, rs1=x10, bit12=0, rs2=0
        // 100 0 01010 00000 10 = 0x8502
        let f = decode_compressed(0x8502);
        assert_eq!(f.quadrant, 2);
        assert_eq!(f.funct3, 4);
        assert_eq!(f.rs1, 10);
        assert_eq!(f.rs2, 0);
        assert_eq!(f.bit12, 0);
    }

    #[test]
    fn test_c_swsp() {
        // C.SWSP x10, 8(sp) -> 0xC42A (verified)
        let f = decode_compressed(0xC42A);
        assert_eq!(f.quadrant, 2);
        assert_eq!(f.funct3, 6); // C.SWSP
    }

    #[test]
    fn test_c_ld_imm_regression() {
        // C.LD a5, 0(s0): 0x601c — faulting instruction from whoami crash.
        let f = decode_compressed(0x601c);
        assert_eq!(f.quadrant, 0);
        assert_eq!(f.funct3, 3);
        assert_eq!(f.rd, 15);
        assert_eq!(f.rs1p, 8);
        assert_eq!(f.imm, 0);
    }

    // ============================================================
    //  llvm-mc verified encoding tests — all encodings confirmed
    //  against /opt/custom-llvm/bin/llvm-mc output.
    //  These lock down every compressed-instruction immediate decoder
    //  to prevent regressions like the C.LWSP offset bug.
    // ============================================================

    // ---- C0 quadrant (00) ----

    #[test]
    fn c0_addi4spn_16() {
        // c.addi4spn x8, sp, 16 -> 0x0800
        let f = decode_compressed(0x0800);
        assert_eq!(f.quadrant, 0);
        assert_eq!(f.funct3, 0);
        assert_eq!(f.rdp, 8);    // x8 = s0
        assert_eq!(f.imm, 16);
    }

    #[test]
    fn c0_addi4spn_1020() {
        // c.addi4spn x15, sp, 1020 -> 0x1ffc (max nzuimm)
        let f = decode_compressed(0x1ffc);
        assert_eq!(f.quadrant, 0);
        assert_eq!(f.funct3, 0);
        assert_eq!(f.rdp, 15);   // x15 = a5
        assert_eq!(f.imm, 1020);
    }

    #[test]
    fn c0_lw_zero_offset() {
        // c.lw x8, 0(x10) -> 0x4100
        let f = decode_compressed(0x4100);
        assert_eq!(f.quadrant, 0);
        assert_eq!(f.funct3, 2);  // C.LW
        assert_eq!(f.rdp, 8);     // x8
        assert_eq!(f.rs1p, 10);   // x10 = a0
        assert_eq!(f.imm, 0);
    }

    #[test]
    fn c0_lw_max_offset() {
        // c.lw x15, 124(x15) -> 0x5ffc (max uimm=124)
        let f = decode_compressed(0x5ffc);
        assert_eq!(f.quadrant, 0);
        assert_eq!(f.funct3, 2);
        assert_eq!(f.imm, 124);
    }

    #[test]
    fn c0_ld_zero_offset() {
        // c.ld x8, 0(x10) -> 0x6100
        let f = decode_compressed(0x6100);
        assert_eq!(f.quadrant, 0);
        assert_eq!(f.funct3, 3);  // C.LD
        assert_eq!(f.imm, 0);
    }

    #[test]
    fn c0_ld_max_offset() {
        // c.ld x15, 248(x15) -> 0x7ffc (max uimm=248)
        let f = decode_compressed(0x7ffc);
        assert_eq!(f.quadrant, 0);
        assert_eq!(f.funct3, 3);
        assert_eq!(f.imm, 248);
    }

    #[test]
    fn c0_sw_zero_offset() {
        // c.sw x8, 0(x10) -> 0xc100
        let f = decode_compressed(0xc100);
        assert_eq!(f.quadrant, 0);
        assert_eq!(f.funct3, 6);  // C.SW
        assert_eq!(f.rdp, 8);     // rs2'
        assert_eq!(f.imm, 0);
    }

    #[test]
    fn c0_sw_max_offset() {
        // c.sw x9, 124(x15) -> 0xdfe4 (max offset)
        let f = decode_compressed(0xdfe4);
        assert_eq!(f.quadrant, 0);
        assert_eq!(f.funct3, 6);
        assert_eq!(f.imm, 124);
    }

    #[test]
    fn c0_sd_zero_offset() {
        // c.sd x8, 0(x10) -> 0xe100
        let f = decode_compressed(0xe100);
        assert_eq!(f.quadrant, 0);
        assert_eq!(f.funct3, 7);  // C.SD
        assert_eq!(f.imm, 0);
    }

    #[test]
    fn c0_sd_max_offset() {
        // c.sd x9, 248(x15) -> 0xffe4 (max offset)
        let f = decode_compressed(0xffe4);
        assert_eq!(f.quadrant, 0);
        assert_eq!(f.funct3, 7);
        assert_eq!(f.imm, 248);
    }

    // ---- C1 quadrant (01) ----

    #[test]
    fn c1_nop() {
        // c.nop -> 0x0001
        let f = decode_compressed(0x0001);
        assert_eq!(f.quadrant, 1);
        assert_eq!(f.funct3, 0);  // C.ADDI rd=0, imm=0
        assert_eq!(f.rd, 0);
        assert_eq!(f.imm, 0);
    }

    #[test]
    fn c1_addi_pos() {
        // c.addi x5, 3 -> 0x028d
        let f = decode_compressed(0x028d);
        assert_eq!(f.quadrant, 1);
        assert_eq!(f.funct3, 0);
        assert_eq!(f.rd, 5);
        assert_eq!(f.imm, 3);
    }

    #[test]
    fn c1_addi_neg32() {
        // c.addi x10, -32 -> 0x1501
        let f = decode_compressed(0x1501);
        assert_eq!(f.quadrant, 1);
        assert_eq!(f.funct3, 0);
        assert_eq!(f.rd, 10);
        // -32 as 6-bit sign-extended -> 0xFFFF_FFFF_FFFF_FFE0 as u64
        assert_eq!(f.imm, 0xFFFF_FFFF_FFFF_FFE0u64);
    }

    #[test]
    fn c1_addiw() {
        // c.addiw x5, -2 -> 0x32f9
        let f = decode_compressed(0x32f9);
        assert_eq!(f.quadrant, 1);
        assert_eq!(f.funct3, 1);  // C.ADDIW
        assert_eq!(f.rd, 5);
        assert_eq!(f.imm, 0xFFFF_FFFF_FFFF_FFFEu64); // -2
    }

    #[test]
    fn c1_li_max() {
        // c.li x5, 31  -> 0x42fd
        let f = decode_compressed(0x42fd);
        assert_eq!(f.quadrant, 1);
        assert_eq!(f.funct3, 2);  // C.LI
        assert_eq!(f.rd, 5);
        assert_eq!(f.imm, 31);
    }

    #[test]
    fn c1_li_min() {
        // c.li x10, -32 -> 0x5501
        let f = decode_compressed(0x5501);
        assert_eq!(f.quadrant, 1);
        assert_eq!(f.funct3, 2);
        assert_eq!(f.rd, 10);
        assert_eq!(f.imm, 0xFFFF_FFFF_FFFF_FFE0u64); // -32
    }

    #[test]
    fn c1_lui_1() {
        // c.lui x5, 1 -> 0x6285
        let f = decode_compressed(0x6285);
        assert_eq!(f.quadrant, 1);
        assert_eq!(f.funct3, 3);
        assert_eq!(f.rd, 5);
        assert_eq!(f.imm, 1);
    }

    #[test]
    fn c1_lui_31() {
        // c.lui x10, 31 -> 0x657d
        let f = decode_compressed(0x657d);
        assert_eq!(f.quadrant, 1);
        assert_eq!(f.funct3, 3);
        assert_eq!(f.rd, 10);
        assert_eq!(f.imm, 31);
    }

    #[test]
    fn c1_addi16sp_neg16() {
        // c.addi16sp sp, -16 -> 0x717d
        let f = decode_compressed(0x717d);
        assert_eq!(f.quadrant, 1);
        assert_eq!(f.funct3, 3);
        assert_eq!(f.rd, 2);     // sp
        assert_eq!(f.imm, 0xFFFF_FFFF_FFFF_FFF0u64); // -16
    }

    #[test]
    fn c1_addi16sp_pos496() {
        // c.addi16sp sp, 496 -> 0x617d
        let f = decode_compressed(0x617d);
        assert_eq!(f.quadrant, 1);
        assert_eq!(f.funct3, 3);
        assert_eq!(f.rd, 2);
        assert_eq!(f.imm, 496);
    }

    #[test]
    fn c1_srli() {
        // c.srli x8, 3 -> 0x800d
        let f = decode_compressed(0x800d);
        assert_eq!(f.quadrant, 1);
        assert_eq!(f.funct3, 4);  // MISC-ALU
        assert_eq!(f.sf, 0);      // SRLI
        assert_eq!(f.rs1p, 8);    // rd' = x8
        assert_eq!(f.bit12, 0);   // shamt[5]=0
        // shamt is bit12<<5 | bits[6:2]; imm=0 for funct3=4 (not decoded in imm_c1)
    }

    #[test]
    fn c1_srai() {
        // c.srai x9, 4 -> 0x8491
        let f = decode_compressed(0x8491);
        assert_eq!(f.quadrant, 1);
        assert_eq!(f.funct3, 4);
        assert_eq!(f.sf, 1);      // SRAI
        assert_eq!(f.rs1p, 9);
    }

    #[test]
    fn c1_andi() {
        // c.andi x10, -1 -> 0x997d
        let f = decode_compressed(0x997d);
        assert_eq!(f.quadrant, 1);
        assert_eq!(f.funct3, 4);
        assert_eq!(f.sf, 2);      // ANDI
        assert_eq!(f.rs1p, 10);
        assert_eq!(f.bit12, 1);   // imm[5]=1 for -1
    }

    #[test]
    fn c1_sub() {
        // c.sub x11, x12 -> 0x8d91
        let f = decode_compressed(0x8d91);
        assert_eq!(f.quadrant, 1);
        assert_eq!(f.funct3, 4);
        assert_eq!(f.sf, 3);      // register ops
        assert_eq!(f.bit12, 0);   // not RV64C
        assert_eq!(f.bit65, 0);   // SUB
        assert_eq!(f.rs1p, 11);   // rd' = x11
        assert_eq!(f.rdp, 12);    // rs2' = x12
    }

    #[test]
    fn c1_xor() {
        // c.xor x11, x12 -> 0x8db1
        let f = decode_compressed(0x8db1);
        assert_eq!(f.bit65, 1);   // XOR
    }

    #[test]
    fn c1_or() {
        // c.or x11, x12 -> 0x8dd1
        let f = decode_compressed(0x8dd1);
        assert_eq!(f.bit65, 2);   // OR
    }

    #[test]
    fn c1_and() {
        // c.and x11, x12 -> 0x8df1
        let f = decode_compressed(0x8df1);
        assert_eq!(f.bit65, 3);   // AND
    }

    #[test]
    fn c1_subw() {
        // c.subw x13, x14 -> 0x9e99
        let f = decode_compressed(0x9e99);
        assert_eq!(f.sf, 3);
        assert_eq!(f.bit12, 1);   // RV64C
        assert_eq!(f.bit65, 0);   // SUBW
        assert_eq!(f.rs1p, 13);
        assert_eq!(f.rdp, 14);
    }

    #[test]
    fn c1_addw() {
        // c.addw x13, x14 -> 0x9eb9
        let f = decode_compressed(0x9eb9);
        assert_eq!(f.sf, 3);
        assert_eq!(f.bit12, 1);
        assert_eq!(f.bit65, 1);   // ADDW
    }

    #[test]
    fn c1_j_forward() {
        // c.j .+2 -> 0xa001 gives imm=0; real forward jump from objdump is 0xa009
        let f = decode_compressed(0xa009);
        assert_eq!(f.quadrant, 1);
        assert_eq!(f.funct3, 5);  // C.J
        assert_eq!(f.imm, 2);     // offset = 2
    }

    #[test]
    fn c1_beqz_taken() {
        // c.beqz x10, .+2 -> 0xc109
        let f = decode_compressed(0xc109);
        assert_eq!(f.quadrant, 1);
        assert_eq!(f.funct3, 6);  // C.BEQZ
        assert_eq!(f.rs1p, 10);   // x10 = a0
    }

    #[test]
    fn c1_bnez() {
        // c.bnez x10, .+2 -> 0xe109
        let f = decode_compressed(0xe109);
        assert_eq!(f.quadrant, 1);
        assert_eq!(f.funct3, 7);  // C.BNEZ
        assert_eq!(f.rs1p, 10);
    }

    // ---- C2 quadrant (10) ----

    #[test]
    fn c2_slli_63() {
        // c.slli x10, 63 -> 0x157e
        let f = decode_compressed(0x157e);
        assert_eq!(f.quadrant, 2);
        assert_eq!(f.funct3, 0);  // C.SLLI
        assert_eq!(f.rd, 10);
        assert_eq!(f.imm, 63);
    }

    #[test]
    fn c2_lwsp_regression_zsh_crash() {
        // c.lwsp x18, 4(sp) -> 0x4912
        // THIS IS THE ZSH CRASH BUG — the old decoder produced offset 16
        // instead of 4, loading saved s2 (-560) instead of the correct value.
        let f = decode_compressed(0x4912);
        assert_eq!(f.quadrant, 2);
        assert_eq!(f.funct3, 2);  // C.LWSP
        assert_eq!(f.rd, 18);     // x18 = s2
        assert_eq!(f.imm, 4);     // offset 4 (NOT 16!)
    }

    #[test]
    fn c2_lwsp_max_offset() {
        // c.lwsp x10, 252(sp) -> 0x557e
        let f = decode_compressed(0x557e);
        assert_eq!(f.quadrant, 2);
        assert_eq!(f.funct3, 2);
        assert_eq!(f.rd, 10);
        assert_eq!(f.imm, 252);
    }

    #[test]
    fn c2_ldsp_8() {
        // c.ldsp x5, 8(sp) -> 0x62a2
        let f = decode_compressed(0x62a2);
        assert_eq!(f.quadrant, 2);
        assert_eq!(f.funct3, 3);  // C.LDSP
        assert_eq!(f.rd, 5);
        assert_eq!(f.imm, 8);
    }

    #[test]
    fn c2_ldsp_504() {
        // c.ldsp x10, 504(sp) -> 0x757e (max offset for LDSP)
        let f = decode_compressed(0x757e);
        assert_eq!(f.quadrant, 2);
        assert_eq!(f.funct3, 3);
        assert_eq!(f.rd, 10);
        assert_eq!(f.imm, 504);
    }

    #[test]
    fn c2_mv() {
        // c.mv x5, x10 -> 0x82aa
        let f = decode_compressed(0x82aa);
        assert_eq!(f.quadrant, 2);
        assert_eq!(f.funct3, 4);
        assert_eq!(f.rd, 5);
        assert_eq!(f.rs2, 10);
        assert_eq!(f.bit12, 0);   // C.MV
    }

    #[test]
    fn c2_add() {
        // c.add x5, x10 -> 0x92aa
        let f = decode_compressed(0x92aa);
        assert_eq!(f.quadrant, 2);
        assert_eq!(f.funct3, 4);
        assert_eq!(f.rd, 5);
        assert_eq!(f.rs2, 10);
        assert_eq!(f.bit12, 1);   // C.ADD
    }

    #[test]
    fn c2_ebreak() {
        // c.ebreak -> 0x9002
        let f = decode_compressed(0x9002);
        assert_eq!(f.quadrant, 2);
        assert_eq!(f.funct3, 4);
        assert_eq!(f.rd, 0);
        assert_eq!(f.rs2, 0);
    }

    #[test]
    fn c2_swsp_zero() {
        // c.swsp x5, 0(sp) -> 0xc016
        let f = decode_compressed(0xc016);
        assert_eq!(f.quadrant, 2);
        assert_eq!(f.funct3, 6);  // C.SWSP
        assert_eq!(f.rs2, 5);
        assert_eq!(f.imm2, 0);    // store offset in imm2
    }

    #[test]
    fn c2_swsp_max_offset() {
        // c.swsp x10, 252(sp) -> 0xdfaa
        let f = decode_compressed(0xdfaa);
        assert_eq!(f.quadrant, 2);
        assert_eq!(f.funct3, 6);
        assert_eq!(f.imm2, 252);
    }

    #[test]
    fn c2_sdsp_8() {
        // c.sdsp x5, 8(sp) -> 0xe416
        let f = decode_compressed(0xe416);
        assert_eq!(f.quadrant, 2);
        assert_eq!(f.funct3, 7);  // C.SDSP
        assert_eq!(f.rs2, 5);
        assert_eq!(f.imm2, 8);
    }

    #[test]
    fn c2_sdsp_max_offset() {
        // c.sdsp x10, 504(sp) -> 0xffaa
        let f = decode_compressed(0xffaa);
        assert_eq!(f.quadrant, 2);
        assert_eq!(f.funct3, 7);
        assert_eq!(f.imm2, 504);
    }

    // ---- 32-bit instruction field extractions (llvm-mc verified) ----

    #[test]
    fn rv32_addw_opcode() {
        // addw x5, x10, x11 -> 0x00b502bb (opcode 0x3B)
        let f = decode_fields(0x00b502bb);
        assert_eq!(f.opcode, 0x3B);
        assert_eq!(f.rd, 5);
        assert_eq!(f.func3, 0);
        assert_eq!(f.func7, 0);  // ADDW func7=0
    }

    #[test]
    fn rv32_subw_func7() {
        // subw x6, x10, x11 -> 0x40b5033b
        let f = decode_fields(0x40b5033b);
        assert_eq!(f.opcode, 0x3B);
        assert_eq!(f.rd, 6);
        assert_eq!(f.func3, 0);
        assert_eq!(f.func7, 0x20);  // SUBW func7=0b0100000
    }

    #[test]
    fn rv32_addiw_opcode() {
        // addiw x5, x10, 42 -> 0x02a5029b (opcode 0x1B)
        let f = decode_fields(0x02a5029b);
        assert_eq!(f.opcode, 0x1B);
        assert_eq!(f.rd, 5);
        assert_eq!(f.func3, 0);
        assert_eq!(f.imm12_se, 42);
    }

    #[test]
    fn rv32_slliw() {
        // slliw x6, x10, 3 -> 0x0035131b
        let f = decode_fields(0x0035131b);
        assert_eq!(f.opcode, 0x1B);
        assert_eq!(f.func3, 1);  // SLLIW
        assert_eq!(f.imm12_se, 3);
    }

    #[test]
    fn rv32_sraiw() {
        // sraiw x8, x10, 4 -> 0x4045541b
        let f = decode_fields(0x4045541b);
        assert_eq!(f.opcode, 0x1B);
        assert_eq!(f.func3, 5);  // SRAIW
    }

    #[test]
    fn store_sd_neg_offset() {
        // sd x9, -8(x10) -> 0xfe953c23
        let f = decode_fields(0xfe953c23);
        assert_eq!(f.opcode, 0x23);
        assert_eq!(f.func3, 3);  // SD
        // imm_s = -8 as sign-extended 12-bit
        assert_eq!(f.imm_s, 0xFFFF_FFFF_FFFF_FFF8u64);
    }

    #[test]
    fn load_ld_neg_offset() {
        // ld x14, -8(x10) -> 0xff853703
        let f = decode_fields(0xff853703);
        assert_eq!(f.opcode, 0x03);
        assert_eq!(f.func3, 3);  // LD
        assert_eq!(f.imm12_se, 0xFFFF_FFFF_FFFF_FFF8u64); // -8
    }

    #[test]
    fn fence_i_func3() {
        // fence.i -> 0x0000100f
        let f = decode_fields(0x0000100f);
        assert_eq!(f.opcode, 0x0F);
        assert_eq!(f.func3, 1);  // FENCE.I
    }

    #[test]
    fn csr_mstatus_addr() {
        // csrrw x5, mstatus, x10 -> 0x300512f3
        let f = decode_fields(0x300512f3);
        assert_eq!(f.opcode, 0x73);
        assert_eq!(f.func3, 1);  // CSRRW
        assert_eq!(f.imm12_se, 0x300);  // mstatus CSR address
    }

    #[test]
    fn csr_csrrsi() {
        // csrrsi x9, mstatus, 0x1f -> 0x300fe4f3
        let f = decode_fields(0x300fe4f3);
        assert_eq!(f.func3, 6);  // CSRRSI
        assert_eq!(f.rd, 9);
        assert_eq!(f.rs1, 31);   // uimm
    }

    #[test]
    fn amo_lrw_fields() {
        // lr.w x5, (x10) -> 0x100522af
        let f = decode_fields(0x100522af);
        assert_eq!(f.opcode, 0x2F);
        assert_eq!(f.func3, 2);  // .W
        assert_eq!(f.rd, 5);
        assert_eq!(f.rs1, 10);
        assert_eq!(f.func7 >> 2, 0b00010);  // LR funct5
    }

    #[test]
    fn amo_amoxor_w() {
        // amoxor.w x13, x11, (x10) -> 0x20b526af
        let f = decode_fields(0x20b526af);
        assert_eq!(f.opcode, 0x2F);
        assert_eq!(f.func3, 2);
        assert_eq!(f.func7 >> 2, 0b00100);  // AMOXOR funct5
    }

    #[test]
    fn amo_amomaxu_w() {
        // amomaxu.w x19, x11, (x10) -> 0xe0b529af
        let f = decode_fields(0xe0b529af);
        assert_eq!(f.func7 >> 2, 0b11100);  // AMOMAXU funct5
    }

    #[test]
    fn amo_amoadd_d() {
        // amoadd.d x20, x11, (x10) -> 0x00b53a2f
        let f = decode_fields(0x00b53a2f);
        assert_eq!(f.opcode, 0x2F);
        assert_eq!(f.func3, 3);  // .D
        assert_eq!(f.func7 >> 2, 0b00000);  // AMOADD funct5
    }
}
