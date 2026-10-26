//! 单次 FFI 调用完成对 RISC-V 指令字段的解码。
//!
//! 每条 32 位指令与每条 16 位压缩指令都只经一次 FFI 调用完成解码。

// ============================================================
//  32 位指令解码
// ============================================================

/// 一次提取全部标准 RISC-V 指令字段。
///
/// 字段顺序与 ctypes 的 `_fields_` 列表一致, 按对齐宽度从大到小排列,
/// 使结构体布局在 FFI 边界两侧逐字节一致。
#[repr(C)]
pub struct DecodedFields {
	// ---- 8 字节对齐 ----
	pub imm12_se: u64,
	pub imm_s: u64,
	pub imm_b: u64,
	pub imm_j: u64,
	pub imm20_raw: u64,

	// ---- 2 字节对齐 ----
	pub func12: u16,

	// ---- 1 字节对齐 ----
	pub opcode: u8,
	pub rd: u8,
	pub func3: u8,
	pub rs1: u8,
	pub rs2: u8,
	pub func7: u8,
	pub is_compressed: u8,
	// ---- F 与 D 扩展的浮点字段: FMA 的 rs3 与 fmt ----
	/// rs3 寄存器号, 取自 FMA 指令的 bits[31:27]。
	pub rs3: u8,
	/// 浮点格式, 取自 bits[26:25]: 取值 0、1、2、3 依次对应 S、D、H、Q。
	pub fmt: u8,
}

/// 将 12 位值符号扩展为 u64。
#[inline(always)]
fn sext12(val: u64) -> u64 {
	(val & 0x7ff).wrapping_sub(val & 0x800) & 0xffff_ffff_ffff_ffff
}

/// 将 13 位值符号扩展为 u64。
#[inline(always)]
fn sext13(val: u64) -> u64 {
	(val & 0xfff).wrapping_sub(val & 0x1000) & 0xffff_ffff_ffff_ffff
}

/// 将 21 位值符号扩展为 u64。
#[inline(always)]
fn sext21(val: u64) -> u64 {
	(val & 0xf_ffff).wrapping_sub(val & 0x10_0000) & 0xffff_ffff_ffff_ffff
}

/// 从 32 位指令中提取全部标准 RISC-V 字段。每条指令经 ctypes 调用一次。
#[no_mangle]
pub extern "C" fn decode_fields(instr: u32) -> DecodedFields {
	let instr = instr as u64;

	DecodedFields {
		// 立即数, 8 字节对齐
		imm12_se: sext12((instr >> 20) & 0xfff),
		imm_s: sext12((((instr >> 25) & 0x7f) << 5) | ((instr >> 7) & 0x1f)),
		imm_b: sext13(
			(((instr >> 31) & 1) << 12) | // imm[12]
				(((instr >> 25) & 0x3f) << 5) | // imm[10:5]
				(((instr >> 8) & 0xf) << 1) | // imm[4:1]
				(((instr >> 7) & 1) << 11), // imm[11]
		),
		imm_j: sext21(
			(((instr >> 31) & 1) << 20) | // imm[20]
				(((instr >> 21) & 0x3ff) << 1) | // imm[10:1]
				(((instr >> 20) & 1) << 11) | // imm[11]
				(((instr >> 12) & 0xff) << 12), // imm[19:12]
		),
		imm20_raw: (instr >> 12) & 0xf_ffff,

		// func12, 2 字节对齐
		func12: ((instr >> 20) & 0xfff) as u16,

		// 标量字段, 1 字节对齐
		opcode: (instr & 0x7f) as u8,
		rd: ((instr >> 7) & 0x1f) as u8,
		func3: ((instr >> 12) & 0x7) as u8,
		rs1: ((instr >> 15) & 0x1f) as u8,
		rs2: ((instr >> 20) & 0x1f) as u8,
		func7: ((instr >> 25) & 0x7f) as u8,
		is_compressed: u8::from((instr & 0x3) != 3),
		// F/D: rs3 取自 bits[31:27], fmt 取自 bits[26:25]
		rs3: ((instr >> 27) & 0x1f) as u8,
		fmt: ((instr >> 25) & 0x3) as u8,
	}
}

// ============================================================
//  16 位压缩指令解码
// ============================================================

/// 一次提取全部压缩指令的寄存器字段与立即数字段。
///
/// Python 侧处理函数按 `quadrant` 与 `funct3` 的取值选取所需字段。
#[repr(C)]
pub struct CompressedFields {
	// ---- 8 字节对齐 ----
	/// `imm` 字段, 有效含义由 `quadrant` 与 `funct3` 的取值决定。
	///
	/// | quadrant |   funct3   | 含义 |
	/// |----------|------------|------|
	/// |    0     |    000     | C.ADDI4SPN 的 nzuimm |
	/// |    0     |   {10}01   | C.FLD 与 C.FSD 的 uimm |
	/// |    0     |   {10}10   | C.LW 与 C.SW 的 uimm |
	/// |    0     |   {10}11   | C.LD 与 C.SD 的 uimm |
	/// |    1     | 000 ~ 010  | C.ADDI、C.ADDIW 与 C.LI 的 6 位有符号立即数 |
	/// |    1     | 011, rd!=2 | C.LUI 的 6 位 nzimm, 由调用方左移 12 位 |
	/// |    1     | 011, rd==2 | C.ADDI16SP 的 10 位 nzimm |
	/// |    1     |    101     | C.J 的 11 位偏移量 |
	/// |    1     |   11{10}   | C.BEQZ 与 C.BNEZ 的 8 位偏移量 |
	/// |    2     |    000     | C.SLLI 的 shamt[5:0] |
	/// |    2     |   0{10}1   | C.FLDSP 与 C.LDSP 的 uimm |
	/// |    2     |    010     | C.LWSP 的 uimm |
	pub imm: u64,

	/// `imm2` 字段, 仅用于 `quadrant` 取值为 2 的存储指令。
	///
	/// | quadrant | funct3 | 含义 |
	/// |----------|--------|------|
	/// | 2        | 1{10}1 | C.FSDSP 与 C.SDSP 的 uimm |
	/// | 2        | 110    | C.SWSP 的 uimm |
	pub imm2: u64,

	// ---- 1 字节对齐 ----
	/// quadrant, 取自 bits[1:0]: 取值 0、1、2 依次对应 C0、C1、C2。
	pub quadrant: u8,
	/// funct3, 取自 bits[15:13]。
	pub funct3: u8,
	/// 目标寄存器, bits[11:7] 的 5 位原始值, 用于 C1 与 C2。
	pub rd: u8,
	/// 源寄存器 1, bits[11:7] 的 5 位原始值, 与 rd 取值相同, 用于 C1 与 C2。
	pub rs1: u8,
	/// 源寄存器 2, bits[6:2] 的 5 位原始值, 用于 C1 与 C2。
	pub rs2: u8,
	/// rd' 经 creg 映射后的值, 即 x8 + bits[4:2]; C0 中作为 rd 与 rs2, C1-ALU 中作为 rs2。
	pub rdp: u8,
	/// rs1' 经 creg 映射后的值, 即 x8 + bits[9:7]; C0 中作为 rs1, C1-ALU 中作为 rd。
	pub rs1p: u8,
	/// C1-ALU 的子功能, 取自 bits[11:10]: 00 为 SRLI, 01 为 SRAI, 10 为 ANDI, 11 为寄存器运算。
	pub sf: u8,
	/// bit[12], 用于 C1-ALU 的分发以及 C.JR 与 C.JALR。
	pub bit12: u8,
	/// bits[6:5], 用于 C1-ALU 的运算选择以及 C.SWSP 与 C.SDSP。
	pub bit65: u8,
}

/// RISC-V 压缩指令的 ``creg`` 映射: 3 位编码加 8 得到实际寄存器号。
/// 映射结果对应 s0、s1 与 a0 至 a5 这 8 个寄存器。
#[inline(always)]
fn creg(raw: u8) -> u8 {
	8 + (raw & 0x7)
}

/// 解码 bits[1:0] 为 0 的压缩指令的立即数。
///
/// 该组的立即数全部为无符号的 uimm 或 nzuimm, 不做符号扩展。
#[inline]
fn imm_c0(h: u64, funct3: u8) -> u64 {
	match funct3 {
		0 => {
			// C.ADDI4SPN: nzuimm[5:4|9:6|2|3]
			(((h >> 6) & 0x1) << 2)
				| (((h >> 5) & 0x1) << 3)
				| (((h >> 11) & 0x1) << 4)
				| (((h >> 12) & 0x1) << 5)
				| (((h >> 7) & 0xf) << 6)
		}
		2 | 6 => {
			// C.LW 与 C.SW: uimm[5:3|2|6]
			(((h >> 6) & 0x1) << 2) | (((h >> 10) & 0x7) << 3) | (((h >> 5) & 0x1) << 6)
		}
		1 | 3 | 5 | 7 => {
			// C.FLD、C.LD、C.FSD 与 C.SD: uimm[5:3|6|7]
			(((h >> 5) & 0x3) << 6) | (((h >> 10) & 0x7) << 3)
		}
		_ => 0u64,
	}
}

/// 解码 bits[1:0] 为 1 的压缩指令的立即数。
#[inline]
fn imm_c1(h: u64, funct3: u8, bit12: u8, bits_6_2: u64, bits_11_7: u64) -> u64 {
	match funct3 {
		0..=2 => {
			// C.ADDI、C.ADDIW 与 C.LI
			// 6 位有符号立即数, 由 bit12 与 bits[6:2] 拼接
			let raw = ((bit12 as u64) << 5) | bits_6_2;
			(raw & 0x1f).wrapping_sub(raw & 0x20) & 0xffff_ffff_ffff_ffff
		}
		3 => {
			if bits_11_7 == 2 {
				// C.ADDI16SP: nzimm[9|4|6|8:7|5], 按 10 位符号扩展
				let nz = (((h >> 6) & 0x1) << 4)
					| (((h >> 2) & 0x1) << 5)
					| (((h >> 5) & 0x1) << 6)
					| (((h >> 3) & 0x3) << 7)
					| (((h >> 12) & 0x1) << 9);
				(nz & 0x1ff).wrapping_sub(nz & 0x200) & 0xffff_ffff_ffff_ffff
			} else {
				// C.LUI: 6 位 nzimm, 由 bit12 与 bits[6:2] 拼接
				let nz = ((bit12 as u64) << 5) | bits_6_2;
				(nz & 0x1f).wrapping_sub(nz & 0x20) & 0xffff_ffff_ffff_ffff
			}
		}
		5 => {
			// C.J: offset[11|4|9:8|10|6|7|3:1|5], 按 11 位符号扩展
			let off = (((h >> 12) & 0x1) << 11)
				| (((h >> 8) & 0x1) << 10)
				| (((h >> 9) & 0x3) << 8)
				| (((h >> 6) & 0x1) << 7)
				| (((h >> 7) & 0x1) << 6)
				| (((h >> 2) & 0x1) << 5)
				| (((h >> 11) & 0x1) << 4)
				| (((h >> 3) & 0x7) << 1);
			(off & 0x7ff).wrapping_sub(off & 0x800) & 0xffff_ffff_ffff_ffff
		}
		6 | 7 => {
			// C.BEQZ 与 C.BNEZ: offset[8|4:3|7:6|2:1|5], 按 9 位符号扩展
			let off = (((h >> 12) & 0x1) << 8)
				| (((h >> 10) & 0x3) << 3)
				| (((h >> 5) & 0x3) << 6)
				| (((h >> 2) & 0x1) << 5)
				| (((h >> 3) & 0x3) << 1);
			(off & 0xff).wrapping_sub(off & 0x100) & 0xffff_ffff_ffff_ffff
		}
		_ => 0u64,
	}
}

/// 解码 bits[1:0] 为 2 的压缩指令的立即数。
/// 返回 ``(imm, imm2)``, 其中 imm2 只用于该组中的存储指令。
#[inline]
fn imm_c2(h: u64, funct3: u8, bit12: u8, bits_6_2: u64) -> (u64, u64) {
	let imm = match funct3 {
		0 => {
			// C.SLLI: shamt[5:0], 由 bit12 与 bits[6:2] 拼接
			((bit12 as u64) << 5) | bits_6_2
		}
		1 | 3 => {
			// C.FLDSP 与 C.LDSP: uimm[5|4:3|8:6]
			(((h >> 2) & 0x7) << 6) | (((h >> 12) & 0x1) << 5) | (((h >> 5) & 0x3) << 3)
		}
		2 => {
			// C.LWSP: uimm[7:6|5|4:2]
			(((h >> 2) & 0x3) << 6) | (((h >> 12) & 0x1) << 5) | (((h >> 4) & 0x7) << 2)
		}
		_ => 0,
	};
	let imm2 = match funct3 {
		6 => {
			// C.SWSP: uimm[5:2|7:6]
			(((h >> 9) & 0xf) << 2) | (((h >> 7) & 0x3) << 6)
		}
		5 | 7 => {
			// C.FSDSP 与 C.SDSP: uimm[5:3|8:6]
			(((h >> 7) & 0x7) << 6) | (((h >> 10) & 0x7) << 3)
		}
		_ => 0,
	};
	(imm, imm2)
}

/// 解码压缩指令的立即数。
///
/// 返回 ``(imm, imm2)``, 其中 ``imm2`` 只用于 ``quadrant`` 取值为 2 的存储指令。
#[inline]
fn decode_comp_imm(
	h: u64,
	quadrant: u8,
	funct3: u8,
	bit12: u8,
	bits_6_2: u64,
	bits_11_7: u64,
) -> (u64, u64) {
	match quadrant {
		0 => (imm_c0(h, funct3), 0u64),

		1 => (imm_c1(h, funct3, bit12, bits_6_2, bits_11_7), 0u64),

		2 => imm_c2(h, funct3, bit12, bits_6_2),

		_ => (0, 0),
	}
}

/// 提取 16 位压缩指令的全部字段。
#[no_mangle]
pub extern "C" fn decode_compressed(half: u16) -> CompressedFields {
	let h = half as u64;
	let quadrant = (h & 0x3) as u8;
	let funct3 = ((h >> 13) & 0x7) as u8;

	// ---- 公共子字段 ----
	let bit12 = ((h >> 12) & 0x1) as u8;
	let bits_6_2 = (h >> 2) & 0x1f;
	let bits_11_7 = (h >> 7) & 0x1f;

	// ---- 寄存器字段 ----
	let rdp = creg(((h >> 2) & 0x7) as u8); // C0 的 rd 与 rs2, C1-ALU 的 rs2
	let rs1p = creg(((h >> 7) & 0x7) as u8); // C0 的 rs1, C1-ALU 的 rd
	let (rd, rs1, rs2) = if quadrant == 0 {
		(rdp, rs1p, rdp)
	} else {
		(bits_11_7 as u8, bits_11_7 as u8, bits_6_2 as u8)
	};

	// ---- C1-ALU 子字段 ----
	let sf = ((h >> 10) & 0x3) as u8;
	let bit65 = ((h >> 5) & 0x3) as u8;

	let (imm, imm2) = decode_comp_imm(h, quadrant, funct3, bit12, bits_6_2, bits_11_7);

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
//  单元测试
// ============================================================

#[cfg(test)]
mod tests {
	use super::*;

	// -- 32 位指令解码测试 --

	#[test]
	fn test_sext12_pos() {
		assert_eq!(sext12(0x7ff), 0x7ff);
		assert_eq!(sext12(0x800), 0xffff_ffff_ffff_f800);
		assert_eq!(sext12(0xfff), 0xffff_ffff_ffff_ffff);
	}

	#[test]
	fn test_addi_x5_x0_42() {
		// addi x5, x0, 42
		let f = decode_fields(0x02a00293);
		assert_eq!(f.opcode, 0x13);
		assert_eq!(f.rd, 5);
		assert_eq!(f.func3, 0);
		assert_eq!(f.rs1, 0);
		// I 型指令: rs2 字段与 imm[4:0] 重叠; imm=42=0b101010, 低 5 位为 10
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
		let f = decode_fields(0x00b50a63);
		assert_eq!(f.opcode, 0x63);
		assert_eq!(f.func3, 0); // BEQ
		assert_eq!(f.imm_b, 20);
		assert_eq!(f.rs1, 10);
		assert_eq!(f.rs2, 11);
	}

	#[test]
	fn test_jal_imm() {
		let f = decode_fields(0x400000ef);
		assert_eq!(f.opcode, 0x6f);
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
		let f = decode_fields(0xfff00293);
		assert_eq!(f.imm12_se, 0xffff_ffff_ffff_ffff);
		assert_eq!(f.opcode, 0x13);
	}

	// -- 16 位压缩指令解码测试 --

	#[test]
	fn test_c_addi4spn() {
		let f = decode_compressed(0x0040);
		assert_eq!(f.quadrant, 0);
		assert_eq!(f.funct3, 0); // C.ADDI4SPN
	}

	#[test]
	fn test_c_addi() {
		let f = decode_compressed(0x050d);
		assert_eq!(f.quadrant, 1);
		assert_eq!(f.funct3, 0); // C.ADDI
		assert_eq!(f.rd, 10); // x10 = a0
		assert_eq!(f.imm, 3); // +3
	}

	#[test]
	fn test_c_li() {
		let f = decode_compressed(0x457d);
		assert_eq!(f.quadrant, 1);
		assert_eq!(f.funct3, 2); // C.LI
		assert_eq!(f.rd, 10);
		assert_eq!(f.imm, 31);
	}

	#[test]
	fn test_c_addi16sp() {
		let f = decode_compressed(0x717d);
		assert_eq!(f.quadrant, 1);
		assert_eq!(f.funct3, 3);
		assert_eq!(f.rd, 2); // sp
		assert_eq!(f.imm, 0xffff_ffff_ffff_fff0); // -16 的 u64 表示
	}

	#[test]
	fn test_c_lui() {
		let f = decode_compressed(0x6505);
		assert_eq!(f.quadrant, 1);
		assert_eq!(f.funct3, 3);
		assert_eq!(f.rd, 10);
		assert_eq!(f.imm, 1); // nzimm=1, 由调用方左移 12 位
	}

	#[test]
	fn test_c_j() {
		let f = decode_compressed(0xb005);
		assert_eq!(f.quadrant, 1);
		assert_eq!(f.funct3, 5); // C.J
		assert!(f.imm > 0);
	}

	#[test]
	fn test_c_beqz() {
		// C.BEQZ x10, +8
		let f = decode_compressed(0xc501);
		assert_eq!(f.quadrant, 1);
		assert_eq!(f.funct3, 6); // C.BEQZ
		assert_eq!(f.rs1, 10); // x10 = a0
	}

	#[test]
	fn test_c_slli() {
		// c.slli x10, 3
		// bit12=0, bits[11:7]=01010, shamt[4:0]=00011, op=10
		// 000 0 01010 00011 10, 即 0x050E
		let f = decode_compressed(0x050e);
		assert_eq!(f.quadrant, 2);
		assert_eq!(f.funct3, 0); // C.SLLI
		assert_eq!(f.rd, 10);
		assert_eq!(f.imm, 3); // shamt
	}

	#[test]
	fn test_c_lwsp() {
		// C.LWSP x10, 4(sp)
		let f = decode_compressed(0x4512); // c.lwsp a0, 4(sp)
		assert_eq!(f.quadrant, 2);
		assert_eq!(f.funct3, 2); // C.LWSP
		assert_eq!(f.rd, 10);
		assert_eq!(f.imm, 4);
	}

	#[test]
	fn test_c_jr() {
		// c.jr x10
		// 100 0 01010 00000 10, 即 0x8502
		let f = decode_compressed(0x8502);
		assert_eq!(f.quadrant, 2);
		assert_eq!(f.funct3, 4);
		assert_eq!(f.rs1, 10);
		assert_eq!(f.rs2, 0);
		assert_eq!(f.bit12, 0);
	}

	#[test]
	fn test_c_swsp() {
		// c.swsp x10, 8(sp)
		let f = decode_compressed(0xc42a);
		assert_eq!(f.quadrant, 2);
		assert_eq!(f.funct3, 6); // C.SWSP
	}

	#[test]
	fn test_c_ld_imm_regression() {
		// c.ld a5, 0(s0), 编码为 0x601c; 该指令曾在 whoami 中触发访存故障。
		let f = decode_compressed(0x601c);
		assert_eq!(f.quadrant, 0);
		assert_eq!(f.funct3, 3);
		assert_eq!(f.rd, 15);
		assert_eq!(f.rs1p, 8);
		assert_eq!(f.imm, 0);
	}

	// ============================================================
	//  全部编码均与 llvm-mc 的输出一致。
	//  防止 C.LWSP 偏移量解码错误再次出现。
	// ============================================================

	// ---- C0: quadrant 为 0 ----

	#[test]
	fn c0_addi4spn_16() {
		// c.addi4spn x8, sp, 16
		let f = decode_compressed(0x0800);
		assert_eq!(f.quadrant, 0);
		assert_eq!(f.funct3, 0);
		assert_eq!(f.rdp, 8); // x8 = s0
		assert_eq!(f.imm, 16);
	}

	#[test]
	fn c0_addi4spn_1020() {
		// c.addi4spn x15, sp, 1020, nzuimm 取最大值
		let f = decode_compressed(0x1ffc);
		assert_eq!(f.quadrant, 0);
		assert_eq!(f.funct3, 0);
		assert_eq!(f.rdp, 15); // x15 = a5
		assert_eq!(f.imm, 1020);
	}

	#[test]
	fn c0_lw_zero_offset() {
		// c.lw x8, 0(x10)
		let f = decode_compressed(0x4100);
		assert_eq!(f.quadrant, 0);
		assert_eq!(f.funct3, 2); // C.LW
		assert_eq!(f.rdp, 8); // x8
		assert_eq!(f.rs1p, 10); // x10 = a0
		assert_eq!(f.imm, 0);
	}

	#[test]
	fn c0_lw_max_offset() {
		// c.lw x15, 124(x15), uimm 最大值为 124
		let f = decode_compressed(0x5ffc);
		assert_eq!(f.quadrant, 0);
		assert_eq!(f.funct3, 2);
		assert_eq!(f.imm, 124);
	}

	#[test]
	fn c0_ld_zero_offset() {
		// c.ld x8, 0(x10)
		let f = decode_compressed(0x6100);
		assert_eq!(f.quadrant, 0);
		assert_eq!(f.funct3, 3); // C.LD
		assert_eq!(f.imm, 0);
	}

	#[test]
	fn c0_ld_max_offset() {
		// c.ld x15, 248(x15), uimm 最大值为 248
		let f = decode_compressed(0x7ffc);
		assert_eq!(f.quadrant, 0);
		assert_eq!(f.funct3, 3);
		assert_eq!(f.imm, 248);
	}

	#[test]
	fn c0_sw_zero_offset() {
		// c.sw x8, 0(x10)
		let f = decode_compressed(0xc100);
		assert_eq!(f.quadrant, 0);
		assert_eq!(f.funct3, 6); // C.SW
		assert_eq!(f.rdp, 8); // rs2'
		assert_eq!(f.imm, 0);
	}

	#[test]
	fn c0_sw_max_offset() {
		// c.sw x9, 124(x15), 偏移量最大
		let f = decode_compressed(0xdfe4);
		assert_eq!(f.quadrant, 0);
		assert_eq!(f.funct3, 6);
		assert_eq!(f.imm, 124);
	}

	#[test]
	fn c0_sd_zero_offset() {
		// c.sd x8, 0(x10)
		let f = decode_compressed(0xe100);
		assert_eq!(f.quadrant, 0);
		assert_eq!(f.funct3, 7); // C.SD
		assert_eq!(f.imm, 0);
	}

	#[test]
	fn c0_sd_max_offset() {
		// c.sd x9, 248(x15), 偏移量最大
		let f = decode_compressed(0xffe4);
		assert_eq!(f.quadrant, 0);
		assert_eq!(f.funct3, 7);
		assert_eq!(f.imm, 248);
	}

	// ---- C1: quadrant 为 1 ----

	#[test]
	fn c1_nop() {
		// c.nop
		let f = decode_compressed(0x0001);
		assert_eq!(f.quadrant, 1);
		assert_eq!(f.funct3, 0); // C.ADDI rd=0, imm=0
		assert_eq!(f.rd, 0);
		assert_eq!(f.imm, 0);
	}

	#[test]
	fn c1_addi_pos() {
		// c.addi x5, 3
		let f = decode_compressed(0x028d);
		assert_eq!(f.quadrant, 1);
		assert_eq!(f.funct3, 0);
		assert_eq!(f.rd, 5);
		assert_eq!(f.imm, 3);
	}

	#[test]
	fn c1_addi_neg32() {
		// c.addi x10, -32
		let f = decode_compressed(0x1501);
		assert_eq!(f.quadrant, 1);
		assert_eq!(f.funct3, 0);
		assert_eq!(f.rd, 10);
		// -32 按 6 位符号扩展, 在 u64 中为 0xFFFF_FFFF_FFFF_FFE0
		assert_eq!(f.imm, 0xffff_ffff_ffff_ffe0u64);
	}

	#[test]
	fn c1_addiw() {
		// c.addiw x5, -2
		let f = decode_compressed(0x32f9);
		assert_eq!(f.quadrant, 1);
		assert_eq!(f.funct3, 1); // C.ADDIW
		assert_eq!(f.rd, 5);
		assert_eq!(f.imm, 0xffff_ffff_ffff_fffeu64); // -2
	}

	#[test]
	fn c1_li_max() {
		// c.li x5, 31
		let f = decode_compressed(0x42fd);
		assert_eq!(f.quadrant, 1);
		assert_eq!(f.funct3, 2); // C.LI
		assert_eq!(f.rd, 5);
		assert_eq!(f.imm, 31);
	}

	#[test]
	fn c1_li_min() {
		// c.li x10, -32
		let f = decode_compressed(0x5501);
		assert_eq!(f.quadrant, 1);
		assert_eq!(f.funct3, 2);
		assert_eq!(f.rd, 10);
		assert_eq!(f.imm, 0xffff_ffff_ffff_ffe0u64); // -32
	}

	#[test]
	fn c1_lui_1() {
		// c.lui x5, 1
		let f = decode_compressed(0x6285);
		assert_eq!(f.quadrant, 1);
		assert_eq!(f.funct3, 3);
		assert_eq!(f.rd, 5);
		assert_eq!(f.imm, 1);
	}

	#[test]
	fn c1_lui_31() {
		// c.lui x10, 31
		let f = decode_compressed(0x657d);
		assert_eq!(f.quadrant, 1);
		assert_eq!(f.funct3, 3);
		assert_eq!(f.rd, 10);
		assert_eq!(f.imm, 31);
	}

	#[test]
	fn c1_addi16sp_neg16() {
		// c.addi16sp sp, -16
		let f = decode_compressed(0x717d);
		assert_eq!(f.quadrant, 1);
		assert_eq!(f.funct3, 3);
		assert_eq!(f.rd, 2); // sp
		assert_eq!(f.imm, 0xffff_ffff_ffff_fff0u64); // -16
	}

	#[test]
	fn c1_addi16sp_pos496() {
		// c.addi16sp sp, 496
		let f = decode_compressed(0x617d);
		assert_eq!(f.quadrant, 1);
		assert_eq!(f.funct3, 3);
		assert_eq!(f.rd, 2);
		assert_eq!(f.imm, 496);
	}

	#[test]
	fn c1_srli() {
		// c.srli x8, 3
		let f = decode_compressed(0x800d);
		assert_eq!(f.quadrant, 1);
		assert_eq!(f.funct3, 4); // MISC-ALU
		assert_eq!(f.sf, 0); // SRLI
		assert_eq!(f.rs1p, 8); // rd' = x8
		assert_eq!(f.bit12, 0); // shamt[5]=0

		// shamt 为 bit12<<5 | bits[6:2]; funct3=4 时 imm 为 0, imm_c1 未解码该值
	}

	#[test]
	fn c1_srai() {
		// c.srai x9, 4
		let f = decode_compressed(0x8491);
		assert_eq!(f.quadrant, 1);
		assert_eq!(f.funct3, 4);
		assert_eq!(f.sf, 1); // SRAI
		assert_eq!(f.rs1p, 9);
	}

	#[test]
	fn c1_andi() {
		// c.andi x10, -1
		let f = decode_compressed(0x997d);
		assert_eq!(f.quadrant, 1);
		assert_eq!(f.funct3, 4);
		assert_eq!(f.sf, 2); // ANDI
		assert_eq!(f.rs1p, 10);
		assert_eq!(f.bit12, 1); // -1 使 imm[5]=1
	}

	#[test]
	fn c1_sub() {
		// c.sub x11, x12
		let f = decode_compressed(0x8d91);
		assert_eq!(f.quadrant, 1);
		assert_eq!(f.funct3, 4);
		assert_eq!(f.sf, 3); // 寄存器运算
		assert_eq!(f.bit12, 0); // 非 RV64C 形式
		assert_eq!(f.bit65, 0); // SUB
		assert_eq!(f.rs1p, 11); // rd' = x11
		assert_eq!(f.rdp, 12); // rs2' = x12
	}

	#[test]
	fn c1_xor() {
		// c.xor x11, x12
		let f = decode_compressed(0x8db1);
		assert_eq!(f.bit65, 1); // XOR
	}

	#[test]
	fn c1_or() {
		// c.or x11, x12
		let f = decode_compressed(0x8dd1);
		assert_eq!(f.bit65, 2); // OR
	}

	#[test]
	fn c1_and() {
		// c.and x11, x12
		let f = decode_compressed(0x8df1);
		assert_eq!(f.bit65, 3); // AND
	}

	#[test]
	fn c1_subw() {
		// c.subw x13, x14
		let f = decode_compressed(0x9e99);
		assert_eq!(f.sf, 3);
		assert_eq!(f.bit12, 1); // RV64C
		assert_eq!(f.bit65, 0); // SUBW
		assert_eq!(f.rs1p, 13);
		assert_eq!(f.rdp, 14);
	}

	#[test]
	fn c1_addw() {
		// c.addw x13, x14
		let f = decode_compressed(0x9eb9);
		assert_eq!(f.sf, 3);
		assert_eq!(f.bit12, 1);
		assert_eq!(f.bit65, 1); // ADDW
	}

	#[test]
	fn c1_j_forward() {
		// c.j .+2 对应 0xa001, imm 为 0; objdump 中的前向跳转编码为 0xa009
		let f = decode_compressed(0xa009);
		assert_eq!(f.quadrant, 1);
		assert_eq!(f.funct3, 5); // C.J
		assert_eq!(f.imm, 2);
	}

	#[test]
	fn c1_beqz_taken() {
		// c.beqz x10, .+2
		let f = decode_compressed(0xc109);
		assert_eq!(f.quadrant, 1);
		assert_eq!(f.funct3, 6); // C.BEQZ
		assert_eq!(f.rs1p, 10); // x10 = a0
	}

	#[test]
	fn c1_bnez() {
		// c.bnez x10, .+2
		let f = decode_compressed(0xe109);
		assert_eq!(f.quadrant, 1);
		assert_eq!(f.funct3, 7); // C.BNEZ
		assert_eq!(f.rs1p, 10);
	}

	// ---- C2: quadrant 为 2 ----

	#[test]
	fn c2_slli_63() {
		// c.slli x10, 63
		let f = decode_compressed(0x157e);
		assert_eq!(f.quadrant, 2);
		assert_eq!(f.funct3, 0); // C.SLLI
		assert_eq!(f.rd, 10);
		assert_eq!(f.imm, 63);
	}

	#[test]
	fn c2_lwsp_regression_zsh_crash() {
		// c.lwsp x18, 4(sp)
		// 偏移量必须解码为 4; 解码为 16 会读出 s2 的保存值 -560。
		let f = decode_compressed(0x4912);
		assert_eq!(f.quadrant, 2);
		assert_eq!(f.funct3, 2); // C.LWSP
		assert_eq!(f.rd, 18); // x18 = s2
		assert_eq!(f.imm, 4); // 偏移量为 4, 不为 16
	}

	#[test]
	fn c2_lwsp_max_offset() {
		// c.lwsp x10, 252(sp)
		let f = decode_compressed(0x557e);
		assert_eq!(f.quadrant, 2);
		assert_eq!(f.funct3, 2);
		assert_eq!(f.rd, 10);
		assert_eq!(f.imm, 252);
	}

	#[test]
	fn c2_ldsp_8() {
		// c.ldsp x5, 8(sp)
		let f = decode_compressed(0x62a2);
		assert_eq!(f.quadrant, 2);
		assert_eq!(f.funct3, 3); // C.LDSP
		assert_eq!(f.rd, 5);
		assert_eq!(f.imm, 8);
	}

	#[test]
	fn c2_ldsp_504() {
		// c.ldsp x10, 504(sp), 编码为 0x757e, 这是 C.LDSP 的最大偏移量
		let f = decode_compressed(0x757e);
		assert_eq!(f.quadrant, 2);
		assert_eq!(f.funct3, 3);
		assert_eq!(f.rd, 10);
		assert_eq!(f.imm, 504);
	}

	#[test]
	fn c2_mv() {
		// c.mv x5, x10
		let f = decode_compressed(0x82aa);
		assert_eq!(f.quadrant, 2);
		assert_eq!(f.funct3, 4);
		assert_eq!(f.rd, 5);
		assert_eq!(f.rs2, 10);
		assert_eq!(f.bit12, 0); // C.MV
	}

	#[test]
	fn c2_add() {
		// c.add x5, x10
		let f = decode_compressed(0x92aa);
		assert_eq!(f.quadrant, 2);
		assert_eq!(f.funct3, 4);
		assert_eq!(f.rd, 5);
		assert_eq!(f.rs2, 10);
		assert_eq!(f.bit12, 1); // C.ADD
	}

	#[test]
	fn c2_ebreak() {
		// c.ebreak
		let f = decode_compressed(0x9002);
		assert_eq!(f.quadrant, 2);
		assert_eq!(f.funct3, 4);
		assert_eq!(f.rd, 0);
		assert_eq!(f.rs2, 0);
	}

	#[test]
	fn c2_swsp_zero() {
		// c.swsp x5, 0(sp)
		let f = decode_compressed(0xc016);
		assert_eq!(f.quadrant, 2);
		assert_eq!(f.funct3, 6); // C.SWSP
		assert_eq!(f.rs2, 5);
		assert_eq!(f.imm2, 0); // 存储指令的偏移量在 imm2 中
	}

	#[test]
	fn c2_swsp_max_offset() {
		// c.swsp x10, 252(sp)
		let f = decode_compressed(0xdfaa);
		assert_eq!(f.quadrant, 2);
		assert_eq!(f.funct3, 6);
		assert_eq!(f.imm2, 252);
	}

	#[test]
	fn c2_sdsp_8() {
		// c.sdsp x5, 8(sp)
		let f = decode_compressed(0xe416);
		assert_eq!(f.quadrant, 2);
		assert_eq!(f.funct3, 7); // C.SDSP
		assert_eq!(f.rs2, 5);
		assert_eq!(f.imm2, 8);
	}

	#[test]
	fn c2_sdsp_max_offset() {
		// c.sdsp x10, 504(sp)
		let f = decode_compressed(0xffaa);
		assert_eq!(f.quadrant, 2);
		assert_eq!(f.funct3, 7);
		assert_eq!(f.imm2, 504);
	}

	// ---- 32 位指令字段提取, 已与 llvm-mc 核对 ----

	#[test]
	fn rv32_addw_opcode() {
		// addw x5, x10, x11
		let f = decode_fields(0x00b502bb);
		assert_eq!(f.opcode, 0x3b);
		assert_eq!(f.rd, 5);
		assert_eq!(f.func3, 0);
		assert_eq!(f.func7, 0); // ADDW 的 func7=0
	}

	#[test]
	fn rv32_subw_func7() {
		// subw x6, x10, x11
		let f = decode_fields(0x40b5033b);
		assert_eq!(f.opcode, 0x3b);
		assert_eq!(f.rd, 6);
		assert_eq!(f.func3, 0);
		assert_eq!(f.func7, 0x20); // SUBW 的 func7=0b0100000
	}

	#[test]
	fn rv32_addiw_opcode() {
		// addiw x5, x10, 42
		let f = decode_fields(0x02a5029b);
		assert_eq!(f.opcode, 0x1b);
		assert_eq!(f.rd, 5);
		assert_eq!(f.func3, 0);
		assert_eq!(f.imm12_se, 42);
	}

	#[test]
	fn rv32_slliw() {
		// slliw x6, x10, 3
		let f = decode_fields(0x0035131b);
		assert_eq!(f.opcode, 0x1b);
		assert_eq!(f.func3, 1); // SLLIW
		assert_eq!(f.imm12_se, 3);
	}

	#[test]
	fn rv32_sraiw() {
		// sraiw x8, x10, 4
		let f = decode_fields(0x4045541b);
		assert_eq!(f.opcode, 0x1b);
		assert_eq!(f.func3, 5); // SRAIW
	}

	#[test]
	fn store_sd_neg_offset() {
		// sd x9, -8(x10)
		let f = decode_fields(0xfe953c23);
		assert_eq!(f.opcode, 0x23);
		assert_eq!(f.func3, 3); // SD
						  // imm_s 为 -8 按 12 位符号扩展后的值
		assert_eq!(f.imm_s, 0xffff_ffff_ffff_fff8u64);
	}

	#[test]
	fn load_ld_neg_offset() {
		// ld x14, -8(x10)
		let f = decode_fields(0xff853703);
		assert_eq!(f.opcode, 0x03);
		assert_eq!(f.func3, 3); // LD
		assert_eq!(f.imm12_se, 0xffff_ffff_ffff_fff8u64); // -8
	}

	#[test]
	fn fence_i_func3() {
		// fence.i
		let f = decode_fields(0x0000100f);
		assert_eq!(f.opcode, 0x0f);
		assert_eq!(f.func3, 1); // FENCE.I
	}

	#[test]
	fn csr_mstatus_addr() {
		// csrrw x5, mstatus, x10
		let f = decode_fields(0x300512f3);
		assert_eq!(f.opcode, 0x73);
		assert_eq!(f.func3, 1); // CSRRW
		assert_eq!(f.imm12_se, 0x300); // mstatus 的 CSR 地址
	}

	#[test]
	fn csr_csrrsi() {
		// csrrsi x9, mstatus, 0x1f
		let f = decode_fields(0x300fe4f3);
		assert_eq!(f.func3, 6); // CSRRSI
		assert_eq!(f.rd, 9);
		assert_eq!(f.rs1, 31); // uimm
	}

	#[test]
	fn amo_lrw_fields() {
		// lr.w x5, (x10)
		let f = decode_fields(0x100522af);
		assert_eq!(f.opcode, 0x2f);
		assert_eq!(f.func3, 2); // .W
		assert_eq!(f.rd, 5);
		assert_eq!(f.rs1, 10);
		assert_eq!(f.func7 >> 2, 0b00010); // LR 的 funct5
	}

	#[test]
	fn amo_amoxor_w() {
		// amoxor.w x13, x11, (x10)
		let f = decode_fields(0x20b526af);
		assert_eq!(f.opcode, 0x2f);
		assert_eq!(f.func3, 2);
		assert_eq!(f.func7 >> 2, 0b00100); // AMOXOR 的 funct5
	}

	#[test]
	fn amo_amomaxu_w() {
		// amomaxu.w x19, x11, (x10)
		let f = decode_fields(0xe0b529af);
		assert_eq!(f.func7 >> 2, 0b11100); // AMOMAXU 的 funct5
	}

	#[test]
	fn amo_amoadd_d() {
		// amoadd.d x20, x11, (x10)
		let f = decode_fields(0x00b53a2f);
		assert_eq!(f.opcode, 0x2f);
		assert_eq!(f.func3, 3); // .D
		assert_eq!(f.func7 >> 2, 0b00000); // AMOADD 的 funct5
	}
}
