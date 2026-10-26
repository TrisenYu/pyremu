//! RISC-V ALU 纯计算操作.
//!
//! 所有函数返回 `Alu64` 结构体:
//!     .value: 存放运算结果
//!     .trap: 记录陷态标志, 0 表示编码合法, 1 表示编码非法并提示产生非法指令陷态.

// ============================================================
//  返回类型
// ============================================================

/// 带陷态标志的 ALU 结果. `trap` 为 1 表示编码非法并提示产生非法指令陷态
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
//  RV64 辅助函数
// ============================================================

/// RISC-V R 型 ALU: funct3 选择运算, funct7 选择变体.
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
				let val = if s2 == 0 {
					u64::MAX
				} else if s1 == i64::MIN && s2 == -1 {
					i64::MIN as u64
				} else {
					(s1 / s2) as u64
				};
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
				let val = if s2 == 0 {
					v1
				} else if s1 == i64::MIN && s2 == -1 {
					0
				} else {
					(s1 % s2) as u64
				};
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
//  I 型立即数 ALU
// ============================================================

/// RISC-V I 型立即数 ALU: funct3 选择运算.
///
/// `imm` 为已经符号扩展的 12 位立即数.
/// `funct7` 的高 6 位即 funct6 字段, 对应 bits[31:26], 用于区分 SRLI 与 SRAI.
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
//  RV64 32 位字运算
// ============================================================

/// RISC-V RV64 32 位字 ALU: 只对低 32 位运算, 结果符号扩展到 64 位.
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
			if funct7 != 0 {
				return Alu64::ill();
			}
			w1 << (w2 & 0x1F)
		}
		0b100 => {
			if funct7 != 1 {
				return Alu64::ill();
			}
			if w2 == 0 {
				-1i32
			} else if w1 == i32::MIN && w2 == -1 {
				i32::MIN
			} else {
				w1 / w2
			}
		}
		0b101 => match funct7 {
			0 => ((w1 as u32) >> (w2 as u32 & 0x1F)) as i32,
			1 => {
				let u1 = w1 as u32;
				let u2 = w2 as u32;
				if u2 == 0 {
					u32::MAX as i32
				} else {
					(u1 / u2) as i32
				}
			}
			0x20 => w1 >> (w2 & 0x1F),
			_ => return Alu64::ill(),
		},
		0b110 => {
			if funct7 != 1 {
				return Alu64::ill();
			}
			if w2 == 0 {
				w1
			} else if w1 == i32::MIN && w2 == -1 {
				0
			} else {
				w1 % w2
			}
		}
		0b111 => {
			if funct7 != 1 {
				return Alu64::ill();
			}
			let u1 = w1 as u32;
			let u2 = w2 as u32;
			if u2 == 0 {
				u1 as i32
			} else {
				(u1 % u2) as i32
			}
		}
		_ => return Alu64::ill(),
	};

	Alu64::ok((result_i32 as i64) as u64)
}

// ============================================================
//  RV64 32 位字立即数运算
// ============================================================

/// RISC-V RV64 32 位字立即数 ALU: 对低 32 位运算, 结果符号扩展到 64 位.
///
/// | funct3 | 运算 |
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
			if funct7 != 0 {
				return Alu64::ill();
			}
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
//  测试
// ============================================================

#[cfg(test)]
mod tests {
	use super::*;

	fn v(r: Alu64) -> u64 {
		assert_eq!(r.trap, 0);
		r.value
	}
	fn trap(r: Alu64) -> bool {
		r.trap != 0
	}

	#[test]
	fn test_add() {
		assert_eq!(v(exec_alu_op(0, 0, 10, 5)), 15);
	}
	#[test]
	fn test_sub() {
		assert_eq!(v(exec_alu_op(0, 0x20, 10, 5)), 5);
	}
	#[test]
	fn test_sll() {
		assert_eq!(v(exec_alu_op(1, 0, 1, 3)), 8);
	}
	#[test]
	fn test_xor() {
		assert_eq!(v(exec_alu_op(4, 0, 0xFF, 0x0F)), 0xF0);
	}
	#[test]
	fn test_or() {
		assert_eq!(v(exec_alu_op(6, 0, 0xF0, 0x0F)), 0xFF);
	}
	#[test]
	fn test_and() {
		assert_eq!(v(exec_alu_op(7, 0, 0xFF, 0x0F)), 0x0F);
	}
	#[test]
	fn test_slt_true() {
		assert_eq!(v(exec_alu_op(2, 0, 5, 10)), 1);
	}
	#[test]
	fn test_slt_false() {
		assert_eq!(v(exec_alu_op(2, 0, 10, 5)), 0);
	}
	#[test]
	fn test_sltu_true() {
		assert_eq!(v(exec_alu_op(3, 0, 5, 10)), 1);
	}
	#[test]
	fn test_srl() {
		assert_eq!(v(exec_alu_op(5, 0, 0x100, 4)), 0x10);
	}
	#[test]
	fn test_sra_negative() {
		assert_eq!(
			v(exec_alu_op(5, 0x20, 0xFFFF_FFFF_FFFF_FF00u64, 4)),
			0xFFFF_FFFF_FFFF_FFF0u64
		);
	}
	#[test]
	fn test_div() {
		assert_eq!(v(exec_alu_op(4, 1, 20, 6)), 3);
	}
	#[test]
	fn test_div_by_zero() {
		assert_eq!(v(exec_alu_op(4, 1, 42, 0)), u64::MAX);
	}
	#[test]
	fn test_div_overflow() {
		assert_eq!(
			v(exec_alu_op(4, 1, i64::MIN as u64, (-1i64) as u64)),
			i64::MIN as u64
		);
	}
	#[test]
	fn test_divu() {
		assert_eq!(v(exec_alu_op(5, 1, 20, 6)), 3);
	}
	#[test]
	fn test_rem() {
		assert_eq!(v(exec_alu_op(6, 1, 20, 6)), 2);
	}
	#[test]
	fn test_remu() {
		assert_eq!(v(exec_alu_op(7, 1, 20, 6)), 2);
	}
	#[test]
	fn test_mul() {
		assert_eq!(v(exec_alu_op(0, 1, 6, 7)), 42);
	}
	#[test]
	fn test_alu_ill_funct7() {
		assert!(trap(exec_alu_op(0, 0x0F, 10, 5)));
	}
	#[test]
	fn test_alu_ill_funct3() {
		assert!(trap(exec_alu_op(0xFF, 0, 10, 5)));
	}

	// ---- 64 位 REM: 检查操作数按完整 64 位参与运算而不被截断 ----
	#[test]
	fn test_rem_positive_large() {
		// 0x3FD9CEF920 占 33 位; REM 必须按完整 64 位运算.
		assert_eq!(v(exec_alu_op(6, 1, 0x3FD9CEF920, 7)), 5);
	}
	#[test]
	fn test_rem_negative() {
		// RISC-V 采用截断除法, 被除数为负时余数为负.
		assert_eq!(v(exec_alu_op(6, 1, (-10i64) as u64, 3)), (-1i64) as u64);
	}
	#[test]
	fn test_rem_negative_divisor() {
		assert_eq!(v(exec_alu_op(6, 1, 20, (-6i64) as u64)), 2);
	}
	#[test]
	fn test_rem_by_zero() {
		// 除数为零时 REM 返回被除数
		assert_eq!(v(exec_alu_op(6, 1, 42, 0)), 42);
	}
	#[test]
	fn test_rem_overflow() {
		assert_eq!(v(exec_alu_op(6, 1, i64::MIN as u64, (-1i64) as u64)), 0);
	}
	#[test]
	fn test_remu_large() {
		assert_eq!(v(exec_alu_op(7, 1, 0x3FD9CEF920, 7)), 5);
	}
	#[test]
	fn test_remu_by_zero() {
		assert_eq!(v(exec_alu_op(7, 1, 42, 0)), 42);
	}

	// ---- REMW: 32 位有符号求余, 结果符号扩展 ----
	#[test]
	fn test_remw_positive() {
		assert_eq!(v(exec_op32(6, 1, 20, 6)), 2);
	}
	#[test]
	fn test_remw_negative_dividend() {
		assert_eq!(v(exec_op32(6, 1, (-10i64) as u64, 3)), (-1i64) as u64);
	}
	#[test]
	fn test_remw_sign_extend_negative_result() {
		// w1 = 0xD9CEF920 按 i32 解释为 -640747232, w2 = 7
		// -640747232 / 7 = -91535318.857, 截断后商为 -91535318
		// 余数 r 为 -6
		// 符号扩展后为 0xFFFF_FFFF_FFFF_FFFA
		assert_eq!(
			v(exec_op32(6, 1, 0x3FD9CEF920, 7)),
			0xFFFF_FFFF_FFFF_FFFAu64
		);
	}
	#[test]
	fn test_remw_by_zero() {
		assert_eq!(v(exec_op32(6, 1, 42, 0)), 42);
	}
	#[test]
	fn test_remw_overflow() {
		assert_eq!(v(exec_op32(6, 1, i32::MIN as u64, (-1i64) as u64)), 0);
	}

	// REMUW: 32 位无符号求余, 结果零扩展
	#[test]
	fn test_remuw_positive() {
		assert_eq!(v(exec_op32(7, 1, 20, 6)), 2);
	}
	#[test]
	fn test_remuw_large_u32() {
		// u1 = 0xD9CEF920 = 3654220064, u2 = 7
		// 3654220064 / 7 = 522031437.714...
		// 7 * 522031437 = 3654220059
		// 3654220064 - 3654220059 = 5
		assert_eq!(v(exec_op32(7, 1, 0x3FD9CEF920, 7)), 5);
	}
	#[test]
	fn test_remuw_by_zero() {
		assert_eq!(v(exec_op32(7, 1, 42, 0)), 42);
	}

	// 对该缺陷触发值, REM 与 REMW 的结果不同
	#[test]
	fn test_rem_vs_remw_crash_value() {
		let v1 = 0x3FD9CEF920u64; // 取自栈中的缺陷触发指针
		let v2 = 7u64;
		let r64 = v(exec_alu_op(6, 1, v1, v2)); // REM: 按完整 64 位运算, 结果为 5
		let r32 = v(exec_op32(6, 1, v1, v2)); // REMW: 低 32 位为 0xD9CEF920, 按有符号数解释为 -640747232, 结果为 -6
		assert_eq!(r64, 5);
		assert_eq!(r32, 0xFFFF_FFFF_FFFF_FFFAu64);
		// 必须不同。若相同, 则说明代码实现混淆了 32 位运算与 64 位运算
		assert_ne!(
			r64, r32,
			"REM and REMW MUST differ for this operand — bit-width confusion?"
		);
	}

	// ---- REM 边界: bit63 为 1 的操作数 ----
	#[test]
	fn test_rem_bit63_1_mod_positive() {
		// u64::MAX 即全部位为 1, 对应 -1
		assert_eq!(v(exec_alu_op(6, 1, u64::MAX, 3)), u64::MAX);
	}
	#[test]
	fn test_rem_bit63_1_mod_negative() {
		assert_eq!(v(exec_alu_op(6, 1, u64::MAX, (-3i64) as u64)), u64::MAX);
	}
	#[test]
	fn test_rem_positive_mod_negative() {
		assert_eq!(v(exec_alu_op(6, 1, 20, (-6i64) as u64)), 2);
	}
	#[test]
	fn test_rem_negative_mod_negative() {
		assert_eq!(
			v(exec_alu_op(6, 1, (-20i64) as u64, (-6i64) as u64)),
			(-2i64) as u64
		);
	}

	#[test]
	fn test_remu_bit63_1() {
		// u64::MAX % 3 = 0.  u64::MAX = 0xFFFFFFFFFFFFFFFF = 18446744073709551615.
		// 以 3 作为取模对象时, 2^64 的余数为 1, 故 2^64 - 1 的余数为 0, 该值可被 3 整除.
		assert_eq!(v(exec_alu_op(7, 1, u64::MAX, 3)), 0);
	}

	// 测试 REMW 指令的边界情况：低 32 位的 bit31 为 1
	#[test]
	fn test_remw_bit31_1_mod_positive() {
		// w1 = 0x80000000, 按 i32 解释为 i32::MIN = -2147483648
		// -2147483648 % 3 = -2, 符号扩展后为 0xFFFFFFFFFFFFFFFE
		assert_eq!(
			v(exec_op32(6, 1, 0x8000_0000u64, 3)),
			0xFFFF_FFFF_FFFF_FFFEu64
		);
	}
	#[test]
	fn test_remw_bit31_1_mod_negative() {
		assert_eq!(
			v(exec_op32(6, 1, 0x8000_0000u64, (-3i64) as u64)),
			0xFFFF_FFFF_FFFF_FFFEu64
		);
	}
	#[test]
	fn test_remw_bit31_0_mod_negative() {
		assert_eq!(v(exec_op32(6, 1, 10, (-3i64) as u64)), 1);
	}
	#[test]
	fn test_remw_bit31_0_sign_extend_positive() {
		assert_eq!(v(exec_op32(6, 1, 0x7FFF_FFFFu64, 3)), 1);
	}

	// REMUW 边界情况：bit31 为 1 时按无符号数解释
	#[test]
	fn test_remuw_bit31_1() {
		assert_eq!(v(exec_op32(7, 1, 0x8000_0000u64, 3)), 2);
	}
	#[test]
	fn test_remuw_contrast_remw_bit31_1() {
		// 低 32 位相同, 但 REMUW 按无符号数运算, REMW 按有符号数运算
		let v1 = 0x8000_0000u64;
		let remw_r = v(exec_op32(6, 1, v1, 3)); // 有符号: -2147483648 % 3 = -2
		let remuw_r = v(exec_op32(7, 1, v1, 3)); // 无符号: 2147483648 % 3 = 2
		assert_eq!(remw_r, 0xFFFF_FFFF_FFFF_FFFEu64); // 符号扩展后的 -2
		assert_eq!(remuw_r, 2); // 零扩展后的 2
		assert_ne!(remw_r, remuw_r, "REMW ≠ REMUW for bit31=1 operand");
	}

	#[test]
	fn test_addi() {
		assert_eq!(v(exec_op_imm(0, 0, 10, 5)), 15);
	}
	#[test]
	fn test_addi_negative() {
		assert_eq!(v(exec_op_imm(0, 0, 10, (-2i64) as u64)), 8);
	}
	#[test]
	fn test_slti() {
		assert_eq!(v(exec_op_imm(2, 0, 5, 10)), 1);
	}
	#[test]
	fn test_xori() {
		assert_eq!(v(exec_op_imm(4, 0, 0xFF, 0x0F)), 0xF0);
	}
	#[test]
	fn test_slli_bad_f6() {
		assert!(trap(exec_op_imm(1, 0x7F, 10, 3)));
	}
	#[test]
	fn test_srli_bad_f6() {
		// funct3 为 101 时 funct6 只允许取 0 或 0x10, 该用例取 0x1F.
		assert!(trap(exec_op_imm(5, 0x3E, 0x100, 2)));
	}

	#[test]
	fn test_addw() {
		assert_eq!(v(exec_op32(0, 0, 10, 5)), 15);
	}
	#[test]
	fn test_addw_sign_extend() {
		assert_eq!(v(exec_op32(0, 0, 0x7FFFFFFF, 1)), 0xFFFF_FFFF_8000_0000u64);
	}
	#[test]
	fn test_subw() {
		assert_eq!(v(exec_op32(0, 0x20, 10, 5)), 5);
	}
	#[test]
	fn test_op32_ill() {
		assert!(trap(exec_op32(0b001, 1, 10, 3)));
	}

	#[test]
	fn test_slliw() {
		assert_eq!(v(exec_op_imm32(1, 0, 0x100, 2)), 0x400);
	}
	#[test]
	fn test_sraiw() {
		assert_eq!(
			v(exec_op_imm32(5, 0x20, 0xFFFF_FFFF_8000_0000u64, 1)),
			0xFFFF_FFFF_C000_0000u64
		);
	}
	#[test]
	fn test_op_imm32_ill() {
		assert!(trap(exec_op_imm32(1, 1, 0x100, 2)));
	}
}
