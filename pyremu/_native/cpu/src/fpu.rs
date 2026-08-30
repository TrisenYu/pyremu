//! RISC-V F/D 浮点运算 — 基于 softfloat-pure (Berkeley SoftFloat 3 纯 Rust 移植)。
//!
//! 设计: 纯计算函数, 输入原始寄存器 bits + 舍入模式, 输出结果 bits +
//! 异常标志 (fflags) + 目标 (GPR/FPR) + 陷阱标志。寄存器读写与 NaN-boxing
//! 由本模块统一处理; 调用方 (op_dispatcher) 只负责按 `to_gpr` 路由结果。
//!
//! Berkeley SoftFloat 的异常标志值与 RISC-V fflags 位序完全一致
//! (invalid=16=NV, infinite=8=DZ, overflow=4=OF, underflow=2=UF, inexact=1=NX),
//! 故 softfloat 返回的 u8 直接就是 fflags, 无需转换。

use softfloat_pure::softfloat::{float32_t, float64_t};
use softfloat_pure::{Float, RoundingMode, TininessMode};

use crate::concurrent::{ModuleState, StopInfo};
use crate::handlers::{pmp_ok, try_handle_virtio, DevCtx, PmpCtx, EXIT_SENTINEL};
use crate::hart_sched::{
	ram_offset, ram_read_raw, ram_write_raw, read_gpr, read_ram_cross_page, write_ram_cross_page,
};
use crate::peripheral::is_device_addr;
use crate::state::{exit_reason, HartState};
use crate::translate::{translate_va, TranslateFault, WalkCtx};
use crate::trap::{deliver_illegal_instruction, deliver_trap, exc_code, mcause_val};

// ============================================================
//  常量
// ============================================================

/// 单精度 canonical quiet NaN。
const CANONICAL_NAN_S: u32 = 0x7FC0_0000;
/// 双精度 canonical quiet NaN。
const CANONICAL_NAN_D: u64 = 0x7FF8_0000_0000_0000;
/// NaN-boxing 掩码: 单精度值存入 64-bit FPR 时高 32 位全置 1。
const NANBOX_S: u64 = 0xFFFF_FFFF_0000_0000;

/// RISC-V 要求 tininess 在舍入后检测。
#[inline]
fn tininess() -> u8 {
	TininessMode::After.to_softfloat()
}

// ============================================================
//  运算结果
// ============================================================

/// 浮点运算输出 (FFI 兼容, 供 Python fallback 直接调用)。
#[repr(C)]
pub struct FpOut {
	/// 结果值: FPR 目标时为 NaN-boxed bits; GPR 目标时为整数/布尔/分类掩码。
	pub value: u64,
	/// 累积异常标志 (RISC-V fflags 位序)。
	pub fflags: u8,
	/// 1 = 结果写入 rd 的 GPR; 0 = 写入 rd 的 FPR。
	pub to_gpr: u8,
	/// 1 = 非法编码 (调用方投递 IllInstr)。
	pub trap: u8,
}

impl FpOut {
	#[inline]
	fn fpr(value: u64, fflags: u8) -> Self {
		FpOut {
			value,
			fflags,
			to_gpr: 0,
			trap: 0,
		}
	}
	#[inline]
	fn gpr(value: u64, fflags: u8) -> Self {
		FpOut {
			value,
			fflags,
			to_gpr: 1,
			trap: 0,
		}
	}
	#[inline]
	fn illegal() -> Self {
		FpOut {
			value: 0,
			fflags: 0,
			to_gpr: 0,
			trap: 1,
		}
	}
}

// ============================================================
//  NaN-boxing 辅助
// ============================================================

/// 单精度值装箱进 64-bit FPR (高 32 位全 1)。
#[inline]
fn box_s(v: u32) -> u64 {
	NANBOX_S | v as u64
}

/// 从 FPR 取单精度操作数: 若未正确 NaN-boxed, 视为 canonical NaN。
#[inline]
fn unbox_s(bits: u64) -> u32 {
	if (bits & NANBOX_S) == NANBOX_S {
		bits as u32
	} else {
		CANONICAL_NAN_S
	}
}

// ============================================================
//  舍入模式解析
// ============================================================

/// 指令 rm 字段 (funct3) + 动态 frm -> softfloat RoundingMode。
/// rm==7 (DYN) 时取 fcsr.frm; 保留值 (5,6, 或 frm 本身为 7) -> None (非法)。
#[inline]
fn resolve_rm(inst_rm: u8, frm: u8) -> Option<RoundingMode> {
	let eff = if inst_rm == 7 { frm } else { inst_rm };
	match eff {
		0 => Some(RoundingMode::RneTiesToEven),
		1 => Some(RoundingMode::RtzTowardZero),
		2 => Some(RoundingMode::RdnTowardNegative),
		3 => Some(RoundingMode::RupTowardPositive),
		4 => Some(RoundingMode::RmmTiesToAway),
		_ => None,
	}
}

// ============================================================
//  FCLASS — 分类掩码 (10 bit)
// ============================================================

fn fclass_s(bits: u32) -> u64 {
	fclass_generic(&float32_t::from_bits(bits))
}

fn fclass_d(bits: u64) -> u64 {
	fclass_generic(&float64_t::from_bits(bits))
}

/// 构建 RISC-V FCLASS 10-bit 分类掩码。
fn fclass_generic<F: Float>(f: &F) -> u64 {
	let mut m = 0u64;
	if f.is_negative_infinity() {
		m |= 1 << 0;
	}
	if f.is_negative_normal() {
		m |= 1 << 1;
	}
	if f.is_negative_subnormal() {
		m |= 1 << 2;
	}
	if f.is_negative_zero() {
		m |= 1 << 3;
	}
	if f.is_positive_zero() {
		m |= 1 << 4;
	}
	if f.is_positive_subnormal() {
		m |= 1 << 5;
	}
	if f.is_positive_normal() {
		m |= 1 << 6;
	}
	if f.is_positive_infinity() {
		m |= 1 << 7;
	}
	if f.is_signaling_nan() {
		m |= 1 << 8;
	} else if f.is_nan() {
		m |= 1 << 9;
	}
	m
}

// ============================================================
//  FSGNJ — 符号注入 (纯位操作, 无 fflags)
// ============================================================

#[inline]
fn sgnj_s(rs1: u32, rs2: u32, funct3: u8) -> Option<u32> {
	let sign_bit = 1u32 << 31;
	let mag = rs1 & !sign_bit;
	let sign = match funct3 {
		0 => rs2 & sign_bit,         // FSGNJ
		1 => !rs2 & sign_bit,        // FSGNJN
		2 => (rs1 ^ rs2) & sign_bit, // FSGNJX
		_ => return None,
	};
	Some(mag | sign)
}

#[inline]
fn sgnj_d(rs1: u64, rs2: u64, funct3: u8) -> Option<u64> {
	let sign_bit = 1u64 << 63;
	let mag = rs1 & !sign_bit;
	let sign = match funct3 {
		0 => rs2 & sign_bit,
		1 => !rs2 & sign_bit,
		2 => (rs1 ^ rs2) & sign_bit,
		_ => return None,
	};
	Some(mag | sign)
}

// ============================================================
//  FMIN / FMAX — RISC-V 2.2 语义
// ============================================================

/// 返回 (结果 bits, fflags)。is_max=false -> FMIN。
fn minmax_s(a: u32, b: u32, is_max: bool) -> (u32, u8) {
	let fa = float32_t::from_bits(a);
	let fb = float32_t::from_bits(b);
	let mut fflags = 0u8;
	// 任一为 signaling NaN -> NV。
	if fa.is_signaling_nan() || fb.is_signaling_nan() {
		fflags |= 16;
	}
	// 两者皆 NaN -> canonical NaN。
	if fa.is_nan() && fb.is_nan() {
		return (CANONICAL_NAN_S, fflags);
	}
	// 一方 NaN -> 返回另一方。
	if fa.is_nan() {
		return (b, fflags);
	}
	if fb.is_nan() {
		return (a, fflags);
	}
	// -0 < +0 特殊处理。
	if fa.is_zero() && fb.is_zero() {
		let a_neg = (a >> 31) & 1 == 1;
		let pick_a = if is_max { !a_neg } else { a_neg };
		return (if pick_a { a } else { b }, fflags);
	}
	let (lt, _) = fa.lt(fb);
	let pick_a = if is_max { !lt } else { lt };
	(if pick_a { a } else { b }, fflags)
}

fn minmax_d(a: u64, b: u64, is_max: bool) -> (u64, u8) {
	let fa = float64_t::from_bits(a);
	let fb = float64_t::from_bits(b);
	let mut fflags = 0u8;
	if fa.is_signaling_nan() || fb.is_signaling_nan() {
		fflags |= 16;
	}
	if fa.is_nan() && fb.is_nan() {
		return (CANONICAL_NAN_D, fflags);
	}
	if fa.is_nan() {
		return (b, fflags);
	}
	if fb.is_nan() {
		return (a, fflags);
	}
	if fa.is_zero() && fb.is_zero() {
		let a_neg = (a >> 63) & 1 == 1;
		let pick_a = if is_max { !a_neg } else { a_neg };
		return (if pick_a { a } else { b }, fflags);
	}
	let (lt, _) = fa.lt(fb);
	let pick_a = if is_max { !lt } else { lt };
	(if pick_a { a } else { b }, fflags)
}

// ============================================================
//  OP-FP 主分派 (opcode 0x53)
// ============================================================

/// 执行 OP-FP 指令。
///
/// - `funct7`: bits[31:25] — funct7[1:0]=fmt (0=S,1=D), funct7[6:2]=op family
/// - `funct3`: bits[14:12] — 算术类为 rm; 其余为子操作选择
/// - `rs2`:    bits[24:20] — CVT 类用于选择整数宽度/方向
/// - `rs1_bits`/`rs2_bits`: 从 FPR 读取的原始 bits (整数源操作数经此传入)
/// - `frm`:    fcsr.frm (rm==DYN 时使用)
#[no_mangle]
pub extern "C" fn fp_exec_op(
	funct7: u8,
	funct3: u8,
	rs2: u8,
	rs1_bits: u64,
	rs2_bits: u64,
	frm: u8,
) -> FpOut {
	let fmt = funct7 & 0x3;
	let op5 = funct7 >> 2;
	let is_double = fmt == 1;
	if fmt != 0 && fmt != 1 {
		return FpOut::illegal(); // 仅支持 S/D
	}

	match op5 {
		0x00 | 0x01 | 0x02 | 0x03 => arith(op5, funct3, is_double, rs1_bits, rs2_bits, frm),
		0x04 => sgnj(funct3, is_double, rs1_bits, rs2_bits),
		0x05 => minmax(funct3, is_double, rs1_bits, rs2_bits),
		0x0B => sqrt(rs2, funct3, is_double, rs1_bits, frm),
		0x14 => compare(funct3, is_double, rs1_bits, rs2_bits),
		0x18 => cvt_f2i(rs2, funct3, is_double, rs1_bits, frm),
		0x1A => cvt_i2f(rs2, funct3, is_double, rs1_bits, frm),
		0x1C => fmv_x_or_class(funct3, is_double, rs1_bits),
		0x1E => fmv_to_fpr(funct3, is_double, rs1_bits),
		0x08 => cvt_f2f(rs2, is_double, rs1_bits, funct3, frm),
		_ => FpOut::illegal(),
	}
}

// ---- 算术: ADD/SUB/MUL/DIV ----

fn arith(op5: u8, rm: u8, is_double: bool, a: u64, b: u64, frm: u8) -> FpOut {
	let Some(rnd) = resolve_rm(rm, frm) else {
		return FpOut::illegal();
	};
	let t = tininess();
	if is_double {
		let fa = float64_t::from_bits(a);
		let fb = float64_t::from_bits(b);
		let (r, fl) = match op5 {
			0x00 => fa.add(fb, rnd, t),
			0x01 => fa.sub(fb, rnd, t),
			0x02 => fa.mul(fb, rnd, t),
			0x03 => fa.div(fb, rnd, t),
			_ => unreachable!(),
		};
		FpOut::fpr(r.to_bits(), fl)
	} else {
		let fa = float32_t::from_bits(unbox_s(a));
		let fb = float32_t::from_bits(unbox_s(b));
		let (r, fl) = match op5 {
			0x00 => fa.add(fb, rnd, t),
			0x01 => fa.sub(fb, rnd, t),
			0x02 => fa.mul(fb, rnd, t),
			0x03 => fa.div(fb, rnd, t),
			_ => unreachable!(),
		};
		FpOut::fpr(box_s(r.to_bits()), fl)
	}
}

// ---- FSQRT ----

fn sqrt(rs2: u8, rm: u8, is_double: bool, a: u64, frm: u8) -> FpOut {
	if rs2 != 0 {
		return FpOut::illegal();
	}
	let Some(rnd) = resolve_rm(rm, frm) else {
		return FpOut::illegal();
	};
	let t = tininess();
	if is_double {
		let (r, fl) = float64_t::from_bits(a).sqrt(rnd, t);
		FpOut::fpr(r.to_bits(), fl)
	} else {
		let (r, fl) = float32_t::from_bits(unbox_s(a)).sqrt(rnd, t);
		FpOut::fpr(box_s(r.to_bits()), fl)
	}
}

// ---- FSGNJ ----

fn sgnj(funct3: u8, is_double: bool, a: u64, b: u64) -> FpOut {
	if is_double {
		match sgnj_d(a, b, funct3) {
			Some(r) => FpOut::fpr(r, 0),
			None => FpOut::illegal(),
		}
	} else {
		match sgnj_s(unbox_s(a), unbox_s(b), funct3) {
			Some(r) => FpOut::fpr(box_s(r), 0),
			None => FpOut::illegal(),
		}
	}
}

// ---- FMIN/FMAX ----

fn minmax(funct3: u8, is_double: bool, a: u64, b: u64) -> FpOut {
	let is_max = match funct3 {
		0 => false,
		1 => true,
		_ => return FpOut::illegal(),
	};
	if is_double {
		let (r, fl) = minmax_d(a, b, is_max);
		FpOut::fpr(r, fl)
	} else {
		let (r, fl) = minmax_s(unbox_s(a), unbox_s(b), is_max);
		FpOut::fpr(box_s(r), fl)
	}
}

// ---- FEQ/FLT/FLE -> GPR ----

fn compare(funct3: u8, is_double: bool, a: u64, b: u64) -> FpOut {
	if is_double {
		let fa = float64_t::from_bits(a);
		let fb = float64_t::from_bits(b);
		let (res, fl) = match funct3 {
			0 => fa.le(fb),          // FLE
			1 => fa.lt(fb),          // FLT
			2 => Float::eq(&fa, fb), // FEQ (quiet); 消歧 PartialEq::eq
			_ => return FpOut::illegal(),
		};
		FpOut::gpr(u64::from(res), fl)
	} else {
		let fa = float32_t::from_bits(unbox_s(a));
		let fb = float32_t::from_bits(unbox_s(b));
		let (res, fl) = match funct3 {
			0 => fa.le(fb),
			1 => fa.lt(fb),
			2 => Float::eq(&fa, fb),
			_ => return FpOut::illegal(),
		};
		FpOut::gpr(u64::from(res), fl)
	}
}

// ---- FCVT float->int -> GPR ----

fn cvt_f2i(rs2: u8, rm: u8, is_double: bool, a: u64, frm: u8) -> FpOut {
	let Some(rnd) = resolve_rm(rm, frm) else {
		return FpOut::illegal();
	};
	// rs2: 0=W(i32), 1=WU(u32), 2=L(i64), 3=LU(u64)
	let (val, fl): (u64, u8) = if is_double {
		let f = float64_t::from_bits(a);
		match rs2 {
			0 => {
				let (v, fl) = f.to_i32(rnd, true);
				(v as i64 as u64, fl) // 符号扩展到 64
			}
			1 => {
				let (v, fl) = f.to_u32(rnd, true);
				(v as i32 as i64 as u64, fl) // W/WU 结果符号扩展 (RV64 约定)
			}
			2 => {
				let (v, fl) = f.to_i64(rnd, true);
				(v as u64, fl)
			}
			3 => {
				let (v, fl) = f.to_u64(rnd, true);
				(v, fl)
			}
			_ => return FpOut::illegal(),
		}
	} else {
		let f = float32_t::from_bits(unbox_s(a));
		match rs2 {
			0 => {
				let (v, fl) = f.to_i32(rnd, true);
				(v as i64 as u64, fl)
			}
			1 => {
				let (v, fl) = f.to_u32(rnd, true);
				(v as i32 as i64 as u64, fl)
			}
			2 => {
				let (v, fl) = f.to_i64(rnd, true);
				(v as u64, fl)
			}
			3 => {
				let (v, fl) = f.to_u64(rnd, true);
				(v, fl)
			}
			_ => return FpOut::illegal(),
		}
	};
	FpOut::gpr(val, fl)
}

// ---- FCVT int->float -> FPR ----

fn cvt_i2f(rs2: u8, rm: u8, is_double: bool, gpr_bits: u64, frm: u8) -> FpOut {
	let Some(rnd) = resolve_rm(rm, frm) else {
		return FpOut::illegal();
	};
	let t = tininess();
	// rs2: 0=W(i32), 1=WU(u32), 2=L(i64), 3=LU(u64)
	if is_double {
		let (r, fl) = match rs2 {
			0 => float64_t::from_i32(gpr_bits as i32, rnd, t),
			1 => float64_t::from_u32(gpr_bits as u32, rnd, t),
			2 => float64_t::from_i64(gpr_bits as i64, rnd, t),
			3 => float64_t::from_u64(gpr_bits, rnd, t),
			_ => return FpOut::illegal(),
		};
		FpOut::fpr(r.to_bits(), fl)
	} else {
		let (r, fl) = match rs2 {
			0 => float32_t::from_i32(gpr_bits as i32, rnd, t),
			1 => float32_t::from_u32(gpr_bits as u32, rnd, t),
			2 => float32_t::from_i64(gpr_bits as i64, rnd, t),
			3 => float32_t::from_u64(gpr_bits, rnd, t),
			_ => return FpOut::illegal(),
		};
		FpOut::fpr(box_s(r.to_bits()), fl)
	}
}

// ---- FMV.X.W/D + FCLASS -> GPR ----

fn fmv_x_or_class(funct3: u8, is_double: bool, a: u64) -> FpOut {
	match funct3 {
		0 => {
			// FMV.X.W / FMV.X.D — 位拷贝到 GPR。
			if is_double {
				FpOut::gpr(a, 0)
			} else {
				// FMV.X.W: 取低 32 位, 符号扩展到 64 (RV64)。
				FpOut::gpr(unbox_s(a) as i32 as i64 as u64, 0)
			}
		}
		1 => {
			// FCLASS。
			let mask = if is_double {
				fclass_d(a)
			} else {
				fclass_s(unbox_s(a))
			};
			FpOut::gpr(mask, 0)
		}
		_ => FpOut::illegal(),
	}
}

// ---- FMV.W.X / FMV.D.X -> FPR ----

fn fmv_to_fpr(funct3: u8, is_double: bool, gpr_bits: u64) -> FpOut {
	if funct3 != 0 {
		return FpOut::illegal();
	}
	if is_double {
		FpOut::fpr(gpr_bits, 0)
	} else {
		FpOut::fpr(box_s(gpr_bits as u32), 0)
	}
}

// ---- FCVT.S.D / FCVT.D.S ----

fn cvt_f2f(rs2: u8, is_double_dst: bool, a: u64, rm: u8, frm: u8) -> FpOut {
	let Some(rnd) = resolve_rm(rm, frm) else {
		return FpOut::illegal();
	};
	let t = tininess();
	if is_double_dst {
		// FCVT.D.S: 源为单精度 (rs2 应为 0)。
		if rs2 != 0 {
			return FpOut::illegal();
		}
		let (r, fl) = float32_t::from_bits(unbox_s(a)).to_f64(rnd, t);
		FpOut::fpr(r.to_bits(), fl)
	} else {
		// FCVT.S.D: 源为双精度 (rs2 应为 1)。
		if rs2 != 1 {
			return FpOut::illegal();
		}
		let (r, fl) = float64_t::from_bits(a).to_f32(rnd, t);
		FpOut::fpr(box_s(r.to_bits()), fl)
	}
}

// ============================================================
//  FMA — FMADD/FMSUB/FNMSUB/FNMADD (opcode 0x43/0x47/0x4B/0x4F)
// ============================================================

/// 执行 FMA 指令。opcode 区分四种变体, fmt 区分 S/D。
/// funct3 = rm。rs1/rs2/rs3 为 FPR 原始 bits。
///
/// - FMADD:  rs1*rs2 + rs3
/// - FMSUB:  rs1*rs2 - rs3
/// - FNMSUB: -(rs1*rs2) + rs3
/// - FNMADD: -(rs1*rs2) - rs3
#[no_mangle]
pub extern "C" fn fp_exec_fma(
	opcode: u8,
	rm: u8,
	fmt: u8,
	rs1: u64,
	rs2: u64,
	rs3: u64,
	frm: u8,
) -> FpOut {
	let Some(rnd) = resolve_rm(rm, frm) else {
		return FpOut::illegal();
	};
	if fmt != 0 && fmt != 1 {
		return FpOut::illegal();
	}
	let t = tininess();
	// 变体的符号处理: 对被乘数取反 (NMSUB/NMADD), 对加数取反 (FMSUB/FNMADD)。
	let neg_prod = matches!(opcode, 0x4B | 0x4F); // FNMSUB / FNMADD
	let neg_add = matches!(opcode, 0x47 | 0x4F); // FMSUB / FNMADD

	if fmt == 1 {
		let a = maybe_neg_d(rs1, neg_prod);
		let b = float64_t::from_bits(rs2);
		let c = maybe_neg_d(rs3, neg_add);
		let (r, fl) = float64_t::from_bits(a).fused_mul_add(b, float64_t::from_bits(c), rnd, t);
		FpOut::fpr(r.to_bits(), fl)
	} else {
		let a = maybe_neg_s(unbox_s(rs1), neg_prod);
		let b = float32_t::from_bits(unbox_s(rs2));
		let c = maybe_neg_s(unbox_s(rs3), neg_add);
		let (r, fl) = float32_t::from_bits(a).fused_mul_add(b, float32_t::from_bits(c), rnd, t);
		FpOut::fpr(box_s(r.to_bits()), fl)
	}
}

#[inline]
fn maybe_neg_s(bits: u32, neg: bool) -> u32 {
	if neg {
		bits ^ (1 << 31)
	} else {
		bits
	}
}

#[inline]
fn maybe_neg_d(bits: u64, neg: bool) -> u64 {
	if neg {
		bits ^ (1 << 63)
	} else {
		bits
	}
}

// ============================================================
//  FP load/store — extracted from concurrent.rs
// ============================================================

const NANBOX_S_CC: u64 = 0xFFFF_FFFF_0000_0000;
pub(crate) const MSTATUS_FS_CC: u64 = 0b11 << 13;
pub(crate) const MSTATUS_SD_CC: u64 = 1 << 63;

pub(crate) fn handle_fp_load_concurrent(
	state: &mut HartState,
	f: &crate::decode::DecodedFields,
	instr: u32,
	ctx: &WalkCtx,
	pmp: &PmpCtx,
	dev: &DevCtx,
	module: &ModuleState,
) -> u64 {
	if (state.mstatus & MSTATUS_FS_CC) == 0 {
		deliver_illegal_instruction(state, instr as u64);
		return 0;
	}
	let base = read_gpr(state, f.rs1);
	let va = base.wrapping_add(f.imm12_se);
	let size: u8 = match f.func3 {
		0b010 => 4, // FLW
		0b011 => 8, // FLD
		_ => {
			deliver_illegal_instruction(state, instr as u64);
			return 0;
		}
	};
	let aligned = va & (size as u64 - 1) == 0;

	let tr = match translate_va(state, ctx, va, false, false) {
		Ok(t) => t,
		Err(TranslateFault::PageFault(cause)) => {
			deliver_trap(state, mcause_val(cause, false), va);
			return 0;
		}
		Err(TranslateFault::AccessFault) => {
			deliver_trap(state, mcause_val(exc_code::LD_ACCESS_FAULT, false), va);
			return 0;
		}
	};
	if !pmp_ok(state, tr.pa, size as u32, false, false, pmp) {
		deliver_trap(state, mcause_val(exc_code::LD_ACCESS_FAULT, false), va);
		return 0;
	}
	// virtio-blk inline check before generic MMIO exit.
	if let Some(data) = try_handle_virtio(tr.pa, false, 0, size, dev) {
		let boxed = if size == 4 {
			0xFFFF_FFFF_0000_0000 | data
		} else {
			data
		};
		state.fprs[f.rd as usize] = boxed;
		state.mstatus |= 0b11 << 13 | 1 << 63; // FS + SD
		return 4;
	}
	// 设备地址退回 Python MMIO 处理 (FP 从 MMIO 加载罕见)。
	if is_device_addr(tr.pa, dev) {
		module.request_stop(StopInfo {
			reason: exit_reason::MMIO,
			hart_id: state.mhartid as u8,
			pc: state.pc,
			instr,
			..StopInfo::empty()
		});
		return EXIT_SENTINEL;
	}
	if ram_offset(
		tr.pa,
		size as u32,
		ctx.ram_base,
		ctx.ram_size,
		ctx.shadow_base,
		ctx.shadow_size,
	)
	.is_none()
	{
		deliver_trap(state, mcause_val(exc_code::LD_ACCESS_FAULT, false), va);
		return 0;
	}

	// Cross-page check must come BEFORE alignment check (see hart_sched::handle_load_concurrent).
	let val = if (va & 0xFFF) + size as u64 > 0x1000 {
		let (v, ok) = read_ram_cross_page(state, ctx, va, tr.pa, size, pmp);
		if !ok {
			return 0;
		}
		v
	} else if aligned {
		ram_read_raw(ctx, tr.pa, size)
	} else {
		let mut r: u64 = 0;
		for b in 0..size {
			r |= ram_read_raw(ctx, tr.pa + b as u64, 1) << (b * 8);
		}
		r
	};
	let boxed = if size == 4 { NANBOX_S_CC | val } else { val };
	state.fprs[f.rd as usize] = boxed;
	state.mstatus |= MSTATUS_FS_CC | MSTATUS_SD_CC;
	4
}

pub(crate) fn handle_fp_store_concurrent(
	state: &mut HartState,
	f: &crate::decode::DecodedFields,
	instr: u32,
	ctx: &WalkCtx,
	pmp: &PmpCtx,
	dev: &DevCtx,
	module: &ModuleState,
) -> u64 {
	if (state.mstatus & MSTATUS_FS_CC) == 0 {
		deliver_illegal_instruction(state, instr as u64);
		return 0;
	}
	let base = read_gpr(state, f.rs1);
	let va = base.wrapping_add(f.imm_s);
	let size: u8 = match f.func3 {
		0b010 => 4, // FSW
		0b011 => 8, // FSD
		_ => {
			deliver_illegal_instruction(state, instr as u64);
			return 0;
		}
	};
	let aligned = va & (size as u64 - 1) == 0;

	let tr = match translate_va(state, ctx, va, true, false) {
		Ok(t) => t,
		Err(TranslateFault::PageFault(cause)) => {
			deliver_trap(state, mcause_val(cause, false), va);
			return 0;
		}
		Err(TranslateFault::AccessFault) => {
			deliver_trap(state, mcause_val(exc_code::ST_ACCESS_FAULT, false), va);
			return 0;
		}
	};
	if !pmp_ok(state, tr.pa, size as u32, true, false, pmp) {
		deliver_trap(state, mcause_val(exc_code::ST_ACCESS_FAULT, false), va);
		return 0;
	}
	// virtio-blk inline check before generic MMIO exit.
	let val_fp = state.fprs[f.rs2 as usize];
	if let Some(_) = try_handle_virtio(tr.pa, true, val_fp, size, dev) {
		return 4;
	}
	if is_device_addr(tr.pa, dev) {
		module.request_stop(StopInfo {
			reason: exit_reason::MMIO,
			hart_id: state.mhartid as u8,
			pc: state.pc,
			instr,
			..StopInfo::empty()
		});
		return EXIT_SENTINEL;
	}
	if ram_offset(
		tr.pa,
		size as u32,
		ctx.ram_base,
		ctx.ram_size,
		ctx.shadow_base,
		ctx.shadow_size,
	)
	.is_none()
	{
		deliver_trap(state, mcause_val(exc_code::ST_ACCESS_FAULT, false), va);
		return 0;
	}

	// Cross-page check must come BEFORE alignment check (see hart_sched::handle_store_concurrent).
	if (va & 0xFFF) + size as u64 > 0x1000 {
		if !write_ram_cross_page(state, ctx, va, tr.pa, val_fp, size, pmp) {
			return 0;
		}
	} else if aligned {
		ram_write_raw(ctx, tr.pa, val_fp, size);
	} else {
		for b in 0..size {
			let byte = (val_fp >> (b * 8)) as u8;
			ram_write_raw(ctx, tr.pa + b as u64, byte as u64, 1);
		}
	}
	4
}

#[cfg(test)]
mod tests {
	use super::*;

	// 单精度: 1.0 = 0x3F800000, 2.0 = 0x40000000, 3.0 = 0x40400000
	const F1_0: u32 = 0x3F80_0000;
	const F2_0: u32 = 0x4000_0000;
	const F3_0: u32 = 0x4040_0000;
	// 双精度: 1.0 = 0x3FF0..., 2.0 = 0x4000...
	const D1_0: u64 = 0x3FF0_0000_0000_0000;
	const D2_0: u64 = 0x4000_0000_0000_0000;
	const D3_0: u64 = 0x4008_0000_0000_0000;

	#[test]
	fn fadd_s_1_plus_2_eq_3() {
		// funct7=0x00 (ADD.S), rm=0(RNE)
		let out = fp_exec_op(0x00, 0, 0, box_s(F1_0), box_s(F2_0), 0);
		assert_eq!(out.trap, 0);
		assert_eq!(out.to_gpr, 0);
		assert_eq!(out.value, box_s(F3_0));
	}

	#[test]
	fn fadd_d_1_plus_2_eq_3() {
		let out = fp_exec_op(0x01, 0, 0, D1_0, D2_0, 0);
		assert_eq!(out.trap, 0);
		assert_eq!(out.value, D3_0);
	}

	#[test]
	fn fmul_s_2_times_3_eq_6() {
		let f6_0: u32 = 0x40C0_0000;
		let out = fp_exec_op(0x08, 0, 0, box_s(F2_0), box_s(F3_0), 0);
		assert_eq!(out.value, box_s(f6_0));
	}

	#[test]
	fn fsub_d_3_minus_1_eq_2() {
		let out = fp_exec_op(0x05, 0, 0, D3_0, D1_0, 0);
		assert_eq!(out.value, D2_0);
	}

	#[test]
	fn fdiv_s_6_div_2_eq_3() {
		let f6_0: u32 = 0x40C0_0000;
		let out = fp_exec_op(0x0C, 0, 0, box_s(f6_0), box_s(F2_0), 0);
		assert_eq!(out.value, box_s(F3_0));
	}

	#[test]
	fn fsqrt_s_4_eq_2() {
		let f4_0: u32 = 0x4080_0000;
		// funct7=0x2C (SQRT.S), rs2=0
		let out = fp_exec_op(0x2C, 0, 0, box_s(f4_0), 0, 0);
		assert_eq!(out.value, box_s(F2_0));
	}

	#[test]
	fn feq_s_equal_returns_1() {
		// funct7=0x50 (CMP.S), funct3=2 (EQ)
		let out = fp_exec_op(0x50, 2, 0, box_s(F1_0), box_s(F1_0), 0);
		assert_eq!(out.to_gpr, 1);
		assert_eq!(out.value, 1);
	}

	#[test]
	fn flt_s_1_lt_2_returns_1() {
		let out = fp_exec_op(0x50, 1, 0, box_s(F1_0), box_s(F2_0), 0);
		assert_eq!(out.to_gpr, 1);
		assert_eq!(out.value, 1);
	}

	#[test]
	fn fcvt_w_s_2_5_rtz_eq_2() {
		let f2_5: u32 = 0x4020_0000; // 2.5
							   // funct7=0x60 (F2I.S), rs2=0 (W), rm=1 (RTZ)
		let out = fp_exec_op(0x60, 1, 0, box_s(f2_5), 0, 0);
		assert_eq!(out.to_gpr, 1);
		assert_eq!(out.value, 2);
	}

	#[test]
	fn fcvt_s_w_3_eq_3_0() {
		// funct7=0x68 (I2F.S), rs2=0 (W), rm=0
		let out = fp_exec_op(0x68, 0, 0, 3, 0, 0);
		assert_eq!(out.to_gpr, 0);
		assert_eq!(out.value, box_s(F3_0));
	}

	#[test]
	fn fclass_s_neg_inf() {
		let neg_inf: u32 = 0xFF80_0000;
		// funct7=0x70 (FMV.X/CLASS.S), funct3=1 (FCLASS)
		let out = fp_exec_op(0x70, 1, 0, box_s(neg_inf), 0, 0);
		assert_eq!(out.to_gpr, 1);
		assert_eq!(out.value, 1 << 0); // -inf
	}

	#[test]
	fn fsgnj_s_copies_sign() {
		let neg1: u32 = 0xBF80_0000; // -1.0
							   // funct7=0x10 (SGNJ.S), funct3=0 (FSGNJ): mag(1.0) sign(-1.0) = -1.0
		let out = fp_exec_op(0x10, 0, 0, box_s(F1_0), box_s(neg1), 0);
		assert_eq!(out.value, box_s(neg1));
	}

	#[test]
	fn fmin_s_picks_smaller() {
		let out = fp_exec_op(0x14, 0, 0, box_s(F1_0), box_s(F2_0), 0);
		assert_eq!(out.value, box_s(F1_0));
	}

	#[test]
	fn fmax_s_picks_larger() {
		let out = fp_exec_op(0x14, 1, 0, box_s(F1_0), box_s(F2_0), 0);
		assert_eq!(out.value, box_s(F2_0));
	}

	#[test]
	fn fmadd_s_2x3_plus_1_eq_7() {
		let f7_0: u32 = 0x40E0_0000;
		// FMADD opcode 0x43, fmt=0, rm=0
		let out = fp_exec_fma(0x43, 0, 0, box_s(F2_0), box_s(F3_0), box_s(F1_0), 0);
		assert_eq!(out.trap, 0);
		assert_eq!(out.value, box_s(f7_0));
	}

	#[test]
	fn fmsub_s_2x3_minus_1_eq_5() {
		let f5_0: u32 = 0x40A0_0000;
		let out = fp_exec_fma(0x47, 0, 0, box_s(F2_0), box_s(F3_0), box_s(F1_0), 0);
		assert_eq!(out.value, box_s(f5_0));
	}

	#[test]
	fn fmv_x_w_bit_copy() {
		// funct7=0x70, funct3=0 (FMV.X.W)
		let out = fp_exec_op(0x70, 0, 0, box_s(F1_0), 0, 0);
		assert_eq!(out.to_gpr, 1);
		assert_eq!(out.value, F1_0 as u64); // 正数无符号扩展影响
	}

	#[test]
	fn illegal_rm_traps() {
		// rm=5 (保留) -> 非法
		let out = fp_exec_op(0x00, 5, 0, box_s(F1_0), box_s(F2_0), 0);
		assert_eq!(out.trap, 1);
	}

	#[test]
	fn dyn_rm_uses_frm() {
		// rm=7 (DYN), frm=1 (RTZ): FCVT.W.S 2.9 -> 2
		let f2_9: u32 = 0x4039_9999; // ~2.9
		let out = fp_exec_op(0x60, 7, 0, box_s(f2_9), 0, 1);
		assert_eq!(out.value, 2);
	}
}
