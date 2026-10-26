//! 纯计算的指令处理函数。
//!
//! 每个处理函数接收 ``&mut HartState`` 与 ``&DecodedFields``, 执行指令后返回
//! PC 的推进量, 取值为 0 或 4。遇到非法编码时调用 ``deliver_illegal_instruction``
//! 并返回 0, 此时 PC 已被设置为 mtvec 的取值。

use crate::alu::{exec_alu_op, exec_op32, exec_op_imm, exec_op_imm32};
use crate::decode::DecodedFields;
use crate::fpu::{fp_exec_fma, fp_exec_op};
use crate::state::HartState;
use crate::trap::deliver_illegal_instruction;

// ============================================================
//  辅助函数
// ============================================================

/// mstatus.FS 字段的掩码, 取自 bits[14:13], 表示浮点单元状态。
const MSTATUS_FS: u64 = 0b11 << 13;
/// mstatus 的 SD 位, 取自 bit 63, 任一扩展状态为 Dirty 时置位。
const MSTATUS_SD: u64 = 1 << 63;

/// 将 ``val`` 写入通用寄存器 ``rd``; x0 恒为 0, 不写入。
#[inline]
fn write_gpr(state: &mut HartState, rd: u8, val: u64) {
	if rd != 0 {
		state.gprs[rd as usize] = val;
	}
}

/// 读取通用寄存器 ``rs``; x0 恒返回 0。
#[inline]
fn read_gpr(state: &HartState, rs: u8) -> u64 {
	if rs == 0 {
		0
	} else {
		state.gprs[rs as usize]
	}
}

/// 检查 ALU 结果的陷阱标记; 置位时投递 IllInstr 并返回 true。
#[inline]
fn check_alu_trap(state: &mut HartState, trap: u8, tval: u64) -> bool {
	if trap == 0 {
		return false;
	}
	deliver_illegal_instruction(state, tval);
	return true;
}

// ============================================================
//  R 型 ALU 指令: ADD、SUB、SLL、SLT、
//  SLTU、XOR、SRL、SRA、OR、AND、MUL、
//  MULH、MULHSU、MULHU、DIV、DIVU、REM、REMU
// ============================================================

pub fn handle_alu(state: &mut HartState, f: &DecodedFields, instr: u32) -> u64 {
	let v1 = read_gpr(state, f.rs1);
	let v2 = read_gpr(state, f.rs2);
	let r = exec_alu_op(f.func3, f.func7, v1, v2);
	if check_alu_trap(state, r.trap, instr as u64) {
		return 0;
	}
	write_gpr(state, f.rd, r.value);
	4
}

// ============================================================
//  I 型 ALU 指令: ADDI、SLLI、SLTI、SLTIU、XORI、SRLI、SRAI、ORI、ANDI
// ============================================================

pub fn handle_op_imm(state: &mut HartState, f: &DecodedFields, instr: u32) -> u64 {
	let v1 = read_gpr(state, f.rs1);
	let r = exec_op_imm(f.func3, f.func7, v1, f.imm12_se);
	if check_alu_trap(state, r.trap, instr as u64) {
		return 0;
	}
	write_gpr(state, f.rd, r.value);
	4
}

// ============================================================
//  RV64 的 32 位字 ALU 指令: ADDW、SUBW、SLLW、SRLW、SRAW、MULW、DIVW、DIVUW、REMW、REMUW
// ============================================================

pub fn handle_op32(state: &mut HartState, f: &DecodedFields, instr: u32) -> u64 {
	let v1 = read_gpr(state, f.rs1);
	let v2 = read_gpr(state, f.rs2);
	let r = exec_op32(f.func3, f.func7, v1, v2);
	if check_alu_trap(state, r.trap, instr as u64) {
		return 0;
	}
	write_gpr(state, f.rd, r.value);
	4
}

// ============================================================
//  RV64 的 32 位立即数 ALU 指令: ADDIW、SLLIW、SRLIW、SRAIW
// ============================================================

pub fn handle_op_imm32(state: &mut HartState, f: &DecodedFields, instr: u32) -> u64 {
	let v1 = read_gpr(state, f.rs1);
	let r = exec_op_imm32(f.func3, f.func7, v1, f.imm12_se);
	if check_alu_trap(state, r.trap, instr as u64) {
		return 0;
	}
	write_gpr(state, f.rd, r.value);
	4
}

// ============================================================
//  F/D 浮点指令
// ============================================================

/// 读取 FPR 的原始位值; 其中 f0 区别于恒为 0 的 x0，是正常的浮点寄存器。
#[inline]
fn read_fpr(state: &HartState, r: u8) -> u64 {
	state.fprs[r as usize]
}

/// 写入 FPR 的原始位值。
#[inline]
fn write_fpr(state: &mut HartState, r: u8, v: u64) {
	state.fprs[r as usize] = v;
}

/// 通过 mstatus.FS 是否为 Off 来判定浮点单元是否被禁用。
#[inline]
fn fp_disabled(state: &HartState) -> bool {
	(state.mstatus & MSTATUS_FS) == 0
}

/// 执行浮点指令后将 FS 置为 Dirty, 即 0b11, 并设置 SD。
#[inline]
fn set_fs_dirty(state: &mut HartState) {
	state.mstatus |= MSTATUS_FS | MSTATUS_SD;
}

/// 当前的动态舍入模式, 取自 fcsr.frm。
#[inline]
fn cur_frm(state: &HartState) -> u8 {
	((state.fcsr >> 5) & 0x7) as u8
}

/// OP-FP, opcode 为 0x53: 算术、转换、比较、符号、分类与移动指令。
pub fn handle_fp_op(state: &mut HartState, f: &DecodedFields, instr: u32) -> u64 {
	if fp_disabled(state) {
		deliver_illegal_instruction(state, instr as u64);
		return 0;
	}
	let op5 = f.func7 >> 2;
	// 0x1A 的整数转浮点与 0x1E 的 FMV.*.X, 其 rs1 源为通用寄存器; 其余指令的 rs1 源为 FPR。
	let rs1_bits = if matches!(op5, 0x1a | 0x1e) {
		read_gpr(state, f.rs1)
	} else {
		read_fpr(state, f.rs1)
	};
	let rs2_bits = read_fpr(state, f.rs2);
	let out = fp_exec_op(f.func7, f.func3, f.rs2, rs1_bits, rs2_bits, cur_frm(state));
	if out.trap != 0 {
		deliver_illegal_instruction(state, instr as u64);
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

/// FMA, opcode 为 0x43、0x47、0x4B 与 0x4F: 合并乘加的四种变体。
pub fn handle_fp_fma(state: &mut HartState, f: &DecodedFields, instr: u32) -> u64 {
	if fp_disabled(state) {
		deliver_illegal_instruction(state, instr as u64);
		return 0;
	}
	let rs1 = read_fpr(state, f.rs1);
	let rs2 = read_fpr(state, f.rs2);
	let rs3 = read_fpr(state, f.rs3);
	let out = fp_exec_fma(f.opcode, f.func3, f.fmt, rs1, rs2, rs3, cur_frm(state));
	if out.trap != 0 {
		deliver_illegal_instruction(state, instr as u64);
		return 0;
	}
	write_fpr(state, f.rd, out.value);
	state.fcsr |= u32::from(out.fflags);
	set_fs_dirty(state);
	4
}

// ============================================================
//  LUI: rd = imm20 << 12, 结果按 32 位符号扩展
// ============================================================

pub fn handle_lui(state: &mut HartState, f: &DecodedFields) -> u64 {
	// imm20_raw 是 bits[31:12] 零扩展为 u64 的值。
	// 左移 12 位后按 bit 31 符号扩展。
	let imm = (f.imm20_raw << 12) as i32 as i64 as u64;
	write_gpr(state, f.rd, imm);
	4
}

// ============================================================
//  AUIPC: rd = pc + (imm20 << 12), 结果按 32 位符号扩展
// ============================================================

pub fn handle_auipc(state: &mut HartState, f: &DecodedFields) -> u64 {
	let imm = (f.imm20_raw << 12) as i32 as i64 as u64;
	write_gpr(state, f.rd, state.pc.wrapping_add(imm));
	4
}

// ============================================================
//  JAL: rd = pc + 4, 随后 pc = pc + imm_j
// ============================================================

pub fn handle_jal(state: &mut HartState, f: &DecodedFields) -> u64 {
	let next_pc = state.pc.wrapping_add(4);
	write_gpr(state, f.rd, next_pc);
	state.pc = state.pc.wrapping_add(f.imm_j);
	0
}

// ============================================================
//  JALR: rd = pc + 4, 随后 pc = (rs1 + imm) & ~1
// ============================================================

pub fn handle_jalr(state: &mut HartState, f: &DecodedFields) -> u64 {
	let next_pc = state.pc.wrapping_add(4);
	let target = read_gpr(state, f.rs1).wrapping_add(f.imm12_se) & !1u64;
	write_gpr(state, f.rd, next_pc);
	state.pc = target;
	0
}

// ============================================================
//  分支指令: BEQ、BNE、BLT、BGE、BLTU、BGEU
// ============================================================

pub fn handle_br(state: &mut HartState, f: &DecodedFields, instr: u32) -> u64 {
	let v1 = read_gpr(state, f.rs1);
	let v2 = read_gpr(state, f.rs2);

	let taken = match f.func3 {
		0b000 => v1 == v2,                   // BEQ
		0b001 => v1 != v2,                   // BNE
		0b100 => (v1 as i64) < (v2 as i64),  // BLT
		0b101 => (v1 as i64) >= (v2 as i64), // BGE
		0b110 => v1 < v2,                    // BLTU
		0b111 => v1 >= v2,                   // BGEU
		_ => {
			deliver_illegal_instruction(state, instr as u64);
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
//  FENCE 与 FENCE.I: 指令执行顺序与访存屏障
// ============================================================
//
// 并发模型中每个 hart 对应一个线程, FENCE 必须发出真实的 CPU 内存屏障。
// OpenSBI 在 hart 之间以无锁方式对 FIFO 入队与出队时依赖 ``fence ow,ow`` 与
// ``fence ir,ir``。缺少真实屏障时编译器与 CPU 可能重排跨屏障的存储与读取, 造成
// FIFO 内容损坏与 TLB shootdown 死锁, 发送方在 opensbi 中的 tlb_update 中反复重试。

pub fn handle_fence(_state: &mut HartState, f: &DecodedFields, instr: u32) -> u64 {
	// func3 == 0b000: FENCE, 由下方分支发出数据内存屏障。
	// func3 == 0b001: FENCE.I, 不执行任何操作; 模拟器没有独立的指令缓存,
	//   取指与数据读取访问同一份 RAM, 无需刷新。
	if f.func3 > 1 {
		deliver_illegal_instruction(_state, instr as u64);
		return 0;
	}
	if f.func3 == 0 {
		// FENCE: 数据内存屏障。fence(AcqRel) 禁止屏障之前的访存与之后的访存相互重排。
		std::sync::atomic::fence(std::sync::atomic::Ordering::AcqRel);
	}
	4
}

// ============================================================
//  测试
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

	// -- LUI --

	#[test]
	fn lui_small() {
		let mut s = new_state();
		// lui x5, 0x42
		let instr: u32 = (0x42u32 << 12) | (5u32 << 7) | 0x37;
		let f = decode_fields(instr);
		let adv = handle_lui(&mut s, &f);
		assert_eq!(adv, 4);
		assert_eq!(s.gprs[5], 0x0004_2000);
		assert_eq!(s.gprs[0], 0); // x0 未被写入
	}

	#[test]
	fn lui_negative() {
		let mut s = new_state();
		// lui x10, 0xFFFFF, 结果应为 0xFFFF_FFFF_FFFF_F000
		let instr: u32 = (0xfffffu32 << 12) | (10u32 << 7) | 0x37;
		let f = decode_fields(instr);
		let adv = handle_lui(&mut s, &f);
		assert_eq!(adv, 4);
		assert_eq!(s.gprs[10], 0xffff_ffff_ffff_f000);
	}

	#[test]
	fn lui_x0_discards() {
		let mut s = new_state();
		let instr: u32 = (0x123u32 << 12) | (0u32 << 7) | 0x37; // rd=x0
		let f = decode_fields(instr);
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
		let adv = handle_auipc(&mut s, &f);
		assert_eq!(adv, 4);
		assert_eq!(s.gprs[5], 0x8000_1000 + 0x0001_0000);
	}

	// -- JAL --

	#[test]
	fn jal_forward() {
		let mut s = new_state();
		s.pc = 0x8000_0000;
		// jal x1, +8
		// imm_j 的位布局交给 decode_fields 计算
		let instr: u32 = 0x0080_00ef; // jal ra, +8
		let f = decode_fields(instr);
		let adv = handle_jal(&mut s, &f);
		assert_eq!(adv, 0);
		assert_eq!(s.gprs[1], 0x8000_0004);
		assert_eq!(s.pc, 0x8000_0008);
	}

	#[test]
	fn jal_backward() {
		let mut s = new_state();
		s.pc = 0x8000_0100;
		// jal x0, -4; x0 不写入, PC 回到低地址
		let instr: u32 = 0xffdf_f06f; // jal x0, -4, imm_j 为 -4 按 21 位有符号数编码
		let f = decode_fields(instr);
		let adv = handle_jal(&mut s, &f);
		assert_eq!(adv, 0);
		assert_eq!(s.gprs[0], 0); // x0 恒为 0
		assert_eq!(s.pc, (0x8000_0100u64).wrapping_add(0xffff_ffff_ffff_fffc));
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
		let adv = handle_jalr(&mut s, &f);
		assert_eq!(adv, 0);
		assert_eq!(s.gprs[1], 0x8000_0004);
		assert_eq!(s.pc, 0x8000_1000);
	}

	#[test]
	fn jalr_lsb_clear() {
		let mut s = new_state();
		s.pc = 0x8000_0000;
		s.gprs[10] = 0x8000_1003; // 奇数地址
		let instr: u32 = (10u32 << 15) | 0x67;
		let f = decode_fields(instr);
		let _adv = handle_jalr(&mut s, &f);
		assert_eq!(s.pc, 0x8000_1002); // 最低位被清零
	}

	// -- 分支指令 --

	#[test]
	fn beq_taken() {
		let mut s = new_state();
		s.gprs[1] = 42;
		s.gprs[2] = 42;
		// beq x1, x2, +16
		let _instr: u32 =
			(0u32 << 31) | (2u32 << 20) | (1u32 << 15) | (0u32 << 12) | (0b000_1000u32 << 7) | 0x63; // imm_b=16, bits[8:7]=01
																							// 同时改用预先算好的编码
		let _pre: u32 = ((1u32 << 12) << 19) | // bit12=1, 偏移量的 bit 12 置位
			(1u32 << 8) | // 偏移量的 bit 4
			(0u32 << 7) | // 偏移量的 bit 11
			(2u32 << 20) | // rs2 = x2
			(1u32 << 15) | // rs1 = x1
			(0u32 << 12) | // func3 = 000
			0x63; // opcode = BR
		let instr: u32 = 0x0020_8863; // beq x1, x2, +16
		let f = decode_fields(instr);
		let adv = handle_br(&mut s, &f, instr);
		assert_eq!(adv, 0, "branch should be taken");
		assert_eq!(s.pc, (0x8000_0000u64).wrapping_add(f.imm_b));
	}

	#[test]
	fn beq_not_taken() {
		let mut s = new_state();
		s.gprs[1] = 42;
		s.gprs[2] = 99;
		let instr: u32 = 0x0020_8863; // beq x1, x2, +16
		let f = decode_fields(instr);
		let adv = handle_br(&mut s, &f, instr);
		assert_eq!(adv, 4, "branch should NOT be taken");
		assert_eq!(s.pc, 0x8000_0000); // pc 未改变
	}

	#[test]
	fn blt_signed() {
		let mut s = new_state();
		// -1 < 1 成立
		s.gprs[1] = 0xffff_ffff_ffff_ffffu64; // -1
		s.gprs[2] = 1;
		// blt 的 func3 为 0b100, 下面的编码按位构造
		let instr: u32 = ((1u32 << 12) << 19) | // bit12=1
			(1u32 << 8) |
			(2u32 << 20) |
			(1u32 << 15) |
			(4u32 << 12) | // func3 = 0b100 = BLT
			0x63;
		let f = decode_fields(instr);
		let adv = handle_br(&mut s, &f, instr);
		assert_eq!(adv, 0, "blt -1 < 1 should be taken");
	}

	#[test]
	fn bgeu_unsigned() {
		let mut s = new_state();
		// 无符号比较 5 >= 3
		s.gprs[1] = 5;
		s.gprs[2] = 3;
		let instr: u32 = ((1u32 << 12) << 19) |
			(1u32 << 8) |
			(2u32 << 20) |
			(1u32 << 15) |
			(7u32 << 12) | // func3 = 0b111 = BGEU
			0x63;
		let f = decode_fields(instr);
		let adv = handle_br(&mut s, &f, instr);
		assert_eq!(adv, 0, "bgeu 5 >= 3 should be taken");
	}

	// -- ALU: 在 Rust 侧直接调用处理函数的集成测试 --

	#[test]
	fn addi_via_handler() {
		let mut s = new_state();
		s.gprs[10] = 100;
		// addi x5, x10, 42
		let instr: u32 =
			((42u32 & 0xfff) << 20) | (10u32 << 15) | (0u32 << 12) | (5u32 << 7) | 0x13;
		let f = decode_fields(instr);
		let adv = handle_op_imm(&mut s, &f, instr);
		assert_eq!(adv, 4);
		assert_eq!(s.gprs[5], 142);
	}

	#[test]
	fn sub_via_handler() {
		let mut s = new_state();
		s.gprs[10] = 100;
		s.gprs[11] = 30;
		// sub x5, x10, x11
		let instr: u32 =
			(0x20u32 << 25) | (11u32 << 20) | (10u32 << 15) | (0u32 << 12) | (5u32 << 7) | 0x33;
		let f = decode_fields(instr);
		let adv = handle_alu(&mut s, &f, instr);
		assert_eq!(adv, 4);
		assert_eq!(s.gprs[5], 70);
	}

	#[test]
	#[allow(dead_code)]
	fn x0_source_is_zero() {
		let mut s = new_state();
		// addi x5, x0, 42; x0 恒为 0
		let instr: u32 = ((42u32 & 0xfff) << 20) | (0u32 << 15) | (0u32 << 12) | (5u32 << 7) | 0x13;
		let f = decode_fields(instr);
		let _adv = handle_op_imm(&mut s, &f, instr);
		assert_eq!(s.gprs[5], 42);
	}

	#[test]
	fn illegal_alu_encoding_traps() {
		let mut s = new_state();
		s.gprs[10] = 1;
		s.gprs[11] = 2;
		// func3=001, func7=0b001_0011, 对 SLL 与 MULH 都是非法组合
		let instr: u32 =
			(0x13u32 << 25) | (11u32 << 20) | (10u32 << 15) | (1u32 << 12) | (5u32 << 7) | 0x33;
		let f = decode_fields(instr);
		let adv = handle_alu(&mut s, &f, instr);
		assert_eq!(adv, 0, "trap -> advance should be 0");
		assert_eq!(
			s.mcause,
			crate::trap::mcause_val(crate::trap::exc_code::ILL_INSTR, false)
		);
	}

	// ============================================================
	//  rd == rs1 的回归测试
	// ============================================================

	/// rd 与 rs1 相同时, ADDI 必须以旧值作为操作数, 然后写入结果。
	#[test]
	fn addi_rd_equals_rs1_uses_old_value() {
		let mut s = new_state();
		s.gprs[5] = 10;
		// addi x5, x5, 42, 结果 x5 = 10 + 42 = 52
		let instr: u32 = (42u32 << 20) | (5u32 << 15) | (0u32 << 12) | (5u32 << 7) | 0x13;
		let f = decode_fields(instr);
		let adv = handle_op_imm(&mut s, &f, instr);
		assert_eq!(adv, 4);
		assert_eq!(
			s.gprs[5], 52,
			"ADDI rd==rs1: must use old value 10, got {}",
			s.gprs[5]
		);
	}

	/// rd 与 rs1 相同时的 SLLI。
	#[test]
	fn slli_rd_equals_rs1_uses_old_value() {
		let mut s = new_state();
		s.gprs[10] = 3;
		// slli x10, x10, 4, 结果 x10 = 3 << 4 = 48
		let instr: u32 =
			(0u32 << 25) | (4u32 << 20) | (10u32 << 15) | (1u32 << 12) | (10u32 << 7) | 0x13;
		let f = decode_fields(instr);
		let adv = handle_op_imm(&mut s, &f, instr);
		assert_eq!(adv, 4);
		assert_eq!(
			s.gprs[10], 48,
			"SLLI rd==rs1: 3 << 4 = 48, got {}",
			s.gprs[10]
		);
	}

	/// rd 与 rs1 相同而 rs2 不同的 ADD。
	#[test]
	fn add_rd_equals_rs1_uses_old_value() {
		let mut s = new_state();
		s.gprs[5] = 20;
		s.gprs[6] = 7;
		// add x5, x5, x6, 结果 x5 = 20 + 7 = 27
		let instr: u32 =
			(0u32 << 25) | (6u32 << 20) | (5u32 << 15) | (0u32 << 12) | (5u32 << 7) | 0x33;
		let f = decode_fields(instr);
		let adv = handle_alu(&mut s, &f, instr);
		assert_eq!(adv, 4);
		assert_eq!(s.gprs[5], 27, "ADD rd==rs1: 20 + 7 = 27, got {}", s.gprs[5]);
	}

	/// rd 与 rs1 相同时, JALR 必须先保存返回地址再跳转。
	#[test]
	fn jalr_rd_equals_rs1_preserves_link_and_jumps() {
		let mut s = new_state();
		s.pc = 0x8000_1000;
		s.gprs[1] = 0x8000_2000; // ra 设为目标地址
						   // jalr ra, ra, 0, rd=1, rs1=1
		let instr: u32 = (0u32 << 20) | (1u32 << 15) | (0u32 << 12) | (1u32 << 7) | 0x67;
		let f = decode_fields(instr);
		let adv = handle_jalr(&mut s, &f);
		assert_eq!(adv, 0, "JALR returns 0 (PC modified directly)");
		// PC 跳到 ra 的旧值, 最低位被清零
		assert_eq!(s.pc, 0x8000_2000, "JALR must jump to old ra");
		// ra 写入返回地址, 即旧 PC + 4
		assert_eq!(
			s.gprs[1], 0x8000_1004,
			"JALR rd==rs1: ra must hold return address, got {:#x}",
			s.gprs[1]
		);
	}

	// -- ld-linux 循环退出回归测试, 崩溃点在偏移 0x3ae8 --

	/// 地址 0x3ab6 处的 bltz: s2=-1 时必须跳转以跳过循环。
	/// 若不跳转, 循环体将以 s2=-1 执行, 访问的地址为
	///   base + (-1)*0xa0 = base - 0xa0
	/// 该地址位于 .dynamic 段内, 读取到的 d_tag 为 7。
	#[test]
	fn ldlinux_bltz_skip_loop_on_counter_zero() {
		let mut s = new_state();
		s.pc = 0x3ab6;
		s.gprs[18] = u64::MAX; // s2 = -1 = 0xFFFF_FFFF_FFFF_FFFF
						 // bltz s2, 0x3b0a
		let instr: u32 = 0x04094a63;
		let f = decode_fields(instr);
		let adv = handle_br(&mut s, &f, instr);
		assert_eq!(adv, 0, "bltz with s2=-1 must take branch (skip loop)");
		assert_eq!(
			s.pc,
			(0x3ab6u64).wrapping_add(84),
			"bltz target must be 0x3b0a"
		);
	}

	/// 地址 0x3b06 处的 bne: s2=s4=-1 时不得跳转, 从而退出循环。
	/// 若跳转, 循环将继续越过数组边界。
	#[test]
	fn ldlinux_bne_exit_on_s2_eq_s4() {
		let mut s = new_state();
		s.pc = 0x3b06;
		s.gprs[18] = u64::MAX; // s2 = -1
		s.gprs[20] = u64::MAX; // s4 = -1
						 // bne s2, s4, 0x3ad0
		let instr: u32 = 0xfd4915e3;
		let f = decode_fields(instr);
		let adv = handle_br(&mut s, &f, instr);
		assert_eq!(adv, 4, "bne with s2=s4=-1 must NOT take branch (exit loop)");
	}
}
