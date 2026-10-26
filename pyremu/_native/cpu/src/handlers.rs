//! 加载与存储、系统、原子操作及压缩指令的处理函数。
//!
//! 每个处理函数接收 ``&mut HartState``、译码结果与访存上下文,
//! 返回 PC 推进量, 取值为 0、2 或 4, 或者返回 ``EXIT_SENTINEL``,
//! 表示必须返回 Python 侧取得对应的处理。
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Mutex;

use crate::csr;
use crate::decode::{decode_compressed, CompressedFields, DecodedFields};
use crate::ffi::InstrToBeExec;
use crate::state::{exit_reason, riscv_mode, HartState, CFG_HAS_SEMIHOSTING};
pub(crate) use crate::translate::{
	lr_check, lr_clear_all, lr_set, ram_read, ram_write, rwx_read, rwx_write, translate_va,
	TranslateFault, TranslateResult, WalkCtx,
};
use crate::trap::{deliver_illegal_instruction, deliver_trap, exc_code, mcause_val};

pub use crate::interrupt::clint::ClintCtx;
pub(crate) use crate::interrupt::clint::{clint_write_msip, try_handle_clint};
use crate::peripheral::plic::try_handle_plic_concurrent;
pub use crate::peripheral::{is_device_addr, virtio::try_handle_virtio, DevCtx};
pub use crate::pmp::{pmp_ok, PmpCtx};

pub use crate::csr::EXIT_SENTINEL;
// ============================================================
//  SBI 扩展号与功能号, 依据 RISC-V SBI 规范 2.0
// ============================================================

#[allow(dead_code)]
mod sbi_eid {
	pub const TIME: u64 = 0x54494d45; // "TIME", 定时器扩展
	pub const IPI: u64 = 0x735049; // "IPI", 核间中断
	pub const RFNC: u64 = 0x52464e43; // "RFNC", 远程 fence, 用于跨核刷新 TLB
	pub const HSM: u64 = 0x48534d; // "HSM", hart 状态管理
}

mod sbi_fid_time {
	pub const SET_TIMER: u64 = 0;
}

mod sbi_fid_ipi {
	pub const SEND_IPI: u64 = 0;
}

// ============================================================
//  通用寄存器的读取与写入辅助函数
// ============================================================

#[inline]
fn read_gpr(state: &HartState, rs: u8) -> u64 {
	if rs == 0 {
		0
	} else {
		state.gprs[rs as usize]
	}
}

#[inline]
fn write_gpr(state: &mut HartState, rd: u8, val: u64) {
	if rd != 0 {
		state.gprs[rd as usize] = val;
	}
}

/// 把 32 位值按符号位扩展为 64 位。
#[inline]
fn sext32(val: u64) -> u64 {
	val as i32 as i64 as u64
}

/// 从指定位宽按符号位扩展为 64 位无符号整数。
#[inline]
fn sext(val: u64, bits: u32) -> u64 {
	let half = 1u64 << (bits - 1);
	(val & (half - 1)).wrapping_sub(val & half) & u64::MAX
}

/// 压缩指令中 sf 为 0b11 的寄存器-寄存器 ALU 操作:
/// SUB、XOR、OR、AND、SUBW 与 ADDW。非法编码返回 ``None``, 由调用方产生 IllInstr 陷态。
#[inline]
fn c1_alu_reg_op(v1: u64, v2: u64, bit12: u8, bit65: u8) -> Option<u64> {
	if bit12 == 0 {
		Some(match bit65 {
			0b00 => v1.wrapping_sub(v2),
			0b01 => v1 ^ v2,
			0b10 => v1 | v2,
			_ => v1 & v2,
		})
	} else {
		match bit65 {
			0b00 => Some(sext32(v1.wrapping_sub(v2) & 0xffff_ffff)),
			0b01 => Some(sext32(v1.wrapping_add(v2) & 0xffff_ffff)),
			_ => None,
		}
	}
}

/// 按 ``sf`` 字段执行一条压缩 ALU 操作。
/// 非法编码返回 ``None``, 由调用方产生 IllInstr 陷态。
#[inline]
fn exec_c1_alu(state: &HartState, cf: &CompressedFields) -> Option<u64> {
	let v1 = read_gpr(state, cf.rs1p);
	match cf.sf {
		0b00 => {
			let shamt = ((cf.bit12 as u64) << 5) | ((cf.rs2 as u64) & 0x1f);
			Some(v1 >> shamt)
		}
		0b01 => {
			let shamt = ((cf.bit12 as u64) << 5) | ((cf.rs2 as u64) & 0x1f);
			Some(((v1 as i64) >> shamt) as u64)
		}
		0b10 => {
			let imm = sext(((cf.bit12 as u64) << 5) | ((cf.rs2 as u64) & 0x1f), 6);
			Some(v1 & imm)
		}
		0b11 => {
			let v2 = read_gpr(state, cf.rdp);
			c1_alu_reg_op(v1, v2, cf.bit12, cf.bit65)
		}
		_ => None,
	}
}

// ============================================================
//  CLINT 的 MMIO 访问处理
// ============================================================

/// 可能由 CLINT 直接处理的一次访存描述。
///
/// 所有字段均为 ``u64``, 避免因内存对齐问题造成运行期间的程序异常。
/// ``is_write`` 为 0 表示读取, 为 1 表示写入。
#[repr(C)]
#[allow(dead_code)]
pub(crate) struct MemAccess {
	pub(crate) pa: u64,
	pub(crate) is_write: u64,
	pub(crate) write_data: u64,
	pub(crate) size: u64,
}

impl MemAccess {
	pub(crate) fn read(pa: u64, size: u8) -> Self {
		Self {
			pa,
			is_write: 0,
			write_data: 0,
			size: size as u64,
		}
	}
	pub(crate) fn write(pa: u64, size: u8, data: u64) -> Self {
		Self {
			pa,
			is_write: 1,
			write_data: data,
			size: size as u64,
		}
	}
}

/// 串行路径上对 IMSIC 地址的处理。
///
/// 地址译码与上下文提取在此完成, 路由、eip 改写与唤醒线程由
/// ``imsic_handle_seteipnum`` 与 ``imsic_handle_clreipnum`` 处理.
/// pa 位于 IMSIC 的 seteipnum 或 clreipnum 寄存器内时返回 true.
fn try_handle_imsic_serial(pa: u64, val: u64, clint: &ClintCtx) -> bool {
	if !crate::state::CFG_AIA {
		return false;
	}
	let addr = match crate::interrupt::decode_imsic_addr(pa, clint.num_harts as u64) {
		Some(a) => a,
		None => {
			return false;
		}
	};
	if addr.reg_off != 0x0000 && addr.reg_off != 0x0008 {
		return false;
	}
	let file_tag: u64 = if addr.is_sfile { 0x1000 } else { 0x0000 };
	let eip_num = val as u32;

	// 读取路径: 按 AIA 规范, IMSIC 的全部 MMIO 读取均返回 0,
	// seteipnum 与 clreipnum 只可写。读取路径的调用点传入 val==0,
	// 以区分读取与写入。
	if eip_num == 0 {
		return true;
	}

	// 写入路径: 全部中断标识都交给并发路径使用的
	// seteipnum 与 clreipnum 辅助函数处理。
	match addr.reg_off {
		0x0000 => {
			let handled = crate::interrupt::imsic::imsic_handle_seteipnum(
				clint.hart_states.get(),
				clint.hart_threads.get(),
				clint.num_harts,
				addr.hart,
				file_tag,
				eip_num,
			);
			handled
		}
		0x0008 => {
			crate::interrupt::imsic::imsic_handle_clreipnum(
				clint.hart_states.get(),
				addr.hart,
				file_tag,
				eip_num,
			);
			true
		}
		_ => unreachable!(),
	}
}

// ============================================================
//  加载处理
// ============================================================
pub fn handle_load(
	state: &mut HartState,
	f: &DecodedFields,
	instr: u32,
	instr_group: &mut InstrToBeExec,
	ctx: &WalkCtx,
	pmp: &PmpCtx,
	dev: &DevCtx,
	clint: &ClintCtx,
) -> u64 {
	let base = read_gpr(state, f.rs1);
	let va = base.wrapping_add(f.imm12_se);
	let (size, signed) = match f.func3 {
		0b000 => (1, true),  // LB
		0b001 => (2, true),  // LH
		0b010 => (4, true),  // LW
		0b011 => (8, false), // LD
		0b100 => (1, false), // LBU
		0b101 => (2, false), // LHU
		0b110 => (4, false), // LWU
		_ => {
			deliver_illegal_instruction(state, instr as u64);
			return 0;
		}
	};

	// 对齐检查: 不支持非对齐的加载
	if (va & ((size as u64) - 1)) != 0 {
		deliver_trap(state, mcause_val(exc_code::LD_MISALIGNED, false), va);
		return 0;
	}

	// 将虚拟地址翻译为物理地址
	let tr = match translate_va(state, ctx, va, rwx_read(), pmp) {
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

	// PMP 检查
	if !pmp_ok(state, tr.pa, size as u32, false, false, pmp) {
		deliver_trap(state, mcause_val(exc_code::LD_ACCESS_FAULT, false), va);
		return 0;
	}

	// 直接读取 CLINT 中的 MSIP、MTIMECMP 与 MTIME,
	// 避免 SMP 启动期间退出加速执行的开销。
	if let Some(data) = try_handle_clint(&MemAccess::read(tr.pa, size), state, clint) {
		let result_val = if signed {
			match size {
				1 => sext(data, 8),
				2 => sext(data, 16),
				4 => sext(data, 32),
				_ => data,
			}
		} else {
			data
		};
		write_gpr(state, f.rd, result_val);
		return 4;
	}

	// IMSIC 的全部 MMIO 读取均返回 0, seteipnum 与 clreipnum
	// 只可写。压缩指令从 IMSIC 读取时不退出加速执行。
	if try_handle_imsic_serial(tr.pa, 0, clint) {
		write_gpr(state, f.rd, 0);
		return 4;
	}

	// virtio-blk 避免设备探测期间退出加速执行的开销。
	if let Some(data) = try_handle_virtio(tr.pa, false, 0, size, dev) {
		let result_val = if signed {
			match size {
				1 => sext(data, 8),
				2 => sext(data, 16),
				4 => sext(data, 32),
				_ => data,
			}
		} else {
			data
		};
		write_gpr(state, f.rd, result_val);
		return 4;
	}

	if let Some(data) = try_handle_plic_concurrent(tr.pa, false, 0, size, state, dev.plic) {
		let result_val = if signed {
			match size {
				1 => sext(data, 8),
				2 => sext(data, 16),
				4 => sext(data, 32),
				_ => data,
			}
		} else {
			data
		};
		write_gpr(state, f.rd, result_val);
		return 4;
	}

	// 未被上述路径处理的设备地址，退出加速执行。
	if is_device_addr(tr.pa, dev) {
		instr_group.exit_reason = exit_reason::MMIO;
		instr_group.exit_instr = instr;
		return EXIT_SENTINEL;
	}

	// 从 RAM 读取
	let val = ram_read(ctx, tr.pa, size);

	let result_val = if signed {
		match size {
			1 => sext(val, 8),
			2 => sext(val, 16),
			4 => sext(val, 32),
			_ => val,
		}
	} else {
		// 无符号加载零扩展; ram_read 已返回零扩展后的值。
		val
	};

	write_gpr(state, f.rd, result_val);
	4
}

// ============================================================
//  存储处理
// ============================================================

pub fn handle_store(
	state: &mut HartState,
	f: &DecodedFields,
	instr: u32,
	instr_group: &mut InstrToBeExec,
	ctx: &WalkCtx,
	pmp: &PmpCtx,
	dev: &DevCtx,
	clint: &ClintCtx,
) -> u64 {
	let base = read_gpr(state, f.rs1);
	let va = base.wrapping_add(f.imm_s);
	let size: u8 = match f.func3 {
		0b000 => 1, // SB
		0b001 => 2, // SH
		0b010 => 4, // SW
		0b011 => 8, // SD
		_ => {
			deliver_illegal_instruction(state, instr as u64);
			return 0;
		}
	};

	// 对齐检查
	if (va & ((size as u64) - 1)) != 0 {
		deliver_trap(state, mcause_val(exc_code::ST_MISALIGNED, false), va);
		return 0;
	}

	// 将虚拟地址翻译为物理地址
	let tr = match translate_va(state, ctx, va, rwx_write(), pmp) {
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

	// PMP 检查
	if !pmp_ok(state, tr.pa, size as u32, true, false, pmp) {
		deliver_trap(state, mcause_val(exc_code::ST_ACCESS_FAULT, false), va);
		return 0;
	}

	let val = read_gpr(state, f.rs2);

	// 就地完成 MSIP、MTIMECMP 与 MTIME 的写入,
	// 避免 SMP 启动期间退出加速执行的开销。
	if let Some(_) = try_handle_clint(&MemAccess::write(tr.pa, size, val), state, clint) {
		return 4;
	}

	// 中断标识 1 与 3 经 CLINT MSIP 处理, 跨 hart 唤醒不退出链接库。
	if try_handle_imsic_serial(tr.pa, val, clint) {
		return 4;
	}

	// virtio-blk 直接处理: 除 QueueNotify 外的全部 MMIO 寄存器就地完成,
	// 避免设备探测期间退出加速执行的开销。
	if let Some(_) = try_handle_virtio(tr.pa, true, val, size, dev) {
		return 4;
	}

	if let Some(_) = try_handle_plic_concurrent(tr.pa, true, val, size, state, dev.plic) {
		return 4;
	}

	// 未被上述路径处理的设备地址则退出加速执行。
	if is_device_addr(tr.pa, dev) {
		instr_group.exit_reason = exit_reason::MMIO;
		instr_group.exit_instr = instr;
		return EXIT_SENTINEL;
	}

	ram_write(ctx, tr.pa, val, size);

	4
}

// ============================================================
//  F/D 浮点加载与存储
// ============================================================

/// mstatus.FS 在 [14:13] 的掩码。
const MSTATUS_FS_LS: u64 = 0b11 << 13;
/// mstatus.SD 在第 63 位的掩码。
const MSTATUS_SD_LS: u64 = 1 << 63;
/// 单精度浮点值写入 64 位寄存器时高 32 位的填充掩码。
const NANBOX_S_LS: u64 = 0xffff_ffff_0000_0000;

/// FLW 与 FLD (opcode 0x07): 从内存加载到 FPR。
/// FLW 的结果写入高 32 位全 1 的 64 位寄存器, FLD 加载完整 64 位。
pub fn handle_fp_load(
	state: &mut HartState,
	f: &DecodedFields,
	instr: u32,
	instr_group: &mut InstrToBeExec,
	ctx: &WalkCtx,
	pmp: &PmpCtx,
	dev: &DevCtx,
	clint: &ClintCtx,
) -> u64 {
	if (state.mstatus & MSTATUS_FS_LS) == 0 {
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

	if (va & ((size as u64) - 1)) != 0 {
		deliver_trap(state, mcause_val(exc_code::LD_MISALIGNED, false), va);
		return 0;
	}

	let tr = match translate_va(state, ctx, va, rwx_read(), pmp) {
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

	// 在通用 MMIO 退出之前做 virtio-blk 直接处理。
	if let Some(data) = try_handle_virtio(tr.pa, false, 0, size, dev) {
		let boxed = if size == 4 { NANBOX_S_LS | data } else { data };
		state.fprs[f.rd as usize] = boxed;
		state.mstatus |= MSTATUS_FS_LS | MSTATUS_SD_LS;
		return 4;
	}

	// 浮点数据极少访问 CLINT; 设备地址交回 Python 按 MMIO 处理。
	if is_device_addr(tr.pa, dev) {
		instr_group.exit_reason = exit_reason::MMIO;
		instr_group.exit_instr = instr;
		return EXIT_SENTINEL;
	}
	let _ = clint;

	let val = ram_read(ctx, tr.pa, size);
	let boxed = if size == 4 { NANBOX_S_LS | val } else { val };
	state.fprs[f.rd as usize] = boxed;
	state.mstatus |= MSTATUS_FS_LS | MSTATUS_SD_LS;
	4
}

/// FSW 与 FSD (opcode 0x27): 将 FPR 的值写入内存。
/// FSW 写入低 32 位, FSD 写入完整 64 位。
pub fn handle_fp_store(
	state: &mut HartState,
	f: &DecodedFields,
	instr: u32,
	instr_group: &mut InstrToBeExec,
	ctx: &WalkCtx,
	pmp: &PmpCtx,
	dev: &DevCtx,
	clint: &ClintCtx,
) -> u64 {
	if (state.mstatus & MSTATUS_FS_LS) == 0 {
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

	if (va & ((size as u64) - 1)) != 0 {
		deliver_trap(state, mcause_val(exc_code::ST_MISALIGNED, false), va);
		return 0;
	}

	let tr = match translate_va(state, ctx, va, rwx_write(), pmp) {
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

	// FSW 只写入低 32 位, 直接取寄存器的低位即可。
	let val = state.fprs[f.rs2 as usize];

	// 在通用 MMIO 退出之前做 virtio-blk 直接处理。
	if let Some(_) = try_handle_virtio(tr.pa, true, val, size, dev) {
		return 4;
	}

	if is_device_addr(tr.pa, dev) {
		instr_group.exit_reason = exit_reason::MMIO;
		instr_group.exit_instr = instr;
		return EXIT_SENTINEL;
	}
	let _ = clint;
	ram_write(ctx, tr.pa, val, size);
	4
}

// ============================================================
//  系统指令处理
// ============================================================

#[inline]
fn try_sbi_time_set_timer(state: &mut HartState, clint: &ClintCtx) -> Option<u64> {
	if read_gpr(state, 17) != sbi_eid::TIME || read_gpr(state, 16) != sbi_fid_time::SET_TIMER {
		return None;
	}
	let stime_val = read_gpr(state, 10); // a0
	let hid = state.mhartid as usize;
	if hid < (clint.num_harts as usize) {
		unsafe {
			*clint.mtimecmp.add(hid) = stime_val;
		}
	}
	state.stimecmp = stime_val;
	state.gprs[10] = 0; // SBI_SUCCESS
	Some(4)
}

#[inline]
fn try_sbi_ipi_send_ipi(
	state: &mut HartState,
	instr: u32,
	result: &mut InstrToBeExec,
	clint: &ClintCtx,
) -> Option<u64> {
	if read_gpr(state, 17) != sbi_eid::IPI || read_gpr(state, 16) != sbi_fid_ipi::SEND_IPI {
		return None;
	}
	let hart_mask = read_gpr(state, 10); // a0
	let mask_base = read_gpr(state, 11); // a1
	if mask_base != 0 {
		// mask_base 不为 0 时交回 Python 处理。
		result.exit_reason = exit_reason::ECALL;
		result.exit_instr = instr;
		return Some(EXIT_SENTINEL);
	}
	// mask_base 为 0 是常见情形。
	let cur = state.mhartid as usize;
	for t in 0..clint.num_harts as u64 {
		if (hart_mask & (1u64 << t)) == 0 {
			continue;
		}
		clint_write_msip(clint, t as usize, cur, 1);
	}
	state.gprs[10] = 0; // SBI_SUCCESS
	Some(4)
}

/// 处理 ECALL。
///
/// SBI 调用约定: a7 即 x17 存放 EID, a6 即 x16 存放 FID。SBI 是 S 模式到 M 模式的接口,
/// 快速路径仅在 S 模式下启用; 非 S 模式发起的 ecall 按特权级分发陷态,
/// 判据与 ``trap::priv_ecall_concurrent`` 相同。其余调用不退出加速执行, 直接分发陷态。
#[inline]
fn handle_ecall_inline(
	state: &mut HartState,
	instr: u32,
	instr_group: &mut InstrToBeExec,
	clint: &ClintCtx,
) -> u64 {
	if state.mode != riscv_mode::S {
		let ecall_cause = match state.mode {
			riscv_mode::U => mcause_val(exc_code::ECALL_UMODE, false),
			_ => mcause_val(exc_code::ECALL_MMODE, false),
		};
		deliver_trap(state, ecall_cause, 0);
		return 0;
	}

	// ---- SBI 快速路径 ----
	if let Some(advance) = try_sbi_time_set_timer(state, clint) {
		return advance;
	}
	if let Some(advance) = try_sbi_ipi_send_ipi(state, instr, instr_group, clint) {
		return advance;
	}

	// ---- 一般 ECALL: 直接分发陷态 ----
	deliver_trap(state, mcause_val(exc_code::ECALL_SMODE, false), 0);
	0
}

/// 分发特权指令: ECALL、EBREAK、MRET、SRET、WFI 与 SFENCE.VMA。
/// ``handle_system`` 在 func3 为 0b000 时调用本函数。
#[inline]
fn dispatch_privileged(
	state: &mut HartState,
	func12: u16,
	instr: u32,
	instr_group: &mut InstrToBeExec,
	clint: &ClintCtx,
	ctx: &WalkCtx,
) -> u64 {
	match func12 {
		0 => handle_ecall_inline(state, instr, instr_group, clint),
		1 => {
			// 调试器服务关闭: 与 Python 引擎的 trap_ebreak 一致, 分发断点异常。
			if !CFG_HAS_SEMIHOSTING {
				let ebreak_pc = state.pc;
				deliver_trap(state, mcause_val(exc_code::BREAKPOINT, false), ebreak_pc);
				return 0;
			}
			// 调试器序列直接处理, 其余情况按 NOP 处理: 没有连接外部调试器。
			if let Some(advance) = try_semihosting(state, ctx, state.pc) {
				return advance;
			}
			4
		}
		0x302 => {
			// MRET
			let mpp = (state.mstatus >> 11) & 0x3;
			let mpie = (state.mstatus >> 7) & 1;
			state.mode = match mpp {
				0 => riscv_mode::U,
				1 => riscv_mode::S,
				3 => riscv_mode::M,
				_ => riscv_mode::M,
			};
			// MIE 取 MPIE 的值, MPIE 置 1
			state.mstatus &= !(1 << 3); // 清除 MIE
			if mpie != 0 {
				state.mstatus |= 1 << 3;
			} // MIE 取 MPIE 的值
			state.mstatus |= 1 << 7; // MPIE 置 1
			state.mstatus &= !(0b11 << 11); // MPP 置 0, 表示 U 模式
			state.pc = state.mepc;
			state.waiting = 0;
			0
		}
		0x102 => {
			// SRET: 检查特权级
			if state.mode < riscv_mode::S {
				deliver_illegal_instruction(state, instr as u64);
				return 0;
			}
			let spp = (state.mstatus >> 8) & 1;
			let spie = (state.mstatus >> 5) & 1;
			state.mode = if spp == 0 {
				riscv_mode::U
			} else {
				riscv_mode::S
			};
			// SIE 取 SPIE 的值, SPIE 置 1
			state.mstatus &= !(1 << 1); // 清除 SIE
			if spie != 0 {
				state.mstatus |= 1 << 1;
			} // SIE 取 SPIE 的值
			state.mstatus |= 1 << 5; // SPIE 置 1
			state.mstatus &= !(1 << 8); // SPP 置 0, 表示 U 模式
			state.pc = state.sepc;
			state.waiting = 0;
			0
		}
		0x105 => {
			// WFI: 等待中断。没有待处理中断时, 循环顶部的检查以
			// WFI_WAIT 结束本 hart 的执行轮次。
			let tw = (state.mstatus >> 21) & 1;
			if tw != 0 && state.mode != riscv_mode::M {
				deliver_illegal_instruction(state, instr as u64);
				return 0;
			}
			// wfi_woken 非 0 表示刚被中断从 WFI 唤醒。MRET 返回到 WFI 之后
			// 的 while 循环, 此时把 WFI 视为 NOP 并推进 PC, 使该轮询循环
			// 重新判定循环条件。
			if state.wfi_woken != 0 {
				state.wfi_woken = 0;
				return 4; // 按 NOP 处理, PC 推进至下一条指令
			}
			let pending = state.mip.load(Ordering::Acquire) & state.mie;
			if pending != 0 {
				return 4;
			} // 已有待处理中断, 按 NOP 处理
			state.waiting = 1;
			4 // 推进 PC 越过 WFI; 等待在循环顶部处理
		}
		// SFENCE.VMA: funct7 位于位域 [31:25], 取值 0b0001001, rs2 位于位域 [24:20]。
		// 编码满足
		//     funct12 = (funct7 << 5) | rs2 = 0x120 | rs2
		// 故匹配 0x120..0x13F 的全部取值, 使按 ASID 刷新的 sfence.vma x0, rs2
		// 也在此处理, 不产生 IllInstr 陷态。
		f if (f >> 5) == 0x09 => {
			// SFENCE.VMA: 刷新全部 TLB 条目, 即 itlb 与 dtlb。
			// 按 ASID 与 VA 过滤尚未实现, 当前执行全量刷新。
			for e in state.itlb.iter_mut() {
				e.valid = 0;
			}
			for e in state.dtlb.iter_mut() {
				e.valid = 0;
			}
			4
		}
		0x5a0 => {
			for e in state.itlb.iter_mut() {
				if e.mdid == state.mdid {
					e.valid = 0;
				}
			}
			for e in state.dtlb.iter_mut() {
				if e.mdid == state.mdid {
					e.valid = 0;
				}
			}
			4
		}
		_ => {
			deliver_illegal_instruction(state, instr as u64);
			0
		}
	}
}

pub fn handle_system(
	state: &mut HartState,
	f: &DecodedFields,
	instr: u32,
	instr_group: &mut InstrToBeExec,
	ctx: &WalkCtx,
	hart_id: u8,
	clint: &ClintCtx,
	pmp: &PmpCtx,
) -> u64 {
	match f.func3 {
		0b000 => dispatch_privileged(state, f.func12, instr, instr_group, clint, ctx),
		0b001 | 0b010 | 0b011 | 0b101 | 0b110 | 0b111 => {
			// CSR 指令可由 funct3 的取值区分:
			//     0b001: CSRRW
			//     0b010: CSRRS
			//     0b011: CSRRC
			//     0b101: CSRRWI
			//     0b110: CSRRSI
			//     0b111: CSRRCI
			// 为 CsrContext 构造临时的 ConcurrentClintCtx, 使计时器中断标识
			// 能够更新 mtimecmp; u64 与 AtomicU64 的内存布局在 x86_64 上相同。
			let conc_clint = crate::concurrent::ConcurrentClintCtx {
				base: clint.base,
				mtime: clint.mtime as *const std::sync::atomic::AtomicU64,
				mtimecmp: clint.mtimecmp as *const std::sync::atomic::AtomicU64,
				msip: clint.msip as *const std::sync::atomic::AtomicU8,
				num_harts: clint.num_harts,
				msip_pending: std::cell::Cell::new(std::ptr::null()),
				hart_threads: std::cell::Cell::new(std::ptr::null()),
				hart_states: std::cell::Cell::new(std::ptr::null()),
				timebase_hz: 0,
			};
			let mut csr_ctx = csr::CsrContext::new(state, hart_id as u32, pmp, &conc_clint);
			let advance = csr::handle_csr(
				&mut csr_ctx,
				f.rd,
				f.rs1,
				f.func12,
				f.func3,
				instr,
				instr_group,
			);
			// 同步 MIP.MSIP 到 CLINT 的 msip: 软件经 CSR 写入清除 MSIP 后,
			// 同时清除 CLINT 的 MSIP 寄存器, 避免 ``build_mip`` 在下次进入
			// 加速执行时再次置位。
			if f.func12 != 0x344 /* MIP */ || f.func12 /* SIP */ == 0x144 {
				return advance;
			}
			let msip_in_mip = (state.mip.load(Ordering::Acquire) >> 3) & 1;
			let msip_in_clint = ((unsafe { *clint.msip.add(hart_id as usize) }) as u64) & 1;
			if msip_in_mip == 0 && msip_in_clint != 0 {
				unsafe {
					*clint.msip.add(hart_id as usize) = 0;
				}
			} else if msip_in_mip != 0 && msip_in_clint == 0 {
				// CLINT 的 MSIP 为 0 而 state.mip 的 MSIP 仍为 1 时, 以 CLINT
				// 为准清除 state.mip; 此处不回写 CLINT。
				state.mip.fetch_and(!(1 << 3), Ordering::AcqRel);
			}
			advance
		}
		_ => {
			deliver_illegal_instruction(state, instr as u64);
			0
		}
	}
}

// ============================================================
//  AMO 取值操作辅助函数: 按宽度分派 AtomicU32 与 AtomicU64
// ============================================================

/// 对一个 32 位值执行一次 AMO 取值操作。``funct5`` 与 ``rs2_val`` 来自
/// 译码结果; 带符号的最小值与最大值操作把操作数按 ``i32`` 解释。
#[inline]
fn amo_fetch_u32(a: &AtomicU32, funct5: u8, rs2_val: u64) -> u32 {
	let rs2_u32 = rs2_val as u32;
	match funct5 {
		0b00001 => a.swap(rs2_u32, Ordering::AcqRel),
		0b00000 => a.fetch_add(rs2_u32, Ordering::AcqRel),
		0b00100 => a.fetch_xor(rs2_u32, Ordering::AcqRel),
		0b01100 => a.fetch_and(rs2_u32, Ordering::AcqRel),
		0b01000 => a.fetch_or(rs2_u32, Ordering::AcqRel),
		0b10000 => a
			.fetch_update(Ordering::AcqRel, Ordering::Acquire, |prev| {
				let s = (prev as i32).min(rs2_val as i32);
				Some(s as u32)
			})
			.unwrap(),
		0b10100 => a
			.fetch_update(Ordering::AcqRel, Ordering::Acquire, |prev| {
				let s = (prev as i32).max(rs2_val as i32);
				Some(s as u32)
			})
			.unwrap(),
		0b11000 => a.fetch_min(rs2_u32, Ordering::AcqRel),
		0b11100 => a.fetch_max(rs2_u32, Ordering::AcqRel),
		_ => unreachable!(),
	}
}

/// 对一个 64 位值执行一次 AMO 取值操作。
#[inline]
fn amo_fetch_u64(a: &AtomicU64, funct5: u8, rs2_val: u64) -> u64 {
	match funct5 {
		0b00001 => a.swap(rs2_val, Ordering::AcqRel),
		0b00000 => a.fetch_add(rs2_val, Ordering::AcqRel),
		0b00100 => a.fetch_xor(rs2_val, Ordering::AcqRel),
		0b01100 => a.fetch_and(rs2_val, Ordering::AcqRel),
		0b01000 => a.fetch_or(rs2_val, Ordering::AcqRel),
		0b10000 => a
			.fetch_update(Ordering::AcqRel, Ordering::Acquire, |prev| {
				Some((prev as i64).min(rs2_val as i64) as u64)
			})
			.unwrap(),
		0b10100 => a
			.fetch_update(Ordering::AcqRel, Ordering::Acquire, |prev| {
				Some((prev as i64).max(rs2_val as i64) as u64)
			})
			.unwrap(),
		0b11000 => a.fetch_min(rs2_val, Ordering::AcqRel),
		0b11100 => a.fetch_max(rs2_val, Ordering::AcqRel),
		_ => unreachable!(),
	}
}

/// AMO 算术操作: AMOSWAP、AMOADD、AMOXOR、AMOAND、AMOOR、AMOMIN、
/// AMOMAX、AMOMINU 与 AMOMAXU。
/// 对 ``tr.pa`` 处的字节数组执行原子取值操作, 清除 LR/SC 保留记录,
/// 并把旧值写入 ``rd``。
#[inline]
fn handle_amo_arithmetic(
	state: &mut HartState,
	funct5: u8,
	rd: u8,
	rs2_val: u64,
	width: u8,
	va: u64,
	tr: &TranslateResult,
	ctx: &WalkCtx,
) -> u64 {
	let off = super::mem::ram_offset_inline(
		tr.pa,
		width as u32,
		ctx.ram_base,
		ctx.ram_size,
		ctx.shadow_base,
		ctx.shadow_size,
	);
	if off.is_none() {
		deliver_trap(state, mcause_val(exc_code::LD_ACCESS_FAULT, false), va);
		return 0;
	}
	let ptr = unsafe { ctx.ram.add(off.unwrap() as usize) };
	let loaded: u64 = match width {
		4 => {
			let a = unsafe { &*(ptr as *const AtomicU32) };
			amo_fetch_u32(a, funct5, rs2_val) as u64
		}
		8 => {
			let a = unsafe { &*(ptr as *const AtomicU64) };
			amo_fetch_u64(a, funct5, rs2_val)
		}
		_ => unreachable!(),
	};
	lr_clear_all(ctx);
	write_gpr(state, rd, if width == 4 { sext32(loaded) } else { loaded });
	4
}

// ============================================================
//  AMO 处理
// ============================================================

pub fn handle_amo(
	state: &mut HartState,
	f: &DecodedFields,
	instr: u32,
	instr_group: &mut InstrToBeExec,
	ctx: &WalkCtx,
	pmp: &PmpCtx,
	dev: &DevCtx,
) -> u64 {
	let _aq = (instr >> 26) & 1;
	let _rl = (instr >> 25) & 1;
	let funct5 = f.func7 >> 2; // funct5 为 func7 的高 5 位, 即指令的位域 [31:27]
	let width: u8 = match f.func3 {
		0b010 => 4, // .W
		0b011 => 8, // .D
		_ => {
			deliver_illegal_instruction(state, instr as u64);
			return 0;
		}
	};

	let base = read_gpr(state, f.rs1);
	let va = base; // AMO: rs1 为地址, rs2 为操作数

	// 对齐检查
	if (va & ((width as u64) - 1)) != 0 {
		deliver_trap(state, mcause_val(exc_code::LD_MISALIGNED, false), va);
		return 0;
	}

	// 将虚拟地址翻译为物理地址
	let tr = match translate_va(state, ctx, va, rwx_write(), pmp) {
		Ok(t) => t,
		Err(TranslateFault::PageFault(cause)) => {
			deliver_trap(state, mcause_val(cause, false), va);
			return 0;
		}
		Err(_) => {
			deliver_trap(state, mcause_val(exc_code::LD_ACCESS_FAULT, false), va);
			return 0;
		}
	};

	// PMP 检查
	if !pmp_ok(state, tr.pa, width as u32, true, false, pmp) {
		deliver_trap(state, mcause_val(exc_code::LD_ACCESS_FAULT, false), va);
		return 0;
	}

	let rd = f.rd;
	let rs2_val = read_gpr(state, f.rs2);

	// virtio-blk 直接处理
	if let Some(_) = try_handle_virtio(tr.pa, true, rs2_val, width, dev) {
		return 4;
	}

	// 其余设备地址: 退出加速执行。
	if is_device_addr(tr.pa, dev) {
		instr_group.exit_reason = exit_reason::MMIO;
		instr_group.exit_instr = instr;
		return EXIT_SENTINEL;
	}

	match funct5 {
		0b00010 => {
			// LR.W 与 LR.D
			let loaded = ram_read(ctx, tr.pa, width);
			lr_set(ctx, state.mhartid as u8, tr.pa);
			state.reservation_valid = 1;
			state.reservation_addr = tr.pa;
			write_gpr(state, rd, if width == 4 { sext32(loaded) } else { loaded });
			4
		}
		0b00011 => {
			// SC.W 与 SC.D
			let has_reservation = lr_check(ctx, state.mhartid as u8, tr.pa)
				|| (state.reservation_valid != 0 && state.reservation_addr == tr.pa);
			if !has_reservation {
				write_gpr(state, rd, 1);
			} else {
				ram_write(ctx, tr.pa, rs2_val, width);
				state.reservation_valid = 0;
				write_gpr(state, rd, 0);
			}
			4
		}
		0b00001 | 0b00000 | 0b00100 | 0b01100 | 0b01000 | 0b10000 | 0b10100 | 0b11000 | 0b11100 => {
			handle_amo_arithmetic(state, funct5, rd, rs2_val, width, va, &tr, ctx)
		}
		_ => {
			deliver_illegal_instruction(state, instr as u64);
			0
		}
	}
}

// ============================================================
//  压缩指令的处理函数
// ============================================================

pub fn handle_compressed(
	state: &mut HartState,
	half: u16,
	instr_word: u32,
	instr_group: &mut InstrToBeExec,
	ctx: &WalkCtx,
	pmp: &PmpCtx,
	dev: &DevCtx,
	clint: &ClintCtx,
) -> u64 {
	let cf = decode_compressed(half);

	match cf.quadrant {
		0 => handle_c0(state, &cf, instr_word, instr_group, ctx, pmp, dev, clint),
		1 => handle_c1(state, &cf, instr_word),
		2 => handle_c2(state, &cf, instr_word, instr_group, ctx, pmp, dev, clint),
		_ => {
			deliver_illegal_instruction(state, half as u64);
			0
		}
	}
}

/// 压缩指令的栈指针加法与访存指令:
/// C.ADDI4SPN、C.FLD、C.LW、C.LD、C.FSD、C.SW 与 C.SD。
fn handle_c0(
	state: &mut HartState,
	cf: &CompressedFields,
	instr_word: u32,
	instr_group: &mut InstrToBeExec,
	ctx: &WalkCtx,
	pmp: &PmpCtx,
	dev: &DevCtx,
	clint: &ClintCtx,
) -> u64 {
	match cf.funct3 {
		0b000 => {
			// C.ADDI4SPN: rd' 取 sp 与 nzuimm 之和
			if cf.imm == 0 {
				deliver_illegal_instruction(state, instr_word as u64);
				return 0;
			}
			let sp = state.gprs[2]; // x2 = sp
			write_gpr(state, cf.rd, sp.wrapping_add(cf.imm));
			2
		}
		0b001 => {
			// C.FLD: 从 rs1' 加 uimm 的地址读取双精度值到 fpr[rd'], 属于 RV64DC。
			let addr = read_gpr(state, cf.rs1p).wrapping_add(cf.imm);
			let prev_mode = state.mode;
			let val = match load_mem_compressed(
				state,
				ctx,
				addr,
				8,
				false,
				instr_word,
				instr_group,
				pmp,
				dev,
				clint,
			) {
				Some(v) => v,
				None => {
					return EXIT_SENTINEL;
				}
			};
			if state.mode != prev_mode {
				return 0;
			} // 已分发陷态
			state.fprs[cf.rdp as usize] = val;
			state.mstatus |= MSTATUS_FS_LS | MSTATUS_SD_LS;
			2
		}
		0b010 => {
			// C.LW: 从 rs1' 加 uimm 的地址读取 32 位值到 rd'。
			let addr = read_gpr(state, cf.rs1p).wrapping_add(cf.imm);
			let prev_mode = state.mode;
			let val = match load_mem_compressed(
				state,
				ctx,
				addr,
				4,
				false,
				instr_word,
				instr_group,
				pmp,
				dev,
				clint,
			) {
				Some(v) => v,
				None => {
					return EXIT_SENTINEL;
				}
			};
			if state.mode != prev_mode {
				return 0;
			} // 已分发陷态
			let final_val = sext32(val);
			write_gpr(state, cf.rd, final_val);
			2
		}
		0b011 => {
			// C.LD: 从 rs1' 加 uimm 的地址读取 64 位值到 rd'。
			let addr = read_gpr(state, cf.rs1p).wrapping_add(cf.imm);
			let prev_mode = state.mode;
			let val = match load_mem_compressed(
				state,
				ctx,
				addr,
				8,
				false,
				instr_word,
				instr_group,
				pmp,
				dev,
				clint,
			) {
				Some(v) => v,
				None => {
					return EXIT_SENTINEL;
				}
			};
			if state.mode != prev_mode {
				return 0;
			} // 已分发陷态
			write_gpr(state, cf.rd, val);
			2
		}
		0b101 => {
			// C.FSD: 将 fpr[rd'] 的双精度值写入 rs1' 加 uimm 的地址, 属于 RV64DC。
			let addr = read_gpr(state, cf.rs1p).wrapping_add(cf.imm);
			let fpr_val = state.fprs[cf.rdp as usize];
			let ret = store_mem_compressed(
				state,
				ctx,
				addr,
				fpr_val,
				8,
				instr_word,
				instr_group,
				pmp,
				dev,
				clint,
			);
			if ret == EXIT_SENTINEL {
				return EXIT_SENTINEL;
			}
			state.mstatus |= MSTATUS_FS_LS | MSTATUS_SD_LS;
			2
		}
		0b110 => {
			// C.SW: 将 rs2' 的低 32 位写入 rs1' 加 uimm 的地址。
			let addr = read_gpr(state, cf.rs1p).wrapping_add(cf.imm);
			let rs2_val = read_gpr(state, cf.rdp) & 0xffff_ffff;
			let ret = store_mem_compressed(
				state,
				ctx,
				addr,
				rs2_val,
				4,
				instr_word,
				instr_group,
				pmp,
				dev,
				clint,
			);
			if ret == EXIT_SENTINEL {
				return EXIT_SENTINEL;
			}
			2
		}
		0b111 => {
			// C.SD: 将 rs2' 写入 rs1' 加 uimm 的地址。
			let addr = read_gpr(state, cf.rs1p).wrapping_add(cf.imm);
			let rs2_val = read_gpr(state, cf.rdp);
			let ret = store_mem_compressed(
				state,
				ctx,
				addr,
				rs2_val,
				8,
				instr_word,
				instr_group,
				pmp,
				dev,
				clint,
			);
			if ret == EXIT_SENTINEL {
				return EXIT_SENTINEL;
			}
			2
		}
		_ => {
			deliver_illegal_instruction(state, instr_word as u64);
			0
		}
	}
}

/// 压缩指令的立即数运算、寄存器间运算、分支与跳转指令:
/// C.ADDI、C.ADDIW、C.LI、C.LUI、C.ADDI16SP、C.SRLI、C.SRAI、C.ANDI、
/// C.SUB、C.XOR、C.OR、C.AND、C.SUBW、C.ADDW、C.J、C.BEQZ 与 C.BNEZ。
fn handle_c1(state: &mut HartState, cf: &CompressedFields, instr_word: u32) -> u64 {
	match cf.funct3 {
		0b000..=0b010 => {
			// C.ADDI、C.ADDIW 与 C.LI
			let imm = cf.imm;
			if cf.funct3 == 0b000 {
				// C.ADDI: rd 增加 imm
				let v = read_gpr(state, cf.rd).wrapping_add(imm);
				write_gpr(state, cf.rd, v);
			} else if cf.funct3 == 0b001 {
				// C.ADDIW: rd 取 rd 与 imm 之和的符号扩展 32 位值
				let v = read_gpr(state, cf.rd).wrapping_add(imm) & 0xffff_ffff;
				write_gpr(state, cf.rd, sext32(v));
			} else {
				// C.LI: rd 取 imm
				write_gpr(state, cf.rd, imm);
			}
			2
		}
		0b011 => {
			if cf.rd == 2 {
				// C.ADDI16SP: sp 增加 nzimm
				if cf.imm == 0 {
					return EXIT_SENTINEL; // 非法编码
				}
				let sp = state.gprs[2];
				state.gprs[2] = sp.wrapping_add(cf.imm);
			} else {
				// C.LUI: rd 取 nzimm 左移 12 位后的值
				if cf.imm == 0 {
					return EXIT_SENTINEL;
				}
				write_gpr(state, cf.rd, (cf.imm << 12) & 0xffff_ffff_ffff_ffff);
			}
			2
		}
		0b100 => {
			// 压缩 ALU 操作: SRLI、SRAI、ANDI、SUB、XOR、OR、AND、SUBW 与 ADDW
			match exec_c1_alu(state, cf) {
				Some(r) => {
					write_gpr(state, cf.rs1p, r);
					2
				}
				None => {
					deliver_illegal_instruction(state, instr_word as u64);
					0
				}
			}
		}
		0b101 => {
			// C.J: pc 增加 offset
			state.pc = state.pc.wrapping_add(cf.imm);
			0
		}
		0b110 | 0b111 => {
			// C.BEQZ 与 C.BNEZ
			let rs1 = cf.rs1p; // rs1 来自位域 [9:7] 的压缩寄存器编码
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
			deliver_illegal_instruction(state, instr_word as u64);
			0
		}
	}
}

/// 压缩指令的栈指针访存与寄存器间运算指令:
/// C.SLLI、C.LWSP、C.LDSP、C.JR、C.MV、C.EBREAK、C.JALR、C.ADD、C.SWSP 与 C.SDSP。
fn handle_c2(
	state: &mut HartState,
	cf: &CompressedFields,
	instr_word: u32,
	instr_group: &mut InstrToBeExec,
	ctx: &WalkCtx,
	pmp: &PmpCtx,
	dev: &DevCtx,
	clint: &ClintCtx,
) -> u64 {
	match cf.funct3 {
		0b000 => {
			// C.SLLI: rd 左移 shamt 位
			let v = read_gpr(state, cf.rd) << cf.imm;
			write_gpr(state, cf.rd, v);
			2
		}
		0b001 => {
			// C.FLDSP: 从 sp 加 uimm 的地址读取双精度值到 fpr[rd], 属于 RV64DC。
			if (state.mstatus & MSTATUS_FS_LS) == 0 {
				deliver_illegal_instruction(state, instr_word as u64);
				return 0;
			}
			let addr = state.gprs[2].wrapping_add(cf.imm);
			let prev_mode = state.mode;
			let val = match load_mem_compressed(
				state,
				ctx,
				addr,
				8,
				false,
				instr_word,
				instr_group,
				pmp,
				dev,
				clint,
			) {
				Some(v) => v,
				None => {
					return EXIT_SENTINEL;
				}
			};
			if state.mode != prev_mode {
				return 0;
			} // 已分发陷态
			state.fprs[cf.rd as usize] = val;
			state.mstatus |= MSTATUS_FS_LS | MSTATUS_SD_LS;
			2
		}
		0b010 => {
			// C.LWSP: 从 sp 加 uimm 的地址读取 32 位值到 rd。
			if cf.rd == 0 {
				deliver_illegal_instruction(state, instr_word as u64);
				return 0;
			}
			let addr = state.gprs[2].wrapping_add(cf.imm);
			let prev_mode = state.mode;
			let val = match load_mem_compressed(
				state,
				ctx,
				addr,
				4,
				false,
				instr_word,
				instr_group,
				pmp,
				dev,
				clint,
			) {
				Some(v) => v,
				None => {
					return EXIT_SENTINEL;
				}
			};
			if state.mode != prev_mode {
				return 0;
			} // 已分发陷态
			let final_val = sext32(val);
			write_gpr(state, cf.rd, final_val);
			2
		}
		0b011 => {
			// C.LDSP: 从 sp 加 uimm 的地址读取 64 位值到 rd。
			if cf.rd == 0 {
				deliver_illegal_instruction(state, instr_word as u64);
				return 0;
			}
			let addr = state.gprs[2].wrapping_add(cf.imm);
			let prev_mode = state.mode;
			let val = match load_mem_compressed(
				state,
				ctx,
				addr,
				8,
				false,
				instr_word,
				instr_group,
				pmp,
				dev,
				clint,
			) {
				Some(v) => v,
				None => {
					return EXIT_SENTINEL;
				}
			};
			if state.mode != prev_mode {
				return 0;
			} // 已分发陷态
			write_gpr(state, cf.rd, val);
			2
		}
		0b100 => {
			let rd_rs1 = cf.rd;
			let rs2 = cf.rs2;
			if rd_rs1 == 0 && rs2 == 0 {
				// 半主机服务关闭: 与 Python 引擎的 trap_ebreak 一致, 分发断点异常。
				if !CFG_HAS_SEMIHOSTING {
					let ebreak_pc = state.pc;
					deliver_trap(state, mcause_val(exc_code::BREAKPOINT, false), ebreak_pc);
					return 0;
				}
				// C.EBREAK: 按 NOP 处理, 见前面 EBREAK 分支。
				return 2;
			} else if rs2 == 0 {
				let target = read_gpr(state, rd_rs1);
				if cf.bit12 != 0 {
					// C.JALR: ra 取 pc 加 2
					state.gprs[1] = state.pc.wrapping_add(2);
				}
				state.pc = target & !1;
				0
			} else if cf.bit12 != 0 {
				// C.ADD: rd 增加 rs2
				let result_val = read_gpr(state, rd_rs1).wrapping_add(read_gpr(state, rs2));
				write_gpr(state, rd_rs1, result_val);
				2
			} else {
				// C.MV: rd 取 rs2
				let v = read_gpr(state, rs2);
				write_gpr(state, rd_rs1, v);
				2
			}
		}
		0b101 => {
			// C.FSDSP: 将 fpr[rs2] 的双精度值写入 sp 加 uimm 的地址, 属于 RV64DC。
			if (state.mstatus & MSTATUS_FS_LS) == 0 {
				deliver_illegal_instruction(state, instr_word as u64);
				return 0;
			}
			let addr = state.gprs[2].wrapping_add(cf.imm2);
			let fpr_val = state.fprs[cf.rs2 as usize];
			let ret = store_mem_compressed(
				state,
				ctx,
				addr,
				fpr_val,
				8,
				instr_word,
				instr_group,
				pmp,
				dev,
				clint,
			);
			if ret == EXIT_SENTINEL {
				return EXIT_SENTINEL;
			}
			state.mstatus |= MSTATUS_FS_LS | MSTATUS_SD_LS;
			2
		}
		0b110 => {
			// C.SWSP: 将 rs2 的低 32 位写入 sp 加 uimm 的地址。
			let addr = state.gprs[2].wrapping_add(cf.imm2);
			let val = read_gpr(state, cf.rs2) & 0xffff_ffff;
			let ret = store_mem_compressed(
				state,
				ctx,
				addr,
				val,
				4,
				instr_word,
				instr_group,
				pmp,
				dev,
				clint,
			);
			if ret == EXIT_SENTINEL {
				return EXIT_SENTINEL;
			}
			2
		}
		0b111 => {
			// C.SDSP: 将 rs2 写入 sp 加 uimm 的地址。
			let addr = state.gprs[2].wrapping_add(cf.imm2);
			let val = read_gpr(state, cf.rs2);
			let ret = store_mem_compressed(
				state,
				ctx,
				addr,
				val,
				8,
				instr_word,
				instr_group,
				pmp,
				dev,
				clint,
			);
			if ret == EXIT_SENTINEL {
				return EXIT_SENTINEL;
			}
			2
		}
		_ => {
			deliver_illegal_instruction(state, instr_word as u64);
			0
		}
	}
}

// ============================================================
//  压缩指令的访存辅助函数
// ============================================================

/// 压缩指令的访存读取。
///
/// 返回 ``None`` 表示命中设备地址, 需要退出本轮加速执行, 退出原因已写入
/// ``instr_group``; ``Some`` 为读到的数据。
#[inline]
fn load_mem_compressed(
	state: &mut HartState,
	ctx: &WalkCtx,
	va: u64,
	size: u8,
	_signed: bool,
	instr_word: u32,
	instr_group: &mut InstrToBeExec,
	pmp: &PmpCtx,
	dev: &DevCtx,
	clint: &ClintCtx,
) -> Option<u64> {
	let tr = match translate_va(state, ctx, va, rwx_read(), pmp) {
		Ok(t) => t,
		Err(TranslateFault::PageFault(cause)) => {
			deliver_trap(state, mcause_val(cause, false), va);
			return Some(0); // 陷态已直接分发; 调用方必须检查 state.mode
		}
		Err(_) => {
			deliver_trap(state, mcause_val(exc_code::LD_ACCESS_FAULT, false), va);
			return Some(0); // 陷态已直接分发; 调用方必须检查 state.mode
		}
	};

	if !pmp_ok(state, tr.pa, size as u32, false, false, pmp) {
		deliver_trap(state, mcause_val(exc_code::LD_ACCESS_FAULT, false), va);
		return Some(0); // 陷态已直接分发; 调用方必须检查 state.mode
	}

	if let Some(data) = try_handle_clint(&MemAccess::read(tr.pa, size), state, clint) {
		return Some(data);
	}

	// IMSIC 的 MMIO 读取一律返回 0, 因为 seteipnum 与 clreipnum 只接受写入。
	// 该分支必须位于 is_device_addr 之前, 否则从 IMSIC 读取的压缩指令会退出到 Python。
	if try_handle_imsic_serial(tr.pa, 0, clint) {
		return Some(0);
	}

	if let Some(data) = try_handle_virtio(tr.pa, false, 0, size, dev) {
		return Some(data);
	}

	// PLIC 直接处理压缩指令的映射访问。
	if let Some(data) = try_handle_plic_concurrent(tr.pa, false, 0, size, state, dev.plic) {
		return Some(data);
	}

	if is_device_addr(tr.pa, dev) {
		instr_group.exit_reason = exit_reason::MMIO;
		instr_group.exit_instr = instr_word;
		return None;
	}

	Some(ram_read(ctx, tr.pa, size))
}

#[inline]
fn store_mem_compressed(
	state: &mut HartState,
	ctx: &WalkCtx,
	va: u64,
	val: u64,
	size: u8,
	instr_word: u32,
	instr_group: &mut InstrToBeExec,
	pmp: &PmpCtx,
	dev: &DevCtx,
	clint: &ClintCtx,
) -> u64 {
	let tr = match translate_va(state, ctx, va, rwx_write(), pmp) {
		Ok(t) => t,
		Err(TranslateFault::PageFault(cause)) => {
			deliver_trap(state, mcause_val(cause, false), va);
			return 0;
		}
		Err(_) => {
			deliver_trap(state, mcause_val(exc_code::ST_ACCESS_FAULT, false), va);
			return 0;
		}
	};

	if !pmp_ok(state, tr.pa, size as u32, true, false, pmp) {
		deliver_trap(state, mcause_val(exc_code::ST_ACCESS_FAULT, false), va);
		return 0;
	}

	if let Some(_) = try_handle_clint(&MemAccess::write(tr.pa, size, val), state, clint) {
		return 0;
	}

	// IMSIC 的写入: 中断号 1 与 3 的 IPI 经 CLINT 的 MSIP 分发, 跨 hart 唤醒不退出加速执行。
	// 该分支必须位于 is_device_addr 之前, 否则写 IMSIC 的 seteipnum 的 C.SW 会退出到 Python 的 MMIO 处理。
	// 缺此分支时, 用压缩指令分发 IPI 的固件在多 hart 的 TLB 刷新中停止推进, 因 IPI 未进入目标 hart 的线程。
	if try_handle_imsic_serial(tr.pa, val, clint) {
		return 0;
	}

	if let Some(_) = try_handle_virtio(tr.pa, true, val, size, dev) {
		return 0;
	}

	// PLIC 直接处理压缩指令的映射访问。
	if let Some(_) = try_handle_plic_concurrent(tr.pa, true, val, size, state, dev.plic) {
		return 0;
	}

	if is_device_addr(tr.pa, dev) {
		instr_group.exit_reason = exit_reason::MMIO;
		instr_group.exit_instr = instr_word;
		return EXIT_SENTINEL;
	}

	ram_write(ctx, tr.pa, val, size);
	0 // 调用方返回 2
}

// ============================================================
// ============================================================
//
// 半主机(semihosting)是固件与宿主调试器或模拟器以特定陷态序列通信的调试机制,
// RISC-V 的半主机调用序列为:
//
//   slli zero, zero, 0x1f    0x01f01013      入口标记
//   ebreak                   0x00100073      陷态
//   srai zero, zero, 0x7     0x40705013      出口标记
//
// OpenSBI 以 CONFIG_SEMIHOSTING=y 构建时用半主机做早期控制台输出。
// 模拟器截获该序列并分派半主机操作, 如 SYS_WRITE 与 SYS_WRITEC。

/// 入口标记, 即 ``slli zero, zero, 0x1f``。
const SEMIHOSTING_PRE: u32 = 0x01f01013;
/// 出口标记, 即 ``srai zero, zero, 0x7``。
const SEMIHOSTING_POST: u32 = 0x40705013;

/// 操作码。
const SH_SYS_OPEN: u64 = 0x01;
const SH_SYS_WRITEC: u64 = 0x03;
const SH_SYS_WRITE: u64 = 0x05;
const SH_SYS_ISTTY: u64 = 0x09;
/// SYS_EXIT: 停机序列的操作码。
pub(crate) const SH_SYS_EXIT: u64 = 0x18;

/// 从 RAM 的物理地址 ``pa`` 读取一个 64 位值。Bare 模式下虚拟地址等于物理地址。
#[inline]
unsafe fn read_ram_u64(ctx: &WalkCtx, pa: u64) -> u64 {
	if pa + 8 > ctx.ram_base + ctx.ram_size || pa < ctx.ram_base {
		return 0;
	}
	let off = (pa - ctx.ram_base) as isize;
	(ctx.ram.offset(off) as *const u64).read_unaligned()
}

/// 从 RAM 的物理地址 ``pa`` 读取一个字节。
#[inline]
unsafe fn read_ram_u8(ctx: &WalkCtx, pa: u64) -> u8 {
	if pa >= ctx.ram_base + ctx.ram_size || pa < ctx.ram_base {
		return 0;
	}
	let off = (pa - ctx.ram_base) as isize;
	*ctx.ram.offset(off)
}

/// 记录最近一次附加 ``[hart X]`` 标记的 hart 编号。
/// 只在输出 hart 改变时附加标记, 不在每次换行时附加, 避免固件逐行打印多个字段时
/// 输出在同一行内交错。
static SH_LAST_TAGGED_HART: Mutex<Option<u64>> = Mutex::new(None);

/// 把 ``data`` 写入 stderr, 仅在输出 hart 改变时附加 ``[hart X]`` 标记,
/// 不在每次换行时附加。
fn write_labeled_stderr(data: &[u8], hart_id: u64) {
	extern "C" {
		fn write(fd: i32, buf: *const u8, count: usize) -> isize;
	}

	// 仅在输出 hart 改变时附加标记。
	{
		let mut last = SH_LAST_TAGGED_HART.lock().unwrap();
		if *last != Some(hart_id) {
			// 粗体蓝色的 ANSI 前缀。
			let prefix = format!("\x1b[1;34m[hart {}]\x1b[0m ", hart_id);
			let prefix_bytes = prefix.as_bytes();
			let mut off = 0;
			while off < prefix_bytes.len() {
				let n = unsafe { write(2, prefix_bytes[off..].as_ptr(), prefix_bytes.len() - off) };
				if n <= 0 {
					break;
				}
				off += n as usize;
			}
			*last = Some(hart_id);
		}
	}

	// 把数据写入 stderr, 不按换行附加标记。
	let mut off = 0;
	while off < data.len() {
		let n = unsafe { write(2, data[off..].as_ptr(), data.len() - off) };
		if n <= 0 {
			break;
		}
		off += n as usize;
	}
}

/// 直接向文件描述符写入, 不经过 Rust 标准输入输出的锁, 避免与 Python 侧
/// rich 与 prompt_toolkit 对终端控制的争用。写入 stderr, 即文件描述符 2,
/// 使半主机输出不改动调试器的终端界面。
#[allow(unused)]
fn raw_write_stderr(data: &[u8]) {
	extern "C" {
		fn write(fd: i32, buf: *const u8, count: usize) -> isize;
	}
	let mut off = 0usize;
	while off < data.len() {
		let remaining = &data[off..];
		let n = unsafe { write(2, remaining.as_ptr(), remaining.len()) };
		if n <= 0 {
			break;
		}
		off += n as usize;
	}
}

/// 执行一次半主机操作, 返回写入 ``a0`` 的值。
fn semihosting_dispatch(ctx: &WalkCtx, op: u64, param_block: u64, hart_id: u64) -> u64 {
	match op {
		SH_SYS_WRITEC => {
			let ch = unsafe { read_ram_u8(ctx, param_block) };
			write_labeled_stderr(&[ch], hart_id);
			0
		}
		SH_SYS_WRITE => {
			let fd = unsafe { read_ram_u64(ctx, param_block) };
			let buf_addr = unsafe { read_ram_u64(ctx, param_block + 8) };
			let len = unsafe { read_ram_u64(ctx, param_block + 16) };
			if len == 0 {
				return 0;
			}
			// 全部输出写入 stderr, 即文件描述符 2; stdout 由调试器的
			// rich 与 prompt_toolkit 终端界面占用。
			if fd != 1 && fd != 2 {
				return -1i64 as u64;
			}
			if buf_addr >= ctx.ram_base && buf_addr + len <= ctx.ram_base + ctx.ram_size {
				let off = (buf_addr - ctx.ram_base) as isize;
				let data = unsafe { std::slice::from_raw_parts(ctx.ram.offset(off), len as usize) };
				write_labeled_stderr(data, hart_id);
				return 0;
			}
			return -1i64 as u64;
		}
		SH_SYS_OPEN => {
			// 返回一个合法的文件描述符, 使固件的探测得到成功结果。
			1
		}
		SH_SYS_ISTTY => {
			let fd = unsafe { read_ram_u64(ctx, param_block) };
			u64::from(fd <= 2)
		}
		_ => -1i64 as u64,
	}
}

/// 检查 ``ebreak_pc`` 前后是否为半主机标记序列
/// ``slli x0, x0, 0x1f; ebreak; srai x0, x0, 7``。
/// 该序列仅在虚拟地址等于物理地址时成立, 即 M 模式且翻译模式为 Bare。
pub(crate) fn semihosting_match(ctx: &WalkCtx, ebreak_pc: u64) -> bool {
	// 半主机仅在虚拟地址等于物理地址时成立, 即 M 模式且翻译模式为 Bare。
	if ebreak_pc < ctx.ram_base || ebreak_pc + 4 > ctx.ram_base + ctx.ram_size {
		return false;
	}
	let ram_off = (ebreak_pc - ctx.ram_base) as isize;
	if ram_off < 4 {
		return false;
	}

	// 检查 ebreak 之前的标记
	let prev: u32 = unsafe { (ctx.ram.offset(ram_off - 4) as *const u32).read_unaligned() };
	if prev != u32::from_le(SEMIHOSTING_PRE) {
		return false;
	}

	// 检查 ebreak 之后的标记
	let next: u32 = unsafe { (ctx.ram.offset(ram_off + 4) as *const u32).read_unaligned() };
	next == u32::from_le(SEMIHOSTING_POST)
}

// 半主机服务由配置项 PYREMU_HAS_SEMIHOSTING 控制, 生成值见
// ``configs_gen.rs`` 的 ``CFG_HAS_SEMIHOSTING``。取值为 0 时标记序列按普通
// ebreak 分发断点陷态: 固件的半主机探测因此判定宿主不支持, 控制台回落到
// UART, 输出经发送环形缓冲区写入每 hart 的日志。

/// 检查 ``ebreak_pc`` 处是否属于一次半主机调用序列。
///
/// 若 ebreak 前后的指令与半主机标记一致, 本函数执行该操作, 更新 ``a0``,
/// 返回 ``Some(8)``, 即跳过 4 字节的 ebreak 与 4 字节的出口标记。
/// 普通 ebreak 返回 ``None``; ``CFG_HAS_SEMIHOSTING`` 为假时所有 ebreak 均返回 ``None``。
pub fn try_semihosting(state: &mut HartState, ctx: &WalkCtx, ebreak_pc: u64) -> Option<u64> {
	if !CFG_HAS_SEMIHOSTING || !semihosting_match(ctx, ebreak_pc) {
		return None;
	}

	let op = state.gprs[10]; // a0
	let param_block = state.gprs[11]; // a1
	let hart_id = state.mhartid;
	state.gprs[10] = semihosting_dispatch(ctx, op, param_block, hart_id);

	// 跳过 ebreak 与出口标记共 8 字节, 其后的 sext.w、bltz、sub 与 j
	// 自行处理返回值。
	Some(8)
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::state::HartState;
	use crate::translate::WalkCtx;
	use std::cell::Cell;
	use std::mem;

	#[test]
	fn sext32_sign_extends() {
		assert_eq!(sext32(0x7fffffff), 0x7fffffff);
		assert_eq!(sext32(0x80000000), 0xffff_ffff_8000_0000u64);
		assert_eq!(sext32(0xffffffff), 0xffff_ffff_ffff_ffffu64);
	}

	#[test]
	fn sext_various_widths() {
		assert_eq!(sext(0xff, 8), 0xffff_ffff_ffff_ffffu64);
		assert_eq!(sext(0x7f, 8), 0x7f);
		assert_eq!(sext(0x8000, 16), 0xffff_ffff_ffff_8000u64);
		assert_eq!(sext(0x7fff, 16), 0x7fff);
	}

	/// 直接测试: MemAccess::write 与 try_handle_clint 共同置位 MSIP。
	#[test]
	fn try_handle_clint_msip_write_direct() {
		let mut s0: HartState = unsafe { mem::zeroed() };
		s0.mhartid = 0;
		let mut s1: HartState = unsafe { mem::zeroed() };
		s1.mhartid = 1;

		let mut mtime: u64 = 0;
		let mut mtimecmp: [u64; 2] = [u64::MAX; 2];
		let mut msip: [u8; 2] = [0; 2];

		let mut states = [s0, s1];
		let ctx = ClintCtx {
			base: 0x200_0000,
			mtime: &mut mtime as *mut u64,
			mtimecmp: mtimecmp.as_mut_ptr(),
			msip: msip.as_mut_ptr(),
			states: states.as_mut_ptr(),
			num_harts: 2,
			timebase_hz: 0, // 单元测试: 时钟源不推进, mtime 保持不变
			yield_for_ipi: Cell::new(false),
			ipi_sender_hart: Cell::new(0),
			ipi_sender_rounds: Cell::new(0),
			msip_pending: Cell::new(std::ptr::null()),
			hart_threads: Cell::new(std::ptr::null()),
			hart_states: Cell::new(std::ptr::null()),
		};

		// 写入 MSIP[1], 目标为另一个 hart
		let access = MemAccess::write(0x200_0004, 4, 1);
		let result = try_handle_clint(&access, &states[0], &ctx);
		assert_eq!(result, Some(0));
		assert_eq!(msip[1], 1, "direct: MSIP[1] must be 1");
		assert!(
			(states[1].mip.load(Ordering::Acquire) & (1 << 3)) != 0,
			"direct: hart 1 mip must have MSIP"
		);
		assert!(
			ctx.yield_for_ipi.get(),
			"direct: cross-hart must set yield flag"
		);
		assert_eq!(
			ctx.ipi_sender_hart.get(),
			0,
			"direct: sender hart must be tracked for short-slice"
		);
		assert_eq!(
			ctx.ipi_sender_rounds.get(),
			16,
			"direct: short-slice rounds must be initialised"
		);

		// 写入 MSIP[0], 自己向自己分发 IPI 时不得置位 yield 标志, 也不记录发送方
		ctx.yield_for_ipi.set(false);
		ctx.ipi_sender_rounds.set(0);
		let access_self = MemAccess::write(0x200_0000, 4, 1);
		let result2 = try_handle_clint(&access_self, &states[0], &ctx);
		assert_eq!(result2, Some(0));
		assert_eq!(msip[0], 1, "direct: MSIP[0] must be 1 after self-IPI");
		assert!(
			!ctx.yield_for_ipi.get(),
			"direct: self-IPI must NOT set yield flag"
		);
		assert_eq!(
			ctx.ipi_sender_rounds.get(),
			0,
			"direct: self-IPI must NOT set short-slice rounds"
		);
	}

	/// MSIP 的清零同样经 ``clint_write_msip`` 完成, 并清除目标 hart 的 mip 位。
	#[test]
	fn try_handle_clint_msip_clear_updates_mip() {
		let mut s0: HartState = unsafe { mem::zeroed() };
		s0.mhartid = 0;
		s0.mip.store(1 << 3, Ordering::Release); // MSIP 处于挂起状态

		let mut mtime: u64 = 0;
		let mut mtimecmp: [u64; 2] = [u64::MAX; 2];
		let mut msip: [u8; 2] = [1, 1]; // 两个 MSIP 均已置位

		let mut states = [s0, unsafe { mem::zeroed() }];
		let ctx = ClintCtx {
			base: 0x200_0000,
			mtime: &mut mtime as *mut u64,
			mtimecmp: mtimecmp.as_mut_ptr(),
			msip: msip.as_mut_ptr(),
			states: states.as_mut_ptr(),
			num_harts: 2,
			timebase_hz: 0, // 单元测试: 时钟源不推进, mtime 保持不变
			yield_for_ipi: Cell::new(false),
			ipi_sender_hart: Cell::new(0),
			ipi_sender_rounds: Cell::new(0),
			msip_pending: Cell::new(std::ptr::null()),
			hart_threads: Cell::new(std::ptr::null()),
			hart_states: Cell::new(std::ptr::null()),
		};

		// 清除 MSIP[0], 目标 hart 与当前 hart 均为 0
		let access = MemAccess::write(0x200_0000, 4, 0);
		let result = try_handle_clint(&access, &states[0], &ctx);
		assert_eq!(result, Some(0));
		assert_eq!(msip[0], 0, "MSIP[0] should be cleared");
		assert_eq!(
			states[0].mip.load(Ordering::Acquire) & (1 << 3),
			0,
			"mip.MSIP must be 0 after clear"
		);
		// 清除自己的 MSIP 不得置位 yield 标志, 也不记录发送方
		assert!(!ctx.yield_for_ipi.get(), "MSIP clear should not set yield");
		assert_eq!(
			ctx.ipi_sender_rounds.get(),
			0,
			"MSIP clear should not set short-slice rounds"
		);
	}

	// ============================================================
	//  半主机测试
	// ============================================================

	/// 在 RAM 的 ``offset`` 处构造半主机陷态序列。
	/// 返回 ebreak 指令的地址, 即 ram_base 加 offset。
	fn write_semihosting_seq(ram: &mut [u8], ram_base: u64, offset: usize) -> u64 {
		let pre: [u8; 4] = (0x01f01013u32).to_le_bytes(); // slli zero, zero, 0x1f
		let ebr: [u8; 4] = (0x00100073u32).to_le_bytes(); // ebreak
		let post: [u8; 4] = (0x40705013u32).to_le_bytes(); // srai zero, zero, 0x7

		ram[offset - 4..offset].copy_from_slice(&pre);
		ram[offset..offset + 4].copy_from_slice(&ebr);
		ram[offset + 4..offset + 8].copy_from_slice(&post);
		ram_base + (offset as u64)
	}

	/// 把 64 位值写入 RAM 的 ``offset`` 处。
	fn write_ram_u64(ram: &mut [u8], offset: usize, val: u64) {
		ram[offset..offset + 8].copy_from_slice(&val.to_le_bytes());
	}

	#[test]
	fn semihosting_writec_outputs_char() {
		if !CFG_HAS_SEMIHOSTING {
			return;
		}

		let mut ram: Vec<u8> = vec![0u8; 0x200];
		let ram_base: u64 = 0x8000_0000;

		// 在偏移 0x100 处放置一个字符
		ram[0x100] = b'X';

		// 在偏移 0x80 处放置半主机序列, 其 ebreak 位于该处
		let ebreak_pc = write_semihosting_seq(&mut ram, ram_base, 0x80);

		let mut state: HartState = unsafe { std::mem::zeroed() };
		state.gprs[10] = super::SH_SYS_WRITEC; // a0 取操作码 WRITEC
		state.gprs[11] = ram_base + 0x100; // a1 指向字符 'X'
		state.pc = ebreak_pc;
		state.mode = crate::state::riscv_mode::M;

		let ctx = WalkCtx {
			ram: ram.as_mut_ptr(),
			ram_size: ram.len() as u64,
			ram_base,
			shadow_base: 0,
			shadow_size: 0,
			tlb_flush_cnt: std::ptr::null(),
			lr_reserved: std::ptr::null_mut(),
			num_harts: 1,
		};

		let result = try_semihosting(&mut state, &ctx, ebreak_pc);
		assert_eq!(result, Some(8), "should return skip-8 for semihosting");
		assert_eq!(state.gprs[10], 0, "WRITEC should return 0 on success");
	}

	#[test]
	fn semihosting_write_outputs_buffer() {
		if !CFG_HAS_SEMIHOSTING {
			return;
		}

		let mut ram: Vec<u8> = vec![0u8; 0x200];
		let ram_base: u64 = 0x8000_0000;

		let msg = b"Hello";
		let msg_off = 0x100;
		ram[msg_off..msg_off + 5].copy_from_slice(msg);

		// 偏移 0x80 处的参数块:
		//     偏移 0: 文件描述符, 1 表示 stdout
		//     偏移 8: 缓冲区地址
		//     偏移 16: 长度
		write_ram_u64(&mut ram, 0x80, 1); // 文件描述符取 1
		write_ram_u64(&mut ram, 0x88, ram_base + (msg_off as u64)); // 缓冲区地址
		write_ram_u64(&mut ram, 0x90, 5); // 长度

		// 在偏移 0xa0 处放置半主机序列, 其 ebreak 位于该处
		let ebreak_pc = write_semihosting_seq(&mut ram, ram_base, 0xa0);

		let mut state: HartState = unsafe { std::mem::zeroed() };
		state.gprs[10] = super::SH_SYS_WRITE; // a0 取操作码 WRITE
		state.gprs[11] = ram_base + 0x80; // a1 指向参数块
		state.pc = ebreak_pc;
		state.mode = crate::state::riscv_mode::M;

		let ctx = WalkCtx {
			ram: ram.as_mut_ptr(),
			ram_size: ram.len() as u64,
			ram_base,
			shadow_base: 0,
			shadow_size: 0,
			tlb_flush_cnt: std::ptr::null(),
			lr_reserved: std::ptr::null_mut(),
			num_harts: 1,
		};

		let result = try_semihosting(&mut state, &ctx, ebreak_pc);
		assert_eq!(result, Some(8), "should return skip-8 for semihosting");
		assert_eq!(state.gprs[10], 0, "WRITE should return 0 on success");
	}

	#[test]
	fn semihosting_regular_ebreak_not_handled() {
		let mut ram: Vec<u8> = vec![0u8; 0x200];
		let ram_base: u64 = 0x8000_0000;

		// 放置没有半主机标记的 ebreak
		let ebr_addr = ram_base + 0x80;
		let ebr: [u8; 4] = (0x00100073u32).to_le_bytes();
		ram[0x80..0x84].copy_from_slice(&ebr);

		let mut state: HartState = unsafe { std::mem::zeroed() };
		state.pc = ebr_addr;
		state.mode = crate::state::riscv_mode::M;

		let ctx = WalkCtx {
			ram: ram.as_mut_ptr(),
			ram_size: ram.len() as u64,
			ram_base,
			shadow_base: 0,
			shadow_size: 0,
			tlb_flush_cnt: std::ptr::null(),
			lr_reserved: std::ptr::null_mut(),
			num_harts: 1,
		};

		let result = try_semihosting(&mut state, &ctx, ebr_addr);
		assert_eq!(result, None, "plain ebreak should NOT be semihosting");
	}

	fn make_dev_ctx(bases: &[u64], ends: &[u64]) -> DevCtx {
		DevCtx {
			bases: if bases.is_empty() {
				std::ptr::null()
			} else {
				bases.as_ptr()
			},
			ends: if ends.is_empty() {
				std::ptr::null()
			} else {
				ends.as_ptr()
			},
			num: bases.len() as u8,
			virtio_base: 0,
			virtio_raw: std::ptr::null_mut(),
			plic: std::ptr::null_mut(),
		}
	}

	/// 覆盖整个物理地址空间的 TOR 条目, R、W 与 X 位均置位, 供访存测试放行。
	fn make_dev_test_pmp(cfg: &mut [u8], addr: &mut [u64]) -> PmpCtx {
		cfg[0] = 0x0f;
		addr[0] = u64::MAX;
		PmpCtx {
			cfg: cfg.as_mut_ptr(),
			addr: addr.as_mut_ptr(),
			num: 1,
		}
	}

	fn make_test_clint(mtime: &mut u64, mtimecmp: &mut [u64], msip: &mut [u8]) -> ClintCtx {
		ClintCtx {
			base: 0x200_0000,
			mtime: mtime as *mut u64,
			mtimecmp: mtimecmp.as_mut_ptr(),
			msip: msip.as_mut_ptr(),
			states: std::ptr::null_mut(),
			num_harts: 1,
			timebase_hz: 0,
			yield_for_ipi: Cell::new(false),
			ipi_sender_hart: Cell::new(0),
			ipi_sender_rounds: Cell::new(0),
			msip_pending: Cell::new(std::ptr::null()),
			hart_threads: Cell::new(std::ptr::null()),
			hart_states: Cell::new(std::ptr::null()),
		}
	}

	/// 内存中的全 1 数据是合法内容, 如内核的 ``~0UL``、``-1`` 以及全 1 的
	/// 浮点 NaN 载荷, 不得与 ``EXIT_SENTINEL`` 混淆而中断加速执行。
	#[test]
	fn compressed_load_all_ones_from_ram_is_not_exit() {
		let mut ram: Vec<u8> = vec![0u8; 0x400];
		let ram_base: u64 = 0x8000_0000;
		ram[0x100..0x108].copy_from_slice(&u64::MAX.to_le_bytes());

		let mut state: HartState = unsafe { mem::zeroed() };
		state.mode = crate::state::riscv_mode::M;
		state.gprs[2] = ram_base + 0x100; // sp
		state.gprs[9] = ram_base + 0x100; // s1

		let ctx = WalkCtx {
			ram: ram.as_mut_ptr(),
			ram_size: ram.len() as u64,
			ram_base,
			shadow_base: 0,
			shadow_size: 0,
			tlb_flush_cnt: std::ptr::null(),
			lr_reserved: std::ptr::null_mut(),
			num_harts: 1,
		};
		let mut cfg = [0u8; 64];
		let mut addr = [0u64; 64];
		let pmp = make_dev_test_pmp(&mut cfg, &mut addr);
		let dev = make_dev_ctx(&[], &[]);
		let mut mtime: u64 = 0;
		let mut mtimecmp = [u64::MAX; 1];
		let mut msip = [0u8; 1];
		let clint = make_test_clint(&mut mtime, &mut mtimecmp, &mut msip);

		// C.LDSP x5, 0(sp)
		let mut group: InstrToBeExec = unsafe { mem::zeroed() };
		let advance = handle_compressed(
			&mut state, 0x6282, 0x6282, &mut group, &ctx, &pmp, &dev, &clint,
		);
		assert_eq!(advance, 2, "读到全 1 的普通内存不得被判为设备退出");
		assert_eq!(state.gprs[5], u64::MAX, "寄存器应拿到内存里真实的全 1 数据");
		assert_eq!(group.exit_reason, exit_reason::NORMAL);

		// C.LD x10, 0(s1)
		let mut group2: InstrToBeExec = unsafe { mem::zeroed() };
		let advance2 = handle_compressed(
			&mut state,
			0x6088,
			0x6088,
			&mut group2,
			&ctx,
			&pmp,
			&dev,
			&clint,
		);
		assert_eq!(advance2, 2, "C.LD 同样不得误判为设备退出");
		assert_eq!(state.gprs[10], u64::MAX);
	}

	/// 设备访问必须退出到 Python 且退出原因标记为 MMIO。
	#[test]
	fn compressed_device_access_exits_with_mmio_reason() {
		let mut ram: Vec<u8> = vec![0u8; 0x400];
		let ram_base: u64 = 0x8000_0000;
		let dev_base: u64 = 0x1000_0000;
		let dev_end: u64 = 0x1000_1000;

		let mut state: HartState = unsafe { mem::zeroed() };
		state.mode = crate::state::riscv_mode::M;
		state.gprs[2] = dev_base; // sp 指向设备
		state.gprs[5] = 0xdead_beef;

		let ctx = WalkCtx {
			ram: ram.as_mut_ptr(),
			ram_size: ram.len() as u64,
			ram_base,
			shadow_base: 0,
			shadow_size: 0,
			tlb_flush_cnt: std::ptr::null(),
			lr_reserved: std::ptr::null_mut(),
			num_harts: 1,
		};
		let mut cfg = [0u8; 64];
		let mut addr = [0u64; 64];
		let pmp = make_dev_test_pmp(&mut cfg, &mut addr);
		let bases = [dev_base];
		let ends = [dev_end];
		let dev = make_dev_ctx(&bases, &ends);
		let mut mtime: u64 = 0;
		let mut mtimecmp = [u64::MAX; 1];
		let mut msip = [0u8; 1];
		let clint = make_test_clint(&mut mtime, &mut mtimecmp, &mut msip);

		// C.SDSP x5, 0(sp)
		let mut group: InstrToBeExec = unsafe { mem::zeroed() };
		let advance = handle_compressed(
			&mut state, 0xe016, 0xe016, &mut group, &ctx, &pmp, &dev, &clint,
		);
		assert_eq!(advance, EXIT_SENTINEL, "设备存储必须退出到 Python");
		assert_eq!(group.exit_reason, exit_reason::MMIO, "退出原因必须为 MMIO");

		// C.LDSP x5, 0(sp)
		let mut group2: InstrToBeExec = unsafe { mem::zeroed() };
		let advance2 = handle_compressed(
			&mut state,
			0x6282,
			0x6282,
			&mut group2,
			&ctx,
			&pmp,
			&dev,
			&clint,
		);
		assert_eq!(advance2, EXIT_SENTINEL, "设备读取必须退出到 Python");
		assert_eq!(group2.exit_reason, exit_reason::MMIO, "退出原因必须为 MMIO");
	}
}
