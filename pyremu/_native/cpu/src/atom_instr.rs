use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use crate::concurrent::{ModuleState, StopInfo};
use crate::handlers::{
	lr_check, lr_clear_all, lr_set, pmp_ok, try_handle_virtio, DevCtx, PmpCtx, EXIT_SENTINEL,
};
use crate::hart_sched::{ram_offset, read_gpr};
use crate::peripheral::is_device_addr;
use crate::state::{exit_reason, HartState};
use crate::translate::{rwx_write, translate_va, TranslateFault, WalkCtx};
use crate::trap::{deliver_illegal_instruction, deliver_trap, exc_code, mcause_val};

pub(crate) fn atomic_minmax_u32(atomic: &AtomicU32, funct5: u8, rs2: u32) -> u64 {
	loop {
		let cur = atomic.load(Ordering::Relaxed);
		let next = match funct5 {
			0b10000 => (cur as i32).min(rs2 as i32) as u32, // AMOMIN
			0b10100 => (cur as i32).max(rs2 as i32) as u32, // AMOMAX
			0b11000 => cur.min(rs2),                        // AMOMINU
			0b11100 => cur.max(rs2),                        // AMOMAXU
			_ => {
				return 0;
			}
		};
		match atomic.compare_exchange_weak(cur, next, Ordering::AcqRel, Ordering::Relaxed) {
			Ok(v) => {
				return v as u64;
			}
			Err(_) => {
				continue;
			}
		}
	}
}

pub(crate) fn atomic_minmax_u64(atomic: &AtomicU64, funct5: u8, rs2: u64) -> u64 {
	loop {
		let cur = atomic.load(Ordering::Relaxed);
		let next = match funct5 {
			0b10000 => (cur as i64).min(rs2 as i64) as u64, // AMOMIN
			0b10100 => (cur as i64).max(rs2 as i64) as u64, // AMOMAX
			0b11000 => cur.min(rs2),                        // AMOMINU
			0b11100 => cur.max(rs2),                        // AMOMAXU
			_ => {
				return 0;
			}
		};
		match atomic.compare_exchange_weak(cur, next, Ordering::AcqRel, Ordering::Relaxed) {
			Ok(v) => {
				return v;
			}
			Err(_) => {
				continue;
			}
		}
	}
}

/// LR.W 与 LR.D: 从 off 读取一个字或双字, 并把读出的值、地址 pa 与有效标志分别写入
/// state.reservation_value、state.reservation_addr 与 state.reservation_valid.
///
/// RISC-V 规范 §8.2: 在 RV64 上 LR.W 把读出的 32 位值符号扩展到 64 位.
#[inline]
fn exec_lr(state: &mut HartState, width: u8, rd: u8, pa: u64, off: usize, ctx: &WalkCtx) {
	let loaded: u64 = if width == 4 {
		let atomic = unsafe { &*(ctx.ram.add(off) as *const AtomicU32) };
		let raw = atomic.load(Ordering::Acquire);
		raw as i32 as i64 as u64
	} else {
		let atomic = unsafe { &*(ctx.ram.add(off) as *const AtomicU64) };
		atomic.load(Ordering::Acquire)
	};
	state.reservation_valid = 1;
	state.reservation_addr = pa;
	state.reservation_value = loaded;
	// 经 lr_set 把 pa 写入各 hart 共享的预留地址数组, 其他 hart 的存储操作经
	// lr_clear_all 将该条目清零.
	lr_set(ctx, state.mhartid as u8, pa);
	state.gprs[rd as usize] = if rd != 0 { loaded } else { 0 };
}

/// SC.W 与 SC.D: 若 state.reservation_valid 为 1 且 state.reservation_addr 等于 pa,
/// 且 lr_check 确认共享预留地址数组中本 hart 的条目仍为 pa, 则把 rs2_val 写入 off.
#[inline]
fn exec_sc(
	state: &mut HartState,
	width: u8,
	rs2_val: u64,
	rd: u8,
	pa: u64,
	off: usize,
	ctx: &WalkCtx,
) {
	let shared_ok = lr_check(ctx, state.mhartid as u8, pa);
	if !shared_ok || state.reservation_valid == 0 || state.reservation_addr != pa {
		state.reservation_valid = 0;
		if rd != 0 {
			state.gprs[rd as usize] = 1;
		}
		return;
	}
	// 在 CAS 前清除全部 hart 的预留地址数组条目, 并设置 SeqCst 屏障,
	// 使其他 hart 在观察到新值之前或同时观察到清除.
	lr_clear_all(ctx);
	std::sync::atomic::fence(Ordering::SeqCst);
	let success = if width == 4 {
		let atomic = unsafe { &*(ctx.ram.add(off) as *const AtomicU32) };
		atomic
			.compare_exchange(
				state.reservation_value as u32,
				rs2_val as u32,
				Ordering::Release,
				Ordering::Relaxed,
			)
			.is_ok()
	} else {
		let atomic = unsafe { &*(ctx.ram.add(off) as *const AtomicU64) };
		atomic
			.compare_exchange(
				state.reservation_value,
				rs2_val,
				Ordering::Release,
				Ordering::Relaxed,
			)
			.is_ok()
	};
	state.reservation_valid = 0;
	if rd != 0 {
		state.gprs[rd as usize] = if success { 0 } else { 1 };
	}
}

/// AMO 指令: AMOSWAP、AMOADD、AMOXOR、AMOAND、AMOOR、AMOMIN、AMOMAX、AMOMINU、AMOMAXU.
/// 从 off 读取一个字或双字, 按 funct5 与 rs2_val 运算,
/// 并把写入前的值写入 rd.
#[inline]
fn exec_amo_rmw(
	state: &mut HartState,
	funct5: u8,
	width: u8,
	rs2_val: u64,
	rd: u8,
	off: usize,
	ctx: &WalkCtx,
) {
	// 在原子写入之前清除全部 hart 的预留地址数组条目, 避免 ABA 问题.
	lr_clear_all(ctx);
	std::sync::atomic::fence(Ordering::SeqCst);
	let old: u64 = if width == 4 {
		let atomic = unsafe { &*(ctx.ram.add(off) as *const AtomicU32) };
		let rv = rs2_val as u32;
		match funct5 {
			0b00001 => atomic.swap(rv, Ordering::AcqRel) as u64,
			0b00000 => atomic.fetch_add(rv, Ordering::AcqRel) as u64,
			0b00100 => atomic.fetch_xor(rv, Ordering::AcqRel) as u64,
			0b01100 => atomic.fetch_and(rv, Ordering::AcqRel) as u64,
			0b01000 => atomic.fetch_or(rv, Ordering::AcqRel) as u64,
			_ => atomic_minmax_u32(atomic, funct5, rv),
		}
	} else {
		let atomic = unsafe { &*(ctx.ram.add(off) as *const AtomicU64) };
		let old_val = match funct5 {
			0b00001 => atomic.swap(rs2_val, Ordering::AcqRel),
			0b00000 => atomic.fetch_add(rs2_val, Ordering::AcqRel),
			0b00100 => atomic.fetch_xor(rs2_val, Ordering::AcqRel),
			0b01100 => atomic.fetch_and(rs2_val, Ordering::AcqRel),
			0b01000 => atomic.fetch_or(rs2_val, Ordering::AcqRel),
			_ => atomic_minmax_u64(atomic, funct5, rs2_val),
		};
		old_val
	};
	if rd != 0 {
		state.gprs[rd as usize] = if width == 4 { old as u64 } else { old };
	}
}

/// 按 `funct5` 字段执行一条 LR / SC / AMO 指令.
/// 返回 PC 的推进量 4; 编码非法时返回 0, 由调用方处理陷态.
#[inline]
fn exec_amo_op(
	state: &mut HartState,
	funct5: u8,
	width: u8,
	rs2_val: u64,
	rd: u8,
	pa: u64,
	off: usize,
	ctx: &WalkCtx,
) -> u64 {
	match funct5 {
		0b00010 => {
			exec_lr(state, width, rd, pa, off, ctx);
			4
		}
		0b00011 => {
			exec_sc(state, width, rs2_val, rd, pa, off, ctx);
			4
		}
		0b00001 | 0b00000 | 0b00100 | 0b01100 | 0b01000 | 0b10000 | 0b10100 | 0b11000 | 0b11100 => {
			exec_amo_rmw(state, funct5, width, rs2_val, rd, off, ctx);
			4
		}
		_ => 0,
	}
}

pub(crate) fn handle_amo_concurrent(
	state: &mut HartState,
	funct5: u8,
	funct3: u8,
	rs2_val: u64,
	rd: u8,
	pa: u64,
	ctx: &WalkCtx,
	pmp: &PmpCtx,
	dev: &DevCtx,
) -> u64 {
	let width: u8 = match funct3 {
		0b010 => 4,
		0b011 => 8,
		_ => {
			return 0;
		} // 编码非法, 由调用方处理
	};

	// 对齐检查: RISC-V AMO 要求自然对齐
	if (pa & ((width as u64) - 1)) != 0 {
		return 0; // 由调用方投递不对齐陷态
	}

	if !pmp_ok(state, pa, width as u32, true, false, pmp) {
		return 0;
	}

	if is_device_addr(pa, dev) {
		return EXIT_SENTINEL; // 返回 Python 侧处理 MMIO
	}

	let off = match ram_offset(
		pa,
		width as u32,
		ctx.ram_base,
		ctx.ram_size,
		ctx.shadow_base,
		ctx.shadow_size,
	) {
		Some(o) => o as usize,
		None => {
			return 0;
		}
	};

	exec_amo_op(state, funct5, width, rs2_val, rd, pa, off, ctx)
}

pub(crate) fn handle_amo_concurrent_dispatch(
	state: &mut HartState,
	f: &crate::decode::DecodedFields,
	instr: u32,
	ctx: &WalkCtx,
	pmp: &PmpCtx,
	dev: &DevCtx,
	module: &ModuleState,
) -> u64 {
	let funct5 = f.func7 >> 2;
	let width: u8 = match f.func3 {
		0b010 => 4,
		0b011 => 8,
		_ => {
			deliver_illegal_instruction(state, instr as u64);
			return 0;
		}
	};

	let base = read_gpr(state, f.rs1);
	let va = base;

	if (va & ((width as u64) - 1)) != 0 {
		deliver_trap(state, mcause_val(exc_code::LD_MISALIGNED, false), va);
		return 0;
	}

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

	if !pmp_ok(state, tr.pa, width as u32, true, false, pmp) {
		deliver_trap(state, mcause_val(exc_code::LD_ACCESS_FAULT, false), va);
		return 0;
	}

	let rs2_val = read_gpr(state, f.rs2);

	// 在通用 MMIO 退出之前先做 virtio-blk 检查.
	if let Some(_) = try_handle_virtio(tr.pa, true, rs2_val, width, dev) {
		return 4;
	}

	if is_device_addr(tr.pa, dev) {
		let info = StopInfo {
			reason: exit_reason::MMIO,
			hart_id: state.mhartid as u8,
			pc: state.pc,
			instr,
			..StopInfo::empty()
		};
		module.request_stop(info);
		return EXIT_SENTINEL;
	}

	let advance =
		handle_amo_concurrent(state, funct5, f.func3, rs2_val, f.rd, tr.pa, ctx, pmp, dev);

	if advance == EXIT_SENTINEL {
		let info = StopInfo {
			reason: exit_reason::MMIO,
			hart_id: state.mhartid as u8,
			pc: state.pc,
			instr,
			..StopInfo::empty()
		};
		module.request_stop(info);
	}
	advance
}
