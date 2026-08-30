use std::cell::Cell;
// use std::time::Instant;
use crate::atom_instr::handle_amo_concurrent_dispatch;
use crate::concurrent::{
	advance_clock_source, ConcurrentClintCtx, FfiExtIrqCtx, ModuleState, SharedDevCtx,
	SharedMemCtx, SharedPmpCtx, StopInfo,
};
use crate::decode::decode_fields;
use crate::fpu::{handle_fp_load_concurrent, handle_fp_store_concurrent};
use crate::handlers::{
	handle_compressed, lr_clear_all, pmp_ok, try_handle_virtio, ClintCtx, DevCtx, PmpCtx,
	EXIT_SENTINEL,
};
use crate::interrupt::{
	check_and_deliver_interrupt_concurrent,
	clint::{sync_msip, sync_mtip, try_handle_clint_concurrent},
	imsic_topei_peek, sync_ext_irq_mip, try_handle_imsic_concurrent,
	try_handle_imsic_read_concurrent,
	wfi::wfi_spin,
	IID_M_IPI, IID_S_IPI,
};
use crate::op_dispatcher::{
	handle_alu, handle_auipc, handle_br, handle_fence, handle_fp_fma, handle_fp_op, handle_jal,
	handle_jalr, handle_lui, handle_op32, handle_op_imm, handle_op_imm32,
};
use crate::peripheral::{
	is_device_addr,
	plic::{plic_sync_mip_if_legacy, try_handle_plic_concurrent},
	uart::try_handle_uart_concurrent,
};
use crate::state::{
	exit_reason, riscv_mode, FfiPlicCtx, FfiUartCtx, HartState, InstrToBeExec, MemCtx,
};
use crate::translate::{tlb_flush_all, tlb_mark_all_dirty, translate_va, TranslateFault, WalkCtx};
use crate::trap::{
	deliver_illegal_instruction, deliver_trap, exc_code, mcause_val, priv_ecall_concurrent,
	priv_mret_concurrent, priv_sret_concurrent, priv_wfi_concurrent,
};
use std::sync::atomic::{self, AtomicU32, AtomicU64, Ordering};

pub(crate) fn ram_offset(
	pa: u64,
	size: u32,
	ram_base: u64,
	ram_size: u64,
	shadow_base: u64,
	shadow_size: u64,
) -> Option<u64> {
	let end = pa + size as u64;
	if pa >= ram_base && end <= ram_base + ram_size {
		return Some(pa - ram_base);
	}
	if shadow_size > 0 {
		let sh_end = shadow_base + shadow_size;
		if pa >= shadow_base && end <= sh_end {
			return Some(pa - shadow_base);
		}
	}
	None
}

pub(crate) fn fetch_instr(ram: *const u8, ram_size: u64, ram_base: u64, pa: u64) -> Option<u32> {
	let offset = pa.wrapping_sub(ram_base);
	if offset > ram_size.saturating_sub(4) {
		return None;
	}
	let ptr = unsafe { ram.add(offset as usize) };
	// Acquire fence pairs with Release stores in ram_write_raw so this hart
	// observes instruction bytes written by another hart (e.g. store instruction
	// writing to a page that is later made executable).
	atomic::fence(Ordering::Acquire);
	let b0 = unsafe { *ptr } as u32;
	let b1 = unsafe { *ptr.add(1) } as u32;
	let b2 = unsafe { *ptr.add(2) } as u32;
	let b3 = unsafe { *ptr.add(3) } as u32;
	Some(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24))
}

pub(crate) fn fetch_instr_safe(
	state: &mut HartState,
	mem: &MemCtx,
	pa: u64,
	pc_before: u64,
	_hart_id: u8,
) -> Option<u32> {
	match fetch_instr(mem.ram, mem.ram_size, mem.ram_base, pa) {
		Some(w) => Some(w),
		None => {
			deliver_trap(
				state,
				mcause_val(exc_code::INSTR_ACCESS_FAULT, false),
				pc_before,
			);
			None
		}
	}
}

pub(crate) fn sext(val: u64, bits: u32) -> u64 {
	let half = 1u64 << (bits - 1);
	(val & (half - 1)).wrapping_sub(val & half) & u64::MAX
}

pub(crate) fn read_gpr(state: &HartState, rs: u8) -> u64 {
	if rs == 0 {
		0
	} else {
		state.gprs[rs as usize]
	}
}

pub(crate) fn write_gpr(state: &mut HartState, rd: u8, val: u64) {
	if rd != 0 {
		state.gprs[rd as usize] = val;
	}
}

pub(crate) fn check_bp_hit(pc: u64, breakpoints: &[u64], fetch_pa: Option<u64>) -> bool {
	if breakpoints.is_empty() {
		return false;
	}
	// Direct VA match — covers breakpoints set on virtual addresses.
	if breakpoints.iter().any(|&bp| bp == pc) {
		return true;
	}
	// PA match — covers breakpoints set on physical addresses when
	// the hart is running with MMU enabled (pc is VA, bp value is PA).
	// Matches Python ``_bp_match_pc`` (breakpoint.py:110-118).
	if let Some(pa) = fetch_pa {
		if pa != pc {
			return breakpoints.iter().any(|&bp| bp == pa);
		}
	}
	false
}

pub(crate) fn dispatch_concurrent(
	state: &mut HartState,
	f: &crate::decode::DecodedFields,
	instr: u32,
	ctx: &WalkCtx,
	hart_id: u8,
	pmp: &PmpCtx,
	dev: &DevCtx,
	clint: &ConcurrentClintCtx,
	uart: &FfiUartCtx,
	module: &ModuleState,
) -> u64 {
	match f.opcode {
		0b01100_11 => {
			// R-type ALU — pure compute, no shared state
			handle_alu(state, f, instr)
		}
		0b00100_11 => handle_op_imm(state, f, instr),
		0b00110_11 => handle_op_imm32(state, f, instr),
		0b01110_11 => handle_op32(state, f, instr),
		0b01101_11 => handle_lui(state, f),
		0b00101_11 => handle_auipc(state, f),
		0b11011_11 => handle_jal(state, f),
		0b11001_11 => handle_jalr(state, f),
		0b11000_11 => handle_br(state, f, instr),
		0b00011_11 => handle_fence(state, f, instr),

		// Loads / Stores — need CLINT inline handling for concurrent path
		0b00000_11 => handle_load_concurrent(state, f, instr, ctx, pmp, dev, clint, uart, module),
		0b01000_11 => handle_store_concurrent(state, f, instr, ctx, pmp, dev, clint, uart, module),

		// FP loads / stores (FLW/FLD/FSW/FSD)
		0b00001_11 => handle_fp_load_concurrent(state, f, instr, ctx, pmp, dev, module),
		0b01001_11 => handle_fp_store_concurrent(state, f, instr, ctx, pmp, dev, module),

		// System
		0b11100_11 => handle_system_concurrent(state, f, instr, ctx, hart_id, clint, pmp, module),

		// AMO — concurrent atomic path
		0b01011_11 => handle_amo_concurrent_dispatch(state, f, instr, ctx, pmp, dev, module),

		// F/D floating point (compute — OP-FP + FMA); pure compute like ALU
		0b10100_11 => handle_fp_op(state, f, instr),
		0b10000_11 | 0b10001_11 | 0b10010_11 | 0b10011_11 => handle_fp_fma(state, f, instr),

		_ => {
			deliver_illegal_instruction(state, instr as u64);
			0
		}
	}
}

pub(crate) fn read_ram_cross_page(
	state: &mut HartState,
	ctx: &WalkCtx,
	va: u64,
	pa_first: u64,
	size: u8,
	pmp: &PmpCtx,
) -> (u64, bool) {
	let page_off = va & 0xFFF;
	let bytes_first = (0x1000 - page_off) as u8;

	// Read bytes on the first page.
	let mut result: u64 = 0;
	for b in 0..bytes_first {
		let byte = ram_read_raw(ctx, pa_first + b as u64, 1);
		result |= byte << (b * 8);
	}

	// Translate the second VA page.
	let va2 = va.wrapping_add(bytes_first as u64);
	let tr2 = match translate_va(state, ctx, va2, false, false) {
		Ok(t) => t,
		Err(TranslateFault::PageFault(cause)) => {
			deliver_trap(state, mcause_val(cause, false), va2);
			return (0, false);
		}
		Err(TranslateFault::AccessFault) => {
			deliver_trap(state, mcause_val(exc_code::LD_ACCESS_FAULT, false), va2);
			return (0, false);
		}
	};

	// PMP check on the second page.
	let second_sz = (size - bytes_first) as u32;
	if !pmp_ok(state, tr2.pa, second_sz, false, false, pmp) {
		deliver_trap(state, mcause_val(exc_code::LD_ACCESS_FAULT, false), va);
		return (0, false);
	}

	// Read bytes on the second page.
	for b in bytes_first..size {
		let byte = ram_read_raw(ctx, tr2.pa.wrapping_add((b - bytes_first) as u64), 1);
		result |= byte << (b * 8);
	}

	(result, true)
}

pub(crate) fn write_ram_cross_page(
	state: &mut HartState,
	ctx: &WalkCtx,
	va: u64,
	pa_first: u64,
	val: u64,
	size: u8,
	pmp: &PmpCtx,
) -> bool {
	let page_off = va & 0xFFF;
	let bytes_first = (0x1000 - page_off) as u8;

	// Write bytes on the first page.
	for b in 0..bytes_first {
		let byte = (val >> (b * 8)) as u8;
		ram_write_raw(ctx, pa_first + b as u64, byte as u64, 1);
	}

	// Translate the second VA page.
	let va2 = va.wrapping_add(bytes_first as u64);
	let tr2 = match translate_va(state, ctx, va2, true, false) {
		Ok(t) => t,
		Err(TranslateFault::PageFault(cause)) => {
			deliver_trap(state, mcause_val(cause, false), va2);
			return false;
		}
		Err(TranslateFault::AccessFault) => {
			deliver_trap(state, mcause_val(exc_code::ST_ACCESS_FAULT, false), va2);
			return false;
		}
	};

	// PMP check on the second page.
	let second_sz = (size - bytes_first) as u32;
	if !pmp_ok(state, tr2.pa, second_sz, true, false, pmp) {
		deliver_trap(state, mcause_val(exc_code::ST_ACCESS_FAULT, false), va);
		return false;
	}

	// Write bytes on the second page.
	for b in bytes_first..size {
		let byte = (val >> (b * 8)) as u8;
		ram_write_raw(
			ctx,
			tr2.pa.wrapping_add((b - bytes_first) as u64),
			byte as u64,
			1,
		);
	}

	true
}

pub(crate) fn handle_load_concurrent(
	state: &mut HartState,
	f: &crate::decode::DecodedFields,
	instr: u32,
	ctx: &WalkCtx,
	pmp: &PmpCtx,
	dev: &DevCtx,
	clint: &ConcurrentClintCtx,
	uart: &FfiUartCtx,
	module: &ModuleState,
) -> u64 {
	let base = read_gpr(state, f.rs1);
	let va = base.wrapping_add(f.imm12_se);
	let (size, signed) = match f.func3 {
		0b000 => (1u8, true),
		0b001 => (2, true),
		0b010 => (4, true),
		0b011 => (8, false),
		0b100 => (1, false),
		0b101 => (2, false),
		0b110 => (4, false),
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

	// CLINT inline
	if let Some(data) = try_handle_clint_concurrent(tr.pa, false, 0, state, clint, module) {
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

	// virtio-blk inline
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

	// UART inline read — prevent TX FIFO polling from exiting speedup
	if let Some(data) = try_handle_uart_concurrent(tr.pa, false, 0, state.mhartid as u8, uart) {
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

	// IMSIC inline read — prevent MMIO exit for kernel IMSIC init reads.
	// All readable IMSIC MMIO registers return 0: seteipnum/clreipnum are
	// write-only, and configuration/topei registers are accessed via CSR.
	if let Some(data) = try_handle_imsic_read_concurrent(tr.pa, clint.num_harts) {
		write_gpr(state, f.rd, data);
		return 4;
	}

	// PLIC inline — kernel 直映射访问 PLIC (priority/pending/enable/threshold/claim)
	// 是 EXT4 recovery 阶段 batch 频繁退出的根因, 必须内联处理.
	// claim 清 pending 后内部立即重算 mip.MEIP/SEIP, 避免延后到 step_interrupts.
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

	// MMIO
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

	// Bounds check: ram_offset returns None for PA outside [ram_base, ram_base+ram_size).
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

	// Cross-page check must come BEFORE alignment check: an 8-byte access
	// at VA 0x1FF8 is naturally aligned but crosses a 4 KiB page boundary.
	// The two VA pages may map to non-consecutive PA pages, so we must
	// translate the second page independently and splice bytes.
	let val = if (va & 0xFFF) + size as u64 > 0x1000 {
		let (v, ok) = read_ram_cross_page(state, ctx, va, tr.pa, size, pmp);
		if !ok {
			return 0;
		}
		v
	} else if aligned {
		ram_read_raw(ctx, tr.pa, size)
	} else {
		// Misaligned within same physical page: simple byte-by-byte.
		let mut result: u64 = 0;
		for b in 0..size {
			let byte = ram_read_raw(ctx, tr.pa + b as u64, 1);
			result |= byte << (b * 8);
		}
		result
	};
	let result_val = if signed {
		match size {
			1 => sext(val, 8),
			2 => sext(val, 16),
			4 => sext(val, 32),
			_ => val,
		}
	} else {
		val
	};
	write_gpr(state, f.rd, result_val);
	4
}

pub(crate) fn handle_store_concurrent(
	state: &mut HartState,
	f: &crate::decode::DecodedFields,
	instr: u32,
	ctx: &WalkCtx,
	pmp: &PmpCtx,
	dev: &DevCtx,
	clint: &ConcurrentClintCtx,
	uart: &FfiUartCtx,
	module: &ModuleState,
) -> u64 {
	let base = read_gpr(state, f.rs1);
	let va = base.wrapping_add(f.imm_s);
	let size: u8 = match f.func3 {
		0b000 => 1,
		0b001 => 2,
		0b010 => 4,
		0b011 => 8,
		_ => {
			deliver_illegal_instruction(state, instr as u64);
			return 0;
		}
	};
	// Diagnostic: log when a small value (1..0xFF) is stored — catches
	// tag values like DT_RELA=7 being written to data structures.

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

	let val = read_gpr(state, f.rs2);

	// CLINT inline
	if let Some(_) = try_handle_clint_concurrent(tr.pa, true, val, state, clint, module) {
		return 4;
	}

	// UART inline — prevent sbi_printf from quiting the speedup execution
	if let Some(_) = try_handle_uart_concurrent(tr.pa, true, val, state.mhartid as u8, uart) {
		return 4;
	}

	// virtio-blk inline
	if let Some(_) = try_handle_virtio(tr.pa, true, val, size, dev) {
		return 4;
	}

	// IMSIC inline: route IPI identities through CLINT MSIP fast path
	if try_handle_imsic_concurrent(tr.pa, val, state, clint) {
		return 4;
	}

	// PLIC inline — priority/enable/threshold/complete 写内联处理; 写后立即
	// 重算 mip.MEIP/SEIP, 覆盖 "先挂起后使能" 的延迟投递与 claim 后电平重挂场景.
	if try_handle_plic_concurrent(tr.pa, true, val, size, state, dev.plic).is_some() {
		return 4;
	}

	// MMIO
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

	// Bounds check
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

	// Cross-page check must come BEFORE alignment check (see handle_load_concurrent).
	if (va & 0xFFF) + size as u64 > 0x1000 {
		if !write_ram_cross_page(state, ctx, va, tr.pa, val, size, pmp) {
			return 0;
		}
	} else if aligned {
		ram_write_raw(ctx, tr.pa, val, size);
	} else {
		// Misaligned within same physical page: simple byte-by-byte.
		for b in 0..size {
			let byte = (val >> (b * 8)) as u8;
			ram_write_raw(ctx, tr.pa + b as u64, byte as u64, 1);
		}
	}
	4
}

pub(crate) fn ram_read_raw(ctx: &WalkCtx, pa: u64, size: u8) -> u64 {
	let off = match ram_offset(
		pa,
		size as u32,
		ctx.ram_base,
		ctx.ram_size,
		ctx.shadow_base,
		ctx.shadow_size,
	) {
		Some(o) => o as usize,
		None => return 0,
	};
	let ptr = ctx.ram as *mut u8;
	match size {
		1 => {
			// Acquire fence before raw byte read: pairs with Release
			// fence (or atomic store) on other hart threads, ensuring
			// the byte value is visible across harts (real-hardware TSO).
			atomic::fence(Ordering::Acquire);
			unsafe { *ptr.add(off) as u64 }
		}
		2 => {
			atomic::fence(Ordering::Acquire);
			let b0 = unsafe { *ptr.add(off) } as u64;
			let b1 = unsafe { *ptr.add(off + 1) } as u64;
			b0 | (b1 << 8)
		}
		4 if (pa & 3) == 0 => {
			let a = unsafe { &*(ptr.add(off) as *const AtomicU32) };
			a.load(Ordering::Acquire) as u64
		}
		4 => {
			atomic::fence(Ordering::Acquire);
			let b0 = unsafe { *ptr.add(off) } as u64;
			let b1 = unsafe { *ptr.add(off + 1) } as u64;
			let b2 = unsafe { *ptr.add(off + 2) } as u64;
			let b3 = unsafe { *ptr.add(off + 3) } as u64;
			b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
		}
		8 if (pa & 7) == 0 => {
			let a = unsafe { &*(ptr.add(off) as *const AtomicU64) };
			let val = a.load(Ordering::Acquire);
			val
		}
		8 => {
			atomic::fence(Ordering::Acquire);
			let b0 = unsafe { *ptr.add(off) } as u64;
			let b1 = unsafe { *ptr.add(off + 1) } as u64;
			let b2 = unsafe { *ptr.add(off + 2) } as u64;
			let b3 = unsafe { *ptr.add(off + 3) } as u64;
			let b4 = unsafe { *ptr.add(off + 4) } as u64;
			let b5 = unsafe { *ptr.add(off + 5) } as u64;
			let b6 = unsafe { *ptr.add(off + 6) } as u64;
			let b7 = unsafe { *ptr.add(off + 7) } as u64;
			let val = b0
				| (b1 << 8) | (b2 << 16)
				| (b3 << 24) | (b4 << 32)
				| (b5 << 40) | (b6 << 48)
				| (b7 << 56);
			val
		}
		_ => 0,
	}
}

pub(crate) fn ram_write_raw(ctx: &WalkCtx, pa: u64, val: u64, size: u8) {
	let off = match ram_offset(
		pa,
		size as u32,
		ctx.ram_base,
		ctx.ram_size,
		ctx.shadow_base,
		ctx.shadow_size,
	) {
		Some(o) => o as usize,
		None => return,
	};
	// RISC-V spec §8.2: invalidate all LR reservations BEFORE the
	// store becomes visible.  With SeqCst ordering, observers see
	// the reservation clear before (or at the same time as) the new
	// value, closing the ABA window where reservation B sees old
	// value V -> store writes new value U -> store from third hart
	// writes back V -> SC on B succeeds (should have failed).
	lr_clear_all(ctx);
	atomic::fence(Ordering::SeqCst);

	let ptr = ctx.ram as *mut u8;
	match size {
		1 => {
			unsafe { *ptr.add(off) = val as u8 };
			atomic::fence(Ordering::Release);
		}
		2 => {
			unsafe { *ptr.add(off) = val as u8 };
			unsafe { *ptr.add(off + 1) = (val >> 8) as u8 };
			atomic::fence(Ordering::Release);
		}
		4 if (pa & 3) == 0 => {
			let a = unsafe { &*(ptr.add(off) as *const AtomicU32) };
			a.store(val as u32, Ordering::Release);
		}
		4 => {
			unsafe { *ptr.add(off) = val as u8 };
			unsafe { *ptr.add(off + 1) = (val >> 8) as u8 };
			unsafe { *ptr.add(off + 2) = (val >> 16) as u8 };
			unsafe { *ptr.add(off + 3) = (val >> 24) as u8 };
			atomic::fence(Ordering::Release);
		}
		8 if (pa & 7) == 0 => {
			let a = unsafe { &*(ptr.add(off) as *const AtomicU64) };
			a.store(val, Ordering::Release);
		}
		8 => {
			unsafe { *ptr.add(off) = val as u8 };
			unsafe { *ptr.add(off + 1) = (val >> 8) as u8 };
			unsafe { *ptr.add(off + 2) = (val >> 16) as u8 };
			unsafe { *ptr.add(off + 3) = (val >> 24) as u8 };
			unsafe { *ptr.add(off + 4) = (val >> 32) as u8 };
			unsafe { *ptr.add(off + 5) = (val >> 40) as u8 };
			unsafe { *ptr.add(off + 6) = (val >> 48) as u8 };
			unsafe { *ptr.add(off + 7) = (val >> 56) as u8 };
			atomic::fence(Ordering::Release);
		}
		_ => {}
	}
}

/// Handle CSR instructions (func3 ≠ 0b000) within the concurrent execution engine.
///
/// Builds a temporary serial ``ClintCtx`` from the concurrent one so that the
/// shared ``csr::handle_csr`` path remains unchanged.
fn handle_csr_concurrent(
	state: &mut HartState,
	f: &crate::decode::DecodedFields,
	instr: u32,
	hart_id: u8,
	clint: &ConcurrentClintCtx,
	pmp: &PmpCtx,
) -> u64 {
	let mut dummy = unsafe { std::mem::zeroed() };
	let mut csr_ctx = crate::csr::CsrContext::new(state, hart_id as u32, pmp, clint);
	let advance = crate::csr::handle_csr(
		&mut csr_ctx,
		f.rd,
		f.rs1,
		f.func12,
		f.func3,
		instr,
		&mut dummy,
	);

	advance
}

pub(crate) fn handle_system_concurrent(
	state: &mut HartState,
	f: &crate::decode::DecodedFields,
	instr: u32,
	_ctx: &WalkCtx,
	hart_id: u8,
	clint: &ConcurrentClintCtx,
	pmp: &PmpCtx,
	module: &ModuleState,
) -> u64 {
	if f.func3 != 0b000 {
		return handle_csr_concurrent(state, f, instr, hart_id, clint, pmp);
	}

	// func3 == 0b000: privileged instructions
	match f.func12 {
		0 => return priv_ecall_concurrent(state, instr, clint, module),
		1 => {
			// EBREAK — semihosting 或普通 breakpoint.
			// SYS_EXIT: 停机序列 — 一个 hart 执行即可停止整个引擎,
			// 状态保留在 live 数组中, 由上层
			// 消费 (调试器接管 / run() 返回). 裸 ebreak 保持 NOP.
			if state.gprs[10] == crate::handlers::SH_SYS_EXIT
				&& crate::handlers::semihosting_match(_ctx, state.pc)
			{
				module.request_stop(StopInfo {
					reason: exit_reason::EBREAK,
					hart_id,
					pc: state.pc,
					..StopInfo::empty()
				});
				return 0;
			}
			if let Some(advance) = crate::handlers::try_semihosting(state, _ctx, state.pc) {
				return advance;
			}
			return 4;
		}
		0x302 => return priv_mret_concurrent(state, instr),
		0x102 => return priv_sret_concurrent(state, instr, _ctx),
		0x105 => return priv_wfi_concurrent(state, instr),
		f_val if (f_val >> 5) == 0x09 => {
			// SFENCE.VMA — flush local TLB + broadcast to all harts.
			//
			// RISC-V SFENCE.VMA only flushes the local hart's TLB; remote
			// shootdown requires an IPI.  We implement broadcast semantics
			// via a global generation counter: every hart that detects a
			// generation mismatch flushes its own TLB at its next instruction
			// boundary.  This is legal because over-invalidation never breaks
			// correctness, and it closes the coherency window where hart A
			// writes a PTE + SFENCE.VMA but hart B's TLB still has the stale
			// entry.
			//
			// Ordering: flush the local TLB first, then increment the global
			// generation (Release).  On x86-64, ``fetch_add`` with Release is
			// ``lock xadd`` — a full hardware barrier that makes all prior
			// PTE stores globally visible before the generation change is
			// observed by other harts.  Paired with the Acquire load in the
			// Mark this hart's own TLB entries dirty; other harts
			// will detect the gen change and mark their own entries
			// dirty at the next instruction boundary.
			tlb_flush_all(&mut state.itlb);
			tlb_flush_all(&mut state.dtlb);
			let new_gen = module
				.tlb_gen
				.fetch_add(1, std::sync::atomic::Ordering::Release)
				.wrapping_add(1);
			module.tlb_gen_per_hart[hart_id as usize]
				.store(new_gen, std::sync::atomic::Ordering::Relaxed);
			return 4;
		}
		0x5A0 => {
			// MFENCE.DID
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
			return 4;
		}
		_ => {
			deliver_illegal_instruction(state, instr as u64);
			return 0;
		}
	}
}

pub(crate) fn exec_compressed_concurrent(
	state: &mut HartState,
	instr_word: u32,
	pc_before: u64,
	hart_id: u8,
	ctx: &WalkCtx,
	pmp: &PmpCtx,
	dev: &DevCtx,
	clint: &ConcurrentClintCtx,
	module: &ModuleState,
	breakpoints: &[u64],
) -> bool {
	let half = (instr_word & 0xFFFF) as u16;
	let mut instr_group: InstrToBeExec = unsafe { std::mem::zeroed() };
	let _serial_clint = ClintCtx {
		base: clint.base,
		mtime: clint.mtime as *mut u64,
		mtimecmp: clint.mtimecmp as *mut u64,
		msip: clint.msip as *mut u8,
		states: std::ptr::null_mut(),
		num_harts: clint.num_harts,
		timebase_hz: clint.timebase_hz,
		yield_for_ipi: Cell::new(false),
		ipi_sender_hart: Cell::new(0),
		ipi_sender_rounds: Cell::new(0),
		// Back-references for concurrent cross-thread IPI notification.
		// ``try_handle_imsic_serial`` uses these to wake the target hart
		// when an IMSIC seteipnum write arrives via a compressed store,
		// preventing TLB-shootdown deadlock in SMP AIA mode.
		msip_pending: Cell::new(clint.msip_pending.get()),
		hart_threads: Cell::new(clint.hart_threads.get()),
		hart_states: Cell::new(clint.hart_states.get()),
	};
	let advance = handle_compressed(
		state,
		half,
		instr_word,
		&mut instr_group,
		ctx,
		pmp,
		dev,
		&_serial_clint,
	);

	if advance == EXIT_SENTINEL {
		let reason = if instr_group.exit_reason != exit_reason::NORMAL {
			instr_group.exit_reason
		} else {
			exit_reason::ECALL
		};
		module.request_stop(StopInfo {
			reason,
			hart_id,
			pc: pc_before,
			instr: instr_word,
			..StopInfo::empty()
		});
		return false;
	}

	advance_pc(state, advance, pc_before);

	post_instr_checks(state, hart_id, module, breakpoints)
}

pub(crate) fn post_instr_checks(
	state: &mut HartState,
	hart_id: u8,
	module: &ModuleState,
	breakpoints: &[u64],
) -> bool {
	state.total_instrs = state.total_instrs.wrapping_add(1);

	if check_bp_hit(state.pc, breakpoints, None) {
		module.request_stop(StopInfo {
			reason: exit_reason::BREAKPOINT,
			hart_id,
			pc: state.pc,
			..StopInfo::empty()
		});
		return false;
	}

	true
}

/// Advance PC for a sequential instruction (``advance != 0`` and PC unchanged).
///
/// ``advance == 0`` means the instruction redirected PC (jump/trap/ecall/WFI),
/// so PC was already updated by the handler and must not be moved here.
#[inline]
pub(crate) fn advance_pc(state: &mut HartState, advance: u64, pc_before: u64) {
	if advance != 0 && state.pc == pc_before {
		state.pc = state.pc.wrapping_add(advance);
	}
}

pub(crate) fn translate_fetch_pc_concurrent(
	state: &mut HartState,
	ctx: &WalkCtx,
	pc: u64,
) -> Option<u64> {
	if state.mmu_mode == 0 || state.mode == riscv_mode::M || state.mode == riscv_mode::D {
		return Some(pc); // Bare mode / M-mode / D-mode: VA == PA
	}
	match translate_va(state, ctx, pc, false, true) {
		Ok(t) => Some(t.pa),
		Err(TranslateFault::PageFault(c)) => {
			deliver_trap(state, mcause_val(c, false), pc);
			None
		}
		Err(TranslateFault::AccessFault) => {
			deliver_trap(state, mcause_val(exc_code::INSTR_ACCESS_FAULT, false), pc);
			None
		}
	}
}

/// Fetch a 32-bit instruction word from physical memory.
///
/// When PC is within the last 2 bytes of a 4 KiB page (`page_offset >= 0xFFE`),
/// the 4-byte fetch crosses into the next virtual page.  The two pages may be
/// mapped to *non-consecutive* physical pages, so we translate the second
/// page independently and splice 2 bytes from each.
///
/// Returns `Some(word)` on success, or `None` if a trap was delivered (the
/// caller must handle the post-instruction boilerplate).
#[inline]
fn fetch_instr_word(
	state: &mut HartState,
	ctx: &WalkCtx,
	mem: &MemCtx,
	pc_before: u64,
	page_offset: u64,
	fetch_pa: u64,
	pmp: &PmpCtx,
	hart_id: u8,
) -> Option<u32> {
	if page_offset < 0xFFE {
		return fetch_instr_safe(state, mem, fetch_pa, pc_before, hart_id);
	}
	// 2 bytes on first page, 2 bytes on second page.
	let fetch_pa2 = translate_fetch_pc_concurrent(state, ctx, pc_before.wrapping_add(2))?;
	// PMP check on the second page as well.
	if !pmp_ok(state, fetch_pa2, 4, false, true, pmp) {
		deliver_trap(
			state,
			mcause_val(exc_code::INSTR_ACCESS_FAULT, false),
			pc_before,
		);
		return None;
	}
	// Read 2 bytes from each physical page and combine (little-endian).
	let lo = ram_read_raw(ctx, fetch_pa, 2) as u32;
	let hi = ram_read_raw(ctx, fetch_pa2, 2) as u32;
	return Some(lo | (hi << 16));
}

// ============================================================
//  hart_worker pipeline helpers
// ============================================================

/// Main-loop control flow sentinel.
enum Step {
	/// Advance to the next pipeline stage.
	Next,
	/// Jump back to the top of the main loop.
	Continue,
	/// Exit the worker function immediately.
	Exit,
}

/// Reconstruct local context structs from the `Send`-safe FFI wrappers.
fn make_contexts(
	mem: SharedMemCtx,
	pmp: SharedPmpCtx,
	dev: SharedDevCtx,
	module: &ModuleState,
) -> (MemCtx, PmpCtx, DevCtx, WalkCtx) {
	let mem_val = MemCtx {
		ram: mem.ram,
		ram_size: mem.ram_size,
		ram_base: mem.ram_base,
		shadow_base: mem.shadow_base,
		shadow_size: mem.shadow_size,
	};
	let pmp_val = PmpCtx {
		cfg: pmp.cfg,
		addr: pmp.addr,
		num: pmp.num,
	};
	let dev_val = DevCtx {
		bases: dev.bases,
		ends: dev.ends,
		num: dev.num,
		virtio_base: dev.virtio_base,
		virtio_raw: dev.virtio_raw,
		plic: dev.plic,
	};
	let walk = WalkCtx {
		ram: mem_val.ram,
		ram_size: mem_val.ram_size,
		ram_base: mem_val.ram_base,
		shadow_base: mem_val.shadow_base,
		shadow_size: mem_val.shadow_size,
		tlb_gen: &module.tlb_gen as *const AtomicU64,
		itlb_hand: Cell::new(0),
		dtlb_hand: Cell::new(0),
		lr_reserved: mem.lr_reserved,
		num_harts: module.wfi_flags.len() as u32,
	};
	(mem_val, pmp_val, dev_val, walk)
}

/// Mark this hart's TLB entries dirty if another hart executed SFENCE.VMA.
/// Dirty entries are re-walked on next access rather than flushed — this
/// preserves cached translations that are still valid.
fn mark_tlb_dirty_if_stale(state: &mut HartState, module: &ModuleState, hart_id: u8) {
	let global_gen = module.tlb_gen.load(Ordering::Acquire);
	let my_gen = module.tlb_gen_per_hart[hart_id as usize].load(Ordering::Relaxed);
	if my_gen == global_gen {
		return;
	}
	tlb_mark_all_dirty(&mut state.itlb);
	tlb_mark_all_dirty(&mut state.dtlb);
	module.tlb_gen_per_hart[hart_id as usize].store(global_gen, Ordering::Relaxed);
}

/// Handle WFI wait / wake.  Returns ``(Step, just_woke)``.
fn handle_wfi_state(
	state: &mut HartState,
	hart_id: u8,
	clint: &ConcurrentClintCtx,
	dev: &DevCtx,
	module: &ModuleState,
	stop_flag: *const u8,
	uart: &FfiUartCtx,
	ext_irq: *mut FfiExtIrqCtx,
) -> (Step, bool) {
	if state.waiting == 0 {
		return (Step::Next, false);
	}
	// Deferred Python-side work (e.g. virtio QueueNotify): exit so
	// Python gets a chance to process I/O before we re-enter WFI spin.
	if dev.has_pending_python_work() {
		module.request_stop(StopInfo {
			reason: exit_reason::WFI_WAIT,
			hart_id,
			pc: state.pc,
			..StopInfo::empty()
		});
		return (Step::Exit, false);
	}
	if !wfi_spin(
		state,
		hart_id as usize,
		clint,
		module,
		stop_flag,
		uart.rx_notify,
		ext_irq,
	) {
		// exit from WFI: flush TLB in case another hart did
		// SFENCE.VMA while we were spinning.
		mark_tlb_dirty_if_stale(state, module, hart_id);
		return (Step::Exit, false);
	}
	(Step::Next, true)
}

/// Synchronise hardware interrupt lines and deliver the highest-priority
/// pending interrupt.  Returns ``true`` when a trap was delivered (caller
/// should ``continue`` the main loop).
fn step_interrupts(
	state: &mut HartState,
	_hart_id: u8,
	clint: &ConcurrentClintCtx,
	just_woke_from_wfi: bool,
	plic: *mut FfiPlicCtx,
) -> bool {
	sync_mtip(state, clint);
	// When just woke from WFI, wfi_sync_and_check already called sync_msip
	// and auto-cleared the CLINT level bit.  Calling sync_msip again here
	// would see level=0 and clear mip.MSIP before check_and_deliver has a
	// chance to deliver the interrupt → TLB-shootdown deadlock.  We still
	// sync to catch re-sends (e.g. tlb_sync spinning), but we must not
	// lose the first MSIP.
	if just_woke_from_wfi {
		// Preserve the MSIP that wfi_sync_and_check set via msip_pending.
		let msip_saved = state.mip.load(Ordering::Acquire) & (1 << 3);
		sync_msip(state, clint);
		if msip_saved != 0 {
			state.mip.fetch_or(1 << 3, Ordering::AcqRel); // restore — don't let sync_msip clear it
		}
	} else {
		sync_msip(state, clint);
	}
	// IMSIC sync was moved out of the per-instruction hot path.
	// Python _native_sync_plic_mip + marshal_imsic already set the
	// correct mip bits at speedup execution start.
	// Within the accerlation, IMSIC state
	// only changes via CSR writes (stopei/mtopei claim, mireg/sireg),
	// and those paths update mip inline via sync_imsic_one().
	// Cross-hart IPIs (imsic_eip_set) also set the target's mip directly.
	let msip_was_pending = (state.mip.load(Ordering::Acquire) & (1 << 3)) != 0;
	if msip_was_pending && (state.mie & (1 << 3)) == 0 {
		state.mie |= 1 << 3;
	}
	// Clean up stale MEIP/SEIP before checking for pending interrupts.
	// sync_imsic was removed from the per-instruction hot path, so
	// mip.MEIP/SEIP can remain set after the IMSIC eip was claimed
	// (e.g. by a prior stopei write on THIS hart, or by a cross-hart
	// imsic_eip_clear).  Without cleanup, check_pending_interrupts sees
	// the stale SEIP, delivers a spurious SEI trap; the kernel reads
	// stopi (0xDB0) → compute_stopi → imsic_topei_peek correctly
	// reports 0 (no IMSIC external pending) → falls through to STIP
	// (timer), handles a pointless timer tick, and writes stopei which
	// finally clears SEIP via sync_imsic_one — the SEI→timer→sret→SEI
	// ping-pong that wastes millions of instructions per second.
	//
	// In AIA mode, ALL IMSIC interrupts — including software IPIs
	// (IID=1, IID=3) — route through the hart's external interrupt
	// lines (MEIP/SEIP).  We clear MEIP/SEIP ONLY when
	// imsic_topei_peek returns 0 (truly no IMSIC interrupt pending),
	// never for valid pending IPI IIDs — otherwise wake-from-WFI
	// sees MEIP, clears it for IID=3, and the IPI is silently lost
	// → SMP boot stalls until a timer rescues the hart.
	//
	// **When eidelivery == 0**, IMSIC does NOT own the MEIP/SEIP
	// lines — ext_irq drain (legacy device interrupts) or legacy PLIC
	// may have set them.  In that case imsic_topei_peek returns 0
	// (no IMSIC IPI pending), but clearing MEIP/SEIP would silently
	// lose the device interrupt.  This path is only valid when
	// IMSIC owns the line (eidelivery != 0).
	//
	// Only call imsic_topei_peek when the mip bit is already set
	// (fast-path: mip==0 → skip entirely, zero per-instruction cost).
	//
	// Two paths:
	// 1) eidelivery != 0: IMSIC owns the line — use topei_peek
	//    which checks IPI fast-path + eidelivery gate for externals.
	// 2) eidelivery == 0: IMSIC does NOT own the line (legacy PLIC
	//    or ext_irq drain).  topei_peek returns 0 for external
	//    interrupts, so we fall back to a raw eip scan: if any eip
	//    bit is set, the interrupt is still pending in IMSIC even
	//    though eidelivery=0 (the daemon set eip via APLIC→IMSIC
	//    before setting ext_irq).  Without this fallback a stale
	//    MEIP/SEIP from a previous ext_irq drain survives forever,
	//    creating an infinite SEI→handler→sret→SEI loop across
	//    speedup execution boundaries.
	if (state.mip.load(Ordering::Acquire) & (1 << 11)) != 0 && state.imsic_m.present != 0 {
		if state.imsic_m.eidelivery != 0 {
			let (val, _) = imsic_topei_peek(&state.imsic_m);
			if val == 0 {
				state.mip.fetch_and(!(1 << 11), Ordering::AcqRel);
			}
		} else {
			// eidelivery == 0: only IPI fast-path bits (IID=1,3) keep
			// MEIP alive (same rationale as the SEIP/S-file block below).
			let ipi_mask: u32 = (1 << IID_S_IPI) | (1 << IID_M_IPI);
			let has_ipi = (state.imsic_m.eip[0].load(Ordering::Acquire) & ipi_mask) != 0;
			if !has_ipi {
				state.mip.fetch_and(!(1 << 11), Ordering::AcqRel);
			}
		}
	}
	if (state.mip.load(Ordering::Acquire) & (1 << 9)) != 0 && state.imsic_s.present != 0 {
		if state.imsic_s.eidelivery != 0 {
			let (val, _) = imsic_topei_peek(&state.imsic_s);
			if val == 0 {
				state.mip.fetch_and(!(1 << 9), Ordering::AcqRel);
			}
		} else {
			// eidelivery == 0: only IPI fast-path bits (IID=1,3) keep
			// SEIP alive.  External interrupt eip bits (IID>=6) are NOT
			// visible through imsic_topei_peek / stopi when eidelivery=0 —
			// the kernel can never claim them, creating a spurious interrupt
			// ping-pong (SEI→stopi→0→sret→SEI).  External interrupts are
			// managed by the legacy path (ext_irq drain / PLIC).
			let ipi_mask: u32 = (1 << IID_S_IPI) | (1 << IID_M_IPI);
			let has_ipi = (state.imsic_s.eip[0].load(Ordering::Acquire) & ipi_mask) != 0;
			if !has_ipi {
				state.mip.fetch_and(!(1 << 9), Ordering::AcqRel);
			}
		}
	}
	// Force-enable MEIE/SEIE ONLY when IMSIC owns the external interrupt
	// path (AIA mode).  OpenSBI's AIA path may not explicitly set
	// mie.MEIE for IMSIC IPIs (IID=1/3 route through MEIP/SEIP), relying
	// on the IMSIC irqchip path (csr_swap MTOPEI) for dispatch.
	//
	// In legacy PLIC mode (imsic not present) this force-enable is FATAL:
	// OpenSBI's PLIC irqchip driver has NO process_hwirqs (only the AIA
	// IMSIC driver does), so sbi_irqchip_init never sets mie.MEIE and
	// sbi_irqchip_process() always returns SBI_ENODEV.  If we force-enable
	// MEIE behind the guest's back, a transient MEIP (e.g. ext_irq.pending
	// latched while UART RX data sat undrained) fires an M-mode external
	// trap into an unprocessable irqchip → sbi_trap_error → sbi_hart_hang
	// → the whole SMP boot freezes (observed: mcause=0x800000000000000b on
	// every hart right after the `riscv-plic: plic@c000000:` DT node).
	if state.imsic_m.present != 0
		&& (state.mip.load(Ordering::Acquire) & (1 << 11)) != 0
		&& (state.mie & (1 << 11)) == 0
	{
		state.mie |= 1 << 11;
	}
	if state.imsic_s.present != 0
		&& (state.mip.load(Ordering::Acquire) & (1 << 9)) != 0
		&& (state.mie & (1 << 9)) == 0
	{
		state.mie |= 1 << 9;
	}
	// Legacy PLIC 模式 (IMSIC 缺席) 的 MEIP/SEIP 对账: ext_irq 机制在 batch 内
	// 置位 MEIP/SEIP 后, 这里按 PLIC 真实仲裁结果 (pending + enable + prio>threshold)
	// 核对 — 覆盖 "device raise 早于 guest enable" 的虚假置位, 以及内联 claim 已清
	// pending 的场景 (claim/complete/enable 写路径已即时重算, 此处为兜底).  门控
	// ``mip ext bits != 0`` 保证零位时零开销; AIA 模式 (imsic present) 由上面的
	// IMSIC 清理块独占, 此处不介入.
	plic_sync_mip_if_legacy(state, plic);
	if check_and_deliver_interrupt_concurrent(state, clint) {
		return true;
	}
	false
}

/// Instruction fetch pipeline: translate VA, PMP execute check, read RAM,
/// breakpoint match.  Returns ``Ok(pc_before, instr_word)`` on success or
/// ``Err(step)`` when a trap was delivered / bp hit.
fn step_fetch_instr(
	state: &mut HartState,
	hart_id: u8,
	ctx: &WalkCtx,
	pmp: &PmpCtx,
	mem: &MemCtx,
	module: &ModuleState,
	breakpoints: &[u64],
) -> Result<(u64, u32), Step> {
	let pc_before = state.pc;
	let page_offset = pc_before & 0xFFF;

	// -- VA -> PA translation --
	let fetch_pa: u64 = match translate_fetch_pc_concurrent(state, ctx, pc_before) {
		Some(pa) => pa,
		None => {
			let step = if post_instr_checks(state, hart_id, module, breakpoints) {
				Step::Continue
			} else {
				Step::Exit
			};
			return Err(step);
		}
	};

	// -- PMP execute check --
	if !pmp_ok(state, fetch_pa, 4, false, true, pmp) {
		deliver_trap(
			state,
			mcause_val(exc_code::INSTR_ACCESS_FAULT, false),
			pc_before,
		);
		let step = if post_instr_checks(state, hart_id, module, breakpoints) {
			Step::Continue
		} else {
			Step::Exit
		};
		return Err(step);
	}

	// -- Read instruction word (cross-page aware) --
	let instr_word = match fetch_instr_word(
		state,
		ctx,
		mem,
		pc_before,
		page_offset,
		fetch_pa,
		pmp,
		hart_id,
	) {
		Some(w) => w,
		None => {
			let step = if post_instr_checks(state, hart_id, module, breakpoints) {
				Step::Continue
			} else {
				Step::Exit
			};
			return Err(step);
		}
	};

	// -- Breakpoint --
	if check_bp_hit(pc_before, breakpoints, Some(fetch_pa)) {
		module.request_stop(StopInfo {
			reason: exit_reason::BREAKPOINT,
			hart_id,
			pc: pc_before,
			instr: instr_word,
			..StopInfo::empty()
		});
		return Err(Step::Exit);
	}

	Ok((pc_before, instr_word))
}

// ============================================================
//  Main per-hart worker
// ============================================================

pub(crate) fn hart_worker(
	state: &mut HartState,
	hart_id: u8,
	mem: SharedMemCtx,
	pmp: SharedPmpCtx,
	dev: SharedDevCtx,
	clint: &ConcurrentClintCtx,
	uart: &FfiUartCtx,
	module: &ModuleState,
	breakpoints: &[u64],
	stop_flag: *const u8,
	ext_irq: *mut FfiExtIrqCtx,
) {
	let (_mem_val, _pmp_val, _dev_val, ctx) = make_contexts(mem, pmp, dev, module);
	let mem = &_mem_val;
	let pmp = &_pmp_val;
	let dev = &_dev_val;

	// HartState persists across FFI calls — stale TLB entries from a
	// previous speedup execution with epoch=0 would match the fresh ModuleState's
	// tlb_gen=0, producing incorrect VA->PA hits and memory corruption
	// (garbage inode metadata -> "Permission denied" / ENOTDIR in ext4).
	tlb_flush_all(&mut state.itlb);
	tlb_flush_all(&mut state.dtlb);

	loop {
		// ---- Guards ----
		if module.stop_flag.load(Ordering::Acquire) {
			return;
		}
		if !stop_flag.is_null() && unsafe { *stop_flag != 0 } {
			return;
		}
		// mtime 按每 hart 指令增量推进 (见 advance_clock_source, NS_PER_INSTR) —
		// 执行循环持续运行, 仅由外部信号 (Ctrl+Q -> stop_flag) 暂停.
		advance_clock_source(module, clint, state);
		// External interrupt (UART, VirtIO, …): daemon set pending after
		// injecting data into UART RX FIFO.  Raise SEIP/MEIP inline so the
		// guest's trap handler processes the interrupt within this acceleration.
		// 仅对未被 IMSIC 占用的线路置位 (eidelivery=1 时 IMSIC 独占, 见
		// sync_ext_irq_mip 注释); 无条件置位会造成 AIA 模式 ~22× 指令吞吐回归.
		sync_ext_irq_mip(state, ext_irq);
		// TermIO RX notification: stdin bytes arrived in ring buffer.
		// Force immediate exit -> Python drain_rx() moves data
		// from ring buffer into UART FIFO -> _native_sync_plic_mip
		// raises SEIP -> guest reads data on next acceleration without
		// waiting for PLIC round-trip or acceleration completion.
		// if !uart.rx_notify.is_null() && unsafe { *uart.rx_notify != 0 } {
		// 	module.stop_flag.store(true, Ordering::Release);
		//	return;
		// }
		if state.halted != 0 {
			// Halted harts exit the worker.  Spinning here would deadlock
			// the engine because the other harts may be waiting for this
			// one in all_in_wfi().
			module.wfi_flags[hart_id as usize].store(1, Ordering::Release);
			module.wfi_count.fetch_add(1, Ordering::Release);
			return;
		}

		// ---- WFI ----
		let (wfi_step, just_woke_from_wfi) =
			handle_wfi_state(state, hart_id, clint, dev, module, stop_flag, uart, ext_irq);
		match wfi_step {
			Step::Exit => return,
			Step::Continue => continue,
			Step::Next => {}
		}

		// ---- Interrupts ----
		if step_interrupts(state, hart_id, clint, just_woke_from_wfi, dev.plic) {
			continue;
		}

		// ---- TLB coherency (broadcast SFENCE.VMA) ----
		mark_tlb_dirty_if_stale(state, module, hart_id);

		// ---- Fetch ----
		let (pc_before, instr_word) =
			match step_fetch_instr(state, hart_id, &ctx, pmp, mem, module, breakpoints) {
				Ok(v) => v,
				Err(Step::Continue) => continue,
				Err(Step::Exit) => return,
				Err(Step::Next) => unreachable!(),
			};

		// ---- Decode & execute ----
		let f = decode_fields(instr_word);
		if f.is_compressed != 0 {
			if !exec_compressed_concurrent(
				state,
				instr_word,
				pc_before,
				hart_id,
				&ctx,
				pmp,
				dev,
				clint,
				module,
				breakpoints,
			) {
				return;
			}
			continue;
		}
		let advance = dispatch_concurrent(
			state, &f, instr_word, &ctx, hart_id, pmp, dev, clint, uart, module,
		);
		if advance == EXIT_SENTINEL {
			if module.stop_flag.load(Ordering::Acquire) {
				return;
			}
			module.request_stop(StopInfo {
				reason: exit_reason::MMIO,
				hart_id,
				pc: pc_before,
				instr: instr_word,
				..StopInfo::empty()
			});
			return;
		}

		advance_pc(state, advance, pc_before);

		if !post_instr_checks(state, hart_id, module, breakpoints) {
			return;
		}
	}
}

// ============================================================
//  Tests
// ============================================================

#[cfg(test)]
mod tests {
	use super::*;
	use crate::concurrent::{ConcurrentClintCtx, ModuleState, StopInfo, NS_PER_INSTR};
	use crate::handlers::{DevCtx, PmpCtx};
	use crate::state::{riscv_mode, FfiUartCtx, HartState, TlbEntry};
	use crate::translate::WalkCtx;
	use std::sync::atomic::{AtomicU64, AtomicU8};

	/// Build a permissive PMP context: single TOR entry covering all of
	/// memory with R/W/X.  Needed because pmp_ok per RISC-V spec §3.7.1
	/// denies S/U access when num_entries==0.
	fn make_permissive_pmp(cfg: &mut [u8], addr: &mut [u64]) -> PmpCtx {
		// TOR entry: covers [0, addr[0]) = [0, u64::MAX) — entire address space.
		// PMP_R(1) | PMP_W(2) | PMP_X(4) | PMP_A_TOR(8)
		cfg[0] = 0x0F;
		addr[0] = u64::MAX;
		PmpCtx {
			cfg: cfg.as_mut_ptr(),
			addr: addr.as_mut_ptr(),
			num: 1,
		}
	}

	/// Build a minimal 4K-page Sv39 page table inside RAM.
	/// Places L2 at PA 0x5000, L1 at 0x6000, L0 at 0x7000.
	/// Returns the `satp` value (Sv39 mode, root PPN = 5).
	fn setup_two_pages(
		ram: &mut [u8],
		va_a: u64,
		pa_a: u64,
		va_b: u64,
		pa_b: u64,
		perm: u64,
	) -> u64 {
		let vpn_a = crate::mmu::sv39_decompose_va(va_a);
		let vpn_b = crate::mmu::sv39_decompose_va(va_b);
		// Both pages must share the same L2/L1 path (same 2 MiB region).
		assert_eq!(vpn_a.vpn2, vpn_b.vpn2);
		assert_eq!(vpn_a.vpn1, vpn_b.vpn1);

		let satp = 8u64 << 60 | 5; // Sv39, root PPN = 5

		// L2 (PA 0x5000): pointer to L1 at PPN=6
		let l2_off = 0x5000usize + (vpn_a.vpn2 * 8) as usize;
		let l2_val = (6u64 << 10) | crate::translate::PTE_V;
		ram[l2_off..l2_off + 8].copy_from_slice(&l2_val.to_le_bytes());

		// L1 (PA 0x6000): pointer to L0 at PPN=7
		let l1_off = 0x6000usize + (vpn_a.vpn1 * 8) as usize;
		let l1_val = (7u64 << 10) | crate::translate::PTE_V;
		ram[l1_off..l1_off + 8].copy_from_slice(&l1_val.to_le_bytes());

		// L0 (PA 0x7000): two leaf entries
		let ppn_a = pa_a >> 12;
		let l0_off_a = 0x7000usize + (vpn_a.vpn0 * 8) as usize;
		let l0_val_a = (ppn_a << 10) | perm | crate::translate::PTE_V;
		ram[l0_off_a..l0_off_a + 8].copy_from_slice(&l0_val_a.to_le_bytes());

		let ppn_b = pa_b >> 12;
		let l0_off_b = 0x7000usize + (vpn_b.vpn0 * 8) as usize;
		let l0_val_b = (ppn_b << 10) | perm | crate::translate::PTE_V;
		ram[l0_off_b..l0_off_b + 8].copy_from_slice(&l0_val_b.to_le_bytes());

		satp
	}

	/// Regression: 8-byte read crossing a 4 KiB page boundary must splice
	/// bytes from two independently-translated PA pages, even when the
	/// VA is naturally aligned (the aligned path must NOT be taken for
	/// cross-page accesses).
	///
	/// Maps VA 0x1000->PA 0x1000 and VA 0x2000->PA 0x3000 (non-consecutive,
	/// skipping PA 0x2000).  A misaligned 8-byte read at VA 0x1FFC (aligned
	/// to 4 but not to 8, crossing into VA 0x2000) must return the bytes from
	/// PA 0x1FFC-0x1FFF concatenated with PA 0x3000-0x3003.
	#[test]
	fn cross_page_read_splices_non_consecutive_pa() {
		const RAM_SIZE: usize = 0x8000;
		let mut ram = vec![0u8; RAM_SIZE];

		// Page A (VA 0x1000): PA 0x1000
		// Page B (VA 0x2000): PA 0x3000 (non-consecutive — gap at PA 0x2000)
		let satp = setup_two_pages(
			&mut ram,
			0x1000,
			0x1000,
			0x2000,
			0x3000,
			crate::translate::PTE_R | crate::translate::PTE_W | crate::translate::PTE_U,
		);

		// Write pattern: 0xAA bytes at end of first PA page, 0xBB at start of second.
		// VA 0x1FFC ->PA 0x1FFC (bytes_first = 4, on page A)
		// VA 0x2000 ->PA 0x3000 (bytes 4-7 of the 8-byte read, on page B)
		for i in 0x1FFC..0x2000 {
			ram[i] = 0xAA;
		}
		for i in 0x3000..0x3004 {
			ram[i] = 0xBB;
		}

		let mut state: HartState = unsafe { std::mem::zeroed() };
		state.mode = riscv_mode::U; // U-mode ->page walk + PMP bypass (num=0)
		state.mmu_mode = 8; // Sv39
		state.satp = satp;
		// Init dtlb as invalid to force page walk.
		for e in state.dtlb.iter_mut() {
			*e = TlbEntry::empty();
		}

		let ctx = WalkCtx {
			ram: ram.as_mut_ptr(),
			ram_size: RAM_SIZE as u64,
			ram_base: 0,
			shadow_base: 0,
			shadow_size: 0,
			tlb_gen: std::ptr::null(),
			itlb_hand: Cell::new(0),
			dtlb_hand: Cell::new(0),
			lr_reserved: std::ptr::null_mut(),
			num_harts: 1,
		};
		// PMP with zero entries — no restrictions (pmp_ok short-circuits on num==0).
		let mut _pmp_cfg = vec![0u8; 64];
		let mut _pmp_addr = vec![0u64; 64];
		let pmp = make_permissive_pmp(&mut _pmp_cfg, &mut _pmp_addr);

		// 8-byte read at VA 0x1FFC — crosses page boundary.
		let (val, ok) = read_ram_cross_page(&mut state, &ctx, 0x1FFC, 0x1FFC, 8, &pmp);
		assert!(ok, "cross-page read must succeed");

		// Expected: little-endian assembly of 4 bytes 0xAA + 4 bytes 0xBB.
		let expected: u64 = 0xBBBB_BBBB_AAAA_AAAA;
		assert_eq!(
			val, expected,
			"cross-page read: expected 0x{expected:016X}, got 0x{val:016X}\
             \n  (4 bytes from PA 0x1FFC + 4 bytes from PA 0x3000)"
		);
	}

	/// Store variant: 8-byte write crossing a page boundary must scatter
	/// bytes to two independently-translated PA pages.
	#[test]
	fn cross_page_write_scatters_non_consecutive_pa() {
		const RAM_SIZE: usize = 0x8000;
		let mut ram = vec![0u8; RAM_SIZE];

		let satp = setup_two_pages(
			&mut ram,
			0x1000,
			0x1000,
			0x2000,
			0x3000,
			crate::translate::PTE_R | crate::translate::PTE_W | crate::translate::PTE_U,
		);

		let mut state: HartState = unsafe { std::mem::zeroed() };
		state.mode = riscv_mode::U;
		state.mmu_mode = 8;
		state.satp = satp;
		for e in state.dtlb.iter_mut() {
			*e = TlbEntry::empty();
		}

		let ctx = WalkCtx {
			ram: ram.as_mut_ptr(),
			ram_size: RAM_SIZE as u64,
			ram_base: 0,
			shadow_base: 0,
			shadow_size: 0,
			tlb_gen: std::ptr::null(),
			itlb_hand: Cell::new(0),
			dtlb_hand: Cell::new(0),
			lr_reserved: std::ptr::null_mut(),
			num_harts: 1,
		};
		let mut _pmp_cfg = vec![0u8; 64];
		let mut _pmp_addr = vec![0u64; 64];
		let pmp = make_permissive_pmp(&mut _pmp_cfg, &mut _pmp_addr);

		// Write 0xCCCCCCCC_DDDDDDDD at VA 0x1FFC (4 bytes to each page).
		let val: u64 = 0xCCCC_CCCC_DDDD_DDDD;
		let ok = write_ram_cross_page(&mut state, &ctx, 0x1FFC, 0x1FFC, val, 8, &pmp);
		assert!(ok, "cross-page write must succeed");

		// Lower 4 bytes (0xDDDDDDDD) go to PA 0x1FFC-0x1FFF (page A).
		assert_eq!(&ram[0x1FFC..0x2000], &[0xDD, 0xDD, 0xDD, 0xDD]);
		// Upper 4 bytes (0xCCCCCCCC) go to PA 0x3000-0x3003 (page B, non-consecutive).
		assert_eq!(&ram[0x3000..0x3004], &[0xCC, 0xCC, 0xCC, 0xCC]);
		// PA 0x2000-0x2FFF must remain untouched.
		assert!(
			ram[0x2000..0x3000].iter().all(|&b| b == 0),
			"gap between mapped pages must not be written"
		);
	}

	// ============================================================
	//  rd == rs1  regression tests
	// ============================================================

	/// A load where rd==rs1 must read the base register BEFORE writing the
	/// loaded value.  Regression: ``lhu a5, 0x336(a5)`` crashes at VA 0x33D
	/// when a5=7 because the old value is lost before address computation.
	#[test]
	fn load_rd_equals_rs1_uses_old_value_for_address() {
		let mut state: HartState = unsafe { std::mem::zeroed() };
		state.mode = riscv_mode::M;
		state.gprs[15] = 0x100; // a5 = low address within RAM
		state.mmu_mode = 0; // Bare mode — VA==PA

		let mut ram = vec![0xCDu8; 0x1000];
		let ctx = WalkCtx {
			ram: ram.as_mut_ptr(),
			ram_size: 0x1000,
			ram_base: 0,
			shadow_base: 0,
			shadow_size: 0,
			tlb_gen: std::ptr::null(),
			itlb_hand: Cell::new(0),
			dtlb_hand: Cell::new(0),
			lr_reserved: std::ptr::null_mut(),
			num_harts: 1,
		};
		let mut _pmp_cfg = vec![0u8; 1];
		let mut _pmp_addr = vec![0u64; 1];
		let pmp = make_permissive_pmp(&mut _pmp_cfg, &mut _pmp_addr);
		let dev = DevCtx {
			bases: std::ptr::null(),
			ends: std::ptr::null(),
			num: 0,
			virtio_base: 0,
			virtio_raw: std::ptr::null_mut(),
				plic: std::ptr::null_mut(),
		};
		// Build minimal inline CLINT/UART contexts for the load handler.
		let _mtime = std::sync::atomic::AtomicU64::new(0);
		let _mtimecmp = std::sync::atomic::AtomicU64::new(0);
		let _msip = std::sync::atomic::AtomicU8::new(0);
		let clint = ConcurrentClintCtx {
			base: 0,
			mtime: &_mtime,
			mtimecmp: &_mtimecmp,
			msip: &_msip,
			num_harts: 1,
			msip_pending: Cell::new(std::ptr::null()),
			hart_threads: Cell::new(std::ptr::null()),
			hart_states: Cell::new(std::ptr::null()),
			timebase_hz: 0, // 单元测试禁用 clock-source 推进
		};
		let _tx_buf = [0u8; 16];
		let _tx_wr = std::sync::atomic::AtomicU32::new(0);
		let uart = FfiUartCtx {
			base: 0,
			tx_buf: _tx_buf.as_ptr() as *mut u8,
			tx_cap: 16,
			tx_wr: &_tx_wr as *const std::sync::atomic::AtomicU32 as *mut u32,
			ie: 0,
			txctrl: 0,
			rxctrl: 0,
			rx_fifo_len: 0,
			tx_notify_fd: -1,
			no_stdout: 0,
			rx_notify: std::ptr::null_mut(),
		};
		let _wfi_flags: Vec<std::sync::atomic::AtomicU8> = (0..1)
			.map(|_| std::sync::atomic::AtomicU8::new(0))
			.collect();
		let _tlb_gen: Vec<std::sync::atomic::AtomicU64> = (0..1)
			.map(|_| std::sync::atomic::AtomicU64::new(0))
			.collect();
		let _msip_pending: Vec<std::sync::atomic::AtomicU64> = (0..1)
			.map(|_| std::sync::atomic::AtomicU64::new(0))
			.collect();
		let module = ModuleState {
			stop_flag: std::sync::atomic::AtomicBool::new(false),
			stop_info: std::sync::Mutex::new(StopInfo::empty()),
			wfi_count: std::sync::atomic::AtomicU32::new(0),
			wfi_flags: _wfi_flags.into_boxed_slice(),
			active_hart_num: 1,
			tlb_gen: std::sync::atomic::AtomicU64::new(0),
			tlb_gen_per_hart: _tlb_gen.into_boxed_slice(),
			lr_reserved: Box::new([]),
			msip_pending: _msip_pending.into_boxed_slice(),
			st_time_val: std::time::Instant::now(),
			time_base_val: 0,
			instr_ref_num: Box::new([]),
		};
		// opcode=0000011, rd=15, func3=101(LHU), rs1=15, imm=0x336
		let instr: u32 = 0x3367d783u32; // lhu a5, 0x336(a5)
		let f = crate::decode::decode_fields(instr);

		// Write known data at the target address (0x100 + 0x336 = 0x436)
		let target_pa = 0x100u64 + 0x336u64;
		ram[target_pa as usize] = 0x42;
		ram[target_pa as usize + 1] = 0x13;

		let adv = handle_load_concurrent(
			&mut state, &f, instr, &ctx, &pmp, &dev, &clint, &uart, &module,
		);
		assert_eq!(adv, 4);
		// After lhu: a5 should be 0x1342 (little-endian: 0x42 | 0x13<<8)
		assert_eq!(
			state.gprs[15], 0x1342,
			"LHU rd==rs1: loaded value must be 0x1342, not the old address"
		);
	}

	/// When a load with rd==rs1 triggers a page fault, the destination
	/// register must NOT be overwritten.  The kernel's trap handler relies
	/// on seeing the original register state.
	#[test]
	fn load_rd_equals_rs1_preserves_rd_on_pagefault() {
		let mut state: HartState = unsafe { std::mem::zeroed() };
		state.mode = riscv_mode::U; // U-mode ->Sv39 active
		state.mmu_mode = 8; // Sv39
		state.satp = 8u64 << 60; // Sv39, root PPN=0
		state.stvec = 0x80000400; // S-mode trap vector
		state.medeleg = 1 << 13; // Delegate LdPageFault to S-mode
		state.gprs[15] = 7; // a5 = 7 (the DT_RELA crash value)
					  // Empty RAM ->any page walk will fail (PTE.V=0)
		let mut ram = vec![0u8; 0x1000];
		let ctx = WalkCtx {
			ram: ram.as_mut_ptr(),
			ram_size: 0x1000,
			ram_base: 0,
			shadow_base: 0,
			shadow_size: 0,
			tlb_gen: std::ptr::null(),
			itlb_hand: Cell::new(0),
			dtlb_hand: Cell::new(0),
			lr_reserved: std::ptr::null_mut(),
			num_harts: 1,
		};
		let mut _pmp_cfg = vec![0u8; 1];
		let mut _pmp_addr = vec![0u64; 1];
		let pmp = make_permissive_pmp(&mut _pmp_cfg, &mut _pmp_addr);
		let dev = DevCtx {
			bases: std::ptr::null(),
			ends: std::ptr::null(),
			num: 0,
			virtio_base: 0,
			virtio_raw: std::ptr::null_mut(),
				plic: std::ptr::null_mut(),
		};
		let _mtime = std::sync::atomic::AtomicU64::new(0);
		let _mtimecmp = std::sync::atomic::AtomicU64::new(0);
		let _msip = std::sync::atomic::AtomicU8::new(0);
		let clint = ConcurrentClintCtx {
			base: 0,
			mtime: &_mtime,
			mtimecmp: &_mtimecmp,
			msip: &_msip,
			num_harts: 1,
			msip_pending: Cell::new(std::ptr::null()),
			hart_threads: Cell::new(std::ptr::null()),
			hart_states: Cell::new(std::ptr::null()),
			timebase_hz: 0, // 单元测试禁用 clock-source 推进
		};
		let _tx_buf = [0u8; 16];
		let _tx_wr = std::sync::atomic::AtomicU32::new(0);
		let uart = FfiUartCtx {
			base: 0,
			tx_buf: _tx_buf.as_ptr() as *mut u8,
			tx_cap: 16,
			tx_wr: &_tx_wr as *const std::sync::atomic::AtomicU32 as *mut u32,
			ie: 0,
			txctrl: 0,
			rxctrl: 0,
			rx_fifo_len: 0,
			tx_notify_fd: -1,
			no_stdout: 0,
			rx_notify: std::ptr::null_mut(),
		};
		let _wfi_flags: Vec<std::sync::atomic::AtomicU8> = (0..1)
			.map(|_| std::sync::atomic::AtomicU8::new(0))
			.collect();
		let _tlb_gen: Vec<std::sync::atomic::AtomicU64> = (0..1)
			.map(|_| std::sync::atomic::AtomicU64::new(0))
			.collect();
		let _msip_pending: Vec<std::sync::atomic::AtomicU64> = (0..1)
			.map(|_| std::sync::atomic::AtomicU64::new(0))
			.collect();
		let module = ModuleState {
			stop_flag: std::sync::atomic::AtomicBool::new(false),
			stop_info: std::sync::Mutex::new(StopInfo::empty()),
			wfi_count: std::sync::atomic::AtomicU32::new(0),
			wfi_flags: _wfi_flags.into_boxed_slice(),
			active_hart_num: 1,
			tlb_gen: std::sync::atomic::AtomicU64::new(0),
			tlb_gen_per_hart: _tlb_gen.into_boxed_slice(),
			lr_reserved: Box::new([]),
			msip_pending: _msip_pending.into_boxed_slice(),
			st_time_val: std::time::Instant::now(),
			time_base_val: 0,
			instr_ref_num: Box::new([]),
		};

		let instr: u32 = 0x3367d783u32; // lhu a5, 0x336(a5)
		let f = crate::decode::decode_fields(instr);

		let adv = handle_load_concurrent(
			&mut state, &f, instr, &ctx, &pmp, &dev, &clint, &uart, &module,
		);
		// Page fault delivered, PC redirected ->advance = 0
		assert_eq!(adv, 0, "page fault must return 0 (PC already redirected)");
		// a5 MUST retain its original value (7), NOT be overwritten
		assert_eq!(
			state.gprs[15], 7,
			"rd==rs1 on page fault: a5 must stay 7, not be corrupted to 0 or loaded data"
		);
		// PC must have changed (to stvec)
		assert_ne!(
			state.pc, 0,
			"PC must be redirected to stvec after page fault"
		);
	}

	/// CSR with rd==rs1: old CSR value read before rs1, then written to rd.
	/// Test CSRRW where rd=rs1 — the old CSR value is swapped into rd.
	#[test]
	fn csrrw_rd_equals_rs1_swaps_correctly() {
		let mut state: HartState = unsafe { std::mem::zeroed() };
		state.mode = riscv_mode::M;
		state.gprs[15] = 0xDEADBEEF; // a5 = value to write to CSR
		state.stvec = 0x80001000; // stvec = old value

		let mut result = unsafe { std::mem::zeroed() };
		let pmp = PmpCtx {
			cfg: std::ptr::null_mut(),
			addr: std::ptr::null_mut(),
			num: 0,
		};
		let clint = ConcurrentClintCtx {
			base: 0,
			mtime: std::ptr::null(),
			mtimecmp: std::ptr::null(),
			msip: std::ptr::null(),
			num_harts: 1,
			msip_pending: std::cell::Cell::new(std::ptr::null()),
			hart_threads: std::cell::Cell::new(std::ptr::null()),
			hart_states: std::cell::Cell::new(std::ptr::null()),
			timebase_hz: 0, // 单元测试禁用 clock-source 推进
		};
		let mut csr_ctx = crate::csr::CsrContext::new(&mut state, 0, &pmp, &clint);
		// CSRRW a5, stvec, a5  -> funct3=001, rd=15, rs1=15, csr=0x105
		let adv = crate::csr::handle_csr(&mut csr_ctx, 15, 15, 0x105, 1, 0, &mut result);
		assert_eq!(adv, 4);
		// a5 should now hold old stvec (0x80001000), not 0xDEADBEEF
		assert_eq!(
			csr_ctx.state.gprs[15], 0x80001000,
			"CSRRW rd==rs1: a5 must swap to old CSR value"
		);
		// stvec should now hold the old a5 value (0xDEADBEEF)
		assert_eq!(
			csr_ctx.state.stvec, 0xDEADBEEF,
			"CSRRW rd==rs1: CSR must get old a5 value"
		);
	}

	/// CSRRS with rd==rs1: rs1 bits are set in CSR, old CSR value ->rd.
	#[test]
	fn csrrs_rd_equals_rs1_sets_bits_correctly() {
		let mut state: HartState = unsafe { std::mem::zeroed() };
		state.mode = riscv_mode::M;
		state.gprs[10] = 0x0000000F; // a0 = mask to set
		state.mie = 0x00000088; // MIE currently has MTIE+MSIE set

		let mut result = unsafe { std::mem::zeroed() };
		let pmp = PmpCtx {
			cfg: std::ptr::null_mut(),
			addr: std::ptr::null_mut(),
			num: 0,
		};
		let clint = ConcurrentClintCtx {
			base: 0,
			mtime: std::ptr::null(),
			mtimecmp: std::ptr::null(),
			msip: std::ptr::null(),
			num_harts: 1,
			msip_pending: std::cell::Cell::new(std::ptr::null()),
			hart_threads: std::cell::Cell::new(std::ptr::null()),
			hart_states: std::cell::Cell::new(std::ptr::null()),
			timebase_hz: 0, // 单元测试禁用 clock-source 推进
		};
		let mut csr_ctx = crate::csr::CsrContext::new(&mut state, 0, &pmp, &clint);
		// CSRRS a0, mie, a0  -> funct3=010, rd=10, rs1=10, csr=0x304
		let adv = crate::csr::handle_csr(&mut csr_ctx, 10, 10, 0x304, 2, 0, &mut result);
		assert_eq!(adv, 4);
		// a0 should hold old MIE value (0x88), not the mask
		assert_eq!(
			csr_ctx.state.gprs[10], 0x88,
			"CSRRS rd==rs1: a0 must hold old MIE, not 0x0F"
		);
		// MIE should now be 0x88 | 0x0F = 0x8F
		assert_eq!(
			csr_ctx.state.mie, 0x8F,
			"CSRRS rd==rs1: MIE must have 0x0F bits set"
		);
	}

	/// 回归: 内核 rdtime 惯用法 ``csrrs rd, time, x0`` (S 模式, mcounteren.TM=1)
	/// 必须完整在一轮加速中完成而不退出到 Python. 修复前 csr_write 无 time 分支,
	/// 写回原值落入 ``_ => CSR_EXIT`` → 每条 rdtime 所引发的退出 (~1.4ms/条)
	/// → 内核吞吐崩溃 → STI 处理尾部长于 tick 周期 → 永久 STI 活锁,
	/// 启动阻塞在 vgaarb: loaded.
	#[test]
	fn csrrs_time_rdtime_stays_in_acceleration() {
		let mut state: HartState = unsafe { std::mem::zeroed() };
		state.mode = riscv_mode::S;
		state.mcounteren = 0b111; // 允许 S 模式读 cycle (bit 0) / time (bit 1) / instret (bit 2)
		let mtime_atomic = AtomicU64::new(0x1234_5678_9ABC_DEF0);

		let mut result = unsafe { std::mem::zeroed() };
		let pmp = PmpCtx {
			cfg: std::ptr::null_mut(),
			addr: std::ptr::null_mut(),
			num: 0,
		};
		let clint = ConcurrentClintCtx {
			base: 0,
			mtime: &mtime_atomic as *const AtomicU64,
			mtimecmp: std::ptr::null(),
			msip: std::ptr::null(),
			num_harts: 1,
			msip_pending: std::cell::Cell::new(std::ptr::null()),
			hart_threads: std::cell::Cell::new(std::ptr::null()),
			hart_states: std::cell::Cell::new(std::ptr::null()),
			timebase_hz: 0, // 单元测试禁用 clock-source 推进
		};
		let mut csr_ctx = crate::csr::CsrContext::new(&mut state, 0, &pmp, &clint);

		// csrrs a0, time, x0 — funct3=010, rd=10, rs1=0, csr=0xC01
		let adv = crate::csr::handle_csr(&mut csr_ctx, 10, 0, 0xC01, 2, 0xc010_2573, &mut result);
		assert_eq!(
			adv, 4,
			"rdtime 必须在一轮加速中完成 (advance=4), 修复前返回 EXIT_SENTINEL"
		);
		assert_eq!(
			result.exit_reason, 0,
			"rdtime 不得设置退出原因, 修复前为 ECALL"
		);
		assert_eq!(
			csr_ctx.state.gprs[10], 0x1234_5678_9ABC_DEF0,
			"a0 必须读到 CLINT 实时 mtime"
		);

		// timeh (0xC81): csrrs a1, timeh, x0 — 高 32 位
		result.exit_reason = 0;
		let adv = crate::csr::handle_csr(&mut csr_ctx, 11, 0, 0xC81, 2, 0xc810_25f3, &mut result);
		assert_eq!(adv, 4, "rdtimeh 必须一轮加速中完成");
		assert_eq!(result.exit_reason, 0, "rdtimeh 不得设置退出原因");
		assert_eq!(
			csr_ctx.state.gprs[11], 0x1234_5678,
			"a1 必须读到 mtime 高 32 位"
		);

		// cycle (0xC00): csrrs a2, cycle, x0 — 同样只读, 写回忽略
		result.exit_reason = 0;
		let adv = crate::csr::handle_csr(&mut csr_ctx, 12, 0, 0xC00, 2, 0xc000_2673, &mut result);
		assert_eq!(adv, 4, "rdcycle 必须一轮加速中完成");
		assert_eq!(result.exit_reason, 0, "rdcycle 不得设置退出原因");
	}

	/// 构造最小 CLINT 上下文 (timer/IPI 全关, 供中断边界测试使用).
	fn clint_ctx_empty() -> ConcurrentClintCtx {
		let _mtime = Box::new(std::sync::atomic::AtomicU64::new(0));
		let _mtimecmp = Box::new(std::sync::atomic::AtomicU64::new(0));
		let _msip = Box::new(std::sync::atomic::AtomicU8::new(0));
		let clint = ConcurrentClintCtx {
			base: 0,
			mtime: &*_mtime,
			mtimecmp: &*_mtimecmp,
			msip: &*_msip,
			num_harts: 1,
			msip_pending: Cell::new(std::ptr::null()),
			hart_threads: Cell::new(std::ptr::null()),
			hart_states: Cell::new(std::ptr::null()),
			timebase_hz: 0, // 单元测试禁用 clock-source 推进
		};
		// 泄漏 Box 使指针在整个测试生命周期有效 (与 interrupt/mod.rs 的
		// clint_with_msip 做法一致).
		std::mem::forget(_mtime);
		std::mem::forget(_mtimecmp);
		std::mem::forget(_msip);
		clint
	}

	/// 回归 (make emu-linux-sh 测速相位 + tick 活锁 + legacy PLIC 相位 tick 风暴):
	/// mtime 曾按单调时钟流逝推进 — 低速模拟 (~0.3 MIPS) 下内核 HZ=250 的定时器
	/// tick (4ms = 40000 ticks) 每真实 4ms 仅隔 ~1100 条指令, tick 处理路径一超长
	/// 即陷入 mret 后立即再 trap 的活锁, 客机在测速/ALSA 相位随机停滞 (guest
	/// 4.5s~18.9s 不等), 且引擎卡在 native batch 无法响应 Ctrl+Q.
	/// 修复: 纯指令计数 (20ns/instr, 无实时分量) — tick 预算 2e5 条, 永不风暴.
	/// 曾尝试 min(实时流逝, 指令计数) 混合模型: 200ns/instr 预算 2e4 条仍实测停滞
	/// (legacy PLIC 相位后 sched_tick + update_vsyscall + timekeeping + tracing
	/// 路径过长, kernel_init 被饿死, 见 /tmp/pyremu_repro/stall_dbg_stack.log),
	/// 且空闲批次实时分量与 Python 侧 clint.tick 补偿叠加成 ~2× real 双倍计数.
	/// 本测试断言 mtime 推进纯粹按指令: 1600 条指令 * 20ns/instr * 10MHz = 320 ticks,
	/// 与流逝时间 (10/100ms) 完全无关 (无实时泄漏).
	#[test]
	fn advance_clock_source_pure_instruction_advance_independent_of_elapsed() {
		let timebase_hz = 10_000_000u64; // 10 MHz: 1 tick = 100ns
		let base_mtime = 1_000_000u64;
		let mtime_atomic = AtomicU64::new(base_mtime);
		let mtimecmp_atomic = AtomicU64::new(0);
		let msip_atomic = AtomicU8::new(0);
		let clint = ConcurrentClintCtx {
			base: 0,
			mtime: &mtime_atomic as *const AtomicU64,
			mtimecmp: &mtimecmp_atomic as *const AtomicU64,
			msip: &msip_atomic as *const AtomicU8,
			num_harts: 1,
			msip_pending: std::cell::Cell::new(std::ptr::null()),
			hart_threads: std::cell::Cell::new(std::ptr::null()),
			hart_states: std::cell::Cell::new(std::ptr::null()),
			timebase_hz,
		};

		// 本轮已执行 1600 条指令 (instr_ref_num=1000 -> total_instrs=2600).
		let mut state: HartState = unsafe { std::mem::zeroed() };
		state.mhartid = 0;
		state.total_instrs = 2600;
		let mut module = ModuleState::new(1, 1, 0, base_mtime, vec![1000u64].into_boxed_slice());

		// 两种流逝下推进量必须一致: 1600 条 * 20ns/instr * 10MHz = 320 ticks.
		for elapsed_ms in [10u64, 100u64] {
			module.st_time_val = std::time::Instant::now()
				.checked_sub(std::time::Duration::from_millis(elapsed_ms))
				.expect("Instant 减法不应下溢");
			mtime_atomic.store(base_mtime, Ordering::Relaxed);
			advance_clock_source(&module, &clint, &state);
			let delta = mtime_atomic.load(Ordering::Relaxed) - base_mtime;
			assert_eq!(
				delta, 320,
				"mtime 必须纯按指令增量推进 (1600 条 -> 320 ticks @20ns/instr), \
				 与流逝 ({elapsed_ms}ms) 无关; 实际 {delta}"
			);
		}
	}

	/// 回归 (混合模型空闲批次双倍计数):
	/// 曾引入 min(实时流逝, 指令计数): 空闲批次 instr_delta=0 时 min 取 0 不推进,
	/// 但 Python ``_wfi_sleep_if_idle`` 同时按真实流逝 clint.tick 补偿 mtime — 若
	/// 批次侧对实时分量取非 0 值 (或引入任何非指令源), 两者叠加令 mtime 以
	/// ~2× real 流逝 (实测 1.65×), 重新制造风暴压力. 本测试断言 0 指令、100ms
	/// 流逝下 mtime 推进必须为 0 (时钟源仅指令, 空闲期由 Python 侧独占补偿).
	#[test]
	fn advance_clock_source_zero_instr_does_not_advance() {
		let timebase_hz = 10_000_000u64;
		let base_mtime = 1_000_000u64;
		let mtime_atomic = AtomicU64::new(base_mtime);
		let mtimecmp_atomic = AtomicU64::new(0);
		let msip_atomic = AtomicU8::new(0);
		let clint = ConcurrentClintCtx {
			base: 0,
			mtime: &mtime_atomic as *const AtomicU64,
			mtimecmp: &mtimecmp_atomic as *const AtomicU64,
			msip: &msip_atomic as *const AtomicU8,
			num_harts: 1,
			msip_pending: std::cell::Cell::new(std::ptr::null()),
			hart_threads: std::cell::Cell::new(std::ptr::null()),
			hart_states: std::cell::Cell::new(std::ptr::null()),
			timebase_hz,
		};

		// 本轮 0 条指令 (total_instrs == instr_ref_num == 1000).
		let mut state: HartState = unsafe { std::mem::zeroed() };
		state.mhartid = 0;
		state.total_instrs = 1000;
		let mut module = ModuleState::new(1, 1, 0, base_mtime, vec![1000u64].into_boxed_slice());

		module.st_time_val = std::time::Instant::now()
			.checked_sub(std::time::Duration::from_millis(100))
			.expect("Instant 减法不应下溢");
		mtime_atomic.store(base_mtime, Ordering::Relaxed);
		advance_clock_source(&module, &clint, &state);
		let delta = mtime_atomic.load(Ordering::Relaxed) - base_mtime;
		assert_eq!(
			delta, 0,
			"0 指令的空闲批次不得推进 mtime (时钟源仅指令, WFI 补偿在 Python 侧); \
			 实际 {delta}"
		);
	}

	/// 回归 (make emu-linux-sh legacy 模式 PLIC 相位后 tick 风暴):
	/// 停滞根因是 HZ=250 的 4ms tick 预算不足 — mret 后立即再 trap 的活锁,
	/// kernel_init 被饿死. legacy PLIC 相位后内核 (tracing 配置) 的 sched_tick +
	/// update_vsyscall + timekeeping trap 路径实测 5~5e4 条指令 (stall_dbg_stack.log:
	/// 三 hart 全在 update_vsyscall / cpu_do_idle / tmigr_requires_handle_remote).
	/// 纯指令计数的 tick 预算 = 1e9 / (NS_PER_INSTR × HZ) (与 timebase 无关):
	///   NS_PER_INSTR=20  => 2e5 条/tick (本值: handler 实测 ≤5e4, 4× 余量);
	///   NS_PER_INSTR=40  => 1e5 条/tick (2× 余量, 上界);
	///   NS_PER_INSTR=80  => 5e4 条/tick (恰好贴 handler 上限, 无余量);
	///   NS_PER_INSTR=200 => 2e4 条/tick (实测仍停滞);
	///   NS_PER_INSTR=500 => 8e3 条/tick (实测仍停滞).
	#[test]
	fn advance_clock_source_instruction_floor_meets_hz250_tick_budget() {
		const HZ: u64 = 250;
		let budget = 1_000_000_000u64 / (NS_PER_INSTR * HZ);
		assert!(
			budget >= 100_000,
			"HZ=250 tick 指令预算不足: {budget} 条/tick < 100000 \
			 (handler 实测上限 ~5e4, 需 ≥2× 余量; NS_PER_INSTR 过大)"
		);
	}

	#[test]
	fn legacy_plic_meip_does_not_force_enable_meie() {
		// 回归 (make emu-linux-sh PLIC 相位启动停滞):
		// 传统 PLIC 模式下 (imsic 不存在), OpenSBI 的 PLIC irqchip 驱动没有
		// process_hwirqs (只有 AIA IMSIC 驱动有), 故 sbi_irqchip_init 不会置
		// mie.MEIE, 且 sbi_irqchip_process() 恒返回 SBI_ENODEV。 若此处对瞬时
		// MEIP 强制使能 MEIE, M 级外部陷阱会落入 OpenSBI 无法处理的 irqchip,
		// sbi_trap_error -> sbi_hart_hang 冻结整个 SMP 启动 —— 实测在
		// `riscv-plic: plic@c000000:` 设备树节点输出后所有 hart 均以
		// mcause=0x800000000000000b (MEIP) 停滞 (见 /tmp/plic_stall_fh.log)。
		let clint = clint_ctx_empty();
		let mut state: HartState = unsafe { std::mem::zeroed() };
		state.mhartid = 0;
		state.mode = riscv_mode::S;
		state.mstatus = 1 << 3; // MIE = 1 — M 级中断可抢占 S 模式
		state.mie = 0; // 客机未使能任何中断 (含 MEIE)
		state.mip.store(1 << 11, Ordering::Release); // 瞬时 MEIP

		step_interrupts(&mut state, 0, &clint, false, std::ptr::null_mut());

		assert_eq!(
			state.mie & (1 << 11),
			0,
			"PLIC 模式下不得强制使能 MEIE — 否则 MEIP 陷阱落入 OpenSBI 无法处理的 irqchip (SBI_ENODEV) -> sbi_hart_hang"
		);
		assert_eq!(
			state.mip.load(Ordering::Acquire) & (1 << 11),
			1 << 11,
			"MEIP 保持 pending (由 Python 侧 batch 边界 _native_sync_plic_mip 同步清除)"
		);
	}

	#[test]
	fn aia_imsic_meip_still_force_enables_meie() {
		// 正控制: AIA 模式下 (imsic present) 强制使能必须保留 —— OpenSBI 的
		// IMSIC IPI 路径 (IID=1/3 经 MEIP) 可能不显式设置 mie.MEIE, 去掉后
		// WFI 中的 hart 永远不会因 IPI 唤醒 -> SMP 启动停滞。
		let clint = clint_ctx_empty();
		let mut state: HartState = unsafe { std::mem::zeroed() };
		state.mhartid = 0;
		state.mode = riscv_mode::S;
		state.mstatus = 1 << 3; // MIE = 1
		state.mie = 0;
		state.mip.store(1 << 11, Ordering::Release); // MEIP (IMSIC IPI)
		state.imsic_m.present = 1;
		// 使 IMSIC M-file 持有 IPI 位, 让 stale-bit 清理保留 MEIP
		// (否则 eidelivery=0 且无 IPI 时会清除 MEIP, 后续断言失去意义).
		state.imsic_m.eip[0].store(1 << crate::interrupt::imsic::IID_M_IPI, Ordering::Release);

		step_interrupts(&mut state, 0, &clint, false, std::ptr::null_mut());

		assert_eq!(
			state.mie & (1 << 11),
			1 << 11,
			"AIA 模式 (IMSIC present) 下仍须强制使能 MEIE"
		);
	}
}
