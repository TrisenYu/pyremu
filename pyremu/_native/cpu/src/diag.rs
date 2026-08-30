//! Diagnostic counter helpers for the concurrent execution engine.
//!
//! Every function body is gated behind ``#[cfg(feature = "diagnostic")]``.
//! When the feature is disabled, each function compiles to a no-op — the
//! call sites stay clean (no ``#[cfg]`` at every caller) and the compiler
//! eliminates the dead stores.
#[cfg(feature = "diagnostic")]
use std::sync::atomic::{AtomicU32, Ordering};
#[cfg(feature = "diagnostic")]
use std::io::Write;

use crate::state::{HartDiag, HartState};
use crate::translate::WalkCtx;
// ============================================================
//  Shared: log file helper
// ============================================================

/// Resolve the diagnostic log path from ``PYREMU_DIAG_LOG`` env var.
#[cfg(feature = "diagnostic")]
fn diag_log_path() -> String {
	std::env::var("PYREMU_DIAG_LOG").unwrap_or_else(|_| "/tmp/pyremu_diag.log".to_string())
}

/// Append a pre-formatted line to the diagnostic log file.
#[inline(always)]
#[allow(unused)]
#[allow(dead_code)]
pub fn log_line(_line: &str) {
	#[cfg(feature = "diagnostic")]
	{
		if let Ok(mut f) = std::fs::OpenOptions::new()
			.create(true)
			.append(true)
			.open(&diag_log_path())
		{
			let _ = writeln!(f, "{}", _line);
		}
	}
}

/// 长 batch 诊断: 采样各 hart 的 PC/mode/waiting/halted/mip 快照.
///
// ============================================================
//  WFI spin-loop diagnostics
// ============================================================

/// Track the reason for exiting the WFI spin loop.
#[inline(always)]
#[allow(unused)]
#[allow(dead_code)]
pub fn wfi_wake_reason(_diag: &mut HartDiag, _pending: u64, _msip_edge: bool) {
	#[cfg(feature = "diagnostic")]
	{
		if _msip_edge || ((_pending >> 3) & 1) != 0 {
			_diag.wfi_wake_msip = _diag.wfi_wake_msip.wrapping_add(1);
		} else if ((_pending >> 7) & 1) != 0 {
			_diag.wfi_wake_mtip = _diag.wfi_wake_mtip.wrapping_add(1);
		} else {
			_diag.wfi_wake_other = _diag.wfi_wake_other.wrapping_add(1);
		}
	}
}

// ============================================================
//  hart_worker diagnostics
// ============================================================

/// Track that MSIE was forced-set because MSIP was pending but masked.
#[inline(always)]
#[allow(dead_code)]
#[allow(unused)]
pub fn msie_forced(_diag: &mut HartDiag) {
	#[cfg(feature = "diagnostic")]
	{
		_diag.msip_masked_by_msie = _diag.msip_masked_by_msie.wrapping_add(1);
	}
}

/// Track the PC where MSIE was explicitly cleared (CSR write).
#[inline(always)]
#[allow(dead_code)]
#[allow(unused)]
pub fn msie_cleared_at(_diag: &mut HartDiag, _pc: u64) {
	#[cfg(feature = "diagnostic")]
	{
		_diag.msie_cleared_at_pc = _pc;
	}
}

// ============================================================
//  ld-linux.so trap trace
// ============================================================

/// Log trap details when trap PC is in the ld-linux.so range.
#[inline(always)]
#[allow(unused)]
#[allow(dead_code)]
pub fn ld_linux_trap(_state: &HartState, _code: u64, _tval: u64) {
	#[cfg(feature = "diagnostic")]
	{
		let pc = _state.pc;
		let is_int = (_code >> 63) != 0;
		let exc = _code & 0x7FFF_FFFF_FFFF_FFFF;
		let lo: u64 = 0x3ff7fdc000;
		let hi: u64 = 0x3ff7ffc000;
		if pc < lo || pc >= hi {
			return;
		}
		log_line(&format!(
			"[ld-trap] pc={:#018x} ({:+}) sepc={:#018x} cause={} {} tval={:#018x} \
             a0={:#018x} a1={:#018x} a5={:#018x} sp={:#018x} ra={:#018x}",
			pc,
			pc as i64 - lo as i64,
			_state.sepc,
			if is_int { "IRQ" } else { "EXC" },
			exc,
			_tval,
			_state.gprs[10],
			_state.gprs[11],
			_state.gprs[15],
			_state.gprs[2],
			_state.gprs[1],
		));
	}
}

// ============================================================
//  sret -> U-mode trace
// ============================================================

/// Track sret -> U-mode transitions: log key register state on first N transitions.
/// Gated by ``PYREMU_TRACE_SRET=N`` env var (runtime, requires "diagnostic" feature).
#[inline(always)]
#[allow(unused)]
#[allow(dead_code)]
pub fn sret_to_umode(_state: &HartState, _ctx: &WalkCtx) {
	#[cfg(feature = "diagnostic")]
	{
		static REMAINING: AtomicU32 = AtomicU32::new(u32::MAX);
		let v = REMAINING.load(Ordering::Relaxed);
		let remaining = if v == u32::MAX {
			let init: u32 = std::env::var("PYREMU_TRACE_SRET")
				.ok()
				.and_then(|s| s.parse().ok())
				.unwrap_or(0);
			REMAINING.store(init, Ordering::Relaxed);
			init
		} else {
			v
		};
		if remaining == 0 {
			return;
		}
		REMAINING.store(remaining - 1, Ordering::Relaxed);
		let sp = _state.gprs[2];
		let stack = read_user_stack_from_sp(_state, _ctx);
		let msg = format!(
			"[sret->U] pc={:#018x} a0(argc)={:#018x} a1(argv)={:#018x} \
             a2(envp)={:#018x} sp={:#018x} ra={:#018x} \
             satp={:#018x} sepc={:#018x}",
			_state.sepc,
			_state.gprs[10],
			_state.gprs[11],
			_state.gprs[12],
			sp,
			_state.gprs[1],
			_state.satp,
			_state.sepc,
		);
		log_line(&msg);
		for (i, chunk) in stack.chunks(4).enumerate() {
			let base = i * 4;
			log_line(&format!(
                "         +0x{:02x}={:#018x} +0x{:02x}={:#018x} +0x{:02x}={:#018x} +0x{:02x}={:#018x}",
                base * 8, chunk[0], (base + 1) * 8, chunk[1],
                (base + 2) * 8, chunk[2], (base + 3) * 8, chunk[3],
            ));
		}
	}
}

/// Read 64 u64 words from the user stack at sp, translating through Sv39 if enabled.
#[cfg(feature = "diagnostic")]
#[inline(always)]
#[allow(unused)]
#[allow(dead_code)]
fn read_user_stack_from_sp(
	_state: &HartState,
	_ctx: &WalkCtx,
) -> [u64; 64] {
	let mut out = [0u64; 64];
	let sp = _state.gprs[2];
	for i in 0..64u64 {
		let va = sp.wrapping_add(i * 8);
		let pa = if _state.mmu_mode == 8 {
			match crate::translate::sv39_walk(_ctx, _state.satp, va, false, false) {
				Some(t) => t.pa,
				None => continue,
			}
		} else {
			va
		};
		if pa >= _ctx.ram_base && pa + 8 <= _ctx.ram_base + _ctx.ram_size {
			let off = (pa - _ctx.ram_base) as usize;
			let mut bytes = [0u8; 8];
			for j in 0..8 {
				bytes[j] = unsafe { *_ctx.ram.add(off + j) };
			}
			out[i as usize] = u64::from_le_bytes(bytes);
		}
	}
	out
}
