//! All ecall wrappers — enclave extension calls and standard SBI calls.
//! Mirrors the pattern in smode_entry/trap_handler.c + ref-emod/enclave_ops.h.

use core::arch::asm;

use crate::constants::*;
use crate::context;

// ---------------------------------------------------------------
//  Low-level ecall primitives
// ---------------------------------------------------------------

/// Execute an ecall with ext_id, func_id, and three args.  Returns (a0, a1).
#[inline(always)]
unsafe fn ecall_3(ext_id: u64, func_id: u64, a0: u64, a1: u64, a2: u64) -> (u64, u64) {
	let ret0: u64;
	let ret1: u64;
	unsafe {
		asm!(
			"ecall",
			in("a7") ext_id,
			in("a6") func_id,
			in("a0") a0,
			in("a1") a1,
			in("a2") a2,
			lateout("a0") ret0,
			lateout("a1") ret1,
		);
	}
	(ret0, ret1)
}

// ---------------------------------------------------------------
//  Enclave extension calls  (ext_id = 0x2022_1222)
// ---------------------------------------------------------------

/// Yield to M-mode.  M-mode returns `(a0, a1, a2)` — typically
/// `(payload_pa, payload_size, argc)` during boot.
#[inline]
pub fn enclave_call_suspend(short_msg: u64) -> (u64, u64, u64) {
	let a0: u64;
	let a1: u64;
	let a2: u64;
	unsafe {
		asm!(
			"ecall",
			in("a7") ENCLAVE_EXT_ID,
			in("a6") ENCLAVE_CALL_SUSPEND,
			in("a0") short_msg,
			lateout("a0") a0,
			lateout("a1") a1,
			lateout("a2") a2,
		);
	}
	(a0, a1, a2)
}

/// Request 2 MiB memory chunks from M-mode.
/// Returns `(allocated_count, physical_address)`.
pub fn enclave_call_mem_alloc(chunk_nums: u64) -> (u64, u64) {
	unsafe { ecall_3(ENCLAVE_EXT_ID, ENCLAVE_CALL_MEM_ALLOC, 0, chunk_nums, 0) }
}

/// Query remaining pool capacity from M-mode.
/// Returns `(free_total, max_contiguous)` in units of 2 MiB partitions.
#[inline]
#[allow(dead_code)]
pub fn enclave_call_get_available_mem() -> (u64, u64) {
	unsafe { ecall_3(ENCLAVE_EXT_ID, ENCLAVE_CALL_GET_AVAILABLE_MEM, 0, 0, 0) }
}

/// Get the current enclave ID.
#[inline]
#[allow(dead_code)]
pub fn enclave_call_get_id() -> u64 {
	let ret: u64;
	unsafe {
		asm!(
			"ecall",
			in("a7") ENCLAVE_EXT_ID,
			in("a6") ENCLAVE_CALL_GET_ID,
			lateout("a0") ret,
		);
	}
	ret
}

/// Get the current hart ID.
#[inline]
#[allow(dead_code)]
pub fn enclave_call_get_hartid() -> u64 {
	let ret: u64;
	unsafe {
		asm!(
			"ecall",
			in("a7") ENCLAVE_EXT_ID,
			in("a6") ENCLAVE_CALL_GET_HARTID,
			lateout("a0") ret,
		);
	}
	ret
}

/// Destroy this enclave and return to host.
pub fn enclave_call_exit(code: u64) -> ! {
	unsafe {
		ecall_3(ENCLAVE_EXT_ID, ENCLAVE_CALL_SHUTDOWN, code, 0, 0);
	}
	crate::hang::fault_halt("enclave_call_exit: ecall returned\n")
}

/// Query M-mode for pending host requests.
/// Returns a flags bitmask: bit0=SHUTDOWN_REQUESTED.
#[inline]
pub fn enclave_call_query_requests() -> u64 {
	let ret: u64;
	unsafe {
		asm!(
			"ecall",
			in("a7") ENCLAVE_EXT_ID,
			in("a6") ENCLAVE_CALL_QUERY_REQUESTS,
			lateout("a0") ret,
		);
	}
	ret
}

/// 请求 M-mode 向 *buf* 写入随机字节, 返回实际写入的字节数.
///
/// 缓冲区地址以物理地址形式交给 M-mode (与 DBCN 同理), 因此只可用于运行时
/// 自身的内存; 载荷的缓冲区是 U 模式虚拟地址, 须先分块拷入本地缓冲.
#[inline]
pub fn enclave_call_get_rand_num(buf: &mut [u8]) -> u64 {
	let pa = va_to_pa(buf.as_ptr() as u64);
	unsafe { ecall_3(ENCLAVE_EXT_ID, ENCLAVE_CALL_GET_RAND_NUM, pa, buf.len() as u64, 0) }.0
}

/// Forward an unmatched access fault to M-mode for diagnosis.
/// Called when S-mode cannot resolve a load/store access fault (scause 5/7).
#[inline]
pub fn enclave_call_unmatched_acc_fault(stval: u64) {
	unsafe {
		ecall_3(
			ENCLAVE_EXT_ID,
			ENCLAVE_CALL_UNMATCHED_ACC_FAULT,
			stval,
			0,
			0,
		);
	}
}

// ---------------------------------------------------------------
//  Standard SBI calls
// ---------------------------------------------------------------

/// Output a single character via the legacy SBI console putchar interface.
#[inline]
#[allow(dead_code)]
pub fn sbi_putchar(c: u8) {
	unsafe {
		asm!(
			"ecall",
			in("a7") SBI_LEGACY_PUTCHAR_EXT,
			in("a6") 0_u64,
			in("a0") c as u64,
		);
	}
}

/// 把运行时的虚拟地址折算为物理地址, 供 SBI DBCN 这类地址按物理地址解释的
/// ecall 使用. 缓冲区是栈上局部量, MMU 使能后其地址落入高 VA 空间
/// (ENCLAVE_MAN_VA_START 区域), 直接传给 M-mode 会被当作 PA 而找不到域内存区域,
/// 导致整条诊断输出被静默丢弃. 低地址 (MMU 未使能时的恒等 PA) 原样返回.
#[inline]
fn va_to_pa(va: u64) -> u64 {
	if va >= ENCLAVE_MAN_VA_START {
		// 运行时自身代码/数据/栈: VA = PA + (ENCLAVE_MAN_VA_START - manager_pa_start).
		// 仅 MMU 使能后才可能落入此区间, 此刻 context 已初始化.
		let va_ofs = ENCLAVE_MAN_VA_START.wrapping_sub(context::ctx().manager_pa_start);
		va.wrapping_sub(va_ofs)
	} else if va >= LINEAR_MAP_OFFSET {
		// 载荷/argv 的线性映射别名: VA = PA + LINEAR_MAP_OFFSET.
		va.wrapping_sub(LINEAR_MAP_OFFSET)
	} else {
		// MMU 未使能时的恒等映射 (VA == PA), 无需折算.
		va
	}
}

/// Output a byte buffer via the SBI DBCN Console Write extension.
/// The entire buffer is written atomically under M-mode's `console_out_lock`,
/// preventing per-character interleaving with concurrent output from other harts.
///
/// Falls back to legacy `sbi_putchar` per byte if DBCN is not available.
#[inline]
pub fn sbi_console_write(buf: &[u8]) {
	let pa = va_to_pa(buf.as_ptr() as u64);
	let len = buf.len() as u64;
	let ret: u64;
	unsafe {
		asm!(
			"ecall",
			in("a7") SBI_DBCN_EXT,
			in("a6") SBI_DBCN_CONSOLE_WRITE,
			in("a0") len,
			in("a1") pa,
			in("a2") 0_u64,
			lateout("a0") ret,
		);
	}
	// DBCN returns 0 on success, negative error code on failure.
	// Fall back to character-by-character legacy putchar.
	if ret != 0 {
		for &byte in buf {
			sbi_putchar(byte);
		}
	}
}

/// Schedule a timer interrupt at `stime_value` (absolute time in ticks).
#[inline]
#[allow(dead_code)]
#[allow(unused)]
pub fn sbi_set_timer(stime_value: u64) {
	unsafe {
		ecall_3(SBI_TIMER_EXT, SBI_SET_TIMER_FUNC, stime_value, 0, 0);
	}
}
