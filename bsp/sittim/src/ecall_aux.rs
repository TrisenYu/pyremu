//! All ecall wrappers — enclave extension calls and standard SBI calls.
//! Mirrors the pattern in smode_entry/trap_handler.c + ref-emod/enclave_ops.h.

use core::arch::asm;

use crate::constants::*;
use crate::context;
use crate::paging;

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

/// 取回本飞地模块请求的交付结果, 返回
/// `(交付结果, 宿主报告的字节数, 接收窗口首字节的物理地址, 实际写入的字节数)`.
///
/// 模块请求的每一次让出 (见 ENCLAVE_SUSPEND_MODULE_*) 都以一次本调用闭合: 飞地经
/// RESUME 取回的是自己让出时的寄存器, 宿主无法经 RESUME 回传数据, 故结果只能另行取回.
/// 读取后 M 模式把交付结果复位为 ENCLAVE_MODULE_STATUS_NONE.
#[inline]
pub fn enclave_call_module_result() -> (u64, u64, u64, u64) {
	let a0: u64;
	let a1: u64;
	let a2: u64;
	let a3: u64;
	unsafe {
		asm!(
			"ecall",
			in("a7") ENCLAVE_EXT_ID,
			in("a6") ENCLAVE_CALL_MODULE_RESULT,
			lateout("a0") a0,
			lateout("a1") a1,
			lateout("a2") a2,
			lateout("a3") a3,
		);
	}
	(a0, a1, a2, a3)
}

/// 登记模块映像接收缓冲区, 入参为缓冲区的起始虚拟地址与字节数。
///
/// 交付的映像字节由 M 模式写入本缓冲区, 故飞地须先分配并映射好缓冲区再登记, 登记之后
/// 才让出请求交付。返回 0 表示已登记, 其余取值为 SBI 错误码。
#[inline]
pub fn enclave_call_module_img_recv_buf(va: u64, len: u64) -> u64 {
	unsafe { ecall_3(ENCLAVE_EXT_ID, ENCLAVE_CALL_MODULE_IMG_RECV_BUF, va, len, 0) }.0
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
/// 地址折算失败时返回 0: 未写入任何字节.
#[inline]
pub fn enclave_call_get_rand_num(buf: &mut [u8]) -> u64 {
	let Some(pa) = va_to_pa(buf.as_ptr() as u64) else {
		return 0;
	};
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
///
/// 未建立映射时返回 None, 由调用方决定退化行为. 不得以 0 代替失败: 0 是合法
/// 物理地址, 会把访问引向域内存之外而不报错.
///
/// 对外可见: 管理器把它作为 `va_to_pa` 回调交给模块, 模块据此折算自身静态存储与
/// 设备缓冲区两处的物理地址 (见 `ext_mod::runtime`).
#[inline]
pub fn va_to_pa(va: u64) -> Option<u64> {
	// 模块窗口的判定必须先于管理器窗口: 模块窗口基址高于管理器窗口基址, 不先
	// 判定则模块地址落进按管理器窗口换算的分支并得到错误的物理地址。
	// 模块各块的物理地址由独立一次分配取得, 块间不保证连续, 无固定偏移可用,
	// 故按页表查询。
	if va >= ENCLAVE_MODULE_LOAD_VA_INIT {
		return paging::get_pa(va);
	}
	if va >= ENCLAVE_MAN_VA_START {
		// 运行时自身代码/数据/栈: VA = PA + (ENCLAVE_MAN_VA_START - manager_pa_start).
		// 仅 MMU 使能后才可能落入此区间, 此刻 context 已初始化.
		let va_ofs = ENCLAVE_MAN_VA_START.wrapping_sub(context::ctx().manager_pa_start);
		Some(va.wrapping_sub(va_ofs))
	} else if va >= LINEAR_MAP_OFFSET {
		// 载荷/argv 的线性映射别名: VA = PA + LINEAR_MAP_OFFSET.
		Some(va.wrapping_sub(LINEAR_MAP_OFFSET))
	} else {
		// MMU 未使能时的恒等映射 (VA == PA), 无需折算.
		Some(va)
	}
}

/// Output a byte buffer via the SBI DBCN Console Write extension.
/// The entire buffer is written atomically under M-mode's `console_out_lock`,
/// preventing per-character interleaving with concurrent output from other harts.
///
/// Falls back to legacy `sbi_putchar` per byte if DBCN is not available, and also
/// when the buffer address does not translate to a physical address (DBCN reads
/// a1 as a physical address, so an untranslatable buffer leaves no target to name).
#[inline]
pub fn sbi_console_write(buf: &[u8]) {
	if let Some(pa) = va_to_pa(buf.as_ptr() as u64) {
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
		if ret == 0 {
			return;
		}
	}
	for &byte in buf {
		sbi_putchar(byte);
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
