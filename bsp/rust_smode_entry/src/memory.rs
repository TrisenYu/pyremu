//! 页池分配器与内存管理。紧跟 ref-emod memory.c + page_pool.c 逻辑。
//!
//! 设计要点（来自原 C 注释）：
//! - 用相对偏移量而非绝对地址，方便日后迁移。
//! - S-mode / U-mode 各自独立页池。
//! - 页池用完后通过 enclave_call_mem_alloc 向 M-mode 申请 CHUNK_2M。

use crate::constants::*;
use crate::context::{self, PoolDesc};
use crate::ecall_aux;
use crate::hang;
use crate::paging;

// ---------------------------------------------------------------
//  对齐辅助
// ---------------------------------------------------------------

#[inline]
pub const fn page_up(x: u64) -> u64 {
	(x + PAGE_SIZE - 1) & !(PAGE_SIZE - 1)
}

#[inline]
pub const fn page_down(x: u64) -> u64 {
	x & !(PAGE_SIZE - 1)
}

#[inline]
pub const fn chunk_2m_up(x: u64) -> u64 {
	(x + CHUNK_2M_SIZE - 1) & !(CHUNK_2M_SIZE - 1)
}

#[inline]
pub const fn chunk_2m_down(x: u64) -> u64 {
	x & !(CHUNK_2M_SIZE - 1)
}

// ---------------------------------------------------------------
//  页池初始化
// ---------------------------------------------------------------

pub fn init_smode_pool(offset: u64, size: u64) {
	context::ctx_mut().smode_pool = PoolDesc {
		offset,
		size,
		used_pages: 0,
	};
}

pub fn init_umode_pool(offset: u64, size: u64) {
	context::ctx_mut().umode_pool = PoolDesc {
		offset,
		size,
		used_pages: 0,
	};
}

// ---------------------------------------------------------------
//  页池分配
// ---------------------------------------------------------------

/// 从指定页池分配 `n` 个物理页，返回物理地址。
fn pool_alloc(n: u64, is_umode: bool) -> u64 {
	let ctx = context::ctx();
	let pool = if is_umode {
		&ctx.umode_pool
	} else {
		&ctx.smode_pool
	};

	let expected = pool.used_pages + n;
	if expected * PAGE_SIZE > pool.size {
		hang::hang_with_msg("pool_alloc: out of pages\n");
	}

	let chunk_start = if is_umode {
		ctx.umode_pool_pa_aligned
	} else {
		ctx.manager_pa_start
	};
	let top = chunk_start + pool.offset + pool.used_pages * PAGE_SIZE;

	let p = if is_umode {
		&mut context::ctx_mut().umode_pool
	} else {
		&mut context::ctx_mut().smode_pool
	};
	p.used_pages = expected;
	top
}

pub fn alloc_smode_page(n: u64) -> u64 {
	pool_alloc(n, false)
}
pub fn alloc_umode_page(n: u64) -> u64 {
	pool_alloc(n, true)
}
pub fn umode_pool_avail() -> u64 {
	context::ctx().umode_pool.avail_bytes()
}

// ---------------------------------------------------------------
//  段映射
// ---------------------------------------------------------------

/// 用链接器符号将 enclave-man 自身各段映射到 VA 空间。
/// 链接器符号已由 entry.s 中的 PIE 重定位调整至运行时地址,
/// 无需再加 load_offset, 否则会 double-count base_pa.
pub fn map_sections() {
	let ctx = context::ctx();
	let start_pa = ctx.manager_pa_start;
	let va_ofs = ENCLAVE_MAN_VA_START.wrapping_sub(start_pa);

	unsafe extern "C" {
		static _text_start: u8;
		static _text_end: u8;
		static _rodata_start: u8;
		static _rodata_end: u8;
		static _data_start: u8;
		static _data_end: u8;
		static _bss_start: u8;
		static _bss_end: u8;
	}

	// 逐个映射各段，避免 Rust 2024 下 macro token pasting 的限制
	unsafe {
		map_one_section(
			&raw const _text_start as u64,
			&raw const _text_end as u64,
			va_ofs,
			PTE_X,
		);
		map_one_section(
			&raw const _rodata_start as u64,
			&raw const _rodata_end as u64,
			va_ofs,
			PTE_R,
		);
		map_one_section(
			&raw const _data_start as u64,
			&raw const _data_end as u64,
			va_ofs,
			PTE_R | PTE_W,
		);
		map_one_section(
			&raw const _bss_start as u64,
			&raw const _bss_end as u64,
			va_ofs,
			PTE_R | PTE_W,
		);
	}
}

/// 映射单个段：start_pa / end_pa -> VA = PA + va_offset。
unsafe fn map_one_section(sec_start: u64, sec_end: u64, va_offset: u64, flags: u8) {
	if sec_start >= sec_end {
		return;
	}
	let size = sec_end.wrapping_sub(sec_start);
	// VA = PA + va_offset (sec_start 是 PIE 重定位后的运行时 PA)
	let va = sec_start.wrapping_add(va_offset);
	// DEBUG: trace first page
	let vpn2 = (va >> 30) & 0x1FF;
	crate::println!("[map_sec] pa=0x{sec_start:x} va=0x{va:x} vpn2=0x{vpn2:x} flags=0x{flags:x}\n");
	for i in 0..(page_up(size) >> PAGE_SHIFT) {
		paging::map_page(
			va + i * PAGE_SIZE,
			page_down(sec_start) + i * PAGE_SIZE,
			flags,
			LEVEL_PAGE,
		);
	}
}

/// 将 S-mode 页池映射到 VA 空间。
pub fn map_smode_page_pool(pool_ofs: u64, pool_size: u64) {
	let ctx = context::ctx();
	let start_pa = ctx.manager_pa_start + pool_ofs;
	let va_ofs = ENCLAVE_MAN_VA_START.wrapping_sub(ctx.manager_pa_start);

	// DEBUG: trace first page mapping
	if pool_size > 0 {
		let first_pa = start_pa;
		let first_va = first_pa.wrapping_add(va_ofs);
		let vpn2 = (first_va >> 30) & 0x1FF;
		crate::println!("[map_pool] pa=0x{first_pa:x} va=0x{first_va:x} vpn2=0x{vpn2:x}\n");
	}

	for i in 0..(pool_size >> PAGE_SHIFT) {
		let pa = start_pa + i * PAGE_SIZE;
		paging::map_page(pa.wrapping_add(va_ofs), pa, PTE_R | PTE_W, LEVEL_PAGE);
	}
}

// ---------------------------------------------------------------
//  栈分配
// ---------------------------------------------------------------

#[unsafe(no_mangle)]
pub unsafe extern "C" fn alloc_smode_stack() -> u64 {
	let ctx = context::ctx();
	let va_ofs = ENCLAVE_MAN_VA_START.wrapping_sub(ctx.manager_pa_start);
	alloc_smode_page(SMODE_STACK_SIZE >> PAGE_SHIFT) + SMODE_STACK_SIZE + va_ofs
}

pub fn alloc_map_umode_stack() -> u64 {
	let pages = UMODE_STACK_SIZE_TOTAL >> PAGE_SHIFT;
	let bottom_pa = alloc_umode_page(pages);
	let bottom_va = UMODE_STACK_TOP_VA - UMODE_STACK_SIZE_TOTAL;

	for i in 0..pages {
		paging::map_page(
			bottom_va + i * PAGE_SIZE,
			bottom_pa + i * PAGE_SIZE,
			PTE_U | PTE_R | PTE_W,
			LEVEL_PAGE,
		);
	}
	UMODE_STACK_TOP_VA
}

/// 将用户 argv 从 PA 映射到 U-mode VA。
pub fn map_user_argv(argv_pa: u64, argc: u64) {
	let va_argv = UMODE_STACK_TOP_VA;
	let argv_ptr = argv_pa.wrapping_add(LINEAR_MAP_OFFSET) as *mut u64;

	for i in 0..argc as usize {
		unsafe {
			let val = argv_ptr.add(i).read_volatile();
			let off = val % PAGE_SIZE;
			argv_ptr.add(i).write_volatile(va_argv + off);
		}
	}
	paging::map_page(va_argv, argv_pa, PTE_U | PTE_R | PTE_W, LEVEL_PAGE);
}

// ---------------------------------------------------------------
//  brk（U-mode 堆扩展）
// ---------------------------------------------------------------

pub fn sys_brk_handler(new_brk: u64) -> u64 {
	let old_top = context::ctx().umode_heap_top;
	if new_brk == 0 {
		return old_top; // 仅查询
	}

	let aligned_new = page_up(new_brk);
	let aligned_old = page_up(old_top);

	if aligned_new <= aligned_old {
		context::ctx_mut().umode_heap_top = new_brk;
		return new_brk;
	}

	let mut remain = aligned_new - aligned_old;
	let avail = umode_pool_avail();

	// 先从页池取
	if avail > 0 {
		let take = if remain <= avail { remain } else { avail };
		let pa = alloc_umode_page(take >> PAGE_SHIFT);
		for i in 0..(take >> PAGE_SHIFT) {
			paging::map_page(
				aligned_old + i * PAGE_SIZE,
				pa + i * PAGE_SIZE,
				PTE_U | PTE_R | PTE_W,
				LEVEL_PAGE,
			);
		}
		remain -= take;
	}

	// 不够再向 M-mode 申请 CHUNK_2M
	if remain <= 0 {
		context::ctx_mut().umode_heap_top = new_brk;
		return new_brk;
	}

	let chunk_new = chunk_2m_up(aligned_new);
	let chunk_old = chunk_2m_up(aligned_old);
	let mut left = chunk_new.saturating_sub(chunk_old);
	let mut va = chunk_old;

	while left > 0 {
		let (allocated, pa) = ecall_aux::enclave_call_mem_alloc(left / CHUNK_2M_SIZE);
		let bytes = allocated * CHUNK_2M_SIZE;
		if bytes == 0 {
			break;
		}
		for i in 0..allocated {
			paging::map_page(
				va + i * CHUNK_2M_SIZE,
				pa + i * CHUNK_2M_SIZE,
				PTE_U | PTE_R | PTE_W,
				LEVEL_MEGA,
			);
		}
		left = left.saturating_sub(bytes);
		va += bytes;
	}
	context::ctx_mut().umode_heap_top = new_brk;
	new_brk
}

// ---------------------------------------------------------------
//  测试
// ---------------------------------------------------------------

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_page_up_aligned() {
		assert_eq!(page_up(0x1000), 0x1000);
		assert_eq!(page_up(0x2000), 0x2000);
	}

	#[test]
	fn test_page_up_rounds() {
		assert_eq!(page_up(0x1001), 0x2000);
		assert_eq!(page_up(0x1), 0x1000);
		assert_eq!(page_up(0xFFF), 0x1000);
	}

	#[test]
	fn test_page_down_rounds() {
		assert_eq!(page_down(0x1FFF), 0x1000);
		assert_eq!(page_down(0x1000), 0x1000);
		assert_eq!(page_down(0x0), 0x0);
	}

	#[test]
	fn test_chunk_2m_up() {
		assert_eq!(chunk_2m_up(0x20_0000), 0x20_0000);
		assert_eq!(chunk_2m_up(0x20_0001), 0x40_0000);
		assert_eq!(chunk_2m_up(0x1), 0x20_0000);
	}

	#[test]
	fn test_chunk_2m_down() {
		assert_eq!(chunk_2m_down(0x3F_FFFF), 0x20_0000);
		assert_eq!(chunk_2m_down(0x0), 0x0);
	}

	#[test]
	fn test_chunk_page_ratio() {
		assert_eq!(CHUNK_2M_SIZE % PAGE_SIZE, 0);
		assert_eq!(CHUNK_2M_SIZE >> PAGE_SHIFT, 512);
	}
}
