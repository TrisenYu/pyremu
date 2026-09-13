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

/// 从指定页池分配 `n` 个物理页，返回物理地址；池耗尽返回 `!0u64`。
///
/// 不再在此处 hang：由调用方决定是向上层报 OOM (U-mode 堆/mmap) 还是视为致命错误
/// (S-mode 页表/栈, 见 `alloc_smode_page`)。
fn pool_alloc(n: u64, is_umode: bool) -> u64 {
	let ctx = context::ctx();
	let pool = if is_umode {
		&ctx.umode_pool
	} else {
		&ctx.smode_pool
	};

	let expected = pool.used_pages + n;
	if expected * PAGE_SIZE > pool.size {
		return !0u64;
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

/// S-mode 页池分配 (页表/栈): 池耗尽属启动期致命错误, fail-stop.
pub fn alloc_smode_page(n: u64) -> u64 {
	let pa = pool_alloc(n, false);
	if pa == !0u64 {
		hang::fault_halt("alloc_smode_page: out of pages\n");
	}
	pa
}

/// S-mode 页池分配 (不挂起): 池耗尽返回 `!0u64`, 由调用方决定向上报错还是视为致命。
///
/// 载荷可控的路径 (brk / mmap 触发的页表扩容) 必须走本函数: 这些请求不得让
/// 可信管理程序自身停止服务, 否则载荷一次越界的内存请求即可使整个飞地失联,
/// 对宿主表现为 (suspended), 与载荷仍在计算无法区分。
pub fn try_alloc_smode_page(n: u64) -> u64 {
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
	#[cfg(feature = "diagnostic")]
	{
		let vpn2 = (va >> 30) & 0x1FF;
		crate::println!("[map_sec] pa=0x{sec_start:x} va=0x{va:x} vpn2=0x{vpn2:x} flags=0x{flags:x}\n");
	}
	for i in 0..(page_up(size) >> PAGE_SHIFT) {
		map_or_halt(
			va + i * PAGE_SIZE,
			page_down(sec_start) + i * PAGE_SIZE,
			flags,
			LEVEL_PAGE,
			"map_sections: page table\n",
		);
	}
}

/// 将 S-mode 页池映射到 VA 空间。
pub fn map_smode_page_pool(pool_ofs: u64, pool_size: u64) {
	let ctx = context::ctx();
	let start_pa = ctx.manager_pa_start + pool_ofs;
	let va_ofs = ENCLAVE_MAN_VA_START.wrapping_sub(ctx.manager_pa_start);

	// DEBUG: trace first page mapping
	#[cfg(feature = "diagnostic")]
	if pool_size > 0 {
		let first_pa = start_pa;
		let first_va = first_pa.wrapping_add(va_ofs);
		let vpn2 = (first_va >> 30) & 0x1FF;
		crate::println!("[map_pool] pa=0x{first_pa:x} va=0x{first_va:x} vpn2=0x{vpn2:x}\n");
	}

	for i in 0..(pool_size >> PAGE_SHIFT) {
		let pa = start_pa + i * PAGE_SIZE;
		map_or_halt(
			pa.wrapping_add(va_ofs),
			pa,
			PTE_R | PTE_W,
			LEVEL_PAGE,
			"map_smode_page_pool: page table\n",
		);
	}
}

/// 启动期建立映射。失败即以 fail-stop 收场并放开中断,
/// 宿主仍可下发终止请求拆掉该飞地。
fn map_or_halt(va: u64, pa: u64, flags: u8, level: u8, msg: &str) {
	if paging::map_page(va, pa, flags, level).is_err() {
		hang::fault_halt(msg);
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

/// 从 S-mode 页池分配一块线程内核栈 (不挂起), 返回栈顶 VA; 池耗尽返回 0。
///
/// 页面由启动期的 map_smode_page_pool 预映射, 本分配只向前推进 used_pages。
/// 调用方 (thread::clone_thread) 在池耗尽时返回 ENOMEM 而非挂死内核。
pub fn try_alloc_thread_kstack(kstack_size: u64) -> u64 {
	let ctx = context::ctx();
	let va_ofs = ENCLAVE_MAN_VA_START.wrapping_sub(ctx.manager_pa_start);
	let pa = pool_alloc(kstack_size >> PAGE_SHIFT, false);
	if pa == !0u64 {
		return 0;
	}
	pa + kstack_size + va_ofs
}

pub fn alloc_map_umode_stack() -> u64 {
	let pages = UMODE_STACK_SIZE_TOTAL >> PAGE_SHIFT;
	let bottom_pa = alloc_umode_page(pages);
	if bottom_pa == !0u64 {
		hang::fault_halt("alloc_map_umode_stack: out of pages\n");
	}
	let bottom_va = UMODE_STACK_TOP_VA - UMODE_STACK_SIZE_TOTAL;

	for i in 0..pages {
		map_or_halt(
			bottom_va + i * PAGE_SIZE,
			bottom_pa + i * PAGE_SIZE,
			PTE_U | PTE_R | PTE_W,
			LEVEL_PAGE,
			"alloc_map_umode_stack: page table\n",
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
	map_or_halt(
		va_argv,
		argv_pa,
		PTE_U | PTE_R | PTE_W,
		LEVEL_PAGE,
		"map_user_argv: page table\n",
	);
}

// ---------------------------------------------------------------
//  brk（U-mode 堆扩展）
// ---------------------------------------------------------------

/// 按 2 MiB 块把 U-mode VA 区间 [from, to) 建成映射, 返回实际映射到的 VA 上界。
///
/// 一个 2 MiB VA 块只能由单一粒度覆盖。块内先落了 4 KiB 池页, 再在块内安装
/// 2 MiB 超页会覆盖整块的叶子项; 反之超页之下不存在下一级页表, 块内再落 4 KiB
/// 页找不到叶子。旧实现两种粒度在同块内混用, 载荷一次越界请求即可命中,
/// 而当时的处理是直接挂起运行时 —— 整个飞地因此失联, 对宿主表现为 (suspended)。
///
/// 故以块为单位在池页与 M-mode 超页之间二选一: 池中不足一整块时放弃余量,
/// 此后全部走 M-mode。块一经使用即不再回退, 同一 VA 永不二次映射。
pub fn grow_umode_va(from: u64, to: u64, flags: u8) -> u64 {
	let mut va = from;

	while va < to {
		if umode_pool_avail() >= CHUNK_2M_SIZE {
			let pages = CHUNK_2M_SIZE >> PAGE_SHIFT;
			let pa = alloc_umode_page(pages);
			if pa == !0u64 {
				return va;
			}
			for i in 0..pages {
				let r = paging::map_page(
					va + i * PAGE_SIZE,
					pa + i * PAGE_SIZE,
					flags,
					LEVEL_PAGE
				);
				if r.is_err() {
					return va + i * PAGE_SIZE;
				}
			}
			va += CHUNK_2M_SIZE;
			continue;
		}

		let want = (to - va) / CHUNK_2M_SIZE;
		let (granted, pa) = ecall_aux::enclave_call_mem_alloc(want);
		if granted == 0 {
			return va;
		}
		for i in 0..granted {
			let r = paging::map_page(
				va + i * CHUNK_2M_SIZE,
				pa + i * CHUNK_2M_SIZE,
				flags,
				LEVEL_MEGA,
			);
			if r.is_err() {
				return va + i * CHUNK_2M_SIZE;
			}
		}
		va += granted * CHUNK_2M_SIZE;
	}

	va
}

/// 在指定 VA 处按 4 KiB 粒度映射 `pages` 页 (MAP_FIXED 用), 页从 U-mode 页池取。
///
/// 只用于调用方指定地址的场景: 该区间不被本运行时的游标管理, 故不做块独占,
/// 落页失败 (池耗尽或该处已被超页覆盖) 一律返回 false, 由调用方回 ENOMEM。
pub fn map_umode_pages(va: u64, pages: u64, flags: u8) -> bool {
	let pa = alloc_umode_page(pages);
	if pa == !0u64 {
		return false;
	}
	for i in 0..pages {
		if paging::map_page(va + i * PAGE_SIZE, pa + i * PAGE_SIZE, flags, LEVEL_PAGE).is_err() {
			return false;
		}
	}
	true
}

// ---------------------------------------------------------------
//  mmap 匿名映射区
// ---------------------------------------------------------------

/// 在 [va, va + pages * PAGE_SIZE) 建立 4 KiB 映射, 物理页自 `pa` 起连续。
///
/// 只用于后备物理内存由调用方保管的场景 (按块切分的 mmap 区): 页的物理地址由
/// 块的物理基址加块内偏移算出, 不经页池, 故不能走 map_umode_pages。
pub fn map_pages_at(va: u64, pa: u64, pages: u64, flags: u8) -> bool {
	for i in 0..pages {
		if paging::map_page(va + i * PAGE_SIZE, pa + i * PAGE_SIZE, flags, LEVEL_PAGE).is_err() {
			return false;
		}
	}
	true
}

/// 为一个按 4 KiB 页切分的 mmap 块取一段连续 2 MiB 物理内存, 返回其物理基址。
///
/// 优先取 U-mode 页池中的整块, 池中不足一整块时向 M-mode 申请一个分区。
/// 失败返回 `!0u64`。
pub fn alloc_mmap_chunk_pa() -> u64 {
	if umode_pool_avail() >= CHUNK_2M_SIZE {
		let pa = alloc_umode_page(CHUNK_2M_PAGES);
		if pa != !0u64 {
			return pa;
		}
	}
	let (granted, pa) = ecall_aux::enclave_call_mem_alloc(1);
	if granted == 0 {
		return !0u64;
	}
	pa
}

/// 从 mmap 匿名映射区取一段 VA 并建成映射, 返回起始 VA; 失败返回 0。
///
/// 映射区按 4 KiB 页计数推进 (umode_mmap_pages_used), 映射 VA 与块边界都由该
/// 计数导出。请求大小只决定交付多少页, 不改变块的划分: 长度非 4 KiB 对齐时向上
/// 取整到页 (与 Linux 一致), 任意长度的请求都可能跨块, 跨块时为下一个块取后备
/// 内存后继续。不足一整块的请求与同一块的其它请求共享该块的后备物理内存 ——
/// 一个 2 MiB 块最多拆成 512 次 4 KiB 页使用。
pub fn take_mmap_region(bytes: u64, pte_flags: u8) -> u64 {
	let pages = page_up(bytes) >> PAGE_SHIFT;
	if pages == 0 {
		return 0;
	}

	if pages >= CHUNK_2M_PAGES {
		// 达到一整块及以上: 用 2 MiB 超页覆盖。超页叶子之下不存在下一级页表,
		// 块内不能再落 4 KiB 页, 故游标不在块边界时先推进到块边界, 当前块的余量
		// 作废 (只损失地址空间, 无物理代价), 并清空切分块记录使后续小块另取后备
		// 内存。
		let used = context::ctx().umode_mmap_pages_used;
		let aligned = if used & (CHUNK_2M_PAGES - 1) != 0 {
			context::ctx_mut().umode_curr_mmap_pa = 0;
			((used >> CHUNK_2M_SHIFT) + 1) << CHUNK_2M_SHIFT
		} else {
			used
		};
		let blocks = chunk_2m_up(pages * PAGE_SIZE) / CHUNK_2M_SIZE;
		let va = UMODE_MMAP_BASE + aligned * PAGE_SIZE;
		let end = va + blocks * CHUNK_2M_SIZE;
		if grow_umode_va(va, end, pte_flags) < end {
			return 0;
		}
		context::ctx_mut().umode_mmap_pages_used = aligned + (blocks << CHUNK_2M_SHIFT);
		return va;
	}

	// 不足一整块: 在块内按 4 KiB 页交付, 游标每跨过块边界就为新的块取后备内存。
	let va = UMODE_MMAP_BASE + context::ctx().umode_mmap_pages_used * PAGE_SIZE;
	let mut done = 0u64;
	while done < pages {
		let in_block = context::ctx().umode_mmap_pages_used & (CHUNK_2M_PAGES - 1);
		if in_block == 0 {
			let pa = alloc_mmap_chunk_pa();
			if pa == !0u64 {
				break;
			}
			context::ctx_mut().umode_curr_mmap_pa = pa;
		}
		let room = CHUNK_2M_PAGES - in_block;
		let n = if pages - done < room { pages - done } else { room };
		let pa = context::ctx().umode_curr_mmap_pa + in_block * PAGE_SIZE;
		if !map_pages_at(va + done * PAGE_SIZE, pa, n, pte_flags) {
			break;
		}
		context::ctx_mut().umode_mmap_pages_used += n;
		done += n;
	}

	// 未交付满: 已交付的部分保留 (游标按实交付页数推进, 不会被二次发放),
	// 返回 0 由调用方回 ENOMEM。
	if done < pages {
		return 0;
	}
	va
}

/// mremap 的原地扩容/收缩: 成功返回 true, 调用方沿用原地址。
///
/// 仅当该映射是映射区的最后一段 (游标恰在其末尾) 时才成立。旧实现只检查「末页
/// 已映射」—— 每次 mmap 独占整块时该条件与前者等价, 而块内切分后同一块的后续
/// 映射也满足它, 原地扩容会覆盖后继映射。mremap 不带 prot, 新增页沿用原映射
/// 末页的权限位。
pub fn grow_mmap_in_place(addr: u64, old_size: u64, new_size: u64) -> bool {
	let old_end = page_up(addr + old_size);
	let new_end = page_up(addr + new_size);

	// 收缩: 只回退游标, 页表项与物理内存保留 (与 munmap 一致, 不回收物理页)。
	if new_end <= old_end {
		if UMODE_MMAP_BASE + context::ctx().umode_mmap_pages_used * PAGE_SIZE != old_end {
			return false;
		}
		context::ctx_mut().umode_mmap_pages_used = (new_end - UMODE_MMAP_BASE) >> PAGE_SHIFT;
		return true;
	}

	if context::ctx().umode_curr_mmap_pa == 0
		|| UMODE_MMAP_BASE + context::ctx().umode_mmap_pages_used * PAGE_SIZE != old_end
	{
		return false;
	}
	// 扩容必须留在同一块内: 块内偏移与后备物理内存一一对应, 换块需先取新的后备
	// 内存, 那属于 mremap 的搬移路径。
	if chunk_2m_down(old_end) != chunk_2m_down(new_end - PAGE_SIZE) {
		return false;
	}
	let flags = match paging::leaf_pte_flags(old_end - PAGE_SIZE) {
		Some(f) => f,
		None => return false,
	};
	let pa = context::ctx().umode_curr_mmap_pa + (old_end - chunk_2m_down(old_end));
	let pages = (new_end - old_end) >> PAGE_SHIFT;
	if !map_pages_at(old_end, pa, pages, flags) {
		return false;
	}
	context::ctx_mut().umode_mmap_pages_used = (new_end - UMODE_MMAP_BASE) >> PAGE_SHIFT;
	true
}

pub fn sys_brk_handler(new_brk: u64) -> u64 {
	let old_top = context::ctx().umode_heap_top;
	if new_brk == 0 {
		return old_top; // 仅查询
	}

	if new_brk <= old_top {
		// 收缩或不变: 只回退 brk。已映射的物理页不回收 —— 页池是 bump 分配器,
		// 2 MiB 池内存统一由 M-mode 在飞地 shutdown 时整块回收。
		context::ctx_mut().umode_heap_top = new_brk;
		return new_brk;
	}

	// 向上扩展: 堆的已映射上界按 2 MiB 块推进, 与 brk 值解耦。
	let need_end = chunk_2m_up(page_up(new_brk));
	let mapped_end = context::ctx().umode_heap_mapped_end;

	if need_end > mapped_end {
		let got = grow_umode_va(mapped_end, need_end, (PTE_U | PTE_R | PTE_W) as u8);
		context::ctx_mut().umode_heap_mapped_end = got;
		if got < need_end {
			// 增长未完全满足 (M-mode 拒绝或池耗尽): 返回旧 brk, 让 musl 判定失败
			// (malloc 返回 NULL), 而不是返回未映射的 new_brk 造成后续缺页。
			return old_top;
		}
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
