//! 内存类系统调用: mmap(222) / munmap(215) / mremap(216)。
//! brk(214) 逻辑仍在 crate::mem::sys_brk_handler，此处仅 re-export。

use crate::constants::{
	CHUNK_2M_SIZE,
	LEVEL_GIGA,
	LEVEL_MEGA,
	LEVEL_PAGE,
	PAGE_SHIFT,
	PAGE_SIZE,
	PTE_R,
	PTE_RSW_SHARED,
	PTE_U,
	PTE_V,
	PTE_W,
	PTE_X,
};
use crate::mem;
use crate::mem_prim::align::{ align_down, align_up };
use crate::paging;

use super::{ EINVAL, ENOMEM, ENOSYS };

// ---------------------------------------------------------------
//  mmap 常量 (Linux rv64 ABI)
// ---------------------------------------------------------------

const MAP_FIXED: u64 = 0x10;
const MAP_ANONYMOUS: u64 = 0x20;
/// 与 MAP_PRIVATE 同为映射类型位, 二者互斥且必须给出其一。取 MAP_TYPE 掩码比对,
/// 与 Linux 的 `switch (flags & MAP_TYPE)` 一致。
const MAP_SHARED: u64 = 0x01;
const MAP_PRIVATE: u64 = 0x02;
const MAP_TYPE: u64 = 0x0f;

const PROT_READ: u64 = 0x1;
const PROT_WRITE: u64 = 0x2;
const PROT_EXEC: u64 = 0x4;

const MREMAP_MAYMOVE: u64 = 0x1;
const MREMAP_FIXED: u64 = 0x2;

/// 本实现识别的 MREMAP_* 位集合。出现其余位返回 EINVAL, 与 Linux
/// `check_mremap_params()` 的首条判据一致。
const MREMAP_KNOWN_FLAGS: u64 = MREMAP_MAYMOVE | MREMAP_FIXED;

/// prot 位图 -> PTE 标志。PROT_NONE 得到一个 R/W/X 全 0 的叶子项:
/// 硬件在末级遇到此编码即报页错误, 正是 PROT_NONE 的语义。
fn prot_to_pte(prot: u64) -> u64 {
	let mut flags: u64 = PTE_U | PTE_V;
	if (prot & PROT_READ) != 0 {
		flags |= PTE_R;
	}
	if (prot & PROT_WRITE) != 0 {
		flags |= PTE_W;
	}
	if (prot & PROT_EXEC) != 0 {
		flags |= PTE_X;
	}
	flags
}

// ---------------------------------------------------------------
//  mmap handler (222)
// ---------------------------------------------------------------

/// mmap(222) 处理函数。
///
/// 映射类型位取 MAP_SHARED 时在叶子项上置 [`PTE_RSW_SHARED`], fork 复制地址空间
/// 时带该位的用户叶子共用同一物理页, 父子进程的写入互相可见; 取 MAP_PRIVATE 时
/// 不带该位, 每个进程各自持有副本。两者都只支持匿名映射, 带文件描述符的调用返回
/// ENOSYS。映射类型位取其余取值时返回 EINVAL。
pub fn mmap_handler(addr: u64, len: u64, prot: u64, flags: u64, _fd: u64, _off: u64) -> u64 {
	// 仅支持匿名映射
	if (flags & MAP_ANONYMOUS) == 0 {
		return ENOSYS;
	}

	let map_type = flags & MAP_TYPE;
	if map_type != MAP_SHARED && map_type != MAP_PRIVATE {
		return EINVAL;
	}

	let pages = align_up(len, PAGE_SIZE) >> PAGE_SHIFT;
	if pages == 0 {
		return EINVAL;
	}
	let total_bytes = pages * PAGE_SIZE;
	let mut pte_flags = prot_to_pte(prot);
	if map_type == MAP_SHARED {
		pte_flags |= PTE_RSW_SHARED;
	}

	// MAP_FIXED: 调用方指定地址, 按 4 KiB 粒度就地映射。
	if (flags & MAP_FIXED) != 0 {
		let va = align_down(addr, PAGE_SIZE);
		if !mem::map_umode_pages(va, pages, pte_flags) {
			return ENOMEM;
		}
		return va;
	}

	// 常规路径: 由映射区的页计数决定放置与粒度 (见 mem::take_mmap_region)。
	let va = mem::take_mmap_region(total_bytes, pte_flags);
	if va == 0 {
		return ENOMEM;
	}
	va
}

// ---------------------------------------------------------------
//  munmap handler (215)
// ---------------------------------------------------------------

/// 解除 [addr, addr + len) 的 VA 映射, 使后续访问落入 access fault 由 S/M 侧裁决。
///
/// 物理页不在本函数归还: umode 页池是 bump 分配器 (无单页归还能力), 2 MiB 池内存
/// 统一由 M-mode 在飞地 shutdown 时经 clear_enclave_mem_region 整块回收。此处只负责
/// 清空页表项并刷新翻译缓存, 与运行时的「不信任应用 free、shutdown 整体回收」约定一致。
pub fn munmap_handler(addr: u64, len: u64) -> u64 {
	// 参数合法性: Linux munmap 要求地址 4 KiB 对齐且长度非零, 违者 EINVAL。
	if (addr & (PAGE_SIZE - 1)) != 0 || len == 0 {
		return EINVAL;
	}

	let start = align_down(addr, PAGE_SIZE);
	let end = align_up(addr + len, PAGE_SIZE);

	// 逐段解除映射, 粒度跟随页表叶子层级:
	//   4 KiB 普通页单页解除; 2 MiB / 1 GiB 超页整块解除并跳过整块,
	//   避免在超页内部再次解引用 (超页之下不存在叶子项)。
	let giga_size = 0x4000_0000_u64;
	let mut va = start;
	while va < end {
		match paging::va_real_page_level(va) {
			None => {
				va += PAGE_SIZE;
			} // 本就未映射, 直接跳过
			Some(LEVEL_PAGE) => {
				paging::unmap_page(va, LEVEL_PAGE);
				va += PAGE_SIZE;
			}
			Some(LEVEL_MEGA) => {
				paging::unmap_page(va, LEVEL_MEGA);
				va += CHUNK_2M_SIZE;
			}
			Some(LEVEL_GIGA) => {
				paging::unmap_page(va, LEVEL_GIGA);
				va += giga_size;
			}
			Some(_) => {
				va += PAGE_SIZE;
			} // 未知层级不应出现, 保守跳过
		}
	}

	0
}

// ---------------------------------------------------------------
//  mprotect handler (226)
// ---------------------------------------------------------------

/// mprotect(226) 处理函数。
///
/// musl 的线程栈按 mmap(PROT_NONE) 之后 mprotect(PROT_READ | PROT_WRITE) 两步取得,
/// 本调用必须真正改写页表项: 只返回 0 而不改映射会使栈保持 R/W/X 全 0 的无权限编码,
/// 线程首次写入线程控制块即触发存储页错误。
///
/// 粒度跟随页表叶子层级: 4 KiB 叶子就地改写标志位; 2 MiB 超页在请求区间覆盖整张
/// 超页时就地改写, 只覆盖一部分时先拆成 4 KiB 叶子再逐页改写。
pub fn mprotect_handler(addr: u64, len: u64, prot: u64) -> u64 {
	if (addr & (PAGE_SIZE - 1)) != 0 {
		return EINVAL;
	}
	if (prot & !(PROT_READ | PROT_WRITE | PROT_EXEC)) != 0 {
		return EINVAL;
	}
	if len == 0 {
		return 0;
	}

	let start = addr;
	let end = align_up(addr + len, PAGE_SIZE);
	let flags = prot_to_pte(prot);

	let mut va = start;
	while va < end {
		match paging::va_real_page_level(va) {
			None => {
				return ENOMEM;
			}
			Some(LEVEL_PAGE) => {
				if !paging::set_leaf_flags(va, flags) {
					return ENOMEM;
				}
				va += PAGE_SIZE;
			}
			Some(LEVEL_MEGA) => {
				let chunk = va & !(CHUNK_2M_SIZE - 1);
				if start <= chunk && end >= chunk + CHUNK_2M_SIZE {
					if !paging::set_leaf_flags(va, flags) {
						return ENOMEM;
					}
					va += CHUNK_2M_SIZE;
				} else if !paging::split_mega_leaf(va) {
					return ENOMEM;
				}
			}
			Some(_) => {
				return ENOMEM;
			}
		}
	}
	0
}

// ---------------------------------------------------------------
//  mremap handler (216)
// ---------------------------------------------------------------

/// mremap 的参数校验, 判据与 Linux `check_mremap_params()` 一致: 未知标志位、
/// 起始地址未按页对齐、新长度为 0 都返回 EINVAL; 指定新地址的调用还要求新地址按页
/// 对齐且同时给出 MREMAP_MAYMOVE。返回 0 表示通过。
///
/// 与 Linux 的一处差别: Linux 允许 old_size 取 0, 此时整段映射按新建立处理, 只用于
/// 复制共享映射区。本实现没有可供复制的共享后备, 故该取值一并返回 EINVAL。
fn check_mremap_params(old_addr: u64, old_size: u64, new_size: u64, flags: u64, new_addr: u64) -> u64 {
	if (flags & !MREMAP_KNOWN_FLAGS) != 0 {
		return EINVAL;
	}
	if (old_addr & (PAGE_SIZE - 1)) != 0 {
		return EINVAL;
	}
	if old_size == 0 || new_size == 0 {
		return EINVAL;
	}
	if (flags & MREMAP_FIXED) != 0 {
		if (new_addr & (PAGE_SIZE - 1)) != 0 {
			return EINVAL;
		}
		// 指定固定地址意味着必然搬移, 故 MREMAP_FIXED 蕴含 MREMAP_MAYMOVE。
		if (flags & MREMAP_MAYMOVE) == 0 {
			return EINVAL;
		}
	}
	0
}

/// mremap(216)。musl 的 realloc 对 mmap 组用 MREMAP_MAYMOVE 扩容。
///
/// 旧实现把它并入「跳过 (无副作用)」一律返回 0, 而 0 在指针语义下是 NULL,
/// realloc 据此判失败 —— 增长型缓冲的使用方 (如 cJSON_Print 的打印缓冲)
/// 直接拿到 NULL。
///
/// 搬移路径的目的映射沿用源映射末页的标志位, 故源为共享映射时目的仍与源共用同一
/// 物理页, 源为私有映射时才逐字节复制。
pub fn mremap_handler(old_addr: u64, old_size: u64, new_size: u64, flags: u64, new_addr: u64) -> u64 {
	let invalid = check_mremap_params(old_addr, old_size, new_size, flags, new_addr);
	if invalid != 0 {
		return invalid;
	}

	// 原地扩容/收缩: 判断条件与页映射都在 memory 侧 (需要映射区的页计数与后备
	// 块物理基址)。
	if (flags & MREMAP_FIXED) == 0 && mem::grow_mmap_in_place(old_addr, old_size, new_size) {
		return old_addr;
	}

	if (flags & MREMAP_MAYMOVE) == 0 {
		return ENOMEM;
	}

	// 目的映射的标志位取源映射末页: mremap 不带 prot, 新映射的权限与共享属性都
	// 只能沿用源映射。
	let flags_from_src = match paging::leaf_pte_flags(old_addr + old_size - PAGE_SIZE) {
		Some(f) => f,
		None => PTE_U | PTE_V | PTE_R | PTE_W,
	};
	let is_shared = (flags_from_src & PTE_RSW_SHARED) != 0;

	// 另择地址: 私有映射复制仍有效的字节数再解除旧映射, 共享映射改为让目的地址
	// 指向源映射的同一批物理页。
	let dst = if (flags & MREMAP_FIXED) != 0 {
		let va = new_addr;
		let pages = align_up(new_size, PAGE_SIZE) >> PAGE_SHIFT;
		let ok = if is_shared {
			mem::alias_umode_pages(old_addr, va, pages, flags_from_src)
		} else {
			mem::map_umode_pages(va, pages, flags_from_src)
		};
		if !ok {
			return ENOMEM;
		}
		va
	} else {
		let va = mem::take_mmap_region(new_size, flags_from_src);
		if va == 0 {
			return ENOMEM;
		}
		va
	};

	let copy_len = if old_size < new_size { old_size } else { new_size };
	if !is_shared {
		unsafe {
			core::ptr::copy_nonoverlapping(
				old_addr as *const u8,
				dst as *mut u8,
				copy_len as usize
			);
		}
	}

	// MREMAP_FIXED 的目标由调用方指定, 旧区间保留由调用方自理。
	if (flags & MREMAP_FIXED) == 0 {
		munmap_handler(old_addr, old_size);
	}

	dst
}
