//! 内存类系统调用: mmap(222) / munmap(215) / mremap(216)。
//! brk(214) 逻辑仍在 crate::mem::sys_brk_handler，此处仅 re-export。

use crate::constants::{
	CHUNK_2M_SIZE, ENCLAVE_MEM_ATTR_DATA, ENCLAVE_MEM_ATTR_INSTR, LEVEL_GIGA, LEVEL_MEGA,
	LEVEL_PAGE, PAGE_SHIFT, PAGE_SIZE, PTE_R, PTE_RSW_SHARED, PTE_U, PTE_V, PTE_W, PTE_X,
	UMODE_VA_LIMIT,
};
use crate::mem;
use crate::paging;
use crate::prim::arith::align_up;

use super::{EINVAL, ENOMEM, ENOSYS};

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
///
/// 请求同时带 PROT_WRITE 与 PROT_EXEC 时返回 EINVAL: 一个 2 MiB 分区整体属于一个
/// 类别, 类别即该分区登记的 PMP 权限, 可写与可执行不能落在同一分区上。
pub fn mmap_handler(addr: u64, len: u64, prot: u64, flags: u64, _fd: u64, _off: u64) -> u64 {
	// 仅支持匿名映射
	if (flags & MAP_ANONYMOUS) == 0 {
		return ENOSYS;
	}

	if (prot & (PROT_WRITE | PROT_EXEC)) == (PROT_WRITE | PROT_EXEC) {
		return EINVAL;
	}

	let map_type = flags & MAP_TYPE;
	if map_type != MAP_SHARED && map_type != MAP_PRIVATE {
		return EINVAL;
	}

	// 长度越界时按页取整会向上溢出而给出一个小值, 后续算式因此接受一个远大于用户
	// 空间的区间。提前拒绝使算式只在无溢出的区间内求值。
	if len == 0 || len > UMODE_VA_LIMIT {
		return EINVAL;
	}
	let pages = align_up(len, PAGE_SIZE) >> PAGE_SHIFT;
	let total_bytes = pages * PAGE_SIZE;
	let mut pte_flags = prot_to_pte(prot);
	if map_type == MAP_SHARED {
		pte_flags |= PTE_RSW_SHARED;
	}

	// MAP_FIXED: 调用方指定地址, 按 4 KiB 粒度就地映射。Linux 要求该地址按页对齐,
	// 未对齐返回 EINVAL; 区间落在用户空间之外时无法建立映射, 返回 ENOMEM。
	if (flags & MAP_FIXED) != 0 {
		if (addr & (PAGE_SIZE - 1)) != 0 {
			return EINVAL;
		}
		let va = addr;
		if !mem::is_umode_va_range(va, total_bytes) {
			return ENOMEM;
		}
		if !mem::map_umode_pages(va, pages, pte_flags) || !seal_instr_mapping(va, pages, pte_flags)
		{
			return ENOMEM;
		}
		return va;
	}

	// 常规路径: 由映射区的页计数决定放置与粒度。
	let va = mem::take_mmap_region(total_bytes, pte_flags);
	if va == 0 || !seal_instr_mapping(va, pages, pte_flags) {
		return ENOMEM;
	}
	va
}

// ---------------------------------------------------------------
//  munmap handler (215)
// ---------------------------------------------------------------

/// 解除 [addr, addr + len) 的 VA 映射, 使后续访问落入 access fault 由 S/M 侧裁决。
///
/// 物理页不在本函数归还: U 模式的物理内存由 M 模式按 2 MiB 分区交付, 本运行时只申请
/// 不归还, 交付的整块内存统一由 M 模式在飞地终止时经 clear_enclave_mem_region 回收。
/// 此处只负责清空页表项并刷新翻译缓存, 与运行时的「不信任应用 free、终止时整体回收」
/// 约定一致。
pub fn munmap_handler(addr: u64, len: u64) -> u64 {
	// 参数合法性: Linux munmap 要求地址 4 KiB 对齐且长度非零, 违者 EINVAL。
	if (addr & (PAGE_SIZE - 1)) != 0 || len == 0 {
		return EINVAL;
	}
	// 区间越过用户空间上界时, 页表遍历按 VPN 的 9 位字段索引根表, 会落到管理器自身
	// 占用的根表项上; 上方溢出同样使末址失去意义。两种情形均返回 EINVAL。
	if !mem::is_umode_va_range(addr, len) {
		return EINVAL;
	}

	let start = addr;
	let end = align_up(addr + len, PAGE_SIZE);

	// 逐段解除映射, 粒度跟随页表叶子层级:
	//   4 KiB 普通页单页解除; 2 MiB / 1 GiB 超页整块解除并跳过整块,
	//   避免在2MB超级页内部再次解引用。
	let giga_size = 0x4000_0000_u64;
	let mut va = start;
	while va < end {
		match paging::va_real_page_level(va) {
			None => {
				va += PAGE_SIZE;
			}
			// 本就未映射, 直接跳过
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

/// 请求区间内是否存在可执行位与 `wants_exec` 不一致的叶子项。
///
/// 4 KiB 叶子逐个比较, 2 MiB 超页整张比较一次。未映射之处与层级不明的叶子返回
/// false, 由随后的改写循环报出错误。
fn range_changes_exec(start: u64, end: u64, wants_exec: bool) -> bool {
	let mut va = start;
	while va < end {
		let Some(level) = paging::va_real_page_level(va) else {
			return false;
		};
		if level != LEVEL_PAGE && level != LEVEL_MEGA {
			return false;
		}
		let Some(flags) = paging::leaf_pte_flags(va) else {
			return false;
		};
		if ((flags & PTE_X) != 0) != wants_exec {
			return true;
		}
		va += if level == LEVEL_MEGA {
			CHUNK_2M_SIZE
		} else {
			PAGE_SIZE
		};
	}
	false
}

/// 把 `[start, end)` 覆盖的每个 2 MiB 分区改判为 `attr`。
///
/// 区间端点按 2 MiB 对齐时, 每个 2 MiB 虚拟块整体落在一个物理分区内, 故由块首的
/// 虚拟地址查得的物理地址下取整到分区边界即是该分区的物理起始地址。
///
/// 改判前先核对覆盖的每个分区都可改判: 分区由本次映射独占, 不与飞地管理器镜像
/// 自身占用的分区共用。任一分区不可改判时返回错误码, 且不改动任何分区。
/// 返回 0 表示全部改判成功, 其余取值为错误码。
fn reclass_covered_partitions(start: u64, end: u64, attr: u64) -> u64 {
	let mut va = start;
	while va < end {
		let Some(pa) = paging::get_pa(va) else {
			return ENOMEM;
		};
		if !mem::is_dedicated_partition(pa) {
			return EINVAL;
		}
		va += CHUNK_2M_SIZE;
	}

	va = start;
	while va < end {
		let Some(pa) = paging::get_pa(va) else {
			return ENOMEM;
		};
		if !mem::set_partition_attr(pa, CHUNK_2M_SIZE, attr) {
			return EINVAL;
		}
		va += CHUNK_2M_SIZE;
	}
	0
}

/// 把指令类映射封口: 覆盖的每个分区改判为指令类, 叶子项恢复为请求的权限。
///
/// 指令类映射的物理区在取得时按数据类登记并映射为可写。mmap 不向新映射写入内容,
/// 取得后立即封口; mremap 的私有映射搬移在复制完成后封口。映射不带 PTE_X 时不涉及
/// 改判, 直接返回 true。
fn seal_instr_mapping(va: u64, pages: u64, flags: u64) -> bool {
	if (flags & PTE_X) == 0 {
		return true;
	}
	if reclass_covered_partitions(va, va + pages * PAGE_SIZE, ENCLAVE_MEM_ATTR_INSTR) != 0 {
		return false;
	}
	for i in 0..pages {
		if !paging::set_leaf_flags(va + i * PAGE_SIZE, flags) {
			return false;
		}
	}
	true
}

/// mprotect(226) 处理函数。
///
/// musl 的线程栈按 mmap(PROT_NONE) 之后 mprotect(PROT_READ | PROT_WRITE) 两步取得,
/// 本调用必须真正改写页表项: 只返回 0 而不改映射会使栈保持 R/W/X 全 0 的无权限编码,
/// 线程首次写入线程控制块即触发存储页错误。
///
/// 粒度跟随页表叶子层级: 4 KiB 叶子就地改写标志位; 2 MiB 超页在请求区间覆盖整张
/// 超页时就地改写, 只覆盖一部分时先拆成 4 KiB 叶子再逐页改写。
///
/// 改可执行位另有两项约束。其一, 2 MiB 分区的类别由该分区内映射的可执行位决定,
/// 故改判可执行位即改判分区类别, 要求 `addr` 与 `len` 均按 2 MiB 对齐, 改判与改写
/// 一并落到整数个分区上; 未对齐即返回 EINVAL, 不触及任何映射。其二, 覆盖的每个分区
/// 都须由本次映射独占, 落在飞地管理器镜像自身所占分区之内的同样返回 EINVAL。
pub fn mprotect_handler(addr: u64, len: u64, prot: u64) -> u64 {
	if (addr & (PAGE_SIZE - 1)) != 0 {
		return EINVAL;
	}
	if (prot & !(PROT_READ | PROT_WRITE | PROT_EXEC)) != 0 {
		return EINVAL;
	}
	if (prot & (PROT_WRITE | PROT_EXEC)) == (PROT_WRITE | PROT_EXEC) {
		return EINVAL;
	}
	if len == 0 {
		return 0;
	}
	// 与 munmap 同一判据: 越界的区间在根表上会落到管理器自身的项。
	if !mem::is_umode_va_range(addr, len) {
		return EINVAL;
	}

	let start = addr;
	let end = align_up(addr + len, PAGE_SIZE);
	let flags = prot_to_pte(prot);

	if range_changes_exec(start, end, (prot & PROT_EXEC) != 0) {
		if (start & (CHUNK_2M_SIZE - 1)) != 0 || (len & (CHUNK_2M_SIZE - 1)) != 0 {
			return EINVAL;
		}
		let attr = if (prot & PROT_EXEC) != 0 {
			ENCLAVE_MEM_ATTR_INSTR
		} else {
			ENCLAVE_MEM_ATTR_DATA
		};
		let err = reclass_covered_partitions(start, end, attr);
		if err != 0 {
			return err;
		}
	}

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
fn check_mremap_params(
	old_addr: u64,
	old_size: u64,
	new_size: u64,
	flags: u64,
	new_addr: u64,
) -> u64 {
	if (flags & !MREMAP_KNOWN_FLAGS) != 0 {
		return EINVAL;
	}
	if (old_addr & (PAGE_SIZE - 1)) != 0 {
		return EINVAL;
	}
	if old_size == 0 || new_size == 0 {
		return EINVAL;
	}
	// 旧区间越出用户空间, 或新长度已不可能落在用户空间内时, 后续的末址算式与页表
	// 遍历都不成立。新长度是否放得进映射区由 mem::take_mmap_region 判定。
	if !mem::is_umode_va_range(old_addr, old_size) || new_size > UMODE_VA_LIMIT {
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
		// 目的区间须落在用户空间内, 与 mmap 的 MAP_FIXED 同一判据与错误码。
		if !mem::is_umode_va_range(new_addr, new_size) {
			return ENOMEM;
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
pub fn mremap_handler(
	old_addr: u64,
	old_size: u64,
	new_size: u64,
	flags: u64,
	new_addr: u64,
) -> u64 {
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
	let pages = align_up(new_size, PAGE_SIZE) >> PAGE_SHIFT;
	// 共享映射搬到固定地址时目的与源指向同一批物理页, 没有新建的映射区要封口。
	let is_src_alias = is_shared && (flags & MREMAP_FIXED) != 0;

	// 另择地址: 私有映射复制仍有效的字节数再解除旧映射, 共享映射改为让目的地址
	// 指向源映射的同一批物理页。
	let dst = if (flags & MREMAP_FIXED) != 0 {
		let va = new_addr;
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

	let copy_len = if old_size < new_size {
		old_size
	} else {
		new_size
	};
	if !is_shared {
		unsafe {
			core::ptr::copy_nonoverlapping(
				old_addr as *const u8,
				dst as *mut u8,
				copy_len as usize,
			);
		}
	}

	// 指令类映射的物理区在取得时按数据类登记并映射为可写, 复制完成后封口。
	if !is_src_alias && !seal_instr_mapping(dst, pages, flags_from_src) {
		return ENOMEM;
	}

	// MREMAP_FIXED 的目标由调用方指定, 旧区间保留由调用方自理。
	if (flags & MREMAP_FIXED) == 0 {
		munmap_handler(old_addr, old_size);
	}

	dst
}
