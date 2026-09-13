//! Sv39 页表管理。逻辑紧跟 ref-emod/emod_manager/memory/page_table.c，
//! 但用卫语句替代原 C 的深层嵌套。
//!
//! 关键约定：
//! - MMU 开启后（satp != 0），中间页表指针通过 LINEAR_MAP_OFFSET 换算。
//! - A/D 位始终置 1（无 swap）。
//! - map / unmap 后 sfence.vma 刷新对应 VA。

use crate::constants::*;
use crate::context;
use crate::csr;
use crate::mem;

// ---------------------------------------------------------------
//  Sv39 PTE
// ---------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct Pte(pub u64);

impl Pte {
	#[cfg(feature = "diagnostic")]
	pub const fn empty() -> Self {
		Self(0)
	}

	/// R | W | X 任一置位即为叶子节点。
	pub fn is_leaf(&self) -> bool {
		self.0 & ((PTE_R | PTE_W | PTE_X) as u64) != 0
	}

	/// 提取 PPN（bits [53:10]）。
	pub fn ppn(&self) -> u64 {
		(self.0 >> 10) & 0xF_FFFF_FFFF
	}

	/// 写入 PPN，保留低 10 位标志。
	pub fn set_ppn(&mut self, ppn: u64) {
		self.0 = (self.0 & 0x3FF) | ((ppn & 0xF_FFFF_FFFF) << 10);
	}

	/// 写入标志位，自动置 A/D。
	pub fn set_flags(&mut self, flags: u8) {
		self.0 |= flags as u64;
		self.0 |= (PTE_A | PTE_D) as u64;
	}

	pub fn clear(&mut self) {
		self.0 = 0;
	}
}

// ---------------------------------------------------------------
//  VA 辅助
// ---------------------------------------------------------------

/// level 0 -> VPN[2] (bits 38:30), 1 -> VPN[1] (29:21), 2 -> VPN[0] (20:12)
#[inline]
fn get_vpn(va: u64, level: u8) -> usize {
	match level {
		0 => ((va >> 30) & 0x1FF) as usize,
		1 => ((va >> 21) & 0x1FF) as usize,
		2 => ((va >> 12) & 0x1FF) as usize,
		_ => 0,
	}
}

/// 物理地址 -> PTE 指针。satp 有效时叠加 LINEAR_MAP_OFFSET。
#[inline]
unsafe fn pte_ptr(pa: u64) -> *mut Pte {
	if csr::read_satp() != 0 {
		pa.wrapping_add(LINEAR_MAP_OFFSET) as *mut Pte
	} else {
		pa as *mut Pte
	}
}

/// 页表根的可解引用指针 (直接以当前映射地址解引用, 不经 pte_ptr)。
///
/// `root_pa()` 返回的是**当前映射下**根表地址: 物理恒等主流程为 PA,
/// 虚拟陷态处理为 VA, 二者均已被映射, 可直接解引用。根表不能像中间表
/// 那样套 pte_ptr: 中间表拿到的恒为 PA, satp 有效时叠加 LINEAR 才对;
/// 根表在虚拟上下文里已经是 VA, 再叠加 LINEAR 会落入 0xffffffa0...
/// 未映射区, 触发页错误 (曾致 brk 扩容走表时 M/S 反复重定向死循环)。
#[inline]
fn root_table() -> *mut Pte {
	context::root_pa() as *mut Pte
}

// ---------------------------------------------------------------
//  页表遍历
// ---------------------------------------------------------------

/// 页表项安装的失败原因。
///
/// 一律经返回值上报, 不挂起运行时: brk / mmap / mremap 由载荷发起, 其请求
/// 不得让可信管理程序自身停止服务。旧实现在这些路径上直接 WFI 冻结, 载荷
/// 一次越界的内存请求即可使整个飞地失联, 对宿主表现为 (suspended), 与载荷
/// 仍在计算无法区分。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MapError {
	/// 目标 VA 已被更粗粒度的叶子覆盖, 其下不存在下一级页表。
	CoveredBySuperPage,
	/// 中间页表或目标 PTE 不存在, 且调用方未要求按需分配。
	NotMapped,
	/// 页表页分配失败 (S-mode 页池耗尽)。
	OutOfPool,
}

impl MapError {
	/// 供启动期诊断输出使用的可读名称 (纯 ASCII)。
	pub fn name(&self) -> &'static str {
		match self {
			MapError::CoveredBySuperPage => "covered by super-page",
			MapError::NotMapped => "not mapped",
			MapError::OutOfPool => "s-mode page pool exhausted",
		}
	}
}

/// 定位给定 VA 在指定 Sv39 层级的叶子 PTE。
/// `alloc=true` 时缺失的中间页表从 S-mode page pool 分配。
///
/// 遍历顺序与硬件页表 walker 一致 (标准 Sv39):
///   root[VPN[2]] -> L2[VPN[1]] -> L3[VPN[0]]
/// `get_vpn(va, i)` 的 `i` 是页表层级: 0=根表(GIGA), 1=中表(MEGA), 2=叶表(PAGE)。
///
/// 中间层级若已是叶子 (超级页), 其下不存在页表页可继续下潜, 返回
/// MapError::CoveredBySuperPage, 由调用方决定该 VA 区间是否已满足需求。
fn get_leaf_pte(vaddr: u64, level: u8, alloc: bool) -> Result<&'static mut Pte, MapError> {
	let mut table = root_table();

	// 遍历中间层级: 从根表 (0) 到目标层级的前一级.
	// level=0 (GIGA): 无中间层级, 直接在根表命中.
	// level=1 (MEGA): 遍历根表 (i=0), 在 L2 命中.
	// level=2 (PAGE): 遍历根表 (i=0) + L2 (i=1), 在 L3 命中.
	for i in 0..level {
		let pte = unsafe { &mut *table.add(get_vpn(vaddr, i)) };

		if pte.0 & (PTE_V as u64) == 0 {
			if !alloc {
				return Err(MapError::NotMapped);
			}
			let next_pa = mem::try_alloc_smode_page(1);
			if next_pa == !0u64 {
				return Err(MapError::OutOfPool);
			}
			// 新分配的页表页必须清零, 否则残留数据可能被误判为超级页
			unsafe {
				core::ptr::write_bytes(pte_ptr(next_pa) as *mut u8, 0, PAGE_SIZE as usize);
			}
			pte.set_ppn(next_pa >> PAGE_SHIFT);
			pte.0 |= PTE_V as u64;
			table = unsafe { pte_ptr(next_pa) };
			continue;
		}

		if pte.is_leaf() {
			return Err(MapError::CoveredBySuperPage);
		}

		let next_pa = pte.ppn() << PAGE_SHIFT;
		table = unsafe { pte_ptr(next_pa) };
	}

	// 命中目标层级
	let pte = unsafe { &mut *table.add(get_vpn(vaddr, level)) };
	if pte.0 & (PTE_V as u64) == 0 && !alloc {
		return Err(MapError::NotMapped);
	}
	Ok(pte)
}

// ---------------------------------------------------------------
//  map / unmap
// ---------------------------------------------------------------

/// 安装一条叶子 PTE。目标 PTE 已存在时按新映射覆盖 (重新映射语义)。
pub fn map_page(vaddr: u64, paddr: u64, flags: u8, level: u8) -> Result<(), MapError> {
	let ppn = paddr >> PAGE_SHIFT;
	let pte = get_leaf_pte(vaddr, level, true)?;

	pte.clear();
	pte.set_ppn(ppn);
	pte.set_flags(flags);
	pte.0 |= PTE_V as u64;

	unsafe { core::arch::asm!("sfence.vma {0}, zero", in(reg) vaddr) };
	Ok(())
}

/// 清除一条叶子 PTE。返回该位置原先是否存在映射。
///
/// 未映射 (含被超级页覆盖) 时返回 false 而非报错: 解除映射是幂等操作,
/// 调用方 (munmap) 只需知道有无实际工作。
#[allow(dead_code)]
pub fn unmap_page(vaddr: u64, level: u8) -> bool {
	match get_leaf_pte(vaddr, level, false) {
		Ok(pte) => {
			pte.clear();
			unsafe { core::arch::asm!("sfence.vma {0}, zero", in(reg) vaddr) };
			true
		}
		Err(_) => false,
	}
}

// ---------------------------------------------------------------
//  VA -> PA 翻译
// ---------------------------------------------------------------

#[cfg(feature = "diagnostic")]
struct WalkResult {
	level: i8,
	pte: Pte,
}

/// 三层 Sv39 页表遍历。未命中时 level = -1。
#[cfg(feature = "diagnostic")]
fn walk_page_table(va: u64) -> WalkResult {
	let idxs: [usize; 3] = [
		((va & 0x7F_C000_0000) >> 30) as usize,
		((va & 0x00_3FE0_0000) >> 21) as usize,
		((va & 0x00_001F_F000) >> 12) as usize,
	];

	let mut table = root_table();

	for i in 0..3 {
		let pte = unsafe { &*table.add(idxs[i]) };

		if pte.0 & (PTE_V as u64) == 0 {
			return WalkResult {
				level: -1,
				pte: Pte::empty(),
			};
		}

		if pte.is_leaf() {
			return WalkResult {
				level: i as i8,
				pte: *pte,
			};
		}

		let next_pa = pte.ppn() << PAGE_SHIFT;
		table = unsafe { pte_ptr(next_pa) };
	}

	WalkResult {
		level: -1,
		pte: Pte::empty(),
	}
}

/// VA -> PA。未映射返回 None。
#[cfg(feature = "diagnostic")]
#[allow(dead_code)]
#[allow(unused)]
pub fn get_pa(va: u64) -> Option<u64> {
	let r = walk_page_table(va);
	if r.level < 0 {
		return None;
	}

	let base = r.pte.ppn() << PAGE_SHIFT;
	let offset_bits = PAGE_SHIFT as usize + (2 - r.level as usize) * SV39_VPN_LEN as usize;
	Some(base | ((1_u64 << offset_bits) - 1) & va)
}

// ---------------------------------------------------------------
//  初始化
// ---------------------------------------------------------------

/// 为 MMU 启用的瞬间建立双重映射 (PA=VA + PA+OFFSET->PA)。
///
/// `csrw satp` 后 PC 仍在低物理地址，必须有一条 PA->PA 的映射
/// 让 CPU 能继续取指，直到代码通过高 VA 访问 trampoline。
///
/// 同时建立 LINEAR_MAP_OFFSET 映射，供 `pte_ptr()` 在 MMU 使能后
/// 通过 PA+OFFSET 访问页表结构体 (根表 + 中间表均在池 PA 范围内).
pub fn identity_map_trampoline(pa: u64) -> Result<(), MapError> {
	let base = crate::mem::chunk_2m_down(pa);
	// 代码执行: identity VA=PA
	map_page(base, base, PTE_R | PTE_W | PTE_X, LEVEL_MEGA)?;
	// 页表访问: VA = PA + LINEAR_MAP_OFFSET (pte_ptr 使用)
	map_page(
		base.wrapping_add(LINEAR_MAP_OFFSET),
		base,
		PTE_R | PTE_W,
		LEVEL_MEGA,
	)
}

/// 构建 satp 值（Sv39 模式，ASID=0）。
pub fn init_satp(root_pa: u64) -> u64 {
	let ppn = root_pa >> PAGE_SHIFT;
	(ppn & 0xF_FFFF_FFFF) | (8_u64 << 60)
}

/// 1 GiB 超级页线性映射。PA [LINEAR_MAP_START, +SIZE) -> VA (PA + OFFSET)。
/// 返回是否全部建成 (启动期不变量, 失败由调用方 fail-stop)。
pub fn setup_linear_map() -> bool {
	let giga = 0x4000_0000_u64;
	let pages = LINEAR_MAP_SIZE.div_ceil(giga);
	let mut pa = LINEAR_MAP_START;

	for _ in 0..pages {
		let va = pa.wrapping_add(LINEAR_MAP_OFFSET);
		if map_page(va, pa, PTE_R | PTE_W, LEVEL_GIGA).is_err() {
			return false;
		}
		pa += giga;
	}

	unsafe { core::arch::asm!("sfence.vma") };
	true
}

/// 返回 vaddr PTE 所在的最远层级
///
/// 与硬件 walker 的遍历顺序一致 (层级 0/1/2 对应 GIGA/MEGA/PAGE);
/// 某层中间表缺失或目标槽为空时返回 None。仅在需要区分"已被 1 GiB /
/// 2 MiB 超级页覆盖"与"完全未映射"时使用 (map_linear_range 据此决定是否
/// 向下一级页表建立页面关系, 避免在既有页面上做二次映射)。
pub fn va_real_page_level(vaddr: u64) -> Option<u8> {
	let mut table = root_table();

	for i in 0..3 {
		let pte = unsafe { &*table.add(get_vpn(vaddr, i)) };
		if pte.0 & (PTE_V as u64) == 0 {
			return None;
		}
		// 末级必是叶子: PROT_NONE 映射的 R/W/X 全 0 编码在此处也按叶子返回,
		// 否则会被当作下一级页表指针解引用 (末级之下并不存在页表)。
		if pte.is_leaf() || i == 2 {
			return Some(i as u8);
		}
		let next_pa = pte.ppn() << PAGE_SHIFT;
		table = unsafe { pte_ptr(next_pa) };
	}
	None
}

/// 该 VA 末级叶子项的标志位 (V/R/W/X/U/A/D), 未映射返回 None。
///
/// mremap 原地扩容需要原映射的权限: mremap 不带 prot 参数, 新增的页只能沿用原
/// 映射的权限位, 否则同一段映射的前后两半权限不一致。
pub fn leaf_pte_flags(vaddr: u64) -> Option<u8> {
	let mut table = root_table();

	for i in 0..3 {
		let pte = unsafe { &*table.add(get_vpn(vaddr, i)) };
		if pte.0 & (PTE_V as u64) == 0 {
			return None;
		}
		// 末级必是叶子: R/W/X 全 0 的 PROT_NONE 映射在此处也按叶子返回,
		// 否则会被当作下一级页表指针解引用 (末级之下并不存在页表)。
		if pte.is_leaf() || i == 2 {
			return Some((pte.0 & 0xFF) as u8);
		}
		let next_pa = pte.ppn() << PAGE_SHIFT;
		table = unsafe { pte_ptr(next_pa) };
	}
	None
}

/// 把物理区间 [pa_start, pa_end) 以 2 MiB 粒度映射到 LINEAR_MAP_OFFSET 别名
/// (VA = PA + LINEAR_MAP_OFFSET), 使 S-mode 可直接经高地址读取该段物理内容。
///
/// 全局线性窗口 setup_linear_map 只覆盖 PA [LINEAR_MAP_START, +SIZE)
/// (5-18 GiB); 落在低物理地址的移交区 (载荷 ELF / argv 块) 须在此补映射。
/// 每个 2 MiB 块若线性别名已由更粗叶子覆盖 (窗口 / trampoline 场景) 则跳过。
pub fn map_linear_range(pa_start: u64, pa_end: u64) -> bool {
	let start = mem::chunk_2m_down(pa_start);
	let end = mem::chunk_2m_up(pa_end);

	let mut pa = start;
	while pa < end {
		let va = pa.wrapping_add(LINEAR_MAP_OFFSET);
		if va_real_page_level(va).is_none() &&
		   map_page(va, pa, PTE_R|PTE_W, LEVEL_MEGA).is_err() {
			return false;
		}
		pa += CHUNK_2M_SIZE;
	}
	true
}

// ---------------------------------------------------------------
//  批量映射辅助（对应 smode_entry 的 set_page_config + map_page_in_range）
// ---------------------------------------------------------------

/// 页映射参数集合。将 VA 基址、PA 基址、标志、层级打包为单一参数，
/// 避免在调用方展开冗长的 for 循环。
pub struct SetPageConfig {
	pub vbase: u64,
	pub pbase: u64,
	pub flags: u8,
	pub level: u8,
}

/// 按 config 连续映射 `n_pages` 页, 返回实际建成的页数。
pub fn map_page_in_range(cfg: &SetPageConfig, n_pages: u64) -> u64 {
	for i in 0..n_pages {
		let r = map_page(
			cfg.vbase + i * PAGE_SIZE,
			cfg.pbase + i * PAGE_SIZE,
			cfg.flags,
			cfg.level,
		);
		if r.is_err() {
			return i;
		}
	}
	n_pages
}
