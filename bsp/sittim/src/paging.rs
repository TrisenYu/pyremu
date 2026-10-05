//! Sv39 页表管理。逻辑紧跟 ref-emod/emod_manager/memory/page_table.c，
//! 但用卫语句替代原 C 的深层嵌套。
//!
//! 关键约定：
//! - MMU 开启后（satp != 0），中间页表指针通过 LINEAR_MAP_OFFSET 换算。
//! - A/D 位始终置 1（无 swap）。
//! - map / unmap 后 sfence.vma 刷新对应 VA。

#[cfg(target_arch = "riscv64")]
use core::arch::asm;
#[cfg(not(target_arch = "riscv64"))]
use core::sync::atomic::fence;
use crate::constants::*;
use crate::context;
use crate::csr;
use crate::mem;
use crate::mem_prim::align::{ align_down, align_up };

// ---------------------------------------------------------------
//  Sv39 PTE
// ---------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct Pte(pub u64);

impl Pte {
	pub const fn empty() -> Self {
		Self(0)
	}

	/// R/W/X 任一置位, 或 R/W/X 全 0 而无访问权限且带 U 位的项, 都是叶子节点。
	pub fn is_leaf(&self) -> bool {
		(self.0 & (PTE_R | PTE_W | PTE_X | PTE_U)) != 0
	}

	/// 提取 PPN（bits [53:10]）。
	pub fn ppn(&self) -> u64 {
		(self.0 >> 10) & 0xf_ffff_ffff
	}

	/// 写入 PPN，保留低 10 位标志。
	pub fn set_ppn(&mut self, ppn: u64) {
		self.0 = (self.0 & 0x3ff) | ((ppn & 0xf_ffff_ffff) << 10);
	}

	/// 写入标志位，自动置 A/D。
	pub fn set_flags(&mut self, flags: u64) {
		self.0 |= flags;
		self.0 |= PTE_A | PTE_D;
	}

	pub fn clear(&mut self) {
		self.0 = 0;
	}
}

// ---------------------------------------------------------------
//  翻译缓存刷新
// ---------------------------------------------------------------

/// 刷新 vaddr 对应的翻译缓存项。
#[inline]
fn sfence_vma(vaddr: u64) {
	#[cfg(target_arch = "riscv64")]
	unsafe {
		asm!("sfence.vma {0}, zero", in(reg) vaddr)
	}
	#[cfg(not(target_arch = "riscv64"))]
	{
		fence(core::sync::atomic::Ordering::SeqCst);
		let _ = vaddr;
	}
}

/// 刷新全部的翻译缓存项。
#[inline]
fn sfence_vma_all() {
	#[cfg(target_arch = "riscv64")]
	unsafe {
		asm!("sfence.vma")
	}
	#[cfg(not(target_arch = "riscv64"))]
	fence(core::sync::atomic::Ordering::SeqCst);
}

// ---------------------------------------------------------------
//  VA 辅助
// ---------------------------------------------------------------

/// level 0 -> VPN[2] (bits 38:30), 1 -> VPN[1] (29:21), 2 -> VPN[0] (20:12)
#[inline]
fn get_vpn(va: u64, level: u8) -> usize {
	match level {
		0 => ((va >> 30) & 0x1ff) as usize,
		1 => ((va >> 21) & 0x1ff) as usize,
		2 => ((va >> 12) & 0x1ff) as usize,
		_ => 0,
	}
}

/// 物理地址 -> PTE 指针。satp 有效时叠加 LINEAR_MAP_OFFSET。
#[inline]
unsafe fn pte_ptr(pa: u64) -> *mut Pte {
	if csr::read_satp() != 0 { pa.wrapping_add(LINEAR_MAP_OFFSET) as *mut Pte } else { pa as *mut Pte }
}

// ---------------------------------------------------------------
//  当前地址空间
// ---------------------------------------------------------------

/// 当前页表根在当前映射下的可解引用地址; 0 表示尚未登记 (启动期)。
static mut ROOT_VA: u64 = 0;

/// 登记当前地址空间的页表根。启动期只登记一次, 此后每次进程切换 (含 fork
/// 建出新地址空间后的首次切入) 由调度路径更新。
///
/// 登记的是根表的可解引用地址, 供页表遍历使用。该值在 MMU 使能前后并不相同 ——
/// 引导期根表在物理恒等映射下以物理地址访问, MMU 使能后同一物理页另有管理器
/// 虚拟地址别名。写 satp 用的物理地址由进程表保存 (见 syscall::concurrency::proc)。
pub fn set_root(root_va: u64) {
	unsafe {
		ROOT_VA = root_va;
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
	let va = unsafe { ROOT_VA };
	if va != 0 {
		va as *mut Pte
	} else {
		context::root_pa() as *mut Pte
	}
}

/// 由页表根物理地址求其可解引用地址。仅适用于自 S-mode 页池分配的根表:
/// 池页落在引导期 identity_map_trampoline 建立的 LINEAR_MAP_OFFSET 别名窗口内,
/// 与中间页表页 pte_ptr 的换算方式一致。
#[inline]
pub fn pool_root_va(root_pa: u64) -> u64 {
	root_pa.wrapping_add(LINEAR_MAP_OFFSET)
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

		if (pte.0 & PTE_V) == 0 {
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
			pte.0 |= PTE_V;
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
	if (pte.0 & PTE_V) == 0 && !alloc {
		return Err(MapError::NotMapped);
	}
	Ok(pte)
}

// ---------------------------------------------------------------
//  map / unmap
// ---------------------------------------------------------------

/// 安装一条叶子 PTE。目标 PTE 已存在时按新映射覆盖 (重新映射语义)。
pub fn map_page(vaddr: u64, paddr: u64, flags: u64, level: u8) -> Result<(), MapError> {
	let ppn = paddr >> PAGE_SHIFT;
	let pte = get_leaf_pte(vaddr, level, true)?;

	pte.clear();
	pte.set_ppn(ppn);
	pte.set_flags(flags);
	pte.0 |= PTE_V;

	sfence_vma(vaddr);
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
			sfence_vma(vaddr);
			true
		}
		Err(_) => false,
	}
}

/// 就地改写一条已存在叶子项的标志位, 保留其物理地址与 RSW; 该处无叶子项时返回 false。
///
/// RSW 记录映射的共享属性而不表示访问权限, 故不随权限位改写: 改前的 RSW 原样留在
/// 新项上, mprotect 因此不改变 MAP_SHARED 映射的共享属性。
pub fn set_leaf_flags(vaddr: u64, flags: u64) -> bool {
	let level = match va_real_page_level(vaddr) {
		Some(l) if l != LEVEL_GIGA => l,
		_ => {
			return false;
		}
	};
	let pte = match get_leaf_pte(vaddr, level, false) {
		Ok(p) => p,
		Err(_) => {
			return false;
		}
	};
	let ppn = pte.ppn();
	let rsw = pte.0 & PTE_RSW_MASK;
	pte.clear();
	pte.set_ppn(ppn);
	pte.set_flags(flags);
	pte.0 |= PTE_V | rsw;
	sfence_vma(vaddr);
	true
}

/// 把覆盖 vaddr 的 2 MiB 超级页拆成 512 条 4 KiB 叶子项, 物理地址与标志位照原叶子重排; 该处不是 2 MiB 超级页时不做改动并返回 true。
pub fn split_mega_leaf(vaddr: u64) -> bool {
	if va_real_page_level(vaddr) != Some(LEVEL_MEGA) {
		return true;
	}
	let slot = match get_leaf_pte(vaddr, LEVEL_MEGA, false) {
		Ok(p) => p,
		Err(_) => {
			return false;
		}
	};
	let leaf = *slot;
	let t = mem::try_alloc_smode_page(1);
	if t == !0u64 {
		return false;
	}
	unsafe {
		core::ptr::write_bytes(pte_ptr(t) as *mut u8, 0, PAGE_SIZE as usize);
	}
	// 2 MiB 叶子的 PPN 低位被 VA 覆盖, 拆分后由 4 KiB 叶子逐页给出同一段物理内存。
	// 低位标志与 RSW 一并照抄, 使拆分不改变映射的共享属性。
	let base_ppn = leaf.ppn() & !0x1ff;
	let low_bits = leaf.0 & (0xff | PTE_RSW_MASK);
	for i in 0..512usize {
		let mut p = Pte(0);
		p.set_ppn(base_ppn + (i as u64));
		p.0 |= low_bits | PTE_V;
		unsafe {
			*pte_ptr(t).add(i) = p;
		}
	}
	slot.clear();
	slot.set_ppn(t >> PAGE_SHIFT);
	slot.0 |= PTE_V;
	sfence_vma(vaddr);
	true
}

// ---------------------------------------------------------------
//  VA -> PA 翻译
// ---------------------------------------------------------------

struct WalkResult {
	level: i8,
	pte: Pte,
}

/// 三层 Sv39 页表遍历。未命中时 level = -1。
fn walk_page_table(va: u64) -> WalkResult {
	let idxs: [usize; 3] = [
		((va & 0x7f_c000_0000) >> 30) as usize,
		((va & 0x00_3fe0_0000) >> 21) as usize,
		((va & 0x00_001f_f000) >> 12) as usize,
	];

	let mut table = root_table();

	for i in 0..3 {
		let pte = unsafe { &*table.add(idxs[i]) };

		if (pte.0 & PTE_V) == 0 {
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
///
/// 供无法用固定偏移换算的映射使用: 模块窗口的物理地址分散, 各块由独立一次向
/// M 模式申请取得, 块间不保证连续, 只能按页表查询。
pub fn get_pa(va: u64) -> Option<u64> {
	let r = walk_page_table(va);
	if r.level < 0 {
		return None;
	}

	let base = r.pte.ppn() << PAGE_SHIFT;
	let offset_bits = (PAGE_SHIFT as usize) + (2 - (r.level as usize)) * (SV39_VPN_LEN as usize);
	Some(base | (((1_u64 << offset_bits) - 1) & va))
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
	let base = align_down(pa, CHUNK_2M_SIZE);
	// 代码执行: identity VA=PA
	map_page(base, base, PTE_R | PTE_W | PTE_X, LEVEL_MEGA)?;
	// 页表访问: VA = PA + LINEAR_MAP_OFFSET (pte_ptr 使用)
	map_page(base.wrapping_add(LINEAR_MAP_OFFSET), base, PTE_R | PTE_W, LEVEL_MEGA)
}

/// 构建 satp 值（Sv39 模式，ASID=0）。
pub fn init_satp(root_pa: u64) -> u64 {
	let ppn = root_pa >> PAGE_SHIFT;
	(ppn & 0xf_ffff_ffff) | (8_u64 << 60)
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

	sfence_vma_all();
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
		if (pte.0 & PTE_V) == 0 {
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

/// 该 VA 末级叶子项的标志位 (V/R/W/X/U/A/D) 与 RSW, 未映射返回 None。
///
/// mremap 原地扩容需要原映射的权限: mremap 不带 prot 参数, 新增的页只能沿用原
/// 映射的权限位, 否则同一段映射的前后两半权限不一致。返回值连同 RSW 一并给出,
/// 使新增的页与原有的页同属共享映射或同属私有映射。
pub fn leaf_pte_flags(vaddr: u64) -> Option<u64> {
	let mut table = root_table();

	for i in 0..3 {
		let pte = unsafe { &*table.add(get_vpn(vaddr, i)) };
		if (pte.0 & PTE_V) == 0 {
			return None;
		}
		// 末级必是叶子: R/W/X 全 0 的 PROT_NONE 映射在此处也按叶子返回,
		// 否则会被当作下一级页表指针解引用 (末级之下并不存在页表)。
		if pte.is_leaf() || i == 2 {
			return Some(pte.0 & (0xff | PTE_RSW_MASK));
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
	let start = align_down(pa_start, CHUNK_2M_SIZE);
	let end = align_up(pa_end, CHUNK_2M_SIZE);

	let mut pa = start;
	while pa < end {
		let va = pa.wrapping_add(LINEAR_MAP_OFFSET);
		if va_real_page_level(va).is_none() && map_page(va, pa, PTE_R | PTE_W, LEVEL_MEGA).is_err() {
			return false;
		}
		pa += CHUNK_2M_SIZE;
	}
	true
}

/// 把设备寄存器窗口映射到 LINEAR_MAP_OFFSET 别名 (VA = PA + LINEAR_MAP_OFFSET),
/// 使 S 模式的设备驱动以线性地址访问 MMIO。
///
/// 窗口按 4 KiB 页映射, 权限取 PTE_R | PTE_W 且不带 PTE_U: 该窗口只由 S 模式
/// 驱动访问, U 模式载荷经系统调用间接访问设备。映射落在根表高半区, 进程式 clone
/// 时对各地址空间共用, 故只在引导期建立一次, 无需随 fork 补建。
///
/// `base` 取 0 表示平台不提供该设备, 此时不建立映射并返回 true。
pub fn map_device_region(base: u64, size: u64) -> bool {
	if base == 0 || size == 0 {
		return true;
	}
	if (base & (PAGE_SIZE - 1)) != 0 {
		return false;
	}
	let pages = size.div_ceil(PAGE_SIZE);
	for i in 0..pages {
		let pa = base + i * PAGE_SIZE;
		let va = pa.wrapping_add(LINEAR_MAP_OFFSET);
		if map_page(va, pa, PTE_R | PTE_W, LEVEL_PAGE).is_err() {
			return false;
		}
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
	pub flags: u64,
	pub level: u8,
}

/// 按 config 连续安装 `n_leaves` 项叶子页, 返回实际建成的项数。
///
/// 步长取该层级叶子页的字节数。若恒以 4 KiB 步进, 以 `LEVEL_MEGA` 调用时落在同一
/// 个 2 MiB 区间的各次写入会反复改写同一条 PTE, 只有最后一次的值留下, 该区间之外
/// 的叶子始终没有映射项, 而函数仍报告全部建成。
pub fn map_page_in_range(cfg: &SetPageConfig, n_leaves: u64) -> u64 {
	let stride = leaf_page_size(cfg.level);
	for i in 0..n_leaves {
		let r = map_page(cfg.vbase + i * stride, cfg.pbase + i * stride, cfg.flags, cfg.level);
		if r.is_err() {
			return i;
		}
	}
	n_leaves
}

/// Sv39 各层级叶子页的字节数: 0 级 1 GiB, 1 级 2 MiB, 2 级 4 KiB。
///
/// 越界层级按最小页处理, 与 `get_vpn` 对越界层级取索引 0 的处理一致。
pub fn leaf_page_size(level: u8) -> u64 {
	1_u64 << (PAGE_SHIFT + (SV39_VPN_LEN as u64) * (LEVEL_PAGE.saturating_sub(level) as u64))
}

// ---------------------------------------------------------------
//  地址空间复制 (进程式 clone)
// ---------------------------------------------------------------

/// Sv39 用户空间的根表索引上界。根索引 < 该值的 VA 落在 [0, 0x40_0000_0000),
/// 即全部用户空间; 不小于该值的 VA 是运行时自身的线性窗口与管理器 VA。
const USER_ROOT_ENTRIES: usize = 256;

/// 复制用户页时使用的临时映射 VA。
///
/// 目标页的物理地址不保证落在 S 模式的线性窗口内 (U 模式页池之外的物理区由
/// M 模式按 2 MiB 分区交付, 地址任意), 故源页与目标页都要先临时映射进当前
/// 地址空间才能读写。四个 VA 取在用户空间上界附近, 远离载荷的段/堆/栈/mmap 区;
/// 按层级分成两组, 因为同一 VA 先以 2 MiB 叶子映射、再以 4 KiB 叶子映射时,
/// 后者会因前者的超级页覆盖而无法安装。
const FORK_SRC_VA_MEGA: u64 = 0x3f_0000_0000;
const FORK_DST_VA_MEGA: u64 = 0x3f_2000_0000;
const FORK_SRC_VA_PAGE: u64 = 0x3f_c000_0000;
const FORK_DST_VA_PAGE: u64 = 0x3f_c000_1000;

/// 复制 4 KiB 页内容。任一映射失败即返回 false。
unsafe fn copy_page_pa(src_pa: u64, dst_pa: u64) -> bool {
	if map_page(FORK_SRC_VA_PAGE, src_pa, PTE_R, LEVEL_PAGE).is_err() {
		return false;
	}
	if map_page(FORK_DST_VA_PAGE, dst_pa, PTE_R | PTE_W, LEVEL_PAGE).is_err() {
		unmap_page(FORK_SRC_VA_PAGE, LEVEL_PAGE);
		return false;
	}
	unsafe {
		core::ptr::copy_nonoverlapping(
			FORK_SRC_VA_PAGE as *const u8,
			FORK_DST_VA_PAGE as *mut u8,
			PAGE_SIZE as usize
		);
	}
	unmap_page(FORK_SRC_VA_PAGE, LEVEL_PAGE);
	unmap_page(FORK_DST_VA_PAGE, LEVEL_PAGE);
	true
}

/// 复制 2 MiB 分区内容 (两个物理基址都按 2 MiB 对齐)。
unsafe fn copy_chunk_pa(src_pa: u64, dst_pa: u64) -> bool {
	if map_page(FORK_SRC_VA_MEGA, src_pa, PTE_R, LEVEL_MEGA).is_err() {
		return false;
	}
	if map_page(FORK_DST_VA_MEGA, dst_pa, PTE_R | PTE_W, LEVEL_MEGA).is_err() {
		unmap_page(FORK_SRC_VA_MEGA, LEVEL_MEGA);
		return false;
	}
	unsafe {
		core::ptr::copy_nonoverlapping(
			FORK_SRC_VA_MEGA as *const u8,
			FORK_DST_VA_MEGA as *mut u8,
			CHUNK_2M_SIZE as usize
		);
	}
	unmap_page(FORK_SRC_VA_MEGA, LEVEL_MEGA);
	unmap_page(FORK_DST_VA_MEGA, LEVEL_MEGA);
	true
}

/// 交付一个 4 KiB 目标页: 先取 U 模式页池, 池中不足时向 M 模式申请一个
/// 2 MiB 分区并逐页切分 (与 mmap 取块的方式一致)。失败返回 `!0u64`。
fn fork_alloc_page() -> u64 {
	let pa = mem::alloc_umode_page(1);
	if pa != !0u64 {
		return pa;
	}
	unsafe {
		if FORK_CHUNK_LEFT == 0 {
			let chunk = mem::alloc_mmap_chunk_pa();
			if chunk == !0u64 {
				return !0u64;
			}
			FORK_CHUNK_PA = chunk;
			FORK_CHUNK_LEFT = CHUNK_2M_PAGES;
		}
		let pa = FORK_CHUNK_PA + (CHUNK_2M_PAGES - FORK_CHUNK_LEFT) * PAGE_SIZE;
		FORK_CHUNK_LEFT -= 1;
		pa
	}
}

/// 复制期间用于交付 4 KiB 目标页的分区块游标, 由 fork_alloc_page 维护。
static mut FORK_CHUNK_PA: u64 = 0;
static mut FORK_CHUNK_LEFT: u64 = 0;

/// 某一层级第 i 个槽位对应的 VA 基址。
#[inline]
fn slot_va(i: usize, level: u8) -> u64 {
	(i as u64) << (PAGE_SHIFT + (SV39_VPN_LEN as u64) * (2 - (level as u64)))
}

/// 递归复制一张页表。
///
/// 叶子项分三类处理:
/// - 带 PTE_U 且不带 RSW 共享位的用户页, 分配新的物理页并复制内容, 两个地址空间
///   自此各自持有独立副本;
/// - 带 PTE_U 且带 RSW 共享位的用户页 (MAP_SHARED 匿名映射), 两个地址空间共用同一
///   物理页, 页表项照抄 —— 这正是共享映射的语义, 父子进程的写入互相可见;
/// - 不带 PTE_U 的是运行时自身的映射 (自身段/线性窗口/管理器 VA), 两个地址空间
///   共用同一物理页, 页表项原样照抄。
///
/// 非叶子项在目标表中新建一张同层级的表并递归, 使两张表的中间层级互不共享 ——
/// 任一地址空间此后新增的用户映射都不会影响另一个。
unsafe fn copy_table_level(src: *const Pte, dst: *mut Pte, level: u8, va_base: u64) -> bool {
	for i in 0..512usize {
		let s = unsafe { *src.add(i) };
		if (s.0 & PTE_V) == 0 {
			continue;
		}
		let va = va_base + slot_va(i, level);

		if s.is_leaf() {
			// 只有私有的用户页需要复制: 运行时自身的映射不带 PTE_U, 共享映射带
			// PTE_RSW_SHARED, 两者都照抄页表项、共用同一物理页。
			let is_private_user_page = (s.0 & PTE_U) != 0 && (s.0 & PTE_RSW_SHARED) == 0;
			if !is_private_user_page {
				unsafe {
					*dst.add(i) = s;
				}
				continue;
			}
			if level == LEVEL_GIGA {
				// 用户空间不产生 1 GiB 叶子 (载荷段/堆/栈/mmap 都按 4 KiB 或
				// 2 MiB 映射), 出现即说明复制规则与实际布局不一致。
				return false;
			}
			let dst_pa = if level == LEVEL_MEGA { mem::alloc_mmap_chunk_pa() } else { fork_alloc_page() };
			if dst_pa == !0u64 {
				return false;
			}
			let src_pa = s.ppn() << PAGE_SHIFT;
			let copied = if level == LEVEL_MEGA {
				unsafe { copy_chunk_pa(src_pa, dst_pa) }
			} else {
				unsafe { copy_page_pa(src_pa, dst_pa) }
			};
			if !copied {
				return false;
			}
			let mut d = s;
			d.set_ppn(dst_pa >> PAGE_SHIFT);
			unsafe {
				*dst.add(i) = d;
			}
			continue;
		}

		let t = mem::try_alloc_smode_page(1);
		if t == !0u64 {
			return false;
		}
		unsafe {
			core::ptr::write_bytes(pte_ptr(t) as *mut u8, 0, PAGE_SIZE as usize);
		}
		let mut d = s;
		d.set_ppn(t >> PAGE_SHIFT);
		unsafe {
			*dst.add(i) = d;
		}
		let ok = unsafe {
			copy_table_level(pte_ptr(s.ppn() << PAGE_SHIFT) as *const Pte, pte_ptr(t), level + 1, va)
		};
		if !ok {
			return false;
		}
	}
	true
}

/// 以当前地址空间为模板建立一份新的用户地址空间, 返回新根表的物理地址;
/// 失败返回 `!0u64` (页池耗尽或映射失败)。
///
/// 根表低半区 (索引 0..256, 即 Sv39 用户空间) 逐级深拷贝; 高半区 (索引
/// 256..512, 线性窗口与管理器 VA) 对所有地址空间完全相同, 直接共用父表的
/// 中间表, 不复制也不新建 —— 复制高半区既无必要, 也会把 S 模式页池按地址
/// 空间的份数重复消耗。
///
/// 本函数只读父地址空间、只写新建的表与新建的物理页, 调用期间不切换 satp,
/// 故可在父进程的陷态处理内直接调用。
pub fn fork_address_space() -> u64 {
	let child_pa = mem::try_alloc_smode_page(1);
	if child_pa == !0u64 {
		return !0u64;
	}
	let parent = root_table();
	let child = unsafe { pte_ptr(child_pa) };

	unsafe {
		core::ptr::write_bytes(child as *mut u8, 0, PAGE_SIZE as usize);
		for i in USER_ROOT_ENTRIES..512usize {
			*child.add(i) = *parent.add(i);
		}
		for i in 0..USER_ROOT_ENTRIES {
			let s = *parent.add(i);
			if (s.0 & PTE_V) == 0 {
				continue;
			}
			if s.is_leaf() {
				// 用户空间不产生根层叶子; 运行时自身的映射照抄共用。
				if (s.0 & PTE_U) != 0 {
					return !0u64;
				}
				*child.add(i) = s;
				continue;
			}
			let t = mem::try_alloc_smode_page(1);
			if t == !0u64 {
				return !0u64;
			}
			core::ptr::write_bytes(pte_ptr(t) as *mut u8, 0, PAGE_SIZE as usize);
			let mut d = s;
			d.set_ppn(t >> PAGE_SHIFT);
			*child.add(i) = d;
			let ok = copy_table_level(
				pte_ptr(s.ppn() << PAGE_SHIFT) as *const Pte,
				pte_ptr(t),
				LEVEL_MEGA,
				slot_va(i, LEVEL_GIGA)
			);
			if !ok {
				return !0u64;
			}
		}
	}
	child_pa
}

// ---------------------------------------------------------------
//  测试
// ---------------------------------------------------------------

#[cfg(test)]
mod tests {
	use crate::paging::*;
	use std::sync::{ Mutex, MutexGuard };

	/// 登记页表根是一项全局状态, 而测试默认并行执行, 故凡调用 `set_root` 的用例先
	/// 取得该锁, 使各自的根表不被其它用例改写。
	static SERIAL: Mutex<()> = Mutex::new(());

	/// 前一个用例 panic 后 `Mutex::lock` 返回 `Err`, 此处经 `into_inner` 取回其中的
	/// 守卫继续, 使后续用例照常执行而不连带失败。
	fn serial() -> MutexGuard<'static, ()> {
		SERIAL.lock().unwrap_or_else(|e| e.into_inner())
	}

	/// 用例内构造的页表页: 页表项给出的物理地址即本数组的地址。
	#[repr(align(4096))]
	struct Table([Pte; 512]);

	/// 在 root 的 VPN[2] 槽写入指向 next 的指针项 (只置 V, 故不是叶子)。
	fn link_root(root: &mut Table, va: u64, next: &Table) {
		let idx = super::get_vpn(va, 0);
		root.0[idx].set_ppn((next.0.as_ptr() as u64) >> PAGE_SHIFT);
		root.0[idx].0 |= PTE_V;
	}

	/// 取 4 KiB 叶子项: 沿中间表的 VPN[1] 项进入末级表, 再取 VPN[0] 项。
	unsafe fn leaf_pte(mid: &Table, va: u64) -> Pte {
		let l3 = (mid.0[super::get_vpn(va, 1)].ppn() << PAGE_SHIFT) as *const Pte;
		unsafe { *l3.add(super::get_vpn(va, 2)) }
	}

	#[test]
	fn test_entry_without_access_permission_is_leaf() {
		assert!(Pte((PTE_V | PTE_U) as u64).is_leaf());
	}

	#[test]
	fn test_pointer_entry_is_not_leaf() {
		assert!(!Pte(PTE_V).is_leaf());
	}

	#[test]
	fn test_empty_entry_is_not_leaf() {
		assert!(!Pte(0).is_leaf());
	}

	#[test]
	fn test_entry_with_access_permission_is_leaf() {
		for flags in [PTE_R, PTE_W, PTE_X, PTE_R | PTE_W | PTE_X] {
			assert!(Pte((PTE_V | flags) as u64).is_leaf(), "flags={flags:#04x}");
		}
	}

	#[test]
	fn test_leaf_page_size_matches_level() {
		assert_eq!(leaf_page_size(LEVEL_GIGA), 1 << 30);
		assert_eq!(leaf_page_size(LEVEL_MEGA), CHUNK_2M_SIZE);
		assert_eq!(leaf_page_size(LEVEL_PAGE), PAGE_SIZE);
	}

	/// 以 2 MiB 叶子页安装三项映射, 每一项都必须落在各自的 PTE 上且物理基址正确。
	///
	/// 修复前 `map_page_in_range` 恒以 4 KiB 步进, 三项写入命中同一张叶子表的同一条
	/// PTE: 只有第三项留下, 其物理基址带上 8 KiB 的偏移, 另外两项没有映射项。
	/// pyremu 的页表遍历按 VA 的 vpn0 重建大页 PPN 的低 9 位, 该偏移被静默吸收,
	/// 故缺陷只表现为后两块始终未映射, 在首次写入时触发存储页错误。
	#[test]
	fn test_mega_page_in_range_installs_each_leaf() {
		let _guard = serial();

		let mut root = Table([Pte(0); 512]);
		let leaf = Table([Pte(0); 512]);

		// 中间表由本用例给出, 安装后的断言因此直接从该表读回。
		link_root(&mut root, ENCLAVE_MODULE_LOAD_VA_INIT, &leaf);
		set_root(root.0.as_ptr() as u64);

		let pbase = 0x8000_0000_u64;
		let cfg = SetPageConfig {
			vbase: ENCLAVE_MODULE_LOAD_VA_INIT,
			pbase,
			flags: PTE_R | PTE_W | PTE_X,
			level: LEVEL_MEGA,
		};
		assert_eq!(map_page_in_range(&cfg, 3), 3);

		let leaf_idx = super::get_vpn(ENCLAVE_MODULE_LOAD_VA_INIT, 1);
		for k in 0..3_usize {
			let pte = leaf.0[leaf_idx + k];
			assert!((pte.0 & PTE_V) != 0, "第 {k} 块没有映射项");
			assert_eq!(
				pte.ppn() << PAGE_SHIFT,
				pbase + (k as u64) * CHUNK_2M_SIZE,
				"第 {k} 块的物理基址不对"
			);
		}

		set_root(0);
	}

	/// mprotect 改写叶子项的标志位时, RSW 与物理地址都必须留下。
	///
	/// RSW 记录映射的共享属性而不表示访问权限。修复前该函数清空页表项后只写回权限位
	/// 与 PPN, MAP_SHARED 映射经一次 mprotect 即退化为私有映射, fork 出的子进程自此
	/// 得到副本而非与父进程共用同一批物理页。
	#[test]
	fn test_set_leaf_flags_preserves_rsw_and_ppn() {
		let _guard = serial();

		let va = 0x1000_0000_u64;
		let pa = 0x8000_0000_u64;

		let mid = Table([Pte(0); 512]);
		let mut root = Table([Pte(0); 512]);
		link_root(&mut root, va, &mid);
		set_root(root.0.as_ptr() as u64);

		// 末级表由 map_page 经 tests/paging_host.rs 的页池桩分配。
		assert!(map_page(va, pa, PTE_U | PTE_R | PTE_W | PTE_RSW_SHARED, LEVEL_PAGE).is_ok());
		assert!(set_leaf_flags(va, PTE_U | PTE_R));

		let pte = unsafe { leaf_pte(&mid, va) };
		assert_eq!(pte.ppn() << PAGE_SHIFT, pa, "物理地址被改写");
		assert!((pte.0 & PTE_RSW_SHARED) != 0, "共享属性丢失");
		assert!((pte.0 & PTE_W) == 0, "写权限未被收回");
		assert!((pte.0 & PTE_R) != 0, "读权限丢失");
		// mremap 搬移共享映射时经该函数取源映射的标志位, 该处同样要给出 RSW。
		assert!(
			(leaf_pte_flags(va).expect("叶子项应存在") & PTE_RSW_SHARED) != 0,
			"共享属性未随标志位给出"
		);

		set_root(0);
	}

	/// 拆分 2 MiB 超级页时, 512 条 4 KiB 叶子都必须保留 RSW 与权限位。
	///
	/// 修复前该函数只照抄叶子的低 8 位标志, 落在位 8 的 PTE_RSW_SHARED 被丢弃:
	/// mprotect 只覆盖超级页的一部分时拆出 4 KiB 叶子, 拆出的页自此按私有页处理,
	/// fork 不再与父进程共用物理页。
	#[test]
	fn test_split_mega_leaf_preserves_rsw() {
		let _guard = serial();

		// VA 取在 2 MiB 区间内的第 7 页, 使拆分不依赖 VA 恰好落在区间起点。
		let va = 0x2000_7000_u64;
		let pa = 0x8000_0000_u64;

		let mid = Table([Pte(0); 512]);
		let mut root = Table([Pte(0); 512]);
		link_root(&mut root, va, &mid);
		set_root(root.0.as_ptr() as u64);

		let flags = PTE_U | PTE_R | PTE_W | PTE_RSW_SHARED;
		assert!(map_page(va, pa, flags, LEVEL_MEGA).is_ok());
		assert!(split_mega_leaf(va));

		let slot = mid.0[super::get_vpn(va, 1)];
		assert!((slot.0 & PTE_V) != 0, "超级页槽位被清空");
		assert!(!slot.is_leaf(), "超级页槽位仍按叶子解释");

		let l3 = (slot.ppn() << PAGE_SHIFT) as *const Pte;
		for i in 0..512_usize {
			let pte = unsafe { *l3.add(i) };
			assert!((pte.0 & PTE_V) != 0, "第 {i} 页没有映射项");
			assert!((pte.0 & PTE_RSW_SHARED) != 0, "第 {i} 页的共享属性丢失");
			assert_eq!(
				pte.0 & (PTE_U | PTE_R | PTE_W),
				flags & (PTE_U | PTE_R | PTE_W),
				"第 {i} 页的权限位不对"
			);
			assert_eq!(pte.ppn() << PAGE_SHIFT, pa + (i as u64) * PAGE_SIZE, "第 {i} 页的物理地址不对");
		}

		set_root(0);
	}
}
