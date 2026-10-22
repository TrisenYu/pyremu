//! `mem` 模块在主机目标上的单元测试入口。
//!
//! 本 crate 是 RISC-V 裸机目标上的可执行程序, 其中的 bin 目标只能在目标上构建, 故
//! `mem` 模块由本入口按源码包含, 由主机上的测试工具链编译。树内引用的平台符号由本
//! 文件声明。
//!
//! 覆盖的是类别改判的准入判定: 待改判的分区是否由单个申请独占。2 MiB 分区整体属于一个
//! 类别, 改判即改掉该分区内全部页的权限; 落在飞地管理器镜像内的分区由运行时自身的段与
//! S 模式页池共用, 改判它们会连带改掉同分区内页表与内核栈的写权限, 或不产生任何立即的
//! 异常而静默改错类别, 故一律拒绝。
//!
//! 另覆盖 U 模式栈的落点: 栈自 argv 分区的末尾起映射, 与 argv 数据分处分区两端。
//!
//! 另覆盖 mmap 映射区的块内切分状态。映射区有页计数与正在切分的块的后备物理内存基址
//! 两份状态, 块内交付按后者加块内偏移算出物理地址, 故 mremap 的原地收缩与扩容都须使
//! 页计数与基址描述同一个 2 MiB 块; 跨越块边界的长度变化由 mremap 的搬移路径处理。

#![allow(dead_code)]

use std::sync::Once;
use std::sync::atomic::Ordering;

#[path = "../src/constants.rs"]
mod constants;

#[path = "../src/prim/mod.rs"]
mod prim;

/// 主机侧的 `paging`, 只声明 `mem` 与 `context` 引用到的符号。
///
/// 页表项安装自身由 `tests/paging_host.rs` 与 `src/paging.rs` 的用例覆盖, 本入口内的用例
/// 不建立页表: 映射请求只被记录。
mod paging {
	use core::sync::atomic::{AtomicU64, Ordering};

	use crate::constants::{PTE_R, PTE_W};

	/// 页表项。`context` 以它声明 4 KiB 对齐的页表根。
	#[derive(Clone, Copy)]
	pub struct Pte(pub u64);

	/// 页表项安装的失败原因。本入口内的用例不安装页表项, 该类型只为 `map_page` 的签名
	/// 而存在。
	#[derive(Debug, Clone, Copy, PartialEq, Eq)]
	pub enum MapError {
		/// 页表页分配失败 (S-mode 页池耗尽)。
		OutOfPool,
	}

	/// 映射请求的次数, 以及首次与最近一次请求的虚拟地址与物理地址。
	pub static MAP_CALLS: AtomicU64 = AtomicU64::new(0);
	pub static FIRST_VA: AtomicU64 = AtomicU64::new(0);
	pub static FIRST_PA: AtomicU64 = AtomicU64::new(0);
	pub static LAST_VA: AtomicU64 = AtomicU64::new(0);
	pub static LAST_PA: AtomicU64 = AtomicU64::new(0);

	/// 记录一次映射请求并报告成功。
	pub fn map_page(vaddr: u64, paddr: u64, _flags: u64, _level: u8) -> Result<(), MapError> {
		if MAP_CALLS.fetch_add(1, Ordering::SeqCst) == 0 {
			FIRST_VA.store(vaddr, Ordering::SeqCst);
			FIRST_PA.store(paddr, Ordering::SeqCst);
		}
		LAST_VA.store(vaddr, Ordering::SeqCst);
		LAST_PA.store(paddr, Ordering::SeqCst);
		Ok(())
	}

	/// 本入口内的用例不查询物理地址。
	pub fn get_pa(_va: u64) -> Option<u64> {
		None
	}

	/// 叶子项的标志位。本入口内的用例不建立页表, 一律给出数据页的标志位, 使原地扩容
	/// 越过标志位读取这一步, 由用例核对实际安装的物理地址。
	pub fn leaf_pte_flags(_va: u64) -> Option<u64> {
		Some(PTE_R | PTE_W)
	}
}

/// 主机侧的 `hang`, 记录挂起请求。本入口内的用例不触发挂起。
mod hang {
	use core::sync::atomic::{AtomicU64, Ordering};

	/// 挂起请求的次数。
	pub static HALT_CALLS: AtomicU64 = AtomicU64::new(0);

	/// 记录一次挂起请求并返回, 不终止进程。
	pub fn fault_halt(_msg: &str) -> ! {
		HALT_CALLS.fetch_add(1, Ordering::SeqCst);
		panic!("fault_halt: {_msg}");
	}
}

/// 主机侧的 `ecall_aux`, 记录发往 M 模式的类别改判请求。
mod ecall_aux {
	use core::sync::atomic::{AtomicU64, Ordering};

	/// 类别改判请求的次数, 以及最近一次请求的物理基址、分区数与类别。
	pub static SET_ATTR_CALLS: AtomicU64 = AtomicU64::new(0);
	pub static SET_ATTR_PA: AtomicU64 = AtomicU64::new(0);
	pub static SET_ATTR_PARTITIONS: AtomicU64 = AtomicU64::new(0);
	pub static SET_ATTR_ATTR: AtomicU64 = AtomicU64::new(0);

	/// 分区申请的次数, 以及最近一次请求的分区块数。
	pub static ALLOC_CALLS: AtomicU64 = AtomicU64::new(0);
	pub static ALLOC_CHUNK_NUMS: AtomicU64 = AtomicU64::new(0);
	/// 分区申请授予的分区块数。取 0 表示拒绝, 即池中无可用分区。
	pub static ALLOC_GRANTED: AtomicU64 = AtomicU64::new(0);
	/// 授予分区的物理起始地址。
	pub static ALLOC_PA: AtomicU64 = AtomicU64::new(0);

	/// 记录一次分区申请。授予的块数取 [`ALLOC_GRANTED`], 请求多于授予时只授予请求的块数。
	pub fn enclave_call_mem_alloc(chunk_nums: u64, _attr: u64) -> (u64, u64) {
		ALLOC_CALLS.fetch_add(1, Ordering::SeqCst);
		ALLOC_CHUNK_NUMS.store(chunk_nums, Ordering::SeqCst);
		let granted = ALLOC_GRANTED.load(Ordering::SeqCst);
		if granted == 0 {
			return (0, !0u64);
		}
		let granted = if granted < chunk_nums {
			granted
		} else {
			chunk_nums
		};
		(granted, ALLOC_PA.load(Ordering::SeqCst))
	}

	/// 记录一次类别改判请求并报告成功。
	pub fn enclave_call_set_mem_attr(paddr: u64, partition_num: u64, attr: u64) -> u64 {
		SET_ATTR_CALLS.fetch_add(1, Ordering::SeqCst);
		SET_ATTR_PA.store(paddr, Ordering::SeqCst);
		SET_ATTR_PARTITIONS.store(partition_num, Ordering::SeqCst);
		SET_ATTR_ATTR.store(attr, Ordering::SeqCst);
		0
	}
}

#[path = "../src/context.rs"]
mod context;

#[path = "../src/mem.rs"]
mod mem;

use constants::*;
use prim::arith::align_up;

// ---------------------------------------------------------------
//  平台布局
// ---------------------------------------------------------------

/// 飞地管理器镜像与 S 模式页池的物理起始地址。
const MAN_PA_START: u64 = 0x8020_0000;
/// 管理器镜像占用的分区数。
const MAN_IMAGE_PARTITIONS: u64 = 2;
/// S 模式页池占用的分区数。
const SMODE_POOL_PARTITIONS: u64 = 1;
/// argv 分区的物理起始地址: argv 数据在该分区的首页, 该分区由单个申请独占。
const ARGV_PA: u64 = 0x9000_0000;
/// 授予区间的末尾, 即管理器镜像与 S 模式页池之外的第一块分区。
const GRANTED_END: u64 =
	MAN_PA_START + (MAN_IMAGE_PARTITIONS + SMODE_POOL_PARTITIONS) * CHUNK_2M_SIZE;
/// 授予区间之外的一块分区, 即由单个申请独占的分区。
const DEDICATED_PA: u64 = 0xA000_0000;

// ---------------------------------------------------------------
//  共享状态
// ---------------------------------------------------------------

/// 飞地上下文只初始化一次, `OnceCell` 的二次初始化会 panic。
static WORLD: Once = Once::new();

/// 建立本入口内的平台布局, 并清空类别改判与映射请求的记录。
fn world() {
	WORLD.call_once(|| {
		context::init_context(MAN_PA_START, 0x1000);
		mem::init_smode_pool(
			MAN_IMAGE_PARTITIONS * CHUNK_2M_SIZE,
			SMODE_POOL_PARTITIONS * CHUNK_2M_SIZE,
		);
	});
	ecall_aux::SET_ATTR_CALLS.store(0, Ordering::SeqCst);
	ecall_aux::ALLOC_CALLS.store(0, Ordering::SeqCst);
	ecall_aux::ALLOC_GRANTED.store(0, Ordering::SeqCst);
	ecall_aux::ALLOC_PA.store(0, Ordering::SeqCst);
	paging::MAP_CALLS.store(0, Ordering::SeqCst);
}

/// 让下一次分区申请授予 *chunk_nums* 块分区, 物理起始地址取 `ALLOC_PA`。
fn grant_alloc(chunk_nums: u64, pa: u64) {
	ecall_aux::ALLOC_GRANTED.store(chunk_nums, Ordering::SeqCst);
	ecall_aux::ALLOC_PA.store(pa, Ordering::SeqCst);
}

/// 类别改判请求的次数, 以及最近一次请求的物理基址、分区数与类别。
fn set_attr_calls() -> u64 {
	ecall_aux::SET_ATTR_CALLS.load(Ordering::SeqCst)
}

fn set_attr_pa() -> u64 {
	ecall_aux::SET_ATTR_PA.load(Ordering::SeqCst)
}

fn set_attr_partitions() -> u64 {
	ecall_aux::SET_ATTR_PARTITIONS.load(Ordering::SeqCst)
}

fn set_attr_attr() -> u64 {
	ecall_aux::SET_ATTR_ATTR.load(Ordering::SeqCst)
}

// ---------------------------------------------------------------
//  独占判定
// ---------------------------------------------------------------

/// 管理器镜像与 S 模式页池之外的分区由单个申请独占, 其中的分区一律可改判。
#[test]
fn test_is_dedicated_partition_accepts_a_granted_partition() {
	world();

	assert!(mem::is_dedicated_partition(GRANTED_END));
	assert!(mem::is_dedicated_partition(DEDICATED_PA));
	assert!(mem::is_dedicated_partition(
		DEDICATED_PA + 7 * CHUNK_2M_SIZE
	));
}

/// 管理器镜像与 S 模式页池覆盖的分区一律不可改判, 首尾两块分区都在内。
#[test]
fn test_is_dedicated_partition_rejects_the_image_and_the_smode_pool() {
	world();

	assert!(!mem::is_dedicated_partition(MAN_PA_START));
	assert!(!mem::is_dedicated_partition(MAN_PA_START + CHUNK_2M_SIZE));
	assert!(!mem::is_dedicated_partition(GRANTED_END - CHUNK_2M_SIZE));
	// 池的最后一个分区内部的偏移仍属于该分区, 一并被拒。
	assert!(!mem::is_dedicated_partition(GRANTED_END - 1));
	// 该分区之上的第一块分区归单个申请独占。
	assert!(mem::is_dedicated_partition(GRANTED_END));
}

/// 判定按分区进行, 分区内部的偏移不改变结果。
#[test]
fn test_is_dedicated_partition_ignores_the_offset_inside_a_partition() {
	world();

	assert!(!mem::is_dedicated_partition(
		MAN_PA_START + CHUNK_2M_SIZE - 1
	));
	assert!(mem::is_dedicated_partition(
		DEDICATED_PA + CHUNK_2M_SIZE - 1
	));
}

// ---------------------------------------------------------------
//  类别改判
// ---------------------------------------------------------------

/// 独占的分区可改判: 请求按分区数发给 M 模式, 类别照传。
#[test]
fn test_set_partition_attr_forwards_a_dedicated_range() {
	world();

	assert!(mem::set_partition_attr(
		DEDICATED_PA,
		CHUNK_2M_SIZE,
		ENCLAVE_MEM_ATTR_INSTR
	));
	assert_eq!(set_attr_calls(), 1);
	assert_eq!(set_attr_pa(), DEDICATED_PA);
	assert_eq!(set_attr_partitions(), 1);
	assert_eq!(set_attr_attr(), ENCLAVE_MEM_ATTR_INSTR);
}

/// 请求的字节数按 2 MiB 向上取整得到分区数。
#[test]
fn test_set_partition_attr_rounds_the_byte_count_up_to_partitions() {
	world();

	assert!(mem::set_partition_attr(
		DEDICATED_PA,
		CHUNK_2M_SIZE + 1,
		ENCLAVE_MEM_ATTR_INSTR
	));
	assert_eq!(set_attr_partitions(), 2);
}

/// 区间越过授予边界而进入管理器镜像时整段被拒, 且不改动任何分区。
///
/// 这是准入判定的回归用例: 判据若只看区间起始的那一块分区, 上界那一块会被连同改判,
/// 管理器镜像首个分区内的页表与内核栈随即失去写权限, 而改判本身不会报错。
#[test]
fn test_set_partition_attr_rejects_a_range_reaching_into_the_image() {
	world();
	let start = MAN_PA_START - CHUNK_2M_SIZE;

	// 上界落在管理器镜像的首个分区之内。
	assert!(!mem::set_partition_attr(
		start,
		2 * CHUNK_2M_SIZE,
		ENCLAVE_MEM_ATTR_INSTR
	));
	assert_eq!(set_attr_calls(), 0);
}

/// 落在 S 模式页池内的分区被拒, 且不改动任何分区。
#[test]
fn test_set_partition_attr_rejects_a_partition_in_the_smode_pool() {
	world();

	assert!(!mem::set_partition_attr(
		GRANTED_END - CHUNK_2M_SIZE,
		CHUNK_2M_SIZE,
		ENCLAVE_MEM_ATTR_INSTR
	));
	assert_eq!(set_attr_calls(), 0);
}

/// 起始地址未按分区对齐时被拒, 且不改动任何分区。
///
/// 类别是分区的属性, 非对齐的起始地址只能覆盖分区的一部分, 改判会波及未请求的页。
#[test]
fn test_set_partition_attr_rejects_an_unaligned_start() {
	world();
	let unaligned = DEDICATED_PA + align_up(1, 8);

	assert!(!mem::set_partition_attr(
		unaligned,
		CHUNK_2M_SIZE,
		ENCLAVE_MEM_ATTR_INSTR
	));
	assert_eq!(set_attr_calls(), 0);
}

/// 字节数为 0 时被拒, 且不改动任何分区。
#[test]
fn test_set_partition_attr_rejects_a_zero_byte_count() {
	world();

	assert!(!mem::set_partition_attr(
		DEDICATED_PA,
		0,
		ENCLAVE_MEM_ATTR_INSTR
	));
	assert_eq!(set_attr_calls(), 0);
}

// ---------------------------------------------------------------
//  U 模式栈
// ---------------------------------------------------------------

/// U 模式栈自 argv 分区的末尾起映射: 栈底页是该分区的倒数第 256 页, 栈顶页是该分区的
/// 最后一页; 虚拟地址一侧, 栈顶页是 `UMODE_STACK_TOP_VA` 之下的一页。
///
/// 这是栈落点的回归用例: 栈若改自 argv 分区的首页起映射, 栈顶页与 argv 数据即落在同一
/// 张物理页上, 载荷写入栈会覆盖 argv。
#[test]
fn test_alloc_map_umode_stack_maps_the_argv_partition_tail() {
	world();

	assert_eq!(mem::alloc_map_umode_stack(ARGV_PA), UMODE_STACK_TOP_VA);

	let calls = paging::MAP_CALLS.load(std::sync::atomic::Ordering::SeqCst);
	let first_va = paging::FIRST_VA.load(std::sync::atomic::Ordering::SeqCst);
	let last_va = paging::LAST_VA.load(std::sync::atomic::Ordering::SeqCst);
	let first_pa = paging::FIRST_PA.load(std::sync::atomic::Ordering::SeqCst);
	let last_pa = paging::LAST_PA.load(std::sync::atomic::Ordering::SeqCst);
	assert_eq!(calls, UMODE_STACK_SIZE_TOTAL / PAGE_SIZE);
	assert_eq!(first_va, UMODE_STACK_TOP_VA - UMODE_STACK_SIZE_TOTAL);
	assert_eq!(last_va, UMODE_STACK_TOP_VA - PAGE_SIZE);
	assert_eq!(first_pa, ARGV_PA + CHUNK_2M_SIZE - UMODE_STACK_SIZE_TOTAL);
	assert_eq!(last_pa, ARGV_PA + CHUNK_2M_SIZE - PAGE_SIZE);
}

// ---------------------------------------------------------------
//  镜像布局核对
// ---------------------------------------------------------------

/// 只读段组越过分区 0 的边界一个字节即拒绝: 越过的字节落在按数据类登记的分区 1, 该分区
/// 只登记读与写权限, 其上的取指被拒绝。
#[test]
fn test_check_img_layout_rejects_read_only_sections_past_partition_0() {
	assert_eq!(
		mem::check_img_layout(
			MAN_PA_START,
			MAN_PA_START + CHUNK_2M_SIZE + 1,
			MAN_PA_START + 2 * CHUNK_2M_SIZE,
			2,
		),
		Err("sittim: RO sections overflow partition 0\n")
	);
}

/// 只读段组的末尾恰落在分区 0 的边界上时接受: 该组占满分区 0 且未侵入分区 1。
#[test]
fn test_check_img_layout_accepts_read_only_sections_ending_at_partition_0() {
	assert_eq!(
		mem::check_img_layout(
			MAN_PA_START,
			MAN_PA_START + CHUNK_2M_SIZE,
			MAN_PA_START + 2 * CHUNK_2M_SIZE,
			2,
		),
		Ok(())
	);
}

/// 镜像末尾不落在分区边界上时拒绝: 页池的起始物理地址随之不按分区对齐, 池内的页会横跨
/// 两个类别的分区。
#[test]
fn test_check_img_layout_rejects_an_unaligned_image_end() {
	assert_eq!(
		mem::check_img_layout(
			MAN_PA_START,
			MAN_PA_START + CHUNK_2M_SIZE,
			MAN_PA_START + 3 * CHUNK_2M_SIZE + PAGE_SIZE,
			4,
		),
		Err("sittim: image ed-addr is not aligned\n")
	);
}

/// 授予的分区数不足以覆盖按分区对齐后的镜像时拒绝。
#[test]
fn test_check_img_layout_rejects_granted_partitions_shorter_than_the_image() {
	assert_eq!(
		mem::check_img_layout(
			MAN_PA_START,
			MAN_PA_START + CHUNK_2M_SIZE,
			MAN_PA_START + 3 * CHUNK_2M_SIZE,
			2,
		),
		Err("sittim: granted partitions do not cover the image\n")
	);
}

// ---------------------------------------------------------------
//  mmap 映射区的块内切分状态
// ---------------------------------------------------------------

/// 正在切分的块的后备物理内存基址, 取一个按 2 MiB 对齐的地址。
const SPLIT_PA: u64 = 0x8800_0000;

/// 把映射区置为 "已交付 *pages* 页, 正在切分的块自 *pa* 起" 这一状态, 返回该状态下的
/// 映射区末端 VA。
///
/// 该状态由不足一块的请求交付后留下 (见 `mem::take_mmap_region`); 本入口内不申请分区,
/// 故直接写入。
fn set_mmap_state(pages: u64, pa: u64) -> u64 {
	world();
	context::ctx_mut().umode_mmap_pages_used = pages;
	context::ctx_mut().umode_curr_mmap_pa = pa;
	UMODE_MMAP_BASE + pages * PAGE_SIZE
}

/// 映射区末端 VA 处的原地长度变化, 参数按 `grow_mmap_in_place` 的 (起始 VA, 原长度,
/// 新长度) 给出。
fn resize_mmap_at_region_end(old_end: u64, new_end: u64) -> bool {
	mem::grow_mmap_in_place(
		UMODE_MMAP_BASE,
		old_end - UMODE_MMAP_BASE,
		new_end - UMODE_MMAP_BASE,
	)
}

/// 收缩后末页与原末页同块: 原地收缩成立, 页计数回退到新长度。
#[test]
fn test_mmap_shrink_inside_one_block_succeeds() {
	let old_end = set_mmap_state(700, SPLIT_PA);
	assert!(resize_mmap_at_region_end(
		old_end,
		UMODE_MMAP_BASE + 600 * PAGE_SIZE
	));
	assert_eq!(context::ctx().umode_mmap_pages_used, 600);
}

/// 收缩跨越块边界时拒绝: 收缩后的末页与原末页分处两个块, 而块内交付按正在切分的块的
/// 物理基址加块内偏移算出物理地址, 页计数落到另一个块之后, 该算式给出的物理页已由
/// 前一个块交付给别的 VA。
#[test]
fn test_mmap_shrink_across_a_block_boundary_is_refused() {
	let old_end = set_mmap_state(700, SPLIT_PA);
	assert!(!resize_mmap_at_region_end(
		old_end,
		UMODE_MMAP_BASE + 100 * PAGE_SIZE
	));
	assert_eq!(context::ctx().umode_mmap_pages_used, 700);
}

/// 收缩到块边界时成立: 收缩后的块内偏移为 0, 下一次不足一块的请求重新取得后备物理
/// 内存, 正在切分的块随之作废。
#[test]
fn test_mmap_shrink_to_a_block_boundary_succeeds() {
	let old_end = set_mmap_state(700, SPLIT_PA);
	assert!(resize_mmap_at_region_end(
		old_end,
		UMODE_MMAP_BASE + 512 * PAGE_SIZE
	));
	assert_eq!(context::ctx().umode_mmap_pages_used, 512);
}

/// 页计数落在块边界且没有正在切分的块时拒绝收缩: 后备物理内存基址为 0 时, 块内交付的
/// 算式给出授予分区之外的物理地址。
#[test]
fn test_mmap_shrink_at_a_block_boundary_without_a_split_block_is_refused() {
	let old_end = set_mmap_state(512, 0);
	assert!(!resize_mmap_at_region_end(
		old_end,
		UMODE_MMAP_BASE + 100 * PAGE_SIZE
	));
	assert_eq!(context::ctx().umode_mmap_pages_used, 512);
}

/// 同块内扩容成立: 新增页接在原末页之后, 物理地址为正在切分的块的物理基址加块内偏移。
#[test]
fn test_mmap_grow_inside_one_block_succeeds() {
	let old_end = set_mmap_state(600, SPLIT_PA);
	assert!(resize_mmap_at_region_end(old_end, old_end + 10 * PAGE_SIZE));
	assert_eq!(context::ctx().umode_mmap_pages_used, 610);
	assert_eq!(paging::FIRST_VA.load(Ordering::SeqCst), old_end);
	assert_eq!(
		paging::FIRST_PA.load(Ordering::SeqCst),
		SPLIT_PA + 88 * PAGE_SIZE
	);
}

/// 扩容跨越块边界时拒绝: 页计数落在块边界时正在切分的块的物理基址描述的是刚填满的
/// 前一个块, 按它加块内偏移算出的物理页与该块的首页重名。
#[test]
fn test_mmap_grow_across_a_block_boundary_is_refused() {
	let old_end = set_mmap_state(512, SPLIT_PA);
	assert!(!resize_mmap_at_region_end(old_end, old_end + PAGE_SIZE));
	assert_eq!(context::ctx().umode_mmap_pages_used, 512);
	assert_eq!(paging::MAP_CALLS.load(Ordering::SeqCst), 0);
}

/// 页计数落在块边界且没有正在切分的块时拒绝扩容。
#[test]
fn test_mmap_grow_at_a_block_boundary_without_a_split_block_is_refused() {
	let old_end = set_mmap_state(512, 0);
	assert!(!resize_mmap_at_region_end(old_end, old_end + PAGE_SIZE));
	assert_eq!(context::ctx().umode_mmap_pages_used, 512);
	assert_eq!(paging::MAP_CALLS.load(Ordering::SeqCst), 0);
}

/// 收缩到块边界后紧接一次块内收缩时拒绝: 前一次收缩使页计数落到块边界, 正在切分的块
/// 随之作废, 后一次收缩按它加块内偏移算出的物理页已由该块交付给别的 VA。
#[test]
fn test_mmap_shrink_after_a_shrink_to_a_block_boundary_is_refused() {
	let old_end = set_mmap_state(700, SPLIT_PA);
	assert!(resize_mmap_at_region_end(
		old_end,
		UMODE_MMAP_BASE + 512 * PAGE_SIZE
	));
	let boundary_end = UMODE_MMAP_BASE + 512 * PAGE_SIZE;
	assert!(!resize_mmap_at_region_end(
		boundary_end,
		boundary_end - 412 * PAGE_SIZE
	));
	assert_eq!(context::ctx().umode_mmap_pages_used, 512);
}

// ---------------------------------------------------------------
//  mmap 映射区的页计数: 交付后的增量与块边界取整
// ---------------------------------------------------------------

/// mmap 映射区可交付的页数。
fn mmap_region_pages() -> u64 {
	(UMODE_VA_LIMIT - UMODE_MMAP_BASE) >> PAGE_SHIFT
}

/// 指令类请求交付一块之后, 页计数推进一整块的页数。
///
/// 页计数以 4 KiB 页为单位, 一块 2 MiB 分区对应 `CHUNK_2M_PAGES` 页。增量若按块移位的
/// 位数计算, 每块会推进 2^21 页即 8 GiB 虚拟地址; 映射区因此很快越过用户空间上界,
/// 而其后的 VA 与低半区的 VA 共用根表项。
#[test]
fn test_take_mmap_region_advances_one_block_of_pages_for_an_instr_request() {
	let _ = set_mmap_state(0, 0);
	grant_alloc(1, SPLIT_PA);

	assert_eq!(
		mem::take_mmap_region(PAGE_SIZE, PTE_U | PTE_R | PTE_X),
		UMODE_MMAP_BASE
	);
	assert_eq!(context::ctx().umode_mmap_pages_used, CHUNK_2M_PAGES);
}

/// 达到一整块的请求先把页计数推进到块边界, 再按块推进, 两者都以 4 KiB 页计数。
///
/// 页计数为 100 时请求 512 页: 块内余量作废, 新块自页 512 起, 交付后页计数为 1024。
/// 边界取整若按块移位的位数计算, 页计数直接跳到 2^21; 增量若按同一移位计算, 交付后
/// 页计数为 2^22。
#[test]
fn test_take_mmap_region_rounds_to_the_block_boundary_in_pages() {
	let _ = set_mmap_state(100, SPLIT_PA);
	grant_alloc(1, 0x9000_0000);

	assert_eq!(
		mem::take_mmap_region(CHUNK_2M_SIZE, PTE_U | PTE_R | PTE_W),
		UMODE_MMAP_BASE + CHUNK_2M_PAGES * PAGE_SIZE
	);
	assert_eq!(context::ctx().umode_mmap_pages_used, 2 * CHUNK_2M_PAGES);
}

/// 任意长度的请求都落在块边界上, 页计数按交付的块数推进。
///
/// 四组取值覆盖块内交付与跨越整数个块: 3 KiB 不足一块, 按 4 KiB 页交付一页; 3 MiB 与
/// 7 MiB 与 11 MiB 各按 2 MiB 取整为 2 与 4 与 6 块, 块内余量作废。交付后的页计数分别
/// 为 1 与 1024 与 2048 与 3072。
#[test]
fn test_take_mmap_region_places_requests_of_various_lengths() {
	// (请求字节数, 交付的页数, 建立映射的次数, 交付后的页计数)
	let cases = [
		(3 * 1024u64, 1u64, 1u64, 1u64),
		(3u64 << 20, 768, 2, 1024),
		(7u64 << 20, 1792, 4, 2048),
		(11u64 << 20, 2816, 6, 3072),
	];

	for (bytes, pages, map_calls, expected_used) in cases {
		let _ = set_mmap_state(0, 0);
		grant_alloc(1, SPLIT_PA);

		assert_eq!(
			mem::take_mmap_region(bytes, PTE_U | PTE_R | PTE_W),
			UMODE_MMAP_BASE,
			"bytes={bytes}"
		);
		assert_eq!(
			align_up(bytes, PAGE_SIZE) >> PAGE_SHIFT,
			pages,
			"bytes={bytes}"
		);
		assert_eq!(
			paging::MAP_CALLS.load(Ordering::SeqCst),
			map_calls,
			"bytes={bytes}"
		);
		assert_eq!(
			context::ctx().umode_mmap_pages_used,
			expected_used,
			"bytes={bytes}"
		);
	}
}

/// 请求越过映射区容量时拒绝交付, 且不申请分区也不建立映射。
#[test]
fn test_take_mmap_region_rejects_a_request_past_the_region_capacity() {
	let region_pages = mmap_region_pages();
	let _ = set_mmap_state(region_pages - 1, SPLIT_PA);
	grant_alloc(1, SPLIT_PA);

	assert_eq!(
		mem::take_mmap_region(2 * PAGE_SIZE, PTE_U | PTE_R | PTE_W),
		0
	);
	assert_eq!(ecall_aux::ALLOC_CALLS.load(Ordering::SeqCst), 0);
	assert_eq!(paging::MAP_CALLS.load(Ordering::SeqCst), 0);
	assert_eq!(context::ctx().umode_mmap_pages_used, region_pages - 1);
}

/// 请求长度为 0 或不小于用户空间上界时拒绝, 且不申请分区。
///
/// 长度不小于上界时按页取整会向上溢出而给出一个小值, 后续算式因此接受一个远大于映射区
/// 的区间。
#[test]
fn test_take_mmap_region_rejects_a_length_outside_the_window() {
	let _ = set_mmap_state(0, 0);
	grant_alloc(1, SPLIT_PA);

	assert_eq!(mem::take_mmap_region(0, PTE_U | PTE_R | PTE_W), 0);
	assert_eq!(
		mem::take_mmap_region(UMODE_VA_LIMIT + PAGE_SIZE, PTE_U | PTE_R | PTE_W),
		0
	);
	assert_eq!(ecall_aux::ALLOC_CALLS.load(Ordering::SeqCst), 0);
}

/// 固定地址映射落在用户空间之外时拒绝, 且不申请分区也不建立映射。
#[test]
fn test_map_umode_pages_rejects_a_range_past_the_user_window() {
	world();
	grant_alloc(1, DEDICATED_PA);

	assert!(!mem::map_umode_pages(
		UMODE_VA_LIMIT - PAGE_SIZE,
		2,
		PTE_U | PTE_R | PTE_W
	));
	assert_eq!(ecall_aux::ALLOC_CALLS.load(Ordering::SeqCst), 0);
	assert_eq!(paging::MAP_CALLS.load(Ordering::SeqCst), 0);
}

/// 按块增长止于用户空间上界: 请求越过上界时只映射到上界, 返回的上界即上界。
#[test]
fn test_grow_umode_va_stops_at_the_user_window() {
	world();
	grant_alloc(1, DEDICATED_PA);

	let from = UMODE_VA_LIMIT - CHUNK_2M_SIZE;
	assert_eq!(
		mem::grow_umode_va(from, UMODE_VA_LIMIT + CHUNK_2M_SIZE, PTE_U | PTE_R | PTE_W),
		UMODE_VA_LIMIT
	);
	assert_eq!(paging::MAP_CALLS.load(Ordering::SeqCst), 1);
}

/// 堆顶越过用户空间上界时拒绝增长, 返回旧值, 且不申请分区。
#[test]
fn test_sys_brk_handler_rejects_a_top_past_the_user_window() {
	world();
	context::ctx_mut().umode_heap_top = UMODE_HEAP_START_ALIGNED;
	context::ctx_mut().umode_heap_mapped_end = UMODE_HEAP_START_ALIGNED;

	assert_eq!(
		mem::sys_brk_handler(UMODE_VA_LIMIT + 1),
		UMODE_HEAP_START_ALIGNED
	);
	assert_eq!(ecall_aux::ALLOC_CALLS.load(Ordering::SeqCst), 0);
	assert_eq!(context::ctx().umode_heap_top, UMODE_HEAP_START_ALIGNED);
}

/// 新长度使末址向上溢出时拒绝原地长度变化。
///
/// 溢出后的末址落在映射区基址之前, 页计数由它减出基址得到的是一个远大于容量的值;
/// 收缩分支据该值回退页计数, 随后的一次不足一块的请求在新的块边界上取下后备物理内存
/// 并按块内偏移算出物理地址, 与已交付的页重名。
#[test]
fn test_mmap_resize_with_an_overflowing_end_is_refused() {
	let _ = set_mmap_state(700, SPLIT_PA);
	let new_size = u64::MAX - UMODE_MMAP_BASE + 1;

	assert!(!mem::grow_mmap_in_place(
		UMODE_MMAP_BASE,
		700 * PAGE_SIZE,
		new_size
	));
	assert_eq!(context::ctx().umode_mmap_pages_used, 700);
}
