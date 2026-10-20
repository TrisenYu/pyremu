//! 内存类系统调用层的参数校验在主机目标上的单元测试入口。
//!
//! 本层在进入映射之前拒绝非法参数: 长度为 0 或越过用户空间上界、MAP_FIXED 的地址未按页
//! 对齐、区间越过用户空间上界。`paging` 由本入口的替身记录调用, 故用例断言的是本层返回
//! 的错误码以及是否触及页表遍历与分区申请。
//!
//! 区间越过用户空间上界有两个来源: 末址的加法向上溢出, 以及末址落在 `UMODE_VA_LIMIT`
//! 之上。后者的危害不在本层之内: Sv39 按 VPN 的 9 位字段索引根表, 越界的 VA 与低半区的
//! 某个 VA 共用根表项, 按它解除映射或改写权限会改到另一个 VA 的页表项上。

#![allow(dead_code)]
#![allow(unused_imports)]

use core::sync::atomic::Ordering;

#[path = "../src/constants.rs"]
mod constants;

#[path = "../src/mem_prim/mod.rs"]
mod mem_prim;

#[path = "../src/syscall/errno.rs"]
mod errno;

pub use errno::*;

/// 主机侧的 `paging`, 只声明 `mem` 与 `context` 引用到的符号, 并记录每一次页表操作。
mod paging {
	use core::sync::atomic::{ AtomicU64, Ordering };

	use crate::constants::{ LEVEL_PAGE, PTE_R, PTE_W };

	/// 页表项。`context` 以它声明 4 KiB 对齐的页表根。
	#[derive(Clone, Copy)]
	pub struct Pte(pub u64);

	/// 页表项安装的失败原因。
	#[derive(Debug, Clone, Copy, PartialEq, Eq)]
	pub enum MapError {
		/// 页表页分配失败 (S-mode 页池耗尽)。
		OutOfPool,
	}

	/// 映射请求的次数。
	pub static MAP_CALLS: AtomicU64 = AtomicU64::new(0);
	/// 页层级查询的次数。
	pub static VA_LEVEL_CALLS: AtomicU64 = AtomicU64::new(0);
	/// 解除映射的次数。
	pub static UNMAP_CALLS: AtomicU64 = AtomicU64::new(0);

	/// 记录一次映射请求并报告成功。
	pub fn map_page(_vaddr: u64, _paddr: u64, _flags: u64, _level: u8) -> Result<(), MapError> {
		MAP_CALLS.fetch_add(1, Ordering::SeqCst);
		Ok(())
	}

	/// 记录一次页层级查询并报告 4 KiB 叶子。
	pub fn va_real_page_level(_va: u64) -> Option<u8> {
		VA_LEVEL_CALLS.fetch_add(1, Ordering::SeqCst);
		Some(LEVEL_PAGE)
	}

	/// 记录一次解除映射。
	pub fn unmap_page(_va: u64, _level: u8) {
		UNMAP_CALLS.fetch_add(1, Ordering::SeqCst);
	}

	/// 本入口内的用例不查询物理地址。
	pub fn get_pa(_va: u64) -> Option<u64> {
		None
	}

	/// 叶子项的标志位: 一律给出数据页的标志位。
	pub fn leaf_pte_flags(_va: u64) -> Option<u64> {
		Some(PTE_R | PTE_W)
	}

	/// 改写叶子项的标志位, 恒成功。
	pub fn set_leaf_flags(_va: u64, _flags: u64) -> bool {
		true
	}

	/// 拆分 2 MiB 超页, 恒成功。
	pub fn split_mega_leaf(_va: u64) -> bool {
		true
	}
}

/// 主机侧的 `hang`。本入口内的用例不触发挂起。
mod hang {
	pub fn fault_halt(_msg: &str) -> ! {
		panic!("fault_halt: {_msg}");
	}
}

/// 主机侧的 `ecall_aux`, 记录分区申请。
mod ecall_aux {
	use core::sync::atomic::{ AtomicU64, Ordering };

	/// 分区申请的次数与授予的分区块数。授予为 0 时按池中无可用分区处理。
	pub static ALLOC_CALLS: AtomicU64 = AtomicU64::new(0);
	pub static ALLOC_GRANTED: AtomicU64 = AtomicU64::new(0);
	/// 授予分区的物理起始地址。
	pub static ALLOC_PA: AtomicU64 = AtomicU64::new(0);

	/// 记录一次分区申请。请求多于授予时只授予请求的块数。
	pub fn enclave_call_mem_alloc(chunk_nums: u64, _attr: u64) -> (u64, u64) {
		ALLOC_CALLS.fetch_add(1, Ordering::SeqCst);
		let granted = ALLOC_GRANTED.load(Ordering::SeqCst);
		if granted == 0 {
			return (0, !0u64);
		}
		let granted = if granted < chunk_nums { granted } else { chunk_nums };
		(granted, ALLOC_PA.load(Ordering::SeqCst))
	}

	/// 本入口内的用例不请求类别改判。
	pub fn enclave_call_set_mem_attr(_paddr: u64, _partition_num: u64, _attr: u64) -> u64 {
		0
	}
}

#[path = "../src/context.rs"]
mod context;

#[path = "../src/mem.rs"]
mod mem;

#[path = "../src/syscall/mmap.rs"]
mod mmap;

use constants::*;

// ---------------------------------------------------------------
//  平台布局
// ---------------------------------------------------------------

/// 飞地管理器镜像的物理起始地址。
const MAN_PA_START: u64 = 0x8020_0000;
/// 授予分区之外的一块分区, 即由单个申请独占的分区。
const DEDICATED_PA: u64 = 0xA000_0000;

// mmap 与 mremap 的标志位与保护位, 与 Linux rv64 ABI 一致。
const MAP_FIXED: u64 = 0x10;
const MAP_ANONYMOUS: u64 = 0x20;
const MAP_PRIVATE: u64 = 0x02;
const PROT_READ: u64 = 0x1;
const PROT_WRITE: u64 = 0x2;
const MREMAP_MAYMOVE: u64 = 0x1;
const MREMAP_FIXED: u64 = 0x2;

// ---------------------------------------------------------------
//  共享状态
// ---------------------------------------------------------------

static WORLD: std::sync::Once = std::sync::Once::new();

/// 建立飞地上下文并清空页表与分区申请的记录, 使下一次分区申请授予一块分区。
fn world() {
	WORLD.call_once(|| {
		context::init_context(MAN_PA_START, 0x1000);
	});
	paging::MAP_CALLS.store(0, Ordering::SeqCst);
	paging::VA_LEVEL_CALLS.store(0, Ordering::SeqCst);
	paging::UNMAP_CALLS.store(0, Ordering::SeqCst);
	ecall_aux::ALLOC_CALLS.store(0, Ordering::SeqCst);
	ecall_aux::ALLOC_GRANTED.store(1, Ordering::SeqCst);
	ecall_aux::ALLOC_PA.store(DEDICATED_PA, Ordering::SeqCst);
}

// ---------------------------------------------------------------
//  mmap
// ---------------------------------------------------------------

/// 长度为 0 返回 EINVAL。
#[test]
fn test_mmap_rejects_a_zero_length() {
	world();

	assert_eq!(
		mmap::mmap_handler(0, 0, PROT_READ, MAP_ANONYMOUS | MAP_PRIVATE, 0, 0),
		EINVAL
	);
	assert_eq!(ecall_aux::ALLOC_CALLS.load(Ordering::SeqCst), 0);
}

/// 长度越过用户空间上界返回 EINVAL, 且不申请分区。
///
/// 长度不小于上界时按页取整会向上溢出而给出一个小值, 后续算式因此接受一个远大于映射区
/// 的区间。
#[test]
fn test_mmap_rejects_a_length_past_the_user_window() {
	world();

	assert_eq!(
		mmap::mmap_handler(
			0,
			UMODE_VA_LIMIT + PAGE_SIZE,
			PROT_READ,
			MAP_ANONYMOUS | MAP_PRIVATE,
			0,
			0
		),
		EINVAL
	);
	assert_eq!(ecall_aux::ALLOC_CALLS.load(Ordering::SeqCst), 0);
}

/// MAP_FIXED 的地址未按页对齐返回 EINVAL, 且不建立映射。
///
/// 按 Linux 的约定, 地址只在该标志之下有意义, 故未对齐的地址是调用方的参数错误, 不得
/// 向下取整后就地映射。
#[test]
fn test_mmap_fields_fixed_rejects_an_unaligned_address() {
	world();

	assert_eq!(
		mmap::mmap_handler(
			UMODE_MMAP_BASE + 1,
			PAGE_SIZE,
			PROT_READ,
			MAP_ANONYMOUS | MAP_PRIVATE | MAP_FIXED,
			0,
			0
		),
		EINVAL
	);
	assert_eq!(paging::MAP_CALLS.load(Ordering::SeqCst), 0);
	assert_eq!(ecall_aux::ALLOC_CALLS.load(Ordering::SeqCst), 0);
}

/// MAP_FIXED 的区间越过用户空间上界返回 ENOMEM, 且不申请分区。
#[test]
fn test_mmap_fields_fixed_rejects_a_range_past_the_user_window() {
	world();

	assert_eq!(
		mmap::mmap_handler(
			UMODE_VA_LIMIT - PAGE_SIZE,
			2 * PAGE_SIZE,
			PROT_READ,
			MAP_ANONYMOUS | MAP_PRIVATE | MAP_FIXED,
			0,
			0
		),
		ENOMEM
	);
	assert_eq!(ecall_aux::ALLOC_CALLS.load(Ordering::SeqCst), 0);
	assert_eq!(paging::MAP_CALLS.load(Ordering::SeqCst), 0);
}

// ---------------------------------------------------------------
//  munmap
// ---------------------------------------------------------------

/// 地址未按页对齐或长度为 0 时返回 EINVAL。
#[test]
fn test_munmap_rejects_malformed_arguments() {
	world();

	assert_eq!(mmap::munmap_handler(UMODE_MMAP_BASE + 1, PAGE_SIZE), EINVAL);
	assert_eq!(mmap::munmap_handler(UMODE_MMAP_BASE, 0), EINVAL);
	assert_eq!(paging::VA_LEVEL_CALLS.load(Ordering::SeqCst), 0);
}

/// 区间越过用户空间上界返回 EINVAL, 且不遍历页表。
///
/// 取管理器自身的线性映射窗口作为越界的地址: 该 VA 的 VPN 与用户空间中的某个 VA 相同,
/// 旧实现会按它走一遍页表并解除所命中的页表项。
#[test]
fn test_munmap_rejects_a_range_past_the_user_window() {
	world();

	assert_eq!(mmap::munmap_handler(LINEAR_MAP_OFFSET, CHUNK_2M_SIZE), EINVAL);
	assert_eq!(paging::VA_LEVEL_CALLS.load(Ordering::SeqCst), 0);
	assert_eq!(paging::UNMAP_CALLS.load(Ordering::SeqCst), 0);
}

/// 末址的加法向上溢出时返回 EINVAL。
#[test]
fn test_munmap_rejects_an_overflowing_end() {
	world();

	assert_eq!(
		mmap::munmap_handler(UMODE_VA_LIMIT - PAGE_SIZE, 2 * PAGE_SIZE),
		EINVAL
	);
	assert_eq!(paging::VA_LEVEL_CALLS.load(Ordering::SeqCst), 0);
}

// ---------------------------------------------------------------
//  mprotect
// ---------------------------------------------------------------

/// 区间越过用户空间上界返回 EINVAL, 且不遍历页表。
#[test]
fn test_mprotect_rejects_a_range_past_the_user_window() {
	world();

	assert_eq!(
		mmap::mprotect_handler(LINEAR_MAP_OFFSET, CHUNK_2M_SIZE, PROT_READ),
		EINVAL
	);
	assert_eq!(paging::VA_LEVEL_CALLS.load(Ordering::SeqCst), 0);
}

/// 末址的加法向上溢出时返回 EINVAL。
#[test]
fn test_mprotect_rejects_an_overflowing_end() {
	world();

	assert_eq!(
		mmap::mprotect_handler(UMODE_VA_LIMIT - PAGE_SIZE, 2 * PAGE_SIZE, PROT_READ),
		EINVAL
	);
	assert_eq!(paging::VA_LEVEL_CALLS.load(Ordering::SeqCst), 0);
}

// ---------------------------------------------------------------
//  mremap
// ---------------------------------------------------------------

/// 旧区间越过用户空间上界返回 EINVAL, 且不申请分区。
#[test]
fn test_mremap_rejects_an_old_range_past_the_user_window() {
	world();

	assert_eq!(
		mmap::mremap_handler(
			UMODE_VA_LIMIT - PAGE_SIZE,
			2 * PAGE_SIZE,
			PAGE_SIZE,
			0,
			0
		),
		EINVAL
	);
	assert_eq!(ecall_aux::ALLOC_CALLS.load(Ordering::SeqCst), 0);
}

/// 新长度越过用户空间上界返回 EINVAL, 且不申请分区。
#[test]
fn test_mremap_rejects_a_new_size_past_the_user_window() {
	world();

	assert_eq!(
		mmap::mremap_handler(UMODE_MMAP_BASE, PAGE_SIZE, UMODE_VA_LIMIT + PAGE_SIZE, 0, 0),
		EINVAL
	);
	assert_eq!(ecall_aux::ALLOC_CALLS.load(Ordering::SeqCst), 0);
}

/// MREMAP_FIXED 的目的区间越过用户空间上界返回 ENOMEM, 且不申请分区。
#[test]
fn test_mremap_fields_fixed_rejects_a_destination_past_the_user_window() {
	world();

	assert_eq!(
		mmap::mremap_handler(
			UMODE_MMAP_BASE,
			PAGE_SIZE,
			PAGE_SIZE,
			MREMAP_MAYMOVE | MREMAP_FIXED,
			UMODE_VA_LIMIT
		),
		ENOMEM
	);
	assert_eq!(ecall_aux::ALLOC_CALLS.load(Ordering::SeqCst), 0);
	assert_eq!(paging::MAP_CALLS.load(Ordering::SeqCst), 0);
}
