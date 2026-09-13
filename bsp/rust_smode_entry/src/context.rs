//! 全局 EnclaveContext —— 替代 ref-emod 中分散在 8+ 个 .c 文件里的 ~30 个 file-static 全局变量。
//! 启动阶段一次写入，此后任意读取。

use core::cell::UnsafeCell;
use core::mem::MaybeUninit;

use crate::constants::PAGE_SIZE;
use crate::paging::Pte;

// ---------------------------------------------------------------
//  PoolDesc
// ---------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct PoolDesc {
	pub offset: u64,
	pub size: u64,
	pub used_pages: u64,
}

impl PoolDesc {
	#[allow(dead_code)]
	pub const fn empty() -> Self {
		Self {
			offset: 0,
			size: 0,
			used_pages: 0,
		}
	}

	pub fn avail_bytes(&self) -> u64 {
		self.size.saturating_sub(self.used_pages * PAGE_SIZE)
	}
}

// ---------------------------------------------------------------
//  SharedBuf
// ---------------------------------------------------------------

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub struct SharedBuf {
	pub pa: u64,
	pub va: u64,
}

// ---------------------------------------------------------------
//  页表根 — 独立 static 强制 4 KiB 对齐
// ---------------------------------------------------------------

/// 页表根（512 项，必须 4 KiB 对齐）。
///
/// 独立 wrapper 结构体以 `#[repr(align(4096))]` 强制对齐,
/// **不依赖 BSS 布局** (BSS 中其他零初始化静态可能破坏
/// `EnclaveContext` 内嵌字段的 4 KiB 对齐约束, 导致
/// satp.PPN 指向错误物理页 -> MMU 开启后全部取指页错误
/// -> trap loop -> hart halt).
#[repr(align(4096))]
#[allow(dead_code)]
pub struct PageTableRoot([Pte; 512]);

static mut PAGE_TABLE_ROOT: PageTableRoot = PageTableRoot([Pte(0); 512]);

// ---------------------------------------------------------------
//  EnclaveContext
// ---------------------------------------------------------------

pub struct EnclaveContext {
	/// 飞地管理器物理起始地址。
	pub manager_pa_start: u64,
	/// 下一个飞地模块的加载 VA。
	#[allow(dead_code)]
	pub enclave_module_load_va: u64,
	/// U-mode 堆顶 (brk 返回值, 字节粒度)。
	pub umode_heap_top: u64,
	/// U-mode 堆已映射到的 VA 上界, 恒为 2 MiB 对齐。
	pub umode_heap_mapped_end: u64,
	/// mmap 匿名映射已交付的 4 KiB 页计数 (自 UMODE_MMAP_BASE 起)。
	/// 右移 CHUNK_2M_SHIFT 位是已彻底写满的 2 MiB 块数, 也就是游标当前所在块的
	/// 序号; 与块内页数掩码按位与是游标当下所在块内已交付的页数。
	/// 例: 511 表示 0 块写满、当前块已用 511 页, 下一页仍在本块内; 512 表示 1 块
	/// 写满、当前块自 0 起用。映射 VA 与块边界都由该计数导出
	pub umode_mmap_pages_used: u64,
	/// 当前 2 MiB 块的物理基址, 仅当该块按 4 KiB 页切分时有效;
	/// 0 表示没有正在切分的块 (游标所在块由 2 MiB 超页覆盖, 或尚无映射)。
	/// 每块的后备物理内存是独立一次分配 (页池或 M-mode 分区), 块间不保证连续,
	/// 无法由页计数导出; 已交付页的物理地址另有页表记录, 故只保留当前块。
	pub umode_curr_mmap_pa: u64,
	pub umode_pool: PoolDesc,
	pub smode_pool: PoolDesc,
	#[allow(dead_code)]
	pub shared_buffer: Option<SharedBuf>,
	pub umode_pool_pa_aligned: u64,
	/// 飞地时间配额 (timer interrupt 次数), 0=不限.
	pub time_quota: u64,
	/// 当前时间片内已消耗的 timer interrupt 次数.
	pub ticks_consumed: u64,
}

// ---------------------------------------------------------------
//  OnceCell —— 单次写入，多次读取
// ---------------------------------------------------------------

struct OnceCell<T> {
	value: UnsafeCell<MaybeUninit<T>>,
	ready: UnsafeCell<bool>,
}

unsafe impl<T> Sync for OnceCell<T> {}

impl<T> OnceCell<T> {
	const fn new() -> Self {
		Self {
			value: UnsafeCell::new(MaybeUninit::uninit()),
			ready: UnsafeCell::new(false),
		}
	}

	/// 逐字段初始化 — 完全避免大结构体的 memcpy。
	///
	/// `f` 接收一个指向未初始化内存的 `*mut T` 裸指针,
	/// 调用方负责逐字段写入。适用于 CTX 已在 BSS 中全零、
	/// 仅需设置少量非零字段的场景。
	fn init_direct(&self, f: impl FnOnce(*mut T)) {
		let ready = unsafe { &mut *self.ready.get() };
		if *ready {
			panic!("EnclaveContext::init_direct called twice");
		}
		f(self.value.get() as *mut T);
		*ready = true;
	}

	#[allow(dead_code)]
	fn set(&self, val: T) {
		self.init_direct(|ptr| unsafe {
			// ptr::write 会逐字段移动, 对 [Pte; 512] 零值数组
			// 编译器应生成简单的循环而非 memcpy.
			core::ptr::write(ptr, val);
		});
	}

	fn get(&self) -> &T {
		assert!(
			*unsafe { &*self.ready.get() },
			"EnclaveContext not initialised"
		);
		unsafe { (*self.value.get()).assume_init_ref() }
	}

	#[allow(clippy::mut_from_ref)]
	fn get_mut(&self) -> &mut T {
		assert!(
			*unsafe { &*self.ready.get() },
			"EnclaveContext not initialised"
		);
		unsafe { (*self.value.get()).assume_init_mut() }
	}
}

static CTX: OnceCell<EnclaveContext> = OnceCell::new();

// ---------------------------------------------------------------
//  Public API
// ---------------------------------------------------------------

/// 初始化 CTX — 逐字段写入, 不依赖任何 memcpy.
/// CTX 位于 BSS 段, 启动时已全零, 故仅需设置非零字段.
/// PAGE_TABLE_ROOT 独立于 CTX 以 `#[repr(align(4096))]` 强制对齐.
pub fn init_context(man_pa_start: u64, enclave_module_load_va: u64) {
	CTX.init_direct(|ptr| unsafe {
		(*ptr).manager_pa_start = man_pa_start;
		(*ptr).enclave_module_load_va = enclave_module_load_va;
		(*ptr).time_quota = crate::constants::TIME_QUOTA;
		// 其余字段 = 0 (umode_heap_top, pools, ticks_consumed, shared_buffer, ...)
		// BSS 已保证全零, 无需显式赋值
	});
}

#[inline]
pub fn ctx() -> &'static EnclaveContext {
	CTX.get()
}

/// 返回页表根的**当前映射地址** (始终 4 KiB 对齐).
///
/// 运行时按 PIE 链接, 代码以 PC 相对方式引用 `PAGE_TABLE_ROOT`:
/// 物理恒等主流程 (MMU 关闭阶段与 MMU 开启后的 after_mmu) 取到 PA,
/// 虚拟映射的陷态处理 (stvec = VA) 取到 VA, 两者指向同一物理页且
/// 均已被映射, 可直接解引用 (见 paging::root_table)。
///
/// 需要真正物理地址的调用方 (如 `init_satp` 计算 PPN) 只能在物理恒等
/// 主流程调用, 此刻返回的即为 PA。
#[inline]
pub fn root_pa() -> u64 {
	&raw const PAGE_TABLE_ROOT as u64
}

#[inline]
pub fn ctx_mut() -> &'static mut EnclaveContext {
	CTX.get_mut()
}
