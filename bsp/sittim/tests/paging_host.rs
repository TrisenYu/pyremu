//! paging 模块树在主机目标上的单元测试入口。
//!
//! 本 crate 是 RISC-V 裸机目标上的可执行程序, 其中的 bin 目标只能在目标上构建, 故
//! paging 模块由本入口按源码包含, 由主机上的测试工具链编译。树内引用的平台符号由本
//! 文件声明。
//!
//! 页表的安装与遍历需要真实的页表根与 S 模式页池。主机上没有二者, 本入口以进程内的
//! 静态内存充当: 页表根由各用例自行构造并经 set_root 登记, 页池由下方 mem 桩交付。
//! 未覆盖的是依赖 satp 换算的部分 —— csr 桩使 read_satp 恒返回 0, 中间页表一律按物理
//! 地址解引用, 线性别名窗口 (LINEAR_MAP_OFFSET) 不参与。

#![allow(dead_code)]

#[path = "../src/constants.rs"]
mod constants;

/// 主机上没有进程上下文, 页表根取 0。
mod context {
	pub fn root_pa() -> u64 {
		0
	}
}

/// 主机上没有 satp 寄存器, 读回 0。
mod csr {
	pub fn read_satp() -> u64 {
		0
	}
}

/// 主机上没有 S 模式页池。以进程内一块 4 KiB 对齐的静态内存充当: 页表页只要求按
/// 4 KiB 对齐、且能经页表项给出的物理地址寻址, 这两点静态数组同样满足。
mod mem {
	use std::sync::atomic::{ AtomicUsize, Ordering };

	const POOL_PAGES: usize = 64;

	#[derive(Clone, Copy)]
	#[repr(align(4096))]
	struct Page([u8; 4096]);

	static mut POOL: [Page; POOL_PAGES] = [Page([0; 4096]); POOL_PAGES];
	static CURSOR: AtomicUsize = AtomicUsize::new(0);

	/// 交付池中连续的 n 页, 返回首页地址; 余量不足时返回 `!0`。
	///
	/// 页表页由调用方自行清零, 故此处不写入内容。游标只前进不回退, 与目标侧的
	/// 池分配器一致; 各用例按各自取得的页写入, 无共享。
	pub fn try_alloc_smode_page(n: u64) -> u64 {
		let first = CURSOR.fetch_add(n as usize, Ordering::SeqCst);
		if first + n as usize > POOL_PAGES {
			return !0u64;
		}
		core::ptr::addr_of_mut!(POOL) as u64 + (first as u64) * 4096
	}

	/// U 模式页池与 mmap 分区块在主机上不交付: 需要它们的用例 (进程式 clone 的地址
	/// 空间复制) 不在本入口的覆盖范围内。
	pub fn alloc_umode_page(_n: u64) -> u64 {
		!0u64
	}

	pub fn alloc_mmap_chunk_pa() -> u64 {
		!0u64
	}
}

#[path = "../src/mem_prim/mod.rs"]
mod mem_prim;

#[path = "../src/paging.rs"]
mod paging;
