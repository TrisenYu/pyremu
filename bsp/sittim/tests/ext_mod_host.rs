//! ext_mod 模块树在主机目标上的单元测试入口。
//!
//! 本 crate 是 RISC-V 裸机目标上的可执行程序, 其中的 bin 目标只能在目标上构建, 故
//! ext_mod 模块由本入口按源码包含, 由主机上的测试工具链编译。树内引用的平台符号由本
//! 文件声明。
//!
//! 取入一个模块时对平台的依赖有三处: 模块窗口内下一个空闲 VA, 映像槽位的物理页与页表
//! 页, 以及验签。`context` 与 `attest` 取 `src/` 内的实现, `mem` 与 `paging` 由本文件
//! 给出, 理由见这两个模块的说明。
//!
//! `ModuleDesc` 的尺寸与字段偏移、`ModuleInterface` 经隐藏指针返回的约定, 在
//! `src/ext_mod/abi.rs` 的用例内, 由本入口一并执行; 映像描述块各字段的核对在
//! `src/ext_mod/img_descriptor.rs` 的用例内。
//!
//! 不在覆盖范围内的一项: `LoadError::AttestFailed`。默认配置下 `ATTEST_ENABLE` 为假,
//! 该分支不可达; 置真时主机侧无法给出带有效签名的映像, 故本入口内到达模块入口的用例在
//! 验签使能时一并跳过。
//!
//! 本入口内的一项限制, 与飞地内的取入无关: 主机侧充当映像的是一段固定的映射 (见
//! [`image_va`]), 而每次取入按槽位的字节数推进 `enclave_module_load_va`, 故同一次 `prepare` 之内
//! 对同一个模块编号只取入一次 —— 第二次会以该映射之后的地址当作映像的加载地址, 那里没有
//! 描述块。模块以无重定位的映像交付 (见 `src/ext_mod/abi.rs`), 在飞地内于模块窗口的
//! 任何地址都能执行, 故飞地内不存在这一限制。

#![allow(dead_code)]

use core::cell::Cell;
use core::ptr;
use core::sync::atomic::{ AtomicU32, AtomicU64, Ordering };
use std::sync::{ Mutex, MutexGuard, Once };

#[path = "../src/constants.rs"]
mod constants;

/// 主机侧的 `mem`, 以固定物理地址回答分区申请与类别改判。
///
/// `ext_mod::loader` 对内存的全部使用只有三项: 建立映像槽位时的 `alloc_adjacent_pages`,
/// 为该槽位建成映射的 `map_pages_at`, 与可执行部分写入完毕后的 `set_partition_attr`。
/// 三者自身的记账由 `src/mem.rs` 的用例覆盖, 本入口因此以本模块代替它, 同时判定五件事:
/// 可执行部分与可写部分各申请一次分区, 两次请求的类别都取数据类; 申请失败 (返回 `!0u64`)
/// 时取入以 `LoadError::MapFailed` 结束, 且不建立任何映射; 映射按 4 KiB 页推进; 改判请求的
/// 长度等于可执行部分的字节数; 改判失败时同样以 `LoadError::MapFailed` 结束。
mod mem {
	use super::constants::{ LEVEL_PAGE, PAGE_SIZE };
	use super::paging;
	use core::sync::atomic::{ AtomicBool, AtomicU64, Ordering };

	/// 分配得到的分区物理起始地址。
	pub const POOL_PA: u64 = 0x9000_0000;

	/// 最近一次分配请求的页数, `ALLOC_CALLS` 为 0 时无意义。
	pub static ALLOC_PAGES: AtomicU64 = AtomicU64::new(0);
	/// 最近一次分配请求的类别。
	pub static ALLOC_ATTR: AtomicU64 = AtomicU64::new(0);
	/// 分配请求的次数。
	pub static ALLOC_CALLS: AtomicU64 = AtomicU64::new(0);
	/// 置位时报告分区池耗尽。
	pub static ALLOC_FAIL: AtomicBool = AtomicBool::new(false);

	/// 改判请求的次数, 以及最近一次改判的物理基址、字节数与类别。
	pub static SET_ATTR_CALLS: AtomicU64 = AtomicU64::new(0);
	pub static SET_ATTR_PA: AtomicU64 = AtomicU64::new(0);
	pub static SET_ATTR_BYTES: AtomicU64 = AtomicU64::new(0);
	pub static SET_ATTR_ATTR: AtomicU64 = AtomicU64::new(0);
	/// 置位时报告改判失败。
	pub static SET_ATTR_FAIL: AtomicBool = AtomicBool::new(false);

	/// 记录一次分区申请, 按 `ALLOC_FAIL` 给出结果。
	pub fn alloc_adjacent_pages(pages: u64, attr: u64) -> u64 {
		ALLOC_CALLS.fetch_add(1, Ordering::SeqCst);
		ALLOC_PAGES.store(pages, Ordering::SeqCst);
		ALLOC_ATTR.store(attr, Ordering::SeqCst);
		if ALLOC_FAIL.load(Ordering::SeqCst) {
			return !0u64;
		}
		POOL_PA
	}

	/// 逐页建立映射, 与 `src/mem.rs` 的同名函数一致: 某一页失败即返回假, 已建立的页保留。
	pub fn map_pages_at(va: u64, pa: u64, pages: u64, flags: u64) -> bool {
		for i in 0..pages {
			if paging::map_page(va + i * PAGE_SIZE, pa + i * PAGE_SIZE, flags, LEVEL_PAGE)
				.is_err()
			{
				return false;
			}
		}
		true
	}

	/// 记录一次类别改判, 按 `SET_ATTR_FAIL` 给出结果。
	pub fn set_partition_attr(pa: u64, bytes: u64, attr: u64) -> bool {
		SET_ATTR_CALLS.fetch_add(1, Ordering::SeqCst);
		SET_ATTR_PA.store(pa, Ordering::SeqCst);
		SET_ATTR_BYTES.store(bytes, Ordering::SeqCst);
		SET_ATTR_ATTR.store(attr, Ordering::SeqCst);
		!SET_ATTR_FAIL.load(Ordering::SeqCst)
	}
}

/// 主机侧的 `paging`, 以固定结果代替 `src/paging.rs` 的页表项安装。
///
/// `ext_mod::loader` 对页表的全部使用只有两项: 映像槽位的逐页 `map_page`, 与可执行部分
/// 写入完毕后的逐页 `set_leaf_flags`。页表项安装自身由 `tests/paging_host.rs` 与
/// `src/paging.rs` 的用例覆盖, 本入口因此以本模块代替它, `src/ext_mod/` 的源码不因测试而
/// 改动, 同时仍能判定五件事: 每一页按给定的虚拟地址与物理地址调用一次映射, 二者逐页
/// 同步推进; 建立时权限取可读可写且层级取 4 KiB 页; 封口时逐页改写成可读可执行; 页表页的
/// 分配失败时以 `LoadError::MapFailed` 结束; 封口失败时同样以 `LoadError::MapFailed`
/// 结束。失败原因取 `src/paging.rs` 的同名变体 `OutOfPool`: 该路径上 `map_page` 只能因
/// 中间页表页无法分配而失败。
///
/// `Pte` 只为 `context` 声明页表根而存在, 本入口不安装任何页表项。
mod paging {
	use core::sync::atomic::{ AtomicBool, AtomicU64, Ordering };

	/// 页表项。`context` 以它声明 4 KiB 对齐的页表根。
	#[derive(Clone, Copy)]
	pub struct Pte(pub u64);

	/// 页表项安装的失败原因。与 `src/paging.rs` 的 `MapError` 同名, 本入口只用到其中
	/// 会在映像接收缓冲区这条路径上出现的一项。
	#[derive(Debug, Clone, Copy, PartialEq, Eq)]
	pub enum MapError {
		/// 页表页分配失败 (S-mode 页池耗尽)。
		OutOfPool,
	}

	/// 首次 `map_page` 的虚拟地址与物理地址, `MAP_CALLS` 为 0 时无意义。
	pub static MAP_FIRST_VA: AtomicU64 = AtomicU64::new(0);
	pub static MAP_FIRST_PA: AtomicU64 = AtomicU64::new(0);
	/// 最近一次 `map_page` 的虚拟地址与物理地址。
	pub static MAP_LAST_VA: AtomicU64 = AtomicU64::new(0);
	pub static MAP_LAST_PA: AtomicU64 = AtomicU64::new(0);
	/// 最近一次 `map_page` 的权限与层级。
	pub static MAP_FLAGS: AtomicU64 = AtomicU64::new(0);
	pub static MAP_LEVEL: AtomicU64 = AtomicU64::new(0);
	/// `map_page` 被调用的次数。
	pub static MAP_CALLS: AtomicU64 = AtomicU64::new(0);
	/// 置位时 `map_page` 报告页表页分配失败, 用于构造该失败。
	pub static PAGE_TABLE_ALLOC_FAIL: AtomicBool = AtomicBool::new(false);

	/// 记录一次映射请求, 按 `PAGE_TABLE_ALLOC_FAIL` 给出结果。
	pub fn map_page(vaddr: u64, paddr: u64, flags: u64, level: u8) -> Result<(), MapError> {
		if MAP_CALLS.load(Ordering::SeqCst) == 0 {
			MAP_FIRST_VA.store(vaddr, Ordering::SeqCst);
			MAP_FIRST_PA.store(paddr, Ordering::SeqCst);
		}
		MAP_LAST_VA.store(vaddr, Ordering::SeqCst);
		MAP_LAST_PA.store(paddr, Ordering::SeqCst);
		MAP_FLAGS.store(flags, Ordering::SeqCst);
		MAP_LEVEL.store(level as u64, Ordering::SeqCst);
		MAP_CALLS.fetch_add(1, Ordering::SeqCst);
		if PAGE_TABLE_ALLOC_FAIL.load(Ordering::SeqCst) {
			return Err(MapError::OutOfPool);
		}
		Ok(())
	}

	/// 封口时首次与最近一次改写的虚拟地址、权限, 以及调用次数。
	pub static SEAL_FIRST_VA: AtomicU64 = AtomicU64::new(0);
	pub static SEAL_LAST_VA: AtomicU64 = AtomicU64::new(0);
	pub static SEAL_FLAGS: AtomicU64 = AtomicU64::new(0);
	pub static SEAL_CALLS: AtomicU64 = AtomicU64::new(0);
	/// 置位时 `set_leaf_flags` 报告失败, 用于构造该失败。
	pub static SEAL_FAIL: AtomicBool = AtomicBool::new(false);

	/// 记录一次叶子项改写, 按 `SEAL_FAIL` 给出结果。
	pub fn set_leaf_flags(vaddr: u64, flags: u64) -> bool {
		if SEAL_CALLS.load(Ordering::SeqCst) == 0 {
			SEAL_FIRST_VA.store(vaddr, Ordering::SeqCst);
		}
		SEAL_LAST_VA.store(vaddr, Ordering::SeqCst);
		SEAL_FLAGS.store(flags, Ordering::SeqCst);
		SEAL_CALLS.fetch_add(1, Ordering::SeqCst);
		!SEAL_FAIL.load(Ordering::SeqCst)
	}
}

/// 地址对齐原语, 取 `src/` 内的实现。本模块在顶层声明, 由 [`mem_prim`] 按原名转出:
/// `#[path]` 若写在 [`mem_prim`] 之内, 其相对路径会以 `mem_prim` 追加为目录。
#[path = "../src/mem_prim/align.rs"]
mod align;

/// 原始内存原语。
///
/// 地址对齐取 `src/` 内的实现; 内存填充在本入口内只记录调用: 取入时模块槽位中超出映像的
/// 部分整块以 0 填充, 该部分在主机上落在测试程序自身的内存之内, 实际写入会改坏测试程序。
mod mem_prim {
	pub(crate) use super::align;

	pub mod string {
		use core::sync::atomic::{ AtomicU64, Ordering };

		/// 0 填充被请求的次数与最近一次的字节数。
		pub static FILL_CALLS: AtomicU64 = AtomicU64::new(0);
		pub static FILL_BYTES: AtomicU64 = AtomicU64::new(0);

		/// 记录一次 0 填充请求, 不写入。
		pub unsafe fn memset(_dst: *mut u8, _byte: u8, size: u64) {
			FILL_CALLS.fetch_add(1, Ordering::SeqCst);
			FILL_BYTES.store(size, Ordering::SeqCst);
		}
	}
}

#[path = "../src/sync_aux.rs"]
mod sync_aux;

#[path = "../src/context.rs"]
mod context;

#[path = "../src/attest.rs"]
mod attest;

#[path = "../src/ext_mod/mod.rs"]
mod ext_mod;

use constants::{
	ATTEST_ENABLE,
	CHUNK_2M_SIZE,
	ENCLAVE_MEM_ATTR_DATA,
	ENCLAVE_MEM_ATTR_INSTR,
	ENCLAVE_MODULE_LOAD_VA_INIT,
	ENCLAVE_MODULE_WINDOW_SIZE,
	LEVEL_PAGE,
	PAGE_SIZE,
	PTE_R,
	PTE_W,
	PTE_X,
	SIG_LEN,
};
use ext_mod::abi::{ GetterFn, ModuleDesc, ModuleInterface, SyscallTable, MAX_MODULES, MODULE_NAME_LEN };
use ext_mod::img_descriptor::{
	 DescError, IMAGE_DESC_ABI_VERSION, IMAGE_DESC_LEN, IMAGE_DESC_MAGIC,
};
use ext_mod::loader::LoadError;
use ext_mod::man::Manager;
use ext_mod::transport::Transport;

/// 本入口内的模块映像的字节数, 取 2 页, 含末尾一个签名。
///
/// 本入口内的映像以本进程内可执行的函数充当 (见 `dummy_img_va`), 故取 2 页即足以跨过该
/// 函数所在的一页。
const IMG_LEN: u64 = 2 * PAGE_SIZE;

/// 可执行部分的字节数: 映像本体 (不含末尾的签名) 按页向上取整。
///
/// `IMG_LEN` 取 2 页时该值同为 2 页, 故可写部分自映像末尾之后的第一个页边界开始。
const EXEC_LEN: u64 = (IMG_LEN - SIG_LEN as u64 + PAGE_SIZE - 1) / PAGE_SIZE * PAGE_SIZE;

/// 一个模块在模块窗口内占用的字节数, 与 `src/ext_mod/loader.rs` 的 `MODULE_SLOT_SIZE` 取值
/// 相同。
const SLOT_SIZE: u64 = CHUNK_2M_SIZE;

// ---------------------------------------------------------------
//  共享状态
// ---------------------------------------------------------------

/// 用例共享的串行锁。
///
/// `context::ctx().enclave_module_load_va`、`mem` 与 `paging` 内的调用记录、`DUMMY_IMG_MODULE_ID`
/// 与 `DUMMY_IMG_SYSCALLS` 都是进程内的全局量, 而测试默认并行执行, 故触及它们的用例先取得本锁。
///
/// 前一个用例 panic 后 `Mutex::lock` 返回 `Err`, 此处经 `into_inner` 取回其中的守卫继续
/// 执行, 使后续用例各自判定, 不被前一个用例的失败牵连。
static SERIAL: Mutex<()> = Mutex::new(());

fn serial() -> MutexGuard<'static, ()> {
	SERIAL.lock().unwrap_or_else(|err| err.into_inner())
}

/// 飞地上下文只初始化一次, `OnceCell` 的二次初始化会 panic。
static WORLD: Once = Once::new();

fn world() {
	WORLD.call_once(|| {
		context::init_context(0, ENCLAVE_MODULE_LOAD_VA_INIT);
	});
}

// ---------------------------------------------------------------
//  模块映像
// ---------------------------------------------------------------

/// `dummy_img_getter` 交付的模块编号, 由用例设置。
static DUMMY_IMG_MODULE_ID: AtomicU32 = AtomicU32::new(0);

/// `dummy_img_getter` 交付的系统调用表指针, 由用例设置。
///
/// `ext_mod::loader` 只判定该指针是否落在本模块的接收缓冲区之内, 从不解引用, 故它可以取
/// 缓冲区内的任意一个 8 字节对齐地址。
static DUMMY_IMG_SYSCALLS: AtomicU64 = AtomicU64::new(0);

/// `dummy_img_getter` 交付的操作表指针, 由用例设置; 取 0 表示本映像不导出操作表。
///
/// `ext_mod::loader` 只按指针本身判定落点, 从不解引用; `ext_mod::acquire_ops_table` 按
/// 调用方给出的表类型判定整张表, 并建立指向它的引用。
static DUMMY_IMG_OPS: AtomicU64 = AtomicU64::new(0);

/// 本入口内的模块映像所交付的取用器: 交付由用例设置的编号、处理函数表指针与操作表指针。
unsafe extern "C" fn dummy_img_getter(_manager: *const Manager) -> ModuleInterface {
	ModuleInterface {
		desc: ModuleDesc {
			module_id: DUMMY_IMG_MODULE_ID.load(Ordering::SeqCst),
			name: [0; MODULE_NAME_LEN],
			signature: 0,
		},
		syscalls: DUMMY_IMG_SYSCALLS.load(Ordering::SeqCst) as *const SyscallTable,
		ops: DUMMY_IMG_OPS.load(Ordering::SeqCst) as *const u8,
	}
}

/// 本入口内的模块入口: 交付取用器。
unsafe extern "C" fn dummy_img_init(_manager: *const Manager) -> GetterFn {
	dummy_img_getter
}

/// 映像缓冲区的字节数。
///
/// 取模块槽位的字节数加一页: 取入方判定的映像字节数上界是槽位的字节数, 而映像的可执行
/// 部分按该字节数取整到页, 故再加一页即覆盖可能交付的全部映像。
const IMAGE_CAP: u64 = SLOT_SIZE + PAGE_SIZE;

/// 映像缓冲区的起始地址; 取 0 表示尚未建立。
static IMAGE_VA: AtomicU64 = AtomicU64::new(0);

/// `mmap` 的保护位与映射标志, 取自 Linux 的 `mman.h`。
const PROT_READ: i32 = 1;
const PROT_WRITE: i32 = 2;
const PROT_EXEC: i32 = 4;
const MAP_PRIVATE: i32 = 0x02;
const MAP_ANONYMOUS: i32 = 0x20;

unsafe extern "C" {
	/// 建立或映射一段内存, 本入口只用它建立匿名映射。
	fn mmap(
		addr: *mut u8, len: usize, prot: i32, flags: i32, fd: i32, offset: i64,
	) -> *mut u8;
}

/// 建立映像缓冲区并返回其起始地址; 已建立时直接返回。
///
/// 飞地内的映像由管理程序在模块窗口内分配并映射, 主机侧没有等价设施: 映像要求可执行,
/// 而本进程内可写的段不可执行。故本入口经 `mmap` 建立一段可读可写可执行的匿名映射充当
/// 映像的加载地址, 其中的内容由 [`stage_image`] 在交付时写入。
fn image_va() -> u64 {
	let established = IMAGE_VA.load(Ordering::SeqCst);
	if established != 0 {
		return established;
	}
	let mapped = unsafe {
		mmap(
			ptr::null_mut(),
			IMAGE_CAP as usize,
			PROT_READ | PROT_WRITE | PROT_EXEC,
			MAP_PRIVATE | MAP_ANONYMOUS,
			-1,
			0,
		)
	};
	assert!(mapped as usize != !0usize, "映像缓冲区建立失败");
	IMAGE_VA.store(mapped as u64, Ordering::SeqCst);
	mapped as u64
}

/// 把映像描述块的第 *index* 个字段改写为 *value*, 字段按小端存放。
fn write_desc_field(index: usize, value: u32) {
	let at = image_va() + index as u64 * 4;
	unsafe {
		ptr::copy_nonoverlapping(value.to_le_bytes().as_ptr(), at as *mut u8, 4);
	}
}

/// 把一个 *size* 字节的映像写入映像缓冲区: 偏移 0 起是描述块, 入口紧随其后。
///
/// 描述块各字段与交付的字节数一致, 使取入方的核对通过; 入口处是转到 [`dummy_img_init`]
/// 的机器码。映像的其余字节不必写入: 取入方只读取描述块。
fn stage_image(size: u64) {
	assert!(size <= IMAGE_CAP, "映像长度为 {size} 字节, 超出缓冲区");
	let body_len = size - SIG_LEN as u64;
	write_desc_field(0, IMAGE_DESC_MAGIC);
	write_desc_field(1, IMAGE_DESC_ABI_VERSION);
	write_desc_field(2, IMAGE_DESC_LEN as u32);
	write_desc_field(3, body_len as u32);
	write_desc_field(4, align::align_up(body_len, PAGE_SIZE) as u32);
	// 入口: 把 `dummy_img_init` 的地址装入 rax 并跳转 (`movabs rax, imm64; jmp rax`)。
	let mut stub = [0u8; 12];
	stub[0] = 0x48;
	stub[1] = 0xB8;
	stub[10] = 0xFF;
	stub[11] = 0xE0;
	stub[2..10].copy_from_slice(&(dummy_img_init as *const () as u64).to_le_bytes());
	unsafe {
		ptr::copy_nonoverlapping(
			stub.as_ptr(),
			(image_va() + IMAGE_DESC_LEN as u64) as *mut u8,
			stub.len(),
		);
	}
}

// ---------------------------------------------------------------
//  管理器回调
// ---------------------------------------------------------------

// 六项回调在本入口内的实现如下, 函数名与 `Manager` 的字段同名。本映像不使用其中任何
// 一项, `ext_mod::loader` 只把集合透传给模块入口。

/// 无依赖模块可取用。
unsafe extern "C" fn alloc_ext_mod(_module_id: u32) -> Option<GetterFn> {
	None
}

/// 主机上没有 S 模式页池, 一律报告失败。
unsafe extern "C" fn fetch_smode_pages(_n: u64) -> u64 {
	!0u64
}

/// 主机上没有控制台。
unsafe extern "C" fn write_console(_bytes: *const u8, _len: u64) {}

/// 主机上不读时钟源。
unsafe extern "C" fn read_time() -> u64 {
	0
}

/// 主机上没有地址翻译, 取恒等映射。
unsafe extern "C" fn va_to_pa(va: u64) -> u64 {
	va
}

/// 管理器回调集合。
///
/// 两个平台量按「不提供」给出: 设备窗口地址与时间源频率取 0。
fn manager() -> Manager {
	Manager {
		alloc_ext_mod,
		fetch_smode_pages,
		write_console,
		read_time,
		va_to_pa,
		device_window_va: 0,
		time_freq: 0,
	}
}

// ---------------------------------------------------------------
//  通路
// ---------------------------------------------------------------

/// 通路在主机上的实现: 按用例给出的字节数、登记结果与交付结果回答, 并记录三项调用各自的
/// 次数与入参。
struct HostTransport {
	size: Option<u64>,
	written: Option<u64>,
	accepts_registration: bool,
	size_requests: Cell<u64>,
	registrations: Cell<u64>,
	registered_va: Cell<u64>,
	registered_len: Cell<u64>,
	fetches: Cell<u64>,
	/// 交付被请求时接收缓冲区是否已登记。
	registered_before_fetch: Cell<bool>,
	/// 交付时对描述块的第几个字段改写为何值; 取 None 表示不改写。用于构造描述块不被
	/// 接受的映像。
	desc_patch: Cell<Option<(usize, u32)>>,
}

impl HostTransport {
	fn new(size: Option<u64>, written: Option<u64>, accepts_registration: bool) -> Self {
		Self {
			size,
			written,
			accepts_registration,
			size_requests: Cell::new(0),
			registrations: Cell::new(0),
			registered_va: Cell::new(0),
			registered_len: Cell::new(0),
			fetches: Cell::new(0),
			registered_before_fetch: Cell::new(false),
			desc_patch: Cell::new(None),
		}
	}

	/// 报告 `size` 字节的映像, 接受登记, 并声明宿主写入了同样多的字节。
	fn writing(size: u64) -> Self {
		Self::new(Some(size), Some(size), true)
	}

	/// 报告 `size` 字节的映像并接受登记, 但拒绝交付。
	fn refusing_delivery(size: u64) -> Self {
		Self::new(Some(size), None, true)
	}

	/// 报告 `size` 字节的映像并拒绝登记接收缓冲区。
	fn rejecting_registration(size: u64) -> Self {
		Self::new(Some(size), Some(size), false)
	}

	fn size_requests(&self) -> u64 {
		self.size_requests.get()
	}

	fn registration_calls(&self) -> u64 {
		self.registrations.get()
	}

	fn fetch_calls(&self) -> u64 {
		self.fetches.get()
	}
}

impl Transport for HostTransport {
	fn request_size(&self, _module_id: u32) -> Option<u64> {
		self.size_requests.set(self.size_requests.get() + 1);
		self.size
	}

	fn register_img_recv_buf(&self, va: u64, len: u64) -> bool {
		self.registrations.set(self.registrations.get() + 1);
		self.registered_va.set(va);
		self.registered_len.set(len);
		self.accepts_registration
	}

	fn fetch(&self, _module_id: u32, size: u64) -> Option<u64> {
		self.fetches.set(self.fetches.get() + 1);
		self.registered_before_fetch.set(self.registrations.get() == 1);
		// 宿主把映像写入接收缓冲区, 在主机侧相当于把映像写入映像缓冲区。
		stage_image(size);
		if let Some((index, value)) = self.desc_patch.get() {
			write_desc_field(index, value);
		}
		self.written
	}
}

// ---------------------------------------------------------------
//  用例辅助
// ---------------------------------------------------------------

/// 把模块窗口内下一个空闲 VA 置到映像的加载地址, 并设置模块映像的编号、处理函数表指针
/// 与操作表指针, 同时清空 `mem`、`paging` 与 `mem_prim` 内各计数器的记录。
///
/// 处理函数表指针取缓冲区内的一个 8 字节对齐地址; 操作表指针取空, 由需要它的用例另行设置。
fn prepare(module_id: u32) -> (u64, Manager) {
	world();
	let va = image_va();
	context::ctx_mut().enclave_module_load_va = va;
	DUMMY_IMG_MODULE_ID.store(module_id, Ordering::SeqCst);
	DUMMY_IMG_SYSCALLS.store(va + 0x1000, Ordering::SeqCst);
	DUMMY_IMG_OPS.store(0, Ordering::SeqCst);
	mem::ALLOC_CALLS.store(0, Ordering::SeqCst);
	mem::ALLOC_FAIL.store(false, Ordering::SeqCst);
	mem::SET_ATTR_CALLS.store(0, Ordering::SeqCst);
	mem::SET_ATTR_FAIL.store(false, Ordering::SeqCst);
	paging::MAP_CALLS.store(0, Ordering::SeqCst);
	paging::PAGE_TABLE_ALLOC_FAIL.store(false, Ordering::SeqCst);
	paging::SEAL_CALLS.store(0, Ordering::SeqCst);
	paging::SEAL_FAIL.store(false, Ordering::SeqCst);
	mem_prim::string::FILL_CALLS.store(0, Ordering::SeqCst);
	mem_prim::string::FILL_BYTES.store(0, Ordering::SeqCst);
	(va, manager())
}

/// 断言取入以给定原因失败。
fn assert_load_fails(result: Result<GetterFn, LoadError>, expected: LoadError) {
	assert_eq!(result.err(), Some(expected));
}

/// 页池的分配请求次数。
fn alloc_calls() -> u64 {
	mem::ALLOC_CALLS.load(Ordering::SeqCst)
}

/// `map_page` 的调用次数。
fn map_calls() -> u64 {
	paging::MAP_CALLS.load(Ordering::SeqCst)
}

/// 模块窗口内下一个空闲 VA。
fn next_free_va() -> u64 {
	context::ctx().enclave_module_load_va
}

/// 一个模块槽位按 4 KiB 页计数的页数。
fn slot_pages() -> u64 {
	SLOT_SIZE / PAGE_SIZE
}

// ---------------------------------------------------------------
//  取入模块
// ---------------------------------------------------------------

/// 编号不小于登记表容量的模块在请求宿主之前即被拒。
#[test]
fn test_load_rejects_a_module_id_beyond_the_registry_capacity() {
	let _guard = serial();
	let (_, man) = prepare(0);
	let transport = HostTransport::writing(IMG_LEN);

	assert_load_fails(ext_mod::acquire_module(MAX_MODULES as u32, &transport, &man), LoadError::BadModuleId);
	assert_eq!(transport.size_requests(), 0);
	assert_eq!(transport.fetch_calls(), 0);
}

/// 宿主报告模块不存在时取入被拒, 且不分配物理页。
#[test]
fn test_load_reports_not_found_when_the_host_has_no_such_module() {
	let _guard = serial();
	let (_, man) = prepare(0x00);
	let transport = HostTransport::new(None, None, true);

	assert_load_fails(ext_mod::acquire_module(0x00, &transport, &man), LoadError::NotFound);
	assert_eq!(transport.size_requests(), 1);
	assert_eq!(alloc_calls(), 0);
	assert_eq!(transport.registration_calls(), 0);
	assert_eq!(transport.fetch_calls(), 0);
}

/// 宿主拒绝交付时取入被拒; 此刻接收缓冲区已建成并登记, 故窗口位置不归还。
#[test]
fn test_load_reports_not_found_when_the_host_refuses_delivery() {
	let _guard = serial();
	let (va, man) = prepare(0x01);
	let transport = HostTransport::refusing_delivery(IMG_LEN);

	assert_load_fails(ext_mod::acquire_module(0x01, &transport, &man), LoadError::NotFound);
	assert_eq!(alloc_calls(), 2);
	assert_eq!(map_calls(), slot_pages());
	assert_eq!(transport.registration_calls(), 1);
	assert_eq!(transport.fetch_calls(), 1);
	assert_eq!(next_free_va(), va + SLOT_SIZE);
}

/// 映像长度不足一个签名时取入被拒。
#[test]
fn test_load_rejects_an_image_shorter_than_one_signature() {
	let _guard = serial();
	let (_, man) = prepare(0x02);

	for size in [0u64, 1, SIG_LEN as u64] {
		let transport = HostTransport::writing(size);
		assert_load_fails(ext_mod::acquire_module(0x02, &transport, &man), LoadError::TooSmall);
		assert_eq!(transport.fetch_calls(), 0, "映像长度为 {size} 字节");
		assert_eq!(alloc_calls(), 0, "映像长度为 {size} 字节");
	}
}

/// 槽位越过模块窗口上界时取入被拒。
#[test]
fn test_load_rejects_an_image_that_overruns_the_module_window() {
	let _guard = serial();
	let (_, man) = prepare(0x03);
	context::ctx_mut().enclave_module_load_va = ENCLAVE_MODULE_LOAD_VA_INIT + ENCLAVE_MODULE_WINDOW_SIZE;
	let transport = HostTransport::writing(IMG_LEN);

	assert_load_fails(ext_mod::acquire_module(0x03, &transport, &man), LoadError::TooLarge);
	assert_eq!(alloc_calls(), 0);
	assert_eq!(map_calls(), 0);
	assert_eq!(transport.fetch_calls(), 0);
}

/// 窗口上界按槽位的字节数判定, 不按映像自身的字节数。
///
/// 该用例的 `enclave_module_load_va` 距上界不足一个槽位, 而上界判定按槽位, 故越界。
#[test]
fn test_load_rejects_a_slot_that_overruns_the_window() {
	let _guard = serial();
	let (_, man) = prepare(0x04);
	context::ctx_mut().enclave_module_load_va =
		ENCLAVE_MODULE_LOAD_VA_INIT + ENCLAVE_MODULE_WINDOW_SIZE - SLOT_SIZE + 1;
	let transport = HostTransport::writing(IMG_LEN);

	assert_load_fails(ext_mod::acquire_module(0x04, &transport, &man), LoadError::TooLarge);
	assert_eq!(alloc_calls(), 0);
	assert_eq!(map_calls(), 0);
	assert_eq!(transport.fetch_calls(), 0);
}

/// 下一个空闲 VA 恰好停在窗口上界时最后一个槽位仍可取入: 上界判定是越过窗口而非到达窗口。
#[test]
fn test_load_admits_the_last_slot_of_the_window() {
	let _guard = serial();
	let (_, man) = prepare(0x05);
	context::ctx_mut().enclave_module_load_va =
		ENCLAVE_MODULE_LOAD_VA_INIT + ENCLAVE_MODULE_WINDOW_SIZE - SLOT_SIZE;
	let transport = HostTransport::writing(IMG_LEN);
	paging::PAGE_TABLE_ALLOC_FAIL.store(true, Ordering::SeqCst);

	// 该槽位落在窗口的最后一个槽位上, 故映射失败; 映射被调用即证明上界判定放行了这一槽位。
	assert_load_fails(ext_mod::acquire_module(0x05, &transport, &man), LoadError::MapFailed);
	assert_eq!(alloc_calls(), 1);
	assert_eq!(mem::ALLOC_PAGES.load(Ordering::SeqCst), EXEC_LEN / PAGE_SIZE);
	assert_eq!(map_calls(), 1);
	assert_eq!(transport.registration_calls(), 0);
}

/// 宿主写入的字节数少于请求的字节数时取入被拒。
#[test]
fn test_load_reports_a_short_write_from_the_host() {
	let _guard = serial();
	let (_, man) = prepare(0x06);
	let transport = HostTransport::new(Some(IMG_LEN), Some(IMG_LEN - 1), true);

	assert_load_fails(ext_mod::acquire_module(0x06, &transport, &man), LoadError::ShortWrite);
	assert_eq!(transport.registration_calls(), 1);
	assert_eq!(transport.fetch_calls(), 1);
}

/// 页池耗尽时取入被拒: 接收缓冲区的物理页与页表页都取自该池, 池无页可用即无处安放。
///
/// 该次失败不建立页表页, 也不推进下一个空闲 VA。
#[test]
fn test_load_reports_an_exhausted_page_pool_and_keeps_the_next_free_va() {
	let _guard = serial();
	let (va, man) = prepare(0x07);
	mem::ALLOC_FAIL.store(true, Ordering::SeqCst);
	let transport = HostTransport::writing(IMG_LEN);

	assert_load_fails(ext_mod::acquire_module(0x07, &transport, &man), LoadError::MapFailed);
	assert_eq!(alloc_calls(), 1);
	assert_eq!(mem::ALLOC_PAGES.load(Ordering::SeqCst), EXEC_LEN / PAGE_SIZE);
	assert_eq!(map_calls(), 0);
	assert_eq!(transport.registration_calls(), 0);
	assert_eq!(transport.fetch_calls(), 0);
	assert_eq!(next_free_va(), va);
}

/// 页表页的分配失败时取入被拒, 且下一个空闲 VA 不变。
#[test]
fn test_load_reports_a_failed_page_table_allocation_and_keeps_the_next_free_va() {
	let _guard = serial();
	let (va, man) = prepare(0x08);
	paging::PAGE_TABLE_ALLOC_FAIL.store(true, Ordering::SeqCst);
	let transport = HostTransport::writing(IMG_LEN);

	assert_load_fails(ext_mod::acquire_module(0x08, &transport, &man), LoadError::MapFailed);
	assert_eq!(map_calls(), 1);
	assert_eq!(paging::MAP_FIRST_VA.load(Ordering::SeqCst), va);
	assert_eq!(paging::MAP_FIRST_PA.load(Ordering::SeqCst), mem::POOL_PA);
	assert_eq!(next_free_va(), va);
	assert_eq!(transport.registration_calls(), 0);
	assert_eq!(transport.fetch_calls(), 0);
}

/// M 模式拒绝登记接收缓冲区时取入被拒, 且不请求交付。
#[test]
fn test_load_reports_a_rejected_receive_buffer_registration() {
	let _guard = serial();
	let (va, man) = prepare(0x09);
	let transport = HostTransport::rejecting_registration(IMG_LEN);

	assert_load_fails(ext_mod::acquire_module(0x09, &transport, &man), LoadError::ImgRecvBufRejected);
	assert_eq!(transport.registration_calls(), 1);
	assert_eq!(transport.fetch_calls(), 0);
	// 缓冲区已建成, 故窗口位置不归还。
	assert_eq!(next_free_va(), va + SLOT_SIZE);
}

/// 登记的是映像接收缓冲区: 起点为映像的加载地址, 字节数为映像长度按页向上取整。
///
/// 登记必须早于请求交付: M 模式把交付的字节写入已登记的缓冲区, 未登记时这次交付无处可写。
#[test]
fn test_load_registers_the_image_receive_buffer_before_the_delivery() {
	if ATTEST_ENABLE {
		return;
	}
	let _guard = serial();
	let (va, man) = prepare(0x0a);
	let transport = HostTransport::writing(IMG_LEN);

	ext_mod::acquire_module(0x0a, &transport, &man).expect("取入应当成功");
	assert_eq!(transport.registered_va.get(), va);
	assert_eq!(transport.registered_len.get(), IMG_LEN);
	assert!(transport.registered_before_fetch.get(), "交付被请求时接收缓冲区尚未登记");
}

/// 同一模块处于取入过程中时再次请求立即失败, 不请求宿主。
///
/// 该结果即模块入口内取用同一个模块时得到的结果: `ext_mod::loader` 在调用模块入口之前
/// 已占据槽位, 且调用入口时不持有取用 `enclave_module_load_va` 的 `LOAD_LOCK`。
#[test]
fn test_load_reports_loading_while_the_same_module_is_in_flight() {
	let _guard = serial();
	let (_, man) = prepare(0x0b);
	assert!(ext_mod::table::begin_load(0x0b));
	let transport = HostTransport::writing(IMG_LEN);

	assert_load_fails(ext_mod::acquire_module(0x0b, &transport, &man), LoadError::Loading);
	assert_eq!(transport.size_requests(), 0);
	assert_eq!(transport.fetch_calls(), 0);
	assert_eq!(map_calls(), 0);
	ext_mod::table::abort_load(0x0b);
}

/// 已登记的模块直接返回既有取用器, 不重复取入。
#[test]
fn test_load_returns_the_registered_getter_without_fetching_again() {
	if ATTEST_ENABLE {
		return;
	}
	let _guard = serial();
	let (_, man) = prepare(0x0c);
	let first = HostTransport::writing(IMG_LEN);
	let getter = ext_mod::acquire_module(0x0c, &first, &man).expect("首次取入应当成功");
	assert_eq!(first.fetch_calls(), 1);

	// 第二次的通路不提供任何模块: 若重新取入一遍, 结果会是 `NotFound`。
	let second = HostTransport::new(None, None, true);
	let again = ext_mod::acquire_module(0x0c, &second, &man).expect("已登记的模块直接返回");
	assert_eq!(second.fetch_calls(), 0);
	assert_eq!(again as usize, getter as usize);
	assert_eq!(
		ext_mod::table::lookup(0x0c).map(|entry| entry as usize),
		Some(getter as usize)
	);
}

/// 接口中的模块编号与请求的编号不一致时取入被拒, 且不登记。
///
/// 接收缓冲区在此之前已经建成, 故模块窗口的位置不回收。
#[test]
fn test_load_rejects_an_interface_whose_module_id_differs() {
	if ATTEST_ENABLE {
		return;
	}
	let _guard = serial();
	let (va, man) = prepare(0x0d);
	DUMMY_IMG_MODULE_ID.store(0x1f, Ordering::SeqCst);
	let transport = HostTransport::writing(IMG_LEN);

	assert_load_fails(ext_mod::acquire_module(0x0d, &transport, &man), LoadError::IdMismatch);
	assert!(ext_mod::table::lookup(0x0d).is_none());
	assert_eq!(next_free_va(), va + SLOT_SIZE);
}

/// 系统调用表指针不落在槽位之内时取入被拒。
///
/// 两种越界各用一个模块编号: 该判定发生在缓冲区建成之后, 故一次取入失败即消耗掉一个窗口
/// 位置, 第二次取入会以被消耗后的位置当作映像的加载地址。
#[test]
fn test_load_rejects_a_syscall_table_outside_the_image() {
	if ATTEST_ENABLE {
		return;
	}
	let _guard = serial();

	// 表的末尾越过槽位末尾。
	let (va, man) = prepare(0x0e);
	DUMMY_IMG_SYSCALLS.store(va + SLOT_SIZE, Ordering::SeqCst);
	let transport = HostTransport::writing(IMG_LEN);
	assert_load_fails(ext_mod::acquire_module(0x0e, &transport, &man), LoadError::BadSyscallTable);
	assert!(ext_mod::table::lookup(0x0e).is_none());
	assert_eq!(next_free_va(), va + SLOT_SIZE);

	// 起点满足 8 字节对齐, 而整张表越过槽位末尾。
	let (va, man) = prepare(0x0f);
	DUMMY_IMG_SYSCALLS.store(va + SLOT_SIZE - 8, Ordering::SeqCst);
	let transport = HostTransport::writing(IMG_LEN);
	assert_load_fails(ext_mod::acquire_module(0x0f, &transport, &man), LoadError::BadSyscallTable);
	assert!(ext_mod::table::lookup(0x0f).is_none());
}

/// 取入成功时逐页建立槽位, 登记取用器并推进下一个空闲 VA。
#[test]
fn test_load_succeeds_and_advances_the_next_free_va() {
	if ATTEST_ENABLE {
		return;
	}
	let _guard = serial();
	let (va, man) = prepare(0x15);
	let transport = HostTransport::writing(IMG_LEN);

	let getter = ext_mod::acquire_module(0x15, &transport, &man).expect("取入应当成功");
	assert_eq!(transport.fetch_calls(), 1);

	// 物理页的入参: 可执行部分与可写部分各申请一个分区, 两次的类别都取数据类; 建立映射时
	// 权限取可读可写 (封口前的内容写入需要), 层级取 4 KiB 页。
	assert_eq!(alloc_calls(), 2);
	assert_eq!(mem::ALLOC_ATTR.load(Ordering::SeqCst), ENCLAVE_MEM_ATTR_DATA);
	assert_eq!(map_calls(), slot_pages());
	assert_eq!(paging::MAP_FIRST_VA.load(Ordering::SeqCst), va);
	assert_eq!(paging::MAP_FIRST_PA.load(Ordering::SeqCst), mem::POOL_PA);
	assert_eq!(paging::MAP_FLAGS.load(Ordering::SeqCst), (PTE_R | PTE_W) as u64);
	assert_eq!(paging::MAP_LEVEL.load(Ordering::SeqCst), LEVEL_PAGE as u64);
	assert_eq!(next_free_va(), va + SLOT_SIZE);

	// 登记的是取用器, 再次调用它得到同一接口。
	assert_eq!(
		ext_mod::table::lookup(0x15).map(|entry| entry as usize),
		Some(getter as usize)
	);
	let interface = unsafe { getter(&man) };
	assert_eq!(interface.desc.module_id, 0x15);
}

/// 封口只涉及可执行部分: 该部分的分区改判为指令类, 其页表项改为可读可执行; 可写部分
/// 保持数据类与可读可写。
///
/// 覆盖两类静态数据的映像: 初值非零的静态数据随映像位于可执行部分之内, 初值为零的静态
/// 数据位于可写部分之内。封口若覆盖整个槽位, 模块对后者的写入会被页表项拒绝。
#[test]
fn test_seal_reclassifies_only_the_executable_part() {
	if ATTEST_ENABLE {
		return;
	}
	let _guard = serial();
	let (va, man) = prepare(0x1e);
	let transport = HostTransport::writing(IMG_LEN);

	ext_mod::acquire_module(0x1e, &transport, &man).expect("取入应当成功");

	// 类别改判: 区间自可执行部分的物理基址起, 长度为可执行部分的字节数。
	assert_eq!(mem::SET_ATTR_CALLS.load(Ordering::SeqCst), 1);
	assert_eq!(mem::SET_ATTR_PA.load(Ordering::SeqCst), mem::POOL_PA);
	assert_eq!(mem::SET_ATTR_BYTES.load(Ordering::SeqCst), EXEC_LEN);
	assert_eq!(mem::SET_ATTR_ATTR.load(Ordering::SeqCst), ENCLAVE_MEM_ATTR_INSTR);

	// 页表项改写: 只改写可执行部分, 自映像的加载地址起共 EXEC_LEN 页。
	assert_eq!(paging::SEAL_CALLS.load(Ordering::SeqCst), EXEC_LEN / PAGE_SIZE);
	assert_eq!(paging::SEAL_FIRST_VA.load(Ordering::SeqCst), va);
	assert_eq!(paging::SEAL_LAST_VA.load(Ordering::SeqCst), va + EXEC_LEN - PAGE_SIZE);
	assert_eq!(paging::SEAL_FLAGS.load(Ordering::SeqCst), (PTE_R | PTE_X) as u64);

	// 槽位中超出映像的部分以 0 填充, 填入模块初值为零的静态数据。
	assert_eq!(mem_prim::string::FILL_CALLS.load(Ordering::SeqCst), 1);
	assert_eq!(mem_prim::string::FILL_BYTES.load(Ordering::SeqCst), SLOT_SIZE - IMG_LEN);
}

/// 映像长度越过可执行部分时, 该部分之后的字节同样以 0 填充。
///
/// 映像本体按页取整可大于映像本体自身的字节数, 此时映像的末页跨过可执行部分的末尾,
/// 该页内自可执行部分末尾起的内容不在交付的写入范围之内, 须另行置零, 其中含映像末尾的
/// 签名字节。填充在验签之后进行, 签名在填充之前已核对。
#[test]
fn test_load_fills_the_page_that_the_image_end_crosses() {
	if ATTEST_ENABLE {
		return;
	}
	let _guard = serial();
	// 映像长度取一页加 4 字节: 可执行部分为一页, 余下 4 字节落在槽位的可写部分之内。
	let (_, man) = prepare(0x1f);
	let size = PAGE_SIZE + 4;
	let transport = HostTransport::writing(size);

	ext_mod::acquire_module(0x1f, &transport, &man).expect("取入应当成功");
	assert_eq!(mem::SET_ATTR_BYTES.load(Ordering::SeqCst), PAGE_SIZE);
	assert_eq!(mem_prim::string::FILL_CALLS.load(Ordering::SeqCst), 2);
	assert_eq!(mem_prim::string::FILL_BYTES.load(Ordering::SeqCst), 4);
}

/// 下一个空闲 VA 按槽位的字节数推进, 不按映像自身的字节数。
#[test]
fn test_load_advances_the_next_free_va_by_the_slot_size() {
	if ATTEST_ENABLE {
		return;
	}
	let _guard = serial();
	let (va, man) = prepare(0x16);
	let size = 3 * PAGE_SIZE;
	let transport = HostTransport::writing(size);

	ext_mod::acquire_module(0x16, &transport, &man).expect("取入应当成功");
	assert_eq!(map_calls(), slot_pages());
	assert_eq!(transport.registered_len.get(), size);
	assert_eq!(next_free_va(), va + SLOT_SIZE);
}

/// 类别改判失败时取入被拒: 缓冲区分区仍为数据类, 自其中取指会被拒绝, 模块入口不可用。
#[test]
fn test_load_reports_a_failed_partition_reclass() {
	if ATTEST_ENABLE {
		return;
	}
	let _guard = serial();
	let (_, man) = prepare(0x1c);
	mem::SET_ATTR_FAIL.store(true, Ordering::SeqCst);
	let transport = HostTransport::writing(IMG_LEN);

	assert_load_fails(ext_mod::acquire_module(0x1c, &transport, &man), LoadError::MapFailed);
	// 改判在映射建立之后, 故映射已按页建成。
	assert_eq!(mem::SET_ATTR_CALLS.load(Ordering::SeqCst), 1);
	assert_eq!(map_calls(), slot_pages());
	assert_eq!(paging::SEAL_CALLS.load(Ordering::SeqCst), 0);
}

/// 页表项改写失败时取入被拒, 且不再改写后续页。
#[test]
fn test_load_reports_a_failed_leaf_rewrite() {
	if ATTEST_ENABLE {
		return;
	}
	let _guard = serial();
	let (_, man) = prepare(0x1d);
	paging::SEAL_FAIL.store(true, Ordering::SeqCst);
	let transport = HostTransport::writing(IMG_LEN);

	assert_load_fails(ext_mod::acquire_module(0x1d, &transport, &man), LoadError::MapFailed);
	assert_eq!(paging::SEAL_CALLS.load(Ordering::SeqCst), 1);
}

// ---------------------------------------------------------------
//  取入操作表
// ---------------------------------------------------------------

// 本节的模块编号取 0x17 至 0x1B, 上一节取 0x00 至 0x0F 与 0x15 至 0x16 与 0x1E, 封口失败
// 的两例取 0x1C 与 0x1D。0x10 至 0x14 留给 `ext_mod::table` 自身的用例, 其中一例把 0x11
// 留在取入状态。各用例的编号互异, 同一编号在本入口内只取入一次。

/// 本入口内作为操作表的示例类型。
///
/// `acquire_ops_table` 只按 `size_of` 与 `align_of` 判定表的落点, 并建立指向它的引用,
/// 故该类型只需可被安全地读取: 三个 64 位字段对任意字节序列都成立。
#[repr(C)]
struct OpsFixture {
	first: u64,
	second: u64,
	third: u64,
}

/// 操作表的起点取槽位内的最小对齐地址。
///
/// 映像的加载 VA 是入口函数的地址, 该地址的对齐由链接器给出, 故此处向本类型的对齐要求
/// 上取整。上取整后仍在槽位之内: 该地址取自槽位起点, 而上取整至多移动 7 字节。
fn aligned_ops_va(va: u64) -> u64 {
	let align = core::mem::align_of::<OpsFixture>() as u64;
	(va + align - 1) & !(align - 1)
}

/// 操作表落在映像的接收缓冲区之内且对齐时, 取入成功并返回指向该表的引用。
#[test]
fn test_acquire_ops_table_returns_a_table_inside_the_image() {
	if ATTEST_ENABLE {
		return;
	}
	let _guard = serial();
	let (va, man) = prepare(0x1b);
	let ops_va = aligned_ops_va(va);
	DUMMY_IMG_OPS.store(ops_va, Ordering::SeqCst);
	let transport = HostTransport::writing(IMG_LEN);

	let ops = ext_mod
		::acquire_ops_table::<OpsFixture>(0x1b, &transport, &man)
		.expect("操作表落在缓冲区内, 取入应当成功");
	assert_eq!(core::ptr::from_ref(ops) as u64, ops_va);
	// 槽位的范围随取用器一同登记, 取用方据此判定整张表的落点。
	assert_eq!(ext_mod::table::image_window(0x1b), Some((va, SLOT_SIZE)));
}

/// 映像不导出操作表时取入失败。
#[test]
fn test_acquire_ops_table_rejects_a_missing_table() {
	if ATTEST_ENABLE {
		return;
	}
	let _guard = serial();
	let (_, man) = prepare(0x17);
	let transport = HostTransport::writing(IMG_LEN);

	assert!(
		matches!(
			ext_mod::acquire_ops_table::<OpsFixture>(0x17, &transport, &man),
			Err(LoadError::BadOpsTable)
		)
	);
	// 模块本身取入成功, 只是操作表不可用。
	assert!(ext_mod::table::lookup(0x17).is_some());
}

/// 操作表的起点落在映像的接收缓冲区之前时取入失败。
#[test]
fn test_acquire_ops_table_rejects_a_table_before_the_image() {
	if ATTEST_ENABLE {
		return;
	}
	let _guard = serial();
	let (va, man) = prepare(0x18);
	DUMMY_IMG_OPS.store(va - 0x1000, Ordering::SeqCst);
	let transport = HostTransport::writing(IMG_LEN);

	assert!(
		matches!(
			ext_mod::acquire_ops_table::<OpsFixture>(0x18, &transport, &man),
			Err(LoadError::BadOpsTable)
		)
	);
}

/// 操作表的起点落在槽位之内而末端越过槽位末尾时取入失败。
///
/// 加载器按指针本身判定落点时该地址通过, 判定来自取用方对整张表的检查。
#[test]
fn test_acquire_ops_table_rejects_a_table_overrunning_the_image() {
	if ATTEST_ENABLE {
		return;
	}
	let _guard = serial();
	let (va, man) = prepare(0x19);
	let size = core::mem::size_of::<OpsFixture>() as u64;
	DUMMY_IMG_OPS.store(va + SLOT_SIZE - size + 1, Ordering::SeqCst);
	let transport = HostTransport::writing(IMG_LEN);

	assert!(
		matches!(
			ext_mod::acquire_ops_table::<OpsFixture>(0x19, &transport, &man),
			Err(LoadError::BadOpsTable)
		)
	);
}

/// 操作表的起点不满足 8 字节对齐时取入失败。
#[test]
fn test_acquire_ops_table_rejects_a_misaligned_table() {
	if ATTEST_ENABLE {
		return;
	}
	let _guard = serial();
	let (va, man) = prepare(0x1a);
	// 偏移 1 字节必不满足 8 字节对齐, 且仍在缓冲区之内。
	DUMMY_IMG_OPS.store(va + 1, Ordering::SeqCst);
	let transport = HostTransport::writing(IMG_LEN);

	assert!(
		matches!(
			ext_mod::acquire_ops_table::<OpsFixture>(0x1a, &transport, &man),
			Err(LoadError::BadOpsTable)
		)
	);
}

// ---------------------------------------------------------------
//  映像描述块与失败的上报
// ---------------------------------------------------------------

/// 描述块不被接受时取入在改判之前被拒: 映像不取得可执行权限, 它的入口一次也不会被调用。
///
/// 两种不符项各取入一次: 魔数不符是交付的字节不是本机制所定义的模块映像; 声明的映像本体
/// 字节数与交付的字节数不符是映像陈旧或被截断。
///
/// 两次之间把 `enclave_module_load_va` 拨回映像的加载地址: 一次取入无论成败都把该值推进
/// 一个槽位 (见 `test_load_reports_not_found_when_the_host_refuses_delivery`), 而主机侧只有
/// 一段映像缓冲区, 交付的仍是这一段的字节。
#[test]
fn test_load_rejects_an_image_whose_descriptor_is_not_accepted() {
	let _guard = serial();
	let (va, man) = prepare(0x10);
	let cases = [
		// 魔数不符
		(0usize, 0u32, DescError::BadMagic),
		// 声明的映像本体字节数少于交付的字节数
		(3, (IMG_LEN - SIG_LEN as u64 - 4) as u32, DescError::BodyLenMismatch),
	];
	for (index, value, expected) in cases {
		context::ctx_mut().enclave_module_load_va = va;
		let transport = HostTransport::writing(IMG_LEN);
		transport.desc_patch.set(Some((index, value)));

		assert_load_fails(
			ext_mod::acquire_module(0x10, &transport, &man),
			LoadError::BadImageDesc(expected)
		);
		// 核对未通过即不改判为指令类, 也不调用映像的入口。
		assert_eq!(mem::SET_ATTR_CALLS.load(Ordering::SeqCst), 0);
		assert_eq!(paging::SEAL_CALLS.load(Ordering::SeqCst), 0);
	}
}

/// 各失败原因折算为上报给可信应用的错误码: 映像与接口的布局不符归 EINVAL, 资源不足归
/// ENOMEM, 宿主未提供该模块归 ENOENT, 该模块正在取入归 EAGAIN。
#[test]
fn test_load_error_folds_into_an_errno() {
	use ext_mod::errno::{ EAGAIN, EINVAL, ENOMEM, ENOENT };
	use ext_mod::loader::LoadError::*;

	let cases = [
		(NotFound, ENOENT),
		(Loading, EAGAIN),
		(BadModuleId, EINVAL),
		(BadImageDesc(DescError::BadMagic), EINVAL),
		(AttestFailed, EINVAL),
		(IdMismatch, EINVAL),
		(BadSyscallTable, EINVAL),
		(BadOpsTable, EINVAL),
		(TooSmall, ENOMEM),
		(TooLarge, ENOMEM),
		(ShortWrite, ENOMEM),
		(MapFailed, ENOMEM),
		(ImgRecvBufRejected, ENOMEM),
	];
	for (err, expected) in cases {
		assert_eq!(err.to_errno(), expected, "{err:?}");
	}
}

/// 只有「宿主未提供该模块」一种失败使运行时的粘性标志置位: 其余失败源于映像或接口自身,
/// 映像重新交付之后的下一次调用可以重试取入。
#[test]
fn test_only_a_missing_module_counts_as_absent() {
	use ext_mod::loader::LoadError::*;

	assert!(NotFound.is_module_absent());
	let others = [
		Loading,
		BadModuleId,
		BadImageDesc(DescError::BadMagic),
		AttestFailed,
		IdMismatch,
		BadSyscallTable,
		BadOpsTable,
		TooSmall,
		TooLarge,
		ShortWrite,
		MapFailed,
		ImgRecvBufRejected,
	];
	for err in others {
		assert!(!err.is_module_absent(), "{err:?}");
	}
}
