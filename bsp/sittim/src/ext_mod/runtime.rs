//! 运行时一侧的模块接入。
//!
//! 取入一个模块需要两样东西: 一条向宿主请求映像的通路, 与一份交给模块的管理器回调
//! 集合。两者都取用平台设施 (ecall、S 模式页池、地址翻译、时钟源), 故本模块只在固件
//! 构建内编译; 主机目标上的用例自带通路与管理器的实现
//! 每个模块一个取用入口: 已登记则直接返回登记的操作表, 未登记则向固件提出交付申请。
//! 模块在首次用到时取入, 不在启动时取入: 平台的设备区间由 `NET_BASE` 决定是否存在,
//! 而文件系统模块只被载荷的路径类系统调用用到, 二者都不必占用启动路径。
//!
//! 一次取入在宿主应答并经 RESUME 交还之前不返回, 期间本飞地不再执行。同一飞地的模块请求
//! 在 M 模式内按到达顺序排在一条有界队列上, 本次取入取回的是队首的结果。
//!
//! 取用失败按 [`LoadError::to_errno`](super::loader::LoadError::to_errno) 交回一个错误码, 由
//! 调用方原样作为系统调用的返回值上报给可信应用: 宿主未提供该模块与映像有缺陷两种情形由
//! 此在可信应用侧可以区分。

use core::ptr;
use core::sync::atomic::{AtomicPtr, Ordering};

use super::abi::GetterFn;
use super::acquire_ops_table;
use super::errno::ENODEV;
use super::loader::load;
use super::man::Manager;
use super::net_ops::NetOps;
use super::transport::EcallTransport;
use super::vfs_ops::VfsOps;
use crate::constants::{LINEAR_MAP_OFFSET, NET_BASE, TIMER_FREQ};
use crate::csr;
use crate::ecall_aux;
use crate::mem;

/// 网卡与协议栈模块的编号
const MODULE_ID_NET: u32 = 4;
/// 文件系统模块的编号
const MODULE_ID_VFS: u32 = 3;

/// 设备窗口的虚拟地址: 设备区间由 `paging::map_device_region` 映射到物理地址加
/// [`LINEAR_MAP_OFFSET`] 处。
///
/// 平台不提供该设备时取 0, 模块据此判定设备不存在
const DEVICE_WINDOW_VA: u64 = if NET_BASE == 0 {
	0
} else {
	NET_BASE + LINEAR_MAP_OFFSET
};

/// 向宿主请求模块映像的通路。
static TRANSPORT: EcallTransport = EcallTransport;

/// 交给模块的管理器回调集合。
static MANAGER: Manager = Manager {
	alloc_ext_mod,
	fetch_smode_pages,
	write_console,
	read_time,
	va_to_pa,
	device_window_va: DEVICE_WINDOW_VA,
	time_freq: TIMER_FREQ,
};

/// 取用另一个模块。
unsafe extern "C" fn alloc_ext_mod(module_id: u32) -> Option<GetterFn> {
	load(module_id, &TRANSPORT, &MANAGER).ok()
}

/// 申请 n 个 S 模式页, 返回物理地址。
unsafe extern "C" fn fetch_smode_pages(n: u64) -> u64 {
	mem::try_alloc_smode_page(n)
}

/// 写控制台, 入参为字节指针与字节数。
unsafe extern "C" fn write_console(buf: *const u8, len: u64) {
	let bytes = unsafe { core::slice::from_raw_parts(buf, len as usize) };
	ecall_aux::sbi_console_write(bytes);
}

/// 读时间源。
unsafe extern "C" fn read_time() -> u64 {
	csr::read_time()
}

/// 把飞地地址空间中的虚拟地址翻译为物理地址; 翻译失败时返回 `!0`。
unsafe extern "C" fn va_to_pa(va: u64) -> u64 {
	ecall_aux::va_to_pa(va).unwrap_or(!0)
}

/// 已接入的网卡模块的操作表; 未接入时为空指针。
static NET_OPS: AtomicPtr<NetOps> = AtomicPtr::new(ptr::null_mut());
/// 已接入的文件系统模块的操作表; 未接入时为空指针。
static VFS_OPS: AtomicPtr<VfsOps> = AtomicPtr::new(ptr::null_mut());

/// 取入 *module_id* 并返回它导出的操作表, 结果按 *cell* 缓存。
///
/// 已登记时直接返回登记的操作表; 未登记时向固件提出交付申请, 由宿主提供映像。本次申请未获
/// 交付只影响本次调用: 运行时不记录该模块的状态, 下一次调用重新提出申请, 故宿主补交映像之后
/// 不需其它动作即可取入。
///
/// *init* 在操作表取到之后、写入缓存之前调用, 返回假即视为该模块不可用。
///
/// 失败返回错误码, 模块自身初始化失败返回 `ENODEV`。
fn acquire<T>(
	cell: &AtomicPtr<T>,
	module_id: u32,
	init: fn(&T) -> bool,
) -> Result<&'static T, u64> {
	let cached = cell.load(Ordering::Acquire);
	if !cached.is_null() {
		// 指针只在取入成功之后写入, 且写入之前模块已经取入并验签, 指向的表此后只读。
		return Ok(unsafe { &*cached });
	}
	let ops =
		acquire_ops_table::<T>(module_id, &TRANSPORT, &MANAGER).map_err(|err| err.to_errno())?;
	let is_module_ready = init(ops);
	if !is_module_ready {
		return Err(ENODEV);
	}

	cell.store(ptr::from_ref(ops).cast_mut(), Ordering::Release);
	Ok(ops)
}

/// 网卡模块的操作表。
///
/// 平台不提供该设备 (`NET_BASE` 取 0) 时返回 [`ENODEV`], 不向宿主提出申请; 其余失败按
/// [`LoadError::to_errno`](super::loader::LoadError::to_errno) 给出错误码。
pub fn acquire_net_ops() -> Result<&'static NetOps, u64> {
	if NET_BASE == 0 {
		return Err(ENODEV);
	}
	acquire(&NET_OPS, MODULE_ID_NET, |ops| unsafe { (ops.init)() })
}

/// 已取入的网卡模块的操作表; 该模块尚未取入时返回 None。
///
/// 本函数不发起取入, 中断上下文的调用方无从向可信应用
/// 上报取入失败, 也不得在其中让出宿主, 故只能取已经登记的操作表。
pub fn loaded_net_ops() -> Option<&'static NetOps> {
	let ops = NET_OPS.load(Ordering::Acquire);
	if ops.is_null() {
		None
	} else {
		// 指针只在取入成功之后写入, 且写入之前模块已经取入并验签, 指向的表此后只读。
		Some(unsafe { &*ops })
	}
}

/// 文件系统模块的操作表; 失败按 [`LoadError::to_errno`](super::loader::LoadError::to_errno) 给出错误码。
pub fn acquire_vfs_ops() -> Result<&'static VfsOps, u64> {
	acquire(&VFS_OPS, MODULE_ID_VFS, |ops| unsafe { (ops.init)() })
}

/// 清除该模块的操作表缓存。
///
/// 模块在运行中出错之后调用: 缓存内的操作表取自出错的那份映像, 本函数之后不再交出它, 该模块
/// 的下一次调用重新提出交付申请。登记表一侧的动作由调用方完成。
pub fn clear_module_cache(module_id: u32) {
	match module_id {
		MODULE_ID_NET => {
			NET_OPS.store(ptr::null_mut(), Ordering::Release);
		}
		MODULE_ID_VFS => {
			VFS_OPS.store(ptr::null_mut(), Ordering::Release);
		}
		_ => {}
	}
}
