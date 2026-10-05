//! 运行时一侧的模块接入。
//!
//! 取入一个模块需要两样东西: 一条向宿主请求映像的通路, 与一份交给模块的管理器回调
//! 集合。两者都取用平台设施 (ecall、S 模式页池、地址翻译、时钟源), 故本模块只在固件
//! 构建内编译; 主机目标上的用例自带通路与管理器的实现, 见 `tests/ext_mod_host.rs`。
//!
//! 每个模块一个取用入口: 已登记则直接返回登记的操作表, 未登记则向固件提出交付申请。
//! 模块在首次用到时取入, 不在启动时取入: 平台的设备区间由 `NET_BASE` 决定是否存在,
//! 而文件系统模块只被载荷的路径类系统调用用到, 二者都不必占用启动路径。
//!
//! 一次取入在宿主应答并经 RESUME 交还之前不返回, 期间本飞地不再执行, 故同一飞地不会有
//! 第二个模块请求与它重叠 —— M 模式为每个飞地只记录一个待决请求。

use core::ptr;
use core::sync::atomic::{AtomicBool, AtomicPtr, Ordering};

use super::abi::GetterFn;
use super::man::Manager;
use super::net_ops::NetOps;
use super::transport::EcallTransport;
use super::vfs_ops::VfsOps;
use super::{acquire_module, acquire_ops_table};
use crate::constants::{LINEAR_MAP_OFFSET, NET_BASE, TIMER_FREQ};
use crate::csr;
use crate::ecall_aux;
use crate::mem;

/// 网卡与协议栈模块的编号, 与 `modules/net` 的 `MODULE_ID` 及 `modules.list` 的条目一致。
const MODULE_ID_NET: u32 = 4;
/// 文件系统模块的编号, 与 `modules/fs` 的 `MODULE_ID` 及 `modules.list` 的条目一致。
const MODULE_ID_VFS: u32 = 3;

/// 设备窗口的虚拟地址: 设备区间由 `paging::map_device_region` 映射到物理地址加
/// [`LINEAR_MAP_OFFSET`] 处。
///
/// 平台不提供该设备 (`NET_BASE` 取 0) 时取 0, 模块据此判定设备不存在, 不再访问设备
/// 寄存器。
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
	acquire_module(module_id, &TRANSPORT, &MANAGER).ok()
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

/// 宿主未提供网卡模块。
static NET_ABSENT: AtomicBool = AtomicBool::new(false);
/// 宿主未提供文件系统模块。
static VFS_ABSENT: AtomicBool = AtomicBool::new(false);

/// 取入 *module_id* 并返回它导出的操作表, 结果按 *cell* 缓存。
///
/// 已登记时直接返回登记的操作表; 未登记时经 [`TRANSPORT`] 向固件提出交付申请, 由宿主
/// 提供映像。宿主未提供该模块的结论一旦得出即记入 *absent*, 后续调用不再提出申请 ——
/// 载荷若反复调用同一个系统调用, 每次都让出到宿主会使其陷入空转。
///
/// *init* 在操作表取到之后、写入缓存之前调用, 返回假即视为取入失败。
fn acquire<T>(
	cell: &AtomicPtr<T>,
	absent: &AtomicBool,
	module_id: u32,
	init: fn(&T) -> bool,
) -> Option<&'static T> {
	let cached = cell.load(Ordering::Acquire);
	if !cached.is_null() {
		// 指针只在取入成功之后写入, 且写入之前模块已经取入并验签, 指向的表此后只读。
		return Some(unsafe { &*cached });
	}
	if absent.load(Ordering::Acquire) {
		return None;
	}
	let ops = match acquire_ops_table::<T>(module_id, &TRANSPORT, &MANAGER) {
		Ok(ops) => ops,
		Err(_) => {
			absent.store(true, Ordering::Release);
			return None;
		}
	};
	if !init(ops) {
		absent.store(true, Ordering::Release);
		return None;
	}
	cell.store(ptr::from_ref(ops).cast_mut(), Ordering::Release);
	Some(ops)
}

/// 网卡模块的操作表; 该模块不可用时返回 None。
///
/// 平台不提供该设备 (`NET_BASE` 取 0) 时直接返回 None, 不提出申请。
pub fn acquire_net_ops() -> Option<&'static NetOps> {
	if NET_BASE == 0 {
		return None;
	}
	acquire(&NET_OPS, &NET_ABSENT, MODULE_ID_NET, |ops| unsafe { (ops.init)() })
}

/// 文件系统模块的操作表; 该模块不可用时返回 None。
pub fn acquire_vfs_ops() -> Option<&'static VfsOps> {
	acquire(&VFS_OPS, &VFS_ABSENT, MODULE_ID_VFS, |ops| unsafe { (ops.init)() })
}
