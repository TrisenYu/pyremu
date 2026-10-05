//! 外部扩展模块机制。
//!
//! 功能模块 (网卡驱动与协议栈、文件系统) 由宿主按需交付: 飞地在该模块首次被用到时取入
//! 模块映像, 验签后调用模块导出的操作表。形态与 ref-impl/emod 的模块管理器一致: 模块
//! 占用的飞地物理内存与虚拟地址都不回收, 模块窗口耗尽即取入失败。
//!
//! 运行时的接入点见 [`runtime`]。
#![allow(dead_code)]

pub mod abi;
pub mod loader;
pub mod man;
pub mod net_ops;
pub mod table;
pub mod transport;
pub mod vfs_ops;

#[cfg(target_arch = "riscv64")]
pub mod runtime;

use abi::GetterFn;
use man::Manager;
use transport::Transport;

/// 取入指定模块并返回它的取用器。已登记的模块直接返回既有的取用器。
///
/// 通路与管理器回调集合由调用方给出: 二者由运行时装配, 见 [`runtime`]。
pub fn acquire_module(
	module_id: u32,
	transport: &dyn Transport,
	manager: &Manager,
) -> Result<GetterFn, loader::LoadError> {
	loader::load(module_id, transport, manager)
}

/// 取入指定模块并返回它导出的操作表。
///
/// 加载器登记接口时不知道操作表的尺寸, 只按指针本身判定落点; 本函数知道表的类型,
/// 故按整张表的范围再判一次。表不落在该模块映像的窗口之内时返回
/// [`BadOpsTable`](loader::LoadError::BadOpsTable)。
pub fn acquire_ops_table<T>(
	module_id: u32,
	transport: &dyn Transport,
	manager: &Manager,
) -> Result<&'static T, loader::LoadError> {
	let getter = acquire_module(module_id, transport, manager)?;
	let interface = unsafe { getter(manager) };
	let (va, win_size) =
		table::image_window(module_id).ok_or(loader::LoadError::BadOpsTable)?;
	if !abi::is_region_in_image(
		interface.ops,
		core::mem::size_of::<T>(),
		core::mem::align_of::<T>(),
		va,
		win_size,
	) {
		return Err(loader::LoadError::BadOpsTable);
	}
	// 整张表的落点已确认在本模块映像之内, 映像在取入时已验签, 表由模块的入口写入
	// 且此后只读, 故可建立静态引用。
	Ok(unsafe { &*interface.ops.cast::<T>() })
}
