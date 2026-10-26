//! 外部扩展模块机制。
//!
//! 功能模块由宿主按需交付: 飞地在该模块首次被用到时取入
//! 模块映像, 验签后调用模块导出的操作表。
//! 模块占用的飞地物理内存与虚拟地址都不回收, 扩展模块虚拟地址空间耗尽即取入失败。
#![allow(dead_code)]

pub mod abi;
pub mod img_descriptor;
pub mod loader;
pub mod man;
pub mod net_ops;
pub mod table;
pub mod transport;
pub mod vfs_ops;

/// 系统调用错误码定义。
#[path = "../syscall/errno.rs"]
pub mod errno;

#[cfg(target_arch = "riscv64")]
pub mod runtime;

use man::Manager;
use transport::Transport;

/// 取入指定模块并返回它导出的操作表。
pub fn acquire_ops_table<T>(
	module_id: u32,
	transport: &dyn Transport,
	manager: &Manager,
) -> Result<&'static T, loader::LoadError> {
	let getter = loader::load(module_id, transport, manager)?;
	let interface = unsafe { getter(manager) };
	let (va, win_size) = table::image_window(module_id).ok_or(loader::LoadError::BadOpsTable)?;
	if !abi::is_region_in_image(
		interface.ops,
		core::mem::size_of::<T>(),
		core::mem::align_of::<T>(),
		va,
		win_size,
	) {
		return Err(loader::LoadError::BadOpsTable);
	}
	Ok(unsafe { &*interface.ops.cast::<T>() })
}
