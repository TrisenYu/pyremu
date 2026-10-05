//! 交给模块的管理器回调集合。
//!
//! 模块经取用器收到本结构体的指针, 由此调用运行时提供的设施。集合相对
//! ref-impl/emod 的同名项裁剪: sittim 已有 `sync_aux::SpinLock`, 且模块与
//! 运行时同处一个地址空间, 故不提供自旋锁回调与测试回调。
//!
//! 集合内既有回调也有平台常量: 二者都只能由运行时给出 —— 设备窗口的虚拟地址在
//! 运行时内由 `paging::map_device_region` 建立, 时间源频率由 `configs_gen` 注入,
//! 模块侧两者都取不到。

use super::abi::GetterFn;

/// 管理器回调集合。字段顺序与模块侧的声明必须一致。
#[repr(C)]
pub struct Manager {
	/// 取用另一个模块并返回它的取用器; 模块不存在或取入失败时返回 None。
	pub alloc_ext_mod: unsafe extern "C" fn(u32) -> Option<GetterFn>,
	/// 申请 n 个 S 模式页, 返回物理地址; 失败返回 `!0`。
	pub fetch_smode_pages: unsafe extern "C" fn(u64) -> u64,
	/// 写控制台, 入参为字节指针与字节数。
	pub write_console: unsafe extern "C" fn(*const u8, u64),
	/// 读时间源, 单位与 `csr::read_time` 相同。
	pub read_time: unsafe extern "C" fn() -> u64,
	/// 把飞地地址空间中的虚拟地址翻译为物理地址。
	///
	/// 设备寄存器按物理地址访问, 而模块的静态存储落在模块窗口内, 其物理地址由 M
	/// 模式按分区交付, 没有固定的偏移可用, 故驱动两处都要经本回调取得物理地址。
	pub va_to_pa: unsafe extern "C" fn(u64) -> u64,
	/// 设备窗口的虚拟地址; 平台不提供该设备时取 0。
	pub device_window_va: u64,
	/// 时间源的频率, 单位为赫兹, 与 [`Manager::read_time`] 的计数单位对应。
	pub time_freq: u64,
}

// ---------------------------------------------------------------
//  测试
// ---------------------------------------------------------------

#[cfg(test)]
mod tests {
	use super::Manager;

	/// 回调集合的尺寸与字段偏移由项数决定, 模块侧按同一布局书写。
	#[test]
	fn test_manager_layout_is_fixed() {
		assert_eq!(core::mem::size_of::<Manager>(), 7 * 8);
		assert_eq!(core::mem::align_of::<Manager>(), 8);
		assert_eq!(core::mem::offset_of!(Manager, alloc_ext_mod), 0);
		assert_eq!(core::mem::offset_of!(Manager, fetch_smode_pages), 8);
		assert_eq!(core::mem::offset_of!(Manager, write_console), 16);
		assert_eq!(core::mem::offset_of!(Manager, read_time), 24);
		assert_eq!(core::mem::offset_of!(Manager, va_to_pa), 32);
		assert_eq!(core::mem::offset_of!(Manager, device_window_va), 40);
		assert_eq!(core::mem::offset_of!(Manager, time_freq), 48);
	}
}
