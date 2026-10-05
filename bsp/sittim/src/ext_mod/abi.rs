//! 模块与运行时之间的二进制接口。
//!
//! 模块以扁平映像交付: 无 ELF 头, 无重定位, 入口在映像偏移 0。模块的入口经
//! [`InitFn`] 交付自己的取用器, 运行时经取用器取得 [`ModuleInterface`]。
//! 形态与 ref-impl/emod 的取用器一致。

use super::man::Manager;

/// 模块名的字节数。
pub const MODULE_NAME_LEN: usize = 32;

/// 登记表的容量, 与 ref-impl/emod 的 `emodule_table` 同容量。
pub const MAX_MODULES: usize = 0x20;

/// 单个模块导出的系统调用处理函数槽位数。
pub const SYSCALL_TABLE_LEN: usize = 256;

/// 系统调用处理函数。
///
/// 六个参数依次对应 a0 至 a5, 返回值为 a0。取六个寄存器而非陷入现场结构体,
/// 使本模块不依赖 `crate::trap` (后者含汇编, 无法在主机目标上编译)。
pub type SyscallHandler = unsafe extern "C" fn(u64, u64, u64, u64, u64, u64) -> u64;

/// 模块导出的系统调用处理函数表。
///
/// 槽位取 `Option` 而非函数指针加有效性位: 空槽位由空指针表示, 调用方据此
/// 判定该编号未实现。
#[repr(C)]
pub struct SyscallTable {
	pub handlers: [Option<SyscallHandler>; SYSCALL_TABLE_LEN],
}

/// 模块的标识信息。
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ModuleDesc {
	pub module_id: u32,
	pub name: [u8; MODULE_NAME_LEN],
	/// 保留字段, 恒为 0。签名本身在映像尾部, 不在本结构体内。
	pub signature: u8,
}

/// 取用器交付给运行时的模块接口: 模块的标识信息与它导出的全部调用能力。
///
/// 取用器按 LP64D 的返回值约定交付本结构体。它大于 16 字节, 故由调用方传入
/// 隐藏指针、被调方在 a0 回传同一指针; 模块侧的 [`InitFn`] 与取用器都必须以
/// 该约定书写, 否则接口在边界处被截断。
#[repr(C)]
#[derive(Clone, Copy)]
pub struct ModuleInterface {
	pub desc: ModuleDesc,
	/// 该模块导出的系统调用处理函数表。
	///
	/// 取裸指针而非引用: 取用时刻模块映像尚未验签, 对未验证的映像构造引用属
	/// 未定义行为。`ext_mod::loader` 在验签并确认该指针落在本模块的窗口之内之后
	/// 才登记接口, 调用方在此之前不得解引用。
	pub syscalls: *const SyscallTable,
	/// 该模块导出的操作表, 形态与 ref-impl/emod 的 `emod_net_api_t` 一致。
	///
	/// 系统调用处理函数表承载载荷发起的调用, 本字段承载运行时自身的调用点 (fd 层的
	/// 套接字读写、调度器对协议栈的推进)。表的类型由运行时与该模块共同约定, 故本
	/// 结构体只给出裸指针; 不导出操作表的模块取空指针。
	///
	/// 取裸指针而非引用, 理由与 `syscalls` 相同。
	pub ops: *const u8,
}

/// 模块取用器。参数为运行时的管理器回调集合, 返回该模块的接口。
pub type GetterFn = unsafe extern "C" fn(manager: *const Manager) -> ModuleInterface;

/// 模块入口, 位于映像偏移 0。参数为运行时的管理器回调集合, 返回该模块的取用器。
///
/// 取用器须可重复调用并返回相同接口: `ext_mod::loader` 在入口之后另调一次取用器,
/// 用于校验模块编号并取得接口。
pub type InitFn = unsafe extern "C" fn(manager: *const Manager) -> GetterFn;

/// 判断 *ptr* 所指的 `[ptr, ptr + size)` 是否落在模块窗口的 `[va, va + window_size)`
/// 之内, 且 *ptr* 对齐满足 *align*。
///
/// 模块在接口中自行给出表指针, 其取值在验签之后仍不可信, 故在登记接口之前必须确认
/// 表的落点。`size` 为 0 时只检查指针本身, 供调用方在表的尺寸未知时使用。
pub fn is_region_in_image(ptr: *const u8, size: usize, align: usize, va: u64, window_size: u64) -> bool {
	let addr = ptr as u64;
	if addr % align as u64 != 0 || addr < va {
		return false;
	}
	match addr.checked_add(size as u64) {
		Some(end) => end <= va + window_size,
		None => false,
	}
}

// ---------------------------------------------------------------
//  测试
// ---------------------------------------------------------------

#[cfg(test)]
mod tests {
	use super::*;
	use crate::constants::CHUNK_2M_SIZE;

	/// `ModuleDesc` 的尺寸与字段偏移固定, 模块侧按同一布局书写。
	#[test]
	fn test_module_desc_layout_is_fixed() {
		assert_eq!(core::mem::size_of::<ModuleDesc>(), 40);
		assert_eq!(core::mem::offset_of!(ModuleDesc, module_id), 0);
		assert_eq!(core::mem::offset_of!(ModuleDesc, name), 4);
		assert_eq!(core::mem::offset_of!(ModuleDesc, signature), 36);
		assert_eq!(core::mem::align_of::<ModuleDesc>(), 4);
	}

	/// 模块接口大于 16 字节, 故按 LP64D 经隐藏指针返回。
	#[test]
	fn test_module_interface_exceeds_register_pair_size() {
		assert_eq!(core::mem::size_of::<ModuleInterface>(), 56);
		assert!(core::mem::size_of::<ModuleInterface>() > 16);
		assert_eq!(core::mem::offset_of!(ModuleInterface, desc), 0);
		assert_eq!(core::mem::offset_of!(ModuleInterface, syscalls), 40);
		assert_eq!(core::mem::offset_of!(ModuleInterface, ops), 48);
		assert_eq!(core::mem::align_of::<ModuleInterface>(), 8);
	}

	/// 处理函数表的尺寸由槽位数决定, 空槽位不占额外空间。
	#[test]
	fn test_syscall_table_is_one_pointer_per_slot() {
		assert_eq!(core::mem::size_of::<SyscallTable>(), SYSCALL_TABLE_LEN * 8);
		assert_eq!(core::mem::size_of::<Option<SyscallHandler>>(), 8);
	}

	/// 表的落点检查覆盖窗口下界、窗口上界与对齐三项。
	#[test]
	fn test_table_region_check_covers_the_window_bounds() {
		let window = 4 * CHUNK_2M_SIZE;
		let base = 0x1_0000_0000_u64;
		let size = core::mem::size_of::<SyscallTable>();
		let align = core::mem::align_of::<SyscallTable>();
		let check = |addr: u64| is_region_in_image(addr as *const u8, size, align, base, window);

		assert!(check(base + 0x1000));

		// 窗口下界之下。
		assert!(!check(base - 0x1000));

		// 表尾越过窗口上界: 窗口末端恰好放不下整个表。
		assert!(!check(base + window - 8));

		// 对齐不满足。
		assert!(!check(base + 0x1004));

		// 空指针落在窗口之外。
		assert!(!is_region_in_image(core::ptr::null(), size, align, base, window));
	}

	/// 尺寸为 0 时只检查指针本身, 供调用方在表的尺寸未知时使用。
	#[test]
	fn test_zero_sized_region_check_admits_the_window_end() {
		let window = CHUNK_2M_SIZE;
		let base = 0x1_0000_0000_u64;
		assert!(is_region_in_image((base + window) as *const u8, 0, 8, base, window));
		assert!(!is_region_in_image((base + window + 8) as *const u8, 0, 8, base, window));
		// 尺寸非零时窗口末尾的地址放不下该区域。
		assert!(!is_region_in_image((base + window) as *const u8, 8, 8, base, window));
	}
}
