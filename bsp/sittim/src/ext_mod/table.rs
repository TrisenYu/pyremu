//! 已取入模块的登记表。
//!
//! 表内存放取用器而非模块接口: 与 ref-impl/emod 的 `emodule_table` 相同。调用方
//! 以取用器取得接口, 故接口在登记之后才被访问。

use super::abi::{GetterFn, MAX_MODULES};
use crate::sync_aux::SpinLock;

/// 登记表的一个槽位。
#[derive(Clone, Copy)]
enum Slot {
	/// 尚未取入。
	Empty,
	/// 正在取入。
	///
	/// 请求宿主之前置入该状态: 同一模块的并发取用与重入取用 (模块的入口内取用同
	/// 一个模块) 因它而立即得到失败结果, 不再请求宿主。
	Loading,
	/// 已取入: 存放该模块的取用器, 以及映像在模块窗口内占用的范围。
	///
	/// 映像范围随取用器一同登记: 加载器只按指针判断接口中各表的位置是否落在映像内,
	/// 判据即本范围, 见 [`image_window`]。
	LoadedModule {
		/// 取得该模块接口的函数。
		getter: GetterFn,
		/// 映像在模块窗口内的起始虚拟地址。
		va: u64,
		/// 映像按 2 MiB 分区取整后在模块窗口内占用的字节数。
		///
		/// 与 `ENCLAVE_MODULE_WINDOW_SIZE` 的区别: 后者是窗口的总容量, 供所有模块
		/// 使用; 本字段是其中被这一个模块用掉的长度, 即映像取整后的长度。
		win_size: u64,
	},
}

/// 模块登记表。容量与 ref-impl/emod 的 `emodule_table` 相同。
static REGISTRY: SpinLock<[Slot; MAX_MODULES]> = SpinLock::new([Slot::Empty; MAX_MODULES]);

/// 取回已登记模块的取用器。该模块未登记或编号越界时返回 None。
pub fn lookup(module_id: u32) -> Option<GetterFn> {
	let registry = REGISTRY.lock();
	match registry.get(module_id as usize) {
		Some(Slot::LoadedModule { getter, .. }) => Some(*getter),
		_ => None,
	}
}

/// 取回已登记模块的映像在模块窗口内占用的范围, 依次为起始虚拟地址与字节数。
///
/// 该模块未登记或编号越界时返回 None。接口中的操作表由模块自行给出, 加载器在登记时
/// 不知道它的尺寸, 故取用方按本范围再判一次整张表的落点。
pub fn image_window(module_id: u32) -> Option<(u64, u64)> {
	let registry = REGISTRY.lock();
	match registry.get(module_id as usize) {
		Some(Slot::LoadedModule { va, win_size, .. }) => Some((*va, *win_size)),
		_ => None,
	}
}

/// 占据登记表的空槽位, 表示该模块开始取入。
///
/// 返回该槽位先前是否为空。该模块已在取入过程中或已登记时返回 false。
pub fn begin_load(module_id: u32) -> bool {
	let mut registry = REGISTRY.lock();
	match registry.get_mut(module_id as usize) {
		Some(slot) if matches!(slot, Slot::Empty) => {
			*slot = Slot::Loading;
			true
		}
		_ => false,
	}
}

/// 释放取入过程中占据的槽位, 使该模块可以重新取入。
pub fn abort_load(module_id: u32) {
	let mut registry = REGISTRY.lock();
	if let Some(slot) = registry.get_mut(module_id as usize) {
		if matches!(slot, Slot::Loading) {
			*slot = Slot::Empty;
		}
	}
}

/// 把取用器与其映像的窗口范围登入取入过程中占据的槽位。
///
/// 返回该槽位先前是否处于取入状态; 该模块不在取入过程中时返回 false 且不登记。
pub fn commit_load(module_id: u32, getter: GetterFn, va: u64, win_size: u64) -> bool {
	let mut registry = REGISTRY.lock();
	match registry.get_mut(module_id as usize) {
		Some(slot) if matches!(slot, Slot::Loading) => {
			*slot = Slot::LoadedModule { getter, va, win_size };
			true
		}
		_ => false,
	}
}

// ---------------------------------------------------------------
//  测试
// ---------------------------------------------------------------

#[cfg(test)]
mod tests {
	use super::{abort_load, begin_load, commit_load, image_window, lookup};
	use crate::ext_mod::abi::{GetterFn, ModuleDesc, ModuleInterface, MODULE_NAME_LEN, MAX_MODULES};
	use crate::ext_mod::man::Manager;

	/// 测试用取用器: 返回模块编号为 0、无系统调用表与操作表的接口。
	unsafe extern "C" fn dummy_getter(_manager: *const Manager) -> ModuleInterface {
		ModuleInterface {
			desc: ModuleDesc {
				module_id: 0,
				name: [0; MODULE_NAME_LEN],
				signature: 0,
			},
			syscalls: core::ptr::null(),
			ops: core::ptr::null(),
		}
	}

	const OTHER_GETTER: GetterFn = dummy_getter;

	/// 用例登记的映像窗口范围。
	const IMAGE_VA: u64 = 0xFFFF_FFF0_0000_0000;
	const IMAGE_WIN_SIZE: u64 = 0x40_0000;

	/// 编号越界的模块与未登记的模块一律查不到取用器。
	#[test]
	fn test_lookup_returns_none_for_unregistered_and_out_of_range_ids() {
		assert!(lookup(0x10).is_none());
		assert!(lookup(MAX_MODULES as u32).is_none());
		assert!(lookup(MAX_MODULES as u32 + 5).is_none());
	}

	/// 空槽位只被占据一次; 占用期间该模块查不到取用器。
	#[test]
	fn test_begin_load_claims_an_empty_slot_once() {
		assert!(begin_load(0x11));
		assert!(!begin_load(0x11));
		assert!(lookup(0x11).is_none());
	}

	/// 取入失败释放槽位后, 同一模块可以重新取入并登记。
	#[test]
	fn test_abort_load_releases_the_slot_for_a_second_attempt() {
		assert!(begin_load(0x12));
		abort_load(0x12);
		assert!(lookup(0x12).is_none());

		assert!(begin_load(0x12));
		assert!(commit_load(0x12, OTHER_GETTER, IMAGE_VA, IMAGE_WIN_SIZE));
		assert!(lookup(0x12).is_some());
	}

	/// 登记只接受处于取入状态的槽位, 且登记后不再接受第二次占据。
	#[test]
	fn test_commit_load_requires_a_loading_slot_and_blocks_further_loads() {
		// 空槽位不能直接登记。
		assert!(!commit_load(0x13, OTHER_GETTER, IMAGE_VA, IMAGE_WIN_SIZE));

		assert!(begin_load(0x13));
		assert!(commit_load(0x13, OTHER_GETTER, IMAGE_VA, IMAGE_WIN_SIZE));
		assert!(!commit_load(0x13, OTHER_GETTER, IMAGE_VA, IMAGE_WIN_SIZE));
		assert!(!begin_load(0x13));
		assert!(lookup(0x13).is_some());
	}

	/// 登记之后可以取回映像的窗口范围; 未登记、取入中与编号越界时取不到。
	#[test]
	fn test_image_window_is_available_only_after_commit() {
		assert!(image_window(0x14).is_none());
		assert!(image_window(MAX_MODULES as u32).is_none());

		assert!(begin_load(0x14));
		assert!(image_window(0x14).is_none());

		assert!(commit_load(0x14, OTHER_GETTER, IMAGE_VA, IMAGE_WIN_SIZE));
		assert_eq!(image_window(0x14), Some((IMAGE_VA, IMAGE_WIN_SIZE)));
	}
}
