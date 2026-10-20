//! 已取入模块的登记表。
//!
//! 表内存放取用器而非模块接口: 调用方取用器取得接口, 故接口在登记之后才被访问。
//!
//! 除取用之外, 本表还回答两个问题: 某一次故障属于哪个模块, 以及某个模块的映像在窗口内
//! 占用了什么范围, 分别见 [`module_of`] 与 [`image_window`]。前者供故障归属使用: 模块取入
//! 之后运行在本飞地的地址空间与栈上, 它在运行中出错时, 记录中只有故障指令地址与故障地址
//! 两项, 归属只能按映像窗口判定。

use super::abi::{GetterFn, MAX_MODULES};
use crate::sync_aux::SpinLock;

/// 登记表的一个槽位。
#[derive(Clone, Copy)]
enum Slot {
	/// 尚未取入。
	Empty,
	/// 正在取入。
	///
	/// 请求宿主之前置入该状态: 同一模块的并发取用与重入取用因它而立即得到失败结果, 不再
	/// 请求宿主。重入指模块的入口内取用同一个模块。
	///
	/// 窗口范围在取入方选定地址之后由 [`note_image_window`] 填入, 在此之前两项取 0:
	/// 取入过程中的故障同样按窗口归属, 而映像的位置要到那时才确定。
	Loading {
		/// 映像在模块窗口内的起始虚拟地址; 尚未选定时为 0。
		va: u64,
		/// 映像在模块窗口内占用的字节数。
		win_size: u64,
	},
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
	/// 运行中出错: 该模块的取用器不再交出, 映像窗口保留。
	///
	/// 取用器不再交出使该模块的下一次调用重新请求宿主交付映像, 出错的那份映像因而不再被
	/// 调入。窗口保留的依据是模块窗口内的地址只增不减: 该范围内的地址不会属于别的模块,
	/// 记下它使落在其中的故障仍归属到本模块。
	Failed {
		/// 映像在模块窗口内的起始虚拟地址。
		va: u64,
		/// 映像在模块窗口内占用的字节数。
		win_size: u64,
	},
}

/// 模块登记表。
static REGISTRY: SpinLock<[Slot; MAX_MODULES]> = SpinLock::new([Slot::Empty; MAX_MODULES]);

/// 取回已登记模块的取用器。
///
/// 该模块未登记、槽位空闲或正在取入时返回 None。
pub fn lookup(module_id: u32) -> Option<GetterFn> {
	let registry = REGISTRY.lock();
	match registry.get(module_id as usize) {
		Some(Slot::LoadedModule { getter, .. }) => Some(*getter),
		_ => None,
	}
}

/// 取回模块映像在模块窗口内占用的范围, 依次为起始虚拟地址与字节数。
///
/// 该范围在登记时随取用器一同写入, 此后不再改动, 直到该模块重新取入。正在取入的槽位在
/// 选定地址之前不给出范围, 见 [`note_image_window`]; 槽位空闲或编号越界时返回 None。
/// 运行中出错的槽位仍给出范围: 出错的那份映像不再被调入, 而它占据的地址范围照旧,
/// 见 [`Slot::Failed`]。
///
/// 接口中的操作表由模块自行给出, 加载器在登记时不知道它的尺寸, 故取用方按本范围再判一次
/// 整张表的位置。
pub fn image_window(module_id: u32) -> Option<(u64, u64)> {
	let registry = REGISTRY.lock();
	match registry.get(module_id as usize) {
		Some(Slot::LoadedModule { va, win_size, .. }) | Some(Slot::Failed { va, win_size }) => {
			Some((*va, *win_size))
		}
		_ => None,
	}
}

/// 判定一次故障属于哪个模块, 返回该模块的编号。
///
/// 先按故障指令地址 *sepc* 判定, *sepc* 不在任何模块的映像窗口之内时再按故障地址 *stval*
/// 判定。故障指令必然属于正在执行的模块, 故 *sepc* 的判定优先; *stval* 的判定覆盖模块把
/// 回调指针交给别处执行这一情形, 那时取指发生在模块的窗口之外而故障地址仍在窗口之内。
/// 两者都不在窗口之内时返回 None, 该次故障不由任何模块引起。
pub fn module_of(sepc: u64, stval: u64) -> Option<u32> {
	owner_of_address(sepc).or_else(|| owner_of_address(stval))
}

/// 判定地址 *addr* 落在哪个模块的映像窗口之内, 返回该模块的编号。
///
/// 已选定窗口的槽位都参与判定: 已登记的模块、正在取入的模块, 以及运行中出错的模块。正在
/// 取入的模块必须参与: 取入过程会在映像之内取指并调用模块的入口, 也会写入映像的字节, 它
/// 在这期间出错时槽位正处在取入状态。
///
/// 窗口互不重叠, 故至多命中一个槽位; 都不命中时返回 None。
fn owner_of_address(addr: u64) -> Option<u32> {
	let registry = REGISTRY.lock();
	for (module_id, slot) in registry.iter().enumerate() {
		let Some((va, win_size)) = slot_window(slot) else {
			continue;
		};
		if win_size != 0 && addr >= va && addr - va < win_size {
			return Some(module_id as u32);
		}
	}
	None
}

/// 槽位已登记的映像窗口范围; 尚未选定窗口的槽位与空闲槽位返回 None。
fn slot_window(slot: &Slot) -> Option<(u64, u64)> {
	match slot {
		Slot::Empty => None,
		Slot::Loading { va, win_size } => Some((*va, *win_size)),
		Slot::LoadedModule { va, win_size, .. } => Some((*va, *win_size)),
		Slot::Failed { va, win_size } => Some((*va, *win_size)),
	}
}

/// 占据槽位, 表示该模块开始取入。
///
/// 返回该槽位是否可被占据。该模块已在取入过程中或已登记时返回 false; 空闲与运行中出错的
/// 槽位都可被占据, 后者使该模块能够重新取入。
pub fn begin_load(module_id: u32) -> bool {
	let mut registry = REGISTRY.lock();
	match registry.get_mut(module_id as usize) {
		Some(slot) if matches!(slot, Slot::Empty | Slot::Failed { .. }) => {
			*slot = Slot::Loading { va: 0, win_size: 0 };
			true
		}
		_ => false,
	}
}

/// 登记正在取入的模块在模块窗口内占用的范围。
///
/// 在取入方为映像选定窗口之后立即调用: 此后直到取入结束, 槽位保持取入状态, 而它的窗口
/// 已经对外可知, 取入期间落在窗口之内的故障因而归属到本模块, 见 [`module_of`]。取入结束
/// 指登记与释放两种结果。
///
/// 返回该槽位是否处于取入状态; 该模块不在取入过程中时返回 false 且不登记。
pub fn note_image_window(module_id: u32, va: u64, win_size: u64) -> bool {
	let mut registry = REGISTRY.lock();
	match registry.get_mut(module_id as usize) {
		Some(Slot::Loading { va: slot_va, win_size: slot_win_size }) => {
			*slot_va = va;
			*slot_win_size = win_size;
			true
		}
		_ => false,
	}
}

/// 释放取入过程中占据的槽位, 使该模块可以重新取入。
pub fn abort_load(module_id: u32) {
	let mut registry = REGISTRY.lock();
	if let Some(slot) = registry.get_mut(module_id as usize) {
		if matches!(slot, Slot::Loading { .. }) {
			*slot = Slot::Empty;
		}
	}
}

/// 登记该模块在运行中出错: 取入过程中的槽位回到空闲, 已登记的槽位转为失败态。
///
/// 取入过程中的故障由陷态路径处置, 请求交付的那个调用不会再返回, [`abort_load`] 因而不会
/// 被执行到。本函数是这一情形下唯一的释放点: 没有它, 该槽位永久停在取入状态, 该模块此后
/// 的每一次调用都只能得到"正在取入"。
///
/// 已登记的槽位转为 [`Slot::Failed`], 该模块的取用器不再交出, 下一次调用重新请求宿主交付
/// 映像。编号越界与空闲的槽位不作改动。
pub fn mark_failed(module_id: u32) {
	let mut registry = REGISTRY.lock();
	if let Some(slot) = registry.get_mut(module_id as usize) {
		*slot = match *slot {
			Slot::LoadedModule { va, win_size, .. } | Slot::Failed { va, win_size } => {
				Slot::Failed { va, win_size }
			}
			Slot::Loading { .. } | Slot::Empty => Slot::Empty,
		};
	}
}

/// 把取用器与其映像的窗口范围登入取入过程中占据的槽位。
///
/// 返回该槽位先前是否处于取入状态; 该模块不在取入过程中时返回 false 且不登记。
pub fn commit_load(module_id: u32, getter: GetterFn, va: u64, win_size: u64) -> bool {
	let mut registry = REGISTRY.lock();
	match registry.get_mut(module_id as usize) {
		Some(slot) if matches!(slot, Slot::Loading { .. }) => {
			*slot = Slot::LoadedModule { getter, va, win_size };
			true
		}
		_ => false,
	}
}



/// 把编号 *module_id* 的槽位归还为空闲。
///
/// 登记表是进程内唯一的静态量, 而编号的取值空间只有 [`MAX_MODULES`] 个, 不足以让每个
/// 用例独占一个。用例在取用编号之前先归还它, 因而与其它用例的执行次序无关。
#[cfg(test)]
pub fn reset_slot(module_id: u32) {
	let mut registry = REGISTRY.lock();
	if let Some(slot) = registry.get_mut(module_id as usize) {
		*slot = Slot::Empty;
	}
}

#[cfg(test)]
mod tests {
	use super::{
		abort_load, begin_load, commit_load, image_window, lookup, mark_failed, module_of,
		note_image_window, owner_of_address, reset_slot,
	};
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

	/// 归属判定用例各自的窗口起始地址。
	///
	/// 判定扫描全表, 而窗口互不重叠的约定使至多命中一个槽位, 故各用例的窗口既与上面登记
	/// 类用例的窗口不同, 相互之间也不重叠。
	const OWNED_VA_LOADING: u64 = 0xFFFF_FFF0_4000_0000;
	const OWNED_VA_BY_SEPC: u64 = 0xFFFF_FFF0_8000_0000;
	const OWNED_VA_BY_STVAL: u64 = 0xFFFF_FFF0_C000_0000;

	/// 落在全部用例的映像窗口之外的地址, 即管理器自身窗口内的地址。
	const OUTSIDE_VA: u64 = 0xFFFF_FFE0_0000_0000;

	/// 运行中出错与重新取入两个用例各自的窗口起始地址。
	const FAILED_VA: u64 = 0xFFFF_FFF0_1000_0000;
	const RELOAD_VA: u64 = 0xFFFF_FFF0_1400_0000;

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
		reset_slot(0x11);
		assert!(begin_load(0x11));
		assert!(!begin_load(0x11));
		assert!(lookup(0x11).is_none());
	}

	/// 取入失败释放槽位后, 同一模块可以重新取入并登记。
	#[test]
	fn test_abort_load_releases_the_slot_for_a_second_attempt() {
		reset_slot(0x12);
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
		reset_slot(0x13);
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
		reset_slot(0x14);
		assert!(image_window(0x14).is_none());
		assert!(image_window(MAX_MODULES as u32).is_none());

		assert!(begin_load(0x14));
		assert!(image_window(0x14).is_none());

		assert!(commit_load(0x14, OTHER_GETTER, IMAGE_VA, IMAGE_WIN_SIZE));
		assert_eq!(image_window(0x14), Some((IMAGE_VA, IMAGE_WIN_SIZE)));
	}

	/// 取入过程中选定窗口之后, 该窗口之内的地址即归属到本模块。
	///
	/// 取入过程本身会在映像之内取指, 也会写入映像的字节, 故槽位处在取入状态时窗口必须
	/// 已经对外可知。
	#[test]
	fn test_owner_of_address_covers_loading_slots() {
		reset_slot(0x11);
		assert!(begin_load(0x11));
		// 选定窗口之前没有范围可归属。
		assert!(owner_of_address(OWNED_VA_LOADING).is_none());

		assert!(note_image_window(0x11, OWNED_VA_LOADING, IMAGE_WIN_SIZE));
		assert_eq!(owner_of_address(OWNED_VA_LOADING), Some(0x11));
		assert_eq!(
			owner_of_address(OWNED_VA_LOADING + IMAGE_WIN_SIZE - 1),
			Some(0x11)
		);
		// 窗口的首字节之前与末字节之后都不归属。
		assert!(owner_of_address(OWNED_VA_LOADING - 1).is_none());
		assert!(owner_of_address(OWNED_VA_LOADING + IMAGE_WIN_SIZE).is_none());
		abort_load(0x11);
	}

	/// 故障归属先按故障指令地址判定, 再按故障地址判定, 两者都不在窗口之内时不属于任何
	/// 模块。
	#[test]
	fn test_module_of_prefers_sepc_and_falls_back_to_stval() {
		reset_slot(0x13);
		reset_slot(0x14);
		assert!(begin_load(0x13));
		assert!(commit_load(0x13, OTHER_GETTER, OWNED_VA_BY_SEPC, IMAGE_WIN_SIZE));
		assert!(begin_load(0x14));
		assert!(commit_load(0x14, OTHER_GETTER, OWNED_VA_BY_STVAL, IMAGE_WIN_SIZE));

		// 两个地址分属两个模块: 取故障指令所属的模块。
		assert_eq!(
			module_of(OWNED_VA_BY_SEPC + 0x1000, OWNED_VA_BY_STVAL + 0x1000),
			Some(0x13)
		);
		// 故障指令不在任何窗口之内: 取故障地址所属的模块。
		assert_eq!(module_of(OUTSIDE_VA, OWNED_VA_BY_STVAL + 0x1000), Some(0x14));
		// 两个地址都不在窗口之内: 该次故障不由任何模块引起。
		assert_eq!(module_of(OUTSIDE_VA, OUTSIDE_VA + 0x1000), None);
	}

	/// 窗口范围只登在处于取入状态的槽位上。
	#[test]
	fn test_note_image_window_requires_a_loading_slot() {
		reset_slot(0x12);
		// 空闲槽位不登记, 也不给出窗口。
		assert!(!note_image_window(0x12, IMAGE_VA, IMAGE_WIN_SIZE));
		assert!(image_window(0x12).is_none());

		assert!(begin_load(0x12));
		assert!(commit_load(0x12, OTHER_GETTER, IMAGE_VA, IMAGE_WIN_SIZE));
		// 已登记的槽位不再接受窗口改动: 窗口由 commit_load 给出。
		assert!(!note_image_window(0x12, IMAGE_VA + IMAGE_WIN_SIZE, IMAGE_WIN_SIZE));
		assert_eq!(image_window(0x12), Some((IMAGE_VA, IMAGE_WIN_SIZE)));
	}

	/// 取入过程中的故障使槽位回到空闲: 该模块可以重新取入, 而不是永久停在取入状态。
	///
	/// 触发条件是模块入口自身的故障。此后请求交付的那个调用不会再返回, 取入方没有机会释放
	/// 槽位, 释放只能由故障处理路径完成。
	#[test]
	fn test_mark_failed_releases_a_loading_slot() {
		reset_slot(0x15);
		assert!(begin_load(0x15));
		assert!(note_image_window(0x15, FAILED_VA, IMAGE_WIN_SIZE));
		assert!(lookup(0x15).is_none());

		mark_failed(0x15);
		// 槽位回到空闲: 该模块可以重新取入, 出错那份映像的窗口不再登记。
		assert!(image_window(0x15).is_none());
		assert!(owner_of_address(FAILED_VA).is_none());
		assert!(begin_load(0x15));
		abort_load(0x15);
	}

	/// 运行中出错的模块不再交出取用器, 出错那份映像的窗口保留。
	///
	/// 窗口保留使落在其中的故障仍归属到该模块; 取用器不再交出使该模块的下一次调用重新请求
	/// 宿主交付映像, 重新取入之后取得新的窗口。
	#[test]
	fn test_mark_failed_hides_the_getter_and_keeps_the_window() {
		reset_slot(0x16);
		assert!(begin_load(0x16));
		assert!(commit_load(0x16, OTHER_GETTER, FAILED_VA, IMAGE_WIN_SIZE));

		mark_failed(0x16);
		assert!(lookup(0x16).is_none());
		assert_eq!(image_window(0x16), Some((FAILED_VA, IMAGE_WIN_SIZE)));
		assert_eq!(module_of(FAILED_VA + 0x1000, OUTSIDE_VA), Some(0x16));

		// 重新取入: 槽位可被占据, 新的窗口取代出错那份。
		assert!(begin_load(0x16));
		assert!(commit_load(0x16, OTHER_GETTER, RELOAD_VA, IMAGE_WIN_SIZE));
		assert!(lookup(0x16).is_some());
		assert_eq!(module_of(RELOAD_VA + 0x1000, OUTSIDE_VA), Some(0x16));
		reset_slot(0x16);
	}

	/// 出错之后的模块在重新取入之前不再被取用。
	///
	/// 模块因自身的缺陷而在调用中出错时, 故障处理路径把槽位转为失败态。故出错的模块在随后
	/// 的调用中重新请求宿主交付映像, 而不是继续使用出错那份映像登记的表。
	#[test]
	fn test_a_failed_module_is_not_handed_out_until_it_is_loaded_again() {
		reset_slot(0x13);
		assert!(begin_load(0x13));
		assert!(commit_load(0x13, OTHER_GETTER, IMAGE_VA, IMAGE_WIN_SIZE));

		// 反复出错: 取用器取不到, 窗口不被改写。
		for _ in 0..3 {
			mark_failed(0x13);
			assert!(lookup(0x13).is_none());
			assert!(!note_image_window(0x13, OWNED_VA_BY_SEPC, IMAGE_WIN_SIZE));
			assert_eq!(image_window(0x13), Some((IMAGE_VA, IMAGE_WIN_SIZE)));
		}
	}
}
