//! 取入一个模块。
//!
//! 取入分五步: 申请映像的字节数; 在模块窗口内取得一个槽位, 为它的可执行部分与可写部分各
//! 建成映射; 登记映像的落点后请求宿主交付; 核对映像描述块并验签; 把可执行部分改判为指令
//! 类, 调用模块入口并校验接口。

use super::abi::{GetterFn, InitFn, MAX_MODULES, SyscallTable, is_region_in_image};
use super::errno::{EAGAIN, EINVAL, ENOENT, ENOMEM};
use super::img_descriptor::{DescError, parse_image_desc};
use super::man::Manager;
use super::table;
use super::transport::Transport;
use crate::attest;
use crate::constants::{
	ATTEST_ENABLE, CHUNK_2M_SIZE, ENCLAVE_MEM_ATTR_DATA, ENCLAVE_MEM_ATTR_INSTR,
	ENCLAVE_MODULE_LOAD_VA_INIT, ENCLAVE_MODULE_WINDOW_SIZE, PAGE_SIZE, PTE_R, PTE_W, PTE_X,
	SIG_LEN,
};
use crate::context;
use crate::mem;
use crate::paging;
use crate::prim::arith::align_up;
use crate::prim::string::memset;
use crate::sync_aux::SpinLock;

/// 模块取入失败的原因。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoadError {
	/// 模块编号超出登记表容量。
	BadModuleId,
	/// 宿主报告模块不存在, 或宿主拒绝交付映像。
	NotFound,
	/// 映像的字节数不足以容纳一个签名。
	TooSmall,
	/// 映像的槽位越过模块窗口的上界。
	TooLarge,
	/// 该模块已在取入过程中: 并发取用, 或模块的入口内重入取用同一个模块。
	Loading,
	/// 宿主写入的字节数少于请求的字节数。
	ShortWrite,
	/// 映像接收缓冲区未能建成: 分区申请失败, 页表项建立失败, 或改判为指令类失败。
	MapFailed,
	/// M 模式拒绝登记映像接收缓冲区。
	ImgRecvBufRejected,
	/// 映像验签失败。
	AttestFailed,
	/// 映像描述块不被接受: 映像声明的布局与本侧的约定不符, 或交付的映像与描述块自身的
	/// 声明不符。
	BadImageDesc(DescError),
	/// 接口中的模块编号与请求的编号不一致。
	IdMismatch,
	/// 接口中的系统调用表不落在本模块的窗口之内, 或对齐不满足要求。
	BadSyscallTable,
	/// 接口中的操作表不落在本模块的窗口之内, 或对齐不满足要求。
	BadOpsTable,
}

impl LoadError {
	/// 上报给可信应用的错误码。
	///
	/// 映像与接口的布局不符归 [`EINVAL`], 资源不足归 [`ENOMEM`], 宿主未提供该模块归
	/// [`ENOENT`], 该模块正在取入归 [`EAGAIN`]。可信应用据此区分"宿主未提供该模块"与
	/// "映像有缺陷"两种情形。
	pub const fn to_errno(self) -> u64 {
		match self {
			LoadError::NotFound => ENOENT,
			LoadError::Loading => EAGAIN,
			LoadError::BadModuleId
			| LoadError::BadImageDesc(_)
			| LoadError::AttestFailed
			| LoadError::IdMismatch
			| LoadError::BadSyscallTable
			| LoadError::BadOpsTable => EINVAL,
			LoadError::TooSmall
			| LoadError::TooLarge
			| LoadError::ShortWrite
			| LoadError::MapFailed
			| LoadError::ImgRecvBufRejected => ENOMEM,
		}
	}
}

/// 保护模块窗口内下一个空闲 VA 的锁。
static LOAD_LOCK: SpinLock<()> = SpinLock::new(());

/// 一个模块在模块窗口内占用的字节数。
///
/// 一个模块占用一个槽位, 槽位自起点起为可执行部分、其后为可写部分, 两部分分处两个 2 MiB
/// 分区。该值取 2 MiB, 使 `MAX_MODULES` 个模块恰好占满模块窗口。
///
/// 模块的可写静态数据全部落在 BSS 段, 且该段的起始地址须等于映像本体按页取整的结果: 映像
/// 中初值非零的部分位于可执行部分之内, 装入后该部分只有读与执行权限。该等式由模块的链接
/// 脚本保证 (把 BSS 段的起始地址对齐到页边界, 并拒绝初值非零的可写静态数据), 并由映像
/// 描述块的 `data_class_off` 字段声明, 取入方在改判之前核对
/// [`img_descriptor`]: super::img_descriptor
const MODULE_SLOT_SIZE: u64 = CHUNK_2M_SIZE;

/// 取入指定模块并返回它的取用器, 调用方以取用器取得模块接口。
///
/// 该模块已登记时直接返回既有的取用器, 不重复取入。取入失败的场合, 失败原因经
/// [`LoadError::to_errno`] 转为错误码交回调用方。
pub fn load(
	module_id: u32,
	transport: &dyn Transport,
	manager: &Manager,
) -> Result<GetterFn, LoadError> {
	if (module_id as usize) >= MAX_MODULES {
		return Err(LoadError::BadModuleId);
	}
	if let Some(getter) = table::lookup(module_id) {
		return Ok(getter);
	}

	// 先占据槽位: 同一模块的并发取用与重入取用在此立即失败, 不再请求宿主。
	// 占据发生在取得 LOAD_LOCK 之前, 而 LOAD_LOCK 在调用模块入口之前释放, 故
	// 模块入口内取用同一个模块得到 Loading, 而不是与自身互等。
	if !table::begin_load(module_id) {
		return Err(LoadError::Loading);
	}
	match load_image(module_id, transport, manager) {
		Ok((getter, va, win_size)) => {
			if !table::commit_load(module_id, getter, va, win_size) {
				return Err(LoadError::Loading);
			}
			Ok(getter)
		}
		Err(err) => {
			table::abort_load(module_id);
			Err(err)
		}
	}
}

/// 完成一次取入, 调用方已确认该模块不在登记表中。
///
/// 返回取用器, 以及映像在模块窗口内的起始虚拟地址与槽位的字节数: 后两项随取用器一同
/// 登记, 供取用方判定接口中整张操作表的落点。
///
/// `LOAD_LOCK` 保护模块窗口内下一个空闲 VA, 并在请求宿主交付之前释放: 让出之后本飞地
/// 不再执行, 直到宿主应答; 持有该锁让出会使同一飞地的其它取用方在锁上自等, 而本 hart
/// 直到交还之前都不会释放它。
fn load_image(
	module_id: u32,
	transport: &dyn Transport,
	manager: &Manager,
) -> Result<(GetterFn, u64, u64), LoadError> {
	// 映像的字节数由宿主报告: 飞地既无从推算它, 也不知道宿主侧模块文件的字节数。
	let size = transport
		.request_size(module_id)
		.ok_or(LoadError::NotFound)?;
	if size <= (SIG_LEN as u64) {
		return Err(LoadError::TooSmall);
	}

	let (va, exec_pa) = open_img_recv_buf(size)?;
	// 窗口在选定之后立即登记: 本行之后落到该范围内的地址都已确定属于本模块, 落在其中的
	// 故障因而归属到本模块, 包括取入方自身在缓冲区内的写入与随后对模块入口的调用。
	if !table::note_image_window(module_id, va, MODULE_SLOT_SIZE) {
		return Err(LoadError::Loading);
	}

	// 登记必须早于请求交付: M 模式把交付的字节写入已登记的缓冲区, 未登记时这次交付
	// 无处可写。缓冲区由本飞地自己分配并映射, 交付逐页按本飞地的页表查询物理地址。
	// 映像的末页可跨过可执行部分的末尾, 故登记到映像取整到页的末尾。
	if !transport.register_img_recv_buf(va, align_up(size, PAGE_SIZE)) {
		return Err(LoadError::ImgRecvBufRejected);
	}
	let written = transport
		.fetch(module_id, size)
		.ok_or(LoadError::NotFound)?;
	if written < size {
		return Err(LoadError::ShortWrite);
	}

	// 映像末尾之后以 0 填充: 映像中初值为零的静态数据 (BSS 段) 落在可执行部分之后,
	// 由本函数提供其初值。
	unsafe {
		memset((va + size) as *mut u8, 0, MODULE_SLOT_SIZE - size);
	}

	// 描述块核对早于验签与改判: 映像声明的布局与本侧的约定不符时不予接受, 映像因而不会
	// 取得可执行权限, 它的入口一次也不会被调用。映像在核对之前已全部写入本飞地自己的
	// 缓冲区, 故按字节读出不含未定义的读取。
	let image = unsafe { core::slice::from_raw_parts(va as *const u8, size as usize) };
	let desc = parse_image_desc(image, MODULE_SLOT_SIZE).map_err(LoadError::BadImageDesc)?;

	// 描述块声明的可写数据起始偏移即本飞地建映射时所用的可执行部分长度: 映像本体的
	// 字节数与 `data_class_off` 两项核对合起来给出该等式
	let exec_len = desc.data_class_off as u64;

	if ATTEST_ENABLE && !attest::attest_image(va, size) {
		return Err(LoadError::AttestFailed);
	}
	// 映像本体按页取整可大于映像本体自身的字节数, 此时映像长度越过可执行部分的末尾,
	// 该部分之后的字节落在槽位的可写部分之内, 同样置零, 其中含映像末尾的签名字节。
	if size > exec_len {
		unsafe {
			memset((va + exec_len) as *mut u8, 0, size - exec_len);
		}
	}
	if !seal_img_recv_buf(va, exec_pa, exec_len) {
		return Err(LoadError::MapFailed);
	}

	let init: InitFn = unsafe { core::mem::transmute((va + (desc.entry as u64)) as *const ()) };

	let getter = unsafe { init(manager) };

	let interface = unsafe { getter(manager) };

	if interface.desc.module_id != module_id {
		return Err(LoadError::IdMismatch);
	}
	// 模块的静态数据落在槽位的可写部分之内, 故核对范围是整个槽位。
	let in_image = |ptr: *const u8, size: usize, align: usize| {
		is_region_in_image(ptr, size, align, va, MODULE_SLOT_SIZE)
	};
	if !in_image(
		interface.syscalls as *const u8,
		core::mem::size_of::<SyscallTable>(),
		core::mem::align_of::<SyscallTable>(),
	) {
		return Err(LoadError::BadSyscallTable);
	}
	// 操作表的类型由运行时与该模块共同约定, 本处不知道它的尺寸, 故只确认指针本身
	// 落在槽位之内且对齐; 整张表的落点由取用方按表的类型再判一次
	if !interface.ops.is_null() && !in_image(interface.ops, 0, 8) {
		return Err(LoadError::BadOpsTable);
	}
	Ok((getter, va, MODULE_SLOT_SIZE))
}

/// 在模块窗口内为 *size* 字节的映像建立一个槽位, 返回槽位的起始虚拟地址与可执行部分的
/// 物理基址。
///
/// 槽位自起点起是可执行部分, 长 `align_up(size - SIG_LEN, PAGE_SIZE)` 字节; 其后是可写
/// 部分, 占满槽位剩余的部分。可执行部分止于页边界, 故模块映像中初值非零的静态数据须落在
/// 该边界之前, 初值为零的静态数据 (BSS 段) 落在其后, 由 [`load_image`] 填入零。BSS 段的
/// 起始地址即该边界
///
/// 可执行部分的字节数不作为返回值给出: 该长度须在取得映像之后由描述块核对确认, 而本函数
/// 在请求交付之前就要定下映射, 只能按交付的字节数算出它。取入方在核对通过之后取用描述块
/// 声明的 `data_class_off`, 与本处算出的取值相等。
///
/// 两部分各自向 M 模式申请一个数据类分区, 不取自 S 模式页池: 池所在的分区与池中的页同属
/// 一个类别, 把池的页改判为指令类会使池内的页表与内核栈失去写权限。两部分都逐页按 4 KiB
/// 映射为可读可写, 映像取入完毕后再把可执行部分改判为指令类。两部分必须分处两个分区:
/// 类别是分区的属性, 取指要求指令类而模块的写入要求数据类, 一个分区不能同时满足两者。
///
/// 起始虚拟地址是模块窗口内下一个未交付的地址 (`enclave_module_load_va`), 该地址只增不减,
/// 已交付的地址不回收。故分区申请失败、页表项无法建立或该地址越过窗口上界, 都表现为取入
/// 失败。
///
/// 分配与映射在 `LOAD_LOCK` 内完成: 该地址的读取与递增之间不得插入另一次取入。请求宿主
/// 交付的调用不在此锁内
fn open_img_recv_buf(size: u64) -> Result<(u64, u64), LoadError> {
	// 映像本体不含末尾的签名字节, 可执行部分按映像本体的字节数取整到页。
	let exec_len = align_up(size - (SIG_LEN as u64), PAGE_SIZE);
	if exec_len > MODULE_SLOT_SIZE {
		return Err(LoadError::TooLarge);
	}
	let exec_pages = exec_len / PAGE_SIZE;
	let rw_pages = MODULE_SLOT_SIZE / PAGE_SIZE - exec_pages;

	let _guard = LOAD_LOCK.lock();
	let va = context::ctx().enclave_module_load_va;
	let va_end = va
		.checked_add(MODULE_SLOT_SIZE)
		.ok_or(LoadError::TooLarge)?;
	if va_end > ENCLAVE_MODULE_LOAD_VA_INIT + ENCLAVE_MODULE_WINDOW_SIZE {
		return Err(LoadError::TooLarge);
	}

	let exec_pa = mem::alloc_adjacent_pages(exec_pages, ENCLAVE_MEM_ATTR_DATA);
	if exec_pa == !0u64 || !mem::map_pages_at(va, exec_pa, exec_pages, PTE_R | PTE_W) {
		return Err(LoadError::MapFailed);
	}

	if rw_pages != 0 {
		let rw_pa = mem::alloc_adjacent_pages(rw_pages, ENCLAVE_MEM_ATTR_DATA);
		if rw_pa == !0u64 || !mem::map_pages_at(va + exec_len, rw_pa, rw_pages, PTE_R | PTE_W) {
			return Err(LoadError::MapFailed);
		}
	}

	context::ctx_mut().enclave_module_load_va = va_end;
	Ok((va, exec_pa))
}

/// 把映像的可执行部分改判为指令类, 并逐页把页表项改为可读可执行。
///
/// 模块入口取自可执行部分的起始处, 而映像的写入在调用本函数时已全部完成: 数据类的分区
/// 只有读与写权限, 自其中取指会被拒绝。可写部分保持数据类, 模块对该部分的写入不受影响。
fn seal_img_recv_buf(va: u64, exec_pa: u64, exec_len: u64) -> bool {
	if !mem::set_partition_attr(exec_pa, exec_len, ENCLAVE_MEM_ATTR_INSTR) {
		return false;
	}
	for offset in (0..exec_len).step_by(PAGE_SIZE as usize) {
		if !paging::set_leaf_flags(va + offset, PTE_R | PTE_X) {
			return false;
		}
	}
	true
}

#[cfg(test)]
mod tests {
	use super::*;

	/// 映像与接口的布局不符的四种原因, 以及编号越界这一项。
	const LAYOUT: [LoadError; 6] = [
		LoadError::BadModuleId,
		LoadError::BadImageDesc(DescError::BadMagic),
		LoadError::AttestFailed,
		LoadError::IdMismatch,
		LoadError::BadSyscallTable,
		LoadError::BadOpsTable,
	];

	/// 资源不足的五个原因。
	const RESOURCE: [LoadError; 5] = [
		LoadError::TooSmall,
		LoadError::TooLarge,
		LoadError::ShortWrite,
		LoadError::MapFailed,
		LoadError::ImgRecvBufRejected,
	];

	/// 每种取入失败都归入它所属的错误码, 可信应用据此区分"宿主未提供该模块"与"映像有
	/// 缺陷": 布局与接口类归 EINVAL, 资源类归 ENOMEM, 未提供归 ENOENT, 正在取入归 EAGAIN。
	#[test]
	fn test_to_errno_maps_every_variant() {
		for err in LAYOUT {
			assert_eq!(err.to_errno(), EINVAL, "{err:?}");
		}
		for err in RESOURCE {
			assert_eq!(err.to_errno(), ENOMEM, "{err:?}");
		}
		assert_eq!(LoadError::NotFound.to_errno(), ENOENT);
		assert_eq!(LoadError::Loading.to_errno(), EAGAIN);
	}
}
