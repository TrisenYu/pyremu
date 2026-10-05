//! 取入一个模块。
//!
//! 取入分四步: 申请映像的字节数, 从本飞地的 S 模式页池分配并映射映像接收缓冲区, 登记
//! 该缓冲区后请求宿主交付, 最后验签、调用模块入口并校验接口。每一步的失败原因见
//! [`LoadError`]。

use super::abi::{is_region_in_image, GetterFn, InitFn, SyscallTable, MAX_MODULES};
use super::man::Manager;
use super::table;
use super::transport::Transport;
use crate::attest;
use crate::constants::{
	ATTEST_ENABLE, ENCLAVE_MODULE_LOAD_VA_INIT, ENCLAVE_MODULE_WINDOW_SIZE, LEVEL_PAGE,
	PAGE_SIZE, PTE_R, PTE_W, PTE_X, SIG_LEN,
};
use crate::context;
use crate::mem;
use crate::mem_prim::align::align_up;
use crate::mem_prim::string::memset;
use crate::paging;
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
	/// 映像取整到页后越过模块窗口的上界。
	TooLarge,
	/// 该模块已在取入过程中: 并发取用, 或模块的入口内重入取用同一个模块。
	Loading,
	/// 宿主写入的字节数少于请求的字节数。
	ShortWrite,
	/// 映像接收缓冲区未能建成: 物理页不足, 或页表项建立失败。
	MapFailed,
	/// M 模式拒绝登记映像接收缓冲区。
	ImgRecvBufRejected,
	/// 映像验签失败。
	AttestFailed,
	/// 接口中的模块编号与请求的编号不一致。
	IdMismatch,
	/// 接口中的系统调用表不落在本模块的窗口之内, 或对齐不满足要求。
	BadSyscallTable,
	/// 接口中的操作表不落在本模块的窗口之内, 或对齐不满足要求。
	BadOpsTable,
}

/// 保护模块窗口内下一个空闲 VA 的锁。
static LOAD_LOCK: SpinLock<()> = SpinLock::new(());

/// 取入指定模块并返回它的取用器, 调用方以取用器取得模块接口。
///
/// 该模块已登记时直接返回既有的取用器, 不重复取入。
pub fn load(
	module_id: u32,
	transport: &dyn Transport,
	manager: &Manager,
) -> Result<GetterFn, LoadError> {
	if module_id as usize >= MAX_MODULES {
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
/// 返回取用器, 以及映像在模块窗口内的起始虚拟地址与接收缓冲区的字节数: 后两项随取用器
/// 一同登记, 供取用方判定接口中整张操作表的落点。
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
	let size = transport.request_size(module_id).ok_or(LoadError::NotFound)?;
	if size <= SIG_LEN as u64 {
		return Err(LoadError::TooSmall);
	}

	let (va, recv_buf_len) = open_img_recv_buf(size)?;

	// 登记必须早于请求交付: M 模式把交付的字节写入已登记的缓冲区, 未登记时这次交付
	// 无处可写。缓冲区由本飞地自己分配并映射, 交付逐页按本飞地的页表查询物理地址。
	if !transport.register_img_recv_buf(va, recv_buf_len) {
		return Err(LoadError::ImgRecvBufRejected);
	}
	let written = transport.fetch(module_id, size).ok_or(LoadError::NotFound)?;
	if written < size {
		return Err(LoadError::ShortWrite);
	}

	// 缓冲区中超出映像的部分以 0 填充, 使模块对未初始化数据的读取有确定结果。
	let tail = recv_buf_len - size;
	if tail != 0 {
		unsafe {
			memset((va + size) as *mut u8, 0, tail);
		}
	}

	if ATTEST_ENABLE && !attest::attest_image(va, size) {
		return Err(LoadError::AttestFailed);
	}

	// 映像的入口在偏移 0, 故映像的加载 VA 就是入口地址。
	let init: InitFn = unsafe { core::mem::transmute(va as *const ()) };
	let getter = unsafe { init(manager) };
	let interface = unsafe { getter(manager) };

	if interface.desc.module_id != module_id {
		return Err(LoadError::IdMismatch);
	}
	let in_image = |ptr: *const u8, size: usize, align: usize| {
		is_region_in_image(ptr, size, align, va, recv_buf_len)
	};
	if !in_image(
		interface.syscalls as *const u8,
		core::mem::size_of::<SyscallTable>(),
		core::mem::align_of::<SyscallTable>(),
	) {
		return Err(LoadError::BadSyscallTable);
	}
	// 操作表的类型由运行时与该模块共同约定, 本处不知道它的尺寸, 故只确认指针本身
	// 落在缓冲区之内且对齐; 整张表的落点由取用方按表的类型再判一次, 见
	// [`acquire_ops_table`](super::acquire_ops_table)。
	if !interface.ops.is_null() && !in_image(interface.ops, 0, 8) {
		return Err(LoadError::BadOpsTable);
	}
	Ok((getter, va, recv_buf_len))
}

/// 在模块窗口内为 *size* 字节的映像建立接收缓冲区, 返回它的起始虚拟地址与字节数。
///
/// 物理页取自本飞地的 S 模式页池, 逐页按 4 KiB 映射为可读可写可执行; 起始虚拟地址取自
/// 模块窗口内的单调游标, 游标只增不减, 已交付的地址不回收。故池的物理页不足、页表项
/// 无法建立或游标越过窗口上界, 都表现为取入失败。
///
/// 分配与映射在 `LOAD_LOCK` 内完成: 游标的读取与递增之间不得插入另一次取入。请求宿主
/// 交付的调用不在此锁内, 见 [`load_image`]。
fn open_img_recv_buf(size: u64) -> Result<(u64, u64), LoadError> {
	let n_pages = align_up(size, PAGE_SIZE) / PAGE_SIZE;
	let recv_buf_len = n_pages * PAGE_SIZE;

	let _guard = LOAD_LOCK.lock();
	let va = context::ctx().enclave_module_load_va;
	let va_end = va.checked_add(recv_buf_len).ok_or(LoadError::TooLarge)?;
	if va_end > ENCLAVE_MODULE_LOAD_VA_INIT + ENCLAVE_MODULE_WINDOW_SIZE {
		return Err(LoadError::TooLarge);
	}

	let pa = mem::try_alloc_smode_page(n_pages);
	if pa == !0u64 {
		return Err(LoadError::MapFailed);
	}
	for i in 0..n_pages {
		let offset = i * PAGE_SIZE;
		let flags = PTE_R | PTE_W | PTE_X;
		if paging::map_page(va + offset, pa + offset, flags, LEVEL_PAGE).is_err() {
			return Err(LoadError::MapFailed);
		}
	}

	context::ctx_mut().enclave_module_load_va = va_end;
	Ok((va, recv_buf_len))
}
