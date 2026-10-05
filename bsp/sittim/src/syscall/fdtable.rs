//! 描述符表与打开文件描述表: fd 编号到打开文件描述的映射, 与描述的生命周期管理。
//!
//! 本层只按 [`FdTarget`] 区分描述绑定的对象属于哪一类, 不解释对象的含义: 控制台由运行时
//! 自身提供, 文件系统节点与套接字由提供它们的文件系统与协议栈解释, 其下标与偏移语义留在
//! 各自所在的一层。本层只保存与转交, 故描述符表不随任何单个子系统划入模块。
//!
//! 描述符表项只记录该描述符自身的标志与它所属的打开文件描述; 偏移、访问模式与 O_APPEND、
//! O_NONBLOCK 属描述, 故 dup 得出的描述符与源描述符读写同一个偏移, 一处改动另一处可见。
//! 描述带引用计数, 最后一条引用释放时才释放对象占用的资源。
//!
//! 取描述只经 [open_file_of] 一个入口, 它按表项给出的下标读描述表, 而释放会把描述表的槽位
//! 置为空闲, 故描述一经释放就不可能再被取到。

use crate::sync_aux::SpinLock;

use super::fs::mount::FileSystem;
use super::net;
use super::EMFILE;

/// 打开文件表容量 (fd 0..MAX_FD)。
pub const MAX_FD: usize = 64;

/// 执行时关闭标志。该位属描述符自身, 故与描述符表同处一层。
pub const O_CLOEXEC: u32 = 0o2000000;
/// 描述符标志允许置位的位, 目前只有 [`O_CLOEXEC`]。打开标志中只有该位属描述符自身,
/// 其余位属打开文件描述。
pub const FD_FLAG_MASK: u32 = O_CLOEXEC;

/// 打开文件描述 (open file description): 由一次打开建立、可被多条描述符共享的对象。
/// 偏移与状态标志属描述, 故 dup 得出的描述符与源描述符读写同一个偏移, 一处改动另一处
/// 可见。
#[derive(Clone, Copy)]
pub struct OpenFile {
	/// 描述绑定的对象, 决定读写走哪条通路。
	pub target: FdTarget,
	/// 读写位置, 语义由绑定对象所属的一层给出。
	pub offset: u64,
	/// 打开标志中的状态标志位, 如 O_APPEND 与 O_NONBLOCK。
	pub flags: u32,
	/// 引用该描述的描述符条数。归零时描述与它绑定的资源一并释放。
	refs: u16,
}

/// 描述符表项: 持有该描述符自身的标志与它所属的描述下标。描述符标志目前只有执行时关闭
/// 一位, 属描述符自身, 不随 dup 复制传递。
#[derive(Clone, Copy)]
pub struct FdEntry {
	open_file_idx: usize,
	/// 该描述符自身的标志位。
	pub flags: u32,
}

/// 描述符表与描述表。两表都以 [`Option`] 的 `None` 表示空闲槽位, 故表项一旦存在, 其
/// 下标与绑定对象都必有取值, 不再是需要按类别解释的位域。
///
/// 两者同处一把锁之下: 描述符的分配与释放、描述的引用计数增减必须一起完成, 分成两把锁
/// 会使一项描述在被其它描述符引用的同时被判为空闲。
pub struct Tables {
	fds: [Option<FdEntry>; MAX_FD],
	open_files: [Option<OpenFile>; MAX_FD],
}

impl Tables {
	const fn empty() -> Self {
		Self { fds: [None; MAX_FD], open_files: [None; MAX_FD] }
	}
}

/// 全局描述符表与描述表。飞地 S-mode 运行在单 hart 上且系统调用不可重入 (中断只登记
/// 抢占、待系统调用返回后才切换线程), 用自旋锁仅为保持与共享资源访问约定一致, 锁的
/// 持有区间不含任何 ecall 或让出, 不会自死锁。
pub static FDS: SpinLock<Tables> = SpinLock::new(Tables::empty());

/// 描述绑定的对象。
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum FdTarget {
	/// 控制台, 无索引字段。
	Console,
	/// 文件系统节点, 字段为该节点所属的文件系统与节点下标。
	Vfs(FileSystem, i16),
	/// 套接字, 字段为协议栈的套接字下标。
	Socket(u64),
}

/// 释放描述符时对描述与资源采取的动作。
pub enum FdRelease {
	/// 描述符无效; 调用方返回 EBADF。
	Invalid,
	/// 描述仍被其它描述符引用, 未释放任何资源。
	Retained,
	/// 描述的最后一条引用被释放; 调用方释放该对象占用的资源。
	Released(FdTarget),
}

/// 取描述下标 *open_file_idx* 处的描述; 该下标落在空闲槽位上时返回 None。
fn open_file_slot(tables: &Tables, open_file_idx: usize) -> Option<&OpenFile> {
	tables.open_files.get(open_file_idx)?.as_ref()
}

/// 取描述符 *fd* 所属的描述下标与描述本身; fd 越界、落在空闲编号上, 或该描述已经释放时
/// 返回 None。
///
/// 这是描述表唯一的查找入口, 故描述一经释放 (其槽位置为空闲) 就不可能再被取到, 而取得
/// 的描述必已绑定对象。取得的下标是该描述在描述表中的位置, 供 [store_open_file] 写回。
pub fn open_file_of(tables: &Tables, fd: u64) -> Option<(usize, OpenFile)> {
	let open_file_idx = fd_slot(tables, fd)?.open_file_idx;
	Some((open_file_idx, *open_file_slot(tables, open_file_idx)?))
}

/// 把改动后的描述 *open_file* 写回它在描述表中的槽位。
///
/// *open_file_idx* 须取自 [open_file_of], 该下标在描述未被释放之前一直指代同一项描述。
pub fn store_open_file(tables: &mut Tables, open_file_idx: usize, open_file: OpenFile) {
	tables.open_files[open_file_idx] = Some(open_file);
}

/// 取描述符 *fd* 在描述符表中的下标; 编号超出下标类型的表示范围, 或落在描述符表之外时
/// 返回 None。
///
/// 本层内的描述符表只经本函数由编号取下标: 编号到下标只此一处转换, 且转换连同范围一并
/// 判定, 故取到的下标必定落在表内。
fn fd_index(tables: &Tables, fd: u64) -> Option<usize> {
	usize::try_from(fd).ok().filter(|index| *index < tables.fds.len())
}

/// 取描述符 *fd* 的表项; fd 越界或落在空闲编号上时返回 None。
pub fn fd_slot(tables: &Tables, fd: u64) -> Option<&FdEntry> {
	tables.fds[fd_index(tables, fd)?].as_ref()
}

/// 取描述符 *fd* 的表项, 供写入; fd 越界或落在空闲编号上时返回 None。
pub fn fd_slot_mut(tables: &mut Tables, fd: u64) -> Option<&mut FdEntry> {
	let index = fd_index(tables, fd)?;
	tables.fds[index].as_mut()
}

/// 取描述符 *fd* 所属的描述下标; fd 越界、落在空闲编号上, 或该描述已经释放时返回 None。
pub fn fd_open_file_idx(tables: &Tables, fd: u64) -> Option<usize> {
	open_file_of(tables, fd).map(|(open_file_idx, _)| open_file_idx)
}

/// 取描述符 *fd* 绑定的对象; 无效描述符返回 None。
pub fn fd_target(tables: &Tables, fd: u64) -> Option<FdTarget> {
	open_file_of(tables, fd).map(|(_, open_file)| open_file.target)
}

/// 取不小于 *from* 的最小编号空闲槽位; 该编号起没有空闲槽位时返回 None。
pub fn take_fd_slot_from(tables: &Tables, from: u64) -> Option<u64> {
	// 编号随迭代以 u64 给出, 与调用方的 fd 类型一致, 不在此处做一次宽度转换。
	tables.fds.iter().zip(0u64..).find(|(entry, fd)| entry.is_none() && *fd >= from).map(|(_, fd)| fd)
}

/// 取编号最小的空闲描述符槽位; 槽位耗尽返回 None。
pub fn take_fd_slot(tables: &Tables) -> Option<u64> {
	take_fd_slot_from(tables, 0)
}

/// 取一个空闲的描述槽位; 描述耗尽返回 None。
fn take_open_file_slot(tables: &Tables) -> Option<usize> {
	tables.open_files.iter().position(|open_file| open_file.is_none())
}

/// 建立一项引用计数为 1 的描述并挂到编号 *fd* 上, 返回是否成功。调用前须确认该编号空闲;
/// 编号落在描述符表之外或描述槽位耗尽时返回假, 且不改动任何表项。
pub fn install_fd(tables: &mut Tables, fd: u64, target: FdTarget, flags: u32, desc_flags: u32) -> bool {
	let Some(index) = fd_index(tables, fd) else {
		return false;
	};
	let Some(open_file_idx) = take_open_file_slot(tables) else {
		return false;
	};
	tables.open_files[open_file_idx] = Some(OpenFile { target, offset: 0, flags, refs: 1 });
	tables.fds[index] = Some(FdEntry { open_file_idx, flags: desc_flags & FD_FLAG_MASK });
	true
}

/// 分配一个 fd 并建立绑定的描述, 返回 fd; 槽位或描述耗尽返回 EMFILE。
pub fn alloc_fd(tables: &mut Tables, target: FdTarget, flags: u32, desc_flags: u32) -> u64 {
	let Some(fd) = take_fd_slot(tables) else {
		return EMFILE;
	};
	if !install_fd(tables, fd, target, flags, desc_flags) {
		return EMFILE;
	}
	fd
}

/// 把描述 *open_file* 挂到编号 *fd* 上, 该描述的引用计数加一, 该描述符自身的标志取
/// *desc_flags* 中属于描述符自身的位; 编号落在描述符表之外时不做任何改动。
///
/// 调用前须确认该编号空闲, 且 *open_file* 为下标 *open_file_idx* 处描述的当前取值。引用
/// 计数按该取值加一, 故取得之后、本调用之前不得改写该描述, 释放被替换的描述的动作须在
/// 本调用之后进行。
pub fn dup_at(tables: &mut Tables, fd: u64, open_file_idx: usize, mut open_file: OpenFile, desc_flags: u32) {
	let Some(index) = fd_index(tables, fd) else {
		return;
	};
	open_file.refs += 1;
	tables.open_files[open_file_idx] = Some(open_file);
	tables.fds[index] = Some(FdEntry { open_file_idx, flags: desc_flags & FD_FLAG_MASK });
}

/// 释放描述下标 *open_file_idx* 处描述的一条引用: 引用计数减一, 归零时释放该描述并报告
/// 它绑定的对象, 槽位置为空闲。槽位本就空闲时没有引用可释放, 报告无效。
///
/// 引用计数按读取该槽位时的取值减一, 故本函数不接受调用方先前取得的描述: 同一次调用
/// 先增后减同一描述时 (如 dup3 的两个编号指向同一描述), 传入旧值会把新增的那条引用
/// 一并减掉。
fn release_one(tables: &mut Tables, open_file_idx: usize) -> FdRelease {
	let Some(mut open_file) = open_file_slot(tables, open_file_idx).copied() else {
		return FdRelease::Invalid;
	};
	open_file.refs -= 1;
	if open_file.refs > 0 {
		tables.open_files[open_file_idx] = Some(open_file);
		return FdRelease::Retained;
	}
	tables.open_files[open_file_idx] = None;
	FdRelease::Released(open_file.target)
}

/// 释放被替换的描述 *replaced*。该项取自编号改写之前的查找, 故编号原本空闲时取 None,
/// 此时没有描述可释放, 报告无效。资源本身的释放由调用方在锁外完成。
pub fn release_replaced(tables: &mut Tables, replaced: Option<usize>) -> FdRelease {
	match replaced {
		Some(open_file_idx) => release_one(tables, open_file_idx),
		None => FdRelease::Invalid,
	}
}

/// 释放描述符 *fd*: 清空描述符表项, 引用计数减一, 归零时释放描述。资源本身的释放由
/// 调用方在锁外完成。
pub fn close_fd(tables: &mut Tables, fd: u64) -> FdRelease {
	let Some((open_file_idx, _)) = open_file_of(tables, fd) else {
		return FdRelease::Invalid;
	};
	let Some(index) = fd_index(tables, fd) else {
		return FdRelease::Invalid;
	};
	tables.fds[index] = None;
	release_one(tables, open_file_idx)
}

/// 释放已无引用的描述所绑定的资源。调用方须先放开描述符表的锁, 因为该路径会调用协议栈。
///
/// 目前只有套接字描述占用需要归还的资源; 控制台由运行时自身持有, 文件系统节点没有按描述
/// 分配的资源, 二者的释放不在此处进行。
pub fn release_target(release: FdRelease) {
	if let FdRelease::Released(FdTarget::Socket(index)) = release {
		net::close_socket(index);
	}
}

/// 取 fd 绑定的对象与状态标志; 无效描述符返回 None。
pub fn fd_entry(fd: u64) -> Option<(FdTarget, u32)> {
	let tables = FDS.lock();
	let (_, open_file) = open_file_of(&tables, fd)?;
	Some((open_file.target, open_file.flags))
}

/// 分配一个 fd 并绑定到套接字 *index*, 返回 fd; 槽位耗尽返回 EMFILE。*desc_flags* 取
/// 描述符标志, 由 socket(198) 的 SOCK_CLOEXEC 给出。
pub fn alloc_socket_fd(index: u64, flags: u32, desc_flags: u32) -> u64 {
	let mut tables = FDS.lock();
	alloc_fd(&mut tables, FdTarget::Socket(index), flags, desc_flags)
}
