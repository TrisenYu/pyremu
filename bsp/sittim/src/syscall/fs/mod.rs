//! 文件系统系统调用: openat/close/read/write/readv/writev/lseek/pread64/pwrite64/
//! fstat/fstatat/getdents/getcwd/mkdirat/unlinkat/faccessat/fcntl/fsync/ftruncate/ppoll。
//!
//! 描述符表与打开文件描述表由运行时持有, 见 [super::fdtable]: 它同时承载控制台、文件系统
//! 节点与套接字三类描述, 不随文件系统划入模块。控制台是标准输入与标准输出/错误的三项
//! 描述, 占用 fd 0/1/2, 由 [vfs_init] 建立; 载荷可关闭这些编号并交由后续的 open 复用。
//! 读与写按描述绑定的对象分派: 绑定套接字时转交 [super::net], 绑定文件系统节点时经模块
//! 导出的操作表调用。
//!
//! 目录树与文件数据由文件系统持有, 其存储形态只在文件系统内, 故本层按不透明的节点下标
//! 与之交互。偏移、O_APPEND、复制与轮询等描述符一级的语义留在本层: 本层取得描述后改动其
//! 偏移或标志, 再经 [super::fdtable::store_open_file] 写回。
//!
//! 文件系统只交出 ino、节点类别、权限位、链接数、长度与块数, musl riscv64 的 `struct stat`
//! 中文件类型的编码 (S_IFDIR/S_IFREG) 由本层按节点类别补齐。
//!
//! 一个文件系统由 [mount] 的挂载表按挂载点选出, 见该模块。根文件系统由宿主交付的文件系统
//! 模块实现, 在 [vfs_init] 中取入。
//!
//! 本层按调用类别分为六个子模块, 另有挂载层与内核自带的文件系统:
//! - [consts] — 容量、打开标志、fcntl 命令、定位基准、文件类型编码与各调用共用的标志位掩码
//! - [path] — 路径参数的读取与路径类调用 (mkdirat/unlinkat/faccessat/getcwd)
//! - [stat] — musl riscv64 `struct stat` 的布局与填充, 以及 stat 类调用 (fstat/fstatat)
//! - [io] — 读写与偏移类调用 (read/write/readv/writev/pread64/pwrite64/lseek/getdents/
//!   ftruncate)
//! - [fdops] — 描述符类调用 (openat/close/dup/dup3/fcntl/fsync)
//! - [poll] — ppoll
//! - [mount] — 挂载表与按路径的分派
//! - [sysfs] — 内核自带的伪文件系统

#![allow(dead_code)]

mod consts;
mod fdops;
mod io;
pub(super) mod mount;
mod path;
mod poll;
mod stat;
mod sysfs;

pub use consts::{O_NONBLOCK, POLL_WAIT_ADDR};
pub use fdops::*;
pub use io::*;
pub use mount::mount_init;
pub use path::*;
pub use poll::*;
pub use stat::*;

use crate::ext_mod::vfs_ops::{VfsOps, VfsStat, VfsStatus};

use super::fdtable::{
	FDS, FdTarget, OpenFile, Tables, fd_slot, install_fd, open_file_of,
};
use super::{
	EBADF, EBUSY, EEXIST, EINVAL, EISDIR, ENAMETOOLONG, ENOENT, ENOMEM, ENOSPC, ENOSYS,
	ENOTDIR, ENOTEMPTY,
};

use consts::CONSOLE_FDS;
use mount::FileSystem;

/// 把模块给出的操作结果折算为 errno。
pub(super) fn vfs_errno(status: VfsStatus) -> u64 {
	match status {
		VfsStatus::Ok => 0,
		VfsStatus::NoEntry => ENOENT,
		VfsStatus::Exists => EEXIST,
		VfsStatus::NotDir => ENOTDIR,
		VfsStatus::NotEmpty => ENOTEMPTY,
		VfsStatus::NoMemory => ENOMEM,
		VfsStatus::NoSpace => ENOSPC,
		VfsStatus::NameTooLong => ENAMETOOLONG,
		VfsStatus::Invalid => EINVAL,
		VfsStatus::NoSys => ENOSYS,
		VfsStatus::IsDir => EISDIR,
		VfsStatus::Busy => EBUSY,
	}
}

/// 取文件 *node* 的长度; 节点不存在时返回 None。
pub(super) fn node_size(ops: &VfsOps, node: i16) -> Option<u64> {
	let mut vstat = VfsStat::zero();
	if unsafe { (ops.stat)(node, &mut vstat) } {
		Some(vstat.size as u64)
	} else {
		None
	}
}

/// 取描述符 *fd* 绑定的文件系统与节点下标; 描述符无效, 或绑定的是控制台与套接字时返回
/// None。
pub(super) fn fd_node(tables: &Tables, fd: u64) -> Option<(FileSystem, i16)> {
	match open_file_of(tables, fd)?.1.target {
		FdTarget::Vfs(fs, node) => Some((fs, node)),
		FdTarget::Console | FdTarget::Socket(_) => None,
	}
}

/// 取 fd 所属的描述下标、描述本身、它绑定的文件系统节点下标与该文件系统的操作表。fd
/// 绑定的不是文件时返回 *not_file_err*, 无效描述符返回 EBADF, 该文件系统尚未交付时返回
/// ENOSYS。
///
/// 描述按取得时的取值交给调用方, 调用方改动其偏移或标志后须经
/// [super::fdtable::store_open_file] 写回; 同一把锁的持有期间没有其它改动者, 故写回不会
/// 覆盖别人的改动。
#[inline]
pub(super) fn fd_vfs_node(
	tables: &Tables, fd: u64, not_file_err: u64,
) -> Result<(usize, OpenFile, i16, &'static VfsOps), u64> {
	let Some((open_file_idx, open_file)) = open_file_of(tables, fd) else {
		return Err(EBADF);
	};
	match open_file.target {
		FdTarget::Vfs(fs, node) =>
			match mount::ops_of(fs) {
				Some(ops) => Ok((open_file_idx, open_file, node, ops)),
				None => Err(ENOSYS),
			}
		FdTarget::Console | FdTarget::Socket(_) => Err(not_file_err),
	}
}

// ---------------------------------------------------------------
//  初始化与载荷注入
// ---------------------------------------------------------------

/// 安装三项控制台描述到 fd 0/1/2: fd 0 为标准输入, 只可读; fd 1 与 2 为标准输出与标准
/// 错误, 只可写。
///
/// 幂等: 只写入空闲槽位, 已在使用的编号不被改写。文件系统的根目录由模块的入口建立,
/// 故本函数不依赖模块是否已取入。
pub fn vfs_init() {
	use crate::ext_mod::vfs_ops::{O_RDONLY, O_WRONLY};

	let mut tables = FDS.lock();
	for fd in 0..CONSOLE_FDS {
		if fd_slot(&tables, fd).is_some() {
			continue;
		}
		let flags = if fd == 0 { O_RDONLY } else { O_WRONLY };
		install_fd(&mut tables, fd, FdTarget::Console, flags, 0);
	}
}

/// 把 *data* 作为文件 *path* 注入文件系统, 沿路目录按需创建。
/// 供启动阶段 (载荷运行前) 用宿主下发的数据预置文件; 路径不属于已登记的挂载点, 或该
/// 文件系统尚未交付时返回假。
pub fn vfs_inject_file(path: &[u8], data: &[u8]) -> bool {
	let Some((_, ops, sub)) = mount::resolve(path) else {
		return false;
	};
	unsafe { (ops.inject_file)(sub.as_ptr(), sub.len() as u64, data.as_ptr(), data.len() as u64) }
}
