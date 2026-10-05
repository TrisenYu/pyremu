//! 描述符类系统调用: openat/close/dup/dup3/fcntl/fsync。
//!
//! 这些调用改动的是描述符表与打开文件描述表本身, 不读写文件数据: 复制、替换与释放都在
//! [super::super::fdtable] 的原语上完成, 文件系统只在 openat 判定一次打开是否成功。
//!
//! 描述符表与描述表的锁在 [close_handler] 与 [dup3_handler] 中显式放开再释放资源: 释放
//! 套接字描述的路径会调用协议栈, 不得在锁的持有期间进行。

use crate::ext_mod::vfs_ops::VfsStatus;

use super::consts::{
	FD_CLOEXEC,
	F_DUPFD,
	F_DUPFD_CLOEXEC,
	F_GETFD,
	F_GETFL,
	F_SETFD,
	F_SETFL,
	NIL,
	OPEN_ONLY_FLAGS,
	PATH_MAX,
	SETFL_MASK,
};
use super::mount;
use super::super::fdtable::{
	FD_FLAG_MASK,
	FDS,
	FdRelease,
	FdTarget,
	MAX_FD,
	O_CLOEXEC,
	alloc_fd,
	close_fd,
	dup_at,
	fd_open_file_idx,
	fd_slot,
	fd_slot_mut,
	fd_target,
	open_file_of,
	release_replaced,
	release_target,
	store_open_file,
	take_fd_slot,
	take_fd_slot_from,
};
use super::super::{ EBADF, EINVAL, EMFILE, ENOSYS };
use super::{ read_path, vfs_errno };

/// openat(56): dirfd + pathname + flags + mode, 返回新 fd。
///
/// 打开标志原样转交文件系统判定; 建立的描述只取其中的状态标志位, 执行时关闭位记在该描述
/// 符自身的表项上, 见 [`OPEN_ONLY_FLAGS`] 与 [`FD_FLAG_MASK`]。路径经 [super::mount] 分派
/// 给它所属的文件系统; 该文件系统尚未交付时返回 ENOSYS。
pub fn openat_handler(dirfd: u64, pathname: u64, flags: u64, mode: u64) -> u64 {
	let mut path_buf = [0u8; PATH_MAX];
	let plen = match unsafe { read_path(pathname as *const u8, dirfd, &mut path_buf) } {
		Ok(n) => n,
		Err(e) => {
			return e;
		}
	};
	let path = &path_buf[..plen];
	let Some((fs, ops, sub)) = mount::resolve(path) else {
		return ENOSYS;
	};
	let mut node: i16 = NIL;
	let status = unsafe {
		(ops.open)(
			sub.as_ptr(),
			sub.len() as u64,
			flags as u32,
			(mode & 0o777) as u32,
			&mut node
		)
	};
	if status != VfsStatus::Ok {
		return vfs_errno(status);
	}
	let mut tables = FDS.lock();
	alloc_fd(
		&mut tables,
		FdTarget::Vfs(fs, node),
		(flags as u32) & !OPEN_ONLY_FLAGS,
		(flags as u32) & FD_FLAG_MASK
	)
}

/// close(57): 释放 fd 的表项。该表项是描述的最后一条引用时, 一并释放描述绑定的资源:
/// 套接字描述释放其协议栈槽位。无效描述符返回 EBADF。
pub fn close_handler(fd: u64) -> u64 {
	let mut tables = FDS.lock();
	let released = close_fd(&mut tables, fd);
	drop(tables);
	if let FdRelease::Invalid = released {
		return EBADF;
	}
	release_target(released);
	0
}

/// dup(23): 把描述符 *old_fd* 复制到编号最小的空闲槽位, 返回新描述符。新旧描述符共用
/// 同一项描述, 故偏移与状态标志共享, 一处改动另一处可见。执行时关闭标志属描述符自身,
/// 新描述符该位为 0。源描述符无效返回 EBADF, 槽位耗尽返回 EMFILE。
pub fn dup_handler(old_fd: u64) -> u64 {
	let mut tables = FDS.lock();
	let Some((open_file_idx, open_file)) = open_file_of(&tables, old_fd) else {
		return EBADF;
	};
	let Some(new_fd) = take_fd_slot(&tables) else {
		return EMFILE;
	};
	dup_at(&mut tables, new_fd, open_file_idx, open_file, 0);
	new_fd
}

/// dup3(24): 把描述符 *old_fd* 复制到编号 *new_fd*。*flags* 只接受 O_CLOEXEC, 含其余位
/// 返回 EINVAL; 两个编号相同返回 EINVAL; *new_fd* 越界返回 EBADF; 源描述符无效返回
/// EBADF。新旧描述符共用同一项描述, 偏移与状态标志共享; *new_fd* 原已打开时其表项被
/// 替换, 被替换的描述释放一条引用, 该描述的最后一条引用即在此释放其绑定的资源。执行时
/// 关闭标志属描述符自身, 按 *flags* 给出。
pub fn dup3_handler(old_fd: u64, new_fd: u64, flags: u64) -> u64 {
	if (flags & !(O_CLOEXEC as u64)) != 0 {
		return EINVAL;
	}
	if old_fd == new_fd {
		return EINVAL;
	}
	if new_fd >= (MAX_FD as u64) {
		return EBADF;
	}
	let mut tables = FDS.lock();
	let Some((open_file_idx, open_file)) = open_file_of(&tables, old_fd) else {
		return EBADF;
	};
	// *new_fd* 原已打开时, 它所指的描述也要释放。该描述的下标须在改写表项之前取得, 改写
	// 之后该编号已指向 *old_fd* 所属的描述; 释放本身按当时的引用计数重新读取该下标, 故
	// 两个编号恰好指向同一描述时, 复制加的那条引用与释放减的那条引用互不抵消。
	let replaced = fd_open_file_idx(&tables, new_fd);
	dup_at(&mut tables, new_fd, open_file_idx, open_file, flags as u32);
	let released = release_replaced(&mut tables, replaced);
	drop(tables);
	release_target(released);
	new_fd
}

/// fcntl(25): 描述符标志查询/设置、状态标志查询/设置与描述符复制。musl 的 fopen 依赖
/// F_SETFD 成功。五个已实现的命令在 fd 越界或落在空闲槽位时均返回 EBADF; 未实现的命令
/// 返回 EINVAL。
///
/// 两组标志分属两张表: F_GETFD 与 F_SETFD 读写描述符表项的执行时关闭位; F_GETFL 回报该
/// 描述的状态标志, F_SETFL 只改动 [`SETFL_MASK`] 覆盖的位, 故 dup 得出的描述符一并可见。
/// F_DUPFD 与 F_DUPFD_CLOEXEC 复制描述本身 (引用计数加一), 只有执行时关闭位按命令给出。
pub fn fcntl_handler(fd: u64, cmd: u64, arg: u64) -> u64 {
	match cmd {
		// 描述符标志只有 FD_CLOEXEC 一位。本运行时不执行 exec, 该位无实际后果, 读回的
		// 值仍与设置的值一致。
		F_GETFD => {
			let tables = FDS.lock();
			let Some(entry) = fd_slot(&tables, fd) else {
				return EBADF;
			};
			if (entry.flags & FD_FLAG_MASK) != 0 {
				FD_CLOEXEC
			} else {
				0
			}
		}
		F_SETFD => {
			let mut tables = FDS.lock();
			let Some(entry) = fd_slot_mut(&mut tables, fd) else {
				return EBADF;
			};
			if (arg & FD_CLOEXEC) != 0 {
				entry.flags |= O_CLOEXEC;
			} else {
				entry.flags &= !O_CLOEXEC;
			}
			0
		}
		F_GETFL => {
			let tables = FDS.lock();
			match open_file_of(&tables, fd) {
				Some((_, open_file)) => open_file.flags as u64,
				None => EBADF,
			}
		}
		F_SETFL => {
			let mut tables = FDS.lock();
			let Some((open_file_idx, mut open_file)) = open_file_of(&tables, fd) else {
				return EBADF;
			};
			open_file.flags = (open_file.flags & !SETFL_MASK) | ((arg as u32) & SETFL_MASK);
			store_open_file(&mut tables, open_file_idx, open_file);
			0
		}
		F_DUPFD | F_DUPFD_CLOEXEC => {
			let mut tables = FDS.lock();
			let Some((open_file_idx, open_file)) = open_file_of(&tables, fd) else {
				return EBADF;
			};
			// arg 是编号下界, 与源描述符的编号无关: 取不小于 arg 的最小编号空闲槽位
			if arg >= (MAX_FD as u64) {
				return EINVAL;
			}
			let Some(new_fd) = take_fd_slot_from(&tables, arg) else {
				return EMFILE;
			};
			// 执行时关闭位不随复制传递: F_DUPFD 清该位, F_DUPFD_CLOEXEC 置该位
			let desc_flags = if cmd == F_DUPFD_CLOEXEC { O_CLOEXEC } else { 0 };
			dup_at(&mut tables, new_fd, open_file_idx, open_file, desc_flags);
			new_fd
		}
		_ => EINVAL,
	}
}

/// fsync(82): 无持久化后端, 已打开的描述符立即成功; 越界 fd 与空闲槽位返回 EBADF。
pub fn fsync_handler(fd: u64) -> u64 {
	let tables = FDS.lock();
	match fd_target(&tables, fd) {
		Some(_) => 0,
		None => EBADF,
	}
}
