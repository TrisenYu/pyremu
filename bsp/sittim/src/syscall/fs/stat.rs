//! musl riscv64 `struct stat` 的布局与填充, 以及 stat 类系统调用: fstat/fstatat。
//!
//! 布局属载荷侧的 ABI, 只在 [Stat] 一处声明; 模块交出的字段经 [fill_stat_from_vfs] 折算
//! 到该布局, 其中文件类型的编码属 ABI 而非模块的表示, 由本层按节点类别补齐。

use core::ptr;

use crate::ext_mod::vfs_ops::{VFS_KIND_DIR, VfsStat, VfsStatus};

use super::consts::{
	AT_EMPTY_PATH, NIL, PATH_MAX, S_IFCHR, S_IFDIR, S_IFREG, S_IFSOCK, STAT_FLAG_BITS,
};
use super::mount;
use super::super::fdtable::{FDS, FdTarget, fd_target};
use super::super::types::Timespec;
use super::super::{EBADF, EFAULT, EINVAL, ENOENT, ENOSYS};
use super::{fd_node, read_path, vfs_errno};

// ---------------------------------------------------------------
//  musl riscv64 `struct stat` 布局 (128 字节, repr(C))
// ---------------------------------------------------------------

#[repr(C)]
#[derive(Clone, Copy)]
struct Stat {
	st_dev: u64,
	st_ino: u64,
	st_mode: u32,
	st_nlink: u32,
	st_uid: u32,
	st_gid: u32,
	st_rdev: u64,
	__pad: u64,
	st_size: i64,
	st_blksize: i32,
	__pad2: i32,
	st_blocks: i64,
	st_atim: Timespec,
	st_mtim: Timespec,
	st_ctim: Timespec,
	__unused: [u32; 2],
}

impl Stat {
	/// 构建一个字段全零的 stat, 仅按需回填非零字段。
	const fn zero() -> Self {
		Self {
			st_dev: 0,
			st_ino: 0,
			st_mode: 0,
			st_nlink: 0,
			st_uid: 0,
			st_gid: 0,
			st_rdev: 0,
			__pad: 0,
			st_size: 0,
			st_blksize: 4096,
			__pad2: 0,
			st_blocks: 0,
			st_atim: Timespec { tv_sec: 0, tv_nsec: 0 },
			st_mtim: Timespec { tv_sec: 0, tv_nsec: 0 },
			st_ctim: Timespec { tv_sec: 0, tv_nsec: 0 },
			__unused: [0; 2],
		}
	}
}

/// 按模块给出的 stat 字段填充 musl 的 stat。文件类型的编码属于载荷侧的 ABI, 由本层按
/// 节点类别补齐。
fn fill_stat_from_vfs(vstat: &VfsStat, st: &mut Stat) {
	st.st_ino = vstat.ino;
	let kind_bits = if vstat.kind == VFS_KIND_DIR { S_IFDIR } else { S_IFREG };
	st.st_mode = kind_bits | (vstat.mode & 0o777);
	st.st_nlink = vstat.nlink;
	st.st_size = vstat.size;
	st.st_blocks = vstat.blocks;
}

/// 填充控制台 (字符设备) stat。
fn fill_stat_console(st: &mut Stat) {
	st.st_mode = S_IFCHR | 0o600;
	st.st_nlink = 1;
	st.st_size = 0;
	st.st_blocks = 0;
}

/// 填充套接字 stat。
fn fill_stat_socket(st: &mut Stat) {
	st.st_mode = S_IFSOCK | 0o777;
	st.st_nlink = 1;
	st.st_size = 0;
	st.st_blocks = 0;
}

/// fstat(80): 按 fd 填充 stat。
pub fn fstat_handler(fd: u64, stat_ptr: u64) -> u64 {
	let stat_ptr = stat_ptr as *mut Stat;
	if stat_ptr.is_null() {
		return EFAULT;
	}
	let mut st = Stat::zero();
	let tables = FDS.lock();
	match fd_target(&tables, fd) {
		Some(FdTarget::Console) => fill_stat_console(&mut st),
		Some(FdTarget::Vfs(fs, node)) => {
			let Some(ops) = mount::ops_of(fs) else {
				return ENOSYS;
			};
			let mut vstat = VfsStat::zero();
			if !(unsafe { (ops.stat)(node, &mut vstat) }) {
				return ENOENT;
			}
			fill_stat_from_vfs(&vstat, &mut st);
		}
		Some(FdTarget::Socket(_)) => fill_stat_socket(&mut st),
		None => {
			return EBADF;
		}
	}
	unsafe {
		ptr::write_volatile(stat_ptr, st);
	}
	0
}

/// fstatat(79): 按路径填充 stat。musl 的 stat/lstat 均落到此调用。
///
/// *flags* 只接受 [`STAT_FLAG_BITS`] 覆盖的位, 出现其余位返回 EINVAL。*flags* 含
/// [`AT_EMPTY_PATH`] 而路径取空时, 目标取 *dirfd* 自身绑定的节点; *dirfd* 无效或绑定的
/// 不是文件系统节点时返回 EBADF。路径经 [mount] 分派给它所属的文件系统; 该文件系统尚未
/// 交付时返回 ENOSYS。
pub fn fstatat_handler(dirfd: u64, pathname: u64, stat_ptr: u64, flags: u64) -> u64 {
	if flags & !STAT_FLAG_BITS != 0 {
		return EINVAL;
	}
	let stat_ptr = stat_ptr as *mut Stat;
	if stat_ptr.is_null() {
		return EFAULT;
	}
	let mut path_buf = [0u8; PATH_MAX];
	let plen = match unsafe { read_path(pathname as *const u8, dirfd, &mut path_buf) } {
		Ok(n) => n,
		Err(e) => {
			return e;
		}
	};
	// 目标节点与它所属文件系统的操作表
	let (ops, node) = if plen == 0 && (flags & AT_EMPTY_PATH) != 0 {
		let (fs, node) = {
			let tables = FDS.lock();
			let Some(found) = fd_node(&tables, dirfd) else {
				return EBADF;
			};
			found
		};
		let Some(ops) = mount::ops_of(fs) else {
			return ENOSYS;
		};
		(ops, node)
	} else {
		let path = &path_buf[..plen];
		let Some((_, ops, sub)) = mount::resolve(path) else {
			return ENOSYS;
		};
		let mut node: i16 = NIL;
		let status = unsafe { (ops.resolve)(sub.as_ptr(), sub.len() as u64, &mut node) };
		if status != VfsStatus::Ok {
			return vfs_errno(status);
		}
		(ops, node)
	};
	let mut vstat = VfsStat::zero();
	if !(unsafe { (ops.stat)(node, &mut vstat) }) {
		return ENOENT;
	}
	let mut st = Stat::zero();
	fill_stat_from_vfs(&vstat, &mut st);
	unsafe {
		ptr::write_volatile(stat_ptr, st);
	}
	0
}
