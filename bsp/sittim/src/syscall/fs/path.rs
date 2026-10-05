//! 路径参数的读取与路径类系统调用: mkdirat/unlinkat/faccessat/getcwd。
//!
//! 路径参数一律经 [read_path] 读入本层的临时缓冲, 再连同长度交给文件系统模块: 模块不访问
//! 载荷的地址空间, 故指针的有效性与路径的长度都在本层判定。

use crate::ext_mod::vfs_ops::{AT_REMOVEDIR, VfsStat};

use super::consts::{
	ACCESS_FLAG_BITS, ACCESS_MODE_BITS, AT_EMPTY_PATH, AT_FDCWD, PATH_MAX,
};
use super::mount;
use super::super::fdtable::FDS;
use super::super::{
	EBADF, EFAULT, EINVAL, ENAMETOOLONG, ENOENT, ENOSYS, ENOTDIR, ERANGE,
};
use super::{fd_node, vfs_errno};

/// 把载荷给出的路径参数读入 *out*, 返回路径的字节数 (不含结尾空字节)。
///
/// 相对路径要求 dirfd 取 [`AT_FDCWD`] (musl 恒以 AT_FDCWD 或绝对路径调用); 指针取空
/// 返回 EFAULT, 非空的相对路径配非 AT_FDCWD 的 dirfd 返回 ENOTDIR, 缓冲区装满而未见到
/// 结尾空字节返回 ENAMETOOLONG (该情形下路径被截断, 截断后的前缀可能指代另一个对象)。
/// 空路径照常返回, 它是否指代对象由调用方按各自的约定判定。
pub(super) unsafe fn read_path(
	pathname: *const u8, dirfd: u64, out: &mut [u8; PATH_MAX],
) -> Result<usize, u64> {
	if pathname.is_null() {
		return Err(EFAULT);
	}
	let mut n = 0;
	let mut is_terminated = false;
	while n < out.len() {
		let b = unsafe { pathname.add(n).read_volatile() };
		out[n] = b;
		n += 1;
		if b == 0 {
			is_terminated = true;
			break;
		}
	}
	if !is_terminated {
		return Err(ENAMETOOLONG);
	}
	// 结尾空字节不计入长度
	let n = n - 1;
	if (dirfd as i64) != AT_FDCWD && n > 0 && !out[..n].starts_with(b"/") {
		return Err(ENOTDIR);
	}
	Ok(n)
}

/// getcwd(17): 返回当前工作目录 (单根, 恒为 "/")。
///
/// 返回值是写入的字节数, 含结尾空字节, 即根目录的两个字节; *size* 放不下这两个字节时
/// 返回 ERANGE, *buf* 取空返回 EFAULT。
pub fn getcwd_handler(buf: u64, size: u64) -> u64 {
	if buf == 0 {
		return EFAULT;
	}
	if size < 2 {
		return ERANGE;
	}
	unsafe {
		(buf as *mut u8).write_volatile(b'/');
		(buf as *mut u8).add(1).write_volatile(0u8);
	}
	2
}

/// mkdirat(34): 创建目录。
pub fn mkdirat_handler(dirfd: u64, pathname: u64, mode: u64) -> u64 {
	let mut path_buf = [0u8; PATH_MAX];
	let plen = match unsafe { read_path(pathname as *const u8, dirfd, &mut path_buf) } {
		Ok(n) => n,
		Err(e) => {
			return e;
		}
	};
	let path = &path_buf[..plen];
	let Some((_, ops, sub)) = mount::resolve(path) else {
		return ENOSYS;
	};
	vfs_errno(unsafe { (ops.mkdir)(sub.as_ptr(), sub.len() as u64, (mode & 0o777) as u32) })
}

/// unlinkat(35): 删除文件或空目录。*flags* 含 [`AT_REMOVEDIR`] 时目标须是目录, 不含时目标须是
/// 文件。*flags* 只接受 [`AT_REMOVEDIR`] 一位, 出现其余位返回 EINVAL; 校验通过的标志位转交
/// 模块。
pub fn unlinkat_handler(dirfd: u64, pathname: u64, flags: u64) -> u64 {
	if flags & !(AT_REMOVEDIR as u64) != 0 {
		return EINVAL;
	}
	let mut path_buf = [0u8; PATH_MAX];
	let plen = match unsafe { read_path(pathname as *const u8, dirfd, &mut path_buf) } {
		Ok(n) => n,
		Err(e) => {
			return e;
		}
	};
	let path = &path_buf[..plen];
	let Some((_, ops, sub)) = mount::resolve(path) else {
		return ENOSYS;
	};
	vfs_errno(unsafe { (ops.unlink)(sub.as_ptr(), sub.len() as u64, flags as u32) })
}

/// faccessat(48): 检查路径存在性 (R_OK/W_OK 简化: 文件恒可读, 写检查恒放行)。
///
/// *mode* 只接受 [`ACCESS_MODE_BITS`] 覆盖的位, *flags* 只接受 [`ACCESS_FLAG_BITS`]
/// 覆盖的位, 出现其余位返回 EINVAL。*flags* 含 [`AT_EMPTY_PATH`] 而路径取空时, 目标取
/// *dirfd* 自身绑定的节点, 见 [empty_path_exists]; *dirfd* 无效或绑定的不是文件系统节点时
/// 返回 EBADF。路径经 [mount] 分派给它所属的文件系统; 该文件系统尚未交付时返回 ENOSYS。
pub fn faccessat_handler(dirfd: u64, pathname: u64, mode: u64, flags: u64) -> u64 {
	if mode & !ACCESS_MODE_BITS != 0 || flags & !ACCESS_FLAG_BITS != 0 {
		return EINVAL;
	}
	let mut path_buf = [0u8; PATH_MAX];
	let plen = match unsafe { read_path(pathname as *const u8, dirfd, &mut path_buf) } {
		Ok(n) => n,
		Err(e) => {
			return e;
		}
	};
	if plen == 0 && (flags & AT_EMPTY_PATH) != 0 {
		return empty_path_exists(dirfd);
	}
	let path = &path_buf[..plen];
	let Some((_, ops, sub)) = mount::resolve(path) else {
		return ENOSYS;
	};
	vfs_errno(unsafe { (ops.access)(sub.as_ptr(), sub.len() as u64) })
}

/// 检查描述符 *dirfd* 绑定的节点是否存在。描述符无效或绑定的不是文件系统节点时返回
/// EBADF, 该节点所属的文件系统尚未交付时返回 ENOSYS。
fn empty_path_exists(dirfd: u64) -> u64 {
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
	// 按节点的存在性判定: 操作表没有按下标检查的入口, stat 是按下标探测的入口
	let mut vstat = VfsStat::zero();
	if unsafe { (ops.stat)(node, &mut vstat) } {
		0
	} else {
		ENOENT
	}
}
