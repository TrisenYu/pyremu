//! 读写与偏移类系统调用: read/write/readv/writev/pread64/pwrite64/lseek/getdents/
//! ftruncate。
//!
//! 每个入口先按描述绑定的对象分流: 控制台走 [console_write_bytes] 与 [read_stdin], 套接字
//! 转交 [net](super::super::net), 文件系统节点经模块的操作表调用。前两者在取描述符表的锁
//! 之前完成, 因为该路径可能阻塞当前线程。
//!
//! 偏移的推进、O_APPEND 的取值与 iovec 的逐段推进都在本层完成: 模块只按下标与偏移读写,
//! 不持有描述符一级的状态。

use crate::ext_mod::vfs_ops::{VfsStatus, mode_readable, mode_writable};

use super::consts::{IOV_MAX, O_APPEND, SEEK_CUR, SEEK_END, SEEK_SET};
use super::super::fdtable::{FDS, FdTarget, fd_entry};
use super::super::io::{console_write_bytes, read_stdin};
use super::super::net;
use super::super::{EBADF, EFAULT, EINVAL, ENOTDIR, EOPNOTSUPP, ESPIPE};
use super::{fd_vfs_node, node_size, vfs_errno};

/// read(63): 从 fd 读入 *len* 字节到 *buf*, 返回实际读取字节数。
/// 套接字 fd 的接收转交 [net](super::super::net), 转交在取文件系统锁之前完成, 因为该路径
/// 可能阻塞当前线程。
pub fn read_handler(fd: u64, buf: *mut u8, len: u64) -> u64 {
	match fd_entry(fd) {
		// 控制台: 只有可读方向的标准输入有输入源
		Some((FdTarget::Console, flags)) => {
			if !mode_readable(flags) {
				return EBADF;
			}
			return unsafe { read_stdin(buf, len) };
		}
		Some((FdTarget::Socket(index), flags)) => {
			return net::read_socket(index, flags, buf, len);
		}
		_ => {}
	}
	let mut tables = FDS.lock();
	let (open_file_idx, mut open_file, node, ops) = match fd_vfs_node(&tables, fd, EOPNOTSUPP) {
		Ok(v) => v,
		Err(e) => {
			return e;
		}
	};
	if !mode_readable(open_file.flags) {
		return EBADF;
	}
	let mut n: u64 = 0;
	let status = unsafe { (ops.read)(node, open_file.offset, buf, len, &mut n) };
	if status != VfsStatus::Ok {
		return vfs_errno(status);
	}
	open_file.offset += n;
	super::super::fdtable::store_open_file(&mut tables, open_file_idx, open_file);
	n
}

/// write(64): 向 fd 写出 *len* 字节, 返回实际写入字节数。
/// 套接字 fd 的发送转交 [net](super::super::net), 转交在取文件系统锁之前完成。
pub fn write_handler(fd: u64, buf: *const u8, len: u64) -> u64 {
	match fd_entry(fd) {
		// 控制台: 只有可写方向的标准输出与标准错误有输出端
		Some((FdTarget::Console, flags)) => {
			if !mode_writable(flags) {
				return EBADF;
			}
			return unsafe { console_write_bytes(buf, len) };
		}
		Some((FdTarget::Socket(index), _)) => {
			return net::write_socket(index, buf, len);
		}
		_ => {}
	}
	let mut tables = FDS.lock();
	let (open_file_idx, mut open_file, node, ops) = match fd_vfs_node(&tables, fd, EOPNOTSUPP) {
		Ok(v) => v,
		Err(e) => {
			return e;
		}
	};
	if !mode_writable(open_file.flags) {
		return EBADF;
	}
	// O_APPEND 时写入偏移为文件末尾
	let offset = if (open_file.flags & O_APPEND) != 0 {
		match node_size(ops, node) {
			Some(size) => size,
			None => {
				return EBADF;
			}
		}
	} else {
		open_file.offset
	};
	let mut written: u64 = 0;
	let status = unsafe { (ops.write)(node, offset, buf, len, &mut written) };
	if status != VfsStatus::Ok {
		return vfs_errno(status);
	}
	open_file.offset = offset + written;
	super::super::fdtable::store_open_file(&mut tables, open_file_idx, open_file);
	written
}

/// writev(66): 汇集多个 iovec 写出, 返回总字节数。iovec 项数超过 [`IOV_MAX`] 返回
/// EINVAL。
/// 套接字 fd 的发送转交 [net](super::super::net), 转交在取文件系统锁之前完成。
pub fn writev_handler(fd: u64, io_vec_arr: u64, io_vec_size: u64) -> u64 {
	if io_vec_size > IOV_MAX {
		return EINVAL;
	}
	match fd_entry(fd) {
		Some((FdTarget::Console, flags)) => {
			if !mode_writable(flags) {
				return EBADF;
			}
			let mut total: u64 = 0;
			for i in 0..io_vec_size {
				let slot = io_vec_arr.wrapping_add(i.wrapping_mul(16));
				let base = unsafe { (slot as *const u64).read_volatile() };
				let len = unsafe { (slot.wrapping_add(8) as *const u64).read_volatile() };
				unsafe {
					console_write_bytes(base as *const u8, len);
				}
				total = total.wrapping_add(len);
			}
			return total;
		}
		Some((FdTarget::Socket(index), _)) => {
			return net::writev_socket(index, io_vec_arr, io_vec_size);
		}
		_ => {}
	}
	// 文件 fd: 逐段写
	let mut tables = FDS.lock();
	let (open_file_idx, mut open_file, node, ops) = match fd_vfs_node(&tables, fd, EOPNOTSUPP) {
		Ok(v) => v,
		Err(e) => {
			return e;
		}
	};
	if !mode_writable(open_file.flags) {
		return EBADF;
	}
	let mut offset = if (open_file.flags & O_APPEND) != 0 {
		match node_size(ops, node) {
			Some(size) => size,
			None => {
				return EBADF;
			}
		}
	} else {
		open_file.offset
	};
	let mut total: u64 = 0;
	for i in 0..io_vec_size {
		let slot = io_vec_arr.wrapping_add(i.wrapping_mul(16));
		let base = unsafe { (slot as *const u64).read_volatile() };
		let len = unsafe { (slot.wrapping_add(8) as *const u64).read_volatile() };
		let mut written: u64 = 0;
		let status = unsafe { (ops.write)(node, offset, base as *const u8, len, &mut written) };
		if status != VfsStatus::Ok {
			// 第一段即失败时给出错误码, 否则返回已写字节数
			if total == 0 {
				return vfs_errno(status);
			}
			break;
		}
		offset += written;
		total += written;
	}
	open_file.offset = offset;
	super::super::fdtable::store_open_file(&mut tables, open_file_idx, open_file);
	total
}

/// readv(65): 汇集多个 iovec 读出, 返回总字节数。iovec 项数超过 [`IOV_MAX`] 返回
/// EINVAL。
/// 套接字 fd 的接收转交 [net](super::super::net), 转交在取文件系统锁之前完成, 因为该路径
/// 可能阻塞当前线程。
pub fn readv_handler(fd: u64, io_vec_arr: u64, io_vec_size: u64) -> u64 {
	if io_vec_size > IOV_MAX {
		return EINVAL;
	}
	match fd_entry(fd) {
		Some((FdTarget::Console, flags)) => {
			if !mode_readable(flags) {
				return EBADF;
			}
			// 逐段读控制台
			let mut total: u64 = 0;
			for i in 0..io_vec_size {
				let slot = io_vec_arr.wrapping_add(i.wrapping_mul(16));
				let base = unsafe { (slot as *const u64).read_volatile() };
				let len = unsafe { (slot.wrapping_add(8) as *const u64).read_volatile() };
				let n = unsafe { read_stdin(base as *mut u8, len) };
				total += n;
				if n < len {
					break;
				}
			}
			return total;
		}
		Some((FdTarget::Socket(index), flags)) => {
			return net::readv_socket(index, flags, io_vec_arr, io_vec_size);
		}
		_ => {}
	}
	// 文件 fd: 逐段读
	let mut tables = FDS.lock();
	let (open_file_idx, mut open_file, node, ops) = match fd_vfs_node(&tables, fd, EOPNOTSUPP) {
		Ok(v) => v,
		Err(e) => {
			return e;
		}
	};
	if !mode_readable(open_file.flags) {
		return EBADF;
	}
	let mut offset = open_file.offset;
	let mut total: u64 = 0;
	for i in 0..io_vec_size {
		let slot = io_vec_arr.wrapping_add(i.wrapping_mul(16));
		let base = unsafe { (slot as *const u64).read_volatile() };
		let len = unsafe { (slot.wrapping_add(8) as *const u64).read_volatile() };
		let mut n: u64 = 0;
		let status = unsafe { (ops.read)(node, offset, base as *mut u8, len, &mut n) };
		if status != VfsStatus::Ok {
			// 第一段即失败时给出错误码, 否则返回已读字节数
			if total == 0 {
				return vfs_errno(status);
			}
			break;
		}
		if n == 0 {
			break;
		}
		offset += n;
		total += n;
	}
	open_file.offset = offset;
	super::super::fdtable::store_open_file(&mut tables, open_file_idx, open_file);
	total
}

/// lseek(62): 重定位 fd 偏移, 返回新偏移。控制台与套接字没有偏移, 对它们返回 ESPIPE;
/// 越界、空闲槽位与空闲 fd 返回 EBADF; *whence* 不是三个 SEEK 之一返回 EINVAL。
///
/// *offset* 与返回值按 Linux 的 off_t 视为有符号量, 故新偏移为负或超出 i64 的表示范围
/// 时返回 EINVAL。偏移量字段是无符号的, 换算须经 i128 进行, 直接按无符号相加会把负的
/// *offset* 折成一个很大的正偏移并写入偏移量字段。
pub fn lseek_handler(fd: u64, offset: u64, whence: u64) -> u64 {
	let mut tables = FDS.lock();
	let (open_file_idx, mut open_file, node, ops) = match fd_vfs_node(&tables, fd, ESPIPE) {
		Ok(v) => v,
		Err(e) => {
			return e;
		}
	};
	let size = match node_size(ops, node) {
		Some(size) => size,
		None => {
			return EBADF;
		}
	};
	let new_off: i128 = match whence {
		SEEK_SET => offset as i64 as i128,
		SEEK_CUR => (open_file.offset as i64 as i128) + (offset as i64 as i128),
		SEEK_END => (size as i64 as i128) + (offset as i64 as i128),
		_ => {
			return EINVAL;
		}
	};
	if new_off < 0 || new_off > (i64::MAX as i128) {
		return EINVAL;
	}
	let new_off = new_off as u64;
	open_file.offset = new_off;
	super::super::fdtable::store_open_file(&mut tables, open_file_idx, open_file);
	new_off
}

/// pread64(67): 在指定偏移读取, 不改变 fd 偏移。套接字无偏移, 对其返回 ESPIPE;
/// 打开标志的访问模式位不允许读出时返回 EBADF。
pub fn pread64_handler(fd: u64, buf: *mut u8, len: u64, offset: u64) -> u64 {
	let tables = FDS.lock();
	let (_, open_file, node, ops) = match fd_vfs_node(&tables, fd, ESPIPE) {
		Ok(v) => v,
		Err(e) => {
			return e;
		}
	};
	if !mode_readable(open_file.flags) {
		return EBADF;
	}
	let mut n: u64 = 0;
	let status = unsafe { (ops.read)(node, offset, buf, len, &mut n) };
	if status != VfsStatus::Ok {
		return vfs_errno(status);
	}
	n
}

/// pwrite64(68): 在指定偏移写入, 不改变 fd 偏移。套接字无偏移, 对其返回 ESPIPE;
/// 打开标志的访问模式位不允许写入时返回 EBADF。
pub fn pwrite64_handler(fd: u64, buf: *const u8, len: u64, offset: u64) -> u64 {
	let tables = FDS.lock();
	let (_, open_file, node, ops) = match fd_vfs_node(&tables, fd, ESPIPE) {
		Ok(v) => v,
		Err(e) => {
			return e;
		}
	};
	if !mode_writable(open_file.flags) {
		return EBADF;
	}
	let mut written: u64 = 0;
	let status = unsafe { (ops.write)(node, offset, buf, len, &mut written) };
	if status != VfsStatus::Ok {
		return vfs_errno(status);
	}
	written
}

/// getdents(61): 读取目录项到 *buf*, 返回写入字节数。套接字不是目录, 对其返回 ENOTDIR。
///
/// *buf* 取空而 *count* 非 0 时返回 EFAULT; 缓冲区容纳不下一个目录项时由模块给出
/// EINVAL, 故 *count* 为 0 同样返回 EINVAL。
pub fn getdents_handler(fd: u64, buf: *mut u8, count: u64) -> u64 {
	if buf.is_null() && count != 0 {
		return EFAULT;
	}
	let mut tables = FDS.lock();
	let (open_file_idx, mut open_file, node, ops) = match fd_vfs_node(&tables, fd, ENOTDIR) {
		Ok(v) => v,
		Err(e) => {
			return e;
		}
	};
	let mut pos = open_file.offset;
	let mut written: u64 = 0;
	let status = unsafe { (ops.getdents)(node, &mut pos, buf, count, &mut written) };
	if status != VfsStatus::Ok {
		return vfs_errno(status);
	}
	open_file.offset = pos;
	super::super::fdtable::store_open_file(&mut tables, open_file_idx, open_file);
	written
}

/// ftruncate(46): 截断文件到指定长度。控制台与套接字无长度, 对其返回 EINVAL;
/// 打开标志的访问模式位不允许写入时同样返回 EINVAL。
pub fn ftruncate_handler(fd: u64, length: u64) -> u64 {
	let tables = FDS.lock();
	let (_, open_file, node, ops) = match fd_vfs_node(&tables, fd, EINVAL) {
		Ok(v) => v,
		Err(e) => {
			return e;
		}
	};
	if !mode_writable(open_file.flags) {
		return EINVAL;
	}
	vfs_errno(unsafe { (ops.truncate)(node, length) })
}
