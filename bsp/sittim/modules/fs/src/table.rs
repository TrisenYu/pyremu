//! 模块操作表的各项: 把节点层的接口按操作表的形态转接给运行时。
//!
//! 本文件由模块的入口与主机侧的测试入口共用, 两侧因此转接同一份代码, 转接过程中的
//! 参数校验与结果编码不会在两侧出现分歧。

use crate::ramfs;
use crate::vfs_ops::{ VfsOps, VfsStat, VfsStatus };

/// 建立本模块的操作表。
pub fn fill_ops() -> VfsOps {
	VfsOps {
		init: ops_init,
		inject_file: ops_inject_file,
		open: ops_open,
		resolve: ops_resolve,
		read: ops_read,
		write: ops_write,
		stat: ops_stat,
		truncate: ops_truncate,
		getdents: ops_getdents,
		mkdir: ops_mkdir,
		unlink: ops_unlink,
		access: ops_access,
	}
}

// ---------------------------------------------------------------
//  调用方缓冲区的借出
// ---------------------------------------------------------------

/// 把调用方给出的 [ptr, ptr + len) 借为字节切片。`len` 取 0 时给出空切片, 不读取 `ptr`。
///
/// 操作表的参数是裸指针而非切片: 切片跨越 `extern "C"` 边界会触发
/// `improper_ctypes_definitions`。
unsafe fn bytes_at<'a>(ptr: *const u8, len: u64) -> &'a [u8] {
	if len == 0 {
		&[]
	} else {
		unsafe { core::slice::from_raw_parts(ptr, len as usize) }
	}
}

// ---------------------------------------------------------------
//  操作表的各项
// ---------------------------------------------------------------

unsafe extern "C" fn ops_init() -> bool {
	ramfs::init()
}

unsafe extern "C" fn ops_inject_file(path: *const u8, path_len: u64, data: *const u8, data_len: u64) -> bool {
	let path = unsafe { bytes_at(path, path_len) };
	let data = unsafe { bytes_at(data, data_len) };
	ramfs::inject_file(path, data)
}

unsafe extern "C" fn ops_open(
	path: *const u8,
	path_len: u64,
	flags: u32,
	mode: u32,
	out_node: *mut i16
) -> VfsStatus {
	let path = unsafe { bytes_at(path, path_len) };
	match ramfs::open(path, flags, mode) {
		Ok(node) => {
			if !out_node.is_null() {
				unsafe {
					out_node.write(node);
				}
			}
			VfsStatus::Ok
		}
		Err(status) => status,
	}
}

unsafe extern "C" fn ops_resolve(path: *const u8, path_len: u64, out_node: *mut i16) -> VfsStatus {
	let path = unsafe { bytes_at(path, path_len) };
	match ramfs::resolve_path(path) {
		Ok(node) => {
			if !out_node.is_null() {
				unsafe {
					out_node.write(node);
				}
			}
			VfsStatus::Ok
		}
		Err(status) => status,
	}
}

unsafe extern "C" fn ops_read(
	node: i16,
	offset: u64,
	buf: *mut u8,
	len: u64,
	out_read: *mut u64
) -> VfsStatus {
	if out_read.is_null() || (len != 0 && buf.is_null()) {
		return VfsStatus::Invalid;
	}
	match ramfs::read(node, offset, buf, len) {
		Ok(read) => {
			unsafe {
				out_read.write(read);
			}
			VfsStatus::Ok
		}
		Err(status) => status,
	}
}

unsafe extern "C" fn ops_write(
	node: i16,
	offset: u64,
	buf: *const u8,
	len: u64,
	out_written: *mut u64
) -> VfsStatus {
	if len != 0 && buf.is_null() {
		return VfsStatus::Invalid;
	}
	match ramfs::write(node, offset, buf, len) {
		Ok(written) => {
			if !out_written.is_null() {
				unsafe {
					out_written.write(written);
				}
			}
			VfsStatus::Ok
		}
		Err(status) => status,
	}
}

unsafe extern "C" fn ops_stat(node: i16, out: *mut VfsStat) -> bool {
	if out.is_null() {
		return false;
	}
	ramfs::stat(node, unsafe { &mut *out })
}

unsafe extern "C" fn ops_truncate(node: i16, len: u64) -> VfsStatus {
	ramfs::truncate(node, len)
}

unsafe extern "C" fn ops_getdents(
	node: i16,
	pos: *mut u64,
	buf: *mut u8,
	count: u64,
	out_written: *mut u64
) -> VfsStatus {
	if pos.is_null() || out_written.is_null() || (count != 0 && buf.is_null()) {
		return VfsStatus::Invalid;
	}
	match ramfs::getdents(node, unsafe { &mut *pos }, buf, count) {
		Ok(written) => {
			unsafe {
				out_written.write(written);
			}
			VfsStatus::Ok
		}
		Err(status) => status,
	}
}

unsafe extern "C" fn ops_mkdir(path: *const u8, path_len: u64, mode: u32) -> VfsStatus {
	let path = unsafe { bytes_at(path, path_len) };
	ramfs::mkdir(path, mode)
}

unsafe extern "C" fn ops_unlink(path: *const u8, path_len: u64, flags: u32) -> VfsStatus {
	let path = unsafe { bytes_at(path, path_len) };
	ramfs::unlink(path, flags)
}

unsafe extern "C" fn ops_access(path: *const u8, path_len: u64) -> VfsStatus {
	let path = unsafe { bytes_at(path, path_len) };
	ramfs::access(path)
}
