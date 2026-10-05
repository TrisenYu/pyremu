//! 内核伪文件系统 sysfs。
//!
//! 对应 Linux 的 `fs/sysfs`: 由内核自身提供、在 `/sys` 挂载的文件系统, 与宿主交付的根
//! 文件系统是并列的两棵目录树, 内容不来自任何交付物。目录树在启动阶段一次建成, 其后的
//! 装载与卸载都不改变它; 节点只增不改, 故整个文件系统只读。
//!
//! 本模块以 [VfsOps] 操作表交付目录树, 与文件系统模块交付根目录树的形式相同; 挂载层
//! 据此把 `/sys` 之下的路径交给本模块, 见 [super::mount]。

use crate::ext_mod::vfs_ops::{
	O_CREAT, VFS_KIND_DIR, VfsOps, VfsStat, VfsStatus, mode_writable,
};
use crate::sync_aux::SpinLock;

/// 节点表容量。
const NODES_MAX: usize = 64;
/// 单个目录名的字节数上限。
const NAME_MAX: usize = 32;
/// 根节点的下标。
const ROOT: i16 = 0;
/// 无父节点, 用于根节点。
const NO_PARENT: i16 = -1;

/// 目录项的定长部分, 即 d_name 之前的字节数; 其后是名字与结尾空字节。
const DIRENT_HEAD: usize = 19;
/// 目录项的类型编码, 取自 Linux 的 DT_* 取值。
const DT_DIR: u8 = 4;
/// 目录的权限位。
const DIR_MODE: u32 = 0o755;

/// 目录节点: 父节点下标与名字。本文件系统只登记目录, 故没有类别字段。
#[derive(Clone, Copy)]
struct Node {
	parent: i16,
	name_len: u8,
	name: [u8; NAME_MAX],
}

/// 节点表。下标加一即该节点的 ino, 见 [ino_of]。
static NODES: SpinLock<[Option<Node>; NODES_MAX]> = SpinLock::new([None; NODES_MAX]);

// ---------------------------------------------------------------
//  目录树的建立
// ---------------------------------------------------------------

/// 建立一个节点。*parent* 取 [`NO_PARENT`] 时是根节点。名字过长或表满时返回 None。
fn add_node(parent: i16, name: &[u8]) -> Option<i16> {
	if name.len() > NAME_MAX {
		return None;
	}
	let mut nodes = NODES.lock();
	let slot = nodes.iter().position(|node| node.is_none())?;
	let mut buf = [0u8; NAME_MAX];
	buf[..name.len()].copy_from_slice(name);
	nodes[slot] = Some(Node {
		parent,
		name_len: name.len() as u8,
		name: buf,
	});
	Some(slot as i16)
}

/// 建立 sysfs 的目录树。可重复调用: 节点表先清空再登记。
///
/// 根节点对应挂载点 `/sys`; `devices` 与 `system` 两级属设备模型, `cpu` 一级与其下的
/// 处理器属处理器子系统。
pub fn init() {
	*NODES.lock() = [None; NODES_MAX];
	if add_node(NO_PARENT, b"").is_none() {
		return;
	}
	let Some(devices) = add_node(ROOT, b"devices") else {
		return;
	};
	let Some(system) = add_node(devices, b"system") else {
		return;
	};
	let Some(cpu) = add_node(system, b"cpu") else {
		return;
	};
	// 本运行时运行在一颗 hart 上, 故只登记一项。各处理器的缓存目录不登记: 平台的
	// 设备树没有给出缓存属性, 登记出来会是编造的内容。
	add_node(cpu, b"cpu0");
}

// ---------------------------------------------------------------
//  目录树的读取
// ---------------------------------------------------------------

/// 取节点 *node* 的副本; 下标无效时返回 None。
fn node_of(node: i16) -> Option<Node> {
	let index = usize::try_from(node).ok()?;
	let nodes = NODES.lock();
	*nodes.get(index)?
}

/// 取节点 *node* 的 ino。
fn ino_of(node: i16) -> u64 {
	node as u64 + 1
}

/// 取节点 *node* 的父节点。根节点没有父节点, 取自身, 与 ".." 在根目录处的约定一致。
fn parent_of(node: i16) -> i16 {
	match node_of(node) {
		Some(item) if item.parent != NO_PARENT => item.parent,
		_ => node,
	}
}

/// 取节点 *node* 之下名为 *name* 的子节点。
fn child_of(node: i16, name: &[u8]) -> Option<i16> {
	let nodes = NODES.lock();
	nodes.iter().enumerate().find_map(|(index, item)| {
		let item = item.as_ref()?;
		let name_len = item.name_len as usize;
		(item.parent == node && name_len == name.len() && item.name[..name_len] == *name)
			.then_some(index as i16)
	})
}

/// 按路径取节点下标。路径以 '/' 分隔, 分量按名字匹配, "." 不移动, ".." 上移一级; 空路径
/// 与 "/" 取根节点。任一分量没有命中时返回 None。
fn lookup(path: &[u8]) -> Option<i16> {
	let mut node = ROOT;
	for part in path.split(|byte| *byte == b'/') {
		if part.is_empty() || part == b"." {
			continue;
		}
		node = if part == b".." {
			parent_of(node)
		} else {
			child_of(node, part)?
		};
	}
	Some(node)
}

/// 取目录 *dir* 的第 *at* 个目录项: 前两项是 "." 与 "..", 其后依次是各子节点。项名写入
/// *name*, 返回该项的 ino 与名字长度; 下标超出目录项数时返回 None。
fn entry_at(dir: i16, at: u64, name: &mut [u8; NAME_MAX]) -> Option<(u64, usize)> {
	if at == 0 {
		name[0] = b'.';
		return Some((ino_of(dir), 1));
	}
	if at == 1 {
		name[0] = b'.';
		name[1] = b'.';
		return Some((ino_of(parent_of(dir)), 2));
	}
	let index = usize::try_from(at - 2).ok()?;
	let nodes = NODES.lock();
	let (child, item) = nodes
		.iter()
		.enumerate()
		.filter(|(_, item)| item.as_ref().is_some_and(|item| item.parent == dir))
		.nth(index)?;
	let item = item.as_ref()?;
	let name_len = item.name_len as usize;
	name[..name_len].copy_from_slice(&item.name[..name_len]);
	Some((ino_of(child as i16), name_len))
}

// ---------------------------------------------------------------
//  操作表的各项
// ---------------------------------------------------------------

/// 把 (指针, 长度) 形式的字节区间取成切片, 只在本次调用期间使用。指针取空或长度取 0 时
/// 给出空切片。
unsafe fn bytes_at<'a>(ptr: *const u8, len: u64) -> &'a [u8] {
	if ptr.is_null() || len == 0 {
		return &[];
	}
	unsafe { core::slice::from_raw_parts(ptr, len as usize) }
}

unsafe extern "C" fn sysfs_init() -> bool {
	init();
	true
}

unsafe extern "C" fn sysfs_inject_file(
	_path: *const u8,
	_path_len: u64,
	_data: *const u8,
	_data_len: u64,
) -> bool {
	// 目录树由本文件系统自己建成, 不接受注入
	false
}

unsafe extern "C" fn sysfs_open(
	path: *const u8,
	path_len: u64,
	flags: u32,
	_mode: u32,
	out_node: *mut i16,
) -> VfsStatus {
	let Some(node) = lookup(unsafe { bytes_at(path, path_len) }) else {
		return VfsStatus::NoEntry;
	};
	if mode_writable(flags) {
		return VfsStatus::IsDir;
	}
	if (flags & O_CREAT) != 0 {
		// 只读文件系统, 不建立节点
		return VfsStatus::Invalid;
	}
	if out_node.is_null() {
		return VfsStatus::Invalid;
	}
	unsafe {
		out_node.write(node);
	}
	VfsStatus::Ok
}

unsafe extern "C" fn sysfs_resolve(
	path: *const u8,
	path_len: u64,
	out_node: *mut i16,
) -> VfsStatus {
	let Some(node) = lookup(unsafe { bytes_at(path, path_len) }) else {
		return VfsStatus::NoEntry;
	};
	if out_node.is_null() {
		return VfsStatus::Invalid;
	}
	unsafe {
		out_node.write(node);
	}
	VfsStatus::Ok
}

unsafe extern "C" fn sysfs_read(
	node: i16,
	_offset: u64,
	_buf: *mut u8,
	_len: u64,
	_out_read: *mut u64,
) -> VfsStatus {
	if node_of(node).is_none() {
		return VfsStatus::Invalid;
	}
	// 本文件系统的全部节点都是目录, 没有可读取的文件数据
	VfsStatus::IsDir
}

unsafe extern "C" fn sysfs_write(
	node: i16,
	_offset: u64,
	_buf: *const u8,
	_len: u64,
	_out_written: *mut u64,
) -> VfsStatus {
	if node_of(node).is_none() {
		return VfsStatus::Invalid;
	}
	// 本文件系统的全部节点都是目录, 没有可写入的文件数据
	VfsStatus::IsDir
}

unsafe extern "C" fn sysfs_stat(node: i16, out: *mut VfsStat) -> bool {
	if out.is_null() || node_of(node).is_none() {
		return false;
	}
	let mut vstat = VfsStat::zero();
	vstat.ino = ino_of(node);
	vstat.kind = VFS_KIND_DIR;
	vstat.mode = DIR_MODE;
	// "." 与 ".." 两项都指向本节点
	vstat.nlink = 2;
	unsafe {
		out.write(vstat);
	}
	true
}

unsafe extern "C" fn sysfs_truncate(_node: i16, _len: u64) -> VfsStatus {
	// 只读文件系统
	VfsStatus::Invalid
}

unsafe extern "C" fn sysfs_getdents(
	node: i16,
	pos: *mut u64,
	buf: *mut u8,
	count: u64,
	out_written: *mut u64,
) -> VfsStatus {
	if pos.is_null() || out_written.is_null() || (count != 0 && buf.is_null()) {
		return VfsStatus::Invalid;
	}
	if node_of(node).is_none() {
		return VfsStatus::Invalid;
	}
	let mut at = unsafe { *pos };
	let mut written: usize = 0;
	loop {
		let mut name_buf = [0u8; NAME_MAX];
		let Some((ino, name_len)) = entry_at(node, at, &mut name_buf) else {
			break;
		};
		let reclen = (DIRENT_HEAD + name_len + 1 + 7) & !7;
		if written + reclen > count as usize {
			// 缓冲区容纳不下当前条目: 已经写入的条目照常返回, 一个条目都写不进去时
			// 返回 Invalid, 使调用方能够区分目录已读完与缓冲区过小。
			if written == 0 {
				return VfsStatus::Invalid;
			}
			break;
		}
		unsafe {
			let entry = buf.add(written);
			(entry as *mut u64).write_volatile(ino);
			(entry.add(8) as *mut i64).write_volatile((at + 1) as i64);
			(entry.add(16) as *mut u16).write_volatile(reclen as u16);
			(entry.add(18) as *mut u8).write_volatile(DT_DIR);
			for i in 0..name_len {
				entry.add(DIRENT_HEAD + i).write_volatile(name_buf[i]);
			}
			entry.add(DIRENT_HEAD + name_len).write_volatile(0u8);
		}
		written += reclen;
		at += 1;
	}
	unsafe {
		*pos = at;
		*out_written = written as u64;
	}
	VfsStatus::Ok
}

unsafe extern "C" fn sysfs_mkdir(_path: *const u8, _path_len: u64, _mode: u32) -> VfsStatus {
	// 只读文件系统
	VfsStatus::Invalid
}

unsafe extern "C" fn sysfs_unlink(_path: *const u8, _path_len: u64, _flags: u32) -> VfsStatus {
	// 只读文件系统
	VfsStatus::Invalid
}

unsafe extern "C" fn sysfs_access(path: *const u8, path_len: u64) -> VfsStatus {
	match lookup(unsafe { bytes_at(path, path_len) }) {
		Some(_) => VfsStatus::Ok,
		None => VfsStatus::NoEntry,
	}
}

/// sysfs 的操作表。表是静态量, 其中的函数指针在启动阶段由重定位修正。
static OPS: VfsOps = VfsOps {
	init: sysfs_init,
	inject_file: sysfs_inject_file,
	open: sysfs_open,
	resolve: sysfs_resolve,
	read: sysfs_read,
	write: sysfs_write,
	stat: sysfs_stat,
	truncate: sysfs_truncate,
	getdents: sysfs_getdents,
	mkdir: sysfs_mkdir,
	unlink: sysfs_unlink,
	access: sysfs_access,
};

/// 取 sysfs 的操作表。
pub fn ops() -> &'static VfsOps {
	&OPS
}
