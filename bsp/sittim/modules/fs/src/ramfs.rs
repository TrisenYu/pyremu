//! ramfs 的节点层: 目录树、文件数据区、路径解析与按节点的文件与目录语义。
//!
//! 移植自 ref-impl/emod/emod_vfs 的 Prex VFS。裁剪其 vnode/dentry/mount 三层抽象为
//! 单一的 ramfs 节点树, 保留核心语义: 路径查找、目录树、读写、截断、创建/删除与目录
//! 遍历。
//!
//! 模块为 no_std 且无全局分配器, 存储改为静态定容: 节点存于定长数组, 文件数据存于文件
//! 数据区, 区的偏移只前进, 不回收。两者都落在 BSS 段, 在挂起与恢复之间保留。
//!
//! 本层不涉及 fd: fd 表由运行时持有, 调用方按节点下标寻址。节点的存储形态只在本层内,
//! 下标对运行时是不透明的。

use core::cmp;

use crate::sync_aux::SpinLock;
use crate::vfs_ops::{
	AT_REMOVEDIR,
	O_CREAT,
	O_DIRECTORY,
	O_EXCL,
	O_TRUNC,
	VFS_KIND_DIR,
	VFS_KIND_FILE,
	VfsStat,
	VfsStatus,
	mode_writable,
};

// ---------------------------------------------------------------
//  容量与哨兵
// ---------------------------------------------------------------

/// 节点总数上限, 含根目录。
const MAX_NODES: usize = 128;
/// 单个节点名的最大长度。名字存于定长数组, 结尾不置空字节, 长度另记于 `name_len`。
const NAME_MAX: usize = 64;
/// 文件数据区的字节数。
const DATA_ARENA: usize = 128 * 1024;

/// 链接字段的空值, 用于 parent/first_child/next_sibling。
const NIL: i16 = -1;

/// 目录项类型: 目录。
const DT_DIR: u8 = 4;
/// 目录项类型: 普通文件。
const DT_REG: u8 = 8;

// ---------------------------------------------------------------
//  节点与全局状态
// ---------------------------------------------------------------

/// ramfs 节点。目录以 first_child 指向孩子链表, 文件以 data_off/size 指出自己的数据在
/// 文件数据区中的位置与长度。所有字段都是可静态零初始化的定长类型, 保证整个 VfsState
/// 落在 BSS 段。
#[derive(Clone, Copy)]
struct RamfsNode {
	name: [u8; NAME_MAX],
	name_len: u16,
	kind: u8,
	parent: i16,
	first_child: i16,
	next_sibling: i16,
	data_off: u32,
	size: u32,
	mode: u32,
}

impl RamfsNode {
	const fn empty() -> Self {
		Self {
			name: [0; NAME_MAX],
			name_len: 0,
			kind: VFS_KIND_FILE,
			parent: NIL,
			first_child: NIL,
			next_sibling: NIL,
			data_off: 0,
			size: 0,
			mode: 0,
		}
	}
}

/// 目录树与文件数据区。
///
/// 节点一旦分配即不回收: 下标在模块的整个生命期内保持有效, 删除只把节点从父目录的孩子
/// 链表中摘除。运行时的调用因此可以只带下标而不必附带校验。
///
/// 文件数据区中已用量之后的字节恒为 0: 该区不单独清零, 而每次写入都把已用量一并推进到
/// 写入区间的末尾。
struct VfsState {
	nodes: [RamfsNode; MAX_NODES],
	data: [u8; DATA_ARENA],
	data_used: usize,
	next_node: u16,
}

impl VfsState {
	const fn empty() -> Self {
		Self {
			nodes: [RamfsNode::empty(); MAX_NODES],
			data: [0; DATA_ARENA],
			data_used: 0,
			next_node: 0,
		}
	}
}

/// 全局目录树状态。飞地 S 模式运行在单 hart 上, 且系统调用不可重入: 中断只登记抢占, 待
/// 系统调用返回后才切换线程。加自旋锁只为与共享资源的访问约定保持一致; 锁的持有区间不含
/// 任何 ecall 与让出, 也就不会再次取用本锁。
static VFS: SpinLock<VfsState> = SpinLock::new(VfsState::empty());

// ---------------------------------------------------------------
//  工具
// ---------------------------------------------------------------

/// 节点下标是否落在已分配的槽位内。
#[inline]
fn is_node_valid(idx: i16) -> bool {
	idx >= 0 && (idx as usize) < MAX_NODES
}

/// 以节点下标 + 1 作为稳定的 inode 编号: 根目录的 ino 取 1, 节点不被回收重排, 编号不变。
#[inline]
fn ino_of(node_idx: i16) -> u64 {
	(node_idx as u64) + 1
}

/// 节点类型折算为目录项 d_type。
#[inline]
fn dtype_of(kind: u8) -> u8 {
	if kind == VFS_KIND_DIR { DT_DIR } else { DT_REG }
}

// ---------------------------------------------------------------
//  路径解析
// ---------------------------------------------------------------

/// 从根目录沿 *path* 解析: 依次取以 '/' 分隔的每一个名字, 给出命中的节点下标。取到名字 "."
/// 时停在当前目录, 取到名字 ".." 时移向父节点, 根目录的父节点取自身。
///
/// 沿途的既有节点须是目录, 取到名字时它所在的节点不是目录即返回
/// [`NotDir`](VfsStatus::NotDir)。*is_create* 为真时把缺失的名字建立为空目录, 为假时遇到
/// 缺失的名字返回 [`NoEntry`](VfsStatus::NoEntry); 名字达到 [`NAME_MAX`] 时返回
/// [`NameTooLong`](VfsStatus::NameTooLong), 节点用尽时返回 [`NoMemory`](VfsStatus::NoMemory)。
fn resolve(fs: &mut VfsState, path: &[u8], is_create: bool) -> Result<i16, VfsStatus> {
	let mut cur: i16 = 0;
	let mut i = 0usize;
	while i < path.len() {
		while i < path.len() && path[i] == b'/' {
			i += 1;
		}
		if i >= path.len() {
			break;
		}
		if fs.nodes[cur as usize].kind != VFS_KIND_DIR {
			return Err(VfsStatus::NotDir);
		}
		let mut j = i;
		while j < path.len() && path[j] != b'/' {
			j += 1;
		}
		let comp = &path[i..j];
		if comp == b"." {
			// 当前目录, 不移动
			i = j;
			continue;
		} else if comp == b".." {
			let p = fs.nodes[cur as usize].parent;
			if p != NIL {
				cur = p;
			}
			i = j;
			continue;
		}
		if comp.len() >= NAME_MAX {
			return Err(VfsStatus::NameTooLong);
		}
		let found = find_child(fs, cur, comp);
		if found != NIL {
			cur = found;
			i = j;
			continue;
		}
		if !is_create {
			return Err(VfsStatus::NoEntry);
		}
		let created = create_child(fs, cur, comp, VFS_KIND_DIR, 0o755);
		if created == NIL {
			return Err(VfsStatus::NoMemory);
		}
		cur = created;
		i = j;
	}
	Ok(cur)
}

/// 把 *path* 切分为最后一个名字所在目录的路径与最后一个名字。
///
/// 切分不含末尾的连续 '/', 它们只要求最后一个名字指代目录。给出两个返回值的三种情形:
/// 得到所在目录与这个名字; 名字是 "." 与 ".." 时得到整条路径与该名字, 它们只指代目录;
/// 路径以 '/' 结尾, 或整个路径只由 '/' 组成时得到整条路径与空名字, 该路径自己即目录。
///
/// 空路径不指代任何对象, 返回 [`NoEntry`](VfsStatus::NoEntry); 名字达到 [`NAME_MAX`] 时
/// 返回 [`NameTooLong`](VfsStatus::NameTooLong)。
///
/// 本函数只切分字节串, 既不查目录树, 也不建立节点。
fn split_path(path: &[u8]) -> Result<(&[u8], &[u8]), VfsStatus> {
	if path.is_empty() {
		return Err(VfsStatus::NoEntry);
	}
	let mut end = path.len();
	while end > 0 && path[end - 1] == b'/' {
		end -= 1;
	}
	if end == 0 {
		// 整个路径只由 '/' 组成
		return Ok((&b"/"[..], &[]));
	}
	let head = &path[..end];
	let slash = head.iter().rposition(|&c| c == b'/');
	let name = match slash {
		None => head,
		Some(at) => &head[at + 1..],
	};
	if name == b"." || name == b".." {
		// "." 与 ".." 只指代目录, 整条路径解析的结果就是该目录
		return Ok((head, name));
	}
	if name.len() >= NAME_MAX {
		return Err(VfsStatus::NameTooLong);
	}
	if end < path.len() {
		// 路径以 '/' 结尾, 最后一个名字只指代目录
		return Ok((head, &[]));
	}
	let dir_path = match slash {
		None | Some(0) => &b"/"[..],
		Some(at) => &head[..at],
	};
	Ok((dir_path, name))
}

/// 路径的最后一个名字所指向的对象。
enum PathTarget<'a> {
	/// 最后一个名字所在目录的下标, 与这个名字。名字长度不足 [`NAME_MAX`]。
	Child(i16, &'a [u8]),
	/// 最后一个名字只指代目录时给出的该目录, 与这个名字: 路径以 '/' 结尾时名字取空,
	/// 名字是 "." 与 ".." 时取该名字本身。
	Dir(i16, &'a [u8]),
}

/// 检查节点是目录: 不是目录返回 [`NotDir`](VfsStatus::NotDir)。
fn as_dir(fs: &VfsState, node: i16) -> Result<i16, VfsStatus> {
	if fs.nodes[node as usize].kind != VFS_KIND_DIR {
		return Err(VfsStatus::NotDir);
	}
	Ok(node)
}

/// 按 [`split_path`] 切分 *path*, 把其中的目录路径解析为目录节点, 给出路径最后一个名字
/// 的目标。
///
/// 路径中间的名字没有命中返回 [`NoEntry`](VfsStatus::NoEntry), 命中的节点不是目录返回
/// [`NotDir`](VfsStatus::NotDir)。
fn split_target<'a>(fs: &mut VfsState, path: &'a [u8]) -> Result<PathTarget<'a>, VfsStatus> {
	let (dir_path, name) = split_path(path)?;
	let dir = resolve(fs, dir_path, false)?;
	let dir = as_dir(fs, dir)?;
	// 名字是 "." 与 ".." 时 dir_path 是整条路径, 解析的结果即它们所指代的目录
	if name.is_empty() || name == b"." || name == b".." {
		Ok(PathTarget::Dir(dir, name))
	} else {
		Ok(PathTarget::Child(dir, name))
	}
}

/// 统计目录 *dir* 的孩子中的子目录个数。
fn count_subdirs(fs: &VfsState, dir: i16) -> u32 {
	let mut count = 0;
	let mut child = fs.nodes[dir as usize].first_child;
	while child != NIL {
		if fs.nodes[child as usize].kind == VFS_KIND_DIR {
			count += 1;
		}
		child = fs.nodes[child as usize].next_sibling;
	}
	count
}

// ---------------------------------------------------------------
//  节点操作: 节点与文件数据区的偏移都只前进, 不回收
// ---------------------------------------------------------------

/// 分配一个空节点, 返回其下标; 节点耗尽返回 NIL。
fn alloc_node(fs: &mut VfsState, kind: u8, mode: u32) -> i16 {
	let idx = fs.next_node as usize;
	if idx >= MAX_NODES {
		return NIL;
	}
	fs.next_node += 1;
	let node = &mut fs.nodes[idx];
	*node = RamfsNode::empty();
	node.kind = kind;
	node.mode = mode;
	node.parent = NIL;
	node.first_child = NIL;
	node.next_sibling = NIL;
	idx as i16
}

/// 在 *parent* 目录下创建名为 *name* 的孩子, 返回其下标; 失败返回 NIL。
fn create_child(fs: &mut VfsState, parent: i16, name: &[u8], kind: u8, mode: u32) -> i16 {
	let child = alloc_node(fs, kind, mode);
	if child == NIL {
		return NIL;
	}
	// 先读出父目录孩子链表头, 避免与子节点的可变借用冲突。
	let head = fs.nodes[parent as usize].first_child;
	{
		let node = &mut fs.nodes[child as usize];
		for (i, &b) in name.iter().enumerate() {
			node.name[i] = b;
		}
		node.name_len = name.len() as u16;
		node.parent = parent;
		node.next_sibling = head;
	}
	// 插入到父目录孩子链表头部
	fs.nodes[parent as usize].first_child = child;
	child
}

/// 在目录中查找名为 *name* 的孩子, 返回其下标; 未命中返回 NIL。
fn find_child(fs: &VfsState, dir: i16, name: &[u8]) -> i16 {
	let mut child = fs.nodes[dir as usize].first_child;
	while child != NIL {
		let node = &fs.nodes[child as usize];
		if (node.name_len as usize) == name.len() && &node.name[..name.len()] == name {
			return child;
		}
		child = node.next_sibling;
	}
	NIL
}

/// 从父目录的孩子链表中移除 *target*。节点与它的文件数据都不回收, 文件数据区的偏移无法
/// 回退, 移除只使该节点不再被路径解析命中。
fn remove_child(fs: &mut VfsState, parent: i16, target: i16) {
	let mut prev = NIL;
	let mut child = fs.nodes[parent as usize].first_child;
	while child != NIL {
		if child != target {
			prev = child;
			child = fs.nodes[child as usize].next_sibling;
			continue;
		}
		if prev == NIL {
			fs.nodes[parent as usize].first_child = fs.nodes[child as usize].next_sibling;
		} else {
			fs.nodes[prev as usize].next_sibling = fs.nodes[child as usize].next_sibling;
		}
		return;
	}
}

/// 把文件节点 *idx* 截断到 *len* 字节。截断只减小长度, 不增大长度。
fn truncate_node(fs: &mut VfsState, idx: i16, len: u64) {
	let node = &mut fs.nodes[idx as usize];
	if len < (node.size as u64) {
		node.size = len as u32;
	}
}

// ---------------------------------------------------------------
//  文件数据读写
// ---------------------------------------------------------------

/// 从文件 *idx* 的 *offset* 处读取至多 *len* 字节到载荷缓冲区 *buf*,
/// 返回实际读取的字节数。
unsafe fn read_file(fs: &VfsState, idx: i16, offset: u64, buf: *mut u8, len: u64) -> u64 {
	let node = &fs.nodes[idx as usize];
	if offset >= (node.size as u64) {
		return 0;
	}
	let avail = ((node.size as u64) - offset) as usize;
	let n = cmp::min(avail, len as usize);
	let base = (node.data_off as usize) + (offset as usize);
	for i in 0..n {
		unsafe {
			buf.add(i).write_volatile(fs.data[base + i]);
		}
	}
	n as u64
}

/// 向文件 *idx* 的 *offset* 处写入 *buf*, 必要时扩大该文件: 数据已在文件数据区末尾时
/// 就地延长, 否则把原有数据拷贝到文件数据区的末尾。返回写入的字节数; 文件数据区余量不足
/// 时返回 `NoSpace`, 此时文件不变。扩大后未被 *buf* 覆盖的字节取 0。
unsafe fn write_file(
	fs: &mut VfsState,
	idx: i16,
	offset: u64,
	buf: *const u8,
	len: u64
) -> Result<u64, VfsStatus> {
	if len == 0 {
		return Ok(0);
	}
	// 偏移取自载荷, 与长度之和可能超出 usize 的表示范围
	let Some(end) = (offset as usize).checked_add(len as usize) else {
		return Err(VfsStatus::NoSpace);
	};
	let node = &fs.nodes[idx as usize];
	if end <= (node.size as usize) {
		let node = &fs.nodes[idx as usize];
		let base = node.data_off as usize;
		for i in 0..len as usize {
			let b = unsafe { buf.add(i).read_volatile() };
			fs.data[base + (offset as usize) + i] = b;
		}
		return Ok(len);
	}
	// 需要增长
	let old_off = node.data_off as usize;
	let old_size = node.size as usize;
	let in_tail = old_off + old_size == fs.data_used;
	let new_off;
	if in_tail && old_off + end <= DATA_ARENA {
		// 数据已在文件数据区末尾, 就地延长
		new_off = old_off;
		fs.data_used += end - old_size;
	} else if fs.data_used + end <= DATA_ARENA {
		// 拷贝到文件数据区末尾的新位置, 原位置留在区内, 偏移不回退
		new_off = fs.data_used;
		for i in 0..old_size {
			fs.data[new_off + i] = fs.data[old_off + i];
		}
		fs.data_used += end;
	} else {
		return Err(VfsStatus::NoSpace);
	}
	for i in 0..len as usize {
		let b = unsafe { buf.add(i).read_volatile() };
		fs.data[new_off + (offset as usize) + i] = b;
	}
	let node = &mut fs.nodes[idx as usize];
	node.data_off = new_off as u32;
	node.size = end as u32;
	Ok(len)
}

// ---------------------------------------------------------------
//  对外接口
// ---------------------------------------------------------------

/// 建立根目录。可重复调用, 已建立时同样返回真。
pub fn init() -> bool {
	let mut fs = VFS.lock();
	if fs.next_node == 0 {
		// 根目录为节点 0
		fs.next_node = 1;
		let root = &mut fs.nodes[0];
		root.kind = VFS_KIND_DIR;
		root.mode = 0o755;
		root.name[0] = b'/';
		root.name_len = 1;
	}
	true
}

/// 把 *data* 作为文件 *path* 注入, 最后一个名字所在目录的各级按需建立。供启动阶段在载荷
/// 运行前用宿主下发的数据预置文件。
///
/// 路径须以一个名字结尾, 由它指定注入的文件; 该名字只指代目录时注入失败, 即路径以 '/'
/// 结尾, 或这个名字是 "." 与 ".."。沿途的既有节点须是目录, 名字达到 [`NAME_MAX`] 或节点
/// 用尽时注入失败。该名字已存在时须是文件, 否则注入失败; 是文件时先清空再写入 *data*,
/// 不存在的名字成为新的文件。
///
/// 注入失败时, 在此之前已建立的各级目录与已清空的文件都留在目录树中, *data* 没有写入。
pub fn inject_file(path: &[u8], data: &[u8]) -> bool {
	let (dir_path, name) = match split_path(path) {
		Ok(parts) => parts,
		Err(_) => {
			return false;
		}
	};
	if name.is_empty() || name == b"." || name == b".." {
		return false;
	}
	let mut fs = VFS.lock();
	let dir = match resolve(&mut fs, dir_path, true) {
		Ok(node) => node,
		Err(_) => {
			return false;
		}
	};
	let node = match find_child(&fs, dir, name) {
		NIL => {
			let created = create_child(&mut fs, dir, name, VFS_KIND_FILE, 0o644);
			if created == NIL {
				return false;
			}
			created
		}
		existing => existing,
	};
	if fs.nodes[node as usize].kind != VFS_KIND_FILE {
		return false;
	}
	truncate_node(&mut fs, node, 0);
	(unsafe { write_file(&mut fs, node, 0, data.as_ptr(), data.len() as u64) }).is_ok()
}

/// 打开 *path*, 成功时返回节点下标。
///
/// 未命中且 *flags* 含 [`O_CREAT`] 时按 *mode* 创建: 同时含 [`O_DIRECTORY`] 时创建目录,
/// 否则创建文件; 命中且含 `O_CREAT | O_EXCL` 时返回 [`Exists`](VfsStatus::Exists)。
/// 最后一个名字是目录而 *flags* 要求写入访问时返回 [`IsDir`](VfsStatus::IsDir); 它不是
/// 目录而 *flags* 含 [`O_DIRECTORY`] 时返回 [`NotDir`](VfsStatus::NotDir)。*flags* 含
/// [`O_TRUNC`] 且要求写入访问且它是文件时截断到 0 字节。
pub fn open(path: &[u8], flags: u32, mode: u32) -> Result<i16, VfsStatus> {
	let mut fs = VFS.lock();
	let node = match split_target(&mut fs, path)? {
		// 最后一个名字只指代目录: O_CREAT 与 O_EXCL 判为已存在
		PathTarget::Dir(dir, _) => {
			if (flags & O_CREAT) != 0 && (flags & O_EXCL) != 0 {
				return Err(VfsStatus::Exists);
			}
			dir
		}
		PathTarget::Child(dir, name) =>
			match find_child(&fs, dir, name) {
				NIL => {
					if (flags & O_CREAT) == 0 {
						return Err(VfsStatus::NoEntry);
					}
					let kind = if (flags & O_DIRECTORY) != 0 { VFS_KIND_DIR } else { VFS_KIND_FILE };
					let created = create_child(&mut fs, dir, name, kind, mode);
					if created == NIL {
						return Err(VfsStatus::NoMemory);
					}
					created
				}
				existing => {
					if (flags & O_CREAT) != 0 && (flags & O_EXCL) != 0 {
						return Err(VfsStatus::Exists);
					}
					existing
				}
			}
	};
	// 最后一个名字所指节点的类别决定可用的访问方向与 O_DIRECTORY 是否成立
	let kind = fs.nodes[node as usize].kind;
	if kind == VFS_KIND_DIR {
		if mode_writable(flags) {
			return Err(VfsStatus::IsDir);
		}
	} else if (flags & O_DIRECTORY) != 0 {
		return Err(VfsStatus::NotDir);
	}
	// 要求写入访问时 O_TRUNC 截断, 目录不截断
	if (flags & O_TRUNC) != 0 && mode_writable(flags) && kind == VFS_KIND_FILE {
		truncate_node(&mut fs, node, 0);
	}
	Ok(node)
}

/// 取 *path* 的节点下标。最后一个名字只指代目录时, 它所指代的节点不是目录即返回
/// [`NotDir`](VfsStatus::NotDir)。
pub fn resolve_path(path: &[u8]) -> Result<i16, VfsStatus> {
	let mut fs = VFS.lock();
	match split_target(&mut fs, path)? {
		PathTarget::Dir(dir, _) => Ok(dir),
		PathTarget::Child(dir, name) =>
			match find_child(&fs, dir, name) {
				NIL => Err(VfsStatus::NoEntry),
				node => Ok(node),
			}
	}
}

/// 从文件 *node* 的 *offset* 处读取至多 *len* 字节到 *buf*, 返回实际读取的字节数。
/// 节点不是文件时返回 [`IsDir`](VfsStatus::IsDir)。
pub fn read(node: i16, offset: u64, buf: *mut u8, len: u64) -> Result<u64, VfsStatus> {
	if !is_node_valid(node) {
		return Err(VfsStatus::Invalid);
	}
	let fs = VFS.lock();
	if fs.nodes[node as usize].kind != VFS_KIND_FILE {
		return Err(VfsStatus::IsDir);
	}
	Ok(unsafe { read_file(&fs, node, offset, buf, len) })
}

/// 向文件 *node* 的 *offset* 处写入 *buf*, 返回写入的字节数。节点不是文件时返回
/// [`IsDir`](VfsStatus::IsDir)。
pub fn write(node: i16, offset: u64, buf: *const u8, len: u64) -> Result<u64, VfsStatus> {
	if !is_node_valid(node) {
		return Err(VfsStatus::Invalid);
	}
	let mut fs = VFS.lock();
	if fs.nodes[node as usize].kind != VFS_KIND_FILE {
		return Err(VfsStatus::IsDir);
	}
	unsafe { write_file(&mut fs, node, offset, buf, len) }
}

/// 取节点 *node* 的 stat 字段; 节点不存在时返回假。
pub fn stat(node: i16, out: &mut VfsStat) -> bool {
	if !is_node_valid(node) {
		return false;
	}
	let fs = VFS.lock();
	let entry = &fs.nodes[node as usize];
	*out = VfsStat {
		ino: ino_of(node),
		kind: entry.kind,
		mode: entry.mode & 0o777,
		// 目录的链接数 = 2 + 子目录数, 其中 2 来自目录自身与父目录中的 ".." 项; 文件为 1
		nlink: if entry.kind == VFS_KIND_DIR {
			2 + count_subdirs(&fs, node)
		} else {
			1
		},
		size: entry.size as i64,
		blocks: ((entry.size as i64) + 511) / 512,
	};
	true
}

/// 把文件 *node* 截断到 *len* 字节。节点不是文件时返回
/// [`Invalid`](VfsStatus::Invalid)。
pub fn truncate(node: i16, len: u64) -> VfsStatus {
	if !is_node_valid(node) {
		return VfsStatus::Invalid;
	}
	let mut fs = VFS.lock();
	if fs.nodes[node as usize].kind != VFS_KIND_FILE {
		return VfsStatus::Invalid;
	}
	truncate_node(&mut fs, node, len);
	VfsStatus::Ok
}

/// 读取目录 *node* 从 *pos* 处的条目起的目录项到 *buf*, 把新的读取位置写回 *pos*,
/// 返回写入的字节数。
///
/// 条目依次为 "."、".." 与各孩子, 孩子的顺序即孩子链表的顺序。单个条目按 Linux
/// dirent64 编码:
/// d_ino(8) + d_off(8) + d_reclen(2) + d_type(1) + d_name(结尾空字节), d_reclen 按
/// 8 字节对齐, d_off 取下一个条目的读取位置。缓冲区放不下下一个条目时停止, 该条目留待
/// 下次读取。
///
/// 节点不是目录时返回 [`NotDir`](VfsStatus::NotDir)。下标无效时返回
/// [`Invalid`](VfsStatus::Invalid); 缓冲区容纳不下一个条目时同样返回 [`Invalid`], 使调用方
/// 能够区分目录已读完与缓冲区过小。
pub fn getdents(node: i16, pos: &mut u64, buf: *mut u8, count: u64) -> Result<u64, VfsStatus> {
	if !is_node_valid(node) {
		return Err(VfsStatus::Invalid);
	}
	let fs = VFS.lock();
	if fs.nodes[node as usize].kind != VFS_KIND_DIR {
		return Err(VfsStatus::NotDir);
	}
	let mut at = *pos;
	let mut written: usize = 0;
	loop {
		// 确定当前条目: 前两项是 "." 与 "..", 其后依次是各孩子
		let mut name_buf = [0u8; NAME_MAX];
		let (ino, dtype, namelen): (u64, u8, usize) = if at == 0 {
			name_buf[0] = b'.';
			(ino_of(node), DT_DIR, 1)
		} else if at == 1 {
			name_buf[0] = b'.';
			name_buf[1] = b'.';
			let parent = fs.nodes[node as usize].parent;
			let p = if parent == NIL { node } else { parent };
			(ino_of(p), DT_DIR, 2)
		} else {
			let mut child = fs.nodes[node as usize].first_child;
			let mut k = at - 2;
			while child != NIL && k > 0 {
				child = fs.nodes[child as usize].next_sibling;
				k -= 1;
			}
			if child == NIL {
				break;
			}
			let n = &fs.nodes[child as usize];
			let nl = n.name_len as usize;
			for i in 0..nl {
				name_buf[i] = n.name[i];
			}
			(ino_of(child), dtype_of(n.kind), nl)
		};
		let reclen = (19 + namelen + 1 + 7) & !7;
		if written + reclen > (count as usize) {
			// 缓冲区容纳不下当前条目: 已经写入的条目照常返回, 一个条目都写不进去时返回
			// Invalid, 使调用方能够区分目录已读完与缓冲区过小。
			if written == 0 {
				return Err(VfsStatus::Invalid);
			}
			break;
		}
		unsafe {
			let entry = buf.add(written);
			(entry as *mut u64).write_volatile(ino);
			(entry.add(8) as *mut i64).write_volatile((at + 1) as i64);
			(entry.add(16) as *mut u16).write_volatile(reclen as u16);
			(entry.add(18) as *mut u8).write_volatile(dtype);
			for i in 0..namelen {
				entry.add(19 + i).write_volatile(name_buf[i]);
			}
			entry.add(19 + namelen).write_volatile(0u8);
		}
		written += reclen;
		at += 1;
	}
	*pos = at;
	Ok(written as u64)
}

/// 创建目录 *path*, *mode* 为权限位。路径的最后一个名字只指代目录, 或该名字已被占用时,
/// 都返回 [`Exists`](VfsStatus::Exists)。
pub fn mkdir(path: &[u8], mode: u32) -> VfsStatus {
	let mut fs = VFS.lock();
	let (dir, name) = match split_target(&mut fs, path) {
		Ok(PathTarget::Child(dir, name)) => (dir, name),
		Ok(PathTarget::Dir(_, _)) => {
			return VfsStatus::Exists;
		}
		Err(status) => {
			return status;
		}
	};
	if find_child(&fs, dir, name) != NIL {
		return VfsStatus::Exists;
	}
	if create_child(&mut fs, dir, name, VFS_KIND_DIR, mode) == NIL {
		return VfsStatus::NoMemory;
	}
	VfsStatus::Ok
}

/// 删除 *path* 所指的对象。
///
/// *flags* 含 [`AT_REMOVEDIR`] 时目标须是目录且须为空, 不含时目标须是文件; 目标为目录
/// 而 *flags* 不含该位时返回 [`IsDir`](VfsStatus::IsDir), 目标为文件而含该位时返回
/// [`NotDir`](VfsStatus::NotDir)。根目录没有父目录可供移除, 返回
/// [`Busy`](VfsStatus::Busy)。
///
/// 最后一个名字是 "." 与 ".." 时给出的目录不以本函数移除: *flags* 不含
/// [`AT_REMOVEDIR`] 时返回 [`IsDir`](VfsStatus::IsDir), 含该位时 ".." 返回
/// [`NotEmpty`](VfsStatus::NotEmpty), "." 返回 [`Invalid`](VfsStatus::Invalid)。
pub fn unlink(path: &[u8], flags: u32) -> VfsStatus {
	let mut fs = VFS.lock();
	let target = match split_target(&mut fs, path) {
		Ok(PathTarget::Child(dir, name)) =>
			match find_child(&fs, dir, name) {
				NIL => {
					return VfsStatus::NoEntry;
				}
				target => target,
			}
		Ok(PathTarget::Dir(node, name)) if name.is_empty() => node,
		Ok(PathTarget::Dir(_, name)) => {
			return if (flags & AT_REMOVEDIR) == 0 {
				VfsStatus::IsDir
			} else if name == b".." {
				VfsStatus::NotEmpty
			} else {
				VfsStatus::Invalid
			};
		}
		Err(status) => {
			return status;
		}
	};
	let is_dir = fs.nodes[target as usize].kind == VFS_KIND_DIR;
	if is_dir != ((flags & AT_REMOVEDIR) != 0) {
		return if is_dir { VfsStatus::IsDir } else { VfsStatus::NotDir };
	}
	let parent = fs.nodes[target as usize].parent;
	if parent == NIL {
		return VfsStatus::Busy;
	}
	if is_dir && fs.nodes[target as usize].first_child != NIL {
		return VfsStatus::NotEmpty;
	}
	remove_child(&mut fs, parent, target);
	VfsStatus::Ok
}

/// 检查 *path* 是否存在。
pub fn access(path: &[u8]) -> VfsStatus {
	match resolve_path(path) {
		Ok(_) => VfsStatus::Ok,
		Err(status) => status,
	}
}

// ---------------------------------------------------------------
//  测试
// ---------------------------------------------------------------

#[cfg(test)]
pub(crate) mod tests {
	use std::sync::{ Mutex, MutexGuard };

	use super::{
		DATA_ARENA,
		DT_DIR,
		DT_REG,
		MAX_NODES,
		NAME_MAX,
		VFS,
		VfsState,
		access,
		getdents,
		init,
		inject_file,
		mkdir,
		open,
		read,
		resolve_path,
		stat,
		truncate,
		unlink,
		write,
	};
	use crate::vfs_ops::{
		AT_REMOVEDIR,
		O_CREAT,
		O_DIRECTORY,
		O_EXCL,
		O_RDONLY,
		O_RDWR,
		O_TRUNC,
		O_WRONLY,
		VFS_KIND_DIR,
		VFS_KIND_FILE,
		VfsStat,
		VfsStatus,
	};

	/// 目录树与文件数据区是全局量, 用例依次取用。
	pub(crate) static LOCK: Mutex<()> = Mutex::new(());

	/// 清空目录树并重建根目录, 使各用例从同一初始状态出发。
	///
	/// 运行时的系统调用层测试入口与本模块的用例同处一个测试二进制, 两者改动的都是本模块
	/// 的全局量, 故该入口取同一把锁并执行同一次清空。
	pub(crate) fn setup() -> MutexGuard<'static, ()> {
		let guard = LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
		{
			let mut fs = VFS.lock();
			*fs = VfsState::empty();
		}
		assert!(init(), "根目录建立失败");
		guard
	}

	/// 取出成功返回的节点下标。状态枚举不实现 `Debug`, 故失败时按编码报告。
	fn ok_node(result: Result<i16, VfsStatus>) -> i16 {
		match result {
			Ok(node) => node,
			Err(status) => panic!("期望成功, 实际返回状态码 {}", status as u32),
		}
	}

	/// 取出成功返回的写入字节数。
	fn ok_written(result: Result<u64, VfsStatus>) -> u64 {
		match result {
			Ok(written) => written,
			Err(status) => panic!("期望成功, 实际返回状态码 {}", status as u32),
		}
	}

	/// 断言状态码等于 *expect*; 状态枚举不实现 `Debug`, 故按编码比较。
	fn assert_status(got: VfsStatus, expect: VfsStatus) {
		assert!((got as u32) == (expect as u32), "期望状态码 {}, 实际 {}", expect as u32, got as u32);
	}

	/// 建一个文件并返回其节点下标。
	fn create_file(path: &[u8]) -> i16 {
		ok_node(open(path, O_CREAT | O_RDWR, 0o644))
	}

	/// 取节点当前的 stat 字段。
	fn stat_of(node: i16) -> VfsStat {
		let mut st = VfsStat::zero();
		assert!(stat(node, &mut st), "节点 {node} 的 stat 失败");
		st
	}

	/// 读出一个文件的全部数据。
	fn read_all(node: i16) -> Vec<u8> {
		let size = stat_of(node).size as u64;
		let mut buf = vec![0u8; size as usize];
		assert_eq!(ok_written(read(node, 0, buf.as_mut_ptr(), size)), size, "未读到全部数据");
		buf
	}

	/// 由一段字节拼出以 '/' 开头的路径。
	fn path_of(content: &[u8]) -> Vec<u8> {
		let mut path = vec![b'/'];
		path.extend_from_slice(content);
		path
	}

	/// 缓冲区中的一个目录项。
	struct Dirent {
		ino: u64,
		off: i64,
		dtype: u8,
		name: Vec<u8>,
	}

	/// 按 Linux dirent64 的编码解码缓冲区中的全部目录项, 并核对各项的长度与结尾空字节。
	fn decode_dirents(buf: &[u8]) -> Vec<Dirent> {
		let mut out = Vec::new();
		let mut at = 0usize;
		while at < buf.len() {
			let base = buf.as_ptr();
			let ino = unsafe { core::ptr::read_unaligned(base.add(at) as *const u64) };
			let off = unsafe { core::ptr::read_unaligned(base.add(at + 8) as *const i64) };
			let reclen = unsafe { core::ptr::read_unaligned(base.add(at + 16) as *const u16) };
			let dtype = buf[at + 18];
			assert!((reclen as usize) >= 20, "目录项长度放不下名称");
			assert!((reclen as usize) % 8 == 0, "目录项长度未按 8 字节对齐");
			assert!(at + (reclen as usize) <= buf.len(), "目录项超出缓冲区");
			let end = at + (reclen as usize);
			let mut name = Vec::new();
			let mut i = at + 19;
			while i < end && buf[i] != 0 {
				name.push(buf[i]);
				i += 1;
			}
			assert!(i < end, "名称没有结尾空字节");
			out.push(Dirent { ino, off, dtype, name });
			at = end;
		}
		out
	}

	/// 建立根目录后根目录可解析, 其 stat 字段为目录, 且重复建立同样成功。
	#[test]
	fn test_init_builds_the_root_directory() {
		let _guard = setup();

		assert_eq!(ok_node(resolve_path(b"/")), 0);
		assert_eq!(ok_node(resolve_path(b"//")), 0, "整个路径只由 '/' 组成时给出根目录");
		assert_status(access(b"/"), VfsStatus::Ok);

		let st = stat_of(0);
		assert_eq!(st.ino, 1);
		assert_eq!(st.kind, VFS_KIND_DIR);
		assert_eq!(st.mode, 0o755);
		assert_eq!(st.nlink, 2);
		assert_eq!(st.size, 0);

		assert!(init());
	}

	/// 空路径不指代任何对象, 各项操作都返回 `NoEntry`, 带 `O_CREAT` 的打开同样不建立节点。
	#[test]
	fn test_empty_path_reports_no_entry() {
		let _guard = setup();

		assert!(matches!(resolve_path(b""), Err(VfsStatus::NoEntry)));
		assert_status(access(b""), VfsStatus::NoEntry);
		assert_status(mkdir(b"", 0o755), VfsStatus::NoEntry);
		assert_status(unlink(b"", 0), VfsStatus::NoEntry);
		assert_status(unlink(b"", AT_REMOVEDIR), VfsStatus::NoEntry);
		assert!(matches!(open(b"", O_CREAT | O_RDWR, 0o644), Err(VfsStatus::NoEntry)));
		assert!(!inject_file(b"", b"x"));
	}

	/// "." 与 ".." 名字按目录树的父子关系解析, 根目录的父目录仍是根目录, 连续斜杠与
	/// 名字前缺斜杠的写法都不引入空名字。
	#[test]
	fn test_resolve_path_handles_dot_and_dotdot() {
		let _guard = setup();
		assert_status(mkdir(b"/a", 0o755), VfsStatus::Ok);
		assert_status(mkdir(b"/a/b", 0o755), VfsStatus::Ok);

		let a = ok_node(resolve_path(b"/a"));
		let b = ok_node(resolve_path(b"/a/b"));
		assert_ne!(a, b, "两级目录是不同的节点");
		assert_eq!(ok_node(resolve_path(b"/a/./b")), b);
		assert_eq!(ok_node(resolve_path(b"/a/b/.")), b);
		assert_eq!(ok_node(resolve_path(b"/a/b/..")), a);
		assert_eq!(ok_node(resolve_path(b"/a/b/../..")), 0);
		assert_eq!(ok_node(resolve_path(b"/..")), 0, "根目录的父目录仍是根目录");
		assert_eq!(ok_node(resolve_path(b"a/b")), b, "名字前缺斜杠时仍从根目录开始");
		assert_eq!(ok_node(resolve_path(b"/a///b")), b, "连续斜杠不引入空名字");
		assert_eq!(ok_node(resolve_path(b"/a/b/")), b, "末尾的 '/' 不引入空名字");
	}

	/// 路径的某一级不存在时返回 `NoEntry`; 某一级不是目录, 或最后一个名字只指代目录而
	/// 它指代的是文件时返回 `NotDir`。
	#[test]
	fn test_resolve_path_reports_a_missing_component() {
		let _guard = setup();
		assert!(matches!(resolve_path(b"/nope"), Err(VfsStatus::NoEntry)));
		assert!(matches!(resolve_path(b"/nope/child"), Err(VfsStatus::NoEntry)));
		assert!(matches!(resolve_path(b"/nope/.."), Err(VfsStatus::NoEntry)));

		create_file(b"/f");
		assert!(matches!(resolve_path(b"/f/child"), Err(VfsStatus::NotDir)));
		assert!(matches!(resolve_path(b"/f/."), Err(VfsStatus::NotDir)));
		assert!(matches!(resolve_path(b"/f/"), Err(VfsStatus::NotDir)));
		assert!(matches!(resolve_path(b"/f/.."), Err(VfsStatus::NotDir)));
	}

	/// 宿主把数据注入为文件, 沿路目录按需创建, 再次注入同一路径时沿用既有节点并替换其
	/// 数据。
	#[test]
	fn test_inject_file_creates_leading_directories() {
		let _guard = setup();

		assert!(inject_file(b"/etc/motd", b"hello"));
		let node = ok_node(resolve_path(b"/etc/motd"));
		assert_eq!(read_all(node), b"hello");
		assert_eq!(stat_of(node).kind, VFS_KIND_FILE);
		assert_eq!(stat_of(node).mode, 0o644);

		let dir = ok_node(resolve_path(b"/etc"));
		assert_eq!(stat_of(dir).kind, VFS_KIND_DIR);
		assert_eq!(stat_of(dir).mode, 0o755);

		assert!(inject_file(b"/etc/motd", b"bye"), "再次注入同一路径");
		assert_eq!(stat_of(node).size, 3);
		assert_eq!(read_all(node), b"bye");

		assert!(!inject_file(b"/", b"x"), "根目录不作为文件注入的目标");
	}

	/// 未带 `O_CREAT` 时路径不存在即返回 `NoEntry`; 带 `O_CREAT` 时创建, 已存在且带
	/// `O_EXCL` 时返回 `Exists`。
	#[test]
	fn test_open_creates_only_with_creat() {
		let _guard = setup();

		assert!(matches!(open(b"/x", O_RDONLY, 0), Err(VfsStatus::NoEntry)));
		let created = ok_node(open(b"/x", O_CREAT | O_RDWR, 0o640));
		assert_eq!(stat_of(created).mode, 0o640);
		assert!(matches!(open(b"/x", O_CREAT | O_EXCL | O_RDWR, 0o644), Err(VfsStatus::Exists)));
		assert_eq!(ok_node(open(b"/x", O_CREAT | O_RDWR, 0o644)), created);

		assert!(
			matches!(open(b"/nodir/x", O_CREAT | O_RDWR, 0o644), Err(VfsStatus::NoEntry)),
			"路径的某一级不存在时 `O_CREAT` 不创建中间的目录"
		);
	}

	/// 末级不是目录时带 `O_DIRECTORY` 的打开返回 `NotDir`, 末级是目录时按目录打开。
	#[test]
	fn test_open_with_directory_flag_requires_a_directory() {
		let _guard = setup();
		create_file(b"/f");

		assert!(matches!(open(b"/f", O_DIRECTORY | O_RDONLY, 0), Err(VfsStatus::NotDir)));
		assert_eq!(ok_node(open(b"/", O_DIRECTORY | O_RDONLY, 0)), 0);
	}

	/// `O_TRUNC` 只在访问模式不是只读时截断, 且只作用于文件; 目录按可写访问打开返回
	/// `IsDir`, 按只读访问打开成功。
	#[test]
	fn test_open_truncates_only_for_a_writable_access() {
		let _guard = setup();
		let node = create_file(b"/f");
		assert_eq!(ok_written(write(node, 0, b"hello".as_ptr(), 5)), 5);

		assert_eq!(ok_node(open(b"/f", O_RDONLY | O_TRUNC, 0)), node);
		assert_eq!(stat_of(node).size, 5);

		assert_eq!(ok_node(open(b"/f", O_WRONLY | O_TRUNC, 0)), node);
		assert_eq!(stat_of(node).size, 0);

		assert_status(mkdir(b"/d", 0o755), VfsStatus::Ok);
		let d = ok_node(resolve_path(b"/d"));
		assert!(matches!(open(b"/d", O_RDWR | O_TRUNC, 0), Err(VfsStatus::IsDir)));
		assert_eq!(ok_node(open(b"/d", O_RDONLY, 0)), d);
		assert_eq!(stat_of(d).kind, VFS_KIND_DIR);
	}

	/// 写入后可读回相同的数据; 读取止于文件末尾; 覆盖写不改变文件长度。
	#[test]
	fn test_write_and_read_round_trip() {
		let _guard = setup();
		let node = create_file(b"/f");
		let src = b"abcdefgh";

		assert_eq!(ok_written(write(node, 0, src.as_ptr(), src.len() as u64)), 8);
		let mut buf = [0u8; 16];
		assert_eq!(ok_written(read(node, 0, buf.as_mut_ptr(), 16)), 8);
		assert_eq!(&buf[..8], src);
		assert_eq!(ok_written(read(node, 4, buf.as_mut_ptr(), 3)), 3);
		assert_eq!(&buf[..3], b"efg");
		assert_eq!(ok_written(read(node, 8, buf.as_mut_ptr(), 4)), 0);
		assert_eq!(ok_written(read(node, 100, buf.as_mut_ptr(), 4)), 0);

		assert_eq!(ok_written(write(node, 2, b"XY".as_ptr(), 2)), 2);
		assert_eq!(stat_of(node).size, 8, "覆盖写不改变文件长度");
		assert_eq!(read_all(node), b"abXYefgh");

		assert_eq!(ok_written(write(node, 0, buf.as_ptr(), 0)), 0, "长度为 0 的写入不改变文件");
		assert_eq!(stat_of(node).size, 8);
	}

	/// 越过文件末尾写入时文件增长, 中间未被写入的字节取数据区的初值 0。
	#[test]
	fn test_write_extends_the_file_and_leaves_the_gap_zeroed() {
		let _guard = setup();
		let node = create_file(b"/f");

		assert_eq!(ok_written(write(node, 0, b"head".as_ptr(), 4)), 4);
		assert_eq!(ok_written(write(node, 8, b"tail".as_ptr(), 4)), 4);
		assert_eq!(stat_of(node).size, 12);

		let mut buf = [0xffu8; 12];
		assert_eq!(ok_written(read(node, 0, buf.as_mut_ptr(), 12)), 12);
		assert_eq!(&buf[0..4], b"head");
		assert_eq!(&buf[4..8], &[0u8; 4]);
		assert_eq!(&buf[8..12], b"tail");
	}

	/// 文件不在数据区尾部时, 增长会把既有数据拷贝到新的区域, 其他文件的数据不受影响。
	#[test]
	fn test_write_relocates_a_file_that_is_not_at_the_arena_tail() {
		let _guard = setup();
		let first = create_file(b"/first");
		let second = create_file(b"/second");

		assert_eq!(ok_written(write(first, 0, b"AAAA".as_ptr(), 4)), 4);
		assert_eq!(ok_written(write(second, 0, b"BBBBBBBB".as_ptr(), 8)), 8);
		assert_eq!(ok_written(write(first, 4, b"CCCC".as_ptr(), 4)), 4);

		assert_eq!(stat_of(first).size, 8);
		assert_eq!(read_all(first), b"AAAACCCC");
		assert_eq!(stat_of(second).size, 8);
		assert_eq!(read_all(second), b"BBBBBBBB");
	}

	/// 数据区容不下增长的写入时返回 `NoSpace`, 文件保持原样。
	#[test]
	fn test_write_reports_no_space_when_the_arena_is_full() {
		let _guard = setup();
		let node = create_file(b"/f");

		let mut data = vec![0x5au8; DATA_ARENA + 1];
		assert!(matches!(write(node, 0, data.as_ptr(), data.len() as u64), Err(VfsStatus::NoSpace)));
		assert_eq!(stat_of(node).size, 0, "被拒的写入不改变文件");

		data.truncate(DATA_ARENA);
		let written = ok_written(write(node, 0, data.as_ptr(), data.len() as u64));
		assert_eq!(written, DATA_ARENA as u64);
		assert_eq!(stat_of(node).size, DATA_ARENA as i64);
		assert_eq!(read_all(node), data);
	}

	/// 偏移与长度之和超出 usize 的表示范围时返回 `NoSpace`, 文件保持原样。
	#[test]
	fn test_write_reports_no_space_when_the_offset_overflows() {
		let _guard = setup();
		let node = create_file(b"/f");
		assert_eq!(ok_written(write(node, 0, b"head".as_ptr(), 4)), 4);

		assert!(matches!(write(node, u64::MAX, b"x".as_ptr(), 1), Err(VfsStatus::NoSpace)));
		assert_eq!(stat_of(node).size, 4, "被拒的写入不改变文件");
		assert_eq!(read_all(node), b"head");
	}

	/// 节点下标越界时读与写返回 `Invalid`、truncate 返回 `Invalid`、stat 返回假。
	#[test]
	fn test_read_and_write_reject_an_invalid_node() {
		let _guard = setup();
		let mut buf = [0u8; 4];
		let invalid = MAX_NODES as i16;

		assert!(matches!(read(-1, 0, buf.as_mut_ptr(), 4), Err(VfsStatus::Invalid)));
		assert!(matches!(read(invalid, 0, buf.as_mut_ptr(), 4), Err(VfsStatus::Invalid)));
		assert!(matches!(write(-1, 0, buf.as_ptr(), 4), Err(VfsStatus::Invalid)));
		assert!(matches!(write(invalid, 0, buf.as_ptr(), 4), Err(VfsStatus::Invalid)));
		assert_status(truncate(-1, 0), VfsStatus::Invalid);
		assert_status(truncate(invalid, 0), VfsStatus::Invalid);
		assert!(!stat(-1, &mut VfsStat::zero()));
	}

	/// 节点是目录时读与写返回 `IsDir`。
	#[test]
	fn test_read_and_write_report_isdir_for_a_directory() {
		let _guard = setup();
		assert_status(mkdir(b"/d", 0o755), VfsStatus::Ok);
		let d = ok_node(resolve_path(b"/d"));
		let mut buf = [0u8; 4];

		assert!(matches!(read(d, 0, buf.as_mut_ptr(), 4), Err(VfsStatus::IsDir)));
		assert!(matches!(write(d, 0, buf.as_ptr(), 4), Err(VfsStatus::IsDir)));
	}

	/// stat 的 ino 取节点下标加 1, 文件的 blocks 按 512 字节向上取整。
	#[test]
	fn test_stat_reports_the_file_fields() {
		let _guard = setup();
		let node = create_file(b"/f");
		let data = vec![7u8; 513];
		assert_eq!(ok_written(write(node, 0, data.as_ptr(), 513)), 513);

		let st = stat_of(node);
		assert_eq!(st.ino, (node as u64) + 1);
		assert_eq!(st.kind, VFS_KIND_FILE);
		assert_eq!(st.mode, 0o644);
		assert_eq!(st.nlink, 1);
		assert_eq!(st.size, 513);
		assert_eq!(st.blocks, 2, "513 字节占两个 512 字节的块");
	}

	/// 截断只收缩不扩展, 目录与越界下标都返回 `Invalid`。
	#[test]
	fn test_truncate_shrinks_only() {
		let _guard = setup();
		let node = create_file(b"/f");
		assert_eq!(ok_written(write(node, 0, b"0123456789".as_ptr(), 10)), 10);

		assert_status(truncate(node, 4), VfsStatus::Ok);
		assert_eq!(read_all(node), b"0123");
		assert_eq!(stat_of(node).size, 4);

		assert_status(truncate(node, 100), VfsStatus::Ok);
		assert_eq!(stat_of(node).size, 4, "截断到更大的长度不改变文件");

		assert_status(truncate(0, 0), VfsStatus::Invalid);
	}

	/// 目录项依次为 "."、".." 与各孩子, 名字、".." 的父目录与类型字段正确, 读完之后
	/// 再取一次不写入任何字节。
	#[test]
	fn test_getdents_lists_dot_dotdot_and_children() {
		let _guard = setup();
		assert_status(mkdir(b"/d", 0o755), VfsStatus::Ok);
		let d = ok_node(resolve_path(b"/d"));
		let f1 = create_file(b"/d/f1");
		let f2 = create_file(b"/d/f2");

		let mut buf = [0u8; 256];
		let mut pos = 0u64;
		let n = ok_written(getdents(d, &mut pos, buf.as_mut_ptr(), 256));
		let entries = decode_dirents(&buf[..n as usize]);
		assert_eq!(entries.len(), 4);

		assert_eq!(entries[0].name, b".");
		assert_eq!(entries[0].ino, (d as u64) + 1);
		assert_eq!(entries[0].dtype, DT_DIR);
		assert_eq!(entries[0].off, 1, "off 取下一个条目的读取位置");

		assert_eq!(entries[1].name, b"..");
		assert_eq!(entries[1].ino, 1, ".. 指向父目录");
		assert_eq!(entries[1].dtype, DT_DIR);
		assert_eq!(entries[1].off, 2);

		let mut children: Vec<u64> = entries[2..]
			.iter()
			.map(|entry| entry.ino)
			.collect();
		children.sort();
		let mut expected = vec![(f1 as u64) + 1, (f2 as u64) + 1];
		expected.sort();
		assert_eq!(children, expected);
		for entry in &entries[2..] {
			assert_eq!(entry.dtype, DT_REG);
		}
		assert_eq!(entries[2].off, 3);
		assert_eq!(entries[3].off, 4);
		assert_eq!(pos, 4);

		let n2 = ok_written(getdents(d, &mut pos, buf.as_mut_ptr(), 256));
		assert_eq!(n2, 0, "读完之后不再写入");
		assert_eq!(pos, 4);
	}

	/// 缓冲区放不下下一个目录项时停止, 该条目留待下次读取; 已经写入的条目其读取位置照常
	/// 推进。
	#[test]
	fn test_getdents_stops_when_the_next_entry_does_not_fit() {
		let _guard = setup();
		assert_status(mkdir(b"/d", 0o755), VfsStatus::Ok);
		let d = ok_node(resolve_path(b"/d"));
		create_file(b"/d/f1");

		// "." 与 ".." 的条目各长 24 字节, 故 24 字节每次只容纳一个条目
		let mut buf = [0u8; 64];
		let mut pos = 0u64;
		let n = ok_written(getdents(d, &mut pos, buf.as_mut_ptr(), 24));
		let entries = decode_dirents(&buf[..n as usize]);
		assert_eq!(entries.len(), 1);
		assert_eq!(entries[0].name, b".");
		assert_eq!(pos, 1, "已写入的条目其读取位置照常推进");

		let n2 = ok_written(getdents(d, &mut pos, buf.as_mut_ptr(), 24));
		let entries = decode_dirents(&buf[..n2 as usize]);
		assert_eq!(entries.len(), 1);
		assert_eq!(entries[0].name, b"..");
		assert_eq!(pos, 2, "放不下的条目留待下次读取");
	}

	/// 缓冲区容纳不下一个目录项时返回 `Invalid`, 不与"目录已读完"返回 0 混淆。
	#[test]
	fn test_getdents_reports_invalid_when_one_entry_does_not_fit() {
		let _guard = setup();
		assert_status(mkdir(b"/d", 0o755), VfsStatus::Ok);
		let d = ok_node(resolve_path(b"/d"));
		create_file(b"/d/f1");

		let mut buf = [0u8; 64];
		let mut pos = 0u64;
		// "." 的条目长 24 字节, 23 字节与 0 字节都容纳不下
		assert!(matches!(getdents(d, &mut pos, buf.as_mut_ptr(), 23), Err(VfsStatus::Invalid)));
		assert!(matches!(getdents(d, &mut pos, buf.as_mut_ptr(), 0), Err(VfsStatus::Invalid)));
		assert_eq!(pos, 0, "未写入任何条目时读取位置不推进");
		// 刚好容纳一个条目
		assert_eq!(ok_written(getdents(d, &mut pos, buf.as_mut_ptr(), 24)), 24);
		assert_eq!(pos, 1);
	}

	/// 节点不是目录时取目录项返回 `NotDir`, 下标无效时返回 `Invalid`。
	#[test]
	fn test_getdents_rejects_a_node_that_is_not_a_directory() {
		let _guard = setup();
		let node = create_file(b"/f");
		let mut buf = [0u8; 64];
		let mut pos = 0u64;

		assert!(matches!(getdents(node, &mut pos, buf.as_mut_ptr(), 64), Err(VfsStatus::NotDir)));
		assert!(matches!(getdents(-1, &mut pos, buf.as_mut_ptr(), 64), Err(VfsStatus::Invalid)));
		assert_eq!(pos, 0);
	}

	/// 创建目录成功时按 `mode` 记录权限, 重名返回 `Exists`, 父目录不存在返回 `NoEntry`,
	/// 路径的某一级不是目录返回 `NotDir`, 最后一个名字达到名字长度上限返回 `NameTooLong`。
	#[test]
	fn test_mkdir_creates_a_directory_and_reports_the_failures() {
		let _guard = setup();

		assert_status(mkdir(b"/a", 0o700), VfsStatus::Ok);
		let a = ok_node(resolve_path(b"/a"));
		assert_eq!(stat_of(a).kind, VFS_KIND_DIR);
		assert_eq!(stat_of(a).mode, 0o700);
		assert_status(mkdir(b"/a", 0o755), VfsStatus::Exists);

		assert_status(mkdir(b"/nodir/x", 0o755), VfsStatus::NoEntry);

		create_file(b"/f");
		assert_status(mkdir(b"/f/x", 0o755), VfsStatus::NotDir);

		assert_status(mkdir(&vec![b'x'; NAME_MAX], 0o755), VfsStatus::NameTooLong);
	}

	/// 路径的最后一个名字只指代目录时创建返回 `Exists`: 路径以 '/' 结尾、整个路径只由 '/'
	/// 组成, 或这个名字是 "." 与 ".."。
	#[test]
	fn test_mkdir_refuses_a_path_that_names_a_directory() {
		let _guard = setup();
		assert_status(mkdir(b"/a", 0o755), VfsStatus::Ok);

		assert_status(mkdir(b"/a/", 0o755), VfsStatus::Exists);
		assert_status(mkdir(b"/", 0o755), VfsStatus::Exists);
		assert_status(mkdir(b"//", 0o755), VfsStatus::Exists);
		assert_status(mkdir(b"/a/.", 0o755), VfsStatus::Exists);
		assert_status(mkdir(b"/a/..", 0o755), VfsStatus::Exists);

		assert_eq!(stat_of(ok_node(resolve_path(b"/a"))).nlink, 2);
	}

	/// 删除可移除文件与空目录, 非空目录返回 `NotEmpty`, 路径不存在返回 `NoEntry`。
	#[test]
	fn test_unlink_removes_files_and_empty_directories() {
		let _guard = setup();
		assert_status(mkdir(b"/d", 0o755), VfsStatus::Ok);
		create_file(b"/d/f");

		assert_status(unlink(b"/d", AT_REMOVEDIR), VfsStatus::NotEmpty);
		assert_status(unlink(b"/d", 0), VfsStatus::IsDir);
		assert_status(unlink(b"/d/f", AT_REMOVEDIR), VfsStatus::NotDir);
		assert_status(unlink(b"/d", AT_REMOVEDIR), VfsStatus::NotEmpty);
		assert_status(unlink(b"/d/f", 0), VfsStatus::Ok);
		assert!(matches!(resolve_path(b"/d/f"), Err(VfsStatus::NoEntry)));
		assert_status(unlink(b"/d/", AT_REMOVEDIR), VfsStatus::Ok);
		assert!(matches!(resolve_path(b"/d"), Err(VfsStatus::NoEntry)));
		assert_status(unlink(b"/d", AT_REMOVEDIR), VfsStatus::NoEntry);
		assert_status(unlink(b"/d", 0), VfsStatus::NoEntry);
		assert_status(access(b"/d"), VfsStatus::NoEntry);
	}

	/// 根目录与 "." 与 ".." 都不以删除操作移除: *flags* 不含 [`AT_REMOVEDIR`] 时返回
	/// `IsDir`, 含时根目录返回 `Busy`、".." 返回 `NotEmpty`、"." 返回 `Invalid`。
	#[test]
	fn test_unlink_refuses_the_root_and_dot_names() {
		let _guard = setup();
		assert_status(mkdir(b"/d", 0o755), VfsStatus::Ok);

		assert_status(unlink(b"/", 0), VfsStatus::IsDir);
		assert_status(unlink(b"/", AT_REMOVEDIR), VfsStatus::Busy);
		assert_status(unlink(b"//", AT_REMOVEDIR), VfsStatus::Busy);

		assert_status(unlink(b"/d/.", 0), VfsStatus::IsDir);
		assert_status(unlink(b"/d/.", AT_REMOVEDIR), VfsStatus::Invalid);
		assert_status(unlink(b"/d/..", 0), VfsStatus::IsDir);
		assert_status(unlink(b"/d/..", AT_REMOVEDIR), VfsStatus::NotEmpty);

		assert_status(access(b"/d"), VfsStatus::Ok);
	}

	/// 移除孩子链表中的一项之后, 其余孩子仍可被路径解析与目录遍历命中。
	#[test]
	fn test_unlink_keeps_the_remaining_children() {
		let _guard = setup();
		assert_status(mkdir(b"/d", 0o755), VfsStatus::Ok);
		let d = ok_node(resolve_path(b"/d"));
		let f1 = create_file(b"/d/f1");
		create_file(b"/d/f2");
		let f3 = create_file(b"/d/f3");

		assert_status(unlink(b"/d/f2", 0), VfsStatus::Ok);

		let mut buf = [0u8; 256];
		let mut pos = 0u64;
		let n = ok_written(getdents(d, &mut pos, buf.as_mut_ptr(), 256));
		let mut names: Vec<Vec<u8>> = decode_dirents(&buf[..n as usize])
			.into_iter()
			.map(|entry| entry.name)
			.collect();
		assert_eq!(names.len(), 4, "两个文件与两项自身");
		names.sort();
		assert_eq!(names, vec![b".".to_vec(), b"..".to_vec(), b"f1".to_vec(), b"f3".to_vec()]);
		assert!(matches!(resolve_path(b"/d/f2"), Err(VfsStatus::NoEntry)));

		let mut st = VfsStat::zero();
		assert!(stat(f1, &mut st));
		assert!(stat(f3, &mut st));
	}

	/// 节点一旦分配即不回收: 删除后重建同名文件取用新的下标, 数据从空开始。
	#[test]
	fn test_unlink_does_not_recycle_the_node() {
		let _guard = setup();
		let before = create_file(b"/f");
		assert_eq!(ok_written(write(before, 0, b"data".as_ptr(), 4)), 4);

		assert_status(unlink(b"/f", 0), VfsStatus::Ok);
		let after = create_file(b"/f");
		assert_ne!(after, before);
		assert_eq!(stat_of(after).size, 0);
	}

	/// 节点用尽时创建返回 `NoMemory`, 注入同样失败。
	#[test]
	fn test_node_exhaustion_reports_no_memory() {
		let _guard = setup();

		for i in 0..MAX_NODES - 1 {
			let path = format!("/n{i}");
			ok_node(open(path.as_bytes(), O_CREAT | O_RDWR, 0o644));
		}
		assert!(matches!(open(b"/overflow", O_CREAT | O_RDWR, 0o644), Err(VfsStatus::NoMemory)));
		assert_status(mkdir(b"/overflow_dir", 0o755), VfsStatus::NoMemory);
		assert!(!inject_file(b"/overflow_file", b"x"));

		assert_status(access(b"/n0"), VfsStatus::Ok);
	}

	/// 检查路径是否存在。
	#[test]
	fn test_access_reports_existence() {
		let _guard = setup();
		assert_status(mkdir(b"/d", 0o755), VfsStatus::Ok);
		create_file(b"/d/f");

		assert_status(access(b"/"), VfsStatus::Ok);
		assert_status(access(b"/d"), VfsStatus::Ok);
		assert_status(access(b"/d/f"), VfsStatus::Ok);
		assert_status(access(b"/d/g"), VfsStatus::NoEntry);
		assert_status(access(b""), VfsStatus::NoEntry);
	}
}
