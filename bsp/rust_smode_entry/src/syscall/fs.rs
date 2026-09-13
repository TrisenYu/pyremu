//! 文件系统系统调用: openat/close/read/write/readv/writev/lseek/pread64/pwrite64/
//! fstat/fstatat/getdents/getcwd/mkdirat/unlinkat/faccessat/fcntl/fsync/ftruncate/ppoll。
//!
//! 移植自 ref-impl/emod/emod_vfs 的 Prex VFS。裁剪其 vnode/dentry/mount 三层抽象为
//! 单一的 ramfs 节点树, 保留核心语义: 路径查找、目录树、fd 偏移、读写、截断、
//! 创建/删除/目录遍历。
//!
//! 运行时为 no_std 且无全局分配器, 无法沿用参考实现的 calloc/realloc, 改用静态
//! 定容存储: 节点存于定长数组, 文件数据存于定长字节竞技场 (bump 分配), 打开文件表
//! 存于定长数组。三者均落在 BSS 段, 跨 SUSPEND/RESUME 存活。

#![allow(dead_code)]

use core::cmp;
use core::ptr;

use crate::concurrency::spinlock::SpinLock;

use super::types::Timespec;
use super::io::{console_write_bytes, read_stdin};

// ---------------------------------------------------------------
//  容量
// ---------------------------------------------------------------

/// 节点总数上限 (含根目录)。
const MAX_NODES: usize = 128;
/// 单个节点名 (含结尾空字节) 的最大长度。
const NAME_MAX: usize = 64;
/// 文件数据竞技场字节数。
const DATA_ARENA: usize = 128 * 1024;
/// 打开文件表容量 (fd 0..MAX_FD)。
const MAX_FD: usize = 64;
/// 路径字符串临时缓冲最大长度 (Linux PATH_MAX)。
const PATH_MAX: usize = 1024;

// ---------------------------------------------------------------
//  节点类型与哨兵
// ---------------------------------------------------------------

const KIND_DIR: u8 = 0;
const KIND_FILE: u8 = 1;
/// 空链接哨兵 (parent/first_child/next_sibling 与 fd 表项的 node 字段共用)。
const NIL: i16 = -1;

// ---------------------------------------------------------------
//  errno (Linux generic)
// ---------------------------------------------------------------

/// 把正 errno 编号折算为 "负 errno" 的 u64 表示 (与 musl 的 errno = -ret 约定一致)。
#[inline]
const fn errno(e: u64) -> u64 {
    (!0u64) - (e - 1)
}

const ENOENT: u64 = errno(2);
const EBADF: u64 = errno(9);
const ENOMEM: u64 = errno(12);
const EACCES: u64 = errno(13);
const EFAULT: u64 = errno(14);
const EEXIST: u64 = errno(17);
const ENOTDIR: u64 = errno(20);
const EISDIR: u64 = errno(21);
const EINVAL: u64 = errno(22);
const EMFILE: u64 = errno(24);
const ENOSPC: u64 = errno(28);
const ESPIPE: u64 = errno(29);
const ENAMETOOLONG: u64 = errno(36);
const ENOTEMPTY: u64 = errno(39);

// ---------------------------------------------------------------
//  open 标志 / 访问模式 / 权限位 / 目录类型
// ---------------------------------------------------------------

const O_ACCMODE: u32 = 0o3;
const O_RDONLY: u32 = 0o0;
const O_WRONLY: u32 = 0o1;
const O_RDWR: u32 = 0o2;
const O_CREAT: u32 = 0o100;
const O_EXCL: u32 = 0o200;
const O_TRUNC: u32 = 0o1000;
const O_APPEND: u32 = 0o2000;
const O_DIRECTORY: u32 = 0o200000;

// fcntl 命令
const F_DUPFD: u64 = 0;
const F_GETFD: u64 = 1;
const F_SETFD: u64 = 2;
const F_GETFL: u64 = 3;
const F_SETFL: u64 = 4;
const F_DUPFD_CLOEXEC: u64 = 1030;

// lseek 定位方式
const SEEK_SET: u64 = 0;
const SEEK_CUR: u64 = 1;
const SEEK_END: u64 = 2;

// 文件类型掩码与目录项类型
const S_IFMT: u32 = 0o170000;
const S_IFDIR: u32 = 0o040000;
const S_IFREG: u32 = 0o100000;
const S_IFCHR: u32 = 0o020000;

const DT_DIR: u8 = 4;
const DT_REG: u8 = 8;
const DT_CHR: u8 = 2;

/// AT_FDCWD (相对路径以当前工作目录为基准)。musl 的 stat/open 恒以此传参。
const AT_FDCWD: i64 = -100;

/// 控制台 fd (标准输入/输出/错误), 特殊处理, 不进入打开文件表。
const STDIN_FD: u64 = 0;
const STDOUT_FD: u64 = 1;
const STDERR_FD: u64 = 2;

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

// ---------------------------------------------------------------
//  节点 / 打开文件表 / 全局状态
// ---------------------------------------------------------------

/// ramfs 节点。目录以 first_child 指向孩子链表, 文件以 data_off/size 指向竞技场内的
/// 数据区。所有字段均为可静态零初始化的定长类型, 保证整个 FsState 落在 BSS。
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
            kind: KIND_FILE,
            parent: NIL,
            first_child: NIL,
            next_sibling: NIL,
            data_off: 0,
            size: 0,
            mode: 0,
        }
    }
}

/// 打开文件表项。node 为 NIL 表示空闲槽位。
#[derive(Clone, Copy)]
struct FdEntry {
    node: i16,
    offset: u64,
    flags: u32,
}

impl FdEntry {
    const fn empty() -> Self {
        Self { node: NIL, offset: 0, flags: 0 }
    }
}

struct FsState {
    nodes: [RamfsNode; MAX_NODES],
    data: [u8; DATA_ARENA],
    data_used: usize,
    fds: [FdEntry; MAX_FD],
    next_node: u16,
}

impl FsState {
    const fn empty() -> Self {
        Self {
            nodes: [RamfsNode::empty(); MAX_NODES],
            data: [0; DATA_ARENA],
            data_used: 0,
            fds: [FdEntry::empty(); MAX_FD],
            next_node: 0,
        }
    }
}

/// 全局文件系统状态。飞地 S-mode 运行在单 hart 上且系统调用不可重入 (中断只登记
/// 抢占、待系统调用返回后才切换线程), 用自旋锁仅为保持与共享资源访问约定一致,
/// 锁的持有区间不含任何 ecall 或让出, 不会自死锁。
static VFS: SpinLock<FsState> = SpinLock::new(FsState::empty());

// ---------------------------------------------------------------
//  工具: 稳定 inode 编号与节点类型
// ---------------------------------------------------------------

/// 以节点下标 + 1 作为稳定 inode (根目录 ino=1, 节点不被回收重排, 编号不变)。
#[inline]
fn ino_of(node_idx: i16) -> u64 {
    (node_idx as u64) + 1
}

/// 节点类型折算为目录项 d_type。
#[inline]
fn dtype_of(kind: u8) -> u8 {
    if kind == KIND_DIR {
        DT_DIR
    } else {
        DT_REG
    }
}

// ---------------------------------------------------------------
//  路径解析
// ---------------------------------------------------------------

/// 读取载荷地址空间内 NUL 结尾的字符串到 *out*, 返回写入的字节数 (含 NUL, 达到
/// 缓冲上限时截断)。
unsafe fn read_cstr(src: *const u8, out: &mut [u8]) -> usize {
    let mut n = 0;
    while n < out.len() {
        let b = unsafe { src.add(n).read_volatile() };
        out[n] = b;
        n += 1;
        if b == 0 {
            break;
        }
    }
    n
}

/// 从根目录沿 *path* 逐分量解析, 返回命中的节点下标, 未命中返回 NIL。
/// 支持 "." 与 ".." 分量, 根目录的父节点仍为根目录。
fn resolve(fs: &FsState, path: &[u8]) -> i16 {
    let mut cur: i16 = 0;
    let mut i = 0usize;
    while i < path.len() {
        while i < path.len() && path[i] == b'/' {
            i += 1;
        }
        if i >= path.len() {
            break;
        }
        let mut j = i;
        while j < path.len() && path[j] != b'/' {
            j += 1;
        }
        let comp = &path[i..j];
        if comp == b"." {
            // 当前目录, 不移动
        } else if comp == b".." {
            let p = fs.nodes[cur as usize].parent;
            if p != NIL {
                cur = p;
            }
        } else {
            let mut child = fs.nodes[cur as usize].first_child;
            let mut found = NIL;
            while child != NIL {
                let node = &fs.nodes[child as usize];
                if node.name_len as usize == comp.len() && &node.name[..comp.len()] == comp {
                    found = child;
                    break;
                }
                child = node.next_sibling;
            }
            if found == NIL {
                return NIL;
            }
            cur = found;
        }
        i = j;
    }
    cur
}

/// 解析 *path* 的父目录并返回末分量 (名字)。用于创建与删除: 需先定位父目录。
/// 返回 (父目录节点下标, 末分量切片)。
fn resolve_parent<'a>(fs: &FsState, path: &'a [u8]) -> Result<(i16, &'a [u8]), u64> {
    // 去除末尾 '/' (但保留根 "/")。
    let mut end = path.len();
    while end > 1 && path[end - 1] == b'/' {
        end -= 1;
    }
    let path = &path[..end];
    let last_slash = path.iter().rposition(|&c| c == b'/');
    let (dir_path, name) = match last_slash {
        None => (&b"/"[..], path),
        Some(idx) => (&path[..idx], &path[idx + 1..]),
    };
    if name.is_empty() || name.len() >= NAME_MAX {
        return Err(errno(ENAMETOOLONG));
    }
    let dir = resolve(fs, dir_path);
    if dir == NIL {
        return Err(errno(ENOENT));
    }
    if fs.nodes[dir as usize].kind != KIND_DIR {
        return Err(errno(ENOTDIR));
    }
    Ok((dir, name))
}

// ---------------------------------------------------------------
//  节点操作 (arena bump 分配, 无释放)
// ---------------------------------------------------------------

/// 在arena尾部分配 *len* 字节, 返回偏移; 空间不足返回 None。
fn arena_alloc(fs: &mut FsState, len: usize) -> Option<u32> {
    if fs.data_used + len > DATA_ARENA {
        return None;
    }
    let off = fs.data_used as u32;
    fs.data_used += len;
    Some(off)
}

/// 分配一个空节点, 返回其下标; 节点耗尽返回 NIL。
fn alloc_node(fs: &mut FsState, kind: u8, mode: u32) -> i16 {
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
fn create_child(fs: &mut FsState, parent: i16, name: &[u8], kind: u8, mode: u32) -> i16 {
    let child = alloc_node(fs, kind, mode);
    if child == NIL {
        return NIL;
    }
    // 先读出父目录孩子链表头, 避免与子节点的可变借用冲突 (E0503)。
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
fn find_child(fs: &FsState, dir: i16, name: &[u8]) -> i16 {
    let mut child = fs.nodes[dir as usize].first_child;
    while child != NIL {
        let node = &fs.nodes[child as usize];
        if node.name_len as usize == name.len() && &node.name[..name.len()] == name {
            return child;
        }
        child = node.next_sibling;
    }
    NIL
}

/// 从父目录孩子链表中摘除 *target*, 并释放其文件数据 (bump 分配无法真正归还,
/// 仅清空引用避免误读)。
fn remove_child(fs: &mut FsState, parent: i16, target: i16) {
    let mut prev = NIL;
    let mut child = fs.nodes[parent as usize].first_child;
    while child != NIL {
        if child == target {
            if prev == NIL {
                fs.nodes[parent as usize].first_child = fs.nodes[child as usize].next_sibling;
            } else {
                fs.nodes[prev as usize].next_sibling = fs.nodes[child as usize].next_sibling;
            }
            return;
        }
        prev = child;
        child = fs.nodes[child as usize].next_sibling;
    }
}

/// 把文件节点 *idx* 截断到 *len* 字节。仅收缩语义 (不扩展)。
fn truncate(fs: &mut FsState, idx: i16, len: u64) {
    let node = &mut fs.nodes[idx as usize];
    if len < node.size as u64 {
        node.size = len as u32;
    }
}

// ---------------------------------------------------------------
//  打开文件表
// ---------------------------------------------------------------

/// 分配一个 fd 并绑定到 *node*, 返回 fd; 失败返回负 errno。
fn alloc_fd(fs: &mut FsState, node: i16, flags: u32) -> u64 {
    for fd in 3..MAX_FD as u64 {
        if fs.fds[fd as usize].node == NIL {
            fs.fds[fd as usize].node = node;
            fs.fds[fd as usize].offset = 0;
            fs.fds[fd as usize].flags = flags;
            return fd;
        }
    }
    errno(EMFILE)
}

/// 释放 fd, 返回 0。
fn free_fd(fs: &mut FsState, fd: u64) {
    if fd < MAX_FD as u64 {
        fs.fds[fd as usize] = FdEntry::empty();
    }
}

/// 返回 fd 绑定的节点下标; 空闲或非法返回 None。
fn fd_node(fs: &FsState, fd: u64) -> Option<i16> {
    if fd < 3 || fd >= MAX_FD as u64 {
        return None;
    }
    let node = fs.fds[fd as usize].node;
    if node == NIL {
        None
    } else {
        Some(node)
    }
}

// ---------------------------------------------------------------
//  文件数据读写 (竞技场内)
// ---------------------------------------------------------------

/// 从文件 *idx* 的 *offset* 处读取至多 *len* 字节到载荷缓冲区 *buf*,
/// 返回实际读取的字节数。
unsafe fn read_file(fs: &FsState, idx: i16, offset: u64, buf: *mut u8, len: u64) -> u64 {
    let node = &fs.nodes[idx as usize];
    if offset >= node.size as u64 {
        return 0;
    }
    let avail = (node.size as u64 - offset) as usize;
    let n = cmp::min(avail, len as usize);
    let base = node.data_off as usize + offset as usize;
    for i in 0..n {
        unsafe { buf.add(i).write_volatile(fs.data[base + i]) };
    }
    n as u64
}

/// 向文件 *idx* 的 *offset* 处写入 *buf*, 必要时在竞技场内增长 (原地尾部增长或
/// 拷贝到新区域)。返回写入的字节数或负 errno (空间不足)。
unsafe fn write_file(fs: &mut FsState, idx: i16, offset: u64, buf: *const u8, len: u64) -> u64 {
    if len == 0 {
        return 0;
    }
    let end = offset as usize + len as usize;
    {
        let node = &fs.nodes[idx as usize];
        if end > node.size as usize {
            // 需要增长
            let old_off = node.data_off as usize;
            let old_size = node.size as usize;
            let in_tail = old_off + old_size == fs.data_used;
            if in_tail && old_off + end <= DATA_ARENA {
                // 竞技场尾部原地增长
                fs.data_used += end - old_size;
            } else if fs.data_used + end <= DATA_ARENA {
                // 拷贝到新区域, 旧区域泄漏 (bump 分配无回收)
                let new_off = fs.data_used;
                for i in 0..old_size {
                    fs.data[new_off + i] = fs.data[old_off + i];
                }
                fs.data_used += end;
                let node = &mut fs.nodes[idx as usize];
                node.data_off = new_off as u32;
            } else {
                return errno(ENOSPC);
            }
            let node = &mut fs.nodes[idx as usize];
            node.size = end as u32;
        }
    }
    let node = &fs.nodes[idx as usize];
    let base = node.data_off as usize;
    for i in 0..len as usize {
        let b = unsafe { buf.add(i).read_volatile() };
        fs.data[base + offset as usize + i] = b;
    }
    len
}

// ---------------------------------------------------------------
//  初始化与载荷注入
// ---------------------------------------------------------------

/// 建立根目录。幂等: 重复调用不重建。
pub fn vfs_init() {
    let mut fs = VFS.lock();
    if fs.next_node == 0 {
        // 根目录为节点 0
        fs.next_node = 1;
        let root = &mut fs.nodes[0];
        root.kind = KIND_DIR;
        root.mode = 0o755;
        root.name[0] = b'/';
        root.name_len = 1;
    }
}

/// 把 *data* 作为文件 *path* 注入文件系统, 沿路目录按需创建。
/// 供启动阶段 (载荷运行前) 用宿主下发的数据预置文件。
pub fn vfs_inject_file(path: &[u8], data: &[u8]) -> bool {
    let mut fs = VFS.lock();
    // 逐级创建目录, 末分量为文件名
    let mut i = 0usize;
    let mut cur: i16 = 0;
    while i < path.len() {
        while i < path.len() && path[i] == b'/' {
            i += 1;
        }
        if i >= path.len() {
            break;
        }
        let mut j = i;
        while j < path.len() && path[j] != b'/' {
            j += 1;
        }
        let comp = &path[i..j];
        let is_last = {
            let mut k = j;
            while k < path.len() && path[k] == b'/' {
                k += 1;
            }
            k >= path.len()
        };
        let kind = if is_last { KIND_FILE } else { KIND_DIR };
        let existing = find_child(&fs, cur, comp);
        let node = if existing != NIL {
            existing
        } else {
            let mode = if kind == KIND_DIR { 0o755 } else { 0o644 };
            let created = create_child(&mut fs, cur, comp, kind, mode);
            if created == NIL {
                return false;
            }
            created
        };
        cur = node;
        i = j;
    }
    // 写入文件数据
    if cur == 0 {
        return false;
    }
    let off = match arena_alloc(&mut fs, data.len()) {
        Some(o) => o,
        None => return false,
    };
    for (i, &b) in data.iter().enumerate() {
        fs.data[off as usize + i] = b;
    }
    let node = &mut fs.nodes[cur as usize];
    node.data_off = off;
    node.size = data.len() as u32;
    true
}

// ---------------------------------------------------------------
//  stat 填充
// ---------------------------------------------------------------

/// 按节点 *idx* 填充 stat。
fn fill_stat(fs: &FsState, idx: i16, st: &mut Stat) {
    let node = &fs.nodes[idx as usize];
    st.st_ino = ino_of(idx);
    st.st_mode = if node.kind == KIND_DIR {
        S_IFDIR | (node.mode & 0o777)
    } else {
        S_IFREG | (node.mode & 0o777)
    };
    st.st_nlink = if node.kind == KIND_DIR {
        // 目录链接数 = 2 (自身 + 父目录中的 ".") + 子目录数, 简化取 2.
        2
    } else {
        1
    };
    st.st_size = node.size as i64;
    st.st_blocks = (node.size as i64 + 511) / 512;
}

/// 填充控制台 (字符设备) stat。
fn fill_stat_console(st: &mut Stat) {
    st.st_mode = S_IFCHR | 0o600;
    st.st_nlink = 1;
    st.st_size = 0;
    st.st_blocks = 0;
}

// ---------------------------------------------------------------
//  系统调用处理器
// ---------------------------------------------------------------

/// openat(56): dirfd + pathname + flags + mode, 返回新 fd。
pub fn openat_handler(dirfd: u64, pathname: u64, flags: u64, mode: u64) -> u64 {
    let pathname = pathname as *const u8;
    if pathname.is_null() {
        return EFAULT;
    }
    let mut path_buf = [0u8; PATH_MAX];
    let plen = unsafe { read_cstr(pathname, &mut path_buf) };
    // 去掉结尾 NUL (read_cstr 返回含 NUL 的长度)
    let path = &path_buf[..plen.saturating_sub(1)];
    let flags = flags as u32;
    let mode = (mode & 0o777) as u32;

    // 相对路径 + 非 AT_FDCWD 的 dirfd 暂不支持 (musl 恒以 AT_FDCWD 或绝对路径调用)。
    if (dirfd as i64) != AT_FDCWD && !path.starts_with(b"/") {
        return errno(ENOTDIR);
    }

    let mut fs = VFS.lock();

    // 打开根目录自身 (path 为 "/" 或空)
    let mut node = if path.is_empty() || path == b"/" {
        0i16
    } else {
        resolve(&fs, path)
    };

    if node == NIL {
        // 未命中: 仅在有 O_CREAT 时创建
        if flags & O_CREAT == 0 {
            return errno(ENOENT);
        }
        let (dir, name) = match resolve_parent(&fs, path) {
            Ok(v) => v,
            Err(e) => return e,
        };
        let created = create_child(&mut fs, dir, name, KIND_FILE, mode);
        if created == NIL {
            return errno(ENOMEM);
        }
        node = created;
    } else {
        // 命中: 校验排他与目录
        if flags & O_CREAT != 0 && flags & O_EXCL != 0 {
            return errno(EEXIST);
        }
        let kind = fs.nodes[node as usize].kind;
        if flags & O_DIRECTORY != 0 && kind != KIND_DIR {
            return errno(ENOTDIR);
        }
        // 写访问 + O_TRUNC 时截断 (目录不可截断)
        if flags & O_TRUNC != 0 && flags & O_ACCMODE != O_RDONLY && kind == KIND_FILE {
            truncate(&mut fs, node, 0);
        }
    }

    alloc_fd(&mut fs, node, flags)
}

/// close(57): 释放 fd。控制台 fd 为无操作。
pub fn close_handler(fd: u64) -> u64 {
    if fd <= STDERR_FD {
        return 0;
    }
    let mut fs = VFS.lock();
    free_fd(&mut fs, fd);
    0
}

/// read(63): 从 fd 读入 *len* 字节到 *buf*, 返回实际读取字节数。
pub fn read_handler(fd: u64, buf: *mut u8, len: u64) -> u64 {
    if fd == STDIN_FD {
        return unsafe { read_stdin(buf, len) };
    }
    let mut fs = VFS.lock();
    let node = match fd_node(&fs, fd) {
        Some(n) => n,
        None => return EBADF,
    };
    let offset = fs.fds[fd as usize].offset;
    let n = unsafe { read_file(&fs, node, offset, buf, len) };
    fs.fds[fd as usize].offset += n;
    n
}

/// write(64): 向 fd 写出 *len* 字节, 返回实际写入字节数。
pub fn write_handler(fd: u64, buf: *const u8, len: u64) -> u64 {
    if fd == STDOUT_FD || fd == STDERR_FD {
        return unsafe { console_write_bytes(buf, len) };
    }
    let mut fs = VFS.lock();
    let node = match fd_node(&fs, fd) {
        Some(n) => n,
        None => return EBADF,
    };
    // O_APPEND 时写入偏移为文件末尾
    let offset = if fs.fds[fd as usize].flags & O_APPEND != 0 {
        fs.nodes[node as usize].size as u64
    } else {
        fs.fds[fd as usize].offset
    };
    let n = unsafe { write_file(&mut fs, node, offset, buf, len) };
    if n < (!0u64) >> 1 {
        // 非负 errno (成功): 推进偏移
        fs.fds[fd as usize].offset = offset + n;
    }
    n
}

/// writev(66): 汇集多个 iovec 写出, 返回总字节数。
pub fn writev_handler(fd: u64, io_vec_arr: u64, io_vec_size: u64) -> u64 {
    if fd == STDOUT_FD || fd == STDERR_FD {
        let mut total: u64 = 0;
        for i in 0..io_vec_size {
            let slot = io_vec_arr.wrapping_add(i.wrapping_mul(16));
            let base = unsafe { (slot as *const u64).read_volatile() };
            let len = unsafe { (slot.wrapping_add(8) as *const u64).read_volatile() };
            unsafe { console_write_bytes(base as *const u8, len) };
            total = total.wrapping_add(len);
        }
        return total;
    }
    // 文件 fd: 逐段写
    let mut fs = VFS.lock();
    let node = match fd_node(&fs, fd) {
        Some(n) => n,
        None => return EBADF,
    };
    let mut offset = if fs.fds[fd as usize].flags & O_APPEND != 0 {
        fs.nodes[node as usize].size as u64
    } else {
        fs.fds[fd as usize].offset
    };
    let mut total: u64 = 0;
    for i in 0..io_vec_size {
        let slot = io_vec_arr.wrapping_add(i.wrapping_mul(16));
        let base = unsafe { (slot as *const u64).read_volatile() };
        let len = unsafe { (slot.wrapping_add(8) as *const u64).read_volatile() };
        let n = unsafe { write_file(&mut fs, node, offset, base as *const u8, len) };
        if n >= (!0u64) >> 1 {
            // 写入失败 (负 errno): 返回已写字节数
            break;
        }
        offset += n;
        total += n;
    }
    fs.fds[fd as usize].offset = offset;
    total
}

/// readv(65): 汇集多个 iovec 读出, 返回总字节数。
pub fn readv_handler(fd: u64, io_vec_arr: u64, io_vec_size: u64) -> u64 {
    if fd != STDIN_FD {
        // 非 stdin: 仅支持文件 fd
        let mut fs = VFS.lock();
        let node = match fd_node(&fs, fd) {
            Some(n) => n,
            None => return EBADF,
        };
        let mut offset = fs.fds[fd as usize].offset;
        let mut total: u64 = 0;
        for i in 0..io_vec_size {
            let slot = io_vec_arr.wrapping_add(i.wrapping_mul(16));
            let base = unsafe { (slot as *const u64).read_volatile() };
            let len = unsafe { (slot.wrapping_add(8) as *const u64).read_volatile() };
            let n = unsafe { read_file(&fs, node, offset, base as *mut u8, len) };
            if n == 0 {
                break;
            }
            offset += n;
            total += n;
        }
        fs.fds[fd as usize].offset = offset;
        return total;
    }
    // stdin: 逐段读控制台 (与 read_stdin 同理)
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
    total
}

/// lseek(62): 重定位 fd 偏移, 返回新偏移。
pub fn lseek_handler(fd: u64, offset: u64, whence: u64) -> u64 {
    if fd <= STDERR_FD {
        return 0; // 控制台无偏移, 保持旧行为避免破坏 musl stdio 退出路径
    }
    let mut fs = VFS.lock();
    let node = match fd_node(&fs, fd) {
        Some(n) => n,
        None => return EBADF,
    };
    let size = fs.nodes[node as usize].size as u64;
    let new_off = match whence {
        SEEK_SET => offset,
        SEEK_CUR => fs.fds[fd as usize].offset.wrapping_add(offset),
        SEEK_END => size.wrapping_add(offset),
        _ => return EINVAL,
    };
    fs.fds[fd as usize].offset = new_off;
    new_off
}

/// pread64(67): 在指定偏移读取, 不改变 fd 偏移。
pub fn pread64_handler(fd: u64, buf: *mut u8, len: u64, offset: u64) -> u64 {
    let fs = VFS.lock();
    let node = match fd_node(&fs, fd) {
        Some(n) => n,
        None => return EBADF,
    };
    unsafe { read_file(&fs, node, offset, buf, len) }
}

/// pwrite64(68): 在指定偏移写入, 不改变 fd 偏移。
pub fn pwrite64_handler(fd: u64, buf: *const u8, len: u64, offset: u64) -> u64 {
    let mut fs = VFS.lock();
    let node = match fd_node(&fs, fd) {
        Some(n) => n,
        None => return EBADF,
    };
    unsafe { write_file(&mut fs, node, offset, buf, len) }
}

/// fstat(80): 按 fd 填充 stat。
pub fn fstat_handler(fd: u64, stat_ptr: u64) -> u64 {
    let stat_ptr = stat_ptr as *mut Stat;
    if stat_ptr.is_null() {
        return EFAULT;
    }
    let mut st = Stat::zero();
    if fd <= STDERR_FD {
        fill_stat_console(&mut st);
    } else {
        let fs = VFS.lock();
        let node = match fd_node(&fs, fd) {
            Some(n) => n,
            None => return EBADF,
        };
        fill_stat(&fs, node, &mut st);
    }
    unsafe { ptr::write_volatile(stat_ptr, st) };
    0
}

/// fstatat(79): 按路径填充 stat。musl 的 stat/lstat 均落到此调用。
pub fn fstatat_handler(dirfd: u64, pathname: u64, stat_ptr: u64, _flags: u64) -> u64 {
    let pathname = pathname as *const u8;
    let stat_ptr = stat_ptr as *mut Stat;
    if pathname.is_null() || stat_ptr.is_null() {
        return EFAULT;
    }
    let mut path_buf = [0u8; PATH_MAX];
    let plen = unsafe { read_cstr(pathname, &mut path_buf) };
    let path = &path_buf[..plen.saturating_sub(1)];
    if (dirfd as i64) != AT_FDCWD && !path.starts_with(b"/") {
        return errno(ENOTDIR);
    }
    let fs = VFS.lock();
    let node = if path.is_empty() || path == b"/" {
        0i16
    } else {
        resolve(&fs, path)
    };
    if node == NIL {
        return errno(ENOENT);
    }
    let mut st = Stat::zero();
    fill_stat(&fs, node, &mut st);
    unsafe { ptr::write_volatile(stat_ptr, st) };
    0
}

/// getdents(61): 读取目录项到 *buf*, 返回写入字节数。
pub fn getdents_handler(fd: u64, buf: *mut u8, count: u64) -> u64 {
    let mut fs = VFS.lock();
    let node = match fd_node(&fs, fd) {
        Some(n) => n,
        None => return EBADF,
    };
    if fs.nodes[node as usize].kind != KIND_DIR {
        return errno(ENOTDIR);
    }
    let mut pos = fs.fds[fd as usize].offset;
    let mut written: usize = 0;
    loop {
        if written >= count as usize {
            break;
        }
        // 确定当前条目: "." / ".." / 第 (pos-2) 个孩子
        let mut name_buf = [0u8; NAME_MAX];
        let (ino, dtype, namelen): (u64, u8, usize) = if pos == 0 {
            name_buf[0] = b'.';
            (ino_of(node), DT_DIR, 1)
        } else if pos == 1 {
            name_buf[0] = b'.';
            name_buf[1] = b'.';
            let parent = fs.nodes[node as usize].parent;
            let p = if parent == NIL { node } else { parent };
            (ino_of(p), DT_DIR, 2)
        } else {
            let mut child = fs.nodes[node as usize].first_child;
            let mut k = pos - 2;
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
        // Linux dirent64: d_ino(8) + d_off(8) + d_reclen(2) + d_type(1) + d_name(NUL),
        // d_reclen 按 8 字节对齐。
        let reclen = (19 + namelen + 1 + 7) & !7;
        if written + reclen > count as usize {
            break;
        }
        unsafe {
            let entry = buf.add(written);
            (entry as *mut u64).write_volatile(ino);
            (entry.add(8) as *mut i64).write_volatile(pos as i64);
            (entry.add(16) as *mut u16).write_volatile(reclen as u16);
            (entry.add(18) as *mut u8).write_volatile(dtype);
            for i in 0..namelen {
                entry.add(19 + i).write_volatile(name_buf[i]);
            }
            entry.add(19 + namelen).write_volatile(0u8);
        }
        written += reclen;
        pos += 1;
    }
    fs.fds[fd as usize].offset = pos;
    written as u64
}

/// getcwd(17): 返回当前工作目录 (单根, 恒为 "/")。
pub fn getcwd_handler(buf: u64, size: u64) -> u64 {
    if buf == 0 {
        return EFAULT;
    }
    if size < 2 {
        return errno(EINVAL);
    }
    unsafe {
        (buf as *mut u8).write_volatile(b'/');
        (buf as *mut u8).add(1).write_volatile(0u8);
    }
    1
}

/// mkdirat(34): 创建目录。
pub fn mkdirat_handler(dirfd: u64, pathname: u64, mode: u64) -> u64 {
    let pathname = pathname as *const u8;
    if pathname.is_null() {
        return EFAULT;
    }
    let mut path_buf = [0u8; PATH_MAX];
    let plen = unsafe { read_cstr(pathname, &mut path_buf) };
    let path = &path_buf[..plen.saturating_sub(1)];
    if (dirfd as i64) != AT_FDCWD && !path.starts_with(b"/") {
        return errno(ENOTDIR);
    }
    let mut fs = VFS.lock();
    let (dir, name) = match resolve_parent(&fs, path) {
        Ok(v) => v,
        Err(e) => return e,
    };
    if find_child(&fs, dir, name) != NIL {
        return errno(EEXIST);
    }
    let created = create_child(&mut fs, dir, name, KIND_DIR, (mode & 0o777) as u32);
    if created == NIL {
        return errno(ENOMEM);
    }
    0
}

/// unlinkat(35): 删除文件 (或空目录, 若 flags 含 AT_REMOVEDIR)。
pub fn unlinkat_handler(dirfd: u64, pathname: u64, _flags: u64) -> u64 {
    let pathname = pathname as *const u8;
    if pathname.is_null() {
        return EFAULT;
    }
    let mut path_buf = [0u8; PATH_MAX];
    let plen = unsafe { read_cstr(pathname, &mut path_buf) };
    let path = &path_buf[..plen.saturating_sub(1)];
    if (dirfd as i64) != AT_FDCWD && !path.starts_with(b"/") {
        return errno(ENOTDIR);
    }
    let mut fs = VFS.lock();
    let (dir, name) = match resolve_parent(&fs, path) {
        Ok(v) => v,
        Err(e) => return e,
    };
    let target = find_child(&fs, dir, name);
    if target == NIL {
        return errno(ENOENT);
    }
    if fs.nodes[target as usize].kind == KIND_DIR {
        // 仅空目录可删
        if fs.nodes[target as usize].first_child != NIL {
            return errno(ENOTEMPTY);
        }
    }
    remove_child(&mut fs, dir, target);
    0
}

/// faccessat(48): 检查路径存在性 (R_OK/W_OK 简化: 文件恒可读, 写检查恒放行)。
pub fn faccessat_handler(dirfd: u64, pathname: u64, _mode: u64, _flags: u64) -> u64 {
    let pathname = pathname as *const u8;
    if pathname.is_null() {
        return EFAULT;
    }
    let mut path_buf = [0u8; PATH_MAX];
    let plen = unsafe { read_cstr(pathname, &mut path_buf) };
    let path = &path_buf[..plen.saturating_sub(1)];
    if (dirfd as i64) != AT_FDCWD && !path.starts_with(b"/") {
        return errno(ENOTDIR);
    }
    let fs = VFS.lock();
    let node = if path.is_empty() || path == b"/" {
        0i16
    } else {
        resolve(&fs, path)
    };
    if node == NIL {
        return errno(ENOENT);
    }
    0
}

/// fcntl(25): 描述符标志查询/复制。musl 的 fopen 依赖 F_SETFD 成功。
pub fn fcntl_handler(fd: u64, cmd: u64, arg: u64) -> u64 {
    match cmd {
        F_GETFD => 0,
        F_SETFD => 0,
        F_GETFL => {
            if fd <= STDERR_FD {
                0
            } else {
                let fs = VFS.lock();
                match fd_node(&fs, fd) {
                    Some(_) => fs.fds[fd as usize].flags as u64,
                    None => EBADF,
                }
            }
        }
        F_SETFL => {
            if fd <= STDERR_FD {
                0
            } else {
                let mut fs = VFS.lock();
                if fd_node(&fs, fd).is_none() {
                    return EBADF;
                }
                // 仅保留 O_APPEND (访问模式不可改)
                fs.fds[fd as usize].flags =
                    (fs.fds[fd as usize].flags & O_ACCMODE) | (arg as u32 & O_APPEND);
                0
            }
        }
        F_DUPFD | F_DUPFD_CLOEXEC => {
            if fd <= STDERR_FD {
                return EINVAL;
            }
            let mut fs = VFS.lock();
            let node = match fd_node(&fs, fd) {
                Some(n) => n,
                None => return EBADF,
            };
            let flags = fs.fds[fd as usize].flags;
            // 从 arg (或当前 fd+1) 起找第一个空闲槽
            let mut new_fd = arg.max(fd + 1);
            while new_fd < MAX_FD as u64 && fs.fds[new_fd as usize].node != NIL {
                new_fd += 1;
            }
            if new_fd >= MAX_FD as u64 {
                return errno(EMFILE);
            }
            fs.fds[new_fd as usize].node = node;
            fs.fds[new_fd as usize].offset = 0;
            fs.fds[new_fd as usize].flags = flags;
            new_fd
        }
        _ => EINVAL,
    }
}

/// fsync(82): 无持久化后端, 立即成功。
pub fn fsync_handler(_fd: u64) -> u64 {
    0
}

/// ftruncate(46): 截断文件到指定长度。
pub fn ftruncate_handler(fd: u64, length: u64) -> u64 {
    if fd <= STDERR_FD {
        return EINVAL;
    }
    let mut fs = VFS.lock();
    let node = match fd_node(&fs, fd) {
        Some(n) => n,
        None => return EBADF,
    };
    if fs.nodes[node as usize].kind != KIND_FILE {
        return EINVAL;
    }
    truncate(&mut fs, node, length);
    0
}

/// ppoll(73): 查询描述符就绪状态。控制台 fd 0 可读, 1/2 可写, 文件 fd 恒可读可写。
pub fn ppoll_handler(fds: u64, nfds: u64, _timeout: u64) -> u64 {
    if fds == 0 {
        return EFAULT;
    }
    const POLLIN: u16 = 0x0001;
    const POLLOUT: u16 = 0x0004;
    const POLLNVAL: u16 = 0x0020;

    let fs = VFS.lock();
    let mut ready: u64 = 0;
    for i in 0..nfds {
        let slot = fds.wrapping_add(i.wrapping_mul(8));
        let fd = unsafe { (slot as *const i32).read_volatile() } as i64;
        let events = unsafe { (slot.wrapping_add(4) as *const u16).read_volatile() };
        let revents = if fd >= 0 && (fd as u64) <= STDERR_FD {
            match fd as u64 {
                STDIN_FD => events & POLLIN,
                _ => events & POLLOUT,
            }
        } else if fd >= 0 && fd_node(&fs, fd as u64).is_some() {
            // 文件 fd: 恒可读可写 (未到 EOF 视为可读)
            events & (POLLIN | POLLOUT)
        } else {
            POLLNVAL
        };
        unsafe { (slot.wrapping_add(6) as *mut u16).write_volatile(revents) };
        if revents != 0 {
            ready += 1;
        }
    }
    ready
}
