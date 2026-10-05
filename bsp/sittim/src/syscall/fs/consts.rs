//! 文件系统层的常量: 容量、打开标志、fcntl 命令、定位基准、文件类型编码与各调用共用的
//! 标志位掩码。
//!
//! 执行时关闭位属描述符自身, 其值与掩码定义在描述符表层 (见 [super::super::fdtable] 的
//! `O_CLOEXEC` 与 `FD_FLAG_MASK`), 本节只在打开标志的掩码中引用它。

use super::super::fdtable::O_CLOEXEC;
use crate::ext_mod::vfs_ops::{O_CREAT, O_EXCL, O_TRUNC};

// ---------------------------------------------------------------
//  容量
// ---------------------------------------------------------------

/// 路径字符串临时缓冲最大长度 (Linux PATH_MAX)。
pub(super) const PATH_MAX: usize = 1024;
/// readv 与 writev 接受的 iovec 项数上限 (Linux UIO_MAXIOV)。
pub(super) const IOV_MAX: u64 = 1024;

/// ppoll 的阻塞地址。等待描述符就绪的线程都阻塞于同一个值: 就绪状态来自协议栈,
/// 与描述符无关, 故网络模块在设备事件到达时按该地址整体唤醒 (见 concurrency/sched.rs)。
pub const POLL_WAIT_ADDR: u64 = 0x7fff_2000_0000;

/// 文件系统节点下标的空值, 由模块的操作表定义。
pub(super) const NIL: i16 = -1;

// ---------------------------------------------------------------
//  open 标志与访问模式
// ---------------------------------------------------------------

// 要与文件系统模块共用的标志定义在 crate::ext_mod::vfs_ops: 运行时把它们原样转交模块。
// 只有转交不到模块的那几位留在本层。

/// 写入偏移取文件末尾。
pub(super) const O_APPEND: u32 = 0o2000;
/// 非阻塞标志, 与 SOCK_NONBLOCK 取同一个位值。
pub const O_NONBLOCK: u32 = 0o4000;
/// 不把打开的设备作为控制终端。本层不使用该位, 声明它是为了把它自状态标志中除去。
pub(super) const O_NOCTTY: u32 = 0o400;
/// 只在 open 的判断中使用、不属于打开文件描述的标志位。F_GETFL 回报描述的状态标志,
/// 故建立描述时把这些位除去。
pub(super) const OPEN_ONLY_FLAGS: u32 = O_CREAT | O_EXCL | O_NOCTTY | O_TRUNC | O_CLOEXEC;
/// fcntl 的 FD_CLOEXEC 位, 与 `O_CLOEXEC` 指代同一状态。
pub(super) const FD_CLOEXEC: u64 = 1;

// fcntl 命令
pub(super) const F_DUPFD: u64 = 0;
pub(super) const F_GETFD: u64 = 1;
pub(super) const F_SETFD: u64 = 2;
pub(super) const F_GETFL: u64 = 3;
pub(super) const F_SETFL: u64 = 4;
pub(super) const F_DUPFD_CLOEXEC: u64 = 1030;

/// F_SETFL 可改动的状态标志位, 对应 Linux 的 SETFL_MASK 中本层建模的两位。
pub(super) const SETFL_MASK: u32 = O_APPEND | O_NONBLOCK;

// lseek 定位方式
pub(super) const SEEK_SET: u64 = 0;
pub(super) const SEEK_CUR: u64 = 1;
pub(super) const SEEK_END: u64 = 2;

// 文件类型编码 (musl 的 `struct stat` 的 st_mode 高位)
pub(super) const S_IFDIR: u32 = 0o040000;
pub(super) const S_IFREG: u32 = 0o100000;
pub(super) const S_IFCHR: u32 = 0o020000;
pub(super) const S_IFSOCK: u32 = 0o140000;

/// AT_FDCWD (相对路径以当前工作目录为基准)。musl 的 stat/open 恒以此传参。
pub(super) const AT_FDCWD: i64 = -100;

/// faccessat 的 *mode* 允许的位: 读 (4), 写 (2), 执行 (1)。存在性检查 F_OK 取 0。
pub(super) const ACCESS_MODE_BITS: u64 = 0o7;
/// 按有效用户而非实际用户判定。本层没有用户模型, 该位不改变结果。
pub(super) const AT_EACCESS: u64 = 0x200;
/// 不跟随符号链接。本层没有符号链接, 该位不改变结果。
pub(super) const AT_SYMLINK_NOFOLLOW: u64 = 0x100;
/// 路径取空时以 *dirfd* 自身为目标。本层按该位取描述符绑定的节点。
pub(super) const AT_EMPTY_PATH: u64 = 0x1000;
/// 不触发自动挂载。本层没有挂载点, 该位不改变结果。
pub(super) const AT_NO_AUTOMOUNT: u64 = 0x800;
/// 缓存同步方式 (强制同步与不强制同步两位)。本层没有缓存, 该位不改变结果。
pub(super) const AT_STATX_SYNC_TYPE: u64 = 0x6000;

/// faccessat 的 *flags* 允许的位。
pub(super) const ACCESS_FLAG_BITS: u64 = AT_EACCESS | AT_SYMLINK_NOFOLLOW | AT_EMPTY_PATH;
/// fstatat 的 *flags* 允许的位。
pub(super) const STAT_FLAG_BITS: u64 =
	AT_SYMLINK_NOFOLLOW | AT_NO_AUTOMOUNT | AT_EMPTY_PATH | AT_STATX_SYNC_TYPE;

/// 建立的控制台表项个数 (标准输入, 标准输出, 标准错误), 占用 fd 0 至 2。
pub(super) const CONSOLE_FDS: u64 = 3;
