//! 文件系统模块导出的操作表。
//!
//! 运行时与文件系统模块共用本文件: 运行时以 `mod vfs_ops` 引入, 模块以 `#[path]` 引入
//! 同一份定义, 两侧不可能漂移。
//!
//! 表内各项不是系统调用, 而是运行时自身的调用点。目录树与文件数据由模块持有, 其存储
//! 形态只在模块内, 故节点以不透明的 i16 下标跨越边界; fd 表由运行时持有, 控制台与
//! 套接字也在运行时一侧实现, 故本表不含 fd 参数, 也不涉及这两类对象。
//!
//! 各项的结果取 [`VfsStatus`], 折算为 errno 由运行时完成; 模块内不出现 errno 编号, 两侧
//! 因此不必共用一套错误码。形态与 ref-impl/emod 的 `emod_vfs_api_t` 一致。

// ---------------------------------------------------------------
//  打开标志
// ---------------------------------------------------------------

// 下列标志即载荷传给 openat 的 flags 字, 其取值与 Linux 一致, 运行时原样转交模块,
// 故两侧共用同一份定义。

/// 访问模式位。
pub const O_ACCMODE: u32 = 0o3;
/// 只读打开。
pub const O_RDONLY: u32 = 0o0;
/// 只写打开。
pub const O_WRONLY: u32 = 0o1;
/// 读写打开。
pub const O_RDWR: u32 = 0o2;
/// 路径不存在时创建。
pub const O_CREAT: u32 = 0o100;
/// 与 [`O_CREAT`] 同用, 路径已存在时失败。
pub const O_EXCL: u32 = 0o200;
/// 打开后截断到 0 字节。
pub const O_TRUNC: u32 = 0o1000;
/// 要求路径末级是目录。
pub const O_DIRECTORY: u32 = 0o200000;

/// 删除目录的标志, 取自 unlinkat 的 flags 字, 与 Linux 的 AT_REMOVEDIR 一致。
pub const AT_REMOVEDIR: u32 = 0x200;

/// 打开标志的访问模式位是否允许读出。`O_RDONLY` 与 `O_RDWR` 允许, `O_WRONLY` 不允许。
pub const fn mode_readable(flags: u32) -> bool {
	flags & O_ACCMODE != O_WRONLY
}

/// 打开标志的访问模式位是否允许写入。`O_WRONLY` 与 `O_RDWR` 允许, `O_RDONLY` 不允许。
pub const fn mode_writable(flags: u32) -> bool {
	flags & O_ACCMODE != O_RDONLY
}

// ---------------------------------------------------------------
//  节点类别
// ---------------------------------------------------------------

/// 节点是目录。
pub const VFS_KIND_DIR: u8 = 0;
/// 节点是文件。
pub const VFS_KIND_FILE: u8 = 1;

/// 操作的结果。运行时按本枚举的取值折算为 errno。
///
/// 取值的编码跨模块映像与运行时的边界传递, 新增取值一律追加在末尾, 既有取值的编码保持不变。
#[repr(u32)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum VfsStatus {
	/// 成功。
	Ok = 0,
	/// 路径或路径的某一级不存在。
	NoEntry,
	/// 创建的目标已存在。
	Exists,
	/// 路径的某一级不是目录, 或末级不是所要求的目录。
	NotDir,
	/// 目录非空, 不可删除。
	NotEmpty,
	/// 节点已用尽。
	NoMemory,
	/// 文件数据区已用尽。
	NoSpace,
	/// 路径的某一级超过名字长度上限。
	NameTooLong,
	/// 参数与对象的类别不符。
	Invalid,
	/// 宿主未提供本模块, 该操作无实现可用。
	NoSys,
	/// 末级是目录, 而该操作只作用于文件。
	IsDir,
	/// 目录是根目录, 不可删除。
	Busy,
}

/// 节点的 stat 字段。
///
/// `mode` 只承载权限位, 文件类型的编码由运行时按 `kind` 补齐: 该编码属于载荷侧的 ABI
/// (musl 的 `struct stat`), 不是模块的语义。
#[repr(C)]
#[derive(Clone, Copy)]
pub struct VfsStat {
	pub ino: u64,
	pub kind: u8,
	pub mode: u32,
	pub nlink: u32,
	pub size: i64,
	pub blocks: i64,
}

impl VfsStat {
	/// 构建一个字段全零的 stat, 供调用方在传入 [`VfsOps::stat`] 之前初始化。
	pub const fn zero() -> Self {
		Self { ino: 0, kind: VFS_KIND_FILE, mode: 0, nlink: 0, size: 0, blocks: 0 }
	}
}

/// 文件系统模块导出的操作表。
///
/// 表由模块在取入时填写, 且必须在模块的入口内填写而不能作为静态初始化项: 静态初始化
/// 会把函数地址作为绝对常量写进映像, 而映像在链接时不知道自己的加载基址。
#[repr(C)]
pub struct VfsOps {
	/// 建立根目录。可重复调用, 已建立时同样返回真。
	pub init: unsafe extern "C" fn() -> bool,
	/// 把 [data, data + data_len) 作为文件 *path* 注入, 沿路目录按需创建。
	pub inject_file: unsafe extern "C" fn(*const u8, u64, *const u8, u64) -> bool,
	/// 打开 *path*, 成功时把节点下标写入 *out_node*。*flags* 取 `O_*` 的组合, *mode*
	/// 为新建节点的权限位。
	pub open: unsafe extern "C" fn(*const u8, u64, u32, u32, *mut i16) -> VfsStatus,
	/// 取 *path* 的节点下标, 成功时写入 *out_node*。*path* 取空或 "/" 时给出根目录。
	pub resolve: unsafe extern "C" fn(*const u8, u64, *mut i16) -> VfsStatus,
	/// 从文件 *node* 的 *offset* 处读取至多 *len* 字节到 [buf, buf + len), 成功时把实际
	/// 读取的字节数写入 *out_read*。末级是目录时返回
	/// [`IsDir`](VfsStatus::IsDir)。
	pub read: unsafe extern "C" fn(i16, u64, *mut u8, u64, *mut u64) -> VfsStatus,
	/// 向文件 *node* 的 *offset* 处写入 [buf, buf + len), 成功时把写入的字节数写入
	/// *out_written*。末级是目录时返回 [`IsDir`](VfsStatus::IsDir)。
	pub write: unsafe extern "C" fn(i16, u64, *const u8, u64, *mut u64) -> VfsStatus,
	/// 取节点 *node* 的 stat 字段; 节点不存在时返回假。
	pub stat: unsafe extern "C" fn(i16, *mut VfsStat) -> bool,
	/// 把文件 *node* 截断到 *len* 字节。节点不是文件时返回
	/// [`Invalid`](VfsStatus::Invalid)。
	pub truncate: unsafe extern "C" fn(i16, u64) -> VfsStatus,
	/// 读取目录 *node* 的目录项到 [buf, buf + count), 从 *pos* 处的条目起读, 并把新的
	/// 读取位置写回 *pos*; 成功时把写入的字节数写入 *out_written*。末级不是目录时返回
	/// [`NotDir`](VfsStatus::NotDir), 下标无效或缓冲区容纳不下更多条目时返回
	/// [`Invalid`](VfsStatus::Invalid)。
	pub getdents: unsafe extern "C" fn(i16, *mut u64, *mut u8, u64, *mut u64) -> VfsStatus,
	/// 创建目录 *path*, *mode* 为权限位。
	pub mkdir: unsafe extern "C" fn(*const u8, u64, u32) -> VfsStatus,
	/// 删除 *path* 所指的对象。*flags* 含 [`AT_REMOVEDIR`] 时目标须是目录, 不含时目标须是
	/// 文件, 与 unlinkat 的约定一致。
	pub unlink: unsafe extern "C" fn(*const u8, u64, u32) -> VfsStatus,
	/// 检查 *path* 是否存在。
	pub access: unsafe extern "C" fn(*const u8, u64) -> VfsStatus,
}

// ---------------------------------------------------------------
//  测试
// ---------------------------------------------------------------

#[cfg(test)]
mod tests {
	use super::*;

	/// 操作表的对齐由其中的函数指针决定, 加载器按该对齐检查表的落点。
	#[test]
	fn test_vfs_ops_is_pointer_aligned() {
		assert_eq!(core::mem::align_of::<VfsOps>(), 8);
		assert_eq!(core::mem::size_of::<VfsOps>() % 8, 0);
	}

	/// 状态枚举按 u32 编码且 `Ok` 取 0, 与 emod 的操作表约定一致。
	#[test]
	fn test_status_is_encoded_as_u32() {
		assert_eq!(core::mem::size_of::<VfsStatus>(), 4);
		assert_eq!(VfsStatus::Ok as u32, 0);
		assert_ne!(VfsStatus::NoEntry as u32, VfsStatus::Exists as u32);
		assert_ne!(VfsStatus::IsDir as u32, VfsStatus::NotDir as u32);
		assert_ne!(VfsStatus::Busy as u32, VfsStatus::NotEmpty as u32);
	}

	/// stat 字段跨边界传递, 其尺寸与字段偏移在两侧一致。
	#[test]
	fn test_vfs_stat_layout_is_fixed() {
		assert_eq!(core::mem::size_of::<VfsStat>(), 40);
		assert_eq!(core::mem::offset_of!(VfsStat, ino), 0);
		assert_eq!(core::mem::offset_of!(VfsStat, kind), 8);
		assert_eq!(core::mem::offset_of!(VfsStat, mode), 12);
		assert_eq!(core::mem::offset_of!(VfsStat, nlink), 16);
		assert_eq!(core::mem::offset_of!(VfsStat, size), 24);
		assert_eq!(core::mem::offset_of!(VfsStat, blocks), 32);
		assert_eq!(core::mem::align_of::<VfsStat>(), 8);
	}

	/// 打开标志取自 Linux, 与载荷传入的取值一致; 访问模式位只占最低两位。
	#[test]
	fn test_open_flags_match_the_payload_encoding() {
		assert_eq!(O_RDONLY, 0);
		assert_eq!(O_WRONLY, 1);
		assert_eq!(O_RDWR, 2);
		assert_eq!(O_ACCMODE, 3);
		assert_eq!(O_CREAT, 0o100);
		assert_eq!(O_EXCL, 0o200);
		assert_eq!(O_TRUNC, 0o1000);
		assert_eq!(O_DIRECTORY, 0o200000);
		for flag in [O_CREAT, O_EXCL, O_TRUNC, O_DIRECTORY] {
			assert_eq!(flag & O_ACCMODE, 0, "标志 {flag:#o} 落在访问模式位内");
		}
	}

	/// 访问模式位的读出与写入判据: 只读允许读出, 只写允许写入, 读写两者都允许。
	#[test]
	fn test_access_mode_predicates() {
		for (flags, readable, writable) in [
			(O_RDONLY, true, false),
			(O_WRONLY, false, true),
			(O_RDWR, true, true),
		] {
			assert_eq!(mode_readable(flags), readable, "标志 {flags:#o} 的读出判据");
			assert_eq!(mode_writable(flags), writable, "标志 {flags:#o} 的写入判据");
		}
	}

	/// 删除目录的标志取自 Linux 的 AT_REMOVEDIR, 且不落在访问模式位内。
	#[test]
	fn test_removedir_flag_matches_the_payload_encoding() {
		assert_eq!(AT_REMOVEDIR, 0x200);
		assert_eq!(AT_REMOVEDIR & O_ACCMODE, 0);
	}
}
