//! 挂载层: 挂载表与按路径的分派。
//!
//! 对应 Linux 的 `fs/namespace.c`。每个挂载点登记一个文件系统, 一条路径交给挂载点前缀
//! 最长的那个文件系统, 且交给它的路径以该挂载点为根。文件系统一律以 [VfsOps] 操作表交付
//! 目录树, 故本层之上不再区分目录树来自宿主交付的模块还是运行时自身。

use crate::ext_mod::runtime;
use crate::ext_mod::vfs_ops::VfsOps;
use crate::sync_aux::SpinLock;

use super::sysfs;

/// 挂载表容量。
const MOUNTS_MAX: usize = 8;
/// 挂载点路径的字节数上限。
const POINT_MAX: usize = 32;

/// 运行时已知的文件系统。
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum FileSystem {
	/// 根文件系统, 由宿主交付的文件系统模块实现。
	Rootfs,
	/// 内核自带的伪文件系统, 见 [super::sysfs]。
	Sysfs,
}

/// 一项挂载: 挂载点与其下的文件系统。
#[derive(Clone, Copy)]
struct Mount {
	point: [u8; POINT_MAX],
	point_len: usize,
	fs: FileSystem,
}

/// 挂载表。
static MOUNTS: SpinLock<[Option<Mount>; MOUNTS_MAX]> = SpinLock::new([None; MOUNTS_MAX]);

/// 建立挂载表: 先建立内核自带的伪文件系统, 再登记各项挂载 —— 根文件系统挂在 `/`,
/// 内核自带的伪文件系统挂在 `/sys`。次序与 Linux 的 `mnt_init()` 先调 `sysfs_init()` 再
/// 建立挂载树相同。
///
/// 可重复调用: 各文件系统的目录树与挂载表都先清空再建立。
pub fn mount_init() {
	sysfs::init();
	*MOUNTS.lock() = [None; MOUNTS_MAX];
	mount(b"/", FileSystem::Rootfs);
	mount(b"/sys", FileSystem::Sysfs);
}

/// 登记一项挂载。挂载点取空、过长或表满时返回假。
fn mount(point: &[u8], fs: FileSystem) -> bool {
	if point.is_empty() || point.len() > POINT_MAX {
		return false;
	}
	let mut mounts = MOUNTS.lock();
	let Some(slot) = mounts.iter().position(|item| item.is_none()) else {
		return false;
	};
	let mut buf = [0u8; POINT_MAX];
	buf[..point.len()].copy_from_slice(point);
	mounts[slot] = Some(Mount {
		point: buf,
		point_len: point.len(),
		fs,
	});
	true
}

/// 取文件系统 *fs* 的操作表。根文件系统由模块交付, 宿主未提供该模块时返回 None。
pub fn ops_of(fs: FileSystem) -> Option<&'static VfsOps> {
	match fs {
		FileSystem::Rootfs => runtime::acquire_vfs_ops(),
		FileSystem::Sysfs => Some(sysfs::ops()),
	}
}

/// 把路径分派给挂载点前缀最长的文件系统, 返回该文件系统、它的操作表, 与它根下的路径。
///
/// 路径不属于任何已登记的挂载点, 或该文件系统尚未交付时返回 None。
pub fn resolve(path: &[u8]) -> Option<(FileSystem, &'static VfsOps, &[u8])> {
	let best = {
		let mounts = MOUNTS.lock();
		mounts
			.iter()
			.flatten()
			.filter(|item| covers(&item.point[..item.point_len], path))
			.max_by_key(|item| item.point_len)
			.copied()?
	};
	let ops = ops_of(best.fs)?;
	Some((best.fs, ops, relative(&best.point[..best.point_len], path)))
}

/// 路径 *path* 是否落在挂载点 *point* 之下。
///
/// 只有以 '/' 开头的路径参与匹配, 其余路径一律交给根文件系统; 比较按路径分量的边界进行,
/// 故 `/sys` 不覆盖 `/sysfoo`。
fn covers(point: &[u8], path: &[u8]) -> bool {
	if point == b"/" {
		return true;
	}
	if path.first() != Some(&b'/') {
		return false;
	}
	path.starts_with(point) && (path.len() == point.len() || path[point.len()] == b'/')
}

/// 取挂载点 *point* 之下文件系统根处的路径。挂载点是 `/` 时路径不变, 路径恰为挂载点时取
/// 该文件系统的根 `/`。
fn relative<'a>(point: &[u8], path: &'a [u8]) -> &'a [u8] {
	if point == b"/" {
		return path;
	}
	if path.len() == point.len() {
		return b"/";
	}
	&path[point.len()..]
}
