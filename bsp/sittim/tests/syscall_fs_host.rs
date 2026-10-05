//! 文件系统系统调用层在主机目标上的单元测试入口。
//!
//! 本层是文件系统模块与载荷之间的转接: 描述符表与打开文件描述、fd 偏移、访问模式判定与
//! 错误码折算都由它给出, 目录树与文件数据由模块持有。两侧各自的存储形态不同, 故本入口按
//! 源码包含三处源码: 运行时的 `src/syscall/fs/`、描述符表 `src/syscall/fdtable.rs` 与模块的
//! 节点层 `modules/fs/src/ramfs.rs`, 后者经 `modules/fs/src/table.rs` 按操作表的形态暴露给
//! 前者。这样用例覆盖的是两侧真实相遇的那一层, 而不是任一侧单独的桩。
//!
//! 本层对平台的依赖集中在三处, 由本文件给出替身: 控制台原语 [`io`]、套接字接口 [`net`]
//! 与模块取用 [`ext_mod::runtime`]。三处的替身只记录调用与给出可设置的返回值, 不改变本层
//! 的判定。
//!
//! 不在覆盖范围内的一项: 本层经 `read_stdin` 读取标准输入时, 飞地内该原语恒返回文件结束;
//! 主机侧由 [`io`] 给出可设置的输入源, 故读入路径的正例只在本入口内成立。

#![allow(dead_code)]

use core::sync::atomic::Ordering;
use std::sync::MutexGuard;

#[path = "../src/sync_aux.rs"]
mod sync_aux;

/// 与 sittim 共用的操作表定义, [`ext_mod`] 以它作为模块接口的形态。
#[path = "../src/ext_mod/vfs_ops.rs"]
pub mod vfs_ops;

#[path = "../src/syscall/errno.rs"]
mod errno;

#[path = "../src/syscall/types.rs"]
mod types;

#[path = "../modules/fs/src/ramfs.rs"]
mod ramfs;

#[path = "../modules/fs/src/table.rs"]
mod fs_table;

/// 描述符表与打开文件描述表。本层与文件系统层都以它为描述的存放处, 描述符表本身不随
/// 文件系统划入任何模块。
#[path = "../src/syscall/fdtable.rs"]
mod fdtable;

#[path = "../src/syscall/fs/mod.rs"]
mod fs;

#[path = "../src/constants.rs"]
mod constants;

#[path = "../src/syscall/concurrency/timer.rs"]
pub mod timer;

pub use errno::*;
pub use vfs_ops::{AT_REMOVEDIR, O_CREAT, O_EXCL, O_RDONLY, O_RDWR, O_TRUNC, O_WRONLY};

/// 时钟源与线程原语的替身。本层只用到「读时钟源」与「阻塞当前线程」两处, 其余由本入口
/// 按用例设置的标志回答。定时器设施与线程原语在飞地内分别位于
/// `crate::syscall::concurrency::{timer, thread}`, 故此处按同一路径重导出。
mod csr {
	use core::sync::atomic::{AtomicU64, Ordering};
	use std::sync::{Mutex, MutexGuard, OnceLock};

	/// 时钟源读数, 由用例设定。
	static TIME: AtomicU64 = AtomicU64::new(0);

	/// 时钟源锁。定时器表与时钟源读数都是进程内的全局量, 用例取该锁串行执行。
	fn clock_mutex() -> &'static Mutex<()> {
		static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
		LOCK.get_or_init(|| Mutex::new(()))
	}

	/// 取时钟源锁。用例在持有该锁期间重置定时器表并设定时钟源读数。
	pub fn clock_lock() -> MutexGuard<'static, ()> {
		let mutex = clock_mutex();
		let guard = mutex.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
		mutex.clear_poison();
		guard
	}

	/// 读时钟源计数。飞地内为 CSR 0xC01 的读取。
	pub fn read_time() -> u64 {
		TIME.load(Ordering::SeqCst)
	}

	/// 设定时钟源计数, 供用例推进时间。
	pub fn set_time(v: u64) {
		TIME.store(v, Ordering::SeqCst);
	}
}

/// 线程原语的替身: 阻塞只作记录, 切换登记与重启请求由用例设定的标志回答。
pub mod thread {
	use core::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

	/// 线程表容量, 与 sittim 一致, 供定时器设施定容。
	pub const NUM_THREADS: usize = 32;

	/// 当前线程标识, 由用例设定。
	pub static CURRENT_THREAD_ID: AtomicUsize = AtomicUsize::new(0);
	/// 最近一次阻塞登记的等待地址。
	pub static BLOCK_WAIT_ADDR: AtomicU64 = AtomicU64::new(0);
	/// 阻塞次数。
	pub static BLOCK_COUNT: AtomicUsize = AtomicUsize::new(0);
	/// 已登记切换目标: 阻塞时由 [`block_current`] 置位。用例清除它即模拟重新进入被重启的
	/// 系统调用。
	pub static SWITCH_PENDING: AtomicBool = AtomicBool::new(false);
	/// 重启请求次数。
	pub static RESTART_COUNT: AtomicUsize = AtomicUsize::new(0);

	pub fn current_thread_id() -> usize {
		CURRENT_THREAD_ID.load(Ordering::SeqCst)
	}

	/// 阻塞当前线程: 记录等待地址与次数, 并置位 [`SWITCH_PENDING`], 与飞地内的阻塞一致 ——
	/// 阻塞在有其它可运行线程时登记切换目标, 调用方据此重启本系统调用。主机侧没有可切换
	/// 的线程, 按已登记目标回答使调用方走重启路径, 循环等待不会在本进程内发生。
	pub fn block_current(wait_addr: u64) {
		BLOCK_WAIT_ADDR.store(wait_addr, Ordering::SeqCst);
		BLOCK_COUNT.fetch_add(1, Ordering::SeqCst);
		SWITCH_PENDING.store(true, Ordering::SeqCst);
	}

	pub fn switch_pending() -> bool {
		SWITCH_PENDING.load(Ordering::SeqCst)
	}

	pub fn request_restart() {
		RESTART_COUNT.fetch_add(1, Ordering::SeqCst);
	}
}

/// 定时器设施与线程原语在飞地内同处 `syscall::concurrency`, 本层按同一路径引用。
pub mod concurrency {
	pub use crate::thread;
	pub use crate::timer;
}

/// 运行时的扩展模块接口。本层只用到模块取用一项, 取入序列不在本入口的执行路径上。
mod ext_mod {
	pub use crate::vfs_ops;

	/// 模块取用。主机侧不执行取入序列, 操作表直接由模块的节点层建立。
	pub mod runtime {
		use core::sync::atomic::{AtomicBool, Ordering};
		use std::sync::OnceLock;

		use crate::vfs_ops::VfsOps;

		/// 置位时 [`acquire_vfs_ops`] 按宿主未提供模块处理。
		pub static VFS_MISSING: AtomicBool = AtomicBool::new(false);

		/// 取用文件系统模块的操作表。首次取用时建立根目录, 与飞地内取入后调用模块入口的
		/// 效果一致。
		pub fn acquire_vfs_ops() -> Option<&'static VfsOps> {
			if VFS_MISSING.load(Ordering::SeqCst) {
				return None;
			}
			static OPS: OnceLock<VfsOps> = OnceLock::new();
			Some(OPS.get_or_init(|| {
				let ops = crate::fs_table::fill_ops();
				unsafe { (ops.init)() };
				ops
			}))
		}
	}
}

/// 控制台原语的替身: 写入落到本进程的缓冲区, 读出取本进程预置的输入源。
mod io {
	use std::collections::VecDeque;
	use std::sync::Mutex;

	/// 写入控制台的字节。
	pub static OUT: Mutex<Vec<u8>> = Mutex::new(Vec::new());
	/// 可供读入的字节。
	pub static IN: Mutex<VecDeque<u8>> = Mutex::new(VecDeque::new());

	/// 把 *len* 个字节追加到 [`OUT`], 返回写入的字节数。
	pub unsafe fn console_write_bytes(buf: *const u8, len: u64) -> u64 {
		let mut out = OUT.lock().unwrap();
		for i in 0..len {
			out.push(unsafe { buf.add(i as usize).read_volatile() });
		}
		len
	}

	/// 从 [`IN`] 取出至多 *len* 个字节, 遇换行符停止, 返回取出的字节数。
	pub unsafe fn read_stdin(buf: *mut u8, len: u64) -> u64 {
		let mut src = IN.lock().unwrap();
		let mut count: u64 = 0;
		while count < len {
			let Some(b) = src.pop_front() else {
				break;
			};
			unsafe { buf.add(count as usize).write_volatile(b) };
			count += 1;
			if b == b'\n' || b == b'\r' {
				break;
			}
		}
		count
	}
}

/// 套接字接口的替身: 只记录转交过来的调用, 并按可设置的标志回答就绪状态。
mod net {
	use std::sync::Mutex;

	/// 可接收的套接字下标。
	pub static CAN_RECV: Mutex<[bool; 8]> = Mutex::new([false; 8]);
	/// 可发送的套接字下标。
	pub static CAN_SEND: Mutex<[bool; 8]> = Mutex::new([false; 8]);
	/// 收到关闭请求的套接字下标。
	pub static CLOSED: Mutex<Vec<u64>> = Mutex::new(Vec::new());
	/// 收到读请求的套接字下标。
	pub static READ: Mutex<Vec<u64>> = Mutex::new(Vec::new());
	/// 收到写请求的套接字下标。
	pub static WRITE: Mutex<Vec<u64>> = Mutex::new(Vec::new());

	pub fn can_recv(index: u64) -> bool {
		CAN_RECV.lock().unwrap()[index as usize]
	}

	pub fn can_send(index: u64) -> bool {
		CAN_SEND.lock().unwrap()[index as usize]
	}

	pub fn close_socket(index: u64) {
		CLOSED.lock().unwrap().push(index);
	}

	pub fn read_socket(index: u64, _open_flags: u32, _buf: *mut u8, _len: u64) -> u64 {
		READ.lock().unwrap().push(index);
		0
	}

	pub fn write_socket(index: u64, _buf: *const u8, len: u64) -> u64 {
		WRITE.lock().unwrap().push(index);
		len
	}

	pub fn readv_socket(index: u64, _open_flags: u32, _io_vec_arr: u64, _io_vec_size: u64) -> u64 {
		READ.lock().unwrap().push(index);
		0
	}

	pub fn writev_socket(index: u64, _io_vec_arr: u64, _io_vec_size: u64) -> u64 {
		WRITE.lock().unwrap().push(index);
		0
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// musl 给出的 AT_FDCWD 取值。
	const AT_FDCWD: u64 = -100i64 as u64;

	/// fcntl 的命令编号, 取 Linux riscv64 的取值。
	const F_DUPFD: u64 = 0;
	const F_GETFD: u64 = 1;
	const F_SETFD: u64 = 2;
	const F_GETFL: u64 = 3;
	const F_SETFL: u64 = 4;
	const F_DUPFD_CLOEXEC: u64 = 1030;

	/// fcntl 的 FD_CLOEXEC 位与打开标志中的 O_CLOEXEC 位, 二者指代同一状态。
	const FD_CLOEXEC: u64 = 1;
	const O_CLOEXEC: u32 = 0o2000000;
	/// 写入偏移取文件末尾的标志位。
	const O_APPEND: u32 = 0o2000;
	/// 要求打开的目标是目录的标志位。
	const O_DIRECTORY: u32 = 0o200000;

	/// 目录项的类型编码, 取自 Linux 的 DT_* 取值。
	const DT_DIR: u8 = 4;

	/// faccessat 的 mode 位: 执行, 写, 读。
	const X_OK: u64 = 1;
	const W_OK: u64 = 2;
	const R_OK: u64 = 4;
	/// 路径类调用共用的标志位。
	const AT_SYMLINK_NOFOLLOW: u64 = 0x100;
	const AT_EACCESS: u64 = 0x200;
	const AT_NO_AUTOMOUNT: u64 = 0x800;
	const AT_EMPTY_PATH: u64 = 0x1000;
	/// fstatat 的缓存同步方式两位。
	const AT_STATX_FORCE_SYNC: u64 = 0x2000;
	const AT_STATX_DONT_SYNC: u64 = 0x4000;

	/// lseek 的基准取值。
	const SEEK_SET: u64 = 0;
	const SEEK_CUR: u64 = 1;
	const SEEK_END: u64 = 2;

	/// `struct pollfd`: 描述符 (i32), 关注的事件 (i16), 发生的事件 (i16)。
	#[repr(C)]
	#[derive(Clone, Copy)]
	struct PollFd {
		fd: i32,
		events: i16,
		revents: i16,
	}

	const POLLIN: i16 = 0x0001;
	const POLLOUT: i16 = 0x0004;
	const POLLNVAL: i16 = 0x0020;

	/// 承载 stat 结果的缓冲。`Stat` 含 u64 字段, 故按 8 字节对齐。
	#[repr(align(8))]
	struct StatBuf([u8; 128]);

	/// 取全局锁, 把挂载表、目录树、文件数据区与打开文件表恢复到初始状态。
	///
	/// 它们都是进程内的全局量, 且本二进制内另有一组直接改动目录树的用例 (由节点层的文件
	/// 自带)。故本入口取用节点层测试入口的同一把锁与同一次清空, 随后释放全部描述符并重新
	/// 安装三条控制台表项。
	///
	/// 另取时钟源锁并清空定时器表: 二者由定时器设施的用例共用, 同一个进程内并发执行会
	/// 使一方的重置抹掉另一方的登记。两把锁的取用顺序固定为「时钟源, 再目录树」。
	fn world() -> (MutexGuard<'static, ()>, MutexGuard<'static, ()>) {
		let clock = crate::csr::clock_lock();
		crate::csr::set_time(0);
		crate::timer::tests::reset_table();
		thread::BLOCK_WAIT_ADDR.store(0, Ordering::SeqCst);
		thread::BLOCK_COUNT.store(0, Ordering::SeqCst);
		thread::RESTART_COUNT.store(0, Ordering::SeqCst);
		thread::SWITCH_PENDING.store(false, Ordering::SeqCst);
		let guard = ramfs::tests::setup();
		for fd in 0..64 {
			fs::close_handler(fd);
		}
		fs::mount_init();
		fs::vfs_init();
		ext_mod::runtime::VFS_MISSING.store(false, Ordering::SeqCst);
		reset(&io::OUT).clear();
		reset(&io::IN).clear();
		*reset(&net::CAN_RECV) = [false; 8];
		*reset(&net::CAN_SEND) = [false; 8];
		reset(&net::CLOSED).clear();
		reset(&net::READ).clear();
		reset(&net::WRITE).clear();
		(clock, guard)
	}

	/// 取替身模块的全局量, 并清除上一次 panic 留在该锁上的毒化标记。
	///
	/// 一条用例在持有该锁时断言失败会使该锁毒化, 其后每条用例都在取锁处失败, 一次失败因此
	/// 表现为几十条失败。这些全局量在本入口被整体重置, 上一次 panic 不构成它们不可用的依据,
	/// 故此处忽略毒化并清除标记, 使后续用例各自判定。
	fn reset<T>(mutex: &'static std::sync::Mutex<T>) -> MutexGuard<'static, T> {
		let guard = mutex.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
		mutex.clear_poison();
		guard
	}

	/// 一条用例在持有全局量的锁时失败会使该锁毒化; 重置入口对该全局量取锁后毒化标记被清除,
	/// 其后用例的取锁形式不再失败, 故前一条用例的失败不牵连后面的用例。
	///
	/// 本用例持有重置入口的锁, 而该锁不重入, 故不得在本用例内再次调用重置入口。
	#[test]
	fn test_world_recovers_from_a_poisoned_lock() {
		let _guard = world();
		let poisoned = std::panic::catch_unwind(|| {
			let _held = net::CLOSED.lock().unwrap();
			panic!("模拟一条用例在持有该锁时失败");
		});
		assert!(poisoned.is_err());
		assert!(net::CLOSED.is_poisoned());
		{
			let _held = reset(&net::CLOSED);
		}
		assert!(!net::CLOSED.is_poisoned());
		// 用例体内的取锁形式
		net::CLOSED.lock().unwrap().clear();
	}

	/// 把路径写成处理器读取的结尾空字节形式。
	fn cstr(path: &str) -> Vec<u8> {
		let mut v = path.as_bytes().to_vec();
		v.push(0);
		v
	}

	/// 建立用例自己的目录, 其余路径都挂在它之下。模块的存储不随用例清空, 故各用例取互不
	/// 相同的目录名。
	fn make_root(tag: &str) -> String {
		let root = format!("/{tag}");
		assert_eq!(fs::mkdirat_handler(AT_FDCWD, cstr(&root).as_ptr() as u64, 0o755), 0);
		root
	}

	/// 在一个用例自己的目录下建立子路径并返回完整路径。
	fn under(root: &str, rest: &str) -> String {
		format!("{root}/{rest}")
	}

	/// 按路径建立目录。
	fn mkdir(path: &str) {
		assert_eq!(fs::mkdirat_handler(AT_FDCWD, cstr(path).as_ptr() as u64, 0o755), 0);
	}

	/// 按路径打开并返回描述符。
	fn open(path: &str, flags: u32) -> u64 {
		let fd = fs::openat_handler(AT_FDCWD, cstr(path).as_ptr() as u64, flags as u64, 0o644);
		assert!(fd < 64, "打开 {path} 得到 {fd}, 不是描述符");
		fd
	}

	/// 解析 getdents 写入的 *written* 个字节, 返回每项的类型编码与名字。
	///
	/// 布局按 musl riscv64 的 `struct dirent`: d_ino (u64) 在偏移 0, d_off (i64) 在偏移 8,
	/// d_reclen (u16) 在偏移 16, d_type 在偏移 18, 名字自偏移 19 起以结尾空字节终止。
	fn dirents(buf: &[u8], written: u64) -> Vec<(u8, Vec<u8>)> {
		/// d_name 之前的字节数。
		const HEAD: usize = 19;
		let end = written as usize;
		let mut entries = Vec::new();
		let mut at = 0;
		while at < end {
			let reclen = u16::from_le_bytes([buf[at + 16], buf[at + 17]]) as usize;
			assert!(reclen > HEAD && at + reclen <= end, "目录项长度 {reclen} 越界");
			let name_end = buf[at + HEAD..at + reclen]
				.iter()
				.position(|byte| *byte == 0)
				.expect("目录项的名字没有结尾空字节");
			entries.push((buf[at + 18], buf[at + HEAD..at + HEAD + name_end].to_vec()));
			at += reclen;
		}
		assert_eq!(at, end, "各目录项的长度之和不等于写入的字节数");
		entries
	}

	/// 宿主未提供文件系统模块时, 涉及路径的入口一律返回 ENOSYS。
	#[test]
	fn test_path_entries_report_enosys_without_the_module() {
		let _guard = world();
		ext_mod::runtime::VFS_MISSING.store(true, Ordering::SeqCst);
		let path = cstr("/unreachable");
		let mut st = StatBuf([0u8; 128]);
		let stat_ptr = st.0.as_mut_ptr() as u64;
		assert_eq!(fs::openat_handler(AT_FDCWD, path.as_ptr() as u64, 0, 0), ENOSYS);
		assert_eq!(fs::mkdirat_handler(AT_FDCWD, path.as_ptr() as u64, 0o755), ENOSYS);
		assert_eq!(fs::unlinkat_handler(AT_FDCWD, path.as_ptr() as u64, 0), ENOSYS);
		assert_eq!(fs::faccessat_handler(AT_FDCWD, path.as_ptr() as u64, 0, 0), ENOSYS);
		assert_eq!(fs::fstatat_handler(AT_FDCWD, path.as_ptr() as u64, stat_ptr, 0), ENOSYS);
	}

	/// 相对路径只在 dirfd 取 AT_FDCWD 时可用; 绝对路径不受 dirfd 限制。
	#[test]
	fn test_relative_path_requires_the_cwd_descriptor() {
		let _guard = world();
		let root = make_root("relpath");
		let relative = cstr("relpath/f");
		assert_eq!(
			fs::openat_handler(0, relative.as_ptr() as u64, O_CREAT as u64, 0o644),
			ENOTDIR
		);
		let fd = open("relpath/f", O_CREAT | O_RDONLY);
		assert_eq!(fs::close_handler(fd), 0);
		let absolute = cstr(&under(&root, "f"));
		assert!(fs::openat_handler(0, absolute.as_ptr() as u64, O_CREAT as u64, 0o644) < 64);
	}

	/// 打开标志的访问模式位限定读与写: 只写的描述符不可读, 只读的描述符不可写。
	#[test]
	fn test_read_and_write_follow_the_access_mode() {
		let _guard = world();
		let root = make_root("mode");
		let path = under(&root, "f");
		let seed = b"abc";
		let fd = open(&path, O_CREAT | O_RDWR);
		assert_eq!(fs::write_handler(fd, seed.as_ptr(), 3), 3);
		assert_eq!(fs::close_handler(fd), 0);

		let wfd = open(&path, O_WRONLY);
		let mut buf = [0u8; 8];
		assert_eq!(fs::read_handler(wfd, buf.as_mut_ptr(), 4), EBADF);
		assert_eq!(fs::readv_handler(wfd, buf.as_mut_ptr() as u64, 0), EBADF);
		assert_eq!(fs::pread64_handler(wfd, buf.as_mut_ptr(), 4, 0), EBADF);
		assert_eq!(fs::write_handler(wfd, seed.as_ptr(), 3), 3);
		assert_eq!(fs::close_handler(wfd), 0);

		let rfd = open(&path, O_RDONLY);
		assert_eq!(fs::write_handler(rfd, seed.as_ptr(), 3), EBADF);
		assert_eq!(fs::writev_handler(rfd, buf.as_mut_ptr() as u64, 0), EBADF);
		assert_eq!(fs::pwrite64_handler(rfd, seed.as_ptr(), 3, 0), EBADF);
		assert_eq!(fs::ftruncate_handler(rfd, 1), EINVAL);
		assert_eq!(fs::read_handler(rfd, buf.as_mut_ptr(), 4), 3);
		assert_eq!(&buf[..3], seed);
	}

	/// 顺序读共享描述符偏移, 读到末尾后返回 0。
	#[test]
	fn test_sequential_reads_share_the_descriptor_offset() {
		let _guard = world();
		let root = make_root("offset");
		let path = under(&root, "f");
		let seed = b"0123456789";
		let fd = open(&path, O_CREAT | O_RDWR);
		assert_eq!(fs::write_handler(fd, seed.as_ptr(), 10), 10);
		assert_eq!(fs::close_handler(fd), 0);

		let fd = open(&path, O_RDONLY);
		let mut buf = [0u8; 8];
		assert_eq!(fs::read_handler(fd, buf.as_mut_ptr(), 4), 4);
		assert_eq!(&buf[..4], b"0123");
		assert_eq!(fs::read_handler(fd, buf.as_mut_ptr(), 4), 4);
		assert_eq!(&buf[..4], b"4567");
		assert_eq!(fs::read_handler(fd, buf.as_mut_ptr(), 4), 2);
		assert_eq!(&buf[..2], b"89");
		assert_eq!(fs::read_handler(fd, buf.as_mut_ptr(), 4), 0);
		// 指定偏移的读不改动描述符偏移
		assert_eq!(fs::pread64_handler(fd, buf.as_mut_ptr(), 2, 0), 2);
		assert_eq!(&buf[..2], b"01");
		assert_eq!(fs::read_handler(fd, buf.as_mut_ptr(), 4), 0);
	}

	/// 目录按可写访问打开失败, 按只读访问打开成功, 对其读返回 EISDIR。
	#[test]
	fn test_directory_reads_report_eisdir() {
		let _guard = world();
		let root = make_root("isdir");
		assert_eq!(
			fs::openat_handler(AT_FDCWD, cstr(&root).as_ptr() as u64, O_RDWR as u64, 0),
			EISDIR
		);
		let fd = open(&root, O_RDONLY);
		let mut buf = [0u8; 8];
		assert_eq!(fs::read_handler(fd, buf.as_mut_ptr(), 4), EISDIR);
		let iovec = [buf.as_mut_ptr() as u64, 4];
		assert_eq!(fs::readv_handler(fd, iovec.as_ptr() as u64, 1), EISDIR);
		assert_eq!(fs::pwrite64_handler(fd, buf.as_ptr(), 4, 0), EBADF);
	}

	/// 对文件取目录项返回 ENOTDIR。
	#[test]
	fn test_getdents_on_a_file_reports_enotdir() {
		let _guard = world();
		let root = make_root("notdir");
		let path = under(&root, "f");
		let fd = open(&path, O_CREAT | O_RDONLY);
		let mut buf = [0u8; 64];
		assert_eq!(fs::getdents_handler(fd, buf.as_mut_ptr(), 64), ENOTDIR);
	}

	/// getdents 的缓冲取空而个数非 0 时返回 EFAULT; 缓冲区容纳不下一个目录项时返回
	/// EINVAL, 与"目录已读完"返回 0 区分开。已写入的条目其读取位置照常推进。
	#[test]
	fn test_getdents_validates_the_buffer() {
		let _guard = world();
		let root = make_root("getdentsbuf");
		let dir = under(&root, "d");
		mkdir(&dir);
		let fd = open(&dir, O_RDONLY);

		let mut buf = [0u8; 64];
		assert_eq!(fs::getdents_handler(fd, core::ptr::null_mut(), 64), EFAULT);
		// 个数为 0 时不读取缓冲, 故指针取空也返回 EINVAL 而不是 EFAULT
		assert_eq!(fs::getdents_handler(fd, core::ptr::null_mut(), 0), EINVAL);
		// "." 的条目长 24 字节
		assert_eq!(fs::getdents_handler(fd, buf.as_mut_ptr(), 23), EINVAL);
		assert_eq!(fs::getdents_handler(fd, buf.as_mut_ptr(), 0), EINVAL);
		assert_eq!(fs::getdents_handler(fd, buf.as_mut_ptr(), 24), 24);
		// 已写入一条之后, 放不下的下一条留待下次读取, 本次返回已写入的字节数
		assert_eq!(fs::getdents_handler(fd, buf.as_mut_ptr(), 40), 24);
	}

	/// sysfs: `/sys/devices/system/cpu` 是目录, 其下恰有一个形如 cpu<数字> 的处理器项。
	///
	/// stress-ng 以 `scandir` 遍历该目录, 按名字前三个字节为 "cpu" 且第四个字节为数字筛选
	/// 处理器, 一项都不命中时报 `no CPUs found in /sys/devices/system/cpu`。本用例按同一
	/// 筛选条件断言命中一项, 并断言该目录本身可打开。
	#[test]
	fn test_sys_cpu_directory_lists_one_processor() {
		let _guard = world();
		let fd = open("/sys/devices/system/cpu", O_RDONLY | O_DIRECTORY);
		let mut buf = [0u8; 256];
		let written = fs::getdents_handler(fd, buf.as_mut_ptr(), buf.len() as u64);
		let entries = dirents(&buf, written);
		let names: Vec<&[u8]> = entries.iter().map(|(_, name)| name.as_slice()).collect();
		assert_eq!(names, [b".".as_slice(), b"..".as_slice(), b"cpu0".as_slice()]);
		assert!(entries.iter().all(|(kind, _)| *kind == DT_DIR));
		let processors = names
			.iter()
			.filter(|name| name.len() >= 4 && name.starts_with(b"cpu") && name[3].is_ascii_digit())
			.count();
		assert_eq!(processors, 1, "按 stress-ng 的筛选条件未命中处理器项");
	}

	/// sysfs 的两棵子树各在 `/sys` 之下: `/sys/devices` 是目录, 其上 `/sys` 自身可打开。
	#[test]
	fn test_sys_root_and_devices_are_directories() {
		let _guard = world();
		let fd = open("/sys", O_RDONLY | O_DIRECTORY);
		let mut buf = [0u8; 256];
		let written = fs::getdents_handler(fd, buf.as_mut_ptr(), buf.len() as u64);
		let names: Vec<Vec<u8>> = dirents(&buf, written).into_iter().map(|(_, name)| name).collect();
		assert_eq!(names, [b".".to_vec(), b"..".to_vec(), b"devices".to_vec()]);
		open("/sys/devices", O_RDONLY | O_DIRECTORY);
	}

	/// faccessat 的 mode 与 flags 只接受登记的位, 出现其余位返回 EINVAL。该判定先于路径
	/// 解析, 故非法取值配不存在的路径同样返回 EINVAL。
	#[test]
	fn test_faccessat_rejects_unknown_mode_bits_and_flags() {
		let _guard = world();
		let missing = cstr("/absent");
		let ptr = missing.as_ptr() as u64;
		for mode in [0u64, X_OK, W_OK, R_OK, X_OK | W_OK | R_OK] {
			assert_eq!(fs::faccessat_handler(AT_FDCWD, ptr, mode, 0), ENOENT);
		}
		assert_eq!(fs::faccessat_handler(AT_FDCWD, ptr, 0o10, 0), EINVAL);
		for flags in [0u64, AT_SYMLINK_NOFOLLOW, AT_EACCESS, AT_EMPTY_PATH, 0x1300] {
			assert_eq!(fs::faccessat_handler(AT_FDCWD, ptr, 0, flags), ENOENT);
		}
		assert_eq!(fs::faccessat_handler(AT_FDCWD, ptr, 0, AT_NO_AUTOMOUNT), EINVAL);
		assert_eq!(fs::faccessat_handler(AT_FDCWD, ptr, 0, 0x400), EINVAL);
	}

	/// fstatat 的 flags 只接受登记的位, 出现其余位返回 EINVAL。该判定先于路径解析与结果
	/// 指针的检查。
	#[test]
	fn test_fstatat_rejects_unknown_flags() {
		let _guard = world();
		let root = make_root("statflags");
		let path = cstr(&root);
		let mut st = StatBuf([0u8; 128]);
		let stat_ptr = st.0.as_mut_ptr() as u64;
		let path_ptr = path.as_ptr() as u64;
		let accepted = [
			0u64,
			AT_SYMLINK_NOFOLLOW,
			AT_NO_AUTOMOUNT,
			AT_EMPTY_PATH,
			AT_STATX_FORCE_SYNC,
			AT_STATX_DONT_SYNC,
			AT_STATX_FORCE_SYNC | AT_STATX_DONT_SYNC,
			AT_SYMLINK_NOFOLLOW | AT_NO_AUTOMOUNT | AT_EMPTY_PATH | AT_STATX_FORCE_SYNC,
		];
		for flags in accepted {
			assert_eq!(fs::fstatat_handler(AT_FDCWD, path_ptr, stat_ptr, flags), 0);
		}
		assert_eq!(fs::fstatat_handler(AT_FDCWD, path_ptr, stat_ptr, AT_EACCESS), EINVAL);
		assert_eq!(fs::fstatat_handler(AT_FDCWD, path_ptr, stat_ptr, 0x20000), EINVAL);
		assert_eq!(fs::fstatat_handler(AT_FDCWD, path_ptr, 0, AT_EACCESS), EINVAL);
	}

	/// 路径取空而 flags 含 AT_EMPTY_PATH 时, fstatat 与 faccessat 以 dirfd 自身绑定的节点
	/// 为目标; dirfd 绑定的不是文件系统节点时返回 EBADF。
	#[test]
	fn test_empty_path_targets_the_descriptor_itself() {
		let _guard = world();
		let root = make_root("emptypath");
		let path = under(&root, "f");
		let fd = open(&path, O_CREAT | O_RDWR);
		assert_eq!(fs::write_handler(fd, b"xy".as_ptr(), 2), 2);
		let empty = cstr("");
		let empty_ptr = empty.as_ptr() as u64;

		let mut st = StatBuf([0u8; 128]);
		assert_eq!(
			fs::fstatat_handler(fd, empty_ptr, st.0.as_mut_ptr() as u64, AT_EMPTY_PATH),
			0
		);
		let st_mode = u32::from_le_bytes(st.0[16..20].try_into().unwrap());
		assert_eq!(st_mode & 0o170000, 0o100000, "目标是 fd 绑定的文本文件");
		let st_size = i64::from_le_bytes(st.0[48..56].try_into().unwrap());
		assert_eq!(st_size, 2, "长度取自 fd 绑定的那个文件");

		assert_eq!(fs::faccessat_handler(fd, empty_ptr, R_OK, AT_EMPTY_PATH), 0);
		// 空路径不带 AT_EMPTY_PATH 时不指代任何对象
		assert_eq!(fs::faccessat_handler(fd, empty_ptr, 0, 0), ENOENT);
		assert_eq!(fs::fstatat_handler(fd, empty_ptr, st.0.as_mut_ptr() as u64, 0), ENOENT);
		// 控制台描述符绑定的不是文件系统节点
		assert_eq!(fs::faccessat_handler(1, empty_ptr, 0, AT_EMPTY_PATH), EBADF);
		assert_eq!(
			fs::fstatat_handler(1, empty_ptr, st.0.as_mut_ptr() as u64, AT_EMPTY_PATH),
			EBADF
		);
	}

	/// unlinkat 把 flags 原样转交模块: 目录要带 AT_REMOVEDIR 才可删除, 不带时给出 EISDIR,
	/// 目录非空时给出 ENOTEMPTY。
	#[test]
	fn test_unlinkat_forwards_the_directory_flag() {
		let _guard = world();
		let root = make_root("unlink");
		let dir = under(&root, "d");
		mkdir(&dir);
		assert_eq!(fs::unlinkat_handler(AT_FDCWD, cstr(&dir).as_ptr() as u64, 0), EISDIR);
		mkdir(&under(&dir, "child"));
		assert_eq!(
			fs::unlinkat_handler(AT_FDCWD, cstr(&dir).as_ptr() as u64, AT_REMOVEDIR as u64),
			ENOTEMPTY
		);
		assert_eq!(
			fs::unlinkat_handler(
				AT_FDCWD,
				cstr(&under(&dir, "child")).as_ptr() as u64,
				AT_REMOVEDIR as u64
			),
			0
		);
		assert_eq!(
			fs::unlinkat_handler(AT_FDCWD, cstr(&dir).as_ptr() as u64, AT_REMOVEDIR as u64),
			0
		);
		assert_eq!(fs::faccessat_handler(AT_FDCWD, cstr(&dir).as_ptr() as u64, 0, 0), ENOENT);
		// 目录上不带标志的删除不得被当作文本文件删除
		mkdir(&dir);
		assert_eq!(fs::unlinkat_handler(AT_FDCWD, cstr(&dir).as_ptr() as u64, 0), EISDIR);
		assert_eq!(fs::faccessat_handler(AT_FDCWD, cstr(&dir).as_ptr() as u64, 0, 0), 0);
	}

	/// unlinkat 只接受 AT_REMOVEDIR 一位: 其余位一律返回 EINVAL, 且不落到模块。
	#[test]
	fn test_unlinkat_rejects_unknown_flags() {
		let _guard = world();
		let root = make_root("unlink_flags");
		let dir = under(&root, "d");
		mkdir(&dir);
		assert_eq!(
			fs::unlinkat_handler(AT_FDCWD, cstr(&dir).as_ptr() as u64, AT_REMOVEDIR as u64 | 4),
			EINVAL
		);
		// 高位同样不接受, 不得因截断到 32 位而被放行
		assert_eq!(
			fs::unlinkat_handler(AT_FDCWD, cstr(&dir).as_ptr() as u64, 1 << 32),
			EINVAL
		);
		// 被拒的调用不改动目录树
		assert_eq!(fs::faccessat_handler(AT_FDCWD, cstr(&dir).as_ptr() as u64, 0, 0), 0);
	}

	/// 根目录不可删除, 按删除目录的请求返回 EBUSY。
	#[test]
	fn test_unlinkat_on_the_root_reports_busy() {
		let _guard = world();
		assert_eq!(fs::unlinkat_handler(AT_FDCWD, cstr("/").as_ptr() as u64, AT_REMOVEDIR as u64), EBUSY);
		assert_eq!(fs::unlinkat_handler(AT_FDCWD, cstr("/").as_ptr() as u64, 0), EISDIR);
	}

	/// 控制台表项按打开标志的方向分流: fd 0 只可读, fd 1 与 fd 2 只可写。
	#[test]
	fn test_console_entries_follow_their_access_mode() {
		let _guard = world();
		io::IN.lock().unwrap().push_back(b'x');
		let mut buf = [0u8; 4];
		assert_eq!(fs::read_handler(0, buf.as_mut_ptr(), 4), 1);
		assert_eq!(buf[0], b'x');
		assert_eq!(fs::write_handler(0, buf.as_ptr(), 1), EBADF);

		assert_eq!(fs::write_handler(1, b"ok".as_ptr(), 2), 2);
		assert_eq!(fs::read_handler(1, buf.as_mut_ptr(), 1), EBADF);
		assert_eq!(io::OUT.lock().unwrap().as_slice(), b"ok");

		assert_eq!(fs::write_handler(2, b"!".as_ptr(), 1), 1);
		assert_eq!(io::OUT.lock().unwrap().as_slice(), b"ok!");
	}

	/// ppoll 按打开标志的方向给出控制台的就绪事件, 越界描述符给出 POLLNVAL。
	#[test]
	fn test_ppoll_reports_console_directions() {
		let _guard = world();
		let mut pfds = [
			PollFd { fd: 0, events: POLLIN | POLLOUT, revents: 0 },
			PollFd { fd: 1, events: POLLIN | POLLOUT, revents: 0 },
			PollFd { fd: 64, events: POLLIN, revents: 0 },
		];
		assert_eq!(fs::ppoll_handler(pfds.as_mut_ptr() as u64, 3, 0), 3);
		assert_eq!(pfds[0].revents, POLLIN);
		assert_eq!(pfds[1].revents, POLLOUT);
		assert_eq!(pfds[2].revents, POLLNVAL);
	}

	/// 套接字描述符的读、写与关闭转交协议栈, 不落到文件系统模块。
	#[test]
	fn test_socket_descriptor_is_routed_to_the_stack() {
		let _guard = world();
		let fd = fdtable::alloc_socket_fd(3, O_RDWR, 0);
		assert!(fd < 64);
		let mut buf = [0u8; 4];
		assert_eq!(fs::write_handler(fd, buf.as_ptr(), 4), 4);
		assert_eq!(fs::read_handler(fd, buf.as_mut_ptr(), 4), 0);
		assert_eq!(*net::WRITE.lock().unwrap(), vec![3]);
		assert_eq!(*net::READ.lock().unwrap(), vec![3]);
		assert_eq!(fs::close_handler(fd), 0);
		assert_eq!(*net::CLOSED.lock().unwrap(), vec![3]);
		assert_eq!(fs::close_handler(fd), EBADF);
	}

	/// 打开取用最低的空闲编号, 关闭后该编号可被后续的打开取回。
	#[test]
	fn test_closed_descriptor_number_is_reused() {
		let _guard = world();
		let root = make_root("reuse");
		let path = under(&root, "f");
		assert_eq!(fs::close_handler(0), 0);
		assert_eq!(fs::close_handler(1), 0);
		let fd = open(&path, O_CREAT | O_RDONLY);
		assert_eq!(fd, 0);
		assert_eq!(fs::close_handler(fd), 0);
		let fd = open(&path, O_RDONLY);
		assert_eq!(fd, 0);
	}

	/// stat 按节点类别补齐文件类型编码: 目录为 S_IFDIR, 文件为 S_IFREG。
	#[test]
	fn test_fstat_reports_the_node_kind() {
		let _guard = world();
		let root = make_root("stat");
		let dir_fd = open(&root, O_RDONLY);
		let mut st = StatBuf([0u8; 128]);
		assert_eq!(fs::fstat_handler(dir_fd, st.0.as_mut_ptr() as u64), 0);
		let mode = u32::from_le_bytes(st.0[16..20].try_into().unwrap());
		assert_eq!(mode & 0o170000, 0o040000);

		let file_fd = open(&under(&root, "f"), O_CREAT | O_RDONLY);
		assert_eq!(fs::fstat_handler(file_fd, st.0.as_mut_ptr() as u64), 0);
		let mode = u32::from_le_bytes(st.0[16..20].try_into().unwrap());
		assert_eq!(mode & 0o170000, 0o100000);
	}

	/// 控制台与套接字无偏移可定位, 对它们返回 ESPIPE; 空闲槽位返回 EBADF。
	#[test]
	fn test_lseek_on_console_and_socket_reports_espipe() {
		let _guard = world();
		assert_eq!(fs::lseek_handler(0, 0, SEEK_SET), ESPIPE);
		assert_eq!(fs::lseek_handler(1, 0, SEEK_SET), ESPIPE);
		assert_eq!(fs::lseek_handler(2, 0, SEEK_END), ESPIPE);
		assert_eq!(fs::lseek_handler(63, 0, SEEK_SET), EBADF);

		let fd = fdtable::alloc_socket_fd(3, O_RDWR, 0);
		assert_eq!(fs::lseek_handler(fd, 0, SEEK_SET), ESPIPE);
		assert_eq!(fs::lseek_handler(fd, 0, SEEK_CUR), ESPIPE);
	}

	/// 新偏移为负或超出 off_t 的表示范围时返回 EINVAL, 且偏移量字段保持原值。
	#[test]
	fn test_lseek_rejects_a_negative_resulting_offset() {
		let _guard = world();
		let root = make_root("seekoff");
		let path = under(&root, "f");
		let seed = b"0123456789";
		let fd = open(&path, O_CREAT | O_RDWR);
		assert_eq!(fs::write_handler(fd, seed.as_ptr(), 10), 10);

		assert_eq!(fs::lseek_handler(fd, 4, SEEK_SET), 4);
		// 负的新偏移: 偏移量字段是无符号的, 按无符号相加会把负值折成很大的正偏移
		assert_eq!(fs::lseek_handler(fd, (-5i64) as u64, SEEK_SET), EINVAL);
		assert_eq!(fs::lseek_handler(fd, 0, SEEK_CUR), 4);
		assert_eq!(fs::lseek_handler(fd, (-5i64) as u64, SEEK_CUR), EINVAL);
		assert_eq!(fs::lseek_handler(fd, (-11i64) as u64, SEEK_END), EINVAL);
		assert_eq!(fs::lseek_handler(fd, 0, SEEK_CUR), 4);
		// 超出 off_t 的表示范围
		assert_eq!(fs::lseek_handler(fd, i64::MAX as u64, SEEK_END), EINVAL);
		assert_eq!(fs::lseek_handler(fd, u64::MAX, SEEK_SET), EINVAL);
		// 未定义的基准
		assert_eq!(fs::lseek_handler(fd, 0, 7), EINVAL);
		assert_eq!(fs::lseek_handler(fd, 0, SEEK_CUR), 4);

		assert_eq!(fs::lseek_handler(fd, 3, SEEK_END), 13);
		assert_eq!(fs::lseek_handler(fd, 0, SEEK_CUR), 13);
	}

	/// F_GETFD 与 F_SETFD 成对读出与写入 FD_CLOEXEC, F_SETFL 不改动该位。
	#[test]
	fn test_fcntl_descriptor_flags_round_trip() {
		let _guard = world();
		let root = make_root("cloexec");
		let fd = open(&under(&root, "f"), O_CREAT | O_RDONLY);
		assert_eq!(fs::fcntl_handler(0, F_GETFD, 0), 0);
		assert_eq!(fs::fcntl_handler(fd, F_GETFD, 0), 0);
		assert_eq!(fs::fcntl_handler(fd, F_SETFD, FD_CLOEXEC), 0);
		assert_eq!(fs::fcntl_handler(fd, F_GETFD, 0), FD_CLOEXEC);
		// F_SETFL 只改状态标志, 描述符标志属另一套位, 不受影响
		assert_eq!(fs::fcntl_handler(fd, F_SETFL, O_APPEND as u64), 0);
		assert_eq!(fs::fcntl_handler(fd, F_GETFD, 0), FD_CLOEXEC);
		assert_eq!(fs::fcntl_handler(fd, F_SETFD, 0), 0);
		assert_eq!(fs::fcntl_handler(fd, F_GETFD, 0), 0);
		// 越界与空闲槽位
		assert_eq!(fs::fcntl_handler(63, F_GETFD, 0), EBADF);
		assert_eq!(fs::fcntl_handler(63, F_SETFD, FD_CLOEXEC), EBADF);
	}

	/// F_DUPFD 取不小于 arg 的最小编号并清除执行时关闭标志, F_DUPFD_CLOEXEC 置该位。
	#[test]
	fn test_fcntl_duplicate_takes_the_lowest_number_at_or_above_arg() {
		let _guard = world();
		let root = make_root("dupfd");
		let fd = open(&under(&root, "f"), O_CREAT | O_RDONLY);
		assert_eq!(fd, 3);
		// 空出编号 1 与 2: arg 取 0 时得到 1, 不受源描述符编号约束
		assert_eq!(fs::close_handler(1), 0);
		assert_eq!(fs::close_handler(2), 0);
		assert_eq!(fs::fcntl_handler(fd, F_SETFD, FD_CLOEXEC), 0);

		let plain = fs::fcntl_handler(fd, F_DUPFD, 0);
		assert_eq!(plain, 1);
		assert_eq!(fs::fcntl_handler(plain, F_GETFD, 0), 0);

		let cloexec = fs::fcntl_handler(fd, F_DUPFD_CLOEXEC, 0);
		assert_eq!(cloexec, 2);
		assert_eq!(fs::fcntl_handler(cloexec, F_GETFD, 0), FD_CLOEXEC);

		// arg 不小于描述符表容量
		assert_eq!(fs::fcntl_handler(fd, F_DUPFD, 64), EINVAL);
	}

	/// dup 与 dup3 复制的是描述本身: 偏移与状态标志两处共享, 执行时关闭标志属描述符自身,
	/// 不随复制传递。
	#[test]
	fn test_dup_shares_the_open_file_description() {
		let _guard = world();
		let root = make_root("dupshare");
		let path = under(&root, "f");
		let seed = b"0123456789";
		let fd = open(&path, O_CREAT | O_RDWR);
		assert_eq!(fs::write_handler(fd, seed.as_ptr(), 10), 10);
		assert_eq!(fs::lseek_handler(fd, 0, SEEK_SET), 0);
		assert_eq!(fs::fcntl_handler(fd, F_SETFD, FD_CLOEXEC), 0);

		let dup_fd = fs::dup_handler(fd);
		assert!(dup_fd < 64);
		// 执行时关闭标志只在源描述符上
		assert_eq!(fs::fcntl_handler(dup_fd, F_GETFD, 0), 0);

		// 偏移属描述: 一条描述符读出的字节, 另一条接着读
		let mut buf = [0u8; 4];
		assert_eq!(fs::read_handler(fd, buf.as_mut_ptr(), 4), 4);
		assert_eq!(&buf[..4], b"0123");
		assert_eq!(fs::read_handler(dup_fd, buf.as_mut_ptr(), 4), 4);
		assert_eq!(&buf[..4], b"4567");
		// 一条描述符重定位, 另一条的偏移随之改变
		assert_eq!(fs::lseek_handler(dup_fd, 0, SEEK_SET), 0);
		assert_eq!(fs::read_handler(fd, buf.as_mut_ptr(), 2), 2);
		assert_eq!(&buf[..2], b"01");

		// 状态标志属描述: 一条描述符设置 O_APPEND, 另一条读到
		assert_eq!(fs::fcntl_handler(fd, F_SETFL, O_APPEND as u64), 0);
		assert_eq!(fs::fcntl_handler(dup_fd, F_GETFL, 0) & O_APPEND as u64, O_APPEND as u64);

		// dup3 同样共享描述, 执行时关闭标志按 flags 给出
		assert_eq!(fs::dup3_handler(dup_fd, 10, O_CLOEXEC as u64), 10);
		assert_eq!(fs::fcntl_handler(10, F_GETFD, 0), FD_CLOEXEC);
		assert_eq!(fs::fcntl_handler(10, F_GETFL, 0) & O_APPEND as u64, O_APPEND as u64);
		assert_eq!(fs::dup3_handler(fd, 11, 0), 11);
		assert_eq!(fs::fcntl_handler(11, F_GETFD, 0), 0);
	}

	/// F_GETFL 只回报状态标志: 建立描述时用到的 O_CREAT、O_EXCL、O_TRUNC 与执行时关闭位
	/// 都不在其中, 访问模式位保留。
	#[test]
	fn test_fgetfl_reports_only_status_flags() {
		let _guard = world();
		let root = make_root("getfl");
		let fd = open(&under(&root, "f"), O_CREAT | O_EXCL | O_TRUNC | O_RDWR | O_CLOEXEC);
		let flags = fs::fcntl_handler(fd, F_GETFL, 0);
		assert_eq!(flags & 0o3, O_RDWR as u64);
		assert_eq!(flags & (O_CREAT | O_EXCL | O_TRUNC) as u64, 0);
		assert_eq!(flags & O_CLOEXEC as u64, 0);
		// 执行时关闭位记在该描述符自身的表项上
		assert_eq!(fs::fcntl_handler(fd, F_GETFD, 0), FD_CLOEXEC);
	}

	/// F_SETFL 只改动状态标志: 访问模式位与执行时关闭位不受影响, 掩码之外的打开标志位
	/// 不会被置入状态标志。
	#[test]
	fn test_fsetfl_only_changes_the_status_flags() {
		let _guard = world();
		let root = make_root("setfl");
		let fd = open(&under(&root, "f"), O_CREAT | O_RDWR);
		assert_eq!(fs::fcntl_handler(fd, F_SETFD, FD_CLOEXEC), 0);
		assert_eq!(fs::fcntl_handler(fd, F_SETFL, (O_APPEND | O_CREAT | O_EXCL) as u64), 0);
		let flags = fs::fcntl_handler(fd, F_GETFL, 0);
		assert_eq!(flags & 0o3, O_RDWR as u64);
		assert_eq!(flags & O_APPEND as u64, O_APPEND as u64);
		assert_eq!(flags & (O_CREAT | O_EXCL) as u64, 0);
		assert_eq!(fs::fcntl_handler(fd, F_GETFD, 0), FD_CLOEXEC);
	}

	/// 描述绑定的套接字槽位在最后一条引用释放时才归还协议栈; 在此之前关闭其中一条描述符
	/// 不释放它, 关闭后该编号的查找一律返回 EBADF。
	#[test]
	fn test_socket_slot_is_released_at_the_last_descriptor() {
		let _guard = world();
		let fd = fdtable::alloc_socket_fd(3, O_RDWR, 0);
		let dup_fd = fs::dup_handler(fd);
		assert!(dup_fd < 64);
		assert_eq!(fs::close_handler(fd), 0);
		assert!(net::CLOSED.lock().unwrap().is_empty());
		// 已关闭的编号取不到描述
		let mut buf = [0u8; 4];
		assert_eq!(fs::read_handler(fd, buf.as_mut_ptr(), 4), EBADF);
		assert_eq!(fs::fcntl_handler(fd, F_GETFL, 0), EBADF);
		assert_eq!(fs::close_handler(fd), EBADF);
		// 另一条描述符仍指向同一槽位
		assert_eq!(fs::write_handler(dup_fd, buf.as_ptr(), 4), 4);
		assert_eq!(fs::close_handler(dup_fd), 0);
		assert_eq!(*net::CLOSED.lock().unwrap(), vec![3]);
	}

	/// dup3 落在已打开的编号上时先释放该编号原有的描述, 该描述是最后一条引用时一并释放
	/// 它绑定的资源。
	#[test]
	fn test_dup3_releases_the_replaced_descriptor() {
		let _guard = world();
		let root = make_root("dup3replace");
		let target = fdtable::alloc_socket_fd(3, O_RDWR, 0);
		assert_eq!(fs::dup3_handler(target, target, 0), EINVAL);

		let fd = open(&under(&root, "f"), O_CREAT | O_RDONLY);
		assert_eq!(fs::dup3_handler(fd, target, 0), target);
		assert_eq!(*net::CLOSED.lock().unwrap(), vec![3]);
		let mut st = StatBuf([0u8; 128]);
		assert_eq!(fs::fstat_handler(target, st.0.as_mut_ptr() as u64), 0);
		let mode = u32::from_le_bytes(st.0[16..20].try_into().unwrap());
		assert_eq!(mode & 0o170000, 0o100000);
	}

	/// 两个编号已指向同一描述时, dup3 的复制与释放落在同一描述上, 引用计数净增零: 关闭
	/// 其中一个编号不释放该描述, 关闭最后一个才释放它绑定的套接字槽位。
	#[test]
	fn test_dup3_onto_a_descriptor_of_the_same_description_keeps_the_reference_count() {
		let _guard = world();
		let first = fdtable::alloc_socket_fd(3, O_RDWR, 0);
		let second = fs::dup_handler(first);
		assert!(second < 64);
		assert_eq!(fs::dup3_handler(first, second, 0), second);
		// 复制加一条引用、替换释放一条引用, 故要两次关闭才归还槽位
		assert_eq!(fs::close_handler(first), 0);
		assert!(net::CLOSED.lock().unwrap().is_empty());
		let buf = [0u8; 4];
		assert_eq!(fs::write_handler(second, buf.as_ptr(), 4), 4);
		assert_eq!(fs::close_handler(second), 0);
		assert_eq!(*net::CLOSED.lock().unwrap(), vec![3]);
	}

	/// fsync 对已打开的描述符成功, 对空闲槽位与越界编号返回 EBADF。
	#[test]
	fn test_fsync_reports_ebadf_for_a_free_slot() {
		let _guard = world();
		let root = make_root("fsync");
		let fd = open(&under(&root, "f"), O_CREAT | O_RDONLY);
		assert_eq!(fs::fsync_handler(0), 0);
		assert_eq!(fs::fsync_handler(fd), 0);
		assert_eq!(fs::close_handler(fd), 0);
		assert_eq!(fs::fsync_handler(fd), EBADF);
		assert_eq!(fs::fsync_handler(63), EBADF);
		assert_eq!(fs::fsync_handler(64), EBADF);
	}

	/// 描述符一经释放, 该编号上的查找一律返回 EBADF, 不再触及原来的对象。
	#[test]
	fn test_a_released_descriptor_number_yields_ebadf() {
		let _guard = world();
		let root = make_root("released");
		let fd = open(&under(&root, "f"), O_CREAT | O_RDWR);
		assert_eq!(fs::close_handler(fd), 0);
		let mut buf = [0u8; 8];
		let mut st = StatBuf([0u8; 128]);
		assert_eq!(fs::read_handler(fd, buf.as_mut_ptr(), 8), EBADF);
		assert_eq!(fs::write_handler(fd, buf.as_ptr(), 8), EBADF);
		assert_eq!(fs::readv_handler(fd, 0, 0), EBADF);
		assert_eq!(fs::writev_handler(fd, 0, 0), EBADF);
		assert_eq!(fs::lseek_handler(fd, 0, SEEK_SET), EBADF);
		assert_eq!(fs::pread64_handler(fd, buf.as_mut_ptr(), 8, 0), EBADF);
		assert_eq!(fs::pwrite64_handler(fd, buf.as_ptr(), 8, 0), EBADF);
		assert_eq!(fs::fstat_handler(fd, st.0.as_mut_ptr() as u64), EBADF);
		assert_eq!(fs::getdents_handler(fd, buf.as_mut_ptr(), 8), EBADF);
		assert_eq!(fs::ftruncate_handler(fd, 0), EBADF);
		assert_eq!(fs::fsync_handler(fd), EBADF);
		assert_eq!(fs::close_handler(fd), EBADF);
		let mut pfds = [PollFd { fd: fd as i32, events: POLLIN, revents: 0 }];
		assert_eq!(fs::ppoll_handler(pfds.as_mut_ptr() as u64, 1, 0), 1);
		assert_eq!(pfds[0].revents, POLLNVAL);
	}

	/// getcwd 返回写入的字节数, 含结尾空字节; 缓冲放不下这两个字节时返回 ERANGE,
	/// 缓冲取空返回 EFAULT。
	#[test]
	fn test_getcwd_reports_the_written_length() {
		let _guard = world();
		let mut buf = [0xFFu8; 8];
		assert_eq!(fs::getcwd_handler(buf.as_mut_ptr() as u64, 8), 2);
		assert_eq!(&buf[..2], b"/\0");
		// 结尾空字节之后的字节不被写入
		assert_eq!(buf[2], 0xFF);
		assert_eq!(fs::getcwd_handler(buf.as_mut_ptr() as u64, 2), 2);
		assert_eq!(fs::getcwd_handler(buf.as_mut_ptr() as u64, 1), ERANGE);
		assert_eq!(fs::getcwd_handler(buf.as_mut_ptr() as u64, 0), ERANGE);
		assert_eq!(fs::getcwd_handler(0, 8), EFAULT);
	}

	/// ppoll 的描述符个数以打开文件表的容量 64 为上界, 超过时返回 EINVAL 且不读取数组。
	/// 内核的上界是 `RLIMIT_NOFILE`, 即最大描述符编号加一, 与本层的表容量指同一个量;
	/// 个数为 0 时不读取数组直接返回 0。
	#[test]
	fn test_ppoll_bounds_the_descriptor_count() {
		let _guard = world();
		let mut pfds = [PollFd { fd: 0, events: POLLIN, revents: 0 }];
		assert_eq!(fs::ppoll_handler(pfds.as_mut_ptr() as u64, 65, 0), EINVAL);
		assert_eq!(fs::ppoll_handler(0, 65, 0), EINVAL);
		// 个数为 0 时不读取数组, 故数组指针取空也返回 0
		let zero = timespec(0, 0);
		assert_eq!(fs::ppoll_handler(0, 0, &zero as *const types::Timespec as u64), 0);
		// 个数非 0 而数组指针取空则返回 EFAULT
		assert_eq!(fs::ppoll_handler(0, 1, 0), EFAULT);
		assert_eq!(fs::ppoll_handler(pfds.as_mut_ptr() as u64, 1, 0), 1);
	}

	/// 造一个时限。返回的值的地址即系统调用收到的时限指针。
	fn timespec(sec: i64, nsec: i64) -> types::Timespec {
		types::Timespec { tv_sec: sec, tv_nsec: nsec }
	}

	/// 落在 [deadline, 期限] 之外且永不就绪的描述符: 编号为负, 就绪事件恒为 0。
	const NEVER_READY_FD: i32 = -1;

	/// ppoll 在描述符已就绪时不进入阻塞, 时限参数不影响该次扫描。
	///
	/// 该次调用登记过定时器 (有时限时), 返回前须收起, 故返回后表中不留下条目。
	#[test]
	fn test_ppoll_returns_ready_descriptors_without_waiting() {
		let _guard = world();
		let mut pfds = [PollFd { fd: 0, events: POLLIN, revents: 0 }];
		let limit = timespec(5, 0);
		let ready = fs::ppoll_handler(pfds.as_mut_ptr() as u64, 1, &limit as *const types::Timespec as u64);
		assert_eq!(ready, 1);
		assert_eq!(pfds[0].revents, POLLIN);
		assert_eq!(thread::BLOCK_COUNT.load(Ordering::SeqCst), 0);
		assert!(!timer::is_armed(thread::current_thread_id()));
	}

	/// 时长为 0 的 ppoll 只扫描一次: 无描述符就绪时返回 0, 不进入阻塞。
	#[test]
	fn test_ppoll_with_a_zero_timeout_does_not_block() {
		let _guard = world();
		let mut pfds = [PollFd { fd: NEVER_READY_FD, events: POLLIN, revents: 0 }];
		let limit = timespec(0, 0);
		assert_eq!(
			fs::ppoll_handler(pfds.as_mut_ptr() as u64, 1, &limit as *const types::Timespec as u64),
			0
		);
		assert_eq!(pfds[0].revents, 0);
		assert_eq!(thread::BLOCK_COUNT.load(Ordering::SeqCst), 0);
		assert!(!timer::is_armed(thread::current_thread_id()));
	}

	/// 时限字段越界的 ppoll 返回 EINVAL 且不读取描述符数组。
	#[test]
	fn test_ppoll_rejects_an_invalid_timeout() {
		let _guard = world();
		let mut pfds = [PollFd { fd: 0, events: POLLIN, revents: 0 }];
		let too_many_nsec = timespec(0, 1_000_000_000);
		let negative_sec = timespec(-1, 0);
		let negative_nsec = timespec(0, -1);
		for limit in [too_many_nsec, negative_sec, negative_nsec] {
			assert_eq!(
				fs::ppoll_handler(
					pfds.as_mut_ptr() as u64,
					1,
					&limit as *const types::Timespec as u64
				),
				EINVAL
			);
		}
		assert_eq!(pfds[0].revents, 0);
		assert_eq!(thread::BLOCK_COUNT.load(Ordering::SeqCst), 0);
		assert!(!timer::is_armed(thread::current_thread_id()));
	}

	/// 无描述符就绪时 ppoll 阻塞于 [`fs::POLL_WAIT_ADDR`], 并按已登记的截止时刻等待: 时钟源
	/// 未到该时刻则再次阻塞, 到达该时刻即返回 0。
	///
	/// 等待时长取 5 ms, 即时间片周期 (config.mk 的 TIMER_INTERVAL 为 10000 个计数单位, 折合
	/// 1 ms) 的 5 倍, 故超时判定只可能来自截止时刻本身。
	#[test]
	fn test_ppoll_waits_until_its_deadline() {
		let _guard = world();
		let me = thread::current_thread_id();
		let mut pfds = [PollFd { fd: NEVER_READY_FD, events: POLLIN, revents: 0 }];
		let limit = timespec(0, 5_000_000);
		let limit_addr = &limit as *const types::Timespec as u64;

		crate::csr::set_time(1_000);
		assert_eq!(fs::ppoll_handler(pfds.as_mut_ptr() as u64, 1, limit_addr), 0);
		assert_eq!(thread::BLOCK_COUNT.load(Ordering::SeqCst), 1);
		assert_eq!(
			thread::BLOCK_WAIT_ADDR.load(Ordering::SeqCst),
			fs::POLL_WAIT_ADDR
		);
		// 阻塞期间登记了切换目标, 故本次调用请求重启, 截止时刻留在表中。
		assert_eq!(thread::RESTART_COUNT.load(Ordering::SeqCst), 1);
		// 5 ms 折合 50_000 个计数单位 (一个计数单位为 100 ns)。
		assert_eq!(timer::next_deadline(), Some(51_000));

		// 重新执行本次调用 (清除切换登记): 截止时刻未到, 沿用同一条目再次阻塞。
		thread::SWITCH_PENDING.store(false, Ordering::SeqCst);
		crate::csr::set_time(50_999);
		assert_eq!(fs::ppoll_handler(pfds.as_mut_ptr() as u64, 1, limit_addr), 0);
		assert_eq!(thread::BLOCK_COUNT.load(Ordering::SeqCst), 2);
		assert_eq!(timer::next_deadline(), Some(51_000));

		// 到达截止时刻: 返回 0 且不再阻塞, 条目收起。
		thread::SWITCH_PENDING.store(false, Ordering::SeqCst);
		crate::csr::set_time(51_000);
		assert_eq!(fs::ppoll_handler(pfds.as_mut_ptr() as u64, 1, limit_addr), 0);
		assert_eq!(thread::BLOCK_COUNT.load(Ordering::SeqCst), 2);
		assert!(!timer::is_armed(me));
	}

}
