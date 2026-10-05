//! 系统调用模块。
//!
//! 细分：
//! - `types` — repr(C) ABI 结构体
//! - `io`   — read/write/writev/close/lseek
//! - `mmap` — mmap (brk 在 crate::mem::sys_brk_handler)
//! - `proc` — 标识/时间/资源类处理函数
//! - `concurrency` — 用户侧并发编程接口 (线程/进程/信号/futex/调度)
//! - `net`  — 套接字接口 (AF_INET 的 SOCK_DGRAM)
//! - `fdtable` — 描述符表与打开文件描述表 (控制台/文件系统节点/套接字三类描述共用)
//!
//! 遵循 smode_entry/syscall.h 的 sysnum 前缀约定。
#![allow(dead_code)]

pub mod concurrency;
mod errno;
mod fdtable;
pub mod fs;
mod io;
mod mmap;
mod net;
mod proc;
mod rand;
mod types;

use crate::constants::LINEAR_MAP_OFFSET;
use crate::mem;
#[cfg(feature = "diagnostic")]
use crate::println;
use crate::trap::TrapGprs;

use types::*;

// ---------------------------------------------------------------
//  Errno
// ---------------------------------------------------------------

pub use errno::*;

// ---------------------------------------------------------------
//  Syscall numbers  (prefixed SYSNUM_ per smode_entry convention)
// ---------------------------------------------------------------

pub const SYSNUM_GETCWD: u64 = 17;
pub const SYSNUM_DUP: u64 = 23;
pub const SYSNUM_DUP3: u64 = 24;
pub const SYSNUM_FCNTL: u64 = 25;
pub const SYSNUM_IOCTL: u64 = 29;
pub const SYSNUM_MKDIRAT: u64 = 34;
pub const SYSNUM_UNLINKAT: u64 = 35;
pub const SYSNUM_STATFS: u64 = 43;
pub const SYSNUM_FTRUNCATE: u64 = 46;
pub const SYSNUM_FACCESSAT: u64 = 48;
pub const SYSNUM_FCHOWN: u64 = 55;
pub const SYSNUM_OPENAT: u64 = 56;
pub const SYSNUM_CLOSE: u64 = 57;
pub const SYSNUM_VHANGUP: u64 = 58;
pub const SYSNUM_GETDENTS: u64 = 61;
pub const SYSNUM_LSEEK: u64 = 62;
pub const SYSNUM_READ: u64 = 63;
pub const SYSNUM_WRITE: u64 = 64;
pub const SYSNUM_READV: u64 = 65;
pub const SYSNUM_WRITEV: u64 = 66;
pub const SYSNUM_PREAD: u64 = 67;
pub const SYSNUM_PWRITE: u64 = 68;
pub const SYSNUM_PPOLL: u64 = 73;
pub const SYSNUM_FSTATAT: u64 = 79;
pub const SYSNUM_FSTAT: u64 = 80;
pub const SYSNUM_FSYNC: u64 = 82;
pub const SYSNUM_EXIT: u64 = 93;
pub const SYSNUM_EXIT_GROUP: u64 = 94;
pub const SYSNUM_SET_TID_ADDRESS: u64 = 96;
pub const SYSNUM_FUTEX: u64 = 98;
pub const SYSNUM_NANOSLEEP: u64 = 101;
pub const SYSNUM_SETITIMER: u64 = 103;
pub const SYSNUM_CLOCK_GETTIME: u64 = 113;
pub const SYSNUM_SCHED_SETAFFINITY: u64 = 122;
pub const SYSNUM_SCHED_GETAFFINITY: u64 = 123;
pub const SYSNUM_SCHED_YIELD: u64 = 124;
pub const SYSNUM_KILL: u64 = 129;
pub const SYSNUM_TKILL: u64 = 130;
pub const SYSNUM_TGKILL: u64 = 131;
pub const SYSNUM_SIGNALSTACK: u64 = 132;
pub const SYSNUM_RT_SIGACTION: u64 = 134;
pub const SYSNUM_SIGPROCMASK: u64 = 135;
pub const SYSNUM_RT_SIGRETURN: u64 = 139;
pub const SYSNUM_TIMES: u64 = 153;
pub const SYSNUM_SETPGID: u64 = 154;
pub const SYSNUM_GETPGID: u64 = 155;
pub const SYSNUM_SETSID: u64 = 157;
pub const SYSNUM_UNAME: u64 = 160;
pub const SYSNUM_UMASK: u64 = 166;
pub const SYSNUM_PRCTL: u64 = 167;
pub const SYSNUM_GETCPU: u64 = 168;
pub const SYSNUM_GETTIMEOFDAY: u64 = 169;
pub const SYSNUM_GETPID: u64 = 172;
pub const SYSNUM_GETPPID: u64 = 173;
pub const SYSNUM_GETUID: u64 = 174;
pub const SYSNUM_GETEUID: u64 = 175;
pub const SYSNUM_GETGID: u64 = 176;
pub const SYSNUM_GETEGID: u64 = 177;
pub const SYSNUM_GETTID: u64 = 178;
pub const SYSNUM_SYSINFO: u64 = 179;
pub const SYSNUM_GETRUSAGE: u64 = 165;
pub const SYSNUM_SOCKET: u64 = 198;
pub const SYSNUM_SOCKETPAIR: u64 = 199;
pub const SYSNUM_BIND: u64 = 200;
pub const SYSNUM_LISTEN: u64 = 201;
pub const SYSNUM_CONNECT: u64 = 203;
pub const SYSNUM_GETSOCKNAME: u64 = 204;
pub const SYSNUM_GETPEERNAME: u64 = 205;
pub const SYSNUM_SENDTO: u64 = 206;
pub const SYSNUM_RECVFROM: u64 = 207;
pub const SYSNUM_SETSOCKOPT: u64 = 208;
pub const SYSNUM_GETSOCKOPT: u64 = 209;
pub const SYSNUM_SHUTDOWN: u64 = 210;
pub const SYSNUM_SENDMSG: u64 = 211;
pub const SYSNUM_RECVMSG: u64 = 212;
pub const SYSNUM_BRK: u64 = 214;
pub const SYSNUM_MUNMAP: u64 = 215;
pub const SYSNUM_MREMAP: u64 = 216;
pub const SYSNUM_CLONE: u64 = 220;
pub const SYSNUM_MMAP: u64 = 222;
pub const SYSNUM_FADVISE64: u64 = 223;
pub const SYSNUM_MPROTECT: u64 = 226;
pub const SYSNUM_MSYNC: u64 = 227;
pub const SYSNUM_MADVISE: u64 = 233;
pub const SYSNUM_ACCEPT4: u64 = 242;
pub const SYSNUM_WAIT4: u64 = 260;
pub const SYSNUM_PRLIMIT64: u64 = 261;
pub const SYSNUM_GETRANDOM: u64 = 278;

// ---------------------------------------------------------------
//  入口：syscall_handler  (per smode_entry/trap_handler.c)
// ---------------------------------------------------------------

/// 系统调用分发。返回的值写入 gprs.a0。
///
/// 陷阱帧以可变引用传入: rt_sigreturn 需要在调用内改写整个帧以恢复信号投递前的
/// 寄存器现场, 其余调用只经 a0 传出返回值。
pub fn syscall_handler(gprs: &mut TrapGprs) -> u64 {
    let num = gprs.a7();
    let a0 = gprs.a0();
    let a1 = gprs.a1();
    let a2 = gprs.a2();
    let a3 = gprs.a3();
    let a4 = gprs.a4();
    let a5 = gprs.a5();

    let ret = match num {
        // ---- I/O / 文件系统 ----
        SYSNUM_OPENAT => fs::openat_handler(a0, a1, a2, a3),
        SYSNUM_CLOSE => fs::close_handler(a0),
        SYSNUM_DUP => fs::dup_handler(a0),
        SYSNUM_DUP3 => fs::dup3_handler(a0, a1, a2),
        SYSNUM_LSEEK => fs::lseek_handler(a0, a1, a2),
        SYSNUM_READ => fs::read_handler(a0, a1 as *mut u8, a2),
        SYSNUM_WRITE => fs::write_handler(a0, a1 as *const u8, a2),
        SYSNUM_READV => fs::readv_handler(a0, a1, a2),
        SYSNUM_WRITEV => fs::writev_handler(a0, a1, a2),
        SYSNUM_PREAD => fs::pread64_handler(a0, a1 as *mut u8, a2, a3),
        SYSNUM_PWRITE => fs::pwrite64_handler(a0, a1 as *const u8, a2, a3),
        SYSNUM_FSTATAT => fs::fstatat_handler(a0, a1, a2, a3),
        SYSNUM_FSTAT => fs::fstat_handler(a0, a1),
        SYSNUM_GETDENTS => fs::getdents_handler(a0, a1 as *mut u8, a2),
        SYSNUM_GETCWD => fs::getcwd_handler(a0, a1),
        SYSNUM_MKDIRAT => fs::mkdirat_handler(a0, a1, a2),
        SYSNUM_UNLINKAT => fs::unlinkat_handler(a0, a1, a2),
        SYSNUM_FACCESSAT => fs::faccessat_handler(a0, a1, a2, a3),
        SYSNUM_FCNTL => fs::fcntl_handler(a0, a1, a2),
        SYSNUM_FSYNC => fs::fsync_handler(a0),
        SYSNUM_FTRUNCATE => fs::ftruncate_handler(a0, a1),
        SYSNUM_PPOLL => fs::ppoll_handler(a0, a1, a2),

        // ---- 随机数 ----
        SYSNUM_GETRANDOM => self::rand::getrandom_handler(a0 as *mut u8, a1, a2),

        // ---- 套接字 ----
        SYSNUM_SOCKET => net::socket_handler(a0, a1, a2),
        SYSNUM_BIND => net::bind_handler(a0, a1, a2),
        SYSNUM_CONNECT => net::connect_handler(a0, a1, a2),
        SYSNUM_GETSOCKNAME => net::getsockname_handler(a0, a1, a2),
        SYSNUM_GETPEERNAME => net::getpeername_handler(a0, a1, a2),
        SYSNUM_SENDTO => net::sendto_handler(a0, a1, a2, a3, a4, a5),
        SYSNUM_RECVFROM => net::recvfrom_handler(a0, a1, a2, a3, a4, a5),
        SYSNUM_SETSOCKOPT => net::setsockopt_handler(a0, a1, a2, a3, a4),
        SYSNUM_GETSOCKOPT => net::getsockopt_handler(a0, a1, a2, a3, a4),
        SYSNUM_SHUTDOWN => net::shutdown_handler(a0, a1),
        SYSNUM_SENDMSG => net::sendmsg_handler(a0, a1, a2),
        SYSNUM_RECVMSG => net::recvmsg_handler(a0, a1, a2),
        // socketpair 只服务 AF_UNIX, listen 与 accept4 只服务 SOCK_STREAM,
        // 三者在本轮的 AF_INET 的 SOCK_DGRAM 范围内均无对应实现。
        SYSNUM_SOCKETPAIR | SYSNUM_LISTEN | SYSNUM_ACCEPT4 => EOPNOTSUPP,

        // ---- 进程控制 ----
        // exit(93) 仅终止当前线程; exit_group(94) 终止整个线程组 (进程)。
        SYSNUM_EXIT => concurrency::thread::exit_handler(a0),
        SYSNUM_EXIT_GROUP => concurrency::proc::exit_group_handler(a0),
        SYSNUM_CLONE => concurrency::thread::clone_handler(gprs, a0, a1, a2, a3, a4),
        SYSNUM_SCHED_YIELD => concurrency::thread::sched_yield_handler(),
        SYSNUM_SET_TID_ADDRESS => concurrency::thread::set_tid_address_handler(a0),
        SYSNUM_FUTEX => concurrency::thread::futex_handler(a0, a1, a2, a3, a4),
        SYSNUM_WAIT4 => concurrency::proc::wait4_handler(a0, a1, a2, a3),
        SYSNUM_SETITIMER => concurrency::proc::setitimer_handler(a0, a1, a2),
        SYSNUM_NANOSLEEP => concurrency::thread::nanosleep_handler(a0, a1),

        // ---- 信号 ----
        SYSNUM_KILL => concurrency::proc::kill_handler(a0, a1),
        SYSNUM_TKILL => concurrency::proc::tkill_handler(a0, a1),
        SYSNUM_TGKILL => concurrency::proc::tgkill_handler(a0, a1, a2),
        SYSNUM_RT_SIGACTION => concurrency::proc::sigaction_handler(a0, a1, a2, a3),
        SYSNUM_SIGPROCMASK => concurrency::proc::sigprocmask_handler(a0, a1, a2, a3),
        SYSNUM_RT_SIGRETURN => concurrency::proc::sigreturn_handler(gprs),

        // ---- 内存 ----
        SYSNUM_BRK => mem::sys_brk_handler(a0),
        SYSNUM_MMAP => mmap::mmap_handler(a0, a1, a2, a3, a4, a5),
        SYSNUM_MUNMAP => mmap::munmap_handler(a0, a1),
        SYSNUM_MREMAP => mmap::mremap_handler(a0, a1, a2, a3, a4),
        SYSNUM_MPROTECT => mmap::mprotect_handler(a0, a1, a2),

        // ---- 标识 ----
        SYSNUM_UNAME => proc::uname_handler(a0 as *mut Utsname),
        SYSNUM_GETPID => concurrency::proc::getpid_handler(),
        SYSNUM_GETPPID => concurrency::proc::getppid_handler(),
        SYSNUM_GETTID => concurrency::thread::gettid_handler(),
        SYSNUM_GETUID => 0,
        SYSNUM_GETEUID => 0,
        SYSNUM_GETGID => 0,
        SYSNUM_GETEGID => 0,

        // ---- 时间 ----
        SYSNUM_GETTIMEOFDAY => proc::gettimeofday_handler(a0 as *mut Timeval),
        SYSNUM_CLOCK_GETTIME => proc::clock_gettime_handler(a0, a1 as *mut Timespec),
        SYSNUM_TIMES => proc::times_handler(a0 as *mut Tms),

        // ---- 资源 ----
        SYSNUM_GETRUSAGE => proc::getrusage_handler(a1 as *mut Rusage),
        SYSNUM_GETCPU => proc::getcpu_handler(a0 as *mut u32, a1 as *mut u32),
        SYSNUM_PRLIMIT64 => {
            proc::prlimit64_handler(a0, a1 as *const Rlimit, a2 as *mut Rlimit)
        }
        SYSNUM_SYSINFO => proc::sysinfo_handler(a0 as *mut Sysinfo),

        // ---- 跳过 (无副作用) ----
        SYSNUM_IOCTL
        | SYSNUM_STATFS
        | SYSNUM_FCHOWN
        | SYSNUM_VHANGUP
        | SYSNUM_SCHED_SETAFFINITY
        | SYSNUM_SCHED_GETAFFINITY
        | SYSNUM_SIGNALSTACK
        | SYSNUM_SETPGID
        | SYSNUM_GETPGID
        | SYSNUM_SETSID
        | SYSNUM_UMASK
        | SYSNUM_PRCTL
        | SYSNUM_FADVISE64
        | SYSNUM_MSYNC
        | SYSNUM_MADVISE => 0,

        // ---- 未知 ----
        _ => {
            #[cfg(feature = "diagnostic")]
            println!("syscall: unknown {}\n", num);
            ENOSYS
        }
    };
    ret
}

// ---------------------------------------------------------------
//  载荷镜像末尾的宿主文件清单 (compound payload trailer)
// ---------------------------------------------------------------

/// trailer 固定长度 (magic 8 字节 + 四个 u32 偏移/长度)。
const MANIFEST_TRAILER: u64 = 24;
/// trailer 起始 8 字节的识别魔数。
const MANIFEST_MAGIC: &[u8; 8] = b"VSFMANIF";

/// 从物理地址 *p* 读取小端 u32。
#[inline]
unsafe fn read_u32_le(p: *const u8) -> u32 {
    let mut b = [0u8; 4];
    for i in 0..4 {
        b[i] = unsafe { p.add(i).read_volatile() };
    }
    u32::from_le_bytes(b)
}

/// 解析载荷镜像末尾的宿主文件清单, 把 (path, data) 逐条注入文件系统。
///
/// 布局 (见 tee_ecall_regress.c 构造的 compound payload, 全部小端):
///   [0..8)   magic = b"VSFMANIF"
///   [8..12)  path_off (u32) — 相对 payload_pa 的路径字节偏移
///   [12..16) path_len (u32)
///   [16..20) data_off (u32) — 相对 payload_pa 的数据字节偏移
///   [20..24) data_len (u32)
/// trailer 紧贴载荷末尾 (payload_pa + payload_size - 24)。
///
/// 返回注入的文件数 (0 或 1)。无合法清单、越界或注入失败时返回 0, 不中断启动:
/// 多数载荷不含清单, 这是常态而非错误。
pub fn inject_manifest(payload_pa: u64, payload_size: u64) -> usize {
    if payload_size < MANIFEST_TRAILER {
        return 0;
    }
    // MMU 已开, 载荷物理区经 LINEAR_MAP_OFFSET 别名可读。
    let base = payload_pa.wrapping_add(LINEAR_MAP_OFFSET);
    let trailer = (base + payload_size - MANIFEST_TRAILER) as *const u8;

    unsafe {
        for i in 0..MANIFEST_MAGIC.len() {
            if trailer.add(i).read_volatile() != MANIFEST_MAGIC[i] {
                return 0;
            }
        }
        let path_off = read_u32_le(trailer.add(8)) as u64;
        let path_len = read_u32_le(trailer.add(12)) as u64;
        let data_off = read_u32_le(trailer.add(16)) as u64;
        let data_len = read_u32_le(trailer.add(20)) as u64;

        // 越界保护: 偏移 + 长度不得越过载荷镜像末尾 (trailer 起点)。
        if path_off + path_len > payload_size - MANIFEST_TRAILER
            || data_off + data_len > payload_size - MANIFEST_TRAILER
        {
            return 0;
        }
        let path =
            core::slice::from_raw_parts((base + path_off) as *const u8, path_len as usize);
        let data =
            core::slice::from_raw_parts((base + data_off) as *const u8, data_len as usize);
        if fs::vfs_inject_file(path, data) {
            1
        } else {
            0
        }
    }
}
