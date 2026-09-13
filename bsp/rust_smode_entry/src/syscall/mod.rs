//! 系统调用模块。
//!
//! 细分：
//! - `types` — repr(C) ABI 结构体
//! - `io`   — read/write/writev/close/lseek
//! - `mem`  — mmap (brk 在 crate::mem::sys_brk_handler)
//! - `proc` — 标识/时间/资源类处理函数
//! - `thread` — 线程生命周期与同步原语粘合 (clone/exit/futex/gettid/...)
//!
//! 遵循 smode_entry/syscall.h 的 sysnum 前缀约定。

#![allow(dead_code)]

mod fs;
mod io;
mod mmap;
mod proc;
mod rand;
mod thread;
mod types;

use crate::constants::LINEAR_MAP_OFFSET;
use crate::ecall_aux;
use crate::mem;
#[cfg(feature = "diagnostic")]
use crate::println;
use crate::trap::TrapGprs;

use types::*;

// ---------------------------------------------------------------
//  Errno
// ---------------------------------------------------------------

// 取值一律为 "负的 errno": musl 的 syscall 包装以 errno = -ret 还原, 因此返回
// !0u64 会被解释为 errno 1 (EPERM) 而不是 ENOSYS, 令调用方的 ENOSYS 判据失效。
pub const EFAULT: u64 = (!0u64) - 13; // errno 14 = EFAULT
pub const ENOMEM: u64 = (!0u64) - 11; // errno 12 = ENOMEM
pub const EINVAL: u64 = (!0u64) - 21; // errno 22 = EINVAL
pub const ENOSYS: u64 = (!0u64) - 37; // errno 38 = ENOSYS

// ---------------------------------------------------------------
//  Syscall numbers  (prefixed SYSNUM_ per smode_entry convention)
// ---------------------------------------------------------------

pub const SYSNUM_GETCWD: u64 = 17;
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
pub const SYSNUM_BRK: u64 = 214;
pub const SYSNUM_MUNMAP: u64 = 215;
pub const SYSNUM_MREMAP: u64 = 216;
pub const SYSNUM_CLONE: u64 = 220;
pub const SYSNUM_MMAP: u64 = 222;
pub const SYSNUM_FADVISE64: u64 = 223;
pub const SYSNUM_MPROTECT: u64 = 226;
pub const SYSNUM_MSYNC: u64 = 227;
pub const SYSNUM_MADVISE: u64 = 233;
pub const SYSNUM_WAIT4: u64 = 260;
pub const SYSNUM_PRLIMIT64: u64 = 261;
pub const SYSNUM_GETRANDOM: u64 = 278;

// ---------------------------------------------------------------
//  入口：syscall_handler  (per smode_entry/trap_handler.c)
// ---------------------------------------------------------------

/// 系统调用分发。返回的值写入 gprs.a0。
pub fn syscall_handler(gprs: &TrapGprs) -> u64 {
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

        // ---- 进程控制 ----
        // exit(93) 仅终止当前线程; exit_group(94) 终止整个线程组 (飞地)。
        SYSNUM_EXIT => thread::exit_handler(a0),
        SYSNUM_EXIT_GROUP => ecall_aux::enclave_call_exit(a0),
        SYSNUM_KILL => thread::signal_handler(a1),
        SYSNUM_TKILL => thread::signal_handler(a1),
        SYSNUM_TGKILL => thread::signal_handler(a2),
        SYSNUM_SCHED_YIELD => thread::sched_yield_handler(),
        SYSNUM_SET_TID_ADDRESS => thread::set_tid_address_handler(a0),
        SYSNUM_SIGPROCMASK => thread::sigprocmask_handler(a0, a2, a3),
        SYSNUM_FUTEX => thread::futex_handler(a0, a1, a2, a3, a4),
        SYSNUM_CLONE => thread::clone_handler(gprs, a0, a1, a2, a3, a4),
        SYSNUM_NANOSLEEP => 0,

        // ---- 内存 ----
        SYSNUM_BRK => mem::sys_brk_handler(a0),
        SYSNUM_MMAP => mmap::mmap_handler(a0, a1, a2, a3, a4, a5),
        SYSNUM_MUNMAP => mmap::munmap_handler(a0, a1),
        SYSNUM_MREMAP => mmap::mremap_handler(a0, a1, a2, a3, a4),

        // ---- 标识 ----
        SYSNUM_UNAME => proc::uname_handler(a0 as *mut Utsname),
        SYSNUM_GETPID => 1,
        SYSNUM_GETPPID => 1,
        SYSNUM_GETTID => thread::gettid_handler(),
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
        | SYSNUM_SETITIMER
        | SYSNUM_SCHED_SETAFFINITY
        | SYSNUM_SCHED_GETAFFINITY
        | SYSNUM_SIGNALSTACK
        | SYSNUM_RT_SIGACTION
        | SYSNUM_SETPGID
        | SYSNUM_GETPGID
        | SYSNUM_SETSID
        | SYSNUM_UMASK
        | SYSNUM_PRCTL
        | SYSNUM_FADVISE64
        | SYSNUM_MPROTECT
        | SYSNUM_MSYNC
        | SYSNUM_MADVISE
        | SYSNUM_WAIT4 => 0,

        // ---- 未知 ----
        _ => {
            #[cfg(feature = "diagnostic")]
            println!("syscall: unknown {}\n", num);
            ENOSYS
        }
    };
    // 逐条追踪 U-mode 载荷的 syscall 序列 (仅 diagnostic 构建)。
    // 用途: 定位载荷在 main 之前因某个 syscall 结果而提前 exit_group 的具体调用点。
    #[cfg(feature = "diagnostic")]
    println!("[sys] {} a0=0x{a0:x} a1=0x{a1:x} a2=0x{a2:x} -> 0x{ret:x}\n", num);
    ret
}

// ---------------------------------------------------------------
//  文件系统初始化与载荷注入 (启动阶段经 main.rs 调用)
// ---------------------------------------------------------------

/// 建立文件系统根目录 (幂等)。
pub fn vfs_init() {
    fs::vfs_init();
}

/// 把 *data* 作为文件 *path* 注入文件系统, 沿路目录按需创建。
/// 供启动阶段 (载荷运行前) 用宿主下发的数据预置文件。
pub fn vfs_inject_file(path: &[u8], data: &[u8]) -> bool {
    fs::vfs_inject_file(path, data)
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
