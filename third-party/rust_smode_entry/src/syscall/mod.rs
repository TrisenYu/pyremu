//! 系统调用模块。
//!
//! 细分：
//! - `types` — repr(C) ABI 结构体
//! - `io`   — read/write/writev/close/lseek
//! - `mem`  — mmap (brk 在 crate::memory::sys_brk_handler)
//! - `proc` — 标识/时间/资源类处理函数
//!
//! 遵循 smode_entry/syscall.h 的 sysnum 前缀约定。

#![allow(dead_code)]

mod io;
mod mem;
mod proc;
mod types;

use crate::ecall_aux;
use crate::memory;
use crate::println;
use crate::trap::TrapGprs;

use types::*;

// ---------------------------------------------------------------
//  Errno
// ---------------------------------------------------------------

pub const ENOSYS: u64 = !0u64;
pub const EINVAL: u64 = (!0u64) - 21;

// ---------------------------------------------------------------
//  Syscall numbers  (prefixed SYSNUM_ per smode_entry convention)
// ---------------------------------------------------------------

pub const SYSNUM_GETCWD: u64 = 17;
pub const SYSNUM_FCNTL: u64 = 25;
pub const SYSNUM_IOCTL: u64 = 29;
pub const SYSNUM_STATFS: u64 = 43;
pub const SYSNUM_FCHOWN: u64 = 55;
pub const SYSNUM_OPENAT: u64 = 56;
pub const SYSNUM_CLOSE: u64 = 57;
pub const SYSNUM_VHANGUP: u64 = 58;
pub const SYSNUM_LSEEK: u64 = 62;
pub const SYSNUM_READ: u64 = 63;
pub const SYSNUM_WRITE: u64 = 64;
pub const SYSNUM_READV: u64 = 65;
pub const SYSNUM_WRITEV: u64 = 66;
pub const SYSNUM_PREAD: u64 = 67;
pub const SYSNUM_PWRITE: u64 = 68;
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

    match num {
        // ---- I/O ----
        SYSNUM_WRITE => io::write_handler(a0, a1 as *const u8, a2),
        SYSNUM_READ => io::read_handler(a0, a1 as *mut u8, a2),
        SYSNUM_WRITEV => io::writev_handler(a0, a1, a2),
        SYSNUM_CLOSE => io::close_handler(a0),

        // ---- 进程控制 ----
        SYSNUM_EXIT | SYSNUM_EXIT_GROUP => ecall_aux::enclave_call_exit(a0),
        SYSNUM_SCHED_YIELD => 0,
        SYSNUM_NANOSLEEP => 0,

        // ---- 内存 ----
        SYSNUM_BRK => memory::sys_brk_handler(a0),
        SYSNUM_MMAP => mem::mmap_handler(a0, a1, a2, a3, a4, a5),

        // ---- 标识 ----
        SYSNUM_UNAME => proc::uname_handler(a0 as *mut Utsname),
        SYSNUM_GETPID => 1,
        SYSNUM_GETPPID => 1,
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
        | SYSNUM_SIGPROCMASK
        | SYSNUM_SETPGID
        | SYSNUM_GETPGID
        | SYSNUM_SETSID
        | SYSNUM_UMASK
        | SYSNUM_PRCTL
        | SYSNUM_FADVISE64
        | SYSNUM_MPROTECT
        | SYSNUM_MSYNC
        | SYSNUM_MADVISE
        | SYSNUM_WAIT4
        | SYSNUM_MUNMAP
        | SYSNUM_MREMAP
        | SYSNUM_KILL
        | SYSNUM_SET_TID_ADDRESS
        | SYSNUM_FUTEX => 0,

        // ---- 未知 ----
        _ => {
            println!("syscall: unknown {}\n", num);
            ENOSYS
        }
    }
}
