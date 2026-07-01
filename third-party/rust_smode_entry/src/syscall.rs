//! 系统调用处理。命名遵循 smode_entry/syscall.h，实现紧跟 ref-emod trap/syscalls.c。
//!
//! 已实现: read(63), write(64), exit(93), exit_group(94), nanosleep(101),
//!         clock_gettime(113), sched_yield(124), times(153), uname(160),
//!         getrusage(165), getcpu(168), gettimeofday(169), getpid(172),
//!         getppid(173), getuid(174), geteuid(175), getgid(176), getegid(177),
//!         sysinfo(179), brk(214), prlimit64(261)
//!
//! `SYSNUM_*` 常量为 Linux rv64 完整系统调用表，未实现编号有意保留供后续补全。

#![allow(dead_code)]

use crate::call;
use crate::constants::*;
use crate::memory;
use crate::println;
use crate::trap::TrapGprs;
use crate::uart;

// ---------------------------------------------------------------
//  Syscall numbers  (prefixed sysnum_ per smode_entry convention)
// ---------------------------------------------------------------

pub const SYSNUM_GETCWD: u64 = 17;
pub const SYSNUM_FCNTL: u64 = 25;
pub const SYSNUM_IOCTL: u64 = 29;
pub const SYSNUM_STATFS: u64 = 43;
pub const SYSNUM_FCHOWN: u64 = 55;
pub const SYSNUM_VHANGUP: u64 = 58;
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
//  Errno
// ---------------------------------------------------------------

pub const ENOSYS: u64 = !0u64;
pub const EINVAL: u64 = (!0u64) - 21;

// ---------------------------------------------------------------
//  Struct definitions — repr(C) to match Linux rv64 ABI
// ---------------------------------------------------------------

#[repr(C)]
pub struct Timeval {
    pub tv_sec: i64,
    pub tv_usec: i64,
}

#[repr(C)]
pub struct Timespec {
    pub tv_sec: i64,
    pub tv_nsec: i64,
}

#[repr(C)]
pub struct Utsname {
    pub sysname: [u8; 65],
    pub nodename: [u8; 65],
    pub release: [u8; 65],
    pub version: [u8; 65],
    pub machine: [u8; 65],
    pub domainname: [u8; 65],
}

#[repr(C)]
pub struct Sysinfo {
    pub uptime: i64,
    pub loads: [u64; 3],
    pub totalram: u64,
    pub freeram: u64,
    pub sharedram: u64,
    pub bufferram: u64,
    pub totalswap: u64,
    pub freeswap: u64,
    pub procs: u16,
    pub totalhigh: u64,
    pub freehigh: u64,
    pub mem_unit: u32,
}

#[repr(C)]
pub struct Rusage {
    pub ru_utime: Timeval,
    pub ru_stime: Timeval,
    pub ru_maxrss: i64,
    pub ru_ixrss: i64,
    pub ru_idrss: i64,
    pub ru_isrss: i64,
    pub ru_minflt: i64,
    pub ru_majflt: i64,
    pub ru_nswap: i64,
    pub ru_inblock: i64,
    pub ru_oublock: i64,
    pub ru_msgsnd: i64,
    pub ru_msgrcv: i64,
    pub ru_nsignals: i64,
    pub ru_nvcsw: i64,
    pub ru_nivcsw: i64,
}

#[repr(C)]
pub struct Tms {
    pub tms_utime: i64,
    pub tms_stime: i64,
    pub tms_cutime: i64,
    pub tms_cstime: i64,
}

#[repr(C)]
pub struct Rlimit {
    pub rlim_cur: u64,
    pub rlim_max: u64,
}

// ---------------------------------------------------------------
//  辅助函数
// ---------------------------------------------------------------

fn read_time() -> u64 {
    let t: u64;
    unsafe { core::arch::asm!("csrr {0}, 0xC01", out(reg) t) };
    t
}

#[inline]
fn is_stdio(fd: u64) -> bool { fd <= 2 }

/// 将固定字符串复制到 utsname 字段，确保 null-terminated。
unsafe fn fill_uts_field(src: &[u8], dst: &mut [u8; 65]) {
    let n = src.len().min(64);
    dst[..n].copy_from_slice(&src[..n]);
    dst[n] = 0;
}

// ---------------------------------------------------------------
//  write handler (64) — 对应 smode_entry 的 write_stdio_handler
// ---------------------------------------------------------------

fn write_handler(fd: u64, buf: *const u8, len: u64) -> u64 {
    // 卫语句：仅 stdout/stderr
    if fd != 1 && fd != 2 {
        return ENOSYS;
    }

    let mut written = 0;
    while written < len {
        let b = unsafe { buf.add(written as usize).read_volatile() };
        uart::uart_putc(b);
        written += 1;
    }
    len
}

// ---------------------------------------------------------------
//  read handler (63)
// ---------------------------------------------------------------

/// 从 UART 阻塞读取字节到 buf，遇换行或满 len 返回。
unsafe fn read_stdin(buf: *mut u8, len: u64) -> u64 {
    let mut count = 0;
    while count < len {
        match uart::uart_getc() {
            Some(b) => {
                unsafe { buf.add(count as usize).write_volatile(b) };
                count += 1;
                if b == b'\n' || b == b'\r' {
                    break;
                }
            }
            None => continue,
        }
    }
    count
}

fn read_handler(fd: u64, buf: *mut u8, len: u64) -> u64 {
    if !is_stdio(fd) {
        return ENOSYS;
    }
    // 仅 stdin 可读
    if fd != 0 {
        return ENOSYS;
    }
    unsafe { read_stdin(buf, len) }
}

// ---------------------------------------------------------------
//  uname handler (160)
// ---------------------------------------------------------------

fn uname_handler(buf: *mut Utsname) -> u64 {
    if buf.is_null() {
        return EINVAL;
    }

    let info = unsafe { &mut *buf };
    unsafe {
        fill_uts_field(b"pyremu-enclave", &mut info.sysname);
        fill_uts_field(b"pyremu", &mut info.nodename);
        fill_uts_field(b"0.1.0", &mut info.release);
        fill_uts_field(b"rust_smode_entry", &mut info.version);
        fill_uts_field(b"riscv64", &mut info.machine);
        fill_uts_field(b"(none)", &mut info.domainname);
    }
    0
}

// ---------------------------------------------------------------
//  gettimeofday handler (169)
// ---------------------------------------------------------------

fn gettimeofday_handler(tv: *mut Timeval) -> u64 {
    if tv.is_null() {
        return EINVAL;
    }

    let t = read_time();
    unsafe {
        (*tv).tv_sec = (t / TIMER_FREQ) as i64;
        (*tv).tv_usec = ((t % TIMER_FREQ) / (TIMER_FREQ / 1_000_000)) as i64;
    }
    0
}

// ---------------------------------------------------------------
//  clock_gettime handler (113)
// ---------------------------------------------------------------

fn clock_gettime_handler(clock_id: u64, tp: *mut Timespec) -> u64 {
    if tp.is_null() {
        return EINVAL;
    }

    // CLOCK_REALTIME=0, CLOCK_MONOTONIC=1
    if clock_id > 1 {
        return EINVAL;
    }

    let t = read_time();
    let nsec_per_tick = 1_000_000_000 / TIMER_FREQ;
    unsafe {
        (*tp).tv_sec = (t / TIMER_FREQ) as i64;
        (*tp).tv_nsec = ((t % TIMER_FREQ) * nsec_per_tick) as i64;
    }
    0
}

// ---------------------------------------------------------------
//  sysinfo handler (179)
// ---------------------------------------------------------------

fn sysinfo_handler(info: *mut Sysinfo) -> u64 {
    if info.is_null() {
        return EINVAL;
    }

    unsafe {
        (*info).uptime = (read_time() / TIMER_FREQ) as i64;
        (*info).loads = [0, 0, 0];
        (*info).totalram = 128 * 1024 * 1024;
        (*info).freeram = 64 * 1024 * 1024;
        (*info).sharedram = 0;
        (*info).bufferram = 0;
        (*info).totalswap = 0;
        (*info).freeswap = 0;
        (*info).procs = 1;
        (*info).totalhigh = 0;
        (*info).freehigh = 0;
        (*info).mem_unit = 1;
    }
    0
}

// ---------------------------------------------------------------
//  getrusage handler (165)
// ---------------------------------------------------------------

fn getrusage_handler(usage: *mut Rusage) -> u64 {
    if !usage.is_null() {
        unsafe { core::ptr::write_bytes(usage as *mut u8, 0, core::mem::size_of::<Rusage>()) };
    }
    0
}

// ---------------------------------------------------------------
//  getcpu handler (168)
// ---------------------------------------------------------------

fn getcpu_handler(cpu: *mut u32, node: *mut u32) -> u64 {
    if !cpu.is_null() {
        unsafe { *cpu = 0 };
    }
    if !node.is_null() {
        unsafe { *node = 0 };
    }
    0
}

// ---------------------------------------------------------------
//  times handler (153)
// ---------------------------------------------------------------

fn times_handler(buf: *mut Tms) -> u64 {
    if buf.is_null() {
        return ENOSYS;
    }
    unsafe { core::ptr::write_bytes(buf as *mut u8, 0, core::mem::size_of::<Tms>()) };
    0
}

// ---------------------------------------------------------------
//  prlimit64 handler (261)
// ---------------------------------------------------------------

const RLIM_INFINITY: u64 = 0xFFFF_FFFF;
const RLIMIT_NOFILE: u64 = 7;

fn prlimit64_handler(
    resource: u64,
    new_limit: *const Rlimit,
    old_limit: *mut Rlimit,
) -> u64 {
    // 设新值：仅接受 RLIMIT_NOFILE
    if !new_limit.is_null() {
        if resource == RLIMIT_NOFILE {
            return 0;
        }
        return EINVAL;
    }

    // 查旧值：返回 INFINITY
    if !old_limit.is_null() {
        unsafe {
            (*old_limit).rlim_cur = RLIM_INFINITY;
            (*old_limit).rlim_max = RLIM_INFINITY;
        }
    }
    0
}

// ---------------------------------------------------------------
//  入口：syscall_handler  (per smode_entry/trap_handler.c)
// ---------------------------------------------------------------

/// 系统调用分发。返回的值写入 gprs.a0。
pub fn syscall_handler(gprs: &TrapGprs) -> u64 {
    let num = gprs.a7();
    let a0 = gprs.a0();
    let a1 = gprs.a1();
    let a2 = gprs.a2();
    // let a3 = gprs.a3();
    // let a4 = gprs.a4();
    // let a5 = gprs.a5();

    match num {
        // ---- I/O ----
        SYSNUM_WRITE => write_handler(a0, a1 as *const u8, a2),
        SYSNUM_READ => read_handler(a0, a1 as *mut u8, a2),

        // ---- 进程控制 ----
        SYSNUM_EXIT | SYSNUM_EXIT_GROUP => call::enclave_call_exit(a0),
        SYSNUM_SCHED_YIELD => 0,
        SYSNUM_NANOSLEEP => 0,

        // ---- 内存 ----
        SYSNUM_BRK => memory::sys_brk_handler(a0),

        // ---- 标识 ----
        SYSNUM_UNAME => uname_handler(a0 as *mut Utsname),
        SYSNUM_GETPID => 1,
        SYSNUM_GETPPID => 1,
        SYSNUM_GETUID => 0,
        SYSNUM_GETEUID => 0,
        SYSNUM_GETGID => 0,
        SYSNUM_GETEGID => 0,

        // ---- 时间 ----
        SYSNUM_GETTIMEOFDAY => gettimeofday_handler(a0 as *mut Timeval),
        SYSNUM_CLOCK_GETTIME => clock_gettime_handler(a0, a1 as *mut Timespec),
        SYSNUM_TIMES => times_handler(a0 as *mut Tms),

        // ---- 资源 ----
        SYSNUM_GETRUSAGE => getrusage_handler(a1 as *mut Rusage),
        SYSNUM_GETCPU => getcpu_handler(a0 as *mut u32, a1 as *mut u32),
        SYSNUM_PRLIMIT64 => {
            prlimit64_handler(a0, a1 as *const Rlimit, a2 as *mut Rlimit)
        }
        SYSNUM_SYSINFO => sysinfo_handler(a0 as *mut Sysinfo),

        // ---- 跳过（无副作用）----
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
        | SYSNUM_SET_TID_ADDRESS => 0,

        // ---- 默认：ENOSYS ----
        _ => {
            println!("syscall: unknown {}\n", num);
            ENOSYS
        }
    }
}
