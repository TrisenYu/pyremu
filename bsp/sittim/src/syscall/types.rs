//! 匹配 Linux rv64 ABI的public repr(C) 结构体。
#![allow(dead_code)]

// ---------------------------------------------------------------
//  时间类型
// ---------------------------------------------------------------

#[repr(C)]
pub struct Timeval {
    pub tv_sec: i64,
    pub tv_usec: i64,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Timespec {
    pub tv_sec: i64,
    pub tv_nsec: i64,
}

// ---------------------------------------------------------------
//  系统信息
// ---------------------------------------------------------------

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

// ---------------------------------------------------------------
//  资源
// ---------------------------------------------------------------

/// A system call that returns
/// Resource Usage measures for
/// [a process] or [its children/threads]
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

/// stores the current process times
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
//  套接字地址
// ---------------------------------------------------------------

/// IPv4 套接字地址 (musl riscv64 的 struct sockaddr_in, 16 字节)。
/// sin_family 按本机字节序, sin_port 与 sin_addr 按网络字节序。
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SockaddrIn {
    pub sin_family: u16,
    pub sin_port: u16,
    pub sin_addr: [u8; 4],
    pub sin_zero: [u8; 8],
}

// ---------------------------------------------------------------
//  报文首部
// ---------------------------------------------------------------

/// 套接字报文的收发首部 (musl riscv64 的 struct msghdr, 56 字节)。
/// msg_controllen 与 msg_flags 之间有一个 int 宽度的填充字段, 使 msg_flags 落在
/// 48 字节处。
#[repr(C)]
#[derive(Clone, Copy)]
pub struct MsgHdr {
    pub msg_name: u64,
    pub msg_namelen: u32,
    pub msg_iov: u64,
    pub msg_iovlen: u32,
    pub msg_control: u64,
    pub msg_controllen: u32,
    pub pad_controllen: u32,
    pub msg_flags: i32,
}
