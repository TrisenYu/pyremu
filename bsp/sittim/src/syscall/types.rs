//! 匹配 Linux rv64 ABI 的公开 repr(C) 结构体。
//!
//! 各结构体的字节数与取用到的字段偏移由紧随其后的编译期断言给出, 取值取自
//! `vendor/musl/include` 与 `vendor/musl/arch/riscv64` 下的头文件。声明与头文件不符时
//! 编译失败, 不在运行期表现为读到错位的字段。
#![allow(dead_code)]

// ---------------------------------------------------------------
//  时间类型
// ---------------------------------------------------------------

#[repr(C)]
pub struct Timeval {
    pub tv_sec: i64,
    pub tv_usec: i64,
}

const _: () = assert!(core::mem::size_of::<Timeval>() == 16);
const _: () = assert!(core::mem::offset_of!(Timeval, tv_usec) == 8);

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Timespec {
    pub tv_sec: i64,
    pub tv_nsec: i64,
}

const _: () = assert!(core::mem::size_of::<Timespec>() == 16);
const _: () = assert!(core::mem::offset_of!(Timespec, tv_nsec) == 8);

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

const _: () = assert!(core::mem::size_of::<Utsname>() == 390);
const _: () = assert!(core::mem::offset_of!(Utsname, domainname) == 325);

/// 内核 struct sysinfo 的布局, 即 sysinfo 调用写入的字节数: 112 字节。
///
/// procs 与 totalhigh 之间另有 4 字节的填充。musl 的用户态结构体在该 112 字节之后还有
/// 256 字节的保留区 (`char __reserved[256]`), 共 368 字节, 内核与本层都不写入该区。
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

const _: () = assert!(core::mem::size_of::<Sysinfo>() == 112);
const _: () = assert!(core::mem::offset_of!(Sysinfo, procs) == 80);
const _: () = assert!(core::mem::offset_of!(Sysinfo, totalhigh) == 88);
const _: () = assert!(core::mem::offset_of!(Sysinfo, mem_unit) == 104);

// ---------------------------------------------------------------
//  资源
// ---------------------------------------------------------------

/// 内核 struct rusage 的布局, 即 getrusage 调用写入的字节数: 2 个 [Timeval] 与 14 个
/// i64, 合计 18 个 8 字节字。
///
/// musl 的用户态结构体在该 144 字节之后还有 16 个 long 的保留区
/// (`long __reserved[16]`), 共 272 字节, 内核与本层都不写入该区。
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

const _: () = assert!(core::mem::size_of::<Rusage>() == 144);
const _: () = assert!(core::mem::offset_of!(Rusage, ru_stime) == 16);
const _: () = assert!(core::mem::offset_of!(Rusage, ru_nivcsw) == 136);

/// 内核 struct tms 的布局, 即 times 调用写入的字节数: 4 个 i64。
#[repr(C)]
pub struct Tms {
    pub tms_utime: i64,
    pub tms_stime: i64,
    pub tms_cutime: i64,
    pub tms_cstime: i64,
}

const _: () = assert!(core::mem::size_of::<Tms>() == 32);

#[repr(C)]
pub struct Rlimit {
    pub rlim_cur: u64,
    pub rlim_max: u64,
}

const _: () = assert!(core::mem::size_of::<Rlimit>() == 16);

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

const _: () = assert!(core::mem::size_of::<SockaddrIn>() == 16);
const _: () = assert!(core::mem::offset_of!(SockaddrIn, sin_port) == 2);
const _: () = assert!(core::mem::offset_of!(SockaddrIn, sin_addr) == 4);

// ---------------------------------------------------------------
//  缓冲区描述与报文首部
// ---------------------------------------------------------------

/// 一个收发缓冲区 (musl riscv64 的 struct iovec, 16 字节)。
///
/// iovec 数组各项之间的间距等于本结构体的字节数。
#[repr(C)]
#[derive(Clone, Copy)]
pub struct IoVec {
    pub iov_base: u64,
    pub iov_len: u64,
}

const _: () = assert!(core::mem::size_of::<IoVec>() == 16);
const _: () = assert!(core::mem::offset_of!(IoVec, iov_len) == 8);

impl IoVec {
    /// 读出载荷地址空间中第 *index* 个 iovec 的缓冲区地址与长度。
    ///
    /// 项与项之间的间距与两个字段的位置都取自本结构体, 调用方不另立常量。
    pub fn at(io_vec_arr: u64, index: u64) -> (u64, u64) {
        let slot =
            io_vec_arr.wrapping_add(index.wrapping_mul(core::mem::size_of::<Self>() as u64));
        let iov = unsafe { core::ptr::read_unaligned(slot as *const Self) };
        (iov.iov_base, iov.iov_len)
    }
}

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

const _: () = assert!(core::mem::size_of::<MsgHdr>() == 56);
const _: () = assert!(core::mem::offset_of!(MsgHdr, msg_iovlen) == 24);
const _: () = assert!(core::mem::offset_of!(MsgHdr, msg_control) == 32);
const _: () = assert!(core::mem::offset_of!(MsgHdr, msg_controllen) == 40);
const _: () = assert!(core::mem::offset_of!(MsgHdr, msg_flags) == 48);

// ---------------------------------------------------------------
//  轮询
// ---------------------------------------------------------------

/// 一个被轮询的描述符 (musl riscv64 的 struct pollfd, 8 字节)。
///
/// 轮询数组各项之间的间距等于本结构体的字节数。
#[repr(C)]
#[derive(Clone, Copy)]
pub struct PollFd {
    pub fd: i32,
    pub events: u16,
    pub revents: u16,
}

const _: () = assert!(core::mem::size_of::<PollFd>() == 8);
const _: () = assert!(core::mem::offset_of!(PollFd, events) == 4);
const _: () = assert!(core::mem::offset_of!(PollFd, revents) == 6);

impl PollFd {
    /// 轮询数组中第 *index* 项在载荷地址空间中的地址。
    pub fn at(fds: u64, index: u64) -> *mut Self {
        fds.wrapping_add(index.wrapping_mul(core::mem::size_of::<Self>() as u64)) as *mut Self
    }
}

