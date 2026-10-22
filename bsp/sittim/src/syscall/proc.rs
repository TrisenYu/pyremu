//! 进程/标识/时间/资源类系统调用。
//!
//! 已实现: uname(160), gettimeofday(169), clock_gettime(113), sysinfo(179),
//!         getrusage(165), getcpu(168), times(153), prlimit64(261),
//!         getpid(172), getppid(173), getuid(174), geteuid(175),
//!         getgid(176), getegid(177)。

use crate::constants::{TIMER_FREQ, TIMER_FREQ_NONZERO};
use crate::prim::arith;

use super::types::{Rlimit, Rusage, Sysinfo, Timespec, Timeval, Tms, Utsname};
use super::{EINVAL, ENOSYS};

// ---------------------------------------------------------------
//  辅助函数
// ---------------------------------------------------------------

fn read_time() -> u64 {
	let t: u64;
	unsafe { core::arch::asm!("csrr {0}, 0xC01", out(reg) t) };
	t
}

/// 将固定字符串复制到 utsname 字段，确保 null-terminated。
unsafe fn fill_uts_field(src: &[u8], dst: &mut [u8; 65]) {
	let n = src.len().min(64);
	dst[..n].copy_from_slice(&src[..n]);
	dst[n] = 0;
}

// ---------------------------------------------------------------
//  uname handler (160)
// ---------------------------------------------------------------

pub fn uname_handler(buf: *mut Utsname) -> u64 {
	if buf.is_null() {
		return EINVAL;
	}

	let info = unsafe { &mut *buf };
	unsafe {
		fill_uts_field(b"sittim-enclave", &mut info.sysname);
		fill_uts_field(b"sittim", &mut info.nodename);
		// release 栏直接取本包的版本号, 不另抄一份字面量以免两处漂移。
		fill_uts_field(env!("CARGO_PKG_VERSION").as_bytes(), &mut info.release);
		fill_uts_field(b"sittim", &mut info.version);
		fill_uts_field(b"riscv64", &mut info.machine);
		fill_uts_field(b"(none)", &mut info.domainname);
	}
	0
}

// ---------------------------------------------------------------
//  gettimeofday handler (169)
// ---------------------------------------------------------------

pub fn gettimeofday_handler(tv: *mut Timeval) -> u64 {
	if tv.is_null() {
		return EINVAL;
	}

	let (sec, usec) = arith::ticks_to_sec_usec(read_time(), TIMER_FREQ_NONZERO);
	unsafe {
		(*tv).tv_sec = sec as i64;
		(*tv).tv_usec = usec as i64;
	}
	0
}

// ---------------------------------------------------------------
//  clock_gettime handler (113)
// ---------------------------------------------------------------

pub fn clock_gettime_handler(clock_id: u64, tp: *mut Timespec) -> u64 {
	if tp.is_null() {
		return EINVAL;
	}

	if clock_id > 1 {
		return EINVAL;
	}

	let (sec, nsec) = arith::ticks_to_sec_nsec(read_time(), TIMER_FREQ_NONZERO);
	unsafe {
		(*tp).tv_sec = sec as i64;
		(*tp).tv_nsec = nsec as i64;
	}
	0
}

// ---------------------------------------------------------------
//  sysinfo handler (179)
// ---------------------------------------------------------------

pub fn sysinfo_handler(info: *mut Sysinfo) -> u64 {
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

pub fn getrusage_handler(usage: *mut Rusage) -> u64 {
	if !usage.is_null() {
		unsafe { core::ptr::write_bytes(usage as *mut u8, 0, core::mem::size_of::<Rusage>()) };
	}
	0
}

// ---------------------------------------------------------------
//  getcpu handler (168)
// ---------------------------------------------------------------

pub fn getcpu_handler(cpu: *mut u32, node: *mut u32) -> u64 {
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

pub fn times_handler(buf: *mut Tms) -> u64 {
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

pub fn prlimit64_handler(resource: u64, new_limit: *const Rlimit, old_limit: *mut Rlimit) -> u64 {
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
