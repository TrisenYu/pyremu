//! 系统调用的错误码。
//!
//! 本文件单独成篇而不与系统调用分发同处, 是为了让按模块划分的主机侧测试入口能够只取
//! 用错误码一项, 而不必连同分发所需的平台符号一并包含。

// ---------------------------------------------------------------
//  Errno
// ---------------------------------------------------------------

/// 由 errno 编号取系统调用返回值。取值为该编号的相反数: musl 的 syscall 包装以
/// errno = -ret 还原, 因此返回 !0u64 会被解释为 errno 1 (EPERM) 而不是 ENOSYS,
/// 令调用方的 ENOSYS 判据失效。
pub const fn errno(number: u32) -> u64 {
	!0u64 - ((number as u64) - 1)
}

pub const ENOENT: u64 = errno(2);
pub const ESRCH: u64 = errno(3);
pub const EBADF: u64 = errno(9);
pub const ECHILD: u64 = errno(10);
pub const EAGAIN: u64 = errno(11);
pub const ENOMEM: u64 = errno(12);
pub const EFAULT: u64 = errno(14);
pub const EBUSY: u64 = errno(16);
pub const EEXIST: u64 = errno(17);
pub const ENODEV: u64 = errno(19);
pub const ENOTDIR: u64 = errno(20);
pub const EISDIR: u64 = errno(21);
pub const EINVAL: u64 = errno(22);
pub const EMFILE: u64 = errno(24);
pub const ENOSPC: u64 = errno(28);
pub const ESPIPE: u64 = errno(29);
pub const ERANGE: u64 = errno(34);
pub const ENAMETOOLONG: u64 = errno(36);
pub const ENOSYS: u64 = errno(38);
pub const ENOTEMPTY: u64 = errno(39);
pub const ENOTSOCK: u64 = errno(88);
pub const EDESTADDRREQ: u64 = errno(89);
pub const EMSGSIZE: u64 = errno(90);
pub const ENOPROTOOPT: u64 = errno(92);
pub const EPROTONOSUPPORT: u64 = errno(93);
pub const EOPNOTSUPP: u64 = errno(95);
pub const EAFNOSUPPORT: u64 = errno(97);
pub const EADDRINUSE: u64 = errno(98);
pub const EADDRNOTAVAIL: u64 = errno(99);
pub const ENETUNREACH: u64 = errno(101);
pub const ENOTCONN: u64 = errno(107);

#[cfg(test)]
mod tests {
	use super::*;

	/// 返回值是该编号的相反数: musl 以 errno = -ret 还原, 故 1 号错误码不能取 !0u64。
	#[test]
	fn test_errno_negates_the_number() {
		assert_eq!(errno(1), !0u64);
		assert_eq!(errno(2), !0u64 - 1);
		assert_eq!(errno(38), !0u64 - 37);
	}

	/// 各常量与 Linux riscv64 的编号一致。
	#[test]
	fn test_errno_numbers_match_linux() {
		assert_eq!(ENOENT, errno(2));
		assert_eq!(EBADF, errno(9));
		assert_eq!(ENOMEM, errno(12));
		assert_eq!(EFAULT, errno(14));
		assert_eq!(EBUSY, errno(16));
		assert_eq!(EEXIST, errno(17));
		assert_eq!(ENOTDIR, errno(20));
		assert_eq!(EISDIR, errno(21));
		assert_eq!(EINVAL, errno(22));
		assert_eq!(EMFILE, errno(24));
		assert_eq!(ENOSPC, errno(28));
		assert_eq!(ESPIPE, errno(29));
		assert_eq!(ERANGE, errno(34));
		assert_eq!(ENAMETOOLONG, errno(36));
		assert_eq!(ENOSYS, errno(38));
		assert_eq!(ENOTEMPTY, errno(39));
		assert_eq!(EOPNOTSUPP, errno(95));
		assert_eq!(ENOTCONN, errno(107));
	}

	/// 同一编号的两个错误码不得取到同一个值。
	#[test]
	fn test_errno_values_are_distinct() {
		let all = [
			ENOENT,
			ESRCH,
			EBADF,
			ECHILD,
			EAGAIN,
			ENOMEM,
			EFAULT,
			EBUSY,
			EEXIST,
			ENODEV,
			ENOTDIR,
			EISDIR,
			EINVAL,
			EMFILE,
			ENOSPC,
			ESPIPE,
			ERANGE,
			ENAMETOOLONG,
			ENOSYS,
			ENOTEMPTY,
			ENOTSOCK,
			EDESTADDRREQ,
			EMSGSIZE,
			ENOPROTOOPT,
			EPROTONOSUPPORT,
			EOPNOTSUPP,
			EAFNOSUPPORT,
			EADDRINUSE,
			EADDRNOTAVAIL,
			ENETUNREACH,
			ENOTCONN,
		];
		for i in 0..all.len() {
			for j in i + 1..all.len() {
				assert_ne!(all[i], all[j], "{i} and {j} should not have the same index");
			}
		}
	}
}
