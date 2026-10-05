//! memcpy / memset 宽字优化实现。紧跟 custom-opensbi mem_man.c:544-647 的
//! `calc_mem_leap` 模式：优先 8 字节搬移，余数按 {4, 2, 1} 字节递减处理。
//! ref-emod string.c 的 memcpy 仍是逐字节循环，此处一并修正。

/// MemLeapCnt 辅助结构 — 将长度分解为 8 字节块数 + 余数 bitmask。
struct MemLeapCnt {
	times8: u64,
	mod8: u8,
}

#[inline]
fn calc_mem_leap(x: u64) -> MemLeapCnt {
	MemLeapCnt {
		times8: x >> 3,
		mod8: (x & 0x7) as u8,
	}
}

/// 等效于 C 标准库 `memset(dst, byte, size)`。
/// 将 byte 广播为 64 位常量后按 8/4/2/1 字节写入。
#[allow(unused)]
pub unsafe fn memset(dst: *mut u8, byte: u8, size: u64) {
	let fill64 = (byte as u64).wrapping_mul(0x0101_0101_0101_0101);
	let cnt = calc_mem_leap(size);
	let mut d = dst;

	// 8-byte chunks
	for _ in 0..cnt.times8 {
		unsafe {
			(d as *mut u64).write_unaligned(fill64);
			d = d.add(8);
		}
	}

	// remainder: 4 / 2 / 1
	if cnt.mod8 & 0x4 != 0 {
		unsafe {
			(d as *mut u32).write_unaligned(fill64 as u32);
			d = d.add(4);
		}
	}
	if cnt.mod8 & 0x2 != 0 {
		unsafe {
			(d as *mut u16).write_unaligned(fill64 as u16);
			d = d.add(2);
		}
	}
	if cnt.mod8 & 0x1 != 0 {
		unsafe {
			d.write(byte);
		}
	}
}

/// 等效于 C 标准库 `memcpy(dst, src, size)`。
/// 优先 8 字节搬移，余数按 {4, 2, 1} 字节递减处理。
#[allow(unused)]
pub unsafe fn memcpy(dst: *mut u8, src: *const u8, size: u64) {
	let cnt = calc_mem_leap(size);
	let mut d = dst;
	let mut s = src;

	// 8-byte chunks
	for _ in 0..cnt.times8 {
		unsafe {
			(d as *mut u64).write_unaligned((s as *const u64).read_unaligned());
			d = d.add(8);
			s = s.add(8);
		}
	}

	// remainder: 4 / 2 / 1
	if cnt.mod8 & 0x4 != 0 {
		unsafe {
			(d as *mut u32).write_unaligned((s as *const u32).read_unaligned());
			d = d.add(4);
			s = s.add(4);
		}
	}
	if cnt.mod8 & 0x2 != 0 {
		unsafe {
			(d as *mut u16).write_unaligned((s as *const u16).read_unaligned());
			d = d.add(2);
			s = s.add(2);
		}
	}
	if cnt.mod8 & 0x1 != 0 {
		unsafe {
			d.write(s.read());
		}
	}
}

#[cfg(test)]
mod tests {
	use super::{memcpy, memset};

	/// 缓冲区总长, 以及其中可供被测函数写入的区间 [REGION_START, REGION_END)。
	const BUF_LEN: usize = 80;
	const REGION_START: usize = 8;
	const REGION_LEN: usize = 64;
	const REGION_END: usize = REGION_START + REGION_LEN;
	/// 缓冲区的初值; 写入越出指定区间时, 区间之外的字节不再是该值。
	const INIT: u8 = 0xA5;

	/// 用 REGION_START 与 REGION_LEN 划出写入区间的缓冲区。
	struct Buf {
		raw: [u8; BUF_LEN],
	}

	impl Buf {
		fn new() -> Self {
			Self {
				raw: [INIT; BUF_LEN],
			}
		}

		/// 区间 [REGION_START, REGION_END) 的起始地址。
		fn ptr(&mut self) -> *mut u8 {
			unsafe { self.raw.as_mut_ptr().add(REGION_START) }
		}

		fn region(&self) -> &[u8] {
			&self.raw[REGION_START..REGION_END]
		}

		/// 指定区间之外的字节是否仍为初值。
		fn is_outside_intact(&self) -> bool {
			self.raw[..REGION_START].iter().all(|&b| b == INIT)
				&& self.raw[REGION_END..].iter().all(|&b| b == INIT)
		}
	}

	/// 长度取 0 时两个函数都不写入任何字节。
	#[test]
	fn test_zero_length_writes_nothing() {
		let mut dst = Buf::new();
		let src = [0x11u8; REGION_LEN];
		unsafe {
			memset(dst.ptr(), 0xAA, 0);
			memcpy(dst.ptr(), src.as_ptr(), 0);
		}
		assert!(dst.region().iter().all(|&b| b == INIT));
		assert!(dst.is_outside_intact());
	}

	/// 长度取 64 以内的每一个值时, 指定长度内全为填充字节, 其余字节保持初值。
	#[test]
	fn test_memset_covers_the_requested_length() {
		for size in 0..=REGION_LEN as u64 {
			let mut dst = Buf::new();
			unsafe { memset(dst.ptr(), 0xAA, size) };
			let n = size as usize;
			let region = dst.region();
			assert!(region[..n].iter().all(|&b| b == 0xAA), "size={size}");
			assert!(region[n..].iter().all(|&b| b == INIT), "size={size}");
			assert!(dst.is_outside_intact(), "size={size}");
		}
	}

	/// 按 8 字节写入的路径与按 4 字节余数写入的路径, 写入的每个字节都等于填充字节。
	#[test]
	fn test_memset_repeats_the_fill_byte_across_the_word() {
		for size in [4u64, 5, 6, 7, 8, 9, 15, 16, 17, 23, 31, 33] {
			let mut dst = Buf::new();
			unsafe { memset(dst.ptr(), 0xAA, size) };
			let n = size as usize;
			let filled = &dst.region()[..n];
			assert!(
				filled.iter().all(|&b| b == 0xAA),
				"size={size} 写入 {filled:02x?}"
			);
		}
	}

	/// 目的地址不对齐时, 指定长度内全为填充字节, 其余字节保持初值。
	#[test]
	fn test_memset_unaligned_destination() {
		for off in 0..REGION_START {
			for size in [1u64, 7, 8, 13] {
				let mut dst = Buf::new();
				let p = unsafe { dst.ptr().add(off) };
				unsafe { memset(p, 0x5A, size) };
				let n = size as usize;
				let region = dst.region();
				assert!(
					region[off..off + n].iter().all(|&b| b == 0x5A),
					"off={off} size={size}"
				);
				assert!(region[..off].iter().all(|&b| b == INIT), "off={off}");
				assert!(
					region[off + n..].iter().all(|&b| b == INIT),
					"off={off} size={size}"
				);
			}
		}
	}

	/// 长度取 64 以内的每一个值时, 复制结果与逐字节复制的写法一致, 且不越界写入。
	#[test]
	fn test_memcpy_matches_bytewise_reference() {
		let mut src = [0u8; REGION_LEN];
		for (i, b) in src.iter_mut().enumerate() {
			*b = (i as u8).wrapping_mul(37).wrapping_add(11);
		}
		for size in 0..=REGION_LEN as u64 {
			let mut dst = Buf::new();
			unsafe { memcpy(dst.ptr(), src.as_ptr(), size) };
			let n = size as usize;
			assert_eq!(&dst.region()[..n], &src[..n], "size={size}");
			assert!(dst.region()[n..].iter().all(|&b| b == INIT), "size={size}");
			assert!(dst.is_outside_intact(), "size={size}");
		}
	}

	/// 源地址与目的地址各自不对齐时, 复制结果与两者均对齐时一致。
	#[test]
	fn test_memcpy_unaligned_ends() {
		let mut src = [0u8; REGION_LEN];
		for (i, b) in src.iter_mut().enumerate() {
			*b = (i as u8) ^ 0x5A;
		}
		for src_off in 0..REGION_START {
			for dst_off in 0..REGION_START {
				for size in [1u64, 3, 4, 5, 8, 9, 16, 17, 23] {
					let mut dst = Buf::new();
					let n = size as usize;
					let s = unsafe { src.as_ptr().add(src_off) };
					let d = unsafe { dst.ptr().add(dst_off) };
					unsafe { memcpy(d, s, size) };
					assert_eq!(
						&dst.region()[dst_off..dst_off + n],
						&src[src_off..src_off + n],
						"src_off={src_off} dst_off={dst_off} size={size}"
					);
				}
			}
		}
	}
}
