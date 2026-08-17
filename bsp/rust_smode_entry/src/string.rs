//! memcpy / memset 宽字优化实现。紧跟 custom-opensbi mem_man.c:544-647 的
//! `calc_mem_leap` 模式：优先 8 字节搬移，余数按 {4, 2, 1} 字节递减处理。
//! ref-emod string.c 的 memcpy 仍是逐字节循环，此处一并修正。

/// mem_leap 辅助结构 — 将长度分解为 8 字节块数 + 余数 bitmask。
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
#[inline]
#[allow(unused)]
pub unsafe fn memset(dst: *mut u8, byte: u8, size: u64) {
	let fill64 = {
		let b = byte as u64;
		b | (b << 8) | (b << 16) | (b << 32)
	};
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
#[inline]
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
