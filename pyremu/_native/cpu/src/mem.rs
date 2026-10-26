//! 直接读取与写入主存, 无需借助 Python 侧 bytes 的访存路径.
//!
//! Python 持有 `bytearray`; Rust 经 ctypes 的 from_buffer 取得指向首字节的裸指针
//! 后直接访问. 地址属于主存范围或 PIE 链接地址区间时完成拷贝并返回 1, 否则返回 0,
//! 由调用方转回 Python 处理. 跨链接库调用之间不保留状态也不分配内存.

// ============================================================
//  辅助函数
// ============================================================

/// 按主存范围与 PIE 链接地址区间计算物理地址在 bytearray 内的偏移.
/// 地址不属于任何一段范围时返回 `None`.
/// 该入口对外公开, 供加速执行引擎的内联访存辅助函数调用.
#[inline(always)]
pub fn ram_offset_inline(
	pa: u64,
	size: u32,
	ram_base: u64,
	ram_size: u64,
	shadow_base: u64,
	shadow_size: u64,
) -> Option<u64> {
	ram_offset(pa, size, ram_base, ram_size, shadow_base, shadow_size)
}

#[inline(always)]
fn ram_offset(
	pa: u64,
	size: u32,
	ram_base: u64,
	ram_size: u64,
	shadow_base: u64,
	shadow_size: u64,
) -> Option<u64> {
	// 调用方传入的 pa 可为任意 64 位值, 其中包含 Python 侧负数地址经 ctypes
	// 转换得到的 2^64-1 一类取值. 若加法按模 2^64 计算, 范围判定会误判为
	// 命中, 随后的拷贝越过 bytearray 造成越界访问.
	let end = pa.checked_add(size as u64)?;
	// 主存范围
	if pa >= ram_base && end <= ram_base + ram_size {
		return Some(pa - ram_base);
	}
	// PIE 链接地址区间: 固件按低地址链接并加载到高地址, 该区间内的地址换算到主存内的偏移.
	if shadow_size > 0 {
		let sh_end = shadow_base + shadow_size;
		if pa >= shadow_base && end <= sh_end {
			return Some(pa - shadow_base);
		}
	}
	None
}

// ============================================================
//  公开接口
// ============================================================

/// 从物理地址 `pa` 读取 `size` 个字节到 `out_buf`.
///
/// `ram_ptr` 指向 Python `bytearray` 的首字节.
/// `out_buf` 至少可容纳 `size` 个字节, 调用方预先分配 8 个字节.
///
/// 成功返回 1; 地址不属于主存范围或 PIE 链接地址区间时返回 0,
/// 调用方在 Python 侧转而访问设备, 查询 L2 缓存或按相应的故障实施处理.
#[no_mangle]
pub extern "C" fn bus_read_ram(
	ram_ptr: *const u8,
	ram_size: u64,
	ram_base: u64,
	shadow_base: u64,
	shadow_size: u64,
	pa: u64,
	size: u32,
	out_buf: *mut u8,
) -> u8 {
	let off = match ram_offset(pa, size, ram_base, ram_size, shadow_base, shadow_size) {
		Some(o) => o as usize,
		None => return 0,
	};
	// 调用方保证 `ram_ptr` 指向长度为 `ram_size` 的 Python bytearray,
	// `off + size <= ram_size` 由 `ram_offset` 保证;
	// `out_buf` 为预先分配的 Python bytearray(8);
	// `copy_nonoverlapping` 可处理非对齐指针.
	unsafe {
		core::ptr::copy_nonoverlapping(ram_ptr.add(off), out_buf, size as usize);
	}
	1
}

/// 把 `data_ptr` 处的 `size` 个字节写入物理地址 `pa`.
///
/// `ram_ptr` 指向 Python `bytearray` 的首字节.
///
/// 成功返回 1; 地址不属于主存范围或 PIE 链接地址区间时返回 0.
#[no_mangle]
pub extern "C" fn bus_write_ram(
	ram_ptr: *mut u8,
	ram_size: u64,
	ram_base: u64,
	shadow_base: u64,
	shadow_size: u64,
	pa: u64,
	data_ptr: *const u8,
	size: u32,
) -> u8 {
	let off = match ram_offset(pa, size, ram_base, ram_size, shadow_base, shadow_size) {
		Some(o) => o as usize,
		None => return 0,
	};
	unsafe {
		core::ptr::copy_nonoverlapping(data_ptr, ram_ptr.add(off), size as usize);
	}
	1
}

// ============================================================
//  测试
// ============================================================

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_read_ram_normal() {
		let ram: Vec<u8> = (0..16).collect(); // [0, 1, 2, ..., 15]
		let mut out = [0u8; 8];
		let ok = bus_read_ram(
			ram.as_ptr(),
			16,
			0x8000_0000,
			0,
			0,
			0x8000_0004,
			4,
			out.as_mut_ptr(),
		);
		assert_eq!(ok, 1);
		assert_eq!(&out[..4], &[4, 5, 6, 7]);
	}

	#[test]
	fn test_read_shadow() {
		let ram: Vec<u8> = (0..16).collect();
		let mut out = [0u8; 8];
		// PIE 链接地址区间: 链接地址 0x0 至 0x10 换算到主存 0x80000000 至 0x80000010
		let ok = bus_read_ram(
			ram.as_ptr(),
			16,
			0x8000_0000,
			0x0,
			16,
			0x0000_0008,
			4,
			out.as_mut_ptr(),
		);
		assert_eq!(ok, 1);
		assert_eq!(&out[..4], &[8, 9, 10, 11]);
	}

	#[test]
	fn test_read_oob() {
		let ram: Vec<u8> = vec![0; 16];
		let mut out = [0u8; 8];
		let ok = bus_read_ram(
			ram.as_ptr(),
			16,
			0x8000_0000,
			0,
			0,
			0x8000_0010,
			4,
			out.as_mut_ptr(),
		);
		assert_eq!(ok, 0); // 地址超出主存范围
	}

	#[test]
	fn test_read_cross_boundary() {
		let ram: Vec<u8> = (0..16).collect();
		let mut out = [0u8; 8];
		// 从偏移 12 起读取 8 个字节, 末端偏移 20 超过 16, 判定为失败
		let ok = bus_read_ram(
			ram.as_ptr(),
			16,
			0x8000_0000,
			0,
			0,
			0x8000_000C,
			8,
			out.as_mut_ptr(),
		);
		assert_eq!(ok, 0);
	}

	#[test]
	fn test_write_ram() {
		let mut ram: Vec<u8> = vec![0; 16];
		let data: [u8; 4] = [0xAA, 0xBB, 0xCC, 0xDD];
		let ok = bus_write_ram(
			ram.as_mut_ptr(),
			16,
			0x8000_0000,
			0,
			0,
			0x8000_0008,
			data.as_ptr(),
			4,
		);
		assert_eq!(ok, 1);
		assert_eq!(&ram[8..12], &[0xAA, 0xBB, 0xCC, 0xDD]);
	}

	#[test]
	fn test_write_oob() {
		let mut ram: Vec<u8> = vec![0; 16];
		let ok = bus_write_ram(
			ram.as_mut_ptr(),
			16,
			0x8000_0000,
			0,
			0,
			0x8000_0010,
			&0u8 as *const u8,
			4,
		);
		assert_eq!(ok, 0);
	}

	/// 地址与长度相加按模 2^64 计算时, 结果不得判为主存范围内的命中.
	///
	/// 触发值 `u64::MAX` 是 Python 侧负数地址经 ctypes 转换后的取值, 例如
	/// fp 减 16 得到 -1; 该值加 8 按模 2^64 计算得 7, 仍属于
	/// [ram_base, ram_base + ram_size) 之内, 若不做上界检查会算出
	/// 0xFFFF_FFFF_7FFF_FFFF 的偏移并越界拷贝.
	#[test]
	fn test_read_addr_wraparound_rejected() {
		let ram: Vec<u8> = vec![0; 16];
		let mut out = [0u8; 8];
		let ok = bus_read_ram(
			ram.as_ptr(),
			16,
			0x8000_0000,
			0,
			0,
			u64::MAX,
			8,
			out.as_mut_ptr(),
		);
		assert_eq!(ok, 0);

		let ok = bus_read_ram(
			ram.as_ptr(),
			16,
			0x8000_0000,
			0,
			0,
			u64::MAX - 3,
			8,
			out.as_mut_ptr(),
		);
		assert_eq!(ok, 0);
	}

	#[test]
	fn test_write_addr_wraparound_rejected() {
		let mut ram: Vec<u8> = vec![0; 16];
		let ok = bus_write_ram(
			ram.as_mut_ptr(),
			16,
			0x8000_0000,
			0,
			0,
			u64::MAX,
			&0u8 as *const u8,
			8,
		);
		assert_eq!(ok, 0);
	}

	#[test]
	fn test_read_sizes() {
		let ram: Vec<u8> = b"\x01\x02\x04\x08\x10\x20\x40\x80".to_vec();
		let mut out = [0u8; 8];

		let ok = bus_read_ram(ram.as_ptr(), 8, 0, 0, 0, 0, 1, out.as_mut_ptr());
		assert_eq!(ok, 1);
		assert_eq!(out[0], 0x01);

		let ok = bus_read_ram(ram.as_ptr(), 8, 0, 0, 0, 0, 2, out.as_mut_ptr());
		assert_eq!(ok, 1);
		assert_eq!(u16::from_le_bytes([out[0], out[1]]), 0x0201);

		let ok = bus_read_ram(ram.as_ptr(), 8, 0, 0, 0, 0, 4, out.as_mut_ptr());
		assert_eq!(ok, 1);
		assert_eq!(
			u32::from_le_bytes([out[0], out[1], out[2], out[3]]),
			0x08040201
		);

		let ok = bus_read_ram(ram.as_ptr(), 8, 0, 0, 0, 0, 8, out.as_mut_ptr());
		assert_eq!(ok, 1);
		assert_eq!(u64::from_le_bytes(out), 0x8040201008040201);
	}
}
