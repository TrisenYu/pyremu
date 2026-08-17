//! Zero-allocation RAM direct read/write — bypasses Python `bytes()` slicing.
//!
//! Python owns the `bytearray`; Rust receives a raw pointer (via ctypes
//! from_buffer), reads/writes directly, and writes results to a pre-allocated
//! output buffer.  No allocation, no state held across calls.

// ============================================================
//  Helpers
// ============================================================

/// Compute offset into the bytearray given RAM and shadow ranges.
/// Returns `None` when the address falls outside all valid ranges.
/// Public alias for use by the speedup execution engine's inline memory access helpers.
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
	let end = pa + size as u64;
	// Primary RAM window
	if pa >= ram_base && end <= ram_base + ram_size {
		return Some(pa - ram_base);
	}
	// VMA shadow alias (firmware linked low, loaded high)
	if shadow_size > 0 {
		let sh_end = shadow_base + shadow_size;
		if pa >= shadow_base && end <= sh_end {
			return Some(pa - shadow_base);
		}
	}
	None
}

// ============================================================
//  Public API
// ============================================================

/// Read *size* bytes from physical address `pa` into `out_buf`.
///
/// `ram_ptr` points to the first byte of the Python `bytearray`.
/// `out_buf` must be at least `size` bytes (caller pre-allocates 8 bytes).
///
/// Returns 1 on success, 0 if `pa` is not in RAM (caller falls back to
/// device / L2-cache / error handling in Python).
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
	// Safety: `ram_ptr` points to a Python bytearray of `ram_size` bytes.
	// `off + size <= ram_size` is guaranteed by `ram_offset`.
	// `out_buf` is a pre-allocated Python bytearray(8).
	// `copy_nonoverlapping` handles unaligned pointers correctly.
	unsafe {
		core::ptr::copy_nonoverlapping(ram_ptr.add(off), out_buf, size as usize);
	}
	1
}

/// Write *size* bytes from `data_ptr` to physical address `pa`.
///
/// `ram_ptr` points to the first byte of the Python `bytearray`.
///
/// Returns 1 on success, 0 if `pa` is not in RAM.
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
//  Tests
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
		// Shadow: VMA 0x0..0x10 aliases RAM 0x80000000..0x80000010
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
		assert_eq!(ok, 0); // address beyond RAM
	}

	#[test]
	fn test_read_cross_boundary() {
		let ram: Vec<u8> = (0..16).collect();
		let mut out = [0u8; 8];
		// size=8 starting at offset 12 extends to offset 20 > 16 -> fail
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
