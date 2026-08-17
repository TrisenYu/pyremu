//! Sv39 page-table helpers — pure functions for VPN decomposition, PTE parsing,
//! and physical address assembly.
//!
//! Physical memory reads stay in Python (Bus pipeline); Rust only does the
//! bit-twiddling that was previously scattered across Python PTE/VPN helpers.

// ============================================================
//  PTE bit layouts
// ============================================================

const PTE_V: u64 = 1 << 0;
const PTE_R: u64 = 1 << 1;
const PTE_W: u64 = 1 << 2;
const PTE_X: u64 = 1 << 3;
const PTE_U: u64 = 1 << 4;

#[allow(dead_code)]
const PTE_PPN0: u64 = 0x3FF << 10; // bits 19:10
#[allow(dead_code)]
const PTE_PPN1: u64 = 0x1FF << 20; // bits 28:20
#[allow(dead_code)]
const PTE_PPN2: u64 = 0x1FFFFFFF << 29; // bits 53:29

const PAGE_SHIFT: u64 = 12;

// ============================================================
//  FFI structs
// ============================================================

/// Decomposed Sv39 virtual address.
#[repr(C)]
pub struct Sv39Vpn {
	/// VPN[2] — L1 (root) index, VA[38:30], 9 bits.
	pub vpn2: u64,
	/// VPN[1] — L2 index, VA[29:21], 9 bits.
	pub vpn1: u64,
	/// VPN[0] — L3 index, VA[20:12], 9 bits.
	pub vpn0: u64,
	/// Page offset, VA[11:0], 12 bits.
	pub offset: u64,
}

/// Parsed fields from a raw 64-bit PTE.
#[repr(C)]
pub struct PteFields {
	/// Physical page number (44 bits).
	pub ppn: u64,
	/// Permission flags: R|W|X|U (bits 1-4 of raw PTE).
	pub perm: u8,
	/// V (valid) flag.
	pub v: u8,
	/// True if this PTE is a leaf (V=1 and R or X set).
	pub is_leaf: u8,
	/// True if this PTE is a pointer to next level (V=1, R=W=X=0).
	pub is_ptr: u8,
}

// ============================================================
//  VPN decomposition
// ============================================================

/// Decompose a 39-bit virtual address into VPN[2:0] and page offset.
#[no_mangle]
pub extern "C" fn sv39_decompose_va(va: u64) -> Sv39Vpn {
	Sv39Vpn {
		vpn2: (va >> 30) & 0x1FF,
		vpn1: (va >> 21) & 0x1FF,
		vpn0: (va >> 12) & 0x1FF,
		offset: va & 0xFFF,
	}
}

// ============================================================
//  PTE parsing
// ============================================================

/// Parse a raw 64-bit PTE value into its component fields.
///
/// This replaces the Python `PTE` class's property-based field extraction,
/// which was called 3× per page-table walk.
#[no_mangle]
pub extern "C" fn pte_parse(raw: u64) -> PteFields {
	// PPN is stored contiguously at PTE[53:10] (RISC-V Privileged Spec §4.3).
	let ppn = (raw >> 10) & 0xFFF_FFFF_FFFFu64; // 44-bit mask
	let v = if raw & PTE_V != 0 { 1u8 } else { 0u8 };
	let r_or_x = raw & (PTE_R | PTE_X);
	PteFields {
		ppn,
		perm: (raw & (PTE_R | PTE_W | PTE_X | PTE_U)) as u8,
		v,
		is_leaf: if v != 0 && r_or_x != 0 { 1 } else { 0 },
		is_ptr: if v != 0 && r_or_x == 0 { 1 } else { 0 },
	}
}

// ============================================================
//  Physical address assembly
// ============================================================

/// Assemble a physical address from a leaf PTE's PPN and the original VA.
///
/// * `ppn` — the 44-bit physical page number extracted from the leaf PTE.
/// * `va`  — the original virtual address (needed for the page offset and,
///           in superpage case, VPN[0]).
/// * `level` — 0 = 4 KiB page (L3 leaf), 1 = 2 MiB superpage (L2 leaf).
///
/// For 4 KiB pages:  PA = (ppn << 12) | (va & 0xFFF)
/// For 2 MiB superpages:  PA = ((ppn[43:21] concatenated with va[20:0]) masked to 64 bits)
///   which is: the PPN's upper bits (≥bit 9) combined with va[20:12] as the low 9 bits,
///   then << 12 and OR with va[11:0].
#[no_mangle]
pub extern "C" fn pte_assemble_pa(ppn: u64, va: u64, level: u8) -> u64 {
	if level == 1 {
		// 2 MiB superpage: PPN[8:0] comes from VA[20:12] (vpn0).
		// PPN bits 9 and above are from the PTE.
		let vpn0 = (va >> 12) & 0x1FF;
		let ppn = (ppn & 0xFFFF_FFFF_FFFF_FE00) | vpn0;
		let offset = va & 0xFFF;
		((ppn << PAGE_SHIFT) | offset) & 0xFFFF_FFFF_FFFF_FFFF
	} else {
		// 4 KiB page: PA = PPN << 12 | offset[11:0]
		let offset = va & 0xFFF;
		((ppn << PAGE_SHIFT) | offset) & 0xFFFF_FFFF_FFFF_FFFF
	}
}

/// Get the page size for a given leaf level.
///
/// * `level` — 0 = 4 KiB, 1 = 2 MiB.
#[no_mangle]
pub extern "C" fn sv39_page_size(level: u8) -> u64 {
	if level == 1 {
		2 * 1024 * 1024 // 2 MiB
	} else {
		4096 // 4 KiB
	}
}

// ============================================================
//  Tests
// ============================================================

#[cfg(test)]
mod tests {
	use super::*;

	// -- VPN decomposition --

	#[test]
	fn test_decompose_va_zero() {
		let v = sv39_decompose_va(0);
		assert_eq!(v.vpn2, 0);
		assert_eq!(v.vpn1, 0);
		assert_eq!(v.vpn0, 0);
		assert_eq!(v.offset, 0);
	}

	#[test]
	fn test_decompose_va_mid() {
		// VA = (0x080 << 21) | (0x0AB << 12) | 0x123 = 0x100A_B123
		//   vpn2 = VA[38:30] = 0 (VA < 2^30)
		//   vpn1 = VA[29:21] = 0x100A_B123 >> 21 = 0x80 = 128
		//   vpn0 = VA[20:12] = 0x100A_B123 >> 12 & 0x1FF = 0xAB = 171
		//   offset = VA[11:0] = 0x123 = 291
		let v = sv39_decompose_va(0x100A_B123u64);
		assert_eq!(v.vpn2, 0);
		assert_eq!(v.vpn1, 128);
		assert_eq!(v.vpn0, 171);
		assert_eq!(v.offset, 0x123);
	}

	#[test]
	fn test_decompose_va_offset() {
		let v = sv39_decompose_va(0xABC);
		assert_eq!(v.vpn2, 0);
		assert_eq!(v.vpn1, 0);
		assert_eq!(v.vpn0, 0);
		assert_eq!(v.offset, 0xABC);
	}

	// -- PTE parsing --

	fn make_pte_raw(flags: u64, ppn: u64) -> u64 {
		flags | ((ppn & 0xFFF_FFFF_FFFFu64) << 10)
	}

	#[test]
	fn test_pte_parse_invalid() {
		let p = pte_parse(0);
		assert_eq!(p.v, 0);
		assert_eq!(p.is_leaf, 0);
		assert_eq!(p.is_ptr, 0);
		assert_eq!(p.ppn, 0);
	}

	#[test]
	fn test_pte_parse_leaf() {
		let raw = make_pte_raw(PTE_V | PTE_R | PTE_W | PTE_X | PTE_U, 0x80100);
		let p = pte_parse(raw);
		assert_eq!(p.v, 1);
		assert_eq!(p.is_leaf, 1);
		assert_eq!(p.is_ptr, 0);
		assert_eq!(p.ppn, 0x80100);
		assert_eq!(p.perm, (PTE_R | PTE_W | PTE_X | PTE_U) as u8);
	}

	#[test]
	fn test_pte_parse_ptr() {
		let raw = make_pte_raw(PTE_V, 0x80001); // V=1, R=W=X=0 -> pointer
		let p = pte_parse(raw);
		assert_eq!(p.v, 1);
		assert_eq!(p.is_leaf, 0);
		assert_eq!(p.is_ptr, 1);
		assert_eq!(p.ppn, 0x80001);
	}

	#[test]
	fn test_pte_ppn_boundary() {
		// PPN with bits set in all three sub-fields
		let ppn = 0x123456789AB; // 44 bits
		let raw = make_pte_raw(PTE_V | PTE_R, ppn);
		let p = pte_parse(raw);
		assert_eq!(p.ppn, ppn & 0xFFFFFFFFFFF);
	}

	// -- PA assembly --

	#[test]
	fn test_assemble_pa_4k() {
		// 4 KiB page: ppn=0x80100, va=0xABC -> pa = (0x80100 << 12) | 0xABC
		// 0x80100 << 12 = 0x8010_0000, + 0xABC = 0x8010_0ABC
		let pa = pte_assemble_pa(0x80100, 0xABC, 0);
		assert_eq!(pa, 0x8010_0ABCu64);
	}

	#[test]
	fn test_assemble_pa_2m_bit9_preserved() {
		// Regression: PPN bit 9 must NOT be cleared for 2 MiB superpages.
		// ppn=0x80200 (bit 9 = 1), va[20:12]=0x123
		// new_ppn = (ppn & ~0x1FF) | vpn0 = 0x80200 | 0x123 = 0x80323
		// pa = 0x80323 << 12 = 0x8032_3000
		let pa = pte_assemble_pa(0x80200, 0x123_000u64, 1);
		// correct: PPN bit 9 is preserved in PA bit 21
		assert_eq!(pa, 0x8032_3000u64);
		// the old buggy mask (~0x3FF, 10 bits) would incorrectly clear bit 9
		let old_buggy = ((0x80200 & !0x3FFu64) | 0x123) << 12;
		assert_ne!(pa, old_buggy, "old 10-bit mask would clear PPN bit 9");
	}

	#[test]
	fn test_page_size() {
		assert_eq!(sv39_page_size(0), 4096);
		assert_eq!(sv39_page_size(1), 2 * 1024 * 1024);
	}
}
