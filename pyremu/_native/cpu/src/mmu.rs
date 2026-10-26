//! Sv39 页表辅助计算函数, 负责虚拟页号分解、PTE 解析与物理地址转换。
//!
//! 本模块设计为纯计算逻辑。

// ============================================================
//  PTE 的位布局
// ============================================================

const PTE_V: u64 = 1 << 0;
const PTE_R: u64 = 1 << 1;
const PTE_W: u64 = 1 << 2;
const PTE_X: u64 = 1 << 3;
const PTE_U: u64 = 1 << 4;

#[allow(dead_code)]
const PTE_PPN0: u64 = 0x3FF << 10; // 位 [19:10]
#[allow(dead_code)]
const PTE_PPN1: u64 = 0x1FF << 20; // 位 [28:20]
#[allow(dead_code)]
const PTE_PPN2: u64 = 0x1FFFFFFF << 29; // 位 [57:29]

const PAGE_SHIFT: u64 = 12;

// ============================================================
//  FFI 结构体定义
// ============================================================

/// Sv39 定义的虚拟地址字段。
#[repr(C)]
pub struct Sv39Vpn {
	/// VPN[2]: L1 页表的索引, 即根页表的索引, 取 VA[38:30], 9 位。
	pub vpn2: u64,
	/// VPN[1]: L2 页表索引, 取 VA[29:21], 9 位。
	pub vpn1: u64,
	/// VPN[0]: L3 页表索引, 取 VA[20:12], 9 位。
	pub vpn0: u64,
	/// 页内偏移, 取 VA[11:0], 12 位。
	pub offset: u64,
}

/// 由 64 位原始 PTE 计算得到的字段。
#[repr(C)]
pub struct PteFields {
	/// 物理页号, 44 位。
	pub ppn: u64,
	/// 权限位: RWX 与 U, 对应原始 PTE 的位 1 至 4。
	pub perm: u8,
	/// V 位, 有效标志。
	pub v: u8,
	/// 通过 V=1 和设置 R 或 X 位判断是否是叶子页表项的标志。
	pub is_leaf: u8,
	/// 通过 V=1, R=W=X=0 判断是否是指向下一级页表的标志。
	pub is_ptr: u8,
}

// ============================================================
//  虚拟页号结构分解
// ============================================================

/// 将 39 位虚拟地址分解为 VPN[2:0] 与页内偏移。
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
//  PTE 解析
// ============================================================

/// 将 64 位原始 PTE 分解为各个字段。
#[no_mangle]
pub extern "C" fn pte_parse(raw: u64) -> PteFields {
	// 按 RISC-V 特权规范 §4.3, PPN 连续存放在 PTE 的位 [53:10]。
	let ppn = (raw >> 10) & 0xFFF_FFFF_FFFFu64;
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
// 通过页偏移与物理页表基地址组装出物理地址的逻辑
// ============================================================

/// 由叶子 PTE 的物理页号与原虚拟地址拼接物理地址。
///
/// `ppn`: 叶子 PTE 中取出的 44 位物理页号。
/// `va`: 原虚拟地址, 用于取页内偏移, 超级页情形下还用于取 VPN[0]。
/// `level`: 0 表示 4 KiB L3 页; 1 表示 2 MiB L2 超级页。
///
/// 4 KiB 页:
///     PA = (ppn << 12) | (va & 0xFFF)
/// 2 MiB 超级页:
///     PA = ((ppn[43:9] 与 va[20:12] 拼接的结果) << 12) | (va & 0xFFF)
#[no_mangle]
pub extern "C" fn pte_assemble_pa(ppn: u64, va: u64, level: u8) -> u64 {
	if level == 1 {
		// 2 MiB 超级页: PPN[8:0] 取自 vpn0,
		// PPN 的第 9 位及以上取自 PTE。
		let vpn0 = (va >> 12) & 0x1FF;
		let ppn = (ppn & 0xFFFF_FFFF_FFFF_FE00) | vpn0;
		let offset = va & 0xFFF;
		((ppn << PAGE_SHIFT) | offset) & 0xFFFF_FFFF_FFFF_FFFF
	} else {
		// 4 KiB 页: PA = (PPN << 12) | offset[11:0]
		let offset = va & 0xFFF;
		((ppn << PAGE_SHIFT) | offset) & 0xFFFF_FFFF_FFFF_FFFF
	}
}

/// 返回给定叶子级别的页大小; level 为 0 时 4 KiB, 为 1 时 2 MiB。
#[no_mangle]
pub extern "C" fn sv39_page_size(level: u8) -> u64 {
	if level == 1 {
		2 * 1024 * 1024 // 2 MiB
	} else {
		4096 // 4 KiB
	}
}

// ============================================================
//  测试
// ============================================================

#[cfg(test)]
mod tests {
	use super::*;

	// -- 虚拟页号分解 --

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
		//   vpn2 = VA[38:30] = 0
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

	// -- PTE 解析 --

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
		let raw = make_pte_raw(PTE_V, 0x80001); // V=1, R、W、X 全为 0, 为指针 PTE
		let p = pte_parse(raw);
		assert_eq!(p.v, 1);
		assert_eq!(p.is_leaf, 0);
		assert_eq!(p.is_ptr, 1);
		assert_eq!(p.ppn, 0x80001);
	}

	#[test]
	fn test_pte_ppn_boundary() {
		// 三个子字段均有置位的 PPN
		let ppn = 0x123456789AB; // 44 位
		let raw = make_pte_raw(PTE_V | PTE_R, ppn);
		let p = pte_parse(raw);
		assert_eq!(p.ppn, ppn & 0xFFFFFFFFFFF);
	}

	// -- 对物理地址的拼接 --

	#[test]
	fn test_assemble_pa_4k() {
		// 4 KiB 页: ppn=0x80100, va=0xABC, 则 pa = (0x80100 << 12) | 0xABC
		// 0x80100 << 12 = 0x8010_0000, 加 0xABC 得 0x8010_0ABC
		let pa = pte_assemble_pa(0x80100, 0xABC, 0);
		assert_eq!(pa, 0x8010_0ABCu64);
	}

	#[test]
	fn test_assemble_pa_2m_bit9_preserved() {
		// 2 MiB 超级页必须保留 PPN 的第 9 位。
		// ppn=0x80200 (第 9 位为 1), va[20:12]=0x123
		// new_ppn = (ppn & ~0x1FF) | vpn0 = 0x80200 | 0x123 = 0x80323
		// pa = 0x80323 << 12 = 0x8032_3000
		let pa = pte_assemble_pa(0x80200, 0x123_000u64, 1);
		// PPN 的第 9 位映射到 PA 的第 21 位
		assert_eq!(pa, 0x8032_3000u64);
		// 10 位掩码 ~0x3FF 会清除 PPN 的第 9 位, 结果与正确值不同
		let old_buggy = ((0x80200 & !0x3FFu64) | 0x123) << 12;
		assert_ne!(pa, old_buggy, "old 10-bit mask would clear PPN bit 9");
	}

	#[test]
	fn test_page_size() {
		assert_eq!(sv39_page_size(0), 4096);
		assert_eq!(sv39_page_size(1), 2 * 1024 * 1024);
	}
}
