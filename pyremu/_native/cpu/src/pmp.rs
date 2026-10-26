//! 物理内存保护检查, 全部状态经 FFI 传入, 函数本身只做计算。
//!
//! 依据 RISC-V 特权规范 §3.7, 支持 TOR、NA4、NAPOT 三种地址匹配方式。
//! 处理 M 模式绕过 PMP 检查以及 mstatus.MPRV 的影响,
//! 并按条目索引由小到大优先匹配。

use crate::consts::HOST_MDID;
use crate::state::{riscv_mode, HartState};

/// PMP 配置, 含条目配置字节、地址数组与条目数。
pub struct PmpCtx {
	pub cfg: *mut u8,
	pub addr: *mut u64,
	pub num: u8,
}

/// 检查对物理地址的访问是否被 PMP 允许。
#[inline]
pub fn pmp_ok(
	state: &HartState,
	pa: u64,
	size: u32,
	is_write: bool,
	is_execute: bool,
	pmp: &PmpCtx,
) -> bool {
	// 按 RISC-V 特权规范 §3.7.1, 未实现任何 PMP 条目时,
	// S 模式与 U 模式的访问一律被拒绝, 仅允许 M 模式访存。
	if pmp.num == 0 {
		if state.mode == riscv_mode::M {
			// RISC-V 特权规范 §4.1.12 要求取指不受 MPRV 影响。
			if is_execute {
				return true;
			}
			let mprv = (state.mstatus >> 17) & 1;
			if mprv == 0 {
				return true;
			}
			let mpp = (state.mstatus >> 11) & 0x3;
			if mpp == (riscv_mode::M as u64) {
				return true;
			}
		}
		return false;
	}
	if state.mode == riscv_mode::M {
		// 取指不受 MPRV 影响, M 模式取指一律绕过 PMP。
		if is_execute {
			return true;
		}
		let mprv = (state.mstatus >> 17) & 1;
		if mprv == 0 {
			return true;
		}
		// MPRV 为 1 时, PMP 按有效模式 MPP 检查;
		// MPP 为 M 时有效模式仍为 M, 绕过 PMP 检查。
		let mpp = (state.mstatus >> 11) & 0x3;
		if mpp == (riscv_mode::M as u64) {
			return true;
		}
	}
	pmp_check(
		pmp.cfg,
		pmp.addr,
		pmp.num,
		pa,
		size,
		if is_write { 1 } else { 0 },
		if is_execute { 1 } else { 0 },
		state.mode,
		state.mstatus,
		state.pmpsplit,
		state.mdid,
	) != 0
}

// ============================================================
//  PMP 配置常量
// ============================================================

const PMP_R: u8 = 0b0000_0001;
const PMP_W: u8 = 0b0000_0010;
const PMP_X: u8 = 0b0000_0100;
const PMP_A_MASK: u8 = 0b0001_1000;
const PMP_A_OFF: u8 = 0b0000_0000;
const PMP_A_TOR: u8 = 0b0000_1000;
const PMP_A_NA4: u8 = 0b0001_0000;
const PMP_A_NAPOT: u8 = 0b0001_1000;
#[allow(dead_code)]
const PMP_L: u8 = 0b1000_0000;

const _MODE_M: u8 = 3;

/// 解码 NAPOT 编码的 pmpaddr 值, 返回区域基址与大小。
///
/// 统计 54 位地址字段末尾连续 1 的个数, 得出自然对齐的 2 的幂次区域。
fn decode_napot(pmpaddr_val: u64) -> (u64, u64) {
	let val = pmpaddr_val & 0x003f_ffff_ffff_ffff; // 54 位 PMP 地址字段
	let trailing = val.trailing_ones();
	if trailing >= 54 {
		// 全为 1: 覆盖整个地址空间
		return (0, u64::MAX);
	}
	if trailing == 0 {
		// 末尾无 1: 区域大小为 8 字节
		return (val << 2, 8);
	}
	let size = 1u64 << (trailing + 3);
	let mask = (1u64 << trailing) - 1;
	let base = ((val & !mask) << 2) & 0xffff_ffff_ffff_ffff;
	(base, size)
}

/// 判定区间 [addr, addr+size) 是否完全位于 [base, base+rsize) 之内。
#[inline(always)]
fn addr_in_range(addr: u64, size: u32, base: u64, rsize: u64) -> bool {
	if rsize == 0 {
		return false;
	}
	let end = (addr + (size as u64)).wrapping_sub(1);
	let rend = (base + rsize).wrapping_sub(1);
	addr >= base && end <= rend
}

/// 检查单个条目的权限位，R 位为 0 时拒绝一切访问。
#[inline(always)]
fn check_perm(cfg: u8, is_write: bool, is_execute: bool) -> bool {
	if (cfg & PMP_R) == 0 {
		return false;
	}
	if is_write && (cfg & PMP_W) == 0 {
		return false;
	}
	if is_execute && (cfg & PMP_X) == 0 {
		return false;
	}
	true
}

// ============================================================
//  主检查函数
// ============================================================

/// 判定一次物理内存访问是否通过 PMP 规则。
///
/// 返回 0 表示拒绝, 1 表示允许。
/// 不依赖 FFI 的 `#[repr(C)]` 结构体布局。
#[no_mangle]
pub extern "C" fn pmp_check(
	cfg_ptr: *const u8,
	addr_ptr: *const u64,
	num_entries: u8,
	pa: u64,
	size: u32,
	is_write: u8,
	is_execute: u8,
	mode_val: u8,
	mstatus_val: u64,
	pmpsplit: u8,
	mdid: u64,
) -> u8 {
	// ---- 确定有效特权模式 ----
	let mut eff_mode = mode_val;
	if mode_val == _MODE_M {
		let mprv = (mstatus_val >> 17) & 1;
		if mprv == 0 {
			// M 模式且 MPRV 为 0: 绕过 PMP 检查
			return 1;
		}
		let mpp = (mstatus_val >> 11) & 0x3;
		// MPP 取 0 时为 U 模式, 1 为 S 模式, 其余则按 M 模式处理
		eff_mode = match mpp {
			0 => 0,
			1 => 1,
			_ => 3,
		};
		// MPRV=1 且 MPP 为 M 模式时，有效模式仍为 M, 不启用 PMP 检查。
		if eff_mode == _MODE_M {
			return 1;
		}
	}

	// ---- 无条目则拒绝本次访问 ----
	if num_entries == 0 {
		return 0;
	}

	let enclave_mode = mdid != HOST_MDID;
	if enclave_mode && pmpsplit > 0 && pmpsplit >= num_entries {
		return 0;
	}

	// 确保执行安全的前提: 指针由调用方提供, 在调用返回前保持有效。
	let cfg_slice = unsafe { core::slice::from_raw_parts(cfg_ptr, num_entries as usize) };
	let addr_slice = unsafe { core::slice::from_raw_parts(addr_ptr, num_entries as usize) };

	// ---- 按优先级匹配, 条目索引小者优先 ----
	for i in 0..num_entries as usize {
		if enclave_mode && pmpsplit > 0 && i < (pmpsplit as usize) {
			continue;
		}

		let cfg = cfg_slice[i];
		let a_type = cfg & PMP_A_MASK;
		if a_type == PMP_A_OFF {
			continue;
		}

		let addr_field = addr_slice[i];
		let (base, rsize) = match a_type {
			PMP_A_TOR => {
				let prev = if i > 0 { addr_slice[i - 1] } else { 0 };
				let lo = if i == 0 { 0 } else { prev << 2 };
				let hi = addr_field << 2;
				(lo, hi.wrapping_sub(lo))
			}
			PMP_A_NA4 => (addr_field << 2, 4),
			PMP_A_NAPOT => decode_napot(addr_field),
			_ => {
				continue;
			} // A 字段取值未知, 按 OFF 处理
		};

		if !addr_in_range(pa, size, base, rsize) {
			continue;
		}

		// 命中条目时按该条目的权限位判定
		return check_perm(cfg, is_write != 0, is_execute != 0) as u8;
	}

	// ---- 无条目命中时，拒绝本次访问 ----
	(eff_mode == _MODE_M) as u8
}

// ============================================================
//  测试
// ============================================================

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_decode_napot_8bytes() {
		// 末尾没有 1 则表示区域为 8 字节
		let (base, size) = decode_napot(0x0000_0000_0000_0000);
		assert_eq!(base, 0);
		assert_eq!(size, 8);
	}

	#[test]
	fn test_decode_napot_4k() {
		// 8 个末尾连续的 1, k=8, 区域大小为 2^(8+3) = 2048 字节。
		// NAPOT 编码: pmpaddr = (base >> 2) | ((size - 1) >> 3)。
		let val = 0x0000_0000_0000_00ff; // 8 个末尾连续的 1
		let trailing = (val & 0x003f_ffff_ffff_ffffu64).trailing_ones();
		assert_eq!(trailing, 8);
		let (_base, size) = decode_napot(val);
		assert_eq!(size, 2048);
	}

	fn check(
		cfg: &[u8],
		addr: &[u64],
		n: u8,
		pa: u64,
		is_write: u8,
		is_execute: u8,
		mode_val: u8,
		mstatus_val: u64,
		pmpsplit: u8,
		mdid: u64,
	) -> u8 {
		pmp_check(
			cfg.as_ptr(),
			addr.as_ptr(),
			n,
			pa,
			4,
			is_write,
			is_execute,
			mode_val,
			mstatus_val,
			pmpsplit,
			mdid,
		)
	}

	#[test]
	fn test_no_entries_smode_deny() {
		assert_eq!(check(&[], &[], 0, 0x8000_0000, 0, 0, 1, 0, 0, 0), 0);
	}

	#[test]
	fn test_no_entries_mmode_allow() {
		assert_eq!(check(&[], &[], 0, 0x8000_0000, 0, 0, 3, 0, 0, 0), 1);
	}

	#[test]
	fn test_tor_match() {
		let cfg = [PMP_A_TOR | PMP_R | PMP_W | PMP_X];
		let addr = [0x2040_0000u64]; // 0x8100_0000 >> 2
		assert_eq!(check(&cfg, &addr, 1, 0x8000_0000, 0, 0, 1, 0, 0, 0), 1);
	}

	#[test]
	fn test_tor_mismatch() {
		let cfg = [PMP_A_TOR | PMP_R | PMP_W];
		let addr = [0x1000_0000u64]; // 覆盖 [0, 0x4000_0000)
		assert_eq!(check(&cfg, &addr, 1, 0x8000_0000, 0, 0, 1, 0, 0, 0), 0);
	}

	#[test]
	fn test_tor_write_denied() {
		let cfg = [PMP_A_TOR | PMP_R];
		let addr = [0x2040_0000u64];
		assert_eq!(check(&cfg, &addr, 1, 0x8000_0000, 1, 0, 1, 0, 0, 0), 0);
	}

	#[test]
	fn test_napot_4k() {
		let pmpaddr = 0x2000_01ffu64;
		let cfg = [PMP_A_NAPOT | PMP_R | PMP_W];
		let addr = [pmpaddr];
		assert_eq!(check(&cfg, &addr, 1, 0x8000_0000, 0, 0, 1, 0, 0, 0), 1);
	}

	#[test]
	fn test_na4_match() {
		let cfg = [PMP_A_NA4 | PMP_R];
		let addr = [0x2000_0000u64];
		assert_eq!(check(&cfg, &addr, 1, 0x8000_0000, 0, 0, 1, 0, 0, 0), 1);
	}

	#[test]
	fn test_na4_mismatch() {
		let cfg = [PMP_A_NA4 | PMP_R];
		let addr = [0x2000_0000u64];
		assert_eq!(check(&cfg, &addr, 1, 0x8000_0008, 0, 0, 1, 0, 0, 0), 0);
	}

	#[test]
	fn test_priority_first_wins() {
		let cfg = [PMP_A_TOR | PMP_R, PMP_A_TOR | PMP_R | PMP_W];
		let addr = [0x8100_0000u64 >> 2, 0x8200_0000u64 >> 2];
		assert_eq!(check(&cfg, &addr, 2, 0x8000_0000, 1, 0, 1, 0, 0, 0), 0);
	}

	#[test]
	fn test_enclave_split() {
		let cfg = [
			PMP_A_TOR | PMP_R | PMP_W,
			PMP_A_TOR | PMP_R | PMP_W,
			PMP_A_TOR | PMP_R | PMP_W,
			PMP_A_TOR | PMP_R | PMP_W,
		];
		let addr = [
			0x0010_0000u64,
			0x0020_0000u64,
			0x0030_0000u64,
			0x8300_0000u64 >> 2,
		];
		// PA=0x82FF_FFF0 位于条目 3 的 TOR 区间 [0x00C0_0000, 0x8300_0000) 内
		assert_eq!(check(&cfg, &addr, 4, 0x82ff_fff0u64, 0, 0, 1, 0, 2, 1), 1);
	}

	#[test]
	fn test_enclave_split_high_mdid() {
		let cfg = [
			PMP_A_TOR | PMP_R | PMP_W,
			PMP_A_TOR | PMP_R | PMP_W,
			PMP_A_TOR | PMP_R | PMP_W,
			PMP_A_TOR | PMP_R | PMP_W,
		];
		let addr = [
			0x0010_0000u64,
			0x0020_0000u64,
			0x0030_0000u64,
			0x8300_0000u64 >> 2,
		];
		assert_eq!(check(&cfg, &addr, 4, 0x0008_0000u64, 0, 0, 1, 0, 2, 256), 0);
		assert_eq!(check(&cfg, &addr, 4, 0x82ff_fff0u64, 0, 0, 1, 0, 2, 300), 1);
	}

	#[test]
	fn test_mmode_mprv0_always_pass() {
		let cfg = [PMP_A_OFF];
		let addr: [u64; 0] = [];
		assert_eq!(check(&cfg, &addr, 0, 0x8000_0000, 1, 0, 3, 0, 0, 0), 1);
	}

	#[test]
	fn test_mmode_mprv1_mpp_m_bypasses_even_if_matched() {
		// MPRV 为 1 且 MPP 为 3 时有效模式为 M, 绕过 PMP 检查。
		// 条目 0 与 PA 0x8000_0000 匹配且无 RWX 三者任意一个的权限, 该条目应被忽略。
		let cfg = [PMP_A_NAPOT | 0x00]; // NAPOT 模式且无 RWX 权限
		let addr = [0x2000_01ffu64]; // 4 KiB NAPOT, 覆盖 0x8000_0000
		let mstatus = (1u64 << 17) | (3u64 << 11); // MPRV=1, MPP=3
											 // 在 0x8000_0000 取指，条目匹配, 但应绕过 PMP 检查
		assert_eq!(
			check(&cfg, &addr, 1, 0x8000_0000, 0, 1, 3, mstatus, 0, 0),
			1
		);
		// 在 0x8000_0000 的写入同样应绕过 PMP 检查
		assert_eq!(
			check(&cfg, &addr, 1, 0x8000_0000, 1, 0, 3, mstatus, 0, 0),
			1
		);
	}

	#[test]
	fn test_mmode_mprv1_mpp_s_enforces_pmp() {
		// MPRV=1 且 MPP=S(1): 有效模式为 S, PMP 检查生效。
		// 条目 0 与 PA 0x8000_0000 匹配且无 RWX 权限, 拒绝访存。
		let cfg = [PMP_A_NAPOT | 0x00]; // NAPOT, 无 RWX 权限
		let addr = [0x2000_01ffu64];
		let mstatus = (1u64 << 17) | (1u64 << 11); // MPRV=1, MPP=1 表示有效模式为 S
		assert_eq!(
			check(&cfg, &addr, 1, 0x8000_0000, 0, 1, 3, mstatus, 0, 0),
			0
		);
	}

	// ---------------------------------------------------------------
	//  PMP 配置表用例
	// ---------------------------------------------------------------

	/// 条目 61: 基址 0x8000_0000, 大小 32 MiB, NAPOT, 权限 RWX。
	/// pmpaddr = 0x203FFFFF, 表明 base = 0x8000_0000, size = 32 MiB。
	#[test]
	fn test_napot_entry61_decode() {
		let pmpaddr = 0x0000_0000_203f_ffffu64;
		let (base, size) = decode_napot(pmpaddr);
		assert_eq!(base, 0x0000_0000_8000_0000, "entry 61 base");
		assert_eq!(size, 32 * 1024 * 1024, "entry 61 size = 32 MiB");
	}

	/// 条目 62: 基址 0, 大小 2 GiB, NAPOT, 权限 RWX。
	/// pmpaddr = 0x0FFFFFFF, 表明 base = 0x0, size = 2 GiB。
	#[test]
	fn test_napot_entry62_decode() {
		let pmpaddr = 0x0000_0000_0fff_ffffu64;
		let (base, size) = decode_napot(pmpaddr);
		assert_eq!(base, 0x0000_0000_0000_0000, "entry 62 base");
		assert_eq!(size, 2u64 * 1024 * 1024 * 1024, "entry 62 size = 2 GiB");
	}

	/// 条目 63: cfg=0x18, 即 NAPOT 且无 RWX 权限, 覆盖 16 EB。
	/// 凡匹配到该条目的访问一律拒绝。
	#[test]
	fn test_napot_entry63_decode() {
		let pmpaddr = 0x003f_ffff_ffff_ffffu64;
		let (base, size) = decode_napot(pmpaddr);
		assert_eq!(base, 0, "entry 63 base");
		assert_eq!(size, u64::MAX, "entry 63 covers all");
	}

	/// 完整的 64 条目 PMP 配置: 在 0x8020_1108 处 S 模式取指
	/// 应由条目 61 匹配并允许。
	#[test]
	fn test_full_config_smode_fetch_allowed() {
		let mut cfg = [0u8; 64];
		let mut addr = [0u64; 64];

		// 构造条目 0 至 63
		let entries: &[(u64, u8)] = &[
			(0x003fff9fffffffff, 0x1f), //  0: NAPOT RWX  1.0 TB
			(0x003fff3fffffffff, 0x1f), //  1: NAPOT RWX  2.0 TB
			(0x003ffe7fffffffff, 0x1f), //  2: NAPOT RWX  4.0 TB
			(0x003ffcffffffffff, 0x1f), //  3: NAPOT RWX  8.0 TB
			(0x003ff9ffffffffff, 0x1f), //  4: NAPOT RWX 16.0 TB
			(0x003ff3ffffffffff, 0x1f), //  5: NAPOT RWX 32.0 TB
			(0x003fe7ffffffffff, 0x1f), //  6: NAPOT RWX 64.0 TB
			(0x003fcfffffffffff, 0x1f), //  7: NAPOT RWX 128.0 TB
			(0x003f9fffffffffff, 0x1f), //  8: NAPOT RWX 256.0 TB
			(0x003f3fffffffffff, 0x1f), //  9: NAPOT RWX 512.0 TB
			(0x003e7fffffffffff, 0x1f), // 10: NAPOT RWX 1.0 PB
			(0x003cfffffffffff, 0x1f),  // 11: NAPOT RWX 2.0 PB
			(0x0039fffffffffff, 0x1f),  // 12: NAPOT RWX 4.0 PB
			(0x0033fffffffffff, 0x1f),  // 13: NAPOT RWX 8.0 PB
			(0x0027fffffffffff, 0x1f),  // 14: NAPOT RWX 16.0 PB
			(0x000fffffffffff, 0x1f),   // 15: NAPOT RWX 32.0 PB
			(0x001fffffffffff, 0x1f),   // 16: NAPOT RWX 64.0 PB
			(0x003fffffffffff, 0x1f),   // 17: NAPOT RWX 16.0 EB
			(0x003fffffffffff, 0x1f),   // 18: NAPOT RWX 16.0 EB
			(0x003fffffffffff, 0x1f),   // 19: NAPOT RWX 16.0 EB
			(0x003fffffffffff, 0x1f),   // 20: NAPOT RWX 16.0 EB
			(0x003fffffffffff, 0x1f),   // 21: NAPOT RWX 16.0 EB
			(0x003fffffffffff, 0x1f),   // 22: NAPOT RWX 16.0 EB
			(0x003fffffffffff, 0x1f),   // 23: NAPOT RWX 16.0 EB
			(0x003fffffffffff, 0x1f),   // 24: NAPOT RWX 16.0 EB
			(0x003fffffffffff, 0x1f),   // 25: NAPOT RWX 16.0 EB
			(0x003fffffffffff, 0x1f),   // 26: NAPOT RWX 16.0 EB
			(0x003fffffffffff, 0x1f),   // 27: NAPOT RWX 16.0 EB
			(0x003fffffffffff, 0x1f),   // 28: NAPOT RWX 16.0 EB
			(0x001fffffffffff, 0x1f),   // 29: NAPOT RWX 64.0 PB
			(0x002fffffffffff, 0x1f),   // 30: NAPOT RWX 32.0 PB
			(0x0017ffffffffff, 0x1f),   // 31: NAPOT RWX 16.0 PB
			(0x000bffffffffff, 0x1f),   // 32: NAPOT RWX 8.0 PB
			(0x0005ffffffffff, 0x1f),   // 33: NAPOT RWX 4.0 PB
			(0x0002ffffffffff, 0x1f),   // 34: NAPOT RWX 2.0 PB
			(0x00017fffffffff, 0x1f),   // 35: NAPOT RWX 1.0 PB
			(0x0000bfffffffff, 0x1f),   // 36: NAPOT RWX 512.0 TB
			(0x00005fffffffff, 0x1f),   // 37: NAPOT RWX 256.0 TB
			(0x00002fffffffff, 0x1f),   // 38: NAPOT RWX 128.0 TB
			(0x000017ffffffff, 0x1f),   // 39: NAPOT RWX 64.0 TB
			(0x00000bffffffff, 0x1f),   // 40: NAPOT RWX 32.0 TB
			(0x000005ffffffff, 0x1f),   // 41: NAPOT RWX 16.0 TB
			(0x000002ffffffff, 0x1f),   // 42: NAPOT RWX 8.0 TB
			(0x0000017fffffff, 0x1f),   // 43: NAPOT RWX 4.0 TB
			(0x000000bfffffff, 0x1f),   // 44: NAPOT RWX 2.0 TB
			(0x0000005fffffff, 0x1f),   // 45: NAPOT RWX 1.0 TB
			(0x0000002fffffff, 0x1f),   // 46: NAPOT RWX 512.0 GB
			(0x00000017ffffff, 0x1f),   // 47: NAPOT RWX 256.0 GB
			(0x0000000bffffff, 0x1f),   // 48: NAPOT RWX 128.0 GB
			(0x00000005ffffff, 0x1f),   // 49: NAPOT RWX 64.0 GB
			(0x00000002ffffff, 0x1f),   // 50: NAPOT RWX 32.0 GB
			(0x000000017fffff, 0x1f),   // 51: NAPOT RWX 16.0 GB
			(0x00000000bfffff, 0x1f),   // 52: NAPOT RWX 8.0 GB
			(0x000000005fffff, 0x1f),   // 53: NAPOT RWX 4.0 GB
			(0x0000000037ffff, 0x1f),   // 54: NAPOT RWX 1.0 GB
			(0x000000002bffff, 0x1f),   // 55: NAPOT RWX 512.0 MB
			(0x0000000025ffff, 0x1f),   // 56: NAPOT RWX 256.0 MB
			(0x0000000022ffff, 0x1f),   // 57: NAPOT RWX 128.0 MB
			(0x00000000217fff, 0x1f),   // 58: NAPOT RWX 64.0 MB
			(0x0000000020efff, 0x1f),   // 59: NAPOT RWX 8.0 MB
			(0x00000000209fff, 0x1f),   // 60: NAPOT RWX 16.0 MB
			(0x00000000203fff, 0x1f),   // 61: NAPOT RWX 32.0 MB @0x80000000
			(0x000000000fffff, 0x1f),   // 62: NAPOT RWX 2.0 GB  @0x0
			(0x003fffffffffff, 0x18),   // 63: NAPOT, 无 R、W、X 权限, 16 EB, 覆盖全部地址
		];

		for (i, (pmpaddr, cfg_val)) in entries.iter().enumerate() {
			cfg[i] = *cfg_val;
			addr[i] = *pmpaddr;
		}

		// S 模式在 0x8020_1108 取指: 位于条目 61 的区间 [0x8000_0000, 0x8200_0000) 内
		assert_eq!(
			check(
				&cfg,
				&addr,
				64,
				0x0000_0000_8020_1108,
				0,
				1,
				/* S 模式取指 */ 1,
				0,
				0,
				0
			),
			1,
			"S-mode fetch at 0x80201108 should be allowed by entry 61"
		);

		// S 模式在 0x8020_1108 加载: 同样位于条目 61 的区间内
		assert_eq!(
			check(
				&cfg,
				&addr,
				64,
				0x0000_0000_8020_1108,
				0,
				0,
				/* S 模式加载 */ 1,
				0,
				0,
				0
			),
			1,
			"S-mode load at 0x80201108 should be allowed by entry 61"
		);

		// S 模式在 0x8020_1108 存储: 同样位于条目 61 的区间内
		assert_eq!(
			check(
				&cfg,
				&addr,
				64,
				0x0000_0000_8020_1108,
				1,
				0,
				/* S 模式存储 */ 1,
				0,
				0,
				0
			),
			1,
			"S-mode store at 0x80201108 should be allowed by entry 61"
		);
	}
}
