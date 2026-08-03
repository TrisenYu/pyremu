//! Physical Memory Protection (PMP) check — pure function, all state passed by FFI.
//!
//! RISC-V Privileged Spec §3.7.  Supports TOR, NA4, and NAPOT matching modes.
//! Handles M-mode bypass (MPRV-aware), enclave pmpsplit partitioning, and
//! the standard priority-based matching rules.

use crate::state::{riscv_mode, HartState};

/// PMP (Physical Memory Protection) configuration for the batch.
pub struct PmpCtx {
    pub cfg: *mut u8,
    pub addr: *mut u64,
    pub num: u8,
}

/// Check PMP for a physical address access.  Returns true if access is allowed.
#[inline]
pub fn pmp_ok(
    state: &HartState,
    pa: u64,
    size: u32,
    is_write: bool,
    is_execute: bool,
    pmp: &PmpCtx,
) -> bool {
    // RISC-V Privileged Spec §3.7.1: when no PMP entries are implemented,
    // S and U mode accesses are denied.  Only M-mode is allowed.
    if pmp.num == 0 {
        if state.mode == riscv_mode::M {
            // Instruction fetches ignore MPRV (RISC-V spec §4.1.12).
            if is_execute { return true; }
            let mprv = (state.mstatus >> 17) & 1;
            if mprv == 0 { return true; }
            let mpp = (state.mstatus >> 11) & 0x3;
            if mpp == riscv_mode::M as u64 { return true; }
        }
        return false;
    }
    if state.mode == riscv_mode::M {
        // RISC-V Privileged Spec §4.1.12: "Instruction access-fault and
        // instruction page-fault exceptions are unaffected by MPRV."
        // Instruction fetches always use the current privilege mode (M)
        // for PMP, so M-mode fetches unconditionally bypass PMP.
        if is_execute {
            return true;
        }
        let mprv = (state.mstatus >> 17) & 1;
        if mprv == 0 {
            return true;
        }
        // MPRV=1: PMP uses effective mode = MPP.
        // If MPP=M, effective mode is still M -> PMP bypass.
        let mpp = (state.mstatus >> 11) & 0x3;
        if mpp == riscv_mode::M as u64 {
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
//  PMP configuration constants
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

/// Decode a NAPOT-encoded pmpaddr value -> (base, size).
///
/// Counts trailing ones in the 54-bit address field to determine
/// the naturally-aligned power-of-two region.
fn decode_napot(pmpaddr_val: u64) -> (u64, u64) {
    let val = pmpaddr_val & 0x003F_FFFF_FFFF_FFFF; // 54-bit PMP address field
    let trailing = val.trailing_ones();
    if trailing >= 54 {
        // All ones: cover entire address space
        return (0, u64::MAX);
    }
    if trailing == 0 {
        // No trailing ones: 8-byte region
        return (val << 2, 8);
    }
    let size = 1u64 << (trailing + 3);
    let mask = (1u64 << trailing) - 1;
    let base = ((val & !mask) << 2) & 0xFFFF_FFFF_FFFF_FFFF;
    (base, size)
}

/// Check whether [addr, addr+size) lies entirely within [base, base+rsize).
#[inline(always)]
fn addr_in_range(addr: u64, size: u32, base: u64, rsize: u64) -> bool {
    if rsize == 0 {
        return false;
    }
    let end = (addr + size as u64).wrapping_sub(1);
    let rend = (base + rsize).wrapping_sub(1);
    addr >= base && end <= rend
}

/// Check per-entry permission bits.
#[inline(always)]
fn check_perm(cfg: u8, is_write: bool, is_execute: bool) -> bool {
    if cfg & PMP_R == 0 {
        return false;
    }
    if is_write && cfg & PMP_W == 0 {
        return false;
    }
    if is_execute && cfg & PMP_X == 0 {
        return false;
    }
    true
}

// ============================================================
//  FFI input struct
// ============================================================

// ============================================================
//  Main check function
// ============================================================

/// Check whether a physical memory access is permitted by the PMP rules.
///
/// Returns `0` (deny) or `1` (allow).  All state is passed via individual
/// parameters to avoid `#[repr(C)]` struct layout issues across FFI.
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
    mdid: u8,
) -> u8 {

    // ---- Determine effective privilege mode ----
    let mut eff_mode = mode_val;
    if mode_val == _MODE_M {
        let mprv = (mstatus_val >> 17) & 1;
        if mprv != 0 {
            let mpp = (mstatus_val >> 11) & 0x3;
            // MPP encoding: 0=U, 1=S, 3=M
            eff_mode = match mpp {
                0 => 0,
                1 => 1,
                _ => 3,
            };
            // MPRV=1 but MPP=M -> effective mode is still M -> PMP bypass
            if eff_mode == _MODE_M {
                return 1;
            }
        } else {
            // M-mode with MPRV=0: PMP does not apply
            return 1;
        }
    }

    // ---- No entries -> S/U mode always denied (RISC-V spec) ----
    if num_entries == 0 {
        return 0;
    }

    // ---- Enclave mode (mdid != 0): entries in [0, pmpsplit) are OFF ----
    let enclave_mode = mdid != 0;
    if enclave_mode && pmpsplit > 0 && pmpsplit >= num_entries {
        return 0;
    }

    // Safety: pointers are valid for the duration of the call (Python holds the GIL).
    let cfg_slice = unsafe { core::slice::from_raw_parts(cfg_ptr, num_entries as usize) };
    let addr_slice = unsafe { core::slice::from_raw_parts(addr_ptr, num_entries as usize) };

    // ---- Priority-based matching (lower index = higher priority) ----
    for i in 0..num_entries as usize {
        // Enclave mode: skip host-side entries
        if enclave_mode && pmpsplit > 0 && i < pmpsplit as usize {
            continue;
        }

        let cfg = cfg_slice[i];
        let a_type = cfg & PMP_A_MASK;
        if a_type == PMP_A_OFF {
            continue;
        }

        // Decode the matching region
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
            _ => continue, // Unknown A field — treat as OFF
        };

        if !addr_in_range(pa, size, base, rsize) {
            continue;
        }

        // Matched: apply this entry's permissions
        return check_perm(cfg, is_write != 0, is_execute != 0) as u8;
    }

    // ---- No matching entry -> M-mode allows, S/U denies ----
    (eff_mode == _MODE_M) as u8
}

// ============================================================
//  Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_napot_8bytes() {
        // No trailing 1 -> 8 bytes
        let (base, size) = decode_napot(0x0000_0000_0000_0000);
        assert_eq!(base, 0);
        assert_eq!(size, 8);
    }

    #[test]
    fn test_decode_napot_4k() {
        // 8 trailing ones -> 2^(8+3) = 2^11 = 2048 ... wait, trailing ones on pmpaddr.
        // pmpaddr with 8 trailing ones: size = 2^(8+3) = 2^11 = 2048 bytes.
        // But pmpaddr is shifted right by 2, so the region is actually 2048 bytes.
        // Wait, let me re-read the spec.
        // NAPOT encoding: pmpaddr = (base >> 2) | ((size-1) >> 3)
        // The trailing ones count gives us k, then size = 2^(k+3).
        // 8 trailing ones -> k=8 -> size=2^11=2048.
        let val = 0x0000_0000_0000_00FF; // 8 trailing ones
        let trailing = (val & 0x003F_FFFF_FFFF_FFFFu64).trailing_ones();
        assert_eq!(trailing, 8);
        let (_base, size) = decode_napot(val);
        assert_eq!(size, 2048);
    }

    fn check(
        cfg: &[u8], addr: &[u64], n: u8, pa: u64,
        is_write: u8, is_execute: u8, mode_val: u8,
        mstatus_val: u64, pmpsplit: u8, mdid: u8,
    ) -> u8 {
        pmp_check(cfg.as_ptr(), addr.as_ptr(), n, pa, 4,
                  is_write, is_execute, mode_val, mstatus_val, pmpsplit, mdid)
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
        let addr = [0x1000_0000u64]; // covers [0, 0x4000_0000)
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
        let pmpaddr = 0x2000_01FFu64;
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
        let addr = [0x0010_0000u64, 0x0020_0000u64, 0x0030_0000u64, 0x8300_0000u64 >> 2];
        // PA=0x82FF_FFF0 inside entry 3's TOR range [0x00C0_0000, 0x8300_0000)
        assert_eq!(check(&cfg, &addr, 4, 0x82FF_FFF0u64, 0, 0, 1, 0, 2, 1), 1);
    }

    #[test]
    fn test_mmode_mprv0_always_pass() {
        let cfg = [PMP_A_OFF];
        let addr: [u64; 0] = [];
        assert_eq!(check(&cfg, &addr, 0, 0x8000_0000, 1, 0, 3, 0, 0, 0), 1);
    }

    #[test]
    fn test_mmode_mprv1_mpp_m_bypasses_even_if_matched() {
        // MPRV=1, MPP=M(3) -> effective mode is M -> PMP bypass.
        // Entry 0 matches PA 0x8000_0000 with no R/W/X — but should be ignored.
        let cfg = [PMP_A_NAPOT | 0x00]; // NAPOT, no R/W/X
        let addr = [0x2000_01FFu64]; // 4K NAPOT covering 0x80000000
        let mstatus = (1u64 << 17) | (3u64 << 11); // MPRV=1, MPP=3(M)
        // Execute at 0x80000000 — PMP entry matches but should bypass
        assert_eq!(check(&cfg, &addr, 1, 0x8000_0000, 0, 1, 3, mstatus, 0, 0), 1);
        // Write at 0x80000000 — same, should bypass
        assert_eq!(check(&cfg, &addr, 1, 0x8000_0000, 1, 0, 3, mstatus, 0, 0), 1);
    }

    #[test]
    fn test_mmode_mprv1_mpp_s_enforces_pmp() {
        // MPRV=1, MPP=S(1) -> effective mode is S -> PMP enforced.
        // Entry 0 matches PA 0x8000_0000 with no R/W/X -> DENY.
        let cfg = [PMP_A_NAPOT | 0x00]; // NAPOT, no R/W/X
        let addr = [0x2000_01FFu64];
        let mstatus = (1u64 << 17) | (1u64 << 11); // MPRV=1, MPP=1(S)
        assert_eq!(check(&cfg, &addr, 1, 0x8000_0000, 0, 1, 3, mstatus, 0, 0), 0);
    }

    // ---------------------------------------------------------------
    //  Regression tests from tobe_fix.txt PMP table
    // ---------------------------------------------------------------

    /// Entry 61: NAPOT 32 MiB @ 0x8000_0000 with RWX.
    /// pmpaddr = 0x203FFFFF -> base=0x8000_0000, size=32 MiB.
    #[test]
    fn test_napot_entry61_decode() {
        let pmpaddr = 0x0000_0000_203F_FFFFu64;
        let (base, size) = decode_napot(pmpaddr);
        assert_eq!(base, 0x0000_0000_8000_0000, "entry 61 base");
        assert_eq!(size, 32 * 1024 * 1024, "entry 61 size = 32 MiB");
    }

    /// Entry 62: NAPOT 2 GiB @ 0x0 with RWX.
    /// pmpaddr = 0x0FFFFFFF -> base=0x0, size=2 GiB.
    #[test]
    fn test_napot_entry62_decode() {
        let pmpaddr = 0x0000_0000_0FFF_FFFFu64;
        let (base, size) = decode_napot(pmpaddr);
        assert_eq!(base, 0x0000_0000_0000_0000, "entry 62 base");
        assert_eq!(size, 2u64 * 1024 * 1024 * 1024, "entry 62 size = 2 GiB");
    }

    /// Entry 63: NAPOT 16 EB catch-all with cfg=0x18 (NAPOT, no R/W/X).
    /// Denies everything that reaches it.
    #[test]
    fn test_napot_entry63_decode() {
        let pmpaddr = 0x003F_FFFF_FFFF_FFFFu64;
        let (base, size) = decode_napot(pmpaddr);
        assert_eq!(base, 0, "entry 63 base");
        assert_eq!(size, u64::MAX, "entry 63 covers all");
    }

    /// Full 64-entry PMP config: S-mode fetch at 0x8020_1108
    /// should be ALLOWED (matched by entry 61).
    #[test]
    fn test_full_config_smode_fetch_allowed() {
        let mut cfg = [0u8; 64];
        let mut addr = [0u64; 64];

        // Build entries 0-63 matching tobe_fix.txt
        let entries: &[(u64, u8)] = &[
            (0x003FFF9FFFFFFFFF, 0x1F), //  0: NAPOT RWX  1.0 TB
            (0x003FFF3FFFFFFFFF, 0x1F), //  1: NAPOT RWX  2.0 TB
            (0x003FFE7FFFFFFFFF, 0x1F), //  2: NAPOT RWX  4.0 TB
            (0x003FFCFFFFFFFFFF, 0x1F), //  3: NAPOT RWX  8.0 TB
            (0x003FF9FFFFFFFFFF, 0x1F), //  4: NAPOT RWX 16.0 TB
            (0x003FF3FFFFFFFFFF, 0x1F), //  5: NAPOT RWX 32.0 TB
            (0x003FE7FFFFFFFFFF, 0x1F), //  6: NAPOT RWX 64.0 TB
            (0x003FCFFFFFFFFFFF, 0x1F), //  7: NAPOT RWX 128.0 TB
            (0x003F9FFFFFFFFFFF, 0x1F), //  8: NAPOT RWX 256.0 TB
            (0x003F3FFFFFFFFFFF, 0x1F), //  9: NAPOT RWX 512.0 TB
            (0x003E7FFFFFFFFFFF, 0x1F), // 10: NAPOT RWX 1.0 PB
            (0x003CFFFFFFFFFFF, 0x1F),  // 11: NAPOT RWX 2.0 PB
            (0x0039FFFFFFFFFFF, 0x1F),  // 12: NAPOT RWX 4.0 PB
            (0x0033FFFFFFFFFFF, 0x1F),  // 13: NAPOT RWX 8.0 PB
            (0x0027FFFFFFFFFFF, 0x1F),  // 14: NAPOT RWX 16.0 PB
            (0x000FFFFFFFFFFF, 0x1F),   // 15: NAPOT RWX 32.0 PB
            (0x001FFFFFFFFFFF, 0x1F),   // 16: NAPOT RWX 64.0 PB
            (0x003FFFFFFFFFFF, 0x1F),   // 17: NAPOT RWX 16.0 EB
            (0x003FFFFFFFFFFF, 0x1F),   // 18: NAPOT RWX 16.0 EB (disabled-like)
            (0x003FFFFFFFFFFF, 0x1F),   // 19: NAPOT RWX 16.0 EB
            (0x003FFFFFFFFFFF, 0x1F),   // 20: NAPOT RWX 16.0 EB
            (0x003FFFFFFFFFFF, 0x1F),   // 21: NAPOT RWX 16.0 EB
            (0x003FFFFFFFFFFF, 0x1F),   // 22: NAPOT RWX 16.0 EB
            (0x003FFFFFFFFFFF, 0x1F),   // 23: NAPOT RWX 16.0 EB
            (0x003FFFFFFFFFFF, 0x1F),   // 24: NAPOT RWX 16.0 EB
            (0x003FFFFFFFFFFF, 0x1F),   // 25: NAPOT RWX 16.0 EB
            (0x003FFFFFFFFFFF, 0x1F),   // 26: NAPOT RWX 16.0 EB
            (0x003FFFFFFFFFFF, 0x1F),   // 27: NAPOT RWX 16.0 EB
            (0x003FFFFFFFFFFF, 0x1F),   // 28: NAPOT RWX 16.0 EB
            (0x001FFFFFFFFFFF, 0x1F),   // 29: NAPOT RWX 64.0 PB
            (0x002FFFFFFFFFFF, 0x1F),   // 30: NAPOT RWX 32.0 PB
            (0x0017FFFFFFFFFF, 0x1F),   // 31: NAPOT RWX 16.0 PB
            (0x000BFFFFFFFFFF, 0x1F),   // 32: NAPOT RWX 8.0 PB
            (0x0005FFFFFFFFFF, 0x1F),   // 33: NAPOT RWX 4.0 PB
            (0x0002FFFFFFFFFF, 0x1F),   // 34: NAPOT RWX 2.0 PB
            (0x00017FFFFFFFFF, 0x1F),   // 35: NAPOT RWX 1.0 PB
            (0x0000BFFFFFFFFF, 0x1F),   // 36: NAPOT RWX 512.0 TB
            (0x00005FFFFFFFFF, 0x1F),   // 37: NAPOT RWX 256.0 TB
            (0x00002FFFFFFFFF, 0x1F),   // 38: NAPOT RWX 128.0 TB
            (0x000017FFFFFFFF, 0x1F),   // 39: NAPOT RWX 64.0 TB
            (0x00000BFFFFFFFF, 0x1F),   // 40: NAPOT RWX 32.0 TB
            (0x000005FFFFFFFF, 0x1F),   // 41: NAPOT RWX 16.0 TB
            (0x000002FFFFFFFF, 0x1F),   // 42: NAPOT RWX 8.0 TB
            (0x0000017FFFFFFF, 0x1F),   // 43: NAPOT RWX 4.0 TB
            (0x000000BFFFFFFF, 0x1F),   // 44: NAPOT RWX 2.0 TB
            (0x0000005FFFFFFF, 0x1F),   // 45: NAPOT RWX 1.0 TB
            (0x0000002FFFFFFF, 0x1F),   // 46: NAPOT RWX 512.0 GB
            (0x00000017FFFFFF, 0x1F),   // 47: NAPOT RWX 256.0 GB
            (0x0000000BFFFFFF, 0x1F),   // 48: NAPOT RWX 128.0 GB
            (0x00000005FFFFFF, 0x1F),   // 49: NAPOT RWX 64.0 GB
            (0x00000002FFFFFF, 0x1F),   // 50: NAPOT RWX 32.0 GB
            (0x000000017FFFFF, 0x1F),   // 51: NAPOT RWX 16.0 GB
            (0x00000000BFFFFF, 0x1F),   // 52: NAPOT RWX 8.0 GB
            (0x000000005FFFFF, 0x1F),   // 53: NAPOT RWX 4.0 GB
            (0x0000000037FFFF, 0x1F),   // 54: NAPOT RWX 1.0 GB
            (0x000000002BFFFF, 0x1F),   // 55: NAPOT RWX 512.0 MB
            (0x0000000025FFFF, 0x1F),   // 56: NAPOT RWX 256.0 MB
            (0x0000000022FFFF, 0x1F),   // 57: NAPOT RWX 128.0 MB
            (0x00000000217FFF, 0x1F),   // 58: NAPOT RWX 64.0 MB
            (0x0000000020EFFF, 0x1F),   // 59: NAPOT RWX 8.0 MB
            (0x00000000209FFF, 0x1F),   // 60: NAPOT RWX 16.0 MB
            (0x00000000203FFF, 0x1F),   // 61: NAPOT RWX 32.0 MB @0x80000000
            (0x000000000FFFFF, 0x1F),   // 62: NAPOT RWX 2.0 GB  @0x0
            (0x003FFFFFFFFFFF, 0x18),   // 63: NAPOT (no R/W/X) 16 EB catch-all
        ];

        for (i, (pmpaddr, cfg_val)) in entries.iter().enumerate() {
            cfg[i] = *cfg_val;
            addr[i] = *pmpaddr;
        }

        // S-mode instruction fetch at 0x8020_1108 -> covered by entry 61 [0x80000000, 0x82000000)
        assert_eq!(
            check(&cfg, &addr, 64, 0x0000_0000_8020_1108, 0, 1, /* S-mode execute */ 1, 0, 0, 0),
            1,
            "S-mode fetch at 0x80201108 should be allowed by entry 61"
        );

        // S-mode load at 0x8020_1108 -> also covered by entry 61
        assert_eq!(
            check(&cfg, &addr, 64, 0x0000_0000_8020_1108, 0, 0, /* S-mode load */ 1, 0, 0, 0),
            1,
            "S-mode load at 0x80201108 should be allowed by entry 61"
        );

        // S-mode store at 0x8020_1108 -> also covered
        assert_eq!(
            check(&cfg, &addr, 64, 0x0000_0000_8020_1108, 1, 0, /* S-mode store */ 1, 0, 0, 0),
            1,
            "S-mode store at 0x80201108 should be allowed by entry 61"
        );
    }
}
