//! Physical Memory Protection (PMP) check — pure function, all state passed by FFI.
//!
//! RISC-V Privileged Spec §3.7.  Supports TOR, NA4, and NAPOT matching modes.
//! Handles M-mode bypass (MPRV-aware), enclave pmpsplit partitioning, and
//! the standard priority-based matching rules.

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

/// Decode a NAPOT-encoded pmpaddr value → (base, size).
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
        } else {
            // M-mode with MPRV=0: PMP does not apply
            return 1;
        }
    }

    // ---- No entries → S/U mode always denied (RISC-V spec) ----
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

    // ---- No matching entry → M-mode allows, S/U denies ----
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
        // No trailing 1 → 8 bytes
        let (base, size) = decode_napot(0x0000_0000_0000_0000);
        assert_eq!(base, 0);
        assert_eq!(size, 8);
    }

    #[test]
    fn test_decode_napot_4k() {
        // 8 trailing ones → 2^(8+3) = 2^11 = 2048 ... wait, trailing ones on pmpaddr.
        // pmpaddr with 8 trailing ones: size = 2^(8+3) = 2^11 = 2048 bytes.
        // But pmpaddr is shifted right by 2, so the region is actually 2048 bytes.
        // Wait, let me re-read the spec.
        // NAPOT encoding: pmpaddr = (base >> 2) | ((size-1) >> 3)
        // The trailing ones count gives us k, then size = 2^(k+3).
        // 8 trailing ones → k=8 → size=2^11=2048.
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
}
