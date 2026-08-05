//! All ecall wrappers — enclave extension calls and standard SBI calls.
//! Mirrors the pattern in smode_entry/trap_handler.c + ref-emod/enclave_ops.h.

use core::arch::asm;

use crate::constants::*;

// ---------------------------------------------------------------
//  Low-level ecall primitives
// ---------------------------------------------------------------

/// Execute an ecall with ext_id, func_id, and three args.  Returns (a0, a1).
#[inline(always)]
unsafe fn ecall_3(ext_id: u64, func_id: u64, a0: u64, a1: u64, a2: u64) -> (u64, u64) {
    let ret0: u64;
    let ret1: u64;
    unsafe {
        asm!(
            "ecall",
            in("a7") ext_id,
            in("a6") func_id,
            in("a0") a0,
            in("a1") a1,
            in("a2") a2,
            lateout("a0") ret0,
            lateout("a1") ret1,
        );
    }
    (ret0, ret1)
}

// ---------------------------------------------------------------
//  Enclave extension calls  (ext_id = 0x2022_1222)
// ---------------------------------------------------------------

/// Yield to M-mode.  M-mode returns `(a0, a1, a2)` — typically
/// `(payload_pa, payload_size, argc)` during boot.
#[inline]
pub fn enclave_call_suspend(short_msg: u64) -> (u64, u64, u64) {
    let a0: u64;
    let a1: u64;
    let a2: u64;
    unsafe {
        asm!(
            "ecall",
            in("a7") ENCLAVE_EXT_ID,
            in("a6") ENCLAVE_CALL_SUSPEND,
            in("a0") short_msg,
            lateout("a0") a0,
            lateout("a1") a1,
            lateout("a2") a2,
        );
    }
    (a0, a1, a2)
}

/// Request 2 MiB memory chunks from M-mode.
/// Returns `(allocated_count, physical_address)`.
pub fn enclave_call_mem_alloc(chunk_nums: u64) -> (u64, u64) {
    unsafe { ecall_3(ENCLAVE_EXT_ID, ENCLAVE_CALL_MEM_ALLOC, 0, chunk_nums, 0) }
}

/// Query remaining pool capacity from M-mode.
/// Returns `(free_total, max_contiguous)` in units of 2 MiB partitions.
#[inline]
#[allow(dead_code)]
pub fn enclave_call_get_available_mem() -> (u64, u64) {
    unsafe { ecall_3(ENCLAVE_EXT_ID, ENCLAVE_CALL_GET_AVAILABLE_MEM, 0, 0, 0) }
}

/// Get the current enclave ID.
#[inline]
#[allow(dead_code)]
pub fn enclave_call_get_id() -> u64 {
    let ret: u64;
    unsafe {
        asm!(
            "ecall",
            in("a7") ENCLAVE_EXT_ID,
            in("a6") ENCLAVE_CALL_GET_ID,
            lateout("a0") ret,
        );
    }
    ret
}

/// Get the current hart ID.
#[inline]
#[allow(dead_code)]
pub fn enclave_call_get_hartid() -> u64 {
    let ret: u64;
    unsafe {
        asm!(
            "ecall",
            in("a7") ENCLAVE_EXT_ID,
            in("a6") ENCLAVE_CALL_GET_HARTID,
            lateout("a0") ret,
        );
    }
    ret
}

/// Destroy this enclave and return to host.
pub fn enclave_call_exit(code: u64) -> ! {
    unsafe {
        ecall_3(ENCLAVE_EXT_ID, ENCLAVE_CALL_SHUTDOWN, code, 0, 0);
    }
    crate::hang::hang()
}

/// Query M-mode for pending host requests.
/// Returns a flags bitmask: bit0=SHUTDOWN_REQUESTED.
#[inline]
pub fn enclave_call_query_requests() -> u64 {
    let ret: u64;
    unsafe {
        asm!(
            "ecall",
            in("a7") ENCLAVE_EXT_ID,
            in("a6") ENCLAVE_CALL_QUERY_REQUESTS,
            lateout("a0") ret,
        );
    }
    ret
}

/// Forward an unmatched access fault to M-mode for diagnosis.
/// Called when S-mode cannot resolve a load/store access fault (scause 5/7).
#[inline]
pub fn enclave_call_unmatched_acc_fault(stval: u64) {
    unsafe {
        ecall_3(
            ENCLAVE_EXT_ID,
            ENCLAVE_CALL_UNMATCHED_ACC_FAULT,
            stval, 0, 0,
        );
    }
}

// ---------------------------------------------------------------
//  Standard SBI calls
// ---------------------------------------------------------------

/// Output a single character via the legacy SBI console putchar interface.
#[inline]
#[allow(dead_code)]
pub fn sbi_putchar(c: u8) {
    unsafe {
        asm!(
            "ecall",
            in("a7") SBI_LEGACY_PUTCHAR_EXT,
            in("a6") 0_u64,
            in("a0") c as u64,
        );
    }
}

/// Output a byte buffer via the SBI DBCN Console Write extension.
/// The entire buffer is written atomically under M-mode's `console_out_lock`,
/// preventing per-character interleaving with concurrent output from other harts.
///
/// Falls back to legacy `sbi_putchar` per byte if DBCN is not available.
#[inline]
pub fn sbi_console_write(buf: &[u8]) {
    let pa = buf.as_ptr() as u64;
    let len = buf.len() as u64;
    let ret: u64;
    unsafe {
        asm!(
            "ecall",
            in("a7") SBI_DBCN_EXT,
            in("a6") SBI_DBCN_CONSOLE_WRITE,
            in("a0") len,
            in("a1") pa,
            in("a2") 0_u64,
            lateout("a0") ret,
        );
    }
    // DBCN returns 0 on success, negative error code on failure.
    // Fall back to character-by-character legacy putchar.
    if ret != 0 {
        for &byte in buf {
            sbi_putchar(byte);
        }
    }
}

/// Schedule a timer interrupt at `stime_value` (absolute time in ticks).
#[inline]
pub fn sbi_set_timer(stime_value: u64) {
    unsafe {
        ecall_3(SBI_TIMER_EXT, SBI_SET_TIMER_FUNC, stime_value, 0, 0);
    }
}
