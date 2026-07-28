// use std::time::Instant;
use std::sync::atomic::Ordering;
use crate::diag;
use crate::state::{exit_reason, HartState};
// use crate::trap::{exc_code, mcause_val, deliver_trap};
use crate::concurrent::{ConcurrentClintCtx, ModuleState, StopInfo};
use crate::interrupt::clint::{sync_mtip, sync_msip};

pub(crate) const TRAP_LOOP_THRESHOLD: u8 = 3;


/// Check whether any interrupt is pending (including MSIP via level-triggered
/// CLINT).  Returns ``(woke, msip_pending)``.
#[inline]
pub(crate) fn wfi_sync_and_check(state: &mut HartState, clint: &ConcurrentClintCtx) -> (bool, bool) {
    sync_mtip(state, clint);
    sync_msip(state, clint);
    // WFI wake-up: any enabled interrupt (mip & mie), OR MSIP pending
    // even when mie.MSIE=0.  MSIP is used by OpenSBI for cross-hart TLB
    // shootdown / IPI — the receiver may have interrupts disabled (mie=0
    // in cpu_do_idle) yet must still wake to acknowledge the request.
    // Python's try_wfi_wakeup has the same MSIP exception.
    let msip_pending = (state.mip & (1 << 3)) != 0;
    let other_pending = (state.mip & state.mie) != 0;
    (other_pending || msip_pending, msip_pending)
}

/// All-idle check: if there's a pending timer deadline, fast-forward
/// mtime toward it by at most ``MTIME_FFWD_CAP`` ticks.  If truly idle
/// (no timer, no MSIP), request batch stop.
///
/// Without a cap, a single fast-forward can jump mtime by billions of
/// ticks (e.g. the kernel's ``deferred_probe_timeout=10`` sets a
/// 10-second timer ->100M ticks at 10 MHz).  The kernel sees this as a
/// multi-second wall-clock jump — kernel log timestamps go from 0.6 s
/// to 5766 s between adjacent printks.
///
/// The cap is 100 000 ticks = 10 ms at the standard 10 MHz timebase.
/// This matches a single scheduler-tick quantum (HZ=100).  When the
/// deadline is farther than the cap, we only advance partway; the next
/// batch iteration will advance again.  The total number of batch
/// round-trips is bounded because each batch also executes instructions
/// that advance mtime (MTIME_DIVISOR throttle in hart_sched.rs).
const MTIME_FFWD_CAP: u64 = 100_000;

pub(crate) fn wfi_check_all_idle(
    state: &mut HartState,
    hart_id: usize,
    clint: &ConcurrentClintCtx,
    module: &ModuleState,
    msip_pending: bool,
) -> Option<bool> {
    let cur_mtime = unsafe { &*clint.mtime }.load(Ordering::Relaxed);
    let mut earliest: u64 = u64::MAX;

    for hid in 0..(clint.num_harts as usize) {
        let cmp = unsafe { &*clint.mtimecmp.add(hid) }.load(Ordering::Relaxed);
        if cmp > 0 && cmp > cur_mtime && cmp < earliest {
            earliest = cmp;
        }
    }

    if earliest != u64::MAX {
        // Cap the fast-forward so a single batch never jumps mtime by
        // more than one scheduler-tick quantum.  This prevents
        // multi-second time jumps when the kernel sets long timeouts
        // (deferred probe, RCU stall detection, watchdog, …).
        let target = if earliest - cur_mtime > MTIME_FFWD_CAP {
            cur_mtime + MTIME_FFWD_CAP
        } else {
            earliest
        };
        unsafe { &*clint.mtime }.store(target, Ordering::Release);
        sync_mtip(state, clint);
        if (state.mip & state.mie) != 0 {
            state.waiting = 0;
            state.wfi_woken = 1;
            return Some(true);
        }
    }

    if !msip_pending {
        // No timer and no pending MSIP — truly idle.
        // After sync_msip, re-check: an MSIP may have arrived between
        // the caller's wfi_sync_and_check and this point.  Exiting the
        // batch would discard the trap and leave the sender spinning in
        // OpenSBI waiting for acknowledgment ->TLB-shootdown deadlock.
        sync_msip(state, clint);
        if (state.mip & (1 << 3)) != 0 {
            state.waiting = 0;
            state.wfi_woken = 1;
            diag::wfi_wake_reason(&mut state.diag, state.mip & state.mie, true);
            return Some(true); // wake — trap will be delivered on re-entry
        }
        module.request_stop(StopInfo {
            reason: exit_reason::WFI_WAIT,
            hart_id: hart_id as u8,
            pc: state.pc,
            ..StopInfo::empty()
        });
        module.wfi_flags[hart_id].store(0, Ordering::Release);
        module.wfi_count.fetch_sub(1, Ordering::Relaxed);
        return Some(false); // exit batch
    }
    // MSIP pending but no timer — keep spinning.
    None
}

pub(crate) fn wfi_spin(
    state: &mut HartState,
    hart_id: usize,
    clint: &ConcurrentClintCtx,
    module: &ModuleState,
    stop_flag: *const u8,
) -> bool {
    module.wfi_flags[hart_id].store(1, Ordering::Release);
    module.wfi_count.fetch_add(1, Ordering::Relaxed);

    // Multi-stage back-off (QEMU-style):
    //  1. Hot-spin  (64 iter)  — catch IPI right after WFI, no syscall
    //  2. Yield     (up to 512) — let sender thread run
    //  3. Park      (> 512)    — park_timeout with thread handle registered
    //     so clint_write_msip_concurrent -> unpark() wakes us immediately.
    //     Timeout tightens from 100 µs to 1 ms after 2k iterations.
    //
    // No fixed-iteration safety valve that exits the batch — the
    // sender is in the same batch and must be given time to reach
    // its CLINT MSIP store.  Batch exit only happens via all-idle
    // detection or external stop flag.
    let mut spin_count: u64 = 0;
    loop {
        spin_count += 1;

        // ---- sync interrupts + check wake ----
        let (woke, msip_pending) = wfi_sync_and_check(state, clint);
        if woke {
            state.waiting = 0;
            state.wfi_woken = 1;
            diag::wfi_wake_reason(&mut state.diag, state.mip & state.mie, msip_pending);
            break;
        }

        // ---- stop flag ----
        if module.stop_flag.load(Ordering::Acquire) {
            sync_msip(state, clint);
            module.wfi_flags[hart_id].store(0, Ordering::Release);
            module.wfi_count.fetch_sub(1, Ordering::Relaxed);
            return false;
        }
        if !stop_flag.is_null() && unsafe { *stop_flag != 0 } {
            sync_msip(state, clint);
            module.wfi_flags[hart_id].store(0, Ordering::Release);
            module.wfi_count.fetch_sub(1, Ordering::Relaxed);
            return false;
        }

        // ---- all-idle detection ----
        if module.all_in_wfi() {
            match wfi_check_all_idle(state, hart_id, clint, module, msip_pending) {
                Some(true) => break,
                Some(false) => return false,
                None => {} // re-check after back-off
            }
        }

        // ---- staged back-off ----
        if spin_count < 64 {
            std::hint::spin_loop();
        } else if spin_count < 512 {
            std::thread::yield_now();
        } else {
            // Park with timeout — clint_write_msip_concurrent calls
            // unpark() to wake us immediately when an MSIP arrives.
            // We register the thread handle BEFORE parking so that
            // unpark() works for BOTH the light-sleep (100 µs) and
            // deep-park (1 ms) phases — the sender doesn't know which
            // stage we're in and must be able to interrupt us in any
            // case.  Without this, light-sleep used std::thread::sleep
            // which ignores unpark(), adding up to ~150 ms of latency
            // per TLB shootdown.
            //
            // The mutex is per-hart (wfi_threads[hart_id]) and only
            // contended when the sender writes to the same CLINT MSIP
            // concurrently — a single lock/unlock pair is ~10 ns.
            {
                let mut slot = module.wfi_threads[hart_id].lock().unwrap();
                *slot = Some(std::thread::current());
            }
            let timeout = if spin_count < 2000 {
                std::time::Duration::from_micros(100)
            } else {
                std::time::Duration::from_millis(1)
            };
            std::thread::park_timeout(timeout);
            {
                let mut slot = module.wfi_threads[hart_id].lock().unwrap();
                *slot = None;
            }
        }
    }

    module.wfi_flags[hart_id].store(0, Ordering::Release);
    module.wfi_count.fetch_sub(1, Ordering::Relaxed);
    true
}
