// use std::time::Instant;
use std::sync::atomic::Ordering;
use crate::diag;
use crate::state::{exit_reason, HartState};
// use crate::trap::{exc_code, mcause_val, deliver_trap};
use crate::concurrent::{ConcurrentClintCtx, FfiExtIrqCtx, ModuleState, StopInfo};
use crate::interrupt::clint::{sync_mtip, sync_msip};

pub(crate) const TRAP_LOOP_THRESHOLD: u8 = 3;


/// Check whether any interrupt is pending (including MSIP via level-triggered
/// CLINT).  Returns ``(woke, msip_pending)``.
#[inline]
/// Check all interrupt sources and sync into ``mip`` before potential park.
/// QEMU equivalent: ``qemu_mutex_lock & qemu_cond_wait`` — the I/O thread
/// updates interrupt state and signals the vCPU thread.  Here the Python
/// daemon writes ``ext_irq.pending`` and we synchronise it into ``mip``.
pub(crate) fn wfi_sync_and_check(
    state: &mut HartState, clint: &ConcurrentClintCtx,
    uart_rx_notify: *const u8,
    ext_irq: *mut FfiExtIrqCtx,
) -> (bool, bool) {
    sync_mtip(state, clint);
    sync_msip(state, clint);
    // Drain ext_irq into mip inline.
    if !ext_irq.is_null() && unsafe { (*ext_irq).pending != 0 } {
        if state.mie & (1 << 9) != 0 { state.mip |= 1 << 9; }
        if state.mie & (1 << 11) != 0 { state.mip |= 1 << 11; }
    }
    let rx_ready = !uart_rx_notify.is_null() && unsafe { *uart_rx_notify != 0 };
    // RX daemon 在 drain 后立即清 _rx_notify, 但 ext_irq.pending 持续置位
    // 直到 guest 处理完中断. 此处补检 ext_irq 防止 WFI 错过唤醒.
    let ext_irq_pending = !ext_irq.is_null() && unsafe { (*ext_irq).pending != 0 };
    let msip_pending = (state.mip & (1 << 3)) != 0;
    let other_pending = (state.mip & state.mie) != 0;
    (other_pending || msip_pending || rx_ready || ext_irq_pending, msip_pending)
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
        module.wfi_count.fetch_sub(1, Ordering::Release);
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
    uart_rx_notify: *const u8,
    ext_irq: *mut FfiExtIrqCtx,
) -> bool {
    module.wfi_flags[hart_id].store(1, Ordering::Release);
    module.wfi_count.fetch_add(1, Ordering::Release);

    // Event-driven park: block the OS thread until a MSIP sender calls
    // unpark() on this thread (in clint_write_msip_concurrent).  This
    // matches QEMU's design: vCPU blocks on pthread_cond_wait, I/O thread
    // signals on data arrival.  Zero CPU usage, zero latency for IPI.
    //
    // Timer interrupts are handled by wfi_check_all_idle fast-forwarding
    // mtime before the park, so we never oversleep past a deadline.
    loop {

        // ---- sync interrupts + check wake ----
        let (woke, msip_pending) = wfi_sync_and_check(state, clint, uart_rx_notify, ext_irq);
        if woke {
            state.waiting = 0;
            state.wfi_woken = 1;
            diag::wfi_wake_reason(&mut state.diag, state.mip & state.mie, msip_pending);
            break;
        }

        // ---- stop flag (debugger pause / batch exit) ----
        if module.stop_flag.load(Ordering::Acquire) {
            sync_msip(state, clint);
            module.wfi_flags[hart_id].store(0, Ordering::Release);
            module.wfi_count.fetch_sub(1, Ordering::Release);
            return false;
        }
        if !stop_flag.is_null() && unsafe { *stop_flag != 0 } {
            sync_msip(state, clint);
            module.wfi_flags[hart_id].store(0, Ordering::Release);
            module.wfi_count.fetch_sub(1, Ordering::Release);
            return false;
        }

        // ---- all-idle detection (fast-forwards mtime to nearest deadline) ----
        if module.all_in_wfi() {
            match wfi_check_all_idle(state, hart_id, clint, module, msip_pending) {
                Some(true) => break,
                Some(false) => return false,
                None => {} // MSIP pending, keep spinning
            }
        }

        // ---- park with deadline timeout ----
        // MSIP: sender calls unpark() -> park returns immediately.
        // Timer: scan mtimecmp, park until nearest deadline.
        // No timer: park for 100 µs (responsive to rx_notify / stop_flag).
        let cur_mtime = unsafe { &*clint.mtime }.load(Ordering::Relaxed);
        let mut nearest: u64 = u64::MAX;
        for hid in 0..(clint.num_harts as usize) {
            let cmp = unsafe { &*clint.mtimecmp.add(hid) }.load(Ordering::Relaxed);
            if cmp > cur_mtime && cmp < nearest { nearest = cmp; }
        }
        let park_us: u64 = if nearest != u64::MAX {
            // ~182 µs per mtime tick, clamp [50, 500] µs.
            // 500 µs cap prevents multi-second boot slowdown from
            // accumulated park time across hundreds of WFI cycles.
            (nearest.wrapping_sub(cur_mtime).saturating_mul(182)).clamp(50, 500)
        } else {
            50 // no timer: minimal spin before re-check
        };
        std::thread::park_timeout(std::time::Duration::from_micros(park_us));
    }

    module.wfi_flags[hart_id].store(0, Ordering::Release);
    module.wfi_count.fetch_sub(1, Ordering::Release);
    true
}
