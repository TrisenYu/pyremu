use std::cell::Cell;
use std::sync::atomic::{AtomicU8, Ordering};
use crate::diag;
use crate::state::{HartState};
use crate::concurrent::{ConcurrentClintCtx, ModuleState};


// ============================================================
//  Serial-engine CLINT inline handlers
// ============================================================
//
// Moved from handlers.rs.  These operate on the batch-engine ClintCtx
// (non-concurrent path) and are re-exported by handlers.rs for
// backward compatibility.

use crate::handlers::MemAccess;

pub(crate) fn sync_mtip(state: &mut HartState, clint: &ConcurrentClintCtx) {
    let hart_id = state.mhartid as usize;
    let cur_mtime = unsafe { &*clint.mtime }.load(Ordering::Acquire);
    let cmp = unsafe { &*clint.mtimecmp.add(hart_id) }.load(Ordering::Acquire);
    let sstc_cmp = state.stimecmp;

    if cmp > 0 && cur_mtime >= cmp {
        state.mip |= 1 << 7;
    } else {
        state.mip &= !(1 << 7);
    }

    let st_pending = (cmp > 0 && cur_mtime >= cmp) || (sstc_cmp > 0 && cur_mtime >= sstc_cmp);
    if st_pending {
        state.mip |= 1 << 5;
    } else {
        state.mip &= !(1 << 5);
    }
}

/// Synchronise ``mip.MSIP`` from the CLINT level bit.
///
/// Level-triggered design (matches real SiFive CLINT / ACLINT MSWI hardware):
/// ``mip.MSIP`` directly follows the CLINT MSIP register bit 0 (level bit).
/// When the level bit is 1, MSIP is pending; when 0, it is not.
///
/// The edge counter in bits 7:1 is maintained by ``clint_write_msip_concurrent``
/// for register-read fidelity but is NOT used for interrupt detection.  This
/// avoids the edge-collapse bug where N MSIP writes before the first
/// ``sync_msip`` call produce only one trap (lost interrupts -> TLB-shootdown
/// deadlock in OpenSBI's ``tlb_process_once`` spin).
///
/// ``mip.MSIP`` is also auto-cleared in ``deliver_trap_mmode`` (RISC-V spec
/// §3.1.15).  After the trap handler calls ``sbi_ipi_raw_clear`` (write-0 to
/// CLINT), the next ``sync_msip`` call sees level=0 and clears ``mip.MSIP``.
/// If a new MSIP arrived before the clear, level stays 1 -> re-asserted ->
/// fires after MRET (when MIE is restored).  This matches real hardware
/// behaviour where the MSIP line is continuously monitored.
#[inline]
pub(crate) fn sync_msip(state: &mut HartState, clint: &ConcurrentClintCtx) {
    let hid = state.mhartid as usize;
    // Atomically drain cross-thread MSIP notifications.
    let pending_ptr = clint.msip_pending.get();
    if pending_ptr.is_null() {
        diag::log_line(&format!(
            "[sync-msip-null] hart={} msip_pending is NULL — cross-thread MSIP delivery disabled",
            state.mhartid,
        ));
    }
    if !pending_ptr.is_null() {
        let cross = unsafe { &*pending_ptr.add(hid) }.swap(0, Ordering::Acquire);
        if cross != 0 {
            state.mip |= cross;
        }
    }
    if hid >= clint.num_harts as usize {
        return;
    }
    // Atomic test-and-clear: single fetch_and replaces load-then-fetch_and,
    // closing the TOCTOU window where a sender could write MSIP=1 between
    // the load and the clear, silently dropping the second IPI.
    let raw = unsafe { &*clint.msip.add(hid) }.fetch_and(0xFE, Ordering::AcqRel);
    let level_set = (raw & 1) != 0;

    // Level-triggered: mip.MSIP directly follows the CLINT level bit.
    // - When level=1: set mip.MSIP.  Do NOT force mie.MSIE=1 — the
    //   event-driven path (clint_write_msip_concurrent) deliberately
    //   avoids it, and wfi_sync_and_check already has an MSIP exception
    //   that wakes the hart regardless of mie.MSIE.  For interrupt
    //   delivery, the hart's MIEs govern masking (matching real hardware).
    //   The level bit was already auto-cleared by fetch_and above.
    // - When level=0: do NOT clear mip.MSIP — deliver_trap_{mmode,smode}
    //   handles that per RISC-V spec §3.1.15.
    if level_set {
        state.mip |= 1 << 3;
        // Diagnostic: count detected MSIP edges locally.
        #[cfg(feature = "diagnostic")] {
            let prev_level: u8 = (state.diag.msip_last_seen & 1) as u8;
            if prev_level == 0 {
                state.diag.clint_msip_set = state.diag.clint_msip_set.wrapping_add(1);
            }
        }
        state.diag.msip_last_seen = (state.diag.msip_last_seen & !1) | (1u64);
    }
}

// ============================================================

pub(crate) fn clint_write_msip_concurrent(clint: &ConcurrentClintCtx, target: usize, val: u8) {
    if target >= clint.num_harts as usize {
        return;
    }
    let p = unsafe { &*clint.msip.add(target) };
    if val & 1 != 0 {
        // Write-1: set the CLINT level bit AND atomically signal the
        // target hart via the msip_pending channel.  This avoids the
        // non-atomic RMW on ``(*states).mip`` which raced with the
        // target's sync_mtip/sync_msip operations on the same u64.
        p.fetch_or(1, Ordering::Release);
        let pending = clint.msip_pending.get();
        if pending.is_null() {
            diag::log_line("[msip-write-null] msip_pending pointer is NULL — cross-thread MSIP delivery disabled");
        } else {
            unsafe { &*pending.add(target) }.fetch_or(1 << 3, Ordering::Release);
        }
    } else {
        // Write-0: clear only the CLINT level bit.  Do NOT touch
        // mip.MSIP on the target — deliver_trap already cleared it
        // when the interrupt was taken, and a second sender may have
        // asserted a *new* MSIP between the trap handler and this
        // acknowledgement.  Clearing mip.MSIP here would lose it.
        p.fetch_and(0xFE, Ordering::Release);
        // sync_msip handles the mip<->level synchronisation for
        // the edge case where deliver_trap hasn't run (stale MSIP).
    }
}

/// Try to handle a CLINT MMIO access inline, concurrent-safe version.
pub(crate) fn try_handle_clint_concurrent(
    pa: u64,
    is_write: bool,
    write_data: u64,
    _state: &mut HartState,
    clint: &ConcurrentClintCtx,
    _module: &ModuleState,
) -> Option<u64> {
    if clint.base == 0 {
        return None;
    }
    let offset = pa.wrapping_sub(clint.base);

    if offset < 0x4000 {
        // MSIP region
        let target = (offset / 4) as usize;
        if target >= clint.num_harts as usize {
            return Some(0);
        }
        if !is_write {
			let val = unsafe { &*clint.msip.add(target) }.load(Ordering::Relaxed) as u64 & 1;
            return Some(val);

        }
		clint_write_msip_concurrent(clint, target, (write_data & 1) as u8);
		return Some(0);
    } else if offset < 0xBFF8 {
        // MTIMECMP region
        let target = ((offset - 0x4000) / 8) as usize;
        if target >= clint.num_harts as usize {
            return Some(0);
        }
        if is_write {
            unsafe { &*clint.mtimecmp.add(target) }.store(write_data, Ordering::Release);
            Some(0)
        } else {
            Some(unsafe { &*clint.mtimecmp.add(target) }.load(Ordering::Relaxed))
        }
    } else if offset < 0xC000 {
        // MTIME region
        if is_write {
            None // rare — fall through to Python
        } else {
            Some(unsafe { &*clint.mtime }.load(Ordering::Relaxed))
        }
    } else {
        None
    }
}

pub struct ClintCtx {
    pub base: u64,
    pub mtime: *mut u64,
    pub mtimecmp: *mut u64,
    pub msip: *mut u8,
    pub states: *mut HartState,
    pub num_harts: u32,
    /// Set when a hart writes MSIP=1 to a *different* hart.  The dispatch
    /// loop checks this flag and yields the current hart's slice early so
    /// the target hart can respond to the IPI within the same batch round.
    pub yield_for_ipi: Cell<bool>,
    /// Hart ID of the most recent cross-hart MSIP sender; this hart receives
    /// short slices so the receiver can complete IPI-triggered work (TLB
    /// flush, sync counter decrement) before the sender's spin-wait resumes.
    pub ipi_sender_hart: Cell<u8>,
    /// Remaining rounds of short slices for *ipi_sender_hart*.  Decremented
    /// once per round-robin round until zero, then the sender resumes full
    /// slices.  Reset to a fresh count each time a new MSIP is sent.
    pub ipi_sender_rounds: Cell<u8>,
}

/// Write MSIP for a target hart, updating its MIP and setting the
/// cross-hart yield flag when the target is a different hart.
#[inline]
pub(crate) fn clint_write_msip(clint: &ClintCtx, target: usize, current: usize, val: u8) {
    if target >= clint.num_harts as usize {
        return;
    }
    if clint.states.is_null() {
        unsafe {
            let atomic_msip = clint.msip as *const AtomicU8;
            (*atomic_msip.add(target)).store(val, Ordering::Release);
        }
        return;
    }
    unsafe {
        *clint.msip.add(target) = val;
    }
    let ts = unsafe { &mut *clint.states.add(target) };
    if val == 0 {
        ts.mip &= !(1 << 3);
        ts.diag.clint_msip_clr = ts.diag.clint_msip_clr.wrapping_add(1);
        return;
    }
    ts.mip |= 1 << 3;
    ts.diag.clint_msip_set = ts.diag.clint_msip_set.wrapping_add(1);
    if target != current {
        clint.yield_for_ipi.set(true);
        clint.ipi_sender_hart.set(current as u8);
        clint.ipi_sender_rounds.set(16);
    }
}

/// Try to handle a CLINT MMIO access inline inside the native batch engine.
pub(crate) fn try_handle_clint(
    access: &MemAccess, state: &HartState, clint: &ClintCtx,
) -> Option<u64> {
    let pa = access.pa;
    let is_write = access.is_write;
    let write_data = access.write_data;

    if clint.base == 0 {
        return None;
    }
    let offset = pa.wrapping_sub(clint.base);

    if offset < 0x4000 {
        let hart_id = (offset / 4) as usize;
        if hart_id >= clint.num_harts as usize {
            return Some(0);
        }
        if is_write == 0 {
            let val = unsafe { *clint.msip.add(hart_id) } as u64 & 1;
            return Some(val);
        }
        clint_write_msip(clint, hart_id, state.mhartid as usize, (write_data & 1) as u8);
        return Some(0);
    } else if offset < 0xBFF8 {
        let hart_id = usize::try_from((offset - 0x4000) / 8).unwrap_or(usize::MAX);
        if hart_id >= clint.num_harts as usize {
            return Some(0);
        }
        if is_write == 0 {
            return Some(unsafe { *clint.mtimecmp.add(hart_id) });
        }
        unsafe {
            *clint.mtimecmp.add(hart_id) = write_data;
        }
        if hart_id < clint.num_harts as usize && !clint.states.is_null() {
            let t = unsafe { &mut *clint.states.add(hart_id) };
            t.diag.clint_mtc_wr = t.diag.clint_mtc_wr.wrapping_add(1);
        }
        return Some(0);
    } else if offset < 0xC000 && is_write == 0 {
        return Some(unsafe { *clint.mtime });
    }
    None
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::HartState;
    use std::sync::atomic::{AtomicU64, AtomicU8};

    #[test]
    fn sync_msip_clears_level_bit_on_detect() {
        let mut state: HartState = unsafe { std::mem::zeroed() };
        state.mhartid = 0;
        state.mie = 1 << 3;

        let msip_byte = std::sync::atomic::AtomicU8::new(1);
        let mtime = std::sync::atomic::AtomicU64::new(0);
        let mtimecmp = std::sync::atomic::AtomicU64::new(0);

        let clint = ConcurrentClintCtx {
            base: 0x2000000,
            mtime: &mtime as *const AtomicU64,
            mtimecmp: &mtimecmp as *const AtomicU64,
            msip: &msip_byte as *const AtomicU8,
            num_harts: 1,
                states: std::ptr::null_mut(),
                msip_pending: Cell::new(std::ptr::null()),};

        assert_eq!(msip_byte.load(Ordering::Relaxed) & 1, 1);
        assert_eq!(state.mip & (1 << 3), 0);

        sync_msip(&mut state, &clint);

        assert_eq!(state.mip & (1 << 3), 1 << 3,
            "sync_msip must set mip.MSIP when level=1");
        assert_eq!(msip_byte.load(Ordering::Relaxed) & 1, 0,
            "sync_msip must auto-clear CLINT level bit after detecting MSIP");
    }

    #[test]
    fn sync_msip_preserves_mip_when_level_zero() {
        // mip.MSIP must NOT be cleared by sync_msip when CLINT level=0.
        // Clearing is the responsibility of deliver_trap_mmode /
        // deliver_trap_smode (RISC-V spec §3.1.15).  If sync_msip cleared
        // mip.MSIP here, the WFI wake path would lose the interrupt:
        //   wfi_sync_and_check -> sync_msip (auto-clears CLINT level)
        //   hart_worker -> sync_msip (sees level=0, clears mip.MSIP)
        //   check_and_deliver_interrupt_concurrent -> no trap -> deadlock.
        let mut state: HartState = unsafe { std::mem::zeroed() };
        state.mhartid = 0;
        state.mip = 1 << 3;

        let msip_byte = std::sync::atomic::AtomicU8::new(0);
        let mtime = std::sync::atomic::AtomicU64::new(0);
        let mtimecmp = std::sync::atomic::AtomicU64::new(0);

        let clint = ConcurrentClintCtx {
            base: 0x2000000,
            mtime: &mtime as *const AtomicU64,
            mtimecmp: &mtimecmp as *const AtomicU64,
            msip: &msip_byte as *const AtomicU8,
            num_harts: 1,
                states: std::ptr::null_mut(),
                msip_pending: Cell::new(std::ptr::null()),};

        sync_msip(&mut state, &clint);

        assert_eq!(state.mip & (1 << 3), 1 << 3,
            "sync_msip must NOT clear mip.MSIP when CLINT level=0 (trap delivery handles clearing)");
    }
}

