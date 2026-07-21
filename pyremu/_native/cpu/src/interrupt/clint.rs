use std::sync::atomic::{Ordering};
use crate::diag;
use crate::state::{HartState};
// use crate::handlers::{ClintCtx, EXIT_SENTINEL};
// use crate::trap::{exc_code, mcause_val};
use crate::concurrent::{ConcurrentClintCtx, ModuleState};

pub(crate) fn sync_mtip(state: &mut HartState, clint: &ConcurrentClintCtx) {
    let hart_id = state.mhartid as usize;
    let cur_mtime = unsafe { &*clint.mtime }.load(Ordering::Relaxed);
    let cmp = unsafe { &*clint.mtimecmp.add(hart_id) }.load(Ordering::Relaxed);
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
/// ``sync_msip`` call produce only one trap (lost interrupts → TLB-shootdown
/// deadlock in OpenSBI's ``tlb_process_once`` spin).
///
/// ``mip.MSIP`` is also auto-cleared in ``deliver_trap_mmode`` (RISC-V spec
/// §3.1.15).  After the trap handler calls ``sbi_ipi_raw_clear`` (write-0 to
/// CLINT), the next ``sync_msip`` call sees level=0 and clears ``mip.MSIP``.
/// If a new MSIP arrived before the clear, level stays 1 → re-asserted →
/// fires after MRET (when MIE is restored).  This matches real hardware
/// behaviour where the MSIP line is continuously monitored.
#[inline]
pub(crate) fn sync_msip(state: &mut HartState, clint: &ConcurrentClintCtx) {
    let hid = state.mhartid as usize;
    if hid >= clint.num_harts as usize {
        return;
    }
    let raw = unsafe { &*clint.msip.add(hid) }.load(Ordering::Acquire);
    let level_set = (raw & 1) != 0;

    // Level-triggered: mip.MSIP directly follows the CLINT level bit.
    // - When level=1: set mip.MSIP, force mie.MSIE=1 (WFI wake needs mip&mie≠0),
    //   auto-clear the CLINT level bit, and increment the diagnostic edge counter.
    // - When level=0: do NOT clear mip.MSIP — deliver_trap_{mmode,smode}
    //   handles that per RISC-V spec §3.1.15.
    //
    // Sender sends MSIP with a *single* atomic ``fetch_or(1)`` — no separate
    // edge-counter increment.  This eliminates a race window where the
    // receiver's auto-clear (``fetch_and(0xFE)``) could interleave between
    // the sender's level-set and edge-increment, silently dropping a second
    // MSIP stored to the same target.
    if level_set {
        state.mip |= 1 << 3;
        if (state.mie & (1 << 3)) == 0 {
            state.mie |= 1 << 3;
        }
        // Diagnostic: count detected MSIP edges locally.
        #[cfg(feature = "diagnostic")] {
			let prev_level: u8 = (state.diag.msip_last_seen & 1) as u8;
		}
        state.diag.msip_last_seen = (state.diag.msip_last_seen & !1) | (1u64);
		#[cfg(feature = "diagnostic")]
		if prev_level == 0 {
			state.diag.clint_msip_set = state.diag.clint_msip_set.wrapping_add(1);
        }
        // Hardware auto-clear: acknowledge the MSIP source so it doesn't
        // re-trigger after deliver_trap_mmode clears mip.MSIP.
        unsafe { &*clint.msip.add(hid) }.fetch_and(0xFE, Ordering::Release);
    }
}

// ============================================================

pub(crate) fn clint_write_msip_concurrent(clint: &ConcurrentClintCtx, target: usize, val: u8) {
    if target >= clint.num_harts as usize {
        return;
    }
    let p = unsafe { &*clint.msip.add(target) };
    if val & 1 != 0 {
        // Write-1: atomically set the level bit.  Single atomic operation
        // — no race with the receiver's auto-clear.  The previous two-step
        // sequence (fetch_or(1) + fetch_add(2)) had a window where the
        // receiver's fetch_and(0xFE) could interleave and clear a level bit
        // that was just set by a second sender, silently dropping MSIPs.
        p.fetch_or(1, Ordering::Release);
    } else {
        // Write-0: clear level while preserving edge counter.
        p.fetch_and(0xFE, Ordering::Release);
    }
}

/// Try to handle a CLINT MMIO access inline, concurrent-safe version.
pub(crate) fn try_handle_clint_concurrent(
    pa: u64,
    is_write: bool,
    write_data: u64,
    state: &mut HartState,
    clint: &ConcurrentClintCtx,
    module: &ModuleState,
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
        if is_write {
            clint_write_msip_concurrent(clint, target, (write_data & 1) as u8);
            diag::clint_msip_write(
                &mut state.diag,
                target as u64,
                state.mhartid,
                (write_data & 1) != 0,
            );
            // If the write sets MSIP (val & 1 == 1), wake a parked
            // WFI thread on the target hart so it sees the edge
            // immediately (QEMU-style thread unpark).
            if (write_data & 1) != 0 {
                if let Ok(slot) = module.wfi_threads[target].lock() {
                    if let Some(t) = slot.as_ref() {
                        t.unpark();
                    }
                }
            }
            Some(0)
        } else {
            let val = unsafe { &*clint.msip.add(target) }.load(Ordering::Relaxed) as u64 & 1;
            Some(val)
        }
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
        };

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
        };

        sync_msip(&mut state, &clint);

        assert_eq!(state.mip & (1 << 3), 1 << 3,
            "sync_msip must NOT clear mip.MSIP when CLINT level=0 (trap delivery handles clearing)");
    }
}
