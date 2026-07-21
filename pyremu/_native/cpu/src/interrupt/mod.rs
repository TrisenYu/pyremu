pub(crate) mod clint;
pub(crate) mod wfi;

use crate::state::{HartState, riscv_mode};
use crate::trap::{mcause_val, deliver_trap, deliver_trap_mmode};
use crate::concurrent::ConcurrentClintCtx;

#[cfg(feature = "diagnostic")]
use crate::diag;

// ============================================================
//  mip / mie bit masks — used by check_pending_interrupts
// ============================================================

const MIE_MEIE: u64 = 1 << 11;  // M-mode external
const MIE_MSIE: u64 = 1 << 3;   // M-mode software
const MIE_MTIE: u64 = 1 << 7;   // M-mode timer
const MIE_SEIE: u64 = 1 << 9;   // S-mode external
const MIE_SSIE: u64 = 1 << 1;   // S-mode software
const MIE_STIE: u64 = 1 << 5;   // S-mode timer


pub(crate) fn check_pending_interrupts(state: &HartState) -> Option<(u64, bool)> {
    let pending = state.mip & state.mie;
    if pending == 0 {
        return None;
    }

    // Priority order: MEI, MSI, MTI, SEI, SSI, STI.
    // Machine-level interrupts (MEI, MSI, MTI) are NOT delegatable by
    // convention — they always trap to M-mode first.  The M-mode handler
    // may then inject a supervisor-level interrupt (SEI, SSI, STI) which
    // IS delegated.  Marking them delegatable here would cause them to be
    // silently skipped when S-mode has interrupts disabled (SIE=0),
    // leading to lost IPIs and TLB-shootdown deadlocks.
    let checks: [(u64, u64, u64); 6] = [
        (MIE_MEIE, 11, 0),
        (MIE_MSIE, 3, 0),
        (MIE_MTIE, 7, 0),
        (MIE_SEIE, 9, 1),
        (MIE_SSIE, 1, 1),
        (MIE_STIE, 5, 1),
    ];

    for (mask, cause, delegatable) in &checks {
        if pending & mask == 0 {
            continue;
        }
        if *delegatable != 0 {
            let delegated = state.mideleg & mask != 0;
            if delegated && state.mode < riscv_mode::M {
                if state.mstatus & (1 << 1) == 0 {
                    continue;
                }
                return Some((*cause, false));
            }
        }
        if state.mode < riscv_mode::M {
            return Some((*cause, true));
        }
        if state.mstatus & (1 << 3) != 0 {
            return Some((*cause, true));
        }
        #[cfg(feature = "diagnostic")]
        if *cause == 3 && (state.mip & (1 << 3)) != 0 {
            diag::log_line(&format!(
                "[mie0-msip-blocked] hart={} pc={:#018x} mode={} mstatus={:#018x} mip={:#010x} mie={:#010x}",
                state.mhartid, state.pc, state.mode, state.mstatus, state.mip, state.mie,
            ));
        }
        return None;
    }
    None
}

pub(crate) fn check_and_deliver_interrupt_concurrent(
    state: &mut HartState,
    _clint: &ConcurrentClintCtx,
) -> bool {
    if let Some((cause, is_m_mode)) = check_pending_interrupts(state) {
        let code = mcause_val(cause, true);
        let mut dummy = unsafe { std::mem::zeroed() };
        // Respect the delegation decision from check_pending_interrupts.
        // Machine-level interrupts (MSI, MTI, MEI) are marked is_m_mode=true
        // and must ALWAYS trap to M-mode, even if mideleg is set — otherwise
        // the S-mode handler receives an unexpected cause code (e.g. MSI=3
        // instead of SSI=1) and cannot process the IPI → TLB-shootdown
        // deadlock.  deliver_trap() does its own mideleg-based delegation
        // check that disagrees with check_pending_interrupts for these
        // interrupts; bypass it by calling deliver_trap_mmode directly.
        if is_m_mode {
            deliver_trap_mmode(state, code, 0, &mut dummy);
        } else {
            deliver_trap(state, code, 0, &mut dummy);
        }
        true
    } else {
        false
    }
}
