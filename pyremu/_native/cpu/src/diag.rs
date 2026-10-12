//! Diagnostic counter helpers for the concurrent execution engine.
//!
//! Every function body is gated behind ``#[cfg(feature = "diagnostic")]``.
//! When the feature is disabled, each function compiles to a no-op — the
//! call sites stay clean (no ``#[cfg]`` at every caller) and the compiler
//! eliminates the dead stores.
use crate::state::{HartDiag};


/// Track the reason for exiting the WFI spin loop.
#[inline(always)]
#[allow(unused)]
#[allow(dead_code)]
pub fn wfi_wake_reason(_diag: &mut HartDiag, _pending: u64, _msip_edge: bool) {
	#[cfg(feature = "diagnostic")]
	{
		if _msip_edge || ((_pending >> 3) & 1) != 0 {
			_diag.wfi_wake_msip = _diag.wfi_wake_msip.wrapping_add(1);
		} else if ((_pending >> 7) & 1) != 0 {
			_diag.wfi_wake_mtip = _diag.wfi_wake_mtip.wrapping_add(1);
		} else {
			_diag.wfi_wake_other = _diag.wfi_wake_other.wrapping_add(1);
		}
	}
}

/// Track that MSIE was forced-set because MSIP was pending but masked.
#[inline(always)]
#[allow(dead_code)]
#[allow(unused)]
pub fn msie_forced(_diag: &mut HartDiag) {
	#[cfg(feature = "diagnostic")]
	{
		_diag.msip_masked_by_msie = _diag.msip_masked_by_msie.wrapping_add(1);
	}
}

/// Track the PC where MSIE was explicitly cleared (CSR write).
#[inline(always)]
#[allow(dead_code)]
#[allow(unused)]
pub fn msie_cleared_at(_diag: &mut HartDiag, _pc: u64) {
	#[cfg(feature = "diagnostic")]
	{
		_diag.msie_cleared_at_pc = _pc;
	}
}
