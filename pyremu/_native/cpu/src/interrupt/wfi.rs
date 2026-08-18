// use std::time::Instant;
use crate::concurrent::{ConcurrentClintCtx, FfiExtIrqCtx, ModuleState, StopInfo};
use crate::interrupt::{clint::sync_msip, clint::sync_mtip, sync_imsic};
use crate::state::{exit_reason, riscv_mode, HartState};
use std::sync::atomic::Ordering;

/// Check whether any interrupt is pending (including MSIP via level-triggered
/// CLINT).  Returns ``(woke, msip_pending)``.
/// Check all interrupt sources and sync into ``mip`` before potential park.
/// QEMU equivalent: ``qemu_mutex_lock & qemu_cond_wait`` — the I/O thread
/// updates interrupt state and signals the vCPU thread.  Here the Python
/// daemon writes ``ext_irq.pending`` and we synchronise it into ``mip``.
#[inline]
pub(crate) fn wfi_sync_and_check(
	state: &mut HartState,
	clint: &ConcurrentClintCtx,
	uart_rx_notify: *const u8,
	ext_irq: *mut FfiExtIrqCtx,
) -> (bool, bool) {
	sync_mtip(state, clint);
	sync_msip(state, clint);
	sync_imsic(state);

	// Drain ext_irq into mip — but only for lines NOT already managed by
	// IMSIC.  When eidelivery=1 the IMSIC file owns that external interrupt
	// line; sync_imsic drives it from eip & eie.  If we set it here too,
	// step_interrupts' sync_imsic call would clear it (no IMSIC pending),
	// silently losing the legacy-raised device interrupt.
	// Must run BEFORE the force-enable below so the newly-set MEIP/SEIP
	// bits are visible when we check whether to force-enable MEIE/SEIE.
	let ext_irq_pending = !ext_irq.is_null() && unsafe { (*ext_irq).pending != 0 };
	if ext_irq_pending {
		let mfile_owns = state.imsic_m.present != 0 && state.imsic_m.eidelivery != 0;
		let sfile_owns = state.imsic_s.present != 0 && state.imsic_s.eidelivery != 0;
		if !sfile_owns {
			state.mip.fetch_or(1 << 9, Ordering::AcqRel); // SEIP
		}
		if !mfile_owns {
			state.mip.fetch_or(1 << 11, Ordering::AcqRel); // MEIP
		}
	}

	// Force-enable MEIE/SEIE so that IMSIC/PLIC interrupts (which route
	// through MEIP/SEIP) are visible to the wake check below.
	// step_interrupts does the same force-enable for the non-WFI path,
	// but wfi_sync_and_check runs outside step_interrupts and must be
	// self-contained.  Without this, a hart in WFI never wakes for an
	// IMSIC IPI because OpenSBI sets mie.MSIE (not mie.MEIE) and
	// (mip & mie) evaluates to 0 even though MEIP=1.
	if (state.mip.load(Ordering::Acquire) & (1 << 11)) != 0 && (state.mie & (1 << 11)) == 0 {
		state.mie |= 1 << 11; // MEIE
	}
	if (state.mip.load(Ordering::Acquire) & (1 << 9)) != 0 && (state.mie & (1 << 9)) == 0 {
		state.mie |= 1 << 9; // SEIE
	}

	let rx_ready = !uart_rx_notify.is_null() && unsafe { *uart_rx_notify != 0 };
	let msip_pending = (state.mip.load(Ordering::Acquire) & (1 << 3)) != 0;

	// In M-mode, exclude delegated (S-level) interrupts from the wake
	// check.  Delegated interrupts (SEI, SSI, STI) are invisible to
	// mtopi and cannot be handled by the M-mode trap handler, so waking
	// for them creates an infinite WFI→wake→skip→WFI loop.  They will
	// be delivered when the hart drops to S-mode.
	// Non-delegatable M-level interrupts (MEI, MSI, MTI) always wake.
	let other_pending = if state.mode == riscv_mode::M {
		(state.mip.load(Ordering::Acquire) & state.mie & !state.mideleg) != 0
	} else {
		(state.mip.load(Ordering::Acquire) & state.mie) != 0
	};
	let woke = other_pending || msip_pending || rx_ready || ext_irq_pending;
	(woke, msip_pending)
}

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

	// Also scan stimecmp (SSTC) deadlines.  stimecmp is managed independently
	// of mtimecmp (matching QEMU / real hardware), so we must check both.
	let hart_states_ptr = clint.hart_states.get();
	if !hart_states_ptr.is_null() {
		for hid in 0..(clint.num_harts as usize) {
			let st = unsafe { &*hart_states_ptr.add(hid) };
			if st.stimecmp > 0 && st.stimecmp > cur_mtime && st.stimecmp < earliest {
				earliest = st.stimecmp;
			}
		}
	}

	if msip_pending {
		// 本 hart 已有挂起的 MSIP — 继续自旋等待投递, 不得因定时器快进而退出.
		// 若在此处先走定时器快进分支, 挂起的 MSIP 会被丢弃 (加速执行退出),
		// 发送方将永远自旋在 OpenSBI tlb_sync 等待确认 -> TLB-shootdown 死锁.
		return None;
	}

	if earliest != u64::MAX {
		// 全部 hart WFI 且有定时器截止时间。
		// 快进 mtime 到截止时间, 使 sync_mtip 在下轮加速执行的入口
		// 当检测到 mtime >= stimecmp 则置位 STIP 以唤醒 hart.
		// advance_clock_source 按真实流逝时间推进 mtime
		// 全部 vCPU 空闲时需要虚拟时钟直接跳到下一个定时器事件
		unsafe { &*clint.mtime }.store(earliest, Ordering::Release);
		module.request_stop(StopInfo {
			reason: exit_reason::WFI_WAIT,
			hart_id: hart_id as u8,
			pc: state.pc,
			..StopInfo::empty()
		});
		module.wfi_flags[hart_id].store(0, Ordering::Release);
		module.wfi_count.fetch_sub(1, Ordering::Release);
		return Some(false);
	}

	let mut any_hart_msip = false;
	for hid in 0..(clint.num_harts as usize) {
		// 仅检查电平位 (bit 0). 该字节还携带 Python 侧 edge counter (bits 7:1),
		// 一旦首个 IPI 过后 edge counter 恒非零, ``byte != 0`` 会永远误判为
		// 有挂起 MSIP -> 所有 WFI hart 永远无法退出加速执行 (活锁).
		if unsafe { (*clint.msip.add(hid)).load(Ordering::Relaxed) } & 1 != 0 {
			any_hart_msip = true;
			break;
		}
	}
	if any_hart_msip {
		return None; // wake in flight for another hart — keep spinning
	}
	// After sync_msip, re-check: an MSIP may have arrived between
	// the caller's wfi_sync_and_check and this point.
	// Exit the speedup would discard the trap and leave the sender spinning in
	// OpenSBI waiting for acknowledgment ->TLB-shootdown deadlock.
	sync_msip(state, clint);
	if (state.mip.load(Ordering::Acquire) & (1 << 3)) != 0 {
		state.waiting = 0;
		state.wfi_woken = 1;
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
	return Some(false); // exit
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

	// 事件驱动阻塞: 阻塞 OS 线程直到被 unpark 唤醒 (解耦通知). 零 CPU
	// 占用, IPI 零延迟. 定时器截止由软件看门狗线程周期性 unpark 本 hart;
	// 本函数自身不做任何 park_timeout 轮询.
	loop {
		// ---- sync interrupts + check wake ----
		let (woke, msip_pending) = wfi_sync_and_check(state, clint, uart_rx_notify, ext_irq);
		if woke {
			state.waiting = 0;
			state.wfi_woken = 1;
			break;
		}

		// ---- stop flag (debugger pause) ----
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
		std::thread::park();
	}

	module.wfi_flags[hart_id].store(0, Ordering::Release);
	module.wfi_count.fetch_sub(1, Ordering::Release);
	true
}

#[cfg(test)]
mod tests {
	use super::*;
	use std::cell::Cell;
	use std::sync::atomic::{AtomicU64, AtomicU8};

	/// Regression: ``wfi_check_all_idle`` 判定 ``any_hart_msip`` 时必须只看电平位
	/// (bit 0). MSIP 字节还携带 Python 侧 edge counter (bits 7:1), 首个 IPI 之后
	/// 恒非零; 若以 ``byte != 0`` 判定, 会误判为有挂起 MSIP -> 所有 WFI hart 永远
	/// 无法退出加速执行 (活锁).
	#[test]
	fn wfi_check_all_idle_ignores_edge_counter_bits() {
		let mut state: HartState = unsafe { std::mem::zeroed() };
		state.mhartid = 0;
		state.mode = riscv_mode::M;
		state.pc = 0x1000;
		state.mip.store(0, Ordering::Release);
		state.mie = 0;

		let mtime = AtomicU64::new(0);
		let mtimecmp = AtomicU64::new(0);
		let msip = AtomicU8::new(0b110); // level=0, edge counter=3
		let clint = ConcurrentClintCtx {
			base: 0x2000000,
			mtime: &mtime as *const AtomicU64,
			mtimecmp: &mtimecmp as *const AtomicU64,
			msip: &msip as *const AtomicU8,
			timebase_hz: 0,
			num_harts: 1,
			msip_pending: Cell::new(std::ptr::null()),
			hart_threads: Cell::new(std::ptr::null()),
			hart_states: Cell::new(std::ptr::null()),
		};
		let module = ModuleState::new(1, 1, 0, 0, vec![0]);

		// msip_pending=false (本 hart 无挂起 MSIP), 无定时器截止, 但 MSIP 字节的
		// edge counter 非零. 修复后只看电平位 -> 正常退出 Some(false);
		// 修复前 byte != 0 -> 误判 any_hart_msip -> 返回 None (活锁).
		let result = wfi_check_all_idle(&mut state, 0, &clint, &module, false);
		assert_eq!(
			result,
			Some(false),
			"edge counter 非零但电平位=0 时 WFI 应正常退出"
		);
	}
}
