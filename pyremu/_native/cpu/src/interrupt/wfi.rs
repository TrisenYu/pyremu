// use std::time::Instant;
use crate::concurrent::{ConcurrentClintCtx, FfiExtIrqCtx, ModuleState, StopInfo, WATCHDOG_POLL_US};
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
	//
	// Gate on IMSIC presence — in legacy PLIC mode (imsic not present)
	// OpenSBI's PLIC irqchip has no process_hwirqs and never sets MEIE;
	// a force-enabled MEIP would fire an M-mode external trap into an
	// unprocessable irqchip (sbi_irqchip_process → SBI_ENODEV) and hang
	// the boot.  The guest's mie is authoritative there.
	if state.imsic_m.present != 0
		&& (state.mip.load(Ordering::Acquire) & (1 << 11)) != 0
		&& (state.mie & (1 << 11)) == 0
	{
		state.mie |= 1 << 11; // MEIE
	}
	if state.imsic_s.present != 0
		&& (state.mip.load(Ordering::Acquire) & (1 << 9)) != 0
		&& (state.mie & (1 << 9)) == 0
	{
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
		// 不得在此把 mtime 快进到截止时间: 快进后 Python 侧 `_wfi_ticks_until_wake`
		// 计算 remaining = 截止 - mtime = 0, `_wfi_sleep_if_idle` 的 wait 永不
		// 阻塞 -> 全 hart WFI 空闲退化为 100% CPU 忙转 (宿主卡顿 / 键盘无响应),
		// 且客机时钟以批量速度 (≈20×真实时间) 狂飙. 正确的实时事件驱动:
		// 仅退出加速执行 (WFI_WAIT), 由 Python 侧休眠 (remaining/timebase, 上限
		// _WFI_MAX_SLEEP), 睡眠期间按真实流逝时间推进 mtime (clint.tick),
		// 定时器因而按真实节奏触发. 此处保持 mtime 不变, 让 Python 计算真实剩余.
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

	// 事件驱动阻塞: 以 park_timeout 自醒阻塞 (解耦通知的安全网). 外部
	// unpark (MSIP 发送方/看门狗) 立即唤醒本 hart 获得零延迟投递; 自醒周期
	// 与看门狗轮询周期对齐 (WATCHDOG_POLL_US = 5ms), 即使看门狗已退出且
	// 最后一次 unpark 被其他路径消费, 本 hart 仍会自行醒来重新检查停止标志
	// 与中断挂起, 杜绝"带着已置位 stop_flag 永久 park"的批次终结竞态 (曾导致
	// Linux 启动间歇性停滞, run_harts 的 join 永不返回).
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
				Some(true) => {
					break;
				}
				Some(false) => {
					return false;
				}
				None => {} // MSIP pending, keep spinning
			}
		}
		std::thread::park_timeout(std::time::Duration::from_micros(WATCHDOG_POLL_US));
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
	use std::sync::mpsc;
	use std::sync::Arc;
	use std::time::Duration;

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
		let module = ModuleState::new(1, 1, 0, 0, vec![0].into_boxed_slice());

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

	/// Regression: ``wfi_spin`` 停驻的 hart 在外部 unpark 全部消失 (看门狗已按
	/// stop 退出, 最后一次 unpark 被消费) 后, 必须自行醒来重查 stop_flag 并退出.
	///
	/// 修复前 ``wfi_spin`` 使用无限 ``std::thread::park()``, 唤醒完全依赖看门狗
	/// 每 5ms 的 unpark 或退出 hart 的完成信号; 当 hart 在最后一次 unpark 之后
	/// park、且看门狗已消失时, 该 hart 会带着已置位的 stop_flag 永久睡眠,
	/// ``run_harts`` 的 join 永不返回 (Linux 启动间歇性停滞). 修复后以
	/// ``park_timeout(WATCHDOG_POLL_US)`` 自醒, 有界时间内必定重新检查 stop_flag.
	#[test]
	fn wfi_spin_exits_on_stop_without_any_unpark() {
		let mut state: HartState = unsafe { std::mem::zeroed() };
		state.mhartid = 0;
		state.mode = riscv_mode::M;
		state.pc = 0x1000;
		state.mip.store(0, Ordering::Release);
		state.mie = 0;

		let mtime = AtomicU64::new(0);
		let mtimecmp = AtomicU64::new(0);
		let msip = AtomicU8::new(0);
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
		// active_hart_num=4 > wfi_count=1 -> all_in_wfi() 为 false, 本 hart 不会走
		// all-idle 快速退出 (WFI_WAIT) 分支, 而是真正 park 等待 -> 恰好落入
		// park_timeout 自醒路径.
		let module = Arc::new(ModuleState::new(1, 4, 0, 0, vec![0].into_boxed_slice()));
		let module2 = Arc::clone(&module);
		let (tx, rx) = mpsc::channel();

		std::thread::spawn(move || {
			let s = &mut state;
			let result = wfi_spin(
				s,
				0,
				&clint,
				&module2,
				std::ptr::null(),
				std::ptr::null(),
				std::ptr::null_mut(),
			);
			let _ = tx.send(result);
		});

		// 等 hart 进入 park 等待后, 仅置位 stop_flag, 不做任何 unpark.
		std::thread::sleep(Duration::from_millis(50));
		module.request_stop(StopInfo {
			reason: exit_reason::TIMEOUT,
			..StopInfo::empty()
		});

		// 有界等待: park_timeout(5ms) 应使 hart 在 ~5ms 内自行醒来并退出.
		// 修复前 (无限 park) 此处 recv_timeout 超时 -> 回归暴露.
		match rx.recv_timeout(Duration::from_millis(500)) {
			Ok(false) => {}
			Ok(v) => panic!("stop_flag 置位后 WFI hart 应返回 false (exit), 实际 {v:?}"),
			Err(_) => panic!("stop_flag 置位后 WFI hart 未在有界时间内退出 (旧行为: 无限 park)"),
		}
	}
}
