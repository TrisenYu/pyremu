use crate::concurrent::{ConcurrentClintCtx, ModuleState, StopInfo, WATCHDOG_POLL_US};
use crate::ffi::FfiExtIrqCtx;
use crate::interrupt::{clint::sync_msip, clint::sync_mtip, imsic_topei_peek, sync_imsic};
use crate::state::{exit_reason, riscv_mode, HartState};
use std::sync::atomic::Ordering;

/// 进入等待前检查全部中断源并同步到 ``mip``, 返回 ``(woke, msip_pending)``.
///
/// 中断源包括 CLINT 的定时器与电平位 MSIP、IMSIC 文件挂起的中断、终端输入输出
/// 守护线程写入的 ``ext_irq.pending``, 以及 ``uart_rx_notify`` 标志.
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

	let ext_irq_pending = !ext_irq.is_null() && unsafe { (*ext_irq).pending != 0 };
	if ext_irq_pending {
		let mfile_owns = state.imsic_m.present != 0 && state.imsic_m.eidelivery != 0;
		let sfile_owns = state.imsic_s.present != 0 && state.imsic_s.eidelivery != 0;
		if !sfile_owns {
			state.mip.fetch_or(1 << 9, Ordering::AcqRel); // SEIP 置位
		}
		if !mfile_owns {
			state.mip.fetch_or(1 << 11, Ordering::AcqRel); // MEIP 置位
		}
	}

	if state.imsic_m.present != 0
		&& (state.mip.load(Ordering::Acquire) & (1 << 11)) != 0
		&& (state.mie & (1 << 11)) == 0
	{
		state.mie |= 1 << 11; // MEIE 置位
	}
	if state.imsic_s.present != 0
		&& (state.mip.load(Ordering::Acquire) & (1 << 9)) != 0
		&& (state.mie & (1 << 9)) == 0
	{
		state.mie |= 1 << 9; // SEIE 置位
	}

	let rx_ready = !uart_rx_notify.is_null() && unsafe { *uart_rx_notify != 0 };
	let msip_pending = (state.mip.load(Ordering::Acquire) & (1 << 3)) != 0;

	// 唤醒判定不使用 mip.MEIP. IMSIC M 文件存在且 eidelivery 为 0 时, 外部中断使
	// mip.MEIP 置位后没有来源清除该位, 受调试程序经 mtopei 响应中断只清除 eip, 而强制
	// 使能又置位 MEIE; 按 mip 与 mie 判定的唤醒条件因此恒成立, WFI 无法进入等待.
	//
	// 故此处与 trap 分发路径 compute_mtopi 一致, 不采信 mip.MEIP, 改为直接查询
	// IMSIC 是否仍有挂起的中断.
	let mfile_pending = state.imsic_m.present != 0 && imsic_topei_peek(&state.imsic_m).0 != 0;
	let mut wake_mask = state.mie;
	if state.imsic_m.present != 0 {
		wake_mask &= !(1 << 11); // MEIP 排除, 改由上面的 IMSIC 直查承担
	}

	// M 模式的唤醒判定排除已委派的中断, 即 SEI、SSI、STI: 这些中断在 M 模式保持挂起,
	// M 模式陷态处理函数也不处理它们, 为其醒来只会在 WFI 与跳过之间反复, hart 降到
	// S 模式后才投递. 未委派的机器级中断 MEI、MSI、MTI 一律唤醒.
	let level_pending = if state.mode == riscv_mode::M {
		(state.mip.load(Ordering::Acquire) & wake_mask & !state.mideleg) != 0
	} else {
		(state.mip.load(Ordering::Acquire) & wake_mask) != 0
	};
	// IMSIC 有挂起的中断时, 仍要求 mie 中的 MEIE 已使能, 才作为唤醒来源.
	let imsic_pending = mfile_pending && (state.mie & (1 << 11)) != 0;
	let other_pending = level_pending || imsic_pending;
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

	// 同时扫描 stimecmp 的截止时刻; stimecmp 由 SSTC 管理, 与 mtimecmp 相互独立,
	// 两个来源都需检查.
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
		// 本 hart 已有挂起的 MSIP, 继续自旋, 直到该中断得到处理; 不得因定时器截止而退出.
		// 若在此处先走定时器分支, 挂起的 MSIP 会随加速执行退出而不再被检查, 发送方
		// 永远等待该 IPI 的确认, 形成 TLB shootdown 死锁.
		return None;
	}

	if earliest != u64::MAX {
		// 全部 hart 处于 WFI 且存在定时器截止时刻, 仅退出加速执行
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
		if unsafe { (*clint.msip.add(hid)).load(Ordering::Relaxed) } & 1 != 0 {
			any_hart_msip = true;
			break;
		}
	}
	if any_hart_msip {
		return None; // 其他 hart 的唤醒正在进行, 继续自旋
	}
	sync_msip(state, clint);
	if (state.mip.load(Ordering::Acquire) & (1 << 3)) != 0 {
		state.waiting = 0;
		state.wfi_woken = 1;
		return Some(true); // 退出 WFI 后投递陷态
	}
	module.request_stop(StopInfo {
		reason: exit_reason::WFI_WAIT,
		hart_id: hart_id as u8,
		pc: state.pc,
		..StopInfo::empty()
	});
	module.wfi_flags[hart_id].store(0, Ordering::Release);
	module.wfi_count.fetch_sub(1, Ordering::Release);
	return Some(false); // 退出加速执行
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

	// park_timeout 在 unpark 或超时之后返回, 循环体重新执行并重查停止标志. unpark 来自
	// 看门狗每轮调用的 unpark_all_harts、发送 MSIP 或写 IMSIC eip 的 hart, 以及退出的
	// hart; 超时取 WATCHDOG_POLL_US, 上述来源都不发生时由该超时使循环继续.
	loop {
		// 同步中断并检查唤醒
		let (woke, msip_pending) = wfi_sync_and_check(state, clint, uart_rx_notify, ext_irq);
		if woke {
			state.waiting = 0;
			state.wfi_woken = 1;
			break;
		}

		// 调试器暂停标志
		if module.stop_flag.load(Ordering::Acquire)
			|| !stop_flag.is_null() && unsafe { *stop_flag != 0 }
		{
			sync_msip(state, clint);
			module.wfi_flags[hart_id].store(0, Ordering::Release);
			module.wfi_count.fetch_sub(1, Ordering::Release);
			return false;
		}

		// ---- 全部 hart 空闲判定 ----
		if module.all_in_wfi() {
			match wfi_check_all_idle(state, hart_id, clint, module, msip_pending) {
				Some(true) => {
					break;
				}
				Some(false) => {
					return false;
				}
				None => {} // 仍有挂起的 MSIP, 继续自旋
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
	use crate::interrupt::{imsic_topei_claim_iid, IID_M_IPI};
	use std::cell::Cell;
	use std::sync::atomic::{AtomicU64, AtomicU8};
	use std::sync::mpsc;
	use std::sync::Arc;
	use std::time::Duration;

	/// ``wfi_check_all_idle`` 判定 ``any_hart_msip`` 时只检查 bit 0, MSIP 字节的
	/// bits 7:1 为边沿计的二进制值. 电平位为 0 而边沿计数非零时,
	/// 该函数返回 ``Some(false)``, 即退出加速执行.
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
		let msip = AtomicU8::new(0b110); // 电平位为 0, 边沿计数为 3
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
		let module = ModuleState::new(1, 1, 0, vec![0].into_boxed_slice(), false, 0, Box::new([]));

		// 本 hart 无挂起的 MSIP, 也无定时器截止时刻, 但 MSIP 字节的边沿计数非零.
		// 判定只检查电平位, 故返回 Some(false).
		let result = wfi_check_all_idle(&mut state, 0, &clint, &module, false);
		assert_eq!(
			result,
			Some(false),
			"edge counter 非零但电平位=0 时 WFI 应正常退出"
		);
	}

	/// ``wfi_spin`` 以 ``park_timeout(WATCHDOG_POLL_US)`` 阻塞,
	/// 超时后重新检查 stop_flag. 没有 unpark 时, 只有该超时使 ``wfi_spin`` 返回.
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
		// active_hart_num 取 4, 大于 wfi_count 的 1, all_in_wfi() 返回 false, 本 hart
		// 不进入全部 hart 空闲的分支, 而在 park_timeout 中阻塞, 直到超时返回.
		let module = Arc::new(ModuleState::new(
			1,
			4,
			0,
			vec![0].into_boxed_slice(),
			false,
			0,
			Box::new([]),
		));
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

		// 使 hart 进入 park_timeout, 随后只设置 stop_flag, 不调用 unpark.
		std::thread::sleep(Duration::from_millis(50));
		module.request_stop(StopInfo {
			reason: exit_reason::TIMEOUT,
			..StopInfo::empty()
		});

		// park_timeout 的超时取 WATCHDOG_POLL_US, 即 5ms; 超时后循环重查 stop_flag,
		// 本 hart 返回 false.
		match rx.recv_timeout(Duration::from_millis(500)) {
			Ok(false) => {}
			Ok(v) => panic!("stop_flag 置位后 WFI hart 应返回 false (exit), 实际 {v:?}"),
			Err(_) => panic!("stop_flag 置位后 WFI hart 未在有界时间内退出 (旧行为: 无限 park)"),
		}
	}

	/// 构造 IMSIC 存在且 M 文件与 S 文件的 eidelivery 均为 0 的 hart 状态, 用于验证
	/// WFI 唤醒判定只查询 IMSIC 的挂起状态, 不使用 ``mip.MEIP``.
	fn make_state_aia_no_mfile_delivery() -> (HartState, ConcurrentClintCtx) {
		let mut state: HartState = unsafe { std::mem::zeroed() };
		state.mhartid = 0;
		state.mode = riscv_mode::M;
		state.pc = 0x1000;
		state.mip.store(0, Ordering::Release);
		state.mie = 0;
		state.imsic_m.present = 1;
		state.imsic_m.eidelivery = 0;
		state.imsic_s.present = 1;
		state.imsic_s.eidelivery = 0;
		(state, make_clint_ctx())
	}

	/// 构造全零的 CLINT 上下文: mtime 与 mtimecmp 均为 0, MSIP 无挂起.
	fn make_clint_ctx() -> ConcurrentClintCtx {
		// ``ConcurrentClintCtx`` 只保存这三个原子变量的原始指针, 这些变量必须在上下文的
		// 使用期内保持有效, 故用 ``Box::leak`` 保留到进程结束.
		let mtime: &'static AtomicU64 = Box::leak(Box::new(AtomicU64::new(0)));
		let mtimecmp: &'static AtomicU64 = Box::leak(Box::new(AtomicU64::new(0)));
		let msip: &'static AtomicU8 = Box::leak(Box::new(AtomicU8::new(0)));
		ConcurrentClintCtx {
			base: 0x2000000,
			mtime: mtime as *const AtomicU64,
			mtimecmp: mtimecmp as *const AtomicU64,
			msip: msip as *const AtomicU8,
			timebase_hz: 0,
			num_harts: 1,
			msip_pending: Cell::new(std::ptr::null()),
			hart_threads: Cell::new(std::ptr::null()),
			hart_states: Cell::new(std::ptr::null()),
		}
	}

	#[test]
	fn wfi_meip_not_stale_after_mfile_ipi_claim_without_eidelivery() {
		let (mut state, clint) = make_state_aia_no_mfile_delivery();
		// M 文件中有一个挂起的软件中断, IID 为 3
		state.imsic_m.eip[0].store(1 << IID_M_IPI, Ordering::Relaxed);

		let (woke_before, _) =
			wfi_sync_and_check(&mut state, &clint, std::ptr::null(), std::ptr::null_mut());
		assert!(
			woke_before,
			"挂起 IPI 期间 WFI 必须醒来处理该 IPI, 唤醒判定不可过弱"
		);

		// 受调试程序读取 mtopei 响应该中断, IMSIC 的 eip 对应位随之清除
		imsic_topei_claim_iid(&mut state.imsic_m, IID_M_IPI);

		// 响应之后 IMSIC 已无挂起的中断, WFI 应当能够进入等待.
		let (woke_after, _) =
			wfi_sync_and_check(&mut state, &clint, std::ptr::null(), std::ptr::null_mut());
		assert!(
			!woke_after,
			"mtopei claim 之后 MEIP 残留 (mip=0x{:x} mie=0x{:x}) -> WFI 永远无法进入等待",
			state.mip.load(Ordering::Acquire),
			state.mie
		);
	}
}
