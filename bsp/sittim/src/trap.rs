//! 陷态分发。

use crate::constants::*;
use crate::csr;
use crate::ecall_aux;
use crate::ext_mod::table;
use crate::hang;
use crate::syscall;
use crate::syscall::concurrency::{proc, sched, thread, timer};
use crate::trap_frame::TrapGprs;

// ---------------------------------------------------------------
//  C ABI 入口
// ---------------------------------------------------------------

/// entry.s调用。通过csrr设置函数参数。
#[unsafe(no_mangle)]
pub unsafe extern "C" fn trap_dispatch(
	gprs: &mut TrapGprs,
	sepc: u64,
	scause: u64,
	stval: u64,
) -> u64 {
	// 返回值约定: 0 表示不切换线程; 非 0 为目标线程陷态栈帧基址,
	// entry.s 据此 mv sp, a0 换到目标帧后走统一的恢复与 sret 路径.
	// sstatus.SPP 取 0 表示本次陷阱来自 U 模式, 取 1 表示来自 S 模式.
	let is_from_user = (csr::read_sstatus() & csr::SSTATUS_SPP) == 0;
	if is_from_user {
		// 陷阱来自 U 模式时, 为当前线程记下 U 模式停在哪条指令上: ecall 记 ecall 之后的那条
		// 指令, 其余记被中断的那条指令. S 模式之内发生的陷阱会覆盖 sepc, 故模块在运行中出错
		// 时, 被放弃的系统调用是哪一个、失败结果写回哪里, 只能取自这一条记录
		let resume = if scause == 0x8 {
			thread::TrapResume::Ecall { next: sepc + 4 }
		} else {
			thread::TrapResume::Interrupted { pc: sepc }
		};
		thread::note_trap_resume(resume);
	}

	// 中断
	if (scause & (1 << 63)) != 0 {
		let cause = InterruptCause::from_code(scause & !(1 << 63));
		interrupt_dispatch(cause);
		// 信号投递: 定时器到期置入的 SIGALRM 即在此投递, 恢复点取被中断的 pc.
		// 默认处置为终止时本进程就此终结并登记切换目标, 故投递先于切换检查, 使
		// 两种来源 (中断自身与信号投递) 的切换都在下方统一处理.
		let next_sepc = proc::deliver_pending(gprs, sepc).unwrap_or(sepc);
		// 有已登记的切换目标则记录当前线程现场并激活目标线程, 返回目标帧基址
		// 交给 entry.s 换栈.
		if thread::switch_pending() {
			return thread::finalize_switch(next_sepc);
		}
		csr::write_sepc(next_sepc);
		return 0;
	}

	// ECALL from U-mode
	if scause == 0x8 {
		let ret = syscall_dispatch(gprs);
		// rt_sigreturn 恢复的现场覆盖整个陷态栈帧 (含 a0), 且恢复点是被中断的 pc;
		// 阻塞后重启的调用要退回 ecall 本身重新执行, 也不写回返回值. 两者都在
		// 写回 a0 与推进 sepc 之前取走.
		let resume_sepc = match proc::take_sigreturn() {
			Some(sigreturn_sepc) => sigreturn_sepc,
			None if thread::take_restart() => sepc,
			None => {
				gprs.set_a0(ret);
				sepc + 4
			}
		};
		// 有用户处理函数则改为以处理函数地址恢复, 使返回时先进入处理
		// 函数; 默认处置为终止时本进程就此终结并登记切换目标, 故投递先于切换检查.
		let next_sepc = proc::deliver_pending(gprs, resume_sepc).unwrap_or(resume_sepc);
		// syscall 与信号投递都可能已登记线程切换 (clone / futex 阻塞 / exit /
		// 让出 / 信号终止): 有则记录当前线程现场并激活目标线程.
		if thread::switch_pending() {
			return thread::finalize_switch(next_sepc);
		}
		csr::write_sepc(next_sepc);
		return 0;
	}

	if is_unmatched_access_fault(scause) {
		// M 模式加载该地址所属的区域之后方可重试原访问. 加载无法使该次访问成立时
		// 返回非 0, 落到下方按故障来源处置: U 模式的故障终止该进程, 其余按模块归属
		// 或管理程序自身的缺陷处置.
		if ecall_aux::enclave_call_unmatched_acc_fault(stval, scause) == 0 {
			return 0;
		}
	}

	// 用户态 不可恢复的同步异常: 终止该进程, 退出码取 128+sig
	if is_from_user {
		proc::exit_process(proc::current_slot(), 128 + user_fault_signal(scause));
		// 进程层已登记切换目标;
		// 无其它可运行线程时终止飞地且不返回
		if thread::switch_pending() {
			return thread::finalize_switch(sepc);
		}
		return 0;
	}

	// 故障指令落在模块映像的窗口之内时, 该次故障归属到该模块: 先按 sepc 判定, sepc 不在
	// 任何窗口之内时再按 stval 判定。清空该模块的操作表缓存并把
	// 故障交回用户态; 当前执行流不来自 U 模式时该次故障没有
	// 可交回的可信应用, 落到下方按管理程序自身的缺陷挂起。
	if let Some(module_id) = table::module_of(sepc, stval) {
		if let Some(target_frame) = proc::deliver_module_fault(module_id) {
			return target_frame;
		}
	}

	// 管理程序自身的同步缺陷, 或无法归属到任何模块的同步异常 —— 输出原因、scause / stval /
	// sepc 与本次陷态携带的陷态栈帧后冻结挂起, 但仍响应宿主发来的终止请求。
	hang::fault_halt_exc(gprs, scause, stval, sepc);
}

/// 将 U 模式同步异常 scause 换算为进程收到该异常时的默认信号编号 (Linux 语义),
/// 使自毁退出码 128+sig 与真实 Linux 下同缺陷进程的退出码一致 (如 NULL 解引用
/// 触发页错误, SIGSEGV=11, 退出码 139), 便于可信应用管理器按退出码推断故障类型。
fn user_fault_signal(cause: u64) -> u64 {
	match cause {
		// 取指/访存地址错位
		0x0 | 0x4 | 0x6 => 7, // SIGBUS
		// 非法指令
		0x2 => 4, // SIGILL
		// 断点
		0x3 => 5, // SIGTRAP
		// 访问错误与三类页错误 (取指/读/写) 归并到段错误
		_ => 11, // SIGSEGV
	}
}

/// 根据 scause 返回可读原因字符串
///
/// 独立成函数便于 fault_halt 现场打印与后续诊断复用。
pub fn get_trap_reason(cause: u64) -> &'static str {
	match cause {
		0x0 => "instruction address misaligned",
		0x1 => "instruction access fault",
		0x2 => "illegal instruction",
		0x3 => "breakpoint",
		0x4 => "load address misaligned",
		0x6 => "store/amo address misaligned",
		0x9 => "ecall from S-mode",
		0xb => "ecall from M-mode",
		0xc => "instruction page fault",
		0xd => "load page fault",
		0xf => "store/amo page fault",
		_ => "unknown exception",
	}
}

// ---------------------------------------------------------------
//  系统调用, 后续可按需扩展 syscall 号
// ---------------------------------------------------------------

fn syscall_dispatch(gprs: &mut TrapGprs) -> u64 {
	syscall::syscall_handler(gprs)
}

// ---------------------------------------------------------------
//  中断, 当前仅 timer
// ---------------------------------------------------------------

/// S 模式中断的原因编号, 取 `scause` 去掉最高位后的值。
enum InterruptCause {
	/// 监督级软件中断 (SSI): M 模式注入 `sip.SSIP`, 通知有 host 请求待处理。
	SupervisorSoftware = 1,
	/// 监督级定时器中断 (STI): 定时器截止时间到达。
	SupervisorTimer = 5,
}

impl InterruptCause {
	/// 由中断原因编号取原因。本运行时不处理的中断返回 None。
	fn from_code(code: u64) -> Option<Self> {
		match code {
			1 => Some(Self::SupervisorSoftware),
			5 => Some(Self::SupervisorTimer),
			_ => None,
		}
	}
}

/// 处理一次 S 模式中断。
///
/// 两个已处理的中断都可能使线程变为可运行 —— SSI 处理网卡事件时唤醒等待接收
/// 的线程, STI 既切时间片又使定时等待到期 —— 故服务完在存在其它可运行线程时
/// 登记轮转抢占。
fn interrupt_dispatch(cause: Option<InterruptCause>) {
	match cause {
		Some(InterruptCause::SupervisorSoftware) => {
			// 软件中断 (SSIP): M-mode 通过 IPI 注入,
			// 通知 S-mode 有 host 请求待处理.
			csr::clear_csr!(sip, csr::SSI);
			sched::check_pending_requests();
		}
		Some(InterruptCause::SupervisorTimer) => {
			// 定时器中断 — 重设定时器截止时间并记录当前时间片.
			let now: u64;
			unsafe {
				core::arch::asm!("csrr {0}, time", out(reg) now);
			}
			// 定时等待到期检查: 逐个取走截止时刻已到的条目并唤醒其所属线程.
			while let Some(waiter) = timer::elapse_due(now) {
				thread::wake_thread(waiter);
			}

			// 下次中断取「调度定时器的下一次触发时刻」与「表中最早的尚未到期时刻」中的较早者:
			// 定时等待在截止时刻本身触发, 其精度不受调度定时器的触发间隔限制.
			// 取走的条目已不参与 next_deadline, 故不会为同一个截止时刻反复设定.
			let slice_end = now + TICKS_PER_SCHED_TICK;
			let deadline = match timer::next_deadline() {
				Some(d) if d < slice_end => d,
				_ => slice_end,
			};
			// 本平台已在设备树声明Sstc, S 模式可直接写该 CSR;
			// 若改经 SBI TIME ecall 由 M 模式代写,
			// M 模式在未识别 Sstc 时会改设 M 模式 mtimecmp, stimecmp 保持旧值,
			// 使 STIP 持续置位.
			csr::write_stimecmp(deadline);
			csr::clear_csr!(sip, csr::STI);
			// 间隔定时器检查. 遍历全部进程而不只当前进程: 父进程的
			// SIGALRM 常在子进程正在运行时到期.
			proc::tick();
			// 时间片检查. TIME_QUOTA=0 时不触发 SUSPEND.
			sched::tick_and_check_quota();
		}
		None => {
			return;
		}
	}
	thread::maybe_preempt();
}
