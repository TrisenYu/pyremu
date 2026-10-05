//! 陷态分发。紧跟 smode_entry/trap_handler.c + ref-emod trap/exceptions.c。
use crate::constants::*;
use crate::csr;
use crate::ecall_aux;
use crate::hang;
use crate::syscall;
use crate::syscall::concurrency::{ proc, sched, thread, timer };

#[cfg(feature = "diagnostic")]
use crate::diag;
// ---------------------------------------------------------------
//  GPR 寄存器别名, 对应 smode_entry/trap_handler.h 中 CTX_INDEX_*
// ---------------------------------------------------------------

#[allow(unused)]
pub const RA: usize = 1;
pub const SP: usize = 2;
pub const GP: usize = 3;
pub const TP: usize = 4;
pub const T0: usize = 5;
pub const T1: usize = 6;
pub const T2: usize = 7;
pub const S0: usize = 8;
pub const S1: usize = 9;
pub const A0: usize = 10;
pub const A1: usize = 11;
pub const A2: usize = 12;
pub const A3: usize = 13;
pub const A4: usize = 14;
pub const A5: usize = 15;
pub const A6: usize = 16;
pub const A7: usize = 17;
pub const S2: usize = 18;
pub const S3: usize = 19;
pub const S4: usize = 20;
pub const S5: usize = 21;
pub const S6: usize = 22;
pub const S7: usize = 23;
pub const S8: usize = 24;
pub const S9: usize = 25;
pub const S10: usize = 26;
pub const S11: usize = 27;
pub const T3: usize = 28;
pub const T4: usize = 29;
pub const T5: usize = 30;
pub const T6: usize = 31;

// ---------------------------------------------------------------
//  TrapGprs —— 与 entry.s 的 SAVE_CONTEXT 布局一致
// ---------------------------------------------------------------

/// 汇编 SAVE_CONTEXT 后的寄存器快照。
/// 31 个 GPR + 用户 sp 槽 = 32 × 8 = 256 + 8 = 264 字节。
#[repr(C)]
pub struct TrapGprs {
	/// x0..x31，x0 恒为 0，该字段为占位。
	pub regs: [u64; 32],
	/// 用户 sp, SAVE_CONTEXT 后由 csrrw t1, sscratch, x0; STORE t1, 偏移(sp) 填入。
	pub usp: u64,
}

#[allow(unused)]
impl TrapGprs {
	// 所有 32 GPR 的 getter
	#[inline]
	pub fn zero(&self) -> u64 {
		self.regs[0]
	}
	#[inline]
	pub fn ra(&self) -> u64 {
		self.regs[RA]
	}
	#[inline]
	pub fn sp(&self) -> u64 {
		self.regs[SP]
	}
	#[inline]
	pub fn gp(&self) -> u64 {
		self.regs[GP]
	}
	#[inline]
	pub fn tp(&self) -> u64 {
		self.regs[TP]
	}
	#[inline]
	pub fn t0(&self) -> u64 {
		self.regs[T0]
	}
	#[inline]
	pub fn t1(&self) -> u64 {
		self.regs[T1]
	}
	#[inline]
	pub fn t2(&self) -> u64 {
		self.regs[T2]
	}
	#[inline]
	pub fn s0(&self) -> u64 {
		self.regs[S0]
	}
	#[inline]
	pub fn s1(&self) -> u64 {
		self.regs[S1]
	}
	#[inline]
	pub fn a0(&self) -> u64 {
		self.regs[A0]
	}
	#[inline]
	pub fn a1(&self) -> u64 {
		self.regs[A1]
	}
	#[inline]
	pub fn a2(&self) -> u64 {
		self.regs[A2]
	}
	#[inline]
	pub fn a3(&self) -> u64 {
		self.regs[A3]
	}
	#[inline]
	pub fn a4(&self) -> u64 {
		self.regs[A4]
	}
	#[inline]
	pub fn a5(&self) -> u64 {
		self.regs[A5]
	}
	#[inline]
	pub fn a6(&self) -> u64 {
		self.regs[A6]
	}
	#[inline]
	pub fn a7(&self) -> u64 {
		self.regs[A7]
	}
	#[inline]
	pub fn s2(&self) -> u64 {
		self.regs[S2]
	}
	#[inline]
	pub fn s3(&self) -> u64 {
		self.regs[S3]
	}
	#[inline]
	pub fn s4(&self) -> u64 {
		self.regs[S4]
	}
	#[inline]
	pub fn s5(&self) -> u64 {
		self.regs[S5]
	}
	#[inline]
	pub fn s6(&self) -> u64 {
		self.regs[S6]
	}
	#[inline]
	pub fn s7(&self) -> u64 {
		self.regs[S7]
	}
	#[inline]
	pub fn s8(&self) -> u64 {
		self.regs[S8]
	}
	#[inline]
	pub fn s9(&self) -> u64 {
		self.regs[S9]
	}
	#[inline]
	pub fn s10(&self) -> u64 {
		self.regs[S10]
	}
	#[inline]
	pub fn s11(&self) -> u64 {
		self.regs[S11]
	}
	#[inline]
	pub fn t3(&self) -> u64 {
		self.regs[T3]
	}
	#[inline]
	pub fn t4(&self) -> u64 {
		self.regs[T4]
	}
	#[inline]
	pub fn t5(&self) -> u64 {
		self.regs[T5]
	}
	#[inline]
	pub fn t6(&self) -> u64 {
		self.regs[T6]
	}

	// syscall 返回时写回的 setter
	#[inline]
	pub fn set_a0(&mut self, v: u64) {
		self.regs[A0] = v;
	}
	#[inline]
	pub fn set_a7(&mut self, v: u64) {
		self.regs[A7] = v;
	}
}

// ---------------------------------------------------------------
//  C ABI 入口
// ---------------------------------------------------------------

/// entry.s调用。通过csrr设置函数参数。
#[unsafe(no_mangle)]
pub unsafe extern "C" fn trap_dispatch(gprs: &mut TrapGprs, sepc: u64, scause: u64, stval: u64) -> u64 {
	// 返回值约定: 0 表示不切换线程; 非 0 为目标线程陷阱帧基址,
	// entry.s 据此 mv sp, a0 换到目标帧后走统一的恢复与 sret 路径.
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
		// rt_sigreturn 恢复的现场覆盖整个陷阱帧 (含 a0), 且恢复点是被中断的 pc;
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
		// 投递信号: 有用户处理函数则改为以处理函数地址恢复, 使返回时先进入处理
		// 函数; 默认处置为终止时本进程就此终结并登记切换目标, 故投递先于切换
		// 检查.
		let next_sepc = proc::deliver_pending(gprs, resume_sepc).unwrap_or(resume_sepc);
		// syscall 与信号投递都可能已登记线程切换 (clone / futex 阻塞 / exit /
		// 让出 / 信号终止): 有则记录当前线程现场并激活目标线程.
		if thread::switch_pending() {
			return thread::finalize_switch(next_sepc);
		}
		csr::write_sepc(next_sepc);
		return 0;
	}

	// 访问故障：转发到 M-mode
	if scause == 0x5 || scause == 0x7 {
		ecall_aux::enclave_call_unmatched_acc_fault(stval);
		return 0;
	}

	// 用户态 (sstatus.SPP=0) 不可恢复的同步异常: 载荷自身的缺陷 (页错误/非法指令/
	// 地址错位/断点…)。按 Linux 的信号默认处置终止该进程, 而不是冻结挂起, 也不
	// 牵连同一飞地内的其它进程。退出码取 128+sig, host 经 shutdown 收到
	// EXITED_ERR(code) 精确反馈载荷异常终止, 由可信应用管理器决定后续处置
	// (终止/重启/清空并上报)。被终止的进程是启动进程时整个飞地退出, 见
	// proc::exit_process。
	if (csr::read_sstatus() & csr::SSTATUS_SPP) == 0 {
		#[cfg(feature = "diagnostic")]
		diag::log(
			format_args!(
				"[trap] u-fault scause=0x{:x} stval=0x{:x} sepc=0x{:x} sig={}\n",
				scause,
				stval,
				sepc,
				user_fault_signal(scause)
			)
		);
		proc::exit_process(proc::current_slot(), 128 + user_fault_signal(scause));
		// 进程层已登记切换目标; 无其它可运行线程时它已终止飞地, 不会返回。
		if thread::switch_pending() {
			return thread::finalize_switch(sepc);
		}
		return 0;
	}

	// S 模式自身 (runtime 缺陷) 或无法归类的同步异常 —— 输出原因与完整现场
	// (scause/stval/sepc) 后冻结挂起, 但仍响应 host 的终止请求
	hang::fault_halt_exc(scause, stval, sepc, get_trap_reason(scause));
}

/// 将 U 模式同步异常 scause 折算为进程收到该异常时的默认信号编号 (Linux 语义),
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
fn get_trap_reason(cause: u64) -> &'static str {
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
			// 定时器中断 — 重设定时器截止时间并记账时间片配额.
			// 不调用 check_pending_requests(): host 请求 SHUTDOWN，
			// 由 M-mode 经 IPI 注入 SSIP 异步通知.
			let now: u64;
			unsafe {
				core::arch::asm!("csrr {0}, time", out(reg) now);
			}
			// 定时等待到期检查 (见 concurrency/timer.rs): 逐个取走截止时刻已到的条目
			// 并唤醒其所属线程, 由本次中断末尾的轮转登记交给它运行.
			while let Some(waiter) = timer::elapse_due(now) {
				thread::wake_thread(waiter);
			}

			// 下次中断取「下一个时间片边界」与「表中最早的尚未到期时刻」中的较早者:
			// 定时等待在截止时刻本身触发, 其精度不受时间片周期 (config.mk 的
			// TIMER_INTERVAL) 限制. 取走的条目已不参与 next_deadline, 故不会为同一个
			// 截止时刻反复设定.
			let slice_end = now + TIMER_INTERVAL;
			let deadline = match timer::next_deadline() {
				Some(d) if d < slice_end => d,
				_ => slice_end,
			};
			// 直接改写 stimecmp (Sstc) 重设 S 模式定时器. 本平台已在设备树声明
			// Sstc, S 模式可直接写该 CSR; 若改经 SBI TIME ecall 由 M 模式代写,
			// M 模式在未识别 Sstc 时会改设 M 模式 mtimecmp, stimecmp 保持旧值,
			// 使 STIP 持续置位.
			csr::write_stimecmp(deadline);
			csr::clear_csr!(sip, csr::STI);
			// 间隔定时器检查 (见 proc.rs). 遍历全部进程而不只当前进程: 父进程的
			// SIGALRM 常在子进程正在运行时到期.
			proc::tick();
			// 时间片检查 (见 sched.rs). TIME_QUOTA=0 时不触发 SUSPEND.
			sched::tick_and_check_quota();
		}
		None => {
			return;
		}
	}
	thread::maybe_preempt();
}
