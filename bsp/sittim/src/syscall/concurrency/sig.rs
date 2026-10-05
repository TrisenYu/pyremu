//! 信号: 每个进程的处置表、屏蔽集合、待投递集合与信号帧。
//!
//! 本模块只维护信号自身的状态与状态变换, 不含调度决策:
//! - 处置表 (rt_sigaction)、屏蔽集合 (rt_sigprocmask)、待投递集合三者在 `SigState`,
//!   作为进程控制块的一个字段随进程保存 (见 crate::syscall::concurrency::proc)。
//! - 信号帧的构造与恢复 (`build_frame` / `restore_frame`) 是纯地址运算, 只读写
//!   用户栈与陷阱帧, 不触碰进程表。
//! - 决定「投递给谁、投不投递、是否终止进程」的部分在 syscall::concurrency::proc ——
//!   终止进程要动进程表与调度器, 属进程层职责。
//!
//! 投递时机: 无独立投递线程。待投递集合只被置位, 实际投递发生在被投递进程的
//! 某个线程返回 U 模式之前的陷阱返回路径上 (见 trap.rs 的两处返回分支)。

use core::ptr;

use crate::csr;
use crate::syscall::EINVAL;
use crate::trap::{ TrapGprs, A0, A1, A2, RA, SP };

/// 登记的信号编号上界。只登记 1..=31 号标准信号: 待投递集合与屏蔽集合都是
/// 单字位图, 位下标即信号编号; 实时信号 (32..=64) 本运行时未使用。
pub const NSIG: usize = 32;

/// SIG_DFL 与 SIG_IGN 的取值 (musl signal.h)。
const SIG_DFL: u64 = 0;
const SIG_IGN: u64 = 1;

/// SIGALRM (musl signal.h), 间隔定时器到期时投递的信号。
pub const SIGALRM: usize = 14;

/// rt_sigaction 的 new_action/old_action 参数中处理函数地址的字节偏移
/// (struct k_sigaction: handler, flags, mask[2], unused; riscv64 无 SA_RESTORER)。
const SIGACTION_HANDLER_OFF: u64 = 0;

/// struct k_sigaction 的字节数 (handler, flags, mask[2], unused)。
const SIGACTION_SIZE: u64 = 32;

/// 进程被信号终止时的退出码 = 128 + 信号编号, 与 U 模式同步异常终止载荷的
/// 约定一致 (见 trap.rs 的 user_fault_signal)。
pub const SIGNAL_EXIT_OFFSET: u64 = 128;

// ---------------------------------------------------------------
//  信号帧布局
// ---------------------------------------------------------------
//
// 帧在用户栈上自高地址向低地址生长, 全部字段 8 字节对齐。
//
//   0    帧首自带的 rt_sigreturn 指令序列 (li a7,139; ecall)
//   8    信号送达时被中断的 sepc
//   16   信号送达时的 sstatus
//   24   信号送达时的信号屏蔽集合
//   32   TrapGprs 快照 (264 字节)
//   296  信号送达时的用户 sp
//   304  siginfo 区 (128 字节)
//   432  ucontext 区 (256 字节), 保持为零
//   688  尾部填充, 使帧长为 704 字节
//
// 本运行时不为 rt_sigreturn 提供 vDSO 符号, 处理函数经哪个地址回到内核完全由
// 这里决定: 帧首写入 `li a7, 139; ecall` 两条指令, 并把处理函数的 ra 指向帧首,
// 处理函数 ret 即执行该序列进入 rt_sigreturn。
//
// ucontext 区只用于满足 SA_SIGINFO 处理函数的第三个参数: 本运行时不构造完整的
// ucontext_t, 现场恢复走帧内的 sepc/sstatus/屏蔽集合三栏与帧首的指令序列, 故该区
// 保持为零。
const FRAME_RESTORE_OFF: u64 = 0;
const FRAME_SEPC_OFF: u64 = 8;
const FRAME_SSTATUS_OFF: u64 = 16;
const FRAME_BLOCKED_OFF: u64 = 24;
const FRAME_GPRS_OFF: u64 = 32;
const FRAME_USP_OFF: u64 = 296;
const FRAME_SIGINFO_OFF: u64 = 304;
const FRAME_UCONTEXT_OFF: u64 = 432;
const FRAME_SIZE: u64 = 704;

/// siginfo 区长度与 ucontext 区长度。
const SIGINFO_SIZE: u64 = 128;
const UCONTEXT_SIZE: u64 = 256;

/// 帧首的 rt_sigreturn 序列: `addi a7, x0, 139` 与 `ecall` 两条指令的编码。
const SIGNAL_RESTORE_CODE: u64 = 0x0000_0073_08b0_0893;

/// siginfo 的 si_code: SI_USER, 表示信号由 kill 一类调用产生。
const SI_USER: i32 = 0;

// ---------------------------------------------------------------
//  信号状态
// ---------------------------------------------------------------

/// 每个进程一份的信号状态。随进程控制块保存, 随进程切换换入换出。
#[derive(Clone, Copy)]
pub struct SigState {
	/// 信号处置函数地址, 下标即信号编号 (SIG_DFL / SIG_IGN / 用户处理函数)。
	handler: [u64; NSIG],
	/// 信号屏蔽集合位图, 位下标即信号编号, 0 号位不使用。
	blocked: u64,
	/// 待投递信号集合位图, 位下标即信号编号。
	pending: u64,
}

impl SigState {
	pub const fn new() -> Self {
		Self {
			handler: [SIG_DFL; NSIG],
			blocked: 0,
			pending: 0,
		}
	}

	/// 进程式 clone 的子进程取值: 处置表与屏蔽集合同父进程, 待投递集合清空
	/// (信号不随 fork 继承待投递状态)。
	pub fn inherited_from(&self) -> Self {
		Self {
			handler: self.handler,
			blocked: self.blocked,
			pending: 0,
		}
	}

	/// 清空待投递集合 (进程终止时调用)。
	pub fn clear_pending(&mut self) {
		self.pending = 0;
	}

	/// 某信号的处置函数地址。
	#[inline]
	pub fn handler_of(&self, sig: usize) -> u64 {
		self.handler[sig]
	}
}

/// 处置的类别。
pub enum Disposition {
	/// 忽略该信号。
	Ignore,
	/// 按默认处置终止进程。
	Terminate,
	/// 交用户处理函数处理。
	Handle(u64),
}

/// 取某信号当前的处置。
pub fn disposition(st: &SigState, sig: usize) -> Disposition {
	let handler = st.handler[sig];
	if handler == SIG_IGN {
		return Disposition::Ignore;
	}
	if handler == SIG_DFL {
		return match default_action(sig) {
			DefaultAction::Ignore => Disposition::Ignore,
			DefaultAction::Terminate => Disposition::Terminate,
		};
	}
	Disposition::Handle(handler)
}

/// 信号默认处置。
enum DefaultAction {
	/// 终止进程。
	Terminate,
	/// 忽略。
	Ignore,
}

/// 取 Linux 的信号默认处置。停止与继续两类作业控制动作本运行时不实现, 按忽略
/// 处理, 飞地内没有作业控制。
fn default_action(sig: usize) -> DefaultAction {
	match sig {
		// SIGCHLD 17, SIGCONT 18, SIGURG 23, SIGWINCH 28。
		17 | 18 | 23 | 28 => DefaultAction::Ignore,
		// SIGSTOP 19, SIGTSTP 20, SIGTTIN 21, SIGTTOU 22。
		19..=22 => DefaultAction::Ignore,
		// 其余标准信号的默认处置都是终止进程。
		_ => DefaultAction::Terminate,
	}
}

// ---------------------------------------------------------------
//  rt_sigaction (134) 与 rt_sigprocmask (135)
// ---------------------------------------------------------------

/// rt_sigaction (134): 参数为 sig = a0, new_action = a1, old_action = a2,
/// sigsetsize = a3。
///
/// 只记录处理函数地址。new_action 与 old_action 指向 musl 的
/// struct k_sigaction; 本运行时自有信号恢复路径, 既不读也不写该结构的 flags
/// 与 mask 两栏, 回写 old_action 时这两栏置零。
pub fn action(st: &mut SigState, sig: u64, new_action: u64, old_action: u64) -> u64 {
	if sig == 0 || sig >= (NSIG as u64) {
		return EINVAL;
	}
	let s = sig as usize;
	unsafe {
		if old_action != 0 {
			ptr::write_bytes(old_action as *mut u8, 0, SIGACTION_SIZE as usize);
			ptr::write_volatile((old_action + SIGACTION_HANDLER_OFF) as *mut u64, st.handler[s]);
		}
		if new_action != 0 {
			let handler = ptr::read_volatile((new_action + SIGACTION_HANDLER_OFF) as *const u64);
			st.handler[s] = handler;
			// 处置改回 SIG_DFL 或改为 SIG_IGN 时, 此前挂起的该信号不再有意义。
			if handler == SIG_DFL || handler == SIG_IGN {
				st.pending &= !(1u64 << s);
			}
		}
	}
	0
}

/// rt_sigprocmask (135): 参数为 how = a0, set = a1, oldset = a2, sigsetsize = a3。
///
/// how 的取值: 0 = SIG_BLOCK, 1 = SIG_UNBLOCK, 2 = SIG_SETMASK。集合按单字位图
/// 解释, 位下标即信号编号; 集合宽度超出单字的部分本运行时不表示, 按未屏蔽处理。
/// 其余 how 取值返回 EINVAL —— libunwind 以 ~0 充当 how 调用本接口, 断言该调用
/// 必然失败并置 errno, 据此判定地址不可读。
pub fn procmask(st: &mut SigState, how: u64, set: u64, oldset: u64, sigsetsize: u64) -> u64 {
	/// SIG_BLOCK / SIG_UNBLOCK / SIG_SETMASK。
	const SIG_BLOCK: u64 = 0;
	const SIG_UNBLOCK: u64 = 1;
	const SIG_SETMASK: u64 = 2;
	/// 内核侧 sigset 的字节数 (rv64), 也是本运行时能表示的位图宽度。
	const KERNEL_SIGSET_SIZE: u64 = 8;

	if how > SIG_SETMASK {
		return EINVAL;
	}
	let n = core::cmp::min(sigsetsize, KERNEL_SIGSET_SIZE) as usize;

	unsafe {
		if oldset != 0 && n > 0 {
			ptr::write_bytes(oldset as *mut u8, 0, n);
			ptr::write_volatile(oldset as *mut u64, st.blocked);
		}
		if set != 0 && n > 0 {
			let mut word: u64 = 0;
			for i in 0..n {
				let b = ptr::read_volatile((set as *const u8).add(i)) as u64;
				word |= b << (8 * i);
			}
			// 0 号位不是有效信号; SIGKILL 与 SIGSTOP 不可屏蔽。
			word &= !(1u64 << 0);
			word &= !(1u64 << 9);
			word &= !(1u64 << 19);
			match how {
				SIG_BLOCK => {
					st.blocked |= word;
				}
				SIG_UNBLOCK => {
					st.blocked &= !word;
				}
				_ => {
					st.blocked = word;
				}
			}
		}
	}
	0
}

// ---------------------------------------------------------------
//  待投递集合
// ---------------------------------------------------------------

/// 把信号置入待投递集合。
pub fn post(st: &mut SigState, sig: usize) {
	st.pending |= 1u64 << sig;
}

/// 取出一个既未屏蔽又已置位的信号并清除其置位, 按信号编号自小到大取第一个,
/// 与 Linux 对标准信号的处理次序一致。无此类信号时返回 None。
pub fn take_pending(st: &mut SigState) -> Option<usize> {
	let deliverable = st.pending & !st.blocked;
	if deliverable == 0 {
		return None;
	}
	let s = deliverable.trailing_zeros() as usize;
	st.pending &= !(1u64 << s);
	Some(s)
}

// ---------------------------------------------------------------
//  信号帧
// ---------------------------------------------------------------

/// 在用户栈上构造信号帧, 改写陷阱帧使返回时进入处理函数, 返回处理函数地址
/// (即应写入 sepc 的值)。
pub fn build_frame(st: &mut SigState, sig: usize, handler: u64, gprs: &mut TrapGprs, sepc: u64) -> u64 {
	let usp = gprs.usp;
	let frame = usp.wrapping_sub(FRAME_SIZE) & !0xf_u64;
	let blocked_before = st.blocked;

	unsafe {
		ptr::write_volatile((frame + FRAME_RESTORE_OFF) as *mut u64, SIGNAL_RESTORE_CODE);
		ptr::write_volatile((frame + FRAME_SEPC_OFF) as *mut u64, sepc);
		ptr::write_volatile((frame + FRAME_SSTATUS_OFF) as *mut u64, csr::read_sstatus());
		ptr::write_volatile((frame + FRAME_BLOCKED_OFF) as *mut u64, blocked_before);
		ptr::copy_nonoverlapping(
			gprs as *const TrapGprs as *const u8,
			(frame + FRAME_GPRS_OFF) as *mut u8,
			core::mem::size_of::<TrapGprs>()
		);
		ptr::write_volatile((frame + FRAME_USP_OFF) as *mut u64, usp);

		// siginfo: 只写 si_signo 与 si_code 两栏, 其余保持为零。
		ptr::write_bytes((frame + FRAME_SIGINFO_OFF) as *mut u8, 0, SIGINFO_SIZE as usize);
		ptr::write_volatile((frame + FRAME_SIGINFO_OFF) as *mut i32, sig as i32);
		ptr::write_volatile((frame + FRAME_SIGINFO_OFF + 8) as *mut i32, SI_USER);
		ptr::write_bytes((frame + FRAME_UCONTEXT_OFF) as *mut u8, 0, UCONTEXT_SIZE as usize);

		// 处理函数执行期间屏蔽本信号 (未置 SA_NODEFER 时与 Linux 一致)。
		st.blocked |= 1u64 << sig;
	}

	gprs.usp = frame;
	gprs.regs[SP] = frame;
	gprs.regs[A0] = sig as u64;
	gprs.regs[A1] = frame + FRAME_SIGINFO_OFF;
	gprs.regs[A2] = frame + FRAME_UCONTEXT_OFF;
	// 处理函数返回时经帧首的 rt_sigreturn 序列回到内核。
	gprs.regs[RA] = frame + FRAME_RESTORE_OFF;

	handler
}

/// rt_sigreturn (139) 的核心: 从当前用户 sp 指向的信号帧恢复现场, 返回被中断的
/// sepc (即恢复后应写入 sepc 的值, 而非 ecall 的下一条指令)。
pub fn restore_frame(st: &mut SigState, gprs: &mut TrapGprs) -> u64 {
	let frame = gprs.usp;
	unsafe {
		let sepc = ptr::read_volatile((frame + FRAME_SEPC_OFF) as *const u64);
		let sstatus = ptr::read_volatile((frame + FRAME_SSTATUS_OFF) as *const u64);
		let blocked = ptr::read_volatile((frame + FRAME_BLOCKED_OFF) as *const u64);
		ptr::copy_nonoverlapping(
			(frame + FRAME_GPRS_OFF) as *const u8,
			gprs as *mut TrapGprs as *mut u8,
			core::mem::size_of::<TrapGprs>()
		);
		gprs.usp = ptr::read_volatile((frame + FRAME_USP_OFF) as *const u64);
		// 帧内 sstatus 取自信号送达时刻, 其 SPP = 0 (来自 U 模式) 与 SPIE = 1
		// 正是 sret 返回 U 模式所需的取值。
		csr::write_sstatus(sstatus);
		st.blocked = blocked;
		sepc
	}
}
