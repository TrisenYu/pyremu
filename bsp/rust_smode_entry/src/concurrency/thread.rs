//! 单 hart 用户线程运行时: 线程表、clone / exit、时间片轮转调度。
//!
//! futex 语义在独立的 futex.rs; 两者经本模块暴露的「阻塞队列原语」配合:
//!   - futex WAIT   阻塞当前线程: thread::block_current(wait_addr)
//!   - futex WAKE   唤醒等待者:   thread::wake_blocked(addr, nr)
//!   - futex REQUEUE 迁移等待者:  thread::requeue_blocked(...)
//!
//! 调度模型
//! - 线程表是 BSS 中的定长数组 (跨 SUSPEND/RESUME 存活), 槽下标即内部
//!   thread_id, 0 为启动主线程。对用户可见的线程标识 = 槽下标 + 1, 恒非零:
//!   musl 以 __thread_list_lock 是否等于自身 tid 判定可重入锁持有者, 而该锁
//!   初值为 0, 故对外线程标识绝不能是 0 (见 vendor/musl env/__init_tls.c)。
//! - 每个线程有独立的 S-mode 内核栈, 栈顶下方 264 字节即汇编 SAVE_CONTEXT
//!   保存的陷阱帧 (crate::trap::TrapGprs)。线程切出后寄存器与用户 sp 留在
//!   自身帧; sepc/sstatus 记入控制块, 切入时经 activate 写回 CSR 后 sret。
//! - 切换统一经 trap_dispatch 的返回值驱动: 返回 0 不切换; 非 0 为目标线程
//!   帧基址, entry.s 据此 `mv sp, a0` 换栈后走统一恢复路径 (见 entry.s)。
//!   决策函数只改状态并把目标登记进 PENDING_TARGET, trap.rs 在各自分支末尾
//!   调用 finalize_switch 完成「记录当前线程恢复现场 + 激活目标线程」。
//!
//! 覆盖
//!   - clone / gettid / sched_yield / set_tid_address / exit(93)
//!   - 阻塞/唤醒/迁移原语 (futex 与线程退出释放 clear_child_thread_id)
//!   - STIP 时间片轮转 (trap.rs 中断分支调用 maybe_preempt)

#![allow(dead_code)]

use core::ptr;

use crate::csr;
#[cfg(feature = "diagnostic")]
use crate::diag;
use crate::ecall_aux;
use crate::mem;
use crate::trap::{TrapGprs, A0, TP};
use crate::constants::ENCLAVE_SUSPEND_VOLUNTARY;

/// 汇编 SAVE_CONTEXT 的陷阱帧大小 (见 entry.s, 264 字节)。
pub const TRAP_FRAME_SIZE: u64 = 264;

/// 每个新建线程的 S-mode 内核栈大小 (16 KiB)。
pub const THREAD_KSTACK_SIZE: u64 = 0x4000;

/// 线程表容量 (含主线程)。
pub const NUM_THREADS: usize = 32;

// clone 标志位 (Linux 通用定义子集, include/uapi/linux/sched.h)。
const CLONE_VM: u64 = 0x0000_0100;
const CLONE_THREAD: u64 = 0x0001_0000;
const CLONE_SETTLS: u64 = 0x0008_0000;
const CLONE_PARENT_SETTID: u64 = 0x0010_0000;
const CLONE_CHILD_CLEARTID: u64 = 0x0020_0000;
const CLONE_CHILD_SETTID: u64 = 0x0100_0000;

// errno 负值以无符号补码表示 (号数见 musl errno.h)。
const EAGAIN: u64 = (!0u64) - 10; // errno 11
const ENOMEM: u64 = (!0u64) - 11; // errno 12
const EINVAL: u64 = (!0u64) - 21; // errno 22

/// 线程状态机。
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ThreadState {
	/// 槽位空闲, 可被 clone 复用 (内核栈保留)。
	Free,
	/// 已就绪, 等待被调度。
	Runnable,
	/// 正在内核中执行 (仅当前线程)。
	Running,
	/// 阻塞于等待地址 (wait_addr 有效), 由唤醒原语转为 Runnable。
	Blocked,
}

/// 线程控制块。线程表槽下标即内部线程标识 thread_id, 0 为启动主线程。
#[derive(Clone, Copy)]
pub struct ThreadCtl {
	pub state: ThreadState,
	/// 本线程 S-mode 内核栈顶 VA; 陷阱帧基址 = kstack_top - TRAP_FRAME_SIZE。
	pub kstack_top: u64,
	/// 本线程切出时用户态下一条指令, 切入时写回 CSR sepc。
	pub sepc: u64,
	/// 本线程切出时的 sstatus, 切入时写回 CSR sstatus 供 sret 使用。
	pub sstatus: u64,
	/// 阻塞等待地址 (state == Blocked 时有效)。
	pub wait_addr: u64,
	/// 用户 TLS 控制块地址 (clone 的 tls 参数, 即 tp 寄存器的用户值)。
	pub thread_meta_info_addr: u64,
	/// 线程退出时需清零并唤醒的用户地址 (set_tid_address / CLONE_CHILD_CLEARTID)。
	pub clear_child_thread_id: u64,
}

impl ThreadCtl {
	pub const fn free() -> Self {
		Self {
			state: ThreadState::Free,
			kstack_top: 0,
			sepc: 0,
			sstatus: 0,
			wait_addr: 0,
			thread_meta_info_addr: 0,
			clear_child_thread_id: 0,
		}
	}
}

/// 线程表: BSS 零初始化后再以 init_main_thread 写入线程 0。
static mut THREADS: [ThreadCtl; NUM_THREADS] = [ThreadCtl::free(); NUM_THREADS];

/// 当前线程标识 (仅调度代码在内核态读写, SIE = 0 下天然串行)。
static mut CURRENT: usize = 0;

/// 启动阶段分配的主线程内核栈顶, 由 rust_main_before_mmu 登记。
static mut BOOT_KSTACK_TOP: u64 = 0;

/// 已登记待切换目标线程 (None 表示本次无切换)。
static mut PENDING_TARGET: Option<usize> = None;

// ---------------------------------------------------------------
//  登记与查询
// ---------------------------------------------------------------

/// 登记启动阶段已分配并在使用的主线程内核栈顶 VA。
pub fn note_boot_kstack_top(kstack_top: u64) {
	unsafe {
		BOOT_KSTACK_TOP = kstack_top;
	}
}

/// 登记主线程 (thread_id = 0) 并初始化调度状态。
///
/// 由 rust_main_after_mmu 在首次 sret 前调用一次。线程 0 的用户现场
/// (入口 sepc 与 sstatus) 以参数显式传入, 不再从 CSR 回读: 该现场 CSR 必须在
/// 全部 ecall 之后才写入 (理由见 rust_main_after_mmu 中的说明), 而本函数又必须
/// 在写入之前完成登记, 两者顺序互斥, 故改由调用方传递。
pub fn init_main_thread(user_sepc: u64, user_sstatus: u64) {
	unsafe {
		THREADS[0] = ThreadCtl {
			state: ThreadState::Running,
			kstack_top: BOOT_KSTACK_TOP,
			sepc: user_sepc,
			sstatus: user_sstatus,
			wait_addr: 0,
			thread_meta_info_addr: 0,
			clear_child_thread_id: 0,
		};
		CURRENT = 0;
		PENDING_TARGET = None;
	}
}

/// 返回当前线程的内部标识 (槽下标)。
pub fn current_thread_id() -> usize {
	unsafe { CURRENT }
}

/// 返回当前线程的对外线程标识 (槽下标 + 1, 恒非零)。
pub fn current_external_thread_id() -> u64 {
	unsafe { (CURRENT + 1) as u64 }
}

/// 返回某线程陷阱帧基址 (其内核栈顶减帧大小)。
pub fn frame_base(thread_id: usize) -> u64 {
	unsafe { THREADS[thread_id].kstack_top - TRAP_FRAME_SIZE }
}

/// 是否有已登记的待切换目标 (供 trap.rs 判定是否走切换路径)。
pub fn switch_pending() -> bool {
	unsafe {
		// Rust 2024 禁止对 static mut 自动取引用, 以拷贝读取判定.
		match PENDING_TARGET {
			Some(_) => true,
			None => false,
		}
	}
}

/// 自 1 起找第一个空闲槽位 (槽 0 恒为主线程)。
fn next_free_slot() -> Option<usize> {
	unsafe {
		for s in 1..NUM_THREADS {
			if THREADS[s].state == ThreadState::Free {
				return Some(s);
			}
		}
		None
	}
}

/// 自 after 起按槽位顺序找下一个可运行线程 (Round-robin, 不含自身)。
fn next_runnable(after: usize) -> Option<usize> {
	unsafe {
		for off in 1..NUM_THREADS {
			let s = (after + off) % NUM_THREADS;
			if THREADS[s].state == ThreadState::Runnable {
				return Some(s);
			}
		}
		None
	}
}

/// 向用户地址写 32 位字 (S 模式以 sstatus.SUM = 1 直写 U 页)。
fn store_user_u32(addr: u64, val: u32) {
	unsafe { ptr::write_volatile(addr as *mut u32, val) };
}

// ---------------------------------------------------------------
//  clone (220)
// ---------------------------------------------------------------

/// 创建共享地址空间线程。
///
/// 参数 (Linux rv64 clone): flags = a0, stack = a1, ptid = a2, tls = a3,
/// ctid = a4。返回子线程对外标识; 失败返回负 errno (EAGAIN / ENOMEM / EINVAL)。
///
/// 子线程的首个用户现场为父帧拷贝后按 clone 语义改写:
/// - a0 = 0 (子线程 clone 返回 0, 见 musl riscv64/clone.s 的 beqz 分支);
/// - CLONE_SETTLS 时 tp = tls (Linux copy_thread 对子线程同样改写);
/// - 用户 sp = stack (a1): musl 已把 func/arg 存到该栈顶, 子线程自 [sp] 取用,
///   故子线程用户 sp 必须是传入的 stack, 而非父线程 sp (父 sp 未变)。
pub fn clone_thread(
	gprs: &TrapGprs,
	flags: u64,
	stack: u64,
	ptid: u64,
	tls: u64,
	ctid: u64,
) -> u64 {
	unsafe {
		// 仅支持共享地址空间的兄弟线程; fork 式独立地址空间不在运行时范围。
		if flags & (CLONE_VM | CLONE_THREAD) != CLONE_VM | CLONE_THREAD {
			return EINVAL;
		}
		let child = match next_free_slot() {
			Some(c) => c,
			None => return EAGAIN,
		};
		// 首次占用的槽位新分配内核栈; 复用的槽位保留旧栈避免页池反复消耗。
		if THREADS[child].kstack_top == 0 {
			let top = mem::try_alloc_thread_kstack(THREAD_KSTACK_SIZE);
			if top == 0 {
				return ENOMEM;
			}
			THREADS[child].kstack_top = top;
		}

		// 构造子线程陷阱帧: 先整体拷贝父帧, 再按 clone 语义改写个别槽。
		let dst = frame_base(child) as *mut TrapGprs;
		ptr::copy_nonoverlapping(gprs as *const TrapGprs, dst, 1);
		let cf = &mut *dst;
		cf.regs[A0] = 0;
		if flags & CLONE_SETTLS != 0 {
			cf.regs[TP] = tls;
		}
		cf.usp = stack;

		// 登记子线程控制块; 恢复现场取父线程此刻的 CSR (与父共享同一现场值)。
		THREADS[child].state = ThreadState::Runnable;
		THREADS[child].sepc = csr::read_sepc().wrapping_add(4);
		THREADS[child].sstatus = csr::read_sstatus();
		THREADS[child].wait_addr = 0;
		THREADS[child].clear_child_thread_id = 0;
		if flags & CLONE_SETTLS != 0 {
			THREADS[child].thread_meta_info_addr = tls;
		} else {
			THREADS[child].thread_meta_info_addr = gprs.tp();
		}
		// PARENT_SETTID: 父线程可经 ptid 读取子线程标识 (musl 的 new->tid)。
		if flags & CLONE_PARENT_SETTID != 0 {
			store_user_u32(ptid, (child + 1) as u32);
		}
		// CHILD_SETTID: 子线程自身可经 ctid 读取标识 (当前载荷不使用)。
		if flags & CLONE_CHILD_SETTID != 0 {
			store_user_u32(ctid, (child + 1) as u32);
		}
		// CHILD_CLEARTID: 线程退出时内核清零该地址并唤醒 (musl 的 __tl_unlock
		// 延后提交, 见 pthread_create.c __pthread_exit 尾部注释)。
		if flags & CLONE_CHILD_CLEARTID != 0 {
			THREADS[child].clear_child_thread_id = ctid;
		}

		// 子线程优先运行 (与 Linux 一致); 父线程现场由 finalize_switch 保存。
		PENDING_TARGET = Some(child);
		(child + 1) as u64
	}
}

// ---------------------------------------------------------------
//  退出 (93)
// ---------------------------------------------------------------

/// 终止当前线程 (exit 93)。主线程或组内最后可运行线程退出即整个飞地退出。
pub fn exit_current_thread(code: u64) -> u64 {
	unsafe {
		let cur = CURRENT;
		// 主线程退出等价于整组退出 (Linux 首线程回收即销毁整个线程组)。
		if cur == 0 {
			ecall_aux::enclave_call_exit(code);
		}
		// 清零 clear_child_thread_id 并唤醒一位等待者, 释放仍被本线程持有的
		// __thread_list_lock (musl 在 SYS_exit 前一直持有该锁, 见 __pthread_exit)。
		let ctid = THREADS[cur].clear_child_thread_id;
		if ctid != 0 {
			store_user_u32(ctid, 0);
			wake_blocked(ctid, 1);
		}
		// 释放槽位 (内核栈保留供复用)。
		THREADS[cur].state = ThreadState::Free;
		THREADS[cur].sepc = 0;
		THREADS[cur].sstatus = 0;
		THREADS[cur].wait_addr = 0;
		THREADS[cur].thread_meta_info_addr = 0;
		THREADS[cur].clear_child_thread_id = 0;
		// 有就绪线程则让出切换; 组内已无活线程则整飞地退出。
		match next_runnable(cur) {
			Some(next) => {
				PENDING_TARGET = Some(next);
				0
			}
			None => ecall_aux::enclave_call_exit(code),
		}
	}
}

// ---------------------------------------------------------------
//  让出 / 抢占 / set_tid_address
// ---------------------------------------------------------------

/// 主动让出 CPU (sched_yield 124): 有他线程就绪则登记切换目标。
pub fn sched_yield_current() {
	unsafe {
		if let Some(next) = next_runnable(CURRENT) {
			PENDING_TARGET = Some(next);
		}
	}
}

/// STIP 时间片边界抢占 (trap.rs 中断分支调用): 有他线程就绪则轮转。
pub fn maybe_preempt() {
	unsafe {
		if let Some(next) = next_runnable(CURRENT) {
			PENDING_TARGET = Some(next);
		}
	}
}

/// 登记本线程的 clear_child_thread_id (set_tid_address 96), 返回本线程对外标识。
///
/// musl 主线程在 libc 初始化期以此登记 __thread_list_lock 地址, 并把返回值
/// 当作自身 tid (见 env/__init_tls.c), 故必须返回非零的线程标识。
pub fn set_tid_address(addr: u64) -> u64 {
	unsafe {
		THREADS[CURRENT].clear_child_thread_id = addr;
	}
	current_external_thread_id()
}

// ---------------------------------------------------------------
//  阻塞队列原语 (futex 与线程退出释放 clear_child_thread_id 共用)
// ---------------------------------------------------------------

/// 阻塞当前线程于 wait_addr 并让出调度。
///
/// 有他线程就绪则登记切换目标后返回 (由 trap.rs 完成切换); 组内已无任何
/// 可运行线程 (死锁) 时循环让出给 host, host 可 SHUTDOWN 回收本飞地。
pub fn block_current(wait_addr: u64) {
	unsafe {
		let cur = CURRENT;
		THREADS[cur].state = ThreadState::Blocked;
		THREADS[cur].wait_addr = wait_addr;
		loop {
			if let Some(next) = next_runnable(cur) {
				PENDING_TARGET = Some(next);
				return;
			}
			// 让出给 host: 本飞地已无运行空间, 等待 host RESUME 或 SHUTDOWN。
			// 属自愿让出, 不参与时间片轮转 — 须交还宿主而非就地续期。
			#[cfg(feature = "diagnostic")]
			diag::log(format_args!(
				"[blk] suspend t={}\n",
				diag::read_mtime()
			));
			let _ = ecall_aux::enclave_call_suspend(
				ENCLAVE_SUSPEND_VOLUNTARY
			);
		}
	}
}

/// 唤醒至多 nr 位阻塞于 wait_addr 的线程, 返回实际唤醒数。
pub fn wake_blocked(wait_addr: u64, nr: u64) -> u64 {
	unsafe {
		let mut woken: u64 = 0;
		for s in 0..NUM_THREADS {
			if woken >= nr {
				break;
			}
			if THREADS[s].state == ThreadState::Blocked && THREADS[s].wait_addr == wait_addr
			{
				THREADS[s].state = ThreadState::Runnable;
				THREADS[s].wait_addr = 0;
				woken += 1;
			}
		}
		woken
	}
}

/// FUTEX_REQUEUE 内核动作: 先唤醒至多 nr_wake 位, 再把至多 nr_requeue 位
/// 仍阻塞于 wait_addr 的等待者迁移到 target_addr (保持阻塞)。返回唤醒数。
///
/// 供 cond 广播/移交把等待者从 cond 挂到互斥锁字上, 待锁释放时一并唤醒。
pub fn requeue_blocked(
	wait_addr: u64,
	nr_wake: u64,
	nr_requeue: u64,
	target_addr: u64,
) -> u64 {
	unsafe {
		let woken = wake_blocked(wait_addr, nr_wake);
		let mut requeued: u64 = 0;
		for s in 0..NUM_THREADS {
			if requeued >= nr_requeue {
				break;
			}
			if THREADS[s].state == ThreadState::Blocked && THREADS[s].wait_addr == wait_addr
			{
				THREADS[s].wait_addr = target_addr;
				requeued += 1;
			}
		}
		woken
	}
}

// ---------------------------------------------------------------
//  切换收尾
// ---------------------------------------------------------------

/// 完成一次切换: 先记录当前线程的恢复现场, 再激活 PENDING_TARGET。
///
/// resume_sepc 为当前线程下次恢复的用户 PC (系统调用为 ecall + 4, 中断为
/// 被中断的 pc)。返回目标线程帧基址 (entry.s 据此 mv sp, a0); 无目标返回 0。
pub fn finalize_switch(resume_sepc: u64) -> u64 {
	unsafe {
		let cur = CURRENT;
		// 当前线程存活则保存现场 (已 Free 的退出线程无需保存)。
		if THREADS[cur].state != ThreadState::Free {
			THREADS[cur].sepc = resume_sepc;
			THREADS[cur].sstatus = csr::read_sstatus();
			if THREADS[cur].state == ThreadState::Running {
				THREADS[cur].state = ThreadState::Runnable;
			}
		}
		// Rust 2024 禁止对 static mut 取引用, 以拷贝方式取出并清空目标.
		let pending = PENDING_TARGET;
		PENDING_TARGET = None;
		match pending {
			Some(target) => {
				THREADS[target].state = ThreadState::Running;
				CURRENT = target;
				csr::write_sepc(THREADS[target].sepc);
				csr::write_sstatus(THREADS[target].sstatus);
				THREADS[target].kstack_top - TRAP_FRAME_SIZE
			}
			None => 0,
		}
	}
}
