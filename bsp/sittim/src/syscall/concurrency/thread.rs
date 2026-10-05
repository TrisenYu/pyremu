//! 用户线程运行时: 线程表、clone / exit、时间片轮转调度, 以及线程类系统调用。
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
//!   决策函数只改状态并把目标登记进调度状态, trap.rs 在各自分支末尾调用
//!   finalize_switch 完成「记录当前线程恢复现场 + 激活目标线程」。
//!
//! 覆盖
//!   - clone / gettid / sched_yield / set_tid_address / exit(93)
//!   - 阻塞/唤醒/迁移原语 (futex 与线程退出释放 clear_child_thread_id)
//!   - STIP 时间片轮转 (trap.rs 中断分支调用 maybe_preempt)
//!   - 线程类系统调用粘合 (见文件末节), 只做寄存器参数到上述原语的搬运

#![allow(dead_code)]

use core::cell::UnsafeCell;
use core::ptr;
use core::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

use crate::csr;
use crate::ecall_aux;
use crate::mem;
use crate::syscall::concurrency::{ futex, proc, timer };
use crate::syscall::types::Timespec;
use crate::syscall::{ EAGAIN, EFAULT, EINVAL, ENOMEM };
use crate::trap::{ TrapGprs, A0, TP };
use crate::constants::ENCLAVE_SUSPEND_IDLE;

/// 汇编 SAVE_CONTEXT 的陷阱帧大小 (见 entry.s, 264 字节)。
pub const TRAP_FRAME_SIZE: u64 = 264;

/// 每个新建线程的 S-mode 内核栈大小 (16 KiB)。
pub const THREAD_KSTACK_SIZE: u64 = 0x4000;

/// 线程表容量 (含主线程)。
pub const NUM_THREADS: usize = 32;

/// nanosleep 的阻塞地址。该地址不与任何唤醒原语的等待地址相同, 故睡眠中的线程只由
/// 自身的等待定时器经 [wake_thread] 唤醒, 不被别处的唤醒波及。
const SLEEP_WAIT_ADDR: u64 = 0x7fff_3000_0000;

// clone 标志位 (Linux 通用定义子集, include/uapi/linux/sched.h)。
const CLONE_VM: u64 = 0x0000_0100;
const CLONE_THREAD: u64 = 0x0001_0000;
const CLONE_SETTLS: u64 = 0x0008_0000;
const CLONE_PARENT_SETTID: u64 = 0x0010_0000;
const CLONE_CHILD_CLEARTID: u64 = 0x0020_0000;
const CLONE_CHILD_SETTID: u64 = 0x0100_0000;

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
	/// 所属进程的进程表槽下标 (见 concurrency::proc)。
	pub proc: usize,
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
			proc: 0,
			kstack_top: 0,
			sepc: 0,
			sstatus: 0,
			wait_addr: 0,
			thread_meta_info_addr: 0,
			clear_child_thread_id: 0,
		}
	}
}

/// 待切换目标的空值。线程槽下标恒小于 [NUM_THREADS], 故取该值表示本次无切换。
const NO_TARGET: usize = usize::MAX;

/// 线程表。表本身不外露, 访问一律经本类型的方法。
///
/// 表以 [UnsafeCell] 承载: 对表的访问都发生在单 hart 的 S-mode 内核态, 系统调用与中断
/// 处理之间不重入 (SIE = 0), 表在其间不存在并发访问, 故把这一不变量的举证收在本类型内。
struct ThreadTable(UnsafeCell<[ThreadCtl; NUM_THREADS]>);

// SAFETY: 见 [ThreadTable] 的说明。全表只在单 hart 的 S-mode 内核态被访问, 且各次访问
// 之间不存在并发。
unsafe impl Sync for ThreadTable {}

impl ThreadTable {
	/// 读取线程 *thread_id* 的控制块。
	fn get(&self, thread_id: usize) -> ThreadCtl {
		unsafe { (*self.0.get())[thread_id] }
	}

	/// 写入线程 *thread_id* 的控制块。
	fn set(&self, thread_id: usize, ctl: ThreadCtl) {
		unsafe {
			(*self.0.get())[thread_id] = ctl;
		}
	}

	/// 就地改动线程 *thread_id* 的控制块, 返回 *f* 的结果。读改写的多项改动经此处一次
	/// 完成, 调用点不必先取出整个控制块再写回。
	fn with<R>(&self, thread_id: usize, f: impl FnOnce(&mut ThreadCtl) -> R) -> R {
		unsafe { f(&mut (*self.0.get())[thread_id]) }
	}

	/// 把全表复位为初始状态 (全部槽位空闲)。
	fn reset(&self) {
		unsafe {
			*self.0.get() = [ThreadCtl::free(); NUM_THREADS];
		}
	}
}

/// 线程表: BSS 零初始化后再以 init_main_thread 写入线程 0。
static THREADS: ThreadTable = ThreadTable(UnsafeCell::new([ThreadCtl::free(); NUM_THREADS]));

/// 调度状态: 当前线程标识、已登记的待切换目标、主线程内核栈顶与本次系统调用的重启请求。
///
/// 四项都是单 hart 的 S-mode 内核态内的标量, 故以原子量承载: 读改写都是普通的方法调用,
/// 调用点无需 `unsafe`。
struct SchedState {
	/// 当前线程的槽下标。
	current: AtomicUsize,
	/// 已登记的待切换目标槽下标; [`NO_TARGET`] 表示本次无切换。
	pending_target: AtomicUsize,
	/// 启动阶段分配的主线程内核栈顶, 由 rust_main_before_mmu 登记。
	boot_kstack_top: AtomicU64,
	/// 本次系统调用是否请求重启 (恢复点取 ecall 自身而非 ecall + 4)。
	///
	/// wait4 阻塞后被唤醒时需要重新扫描子进程, 而在切换回本线程之前无法重扫, 故只
	/// 能请求重启: 恢复点退回 ecall, a0 也不写回返回值, 由调用方在重新执行时重新
	/// 传入全部参数。
	is_restart_requested: AtomicBool,
}

impl SchedState {
	/// 当前线程的槽下标。
	fn current(&self) -> usize {
		self.current.load(Ordering::SeqCst)
	}

	/// 登记 *thread_id* 为当前线程。
	fn set_current(&self, thread_id: usize) {
		self.current.store(thread_id, Ordering::SeqCst);
	}

	/// 已登记的待切换目标; 本次无切换时为 None。
	fn pending_target(&self) -> Option<usize> {
		match self.pending_target.load(Ordering::SeqCst) {
			NO_TARGET => None,
			target => Some(target),
		}
	}

	/// 登记待切换目标。
	fn set_pending_target(&self, thread_id: usize) {
		self.pending_target.store(thread_id, Ordering::SeqCst);
	}

	/// 清除已登记的待切换目标。
	fn clear_pending_target(&self) {
		self.pending_target.store(NO_TARGET, Ordering::SeqCst);
	}

	/// 取走已登记的待切换目标, 并清除登记。
	fn take_pending_target(&self) -> Option<usize> {
		match self.pending_target.swap(NO_TARGET, Ordering::SeqCst) {
			NO_TARGET => None,
			target => Some(target),
		}
	}

	/// 主线程内核栈顶。
	fn boot_kstack_top(&self) -> u64 {
		self.boot_kstack_top.load(Ordering::SeqCst)
	}

	/// 登记主线程内核栈顶。
	fn set_boot_kstack_top(&self, kstack_top: u64) {
		self.boot_kstack_top.store(kstack_top, Ordering::SeqCst);
	}

	/// 是否请求了系统调用重启。
	fn restart_requested(&self) -> bool {
		self.is_restart_requested.load(Ordering::SeqCst)
	}

	/// 登记系统调用重启请求。
	fn set_restart_requested(&self, is_requested: bool) {
		self.is_restart_requested.store(is_requested, Ordering::SeqCst);
	}

	/// 取走系统调用重启请求: 返回值并清除登记。
	fn take_restart_requested(&self) -> bool {
		self.is_restart_requested.swap(false, Ordering::SeqCst)
	}
}

static SCHED: SchedState = SchedState {
	current: AtomicUsize::new(0),
	pending_target: AtomicUsize::new(NO_TARGET),
	boot_kstack_top: AtomicU64::new(0),
	is_restart_requested: AtomicBool::new(false),
};

// ---------------------------------------------------------------
//  登记与查询
// ---------------------------------------------------------------

/// 登记启动阶段已分配并在使用的主线程内核栈顶 VA。
pub fn note_boot_kstack_top(kstack_top: u64) {
	SCHED.set_boot_kstack_top(kstack_top);
}

/// 登记主线程 (thread_id = 0) 并初始化调度状态。
///
/// 由 rust_main_after_mmu 在首次 sret 前调用一次。线程 0 的用户现场
/// (入口 sepc 与 sstatus) 以参数显式传入, 不再从 CSR 回读: 该现场 CSR 必须在
/// 全部 ecall 之后才写入 (理由见 rust_main_after_mmu 中的说明), 而本函数又必须
/// 在写入之前完成登记, 两者顺序互斥, 故改由调用方传递。
pub fn init_main_thread(user_sepc: u64, user_sstatus: u64) {
	THREADS.set(
		0,
		ThreadCtl {
			state: ThreadState::Running,
			proc: 0,
			kstack_top: SCHED.boot_kstack_top(),
			sepc: user_sepc,
			sstatus: user_sstatus,
			wait_addr: 0,
			thread_meta_info_addr: 0,
			clear_child_thread_id: 0,
		},
	);
	SCHED.set_current(0);
	SCHED.clear_pending_target();
	SCHED.set_restart_requested(false);
}

/// 返回当前线程的内部标识 (槽下标)。
pub fn current_thread_id() -> usize {
	SCHED.current()
}

/// 返回当前线程的对外线程标识 (槽下标 + 1, 恒非零)。
pub fn current_external_thread_id() -> u64 {
	(SCHED.current() + 1) as u64
}

/// 返回某线程陷阱帧基址 (其内核栈顶减帧大小)。
pub fn frame_base(thread_id: usize) -> u64 {
	THREADS.get(thread_id).kstack_top - TRAP_FRAME_SIZE
}

/// 是否有已登记的待切换目标 (供 trap.rs 判定是否走切换路径)。
pub fn switch_pending() -> bool {
	SCHED.pending_target().is_some()
}

/// 自 1 起找第一个空闲槽位 (槽 0 恒为主线程)。
fn next_free_slot() -> Option<usize> {
	(1..NUM_THREADS).find(|s| THREADS.get(*s).state == ThreadState::Free)
}

/// 自 after 起按槽位顺序找下一个可运行线程 (Round-robin, 不含自身)。
fn next_runnable(after: usize) -> Option<usize> {
	(1..NUM_THREADS)
		.map(|off| (after + off) % NUM_THREADS)
		.find(|s| THREADS.get(*s).state == ThreadState::Runnable)
}

/// 向用户地址写 32 位字 (S 模式以 sstatus.SUM = 1 直写 U 页)。
fn store_user_u32(addr: u64, val: u32) {
	unsafe { ptr::write_volatile(addr as *mut u32, val) }
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
pub fn clone_thread(gprs: &TrapGprs, flags: u64, stack: u64, ptid: u64, tls: u64, ctid: u64) -> u64 {
	// 未请求共享地址空间即为进程式 clone: 交由进程层复制地址空间。
	if proc::is_process_clone(flags) {
		return proc::clone_process(gprs, flags);
	}
	// 共享地址空间但不属同一线程组 (CLONE_VM 而缺 CLONE_THREAD) 的用法本运行时不支持。
	if (flags & (CLONE_VM | CLONE_THREAD)) != (CLONE_VM | CLONE_THREAD) {
		return EINVAL;
	}
	let child = match next_free_slot() {
		Some(c) => c,
		None => {
			return EAGAIN;
		}
	};
	// 首次占用的槽位新分配内核栈; 复用的槽位保留旧栈避免页池反复消耗。
	if THREADS.get(child).kstack_top == 0 {
		let top = mem::try_alloc_thread_kstack(THREAD_KSTACK_SIZE);
		if top == 0 {
			return ENOMEM;
		}
		THREADS.with(child, |ctl| ctl.kstack_top = top);
	}

	// 构造子线程陷阱帧: 先整体拷贝父帧, 再按 clone 语义改写个别槽。
	let dst = frame_base(child) as *mut TrapGprs;
	unsafe {
		ptr::copy_nonoverlapping(gprs as *const TrapGprs, dst, 1);
		let cf = &mut *dst;
		cf.regs[A0] = 0;
		if (flags & CLONE_SETTLS) != 0 {
			cf.regs[TP] = tls;
		}
		cf.usp = stack;
	}

	// 登记子线程控制块; 恢复现场取父线程此刻的 CSR (与父共享同一现场值)。
	// clear_child_thread_id 是线程退出时的清零并唤醒地址, 由 CHILD_CLEARTID 给出
	// (musl 的 __tl_unlock 延后提交, 见 pthread_create.c __pthread_exit 尾部注释)。
	let parent = SCHED.current();
	THREADS.with(child, |ctl| {
		ctl.state = ThreadState::Runnable;
		ctl.proc = THREADS.get(parent).proc;
		ctl.sepc = csr::read_sepc().wrapping_add(4);
		ctl.sstatus = csr::read_sstatus();
		ctl.wait_addr = 0;
		ctl.thread_meta_info_addr = if (flags & CLONE_SETTLS) != 0 { tls } else { gprs.tp() };
		ctl.clear_child_thread_id = if (flags & CLONE_CHILD_CLEARTID) != 0 { ctid } else { 0 };
	});
	// PARENT_SETTID: 父线程可经 ptid 读取子线程标识 (musl 的 new->tid)。
	if (flags & CLONE_PARENT_SETTID) != 0 {
		store_user_u32(ptid, (child + 1) as u32);
	}
	// CHILD_SETTID: 子线程自身可经 ctid 读取标识 (当前载荷不使用)。
	if (flags & CLONE_CHILD_SETTID) != 0 {
		store_user_u32(ctid, (child + 1) as u32);
	}

	// 子线程优先运行 (与 Linux 一致); 父线程现场由 finalize_switch 保存。
	SCHED.set_pending_target(child);
	(child + 1) as u64
}

// ---------------------------------------------------------------
//  退出 (93)
// ---------------------------------------------------------------

/// 终止当前线程 (exit 93)。
///
/// 线程组即进程: 组内还有其它线程时只释放本线程的槽位, 由调度器切到其它线程;
/// 本线程是组内最后一个线程时整个进程退出 (见 proc::exit_process), 该进程是启动
/// 进程时终止整个飞地。
pub fn exit_current_thread(code: u64) -> u64 {
	let cur = SCHED.current();
	let slot = THREADS.get(cur).proc;
	if live_thread_count(slot) <= 1 {
		proc::exit_process(slot, code);
		return 0;
	}
	// 清零 clear_child_thread_id 并唤醒一位等待者, 释放仍被本线程持有的
	// __thread_list_lock (musl 在 SYS_exit 前一直持有该锁, 见 __pthread_exit)。
	let ctid = THREADS.get(cur).clear_child_thread_id;
	if ctid != 0 {
		store_user_u32(ctid, 0);
		wake_blocked(ctid, 1);
	}
	release_slot(cur);
	// 有就绪线程则让出切换; 组内已无活线程则整飞地退出。
	match next_runnable(cur) {
		Some(next) => {
			SCHED.set_pending_target(next);
			0
		}
		None => ecall_aux::enclave_call_exit(code),
	}
}

/// 释放线程槽位并把控制块复位 (内核栈保留供复用)。
///
/// 一并复位本线程的等待定时器条目 (见 [timer::reset]): 槽位释放后没有调用方再取回该
/// 结果, 留下的条目会使定时器中断一直按它的截止时刻设定 stimecmp, 而该槽位被新线程
/// 复用时还会继承这个截止时刻。
fn release_slot(thread_id: usize) {
	THREADS.with(thread_id, |ctl| {
		ctl.state = ThreadState::Free;
		ctl.proc = 0;
		ctl.sepc = 0;
		ctl.sstatus = 0;
		ctl.wait_addr = 0;
		ctl.thread_meta_info_addr = 0;
		ctl.clear_child_thread_id = 0;
	});
	timer::reset(thread_id);
}

/// 统计某进程尚未释放的线程数 (含当前正在运行的线程)。
fn live_thread_count(slot: usize) -> u64 {
	(0..NUM_THREADS).filter(|s| is_live_thread_of(*s, slot)).count() as u64
}

/// 判断线程 *thread_id* 是否属于进程 *slot* 且尚未释放。
fn is_live_thread_of(thread_id: usize, slot: usize) -> bool {
	let ctl = THREADS.get(thread_id);
	ctl.state != ThreadState::Free && ctl.proc == slot
}

/// 释放某进程的全部线程 (进程退出时调用)。
pub fn free_process_threads(slot: usize) {
	for s in 0..NUM_THREADS {
		if is_live_thread_of(s, slot) {
			release_slot(s);
		}
	}
}

/// 把某进程全部阻塞的线程置回就绪: 有信号待投递时, 阻塞中的线程也必须回到可运行
/// 状态, 才能在其陷阱返回路径上取走信号。
pub fn wake_process(slot: usize) {
	for s in 0..NUM_THREADS {
		THREADS.with(s, |ctl| {
			if ctl.state == ThreadState::Blocked && ctl.proc == slot {
				ctl.state = ThreadState::Runnable;
				ctl.wait_addr = 0;
			}
		});
	}
}

// ---------------------------------------------------------------
//  让出 / 抢占 / set_tid_address
// ---------------------------------------------------------------

/// 主动让出 CPU (sched_yield 124): 有他线程就绪则登记切换目标。
pub fn sched_yield_current() {
	if let Some(next) = next_runnable(SCHED.current()) {
		SCHED.set_pending_target(next);
	}
}

/// STIP 时间片边界抢占 (trap.rs 中断分支调用): 有他线程就绪则轮转。
pub fn maybe_preempt() {
	if let Some(next) = next_runnable(SCHED.current()) {
		SCHED.set_pending_target(next);
	}
}

/// 以 gprs 为模板建立某进程的首个线程, 返回其槽位; 无空闲槽位返回 None。
///
/// 供进程式 clone 调用 (见 proc::clone_process)。子线程现场为父帧拷贝后只把 a0
/// 改写为 0 —— 子进程中 clone 返回 0; 用户 sp 由调用方传入, fork 语义下即父进程
/// 此刻的用户 sp。
pub fn spawn_process_thread(slot: usize, gprs: &TrapGprs, usp: u64) -> Option<usize> {
	let child = next_free_slot()?;
	// 首次占用的槽位新分配内核栈; 复用的槽位保留旧栈避免页池反复消耗。
	if THREADS.get(child).kstack_top == 0 {
		let top = mem::try_alloc_thread_kstack(THREAD_KSTACK_SIZE);
		if top == 0 {
			return None;
		}
		THREADS.with(child, |ctl| ctl.kstack_top = top);
	}

	let dst = frame_base(child) as *mut TrapGprs;
	unsafe {
		ptr::copy_nonoverlapping(gprs as *const TrapGprs, dst, 1);
		let cf = &mut *dst;
		cf.regs[A0] = 0;
		cf.usp = usp;
	}

	THREADS.with(child, |ctl| {
		ctl.state = ThreadState::Runnable;
		ctl.proc = slot;
		ctl.sepc = csr::read_sepc().wrapping_add(4);
		ctl.sstatus = csr::read_sstatus();
		ctl.wait_addr = 0;
		ctl.thread_meta_info_addr = gprs.tp();
		ctl.clear_child_thread_id = 0;
	});
	Some(child)
}

/// 登记待切换目标线程 (供进程层在 clone 后指定子线程优先运行)。
pub fn set_pending_target(thread_id: usize) {
	SCHED.set_pending_target(thread_id);
}

/// 登记切换到任一线程 (Round-robin, 不含自身), 返回是否找到。
///
/// 与 sched_yield_current 的区别在于返回值: 进程退出路径需据此判定本飞地是否
/// 还有可运行线程, 无则终止飞地。
pub fn switch_away() -> bool {
	match next_runnable(SCHED.current()) {
		Some(next) => {
			SCHED.set_pending_target(next);
			true
		}
		None => false,
	}
}

/// 请求本系统调用被重启: 恢复点取 ecall 自身, 且不写回 a0。
pub fn request_restart() {
	SCHED.set_restart_requested(true);
}

/// 取走重启请求。
pub fn take_restart() -> bool {
	SCHED.take_restart_requested()
}

/// 登记本线程的 clear_child_thread_id (set_tid_address 96), 返回本线程对外标识。
///
/// musl 主线程在 libc 初始化期以此登记 __thread_list_lock 地址, 并把返回值
/// 当作自身 tid (见 env/__init_tls.c), 故必须返回非零的线程标识。
pub fn set_tid_address(addr: u64) -> u64 {
	THREADS.with(SCHED.current(), |ctl| ctl.clear_child_thread_id = addr);
	current_external_thread_id()
}

// ---------------------------------------------------------------
//  阻塞队列原语 (futex 与线程退出释放 clear_child_thread_id 共用)
// ---------------------------------------------------------------

/// 阻塞当前线程于 wait_addr 并让出调度。
///
/// 四种返回情形, 返回值一律为单元, 由调用方重新判定等待条件:
///   - 本线程被唤醒: 唤醒方只把状态改回 Runnable, 不登记切换目标, 故此处
///     直接返回调用方, 由调用方重新判定条件是否已满足;
///   - 本飞地内另有可运行线程: 登记切换目标后返回, 由 trap.rs 完成切换;
///   - 本线程的截止时刻已到: 结束本次等待, 由调用方取回到期结果;
///   - 本飞地内已无任何可运行线程, 且表中没有尚未到期的截止时刻: 循环交还宿主。
///
/// 定时等待不在此处登记截止时刻: 截止时刻由 [timer] 设施持有 (见 concurrency/timer.rs),
/// 唤醒来源是定时器中断而非等待地址。
pub fn block_current(wait_addr: u64) {
	let cur = SCHED.current();
	THREADS.with(cur, |ctl| {
		ctl.state = ThreadState::Blocked;
		ctl.wait_addr = wait_addr;
	});
	loop {
		// 被唤醒即返回。该判定必须在选取切换目标之前: 本线程重新可运行后不再
		// 是等待者, 继续停在选取上会把唤醒丢失, 调用方无从重新判定条件。
		if !is_blocked(cur) {
			return;
		}
		if let Some(next) = next_runnable(cur) {
			SCHED.set_pending_target(next);
			return;
		}
		// 本线程的截止时刻已到而定时器中断尚未送达时, 与 trap.rs 的 STIP 分支
		// 同样地结束本次等待, 使等待不依赖中断的送达时机。
		if timer::is_expired(cur) {
			wake_thread(cur);
			return;
		}
		// 表中仍有尚未到期的截止时刻时不得交还宿主: 交还宿主时飞地的 stimecmp
		// 随上下文保存, 该时刻到期不再触发定时器中断, 发起等待的系统调用因此
		// 没有唤醒来源。此时留在本 hart 上, 由定时器中断结束等待。
		if timer::next_deadline().is_none() {
			// 属空闲让出, 不参与时间片轮转: 本飞地已无运行空间且无尚未到期的
			// 截止时刻, 交还宿主等待网卡事件。
			let _ = ecall_aux::enclave_call_suspend(ENCLAVE_SUSPEND_IDLE);
		}
	}
}

/// 判断线程 *thread_id* 当前是否处于阻塞状态。
fn is_blocked(thread_id: usize) -> bool {
	THREADS.get(thread_id).state == ThreadState::Blocked
}

/// 判断线程 *thread_id* 是否阻塞于等待地址 *wait_addr*。
fn is_blocked_on(thread_id: usize, wait_addr: u64) -> bool {
	let ctl = THREADS.get(thread_id);
	ctl.state == ThreadState::Blocked && ctl.wait_addr == wait_addr
}

/// 把线程 *thread_id* 由阻塞改为就绪并清除其等待地址。
fn make_runnable(thread_id: usize) {
	THREADS.with(thread_id, |ctl| {
		ctl.state = ThreadState::Runnable;
		ctl.wait_addr = 0;
	});
}

/// 把线程 *thread_id* 由阻塞改为就绪并清除其等待地址, 返回是否确有该线程处于阻塞。
///
/// 与按等待地址匹配的 [wake_blocked] 并列: 定时器中断按线程标识唤醒, 不经等待地址。
pub fn wake_thread(thread_id: usize) -> bool {
	if THREADS.get(thread_id).state != ThreadState::Blocked {
		return false;
	}
	make_runnable(thread_id);
	true
}

/// 唤醒至多 nr 位阻塞于 wait_addr 的线程, 返回实际唤醒数。
pub fn wake_blocked(wait_addr: u64, nr: u64) -> u64 {
	let mut woken: u64 = 0;
	for s in 0..NUM_THREADS {
		if woken >= nr {
			break;
		}
		if is_blocked_on(s, wait_addr) {
			make_runnable(s);
			woken += 1;
		}
	}
	woken
}

/// FUTEX_REQUEUE 内核动作: 先唤醒至多 nr_wake 位, 再把至多 nr_requeue 位
/// 仍阻塞于 wait_addr 的等待者迁移到 target_addr (保持阻塞)。返回唤醒数。
///
/// 供 cond 广播/移交把等待者从 cond 挂到互斥锁字上, 待锁释放时一并唤醒。
pub fn requeue_blocked(wait_addr: u64, nr_wake: u64, nr_requeue: u64, target_addr: u64) -> u64 {
	let woken = wake_blocked(wait_addr, nr_wake);
	let mut requeued: u64 = 0;
	for s in 0..NUM_THREADS {
		if requeued >= nr_requeue {
			break;
		}
		if is_blocked_on(s, wait_addr) {
			THREADS.with(s, |ctl| ctl.wait_addr = target_addr);
			requeued += 1;
		}
	}
	woken
}

// ---------------------------------------------------------------
//  切换收尾
// ---------------------------------------------------------------

/// 完成一次切换: 先记录当前线程的恢复现场, 再激活已登记的切换目标。
///
/// resume_sepc 为当前线程下次恢复的用户 PC (系统调用为 ecall + 4, 中断为
/// 被中断的 pc)。返回目标线程帧基址 (entry.s 据此 mv sp, a0); 无目标返回 0。
pub fn finalize_switch(resume_sepc: u64) -> u64 {
	let cur = SCHED.current();
	// 当前线程存活则保存现场 (已 Free 的退出线程无需保存)。
	if THREADS.get(cur).state != ThreadState::Free {
		THREADS.with(cur, |ctl| {
			ctl.sepc = resume_sepc;
			ctl.sstatus = csr::read_sstatus();
			if ctl.state == ThreadState::Running {
				ctl.state = ThreadState::Runnable;
			}
		});
	}
	match SCHED.take_pending_target() {
		Some(target) => {
			THREADS.with(target, |ctl| ctl.state = ThreadState::Running);
			SCHED.set_current(target);
			// 目标线程属另一进程时切换地址空间. 必须在写完 sepc/sstatus 之前
			// 完成: 此后不再有对本地址空间的访问, 而目标帧位于共享的 S 模式
			// 内核栈上, 切换后仍可达.
			let target_ctl = THREADS.get(target);
			proc::activate(target_ctl.proc);
			csr::write_sepc(target_ctl.sepc);
			csr::write_sstatus(target_ctl.sstatus);
			frame_base(target)
		}
		None => 0,
	}
}

// ---------------------------------------------------------------
//  线程类系统调用粘合
// ---------------------------------------------------------------
//
// 只做寄存器参数到本模块与 futex 模块原语的搬运; 返回值为要写入 a0 的值,
// 负 errno 以无符号补码表示。

/// clone (220): 创建共享地址空间线程, 或按进程式标志复制地址空间建立新进程。
/// 参数 (Linux rv64): flags = a0, stack = a1, ptid = a2, tls = a3, ctid = a4。
pub fn clone_handler(gprs: &TrapGprs, flags: u64, stack: u64, ptid: u64, tls: u64, ctid: u64) -> u64 {
	clone_thread(gprs, flags, stack, ptid, tls, ctid)
}

/// exit (93): 终止当前线程。本线程是组内最后一个线程时整个进程退出。
pub fn exit_handler(code: u64) -> u64 {
	exit_current_thread(code)
}

/// gettid (178): 返回当前线程的对外标识。
pub fn gettid_handler() -> u64 {
	current_external_thread_id()
}

/// sched_yield (124): 主动让出 CPU。若存在其它可运行线程, 由 trap.rs 完成切换。
pub fn sched_yield_handler() -> u64 {
	sched_yield_current();
	0
}

/// set_tid_address (96): 登记线程退出时的清零并唤醒地址, 返回当前线程对外标识。
pub fn set_tid_address_handler(addr: u64) -> u64 {
	set_tid_address(addr)
}

/// futex (98): 参数 (Linux rv64) 为 uaddr = a0, op = a1, val = a2,
/// timeout = a3, uaddr2 = a4。REQUEUE 以 a3 作 nr_requeue。
pub fn futex_handler(uaddr: u64, op: u64, val: u64, arg3: u64, uaddr2: u64) -> u64 {
	futex::futex(uaddr, op, val, arg3, uaddr2)
}

/// nanosleep (101): 参数为 req = a0 (指向 const struct timespec), rem = a1。
///
/// 睡眠时长交给 [timer] 设施, 故等待 5 ms 的调用在 5 ms 后唤醒, 不受时间片周期限制。
/// *rem* 不写回: 本运行时的阻塞系统调用在唤醒后重启整个调用 (见 trap.rs 的
/// 取用 restart), 不向载荷返回 EINTR, 没有剩余时长需要报告。
///
/// 时长为 0 或为负时不进入阻塞, 立即返回 0; 指针为 0 返回 EFAULT, 时长字段越界
/// (tv_sec 为负或 tv_nsec 不在 [0, 1_000_000_000) 内) 返回 EINVAL。
pub fn nanosleep_handler(req: u64, rem: u64) -> u64 {
	let _ = rem;
	if req == 0 {
		return EFAULT;
	}
	let ts = unsafe { ptr::read_unaligned(req as *const Timespec) };
	if ts.tv_sec < 0 || ts.tv_nsec < 0 || ts.tv_nsec >= 1_000_000_000 {
		return EINVAL;
	}
	let nsec = (ts.tv_sec as u64).saturating_mul(1_000_000_000).saturating_add(ts.tv_nsec as u64);

	let me = current_thread_id();
	// 被中断后重新执行本调用时沿用表中已有的截止时刻, 不重新计一段完整的时长。
	if !timer::is_armed(me) {
		timer::arm_after(me, nsec);
	}
	loop {
		if timer::is_expired(me) {
			timer::reset(me);
			return 0;
		}
		block_current(SLEEP_WAIT_ADDR);
		// 有切换目标时退回 ecall 之前重新执行本次调用, 使睡眠期间让出的线程在别处
		// 运行; 无切换目标说明是被定时器唤醒而来不及登记切换, 重新检查到期。
		if switch_pending() {
			request_restart();
			return 0;
		}
	}
}

#[cfg(test)]
pub mod tests {
	use super::*;

	/// 把线程表与调度状态复位: 槽位全部空闲, 当前线程为主线程, 已登记的调度目标与重启
	/// 请求清除。主线程的内核栈顶取 *main_kstack_top*。
	///
	/// 线程表、调度目标与重启请求都是进程内的全局量, 飞地内由 SIE = 0 保证串行, 主机侧
	/// 用例须先取得入口的全局锁。
	pub fn reset_scheduler(main_kstack_top: u64) {
		THREADS.reset();
		SCHED.set_boot_kstack_top(main_kstack_top);
		init_main_thread(0, 0);
	}

	/// 清除已登记的调度目标。
	///
	/// 用例以它模拟「被重启的系统调用重新执行」这一步: 本次切换已由引擎完成, 下一次进入
	/// 调用时 [switch_pending] 为假。
	pub fn clear_switch() {
		SCHED.clear_pending_target();
	}

	/// 判断线程 *thread_id* 当前是否处于阻塞状态。
	pub fn is_blocked(thread_id: usize) -> bool {
		super::is_blocked(thread_id)
	}
}
