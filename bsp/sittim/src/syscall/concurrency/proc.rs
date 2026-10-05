//! 进程模型: 进程表、独立地址空间、父子关系与等待, 以及间隔定时器。
//!
//! 本模块同时承载进程类与信号类系统调用的粘合层: 只做寄存器参数到本模块原语的
//! 搬运, 与线程类系统调用落在 thread.rs 的分工一致。
//!
//! 与 `thread` 模块的分工: 线程表 (槽位、内核栈、就绪队列、切换) 在 thread.rs,
//! 本模块只维护进程级状态 —— 地址空间根表、父子关系与退出状态、间隔定时器。
//! 信号自身的状态与信号帧在 sig.rs, 本模块只做投递决策与进程终止。
//! 每个线程经 `ThreadCtl::proc` 归属一个进程, 线程组即进程。
//!
//! 调度模型
//! - 进程表是 BSS 中的定长数组, 槽下标即内部进程标识, 0 为启动进程。对用户可见
//!   的进程标识 (getpid) 自 1 起单调递增, 与槽下标无关: 槽位会被回收复用, 而
//!   载荷会把 pid 当作长期有效的键 (stress-ng 以 getpid() 判定自己是父进程还是
//!   子进程)。
//! - 地址空间之间共享 S 模式的那一半页表 (管理器 VA 区段、线性窗口、内核栈与页
//!   池), U 模式的那一半在进程式 clone 时逐页复制, 见 paging::fork_address_space。
//!   进程切换时由 thread::finalize_switch 经 activate 写 satp 完成地址空间切换。
//! - 间隔定时器 (ITIMER_REAL) 的检查在每次定时器中断时进行, 且遍历全部进程而不
//!   只当前进程 —— 父进程的 SIGALRM 常在子进程正在运行时到期。

use core::ptr;

use crate::syscall::concurrency::sig::{ self, SigState };
use crate::syscall::concurrency::thread;
use crate::syscall::{ EAGAIN, ECHILD, EINVAL, ENOMEM, ESRCH };
use crate::constants::{ CHUNK_2M_PAGES, PAGE_SHIFT, TIMER_FREQ };
use crate::mem_prim::align::align_up;
use crate::csr;
use crate::ecall_aux;
use crate::paging;
use crate::trap::{ TrapGprs, A0 };

/// 进程表容量。启动进程占用槽位 0。
pub const NUM_PROCS: usize = 8;

/// 无父进程 (启动进程) 的 parent 取值。
const NO_PARENT: usize = !0;

/// wait4 的阻塞地址基址。该地址与 futex 的等待地址同属一类: 只参与唤醒匹配,
/// 内核从不解引用。每个进程一个, 父进程等待子进程与子进程退出唤醒父进程使用
/// 同一个值。
const WAIT_ADDR_BASE: u64 = 0x7fff_0000_0000;

// clone 标志位中与进程式 clone 有关的取值 (Linux 通用定义子集)。
const CLONE_VM: u64 = 0x0000_0100;
const CLONE_VFORK: u64 = 0x0000_4000;

// ---------------------------------------------------------------
//  进程控制块
// ---------------------------------------------------------------

/// 进程控制块。所有字段都在 BSS 中, 跨 SUSPEND/RESUME 存活。
#[derive(Clone, Copy)]
pub struct ProcCtl {
	/// 槽位是否占用。退出后仍为 true, 直到被父进程回收。
	used: bool,
	/// 对用户可见的进程标识, 自 1 起单调递增。
	pid: u64,
	/// 父进程的对用户可见标识; 启动进程为 0。
	ppid: u64,
	/// 已退出待回收, 此时不再有线程被调度。
	exited: bool,
	/// 退出码。
	exit_code: u64,
	/// 页表根物理地址。
	root_pa: u64,
	/// 父进程槽下标; NO_PARENT 表示无父进程。
	parent: usize,
	/// 信号状态 (处置表、屏蔽集合、待投递集合)。
	sig: SigState,
	/// 间隔定时器的下次到期时刻 (mtime 计数值); 0 表示未装载。
	itimer_next: u64,
	/// 间隔定时器的重装周期 (mtime 计数值); 0 表示单次。
	itimer_interval: u64,
	/// 本应用地址空间的内存上下文。
	mem_ctx: TaMemCtx,
}

/// 一个可信应用的内存上下文: 其地址空间上四个推进位置的取值。
///
/// 四个取值随地址空间走而不随线程走。进程式 clone 之后父子进程各自继续调用 brk 与
/// mmap, 共享同一组取值会使一方的映射落在另一方已经用过的虚拟地址上, 或使双方从
/// 同一段物理内存取得后备。正在运行的那个应用的一份在 EnclaveContext 中, 其余应用
/// 的一份随进程控制块保存, 由 activate 在进程切换时换入换出。
#[derive(Clone, Copy)]
pub struct TaMemCtx {
	/// U 模式堆顶, 即 brk 的返回值。
	pub heap_top: u64,
	/// U 模式堆已映射到的虚拟地址上界, 按 2 MiB 对齐。
	pub heap_mapped_end: u64,
	/// 映射区已交付的 4 KiB 页计数, 自映射区基址起算。
	pub mmap_pages_used: u64,
	/// 映射区当前 2 MiB 分块的物理基址; 0 表示当前分块未取后备内存。
	pub curr_mmap_pa: u64,
}

impl ProcCtl {
	const fn free() -> Self {
		Self {
			used: false,
			pid: 0,
			ppid: 0,
			exited: false,
			exit_code: 0,
			root_pa: 0,
			parent: NO_PARENT,
			sig: SigState::new(),
			itimer_next: 0,
			itimer_interval: 0,
			mem_ctx: TaMemCtx {
				heap_top: 0,
				heap_mapped_end: 0,
				mmap_pages_used: 0,
				curr_mmap_pa: 0,
			},
		}
	}
}

static mut PROCS: [ProcCtl; NUM_PROCS] = [ProcCtl::free(); NUM_PROCS];

/// 当前进程槽下标。
static mut CURRENT_PROC: usize = 0;

/// 下一个对外进程标识。
static mut NEXT_PID: u64 = 1;

/// rt_sigreturn 恢复出的 sepc, 由 trap.rs 在系统调用返回后取走。
static mut SIGRETURN_SEPC: Option<u64> = None;

// ---------------------------------------------------------------
//  登记与查询
// ---------------------------------------------------------------

/// 登记启动进程 (槽位 0) 及其地址空间。
///
/// 由 rust_main_after_mmu 在首次 sret 前调用。启动页表根不是从 S 模式页池分配的,
/// 其可解引用地址是 `context::root_pa()` 在当前映射下取到的值, 物理地址则从 satp
/// 反解 —— 两者不能由同一个表达式得到, 故在此一并登记。
pub fn init_main_process() {
	let satp = csr::read_satp();
	let root_pa = (satp & 0xf_ffff_ffff) << PAGE_SHIFT;
	let root_va = crate::context::root_pa();
	paging::set_root(root_va);
	unsafe {
		PROCS[0] = ProcCtl::free();
		PROCS[0].used = true;
		PROCS[0].pid = 1;
		PROCS[0].ppid = 0;
		PROCS[0].root_pa = root_pa;
		PROCS[0].parent = NO_PARENT;
		CURRENT_PROC = 0;
		NEXT_PID = 2;
		SIGRETURN_SEPC = None;
	}
}

/// 取当前进程的信号状态。
fn current_sig() -> &'static mut SigState {
	unsafe {
		let slot = CURRENT_PROC;
		&mut *core::ptr::addr_of_mut!(PROCS[slot].sig)
	}
}

/// 当前进程槽下标。
#[inline]
pub fn current_slot() -> usize {
	unsafe { CURRENT_PROC }
}

/// 当前进程的对用户可见标识 (getpid 172)。
pub fn getpid() -> u64 {
	unsafe { PROCS[CURRENT_PROC].pid }
}

/// 当前进程的父进程标识 (getppid 173)。
pub fn getppid() -> u64 {
	unsafe { PROCS[CURRENT_PROC].ppid }
}

/// 按对用户可见的标识查找进程槽位。
fn slot_of_pid(pid: u64) -> Option<usize> {
	unsafe {
		for s in 0..NUM_PROCS {
			let p = &*core::ptr::addr_of!(PROCS[s]);
			if p.used && p.pid == pid {
				return Some(s);
			}
		}
	}
	None
}

/// 自 1 起找第一个空闲进程槽位。
fn next_free_proc_slot() -> Option<usize> {
	unsafe {
		for s in 1..NUM_PROCS {
			let p = &*core::ptr::addr_of!(PROCS[s]);
			if !p.used {
				return Some(s);
			}
		}
	}
	None
}

/// 进程的 wait4 阻塞地址。
#[inline]
fn wait_addr(slot: usize) -> u64 {
	WAIT_ADDR_BASE + (slot as u64) * 8
}

// ---------------------------------------------------------------
//  地址空间切换
// ---------------------------------------------------------------

/// 正在运行的可信应用的内存上下文。
///
/// 取 EnclaveContext 中的实时取值。进程控制块中的同名副本由 save_cursors 在应用被
/// 切换走时写入, 正在运行的应用在其中的副本停留在它上一次被切换走时的取值 ——
/// 启动应用自引导起从未被切换走, 副本仍是初值。
fn current_ta_mem_ctx() -> TaMemCtx {
	let ctx = crate::context::ctx();
	TaMemCtx {
		heap_top: ctx.umode_heap_top,
		heap_mapped_end: ctx.umode_heap_mapped_end,
		mmap_pages_used: ctx.umode_mmap_pages_used,
		curr_mmap_pa: ctx.umode_curr_mmap_pa,
	}
}

/// fork 之后子进程的内存上下文。
///
/// 堆的两个取值直接继承: 子进程的堆页是父进程那些页的副本, 双方此后的 brk 扩堆各自
/// 经 alloc_umode_page 取新的物理页, 同一虚拟地址在两进程中对应不同的物理页。
///
/// 映射区的两个取值须调整。curr_mmap_pa 指向父进程映射区当前分块的物理内存, 而该
/// 分块的页已按父进程的映射区虚拟地址段交出; 子进程若沿用, 双方的下一次小块映射会
/// 从同一物理地址取后备内存并映射到相同的虚拟地址, 匿名映射因此不再私有。子进程的
/// 页计数推进到下一个 2 MiB 分块的边界并清空分块基址, 使其另取后备内存; 代价是当前
/// 分块剩余的虚拟地址空间被放弃, 与请求达到一整块时 (见 mem::take_mmap_region) 的
/// 处理一致, 无物理代价。
fn forked_mem_ctx(parent: TaMemCtx) -> TaMemCtx {
	TaMemCtx {
		heap_top: parent.heap_top,
		heap_mapped_end: parent.heap_mapped_end,
		mmap_pages_used: if parent.curr_mmap_pa == 0 {
			parent.mmap_pages_used
		} else {
			align_up(parent.mmap_pages_used, CHUNK_2M_PAGES)
		},
		curr_mmap_pa: 0,
	}
}

/// 把正在运行的应用的内存上下文保存进其进程控制块。
fn save_cursors(slot: usize) {
	unsafe {
		PROCS[slot].mem_ctx = current_ta_mem_ctx();
	}
}

/// 把进程控制块中的内存上下文装入 EnclaveContext, 使其成为正在运行的应用的取值。
fn load_cursors(slot: usize) {
	let mem_ctx = unsafe { PROCS[slot].mem_ctx };
	let ctx = crate::context::ctx_mut();
	ctx.umode_heap_top = mem_ctx.heap_top;
	ctx.umode_heap_mapped_end = mem_ctx.heap_mapped_end;
	ctx.umode_mmap_pages_used = mem_ctx.mmap_pages_used;
	ctx.umode_curr_mmap_pa = mem_ctx.curr_mmap_pa;
}

/// 切换到某进程的地址空间。
///
/// 由 thread::finalize_switch 在目标线程属于另一个进程时调用。两个地址空间共享
/// S 模式的全部映射 (运行时自身段、线性窗口、页池、内核栈), 故切换 satp 后仍在
/// 执行的陷阱处理代码、正要返回的陷阱帧与页表根自身都依然可达。
pub fn activate(slot: usize) {
	let from = unsafe { CURRENT_PROC };
	if from == slot {
		return;
	}
	save_cursors(from);
	load_cursors(slot);

	let root_pa = unsafe { PROCS[slot].root_pa };
	paging::set_root(paging::pool_root_va(root_pa));
	csr::write_satp(paging::init_satp(root_pa));
	// satp 的 ASID 恒为 0, 硬件不区分地址空间, 切换后必须整体刷新 TLB。
	unsafe {
		core::arch::asm!("sfence.vma");
	}
	unsafe {
		CURRENT_PROC = slot;
	}
}

// ---------------------------------------------------------------
//  进程式 clone (fork)
// ---------------------------------------------------------------

/// clone 是否为进程式: 未请求共享地址空间。
///
/// musl 的 fork 与 _Fork 都调用 `clone(SIGCHLD, 0)`, 标志位里没有 CLONE_VM;
/// 线程创建 pthread_create 则必定带 CLONE_VM | CLONE_THREAD。
#[inline]
pub fn is_process_clone(flags: u64) -> bool {
	(flags & CLONE_VM) == 0
}

/// 进程式 clone: 复制当前地址空间, 建立新进程并把当前线程现场作为其首个线程。
///
/// 参数与线程式 clone 相同 (Linux rv64 clone): flags = a0, stack = a1, ptid = a2,
/// tls = a3, ctid = a4。fork 语义下 stack 为 0, 子进程沿用父进程的用户 sp; 子线程
/// 现场为父帧拷贝, 仅把 a0 改写为 0 (子进程中 clone 返回 0)。
///
/// 返回新进程的对用户可见标识; 失败返回负 errno。
pub fn clone_process(gprs: &TrapGprs, flags: u64) -> u64 {
	// CLONE_VFORK 要求父进程阻塞至子进程 exec 或退出, 且总是与 CLONE_VM 同时出现
	// (musl 的 vfork 即 0x100 | 0x4000 | SIGCHLD)。本运行时未实现「共享地址空间
	// 加父进程阻塞」这一组合, 明确拒绝而不退化处理, 以免调用方在共享内存的前提下
	// 按独立副本推断行为。
	if (flags & CLONE_VFORK) != 0 {
		return EINVAL;
	}

	let child_slot = match next_free_proc_slot() {
		Some(s) => s,
		None => {
			return EAGAIN;
		}
	};

	let child_root = paging::fork_address_space();
	if child_root == !0u64 {
		return ENOMEM;
	}

	let child_thread = match thread::spawn_process_thread(child_slot, gprs, gprs.usp) {
		Some(t) => t,
		None => {
			return EAGAIN;
		}
	};

	// 子进程的内存上下文取自正在运行的父进程, 不取自父进程控制块中的副本: 该副本
	// 只在父进程被切换走时写入, 而 fork 可以发生在父进程从未被切换走的时候。
	let child_mem_ctx = forked_mem_ctx(current_ta_mem_ctx());

	unsafe {
		let parent_slot = CURRENT_PROC;
		let parent_pid = PROCS[parent_slot].pid;
		let mut child = PROCS[parent_slot];
		child.used = true;
		child.exited = false;
		child.exit_code = 0;
		child.pid = NEXT_PID;
		NEXT_PID += 1;
		child.ppid = parent_pid;
		child.root_pa = child_root;
		child.parent = parent_slot;
		child.sig = PROCS[parent_slot].sig.inherited_from();
		// 间隔定时器不随 fork 继承 (Linux 的 ITIMER_REAL 亦不继承)。
		child.itimer_next = 0;
		child.itimer_interval = 0;
		child.mem_ctx = child_mem_ctx;
		let child_pid = child.pid;
		PROCS[child_slot] = child;

		// 子进程先运行 (与 Linux 一致); 父进程现场由 finalize_switch 保存。
		thread::set_pending_target(child_thread);
		child_pid
	}
}

// ---------------------------------------------------------------
//  退出与回收
// ---------------------------------------------------------------

/// 终止整个进程 (exit_group 94, 以及信号默认处置为终止时)。
///
/// 启动进程退出即整个飞地退出。其余进程转为已退出待回收: 其全部线程被释放,
/// 地址空间不再被调度, 退出码保留到父进程 wait4 回收。子进程的物理页与页表不在
/// 此归还 —— 本运行时没有物理页回收路径, 页池只前进不回退, 释放线程槽位即代表
/// 该进程不再占用调度资源。
pub fn exit_process(slot: usize, code: u64) {
	if slot == 0 {
		ecall_aux::enclave_call_exit(code);
	}

	let parent = unsafe { PROCS[slot].parent };
	unsafe {
		PROCS[slot].exited = true;
		PROCS[slot].exit_code = code;
		PROCS[slot].sig.clear_pending();
		PROCS[slot].itimer_next = 0;
		PROCS[slot].itimer_interval = 0;
	}
	// 若当前正在运行的就是本进程的线程, 其地址空间必须换出: 换出动作由随后的
	// 进程切换完成, 这里先把本进程的线程全部释放, 使调度器不会再次选中它们。
	thread::free_process_threads(slot);

	// 唤醒父进程: 父进程可能正阻塞在 wait4 上等待本进程退出。
	if parent != NO_PARENT {
		thread::wake_blocked(wait_addr(parent), 1);
	}

	if !thread::switch_away() {
		// 组内已无任何可运行线程: 飞地没有继续执行的主体, 终止飞地。
		ecall_aux::enclave_call_exit(code);
	}
}

/// 终止当前线程所属的整个进程 (exit_group 94)。
pub fn exit_group_current(code: u64) -> u64 {
	exit_process(unsafe { CURRENT_PROC }, code);
	0
}

/// wait4 (260)。
///
/// 参数: pid = a0, status = a1, options = a2, rusage = a3。回收一个已退出的
/// 子进程并回写其退出状态; 无已退出的子进程时, 若仍有存活子进程则阻塞等待,
/// WNOHANG 置位时立即返回 0; 完全没有子进程时返回 -ECHILD。
///
/// 阻塞经线程的等待地址实现: 本线程置为阻塞后调度到子进程, 子进程退出时按父进程
/// 的等待地址唤醒。被唤醒后需要重新扫描子进程, 故本系统调用在登记切换的同时请求
/// 重启 (thread::request_restart), 唤醒后从 ecall 处重新执行。
pub fn wait4(pid: u64, status: u64, options: u64) -> u64 {
	/// WNOHANG (musl bits/waitflags.h)。
	const WNOHANG: u64 = 1;
	/// pid 取该值表示等待任意子进程。
	const WAIT_ANY: u64 = !0u64;

	let parent = unsafe { CURRENT_PROC };
	loop {
		let mut has_child = false;
		let mut zombie: Option<usize> = None;
		for s in 1..NUM_PROCS {
			let p = unsafe { &*core::ptr::addr_of!(PROCS[s]) };
			if !p.used || p.parent != parent {
				continue;
			}
			has_child = true;
			let wanted = pid == WAIT_ANY || pid == 0 || pid == p.pid;
			if p.exited && wanted {
				zombie = Some(s);
				break;
			}
		}

		if let Some(s) = zombie {
			let (child_pid, code) = unsafe { (PROCS[s].pid, PROCS[s].exit_code) };
			if status != 0 {
				// 正常退出: 退出码置于状态字的高字节 (wait 状态字约定)。
				let word = ((code & 0xff) << 8) as u32;
				unsafe {
					ptr::write_volatile(status as *mut u32, word);
				}
			}
			// 回收槽位; 至此该进程不再出现在进程表中。
			unsafe {
				PROCS[s].used = false;
			}
			return child_pid;
		}

		if !has_child {
			return ECHILD;
		}
		if (options & WNOHANG) != 0 {
			return 0;
		}

		thread::block_current(wait_addr(parent));
		if thread::switch_pending() {
			// 已调度到子进程; 唤醒后必须重新扫描, 故请求从 ecall 处重启本调用。
			thread::request_restart();
			return 0;
		}
	}
}

// ---------------------------------------------------------------
//  信号: 处置、屏蔽与投递
// ---------------------------------------------------------------

/// rt_sigaction (134)。
pub fn sigaction(sig_num: u64, new_action: u64, old_action: u64) -> u64 {
	sig::action(current_sig(), sig_num, new_action, old_action)
}

/// rt_sigprocmask (135)。
pub fn sigprocmask(how: u64, set: u64, oldset: u64, sigsetsize: u64) -> u64 {
	sig::procmask(current_sig(), how, set, oldset, sigsetsize)
}

/// kill (129) / tkill (130) / tgkill (131): 把信号置入目标进程的待投递集合。
///
/// pid 为目标进程的对用户可见标识; 0 与自身标识都指向当前进程。信号 0 只做存在性
/// 检查, 不投递。信号编号超出登记范围返回 EINVAL。
pub fn kill(pid: u64, sig: u64) -> u64 {
	if sig == 0 {
		return 0;
	}
	if sig >= (sig::NSIG as u64) {
		return EINVAL;
	}
	let cur = unsafe { CURRENT_PROC };
	let target = if pid == 0 || pid == (unsafe { PROCS[cur].pid }) {
		cur
	} else {
		match slot_of_pid(pid) {
			Some(s) => s,
			None => {
				return ESRCH;
			}
		}
	};

	unsafe {
		sig::post(&mut PROCS[target].sig, sig as usize);
	}
	// 目标进程若有线程正阻塞, 需要让它回到可运行状态, 才能在其陷阱返回路径上
	// 取走信号。
	if target != cur {
		thread::wake_process(target);
	}
	0
}

/// 取出当前进程一个可投递的信号并构造信号帧, 返回处理函数地址 (即恢复时应写入
/// sepc 的值); 不需要投递时返回 None。
///
/// 由 trap.rs 在返回 U 模式之前调用。
pub fn deliver_pending(gprs: &mut TrapGprs, sepc: u64) -> Option<u64> {
	let slot = unsafe { CURRENT_PROC };
	let state = current_sig();
	let sig_num = sig::take_pending(state)?;
	match sig::disposition(state, sig_num) {
		sig::Disposition::Ignore => None,
		sig::Disposition::Terminate => {
			exit_process(slot, sig::SIGNAL_EXIT_OFFSET + (sig_num as u64));
			None
		}
		sig::Disposition::Handle(handler) => { Some(sig::build_frame(state, sig_num, handler, gprs, sepc)) }
	}
}

/// rt_sigreturn (139): 从当前用户 sp 指向的信号帧恢复现场。
///
/// 恢复出的 sepc 经 SIGRETURN_SEPC 交给 trap.rs —— 本系统调用返回后不能按常规
/// 路径把 sepc 推进到 ecall 的下一条指令, 而要回到信号送达时被中断的位置。
pub fn sigreturn(gprs: &mut TrapGprs) -> u64 {
	let sepc = sig::restore_frame(current_sig(), gprs);
	unsafe {
		SIGRETURN_SEPC = Some(sepc);
	}
	gprs.regs[A0]
}

/// 取走 rt_sigreturn 恢复出的 sepc。
pub fn take_sigreturn() -> Option<u64> {
	unsafe {
		let v = SIGRETURN_SEPC;
		SIGRETURN_SEPC = None;
		v
	}
}

// ---------------------------------------------------------------
//  间隔定时器 (setitimer 103)
// ---------------------------------------------------------------

/// mtime 当前计数值 (自由运行计数器, 频率见 config.mk 的 TIMER_FREQ)。
#[inline]
fn read_mtime() -> u64 {
	let v: u64;
	unsafe {
		core::arch::asm!("csrr {0}, time", out(reg) v);
	}
	v
}

/// 由秒与微秒折算 mtime 计数值。
#[inline]
fn to_ticks(sec: u64, usec: u64) -> u64 {
	sec.saturating_mul(TIMER_FREQ) + usec.saturating_mul(TIMER_FREQ) / 1_000_000
}

/// 由 mtime 计数值折算秒与微秒。
#[inline]
fn from_ticks(ticks: u64) -> (u64, u64) {
	(ticks / TIMER_FREQ, ((ticks % TIMER_FREQ) * 1_000_000) / TIMER_FREQ)
}

/// setitimer (103): 参数为 which = a0, new_value = a1, old_value = a2。
///
/// 只实现 ITIMER_REAL (which = 0), 到期投递 SIGALRM; ITIMER_VIRTUAL 与
/// ITIMER_PROF 按无操作处理 (返回 0 而不装载), 因为本运行时不区分用户态与内核态
/// 的执行时间。new_value 与 old_value 都指向 struct itimerval
/// (it_interval.tv_sec, it_interval.tv_usec, it_value.tv_sec, it_value.tv_usec),
/// 每栏 8 字节; it_value 为零表示撤销定时器。
///
/// musl 的 alarm() 即经本接口实现, 故此处同时是 alarm 的实现。
pub fn setitimer(which: u64, new_value: u64, old_value: u64) -> u64 {
	/// ITIMER_REAL。
	const ITIMER_REAL: u64 = 0;

	let now = read_mtime();
	unsafe {
		let slot = CURRENT_PROC;
		let p = &mut *core::ptr::addr_of_mut!(PROCS[slot]);
		if old_value != 0 {
			let (isec, iusec) = from_ticks(p.itimer_interval);
			let (rsec, rusec) = from_ticks(p.itimer_next.saturating_sub(now));
			ptr::write_volatile(old_value as *mut u64, isec);
			ptr::write_volatile((old_value + 8) as *mut u64, iusec);
			ptr::write_volatile((old_value + 16) as *mut u64, rsec);
			ptr::write_volatile((old_value + 24) as *mut u64, rusec);
		}
		if which != ITIMER_REAL {
			return 0;
		}
		if new_value != 0 {
			let isec = ptr::read_volatile(new_value as *const u64);
			let iusec = ptr::read_volatile((new_value + 8) as *const u64);
			let vsec = ptr::read_volatile((new_value + 16) as *const u64);
			let vusec = ptr::read_volatile((new_value + 24) as *const u64);
			let value = to_ticks(vsec, vusec);
			if value == 0 {
				p.itimer_next = 0;
				p.itimer_interval = 0;
			} else {
				p.itimer_next = now + value;
				p.itimer_interval = to_ticks(isec, iusec);
			}
		}
	}
	0
}

/// 让某进程的间隔定时器到达下一次到期。返回本次是否投递了 SIGALRM。
fn advance_itimer(slot: usize, now: u64) -> bool {
	unsafe {
		let p = &mut *core::ptr::addr_of_mut!(PROCS[slot]);
		if !p.used || p.exited || p.itimer_next == 0 || now < p.itimer_next {
			return false;
		}
		p.itimer_next = if p.itimer_interval != 0 { p.itimer_next + p.itimer_interval } else { 0 };
		sig::post(&mut p.sig, sig::SIGALRM);
		true
	}
}

/// 定时器中断的间隔定时器检查 (trap.rs 的 STIP 分支调用)。
///
/// 遍历全部进程而不只当前进程: 父进程的 SIGALRM 常在子进程正在运行时到期, 只检查
/// 当前进程会使父进程的定时器永不触发。
pub fn tick() {
	let now = read_mtime();
	for s in 0..NUM_PROCS {
		if advance_itimer(s, now) {
			// 该进程若有线程正阻塞, 需要让它回到可运行状态, 才能在其陷阱返回路径
			// 上取走信号。
			thread::wake_process(s);
		}
	}
}

// ---------------------------------------------------------------
//  进程类与信号类系统调用粘合
// ---------------------------------------------------------------
//
// 只做寄存器参数到本模块原语的搬运; 返回值为要写入 a0 的值, 负 errno 以无符号
// 补码表示。

/// exit_group (94): 终止当前线程所属的整个进程。该进程是启动进程时终止整个飞地。
/// musl 的 _exit 与 exit 都归到本调用。
pub fn exit_group_handler(code: u64) -> u64 {
	exit_group_current(code)
}

/// getpid (172): 返回当前进程的对用户可见标识。
pub fn getpid_handler() -> u64 {
	getpid()
}

/// getppid (173): 返回当前进程的父进程标识。
pub fn getppid_handler() -> u64 {
	getppid()
}

/// wait4 (260): 参数 (Linux rv64) 为 pid = a0, status = a1, options = a2,
/// rusage = a3。rusage 本运行时不写。
pub fn wait4_handler(pid: u64, status: u64, options: u64, _rusage: u64) -> u64 {
	wait4(pid, status, options)
}

/// setitimer (103): 参数 (Linux rv64) 为 which = a0, new_value = a1,
/// old_value = a2。
pub fn setitimer_handler(which: u64, new_value: u64, old_value: u64) -> u64 {
	setitimer(which, new_value, old_value)
}

/// kill (129): pid = a0, sig = a1。本运行时以进程为信号的处置单位, 目标进程的
/// 任一线程返回 U 模式之前即可取走信号。
pub fn kill_handler(pid: u64, sig: u64) -> u64 {
	kill(pid, sig)
}

/// tkill (130): tid = a0, sig = a1。本运行时不区分线程标识与进程标识, 按进程处理。
pub fn tkill_handler(tid: u64, sig: u64) -> u64 {
	kill(tid, sig)
}

/// tgkill (131): tgid = a0, tid = a1, sig = a2。以线程组标识确定目标进程。
///
/// 这是 musl abort() 的必经之路: 它先 raise(SIGABRT), 若信号被处理函数拦下且
/// 处理函数返回, 才落到 a_crash()。若此处返回 ENOSYS, abort 直接走到 a_crash,
/// 载荷以 139 (SIGSEGV) 退出并掩盖真实终止原因 (例如 Rust std 的 panic)。
pub fn tgkill_handler(tgid: u64, _tid: u64, sig: u64) -> u64 {
	kill(tgid, sig)
}

/// rt_sigaction (134): 参数 (Linux rv64) 为 sig = a0, new_action = a1,
/// old_action = a2, sigsetsize = a3。sigsetsize 只描述信号集合宽度, 本运行时不读
/// new_action 的 mask 栏, 故不参与处理。
pub fn sigaction_handler(sig_num: u64, new_action: u64, old_action: u64, _sigsetsize: u64) -> u64 {
	sigaction(sig_num, new_action, old_action)
}

/// rt_sigprocmask (135): 参数 (Linux rv64) 为 how = a0, set = a1, oldset = a2,
/// sigsetsize = a3。
///
/// 非法的 how 返回 EINVAL, 这是 libunwind 判断地址可读性的依赖 —— 它以 ~0 充当
/// how 调用本接口, 断言该调用必然失败并置 errno
/// (见 vendor/riscv-llvm-toolchain/libunwind/src/UnwindCursor.hpp 的 isReadableAddr)。
pub fn sigprocmask_handler(how: u64, set: u64, oldset: u64, sigsetsize: u64) -> u64 {
	sigprocmask(how, set, oldset, sigsetsize)
}

/// rt_sigreturn (139): 从当前用户 sp 指向的信号帧恢复现场。返回值写入 a0 后又会被
/// 帧内快照覆盖 (帧内含 a0), 故返回值只用于满足调用点的形状。
pub fn sigreturn_handler(gprs: &mut TrapGprs) -> u64 {
	sigreturn(gprs)
}
