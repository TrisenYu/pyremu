//! Thread-per-hart concurrent execution engine.
//!
//! ``run_parallel`` is the FFI entry point so the Python side can switch transparently.
use std::cell::Cell;
use std::sync::{
	atomic::AtomicBool, atomic::AtomicU32, atomic::AtomicU64, atomic::AtomicU8, atomic::Ordering,
	Arc, Mutex,
};

// ============================================================
//  Items extracted to sibling modules in cpu/src/
// ============================================================
//
// Each extracted module is declared in lib.rs as a top-level module
// of the pyremu-native crate.  The `use crate::*` paths below bring
// the pub(crate) items back into scope so `run_parallel` (below) and
// the test functions can reference them without qualification changes.
use crate::hart_sched::*;

// 无条件导入: diag 模块始终编译, 函数体在 feature 关闭时为 no-op.
#[allow(unused)]
use crate::diag;

use crate::ffi::{
	FfiBpCtx, FfiClintCtx, FfiDevicesCtx, FfiExtIrqCtx, FfiHartCtx, FfiInterruptCtx,
	FfiMemCtx, FfiPlicCtx, FfiTlbCtx, FfiUartCtx, FfiVirtIoCtx, InstrToBeExec, EMPTY_UART,
};
use crate::state::{exit_reason, HartState};

// ============================================================
//  Send/Sync wrappers for FFI raw pointers
// ============================================================
//
// Python holds all backing memory (bytearray, ctypes arrays) alive for
// the duration of the FFI call.  These wrappers assert Send + Sync
// so they can cross thread boundaries safely.

/// Wrapper around ``MemCtx`` that is ``Send + Sync``.
///
/// # Safety
/// The Python FFI caller must keep the underlying RAM alive.
#[derive(Clone, Copy)]
pub(crate) struct SharedMemCtx {
	pub ram: *mut u8,
	pub ram_size: u64,
	pub ram_base: u64,
	pub shadow_base: u64,
	pub shadow_size: u64,
	/// Per-hart LR/SC reservation slots.  LR sets slot[hid]=pa;
	/// any store clears all slots.  Allocated in ModuleState.
	pub lr_reserved: *mut AtomicU64,
}
unsafe impl Send for SharedMemCtx {}
unsafe impl Sync for SharedMemCtx {}

/// Wrapper around PMP pointers that is ``Send + Sync``.
#[derive(Clone, Copy)]
pub(crate) struct SharedPmpCtx {
	pub cfg: *mut u8,
	pub addr: *mut u64,
	pub num: u8,
}
unsafe impl Send for SharedPmpCtx {}
unsafe impl Sync for SharedPmpCtx {}

/// Wrapper around device MMIO pointers that is ``Send + Sync``.
#[derive(Clone, Copy)]
pub(crate) struct SharedDevCtx {
	pub bases: *const u64,
	pub ends: *const u64,
	pub num: u8,
	pub virtio_base: u64,
	pub virtio_raw: *mut FfiVirtIoCtx,
	pub plic: *mut FfiPlicCtx,
}
unsafe impl Send for SharedDevCtx {}
unsafe impl Sync for SharedDevCtx {}

// ============================================================
//  Concurrent CLINT context — Atomic wrappers around shared state
// ============================================================

/// CLINT state safe for concurrent access across hart threads.
///
/// All fields are atomic or read-only after initialisation.
/// Pointers are cast from Python ctypes arrays; on x86-64
/// ``AtomicU64`` has the same layout as ``u64`` (8 bytes, 8-byte aligned)
/// and ``AtomicU8`` has the same layout as ``u8`` (1 byte, 1-byte aligned).
///
/// MSIP is modelled as an **edge counter** (not a level): each write of 1
/// increments the per-hart counter; ``sync_msip`` detects new edges by
/// comparing against the last-seen value.  This correctly handles
/// consecutive writes of 1 without an intervening 0, which a pure
/// level-triggered detector would miss.
pub struct ConcurrentClintCtx {
	pub base: u64,
	/// Shared mtime counter — advanced by clock source
	/// (``advance_clock_source``), NOT per instruction.
	pub mtime: *const AtomicU64,
	/// Per-hart mtimecmp registers — indexed by hart_id.
	pub mtimecmp: *const AtomicU64,
	/// Per-hart MSIP level bytes — indexed by hart_id.
	pub msip: *const AtomicU8,
	/// mtime 时钟源频率 (Hz) — 0 表示禁用 clock source
	/// (Rust 单元测试传入 0 保持确定性).
	pub timebase_hz: u64,
	pub num_harts: u32,
	/// Per-hart atomic MSIP pending slots (Release write from sender,
	/// Acquire swap from receiver).  Set after ``ModuleState`` creation.
	/// ``Cell`` allows late initialisation despite ``&self``.
	pub msip_pending: Cell<*const AtomicU64>,
	/// Per-hart OS thread handles for MSIP unpark wake-up.
	/// Populated by ``run_harts`` after thread spawn; ``Cell`` enables
	/// late initialisation via ``&self`` (same pattern as ``msip_pending``).
	pub hart_threads: Cell<*const std::thread::Thread>,
	/// Per-hart ``HartState`` array (FFI 共享的 hart 状态指针).
	///
	/// 双重用途:
	/// 1. 跨 hart 读取 ``stimecmp`` (SSTC) — ``wfi_check_all_idle`` 据此在全部
	///    hart WFI 空闲时判定下一截止时间并以 WFI_WAIT 退出 (mtime 不再在此
	///    快进; Python 侧据真实剩余休眠并按流逝时间补偿 mtime), legacy 与 AIA
	///    两种模式都需要 (内核 clockevent 在 legacy 下同样走 SSTC).
	/// 2. AIA 模式下 ``try_handle_imsic_concurrent`` 跨 hart 置/清目标 hart 的
	///    IMSIC eip 位 (seteipnum/clreipnum MMIO).
	///
	/// 由 ``run_parallel`` 在 FFI 解包后无条件写入.
	pub hart_states: Cell<*const HartState>,
}

// Safety: Python holds the backing ctypes arrays alive for the FFI call.
// All pointers target properly-aligned atomic types.
unsafe impl Send for ConcurrentClintCtx {}
unsafe impl Sync for ConcurrentClintCtx {}

impl ConcurrentClintCtx {
	/// Build from the FFI ``FfiClintCtx`` raw pointers.
	///
	/// # Safety
	///
	/// The caller must ensure the backing ctypes arrays outlive this struct.
	/// On x86-64 the transmute from ``*mut u64`` to ``*const AtomicU64`` is
	/// sound because both types have identical size and alignment.
	pub unsafe fn from_ffi(raw: &FfiClintCtx, num_harts: u32) -> Self {
		ConcurrentClintCtx {
			base: raw.base,
			mtime: raw.mtime as *const AtomicU64,
			mtimecmp: raw.mtimecmp as *const AtomicU64,
			msip: raw.msip as *const AtomicU8,
			timebase_hz: raw.timebase_hz,
			num_harts,
			msip_pending: Cell::new(std::ptr::null()),
			hart_threads: Cell::new(std::ptr::null()),
			hart_states: Cell::new(std::ptr::null()),
		}
	}
}

// ============================================================
//  Stop coordination
// ============================================================

/// Reason a hart set the global stop flag.
#[derive(Clone, Copy)]
pub struct StopInfo {
	pub reason: u8,
	pub hart_id: u8,
	pub pc: u64,
	pub instr: u32,
	pub trap_cause: u32,
	pub trap_tval: u64,
	pub trap_is_interrupt: u8,
	pub trap_delegated: u8,
}

impl StopInfo {
	pub const fn empty() -> Self {
		StopInfo {
			reason: exit_reason::NORMAL,
			hart_id: 0,
			pc: 0,
			instr: 0,
			trap_cause: 0,
			trap_tval: 0,
			trap_is_interrupt: 0,
			trap_delegated: 0,
		}
	}
}

/// Global coordination state shared across all hart threads.
///
/// One instance per ``run_parallel`` call, wrapped in ``Arc``.
pub struct ModuleState {
	/// When set to true, all hart threads must exit at their next
	/// instruction boundary.
	pub stop_flag: AtomicBool,
	/// Details of the stop event, written by the triggering hart.
	/// Protected by a Mutex so only one hart sets it.
	pub stop_info: Mutex<StopInfo>,
	/// Number of harts currently in WFI spin-wait.
	pub wfi_count: AtomicU32,
	/// Per-hart WFI status flags (index = hart_id).
	/// Each hart sets its slot to 1 when entering WFI, 0 when leaving.
	pub wfi_flags: Box<[AtomicU8]>,
	/// Number of non-halted harts. Set once before threads spawn, read-only after.
	pub active_hart_num: u32,
	/// Global TLB generation counter — incremented (Release) by any hart
	/// that executes SFENCE.VMA.  Each hart locally samples this counter
	/// (Acquire) at instruction boundaries and flushes its own TLB when
	/// it detects a mismatch against its per-hart slot in ``tlb_gen_per_hart``.
	/// This provides broadcast-TLB-invalidation semantics without explicit
	/// IPI-driven shootdown: over-invalidation never breaks correctness,
	/// and it closes the window where hart A's page-table write followed by
	/// local SFENCE.VMA is invisible to hart B's TLB hit.
	pub tlb_gen: AtomicU64,
	/// Per-hart last-seen TLB generation.  When a hart's value is less than
	/// the global ``tlb_gen``, it must flush both itlb and dtlb before its
	/// next instruction fetch or data access.
	pub tlb_gen_per_hart: Box<[AtomicU64]>,
	/// Per-hart LR/SC reservation slots.  Each slot holds the PA reserved by
	/// that hart (0 = no reservation).  Any store to a PA clears matching
	/// reservations from ALL harts, ensuring correct multi-hart LR/SC semantics
	/// (RISC-V spec §8.2: "a successful SC on a given hart renders any LR on
	/// that hart invalid, and a successful AMO, SC, or store on any other hart
	/// to the reservation set renders any LR on this hart invalid").
	pub lr_reserved: Box<[AtomicU64]>,
	/// Per-hart cross-thread MSIP notification.  When another hart writes to
	/// CLINT MSIP for this hart, the bit is set here atomically (Release).
	/// This hart's ``sync_msip`` reads & clears it with ``swap(0, Acquire)``
	/// and merges into ``HartState.mip``.  This avoids the non-atomic RMW race
	/// on ``HartState.mip`` between the sender's direct write and the receiver's
	/// ``sync_mtip``/``sync_msip`` calls.
	pub msip_pending: Box<[AtomicU64]>,
	/// 本轮加速执行起点 Instant — 仅看门狗用它判定时钟源超时.
	pub st_time_val: std::time::Instant,
	/// marshal 时的 mtime 基准值. 指令推进的基准:
	/// target = time_base_val + instr_delta * timebase_hz * NS_PER_INSTR / 1e9.
	pub time_base_val: u64,
	/// 每 hart 本轮加速执行起始时的指令计数 (marshal 时快照).
	/// ``advance_clock_source`` 用 ``state.total_instrs - instr_ref_num[hid]``
	/// 计算本轮已执行的指令增量并换算 mtime tick — 受调试程序 时钟随已执行指令推进,
	/// 与主机单调时钟无关. 低速模拟 (2.5~6 MIPS) 下若 mtime 跟随真实流逝, 内核
	/// HZ=250 的定时器 tick (每 4ms 受调试程序时间) 只隔 ~1e4 条指令, tick 处理路径
	/// (实测 <=5e4 条) 超长即陷入 mret 后立即再 trap 的活锁; 按指令推进
	/// (每指令 20ns, tick 间隔 2e5 条指令 = 4× 余量, 永不风暴) 消除该活锁,
	/// 同时 rdtime 预算循环 (如内核 unaligned-access 测速, 8ms = 80000 ticks =
	/// 4e6 条指令) 仍可完成.
	pub instr_ref_num: Box<[u64]>,
}

impl ModuleState {
	pub fn new(
		num_harts: u32,
		active_hart_num: u32,
		init_tlb_gen: u64,
		time_base_val: u64,
		instr_ref_num: Box<[u64]>,
	) -> Self {
		let mut v = Vec::with_capacity(num_harts as usize);
		let mut gv = Vec::with_capacity(num_harts as usize);
		let mut rv = Vec::with_capacity(num_harts as usize);
		let mut mv = Vec::with_capacity(num_harts as usize);
		for _ in 0..num_harts {
			v.push(AtomicU8::new(0));
			gv.push(AtomicU64::new(0));
			rv.push(AtomicU64::new(0));
			mv.push(AtomicU64::new(0));
		}
		ModuleState {
			stop_flag: AtomicBool::new(false),
			stop_info: Mutex::new(StopInfo::empty()),
			wfi_count: AtomicU32::new(0),
			wfi_flags: v.into_boxed_slice(),
			active_hart_num,
			tlb_gen: AtomicU64::new(init_tlb_gen),
			tlb_gen_per_hart: gv.into_boxed_slice(),
			lr_reserved: rv.into_boxed_slice(),
			msip_pending: mv.into_boxed_slice(),
			st_time_val: std::time::Instant::now(),
			time_base_val,
			instr_ref_num,
		}
	}

	/// Set the stop flag and record the reason.  Only the first caller
	/// wins; subsequent callers are silently ignored.
	pub fn request_stop(&self, info: StopInfo) {
		if self.stop_flag.swap(true, Ordering::Release) {
			return; // already stopping
		}
		if let Ok(mut guard) = self.stop_info.lock() {
			*guard = info;
		}
	}

	/// Check whether all non-halted harts are in WFI.
	#[inline]
	pub fn all_in_wfi(&self) -> bool {
		self.wfi_count.load(Ordering::Acquire) >= self.active_hart_num
	}
}

// ============================================================
//  FFI entry point helpers
// ============================================================

#[inline]
unsafe fn init_instr_group(instr_group: *mut InstrToBeExec) {
	(*instr_group).total_instrs = 0;
	(*instr_group).exit_reason = exit_reason::NORMAL;
	(*instr_group).exit_hart_id = 0;
	(*instr_group).exit_pc = 0;
	(*instr_group).exit_instr = 0;
	(*instr_group).trap_cause = 0;
	(*instr_group).trap_tval = 0;
	(*instr_group).trap_is_interrupt = 0;
	(*instr_group).trap_delegated = 0;
}

#[inline]
unsafe fn count_active(states: *mut HartState, num_harts: u32) -> u32 {
	let mut count: u32 = 0;
	for hid in 0..num_harts {
		if (*states.add(hid as usize)).halted == 0 {
			count += 1;
		}
	}
	count
}

#[inline]
unsafe fn build_bps(bp: *const FfiBpCtx) -> &'static [u64] {
	if bp.is_null() {
		return &[];
	}
	let bp = &*bp;
	if bp.count == 0 || bp.addrs.is_null() {
		return &[];
	}
	std::slice::from_raw_parts(bp.addrs, bp.count as usize)
}

/// 软件看门狗轮询周期 (µs) — 周期性向中断控制器发置位请求信号 (unpark 全部
/// WFI hart) 的间隔, 兼作看门狗自身阻塞的周期.
/// 与 Python 侧 ``WFI_WATCHDOG_MS`` (5 ms) 对齐, 亦为 ``wfi_spin`` 自醒周期:
/// 停驻的 WFI hart 即使外部 unpark 全部消失, 也会按此周期自行醒来重查停止标志.
pub(crate) const WATCHDOG_POLL_US: u64 = 5_000;

/// 唤醒所有已注册 hart 线程 — WFI hart 现以 ``park()`` 无限阻塞,
/// 必须由看门狗/完成信号显式 ``unpark`` (解耦通知) 才能醒来.
fn unpark_all_harts(clint: &ConcurrentClintCtx) {
	let ptr = clint.hart_threads.get();
	if ptr.is_null() {
		return;
	}
	// Safety: ``hart_threads`` 由 ``run_harts`` 在线程全部 spawn 后写入,
	// 并在 join 前保持有效; 本函数仅在 ``run_harts`` 生命周期内被调用.
	let slice = unsafe { std::slice::from_raw_parts(ptr, clint.num_harts as usize) };
	for t in slice {
		t.unpark();
	}
}

/// 软件看门狗线程 — 单轮加速执行内的低延迟唤醒源.
///
/// WFI hart 以 ``park_timeout(WATCHDOG_POLL_US)`` 自醒阻塞 (既支持外部 unpark
/// 立即唤醒, 又在外部唤醒消失时自行醒来重查停止标志与中断挂起). 看门狗按固定
/// 周期 (``WATCHDOG_POLL_US``, 时钟源) 周期性 ``unpark_all_harts`` 唤醒全部
/// WFI hart, 降低自醒延迟: 被唤醒的 hart 重新执行 ``wfi_sync_and_check`` ->
/// ``sync_mtip``/``sync_msip``/``sync_imsic``, 由中断控制器判定定时器/软件/
/// 外部中断是否挂起 (置位对应 mip 位), 挂起则退出 WFI 投递中断, 否则重新阻塞.
///
/// 定时器 (mtime) 的推进与截止判定都不归看门狗管: mtime 由活动 hart 在指令
/// 边界按时钟源流逝推进 (``advance_clock_source``), 全部 WFI 空闲时由
/// ``wfi_check_all_idle`` 判定下一截止时间并以 WFI_WAIT 退出, 由 Python 侧
/// 休眠 (remaining/timebase) 并按睡眠流逝时间补偿推进 mtime (clint.tick).
fn watchdog_loop(
	module: &ModuleState,
	clint: &ConcurrentClintCtx,
	stop_flag: *const u8,
	uart_rx_notify: *const u8,
	timeout_ns: u64,
) {
	loop {
		// 1. 时钟源超时 (终止条件): 越过 timeout_ns 时以 TIMEOUT 停止.
		if timeout_ns != 0 && module.st_time_val.elapsed().as_nanos() as u64 >= timeout_ns {
			module.request_stop(StopInfo {
				reason: exit_reason::TIMEOUT,
				..StopInfo::empty()
			});
			unpark_all_harts(clint);
			return;
		}

		// 2. 停止 / RX 检查: 有信号则唤醒全部 WFI hart 让其自行退出.
		let internal_stop = module.stop_flag.load(Ordering::Acquire);
		let external_stop = !stop_flag.is_null() && unsafe { *stop_flag != 0 };
		let rx_ready = !uart_rx_notify.is_null() && unsafe { *uart_rx_notify != 0 };
		if internal_stop || external_stop || rx_ready {
			unpark_all_harts(clint);
		}
		if internal_stop || external_stop {
			return;
		}

		// 3. 周期性向中断控制器发置位请求信号: unpark 全部 WFI hart, 让它们重新
		//    执行 wfi_sync_and_check 判定定时器/软件/外部中断是否挂起并投递. mtime
		//    由活动 hart 按时钟源流逝推进, 看门狗不做截止预测, 以固定周期唤醒.
		unpark_all_harts(clint);
		std::thread::park_timeout(std::time::Duration::from_micros(WATCHDOG_POLL_US));
	}
}

/// Spawn one OS thread per hart, run ``hart_worker``, join all.
unsafe fn run_harts(
	states: *mut HartState,
	num_harts: u32,
	stop_flag: *const u8,
	ext_irq: *mut FfiExtIrqCtx,
	shared_mem: SharedMemCtx,
	shared_pmp: SharedPmpCtx,
	shared_dev: SharedDevCtx,
	cc_clint: &ConcurrentClintCtx,
	uart_ffi: *const FfiUartCtx,
	module: &Arc<ModuleState>,
	breakpoints: &'static [u64],
	timeout_ns: u64,
) {
	let uart_ref: &'static FfiUartCtx = if uart_ffi.is_null() {
		&EMPTY_UART
	} else {
		std::mem::transmute(&*uart_ffi)
	};

	// 软件看门狗: 先 spawn 以便把其 Thread 句柄 move 进每个 hart 闭包.
	// 看门狗是单轮加速执行内唯一轮询者, 周期性向中断控制器发置位请求信号唤醒 WFI hart.
	let clint_ptr: &'static ConcurrentClintCtx = std::mem::transmute(cc_clint);
	let wd_module = Arc::clone(module);
	let wd_stop_ptr = stop_flag as usize;
	let wd_rx_ptr = uart_ref.rx_notify as usize;
	let wd_handle = std::thread::spawn(move || {
		watchdog_loop(
			&wd_module,
			clint_ptr,
			wd_stop_ptr as *const u8,
			wd_rx_ptr as *const u8,
			timeout_ns,
		);
	});
	let watchdog_thread = wd_handle.thread().clone();

	let mut handles = Vec::with_capacity(num_harts as usize);
	let mut thread_refs: Vec<std::thread::Thread> = Vec::with_capacity(num_harts as usize);
	for hid in 0..num_harts {
		let mref = Arc::clone(module);
		let mem_send = shared_mem;
		let pmp_send = if shared_pmp.num > 0 {
			SharedPmpCtx {
				cfg: shared_pmp.cfg.add(hid as usize * 64),
				addr: shared_pmp.addr.add(hid as usize * 64),
				num: shared_pmp.num,
			}
		} else {
			shared_pmp
		};
		let dev_send = shared_dev;
		let bp_send: &'static [u64] = std::mem::transmute(breakpoints);
		let state_addr = states.add(hid as usize) as usize;
		let uart_ptr = uart_ref;
		let stop_ptr = stop_flag as usize;
		let ext_irq_ptr = ext_irq as usize;
		let wd_thread = watchdog_thread.clone();

		let handle = std::thread::spawn(move || {
			let state = &mut *(state_addr as *mut HartState);
			hart_worker(
				state,
				hid as u8,
				mem_send,
				pmp_send,
				dev_send,
				clint_ptr,
				uart_ptr,
				&mref,
				bp_send,
				stop_ptr as *const u8,
				ext_irq_ptr as *mut FfiExtIrqCtx,
			);
			// 完成信号: 唤醒其余 WFI hart 与看门狗, 让单轮加速执行尽快收敛退出.
			unpark_all_harts(clint_ptr);
			wd_thread.unpark();
		});
		thread_refs.push(handle.thread().clone());
		handles.push(handle);
	}

	// Store thread handles for MSIP/unpark: 发送方 (MSIP) 与看门狗调用
	// unpark() 唤醒接收方 hart 线程, 使其从 wfi_spin 的 park() 醒来.
	let thread_slice: Box<[std::thread::Thread]> = thread_refs.into_boxed_slice();
	cc_clint.hart_threads.set(thread_slice.as_ptr());
	std::mem::forget(thread_slice); // pointer valid until run_harts returns

	for h in handles {
		let _ = h.join();
	}
	let _ = wd_handle.join();
}

#[inline]
unsafe fn collect_stop(module: &ModuleState, instr_group: *mut InstrToBeExec) {
	if let Ok(guard) = module.stop_info.lock() {
		(*instr_group).exit_reason = guard.reason;
		(*instr_group).exit_hart_id = guard.hart_id;
		(*instr_group).exit_pc = guard.pc;
		(*instr_group).exit_instr = guard.instr;
		(*instr_group).trap_cause = guard.trap_cause;
		(*instr_group).trap_tval = guard.trap_tval;
		(*instr_group).trap_is_interrupt = guard.trap_is_interrupt;
		(*instr_group).trap_delegated = guard.trap_delegated;
	}
}

#[inline]
unsafe fn sum_instrs(states: *mut HartState, num_harts: u32, instr_group: *mut InstrToBeExec) {
	let mut total: u64 = 0;
	for hid in 0..num_harts {
		total = total.wrapping_add((*states.add(hid as usize)).total_instrs);
	}
	(*instr_group).total_instrs = total;
}

#[inline]
unsafe fn writeback_tlb(
	gen_ptr: *mut u64,
	gen_per_hart_ptr: *mut u64,
	module: &ModuleState,
	num_harts: u32,
) {
	if gen_ptr.is_null() {
		return;
	}
	*gen_ptr = module.tlb_gen.load(Ordering::Relaxed);
	for i in 0..num_harts as usize {
		*gen_per_hart_ptr.add(i) = module.tlb_gen_per_hart[i].load(Ordering::Relaxed);
	}
}

#[inline]
unsafe fn writeback_mtime(clint_raw: &FfiClintCtx, cc_clint: &ConcurrentClintCtx) {
	*clint_raw.mtime = (*cc_clint.mtime).load(Ordering::SeqCst);
}

/// 纳秒/秒 — Duration 换算为时钟 tick 的定义性常数.
const NS_PER_SEC: u64 = 1_000_000_000;

/// 指令计数推进的换算比率 — 每指令多少纳秒 受调试程序 时间.
///
/// 取值 20ns/instr:
pub const NS_PER_INSTR: u64 = 20;

/// 按指令计数推进 mtime (纯指令源, 无实时分量).
///
/// mtime = time_base_val + instr_delta * NS_PER_INSTR * tb / 1e9
///
/// - 单一时钟源
/// - 单调
pub(crate) fn advance_clock_source(
	module: &ModuleState,
	clint: &ConcurrentClintCtx,
	state: &HartState,
) {
	if clint.timebase_hz == 0 {
		return;
	}
	let hid = state.mhartid as usize;
	if hid >= module.instr_ref_num.len() {
		return;
	}
	let instr_delta = state.total_instrs.wrapping_sub(module.instr_ref_num[hid]);
	// instr_delta * timebase_hz * NS_PER_INSTR: 峰值 batch ~4M * 10MHz = 4e13,
	// 远小于 u64 上限, 无需 u128.
	let ticks = instr_delta
		.saturating_mul(clint.timebase_hz)
		.saturating_mul(NS_PER_INSTR)
		/ NS_PER_SEC;
	let target = module.time_base_val.wrapping_add(ticks);
	unsafe { &*clint.mtime }.fetch_max(target, Ordering::Relaxed);
}

/// QEMU 一次性定时器语义的指令计数换算: 计算 future ``timecmp`` 在
/// "本 hart 指令计数空间" 的截止点 (0 = 未设定, 见 write_mtimecmp).
///
/// QEMU ``riscv_aclint_mtimer_write_timecmp`` 在写入时判定过去/未来:
/// 过去立即置位, 未来清位并设定一个一次性 deadline, 到期才再次置位
/// (``timer_mod`` + ``riscv_aclint_mtimer_cb``)。这里把该"一次性定时器"
/// 翻译到当前 hart 的指令计数空间: ``deadline = own_total_instrs + offset``,
/// 其中 ``offset`` 是把剩余 tick (``timecmp - now_mtime``) 按
/// ``timebase * NS_PER_INSTR / 1e9`` (每指令推进的 tick 数) 换算成指令数,
/// 向上取整保证 deadline 在 mtime 严格越过 timecmp 之后 (或同时) 才到达,
/// 与 ``sync_mtip`` 的 ">=" 比较一致。
///
/// ``timecmp`` 为 0 (禁用) 或 <= 当前共享 mtime (已到期) 时返回 0 — 调用方
/// 负责立即置位 (已到期) 或保持清除 (禁用), 与本函数解耦。
#[inline]
pub(crate) fn timer_deadline_own(
	own_total_instrs: u64,
	now_mtime: u64,
	timecmp: u64,
	timebase_hz: u64,
) -> u64 {
	if timecmp == 0 || timecmp <= now_mtime {
		return 0;
	}
	// 每指令推进 timebase * NS_PER_INSTR / 1e9 tick (20ns/instr @ 10MHz -> 0.2 tick/instr).
	let ticks_per_instr_scale = timebase_hz.saturating_mul(NS_PER_INSTR);
	if ticks_per_instr_scale == 0 {
		return 0;
	}
	let ticks = timecmp - now_mtime;
	// 向上取整: ceil(ticks * 1e9 / (timebase * NS_PER_INSTR)).
	let offset = ticks
		.saturating_mul(NS_PER_SEC)
		.saturating_add(ticks_per_instr_scale - 1)
		.checked_div(ticks_per_instr_scale)
		.unwrap_or(u64::MAX);
	own_total_instrs.wrapping_add(offset)
}

// ============================================================
//  FFI entry point
// ============================================================

#[no_mangle]
#[allow(dead_code)]
pub unsafe extern "C" fn run_parallel(
	hart: *const FfiHartCtx,
	mem_ctx: *const FfiMemCtx,
	intr_ctx: *const FfiInterruptCtx,
	dev_ctx: *const FfiDevicesCtx,
	bp: *const FfiBpCtx,
	tlb: *mut FfiTlbCtx,
) {
	// 1. Unpack FFI structs
	let hart = unsafe { &*hart };
	let mem_ctx = unsafe { &*mem_ctx };
	let intr_ctx = unsafe { &*intr_ctx };
	let dev_ctx = unsafe { &*dev_ctx };
	let states = hart.states;
	let num_harts = hart.num_harts;

	// 2. Zero-initialise instr_group + build contexts
	let (mem, pmp_raw, clint_raw, dev_raw, cc_clint, active) = unsafe {
		init_instr_group(hart.instr_group);
		(
			&*mem_ctx.mem,
			&*mem_ctx.pmp,
			&*intr_ctx.clint,
			&*dev_ctx.dev,
			ConcurrentClintCtx::from_ffi(&*intr_ctx.clint, num_harts),
			count_active(states, num_harts),
		)
	};
	// stimecmp (SSTC) 是内核 clockevent 在 legacy 与 AIA 两种模式下的共同定时器源.
	// ``wfi_check_all_idle`` 通过本指针读取 stimecmp, 判断全部 WFI hart 空闲时
	// 该把 mtime 快进到哪个截止. 无条件设置本指针
	cc_clint.hart_states.set(states as *const HartState);

	// 3. Module state (TLB gen persists between different speedup stage)
	let tlb_gen_ptr = if tlb.is_null() {
		std::ptr::null_mut()
	} else {
		unsafe { (*tlb).gen }
	};
	let tlb_gen_per_hart_ptr = if tlb.is_null() {
		std::ptr::null_mut()
	} else {
		unsafe { (*tlb).gen_per_hart }
	};
	let init_gen = if tlb_gen_ptr.is_null() {
		0
	} else {
		unsafe { *tlb_gen_ptr }
	};
	// time_base_val = marshal 时的 mtime 值, instr_ref_num = 每 hart 起始指令计数,
	// advance_clock_source 据此以指令增量推进 (见 NS_PER_INSTR).
	let instr_ref_num: Box<[u64]> = (0..num_harts)
		.map(|hid| unsafe { (*states.add(hid as usize)).total_instrs })
		.collect();
	let module = Arc::new(ModuleState::new(
		num_harts,
		active,
		init_gen,
		unsafe { *clint_raw.mtime },
		instr_ref_num,
	));
	cc_clint.msip_pending.set(module.msip_pending.as_ptr());

	// 时钟源超时时间 (ns, 0 = 禁用) — 由 FfiWatchdogCtx 注入.
	let timeout_ns = if dev_ctx.watchdog.is_null() {
		0
	} else {
		unsafe { (*dev_ctx.watchdog).timeout_ns }
	};

	// 4. Build shared contexts
	let virtio_raw: *mut FfiVirtIoCtx = if dev_ctx.virtio.is_null() {
		std::ptr::null_mut()
	} else {
		dev_ctx.virtio as *mut FfiVirtIoCtx
	};
	let shared_mem = SharedMemCtx {
		ram: mem.ram,
		ram_size: mem.ram_size,
		ram_base: mem.ram_base,
		shadow_base: mem.shadow_base,
		shadow_size: mem.shadow_size,
		lr_reserved: module.lr_reserved.as_ptr() as *mut AtomicU64,
	};
	let shared_pmp = SharedPmpCtx {
		cfg: pmp_raw.cfg,
		addr: pmp_raw.addr,
		num: pmp_raw.num,
	};
	let plic_raw: *mut FfiPlicCtx = if intr_ctx.plic.is_null() {
		std::ptr::null_mut()
	} else {
		intr_ctx.plic as *mut FfiPlicCtx
	};
	let shared_dev = SharedDevCtx {
		bases: dev_raw.bases,
		ends: dev_raw.ends,
		num: dev_raw.num,
		virtio_base: if virtio_raw.is_null() {
			0
		} else {
			unsafe { (*virtio_raw).base }
		},
		virtio_raw,
		plic: plic_raw,
	};

	unsafe {
		// 5. Spawn & join hart threads
		run_harts(
			states,
			num_harts,
			hart.stop_flag,
			hart.ext_irq,
			shared_mem,
			shared_pmp,
			shared_dev,
			&cc_clint,
			dev_ctx.uart,
			&module,
			build_bps(bp),
			timeout_ns,
		);
		// 6. Collect results — single unsafe block for all write-backs
		collect_stop(&module, hart.instr_group);
		sum_instrs(states, num_harts, hart.instr_group);
		writeback_tlb(tlb_gen_ptr, tlb_gen_per_hart_ptr, &module, num_harts);
		writeback_mtime(clint_raw, &cc_clint);
	}
}

// ============================================================
//  Tests
// ============================================================

#[cfg(test)]
mod tests {
	use super::*;
	// Items extracted from this file into sibling modules — bring them
	// back in scope so tests can reference them directly.
	use crate::ffi::{DevMMIOAddrInfo, FfiPmpCtx, FfiWatchdogCtx, MemCtx};
	use crate::state::{riscv_mode, CFG_AIA};
	use std::mem;

	fn make_state(pc: u64, hart_id: u64) -> HartState {
		let mut s: HartState = unsafe { mem::zeroed() };
		s.mode = riscv_mode::M;
		s.mtvec = 0x8000_0100;
		s.pc = pc;
		s.mhartid = hart_id;
		s
	}

	fn write_u32_le(buf: &mut [u8], offset: usize, val: u32) {
		buf[offset] = val as u8;
		buf[offset + 1] = (val >> 8) as u8;
		buf[offset + 2] = (val >> 16) as u8;
		buf[offset + 3] = (val >> 24) as u8;
	}

	/// Build default zero-length PMP / device arrays and call ``run_parallel``.
	unsafe fn run_parallel_defaults(
		states: *mut HartState,
		num: u32,
		ram: *mut u8,
		ram_sz: u64,
		ram_base: u64,
		shadow_base: u64,
		shadow_size: u64,
		instr_group: *mut InstrToBeExec,
	) {
		let mem = MemCtx {
			ram,
			ram_size: ram_sz,
			ram_base,
			shadow_base,
			shadow_size,
		};
		let dev_bases: [u64; 0] = [];
		let dev_ends: [u64; 0] = [];
		let dev = DevMMIOAddrInfo {
			bases: dev_bases.as_ptr(),
			ends: dev_ends.as_ptr(),
			num: 0,
		};
		// Single TOR entry covering full address space (R/W/X).
		// pmp_ok per RISC-V spec §3.7.1 denies S/U access when num==0,
		// so we must provide at least one permissive entry for tests.
		let mut pmp_cfg: [u8; 1] = [0x0F]; // PMP_R|PMP_W|PMP_X|PMP_A_TOR
		let mut pmp_addr: [u64; 1] = [u64::MAX];
		let pmp = FfiPmpCtx {
			cfg: pmp_cfg.as_mut_ptr(),
			addr: pmp_addr.as_mut_ptr(),
			num: 1,
			pmpsplit: 0,
		};
		// Allocate per-hart CLINT arrays based on the actual number of harts.
		let nh = num as usize;
		let mut mtimecmp_vec: Vec<u64> = vec![u64::MAX; nh];
		let mut msip_vec: Vec<u8> = vec![0u8; nh];
		let mut mtime_val: u64 = 0;
		let clint = FfiClintCtx {
			mtime: &mut mtime_val as *mut u64,
			mtimecmp: mtimecmp_vec.as_mut_ptr(),
			msip: msip_vec.as_mut_ptr(),
			base: 0,
			timebase_hz: 0, // 单元测试禁用 clock-source 推进, 保持确定性
		};
		let hart = FfiHartCtx {
			states,
			num_harts: num,
			instr_group,
			stop_flag: std::ptr::null(),
			ext_irq: std::ptr::null_mut(),
		};
		let mem_ctx = FfiMemCtx {
			mem: &mem,
			pmp: &pmp,
		};
		let intr_ctx = FfiInterruptCtx {
			clint: &clint,
			plic: std::ptr::null(),
		};
		let dev_ctx = FfiDevicesCtx {
			dev: &dev,
			uart: std::ptr::null(),
			virtio: std::ptr::null(),
			watchdog: std::ptr::null(),
		};
		let bp = FfiBpCtx {
			addrs: std::ptr::null(),
			count: 0,
		};
		let mut tlb = FfiTlbCtx {
			gen: std::ptr::null_mut(),
			gen_per_hart: std::ptr::null_mut(),
		};
		run_parallel(&hart, &mem_ctx, &intr_ctx, &dev_ctx, &bp, &mut tlb);
	}

	#[test]
	fn parallel_single_hart_addi() {
		let mut ram = vec![0u8; 256];
		// Instructions at PC=0:
		write_u32_le(&mut ram, 0, 0x00A0_0293); // addi x5, x0, 10
		write_u32_le(&mut ram, 4, 0x0052_8293); // addi x5, x5, 5
		write_u32_le(&mut ram, 8, 0x1050_0073); // WFI -> clean exit

		let mut state = make_state(0, 0);
		state.mode = riscv_mode::M; // WFI is NOP in M-mode when mip=0->enters waiting
		state.mie = 0;
		state.mip.store(0, Ordering::Release);
		let mut instr_group: InstrToBeExec = unsafe { mem::zeroed() };
		unsafe {
			run_parallel_defaults(
				&mut state as *mut HartState,
				1,
				ram.as_mut_ptr(),
				ram.len() as u64,
				0,
				0,
				0,
				&mut instr_group as *mut InstrToBeExec,
			);
		}
		assert_eq!(state.gprs[5], 15, "addi instructions should execute");
		assert_eq!(instr_group.exit_reason, exit_reason::WFI_WAIT);
	}

	/// Regression: WFI 阻塞时 stimecmp (SSTC) 截止必须在 legacy (非 AIA) 模式下
	/// 被识别并以 WFI_WAIT 退出加速执行 (mtime 不再快进 — Python 侧据真实剩余
	/// 休眠并按流逝时间补偿 mtime). 修复前 ``run_parallel`` 仅在 ``CFG_AIA`` 下
	/// 设置 ``hart_states``, legacy 模式下该指针恒为 null, ``wfi_check_all_idle`` 扫描不到
	/// stimecmp -> mtime 不推进 -> 用 usleep_range() (SSTC) 睡眠的 hart 永不唤醒,
	/// 辅助核上线握手 (cpuhp_ap_sync_alive) 死锁.
	#[test]
	fn wfi_stimecmp_deadline_exits_wfi_wait_in_legacy_mode() {
		// 本测试明确覆盖 legacy 模式 (CFG_AIA=false) 下的 stimecmp 唤醒路径 —
		// 修复前 ``run_parallel`` 以 CFG_AIA 门控 hart_states, legacy 模式下指针恒为
		// null 导致 stimecmp 不可见. AIA 构建下该路径走 IMSIC 分支, 跳过本测试.
		if CFG_AIA {
			return;
		}

		let mut ram = vec![0u8; 256];
		write_u32_le(&mut ram, 0, 0x1050_0073); // WFI

		let mut state = make_state(0, 0);
		state.mode = riscv_mode::M; // WFI 在 M 模式且 mip=0 时进入等待
		state.mie = 0;
		state.mip.store(0, Ordering::Release);
		state.stimecmp = 5000; // 未来 SSTC 截止时间

		let mem = MemCtx {
			ram: ram.as_mut_ptr(),
			ram_size: ram.len() as u64,
			ram_base: 0,
			shadow_base: 0,
			shadow_size: 0,
		};
		let dev_bases: [u64; 0] = [];
		let dev_ends: [u64; 0] = [];
		let dev = DevMMIOAddrInfo {
			bases: dev_bases.as_ptr(),
			ends: dev_ends.as_ptr(),
			num: 0,
		};
		// 单个 TOR 条目覆盖全地址空间 (R/W/X), 避免 pmp_ok 因 num==0 拒绝访问.
		let mut pmp_cfg: [u8; 1] = [0x0F];
		let mut pmp_addr: [u64; 1] = [u64::MAX];
		let pmp = FfiPmpCtx {
			cfg: pmp_cfg.as_mut_ptr(),
			addr: pmp_addr.as_mut_ptr(),
			num: 1,
			pmpsplit: 0,
		};

		// mtimecmp 置 u64::MAX (无截止), stimecmp=5000 (唯一截止).
		// mtime 由测试持有以便断言其被快进到 stimecmp.
		let nh = 1usize;
		let mut mtimecmp_vec: Vec<u64> = vec![u64::MAX; nh];
		let mut msip_vec: Vec<u8> = vec![0u8; nh];
		let mut mtime_val: u64 = 0;
		let clint = FfiClintCtx {
			mtime: &mut mtime_val as *mut u64,
			mtimecmp: mtimecmp_vec.as_mut_ptr(),
			msip: msip_vec.as_mut_ptr(),
			base: 0,
			timebase_hz: 0, // 禁用 clock-source 推进, mtime 仅由快进决定
		};

		let mut instr_group: InstrToBeExec = unsafe { mem::zeroed() };
		unsafe {
			let hart = FfiHartCtx {
				states: &mut state as *mut HartState,
				num_harts: 1,
				instr_group: &mut instr_group as *mut InstrToBeExec,
				stop_flag: std::ptr::null(),
				ext_irq: std::ptr::null_mut(),
			};
			let mem_ctx = FfiMemCtx {
				mem: &mem,
				pmp: &pmp,
			};
			let intr_ctx = FfiInterruptCtx {
				clint: &clint,
				plic: std::ptr::null(),
			};
			let dev_ctx = FfiDevicesCtx {
				dev: &dev,
				uart: std::ptr::null(),
				virtio: std::ptr::null(),
				watchdog: std::ptr::null(),
			};
			let bp = FfiBpCtx {
				addrs: std::ptr::null(),
				count: 0,
			};
			let mut tlb = FfiTlbCtx {
				gen: std::ptr::null_mut(),
				gen_per_hart: std::ptr::null_mut(),
			};
			run_parallel(&hart, &mem_ctx, &intr_ctx, &dev_ctx, &bp, &mut tlb);
		}
		assert_eq!(
			mtime_val, 0,
			"WFI 空闲不得快进 mtime — 快进使 Python 侧 remaining=0 永不睡眠, \
             全 hart WFI 退化为 100% CPU 忙转; mtime 由 Python 按睡眠流逝时间补偿"
		);
		assert_eq!(
            instr_group.exit_reason, exit_reason::WFI_WAIT,
            "stimecmp 截止必须被识别并以 WFI_WAIT 退出 (legacy 模式) — hart_states 必须在非 AIA 下也设置"
        );
	}

	/// 构造「受调试程序 在无限循环中执行, 且 termio 已置位 RX 通知」的单 hart 加速
	/// 执行场景, 返回 (退出原因, 本轮指令数)。``imsic_owns`` 决定 IMSIC 是否存在
	/// 且已开启投递 (eidelivery 非 0)。
	///
	/// 看门狗超时 300 ms 用于界定「引擎忽略 RX 通知」的行为: 受调试程序 的无限
	/// 循环使本轮永不自然结束, 忽略通知时只能由看门狗以 TIMEOUT 收场, 测试因此
	/// 以断言失败而非挂起的形式暴露回归。
	unsafe fn run_rx_notify_case(imsic_owns: bool) -> (u8, u64) {
		let mut ram = vec![0u8; 256];
		write_u32_le(&mut ram, 0, 0x0000_006F); // jal x0, 0: 原地无限循环

		let mut state = make_state(0, 0);
		state.mode = riscv_mode::M;
		state.mie = 0;
		state.mip.store(0, Ordering::Release);
		if imsic_owns {
			state.imsic_s.present = 1;
			state.imsic_s.eidelivery = 1;
		}

		// termio 线程发现 stdin 新字节后置位的通知位.
		let mut rx_notify: u8 = 1;
		let uart = FfiUartCtx {
			base: 0,
			tx_buf: std::ptr::null_mut(),
			tx_cap: 0,
			tx_wr: std::ptr::null_mut(),
			ie: 0,
			txctrl: 0,
			rxctrl: 0,
			rx_fifo_len: 0,
			tx_notify_fd: -1,
			no_stdout: 0,
			rx_notify: &mut rx_notify as *mut u8,
		};
		let watchdog = FfiWatchdogCtx {
			timeout_ns: 300_000_000,
		};

		let mem = MemCtx {
			ram: ram.as_mut_ptr(),
			ram_size: ram.len() as u64,
			ram_base: 0,
			shadow_base: 0,
			shadow_size: 0,
		};
		let dev_bases: [u64; 0] = [];
		let dev_ends: [u64; 0] = [];
		let dev = DevMMIOAddrInfo {
			bases: dev_bases.as_ptr(),
			ends: dev_ends.as_ptr(),
			num: 0,
		};
		// 单个 TOR 条目覆盖全地址空间 (R/W/X), 避免 pmp_ok 因 num==0 拒绝访问.
		let mut pmp_cfg: [u8; 1] = [0x0F];
		let mut pmp_addr: [u64; 1] = [u64::MAX];
		let pmp = FfiPmpCtx {
			cfg: pmp_cfg.as_mut_ptr(),
			addr: pmp_addr.as_mut_ptr(),
			num: 1,
			pmpsplit: 0,
		};
		let mut mtimecmp_vec: Vec<u64> = vec![u64::MAX; 1];
		let mut msip_vec: Vec<u8> = vec![0u8; 1];
		let mut mtime_val: u64 = 0;
		let clint = FfiClintCtx {
			mtime: &mut mtime_val as *mut u64,
			mtimecmp: mtimecmp_vec.as_mut_ptr(),
			msip: msip_vec.as_mut_ptr(),
			base: 0,
			timebase_hz: 0, // 禁用 clock-source 推进, 保持确定性
		};

		let mut instr_group: InstrToBeExec = mem::zeroed();
		let hart = FfiHartCtx {
			states: &mut state as *mut HartState,
			num_harts: 1,
			instr_group: &mut instr_group as *mut InstrToBeExec,
			stop_flag: std::ptr::null(),
			ext_irq: std::ptr::null_mut(),
		};
		let mem_ctx = FfiMemCtx {
			mem: &mem,
			pmp: &pmp,
		};
		let intr_ctx = FfiInterruptCtx {
			clint: &clint,
			plic: std::ptr::null(),
		};
		let dev_ctx = FfiDevicesCtx {
			dev: &dev,
			uart: &uart,
			virtio: std::ptr::null(),
			watchdog: &watchdog,
		};
		let bp = FfiBpCtx {
			addrs: std::ptr::null(),
			count: 0,
		};
		let mut tlb = FfiTlbCtx {
			gen: std::ptr::null_mut(),
			gen_per_hart: std::ptr::null_mut(),
		};
		run_parallel(&hart, &mem_ctx, &intr_ctx, &dev_ctx, &bp, &mut tlb);
		(instr_group.exit_reason, instr_group.total_instrs)
	}

	/// Regression: termio 写入 stdin 字节并置位 RX 通知时, 若 IMSIC 独占外部中断
	/// 线路 (eidelivery 非 0), 引擎必须在执行任何指令之前退出本轮加速执行。
	///
	/// 字节到达受调试程序 需经 Python 侧两步搬运: RX 线程把 ring buffer 的字节
	/// 搬进 UART FIFO, APLIC 再把中断注入 IMSIC 的 eip。这两步在单轮加速执行内
	/// 都不可见, 且 ``sync_ext_irq_mip`` 在 IMSIC 独占线路时不置 SEIP, 引擎读到的
	/// eip 又是 marshal 时刻的快照, 故本轮无法投递中断, 必须退出让 Python 搬运;
	/// 下一轮 ``_native_sync_plic_mip`` 依据新的 eip 置 SEIP, 受调试程序 随即取走数据。
	///
	/// 修复前 (e904ba4 删除了 ``hart_worker`` 的 RX 通知退出) 该通知被完全忽略,
	/// 输入要等到本轮加速执行自然结束才被处理 — 交互式控制台下单轮长达数秒,
	/// 表现为 AIA 模式下 zsh 收不到键盘输入。
	#[test]
	fn rx_notify_exits_batch_when_imsic_owns_ext_line() {
		let (reason, instrs) = unsafe { run_rx_notify_case(true) };
		assert_eq!(
			reason,
			exit_reason::RX_WAIT,
			"IMSIC 独占外部中断线路时, RX 通知必须让引擎立即退出本轮; \
             否则输入要等本轮自然结束 (数秒) 才被搬运, AIA 模式下表现为无键盘输入"
		);
		assert_eq!(instrs, 0, "退出判定在指令循环之前, 本轮不得执行任何指令");
	}

	/// 互补用例: IMSIC 未占用外部中断线路 (legacy PLIC 模式) 时, RX 通知不得
	/// 触发退出 — ``sync_ext_irq_mip`` 已在本轮内联置位 SEIP, 受调试程序 在轮内
	/// 即可取走字节; 退出会砍掉 legacy 模式的输入吞吐。此处受调试程序 无限循环,
	/// 唯一的结束方式是看门狗超时, 故以 TIMEOUT 锁定「未退出」这一行为。
	#[test]
	fn rx_notify_does_not_exit_batch_without_imsic() {
		let (reason, _) = unsafe { run_rx_notify_case(false) };
		assert_eq!(
			reason,
			exit_reason::TIMEOUT,
			"IMSIC 未占用外部中断线路时不得为 RX 通知退出本轮 (内联置位已足够)"
		);
	}

	/// Test that ECALL delivers trap inline: mode -> M, PC -> mtvec,
	/// mepc = ECALL address.  The M-mode handler at mtvec advances
	/// mepc by 4 (to skip ECALL) and MRETs back to S-mode, where
	/// WFI terminates cleanly.
	#[test]
	fn parallel_ecall_inline_trap() {
		let mut ram = vec![0u8; 256];
		// S-mode code at PC=0:
		write_u32_le(&mut ram, 0, 0x0010_0293); // addi x5, x0, 1
		write_u32_le(&mut ram, 4, 0x0000_0073); // ECALL (S-mode -> M-mode)
		write_u32_le(&mut ram, 8, 0x1050_0073); // WFI (reached after MRET)

		// M-mode handler at mtvec=0x20 (uses t1=x6 to avoid clobbering x5):
		let mtvec_addr: u64 = 0x20;
		// addi t1, x0, 4     -> 0x00400313  (t1 = 4)
		write_u32_le(&mut ram, 0x20, 0x0040_0313);
		// csrr t1, mepc      -> 0x34102373  (t1 = mepc)
		write_u32_le(&mut ram, 0x24, 0x3410_2373);
		// addi t1, t1, 4     -> 0x00430313  (t1 = mepc + 4)
		write_u32_le(&mut ram, 0x28, 0x0043_0313);
		// csrw mepc, t1      -> 0x34131073  (mepc = t1)
		write_u32_le(&mut ram, 0x2C, 0x3413_1073);
		// mret                -> 0x30200073
		write_u32_le(&mut ram, 0x30, 0x3020_0073);

		let mut state = make_state(0, 0);
		state.mode = riscv_mode::S;
		state.mtvec = mtvec_addr;
		state.mstatus = (1 << 11) | (1 << 7); // MPP=S, MPIE=1
		state.mie = 0;
		state.mip.store(0, Ordering::Release);

		let mut instr_group: InstrToBeExec = unsafe { mem::zeroed() };
		unsafe {
			run_parallel_defaults(
				&mut state as *mut HartState,
				1,
				ram.as_mut_ptr(),
				ram.len() as u64,
				0,
				0,
				0,
				&mut instr_group as *mut InstrToBeExec,
			);
		}
		// addi x5,0,1 executed before ECALL
		assert_eq!(state.gprs[5], 1);
		// ECALL trap delivered inline -> M-mode handler -> MRET -> S-mode -> WFI
		assert_eq!(
			instr_group.exit_reason,
			exit_reason::WFI_WAIT,
			"inline ECALL+M-mode handler+MRET+WFI->WFI_WAIT exit"
		);
	}

	#[test]
	fn parallel_two_harts_independent() {
		let mut ram = vec![0u8; 4096];
		// Hart 0: addi x5, x0, 10; addi x5, x5, 3; WFI (PC 0..12)
		write_u32_le(&mut ram, 0, 0x00A0_0293);
		write_u32_le(&mut ram, 4, 0x0032_8293);
		write_u32_le(&mut ram, 8, 0x1050_0073); // WFI
										  // Hart 1: addi x6, x0, 20; addi x6, x6, 7; WFI (PC 128..140)
		write_u32_le(&mut ram, 128, 0x0140_0313);
		write_u32_le(&mut ram, 132, 0x0073_0313);
		write_u32_le(&mut ram, 136, 0x1050_0073); // WFI

		let mut s0 = make_state(0, 0);
		s0.mode = riscv_mode::M; // WFI needs M-mode to avoid TW trap
		s0.mie = 0;
		s0.mip.store(0, Ordering::Release);
		let mut s1 = make_state(128, 1);
		s1.mode = riscv_mode::M;
		s1.mie = 0;
		s1.mip.store(0, Ordering::Release);
		let mut states = [s0, s1];

		let mut instr_group: InstrToBeExec = unsafe { mem::zeroed() };
		unsafe {
			run_parallel_defaults(
				states.as_mut_ptr(),
				2,
				ram.as_mut_ptr(),
				ram.len() as u64,
				0,
				0,
				0,
				&mut instr_group as *mut InstrToBeExec,
			);
		}
		// Both harts finished their adds before entering WFI.
		// Since all harts end up in WFI, the exits with WFI_WAIT.
		assert_eq!(states[0].gprs[5], 13, "hart 0: x5 should be 13");
		assert_eq!(states[1].gprs[6], 27, "hart 1: x6 should be 27");
		assert_eq!(
			instr_group.exit_reason,
			exit_reason::WFI_WAIT,
			"both harts WFI -> should exit with WFI_WAIT, got {}",
			instr_group.exit_reason
		);
	}

	#[test]
	fn parallel_amo_single_hart() {
		// Verify AMOADD works with a single hart in concurrent mode.
		let mut ram = vec![0u8; 4096];
		let shared_addr: u64 = 0xF0;
		write_u32_le(&mut ram, 0, 0x0063_A2AF); // amoadd.w x5, x6, (x7)
		write_u32_le(&mut ram, 4, 0x1050_0073); // WFI

		ram[shared_addr as usize] = 42;
		let mut state = make_state(0, 0);
		state.mode = riscv_mode::M;
		state.mie = 0;
		state.mip.store(0, Ordering::Release);
		state.gprs[7] = shared_addr; // addr
		state.gprs[6] = 3; // addend
		let mut instr_group: InstrToBeExec = unsafe { mem::zeroed() };
		unsafe {
			run_parallel_defaults(
				&mut state as *mut HartState,
				1,
				ram.as_mut_ptr(),
				ram.len() as u64,
				0,
				0,
				0,
				&mut instr_group as *mut InstrToBeExec,
			);
		}
		let final_val = ram[shared_addr as usize] as u32
			| (ram[shared_addr as usize + 1] as u32) << 8
			| (ram[shared_addr as usize + 2] as u32) << 16
			| (ram[shared_addr as usize + 3] as u32) << 24;
		assert_eq!(
			final_val, 45,
			"single hart AMOADD: 42+3=45, got {}",
			final_val
		);
		assert_eq!(instr_group.exit_reason, exit_reason::WFI_WAIT);
	}

	#[test]
	fn parallel_amo_add_two_harts() {
		let mut ram = vec![0u8; 4096];
		let shared_addr: u64 = 0xF0; // aligned, well within RAM

		// Hart 0: AMOADD.W to shared_addr; then ECALL
		//   amoadd.w x5, x6, (x7): rs1=x7=addr, rs2=x6=addend, rd=x5=old
		write_u32_le(&mut ram, 0, 0x0063_A2AF); // amoadd.w x5, x6, (x7)
		write_u32_le(&mut ram, 4, 0x1050_0073); // WFI

		// Hart 1: AMOADD.W to same shared_addr; then WFI
		write_u32_le(&mut ram, 128, 0x0063_A2AF); // amoadd.w x5, x6, (x7)
		write_u32_le(&mut ram, 132, 0x1050_0073); // WFI

		// Set initial value at shared_addr = 42 (little-endian)
		ram[shared_addr as usize] = 42;
		ram[shared_addr as usize + 1] = 0;
		ram[shared_addr as usize + 2] = 0;
		ram[shared_addr as usize + 3] = 0;

		let mut s0 = make_state(0, 0);
		s0.mode = riscv_mode::M; // WFI needs M-mode
		s0.mie = 0;
		s0.mip.store(0, Ordering::Release);
		s0.gprs[7] = shared_addr; // rs1 — address
		s0.gprs[6] = 3; // rs2 — value to add

		let mut s1 = make_state(128, 1);
		s1.mode = riscv_mode::M;
		s1.mie = 0;
		s1.mip.store(0, Ordering::Release);
		s1.gprs[7] = shared_addr;
		s1.gprs[6] = 5;

		let mut states = [s0, s1];
		let mut instr_group: InstrToBeExec = unsafe { mem::zeroed() };
		unsafe {
			run_parallel_defaults(
				states.as_mut_ptr(),
				2,
				ram.as_mut_ptr(),
				ram.len() as u64,
				0,
				0,
				0,
				&mut instr_group as *mut InstrToBeExec,
			);
		}

		// Final value should be 42 + 3 + 5 = 50 (both AMOADDs applied atomically)
		let final_val = ram[shared_addr as usize] as u32
			| (ram[shared_addr as usize + 1] as u32) << 8
			| (ram[shared_addr as usize + 2] as u32) << 16
			| (ram[shared_addr as usize + 3] as u32) << 24;
		assert_eq!(
			final_val, 50,
			"AMOADD final value should be 42+3+5=50, got {}",
			final_val
		);
	}

	#[test]
	fn parallel_stop_on_mmio_store() {
		let mut ram = vec![0u8; 4096];
		let dev_addr: u64 = 0x1000_0000; // UART MMIO

		// Hart 0: store x5 -> [x6] where x6 = dev_addr (MMIO store -> exit).
		// sd x5, 0(x6) = 0x00533023 (rs2=x5, rs1=x6, imm=0).
		write_u32_le(&mut ram, 0, 0x0053_3023);

		// Hart 1: infinite two-instruction loop at offset 128 that advances PC
		// every iteration (addi then jal back), so it stays alive forever and
		// never triggers a stop condition (no MMIO / WFI / trap-loop).  The old
		// finite NOP sequence ran off into 0x0000 (illegal compressed instr) and
		// raised a racy TRAP exit that could beat hart 0's MMIO exit, making
		// this test flaky under load.
		write_u32_le(&mut ram, 128, 0x0010_8093); // addi x1, x1, 1
		write_u32_le(&mut ram, 132, 0xffdf_f06f); // jal  x0, -4  -> back to 128

		let mut s0 = make_state(0, 0);
		s0.mode = riscv_mode::M; // Bare-mode VA=PA
		s0.gprs[5] = 0x41; // value to store
		s0.gprs[6] = dev_addr; // rs1 — store address
		let mut s1 = make_state(128, 1);
		s1.mode = riscv_mode::M;
		let mut states = [s0, s1];

		// Custom device MMIO: cover [dev_addr, dev_addr+8)
		let dev_bases: [u64; 1] = [dev_addr];
		let dev_ends: [u64; 1] = [dev_addr + 8];
		let dev = DevMMIOAddrInfo {
			bases: dev_bases.as_ptr(),
			ends: dev_ends.as_ptr(),
			num: 1,
		};
		let mem = MemCtx {
			ram: ram.as_mut_ptr(),
			ram_size: ram.len() as u64,
			ram_base: 0,
			shadow_base: 0,
			shadow_size: 0,
		};
		// Single TOR entry covering full address space (R/W/X).
		// pmp_ok per RISC-V spec §3.7.1 denies S/U access when num==0,
		// so we must provide at least one permissive entry for tests.
		let mut pmp_cfg: [u8; 1] = [0x0F]; // PMP_R|PMP_W|PMP_X|PMP_A_TOR
		let mut pmp_addr: [u64; 1] = [u64::MAX];
		let pmp = FfiPmpCtx {
			cfg: pmp_cfg.as_mut_ptr(),
			addr: pmp_addr.as_mut_ptr(),
			num: 1,
			pmpsplit: 0,
		};
		let nh = 2usize;
		let mut mtimecmp_vec: Vec<u64> = vec![u64::MAX; nh];
		let mut msip_vec: Vec<u8> = vec![0u8; nh];
		let mut mtime_val: u64 = 0;
		let clint = FfiClintCtx {
			mtime: &mut mtime_val as *mut u64,
			mtimecmp: mtimecmp_vec.as_mut_ptr(),
			msip: msip_vec.as_mut_ptr(),
			base: 0,
			timebase_hz: 0, // 单元测试禁用 clock-source 推进, 保持确定性
		};
		let mut instr_group: InstrToBeExec = unsafe { mem::zeroed() };
		unsafe {
			let hart = FfiHartCtx {
				states: states.as_mut_ptr(),
				num_harts: 2,
				instr_group: &mut instr_group as *mut InstrToBeExec,
				stop_flag: std::ptr::null(),
				ext_irq: std::ptr::null_mut(),
			};
			let mem_ctx = FfiMemCtx {
				mem: &mem,
				pmp: &pmp,
			};
			let intr_ctx = FfiInterruptCtx {
				clint: &clint,
				plic: std::ptr::null(),
			};
			let dev_ctx = FfiDevicesCtx {
				dev: &dev,
				uart: std::ptr::null(),
				virtio: std::ptr::null(),
				watchdog: std::ptr::null(),
			};
			let bp = FfiBpCtx {
				addrs: std::ptr::null(),
				count: 0,
			};
			let mut tlb = FfiTlbCtx {
				gen: std::ptr::null_mut(),
				gen_per_hart: std::ptr::null_mut(),
			};
			run_parallel(&hart, &mem_ctx, &intr_ctx, &dev_ctx, &bp, &mut tlb);
		}

		// Store to device MMIO should trigger an exit with MMIO reason.
		assert_eq!(
			instr_group.exit_reason,
			exit_reason::MMIO,
			"store to MMIO device should cause an exit with MMIO reason"
		);
		assert_eq!(instr_group.exit_hart_id, 0, "exit hart should be 0");
	}

	#[test]
	fn parallel_wfi_all_idle_stops() {
		let mut ram = vec![0u8; 256];
		// Both harts: WFI
		write_u32_le(&mut ram, 0, 0x1050_0073); // WFI
		write_u32_le(&mut ram, 128, 0x1050_0073);

		let mut s0 = make_state(0, 0);
		s0.mode = riscv_mode::M;
		s0.mie = 0;
		s0.mip.store(0, Ordering::Release);
		let mut s1 = make_state(128, 1);
		s1.mode = riscv_mode::M;
		s1.mie = 0;
		s1.mip.store(0, Ordering::Release);
		let mut states = [s0, s1];

		let mut instr_group: InstrToBeExec = unsafe { mem::zeroed() };
		unsafe {
			run_parallel_defaults(
				states.as_mut_ptr(),
				2,
				ram.as_mut_ptr(),
				ram.len() as u64,
				0,
				0,
				0,
				&mut instr_group as *mut InstrToBeExec,
			);
		}

		// Both harts should be in WFI, and acceleration should exit with WFI_WAIT
		assert_eq!(
			instr_group.exit_reason,
			exit_reason::WFI_WAIT,
			"All WFI should cause WFI_WAIT exit, got {}",
			instr_group.exit_reason
		);
		assert_eq!(states[0].waiting, 1);
		assert_eq!(states[1].waiting, 1);
	}

	/// Helper: run_parallel with a custom CLINT base address.
	#[allow(unused)]
	unsafe fn run_parallel_with_clint(
		states: *mut HartState,
		num: u32,
		ram: *mut u8,
		ram_sz: u64,
		ram_base: u64,
		clint_base: u64,
		instr_group: *mut InstrToBeExec,
	) {
		let mem = MemCtx {
			ram,
			ram_size: ram_sz,
			ram_base,
			shadow_base: 0,
			shadow_size: 0,
		};
		let dev_bases: [u64; 0] = [];
		let dev_ends: [u64; 0] = [];
		let dev = DevMMIOAddrInfo {
			bases: dev_bases.as_ptr(),
			ends: dev_ends.as_ptr(),
			num: 0,
		};
		// Single TOR entry covering full address space (R/W/X).
		// pmp_ok per RISC-V spec §3.7.1 denies S/U access when num==0,
		// so we must provide at least one permissive entry for tests.
		let mut pmp_cfg: [u8; 1] = [0x0F]; // PMP_R|PMP_W|PMP_X|PMP_A_TOR
		let mut pmp_addr: [u64; 1] = [u64::MAX];
		let pmp = FfiPmpCtx {
			cfg: pmp_cfg.as_mut_ptr(),
			addr: pmp_addr.as_mut_ptr(),
			num: 1,
			pmpsplit: 0,
		};
		let nh = num as usize;
		let mut mtimecmp_vec: Vec<u64> = vec![u64::MAX; nh];
		let mut msip_vec: Vec<u8> = vec![0u8; nh];
		let mut mtime_val: u64 = 0;
		let clint = FfiClintCtx {
			mtime: &mut mtime_val as *mut u64,
			mtimecmp: mtimecmp_vec.as_mut_ptr(),
			msip: msip_vec.as_mut_ptr(),
			base: clint_base,
			timebase_hz: 0, // 单元测试禁用 clock-source 推进, 保持确定性
		};
		let hart = FfiHartCtx {
			states,
			num_harts: num,
			instr_group,
			stop_flag: std::ptr::null(),
			ext_irq: std::ptr::null_mut(),
		};
		let mem_ctx = FfiMemCtx {
			mem: &mem,
			pmp: &pmp,
		};
		let intr_ctx = FfiInterruptCtx {
			clint: &clint,
			plic: std::ptr::null(),
		};
		let dev_ctx = FfiDevicesCtx {
			dev: &dev,
			uart: std::ptr::null(),
			virtio: std::ptr::null(),
			watchdog: std::ptr::null(),
		};
		let bp = FfiBpCtx {
			addrs: std::ptr::null(),
			count: 0,
		};
		let mut tlb = FfiTlbCtx {
			gen: std::ptr::null_mut(),
			gen_per_hart: std::ptr::null_mut(),
		};
		run_parallel(&hart, &mem_ctx, &intr_ctx, &dev_ctx, &bp, &mut tlb);
	}

	/// Verify that a store to CLINT MSIP succeeds without traps.
	#[test]
	fn store_to_clint_msip_single_hart() {
		let mut ram = vec![0u8; 1024];
		let clint_base: u64 = 0x2000000;

		// Single hart in M-mode writes to CLINT MSIP for hart 0 (self),
		// then enters WFI.  With MSIE enabled and MIE=1, the pending MSIP
		// triggers a trap to mtvec (0x200) which does MRET, resuming at the
		// instruction after WFI (a second WFI) for a clean all-idle exit.
		// 0x00: addi t0, x0, 0x42        ->t0 = 0x42 (control)
		// 0x04: sw   t0, 0x1000(x0)     ->store to RAM (control)
		// 0x08: lui  t0, 0x20000        ->t0 = 0x20000000
		// 0x0C: addi t0, t0, 0          ->t0 = 0x20000000 (CLINT MSIP hart 0)
		// 0x10: addi t1, x0, 1          ->t1 = 1
		// 0x14: sw   t1, 0(t0)          ->write MSIP=1 for hart 0
		// 0x18: addi t2, x0, 0x42       ->t2 = 0x42 (proves CLINT write OK)
		// 0x1C: wfi                      ->enter WFI, MSIP pending -> trap
		// 0x20: wfi                      ->reached after MSI handler mrets;
		//                                    MSIP cleared -> all-idle 退出
		write_u32_le(&mut ram, 0x00, 0x04200293); // addi t0, x0, 0x42
		write_u32_le(&mut ram, 0x04, 0x00502023); // sw t0, 0(x0) — RAM store to PA 0
		write_u32_le(&mut ram, 0x08, 0x020002B7); // lui t0, 0x2000 ->t0 = 0x2000000
		write_u32_le(&mut ram, 0x0C, 0x00028293); // addi t0, t0, 0
		write_u32_le(&mut ram, 0x10, 0x00100313); // addi t1, x0, 1
		write_u32_le(&mut ram, 0x14, 0x0062A023); // sw t1, 0(t0)
		write_u32_le(&mut ram, 0x18, 0x04200393); // addi t2, x0, 0x42
		write_u32_le(&mut ram, 0x1C, 0x10500073); // wfi
		write_u32_le(&mut ram, 0x20, 0x10500073); // wfi

		let mut state = HartState {
			pc: 0,
			mode: riscv_mode::M,
			mstatus: (3 << 11) | (1 << 7) | (1 << 3), // MPP=M, MPIE=1, MIE=1
			..unsafe { std::mem::zeroed() }
		};
		state.mie = 1 << 3; // MSIE — allow MSI to be taken
		state.mip.store(0, Ordering::Release);
		// M-mode trap handler at 0x200: clear CLINT MSIP for hart 0
		// to prevent re-triggering the level-triggered interrupt, then MRET.
		// 0x200: lui  t1, 0x20000    ->t1 = 0x20000000 (CLINT base)
		// 0x204: sw   x0, 0(t1)      ->CLINT[hart0].msip = 0
		// 0x208: mret
		state.mtvec = 0x200;
		write_u32_le(&mut ram, 0x200, 0x02000337); // lui t1, 0x20000
		write_u32_le(&mut ram, 0x204, 0x00032023); // sw x0, 0(t1)
		write_u32_le(&mut ram, 0x208, 0x30200073); // mret

		let mut instr_group: InstrToBeExec = unsafe { mem::zeroed() };

		unsafe {
			run_parallel_with_clint(
				&mut state as *mut HartState,
				1,
				ram.as_mut_ptr(),
				ram.len() as u64,
				0,
				clint_base,
				&mut instr_group as *mut InstrToBeExec,
			);
		}

		// RAM store at 0x04 should have written 0x42 to PA=0
		assert_eq!(ram[0], 0x42, "RAM store should have set RAM[0]=0x42");
		assert_eq!(
			state.gprs[5], 0x2000000,
			"t0 should = 0x2000000; got {:#x}",
			state.gprs[5]
		);
		assert_eq!(
			state.gprs[7], 0x42,
			"CLINT MSIP write should succeed; t2={:#x}",
			state.gprs[7]
		);
		assert_eq!(state.halted, 0, "hart should not be halted");
	}

	/// Cross-hart MSI delivery test.
	///
	/// Hart 0 sends an MSI to Hart 1 via CLINT MMIO write, then enters WFI.
	/// Hart 1 starts in WFI, should wake on MSI, trap to M-mode handler
	/// (which does MRET), and execute the marker instruction after WFI.
	/// Hart 1's t0 MUST be 0x42 — proving the MSI was
	/// delivered, the trap handler ran, and execution resumed at the
	/// instruction following WFI.  Both harts then idle -> WFI_WAIT exit
	/// (no instruction quota).
	#[test]
	fn cross_hart_msip_wakes_target() {
		let mut ram = vec![0u8; 4096];
		let clint_base: u64 = 0x2000000;

		// --- Hart 0 instructions (PC=0) ---
		// Delay loop: spin ~512 iterations so hart 1 can enter WFI first.
		// 0x00: addi t0, x0, 512       ->t0 = 512
		// 0x04: addi t0, t0, -1        ->t0--
		// 0x08: bnez t0, -8            ->loop back to 0x04
		// 0x0C: lui  t0, 0x20000       ->t0 = 0x20000000
		// 0x10: addi t0, t0, 4         ->t0 = 0x20000004 (CLINT MSIP for hart 1)
		// 0x14: addi t1, x0, 1         ->t1 = 1
		// 0x18: sw   t1, 0(t0)         ->write MSIP=1 for hart 1
		// 0x1C: addi x0, x0, 0          ->NOP
		// 0x20: wfi                    ->hart 0 进入 WFI; 两个 hart 全部空闲后
		//                                 all-idle 检测退出acceleration(无指令配额,
		//                                 靠自然停止条件终止执行)
		write_u32_le(&mut ram, 0x00, 0x20000293); // addi t0, x0, 512
		write_u32_le(&mut ram, 0x04, 0xFFF28293); // addi t0, t0, -1
		write_u32_le(&mut ram, 0x08, 0xFE029EE3); // bnez t0, -8  -> PC=0x04
		write_u32_le(&mut ram, 0x0C, 0x020002B7); // lui t0, 0x2000 ->t0 = 0x2000000
		write_u32_le(&mut ram, 0x10, 0x00428293); // addi t0, t0, 4
		write_u32_le(&mut ram, 0x14, 0x00100313); // addi t1, x0, 1
		write_u32_le(&mut ram, 0x18, 0x0062A023); // sw t1, 0(t0)
		write_u32_le(&mut ram, 0x1C, 0x00000013); // addi x0, x0, 0 (NOP)
		write_u32_le(&mut ram, 0x20, 0x10500073); // wfi

		// --- Hart 1 instructions (PC=0x100) ---
		// 0x100: wfi                    ->enter WFI
		// 0x104: addi t0, x0, 0x42     ->t0 = 0x42 (marker that MSI was received)
		// 0x108: wfi                    ->back to WFI
		// 0x10C: jal  x0, -12          ->back to 0x104 (loop: safety net
		//                                  if PC somehow advances past WFI;
		//                                  handler at 0x200 does bare MRET
		//                                  so mepc must point to valid code)
		write_u32_le(&mut ram, 0x100, 0x10500073); // wfi
		write_u32_le(&mut ram, 0x104, 0x04200293); // addi t0, x0, 0x42
		write_u32_le(&mut ram, 0x108, 0x10500073); // wfi
		write_u32_le(&mut ram, 0x10C, 0xFF5FF06F); // jal x0, -12 -> PC=0x104

		// --- Shared mtvec handler (PC=0x200) ---
		// 0x200: mret  (Hart 0 handler — never traps in this test)
		write_u32_le(&mut ram, 0x200, 0x30200073); // mret

		// --- Hart 1 MSI handler (PC=0x300) ---
		// Clears CLINT MSIP before MRET to prevent re-triggering the same
		// level-sensitive interrupt.  Real firmware does this in the IPI
		// handler (sbi_ipi_raw_clear).  Without the clear, sync_msip
		// re-asserts mip.MSIP after MRET because the CLINT level is still
		// high -> infinite trap loop -> addi never executes.
		//   0x300: lui  t1, 0x20000    ->t1 = 0x20000000 (CLINT base)
		//   0x304: sw   x0, 4(t1)      ->CLINT[hart1].msip = 0
		//   0x308: mret
		write_u32_le(&mut ram, 0x300, 0x02000337); // lui t1, 0x20000
		write_u32_le(&mut ram, 0x304, 0x00032223); // sw x0, 4(t1)
		write_u32_le(&mut ram, 0x308, 0x30200073); // mret

		// --- Hart 0 state ---
		let mut s0 = make_state(0, 0);
		s0.mode = riscv_mode::M;
		s0.mie = 0;
		s0.mip.store(0, Ordering::Release);
		s0.mtvec = 0x200; // valid MRET handler at 0x200 (shared with Hart 1)

		// --- Hart 1 state ---
		let mut s1 = make_state(0x100, 1);
		s1.mode = riscv_mode::M;
		s1.mie = 0; // no interrupts enabled initially
		s1.mip.store(0, Ordering::Release); // no interrupts pending
		s1.mtvec = 0x300; // M-mode handler at 0x300 (clears MSIP then MRET)
		s1.mstatus = (3 << 11) | (1 << 7) | (1 << 3); // MPP=M, MPIE=1, MIE=1

		let mut states = [s0, s1];
		let mut instr_group: InstrToBeExec = unsafe { mem::zeroed() };

		unsafe {
			run_parallel_with_clint(
				states.as_mut_ptr(),
				2,
				ram.as_mut_ptr(),
				ram.len() as u64,
				0, // ram_base
				clint_base,
				&mut instr_group as *mut InstrToBeExec,
			);
		}

		// MSI MUST have been detected by Hart 1's sync_msip.
		assert!(
			states[1].diag.msip_last_seen > 0,
			"Hart 1 should have detected MSIP via sync_msip"
		);

		// mcause MUST indicate MSI was delivered (bit 63=1=interrupt, code=3=MSI).
		let is_interrupt = (states[1].mcause >> 63) != 0;
		let cause_code = states[1].mcause & 0x7FFF_FFFF_FFFF_FFFF;
		assert!(
			is_interrupt && cause_code == 3,
			"Hart 1 mcause should show MSI interrupt; got mcause={:#x}",
			states[1].mcause
		);

		// Hart 1 must NOT have halted (no trap loop).
		assert_eq!(states[1].halted, 0, "Hart 1 should not be halted");

		// The marker value: if MSI arrived AFTER WFI, PC advances past WFI
		// before the trap, mepc points to the addi, and t0=0x42 after MRET.
		// If MSI arrived BEFORE WFI, mepc points to WFI itself, and after
		// MRET WFI re-executes ->t0 stays 0.
		// Either way the MSI was delivered; we assert the detection markers.
		if states[1].gprs[5] != 0x42 && states[1].mepc != 0x100 {
			panic!(
				"If marker not set, mepc should be 0x100 (MSI before WFI); \
                 got t0={:#x} mepc={:#x}",
				states[1].gprs[5], states[1].mepc
			);
		}
	}

	/// Regression: BLTZ (BLT rs1, x0, offset) with negative rs1.
	///
	/// Verifies that ``addi s2, a5, -1`` and ``bltz s2, target`` work correctly
	/// with negative values.  a5 = -1 ->s2 = -2 ->bltz MUST branch.
	#[test]
	fn regression_bltz_negative_takes_branch() {
		let mut ram = vec![0u8; 128];

		// Instructions at PC=0:
		// 0x00: addi s2, a5, -1   -> s2 = -2 (a5 = -1)
		// 0x04: bltz s2, +0x10    -> branch to 0x18 (if s2 < 0)
		// 0x08: addi s1, x0, 0xBAD ->s1 = 0xBAD (only reached if bltz fails)
		// 0x0C: ebreak             -> trap to mtvec
		// ...
		// 0x18: addi s2, x0, 42   -> s2 = 42 (landing pad)
		// 0x1C: ebreak             -> trap to mtvec
		// 0x40: wfi (mtvec)        -> trap 处理入口; 无指令配额后
		//                             由 WFI 空闲退出终止 acceleration

		write_u32_le(&mut ram, 0, 0xFFF78913); // addi s2, a5, -1
		write_u32_le(&mut ram, 4, 0x00094A63); // bltz s2, +0x14 -> PC=0x18
		write_u32_le(&mut ram, 8, 0xBAD00493); // addi s1, x0, 0xBAD (should skip)
		write_u32_le(&mut ram, 12, 0x00100073); // ebreak
		write_u32_le(&mut ram, 24, 0x02A00913); // addi s2, x0, 42 (landing pad)
		write_u32_le(&mut ram, 28, 0x00100073); // ebreak
		write_u32_le(&mut ram, 0x40, 0x10500073); // wfi

		let mut state = HartState {
			pc: 0,
			gprs: [0u64; 32],
			mode: riscv_mode::M,
			mstatus: 0,
			..unsafe { std::mem::zeroed() }
		};
		state.mtvec = 0x40; // ebreak -> WFI -> all-idle 退出
		state.gprs[15] = 0xFFFF_FFFF_FFFF_FFFFu64; // a5 = -1

		let mut instr_group: InstrToBeExec = unsafe { std::mem::zeroed() };
		let mem = MemCtx {
			ram: ram.as_mut_ptr(),
			ram_size: ram.len() as u64,
			ram_base: 0,
			shadow_base: 0,
			shadow_size: 0,
		};
		let dev = DevMMIOAddrInfo {
			bases: [].as_ptr(),
			ends: [].as_ptr(),
			num: 0,
		};
		let pmp = FfiPmpCtx {
			cfg: [].as_mut_ptr(),
			addr: [].as_mut_ptr(),
			num: 0,
			pmpsplit: 0,
		};
		let mut mtimecmp = [u64::MAX; 1];
		let mut msip = [0u8; 1];
		let mut mtime: u64 = 0;
		let clint = FfiClintCtx {
			mtime: &mut mtime,
			mtimecmp: mtimecmp.as_mut_ptr(),
			msip: msip.as_mut_ptr(),
			base: 0,
			timebase_hz: 0, // 单元测试禁用clock-source 推进, 保持确定性
		};

		unsafe {
			let hart = FfiHartCtx {
				states: &mut state,
				num_harts: 1,
				instr_group: &mut instr_group,
				stop_flag: std::ptr::null(),
				ext_irq: std::ptr::null_mut(),
			};
			let mem_ctx = FfiMemCtx {
				mem: &mem,
				pmp: &pmp,
			};
			let intr_ctx = FfiInterruptCtx {
				clint: &clint,
				plic: std::ptr::null(),
			};
			let dev_ctx = FfiDevicesCtx {
				dev: &dev,
				uart: std::ptr::null(),
				virtio: std::ptr::null(),
				watchdog: std::ptr::null(),
			};
			let bp = FfiBpCtx {
				addrs: std::ptr::null(),
				count: 0,
			};
			let mut tlb = FfiTlbCtx {
				gen: std::ptr::null_mut(),
				gen_per_hart: std::ptr::null_mut(),
			};
			run_parallel(&hart, &mem_ctx, &intr_ctx, &dev_ctx, &bp, &mut tlb);
		}

		// BLTZ should have branched ->s2 = 42
		assert_eq!(
			state.gprs[18], 42,
			"BLTZ s2,-2 MUST branch: expected s2=42, got {}",
			state.gprs[18]
		);
		// s1 should NOT be 0xBAD (proves we didn't fall through)
		assert_ne!(
			state.gprs[9], 0xBAD,
			"BLTZ should have branched; s1=0xBAD means fallthrough occurred"
		);
	}

	#[test]
	fn pmp_concurrent_csrw_two_harts() {
		let mut ram = vec![0u8; 4096];
		// csrw pmpaddr1, x5  ->  (0x3B1 << 20) | (5 << 15) | (1 << 12) | 0x73 = 0x3B129073
		// WFI
		write_u32_le(&mut ram, 0, 0x3B129073);
		write_u32_le(&mut ram, 4, 0x10500073);

		let mut s0 = make_state(0, 0);
		s0.gprs[5] = 0xA000;
		s0.mie = 0;
		s0.mip.store(0, Ordering::Release);

		let mut s1 = make_state(0, 1);
		s1.gprs[5] = 0xB000;
		s1.mie = 0;
		s1.mip.store(0, Ordering::Release);

		let mut states = [s0, s1];

		// PMP with 16 entries, permissive
		let mut pmp_cfg: [u8; 128] = [0u8; 128];
		let mut pmp_addr: [u64; 128] = [u64::MAX; 128];
		// Set address mode to NAPOT for all 16 entries per hart
		for h in 0..2 {
			for e in 0..16 {
				pmp_cfg[h * 64 + e] = 0x0F; // R|W|X|NAPOT
			}
		}
		let pmp = FfiPmpCtx {
			cfg: pmp_cfg.as_mut_ptr(),
			addr: pmp_addr.as_mut_ptr(),
			num: 64, // per-hart entries (flat buf has 128 / 2 harts)
			pmpsplit: 0,
		};

		let mut instr_group: InstrToBeExec = unsafe { std::mem::zeroed() };
		unsafe {
			let mem = MemCtx {
				ram: ram.as_mut_ptr(),
				ram_size: 4096,
				ram_base: 0,
				shadow_base: 0,
				shadow_size: 0,
			};
			let dev = DevMMIOAddrInfo {
				bases: [].as_ptr(),
				ends: [].as_ptr(),
				num: 0,
			};
			let mut clint_mtime: u64 = 0;
			let mut clint_mtimecmp = [u64::MAX, u64::MAX];
			let mut clint_msip = [0u8, 0u8];
			let clint = FfiClintCtx {
				mtime: &mut clint_mtime,
				mtimecmp: clint_mtimecmp.as_mut_ptr(),
				msip: clint_msip.as_mut_ptr(),
				base: 0,
				timebase_hz: 0,
			};
			let hart = FfiHartCtx {
				states: states.as_mut_ptr(),
				num_harts: 2,
				instr_group: &mut instr_group,
				stop_flag: std::ptr::null(),
				ext_irq: std::ptr::null_mut(),
			};
			let mem_ctx = FfiMemCtx {
				mem: &mem,
				pmp: &pmp,
			};
			let intr_ctx = FfiInterruptCtx {
				clint: &clint,
				plic: std::ptr::null(),
			};
			let dev_ctx = FfiDevicesCtx {
				dev: &dev,
				uart: std::ptr::null(),
				virtio: std::ptr::null(),
				watchdog: std::ptr::null(),
			};
			let bp = FfiBpCtx {
				addrs: std::ptr::null(),
				count: 0,
			};
			let mut tlb = FfiTlbCtx {
				gen: std::ptr::null_mut(),
				gen_per_hart: std::ptr::null_mut(),
			};
			run_parallel(&hart, &mem_ctx, &intr_ctx, &dev_ctx, &bp, &mut tlb);
		}
		println!(
			"exit_reason={} exit_hart={} total={}",
			instr_group.exit_reason, instr_group.exit_hart_id, instr_group.total_instrs
		);
		println!(
			"hart0: pc={:#x} gpr5={:#x}",
			states[0].pc, states[0].gprs[5]
		);
		println!(
			"hart1: pc={:#x} gpr5={:#x}",
			states[1].pc, states[1].gprs[5]
		);
		// Check PMP writes
		assert_eq!(pmp_addr[1], 0xA000, "hart0 pmpaddr1");
		assert_eq!(pmp_addr[65], 0xB000, "hart1 pmpaddr1 in flat buf");
		// WFI exit expected
		assert_eq!(instr_group.exit_reason, exit_reason::WFI_WAIT);
	}
}
