//! 每个 hart 分配一个线程的并发执行引擎.
//!
//! ``run_parallel`` 是 FFI 入口, Python 侧得到链接库的调用执行能力.
use std::cell::Cell;
use std::sync::{
	atomic::AtomicBool, atomic::AtomicU32, atomic::AtomicU64, atomic::AtomicU8, atomic::Ordering,
	Arc, Mutex,
};

// ============================================================
// ============================================================
//
use crate::hart_sched::*;

use crate::ffi::{
	FfiBpCtx, FfiClintCtx, FfiDevicesCtx, FfiExtIrqCtx, FfiHartCtx, FfiInterruptCtx, FfiMemCtx,
	FfiPlicCtx, FfiTlbCtx, FfiUartCtx, FfiVirtIoCtx, InstrToBeExec, EMPTY_UART,
};
use crate::state::{exit_reason, HartState};

// ============================================================
//  FFI 裸指针的 Send 与 Sync 包装类型
// ============================================================
//
// Python 在整个 FFI 调用期间占有对内存的使用权，例如 bytearray, ctypes 数组.
// 这些包装类型断言 Send 与 Sync, 使其可以安全地跨线程边界传递.

/// 设置 ``MemCtx`` 的 Send 与 Sync 类型.
///
/// # Safety
/// Python 侧的 FFI 调用方必须保持其传递的内存是有效的.
#[derive(Clone, Copy)]
pub(crate) struct SharedMemCtx {
	pub ram: *mut u8,
	pub ram_size: u64,
	pub ram_base: u64,
	pub shadow_base: u64,
	pub shadow_size: u64,
	/// 每个 hart 的 LR 与 SC 预留槽位. LR 把 slot[hid] 置为物理地址,
	/// 任一存储指令清除所有槽位. 该数组在 ``ModuleState`` 中分配.
	pub lr_reserved: *mut AtomicU64,
}
unsafe impl Send for SharedMemCtx {}
unsafe impl Sync for SharedMemCtx {}

/// 包装 PMP 指针的 Send 与 Sync 类型.
#[derive(Clone, Copy)]
pub(crate) struct SharedPmpCtx {
	pub cfg: *mut u8,
	pub addr: *mut u64,
	pub num: u8,
}
unsafe impl Send for SharedPmpCtx {}
unsafe impl Sync for SharedPmpCtx {}

/// 包装设备 MMIO 指针的 Send 与 Sync 类型.
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
//  并发安全的 CLINT 上下文结构
// ============================================================

/// 可供各 hart 线程并发访问的 CLINT 状态.
///
/// 全部字段为原子类型, 或在初始化之后只读.
/// 指针来自 Python ctypes 数组的类型转换; 在 x86-64 上 ``AtomicU64`` 与 ``u64`` 的
/// 布局相同, 即 8 字节大小与 8 字节对齐; ``AtomicU8`` 与 ``u8`` 的布局相同,
/// 即 1 字节大小与 1 字节对齐.
///
/// MSIP 建模为边沿计数器而非电平: 每次写入 1 都递增该 hart 的计数器,
/// ``sync_msip`` 与此前记录的取值比较以识别新的边沿. 连续写入 1 而中间没有
/// 写入 0 的情形因此不会漏检, 纯电平检测会漏掉这种情形.
pub struct ConcurrentClintCtx {
	pub base: u64,
	/// 共享 mtime 计数器, 由 ``advance_clock_source`` 按已执行的指令条数
	/// 推进, 据此换算受调试程序的 mtime.
	pub mtime: *const AtomicU64,
	/// 每个 hart 的 mtimecmp 寄存器, 按 hart_id 索引.
	pub mtimecmp: *const AtomicU64,
	/// 每个 hart 的 MSIP 电平字节, 按 hart_id 索引.
	pub msip: *const AtomicU8,
	/// mtime 时钟源频率, 单位为 Hz; 为 0 时表示禁用时钟源.
	pub timebase_hz: u64,
	pub num_harts: u32,
	pub msip_pending: Cell<*const AtomicU64>,
	pub hart_threads: Cell<*const std::thread::Thread>,
	/// 每个 hart 的 ``HartState`` 数组.
	///
	/// 双重用途:
	/// 1. 跨 hart 读取 ``stimecmp`` (SSTC). ``wfi_check_all_idle`` 据此在全部
	///    hart 处于 WFI 空闲时确定下一截止时间并以 WFI_WAIT 退出. 此处不推进
	///    mtime; Python 侧按真实剩余休眠时间与流逝时间补偿 mtime. 传统 PLIC 模式与
	///    AIA 模式都需要该指针, 内核的时钟事件设备在传统 PLIC 模式下同样使用 SSTC.
	/// 2. AIA 模式下 ``try_handle_imsic_concurrent`` 跨 hart 置位或清除目标 hart 的
	///    IMSIC eip 位, 即 seteipnum 与 clreipnum 的 MMIO 访问.
	///
	/// 由 ``run_parallel`` 在成功解析 FFI 后无条件写入.
	pub hart_states: Cell<*const HartState>,
}

// 安全性: Python 在 FFI 调用期间保持 ctypes 数组有效, 全部指针都指向正确对齐的原子类型.
unsafe impl Send for ConcurrentClintCtx {}
unsafe impl Sync for ConcurrentClintCtx {}

impl ConcurrentClintCtx {
	/// 由 FFI 的 ``FfiClintCtx`` 裸指针构造.
	///
	/// # Safety
	///
	/// 调用方必须保证底层 ctypes 数组的生命周期长于本结构.
	/// 在 x86-64 上把 ``*mut u64`` 转换为 ``*const AtomicU64`` 是合法的,
	/// 因为两种类型的大小与对齐相同.
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

	/// 取 hart_id 对应的 MSIP 待处理状态, 未初始化时返回空指针.
	///
	/// 槽位地址为 ``msip_pending`` 数组首地址加 ``hart_id * MSIP_PENDING_STRIDE``,
	/// 与 ``ModuleState::new`` 的分配跨步一致.
	#[inline]
	pub(crate) fn msip_pending_slot(&self, hart_id: usize) -> *const AtomicU64 {
		let base = self.msip_pending.get();
		if base.is_null() {
			return std::ptr::null();
		}
		unsafe { base.add(hart_id * MSIP_PENDING_STRIDE) }
	}
}

// ============================================================
// ============================================================

/// hart 置位全局停止标志的原因.
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

/// 每 hart 的 MSIP 待处理槽位占用的元素个数 — hart 索引乘本值即该 hart 的槽位下标,
/// 相邻 hart 的槽位相距 64 字节.
///
/// 槽位由向其发送 IPI 的 hart 置位, 由本 hart 的 ``sync_msip`` 读取并写入 0.
pub const MSIP_PENDING_STRIDE: usize = 8;

/// 各 hart 线程共享的全局协调状态.
///
/// 每次 ``run_parallel`` 调用对应一个实例
pub struct ModuleState {
	/// 为真时, 全部 hart 线程必须在下一条指令退出.
	pub stop_flag: AtomicBool,
	/// 停止事件的具体信息, 由触发停止的 hart 写入.
	/// 由 ``Mutex`` 保护, 保证只有一个 hart 写入.
	pub stop_info: Mutex<StopInfo>,
	/// 当前处于 WFI 自旋等待的 hart 数量.
	pub wfi_count: AtomicU32,
	/// 每个 hart 的 WFI 状态标志, 下标为 hart_id.
	/// hart 进入 WFI 时把自身槽位置为 1, 离开时置为 0.
	pub wfi_flags: Box<[AtomicU8]>,
	/// 未停机的 hart 数量. 在线程创建之前设置一次, 之后只读.
	pub active_hart_num: u32,
	/// 已发生的 TLB 刷新次数. 使既有翻译条目失效的动作各算一次: 受调试程序
	/// 执行 SFENCE.VMA 或改写 satp, 调试器刷新 TLB 或回滚快照, 装入固件.
	/// 该计数由调用方持有, 引擎在每轮开始时装入, 结束时写回, 故两次调用之间
	/// 发生的刷新对引擎内的条目同样生效.
	///
	/// 用计数而不是一个"已失效"标志: 标志上的清除与置位并发时, 清除可能覆盖
	/// 紧随其后的置位, 使某个 hart 漏掉一次失效; 计数只增不减, 各 hart 比较
	/// 取值即可判定, 多标记不影响正确性. 由此不必为跨 hart 失效发送核间中断.
	pub tlb_flush_cnt: AtomicU64,
	/// 各 hart 上次见到的 ``tlb_flush_cnt``. 与 ``tlb_flush_cnt`` 不相等时,
	/// 该 hart 在取下一条指令或访问数据之前把自己两个 TLB 的条目全部标为陈旧,
	/// 下一次访问重新遍历页表.
	pub tlb_flush_cnt_per_hart: Box<[AtomicU64]>,
	/// 调用方是否给出了上述两处计数的存放位置. 为真时计数跨调用保留, 引擎
	/// 进入执行循环时不清空任何条目. 为假时计数只在本轮内有效, 各 hart 在
	/// 进入执行循环之前清空自己的两个 TLB — 上一轮留下的条目是否仍然有效
	/// 没有判定依据.
	pub is_flush_cnt_persistent: bool,
	/// 每个 hart 的 LR 与 SC 预留内存空间. 用于保存该 hart 预留的物理地址,
	/// 0 表示没有预留. 任一存储指令写入同一物理地址时清除所有 hart 上匹配
	/// 的预留, 保证多 hart 下 LR 与 SC 的语义正确. RISC-V 规范 §8.2 规定:
	/// 某 hart 上成功的 SC 使该 hart 上的任何 LR 失效, 任何 hart 上成功的
	/// AMO、SC 或存储指令写入预留集合时, 使该 hart 上的任何 LR 失效.
	pub lr_reserved: Box<[AtomicU64]>,
	/// 每个 hart 的跨线程 MSIP 通知. 其余 hart 写入该 hart 的 CLINT MSIP 时,
	/// 在此处以 Release 原子地置位; 本 hart 的 ``sync_msip`` 以 swap(0, Acquire)
	/// 读出并清除, 再合并进 ``HartState.mip``. 这样避免了发送方直接写入与接收方
	/// ``sync_mtip`` 与 ``sync_msip`` 调用之间对 ``HartState.mip`` 的非原子读改写竞争.
	pub msip_pending: Box<[AtomicU64]>,
	pub st_time_val: std::time::Instant,
	/// ``marshal_hart`` 写入的 mtime 基准值.
	pub time_base_val: u64,
	/// 每 hart 在本轮加速执行开始时的指令计数, 由 ``marshal_hart`` 记录.
	/// ``advance_clock_source`` 用 ``state.total_instrs - instr_ref_num[hid]``
	pub instr_ref_num: Box<[u64]>,
}

impl ModuleState {
	pub fn new(
		num_harts: u32,
		active_hart_num: u32,
		init_flush_cnt: u64,
		init_flush_cnt_per_hart: Box<[u64]>,
		is_flush_cnt_persistent: bool,
		time_base_val: u64,
		instr_ref_num: Box<[u64]>,
	) -> Self {
		let mut v = Vec::with_capacity(num_harts as usize);
		let mut gv = Vec::with_capacity(num_harts as usize);
		let mut rv = Vec::with_capacity(num_harts as usize);
		let mut mv = Vec::with_capacity(num_harts as usize * MSIP_PENDING_STRIDE);
		for hid in 0..num_harts as usize {
			v.push(AtomicU8::new(0));
			gv.push(AtomicU64::new(init_flush_cnt_per_hart[hid]));
			rv.push(AtomicU64::new(0));
		}
		for _ in 0..num_harts as usize * MSIP_PENDING_STRIDE {
			mv.push(AtomicU64::new(0));
		}
		ModuleState {
			stop_flag: AtomicBool::new(false),
			stop_info: Mutex::new(StopInfo::empty()),
			wfi_count: AtomicU32::new(0),
			wfi_flags: v.into_boxed_slice(),
			active_hart_num,
			tlb_flush_cnt: AtomicU64::new(init_flush_cnt),
			tlb_flush_cnt_per_hart: gv.into_boxed_slice(),
			is_flush_cnt_persistent,
			lr_reserved: rv.into_boxed_slice(),
			msip_pending: mv.into_boxed_slice(),
			st_time_val: std::time::Instant::now(),
			time_base_val,
			instr_ref_num,
		}
	}

	/// 使能停止标志并记录原因. 只有第一个调用方生效, 后续调用方被忽略.
	pub fn request_stop(&self, info: StopInfo) {
		if self.stop_flag.swap(true, Ordering::Release) {
			return; // 已处于停止状态
		}
		if let Ok(mut guard) = self.stop_info.lock() {
			*guard = info;
		}
	}

	/// 判定未停机的 hart 是否全部处于 WFI 等待.
	#[inline]
	pub fn all_in_wfi(&self) -> bool {
		self.wfi_count.load(Ordering::Acquire) >= self.active_hart_num
	}
}

// ============================================================
//  FFI 入口辅助函数
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

/// 软件看门狗轮询周期, 单位为微秒, 兼作看门狗自身阻塞的周期.
///
/// 处于 WFI 等待的 hart 按此周期自行醒来重新检查停止标志.
pub(crate) const WATCHDOG_POLL_US: u64 = 5_000;

/// 唤醒所有已注册的 hart 线程. WFI 中的 hart 以 ``park_timeout`` 阻塞,
/// 由看门狗或完成信号调用 ``unpark`` 显式唤醒.
fn unpark_all_harts(clint: &ConcurrentClintCtx) {
	let ptr = clint.hart_threads.get();
	if ptr.is_null() {
		return;
	}
	// 安全前提: ``hart_threads`` 由 ``run_harts`` 在线程全部创建后写入,
	// 并在 ``join`` 之前保持有效; 本函数只在 ``run_harts`` 的生命周期内调用.
	let slice = unsafe { std::slice::from_raw_parts(ptr, clint.num_harts as usize) };
	for t in slice {
		t.unpark();
	}
}

/// 软件看门狗线程
///
/// WFI 中的 hart 以 ``park_timeout(WATCHDOG_POLL_US)`` 从睡眠中唤醒. 看门狗按固定
/// 周期调用 ``unpark_all_harts`` 唤醒全部 WFI hart, 被唤醒的 hart 重新执行
/// ``wfi_sync_and_check``, 经 ``sync_mtip``、``sync_msip`` 与 ``sync_imsic``
/// 判定定时器、软件与外部中断是否挂起. 挂起则投递中断, 否则重新阻塞.
///
/// 由 ``advance_clock_source`` 推进 mtime; 全部 hart 处于 WFI 时由
/// ``wfi_check_all_idle`` 确定下一截止时间并以 WFI_WAIT 退出, Python 侧按
/// 剩余时间休眠并按睡眠流逝时间补偿推进 mtime.
fn watchdog_loop(
	module: &ModuleState,
	clint: &ConcurrentClintCtx,
	stop_flag: *const u8,
	uart_rx_notify: *const u8,
	timeout_ns: u64,
) {
	loop {
		// 1. 时钟源超时, 即终止条件: 越过 timeout_ns 时以 TIMEOUT 原因停止.
		if timeout_ns != 0 && module.st_time_val.elapsed().as_nanos() as u64 >= timeout_ns {
			module.request_stop(StopInfo {
				reason: exit_reason::TIMEOUT,
				..StopInfo::empty()
			});
			unpark_all_harts(clint);
			return;
		}

		// 2. 停止与 RX 检查: 有信号则唤醒全部 WFI hart, 使其自行退出.
		let internal_stop = module.stop_flag.load(Ordering::Acquire);
		let external_stop = !stop_flag.is_null() && unsafe { *stop_flag != 0 };
		let rx_ready = !uart_rx_notify.is_null() && unsafe { *uart_rx_notify != 0 };
		if internal_stop || external_stop || rx_ready {
			unpark_all_harts(clint);
		}
		if internal_stop || external_stop {
			return;
		}

		// 3. 周期性地向中断控制器发置位请求信号: unpark 全部 WFI hart, 使其重新
		//    执行 wfi_sync_and_check 判定定时器、软件与外部中断是否挂起并投递.
		//    mtime 由活动的 hart 按指令推进, 看门狗不做截止预测, 按固定周期唤醒.
		unpark_all_harts(clint);
		std::thread::park_timeout(std::time::Duration::from_micros(WATCHDOG_POLL_US));
	}
}

/// 每个 hart 创建一个操作系统线程, 运行 ``hart_worker``, 并等待全部线程结束.
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

	// 看门狗是单轮加速执行内唯一的轮询者, 周期性唤醒处于 WFI 的 hart.
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
			// 完成信号: 唤醒其余处于 WFI 的 hart 与看门狗, 使单轮加速执行尽快结束.
			unpark_all_harts(clint_ptr);
			wd_thread.unpark();
		});
		thread_refs.push(handle.thread().clone());
		handles.push(handle);
	}

	let thread_slice: Box<[std::thread::Thread]> = thread_refs.into_boxed_slice();
	cc_clint.hart_threads.set(thread_slice.as_ptr());
	std::mem::forget(thread_slice); // 指针在 run_harts 返回之前保持有效

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

/// 把本轮结束时的刷新计数写回调用方持有的两处存放位置.
///
/// 两处指针必须同时给出: 只给出其中之一时, 本轮执行循环的刷新计数从 0 起算,
/// 写回会把调用方持有的取值改小, 故一律不写回.
#[inline]
unsafe fn writeback_tlb_flush_cnt(
	flush_cnt_ptr: *mut u64,
	flush_cnt_per_hart_ptr: *mut u64,
	module: &ModuleState,
	num_harts: u32,
) {
	if flush_cnt_ptr.is_null() || flush_cnt_per_hart_ptr.is_null() {
		return;
	}
	*flush_cnt_ptr = module.tlb_flush_cnt.load(Ordering::Relaxed);
	for i in 0..num_harts as usize {
		*flush_cnt_per_hart_ptr.add(i) = module.tlb_flush_cnt_per_hart[i].load(Ordering::Relaxed);
	}
}

#[inline]
unsafe fn writeback_mtime(clint_raw: &FfiClintCtx, cc_clint: &ConcurrentClintCtx) {
	*clint_raw.mtime = (*cc_clint.mtime).load(Ordering::SeqCst);
}

/// 纳秒与秒的换算常数, 用于把 Duration 换算为时钟计数.
const NS_PER_SEC: u64 = 1_000_000_000;

/// 指令计数推进的换算比率, 即每条指令对应的受调试程序时间, 单位为纳秒.
///
/// 取值为 20, 即每条指令 20 纳秒.
pub const NS_PER_INSTR: u64 = 20;

/// 按指令计数推进 mtime, 时钟源只取指令条数, 不含实时分量.
///
/// - 单一时钟源
/// - 单调递增
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
	// instr_delta * timebase_hz * NS_PER_INSTR 的峰值约为 4M * 10MHz = 4e13,
	// 远小于 u64 上限, 无需使用 u128.
	let ticks = instr_delta
		.saturating_mul(clint.timebase_hz)
		.saturating_mul(NS_PER_INSTR)
		/ NS_PER_SEC;
	let target = module.time_base_val.wrapping_add(ticks);
	// mtime 只增不减, 目标值不超过当前值时该操作不改变任何值.
	if unsafe { &*clint.mtime }.load(Ordering::Relaxed) < target {
		unsafe { &*clint.mtime }.fetch_max(target, Ordering::Relaxed);
	}
}

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
	let ticks_per_instr_scale = timebase_hz.saturating_mul(NS_PER_INSTR);
	if ticks_per_instr_scale == 0 {
		return 0;
	}
	let ticks = timecmp - now_mtime;
	let offset = ticks
		.saturating_mul(NS_PER_SEC) // 向上取整
		.saturating_add(ticks_per_instr_scale - 1)
		.checked_div(ticks_per_instr_scale)
		.unwrap_or(u64::MAX);
	own_total_instrs.wrapping_add(offset)
}

// ============================================================
//  FFI 入口
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
	// 1. 解包 FFI 结构体
	let hart = unsafe { &*hart };
	let mem_ctx = unsafe { &*mem_ctx };
	let intr_ctx = unsafe { &*intr_ctx };
	let dev_ctx = unsafe { &*dev_ctx };
	let states = hart.states;
	let num_harts = hart.num_harts;

	// 2. 把 instr_group 清零并构造上下文
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
	// 无条件设置 stimecmp (SSTC).
	cc_clint.hart_states.set(states as *const HartState);

	// 3. 模块状态, 其中 TLB 刷新计数跨调用保留, 见 FfiTlbCtx
	let flush_cnt_ptr = if tlb.is_null() {
		std::ptr::null_mut()
	} else {
		unsafe { (*tlb).flush_cnt }
	};
	let flush_cnt_per_hart_ptr = if tlb.is_null() {
		std::ptr::null_mut()
	} else {
		unsafe { (*tlb).flush_cnt_per_hart }
	};
	// 起算. 从 0 起算时, 只要调用方一侧的刷新次数不为 0, 每个 hart 在每轮的第一条
	// 指令处都会判定为陈旧, 引擎内的翻译在每轮边界全部作废.
	let is_flush_cnt_persistent = !flush_cnt_ptr.is_null() && !flush_cnt_per_hart_ptr.is_null();
	let init_flush_cnt = if is_flush_cnt_persistent {
		unsafe { *flush_cnt_ptr }
	} else {
		0
	};
	let init_flush_cnt_per_hart: Box<[u64]> = if is_flush_cnt_persistent {
		(0..num_harts as usize)
			.map(|hid| unsafe { *flush_cnt_per_hart_ptr.add(hid) })
			.collect()
	} else {
		vec![0u64; num_harts as usize].into_boxed_slice()
	};
	let instr_ref_num: Box<[u64]> = (0..num_harts)
		.map(|hid| unsafe { (*states.add(hid as usize)).total_instrs })
		.collect();
	let module = Arc::new(ModuleState::new(
		num_harts,
		active,
		init_flush_cnt,
		init_flush_cnt_per_hart,
		is_flush_cnt_persistent,
		unsafe { *clint_raw.mtime },
		instr_ref_num,
	));
	cc_clint.msip_pending.set(module.msip_pending.as_ptr());

	let timeout_ns = if dev_ctx.watchdog.is_null() {
		0
	} else {
		unsafe { (*dev_ctx.watchdog).timeout_ns }
	};

	// 4. 构造共享上下文
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
		// 5. 创建并等待 hart 线程
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
		// 6. 在同一个 unsafe 块中收集结果
		collect_stop(&module, hart.instr_group);
		sum_instrs(states, num_harts, hart.instr_group);
		writeback_tlb_flush_cnt(flush_cnt_ptr, flush_cnt_per_hart_ptr, &module, num_harts);
		writeback_mtime(clint_raw, &cc_clint);
	}
}

// ============================================================
//  测试
// ============================================================

#[cfg(test)]
mod tests {
	use super::*;
	// 从本文件拆分到同级模块的条目, 在此重新引入作用域,
	// 使测试可以直接引用.
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

	/// 构造长度为零的 PMP 与设备数组, 并调用 ``run_parallel``.
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
		// 单条 TOR 条目覆盖全部地址空间, 权限为读、写与执行.
		// 按 RISC-V 规范 §3.7.1, num 为 0 时 pmp_ok 拒绝 S 模式与 U 模式的访问,
		// 故测试至少需要一条宽松条目.
		let mut pmp_cfg: [u8; 1] = [0x0F]; // PMP_R, PMP_W, PMP_X 与 PMP_A_TOR
		let mut pmp_addr: [u64; 1] = [u64::MAX];
		let pmp = FfiPmpCtx {
			cfg: pmp_cfg.as_mut_ptr(),
			addr: pmp_addr.as_mut_ptr(),
			num: 1,
			pmpsplit: 0,
		};
		// 按实际的 hart 数量分配每个 hart 的 CLINT 数组.
		let nh = num as usize;
		let mut mtimecmp_vec: Vec<u64> = vec![u64::MAX; nh];
		let mut msip_vec: Vec<u8> = vec![0u8; nh];
		let mut mtime_val: u64 = 0;
		let clint = FfiClintCtx {
			mtime: &mut mtime_val as *mut u64,
			mtimecmp: mtimecmp_vec.as_mut_ptr(),
			msip: msip_vec.as_mut_ptr(),
			base: 0,
			timebase_hz: 0, // 单元测试中不推进时钟源, 保持确定性
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
			flush_cnt: std::ptr::null_mut(),
			flush_cnt_per_hart: std::ptr::null_mut(),
		};
		run_parallel(&hart, &mem_ctx, &intr_ctx, &dev_ctx, &bp, &mut tlb);
	}

	#[test]
	fn parallel_single_hart_addi() {
		let mut ram = vec![0u8; 256];
		// PC=0 处的指令:
		write_u32_le(&mut ram, 0, 0x00A0_0293); // addi x5, x0, 10
		write_u32_le(&mut ram, 4, 0x0052_8293); // addi x5, x5, 5
		write_u32_le(&mut ram, 8, 0x1050_0073); // WFI, 用于正常退出

		let mut state = make_state(0, 0);
		state.mode = riscv_mode::M; // M 模式下 mip 为 0 时 WFI 进入等待
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

	/// 辅助核上线握手 (cpuhp_ap_sync_alive) 死锁.
	#[test]
	fn wfi_stimecmp_deadline_exits_wfi_wait_in_legacy_mode() {
		// 设置 CFG_AIA 为 false 时的 stimecmp 唤醒路径.
		// AIA 构建下该路径进入 IMSIC 分支, 故跳过本用例.
		if CFG_AIA {
			return;
		}

		let mut ram = vec![0u8; 256];
		write_u32_le(&mut ram, 0, 0x1050_0073); // WFI

		let mut state = make_state(0, 0);
		state.mode = riscv_mode::M; // WFI 在 M 模式且 mip=0 时进入等待
		state.mie = 0;
		state.mip.store(0, Ordering::Release);
		state.stimecmp = 5000; // 大于当前 mtime 的 SSTC 截止时间

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
		// 单条 TOR 条目覆盖全部地址空间, 权限为 RWX, 避免 pmp_ok 因 num 为 0 拒绝访问.
		let mut pmp_cfg: [u8; 1] = [0x0F];
		let mut pmp_addr: [u64; 1] = [u64::MAX];
		let pmp = FfiPmpCtx {
			cfg: pmp_cfg.as_mut_ptr(),
			addr: pmp_addr.as_mut_ptr(),
			num: 1,
			pmpsplit: 0,
		};

		// mtimecmp 置为 u64::MAX, 即没有截止; stimecmp 为 5000, 即唯一截止.
		// mtime 由本用例保持初值, 以便断言其未被改写.
		let nh = 1usize;
		let mut mtimecmp_vec: Vec<u64> = vec![u64::MAX; nh];
		let mut msip_vec: Vec<u8> = vec![0u8; nh];
		let mut mtime_val: u64 = 0;
		let clint = FfiClintCtx {
			mtime: &mut mtime_val as *mut u64,
			mtimecmp: mtimecmp_vec.as_mut_ptr(),
			msip: msip_vec.as_mut_ptr(),
			base: 0,
			timebase_hz: 0, // 不推进时钟源, mtime 保持初值以断言未被改写
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
				flush_cnt: std::ptr::null_mut(),
				flush_cnt_per_hart: std::ptr::null_mut(),
			};
			run_parallel(&hart, &mem_ctx, &intr_ctx, &dev_ctx, &bp, &mut tlb);
		}
		assert_eq!(
			mtime_val, 0,
			"timebase_hz 为 0 时引擎不按指令推进 mtime, 本轮以 WFI_WAIT 退出, mtime 保持初值"
		);
		assert_eq!(
            instr_group.exit_reason, exit_reason::WFI_WAIT,
            "stimecmp 截止必须被识别并以 WFI_WAIT 退出 (legacy 模式) — hart_states 必须在非 AIA 下也设置"
        );
	}

	///
	/// 无限循环不会自行结束, 故本轮的出口只有两个: 引擎因 RX 通知退出, 或看门狗在
	/// 300 毫秒后以超时结束本轮. 后者使本用例表现为断言失败而不是挂起.
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

		// termio 线程发现标准输入的新字节后置位的通知位.
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
		// 单条 TOR 条目覆盖全部地址空间, 权限为读、写与执行, 避免 pmp_ok 因 num 为 0 拒绝访问.
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
			timebase_hz: 0, // 不推进时钟源, 保持确定性
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
			flush_cnt: std::ptr::null_mut(),
			flush_cnt_per_hart: std::ptr::null_mut(),
		};
		run_parallel(&hart, &mem_ctx, &intr_ctx, &dev_ctx, &bp, &mut tlb);
		(instr_group.exit_reason, instr_group.total_instrs)
	}

	/// termio 把标准输入的字节写入 RX 环形缓冲区并置位 RX 通知时, 若 IMSIC 独占
	/// 外部中断线路, 即 eidelivery 非 0, 引擎必须在执行任何指令之前退出本轮加速执行.
	///
	/// 字节进入 UART FIFO 需要 Python 侧从 RX 环形缓冲区写入, 该步骤在单轮
	/// 加速执行内不可见, 故必须退出, 使 Python 尽快完成写入. 中断投递本身不依赖
	/// 此退出: eip 的新增位由 Python 经 ``IMSIC._publish_eip`` 写入共享数组,
	/// ``sync_ext_irq_mip`` 在轮内即可据此置位 SEIP.
	///
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

	/// IMSIC 未占用外部中断线路, 即传统 PLIC 模式时, RX 通知不得
	/// 触发退出. ``sync_ext_irq_mip`` 已在本轮内联置位 SEIP, 受调试程序在轮内
	/// 即可读出字节, 退出会降低传统 PLIC 模式的输入吞吐. 受调试程序为无限循环,
	/// 唯一的结束方式是看门狗超时, 故以 TIMEOUT 确认未退出.
	#[test]
	fn rx_notify_does_not_exit_batch_without_imsic() {
		let (reason, _) = unsafe { run_rx_notify_case(false) };
		assert_eq!(
			reason,
			exit_reason::TIMEOUT,
			"IMSIC 未占用外部中断线路时不得为 RX 通知退出本轮 (内联置位已足够)"
		);
	}

	/// ECALL 在轮内投递陷阱: 模式变为 M, PC 变为 mtvec, mepc 为 ECALL 的地址.
	/// mtvec 处的 M 模式处理函数把 mepc 加 4 以跳过 ECALL, 执行 MRET 返回 S 模式,
	/// 随后执行 WFI 退出.
	#[test]
	fn parallel_ecall_inline_trap() {
		let mut ram = vec![0u8; 256];
		// PC=0 处的 S 模式代码:
		write_u32_le(&mut ram, 0, 0x0010_0293); // addi x5, x0, 1
		write_u32_le(&mut ram, 4, 0x0000_0073); // ECALL, 从 S 模式陷入 M 模式
		write_u32_le(&mut ram, 8, 0x1050_0073); // WFI, MRET 之后执行

		// mtvec 为 0x20 处的 M 模式处理函数, 使用 t1, 即 x6, 以免改写 x5:
		let mtvec_addr: u64 = 0x20;
		// addi t1, x0, 4 编码为 0x00400313, 使 t1 为 4
		write_u32_le(&mut ram, 0x20, 0x0040_0313);
		// csrr t1, mepc 编码为 0x34102373, 读取 mepc
		write_u32_le(&mut ram, 0x24, 0x3410_2373);
		// addi t1, t1, 4 编码为 0x00430313, 使 t1 为 mepc + 4
		write_u32_le(&mut ram, 0x28, 0x0043_0313);
		// csrw mepc, t1 编码为 0x34131073, 写入 mepc
		write_u32_le(&mut ram, 0x2C, 0x3413_1073);
		// mret 编码为 0x30200073
		write_u32_le(&mut ram, 0x30, 0x3020_0073);

		let mut state = make_state(0, 0);
		state.mode = riscv_mode::S;
		state.mtvec = mtvec_addr;
		state.mstatus = (1 << 11) | (1 << 7); // MPP 为 S, MPIE 为 1
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
		// ECALL 之前已执行 addi x5,0,1
		assert_eq!(state.gprs[5], 1);
		// ECALL 进入 M 模式处理函数, MRET 返回 S 模式后执行 WFI
		assert_eq!(
			instr_group.exit_reason,
			exit_reason::WFI_WAIT,
			"inline ECALL+M-mode handler+MRET+WFI->WFI_WAIT exit"
		);
	}

	#[test]
	fn parallel_two_harts_independent() {
		let mut ram = vec![0u8; 4096];
		// hart 0 在 PC 0、4、8 处依次执行 addi x5, x0, 10、addi x5, x5, 3 与 WFI,
		write_u32_le(&mut ram, 0, 0x00A0_0293);
		write_u32_le(&mut ram, 4, 0x0032_8293);
		write_u32_le(&mut ram, 8, 0x1050_0073); // WFI
										  // hart 1 在 PC 128、132、136 处依次执行 addi x6, x0, 20、addi x6, x6, 7
		write_u32_le(&mut ram, 128, 0x0140_0313);
		write_u32_le(&mut ram, 132, 0x0073_0313);
		write_u32_le(&mut ram, 136, 0x1050_0073); // WFI

		let mut s0 = make_state(0, 0);
		s0.mode = riscv_mode::M; // WFI 需在 M 模式执行, 避免 TW 触发陷阱
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
		// 两个 hart 在进入 WFI 之前都已执行完各自的加法.
		// 全部 hart 处于 WFI, 故以 WFI_WAIT 退出.
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
		// 验证并发模式下单个 hart 的 AMOADD 可以正确执行.
		let mut ram = vec![0u8; 4096];
		let shared_addr: u64 = 0xF0;
		write_u32_le(&mut ram, 0, 0x0063_A2AF); // amoadd.w x5, x6, (x7)
		write_u32_le(&mut ram, 4, 0x1050_0073); // WFI

		ram[shared_addr as usize] = 42;
		let mut state = make_state(0, 0);
		state.mode = riscv_mode::M;
		state.mie = 0;
		state.mip.store(0, Ordering::Release);
		state.gprs[7] = shared_addr; // rs1, 地址
		state.gprs[6] = 3; // rs2, 加数
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
		let shared_addr: u64 = 0xF0; // 已对齐, 位于 RAM 范围内

		// hart 0: 对 shared_addr 执行 AMOADD.W
		//   amoadd.w x5, x6, (x7): rs1 为 x7 即地址, rs2 为 x6 即加数, rd 为 x5 即旧值
		write_u32_le(&mut ram, 0, 0x0063_A2AF); // amoadd.w x5, x6, (x7)
		write_u32_le(&mut ram, 4, 0x1050_0073); // WFI

		// hart 1: 对同一 shared_addr 执行 AMOADD.W, 随后执行 WFI
		write_u32_le(&mut ram, 128, 0x0063_A2AF); // amoadd.w x5, x6, (x7)
		write_u32_le(&mut ram, 132, 0x1050_0073); // WFI

		// 在 shared_addr 处写入初值 42, 按小端序组织存放
		ram[shared_addr as usize] = 42;
		ram[shared_addr as usize + 1] = 0;
		ram[shared_addr as usize + 2] = 0;
		ram[shared_addr as usize + 3] = 0;

		let mut s0 = make_state(0, 0);
		s0.mode = riscv_mode::M; // WFI 需在 M 模式执行
		s0.mie = 0;
		s0.mip.store(0, Ordering::Release);
		s0.gprs[7] = shared_addr; // rs1, 地址
		s0.gprs[6] = 3; // rs2, 加数

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

		// 最终值应为 42 + 3 + 5 = 50, 两次 AMOADD 都以原子方式生效
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
		let dev_addr: u64 = 0x1000_0000; // UART 的 MMIO 地址

		// hart 0: 把 x5 存储到 x6 指向的地址, 其中 x6 为 dev_addr, 设备存储触发退出.
		// sd x5, 0(x6) 编码为 0x00533023, 即 rs2 为 x5, rs1 为 x6, imm 为 0.
		write_u32_le(&mut ram, 0, 0x0053_3023);

		// hart 1: 在偏移 128 处执行两条指令的无限循环, 每次迭代推进 PC,
		// 即先 addi 再 jal 跳回, 故永远不会触发停止条件, 没有 MMIO、WFI 或
		// 陷阱循环. 有限的 NOP 序列会执行到 0x0000 处的非法压缩指令,
		// 产生与 hart 0 的 MMIO 退出竞争的 ``TRAP`` 原因退出, 使本用例在负载下不稳定.
		write_u32_le(&mut ram, 128, 0x0010_8093); // addi x1, x1, 1
		write_u32_le(&mut ram, 132, 0xffdf_f06f); // jal x0, -4, 跳回 128

		let mut s0 = make_state(0, 0);
		s0.mode = riscv_mode::M; // Bare 模式: 虚拟地址即物理地址
		s0.gprs[5] = 0x41; // 要存储的值
		s0.gprs[6] = dev_addr; // rs1, 存储地址
		let mut s1 = make_state(128, 1);
		s1.mode = riscv_mode::M;
		let mut states = [s0, s1];

		// 自定义设备 MMIO, 覆盖 [dev_addr, dev_addr + 8)
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
		// 单条 TOR 条目覆盖全部地址空间, 权限为读、写与执行.
		// 按 RISC-V 规范 §3.7.1, num 为 0 时 pmp_ok 拒绝 S 模式与 U 模式的访问,
		// 故测试至少需要一条宽松条目.
		let mut pmp_cfg: [u8; 1] = [0x0F]; // PMP_R, PMP_W, PMP_X 与 PMP_A_TOR
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
			timebase_hz: 0, // 单元测试中不推进时钟源, 保持确定性
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
				flush_cnt: std::ptr::null_mut(),
				flush_cnt_per_hart: std::ptr::null_mut(),
			};
			run_parallel(&hart, &mem_ctx, &intr_ctx, &dev_ctx, &bp, &mut tlb);
		}

		// 对设备 MMIO 的存储应触发 MMIO 原因的退出.
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
		// 两个 hart 都执行 WFI
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

		// 两个 hart 都应处于 WFI, 加速执行应以 WFI_WAIT 退出
		assert_eq!(
			instr_group.exit_reason,
			exit_reason::WFI_WAIT,
			"All WFI should cause WFI_WAIT exit, got {}",
			instr_group.exit_reason
		);
		assert_eq!(states[0].waiting, 1);
		assert_eq!(states[1].waiting, 1);
	}

	/// 辅助函数: 以自定义的 CLINT 基址调用 ``run_parallel``.
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
		// 单条 TOR 条目覆盖全部地址空间, 权限为读、写与执行.
		// 按 RISC-V 规范 §3.7.1, num 为 0 时 pmp_ok 拒绝 S 模式与 U 模式的访问,
		// 故测试至少需要一条宽松条目.
		let mut pmp_cfg: [u8; 1] = [0x0F]; // PMP_R, PMP_W, PMP_X 与 PMP_A_TOR
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
			timebase_hz: 0, // 单元测试中不推进时钟源, 保持确定性
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
			flush_cnt: std::ptr::null_mut(),
			flush_cnt_per_hart: std::ptr::null_mut(),
		};
		run_parallel(&hart, &mem_ctx, &intr_ctx, &dev_ctx, &bp, &mut tlb);
	}

	/// 验证对 CLINT MSIP 的存储正常完成: MSIP 挂起后触发陷阱, MRET 之后继续执行.
	#[test]
	fn store_to_clint_msip_single_hart() {
		let mut ram = vec![0u8; 1024];
		let clint_base: u64 = 0x2000000;

		// M 模式下单个 hart 向 hart 0 自身的 CLINT MSIP 写入, 随后进入 WFI.
		// MSIE 使能且 MIE 为 1 时, 挂起的 MSIP 触发陷阱进入 mtvec 指向的 0x200,
		// 该处执行 MRET, 从 WFI 的下一条指令继续, 即第二个 WFI, 随后全部空闲退出.
		// 0x00: addi t0, x0, 0x42, 使 t0 为 0x42
		// 0x04: sw t0, 0x1000(x0), 向 RAM 存储
		// 0x08: lui t0, 0x2000
		// 0x0C: addi t0, t0, 0, 使 t0 保持为 0x2000000, 即 hart 0 的 CLINT MSIP
		// 0x10: addi t1, x0, 1, 使 t1 为 1
		// 0x14: sw t1, 0(t0), 为 hart 0 写入 MSIP=1
		// 0x18: addi t2, x0, 0x42, 使 t2 为 0x42, 证明 CLINT 写入成功
		// 0x1C: wfi, 进入 WFI, MSIP 挂起后触发陷阱
		// 0x20: wfi, MSI 处理函数 MRET 之后到达; MSIP 已清除, 全部空闲退出
		write_u32_le(&mut ram, 0x00, 0x04200293); // addi t0, x0, 0x42
		write_u32_le(&mut ram, 0x04, 0x00502023); // sw t0, 0(x0), 向物理地址 0 的 RAM 存储
		write_u32_le(&mut ram, 0x08, 0x020002B7); // lui t0, 0x2000, 使 t0 为 0x2000000
		write_u32_le(&mut ram, 0x0C, 0x00028293); // addi t0, t0, 0
		write_u32_le(&mut ram, 0x10, 0x00100313); // addi t1, x0, 1
		write_u32_le(&mut ram, 0x14, 0x0062A023); // sw t1, 0(t0)
		write_u32_le(&mut ram, 0x18, 0x04200393); // addi t2, x0, 0x42
		write_u32_le(&mut ram, 0x1C, 0x10500073); // wfi
		write_u32_le(&mut ram, 0x20, 0x10500073); // wfi

		let mut state = HartState {
			pc: 0,
			mode: riscv_mode::M,
			mstatus: (3 << 11) | (1 << 7) | (1 << 3), // MPP 为 M, MPIE 为 1, MIE 为 1
			..unsafe { std::mem::zeroed() }
		};
		state.mie = 1 << 3; // MSIE, 允许接收 MSI
		state.mip.store(0, Ordering::Release);
		// 0x200 处的 M 模式处理函数先清除 hart 0 的 CLINT MSIP, 避免电平触发的
		// 中断再次触发, 然后执行 MRET.
		// 0x200: lui t1, 0x2000, 使 t1 为 0x2000000, 即 CLINT 基址
		// 0x204: sw x0, 0(t1), 即 CLINT 中 hart 0 的 msip 置 0
		// 0x208: mret
		state.mtvec = 0x200;
		write_u32_le(&mut ram, 0x200, 0x02000337); // lui t1, 0x2000
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

		// 0x04 处的 RAM 存储在物理地址 0 处写入 0x42
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

	/// 跨 hart 的 MSI 投递用例.
	///
	/// hart 0 经 CLINT MMIO 向 hart 1 发送 MSI, 随后进入 WFI.
	/// hart 1 从 WFI 开始, 应被 MSI 唤醒, 陷入 M 模式处理函数, 该函数执行 MRET,
	/// 随后执行 WFI 之后的标记指令. hart 1 的 t0 必须为 0x42, 以证明 MSI 已投递,
	/// 处理函数已执行, 且执行从 WFI 之后继续. 两个 hart 随后都空闲, 以 WFI_WAIT
	/// 退出, 没有指令配额.
	#[test]
	fn cross_hart_msip_wakes_target() {
		let mut ram = vec![0u8; 4096];
		let clint_base: u64 = 0x2000000;

		// --- hart 0 的指令, PC=0 ---
		// 延迟循环: 自旋约 512 次, 使 hart 1 先进入 WFI.
		// 0x00: addi t0, x0, 512, 使 t0 为 512
		// 0x04: addi t0, t0, -1, t0 减 1
		// 0x08: bnez t0, -8, 跳回 0x04
		// 0x0C: lui t0, 0x2000, 使 t0 为 0x2000000
		// 0x10: addi t0, t0, 4, 使 t0 为 0x2000004, 即 hart 1 的 CLINT MSIP
		// 0x14: addi t1, x0, 1, 使 t1 为 1
		// 0x18: sw t1, 0(t0), 为 hart 1 写入 MSIP=1
		// 0x1C: addi x0, x0, 0, 即 NOP
		// 0x20: wfi, hart 0 进入 WFI. 两个 hart 都空闲后, ``wfi_check_all_idle``
		//       使本轮加速执行以 WFI_WAIT 结束.
		write_u32_le(&mut ram, 0x00, 0x20000293); // addi t0, x0, 512
		write_u32_le(&mut ram, 0x04, 0xFFF28293); // addi t0, t0, -1
		write_u32_le(&mut ram, 0x08, 0xFE029EE3); // bnez t0, -8, 跳回 0x04
		write_u32_le(&mut ram, 0x0C, 0x020002B7); // lui t0, 0x2000, 使 t0 为 0x2000000
		write_u32_le(&mut ram, 0x10, 0x00428293); // addi t0, t0, 4
		write_u32_le(&mut ram, 0x14, 0x00100313); // addi t1, x0, 1
		write_u32_le(&mut ram, 0x18, 0x0062A023); // sw t1, 0(t0)
		write_u32_le(&mut ram, 0x1C, 0x00000013); // addi x0, x0, 0, 即 NOP
		write_u32_le(&mut ram, 0x20, 0x10500073); // wfi

		// --- hart 1 的指令, PC=0x100 ---
		// 0x100: wfi, 进入 WFI
		// 0x104: addi t0, x0, 0x42, 使 t0 为 0x42, 标记 MSI 已接收
		// 0x108: wfi, 再次进入 WFI
		// 0x10C: jal x0, -12, 跳回 0x104. 该循环用作安全保障: 若 PC 越过 WFI
		//        继续推进, 0x200 处的处理函数只执行 MRET, 故 mepc 必须指向有效代码
		write_u32_le(&mut ram, 0x100, 0x10500073); // wfi
		write_u32_le(&mut ram, 0x104, 0x04200293); // addi t0, x0, 0x42
		write_u32_le(&mut ram, 0x108, 0x10500073); // wfi
		write_u32_le(&mut ram, 0x10C, 0xFF5FF06F); // jal x0, -12, 跳回 0x104

		// --- 共用的 mtvec 处理函数, PC=0x200 ---
		// 0x200: mret, hart 0 的处理函数, 本用例中不触发陷阱
		write_u32_le(&mut ram, 0x200, 0x30200073); // mret

		// --- hart 1 的 MSI 处理函数, PC=0x300 ---
		// 在 MRET 之前清除 CLINT MSIP, 避免同一电平触发中断再次触发.
		// 实际固件在核间中断处理函数中完成该清除, 即 sbi_ipi_raw_clear.
		// 若不清除, sync_msip 在 MRET 之后因 CLINT 电平仍为高而重新置位
		// mip.MSIP, 形成无限陷阱循环, addi 永不执行.
		//   0x300: lui t1, 0x2000, 使 t1 为 0x2000000, 即 CLINT 基址
		//   0x304: sw x0, 4(t1), 即 CLINT 中 hart 1 的 msip 置 0
		//   0x308: mret
		write_u32_le(&mut ram, 0x300, 0x02000337); // lui t1, 0x2000
		write_u32_le(&mut ram, 0x304, 0x00032223); // sw x0, 4(t1)
		write_u32_le(&mut ram, 0x308, 0x30200073); // mret

		// --- hart 0 的状态 ---
		let mut s0 = make_state(0, 0);
		s0.mode = riscv_mode::M;
		s0.mie = 0;
		s0.mip.store(0, Ordering::Release);
		s0.mtvec = 0x200; // 0x200 处有效的 MRET 处理函数, 与 hart 1 共用

		// --- hart 1 的状态 ---
		let mut s1 = make_state(0x100, 1);
		s1.mode = riscv_mode::M;
		s1.mie = 0; // 初始不使能任何中断
		s1.mip.store(0, Ordering::Release); // 初始没有挂起中断
		s1.mtvec = 0x300; // 0x300 处的 M 模式处理函数, 先清除 MSIP 再 MRET
		s1.mstatus = (3 << 11) | (1 << 7) | (1 << 3); // MPP 为 M, MPIE 为 1, MIE 为 1

		let mut states = [s0, s1];
		let mut instr_group: InstrToBeExec = unsafe { mem::zeroed() };

		unsafe {
			run_parallel_with_clint(
				states.as_mut_ptr(),
				2,
				ram.as_mut_ptr(),
				ram.len() as u64,
				0, // RAM 基址
				clint_base,
				&mut instr_group as *mut InstrToBeExec,
			);
		}

		// mcause 必须表明 MSI 已投递: 第 63 位为 1 表示中断, 原因编码 3 表示 MSI.
		let is_interrupt = (states[1].mcause >> 63) != 0;
		let cause_code = states[1].mcause & 0x7FFF_FFFF_FFFF_FFFF;
		assert!(
			is_interrupt && cause_code == 3,
			"Hart 1 mcause should show MSI interrupt; got mcause={:#x}",
			states[1].mcause
		);

		// hart 1 不得停机, 即没有陷入陷阱循环.
		assert_eq!(states[1].halted, 0, "Hart 1 should not be halted");

		// 标记值的含义: MSI 在 WFI 之后到达时, PC 先越过 WFI 再陷入陷阱,
		// mepc 指向 addi, MRET 之后 t0 为 0x42. MSI 在 WFI 之前到达时,
		// mepc 指向 WFI 本身, MRET 之后重新执行 WFI, t0 保持 0.
		// 两种情形下 MSI 都已投递, 此处的断言识别实际发生的情形.
		if states[1].gprs[5] != 0x42 && states[1].mepc != 0x100 {
			panic!(
				"If marker not set, mepc should be 0x100 (MSI before WFI); \
                 got t0={:#x} mepc={:#x}",
				states[1].gprs[5], states[1].mepc
			);
		}
	}

	/// BLTZ 在 rs1 为负数时的用例, 即 BLT rs1, x0, 偏移量.
	///
	/// 验证 ``addi s2, a5, -1`` 与 ``bltz s2, target`` 在负数下的行为:
	/// a5 为 -1 时 s2 为 -2, bltz 必须跳转.
	#[test]
	fn regression_bltz_negative_takes_branch() {
		let mut ram = vec![0u8; 128];

		// PC=0 处的指令:
		// 0x00: addi s2, a5, -1, 使 s2 为 -2, 其中 a5 为 -1
		// 0x04: bltz s2, +0x10, s2 小于 0 时跳转到 0x18
		// 0x08: addi s1, x0, 0xBAD, 使 s1 为 0xBAD, 仅在 bltz 未跳转时执行
		// 0x0C: ebreak, 陷入 mtvec
		// ...
		// 0x18: addi s2, x0, 42, 使 s2 为 42, 即跳转目标
		// 0x1C: ebreak, 陷入 mtvec
		// 0x40: wfi, mtvec 指向此处，在后续处理中自动退出

		write_u32_le(&mut ram, 0, 0xFFF78913); // addi s2, a5, -1
		write_u32_le(&mut ram, 4, 0x00094A63); // bltz s2, +0x14, 跳转到 0x18
		write_u32_le(&mut ram, 8, 0xBAD00493); // addi s1, x0, 0xBAD, 应当被跳过
		write_u32_le(&mut ram, 12, 0x00100073); // ebreak
		write_u32_le(&mut ram, 24, 0x02A00913); // addi s2, x0, 42, 即跳转目标
		write_u32_le(&mut ram, 28, 0x00100073); // ebreak
		write_u32_le(&mut ram, 0x40, 0x10500073); // wfi

		let mut state = HartState {
			pc: 0,
			gprs: [0u64; 32],
			mode: riscv_mode::M,
			mstatus: 0,
			..unsafe { std::mem::zeroed() }
		};
		state.mtvec = 0x40; // ebreak 进入 WFI, 随后全部空闲退出
		state.gprs[15] = 0xFFFF_FFFF_FFFF_FFFFu64; // a5 为 -1

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
			timebase_hz: 0, // 单元测试中不推进时钟源, 保持确定性
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
				flush_cnt: std::ptr::null_mut(),
				flush_cnt_per_hart: std::ptr::null_mut(),
			};
			run_parallel(&hart, &mem_ctx, &intr_ctx, &dev_ctx, &bp, &mut tlb);
		}

		// BLTZ 应当已跳转, 故 s2 为 42
		assert_eq!(
			state.gprs[18], 42,
			"BLTZ s2,-2 MUST branch: expected s2=42, got {}",
			state.gprs[18]
		);
		// s1 不应为 0xBAD, 以证明没有顺序执行到 0x08
		assert_ne!(
			state.gprs[9], 0xBAD,
			"BLTZ should have branched; s1=0xBAD means fallthrough occurred"
		);
	}

	#[test]
	fn pmp_concurrent_csrw_two_harts() {
		let mut ram = vec![0u8; 4096];
		// csrw pmpaddr1, x5 编码为 (0x3B1 << 20) | (5 << 15) | (1 << 12) | 0x73,
		// 即 0x3B129073; 随后执行 WFI.
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

		// 每个 hart 使用扁平缓冲区的 64 项, 前 16 项配置为宽松条目
		let mut pmp_cfg: [u8; 128] = [0u8; 128];
		let mut pmp_addr: [u64; 128] = [u64::MAX; 128];
		// 前 16 项设为 NAPOT 地址模式
		for h in 0..2 {
			for e in 0..16 {
				pmp_cfg[h * 64 + e] = 0x0F; // 读、写、执行与 NAPOT
			}
		}
		let pmp = FfiPmpCtx {
			cfg: pmp_cfg.as_mut_ptr(),
			addr: pmp_addr.as_mut_ptr(),
			num: 64, // 每个 hart 的条目数; 扁平缓冲区共 128 项, 供 2 个 hart 使用
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
				flush_cnt: std::ptr::null_mut(),
				flush_cnt_per_hart: std::ptr::null_mut(),
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
		// 检查 PMP 写入结果
		assert_eq!(pmp_addr[1], 0xA000, "hart0 pmpaddr1");
		assert_eq!(pmp_addr[65], 0xB000, "hart1 pmpaddr1 in flat buf");
		// 预期以 WFI 退出
		assert_eq!(instr_group.exit_reason, exit_reason::WFI_WAIT);
	}
}
