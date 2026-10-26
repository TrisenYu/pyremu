//! FFI 边界上下文结构体 — 与 Python ctypes 逐一对应的 `#[repr(C)]` 定义.
//!
//! 集中存放所有跨 FFI 边界的参数结构体: 内存、PMP、CLINT、PLIC、UART、virtio、
//! 看门狗、断点与 TLB, 以及设备快速路径接口 ``FfiDevCtx``. 处理器自身的状态
//! ``HartState``、``TlbEntry`` 与 ``ImsicFile`` 位于 [state](crate::state).
//!
//! 布局约束: 每个结构体在 Python 侧都有 ctypes 镜像, 两侧必须逐字段一致,
//! 否则 Rust 在错误的偏移处读取字段而引发段错误.

use crate::peripheral::uart::uart_tx_push;
use crate::state::HartState;

#[repr(C)]
pub struct InstrToBeExec {
	pub total_instrs: u64,
	pub exit_reason: u8,
	pub exit_hart_id: u8,
	pub exit_pc: u64,
	pub exit_instr: u32,
	pub trap_cause: u32,
	pub trap_tval: u64,
	pub trap_is_interrupt: u8,
	pub trap_delegated: u8,
	pub _pad: [u8; 6],
}

// ============================================================
//  FFI 上下文结构体
// ============================================================

/// 主存范围与 PIE 链接地址区间构成的内存上下文.
#[repr(C)]
pub struct MemCtx {
	pub ram: *mut u8,
	pub ram_size: u64,
	pub ram_base: u64,
	pub shadow_base: u64,
	pub shadow_size: u64,
}

/// 跨 FFI 边界的 PMP 配置.
/// Rust 借此就地写回 PMP 的 CSR 取值.
#[repr(C)]
pub struct FfiPmpCtx {
	pub cfg: *mut u8,
	pub addr: *mut u64,
	pub num: u8,
	pub pmpsplit: u8,
}

/// 跨 FFI 边界的 CLINT 状态.
/// Rust 借此就地更新 MSIP、MTIMECMP 与 MTIME.
#[repr(C)]
pub struct FfiClintCtx {
	pub mtime: *mut u64,
	pub mtimecmp: *mut u64,
	pub msip: *mut u8,
	pub base: u64,
	/// mtime 的计数频率 (Hz); 取 0 表示不按指令推进 mtime.
	/// 加速执行内按本 hart 本轮已执行的指令数 instr_delta 换算 mtime:
	/// target = time_base_val + instr_delta * timebase_hz * NS_PER_INSTR / NS_PER_SEC
	pub timebase_hz: u64,
}

/// 跨 FFI 边界的 PLIC 状态. Rust 就地更新优先级、pending、使能、
/// 阈值与 claimed 等状态, 并执行 claim 与 complete 语义, 避免受调试程序直接映射访问
/// PLIC 时退出加速执行. 电平由 Python 在设备中断线变化时写入.
///
/// Python 持有底层 ctypes 数组, 并在每轮加速执行边界同步两个方向的状态;
/// 设备中断的投递目标经 ``set_irq`` 直接写入该数组, 本轮加速执行内新到达的
/// 设备中断因而对 Rust 可见.
#[repr(C)]
pub struct FfiPlicCtx {
	/// PLIC 的 MMIO 基址; 取 0 表示不存在 PLIC 设备.
	pub base: u64,
	/// 中断源个数; 中断源 0 保留, 有效中断源为 1..=num_sources.
	pub num_sources: u32,
	/// 上下文个数; 每个 hart 通常有 2 个, 分别为 M 模式与 S 模式.
	pub num_contexts: u32,
	/// 每个中断源的优先级. 索引范围 0..=num_sources.
	pub priority: *mut u8,
	/// 每个中断源的 pending 标志, 与 Python 设备侧的 set_irq 共享.
	pub pending: *mut u8,
	/// 每个中断源的电平标志. complete 时电平仍为高则重新置位 pending.
	pub level: *mut u8,
	/// 每个上下文的中断使能位图: enable[ctx * num_words + word_idx], 每个元素为一个 u32 字,
	/// 位 i 对应中断源 word_idx * 32 + i; 中断源 0 保留.
	/// num_words = (num_sources + 31) / 32
	pub enable: *mut u32,
	/// 每个上下文的阈值.
	pub threshold: *mut u8,
	/// 每个上下文已 claim 的中断源; 取 0 表示没有.
	pub claimed: *mut u32,
}

/// 跨 FFI 边界的宿主侧执行看门狗.
///
/// 同样定义于 FfiDevicesCtx, 但不含任何设备寄存器,
/// 只携带本轮加速执行的超时时间 ``timeout_ns``, 取 0 表示不启用.
/// 看门狗线程 ``watchdog_loop`` 以 ``module.st_time_val`` 为起点, 真实流逝时间达到
/// ``timeout_ns`` 后经 ``request_stop`` 与 ``unpark_all_harts`` 以
/// ``exit_reason::TIMEOUT`` 终止本轮加速执行.
#[repr(C)]
pub struct FfiWatchdogCtx {
	pub timeout_ns: u64,
}

/// 跨 FFI 边界的 UART 上下文.
#[repr(C)]
pub struct FfiUartCtx {
	/// UART 的 MMIO 基址, 如 0x10000000.
	pub base: u64,
	/// 发送环形缓冲区: 每条目为 ``[hid, byte]`` 两个字节, 故条目容量为 ``tx_cap / 2``
	pub tx_buf: *mut u8,
	/// ``tx_buf`` 的字节容量.
	pub tx_cap: u32,
	/// ``tx_buf`` 的写索引, 以条目为单位; 每写入一条目加 1, 以 2^32 为取模对象,
	/// 槽位下标为写索引对条目容量取模.
	pub tx_wr: *mut u32,
	/// UART IE 寄存器(偏移 0x10)的副本
	pub ie: u32,
	/// UART TXCTRL 寄存器 (偏移 0x08) 的副本, 供寄存器读取与 IP 的 TX 位计算.
	pub txctrl: u32,
	/// UART RXCTRL 寄存器 (偏移 0x0C) 的副本, 供 IP 的 RX 位计算.
	pub rxctrl: u32,
	/// RX FIFO 的近似占用量, 由 Python 在取值变化时写入.
	pub rx_fifo_len: u32,
	/// 兼容性字段, 取 -1 表示未设置
	pub tx_notify_fd: i32,
	/// 取非 0 时 Rust 只写发送环形缓冲区, 不调用 libc::write 写标准输出;
	/// 标准输出由 Python 的 _tx_callback 负责.
	pub no_stdout: u8,
	/// 接收通知标志. TermIO 守护线程把标准输入的新字节写入接收环形缓冲区后置位.
	/// ``hart_worker`` 每轮读取该标志: IMSIC 独占外部中断线路时置位即结束本轮,
	/// 退出原因为 ``exit_reason::RX_WAIT``; 进入 WFI 等待前的检查也以该标志作为唤醒条件.
	pub rx_notify: *mut u8,
}

// 安全性: Python 在 FFI 调用期间保持底层 ctypes 数组不被 gc 回收.
unsafe impl Send for FfiUartCtx {}
unsafe impl Sync for FfiUartCtx {}

// ============================================================
// ============================================================

/// 跨 FFI 边界的各设备 MMIO 地址范围.
///
/// 由 ``FfiDevicesCtx`` 携带.
#[repr(C)]
pub struct DevMMIOAddrInfo {
	pub bases: *const u64,
	pub ends: *const u64,
	pub num: u8,
}

/// 处理受调试程序发起的 ecall.
///
/// ``priv_ecall_concurrent`` 经 ``&dyn FfiDevCtx`` 调用 ``try_ecall``,
/// 调用方无需知道具体设备类型.
/// ```ignore
/// impl FfiDevCtx for FfiUartCtx { ... }
/// ```
///
/// 新增设备只需实现 trait, 无需改动 ecall 分发代码.
pub trait FfiDevCtx: Send + Sync {
	/// 尝试加速处理 ecall. 返回 ``Some(advance)`` 表示已处理,
	/// advance 为指令前进的字节数, 通常为 4; 返回 ``None`` 表示未处理,
	/// 调用方转入通用 trap 投递.
	fn try_ecall(&self, state: &mut HartState, ext_id: u64, func_id: u64) -> Option<u64>;

	/// 设备名称.
	fn name(&self) -> &'static str;
}

/// ``FfiUartCtx`` 的 ``FfiDevCtx`` 实现: ``sbi_putchar`` 直接写入发送环形缓冲区.
///
/// 受调试程序经 SBI 控制台输出字符时不写 UART TXDATA, 因而不退出加速执行;
/// 发送环形缓冲区不可写入时返回 ``None``, 由调用方转入通用 trap 投递.
impl FfiDevCtx for FfiUartCtx {
	fn try_ecall(&self, state: &mut HartState, ext_id: u64, func_id: u64) -> Option<u64> {
		// SBI 传统 putchar: ext_id 为 0x01, func_id 为 0, a0 为字符.
		if ext_id != 0x01 || func_id != 0 {
			return None;
		}
		let c = state.gprs[10] as u8;
		if uart_tx_push(self, state.mhartid as u8, c) {
			state.gprs[10] = 0; // 按 SBI 约定返回成功
			Some(4) // 前进越过 ecall 指令
		} else {
			None
		}
	}

	fn name(&self) -> &'static str {
		"uart"
	}
}

/// 跨 FFI 边界的 virtio-blk MMIO 就地处理上下文.
///
/// 除下列两类写操作外, Rust 在加速执行内就地处理全部 MMIO 寄存器读写,
/// 设备探测过程无需退出至 Python. 驱动写 ``QueueNotify`` (偏移 0x050) 时返回
/// ``None`` 强制退出, 由 Python 处理虚拟队列描述符; 清空全部中断位的
/// ``InterruptACK`` (偏移 0x064) 同样强制退出, 由 Python 拉低 PLIC 中断线.
///
/// 字段与 Python ``VirtIOBlock`` 的 MMIO 状态逐项对应; Python 在每轮加速执行前
/// 写入这些字段, 结束后读回被 Rust 改动的字段.
#[repr(C)]
pub struct FfiVirtIoCtx {
	/// virtio 的 MMIO 基址; 取 0 表示不存在 virtio 设备.
	pub base: u64,
	/// 磁盘容量, 以 512 字节扇区为单位, 供偏移 0x100 起的配置空间读取.
	pub capacity: u64,
	/// 队列最大条目数, 只读.
	pub queue_num_max: u32,

	// ---- 特性协商 ----
	/// 设备特性页选择寄存器, 写偏移 0x014.
	pub device_features_sel: u32,
	/// 驱动特性页选择寄存器, 写偏移 0x024.
	pub driver_features_sel: u32,
	/// 受调试程序接受的驱动特性, 写偏移 0x020, 共 64 位, 按页选择分两次写入.
	pub driver_features: u64,

	// ---- 队列设置 ----
	/// 当前选中的队列索引, 写偏移 0x030.
	pub queue_sel: u32,
	/// 选定的队列长度, 写偏移 0x038; 取值不超过 ``queue_num_max``.
	pub queue_num: u32,
	/// 队列就绪标志, 写偏移 0x044.
	pub queue_ready: u8,

	// 队列地址
	pub queue_desc: u64,
	pub queue_driver: u64,
	pub queue_device: u64,

	// ---- 设备状态与中断 ----
	/// 设备状态寄存器, 偏移 0x070; 写入 0 复位设备.
	pub status: u32,
	/// 中断状态, 偏移 0x060; 位 0 表示已用缓冲区通知.
	pub interrupt_status: u32,

	// QueueNotify 待处理标志
	/// 驱动写 ``QueueNotify`` (偏移 0x050) 时 Rust 置 1, 并强制退出加速执行.
	/// Python 在进入虚拟队列处理前将该字段清零.
	pub notify_pending: u8,
	/// 取 1 时 Python 在加速执行结束后调用 ``lower_irq_if_idle`` 拉低 PLIC 中断线.
	/// 当前没有代码置位该字段.
	pub irq_maybe_lower: u8,
	pub _pad: [u8; 6],
}

// 安全性: Python 在 FFI 调用期间保持底层 ctypes 数组不被 gc 回收.
unsafe impl Send for FfiVirtIoCtx {}
unsafe impl Sync for FfiVirtIoCtx {}

// ============================================================
//  FFI 结构体 — run_parallel 的分组参数
// ============================================================

#[repr(C)]
/// Python 持有的共享外部中断上下文, 由 Rust 在 hart 循环内读取.
///
/// PLIC 与设备状态位于 Python 侧. 设备置位或拉低中断线时 Python 更新本结构体.
/// 链接库在 hart 循环内读取 ``pending``: 置位时对未被 IMSIC 文件占用的线路就地置位
/// ``mip`` 的 MEIP 与 SEIP, 对已被占用的线路改为由该文件的 eip 派生对应的 ``mip`` 位.
/// 该标志同时是 WFI 的唤醒条件; 每轮加速执行结束时, 接收环形缓冲区与
/// UART FIFO 均为空则 Python 将其清零.
#[repr(C)]
pub struct FfiExtIrqCtx {
	/// 非 0 表示至少一个外部中断源已置位, PLIC 状态可能已改变. Rust 在 hart 循环内读取该字段.
	pub pending: u8,
	/// 待处理中断源的位图, 位 i 对应 PLIC 中断源 i, 例如 1 为 virtio 块设备、10 为 UART.
	/// 当前没有代码读取或写入该字段.
	pub sources: u32,
	/// 当前待处理中断源中的最高优先级, 无待处理源时为 0.
	/// 当前没有代码读取或写入该字段.
	pub max_priority: u8,
	pub _pad: [u8; 2],
}

#[repr(C)]
pub struct FfiHartCtx {
	pub states: *mut HartState,
	pub num_harts: u32,
	pub instr_group: *mut InstrToBeExec,
	pub stop_flag: *const u8,
	pub ext_irq: *mut FfiExtIrqCtx,
}

/// 跨 FFI 边界的内存与保护上下文，暂存有主存范围、PIE 链接地址区间与 PMP.
#[repr(C)]
pub struct FfiMemCtx {
	pub mem: *const MemCtx,
	pub pmp: *const FfiPmpCtx,
}

/// 跨 FFI 边界的中断控制器上下文，记录 CLINT 提供定时器与 IPI, PLIC 提供外部中断.
#[repr(C)]
pub struct FfiInterruptCtx {
	pub clint: *const FfiClintCtx,
	pub plic: *const FfiPlicCtx,
}

/// 跨 FFI 边界的设备上下文，记录 MMIO 地址范围, 以及 UART、virtio 与看门狗设备指针.
#[repr(C)]
pub struct FfiDevicesCtx {
	pub dev: *const DevMMIOAddrInfo,
	pub uart: *const FfiUartCtx,
	pub virtio: *const FfiVirtIoCtx,
	pub watchdog: *const FfiWatchdogCtx,
}

/// 跨 FFI 边界传递的断点配置.
#[repr(C)]
pub struct FfiBpCtx {
	pub addrs: *const u64,
	pub count: u32,
}

/// TLB 刷新计数跨 FFI 边界传递的两处存放位置. 两处均由调用方持有, 引擎每轮
/// 结束把取值写回, 故两个调用之间积累的刷新对下一轮仍然可见.
#[repr(C)]
pub struct FfiTlbCtx {
	pub flush_cnt: *mut u64,
	pub flush_cnt_per_hart: *mut u64,
}

/// 未配置 UART 时使用的空上下文.
pub(crate) static EMPTY_UART: FfiUartCtx = FfiUartCtx {
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
	rx_notify: std::ptr::null_mut(),
};
