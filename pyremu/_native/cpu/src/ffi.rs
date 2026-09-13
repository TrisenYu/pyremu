//! FFI 边界上下文结构体 — 与 Python ctypes 逐一对应的 `#[repr(C)]` 定义.
//!
//! 集中存放所有跨 FFI 边界的参数结构体: 内存/PMP/CLINT/PLIC/UART/virtio/
//! watchdog/断点/TLB, 以及设备快速路径接口 (``FfiDevCtx``). 处理器自身的状态
//! (``HartState`` / ``TlbEntry`` / ``ImsicFile`` / ``HartDiag``) 保留在
//! [state](crate::state).
//!
//! 布局约束: 每个结构体在 Python 侧都有 ctypes 镜像, 两侧必须逐字段一致,
//! 否则 Rust 会在错误偏移读取字段而引发段错误。

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
//  FFI context structs — group related parameters
// ============================================================

/// Memory context: RAM + shadow region.
#[repr(C)]
pub struct MemCtx {
	pub ram: *mut u8,
	pub ram_size: u64,
	pub ram_base: u64,
	pub shadow_base: u64,
	pub shadow_size: u64,
}

/// PMP configuration (crosses FFI boundary).  Pointers are mutable
/// so that Rust can write PMP CSR values back inline.
#[repr(C)]
pub struct FfiPmpCtx {
	pub cfg: *mut u8,
	pub addr: *mut u64,
	pub num: u8,
	pub pmpsplit: u8,
}

/// CLINT state (crosses FFI boundary).  Pointers inside are mutable
/// so that Rust can update MSIP / MTIMECMP / MTIME inline.
#[repr(C)]
pub struct FfiClintCtx {
	pub mtime: *mut u64,
	pub mtimecmp: *mut u64,
	pub msip: *mut u8,
	pub base: u64,
	/// mtime 时钟源频率 (Hz, 与 DTB timebase-frequency 同源).
	/// 用于acceleration内按 clock-source 推进 mtime (QEMU QEMU_CLOCK_VIRTUAL 等价):
	/// target = time_base_val + elapsed_ns * timebase_hz / 1e9.  0 = 禁用.
	pub timebase_hz: u64,
}

/// PLIC state (crosses FFI boundary).  Pointers are mutable so Rust can
/// update priority / pending / level / enable / threshold / claimed inline
/// (claim/complete 语义), 避免 kernel 直映射访问 PLIC 时的 batch 退出.
///
/// Python 持有底层 ctypes 数组, 并在每次加速执行边界 marshal/unmarshal;
/// 设备侧 ``raise_device_irq`` 也会直接写入本数组 (write-through), 保证 batch
/// 内新到达的设备中断对 Rust 可见.
#[repr(C)]
pub struct FfiPlicCtx {
	/// PLIC MMIO base address (0 = no PLIC device present).
	pub base: u64,
	/// Number of interrupt sources (source 0 reserved, sources 1..=num_sources).
	pub num_sources: u32,
	/// Number of contexts (2 per hart typically: M + S).
	pub num_contexts: u32,
	/// Per-source priority (u8, bits [2:0] valid).  Index 0..=num_sources.
	pub priority: *mut u8,
	/// Per-source pending flag (0/1).  Shared with Python device set_irq.
	pub pending: *mut u8,
	/// Per-source level flag (0/1) — gateway 语义: complete 时电平仍高则重挂 pending.
	pub level: *mut u8,
	/// Per-context enable 位图: enable[ctx * num_words + word], 每 u32 一个 word,
	/// bit i = source (word*32 + i); source 0 保留恒为 0 (与 Python plic.py 一致).
	/// num_words = (num_sources + 31) / 32.
	pub enable: *mut u32,
	/// Per-context threshold (u8, bits [2:0] valid).
	pub threshold: *mut u8,
	/// Per-context claimed source (0 = none).
	pub claimed: *mut u32,
}

/// Host-side execution watchdog (crosses FFI boundary).
///
/// 与 DevMMIOAddrInfo / FfiUartCtx / FfiVirtIoCtx 同属设备上下文组 (由 FfiDevicesCtx
/// 携带), 但不镜像任何 受调试程序 可见 MMIO: 仅携带一次加速执行的时钟源超时时间
/// ``timeout_ns`` (0 = 禁用).  看门狗线程 ``watchdog_loop`` 以 ``module.st_time_val``
/// 为基准, 越过该时间后经通用停止接口 (``request_stop`` + ``unpark_all_harts``)
/// 以 ``exit_reason::TIMEOUT`` 退出.
#[repr(C)]
pub struct FfiWatchdogCtx {
	pub timeout_ns: u64,
}

/// UART context (crosses FFI boundary).  Rust writes TX characters
/// directly into ``tx_buf`` so that ``sbi_printf`` does NOT cause a
/// acceleration exit.  Python reads the buffer after the acceleration completes.
#[repr(C)]
pub struct FfiUartCtx {
	/// UART MMIO base address (e.g. 0x10000000).
	pub base: u64,
	/// Ring buffer for TX characters (1-byte entries, ``tx_cap`` slots).
	pub tx_buf: *mut u8,
	/// Capacity of ``tx_buf`` (must be power of two, <= 256 MiB).
	pub tx_cap: u32,
	/// Write-index into ``tx_buf`` (monotonic, wraps at ``tx_cap``).
	/// Updated atomically by Rust; Python reads after acceleration.
	pub tx_wr: *mut u32,
	/// Shadow copy of UART IE register (offset 0x10), written by Python
	/// during marshal so Rust can handle IE/IP reads inline.
	pub ie: u32,
	/// Shadow copy of UART TXCTRL register (offset 0x08).
	pub txctrl: u32,
	/// Shadow copy of UART RXCTRL register (offset 0x0C).
	pub rxctrl: u32,
	/// Approximate RX FIFO fill level (Python sets during marshal).
	pub rx_fifo_len: u32,
	/// Pipe write-end for TX notification: Rust writes 1 byte per TXDATA
	/// write; Python TX thread select()s the read-end and drains to stdout.
	/// -1 = disabled.
	pub tx_notify_fd: i32,
	/// When 1, Rust writes to ring buffer only (no libc::write).
	/// Python _tx_callback handles all stdout output.
	pub no_stdout: u8,
	/// RX notification — set to 1 by TermIO daemon when new stdin data
	/// arrives in the ring buffer.  ``hart_worker`` polls this flag and
	/// wakes WFI / triggers SEIP inline.
	pub rx_notify: *mut u8,
}

// Safety: Python holds the backing ctypes arrays alive for the FFI call.
unsafe impl Send for FfiUartCtx {}
unsafe impl Sync for FfiUartCtx {}

// ============================================================
//  设备快速路径接口 — 避免 MMIO batch 退出的统一抽象
// ============================================================

/// MMIO 地址范围 — 设备基址/结束地址对 (crosses FFI boundary).
///
/// 从原 ``FfiDevCtx`` struct 中提取, 供 ``FfiDevicesCtx`` 直接携带.
#[repr(C)]
pub struct DevMMIOAddrInfo {
	pub bases: *const u64,
	pub ends: *const u64,
	pub num: u8,
}

/// 设备上下文接口 — 统一抽象各设备 (UART/SPI/I2C/块设备) 的快速路径.
///
/// 每个设备类型实现此 trait, ``priv_ecall_concurrent`` 通过
/// ``&dyn FfiDevCtx`` 调用, 无需知道具体设备类型:
///
/// ```ignore
/// // UART: sbi_putchar 直接写入 TX ring buffer
/// impl FfiDevCtx for FfiUartCtx { ... }
/// // SPI: 飞地 SPI 闪存快速读写 (未来)
/// impl FfiDevCtx for FfiSpiCtx { ... }
/// // I2C: 飞地传感器快速读取 (未来)
/// impl FfiDevCtx for FfiI2cCtx { ... }
/// ```
///
/// 新增设备只需实现 trait, 无需改 ecall 分发代码.
pub trait FfiDevCtx: Send + Sync {
	/// 尝试处理 ecall 快速路径. 返回 ``Some(advance)`` 表示已处理
	/// (advance = 指令前进字节数, 通常 4); ``None`` 表示不匹配,
	/// 回落通用 trap 投递.
	fn try_ecall(&self, state: &mut HartState, ext_id: u64, func_id: u64) -> Option<u64>;

	/// 向设备写入单字节 (供通用 putchar 路径使用).
	/// 返回 true 表示成功.
	fn write_byte(&self, _byte: u8) -> bool {
		false
	}

	/// 设备名称 (供诊断日志使用).
	fn name(&self) -> &'static str;
}

/// FfiUartCtx 的 FfiDevCtx 实现 — sbi_putchar 直接写入 TX ring buffer.
///
/// 避免走完整 M-mode trap 路径后写 UART TXDATA 触发 MMIO batch 退出.
/// 每个字符一次 batch 退出 (~1-5 ms marshal/unmarshal) 是飞地输出密集时
/// 数百倍减速的根因之一.
impl FfiDevCtx for FfiUartCtx {
	fn try_ecall(&self, state: &mut HartState, ext_id: u64, func_id: u64) -> Option<u64> {
		// SBI legacy putchar: ext=0x01, func=0, a0=character
		if ext_id != 0x01 || func_id != 0 {
			return None;
		}
		let c = state.gprs[10] as u8;
		if self.write_byte(c) {
			state.gprs[10] = 0; // SBI success
			Some(4) // advance past ecall instruction
		} else {
			None
		}
	}

	fn write_byte(&self, byte: u8) -> bool {
		if self.tx_buf.is_null() || self.tx_wr.is_null() || self.tx_cap == 0 {
			return false;
		}
		unsafe {
			let wr = self.tx_wr.read_volatile();
			let mask = self.tx_cap - 1;
			*self.tx_buf.add((wr & mask) as usize) = byte;
			self.tx_wr.write_volatile(wr.wrapping_add(1));
		}
		true
	}

	fn name(&self) -> &'static str {
		"uart"
	}
}

/// virtio-blk MMIO inline context (crosses FFI boundary).
///
/// Rust handles all MMIO register reads/writes inline inside the acceleration,
/// avoiding expensive exits to Python during the device probe sequence.
/// Only ``QueueNotify`` (offset 0x050) triggers an exit to Python
/// then processes the virtqueue descriptors (disk I/O).
///
/// Fields are laid out to match the Python ``VirtIOBlock`` MMIO state
/// 1:1; Python initialises them before the acceleration and reads back changed
/// fields after the acceleration.
#[repr(C)]
pub struct FfiVirtIoCtx {
	/// virtio MMIO base address (0 = no virtio device present).
	pub base: u64,
	/// Disk capacity in 512-byte sectors (for Config reads at offset 0x100).
	pub capacity: u64,
	/// Maximum queue entries (read-only, set by Python at init).
	pub queue_num_max: u32,

	// ---- Feature negotiation ----
	/// Device features page selector (written at offset 0x014).
	pub device_features_sel: u32,
	/// Driver features page selector (written at offset 0x024).
	pub driver_features_sel: u32,
	/// Driver features accepted by the 受调试程序 (written at offset 0x020, 64-bit).
	pub driver_features: u64,

	// ---- Queue setup ----
	/// Currently selected queue index (written at offset 0x030).
	pub queue_sel: u32,
	/// Chosen queue size (written at offset 0x038).
	pub queue_num: u32,
	/// Queue is ready / activated (written at offset 0x044).
	pub queue_ready: u8,

	// ---- Queue addresses (split 64-bit, written at offsets 0x080-0x0A4) ----
	pub queue_desc: u64,
	pub queue_driver: u64,
	pub queue_device: u64,

	// ---- Device status + interrupts ----
	/// Device status register (offset 0x070).  Writing 0 resets the device.
	pub status: u32,
	/// Interrupt status (offset 0x060).  Bit 0 = used buffer notification.
	pub interrupt_status: u32,

	// ---- QueueNotify pending flag ----
	/// Set to 1 by Rust when the driver writes to QueueNotify (offset 0x050).
	/// Python reads and clears this after processing the virtqueue.
	pub notify_pending: u8,
	/// Set to 1 by Rust when InterruptACK clears all interrupt bits and the
	/// PLIC IRQ should be lowered (``_lower_irq_if_idle``).
	pub irq_maybe_lower: u8,
	pub _pad: [u8; 6],
}

// Safety: Python holds the backing ctypes arrays alive for the FFI call.
unsafe impl Send for FfiVirtIoCtx {}
unsafe impl Sync for FfiVirtIoCtx {}

// ============================================================
//  FFI structs (grouped parameters for run_parallel)
// ============================================================

/// Hart execution parameters passed across the FFI boundary.
#[repr(C)]
/// Shared external-interrupt context owned by Python, polled by Rust.
///
/// PLIC and device state lives on the Python side.  When a device raises
/// (or lowers) an interrupt, Python updates this struct.  Rust checks
/// ``pending`` periodically inside the hart loop; when set it exits,
/// Python can call ``_native_sync_plic_mip()`` to update each
/// hart's ``mip`` with the latest PLIC-driven MEIP/SEIP bits.
#[repr(C)]
pub struct FfiExtIrqCtx {
	/// Non-zero: at least one external interrupt source is asserted and
	/// the PLIC state may have changed.  Rust exits on next check.
	pub pending: u8,
	/// Bitmap of pending interrupt sources.  Bit *i* corresponds to PLIC
	/// interrupt source *i* (1 = UART, 2 = VirtIO, …).  Updated atomically
	/// by Python; currently informational, may drive inline delivery later.
	pub sources: u32,
	/// Highest priority among currently-pending sources, or 0 if none.
	/// Rust may skip and exit when priority <= the current hart's
	/// PLIC threshold (not yet implemented — always exits when pending≠0).
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

/// 内存与保护上下文 (FFI 边界): RAM/shadow + PMP.
#[repr(C)]
pub struct FfiMemCtx {
	pub mem: *const MemCtx,
	pub pmp: *const FfiPmpCtx,
}

/// 中断控制器上下文 (FFI 边界): CLINT (定时器/IPI) + PLIC (外部中断).
#[repr(C)]
pub struct FfiInterruptCtx {
	pub clint: *const FfiClintCtx,
	pub plic: *const FfiPlicCtx,
}

/// 设备上下文 (FFI 边界): MMIO 范围 + UART/virtio/watchdog 设备指针.
#[repr(C)]
pub struct FfiDevicesCtx {
	pub dev: *const DevMMIOAddrInfo,
	pub uart: *const FfiUartCtx,
	pub virtio: *const FfiVirtIoCtx,
	pub watchdog: *const FfiWatchdogCtx,
}

/// Breakpoint configuration passed across the FFI boundary.
#[repr(C)]
pub struct FfiBpCtx {
	pub addrs: *const u64,
	pub count: u32,
}

/// TLB generation counter passed across the FFI boundary (mutable —
/// Rust writes back the values so they persist across calls).
#[repr(C)]
pub struct FfiTlbCtx {
	pub gen: *mut u64,
	pub gen_per_hart: *mut u64,
}

/// Empty UART context for when no UART is configured.
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
