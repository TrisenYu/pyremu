//! `#[repr(C)]` state structures that cross the Python FFI boundary.

use core::fmt;
use core::sync::atomic::{AtomicU32, AtomicU64, Ordering};

include!("config_gen.rs");

// ============================================================
//  TLB entry
// ============================================================

/// A single TLB entry — 40 bytes, 8-byte aligned.
/// Layout is FFI-locked; must match Python ``TlbEntry`` ctypes definition.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct TlbEntry {
	pub vpn: u64,
	pub ppn: u64,
	pub perm: u8,
	pub level: u8,
	pub valid: u8,
	/// Memory domain ID — full 64-bit.  Must NOT be narrowed to u8:
	/// enclave IDs ≥ 256 would alias to 0 (= host) and break both
	/// `mfence.did` domain isolation and PMP enclave-mode checks.
	pub mdid: u64,
	pub tlb_epoch: u32,
	pub dirty: u8,
	pub accessed: u8,
	/// ASID from satp at insertion time.  Lookup must match; different ASID
	/// (process switch without SFENCE.VMA) is a miss — prevents stale entries
	/// from leaking across address spaces when the kernel uses ASID-tagged TLBs.
	pub asid: u16,
}

impl TlbEntry {
	pub const fn empty() -> Self {
		TlbEntry {
			vpn: 0,
			ppn: 0,
			perm: 0,
			level: 0,
			valid: 0,
			mdid: 0,
			tlb_epoch: 0,
			dirty: 0,
			accessed: 0,
			asid: 0,
		}
	}
}

impl fmt::Debug for TlbEntry {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(
			f,
			"TlbEntry(vpn={:#x} ppn={:#x} perm={} level={} v={})",
			self.vpn, self.ppn, self.perm, self.level, self.valid
		)
	}
}

// ============================================================
//  IMSIC interrupt file (AIA)
// ============================================================

/// Single IMSIC interrupt file — M-mode or S-mode per hart.
/// eip/eie are u32 arrays (2048 bits / 32 bits per word = 64 words)
/// for compact FFI transfer; RV64 CSR reads pack two adjacent u32s.
#[repr(C)]
pub struct ImsicFile {
	pub eip: [AtomicU32; 64],
	pub eie: [AtomicU32; 64],
	pub eidelivery: u8,
	pub eithreshold: u8,
	pub select: u32,
	pub present: u8,
	pub eip_ext_any: u8,
}

// Manual Clone: AtomicU32 is intentionally !Clone / !Copy:
// a value-level copy is safe here because HartState cloning only happens
// at snapshot / init time, never during concurrent cross-hart access.
impl Clone for ImsicFile {
	fn clone(&self) -> Self {
		let mut out = Self::empty();
		for i in 0..64 {
			out.eip[i] = AtomicU32::new(self.eip[i].load(Ordering::Relaxed));
			out.eie[i] = AtomicU32::new(self.eie[i].load(Ordering::Relaxed));
		}
		out.eidelivery = self.eidelivery;
		out.eithreshold = self.eithreshold;
		out.select = self.select;
		out.present = self.present;
		out.eip_ext_any = self.eip_ext_any;
		out
	}
}
impl ImsicFile {
	pub fn empty() -> Self {
		ImsicFile {
			eip: std::array::from_fn(|_| AtomicU32::new(0)),
			eie: std::array::from_fn(|_| AtomicU32::new(0)),
			eidelivery: 0,
			eithreshold: 0,
			select: 0,
			present: 1,
			eip_ext_any: 0,
		}
	}
}

// ============================================================
//  Per-hart state
// ============================================================

#[repr(C)]
pub struct HartState {
	// ---- GPRs ----
	pub gprs: [u64; 32],

	// ---- Key CSRs (u64) ----
	pub mstatus: u64,
	pub mtvec: u64,
	pub stvec: u64,
	pub mepc: u64,
	pub sepc: u64,
	pub mcause: u64,
	pub scause: u64,
	pub mtval: u64,
	pub stval: u64,
	pub satp: u64,
	pub mie: u64,
	pub mip: AtomicU64,
	pub medeleg: u64,
	pub mideleg: u64,

	// ---- PC ----
	pub pc: u64,

	// ---- Reservation (LR/SC) ----
	pub reservation_addr: u64,
	/// Value loaded by LR — used as the expected value in SC's CAS.
	/// Without this, SC does a fresh load for the CAS expected value,
	/// which always equals the current value ->SC never fails due to
	/// a concurrent store from another hart (reservation-invalidation
	/// bug: two harts can simultaneously enter the same critical section).
	pub reservation_value: u64,

	// ---- single-byte fields ----
	pub reservation_valid: u8,
	pub mode: u8,
	pub mmu_mode: u8,
	pub waiting: u8,
	pub wfi_woken: u8,
	pub halted: u8,
	/// Memory domain ID — full 64-bit (see ``TlbEntry.mdid`` for why not u8).
	/// u64 对齐后由偏移 6 移到偏移 8, 整个字节块从 16B 涨到 24B.
	pub mdid: u64,
	pub pmpsplit: u8,
	pub _pad: [u8; 7],

	// ---- Phase B: TLB entries ----
	pub itlb: [TlbEntry; TLB_ENTRIES],
	pub dtlb: [TlbEntry; TLB_ENTRIES],

	// ---- Phase C: Additional CSRs ----
	pub mscratch: u64,
	pub sscratch: u64,
	pub mhartid: u64,
	pub mcounteren: u64,
	pub scounteren: u64,

	// ---- Phase D: Sstc stimecmp ----
	pub stimecmp: u64,

	// ---- Phase E: interrupt cache ----
	pub _mmu_mode_pad: u64,

	// ---- Phase F: per-hart instruction counter ----
	/// Number of instructions executed by this hart across all accelerationes.
	/// Incremented after each slice; never reset.
	/// Used to provide per-hart ``minstret`` and debugger display.
	pub total_instrs: u64,

	// ---- Diagnostic counters & MSIP edge tracking ----
	pub diag: HartDiag,

	// ---- Phase G: F/D floating point ----
	/// 32 个浮点寄存器, 存原始 bits (NaN-boxed), 非 f64 —
	/// 保留 sNaN/qNaN payload 与单精度 NaN-boxing 语义。
	pub fprs: [u64; 32],
	/// 浮点控制状态寄存器: bits[7:5]=frm, bits[4:0]=fflags。
	pub fcsr: u32,
	pub _fpad: [u8; 4],

	// ---- Phase H: AIA IMSIC register files (M + S per hart) ----
	pub imsic_m: ImsicFile,
	pub imsic_s: ImsicFile,

	// ---- Phase I: per-hart one-shot timer deadlines (QEMU ACLINT 模型) ----
	//
	// QEMU 在 timecmp/stimecmp 写入时判定过去/未来: 过去立即置位, 未来清位并
	// 设定一个一次性 deadline (``riscv_aclint_mtimer_write_timecmp``), 定时器到期
	// 才再次置位; 电平保持到前向重设 deadline。本字段把"一次性定时器"翻译到当前 hart
	// 的指令计数空间: 写入 future 值时, deadline = 写入时刻的 total_instrs +
	// ceil(剩余 tick / (timebase × NS_PER_INSTR / 1e9)) (见 timer_deadline_own)。
	// ``sync_mtip`` 对 ACTIVE hart 仅在 total_instrs >= deadline 时置位
	// (永不清除 — 清除仅发生在 timecmp/stimecmp 写入: 未来值→清位, 0→禁用),
	// 从而与其他活跃 hart 的共享 mtime 膨胀解耦。这是与"每指令持续比较
	// cur_mtime >= cmp"模型的关键差异: 共享 mtime 由所有活跃 hart 以 fetch_max
	// 推进, 快 hart 会过早触发慢 hart 刚重设 deadline 的定时器 → mret 后立即再 trap
	// 的活锁。0 = 未设定 deadline (写入即到期/禁用时置 0)。
	pub stip_deadline: u64,
	pub mtip_deadline: u64,
}

// ============================================================
//  Diagnostic / MSIP-edge sub-struct (always present, FFI-stable)
// ============================================================

/// Diagnostic counters and MSIP edge-detection state.
///
/// Kept as a sub-struct so all diag fields are namespaced under
/// ``state.diag.*``.  The layout is ``#[repr(C)]`` and every field is
/// always present — the ``diagnostic`` feature only gates the code that
/// *writes* to the counters, not the counters themselves.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct HartDiag {
	// ---- CLINT inline handler call counters ----
	pub clint_msip_set: u64,      // sync_msip observed edge transitions
	pub clint_msip_clr: u64,      // (unused, reserved)
	pub clint_mtc_wr: u64,        // direct CLINT MSIP writes seen in WFI
	pub clint_msip_wr0: u64,      // MMIO write-0 to MSIP
	pub clint_msip_wr1: u64,      // MMIO write-1 to MSIP
	pub clint_wr1_remote: u64,    // write-1 to another hart
	pub clint_wr1_self: u64,      // write-1 to self
	pub cooldown_start: u64,      // MRET/SRET with CLINT MSIP=1
	pub wfi_wake_msip: u64,       // WFI woke due to MSIP
	pub wfi_wake_mtip: u64,       // WFI woke due to timer
	pub wfi_wake_other: u64,      // WFI woke due to other interrupt
	pub trap_msip_total: u64,     // MSIP traps delivered
	pub trap_msip_delegated: u64, // MSIP traps delegated to S-mode
	/// Last CLINT MSIP edge-counter value.  ``sync_msip`` compares the
	/// current edge counter against this to detect new MSIP edges.
	pub msip_last_seen: u64,
	pub msip_masked_by_msie: u64,   // MSIP pending but MSIE=0
	pub msip_pending_no_trap: u64,  // MSIP pending but no trap delivered
	pub nt_mip_snapshot: u64,       // mip at first no_trap
	pub nt_mie_snapshot: u64,       // mie at first no_trap
	pub msie_cleared_at_pc: u64,    // PC where MSIE was explicitly cleared
	pub wfi_wake_no_msip_trap: u64, // WFI woke by MSIP edge but no trap
	pub nt_mode: u8,                // mode at first no_trap
	pub nt_clint_raw: u8,           // CLINT MSIP raw byte at first no_trap
	pub _pad2: [u8; 6],
	pub nt_pending: u64, // mip & mie at first no_trap
}

impl HartDiag {
	pub const fn zeroed() -> Self {
		HartDiag {
			clint_msip_set: 0,
			clint_msip_clr: 0,
			clint_mtc_wr: 0,
			clint_msip_wr0: 0,
			clint_msip_wr1: 0,
			clint_wr1_remote: 0,
			clint_wr1_self: 0,
			cooldown_start: 0,
			wfi_wake_msip: 0,
			wfi_wake_mtip: 0,
			wfi_wake_other: 0,
			trap_msip_total: 0,
			trap_msip_delegated: 0,
			msip_last_seen: 0,
			msip_masked_by_msie: 0,
			msip_pending_no_trap: 0,
			nt_mip_snapshot: 0,
			nt_mie_snapshot: 0,
			msie_cleared_at_pc: 0,
			wfi_wake_no_msip_trap: 0,
			nt_mode: 0,
			nt_clint_raw: 0,
			_pad2: [0; 6],
			nt_pending: 0,
		}
	}
}

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

/// Device MMIO ranges (crosses FFI boundary).
#[repr(C)]
pub struct FfiDevCtx {
	pub bases: *const u64,
	pub ends: *const u64,
	pub num: u8,
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
/// 与 FfiDevCtx / FfiUartCtx / FfiVirtIoCtx 同属设备上下文组 (由 FfiPeriphCtx
/// 携带), 但不镜像任何 guest 可见 MMIO: 仅携带一次加速执行的时钟源超时时间
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
	/// Capacity of ``tx_buf`` (must be power of two, ≤ 256 MiB).
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
	/// Driver features accepted by the guest (written at offset 0x020, 64-bit).
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
//  Constants
// ============================================================

#[allow(dead_code)]
pub mod exit_reason {
	pub const NORMAL: u8 = 0;
	pub const TRAP: u8 = 1;
	pub const MMIO: u8 = 2;
	pub const ECALL: u8 = 3;
	pub const EBREAK: u8 = 4;
	pub const WFI_WAIT: u8 = 5;
	pub const ERROR: u8 = 6;
	pub const BREAKPOINT: u8 = 7;
	pub const TIMEOUT: u8 = 8;
}

#[allow(dead_code)]
pub mod riscv_mode {
	pub const U: u8 = 0;
	pub const S: u8 = 1;
	pub const H: u8 = 2;
	pub const M: u8 = 3;
	pub const D: u8 = 8;
}

// ============================================================
//  Tests
// ============================================================

#[cfg(test)]
mod tests {
	use super::*;
	use std::mem::{align_of, size_of};

	#[test]
	fn tlb_entry_size() {
		// mdid 加宽 u8→u64 后 TlbEntry 由 32B 涨到 40B (mdid 对齐到偏移 24).
		assert_eq!(size_of::<TlbEntry>(), 40);
		assert_eq!(align_of::<TlbEntry>(), 8);
	}

	#[test]
	fn hart_state_alignment() {
		assert_eq!(align_of::<HartState>(), 8);
	}

	#[test]
	fn hart_state_size_reasonable() {
		let sz = size_of::<HartState>();
		// With TLB_ENTRIES entries per TLB, HartState grows proportionally.
		// 256 entries × 32 bytes × 2 (itlb+dtlb) = 16384 bytes for TLB alone.
		assert!(sz < 65536, "HartState size {} should be < 65536", sz);
	}

	#[test]
	fn instr_group_alignment() {
		assert_eq!(align_of::<InstrToBeExec>(), 8);
	}

	#[test]
	fn hart_state_defaults() {
		let mut hs = HartState {
			gprs: [0u64; 32],
			mstatus: 0,
			mtvec: 0,
			stvec: 0,
			mepc: 0,
			sepc: 0,
			mcause: 0,
			scause: 0,
			mtval: 0,
			stval: 0,
			satp: 0,
			mie: 0,
			mip: AtomicU64::new(0),
			medeleg: 0,
			mideleg: 0,
			pc: 0,
			reservation_addr: 0,
			reservation_value: 0,
			reservation_valid: 0,
			mode: riscv_mode::M,
			mmu_mode: 0,
			waiting: 0,
			wfi_woken: 0,
			halted: 0,
			mdid: 0,
			pmpsplit: 0,
			_pad: [0; 7],
			itlb: [TlbEntry::empty(); TLB_ENTRIES],
			dtlb: [TlbEntry::empty(); TLB_ENTRIES],
			mscratch: 0,
			sscratch: 0,
			mhartid: 0,
			mcounteren: 0,
			scounteren: 0,
			stimecmp: 0,
			_mmu_mode_pad: 0,
			total_instrs: 0,
			diag: HartDiag::zeroed(),
			fprs: [0u64; 32],
			fcsr: 0,
			_fpad: [0; 4],
			imsic_m: ImsicFile::empty(),
			imsic_s: ImsicFile::empty(),
			stip_deadline: 0,
			mtip_deadline: 0,
		};
		// Enable IMSIC state tracking when AIA is active so that
		// sync_imsic() and imsic_topei_peek() see eip/eie state.
		if crate::PYREMU_AIA {
			hs.imsic_m.present = 1;
			hs.imsic_s.present = 1;
		};
		assert_eq!(hs.gprs[0], 0);
		assert_eq!(hs.mode, riscv_mode::M);
	}
}
