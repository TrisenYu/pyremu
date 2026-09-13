//! `#[repr(C)]` state structures that cross the Python FFI boundary.

use core::fmt;
use core::sync::atomic::{AtomicU32, AtomicU64, AtomicU8, Ordering};

include!("configs_gen.rs");

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
	/// enclave IDs >= 256 would alias to 0 (= host) and break both
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
	/// 外部中断位的存在性缓存, 供 ``imsic_topei_peek`` 快速排除无外部中断的
	/// 情形 (见该函数注释). 由属主 hart 在 eip 变更后重算, 另由跨 hart 的
	/// ``imsic_eip_set`` 置位 — 该调用可能来自另一个 hart 的加速执行线程, 故
	/// 必须为原子类型. 类型与 u8 同布局 (1 字节, 对齐 1), FFI 布局不变.
	pub eip_ext_any: AtomicU8,
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
		out.eip_ext_any = AtomicU8::new(self.eip_ext_any.load(Ordering::Relaxed));
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
			eip_ext_any: AtomicU8::new(0),
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
	// (永不清除 — 清除仅发生在 timecmp/stimecmp 写入: 未来值->清位, 0->禁用),
	// 从而与其他活跃 hart 的共享 mtime 膨胀解耦。这是与"每指令持续比较
	// cur_mtime >= cmp"模型的关键差异: 共享 mtime 由所有活跃 hart 以 fetch_max
	// 推进, 快 hart 会过早触发慢 hart 刚重设 deadline 的定时器 -> mret 后立即再 trap
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
	/// TermIO RX 通知 — 引擎为让 Python 搬运 stdin 字节而主动退出本轮,
	/// 不是调试器可见的停止事件 (见 hart_sched.rs 的 hart_worker)。
	pub const RX_WAIT: u8 = 9;
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
	use crate::ffi::InstrToBeExec;
	use std::mem::{align_of, size_of};

	#[test]
	fn tlb_entry_size() {
		// mdid 加宽 u8->u64 后 TlbEntry 由 32B 涨到 40B (mdid 对齐到偏移 24).
		assert_eq!(size_of::<TlbEntry>(), 40);
		assert_eq!(align_of::<TlbEntry>(), 8);
	}

	#[test]
	fn hart_state_alignment() {
		assert_eq!(align_of::<HartState>(), 8);
	}

	/// ImsicFile 的字段偏移锁定 — 必须与 Python 侧 ``ImsicFileC``
	/// (hart.py) 的 ctypes 字段顺序逐一对应, 否则 marshal 时 Python 写入
	/// 会落到错误的字节 (eip_ext_any 改为 AtomicU8 时布局不变, 由本测试守住).
	#[test]
	fn imsic_file_offsets_match_python_layout() {
		use core::mem::offset_of;
		// eip[64] + eie[64] 各 256 字节.
		assert_eq!(offset_of!(ImsicFile, eidelivery), 512);
		assert_eq!(offset_of!(ImsicFile, eithreshold), 513);
		assert_eq!(offset_of!(ImsicFile, select), 516);
		assert_eq!(offset_of!(ImsicFile, present), 520);
		assert_eq!(offset_of!(ImsicFile, eip_ext_any), 521);
		assert_eq!(size_of::<ImsicFile>(), 524);
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
		if crate::CFG_AIA {
			hs.imsic_m.present = 1;
			hs.imsic_s.present = 1;
		};
		assert_eq!(hs.gprs[0], 0);
		assert_eq!(hs.mode, riscv_mode::M);
	}
}
