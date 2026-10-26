//! 跨 Python FFI 边界的 `#[repr(C)]` 状态结构体.

use core::fmt;
use core::sync::atomic::{AtomicU32, AtomicU64, AtomicU8, Ordering};

include!("configs_gen.rs");

// ============================================================
//  TLB 条目
// ============================================================

/// 单个 TLB 条目, 占 40 字节, 8 字节对齐.
/// 布局由 FFI 锁定, 必须与 Python 侧 ``TlbEntry`` 的 ctypes 定义一致.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct TlbEntry {
	pub vpn: u64,
	pub ppn: u64,
	pub perm: u8,
	pub level: u8,
	pub valid: u8,
	pub mdid: u64,
	pub tlb_epoch: u32,
	pub dirty: u8,
	pub accessed: u8,
	/// 插入条目时取自 satp 的 ASID. 查找时 asid 与条目 asid 均非零且不相等即未命中;
	/// 条目 asid 为 0 时匹配任何地址空间.
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
//  IMSIC 中断文件
// ============================================================

/// 单个 IMSIC 中断文件, 每个 hart 各有一个 M 模式文件与一个 S 模式文件.
/// eip 与 eie 各 2048 位, 按每字 32 位存为 64 个 u32,
/// 便于 FFI 传输; RV64 的 CSR 读取把相邻的两个 u32 合并为一个 64 位值.
#[repr(C)]
pub struct ImsicFile {
	pub eip: [AtomicU32; 64],
	pub eie: [AtomicU32; 64],
	pub eidelivery: u8,
	pub eithreshold: u8,
	pub select: u32,
	pub present: u8,
	/// 供 ``imsic_topei_peek`` 在没有外部中断时直接返回.
	/// 本 hart 在 eip 变更后、跨 hart 调用的 ``imsic_eip_set`` 会重新计算这一用于暂存外部中断位的原子状态;
	/// 该类型与 u8 布局相同, 占 1 字节, 对齐为 1, FFI 布局不变.
	pub eip_ext_any: AtomicU8,
}

// AtomicU32 不实现 Clone 与 Copy. 克隆只发生在快照与初始化时,
// 不与跨 hart 的并发访问同时进行, 故逐字段按值复制.
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
//  每个 hart 的状态
// ============================================================

#[repr(C)]
pub struct HartState {
	// ---- GPRs ----
	pub gprs: [u64; 32],

	// ---- 关键 CSR ----
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

	// ---- 预留 LR/SC ----
	pub reservation_addr: u64,
	/// LR 读入的值, 作为 SC 判定是否写入的期望值.
	/// 若 SC 在比较时重新读取, 读到的总是当前值, 另一个 hart 的并发写入
	/// 不会使 SC 失败, 两个 hart 可同时进入同一临界区.
	pub reservation_value: u64,

	// ---- 单字节字段 ----
	pub reservation_valid: u8,
	pub mode: u8,
	pub mmu_mode: u8,
	pub waiting: u8,
	pub wfi_woken: u8,
	pub halted: u8,
	pub mdid: u64,
	pub pmpsplit: u8,
	pub _pad: [u8; 7],

	// TLB 条目
	pub itlb: [TlbEntry; TLB_ENTRIES],
	pub dtlb: [TlbEntry; TLB_ENTRIES],

	// 其余 CSR
	pub mscratch: u64,
	pub sscratch: u64,
	pub mhartid: u64,
	pub mcounteren: u64,
	pub scounteren: u64,

	// Sstc stimecmp ----
	pub stimecmp: u64,

	// 中断缓存
	pub _mmu_mode_pad: u64,

	// 每 hart 指令计数
	/// 该 hart 已执行完成的指令总数, 每轮加速执行结束后累加, 从不清零.
	/// 用于提供该 hart 的 ``minstret`` 与调试器显示.
	pub total_instrs: u64,

	// F/D 浮点 ----
	/// 32 个浮点寄存器, 保存原始位模式而非 f64, 以保留 sNaN 与 qNaN
	/// 的载荷以及单精度数的 NaN 语义.
	pub fprs: [u64; 32],
	/// 浮点控制状态寄存器: 位 [7:5] 为 frm, 位 [4:0] 为 fflags.
	pub fcsr: u32,
	pub _fpad: [u8; 4],

	// AIA IMSIC 寄存器文件, 每 hart 一个 M 文件与一个 S 文件
	pub imsic_m: ImsicFile,
	pub imsic_s: ImsicFile,

	pub stip_deadline: u64,
	pub mtip_deadline: u64,

	// ---- 加速执行引擎的 TLB 计数 ----
	//
	// 在 translate_va 的翻译点上保持累加;
	// Python 侧只读取不回写, 故其取值覆盖该引擎此前记录的全部翻译活动.
	// 权限检查未通过而返回故障的命中计入命中数: 该情形下 TLB 已给出翻译,
	// 与未命中不同.
	pub itlb_hits: u64,
	pub itlb_misses: u64,
	pub dtlb_hits: u64,
	pub dtlb_misses: u64,
}

// ============================================================
//  常量
// ============================================================

#[allow(dead_code)]
pub mod exit_reason {
	pub const NORMAL: u8 = 0;
	pub const TRAP: u8 = 1;
	pub const MMIO: u8 = 2;
	pub const ECALL: u8 = 3;
	pub const EBREAK: u8 = 4;
	pub const WFI_WAIT: u8 = 5;
	/// 取指的目标物理地址不属于主存, 引擎只从主存读取指令字节, 故本轮退出.
	/// 由 Python 取指: 目标属于设备 MMIO 时读取设备的寄存器, 属于未定义物理地址范围时投递访问故障.
	pub const NOT_MAINRAM: u8 = 6;
	pub const BREAKPOINT: u8 = 7;
	pub const TIMEOUT: u8 = 8;
	/// TermIO 接收通知: 为能让 Python 读取标准输入的字节而主动退出。
	/// 非调试器可见的停止事件.
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
//  测试
// ============================================================

#[cfg(test)]
mod tests {
	use super::*;
	use crate::ffi::InstrToBeExec;
	use std::mem::{align_of, size_of};

	#[test]
	fn tlb_entry_size() {
		assert_eq!(size_of::<TlbEntry>(), 40);
		assert_eq!(align_of::<TlbEntry>(), 8);
	}

	#[test]
	fn hart_state_alignment() {
		assert_eq!(align_of::<HartState>(), 8);
	}

	/// ImsicFile 的字段偏移必须与 Python 侧 ``ImsicFileC`` 的 ctypes 定义
	/// 逐一对应, 否则 marshal 时 Python 会写到错误的偏移而引发执行期间的异常.
	#[test]
	fn imsic_file_offsets_match_python_layout() {
		use core::mem::offset_of;
		// eip 与 eie 各 64 个 u32, 各占 256 字节.
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
		// 每个 TLB 有 TLB_ENTRIES 个条目, 每个条目 40 字节,
		// itlb 与 dtlb 合计 TLB_ENTRIES * 80 字节.
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
			fprs: [0u64; 32],
			fcsr: 0,
			_fpad: [0; 4],
			imsic_m: ImsicFile::empty(),
			imsic_s: ImsicFile::empty(),
			stip_deadline: 0,
			mtip_deadline: 0,
			itlb_hits: 0,
			itlb_misses: 0,
			dtlb_hits: 0,
			dtlb_misses: 0,
		};
		// AIA 使能时标记 IMSIC 状态可用, 使 sync_imsic() 与 imsic_topei_peek()
		// 能读到 eip 与 eie 状态.
		if crate::CFG_AIA {
			hs.imsic_m.present = 1;
			hs.imsic_s.present = 1;
		}
		assert_eq!(hs.gprs[0], 0);
		assert_eq!(hs.mode, riscv_mode::M);
	}
}
