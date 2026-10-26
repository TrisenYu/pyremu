//! 含支持中断委派判定的陷态处理逻辑。
//!
//! 依据 ``medeleg`` 与 ``mideleg`` 把陷态委派到 S 模式

use crate::concurrent::{ConcurrentClintCtx, ModuleState};
use crate::ffi::FfiDevCtx;
use crate::hart_sched::read_gpr;
use crate::interrupt::clint::write_mtimecmp;
use crate::interrupt::imsic_clear_ipi_on_trap;
use crate::state::{riscv_mode, HartState};
use std::sync::atomic::Ordering;
// ============================================================
//  陷态原因码, 与 mcause、scause 的编码兼容
// ============================================================

/// RISC-V 异常的原因码, 由 mcause 第 63 位的取值确定是异常还是中断。
#[allow(dead_code)]
pub mod exc_code {
	pub const INSTR_ACCESS_FAULT: u64 = 1;
	pub const ILL_INSTR: u64 = 2;
	pub const BREAKPOINT: u64 = 3;
	pub const LD_MISALIGNED: u64 = 4;
	pub const LD_ACCESS_FAULT: u64 = 5;
	pub const ST_MISALIGNED: u64 = 6;
	pub const ST_ACCESS_FAULT: u64 = 7;
	pub const ECALL_UMODE: u64 = 8;
	pub const ECALL_SMODE: u64 = 9;
	pub const ECALL_MMODE: u64 = 11;
	pub const INSTR_PAGE_FAULT: u64 = 12;
	pub const LD_PAGE_FAULT: u64 = 13;
	pub const ST_PAGE_FAULT: u64 = 15;
}

/// 构造与 ``mcause`` 兼容的值: bit 63 为中断标志, 低位为原因码。
#[inline]
pub fn mcause_val(code: u64, is_interrupt: bool) -> u64 {
	if is_interrupt {
		code | (1u64 << 63)
	} else {
		code
	}
}

// ============================================================
//  mstatus 位定义的相关常量
// ============================================================

const MSTATUS_SIE: u64 = 1 << 1;
const MSTATUS_MIE: u64 = 1 << 3;
const MSTATUS_SPIE: u64 = 1 << 5;
const MSTATUS_MPIE: u64 = 1 << 7;
const MSTATUS_SPP: u64 = 1 << 8;
const MSTATUS_MPP: u64 = 0b11 << 11;

/// 把特权模式映射为 MPP 的编码。
#[inline]
fn mode_to_mpp(mode: u8) -> u64 {
	match mode {
		riscv_mode::U => 0,
		riscv_mode::S => 1,
		riscv_mode::M => 3,
		_ => 0,
	}
}

/// 把特权模式映射为 SPP 的编码, U 模式为 0, S 模式为 1。
#[inline]
fn mode_to_spp(mode: u8) -> u64 {
	match mode {
		riscv_mode::U => 0,
		riscv_mode::S => 1,
		_ => 0,
	}
}

// ============================================================
// 含委派判定的入口
// ============================================================

/// 派发一次陷态, 依据 ``medeleg`` 与 ``mideleg`` 判定是否委派。
///
/// 当异常或中断已被委派且 hart 不处于 M 模式时, 把陷态交给 S 模式;
/// 其余情形交给 M 模式。行为与 Python 侧 ``trap_handler.py`` 的
/// ``deliver_trap`` 一致。
///
/// 返回 ``0``, 调用方不得推进 PC, 因为 PC 已改为陷态入口地址。
pub fn deliver_trap(state: &mut HartState, code: u64, tval: u64) -> u64 {
	let is_interrupt = (code >> 63) != 0;
	let exc_code = code & 0x7fff_ffff_ffff_ffff;

	// M 模式发生的陷态永不派发到其他模式处理。
	let delegate = if state.mode != riscv_mode::M {
		if is_interrupt {
			(state.mideleg & (1 << exc_code)) != 0
		} else {
			(state.medeleg & (1 << exc_code)) != 0
		}
	} else {
		false
	};

	if delegate {
		deliver_trap_smode(state, code, tval)
	} else {
		deliver_trap_mmode(state, code, tval)
	}
}

// ============================================================
//  S 模式陷态派发
// ============================================================

/// 把陷态分发到 S 模式, 保存 PC 与相关 CSR, 并跳转到 ``stvec``。
///
/// 由 ``deliver_trap`` 在满足委派条件时调用, 行为与 Python 侧
/// ``trap_handler.py`` 的 ``_trap_deliver_smode`` 一致。
pub fn deliver_trap_smode(state: &mut HartState, code: u64, tval: u64) -> u64 {
	// 任何陷态都清除 LR/SC 的保留地址。
	state.reservation_valid = 0;
	state.reservation_addr = 0;

	// 从 WFI 等待状态唤醒。
	state.waiting = 0;

	// 把 PC 保存到 sepc。
	state.sepc = state.pc;

	// 保存原因码。
	state.scause = code;

	// 保存 tval。
	state.stval = tval;

	// mstatus: SPIE 取原 SIE, SIE 清 0, SPP 取当前模式。
	let sie_present = (state.mstatus & MSTATUS_SIE) != 0;
	let spp_bits = mode_to_spp(state.mode) << 8;
	if sie_present {
		state.mstatus |= MSTATUS_SPIE;
	} else {
		state.mstatus &= !MSTATUS_SPIE;
	}
	state.mstatus &= !MSTATUS_SIE;
	state.mstatus = (state.mstatus & !MSTATUS_SPP) | spp_bits;

	// 切换到 S 模式。
	state.mode = riscv_mode::S;

	// 跳转到 stvec。
	let stvec = state.stvec;
	let tvec_mode = stvec & 0x3;
	let tvec_base = stvec & !0x3;
	let exc_code = code & 0x7fff_ffff_ffff_ffffu64;
	let is_interrupt = (code >> 63) != 0;

	if tvec_mode == 0 || !is_interrupt {
		// direct 模式或异常: 跳转到基址。
		state.pc = tvec_base;
	} else {
		// vectored 模式且为中断: 跳转到基址加 4 乘原因码。
		state.pc = tvec_base + 4 * exc_code;
	}

	// MSIP 与 SSIP 的自动清除按模式区分。
	// AIA 模式下 sbi_ipi_raw_clear(false) 不产生效果, eip 的清除推迟到
	// MTOPEI 或 STOPEI 得到响应, 使处理程序读到中断标识。
	// 传统模式下清除对应的 mip 位与 IMSIC 的 eip, 依据 RISC-V 规范 §3.1.15。
	if !is_interrupt {
		return 0;
	}
	// AIA 模式下 IPI 经 MEIP 或 SEIP 投递, 原因码为 11 或 9, 不经 MSIP 或 SSIP
	// 的原因码 3 或 1。中断以原因码 3 或 1 到达时按传统模式处理。
	let in_aia_s = state.imsic_s.present != 0 && state.imsic_s.eidelivery != 0;
	if exc_code == 3 {
		if in_aia_s {
			// AIA 模式推迟 MSIP 与 eip 的清除。
		} else {
			state.mip.fetch_and(!(1u64 << 3), Ordering::AcqRel); // MSIP
			imsic_clear_ipi_on_trap(state, 3);
		}
	} else if exc_code == 1 {
		if in_aia_s {
			// AIA 模式推迟 SSIP 与 eip 的清除。
		} else {
			state.mip.fetch_and(!(1u64 << 1), Ordering::AcqRel); // SSIP
			imsic_clear_ipi_on_trap(state, 1);
		}
	}
	0
}

// ============================================================
//  M 模式陷态分发, 未经委派时的去向
// ============================================================

/// 把陷态分发给 M 模式, 保存 PC 与相关 CSR, 并跳转到 ``mtvec``。
///
/// 由 ``deliver_trap`` 在陷态未被委派或 hart 已处于 M 模式时调用,
/// 行为与 Python 侧 ``trap_handler.py`` 的 ``_trap_deliver_mmode`` 一致。
pub fn deliver_trap_mmode(state: &mut HartState, code: u64, tval: u64) -> u64 {
	// 任何陷态都清除 LR/SC 的保留地址。
	state.reservation_valid = 0;
	state.reservation_addr = 0;

	// RISC-V 规范 §3.1.15 规定未委派的机器级中断在进入陷态时由硬件自动清除,
	// 否则 CLINT 的 MSIP 保持置位期间, MRET 之后中断立即再次触发。
	// AIA 模式下 MSIP 与 eip 的清除推迟到 compute_mtopi 或 MTOPEI 得到响应,
	// 见下面的分支说明。
	let is_interrupt = (code >> 63) != 0;
	let exc_code = code & 0x7fff_ffff_ffff_ffffu64;

	// AIA 模式下 eidelivery 为 1, OpenSBI 经 MTOPI 分派, 在 compute_mtopi 中清除
	// MSIP。若在此处清除 MSIP 与 IMSIC 的 eip, MTOPI 读不到中断标识, IPI 丢失,
	// 引起 RCU 停滞与多 hart 启动超时。
	// 传统模式下 eidelivery 为 0, OpenSBI 按 mcause 分派, 并在此处按规范 §3.1.15 清除 MSIP。
	let in_aia = state.imsic_m.present != 0 && state.imsic_m.eidelivery != 0;
	if is_interrupt && exc_code == 3 {
		if in_aia {
			// AIA 模式推迟 MSIP 与 eip 的清除, 使 sbi_trap_aia_irq 经 MTOPI
			// 读到中断标识后正确分派。
		} else {
			// 传统模式按 RISC-V 规范 §3.1.15 自动清除 MSIP。
			state.mip.fetch_and(!(1u64 << 3), Ordering::AcqRel);
			imsic_clear_ipi_on_trap(state, 3);
		}
	}

	// 从 WFI 等待状态唤醒。
	state.waiting = 0;

	// 把 PC 保存到 mepc。
	state.mepc = state.pc;

	// 保存原因码。
	state.mcause = code;

	// 保存 tval。
	state.mtval = tval;

	// mstatus: MPIE 取原 MIE, MIE 清 0, MPP 取当前模式。
	let mie_present = (state.mstatus & MSTATUS_MIE) != 0;
	let mpp_bits = mode_to_mpp(state.mode) << 11;
	if mie_present {
		state.mstatus |= MSTATUS_MPIE;
	} else {
		state.mstatus &= !MSTATUS_MPIE;
	}
	state.mstatus &= !MSTATUS_MIE;
	state.mstatus = (state.mstatus & !MSTATUS_MPP) | mpp_bits;

	// RISC-V 规范规定进入 M 模式陷态时清除 MPRV, 使处理程序访问自身的栈与
	// 数据时不经过上一特权模式的 MMU 翻译。MPRV 为 1 期间发生嵌套中断时,
	// 若不清除该位, M 模式的栈访问会使用 S 模式页表, 破坏 M 模式的栈状态。
	state.mstatus &= !(1u64 << 17); // 清除 MPRV

	// 切换到 M 模式。
	state.mode = riscv_mode::M;

	// 跳转到 mtvec。
	let mtvec = state.mtvec;
	let tvec_mode = mtvec & 0x3;
	let tvec_base = mtvec & !0x3;
	let exc_code = code & 0x7fff_ffff_ffff_ffffu64;
	let is_interrupt = (code >> 63) != 0;

	if tvec_mode == 0 || !is_interrupt {
		// direct 模式或异常: 跳转到基址。
		state.pc = tvec_base;
	} else {
		// vectored 模式且为中断: 跳转到基址加 4 乘原因码。
		state.pc = tvec_base + 4 * exc_code;
	}

	0 // 推进量为 0, PC 已被改写。
}

/// 为非法编码产生 IllInstr 陷态, 使用 ``deliver_trap`` 做委派判定。
#[inline]
pub fn deliver_illegal_instruction(state: &mut HartState, tval: u64) -> u64 {
	deliver_trap(state, mcause_val(exc_code::ILL_INSTR, false), tval)
}

// ============================================================
//  特权指令的处理函数
// ============================================================

pub(crate) fn priv_ecall_concurrent(
	state: &mut HartState,
	_instr: u32,
	clint: &ConcurrentClintCtx,
	_module: &ModuleState,
	dev: &dyn FfiDevCtx,
) -> u64 {
	// SBI 是 S 模式到 M 模式的接口, 只有 S 模式发起的 ecall 才可能命中下面的
	// 快速路径: U 模式的 ecall 一律以 ECALL_UMODE 分发给 S 模式内核.
	// U 模式的退出序列 li a0,0、li a7,1、ecall 与 SBI 传统 putchar 的扩展号与
	// 功能号相同, 缺少模式判据时会被串口快速路径得到响应, a0 被当作字符写出且
	// 不分发陷态, 程序越过退出 ecall 继续执行。
	if state.mode != riscv_mode::S {
		let ecall_cause = match state.mode {
			riscv_mode::U => mcause_val(exc_code::ECALL_UMODE, false),
			_ => mcause_val(exc_code::ECALL_MMODE, false),
		};
		deliver_trap(state, ecall_cause, 0);
		return 0;
	}

	let a7 = read_gpr(state, 17);
	let a6 = read_gpr(state, 16);

	// 设备快速路径, 如 UART 的 sbi_putchar, 在当前调用内直接完成, 不退出
	// 本轮加速执行。每输出一个字符退出一次, 每次退出要在 Python 与动态库
	// 之间复制 hart 状态, 约需 1 至 5 毫秒, 是飞地输出密集时吞吐大幅下降的
	// 原因之一。新设备只需实现 FfiDevCtx trait, 无需修改此处的分派逻辑。
	if let Some(advance) = dev.try_ecall(state, a7, a6) {
		return advance;
	}

	if a7 == 0x54494d45 && a6 == 0 {
		// SBI_TIME 的 set_timer: 写入 M 模式的 mtimecmp, 使 MTIP 在请求时触发。
		let stime_val = read_gpr(state, 10);
		let hid = state.mhartid as usize;
		if hid < (clint.num_harts as usize) {
			(unsafe { &*clint.mtimecmp.add(hid) }).store(stime_val, Ordering::Release);
			// 与 CLINT 的 MMIO 写路径 write_mtimecmp_for_target 一致, 设定一次性的
			// mtip_deadline。否则 mtip_deadline 保持 0, sync_mtip 落入共享 mtime 的
			// 比较分支, 其他活跃 hart 经 fetch_max 推高的共享 mtime 使 MTIP 提前置位,
			// MRET 后立即再次进入陷态而无法推进。飞地经 sbi_set_timer 而非 SSTC 的
			// stimecmp 设定定时器, 是唯一命中该分支的调用方。
			let now = (unsafe { &*clint.mtime }).load(Ordering::Acquire);
			write_mtimecmp(state, now, stime_val, clint.timebase_hz);
		}
		state.gprs[10] = 0;
		return 4;
	}

	// 改为生成 ECALL 陷态, 由 M 模式处理程序设置 S 模式下的跨核请求同步 TLB。

	// 其余 ECALL 一律引发处理器陷态, 在本轮加速执行内继续运行。
	// 至此 state.mode 必为 S: 非 S 模式已在函数开头引发陷态并在得到处理后返回。
	deliver_trap(state, mcause_val(exc_code::ECALL_SMODE, false), 0);
	// deliver_trap 已把 PC 改为 mtvec 或 stvec 的入口, 并把模式改为 M 或 S。
	// 返回 0 使陷态处理重复执行而不退出当前处于链接库下的加速执行。
	0
}

pub(crate) fn priv_mret_concurrent(state: &mut HartState, instr: u32) -> u64 {
	// MRET 只在 M 模式下合法
	if state.mode != riscv_mode::M {
		deliver_illegal_instruction(state, instr as u64);
		return 0;
	}
	let mpp = (state.mstatus >> 11) & 0x3;
	let mpie = (state.mstatus >> 7) & 1;
	state.mode = match mpp {
		0 => riscv_mode::U,
		1 => riscv_mode::S,
		3 => riscv_mode::M,
		_ => riscv_mode::M,
	};
	state.mstatus &= !(1 << 3);
	if mpie != 0 {
		state.mstatus |= 1 << 3;
	}
	state.mstatus |= 1 << 7;
	state.mstatus &= !(0b11 << 11);
	state.pc = state.mepc;
	state.waiting = 0;
	0
}

pub(crate) fn priv_sret_concurrent(
	state: &mut HartState,
	instr: u32,
	_ctx: &crate::translate::WalkCtx,
) -> u64 {
	if state.mode < riscv_mode::S {
		deliver_illegal_instruction(state, instr as u64);
		return 0;
	}
	let spp = (state.mstatus >> 8) & 1;
	let spie = (state.mstatus >> 5) & 1;
	state.mode = if spp == 0 {
		riscv_mode::U
	} else {
		riscv_mode::S
	};
	state.mstatus &= !(1 << 1);
	if spie != 0 {
		state.mstatus |= 1 << 1;
	}
	state.mstatus |= 1 << 5;
	state.mstatus &= !(1 << 8);
	state.pc = state.sepc;
	state.waiting = 0;
	0
}

pub(crate) fn priv_wfi_concurrent(state: &mut HartState, instr: u32) -> u64 {
	let tw = (state.mstatus >> 21) & 1;
	if tw != 0 && state.mode != riscv_mode::M {
		deliver_illegal_instruction(state, instr as u64);
		return 0;
	}
	if state.wfi_woken != 0 {
		state.wfi_woken = 0;
		return 4;
	}
	let pending = state.mip.load(Ordering::Acquire) & state.mie;
	if pending != 0 {
		return 4;
	}
	state.waiting = 1;
	4
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::ffi::FfiUartCtx;
	use std::cell::Cell;
	use std::sync::atomic::{AtomicU64, AtomicU8};

	fn default_state() -> HartState {
		let mut s: HartState = unsafe { std::mem::zeroed() };
		s.mode = riscv_mode::M;
		s.mtvec = 0x80000000;
		s.stvec = 0x80004000;
		s.pc = 0x1000;
		s
	}

	/// 构造单 hart 的计时器上下文, 供 ecall 用例使用.
	fn single_hart_clint(
		mtime: &AtomicU64,
		mtimecmp: &AtomicU64,
		msip: &AtomicU8,
	) -> ConcurrentClintCtx {
		ConcurrentClintCtx {
			base: 0x2000000,
			mtime: mtime as *const AtomicU64,
			mtimecmp: mtimecmp as *const AtomicU64,
			msip: msip as *const AtomicU8,
			num_harts: 1,
			msip_pending: Cell::new(std::ptr::null()),
			hart_threads: Cell::new(std::ptr::null()),
			hart_states: Cell::new(std::ptr::null()),
			timebase_hz: 10_000_000,
		}
	}

	/// 构造一个发送缓冲区可用的串口上下文。
	/// ``tx_wr`` 指向调用方的局部变量, 供调用后读取写指针。
	fn uart_ctx_with_tx_buffer(tx_buf: *mut u8, tx_wr: *mut u32) -> FfiUartCtx {
		FfiUartCtx {
			base: 0x1000_0000,
			tx_buf,
			tx_cap: 8,
			tx_wr,
			ie: 0,
			txctrl: 0,
			rxctrl: 0,
			rx_fifo_len: 0,
			tx_notify_fd: -1,
			no_stdout: 1,
			rx_notify: std::ptr::null_mut(),
		}
	}

	// ---- 委派判定测试用例 ----

	#[test]
	fn trap_delegates_to_smode_when_medeleg_set() {
		let mut s = default_state();
		s.mode = riscv_mode::S;
		s.medeleg = 1 << exc_code::ILL_INSTR; // 委派 IllInstr
		deliver_trap(&mut s, mcause_val(exc_code::ILL_INSTR, false), 0xdead);
		assert_eq!(
			s.mode,
			riscv_mode::S,
			"should stay in S-mode when delegated"
		);
		assert_eq!(s.sepc, 0x1000, "sepc should save faulting PC");
		assert_eq!(s.scause, mcause_val(exc_code::ILL_INSTR, false));
		assert_eq!(s.stval, 0xdead);
		assert_eq!(s.pc, 0x80004000, "should jump to stvec base");
	}

	#[test]
	fn trap_not_delegated_from_mmode() {
		let mut s = default_state();
		s.mode = riscv_mode::M;
		s.medeleg = 1 << exc_code::ILL_INSTR; // M 模式下不进行委派判定
		deliver_trap(&mut s, mcause_val(exc_code::ILL_INSTR, false), 0xbeef);
		assert_eq!(s.mode, riscv_mode::M, "M-mode traps never delegate");
		assert_eq!(s.mepc, 0x1000);
		assert_eq!(s.mcause, mcause_val(exc_code::ILL_INSTR, false));
		assert_eq!(s.pc, 0x80000000, "should jump to mtvec base");
	}

	#[test]
	fn trap_not_delegated_when_medeleg_bit_clear() {
		let mut s = default_state();
		s.mode = riscv_mode::S;
		s.medeleg = 0; // 未委派任何异常
		deliver_trap(&mut s, mcause_val(exc_code::ILL_INSTR, false), 0xbeef);
		assert_eq!(s.mode, riscv_mode::M, "should switch to M-mode");
		assert_eq!(s.mepc, 0x1000);
	}

	#[test]
	fn interrupt_delegates_to_smode_via_mideleg() {
		let mut s = default_state();
		s.mode = riscv_mode::S;
		s.mideleg = 1 << 7; // 委派 MTI, 即定时器中断
		deliver_trap(&mut s, mcause_val(7, true), 0);
		assert_eq!(s.mode, riscv_mode::S, "delegated interrupt stays in S-mode");
		assert_eq!(s.sepc, 0x1000);
		assert_eq!(s.scause, mcause_val(7, true));
	}

	// ---- S 模式陷态分发测试 ----

	#[test]
	fn smode_trap_saves_sie_to_spie() {
		let mut s = default_state();
		s.mode = riscv_mode::S;
		s.mstatus = MSTATUS_SIE; // SIE 为 1
		s.medeleg = 1 << exc_code::ILL_INSTR;
		deliver_trap(&mut s, mcause_val(exc_code::ILL_INSTR, false), 0);
		assert_eq!(s.mstatus & MSTATUS_SIE, 0, "SIE should be cleared");
		assert_ne!(s.mstatus & MSTATUS_SPIE, 0, "SPIE should be set");
	}

	#[test]
	fn smode_trap_sets_spp() {
		let mut s = default_state();
		s.mode = riscv_mode::S;
		s.medeleg = 1 << exc_code::ILL_INSTR;
		deliver_trap(&mut s, mcause_val(exc_code::ILL_INSTR, false), 0);
		let spp = (s.mstatus & MSTATUS_SPP) >> 8;
		assert_eq!(spp, 1, "SPP should be 1 (was S-mode)");
	}

	#[test]
	fn smode_trap_from_umode_sets_spp_0() {
		let mut s = default_state();
		s.mode = riscv_mode::U;
		s.medeleg = 1 << exc_code::ILL_INSTR;
		deliver_trap(&mut s, mcause_val(exc_code::ILL_INSTR, false), 0);
		let spp = (s.mstatus & MSTATUS_SPP) >> 8;
		assert_eq!(spp, 0, "SPP should be 0 (was U-mode)");
	}

	// ---- M 模式陷态分发测试 ----

	#[test]
	fn deliver_ill_instr_saves_mepc() {
		let mut s = default_state();
		s.pc = 0xbeef;
		deliver_illegal_instruction(&mut s, 0xdead_beef);
		assert_eq!(s.mepc, 0xbeef);
		assert_eq!(s.mcause, mcause_val(exc_code::ILL_INSTR, false));
		assert_eq!(s.mtval, 0xdead_beef);
	}

	#[test]
	fn deliver_trap_clears_mie() {
		let mut s = default_state();
		s.mstatus = MSTATUS_MIE; // MIE 为 1
		deliver_illegal_instruction(&mut s, 0);
		assert_eq!(s.mstatus & MSTATUS_MIE, 0, "MIE should be cleared");
		assert_eq!(
			s.mstatus & MSTATUS_MPIE,
			MSTATUS_MPIE,
			"MPIE should be set because MIE was 1"
		);
	}

	#[test]
	fn deliver_trap_sets_mpie_correctly_when_mie_was_0() {
		let mut s = default_state();
		s.mstatus = 0; // MIE 为 0
		deliver_illegal_instruction(&mut s, 0);
		assert_eq!(
			s.mstatus & MSTATUS_MPIE,
			0,
			"MPIE should be 0 when MIE was 0"
		);
	}

	#[test]
	fn deliver_trap_switches_to_mmode() {
		let mut s = default_state();
		s.mode = riscv_mode::S;
		deliver_illegal_instruction(&mut s, 0);
		assert_eq!(s.mode, riscv_mode::M);
	}

	#[test]
	fn deliver_trap_saves_mpp() {
		let mut s = default_state();
		s.mode = riscv_mode::S;
		deliver_illegal_instruction(&mut s, 0);
		let mpp = (s.mstatus & MSTATUS_MPP) >> 11;
		assert_eq!(mpp, mode_to_mpp(riscv_mode::S));
	}

	#[test]
	fn deliver_trap_direct_mtvec() {
		let mut s = default_state();
		s.mtvec = 0x80004000; // mode 为 0, 即 direct
		deliver_illegal_instruction(&mut s, 0);
		assert_eq!(s.pc, 0x80004000); // direct 模式: 跳转到基址
	}

	#[test]
	fn deliver_trap_vectored_exception() {
		let mut s = default_state();
		s.mtvec = 0x80004000 | 1; // vectored 模式
		deliver_illegal_instruction(&mut s, 0);
		// 异常: 跳转到基址, 而非基址加 4 乘原因码
		assert_eq!(s.pc, 0x80004000);
	}

	#[test]
	fn deliver_trap_clears_reservation() {
		let mut s = default_state();
		s.reservation_valid = 1;
		s.reservation_addr = 0x8000_0000;
		deliver_illegal_instruction(&mut s, 0);
		assert_eq!(s.reservation_valid, 0);
	}

	#[test]
	fn mcause_encoding() {
		assert_eq!(mcause_val(2, false), 2);
		assert_eq!(mcause_val(7, true), (1u64 << 63) | 7);
	}

	// ============================================================
	//  进入陷态时自动清除 MSIP 的用例
	// ============================================================
	//
	// 为避免 MSIP 保持置位引起同一中断反复进入 trap 处理函数，对 MSIP 与 SSIP 的清除推迟到 MTOPI 得到响应为止

	#[test]
	fn msip_cleared_in_deliver_trap_mmode() {
		// deliver_trap_mmode 对原因码为 3 的中断清除 MSIP
		let mut s = default_state();
		s.mip.store(1 << 3, Ordering::Release); // 置位 MSIP
		deliver_trap_mmode(&mut s, mcause_val(3, true), 0);
		assert_eq!(
			s.mip.load(Ordering::Acquire) & (1 << 3),
			0,
			"MSIP must be cleared by deliver_trap_mmode"
		);
	}

	#[test]
	fn msip_not_cleared_for_non_interrupt_in_deliver_trap_mmode() {
		// deliver_trap_mmode 只对中断清除 MSIP, 不对异常清除
		let mut s = default_state();
		s.mip.store(1 << 3, Ordering::Release); // 置位 MSIP
		deliver_trap_mmode(&mut s, mcause_val(exc_code::ILL_INSTR, false), 0);
		assert_eq!(
			s.mip.load(Ordering::Acquire) & (1 << 3),
			1 << 3,
			"MSIP must NOT be cleared for exceptions"
		);
	}

	#[test]
	fn ssip_cleared_in_deliver_trap_for_delegated_interrupt() {
		// SSIP 的原因码为 1, 可以被委派, 验证 deliver_trap 会清除该位
		let mut s = default_state();
		s.mode = riscv_mode::S;
		s.mip.store(1 << 1, Ordering::Release); // 置位 SSIP
		s.mideleg = 1 << 1; // 委派 SSIP
		s.mstatus |= 1 << 1; // SIE 为 1
		deliver_trap(&mut s, mcause_val(1, true), 0);
		assert_eq!(
			s.mip.load(Ordering::Acquire) & (1 << 1),
			0,
			"SSIP must be cleared by deliver_trap for delegated interrupt"
		);
	}

	#[test]
	fn msip_cleared_for_non_mmode_interrupt_in_deliver_trap() {
		// 非 M 模式下发生中断且该中断未被委派时, deliver_trap 转入
		// deliver_trap_mmode, 验证 S 模式到 M 模式的路径清除 MSIP。
		let mut s = default_state();
		s.mode = riscv_mode::S;
		s.mip.store(1 << 3, Ordering::Release); // 置位 MSIP, 该中断不可委派
		s.mstatus |= 1 << 1; // SIE 为 1
					   // MSIP 不可委派, 因此经 deliver_trap 的 M 模式路径分发,
					   // 该路径内部调用 _trap_deliver_mmode。
		deliver_trap(&mut s, mcause_val(3, true), 0);
		assert_eq!(
			s.mip.load(Ordering::Acquire) & (1 << 3),
			0,
			"MSIP must be cleared in deliver_trap non-delegated M-interrupt path"
		);
	}

	// ============================================================
	// ============================================================
	//
	// 只写 CLINT 的 mtimecmp 而不设 mtip_deadline 时, sync_mtip 落入
	// 共享 mtime 的比较分支, 其他活跃 hart 经 fetch_max 使 MTIP
	// 提前被设置, 并让当前 hart mret 后立即再次进入陷态。

	#[test]
	fn sbi_set_timer_fast_path_sets_mtip_deadline() {
		let mut state: HartState = unsafe { std::mem::zeroed() };
		state.mhartid = 0;
		state.total_instrs = 1000;
		// SBI 只在 S 模式下可用; 零初始化的 mode 为 U, 必须显式置为 S,
		state.mode = riscv_mode::S;
		// a0 为目标绝对时间, 取未来值; a6 为 0, a7 为 SBI_TIME 即 0x54494D45。
		state.gprs[10] = 11000;
		state.gprs[16] = 0;
		state.gprs[17] = 0x5449_4d45;

		let mtime = AtomicU64::new(1000);
		let mtimecmp = AtomicU64::new(0);
		let msip = AtomicU8::new(0);

		let clint = ConcurrentClintCtx {
			base: 0x2000000,
			mtime: &mtime as *const AtomicU64,
			mtimecmp: &mtimecmp as *const AtomicU64,
			msip: &msip as *const AtomicU8,
			num_harts: 1,
			msip_pending: Cell::new(std::ptr::null()),
			hart_threads: Cell::new(std::ptr::null()),
			hart_states: Cell::new(std::ptr::null()),
			timebase_hz: 10_000_000,
		};
		let module = ModuleState::new(1, 1, 0, vec![0].into_boxed_slice(), false, 0, Box::new([]));
		let uart: FfiUartCtx = unsafe { std::mem::zeroed() };

		let advance = priv_ecall_concurrent(&mut state, 0, &clint, &module, &uart);

		assert_eq!(advance, 4, "sbi_set_timer 应返回 advance=4");
		assert_eq!(state.gprs[10], 0, "a0 应清零为 0");
		assert_eq!(mtimecmp.load(Ordering::Acquire), 11000, "mtimecmp 应被写入");
		assert_ne!(
			state.mtip_deadline, 0,
			"sbi_set_timer 必须设定 mtip_deadline"
		);
		assert!(
			state.mtip_deadline > state.total_instrs,
			"deadline 必须落在未来, 而非立即到期"
		);
		assert_eq!(
			state.mip.load(Ordering::Acquire) & (1 << 7),
			0,
			"未来定时器不应立即置位 MTIP"
		);
	}

	// ---- ecall 特权级判定的用例 ----
	//
	// ``dev.try_ecall`` 与 SBI set_timer 的快速路径只在 S 模式下启用。
	// 用户程序的退出序列 li a0,0、li a7,1、ecall 中, a7 与 a6 分别与 SBI 传统
	// putchar 的扩展号 0x01 和功能号 0 相同, 缺少该判定时会被串口快速路径
	// 得到响应: a0 被当作字符写出且不分发陷态, 程序越过退出 ecall 继续执行。
	// 下面两条用例分别覆盖该判定的两侧。

	#[test]
	fn umode_ecall_with_sbi_putchar_encoding_never_takes_fast_path() {
		let mut state = default_state();
		state.mode = riscv_mode::U;
		state.medeleg = 1 << exc_code::ECALL_UMODE;
		// 用户程序的退出序列: a0 取 0, a7 取 1 即 SBI putchar 的扩展号, a6 取 0。
		state.gprs[10] = 0;
		state.gprs[16] = 0;
		state.gprs[17] = 1;

		let mtime = AtomicU64::new(0);
		let mtimecmp = AtomicU64::new(0);
		let msip = AtomicU8::new(0);
		let clint = single_hart_clint(&mtime, &mtimecmp, &msip);
		let module = ModuleState::new(1, 1, 0, vec![0].into_boxed_slice(), false, 0, Box::new([]));

		let mut tx_buf = [0u8; 8];
		let mut tx_wr: u32 = 0;
		let uart = uart_ctx_with_tx_buffer(tx_buf.as_mut_ptr(), &mut tx_wr as *mut u32);

		priv_ecall_concurrent(&mut state, 0, &clint, &module, &uart);

		assert_eq!(tx_wr, 0, "U 模式 ecall 不得被串口快速路径认领而写出字符");
		assert_eq!(
			state.scause,
			mcause_val(exc_code::ECALL_UMODE, false),
			"U 模式 ecall 必须以 ECALL_UMODE 投递给 S 模式内核"
		);
		assert_eq!(state.sepc, 0x1000, "sepc 应保存发起 ecall 的 PC");
		assert_eq!(state.pc, 0x80004000, "应跳转到 stvec");
	}

	#[test]
	fn smode_ecall_with_sbi_putchar_encoding_takes_fast_path() {
		let mut state = default_state();
		state.mode = riscv_mode::S;
		// S 模式发出相同的编码: a0 取 'A', a6 取 0, a7 取 1。
		state.gprs[10] = 0x41;
		state.gprs[16] = 0;
		state.gprs[17] = 1;

		let mtime = AtomicU64::new(0);
		let mtimecmp = AtomicU64::new(0);
		let msip = AtomicU8::new(0);
		let clint = single_hart_clint(&mtime, &mtimecmp, &msip);
		let module = ModuleState::new(1, 1, 0, vec![0].into_boxed_slice(), false, 0, Box::new([]));

		// 归档侧按两个字节的条目解码, 第一个字节为 hart 标识, 第二个字节为字符,
		// 故该快速路径写入的条目必须带发出该字符的 hart 标识。
		state.mhartid = 2;
		let mut tx_buf = [0u8; 8];
		let mut tx_wr: u32 = 0;
		let uart = uart_ctx_with_tx_buffer(tx_buf.as_mut_ptr(), &mut tx_wr as *mut u32);

		let advance = priv_ecall_concurrent(&mut state, 0, &clint, &module, &uart);

		assert_eq!(advance, 4, "should gain advance=4");
		assert_eq!(tx_wr, 1, "should write one entry");
		assert_eq!(tx_buf[0], 2, "entry byte 0 should carry the hart id");
		assert_eq!(tx_buf[1], 0x41, "entry byte 1 should carry the character");
		assert_eq!(state.gprs[10], 0, "should return 0");
		assert_eq!(state.pc, 0x1000, "should not trap");
	}
}
