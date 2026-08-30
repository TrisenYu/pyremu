//! 陷态分发。紧跟 smode_entry/trap_handler.c + ref-emod trap/exceptions.c。

use crate::constants::*;
use crate::csr;
use crate::ecall_aux;
use crate::hang;
use crate::sched;
use crate::syscall;

#[cfg(feature = "diagnostic")]
use crate::paging;
#[cfg(feature = "diagnostic")]
use crate::println;
// ---------------------------------------------------------------
//  GPR 寄存器别名（对应 smode_entry/trap_handler.h 中 CTX_INDEX_*）
// ---------------------------------------------------------------

#[allow(unused)]
pub const RA: usize = 1;
pub const SP: usize = 2;
pub const GP: usize = 3;
pub const TP: usize = 4;
pub const T0: usize = 5;
pub const T1: usize = 6;
pub const T2: usize = 7;
pub const S0: usize = 8;
pub const S1: usize = 9;
pub const A0: usize = 10;
pub const A1: usize = 11;
pub const A2: usize = 12;
pub const A3: usize = 13;
pub const A4: usize = 14;
pub const A5: usize = 15;
pub const A6: usize = 16;
pub const A7: usize = 17;
pub const S2: usize = 18;
pub const S3: usize = 19;
pub const S4: usize = 20;
pub const S5: usize = 21;
pub const S6: usize = 22;
pub const S7: usize = 23;
pub const S8: usize = 24;
pub const S9: usize = 25;
pub const S10: usize = 26;
pub const S11: usize = 27;
pub const T3: usize = 28;
pub const T4: usize = 29;
pub const T5: usize = 30;
pub const T6: usize = 31;

// ---------------------------------------------------------------
//  TrapGprs —— 与 entry.s 的 SAVE_CONTEXT 布局一致
// ---------------------------------------------------------------

/// 汇编 SAVE_CONTEXT 后的寄存器快照。
/// 31 个 GPR + 用户 sp 槽 = 32 × 8 = 256 + 8 = 264 字节。
#[repr(C)]
pub struct TrapGprs {
	/// x0..x31，x0 恒为 0（占位）。
	pub regs: [u64; 32],
	/// 用户 sp（SAVE_CONTEXT 后由 csrrw t1, sscratch, x0; STORE t1, 偏移(sp) 填入）。
	pub usp: u64,
}

#[allow(unused)]
impl TrapGprs {
	// 所有 32 GPR 的 getter
	#[inline]
	pub fn zero(&self) -> u64 {
		self.regs[0]
	}
	#[inline]
	pub fn ra(&self) -> u64 {
		self.regs[RA]
	}
	#[inline]
	pub fn sp(&self) -> u64 {
		self.regs[SP]
	}
	#[inline]
	pub fn gp(&self) -> u64 {
		self.regs[GP]
	}
	#[inline]
	pub fn tp(&self) -> u64 {
		self.regs[TP]
	}
	#[inline]
	pub fn t0(&self) -> u64 {
		self.regs[T0]
	}
	#[inline]
	pub fn t1(&self) -> u64 {
		self.regs[T1]
	}
	#[inline]
	pub fn t2(&self) -> u64 {
		self.regs[T2]
	}
	#[inline]
	pub fn s0(&self) -> u64 {
		self.regs[S0]
	}
	#[inline]
	pub fn s1(&self) -> u64 {
		self.regs[S1]
	}
	#[inline]
	pub fn a0(&self) -> u64 {
		self.regs[A0]
	}
	#[inline]
	pub fn a1(&self) -> u64 {
		self.regs[A1]
	}
	#[inline]
	pub fn a2(&self) -> u64 {
		self.regs[A2]
	}
	#[inline]
	pub fn a3(&self) -> u64 {
		self.regs[A3]
	}
	#[inline]
	pub fn a4(&self) -> u64 {
		self.regs[A4]
	}
	#[inline]
	pub fn a5(&self) -> u64 {
		self.regs[A5]
	}
	#[inline]
	pub fn a6(&self) -> u64 {
		self.regs[A6]
	}
	#[inline]
	pub fn a7(&self) -> u64 {
		self.regs[A7]
	}
	#[inline]
	pub fn s2(&self) -> u64 {
		self.regs[S2]
	}
	#[inline]
	pub fn s3(&self) -> u64 {
		self.regs[S3]
	}
	#[inline]
	pub fn s4(&self) -> u64 {
		self.regs[S4]
	}
	#[inline]
	pub fn s5(&self) -> u64 {
		self.regs[S5]
	}
	#[inline]
	pub fn s6(&self) -> u64 {
		self.regs[S6]
	}
	#[inline]
	pub fn s7(&self) -> u64 {
		self.regs[S7]
	}
	#[inline]
	pub fn s8(&self) -> u64 {
		self.regs[S8]
	}
	#[inline]
	pub fn s9(&self) -> u64 {
		self.regs[S9]
	}
	#[inline]
	pub fn s10(&self) -> u64 {
		self.regs[S10]
	}
	#[inline]
	pub fn s11(&self) -> u64 {
		self.regs[S11]
	}
	#[inline]
	pub fn t3(&self) -> u64 {
		self.regs[T3]
	}
	#[inline]
	pub fn t4(&self) -> u64 {
		self.regs[T4]
	}
	#[inline]
	pub fn t5(&self) -> u64 {
		self.regs[T5]
	}
	#[inline]
	pub fn t6(&self) -> u64 {
		self.regs[T6]
	}

	// syscall 返回时写回的 setter
	#[inline]
	pub fn set_a0(&mut self, v: u64) {
		self.regs[A0] = v;
	}
	#[inline]
	pub fn set_a7(&mut self, v: u64) {
		self.regs[A7] = v;
	}
}

// ---------------------------------------------------------------
//  C ABI 入口
// ---------------------------------------------------------------

#[unsafe(no_mangle)]
pub unsafe extern "C" fn trap_dispatch(gprs: &mut TrapGprs, sepc: u64, scause: u64, stval: u64) {
	// 中断
	if scause & (1 << 63) != 0 {
		interrupt_dispatch(scause & !(1 << 63));
		return;
	}

	// ECALL from U-mode
	if scause == 0x8 {
		let ret = syscall_dispatch(gprs);
		gprs.set_a0(ret);
		csr::write_sepc(sepc + 4);
		return;
	}

	// 访问故障：转发到 M-mode
	if scause == 0x5 || scause == 0x7 {
		ecall_aux::enclave_call_unmatched_acc_fault(stval);
		return;
	}

	// 页错误：额外输出 VA -> PA 诊断信息
	#[cfg(feature = "diagnostic")]
	if scause == 0xc || scause == 0xd || scause == 0xf {
		if let Some(pa) = paging::get_pa(stval) {
			println!("page_fault: va=0x{stval:x} pa=0x{pa:x}\n");
		}
	}

	// 其余同步异常 —— 输出具体原因后挂起
	hang::hang_with_msg(match scause {
		0x0 => "trap: instruction address misaligned\n",
		0x1 => "trap: instruction access fault\n",
		0x2 => "trap: illegal instruction\n",
		0x3 => "trap: breakpoint\n",
		0x4 => "trap: load address misaligned\n",
		0x6 => "trap: store/amo address misaligned\n",
		0x9 => "trap: ecall from S-mode\n",
		0xb => "trap: ecall from M-mode\n",
		0xc => "trap: instruction page fault\n",
		0xd => "trap: load page fault\n",
		0xf => "trap: store/amo page fault\n",
		_ => "trap: unknown exception\n",
	});
}

// ---------------------------------------------------------------
//  系统调用（后续可按需扩展 syscall 号）
// ---------------------------------------------------------------

fn syscall_dispatch(gprs: &TrapGprs) -> u64 {
	syscall::syscall_handler(gprs)
}

// ---------------------------------------------------------------
//  中断（当前仅 timer）
// ---------------------------------------------------------------

fn interrupt_dispatch(cause: u64) {
	match cause {
		1 => {
			// 软件中断 (SSIP): M-mode 通过 IPI 注入,
			// 通知 S-mode 有 host 请求待处理.
			csr::clear_csr!(sip, csr::SSI);
			sched::check_pending_requests();
		}
		5 => {
			// 定时器中断
			unsafe {
				let now: u64;
				core::arch::asm!("csrr {0}, 0xC01", out(reg) now);
				ecall_aux::sbi_set_timer(now + TIMER_INTERVAL);
			}
			csr::clear_csr!(sip, csr::STI);

			// 时间片检查 (见 sched.rs).
			sched::tick_and_check_quota();
			// 同时检查是否有 host 发来的待处理请求.
			sched::check_pending_requests();
		}
		_ => {}
	}
}
