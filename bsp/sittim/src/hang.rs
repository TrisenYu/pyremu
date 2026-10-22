//! Panic-halt 辅助函数。。
use core::arch::asm;
use core::ptr;

use crate::csr;
use crate::println;
use crate::trap::get_trap_reason;
use crate::trap_frame::TrapGprs;

/// 用于飞地无恢复路径时挂起当前 hart。
pub fn hang() -> ! {
	loop {
		unsafe {
			asm!("wfi");
		}
	}
}

/// 重新打开全局中断后冻结挂起
///
/// 陷态入口已将 sstatus.SIE 清 0, 若直接 WFI, host 后续
/// 发来的终止请求会因 SIE=0 被
/// 屏蔽, WFI 不唤醒, 在 Linux 侧表现为 rcu_sched stall 且
/// NMI 无响应。因此这里重新打开全局中断并使能 SSIP 与 STIP, 让中断照常投递
/// 到 interrupt_dispatch
fn fault_halt_enable_irq() -> ! {
	csr::write_sie(csr::SSI | csr::STI);
	csr::write_sstatus(csr::read_sstatus() | csr::SSTATUS_SIE);
	hang();
}

/// 冻结挂起并输出陷态现场
pub fn fault_halt(msg: &str) -> ! {
	println!("{msg}\n");
	fault_halt_enable_irq();
}

/// 冻结挂起并输出 S 模式陷态现场。
///
/// 报告本次陷态详情: 来源特权级、陷态栈帧的地址、帧内的 32 个通用寄存器、陷态入口读到的
/// sepc 与当前的 satp。
pub fn fault_halt_exc(gprs: &TrapGprs, scause: u64, stval: u64, sepc: u64) -> ! {
	println!(
		"\n!!!!!!!!!sittim-trap!!!!!!!!!\n{}\n\n",
		get_trap_reason(scause)
	);
	println!(
		"cause, tval: {:016x} {:016x}\n\
		ctxAddr,epc: {:016x} {:016x}\n\
		ra,sp,gp,tp: {:016x} {:016x} {:016x} {:016x}\n\
		t0,t1,t2   : {:016x} {:016x} {:016x}\n\
		s0,s1,a0,a1: {:016x} {:016x} {:016x} {:016x}\n\
		a2,a3,a4,a5: {:016x} {:016x} {:016x} {:016x}\n\
		a6,a7,s2,s3: {:016x} {:016x} {:016x} {:016x}\n\
		s4,s5,s6,s7: {:016x} {:016x} {:016x} {:016x}\n\
		s8,s9,sa,sb: {:016x} {:016x} {:016x} {:016x}\n\
		t3,t4,t5,t6: {:016x} {:016x} {:016x} {:016x}\n\
		!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n\n",
		scause,
		stval,
		ptr::from_ref(gprs) as u64,
		sepc,
		gprs.ra(),
		gprs.sp_before_trap,
		gprs.gp(),
		gprs.tp(),
		gprs.t0(),
		gprs.t1(),
		gprs.t2(),
		gprs.s0(),
		gprs.s1(),
		gprs.a0(),
		gprs.a1(),
		gprs.a2(),
		gprs.a3(),
		gprs.a4(),
		gprs.a5(),
		gprs.a6(),
		gprs.a7(),
		gprs.s2(),
		gprs.s3(),
		gprs.s4(),
		gprs.s5(),
		gprs.s6(),
		gprs.s7(),
		gprs.s8(),
		gprs.s9(),
		gprs.s10(),
		gprs.s11(),
		gprs.t3(),
		gprs.t4(),
		gprs.t5(),
		gprs.t6()
	);
	fault_halt_enable_irq();
}
