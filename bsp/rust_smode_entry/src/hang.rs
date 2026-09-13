//! Panic-halt 辅助函数。对应 smode_entry/hang.c。

use core::arch::asm;

use crate::csr;
use crate::println;

/// 无限 WFI 循环。用于飞地无恢复路径时挂起当前 hart (纯 fail-stop)。
pub fn hang() -> ! {
	loop {
		unsafe {
			asm!("wfi");
		}
	}
}

/// 重新打开全局中断后冻结挂起, 不输出任何内容。
///
/// 陷态入口已将 sstatus.SIE 清 0, 若直接 WFI, host 后续
/// 发来的终止请求 (REQUEST_SHUTDOWN 经 OpenSBI 反射为 SSIP) 会因 SIE=0 被
/// 屏蔽, WFI 不唤醒, 该 hart 永久失联 (Linux 侧表现为 rcu_sched stall 且
/// NMI 无响应)。因此这里重新打开全局中断并使能 SSIP 与 STIP, 让中断照常投递
/// 到 interrupt_dispatch
fn fault_halt_enable_irq() -> ! {
	csr::write_sie(csr::SSI | csr::STI);
	csr::write_sstatus(csr::read_sstatus() | csr::SSTATUS_SIE);
	hang();
}

/// 冻结挂起并输出消息 (调用方自带现场信息)。
pub fn fault_halt(msg: &str) -> ! {
	println!("{msg}\n");
	fault_halt_enable_irq();
}

/// 冻结挂起并输出 S 模式陷态的完整现场。
///
/// 仅打印一句原因不足以定位 runtime 缺陷: 必须同时留下 ``scause`` /
/// ``stval`` / ``sepc`` 三元组。``sepc`` 是出错指令自身的地址, ``stval``
/// 是出错的内存地址 (页错误/访问错误/地址错位) 或指令编码 (非法指令),
/// 两者合起来即"哪条指令访问了哪个地址"。此路径为 fail-stop 且不可恢复,
/// 故现场打印不受 ``diagnostic`` feature 约束 —— 缺了它缺陷无法定位。
pub fn fault_halt_exc(scause: u64, stval: u64, sepc: u64, msg: &str) -> ! {
	println!("{msg}");
	println!("[trap] s-fault scause=0x{scause:x} stval=0x{stval:x} sepc=0x{sepc:x}\n");
	fault_halt_enable_irq();
}
