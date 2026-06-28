//! Panic-halt 辅助函数。对应 smode_entry/hang.c。

use core::arch::asm;

use crate::println;

/// 无限 WFI 循环。用于飞地无恢复路径时挂起当前 hart。
pub fn hang() -> ! {
    loop {
        unsafe { asm!("wfi"); }
    }
}

/// 挂起并输出消息。消息应为纯 ASCII。
pub fn hang_with_msg(msg: &str) -> ! {
    println!("{msg}\n");
    hang();
}
