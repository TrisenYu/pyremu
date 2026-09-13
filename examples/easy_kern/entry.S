// entry.S — easy_kern 入口: 清零 BSS, 设置栈, 进入 C 主函数.
//
// OpenSBI fw_jump 完成 M 模式初始化后, 以 S 模式跳转到 0x80200000,
// 从 _start 开始执行 (satp=Bare, 虚拟地址即物理地址).
	.section .text.init,"ax",@progbits
	.globl _start
_start:
	// 清零 .bss: 从 _bss_start 到 _bss_end 按 8 字节推进
	la t0, _bss_start
	la t1, _bss_end
	// 逐 8 字节清零
1:
	bgeu t0, t1, 2f
	sd zero, 0(t0)
	addi t0, t0, 8
	j 1b
2:
	// 设置栈顶并调用 C 主函数
	la sp, _stack_top
	call main
	// main 返回后原地低功耗等待 (示例程序执行完毕即停)
3:
	wfi
	j 3b

	// 提示链接器该段不需可执行栈, 避免 ld.lld 告警
	.section .note.GNU-stack,"",@progbits
