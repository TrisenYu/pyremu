/* easy_kern.ld — easy_kern 链接脚本 (S 模式裸机 host, 加载地址 0x80200000)
 *
 * 布局 (地址递增):
 *   .text  入口 _start (entry.S) + main 及辅助函数
 *   .rodata 含 payload_embed.S 内嵌的飞地载荷 (cfrac 静态 ELF)
 *   .data  静态数据
 *   .bss   entry.S 在启动时清零 (不进 raw 二进制)
 *   [栈]   0x4000 字节, 位于 _stack_top
 *
 * OpenSBI fw_jump 完成 M 模式初始化后 mret 跳转到 0x80200000 (FW_JUMP_ADDR),
 * S 模式 satp=Bare, 故链接地址即物理地址, raw 展开与运行时地址一致.
 */
OUTPUT_ARCH(riscv)
ENTRY(_start)

BASE = 0x80200000;

SECTIONS {
	. = BASE;

	.text : {
		*(.text.init)
		*(.text .text.*)
	}

	. = ALIGN(16);
	.rodata : {
		*(.srodata .srodata.*)
		*(.rodata .rodata.*)
	}

	. = ALIGN(16);
	.data : {
		*(.sdata .sdata.*)
		*(.data .data.*)
	}

	. = ALIGN(16);
	.bss : {
		_bss_start = .;
		*(.sbss .sbss.*)
		*(.bss .bss.*)
		*(COMMON)
		_bss_end = .;
	}

	. = ALIGN(16);
	. += 0x4000;
	_stack_top = .;
	_end = .;
}
