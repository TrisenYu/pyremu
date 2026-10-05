/* 扩展模块映像的链接脚本。
 *
 * 本脚本不指定基址, 各段自地址 0 起依次排布, 映像内的地址即文件偏移, 运行时按偏移 0
 * 调用模块入口。故 .text.init 排在首位, 模块入口由 #[link_section = ".text.init"]
 * 放入该段。
 *
 * .eh_frame 与 .note.* 等段只在 ELF 中有意义, 保留会改变模块映像的字节, 故丢弃。
 */
OUTPUT_ARCH("riscv")
ENTRY(module_init)

SECTIONS
{
	.text : ALIGN(8)
	{
		*(.text.init)
		*(.text .text.*)
		. = ALIGN(8);
	}

	.rodata : ALIGN(8)
	{
		*(.rodata .rodata.*)
		*(.srodata .srodata.*)
		. = ALIGN(8);
	}

	.data : ALIGN(8)
	{
		*(.data .data.*)
		*(.sdata .sdata.*)
		. = ALIGN(8);
	}

	.bss : ALIGN(8)
	{
		*(.bss .bss.*)
		*(.sbss .sbss.*)
		*(COMMON)
		. = ALIGN(8);
	}

	/DISCARD/ :
	{
		*(.eh_frame .eh_frame.*)
		*(.note .note.*)
		*(.comment)
		*(.riscv.attributes)
	}
}
