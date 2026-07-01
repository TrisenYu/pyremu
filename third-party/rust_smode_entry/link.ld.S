OUTPUT_ARCH("riscv")
ENTRY(_start)

SECTIONS {
	. = LINK_BASE;
	_start_ld = .;

	_text_start = .;
	.text : {
		*(.text.init)
		*(.text)
		*(.text.*)
		. = ALIGN(8);
	}
	_text_end = .;

	. = ALIGN(0x1000);

	_rodata_start = .;
	.rodata : {
		*(.rodata)
		*(.rodata.*)
		. = ALIGN(8);
	}
	_rodata_end = .;

	. = ALIGN(0x1000);

	_data_start = .;
	.data : {
		*(.data)
		*(.data.*)
		*(.sdata)
		. = ALIGN(8);
	}
	_data_end = .;

	. = ALIGN(0x1000);

	_bss_start = .;
	.bss : {
		*(.bss)
		*(.bss.*)
		*(.sbss)
		. = ALIGN(8);
	}
	_bss_end = .;

	. = ALIGN(0x1000);
	_end = .;

	/* 动态重定位表 — 供 entry.s 在运行时修正 GOT 中的绝对地址.
	   R_RISCV_RELATIVE (type=3) 条目: *(load_base + r_offset) = load_base + r_addend */
	.rela.dyn : {
		_rela_dyn_start = .;
		*(.rela.dyn)
		*(.rela.*)
		_rela_dyn_end = .;
	}
}
