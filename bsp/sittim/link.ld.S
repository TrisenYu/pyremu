OUTPUT_ARCH("riscv")
ENTRY(_start)

SECTIONS {
	. = LINK_BASE;
	_start_ld = .;

	//> .text Section
	_text_start = .;
	.text : {
		*(.text.init)
		*(.text)
		*(.text.*)
		. = ALIGN(8);
	}
	_text_end = .;

	. = ALIGN(0x1000);

	//> .rodata Section
	_rodata_start = .;
	.rodata : {
		*(.rodata)
		*(.rodata.*)
		. = ALIGN(8);
	}
	_rodata_end = .;

	. = ALIGN(0x1000);

	//> .data Section
	_data_start = .;
	.data : {
		*(.data)
		*(.data.*)
		*(.sdata)
		. = ALIGN(8);
	}
	_data_end = .;

	. = ALIGN(0x1000);

	//> .bss Section
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

	/*
		relocation table section
		correct the base address of GOT in entry.S during the runtime.
		Especially the R_RISCV_RELATIVE (type=3):
			*(load_base + r_offset) = load_base + r_addend
	*/
	.rela.dyn : {
		_rela_dyn_start = .;
		*(.rela.dyn)
		*(.rela.*)
		_rela_dyn_end = .;
	}
}
