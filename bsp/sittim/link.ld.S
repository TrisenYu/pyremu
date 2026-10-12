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

	//> section for elf header
	.eh_frame : {
		*(.eh_frame)
		*(.eh_frame.*)
		. = ALIGN(8);
	}
	.dynsym : {
		*(.dynsym)
		. = ALIGN(8);
	}
	.dynstr : {
		*(.dynstr)
		. = ALIGN(8);
	}
	.gnu.hash : {
		*(.gnu.hash)
		. = ALIGN(8);
	}
	.hash : {
		*(.hash)
		. = ALIGN(8);
	}

	.rela.dyn : {
		_rela_dyn_start = .;
		*(.rela.dyn)
		*(.rela.*)
		_rela_dyn_end = .;
	}

	. = ALIGN(0x1000);

	_instr_class_end = .;

	. = ALIGN(0x200000);
	_data_class_start = .;
	.dynamic : {
		*(.dynamic)
		. = ALIGN(8);
	}
	.got : {
		*(.got)
		*(.got.*)
		. = ALIGN(8);
	}

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

	/* start address of memory pool */
	. = ALIGN(0x200000);
	_end = .;
}
