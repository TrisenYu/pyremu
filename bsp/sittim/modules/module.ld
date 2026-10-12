OUTPUT_ARCH("riscv")
ENTRY(module_init)

SECTIONS
{
	.module_desc : ALIGN(8) {
		LONG(0x31444d53)                           /* magic: "SMD1" read in little endian */
		LONG(1)                                    /* abi_version */
		LONG(module_init - ADDR(.module_desc))     /* entry */
		LONG(_image_body_end - ADDR(.module_desc)) /* body_len */
		LONG(ADDR(.bss) - ADDR(.module_desc))      /* data_class_off */
		LONG(SIZEOF(.bss))                         /* bss_size */
		LONG(0)                                    /* flags */
		. += 36;                                   /* reserved[9]: 0 */
	}

	ASSERT(
		SIZEOF(.module_desc) == 64,
		"module image descriptor: unexpected size"
	)

	.text : ALIGN(8) {
		*(.text.init)
		*(.text .text.*)
		. = ALIGN(8);
	}

	.rodata : ALIGN(8) {
		*(.rodata .rodata.*)
		*(.srodata .srodata.*)
		. = ALIGN(8);
	}

	.data : ALIGN(8) {
		*(.data .data.*)
		*(.sdata .sdata.*)
		. = ALIGN(8);
	}

	_image_body_end = .;

	.bss : ALIGN(0x1000) {
		*(.bss .bss.*)
		*(.sbss .sbss.*)
		*(COMMON)
		. = ALIGN(8);
	}

	ASSERT(
		SIZEOF(.data) == 0,
		"module image: writable statics with non-zero initial value belong in the BSS section")

	/DISCARD/ : {
		*(.eh_frame .eh_frame.*)
		*(.note .note.*)
		*(.comment)
		*(.riscv.attributes)
	}
}
