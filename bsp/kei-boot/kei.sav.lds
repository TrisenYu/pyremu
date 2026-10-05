OUTPUT_ARCH(riscv)
ENTRY(_start)

SECTIONS {
    . = 0x0;
    .text : { *(.text) }

    /* BSS is 8 bytes aligned */
    . = ALIGN(8);
    _bss_start = .;
    .bss : { *(.sbss) *(.bss) *(COMMON) }
    . = ALIGN(8);
    _bss_end = .;
}
