OUTPUT_ARCH(riscv)
ENTRY(_start)

SECTIONS {
    . = 0x0;
    .text : { *(.text) }

    /* BSS 按 8 字节对齐 */
    . = ALIGN(8);
    _bss_start = .;
    .bss : { *(.sbss) *(.bss) *(COMMON) }
    . = ALIGN(8);
    _bss_end = .;
}
