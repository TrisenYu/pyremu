/*
 * 多程序内核链接脚本.
 *
 * 内核 (M+S) + 多个 U 模式用户程序共同链接.
 * 布局: .tohost -> .text (所有输入文件合并) -> .rodata -> .data -> .bss
 *
 * 输出入口: _start (kernel.s 定义)
 */

OUTPUT_ARCH("riscv")
ENTRY(_start)

SECTIONS {
    . = 0x80000000;

    .tohost : ALIGN(0x1000) {
        *(.tohost)
    }

    . += 0x1000;

    .text : ALIGN(0x1000) {
        __TEXT_START__ = .;
        *(.text .text.*)
        __TEXT_END__ = .;
    }

    .rodata : ALIGN(8) {
        *(.rodata .rodata.*)
    }

    .data : ALIGN(8) {
        *(.data .data.*)
    }

    .bss : ALIGN(8) {
        __BSS_START__ = .;
        *(.bss .bss.*)
        __BSS_END__ = .;
    }

    _end = .;
}
