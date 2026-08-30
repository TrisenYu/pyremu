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

    /*
     * U 模式用户程序专用区域 — 独立页 (0x80020000).
     *
     * 内核在 setup_sv39 中将其 identity 映射为 U=1 页 (内核页 U=0),
     * 这样 S 模式内核取指与 U 模式进程执行互不冲突 (Sv39 U-bit 规则:
     * S 模式永不从 U=1 页取指, 见 RISC-V §4.3.2). 若 U 程序与内核文本
     * 混在同一页, 页粒度的 U 位无法同时服务 S 与 U 两种模式.
     *
     * __UPROG_START__ / __UPROG_END__ 由内核 setup_sv39 引用, 决定映射范围.
     */
    . = 0x80020000;
    __UPROG_START__ = .;

    .uprog : ALIGN(8) {
        *(.uprog.text .uprog.text.*)
        *(.uprog.rodata .uprog.rodata.*)
        *(.uprog.data .uprog.data.*)
    }

    .uprog.bss : ALIGN(8) {
        *(.uprog.bss .uprog.bss.*)
    }

    __UPROG_END__ = .;
}
