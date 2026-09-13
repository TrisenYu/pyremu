/*
 * 最小 ELF 布局, 适配 Rust S-mode 运行时的 ELF 加载器:
 *   载荷从 0x10000 开始 (U-mode 低地址区), 与 picolibc 的 libc.a 兼容.
 *
 * Rust 运行时将 PT_LOAD 段映射到 U-mode 地址空间,
 * 通过 Sv39 页表隔离不同飞地的物理页.
 */

OUTPUT_ARCH("riscv")
ENTRY(_start)

SECTIONS {
    . = 0x10000;

    .text : {
        *(.text.init)
        *(.text)
        *(.text.*)
    }

    .rodata : {
        *(.rodata)
        *(.rodata.*)
    }

    .data : {
        *(.data)
        *(.data.*)
    }

    .bss : {
        *(.bss)
        *(.bss.*)
        *(COMMON)
    }

    . = ALIGN(8);
    _end = .;
}
