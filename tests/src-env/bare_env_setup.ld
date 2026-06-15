/*
    MEMORY {
        FLASH (rx)  : ORIGIN = 0x08000000, LENGTH = 512K
        RAM   (rw)  : ORIGIN = 0x20000000, LENGTH = 128K
    }
*/
OUTPUT_ARCH("riscv")
ENTRY(_start)

SECTIONS {
    . = 0x80000000;
    .tohost : ALIGN(0x1000) { *(.tohost) }  
    . += 0x1000;
    .text : ALIGN(0x1000) { 
        __CODE_TEXT_START__ = .; /* 这样可以在C代码内用extern const unsigned long long __CODE_TEXT_START__ */
        *(.text) 
        __CODE_TEXT_END__ = .; /* 同理 __CODE_TEXT_END__ */
    } 
    .data : { *(.data) }
    .data.string : { *(.data.string)}
    /* 在段块后面添加> RAM*/
    .bss : { *(.bss) } /* 
        > RAM 
        AT > FLASH 
        AT > 表示指定【加载地址】
        RAM
    */
    _end = .;
}