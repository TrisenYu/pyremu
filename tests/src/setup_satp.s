
.option norvc
.section .tohost, "aw", @progbits

.align 6
.globl tohost
tohost: .dword 0

.align 6
.globl fromhost
fromhost: .dword 0

// text节放程序入口
.section .text
.globl _start
_start:



.section .bss
.align 4
    stack_bottom:
    // 4KB 栈
    .skip 4096
    stack_top: