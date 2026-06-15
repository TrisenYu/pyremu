// SPDX-LICENSE-IDENTIFIER: GPL2.0
// (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

// ECALL wrapper — 供 C 代码调用的汇编函数.
// 避免 C 内联 asm 的寄存器分配歧义.

.section .text

.globl ecall_putc
.globl ecall_puts
.globl ecall_putdec
.globl ecall_exit

// ecall_putc(char c) — 通过 ECALL a7=2 输出一个字符
ecall_putc:
    li   a7, 2
    ecall
    ret

// ecall_puts(const char *s) — 通过 ECALL a7=3 输出字符串
ecall_puts:
    li   a7, 3
    ecall
    ret

// ecall_putdec(int v) — 通过 ECALL a7=2 输出十进制整数
//   递归实现, 复用 ecall_putc
ecall_putdec:
    addi sp, sp, -32
    sd   ra,  0(sp)
    sd   s0,  8(sp)
    sd   s1, 16(sp)

    mv   s0, a0                 // s0 = v
    li   s1, 10

    // if v >= 10: ecall_putdec(v / 10)
    blt  s0, s1, putdec_leaf
    div  a0, s0, s1
    call ecall_putdec

putdec_leaf:
    // ecall_putc('0' + v % 10)
    rem  a0, s0, s1
    addi a0, a0, '0'
    call ecall_putc

    ld   ra,  0(sp)
    ld   s0,  8(sp)
    ld   s1, 16(sp)
    addi sp, sp, 32
    ret

// ecall_exit(int code) — 通过 ECALL a7=1 退出进程
ecall_exit:
    li   a7, 1
    ecall
    ret
