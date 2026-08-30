// SPDX-LICENSE-IDENTIFIER: GPL2.0
// (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

// U 模式 Fibonacci 用户程序.
//
// 通过 ECALL 进行所有 I/O (无直接 MMIO 访问).
// Syscall 约定 (ECALL, a7 = number):
//   0 = report fib:  a0 = fib 结果, a1 = n
//   1 = exit:        a0 = exit code
//   2 = uart_putc:   a0 = char
//   3 = uart_puts:   a0 = string ptr
//   4 = uart_getc:   返回 a0 = char
//
// 导出: u_fib_entry — 内核调度器通过 sepc 跳转至此.

.section .uprog.text, "ax", @progbits
.globl u_fib_entry

.equ LINE_MAX, 20

// ============================================================
//  入口 — 无限循环: 读输入, 约束, fib, 报告
// ============================================================
u_fib_entry:
    // sp 由内核在 PCB 中设置, 此处不修改

    // 提示
    la   a0, str_prompt
    li   a7, 3                 // uart_puts
    ecall

1:
    // 读一行
    la   a0, input_buf
    li   a1, LINE_MAX
    call readline
    // readline 返回 a0 = 行长度; 0 = 输入流耗尽 (EOF), 正常退出
    // (配合 loader 的 run(timeout=0): 进程必须终止, 内核才能 stop_machine 停机)
    beqz a0, fib_eof

    // 解析
    la   a0, input_buf
    call parse_int_simple
    bnez a1, fib_exit

    // 约束到 [1, 16]
    andi a0, a0, 0xF
    bnez a0, 2f
    li   a0, 1
2:
    mv   s4, a0

    // 回显 n
    call uart_putdec_ecall
    li   a0, '\n'
    li   a7, 2                 // uart_putc
    ecall

    // fib(n)
    mv   a0, s4
    call fib

    // 报告
    mv   a1, s4
    li   a7, 0                 // report fib
    ecall

    j    1b

fib_exit:
    li   a0, 1
    li   a7, 1                 // exit
    ecall
    j    fib_exit

// ============================================================
//  fib_eof — 输入流耗尽: 正常结束进程 (exit 0)
// ============================================================
fib_eof:
    li   a0, 0
    li   a7, 1                 // exit
    ecall
    j    fib_eof


// ============================================================
//  fib — 递归 Fibonacci
// ============================================================
fib:
    addi sp, sp, -32
    sd   ra, 24(sp)
    sd   fp, 16(sp)
    sd   s1, 8(sp)
    addi fp, sp, 32

    mv   s1, a0
    li   t0, 1
    ble  s1, t0, fib_base

    addi a0, s1, -1
    call fib
    sd   a0, 0(sp)

    addi a0, s1, -2
    call fib
    ld   t0, 0(sp)
    add  a0, a0, t0

    j    fib_ret

fib_base:

fib_ret:
    ld   ra, 24(sp)
    ld   fp, 16(sp)
    ld   s1, 8(sp)
    addi sp, sp, 32
    ret


// ============================================================
//  readline — 行缓冲 (ECALL uart_getc / uart_putc)
// ============================================================
readline:
    addi sp, sp, -32
    sd   ra,  0(sp)
    sd   s0,  8(sp)
    sd   s1, 16(sp)
    sd   s2, 24(sp)
    mv   s0, a0
    addi s2, a0, -1
    add  s2, s2, a1
    mv   s1, a0
1:
    // uart_getc via ECALL
    li   a7, 4
    ecall
    // a0 = char (0 = 无数据, 输入流已耗尽)
    // 视为 EOF: 结束本行 (s1 未前移, 返回长度为 0), 由主循环 exit(0)
    beqz a0, readline_done

    li   t0, '\n'
    beq  a0, t0, readline_done
    li   t0, '\r'
    beq  a0, t0, readline_done
    li   t0, 0x08
    beq  a0, t0, readline_bs
    li   t0, 0x7F
    beq  a0, t0, readline_bs
    li   t0, ' '
    blt  a0, t0, 1b
    bge  s1, s2, 1b
    sb   a0, 0(s1)
    addi s1, s1, 1
    // echo via ECALL
    li   a7, 2
    ecall
    j    1b
readline_bs:
    ble  s1, s0, 1b
    addi s1, s1, -1
    li   a0, 0x08
    li   a7, 2
    ecall
    li   a0, ' '
    li   a7, 2
    ecall
    li   a0, 0x08
    li   a7, 2
    ecall
    j    1b
readline_done:
    sb   zero, 0(s1)
    sub  a0, s1, s0
    ld   ra,  0(sp)
    ld   s0,  8(sp)
    ld   s1, 16(sp)
    ld   s2, 24(sp)
    addi sp, sp, 32
    ret


// ============================================================
//  parse_int_simple
// ============================================================
parse_int_simple:
    li   t2, 0
    li   t3, 0
    mv   t4, a0
1:
    lbu  t0, 0(t4)
    addi t4, t4, 1
    beqz t0, 3f
    li   t1, '0'
    blt  t0, t1, 1b
    li   t1, '9'
    bgt  t0, t1, 1b
    addi t3, t3, 1
    li   t1, 10
    mul  t2, t2, t1
    addi t0, t0, -'0'
    add  t2, t2, t0
    j    1b
3:
    beqz t3, parse_fail
    mv   a0, t2
    li   a1, 0
    ret
parse_fail:
    li   a0, 0
    li   a1, 1
    ret


// ============================================================
//  uart_putdec_ecall — 通过 ECALL uart_putc 输出十进制
// ============================================================
uart_putdec_ecall:
    addi sp, sp, -48
    sd   ra,  0(sp)
    sd   s0,  8(sp)
    sd   s1, 16(sp)
    mv   s0, a0
    addi s1, sp, 40
    sb   zero, 0(s1)
    addi s1, s1, -1
    bnez s0, putdec_loop
    li   t0, '0'
    sb   t0, 0(s1)
    addi s1, s1, -1
    j    putdec_out
putdec_loop:
    li   t0, 10
    divu t1, s0, t0
    remu t2, s0, t0
    addi t2, t2, '0'
    sb   t2, 0(s1)
    addi s1, s1, -1
    mv   s0, t1
    bnez s0, putdec_loop
putdec_out:
    addi a0, s1, 1
    li   a7, 3               // uart_puts via ECALL
    ecall
    ld   ra,  0(sp)
    ld   s0,  8(sp)
    ld   s1, 16(sp)
    addi sp, sp, 48
    ret


// ============================================================
//  数据
// ============================================================
.section .uprog.rodata, "a", @progbits
.align 2
str_prompt:
    .asciz "fib> "

.section .uprog.bss, "aw", @nobits
input_buf:
    .skip LINE_MAX
