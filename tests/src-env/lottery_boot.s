// SPDX-LICENSE-IDENTIFIER: GPL2.0
// OpenSBI 风格彩票启动 — 多 hart 竞争互斥锁 (AMOSWAP)
//
//   冷启动 hart (抢到锁):
//     1. 打印 "[hart N] cold boot"
//     2. 设置 boot_ready = 1
//     3. 释放锁
//
//   热启动 hart (未抢到锁):
//     1. 自旋等待 boot_ready
//     2. 打印 "[hart N] warm boot"
//
//   所有 hart 最终打印 "[hart N] running" 后 WFI.

.section .text
.globl _start

// ============================================================
//  常量
// ============================================================
.equ UART_BASE,      0x10000000
.equ UART_TXDATA,    0x00
.equ UART_TXCTRL,    0x08

// ============================================================
//  _start — 彩票启动主入口 (必须在 .text 最前面以保证正确对齐)
// ============================================================
_start:
    // 启用 UART TX
    li   t0, UART_BASE + UART_TXCTRL
    li   t1, 1
    sw   t1, 0(t0)

    // 每个 hart 有自己的栈 (sp = 0x80010000 + hart_id * 4096)
    csrr t0, mhartid
    slli t0, t0, 12
    li   t1, 0x80010000
    add  sp, t1, t0

    // ========================================================
    //  彩票锁: AMOSWAP 原子获取 boot_lock
    //  公平性由仿真器保证 — hart 执行顺序每周期随机打乱.
    // ========================================================
    la   t0, boot_lock
    li   t1, 1
    amoswap.w t2, t1, (t0)       // t2 = 旧值, [t0] = 1
    bnez t2, warm_boot            // 锁已被占 -> 热启动

    // ---- 冷启动路径 ----
    la   a0, str_cold
    call uart_puts

    // 设置 boot_ready = 1 (通知热启动 hart)
    la   t0, boot_ready
    li   t1, 1
    sw   t1, 0(t0)

    // 释放锁
    la   t0, boot_lock
    sw   zero, 0(t0)

    j    continue_boot

    // ---- 热启动路径 ----
warm_boot:
    // 自旋等待 boot_ready
    la   t0, boot_ready
1:
    lw   t1, 0(t0)
    beqz t1, 1b

    la   a0, str_warm
    call uart_puts

    // ---- 公共入口 ----
continue_boot:
    la   a0, str_running
    call uart_puts

idle:
    wfi
    j    idle


// ============================================================
//  UART 辅助 (必须在 _start 之后, 避免影响入口对齐)
// ============================================================

// uart_putc — 发送单个字符 (a0 = char)
uart_putc:
    li   t0, UART_BASE
    li   t2, 0x80000000
1:
    lw   t1, UART_TXCTRL(t0)
    and  t1, t1, t2
    bnez t1, 1b
    sb   a0, UART_TXDATA(t0)
    ret

// uart_puts — 发送字符串 (a0 = ptr)
uart_puts:
    addi sp, sp, -16
    sd   ra, 0(sp)
    sd   s0, 8(sp)
    mv   s0, a0
1:
    lbu  a0, 0(s0)
    beqz a0, 2f
    call uart_putc
    addi s0, s0, 1
    j    1b
2:
    ld   ra, 0(sp)
    ld   s0, 8(sp)
    addi sp, sp, 16
    ret

// uart_putdec — 输出整数十进制 (a0 = value)
uart_putdec:
    addi sp, sp, -48
    sd   ra, 0(sp)
    sd   s0, 8(sp)
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
    call uart_puts
    ld   ra, 0(sp)
    ld   s0, 8(sp)
    ld   s1, 16(sp)
    addi sp, sp, 48
    ret

// print_hart_id — 输出 "[hart N] "
print_hart_id:
    addi sp, sp, -16
    sd   ra, 0(sp)

    li   a0, '['
    call uart_putc
    li   a0, 'h'
    call uart_putc
    li   a0, 'a'
    call uart_putc
    li   a0, 'r'
    call uart_putc
    li   a0, 't'
    call uart_putc
    li   a0, ' '
    call uart_putc

    csrr a0, mhartid
    call uart_putdec

    li   a0, ']'
    call uart_putc
    li   a0, ' '
    call uart_putc

    ld   ra, 0(sp)
    addi sp, sp, 16
    ret


// ============================================================
//  BSS 共享变量
// ============================================================
.section .bss
.align 4
boot_lock:
    .skip 4
boot_ready:
    .skip 4

// ============================================================
//  字符串
// ============================================================
.section .rodata
.align 2
str_cold:
    .asciz "cold boot\n"
str_warm:
    .asciz "warm boot\n"
str_running:
    .asciz "running\n"
