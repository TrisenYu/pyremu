// SPDX-LICENSE-IDENTIFIER: GPL2.0
// (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

// M 模式 → S 模式启动移交: 模拟 OpenSBI 将控制权移交给操作系统的 boot 片段.
//
// 流程:
//   1. M 模式初始化 UART, 保存 mstatus 到共享内存, 设置 mtvec 和 mstatus.MPP=S
//   2. MRET 切换到 S 模式
//   3. S 模式输出 mstatus 的完整值 (16 进制) 和 "Hello World!\n"
//   4. 执行 WFI 后进入死循环
//
// 验证方式:
//   python -m pyremu.debugger tests/bins/elf/m_mode_to_s_mode.elf
//   命令: r 100  然后检查串口输出

.section .text
.globl _start

// ============================================================
//  UART 基址 (qemu_virt 默认)
// ============================================================
.equ UART_BASE,    0x10000000
.equ UART_TX,      0x00
.equ UART_TXCTRL,  0x08

// ============================================================
//  M 模式入口 — _start
// ============================================================
_start:
    // 设置栈指针 (M 模式用)
    la   sp, stack_top_m

    // -- 使能 UART 发送 --
    li   t0, UART_BASE + UART_TXCTRL
    li   t1, 1                // txen = 1
    sw   t1, 0(t0)

    // -- 配置 PMP: NAPOT 放行整个地址空间 (R+W+X) --
    //  PMP 默认拒绝所有 S/U 模式内存访问; 必须至少配置一条条目.
    //  pmpaddr0 = 全 1 (NAPOT 覆盖 2^64 整个空间, 54-bit 地址字段)
    //  pmpcfg0  = 0x1B (A=NAPOT, R, W, X)
    li   t0, 0x1B
    csrw pmpcfg0, t0
    li   t0, -1               // 全 1 = NAPOT 编码覆盖整个地址空间
    srli t0, t0, 10           // 取低 54 位 (pmpaddr 字段宽度)
    csrw pmpaddr0, t0

    // -- 设置 M 模式 trap 向量 --
    la   t0, m_trap_handler
    csrw mtvec, t0

    // -- 配置 mstatus: MPP=S, MPIE=1 --
    //  MPP 位于 bits 12:11, S 模式编码为 1
    //  MPIE 位于 bit 7, 置 1 以便 mret 后 S 模式中断使能
    csrr t0, mstatus
    li   t1, ~(3 << 11)
    and  t0, t0, t1          // 清零 MPP
    li   t1, (1 << 11) | (1 << 7)
    or   t0, t0, t1          // MPP=S, MPIE=1
    csrw mstatus, t0

    // -- 将配置后的 mstatus 值存入共享内存, 供 S 模式读取 --
    la   t0, saved_mstatus
    csrr t1, mstatus
    sd   t1, 0(t0)

    // -- 设置 S 模式入口地址 --
    la   t0, s_mode_boot
    csrw mepc, t0

    // MRET: PC ← mepc, mode ← MPP (S), MIE ← MPIE
    mret


// ============================================================
//  M 模式 trap 处理 — 简洁: 跳过故障指令
// ============================================================
m_trap_handler:
    csrr t0, mepc
    addi t0, t0, 4
    csrw mepc, t0
    mret


// ============================================================
//  S 模式入口 (模拟 OS boot)
// ============================================================
s_mode_boot:
    // 设置 S 模式自己的栈
    la   sp, stack_top_s

    // -- 输出 "mstatus=0x" --
    la   a0, str_mstatus
    call uart_puts

    // -- 读取 saved_mstatus 并以 hex 输出 --
    la   t0, saved_mstatus
    ld   a0, 0(t0)
    call uart_puthex64

    // -- 输出换行 --
    li   a0, '\n'
    call uart_putc

    // -- 输出 "Hello World!\n" --
    la   a0, str_hello
    call uart_puts

    // -- WFI 等待中断; 若立即唤醒则死循环 --
    wfi
spin:
    j    spin


// ============================================================
//  UART 子程序
// ============================================================

// uart_putc — 通过 UART 发送一个字符
//   a0: 要发送的字符 (低 8 位有效)
//   不修改任何寄存器 ( caller-saved 除外)
uart_putc:
    li   t0, UART_BASE + UART_TX
    sb   a0, 0(t0)
    ret


// uart_puts — 通过 UART 发送以 null 结尾的字符串
//   a0: 字符串首地址
uart_puts:
    addi sp, sp, -16
    sd   ra, 0(sp)
    sd   s0, 8(sp)
    mv   s0, a0              // s0 = 字符串指针
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


// uart_puthex64 — 以 16 进制格式输出 64-bit 值 (16 个 hex 数字)
//   a0: 要输出的值
uart_puthex64:
    addi sp, sp, -32
    sd   ra, 0(sp)
    sd   s0, 8(sp)
    sd   s1, 16(sp)
    mv   s0, a0              // s0 = 待输出的值
    li   s1, 60              // s1 = 当前位移 (从最高 nibble 开始)
1:
    srl  a0, s0, s1          // 右移得当前 nibble
    andi a0, a0, 0xF
    // 转换为 ASCII hex 字符
    li   t0, 10
    blt  a0, t0, 2f          // a0 < 10 → 数字
    addi a0, a0, 'a' - 10    // a-f
    j    3f
2:
    addi a0, a0, '0'         // 0-9
3:
    call uart_putc
    addi s1, s1, -4
    bgez s1, 1b              // s1 >= 0 继续
    ld   ra, 0(sp)
    ld   s0, 8(sp)
    ld   s1, 16(sp)
    addi sp, sp, 32
    ret


// ============================================================
//  数据段
// ============================================================
.section .data

// 共享内存: M 模式在此保存 mstatus, S 模式从此读取
.align 3
saved_mstatus:
    .dword 0

.section .rodata

.align 2
str_mstatus:
    .asciz "mstatus=0x"

str_hello:
    .asciz "Hello World!\n"


// ============================================================
//  BSS — 栈空间
// ============================================================
.section .bss
.align 4

// M 模式栈 (2 KiB)
.skip 2048
stack_top_m:

// S 模式栈 (2 KiB)
.skip 2048
stack_top_s:
