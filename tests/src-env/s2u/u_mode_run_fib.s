// SPDX-LICENSE-IDENTIFIER: GPL2.0
// (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

// M → S → U 特权级切换 + Sv39 栈保护 + UART 输入 + 递归 Fibonacci.
//
// ═══════════════════════════════════════════════════════════════
//  架构概述
// ═══════════════════════════════════════════════════════════════
//
//  M 模式: PMP / medeleg / mstatus → MRET → S
//  S 模式: stvec / sscratch / sstatus → Sv39 页表 → SRET → U
//  U 模式: 读 UART → 约束 n → fib(n) → ECALL 报告
//
//  Sv39 identity 映射 (4 KiB 页):
//    VA 0x8000_0000–0x8000_FFFF → PA 同 (16 页, R+W+X+U, 代码/数据/BSS)
//    VA 0x1000_0000–0x1000_0FFF → PA 同 ( 1 页, R+W+U,    UART MMIO)
//    VA 0x8010_0000–0x8010_0FFF → PA 同 ( 1 页, R+W+U,    U 模式栈)
//    VA 0x800F_F000–0x800F_FFFF   (保护页, 未映射 → StorePageFault 15)
//
//  U 栈顶 = 0x8010_1000.  栈向下增长触及 0x800F_FFFF → trap → S 终止进程.
//
//  sscratch 交换: S 模式进入前将 sscratch 指向 S 栈顶;
//  trap handler 第一条指令用 csrrw sp, sscratch, sp 交换,
//  保证 S 模式始终有合法 sp (即使 U 模式 sp 已落入保护页).
//
// ═══════════════════════════════════════════════════════════════
//  UART 输入
// ═══════════════════════════════════════════════════════════════
//
//   uart_poll → ring buffer (64B) → uart_getc → readline → parse_int_simple

.section .text
.globl _start
.globl u_mode_bad

// ============================================================
//  常量
// ============================================================
.equ UART_BASE,     0x10000000
.equ UART_TX,       0x00
.equ UART_RXDATA,   0x04
.equ UART_TXCTRL,   0x08
.equ UART_IP,       0x14

.equ RBUF_SIZE,     64
.equ RBUF_MASK,     63
.equ LINE_MAX,      20

.equ STACK_TOP_U,   0x80101000   // U 模式栈顶 (独立于 BSS)
.equ PAGE_SHIFT,    12

// PTE flags
.equ PTE_V,    (1 << 0)
.equ PTE_RWXU, PTE_V | (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4)
.equ PTE_RWU,  PTE_V | (1 << 1) | (1 << 2) | (1 << 4)


// ============================================================
//  M 模式入口
// ============================================================
_start:
    la   sp, stack_top_m

    // UART 发送使能
    li   t0, UART_BASE + UART_TXCTRL
    li   t1, 1
    sw   t1, 0(t0)

    // PMP: NAPOT 全地址空间 R+W+X
    li   t0, 0x1F
    csrw pmpcfg0, t0
    li   t0, -1
    srli t0, t0, 10
    csrw pmpaddr0, t0

    // M trap 向量
    la   t0, m_trap_handler
    csrw mtvec, t0

    // 委派: ECALL + page faults 从 U/S 模式 → S 模式处理
    //   bits: 8 (ECALL-U) + 12 (InstrPF) + 13 (LdPF) + 15 (StPF)
    li   t0, 0xA100
    csrw medeleg, t0

    // mstatus: MPP=S, MPIE=1
    csrr t0, mstatus
    li   t1, ~(3 << 11)
    and  t0, t0, t1
    li   t1, (1 << 11) | (1 << 7)
    or   t0, t0, t1
    csrw mstatus, t0

    // → S 模式入口
    la   t0, s_mode_boot
    csrw mepc, t0
    mret


m_trap_handler:
    csrrw t0, mscratch, t0
    csrr t0, mepc
    addi t0, t0, 4
    csrw mepc, t0
    csrrw t0, mscratch, t0
    mret


// ============================================================
//  S 模式入口
// ============================================================
s_mode_boot:
    la   sp, stack_top_s
    addi fp, sp, 0

    // 配置 sscratch ← S 栈顶 (trap handler 入口用 csrrw 交换)
    la   t0, stack_top_s
    csrw sscratch, t0

    // S trap 向量
    la   t0, s_trap_handler
    csrw stvec, t0

    // sstatus: SPIE=1, SIE=1, SPP=U
    li   t0, (1 << 5)
    csrw sstatus, t0
    li   t0, (1 << 1)
    csrs sstatus, t0
    li   t0, (1 << 8)
    csrc sstatus, t0

    // 建立 Sv39 页表 (含 U 栈保护页)
    call setup_sv39

    // → U 模式入口
    la   t0, u_mode_main
    csrw sepc, t0

    la   a0, str_sboot
    call uart_puts

    sret


// ============================================================
//  setup_sv39 — Sv39 identity 映射
//
//  页表物理布局 (BSS, 4 KiB 对齐):
//    L1      @ page_tables + 0x0000
//    L2_hi   @ page_tables + 0x1000  (VPN[2]=2, 覆盖 0x80000000+)
//    L2_lo   @ page_tables + 0x2000  (VPN[2]=0, 覆盖 0x00000000+)
//    L3_main @ page_tables + 0x3000  (代码/数据/BSS)
//    L3_uart @ page_tables + 0x4000  (UART)
//
//  映射:
//    16 页 @ 0x80000–0x8000F  R+W+X+U  (代码/数据/BSS/栈)
//     1 页 @ 0x10000           R+W+U    (UART)
//     1 页 @ 0x80100           R+W+U    (U 模式栈)
//     1 页 @ 0x800FF           未映射    (保护页)
// ============================================================
setup_sv39:
    addi sp, sp, -32
    sd   ra,  0(sp)
    sd   s0,  8(sp)
    sd   s1, 16(sp)
    sd   s2, 24(sp)

    // 页表在 BSS 中, RAM 初始化为零, 无需显式清零.
    // 仅需覆写用到的条目, 其余 V=0 即未映射.
    la   s0, page_tables       // s0 = L1

    // -- L1[2] → L2_hi (VPN[2]=2, VA 0x8000_0000–0xBFFF_FFFF) --
    li   t0, 0x0000000040001001  // V=1, PPN=0x80004 (page_tables+0x1000)
    sd   t0, 16(s0)            // L1[2] = L1 + 16

    // -- L1[0] → L2_lo (VPN[2]=0, VA 0x0000_0000–0x3FFF_FFFF) --
    li   t0, 0x0000000040001401  // V=1, PPN=0x80005 (page_tables+0x2000)
    sd   t0, 0(s0)             // L1[0] = L1 + 0

    // -- L2_hi[0] → L3_main --
    li   t0, 0x1000
    add  s1, s0, t0            // s1 = L2_hi (+1 page)
    li   t0, 0x0000000040001801  // V=1, PPN=0x80006 (page_tables+0x3000)
    sd   t0, 0(s1)

    // -- L2_lo[128] → L3_uart (VPN[1]=128, VA 0x1000_0000) --
    li   t0, 0x2000
    add  s2, s0, t0            // s2 = L2_lo (+2 pages)
    li   t0, 0x0000000040001c01  // V=1, PPN=0x80007 (page_tables+0x4000)
    sd   t0, 1024(s2)          // L2_lo[128] = L2_lo + 128×8

    // -- L3_main: 16 页 identity 映射 (循环) --
    li   t0, 0x3000
    add  s1, s0, t0            // s1 = L3_main (+3 pages)
    li   t0, 0x000000004000001f  // 第 0 页 PTE (R+W+X+U, PPN=0x80000)
    li   t1, 16                // 16 页
1:
    sd   t0, 0(s1)
    addi s1, s1, 8
    li   t2, (1 << 10)         // 下一 PPN → PTE += 0x400
    add  t0, t0, t2
    addi t1, t1, -1
    bnez t1, 1b

    // -- L3_main[256] U 模式栈 (PPN=0x80100, R+W+U) --
    li   t0, 0x3000
    add  s1, s0, t0            // s1 = L3_main base (+3 pages)
    li   t0, 0x0000000040040017
    li   t2, 2048
    add  t2, s1, t2
    sd   t0, 0(t2)             // L3_main[256] = L3_main + 256×8

    // -- L3_uart[0] UART (PPN=0x10000, R+W+U) --
    li   t0, 0x4000
    add  s1, s0, t0            // s1 = L3_uart (+4 pages)
    li   t0, 0x0000000008000017
    sd   t0, 0(s1)

    // 启用 Sv39
    li   t0, 8                 // MODE = Sv39
    slli t0, t0, 60            // t0 = (8 << 60)
    li   t1, 0x80003           // PPN of L1 (= page_tables PA >> 12)
    or   t0, t0, t1            // satp = (Sv39 << 60) | root_ppn
    csrw satp, t0
    sfence.vma zero, zero

    ld   ra,  0(sp)
    ld   s0,  8(sp)
    ld   s1, 16(sp)
    ld   s2, 24(sp)
    addi sp, sp, 32
    ret


// ============================================================
//  S 模式 trap handler
//  入口: csrrw sp, sscratch, sp 交换栈指针
//        sp ← S 栈顶, sscratch ← 故障时的 U sp
// ============================================================
s_trap_handler:
    // 交换栈: sp ↔ sscratch
    csrrw sp, sscratch, sp     // sp = S 栈, sscratch = 旧 sp

    addi sp, sp, -96
    sd   ra,  0(sp)
    sd   fp,  8(sp)
    sd   a0, 16(sp)
    sd   a1, 24(sp)
    sd   a7, 32(sp)
    sd   t0, 40(sp)
    sd   t1, 48(sp)
    sd   s2, 56(sp)
    sd   s3, 64(sp)
    // 保存旧 sp (现在在 sscratch 中)
    csrr t0, sscratch
    sd   t0, 72(sp)
    addi fp, sp, 96

    csrr t0, scause
    li   t1, 8                  // ECALL from U-mode
    bne  t0, t1, s_trap_default

    // syscall 分发
    li   t1, 0                  // report fib
    beq  a7, t1, report_fib
    li   t1, 1                  // exit
    beq  a7, t1, terminate_proc

    li   a0, 255
    j    terminate_proc

s_trap_default:
    // 非 ECALL 异常: 打印 scause + sepc + 故障时 sp
    csrr s2, scause
    csrr s3, sepc
    ld   s4, 72(sp)            // 故障时的 U sp (之前保存的 sscratch)

    la   a0, str_fault
    call uart_puts
    mv   a0, s2
    call uart_putdec
    la   a0, str_at
    call uart_puts
    mv   a0, s3
    call uart_puthex
    la   a0, str_sp
    call uart_puts
    mv   a0, s4
    call uart_puthex
    li   a0, '\n'
    call uart_putc

    mv   a0, s2
    j    terminate_proc

report_fib:
    mv   s2, a0                 // fib result
    mv   s3, a1                 // n

    la   a0, str_fib
    call uart_puts
    mv   a0, s3
    call uart_putdec
    la   a0, str_eq
    call uart_puts
    mv   a0, s2
    call uart_putdec
    li   a0, '\n'
    call uart_putc

    la   t0, fib_result
    sd   s2, 0(t0)

    // sepc += 4 (skip ECALL)
    csrr t0, sepc
    addi t0, t0, 4
    csrw sepc, t0
    j    s_trap_done

terminate_proc:
    mv   s2, a0

    la   a0, str_exit
    call uart_puts
    mv   a0, s2
    call uart_putdec
    li   a0, '\n'
    call uart_putc

    // 重新导向到 S 空闲循环
    la   t0, s_mode_idle
    csrw sepc, t0
    li   t0, (1 << 8)
    csrs sstatus, t0           // SPP=S

s_trap_done:
    // 恢复 sscratch (指向 S 栈顶, 供下次 trap 使用)
    la   t0, stack_top_s
    csrw sscratch, t0

    ld   ra,  0(sp)
    ld   fp,  8(sp)
    ld   a0, 16(sp)
    ld   a1, 24(sp)
    ld   a7, 32(sp)
    ld   t0, 40(sp)
    ld   t1, 48(sp)
    ld   s2, 56(sp)
    ld   s3, 64(sp)
    ld   t0, 72(sp)            // 恢复故障时的 sp 到 sscratch (供 csrrw 交换)
    csrw sscratch, t0
    addi sp, sp, 96
    csrrw sp, sscratch, sp     // sp = 故障时的 sp, sscratch = S 栈顶
    sret


s_mode_idle:
    wfi
    j    s_mode_idle


// ============================================================
//  U 模式入口 (well-behaved) — 约束输入 → fib → 报告
// ============================================================
u_mode_main:
    li   sp, STACK_TOP_U
    addi fp, sp, 0

u_input_loop:
    la   a0, str_prompt
    call uart_puts

    la   a0, input_buf
    li   a1, LINE_MAX
    call readline

    la   a0, input_buf
    call parse_int_simple
    bnez a1, u_exit

    // well-behaved: 约束到 [1, 16]
    andi a0, a0, 0xF
    bnez a0, 1f
    li   a0, 1
1:
    mv   s4, a0

    call uart_putdec
    li   a0, '\n'
    call uart_putc

    mv   a0, s4
    call fib

    mv   a1, s4
    li   a7, 0
    ecall

    j    u_input_loop

u_exit:
    li   a0, 1
    li   a7, 1
    ecall
    j    u_exit


// ============================================================
//  U 模式入口 (pathological) — 不约束, 输入 0 → stack_bomb
// ============================================================
u_mode_bad:
    li   sp, STACK_TOP_U
    addi fp, sp, 0

    la   a0, str_bad_boot
    call uart_puts

    la   a0, str_prompt
    call uart_puts

    la   a0, input_buf
    li   a1, LINE_MAX
    call readline

    la   a0, input_buf
    call parse_int_simple
    bnez a1, u_bad_exit

    mv   s4, a0

    // 输入 0 → 栈溢出演示
    beqz s4, stack_bomb

    call uart_putdec
    li   a0, '\n'
    call uart_putc

    // 无约束 fib
    mv   a0, s4
    call fib

    mv   a1, s4
    li   a7, 0
    ecall

    j    u_mode_bad

u_bad_exit:
    li   a0, 1
    li   a7, 1
    ecall
    j    u_bad_exit


// ============================================================
//  stack_bomb — 分配超大栈帧, 立即触及保护页
//
//  U 栈仅 1 页 (0x8010_0000–0x8010_0FFF), sp 初值 0x8010_1000.
//  减去 0x2000 → sp=0x800F_F000 (保护页).
//  随后 sd → StorePageFault (scause=15) → S 终止进程.
// ============================================================
stack_bomb:
    addi sp, sp, -16
    sd   ra, 0(sp)

    la   a0, str_bomb
    call uart_puts

    // 重新加载: uart_puts 破坏 t0–t2
    li   t0, -0x2000
    add  sp, sp, t0
    sd   zero, 0(sp)            // ← StorePageFault

    // 不会到达此处
    li   t0, 0x2000
    add  sp, sp, t0
    ld   ra, 0(sp)
    addi sp, sp, 16
    ret


// ============================================================
//  fib — 递归 Fibonacci (标准帧指针)
//
//  栈帧 32 字节, fp = 旧 sp
//    fp-8  (sp+24): RA
//    fp-16 (sp+16): FP
//    fp-24 (sp+8):  s1 (n)
//    fp-32 (sp+0):  fib(n-1) 暂存
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
//  UART 发送
// ============================================================

uart_putc:
    li   t0, UART_BASE + UART_TX
    li   t2, 0x80000000
1:
    lw   t1, UART_TXCTRL(t0)
    and  t1, t1, t2
    bnez t1, 1b
    sb   a0, 0(t0)
    ret


// ============================================================
//  UART 接收
// ============================================================

uart_poll:
    addi sp, sp, -16
    sd   ra, 0(sp)
    li   a0, 0
    li   t2, 16
1:
    beqz t2, 2f
    li   t0, UART_BASE
    lw   t1, UART_IP(t0)
    andi t1, t1, 2
    beqz t1, 2f
    la   t0, rbuf_count
    lw   t1, 0(t0)
    li   t0, RBUF_SIZE
    bge  t1, t0, 2f
    li   t0, UART_BASE
    lbu  t1, UART_RXDATA(t0)
    mv   a1, t1
    call rbuf_put
    addi a0, a0, 1
    addi t2, t2, -1
    j    1b
2:
    ld   ra, 0(sp)
    addi sp, sp, 16
    ret


rbuf_put:
    la   t0, rbuf_count
    lw   t1, 0(t0)
    li   t2, RBUF_SIZE
    bge  t1, t2, rbuf_put_full
    la   t0, rbuf_head
    lw   t2, 0(t0)
    andi t2, t2, RBUF_MASK
    la   t0, rbuf
    add  t2, t0, t2
    sb   a1, 0(t2)
    la   t0, rbuf_head
    lw   t2, 0(t0)
    addi t2, t2, 1
    sw   t2, 0(t0)
    la   t0, rbuf_count
    lw   t2, 0(t0)
    addi t2, t2, 1
    sw   t2, 0(t0)
    li   a0, 0
    ret
rbuf_put_full:
    li   a0, 1
    ret


rbuf_get:
    la   t0, rbuf_count
    lw   t1, 0(t0)
    beqz t1, rbuf_get_empty
    la   t0, rbuf_tail
    lw   t2, 0(t0)
    andi t2, t2, RBUF_MASK
    la   t0, rbuf
    add  t2, t0, t2
    lbu  a0, 0(t2)
    la   t0, rbuf_tail
    lw   t2, 0(t0)
    addi t2, t2, 1
    sw   t2, 0(t0)
    la   t0, rbuf_count
    lw   t2, 0(t0)
    addi t2, t2, -1
    sw   t2, 0(t0)
    li   a1, 0
    ret
rbuf_get_empty:
    li   a0, 0
    li   a1, 1
    ret


uart_getc:
    addi sp, sp, -16
    sd   ra, 0(sp)
1:
    call rbuf_get
    beqz a1, 2f
    call uart_poll
    call rbuf_get
    beqz a1, 2f
    j    1b
2:
    ld   ra, 0(sp)
    addi sp, sp, 16
    ret


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
    call uart_getc
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
    call uart_putc
    j    1b
readline_bs:
    ble  s1, s0, 1b
    addi s1, s1, -1
    li   a0, 0x08
    call uart_putc
    li   a0, ' '
    call uart_putc
    li   a0, 0x08
    call uart_putc
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
//  输出辅助
// ============================================================

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


uart_putdec:
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
    call uart_puts
    ld   ra,  0(sp)
    ld   s0,  8(sp)
    ld   s1, 16(sp)
    addi sp, sp, 48
    ret


uart_puthex:
    addi sp, sp, -32
    sd   ra,  0(sp)
    sd   s0,  8(sp)
    sd   s1, 16(sp)
    sd   s2, 24(sp)
    mv   s0, a0
    li   a0, '0'
    call uart_putc
    li   a0, 'x'
    call uart_putc
    li   s1, 60
1:
    srl  s2, s0, s1
    andi s2, s2, 0xF
    addi s2, s2, '0'
    li   t0, '9'
    ble  s2, t0, 2f
    addi s2, s2, 'a' - '0' - 10
2:
    mv   a0, s2
    call uart_putc
    addi s1, s1, -4
    bgez s1, 1b
    ld   ra,  0(sp)
    ld   s0,  8(sp)
    ld   s1, 16(sp)
    ld   s2, 24(sp)
    addi sp, sp, 32
    ret


// ============================================================
//  数据段
// ============================================================
.section .data
.align 3
fib_result:
    .dword 0

.section .rodata
.align 2
str_sboot:
    .asciz "S-mode boot (Sv39 guard page enabled)\n"

str_prompt:
    .asciz "Enter n (1-16): "

str_fib:
    .asciz "fib("

str_eq:
    .asciz ")="

str_exit:
    .asciz "Process terminated, exit code: "

str_fault:
    .asciz "Fault caught by S-mode, scause="

str_at:
    .asciz " @ sepc="

str_sp:
    .asciz " sp="

str_bad_boot:
    .asciz "S-mode: UNTRUSTED U-mode proc (no bounds check, guard page active)\n"

str_bomb:
    .asciz "Stack overflow via guard page...\n"


// ============================================================
//  BSS — 页表 + ring buffer + input buffer + M/S 栈
// ============================================================
.section .bss

// Sv39 页表 (5 × 4 KiB, 页对齐)
.align 12
page_tables:
    .skip 5 * 4096

// Ring buffer
.align 4
rbuf:
    .skip RBUF_SIZE
rbuf_head:
    .skip 4
rbuf_tail:
    .skip 4
rbuf_count:
    .skip 4

// Line buffer
input_buf:
    .skip LINE_MAX

// M / S 模式栈 (U 栈独立于 0x8010_1000)
.align 4
    .skip 4096
stack_top_m:
    .skip 4096
stack_top_s:
