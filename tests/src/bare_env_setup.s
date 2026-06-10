// 这一选项禁用压缩扩展
// .option norvc
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
    // 设置栈顶指针
    la      sp, stack_top
    la      t0, m_mode_interrupt_dispatcher
    csrw    mtvec, t0 // 设置异常基向量地址
    csrr    t0, mstatus
    li      t1, (0x3<<11)
    // 配置mstatus.MPP为M模式
    or      t0, t0, t1
    csrw    mstatus, t0
    nop
    li a0, 0x5
    li a1, 0
    li a2, 0
    li a3, 0
    li a4, 0
    li a5, 0
    li a6, 0x1234
    li a7, 0x2
    ecall
halt_label:
    mfence.did
    fence.i
    j halt_label


// 中断处理函数入口
m_mode_interrupt_dispatcher:
    addi    sp, sp, -56
    sw      ra,  0(sp)
    sw      s0,  4(sp)
    sw      s1,  8(sp)
    sw      s2, 12(sp)
    sw      s3, 16(sp)
    sw      s4, 20(sp)
    sw      s5, 24(sp)
    sw      s6, 28(sp)
    sw      s7, 32(sp)
    sw      s8, 36(sp)
    sw      s9, 40(sp)
    sw      s10,44(sp)
    sw      s11,48(sp)
    // 获取中断原因
    csrr    t0, mcause
    // 关闭中断
    csrr    t3, mie
    csrw    mie, zero

    // 用户态 ecall
    li      t1, 0x8
    beq     t0, t1, baby_ecall 

    // supervisior-mode ecall
    addi    t1, t1, 1
    beq     t0, t1, baby_ecall 

    // machine-mode ecall
    addi    t1, t1, 2
    beq     t0, t1, baby_ecall 
// 具体中断处理
baby_ecall:
    la t0, hello_str
keep_print:
    lbu t1, 0(t0)
    beq t1, zero, baby_ecall_done
    li t2, 0x10000000 // uart 串口地址
    sb t1, 0(t2)
    addi t0, t0, 1
    j keep_print
baby_ecall_done:
    csrr t0, mepc
    addi t0, t0, 4
    csrw mepc, t0
    lw ra,  0(sp)
    lw s0,  4(sp)
    lw s1,  8(sp)
    lw s2, 12(sp)
    lw s3, 16(sp)
    lw s4, 20(sp)
    lw s5, 24(sp)
    lw s6, 28(sp)
    lw s7, 32(sp)
    lw s8, 36(sp)
    lw s9, 40(sp)
    lw s10,44(sp)
    lw s11,48(sp)
    addi sp, sp, 56
    csrw mie, t3
    mret // 回到设置mepc之后的地址

.section .bss
.align 4
    stack_bottom:
    // 4KB 栈
    .skip 4096
    stack_top:

.section .rodata
.align 2
hello_str:
    .string "Hello World\n"