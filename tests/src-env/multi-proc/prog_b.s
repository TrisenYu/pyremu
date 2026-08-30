// U 模式程序 B — 输出 'B' 200 次, 然后 exit(0).
// 用于抢占测试的最简样例.

.section .uprog.text, "ax", @progbits
.globl u_prog_b_entry

u_prog_b_entry:
    li   s0, 0                  // 计数器

1:  li   a0, 'B'
    li   a7, 2                  // uart_putc
    ecall

    addi s0, s0, 1
    li   s1, 200
    blt  s0, s1, 1b             // 未到 200 次, 继续

    // exit(0)
    li   a0, 0
    li   a7, 1
    ecall
    j    u_prog_b_entry
