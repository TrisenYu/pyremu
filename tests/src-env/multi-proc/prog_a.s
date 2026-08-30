// U 模式程序 A — 输出 'A' 100 次, 然后 exit(0).
// 用于抢占测试的最简样例.

.section .uprog.text, "ax", @progbits
.globl u_prog_a_entry

u_prog_a_entry:
    li   s0, 0                  // 计数器
    li   s1, 100                // 上限 (callee-saved, 不会被 ecall 破坏)

1:  li   a0, 'A'
    li   a7, 2                  // uart_putc
    ecall

    addi s0, s0, 1
    blt  s0, s1, 1b             // 未到 100 次, 继续

    // exit(0)
    li   a0, 0
    li   a7, 1
    ecall
    j    u_prog_a_entry
