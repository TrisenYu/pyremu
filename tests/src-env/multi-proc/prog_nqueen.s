// SPDX-LICENSE-IDENTIFIER: GPL2.0
// (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

// U 模式 n-Queens — 递归位掩码回溯.
// I/O 通过 ECALL (a7=2 putc, a7=3 puts, a7=1 exit).
// 导出: u_nqueen_entry

.section .text
.globl u_nqueen_entry

// ============================================================
//  入口
// ============================================================
u_nqueen_entry:

    // 对 n=1..4 求解
    li   s4, 1                 // s4 = n

1:
    // nqueen_solve(n) -> a0 = solutions
    mv   a0, s4
    call nqueen_solve
    mv   s5, a0                // s5 = solutions

    // 输出 "nq(N)=S\n"
    la   a0, str_nq
    li   a7, 3
    ecall
    mv   a0, s4
    call putdec
    la   a0, str_eq
    li   a7, 3
    ecall
    mv   a0, s5
    call putdec
    li   a0, '\n'
    li   a7, 2
    ecall

    addi s4, s4, 1
    li   t0, 5
    blt  s4, t0, 1b

    // exit(0)
    li   a0, 0
    li   a7, 1
    ecall
    j    u_nqueen_entry


// ============================================================
//  nqueen_solve(n) -> solutions
// ============================================================
nqueen_solve:
    addi sp, sp, -16
    sd   ra, 0(sp)

    // 初始化全局
    la   t0, nq_solutions
    sw   zero, 0(t0)
    la   t0, nq_n
    sw   a0, 0(t0)

    // dfs(row=0, col=0, d1=0, d2=0)
    li   a0, 0
    li   a1, 0
    li   a2, 0
    li   a3, 0
    call dfs

    // 返回 solutions
    la   t0, nq_solutions
    lw   a0, 0(t0)

    ld   ra, 0(sp)
    addi sp, sp, 16
    ret


// ============================================================
//  dfs(row, col_mask, d1_mask, d2_mask)
//
//  d1 = 主对角线 (row+col), d2 = 副对角线 (row-col+n)
//  每次放置后: d1 不变但 row+1 意味着下一行检查时 d1 的位已自动右移
//  实际上对于每一列 col_idx: 检查 col|d1|d2 的对应位
//
//  简化: 每行迭代所有列
// ============================================================
dfs:
    addi sp, sp, -56
    sd   ra, 48(sp)
    sd   s0, 40(sp)
    sd   s1, 32(sp)
    sd   s2, 24(sp)
    sd   s3, 16(sp)
    sd   s4, 8(sp)
    sd   s5, 0(sp)

    mv   s0, a0                // row
    mv   s1, a1                // col_mask
    mv   s2, a2                // d1_mask
    mv   s3, a3                // d2_mask

    // load n into s5 (callee-saved, survives recursion)
    la   t0, nq_n
    lw   s5, 0(t0)

    // if row == n: solutions++; return
    bne  s0, s5, dfs_try_cols

    la   t0, nq_solutions
    lw   t1, 0(t0)
    addi t1, t1, 1
    sw   t1, 0(t0)
    j    dfs_ret

dfs_try_cols:
    li   s4, 0                 // col_idx (callee-saved)

1:
    bge  s4, s5, dfs_ret       // col_idx >= n -> 返回

    li   t1, 1
    sll  t1, t1, s4            // bit = 1 << col_idx

    // 冲突检查: (col|d1|d2) & bit
    or   t2, s1, s2
    or   t2, t2, s3
    and  t2, t2, t1
    bnez t2, dfs_next_col

    // 放置 + 递归
    addi a0, s0, 1
    or   a1, s1, t1
    or   t2, s2, t1
    srli a2, t2, 1
    or   t3, s3, t1
    slli a3, t3, 1
    call dfs

dfs_next_col:
    addi s4, s4, 1
    j    1b

dfs_ret:
    ld   ra, 48(sp)
    ld   s0, 40(sp)
    ld   s1, 32(sp)
    ld   s2, 24(sp)
    ld   s3, 16(sp)
    ld   s4, 8(sp)
    ld   s5, 0(sp)
    addi sp, sp, 56
    ret


// ============================================================
//  putdec(v) — 通过 ECALL 输出十进制
// ============================================================
putdec:
    addi sp, sp, -32
    sd   ra,  0(sp)
    sd   s0,  8(sp)
    sd   s1, 16(sp)

    mv   s0, a0
    li   s1, 10

    blt  s0, s1, 1f
    div  a0, s0, s1
    call putdec

1:
    rem  a0, s0, s1
    addi a0, a0, '0'
    li   a7, 2
    ecall

    ld   ra,  0(sp)
    ld   s0,  8(sp)
    ld   s1, 16(sp)
    addi sp, sp, 32
    ret


// ============================================================
//  数据
// ============================================================
.section .data
.align 2
nq_n:
    .word 0
nq_solutions:
    .word 0

.section .rodata
.align 2
str_nq:
    .asciz "nq("
str_eq:
    .asciz ")="
