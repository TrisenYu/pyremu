// SPDX-LICENSE-IDENTIFIER: GPL2.0
// (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

// U 模式 n-Queens — 基于 src-alg/n-queen.cc 改写为固定数组, 无 STL / 无 libc.
//
// I/O 通过 ecall_wrappers.s 中定义的汇编函数 (ecall_putc/ecall_puts 等),
// 避免 C 内联 asm 的寄存器分配歧义.

#define MAX_N 6

// ---- 外部 ECALL 函数 (ecall_wrappers.s) ----
extern void ecall_putc(char c);
extern void ecall_puts(const char *s);
extern void ecall_putdec(int v);
extern void ecall_exit(int code);

// ---- 棋盘状态 ----
static int n, solutions, rec_x[16], rec_y[16];
static unsigned int row_mask, col_mask, xl_mask, xr_mask;

// ---- 位掩码辅助 ----
static int check_pos(int pos, unsigned int v) {
	return (v >> pos) & 1;
}
static void set_pos(int pos, unsigned int *v) {
	*v |= (1u << pos);
}
static void unset_pos(int pos, unsigned int *v) {
	*v &= ~(1u << pos);
}

// ---- DFS (与原版 n-queen.cc 一致) ----
static void dfs(int pos, int num) {
	if (num >= n) {
		solutions++;
		return;
	}
	for (int curr = pos; curr < n * n; curr++) {
		int i = curr / n, j = curr % n;
		if (check_pos(i, row_mask) || check_pos(j, col_mask) || check_pos(i + j, xl_mask)
			|| check_pos(i - j + n, xr_mask)) {
			continue;
		}
		set_pos(i, &row_mask);
		set_pos(j, &col_mask);
		set_pos(i + j, &xl_mask);
		set_pos(i - j + n, &xr_mask);
		rec_x[num] = i;
		rec_y[num] = j;
		dfs(curr + 1, num + 1);
		unset_pos(i, &row_mask);
		unset_pos(j, &col_mask);
		unset_pos(i + j, &xl_mask);
		unset_pos(i - j + n, &xr_mask);
	}
}

// ---- 调度器入口 ----
__attribute__((used)) void u_nqueen_entry(void) {
	for (int qn = 1; qn <= 4; qn++) {
		n		  = qn;
		solutions = 0;
		row_mask = col_mask = xl_mask = xr_mask = 0;
		ecall_puts("nq(");
		ecall_putdec(qn);
		ecall_puts(")=");
		dfs(0, 0);
		ecall_putdec(solutions);
		ecall_puts("\n");
	}
	ecall_exit(0);
}
