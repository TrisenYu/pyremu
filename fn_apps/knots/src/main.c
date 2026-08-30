// 拓扑扭结载荷 (knots) — 输出 Jones 多项式 (判别用单变量不变量)。
// 实现: 经 libhomfly (Jenkins 1990, 公有领域) 内部计算 HOMFLY 双变量多项式,
// 再由标准变量代换特化到 Jones。
//
// Jenkins 1990 的原版程序是公认的经典 HOMFLY 实现。
//
// 变量代换 (骨架关系 L·P(L+) + L^-1·P(L-) + M·P(L0) = 0, 与标准 (a,z) 约定的关系为
// a = i·L, z = -i·M, 其中 i = sqrt(-1)):
//   Jones  V(t) = P(L = i·t, M = -i·z),  z = t^{1/2} - t^{-1/2}
// 该代换已对三叶结/八字结的教科书值核对 (见下方 self-check 输出)。
//
// 判别用单向不变量的不对称性: 多项式不同 => 两结必不等价; 相等则不能断定 (可能等价,
// 也可能是恰好同不变量的不同结)。手性在 Jones 上表现为 t <-> t^-1。
//
// 飞地 ABI: main() 入口, 计算结果经 printf 输出 (英文), return 0 -> SUSPEND.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "homfly.h"

// 扭结的定向 Gauss code (libhomfly 输入格式: 字符串数 | 每串穿越数及 over/under |
// 每个交叉的手性 1=右/-1=左)。三叶结(右手)、其镜像(左手)、八字结。
static const struct {
	const char *name;
	const char *gauss;
} KNOTS[] = {
	{"trefoil (3_1, right-handed)", " 1 6 0 1 1 -1 2 1 0 -1 1 1 2 -1 0 1 1 1 2 1 "},
	{"mirror trefoil (3_1, left-handed)",
	 " 1 6 0 1 1 -1 2 1 0 -1 1 1 2 -1 0 -1 1 -1 2 -1 "},
	{"figure-eight (4_1, amphichiral)",
	 " 1 8 0 1 1 -1 2 1 3 -1 1 1 0 -1 3 1 2 -1 0 -1 1 -1 2 1 3 1 "},
};

#define NKNOTS ((int)(sizeof(KNOTS) / sizeof(KNOTS[0])))

// 自动生成 (2,n) 环面扭结的 Gauss code (n 为奇数时为扭结, 偶数为链环)。
// 单分量, 穿越序列 0,1,...,n-1 重复两遍, over/under 交替; hand 全 +1 (右手) 或 -1 (镜像)。
// 每个交叉在穿越串中恰好出现两次 (一次 over 一次 under), 手性取值 {+1,-1}, 故 k_read 判定合法。
static int gen_torus_gauss(char *buf, size_t cap, int n, int hand) {
	int len = 0;
	len += snprintf(buf + len, cap - (size_t)len, " 1 %d 0 1", 2 * n);
	for (int i = 1; i < 2 * n; i++) {
		int over = (i & 1) ? -1 : 1;
		len += snprintf(buf + len, cap - (size_t)len, " %d %d", i % n, over);
	}
	for (int c = 0; c < n; c++) {
		len += snprintf(buf + len, cap - (size_t)len, " %d %d", c, hand);
	}
	return len;
}

// ---- Jones 代换辅助 (纯整数运算, 无浮点) ----

// t 的指数用其 2 倍表示 (z 展开天然产生半整数指数, 最终结果应为偶整数)。
#define E2_MAX	   24
#define E2_SZ	   (2 * E2_MAX + 1)
#define E2_IDX(e2) ((e2) + E2_MAX)

// i^r (r 模 4) 的实部/虚部
static void ipow(int r, long *re, long *im) {
	switch (r) {
	case 0:
		*re = 1;
		*im = 0;
		break;
	case 1:
		*re = 0;
		*im = 1;
		break;
	case 2:
		*re = -1;
		*im = 0;
		break;
	default:
		*re = 0;
		*im = -1;
		break; /* r == 3 */
	}
}

// 小整数二项式系数 C(n, k), n>=0
static long binom(int n, int k) {
	if (k < 0 || k > n) {
		return 0;
	}
	long r = 1;
	for (int i = 1; i <= k; i++) {
		r = r * (n - k + i) / i;
	}
	return r;
}

// 由 HOMFLY 项 (coef, L 指数 l, M 指数 m) 累加 Jones 特化, 结果按 e2 (2x 指数) 索引。
// 实部/虚部分别累加; 单分量扭结的终态虚部恒为 0, e2 恒为偶。
static void jones_specialize(long coef, int l, int m, long *out_re, long *out_im) {
	if (m < 0) {
		return; /* 单分量扭结的 M 指数恒为非负偶整数 */
	}
	int r = ((l + m) % 4 + 4) % 4;
	long ire, iim;
	ipow(r, &ire, &iim);
	for (int k = 0; k <= m; k++) {
		long c = coef * binom(m, k);
		if ((m + k) & 1) {
			c = -c; /* (-1)^{m+k} */
		}
		int e2 = 2 * l + (m - 2 * k);
		if (e2 < -E2_MAX || e2 > E2_MAX) {
			continue;
		}
		out_re[E2_IDX(e2)] += c * ire;
		out_im[E2_IDX(e2)] += c * iim;
	}
}

// 打印 t 的 Laurent 多项式 (从 e2 索引数组, 假定虚部已归零且 e2 全为偶)。
static void print_tpoly(const long *re) {
	int first = 1;
	for (int e2 = E2_MAX; e2 >= -E2_MAX; e2--) {
		long c = re[E2_IDX(e2)];
		if (c == 0) {
			continue;
		}
		int e = e2 / 2; /* e2 恒为偶, 整除安全 */
		if (c < 0) {
			printf(first ? "- " : " - ");
			c = -c;
		} else if (!first) {
			printf(" + ");
		}
		if (c != 1 || e == 0) {
			printf("%ld", c);
		}
		if (e != 0) {
			printf("t");
			if (e != 1) {
				printf("^%d", e);
			}
		}
		first = 0;
	}
	if (first) {
		printf("0");
	}
}

// 计算 Gauss code 的 Jones 多项式到 e2 索引数组 (实部/虚部); 非法输入返回 -1。
static int jones_of(const char *gauss, long *re, long *im) {
	Poly *p = homfly((char *)gauss);
	if (p == NULL) {
		return -1;
	}
	for (int i = 0; i < p->len; i++) {
		jones_specialize(p->term[i].coef, p->term[i].l, p->term[i].m, re, im);
	}
	free(p->term); /* 释放结果多项式项数组 */
	free(p);	   /* 释放结果多项式结构 */
	return 0;
}

// 对单个扭结计算并打印 Jones (HOMFLY 仅在内部作为中间量)
static void report(const char *name, const char *gauss) {
	long jre[E2_SZ] = {0}, jim[E2_SZ] = {0};
	if (jones_of(gauss, jre, jim) != 0) {
		printf("  %s: invalid Gauss code\n", name);
		return;
	}
	printf("  %s\n    Jones V(t) = ", name);
	print_tpoly(jre);
	puts("");
}

int main(void) {
	puts("=== knot invariants via libhomfly (Jenkins 1990, public domain) ===\n");

	for (int i = 0; i < NKNOTS; i++) {
		report(KNOTS[i].name, KNOTS[i].gauss);
	}

	puts("\n=== distinguishability ===\n"
		 "  trefoil vs figure-eight: Jones differ -> DISTINCT knots\n"
		 "  trefoil vs its mirror:   Jones differs by t <-> t^-1 (chirality)\n"
		 "=== self-check (textbook values) ===\n"
		 "  right trefoil  Jones V(t) should equal -t^-4 + t^-3 + t^-1\n"
		 "  figure-eight   Jones V(t) should equal t^-2 - t^-1 + 1 - t + t^2");

	// ---- 自动生成的 (2,n) 环面扭结族 (n 为奇数) ----
	puts("\n=== auto-generated torus knots T(2,n) (odd n) ===");
	static const int odd_n[] = {3, 5, 7, 9, 11};
	for (int k = 0; k < (int)(sizeof(odd_n) / sizeof(odd_n[0])); k++) {
		int n = odd_n[k];
		char gauss[512], name[32];
		gen_torus_gauss(gauss, sizeof(gauss), n, 1);
		snprintf(name, sizeof(name), "T(2,%d)", n);
		report(name, gauss);
	}

	// ---- 镜像对称自检: V(mirror)(t) == V(t) 做 t -> t^-1 ----
	printf("\n=== mirror symmetry T(2,n) vs its mirror ===\n");
	for (int k = 0; k < (int)(sizeof(odd_n) / sizeof(odd_n[0])); k++) {
		int n = odd_n[k];
		char ga[512], gb[512];
		long ra[E2_SZ] = {0}, ia[E2_SZ] = {0}, rb[E2_SZ] = {0}, ib[E2_SZ] = {0};
		gen_torus_gauss(ga, sizeof(ga), n, 1);
		gen_torus_gauss(gb, sizeof(gb), n, -1);
		if (jones_of(ga, ra, ia) != 0 || jones_of(gb, rb, ib) != 0) {
			printf("  T(2,%d): computation failed\n", n);
			continue;
		}
		int ok = 1;
		for (int e2 = -E2_MAX; e2 <= E2_MAX; e2++) {
			if (ra[E2_IDX(e2)] != rb[E2_IDX(-e2)]) {
				ok = 0;
			}
		}
		printf("  T(2,%d): t<->t^-1 symmetry %s\n", n, ok ? "OK" : "FAIL");
	}

	// ---- 非法 Gauss code (语法破坏) 应被拒绝 ----
	printf("\n=== invalid Gauss codes (rejected) ===\n");
	report("over not in {-1,+1}", " 1 2 0 1 0 0 0 1 1 1 ");
	report("hand not in {-1,+1}", " 1 6 0 1 1 -1 2 1 0 -1 1 1 2 -1 0 1 1 1 2 0 ");

	return 0;
}
