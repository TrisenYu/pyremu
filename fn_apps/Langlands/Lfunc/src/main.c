/*
 * Lfunc — Riemann zeta 与 Dirichlet L 函数的载荷。
 *
 * 在飞地内用 FLINT 的球算术(ball arithmetic)做两类计算, 实数由 arb 表示, 复数由 acb 表示:
 *
 *   1. 取值核对。取 zeta 与若干 L 函数的特殊值与已知闭式比较, 参考值分两种来源,
 *      各自在注释中标明: 初等闭式由 arb 以同一精度独立算出, 没有初等闭式的取值
 *      (如 zeta(3)) 由 Apéry 的递推与级数自行求出, 不依赖十进制字面量。
 *
 *   2. 临界线上的零点搜索。载荷自行扫描并用二分法定位零点, 不调用现成的零点例程。
 *      搜索得到的零点与公开的高精度值比较, 同时打印使搜索结果自洽的中间量。
 *
 * 飞地 ABI: main() 入口, 结果经 printf 输出 (英文), return 0 进入 SUSPEND。
 */

#include <stdio.h>

#include "flint/acb.h"
#include "flint/acb_dirichlet.h"
#include "flint/arb.h"
#include "flint/dirichlet.h"
#include "flint/flint.h"
#include "flint/fmpq.h"
#include "flint/fmpz.h"

/* 球算术精度 (比特)。256 位约合 77 位十进制有效数字。 */
#define PREC 256

/*
 * 零点搜索: 扫描步长的分母, 以及单个零点二分次数的上限。
 *
 * 二分的实际次数由精度决定, 上限只是安全边界, 不构成额外开销。收窄区间时, 区间端点自身
 * 也带有半径, 若以 PREC 计算中点, 端点半径约 2 的 -PREC 次幂, 中点到零点的距离一旦小于
 * 该半径, 中点的符号便无法判定而终止二分。故中点与区间本身都以 BISECT_PREC 计算, 使
 * 区间可以一直收窄到符号判定所支持的最深一级; 二分次数随之约为 BISECT_PREC 的位数。
 */
#define SCAN_STEP_DEN 8
#define BISECT_ITERS 2048
#define BISECT_PREC (2 * PREC)
/* 零点搜索的区间上界, 三个函数均在此区间内取前若干个零点核对。 */
#define T_MAX 30
/* 单个函数在此区间内定位到的零点个数上限。 */
#define MAX_ZEROS 16

static int checks_passed = 0;
static int checks_failed = 0;

/* ---------- 结果报告 ---------- */

/*
 * 报告一项实数比较, 打印算得的球中点与半径。半径即该结果与真值之间误差的上界,
 * 算得的精度由此直接可读。不通过时行内出现 FAIL。
 */
static void report_arb(const char *name, int ok, const arb_t got) {
	if (ok) {
		checks_passed++;
		printf("  ok    %-40s [", name);
	} else {
		checks_failed++;
		printf("  FAIL  %-40s [", name);
	}
	arb_printn(got, 20, ARB_STR_NO_RADIUS);
	printf("] rad ");
	mag_printd(arb_radref(got), 3);
	printf("\n");
}

/* 报告一项复数比较, 打印实部与虚部各自的中点与半径。 */
static void report_acb(const char *name, int ok, const acb_t got) {
	if (ok) {
		checks_passed++;
		printf("  ok    %-40s [", name);
	} else {
		checks_failed++;
		printf("  FAIL  %-40s [", name);
	}
	acb_printn(got, 18, ARB_STR_NO_RADIUS);
	printf("] rad ");
	mag_printd(arb_radref(acb_realref(got)), 3);
	printf(" ");
	mag_printd(arb_radref(acb_imagref(got)), 3);
	printf("\n");
}

/* 报告一项布尔判定。 */
static void report_flag(const char *name, int ok) {
	if (ok) {
		checks_passed++;
		printf("  ok    %s\n", name);
	} else {
		checks_failed++;
		printf("  FAIL  %s\n", name);
	}
}

/*
 * 报告一个定位到的零点。零点位置取 32 位有效数字打印, 随之打印包围区间的半宽:
 * 真零点落在该区间内, 故半宽即定位结果与真值之间误差的上界, 定位精度由此直接可读。
 */
static void report_zero(const char *name, int ok, const arb_t mid, const arb_t half) {
	if (ok) {
		checks_passed++;
		printf("  ok    %-40s ", name);
	} else {
		checks_failed++;
		printf("  FAIL  %-40s ", name);
	}
	printf("t = ");
	arb_printn(mid, 32, ARB_STR_NO_RADIUS);
	printf(" +/- ");
	arb_printn(half, 3, ARB_STR_NO_RADIUS);
	printf("\n");
}

/* ---------- 比较原语 ---------- */

/* 字面量中小数点后的位数, 即该字面量最低十进制位所在的位置。 */
static slong literal_decimals(const char *literal) {
	const char *dot = NULL;
	slong count = 0;

	for (; *literal != '\0'; literal++) {
		if (*literal == '.') {
			dot = literal;
			continue;
		}
		if (*literal < '0' || *literal > '9') {
			break;
		}
		if (dot != NULL) {
			count++;
		}
	}
	return count;
}

/*
 * 与十进制字面量给出的参考值比较。字面量是照着真值截断或四舍五入写就的, 与真值相差不超过
 * 其最低十进制位的一个单位, 故两个球未必相交; 误差界按位取定, 为一乘 10 的负 (小数位数) 次幂。
 * 比较的含义是算得值与该字面量在其给出的全部位数上一致。算得值自身的半径远小于该误差界,
 * 不影响判定。
 */
static int arb_meets_literal(const arb_t x, const char *literal) {
	arb_t ref, diff, tol;
	slong places;
	int ok;

	places = literal_decimals(literal);
	arb_init(ref);
	arb_init(diff);
	arb_init(tol);

	arb_set_str(ref, literal, PREC);
	arb_sub(diff, x, ref, PREC);
	arb_abs(diff, diff);

	arb_set_ui(tol, 10);
	arb_pow_ui(tol, tol, (ulong) places, PREC);
	arb_inv(tol, tol, PREC);

	ok = arb_lt(diff, tol);

	arb_clear(tol);
	arb_clear(diff);
	arb_clear(ref);
	return ok;
}

/* 与有理数 (num / den) 给出的参考值比较。 */
static int arb_meets_ratio(const arb_t x, slong num, ulong den) {
	arb_t ref;
	int ok;

	arb_init(ref);
	arb_set_si(ref, num);
	arb_div_ui(ref, ref, den, PREC);
	ok = arb_overlaps(x, ref);
	arb_clear(ref);
	return ok;
}

/* ---------- zeta 的取值 ---------- */

/* zeta(s) 在整数 s 处的取值。s 为整数时函数值为实数, 取复球的实部。 */
static void zeta_at_int(arb_t out, slong s) {
	acb_t arg, val;

	acb_init(arg);
	acb_init(val);
	acb_set_si(arg, s);
	acb_dirichlet_zeta(val, arg, PREC);
	arb_set(out, acb_realref(val));
	acb_clear(val);
	acb_clear(arg);
}

/*
 * 函数方程。zeta 的完备化形式给出
 *     zeta(s) = 2^s pi^(s-1) sin(pi s / 2) Gamma(1 - s) zeta(1 - s)
 * 本项在 s = 3/4 + 2i 处比较等式两侧, 两侧均为复球, 相交即等式在所设精度内成立。
 */
static void check_functional_equation(void) {
	acb_t s, lhs, rhs, factor, tmp, one_minus_s, s_minus_1;

	acb_init(s);
	acb_init(lhs);
	acb_init(rhs);
	acb_init(factor);
	acb_init(tmp);
	acb_init(one_minus_s);
	acb_init(s_minus_1);

	acb_set_d_d(s, 0.75, 2.0);
	acb_dirichlet_zeta(lhs, s, PREC);

	/* 2^s */
	acb_set_ui(factor, 2);
	acb_pow(factor, factor, s, PREC);

	/* pi^(s-1) */
	acb_sub_ui(s_minus_1, s, 1, PREC);
	acb_const_pi(tmp, PREC);
	acb_pow(tmp, tmp, s_minus_1, PREC);
	acb_mul(factor, factor, tmp, PREC);

	/* sin(pi s / 2) */
	acb_div_ui(tmp, s, 2, PREC);
	acb_sin_pi(tmp, tmp, PREC);
	acb_mul(factor, factor, tmp, PREC);

	/* Gamma(1 - s) */
	acb_set_ui(one_minus_s, 1);
	acb_sub(one_minus_s, one_minus_s, s, PREC);
	acb_gamma(tmp, one_minus_s, PREC);
	acb_mul(factor, factor, tmp, PREC);

	/* zeta(1 - s) */
	acb_dirichlet_zeta(tmp, one_minus_s, PREC);
	acb_mul(rhs, factor, tmp, PREC);

	report_acb("zeta(s) = functional eq. right side, s = 3/4+2i", acb_overlaps(lhs, rhs),
		lhs);

	acb_clear(s_minus_1);
	acb_clear(one_minus_s);
	acb_clear(tmp);
	acb_clear(factor);
	acb_clear(rhs);
	acb_clear(lhs);
	acb_clear(s);
}

/* ---------- zeta(3) 与 Apéry 的递推 ---------- */

/* 三项递推的系数: 34 n^3 + 51 n^2 + 27 n + 5。 */
static void apery_coef(fmpz_t out, ulong n) {
	fmpz_t t;

	fmpz_init(t);
	fmpz_ui_pow_ui(out, n, 3);
	fmpz_mul_ui(out, out, 34);
	fmpz_ui_pow_ui(t, n, 2);
	fmpz_mul_ui(t, t, 51);
	fmpz_add(out, out, t);
	fmpz_add_ui(out, out, 27 * n + 5);
	fmpz_clear(t);
}

/*
 * 三项递推的一步:
 *     (n+1)^3 u_{n+1} = (34 n^3 + 51 n^2 + 27 n + 5) u_n - n^3 u_{n-1}
 * 以精确有理数计算, 不引入舍入。
 */
static void apery_step(fmpq_t out, const fmpq_t prev, const fmpq_t prev2, ulong n) {
	fmpz_t coef, n3, next3;
	fmpq_t acc, sub;

	fmpz_init(coef);
	fmpz_init(n3);
	fmpz_init(next3);
	fmpq_init(acc);
	fmpq_init(sub);

	apery_coef(coef, n);
	fmpq_mul_fmpz(acc, prev, coef);

	fmpz_ui_pow_ui(n3, n, 3);
	fmpq_mul_fmpz(sub, prev2, n3);
	fmpq_sub(acc, acc, sub);

	fmpz_set_ui(next3, n + 1);
	fmpz_pow_ui(next3, next3, 3);
	fmpq_div_fmpz(out, acc, next3);

	fmpq_clear(sub);
	fmpq_clear(acc);
	fmpz_clear(next3);
	fmpz_clear(n3);
	fmpz_clear(coef);
}

/* Apéry 数 a_n 的直接定义: a_n = sum_{k=0}^{n} C(n,k)^2 C(n+k,k)^2。 */
static void apery_a_direct(fmpz_t out, ulong n) {
	fmpz_t c1, c2, term;
	ulong k;

	fmpz_init(c1);
	fmpz_init(c2);
	fmpz_init(term);
	fmpz_zero(out);
	for (k = 0; k <= n; k++) {
		fmpz_bin_uiui(c1, n, k);
		fmpz_bin_uiui(c2, n + k, k);
		fmpz_mul(term, c1, c1);
		fmpz_mul(c1, c2, c2);
		fmpz_mul(term, term, c1);
		fmpz_add(out, out, term);
	}
	fmpz_clear(term);
	fmpz_clear(c2);
	fmpz_clear(c1);
}

/* 三次调和数 H_n^(3) = sum_{j=1}^{n} 1 / j^3。 */
static void cubic_harmonic(fmpq_t out, ulong n) {
	fmpz_t j3, one;
	fmpq_t term;
	ulong j;

	fmpz_init(j3);
	fmpz_init(one);
	fmpq_init(term);
	fmpz_one(one);
	fmpq_zero(out);
	for (j = 1; j <= n; j++) {
		fmpz_ui_pow_ui(j3, j, 3);
		fmpq_set_fmpz_frac(term, one, j3);
		fmpq_add(out, out, term);
	}
	fmpq_clear(term);
	fmpz_clear(one);
	fmpz_clear(j3);
}

/*
 * Apéry 数 b_n 的直接定义:
 *     b_n = sum_{k=0}^{n} C(n,k)^2 C(n+k,k)^2 ( H_n^(3)
 *             + sum_{j=1}^{k} (-1)^(j-1) / (2 j^3 C(n,j) C(n+j,j)) )
 */
static void apery_b_direct(fmpq_t out, ulong n) {
	fmpz_t c1, c2, c3, den, one;
	fmpq_t total, inner, term, harm;
	ulong k, j;

	fmpz_init(c1);
	fmpz_init(c2);
	fmpz_init(c3);
	fmpz_init(den);
	fmpz_init(one);
	fmpq_init(total);
	fmpq_init(inner);
	fmpq_init(term);
	fmpq_init(harm);
	fmpz_one(one);

	cubic_harmonic(harm, n);
	fmpq_zero(total);

	for (k = 0; k <= n; k++) {
		fmpq_zero(inner);
		for (j = 1; j <= k; j++) {
			fmpz_bin_uiui(c1, n, j);
			fmpz_bin_uiui(c2, n + j, j);
			fmpz_mul(c3, c1, c2);
			fmpz_mul_ui(c3, c3, 2);
			fmpz_ui_pow_ui(den, j, 3);
			fmpz_mul(den, den, c3);
			fmpq_set_fmpz_frac(term, one, den);
			if (j % 2 == 0) {
				fmpq_neg(term, term);
			}
			fmpq_add(inner, inner, term);
		}
		fmpq_add(term, harm, inner);

		fmpz_bin_uiui(c1, n, k);
		fmpz_bin_uiui(c2, n + k, k);
		fmpz_mul(c3, c1, c1);
		fmpz_mul(c1, c2, c2);
		fmpz_mul(c3, c3, c1);
		fmpq_mul_fmpz(term, term, c3);
		fmpq_add(total, total, term);
	}

	fmpq_set(out, total);
	fmpq_clear(harm);
	fmpq_clear(term);
	fmpq_clear(inner);
	fmpq_clear(total);
	fmpz_clear(one);
	fmpz_clear(den);
	fmpz_clear(c3);
	fmpz_clear(c2);
	fmpz_clear(c1);
}

/*
 * zeta(3) 的两项检验, 两者都以精确有理数计算, 不使用 zeta(3) 的十进制字面量。
 *
 *   Apéry 的两组序列 a_n 与 b_n 满足同一三项递推, 且 b_n / a_n 以
 *       delta^(4n),  delta = sqrt(2) - 1
 *   的速率趋向 zeta(3)。本项先核对递推与直接定义一致, 再按递推外推 b_n / a_n。
 *
 *   另一条途径是 Apéry 的加速级数
 *       zeta(3) = (5/2) sum_{n>=1} (-1)^(n-1) / (n^3 C(2n,n))
 *   它是各项单调递减的交错级数, 余项不超过首个被舍去的项。
 */
static void check_apery_zeta3(void) {
	const ulong recur_check_a = 12;
	const ulong recur_check_b = 6;
	const ulong extrapolate_n = 20;
	const ulong series_terms = 100;

	fmpq_t a_prev, a_cur, a_next, b_prev, b_cur, b_next, ratio, acc, scale;
	fmpz_t expect_a, c2n, n3, one, denom, sign_num;
	arb_t zeta3, approx, diff, delta, bound;
	ulong n;
	int ok;

	fmpq_init(a_prev);
	fmpq_init(a_cur);
	fmpq_init(a_next);
	fmpq_init(b_prev);
	fmpq_init(b_cur);
	fmpq_init(b_next);
	fmpq_init(ratio);
	fmpq_init(acc);
	fmpq_init(scale);
	fmpz_init(expect_a);
	fmpz_init(c2n);
	fmpz_init(n3);
	fmpz_init(one);
	fmpz_init(denom);
	fmpz_init(sign_num);
	arb_init(zeta3);
	arb_init(approx);
	arb_init(diff);
	arb_init(delta);
	arb_init(bound);
	fmpz_one(one);

	zeta_at_int(zeta3, 3);

	/* 递推与直接定义核对: a_n 为整数, 递推中的除法必须整除 */
	fmpq_one(a_prev); /* a_0 */
	fmpq_set_ui(a_cur, 5, 1); /* a_1 */
	ok = 1;
	for (n = 1; n <= recur_check_a; n++) {
		apery_step(a_next, a_cur, a_prev, n);
		if (!fmpz_is_one(fmpq_denref(a_next))) {
			ok = 0;
			break;
		}
		apery_a_direct(expect_a, n + 1);
		if (!fmpz_equal(fmpq_numref(a_next), expect_a)) {
			ok = 0;
			break;
		}
		fmpq_swap(a_prev, a_cur);
		fmpq_swap(a_cur, a_next);
	}
	report_flag("Apery recurrence reproduces a_n for n <= 12", ok);

	/* 递推与直接定义核对: b_n 为有理数, 与直接定义逐项相等 */
	fmpq_zero(b_prev); /* b_0 */
	fmpq_set_ui(b_cur, 6, 1); /* b_1 */
	ok = 1;
	for (n = 1; n <= recur_check_b; n++) {
		apery_step(b_next, b_cur, b_prev, n);
		apery_b_direct(acc, n + 1);
		if (!fmpq_equal(b_next, acc)) {
			ok = 0;
			break;
		}
		fmpq_swap(b_prev, b_cur);
		fmpq_swap(b_cur, b_next);
	}
	report_flag("Apery recurrence reproduces b_n for n <= 6", ok);

	/* 外推: 由 b_1 / a_1 出发, 两组序列各自递推到 b_N / a_N */
	fmpq_one(a_prev);
	fmpq_set_ui(a_cur, 5, 1);
	fmpq_zero(b_prev);
	fmpq_set_ui(b_cur, 6, 1);
	for (n = 1; n < extrapolate_n; n++) {
		apery_step(a_next, a_cur, a_prev, n);
		apery_step(b_next, b_cur, b_prev, n);
		fmpq_swap(a_prev, a_cur);
		fmpq_swap(a_cur, a_next);
		fmpq_swap(b_prev, b_cur);
		fmpq_swap(b_cur, b_next);
	}
	fmpq_div(ratio, b_cur, a_cur);
	arb_set_fmpq(approx, ratio, PREC);

	/* 判定上界 delta^(4N), delta = sqrt(2) - 1 */
	arb_sqrt_ui(delta, 2, PREC);
	arb_sub_ui(delta, delta, 1, PREC);
	arb_pow_ui(bound, delta, 4 * extrapolate_n, PREC);

	arb_sub(diff, approx, zeta3, PREC);
	arb_abs(diff, diff);
	report_arb("|b_20/a_20 - zeta(3)| < (sqrt(2)-1)^80", arb_lt(diff, bound), diff);

	/* 加速级数: 逐项以精确有理数累加 */
	fmpq_zero(acc);
	for (n = 1; n <= series_terms; n++) {
		fmpz_ui_pow_ui(n3, n, 3);
		fmpz_bin_uiui(c2n, 2 * n, n);
		fmpz_mul(denom, n3, c2n);
		fmpq_set_fmpz_frac(b_next, one, denom);
		if (n % 2 == 0) {
			fmpq_neg(b_next, b_next);
		}
		fmpq_add(acc, acc, b_next);
	}
	fmpq_set_ui(scale, 5, 2);
	fmpq_mul(acc, acc, scale);
	arb_set_fmpq(approx, acc, PREC);

	/* 余项上界取首个被舍去的项: (5/2) / ((N+1)^3 C(2N+2, N+1)) */
	fmpz_set_ui(n3, series_terms + 1);
	fmpz_pow_ui(n3, n3, 3);
	fmpz_bin_uiui(c2n, 2 * series_terms + 2, series_terms + 1);
	fmpz_mul(denom, n3, c2n);
	fmpz_mul_ui(denom, denom, 2);
	fmpz_set_ui(sign_num, 5);
	arb_set_fmpz(bound, sign_num);
	arb_div_fmpz(bound, bound, denom, PREC);

	arb_sub(diff, approx, zeta3, PREC);
	arb_abs(diff, diff);
	report_arb("|accelerated series (100 terms) - zeta(3)| < a_101", arb_lt(diff, bound),
		diff);

	arb_clear(bound);
	arb_clear(delta);
	arb_clear(diff);
	arb_clear(approx);
	arb_clear(zeta3);
	fmpz_clear(sign_num);
	fmpz_clear(denom);
	fmpz_clear(one);
	fmpz_clear(n3);
	fmpz_clear(c2n);
	fmpz_clear(expect_a);
	fmpq_clear(acc);
	fmpq_clear(scale);
	fmpq_clear(ratio);
	fmpq_clear(b_next);
	fmpq_clear(b_cur);
	fmpq_clear(b_prev);
	fmpq_clear(a_next);
	fmpq_clear(a_cur);
	fmpq_clear(a_prev);
}

/* ---------- 临界线上的零点搜索 ---------- */

/*
 * Hardy Z 函数在实数 t 处的取值:
 *     Z(t) = e^{i theta(t)} L(1/2 + i t, chi)
 * 其中 theta 为与特征相配的相位, 使 Z 在实数 t 上取实值。Z 的零点与 L(s, chi) 在
 * 临界线 Re(s) = 1/2 上的零点一一对应。群与特征传空即对应 Riemann zeta。
 */
static void hardy_z_value(arb_t out, const arb_t t, const dirichlet_group_t G,
	const dirichlet_char_t chi, slong prec) {
	acb_t arg, val;

	acb_init(arg);
	acb_init(val);
	acb_set_arb(arg, t);
	acb_dirichlet_hardy_z(val, arg, G, chi, 1, prec);
	arb_set(out, acb_realref(val));
	acb_clear(val);
	acb_clear(arg);
}

/*
 * 取 Z(t) 的符号: 1 为正, -1 为负, 0 表示所给精度内不足以判定。从 prec_from 起逐级加倍
 * 重算, 直到判定或达到 BISECT_PREC。prec_from 由调用方按前一次判定通过的精度给出, 避免
 * 每步都从最低精度重算。
 */
static int hardy_z_sign(
	const arb_t t, const dirichlet_group_t G, const dirichlet_char_t chi, slong prec_from) {
	arb_t val;
	slong prec;
	int sign = 0;

	arb_init(val);
	for (prec = prec_from; prec <= BISECT_PREC; prec *= 2) {
		hardy_z_value(val, t, G, chi, prec);
		if (arb_is_positive(val)) {
			sign = 1;
			break;
		}
		if (arb_is_negative(val)) {
			sign = -1;
			break;
		}
	}
	arb_clear(val);
	return sign;
}

/*
 * 二分法收窄一个已经由符号变化确认的零点。区间端点始终保持异号, 故 [lo, hi] 内必有零点
 * (Z 连续)。中点的符号无法判定时停止细分, 此时 [lo, hi] 即该零点的包围区间。
 */
static void bisect_zero(arb_t lo_out, arb_t hi_out, const arb_t a, const arb_t b,
	const dirichlet_group_t G, const dirichlet_char_t chi) {
	arb_t lo, hi, mid;
	int sign_lo, sign_mid;
	slong prec_hint = PREC;
	slong i;

	arb_init(lo);
	arb_init(hi);
	arb_init(mid);
	arb_set(lo, a);
	arb_set(hi, b);
	sign_lo = hardy_z_sign(lo, G, chi, prec_hint);

	for (i = 0; i < BISECT_ITERS; i++) {
		arb_add(mid, lo, hi, BISECT_PREC);
		arb_mul_2exp_si(mid, mid, -1);
		sign_mid = hardy_z_sign(mid, G, chi, prec_hint);
		if (sign_mid == 0) {
			break;
		}
		prec_hint = BISECT_PREC;
		if (sign_mid == sign_lo) {
			arb_set(lo, mid);
		} else {
			arb_set(hi, mid);
		}
	}

	arb_set(lo_out, lo);
	arb_set(hi_out, hi);
	arb_clear(mid);
	arb_clear(hi);
	arb_clear(lo);
}

/*
 * 扫描 [0, T_MAX] 定位零点。步长取 1/8, 远小于三个函数在该区间内相邻零点的最小间距
 * (约 1.29), 故一次符号变化只对应一个零点。定位到的零点按从小到大写入 out, 所在区间的
 * 半宽写入 half (半宽即该零点的定位精度), 返回个数。
 */
static slong locate_zeros(arb_ptr out, arb_ptr half, slong max_out, const dirichlet_group_t G,
	const dirichlet_char_t chi) {
	arb_t step, prev, cur, lo, hi;
	int sign_prev, sign_cur;
	slong count = 0;
	int i;

	arb_init(step);
	arb_init(prev);
	arb_init(cur);
	arb_init(lo);
	arb_init(hi);

	arb_set_ui(step, 1);
	arb_div_ui(step, step, SCAN_STEP_DEN, PREC);
	arb_zero(prev);
	sign_prev = 0;

	for (i = 1; i <= T_MAX * SCAN_STEP_DEN; i++) {
		arb_mul_ui(cur, step, (ulong) i, PREC);
		sign_cur = hardy_z_sign(cur, G, chi, PREC);
		if (sign_prev != 0 && sign_cur != 0 && sign_prev != sign_cur) {
			if (count < max_out) {
				bisect_zero(lo, hi, prev, cur, G, chi);
				/* 中点作为零点位置, 区间半宽作为定位精度 */
				arb_add(out + count, lo, hi, BISECT_PREC);
				arb_mul_2exp_si(out + count, out + count, -1);
				arb_sub(half + count, hi, lo, BISECT_PREC);
				arb_mul_2exp_si(half + count, half + count, -1);
				arb_abs(half + count, half + count);
				count++;
			}
		}
		arb_set(prev, cur);
		sign_prev = sign_cur;
	}

	arb_clear(hi);
	arb_clear(lo);
	arb_clear(cur);
	arb_clear(prev);
	arb_clear(step);
	return count;
}

/* ---------- Dirichlet 特征与 L 函数 ---------- */

/*
 * 取模 q 的实本原奇特征, 即 Kronecker 符号 (d/.) 这一支: 在模 q 的本原特征中取阶为 2
 * (实特征) 且奇偶性为奇者。q 等于 4 或为模 4 余 3 的素数时该特征唯一, 此时其模等于
 * 虚二次域的判别式绝对值。返回 0 表示未取到。
 */
static int real_odd_primitive_char(dirichlet_char_t chi, const dirichlet_group_t G) {
	dirichlet_char_t cursor;
	int found = 0;

	dirichlet_char_init(cursor, G);
	dirichlet_char_first_primitive(cursor, G);
	do {
		if (dirichlet_order_char(G, cursor) == 2 && dirichlet_parity_char(G, cursor) == 1) {
			dirichlet_char_set(chi, G, cursor);
			found = 1;
			break;
		}
	} while (dirichlet_char_next_primitive(cursor, G) >= 0);

	dirichlet_char_clear(cursor);
	return found;
}

/* L(s, chi) 在整数 s 处的取值。chi 为实特征时函数值为实数, 取复球的实部。 */
static void l_at_int(
	arb_t out, const dirichlet_group_t G, const dirichlet_char_t chi, slong s) {
	acb_t arg, val;

	acb_init(arg);
	acb_init(val);
	acb_set_si(arg, s);
	acb_dirichlet_l(val, arg, G, chi, PREC);
	arb_set(out, acb_realref(val));
	acb_clear(val);
	acb_clear(arg);
}

/*
 * 特征表: 实本原奇特征满足 chi(1) = 1 与 chi(q-1) = -1, 后者即 chi(-1) = -1。
 * 取值经 acb_dirichlet_chi 求得, 结果为复球。
 */
static void check_character_table(
	ulong q, const dirichlet_group_t G, const dirichlet_char_t chi) {
	acb_t val;
	char name[64];

	acb_init(val);

	acb_dirichlet_chi(val, G, chi, 1, PREC);
	snprintf(name, sizeof(name), "q = %lu: chi(1) = 1", (unsigned long) q);
	report_acb(name, arb_contains_si(acb_realref(val), 1), val);

	acb_dirichlet_chi(val, G, chi, q - 1, PREC);
	snprintf(name, sizeof(name), "q = %lu: chi(-1) = -1", (unsigned long) q);
	report_acb(name, arb_contains_si(acb_realref(val), -1), val);

	acb_clear(val);
}

/*
 * 临界线上的零点核对。对每个函数定位 [0, T_MAX] 内的全部零点, 核对个数, 并核对前
 * 三个零点的位置 (位置与包围区间半宽一并打印, 半宽即定位精度)。参考值取自公开的高精度
 * 计算结果, 与本载荷的搜索过程无关。
 *
 * zeta 另有一项独立的核对: FLINT 的零点计数函数给出区间内的零点个数, 与扫描结果
 * 比较, 用于确认扫描步长没有漏掉零点。
 */
static void check_critical_line_zeros(void) {
	static const char *zeta_ref[3] = {
		"14.13472514173469379045725198356247027078",
		"21.02203963877155499262847959389690277733",
		"25.01085758014568876321379099256282181866",
	};
	static const char *beta_ref[3] = {
		"6.020948904697596654902511521612085868864",
		"10.24377030416655455213775747910995902486",
		"12.98809801231242250745310978956299376497",
	};
	static const char *chi3_ref[3] = {
		"8.039737155681466681713623214172965802793",
		"11.2492062077729352497050256788632146487",
		"15.7046191767216255651655508804",
	};
	const slong expected[3] = {3, 10, 8};
	const char *labels[3] = {"zeta", "L(s, chi_-4)", "L(s, chi_-3)"};
	arb_ptr zeros, halves;
	arb_t limit, counted;
	/* 群与特征为空指针即表示 Riemann zeta, 故以指针形式保存, 非空时指向下面两份存储 */
	dirichlet_group_struct *G;
	dirichlet_char_struct *chi;
	dirichlet_group_t group_store;
	dirichlet_char_t char_store;
	slong found[3];
	char name[96];
	int f, k;

	zeros = _arb_vec_init(MAX_ZEROS);
	halves = _arb_vec_init(MAX_ZEROS);
	arb_init(limit);
	arb_init(counted);

	for (f = 0; f < 3; f++) {
		G = NULL;
		chi = NULL;
		if (f > 0) {
			dirichlet_group_init(group_store, (f == 1) ? 4 : 3);
			dirichlet_char_init(char_store, group_store);
			if (!real_odd_primitive_char(char_store, group_store)) {
				checks_failed++;
				printf("  FAIL  %s: no real odd primitive character\n", labels[f]);
				found[f] = 0;
				dirichlet_char_clear(char_store);
				dirichlet_group_clear(group_store);
				continue;
			}
			G = group_store;
			chi = char_store;
		}

		found[f] = locate_zeros(zeros, halves, MAX_ZEROS, G, chi);

		snprintf(name, sizeof(name), "%s: zeros found in (0, 30]", labels[f]);
		printf("  ok    %-40s %ld (expected %ld)\n", name, (long) found[f],
			(long) expected[f]);
		if (found[f] == expected[f]) {
			checks_passed++;
		} else {
			checks_failed++;
			printf("  FAIL  %s: zero count mismatch\n", labels[f]);
		}

		for (k = 0; k < 3 && k < found[f]; k++) {
			const char *ref = (f == 0) ? zeta_ref[k] : (f == 1) ? beta_ref[k] : chi3_ref[k];
			snprintf(name, sizeof(name), "%s: zero #%d at t", labels[f], k + 1);
			report_zero(name, arb_meets_literal(zeros + k, ref), zeros + k, halves + k);
		}

		if (f == 0) {
			/* 独立的零点计数核对 */
			arb_set_ui(limit, T_MAX);
			acb_dirichlet_zeta_nzeros(counted, limit, PREC);
			snprintf(name, sizeof(name), "zeta: zero count in (0, 30] agrees with nzeros");
			report_arb(name,
				arb_contains_si(counted, expected[0]) && found[0] == expected[0], counted);
		}

		if (f > 0) {
			dirichlet_char_clear(char_store);
			dirichlet_group_clear(group_store);
		}
	}

	arb_clear(counted);
	arb_clear(limit);
	_arb_vec_clear(halves, MAX_ZEROS);
	_arb_vec_clear(zeros, MAX_ZEROS);
}

/* 模 4 的实本原奇特征 chi_-4 = (-4/.) 的取值 */
static void check_l_mod_4(void) {
	dirichlet_group_t G;
	dirichlet_char_t chi;
	arb_t got, ref, pi;

	arb_init(got);
	arb_init(ref);
	arb_init(pi);
	arb_const_pi(pi, PREC);
	dirichlet_group_init(G, 4);
	dirichlet_char_init(chi, G);

	if (!real_odd_primitive_char(chi, G)) {
		checks_failed++;
		printf("  FAIL  q = 4: no real odd primitive character\n");
	} else {
		check_character_table(4, G, chi);

		/* L(1, chi_-4) = pi / 4 */
		l_at_int(got, G, chi, 1);
		arb_div_ui(ref, pi, 4, PREC);
		report_arb("L(1, chi_-4) = pi/4", arb_overlaps(got, ref), got);

		/* L(2, chi_-4) = G, 卡塔兰常数 */
		l_at_int(got, G, chi, 2);
		report_arb("L(2, chi_-4) = Catalan G",
			arb_meets_literal(got, "0.9159655941772190150546035149323841107741"), got);

		/* L(3, chi_-4) = pi^3 / 32 */
		l_at_int(got, G, chi, 3);
		arb_mul(ref, pi, pi, PREC);
		arb_mul(ref, ref, pi, PREC);
		arb_div_ui(ref, ref, 32, PREC);
		report_arb("L(3, chi_-4) = pi^3/32", arb_overlaps(got, ref), got);

		/* L(0, chi_-4) = 1/2 */
		l_at_int(got, G, chi, 0);
		report_arb("L(0, chi_-4) = 1/2", arb_meets_ratio(got, 1, 2), got);

		/* L(-1, chi_-4) = 0, 平凡零点 */
		l_at_int(got, G, chi, -1);
		report_arb("L(-1, chi_-4) = 0", arb_contains_si(got, 0), got);
	}

	dirichlet_char_clear(chi);
	dirichlet_group_clear(G);
	arb_clear(pi);
	arb_clear(ref);
	arb_clear(got);
}

/* 模 3 的实本原奇特征 chi_-3 = (-3/.) 的取值 */
static void check_l_mod_3(void) {
	dirichlet_group_t G;
	dirichlet_char_t chi;
	arb_t got, ref, pi;

	arb_init(got);
	arb_init(ref);
	arb_init(pi);
	arb_const_pi(pi, PREC);
	dirichlet_group_init(G, 3);
	dirichlet_char_init(chi, G);

	if (!real_odd_primitive_char(chi, G)) {
		checks_failed++;
		printf("  FAIL  q = 3: no real odd primitive character\n");
	} else {
		check_character_table(3, G, chi);

		/* L(1, chi_-3) = pi / (3 sqrt 3) */
		l_at_int(got, G, chi, 1);
		arb_sqrt_ui(ref, 3, PREC);
		arb_mul_ui(ref, ref, 3, PREC);
		arb_div(ref, pi, ref, PREC);
		report_arb("L(1, chi_-3) = pi/(3 sqrt 3)", arb_overlaps(got, ref), got);

		/* L(2, chi_-3) 没有初等闭式, 参考值为 40 位有效数字的字面量 */
		l_at_int(got, G, chi, 2);
		report_arb("L(2, chi_-3) = 0.781302412896486296867187429...",
			arb_meets_literal(got, "0.7813024128964862968671874296240923563651"), got);

		/* L(0, chi_-3) = 1/3 */
		l_at_int(got, G, chi, 0);
		report_arb("L(0, chi_-3) = 1/3", arb_meets_ratio(got, 1, 3), got);
	}

	dirichlet_char_clear(chi);
	dirichlet_group_clear(G);
	arb_clear(pi);
	arb_clear(ref);
	arb_clear(got);
}

/*
 * 虚二次域的类数公式。判别式为 D 的虚二次域满足
 *     L(1, chi_D) = 2 pi h / (w sqrt|D|)
 * 其中 h 为类数, w 为单位根个数 (D = -3 取 6, D = -4 取 4, 其余取 2)。本项对
 * D = -3, -4, -7, -11 反解 h 并断言该球包含整数 1: 这四个域都是类数为 1 的域,
 * 判定落在整数上, 不受任何十进制字面量影响。
 */
static void check_class_number_formula(void) {
	static const struct {
		ulong q; /* 特征的模, 等于 |D| */
		slong disc; /* 虚二次域判别式 */
		ulong units; /* 单位根个数 w */
	} fields[] = {
		{3, -3, 6},
		{4, -4, 4},
		{7, -7, 2},
		{11, -11, 2},
	};
	size_t i;

	for (i = 0; i < sizeof(fields) / sizeof(fields[0]); i++) {
		dirichlet_group_t G;
		dirichlet_char_t chi;
		arb_t l1, h, pi, sqrt_disc;
		char name[64];

		arb_init(l1);
		arb_init(h);
		arb_init(pi);
		arb_init(sqrt_disc);
		dirichlet_group_init(G, fields[i].q);
		dirichlet_char_init(chi, G);

		if (!real_odd_primitive_char(chi, G)) {
			checks_failed++;
			printf("  FAIL  D = %ld: no real odd primitive character\n",
				(long) fields[i].disc);
		} else {
			l_at_int(l1, G, chi, 1);
			arb_const_pi(pi, PREC);
			arb_sqrt_ui(sqrt_disc, (ulong) (-fields[i].disc), PREC);

			/* h = w sqrt|D| L(1, chi) / (2 pi) */
			arb_mul_ui(h, sqrt_disc, fields[i].units, PREC);
			arb_mul(h, h, l1, PREC);
			arb_mul_ui(pi, pi, 2, PREC);
			arb_div(h, h, pi, PREC);

			snprintf(
				name, sizeof(name), "class number h(%ld) = 1", (long) fields[i].disc);
			report_arb(name, arb_contains_si(h, 1), h);
		}

		dirichlet_char_clear(chi);
		dirichlet_group_clear(G);
		arb_clear(sqrt_disc);
		arb_clear(pi);
		arb_clear(h);
		arb_clear(l1);
	}
}

/* zeta 的取值 */
static void check_zeta_values(void) {
	arb_t got, ref, pi;

	arb_init(got);
	arb_init(ref);
	arb_init(pi);
	arb_const_pi(pi, PREC);

	/* zeta(2) = pi^2 / 6 */
	zeta_at_int(got, 2);
	arb_mul(ref, pi, pi, PREC);
	arb_div_ui(ref, ref, 6, PREC);
	report_arb("zeta(2) = pi^2/6", arb_overlaps(got, ref), got);

	/* zeta(4) = pi^4 / 90 */
	zeta_at_int(got, 4);
	arb_mul(ref, pi, pi, PREC);
	arb_mul(ref, ref, ref, PREC);
	arb_div_ui(ref, ref, 90, PREC);
	report_arb("zeta(4) = pi^4/90", arb_overlaps(got, ref), got);

	/* zeta(0) = -1/2, 以及 zeta(1-2n) = -B_2n/(2n) 给出的有理值 */
	zeta_at_int(got, 0);
	report_arb("zeta(0) = -1/2", arb_meets_ratio(got, -1, 2), got);

	zeta_at_int(got, -1);
	report_arb("zeta(-1) = -1/12", arb_meets_ratio(got, -1, 12), got);

	zeta_at_int(got, -3);
	report_arb("zeta(-3) = 1/120", arb_meets_ratio(got, 1, 120), got);

	/* 平凡零点: 负偶数处的函数值为零 */
	zeta_at_int(got, -2);
	report_arb("zeta(-2) = 0", arb_contains_si(got, 0), got);

	arb_clear(pi);
	arb_clear(ref);
	arb_clear(got);
}

int main(void) {
	printf("Lfunc: zeta and Dirichlet L-functions, FLINT arb/acb, prec = %d bits\n",
		(int) PREC);

	printf("zeta values\n");
	check_zeta_values();
	check_functional_equation();

	printf("zeta(3) via Apery recurrence and series\n");
	check_apery_zeta3();

	printf("Dirichlet characters and L-values\n");
	check_l_mod_4();
	check_l_mod_3();
	check_class_number_formula();

	printf("zeros on the critical line Re(s) = 1/2\n");
	check_critical_line_zeros();

	printf("Lfunc: %d checks, %d failed\n", checks_passed + checks_failed, checks_failed);

	/* 释放 FLINT 与 MPFR 的全局缓存, 否则这些内存由库持有到进程结束 */
	flint_cleanup_master();
	return checks_failed == 0 ? 0 : 1;
}
