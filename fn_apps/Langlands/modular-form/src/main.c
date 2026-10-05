/*
 * modular-form — 模形式与模判别式的载荷。
 *
 * 在飞地内用两种互相独立的方式处理模形式:
 *
 *   1. 精确整数 q 级数。E_4 与 E_6 由因子和 sigma_3 与 sigma_5 生成, 模判别式由一个
 *      Euler 乘积的 24 次幂生成, 两者的关系 E_4^3 - E_6^2 = 1728 Delta 是恒等式,
 *      载荷用整系数多项式的截断运算逐步核对, 全部系数为精确整数。
 *
 *   2. 球算术(ball arithmetic)取值。eta, j 与模判别式由 FLINT 的 acb_modular 例程
 *      以复数球算出, 与初等闭式、特殊值以及上面精确算出的 q 级数比较。
 *
 * 核对用的参考值分两类, 各自在注释中标明来源: 公开数表给出的前若干项系数, 以及由
 * 载荷以同一精度独立算出的闭式。
 *
 * 飞地 ABI: main() 入口, 结果经 printf 输出 (英文), return 0 进入 SUSPEND。
 */

#include <stdio.h>

#include "flint/acb.h"
#include "flint/acb_modular.h"
#include "flint/arb.h"
#include "flint/flint.h"
#include "flint/fmpz.h"
#include "flint/fmpz_poly.h"
#include "flint/fmpz_vec.h"
#include "flint/mag.h"
#include "flint/ulong_extras.h"

/* 球算术精度 (比特)。256 位约合 77 位十进制有效数字。 */
#define PREC 256

/*
 * q 级数保留的项数, 即次数 0 到 SERIES_TERMS - 1。精确级数的各项核对都截断到该长度:
 * 模判别式的系数 tau(n) 需要 n 到 SERIES_TERMS - 1, Hecke 关系与模 691 的同余都在
 * 这批系数上逐项验证。
 */
#define SERIES_TERMS 128

/* tau(n) 的上界, 即精确级数能给出的最大下标。 */
#define TAU_MAX (SERIES_TERMS - 1)

/*
 * 以 q 级数求值时取用的项数。两个求值点 tau = i 与 tau = 0.3 + 1.1 i 上 |q| 分别为
 * 1.87e-3 与 1.12e-3, 取到 64 项时余项已在 1e-150 以下, 远小于比较的误差界。
 */
#define SERIES_TAKE 64

/*
 * 各项比较的误差界取 10 的负 CHECK_DIGITS 次幂。256 位球算术约合 77 位十进制有效数字,
 * 误差界在此留出一档余量, 比较的含义是两侧在前 60 位十进制上一致。
 */
#define CHECK_DIGITS 60

/*
 * Dirichlet 级数与 Euler 乘积比较所用的 s。s = 13 时级数绝对收敛
 */
#define EULER_S 13

/*
 * Euler 乘积两侧比较的误差界取 10 的负 EULER_DIGITS 次幂。两侧之差是下标超出
 * SERIES_TERMS - 1 的那些整数 n 上的余项, 由 Deligne 界与因子个数界控制:
 *
 *   |tau(n)| <= d(n) n^(11/2),  d(n) <= 2 n^(1/2)
 *   sum_{n > N} |tau(n)| n^-s <= 2 sum_{n > N} n^(6 - s) < 1e-13,  N = 127, s = 13
 *
 * 故取 1e-12 作误差界, 留一档余量。
 */
#define EULER_DIGITS 12

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
		printf("  ok    %-44s [", name);
	} else {
		checks_failed++;
		printf("  FAIL  %-44s [", name);
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
		printf("  ok    %-44s [", name);
	} else {
		checks_failed++;
		printf("  FAIL  %-44s [", name);
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
 * 报告一项按误差界判定的比较, 打印两侧之差。差值即该判定的余量, 与误差界之比由此直接可读。
 */
static void report_diff(const char *name, int ok, const arb_t diff) {
	if (ok) {
		checks_passed++;
		printf("  ok    %-44s ", name);
	} else {
		checks_failed++;
		printf("  FAIL  %-44s ", name);
	}
	printf("diff = ");
	arb_printn(diff, 5, ARB_STR_MORE);
	printf("\n");
}

/* ---------- 比较原语 ---------- */

/* 误差界 10 的负 places 次幂。 */
static void tolerance(arb_t out, slong places) {
	arb_set_ui(out, 10);
	arb_pow_ui(out, out, (ulong)places, PREC);
	arb_inv(out, out, PREC);
}

/* out = |x - y|。两侧之差自身带有半径, 判定据此按区间给出。 */
static void arb_distance(arb_t out, const arb_t x, const arb_t y) {
	arb_sub(out, x, y, PREC);
	arb_abs(out, out);
}

/* |x - y| < 10 的负 places 次幂。 */
static int arb_close(const arb_t x, const arb_t y, slong places) {
	arb_t diff, tol;
	int ok;

	arb_init(diff);
	arb_init(tol);
	arb_distance(diff, x, y);
	tolerance(tol, places);
	ok = arb_lt(diff, tol);
	arb_clear(tol);
	arb_clear(diff);
	return ok;
}

/* |x - y| < 10 的负 places 次幂, x 与 y 为复数球。 */
static int acb_close(const acb_t x, const acb_t y, slong places) {
	acb_t diff;
	arb_t mag, tol;
	int ok;

	acb_init(diff);
	arb_init(mag);
	arb_init(tol);
	acb_sub(diff, x, y, PREC);
	acb_abs(mag, diff, PREC);
	tolerance(tol, places);
	ok = arb_lt(mag, tol);
	arb_clear(tol);
	arb_clear(mag);
	acb_clear(diff);
	return ok;
}

/* |x| < 10 的负 places 次幂, x 为复数球。 */
static int acb_small(const acb_t x, slong places) {
	arb_t mag, tol;
	int ok;

	arb_init(mag);
	arb_init(tol);
	acb_abs(mag, x, PREC);
	tolerance(tol, places);
	ok = arb_lt(mag, tol);
	arb_clear(tol);
	arb_clear(mag);
	return ok;
}

/* ---------- 精确整数 q 级数 ---------- */

/* sigma_k(n): n 的全部正因子的 k 次幂之和。 */
static void divisor_sigma(fmpz_t out, ulong n, ulong k) {
	fmpz_t power;
	ulong d;

	fmpz_init(power);
	fmpz_zero(out);
	for (d = 1; d <= n; d++) {
		if (n % d != 0) {
			continue;
		}
		fmpz_ui_pow_ui(power, d, k);
		fmpz_add(out, out, power);
	}
	fmpz_clear(power);
}

/* res = base^exp, 只保留次数小于 len 的项。 */
static void poly_pow_trunc(
	fmpz_poly_t res, const fmpz_poly_t base, ulong exp, slong len) {
	fmpz_poly_t acc, sq;
	ulong e = exp;

	fmpz_poly_init(acc);
	fmpz_poly_init(sq);
	fmpz_poly_one(acc);
	fmpz_poly_set(sq, base);
	while (e > 0) {
		if (e & 1) {
			fmpz_poly_mullow(acc, acc, sq, len);
		}
		e >>= 1;
		if (e > 0) {
			fmpz_poly_mullow(sq, sq, sq, len);
		}
	}
	fmpz_poly_set(res, acc);
	fmpz_poly_clear(sq);
	fmpz_poly_clear(acc);
}

/*
 * E_4 与 E_6, 截断到 len 项:
 *
 *   E_4 = 1 + 240 sum sigma_3(n) q^n
 *   E_6 = 1 - 504 sum sigma_5(n) q^n
 */
static void eisenstein_series(fmpz_poly_t e4, fmpz_poly_t e6, slong len) {
	fmpz_t s;
	slong n;

	fmpz_init(s);
	fmpz_poly_zero(e4);
	fmpz_poly_zero(e6);
	fmpz_poly_set_coeff_si(e4, 0, 1);
	fmpz_poly_set_coeff_si(e6, 0, 1);
	for (n = 1; n < len; n++) {
		divisor_sigma(s, (ulong)n, 3);
		fmpz_mul_ui(s, s, 240);
		fmpz_poly_set_coeff_fmpz(e4, n, s);
		divisor_sigma(s, (ulong)n, 5);
		fmpz_mul_ui(s, s, 504);
		fmpz_neg(s, s);
		fmpz_poly_set_coeff_fmpz(e6, n, s);
	}
	fmpz_clear(s);
}

/* prod_{m>=1} (1 - q^m), 截断到 len 项。 */
static void euler_product_poly(fmpz_poly_t out, slong len) {
	fmpz_poly_t prod, factor;
	slong m;

	fmpz_poly_init(prod);
	fmpz_poly_init(factor);
	fmpz_poly_one(prod);
	for (m = 1; m < len; m++) {
		fmpz_poly_zero(factor);
		fmpz_poly_set_coeff_si(factor, 0, 1);
		fmpz_poly_set_coeff_si(factor, m, -1);
		fmpz_poly_mullow(prod, prod, factor, len);
	}
	fmpz_poly_set(out, prod);
	fmpz_poly_clear(factor);
	fmpz_poly_clear(prod);
}

/*
 * Ramanujan tau 的前 TAU_MAX 项。模判别式的 q 级数为
 *
 *   Delta = q prod_{m>=1} (1 - q^m)^24 = sum_{n>=1} tau(n) q^n
 *
 * 故 tau(n) 即该 Euler 乘积的 24 次幂中 q 的 n - 1 次项系数。系数为精确整数。
 */
static void tau_from_eta_product(fmpz *tau) {
	fmpz_poly_t unit, p24;
	slong n;

	fmpz_poly_init(unit);
	fmpz_poly_init(p24);
	euler_product_poly(unit, SERIES_TERMS);
	poly_pow_trunc(p24, unit, 24, SERIES_TERMS);
	for (n = 1; n <= TAU_MAX; n++) {
		fmpz_poly_get_coeff_fmpz(tau + n, p24, n - 1);
	}
	fmpz_poly_clear(p24);
	fmpz_poly_clear(unit);
}

/* 公开数表给出的 tau(1) 到 tau(10)。 */
static const slong tau_table[10] = {
	1, -24, 252, -1472, 4830, -6048, -16744, 84480, -113643, -115920};

/* 公开数表给出的 j 的 q 展开系数, 从 q^0 起: j = q^-1 (1 + 744 q + 196884 q^2 + ...)。 */
static const slong j_table[8] = {
	1, 744, 196884, 21493760, 864299970, 20245856256, 333202640600, 4252023300096};
#define J_TERMS ((slong)(sizeof(j_table) / sizeof(j_table[0])))

/* 打印一段系数, 由次数 from 起连续 count 项。 */
static void print_coeffs(const fmpz_poly_t poly, slong from, slong count) {
	fmpz_t c;
	slong n;

	fmpz_init(c);
	for (n = from; n < from + count; n++) {
		fmpz_poly_get_coeff_fmpz(c, poly, n);
		printf("%s", n == from ? "" : " ");
		fmpz_print(c);
	}
	fmpz_clear(c);
}

/* ---------- 精确级数核对 ---------- */

static void check_exact_series(const fmpz *tau) {
	fmpz_poly_t e4, e6, e4cub, e6sq, ident, unit, p24, qp24, jser;
	fmpz_t lhs, rhs, p11, mod_a, mod_b;
	ulong p, pk;
	slong m, n;
	int ok;

	fmpz_poly_init(e4);
	fmpz_poly_init(e6);
	fmpz_poly_init(e4cub);
	fmpz_poly_init(e6sq);
	fmpz_poly_init(ident);
	fmpz_poly_init(unit);
	fmpz_poly_init(p24);
	fmpz_poly_init(qp24);
	fmpz_poly_init(jser);
	fmpz_init(lhs);
	fmpz_init(rhs);
	fmpz_init(p11);
	fmpz_init(mod_a);
	fmpz_init(mod_b);

	eisenstein_series(e4, e6, SERIES_TERMS);
	poly_pow_trunc(e4cub, e4, 3, SERIES_TERMS);
	poly_pow_trunc(e6sq, e6, 2, SERIES_TERMS);
	fmpz_poly_sub(ident, e4cub, e6sq);

	euler_product_poly(unit, SERIES_TERMS);
	poly_pow_trunc(p24, unit, 24, SERIES_TERMS);
	fmpz_poly_shift_left(qp24, p24, 1);
	fmpz_poly_truncate(qp24, SERIES_TERMS);
	fmpz_poly_scalar_mul_ui(qp24, qp24, 1728);

	/* E_4^3 - E_6^2 = 1728 q prod (1 - q^m)^24, 两侧同为整系数多项式, 逐步比较 */
	report_flag("E4^3 - E6^2 = 1728 q prod (1-q^m)^24", fmpz_poly_equal(ident, qp24));

	/* j = 1728 E_4^3 / (E_4^3 - E_6^2) = E_4^3 / (q prod (1 - q^m)^24) */
	fmpz_poly_div_series(jser, e4cub, p24, SERIES_TERMS);

	printf("  tau(1..10)   = ");
	print_coeffs(p24, 0, 10);
	printf("\n");
	printf("  j q-series   = ");
	print_coeffs(jser, 0, 8);
	printf(" + ...\n");

	ok = 1;
	for (n = 0; n < 10; n++) {
		if (fmpz_cmp_si(tau + n + 1, tau_table[n]) != 0) {
			ok = 0;
		}
	}
	report_flag("tau(1..10) match published values", ok);

	ok = 1;
	for (n = 0; n < J_TERMS; n++) {
		fmpz_poly_get_coeff_fmpz(lhs, jser, n);
		if (fmpz_cmp_si(lhs, j_table[n]) != 0) {
			ok = 0;
		}
	}
	report_flag("j = q^-1 + 744 + 196884 q + ...", ok);

	/* tau(mn) = tau(m) tau(n), m 与 n 互素 */
	ok = 1;
	for (m = 2; m <= TAU_MAX; m++) {
		for (n = 2; m * n <= TAU_MAX; n++) {
			if (n_gcd((ulong)m, (ulong)n) != 1) {
				continue;
			}
			fmpz_mul(lhs, tau + m, tau + n);
			if (fmpz_cmp(lhs, tau + m * n) != 0) {
				ok = 0;
			}
		}
	}
	report_flag("tau(mn) = tau(m) tau(n), gcd(m,n) = 1", ok);

	/* tau(p^(k+1)) = tau(p) tau(p^k) - p^11 tau(p^(k-1)), 素数 p 与 p^(k+1) 在上界内 */
	ok = 1;
	for (p = 2; p <= (ulong)TAU_MAX; p++) {
		if (!n_is_prime(p)) {
			continue;
		}
		fmpz_ui_pow_ui(p11, p, 11);
		pk = p;
		while (pk <= (ulong)TAU_MAX / p) {
			fmpz_mul(lhs, tau + p, tau + pk);
			fmpz_mul(rhs, p11, tau + pk / p);
			fmpz_sub(lhs, lhs, rhs);
			if (fmpz_cmp(lhs, tau + pk * p) != 0) {
				ok = 0;
			}
			pk *= p;
		}
	}
	report_flag("tau(p^(k+1)) = tau(p)tau(p^k) - p^11 tau(p^(k-1))", ok);

	/*
	 * Deligne 界: |tau(p)| <= 2 p^(11/2), 两端平方后为整数比较。该界即 Ramanujan
	 * 猜想的结论, 与上面逐项核对过的系数相互独立。
	 */
	ok = 1;
	for (p = 2; p <= (ulong)TAU_MAX; p++) {
		if (!n_is_prime(p)) {
			continue;
		}
		fmpz_mul(lhs, tau + p, tau + p);
		fmpz_ui_pow_ui(rhs, p, 11);
		fmpz_mul_ui(rhs, rhs, 4);
		if (fmpz_cmp(lhs, rhs) > 0) {
			ok = 0;
		}
	}
	report_flag("tau(p)^2 <= 4 p^11", ok);

	/* tau(n) = sigma_11(n) mod 691 */
	ok = 1;
	for (n = 1; n <= TAU_MAX; n++) {
		fmpz_mod_ui(mod_a, tau + n, 691);
		divisor_sigma(rhs, (ulong)n, 11);
		fmpz_mod_ui(mod_b, rhs, 691);
		if (fmpz_cmp(mod_a, mod_b) != 0) {
			ok = 0;
		}
	}
	report_flag("tau(n) = sigma_11(n) mod 691", ok);

	fmpz_clear(mod_b);
	fmpz_clear(mod_a);
	fmpz_clear(p11);
	fmpz_clear(rhs);
	fmpz_clear(lhs);
	fmpz_poly_clear(jser);
	fmpz_poly_clear(qp24);
	fmpz_poly_clear(p24);
	fmpz_poly_clear(unit);
	fmpz_poly_clear(ident);
	fmpz_poly_clear(e6sq);
	fmpz_poly_clear(e4cub);
	fmpz_poly_clear(e6);
	fmpz_poly_clear(e4);
}

/* ---------- 模函数取值 ---------- */

/* q = exp(2 pi i tau), 即模形式 q 展开所用的 q。 */
static void nome(acb_t q, const acb_t tau) {
	acb_t e, cpi;

	acb_init(e);
	acb_init(cpi);
	acb_const_pi(cpi, PREC);
	acb_mul(e, tau, cpi, PREC);
	acb_mul_2exp_si(e, e, 1);
	acb_mul_onei(e, e);
	acb_exp(q, e, PREC);
	acb_clear(cpi);
	acb_clear(e);
}

/* j 的精确 q 级数的前 SERIES_TERMS 个系数, 即 j = q^-1 sum c_n q^n 的 c_n。 */
static void j_series_coeffs(fmpz_poly_t jser) {
	fmpz_poly_t e4, e6, e4cub, e6sq, unit, p24;

	fmpz_poly_init(e4);
	fmpz_poly_init(e6);
	fmpz_poly_init(e4cub);
	fmpz_poly_init(e6sq);
	fmpz_poly_init(unit);
	fmpz_poly_init(p24);
	eisenstein_series(e4, e6, SERIES_TERMS);
	poly_pow_trunc(e4cub, e4, 3, SERIES_TERMS);
	poly_pow_trunc(e6sq, e6, 2, SERIES_TERMS);
	euler_product_poly(unit, SERIES_TERMS);
	poly_pow_trunc(p24, unit, 24, SERIES_TERMS);
	fmpz_poly_div_series(jser, e4cub, p24, SERIES_TERMS);
	fmpz_poly_clear(p24);
	fmpz_poly_clear(unit);
	fmpz_poly_clear(e6sq);
	fmpz_poly_clear(e4cub);
	fmpz_poly_clear(e6);
	fmpz_poly_clear(e4);
}

static void check_modular_functions(const fmpz *tau_coeff) {
	acb_t tau, tau1, eta_a, eta_b, lhs, factor, cpi, tmp, q, qinv, sum;
	acb_t g2, g3, g2cub, g3sq, disc, num, j_alt, j_ref, delta, twopi12, rho;
	arb_t pi, quarter, root, ref, qabs, qn, series, diff;
	fmpz_poly_t jser;
	fmpz_t c;
	acb_ptr G;
	slong n;

	acb_init(tau);
	acb_init(tau1);
	acb_init(eta_a);
	acb_init(eta_b);
	acb_init(lhs);
	acb_init(factor);
	acb_init(cpi);
	acb_init(tmp);
	acb_init(q);
	acb_init(qinv);
	acb_init(sum);
	acb_init(g2);
	acb_init(g3);
	acb_init(g2cub);
	acb_init(g3sq);
	acb_init(disc);
	acb_init(num);
	acb_init(j_alt);
	acb_init(j_ref);
	acb_init(delta);
	acb_init(twopi12);
	acb_init(rho);
	arb_init(pi);
	arb_init(quarter);
	arb_init(root);
	arb_init(ref);
	arb_init(qabs);
	arb_init(qn);
	arb_init(series);
	arb_init(diff);
	fmpz_poly_init(jser);
	fmpz_init(c);
	G = _acb_vec_init(2);

	acb_const_pi(cpi, PREC);
	arb_const_pi(pi, PREC);

	/*
	 * eta(i) = Gamma(1/4) / (2 pi^(3/4))。参考值由同一精度的弧函数与幂函数算出,
	 * 不含十进制字面量。
	 */
	acb_set_d_d(tau, 0.0, 1.0);
	acb_modular_eta(eta_a, tau, PREC);
	arb_set_ui(quarter, 1);
	arb_div_ui(quarter, quarter, 4, PREC);
	arb_gamma(ref, quarter, PREC);
	arb_pow_ui(root, pi, 3, PREC);
	arb_root_ui(root, root, 4, PREC);
	arb_mul_2exp_si(root, root, 1);
	arb_div(ref, ref, root, PREC);
	report_arb(
		"eta(i) = Gamma(1/4) / (2 pi^(3/4))",
		arb_close(acb_realref(eta_a), ref, CHECK_DIGITS),
		acb_realref(eta_a));

	/* eta(tau + 1) = exp(i pi / 12) eta(tau), 取半平面内的一般点 */
	acb_set_d_d(tau, 0.3, 1.1);
	acb_modular_eta(eta_a, tau, PREC);
	acb_add_ui(tau1, tau, 1, PREC);
	acb_modular_eta(eta_b, tau1, PREC);
	/* 因子 exp(i pi / 12) 算出后再乘到 eta(tau) 上 */
	acb_div_ui(factor, cpi, 12, PREC);
	acb_mul_onei(factor, factor);
	acb_exp(factor, factor, PREC);
	acb_mul(factor, factor, eta_a, PREC);
	report_acb(
		"eta(tau+1) = e^(i pi/12) eta(tau)",
		acb_close(eta_b, factor, CHECK_DIGITS),
		eta_b);

	/* eta(-1/tau) = sqrt(-i tau) eta(tau), 取 tau = 2i 使根式无分支歧义 */
	acb_set_d_d(tau, 0.0, 2.0);
	acb_modular_eta(eta_a, tau, PREC);
	acb_inv(tmp, tau, PREC);
	acb_neg(tmp, tmp);
	acb_modular_eta(eta_b, tmp, PREC);
	acb_mul_onei(lhs, tau);
	acb_neg(lhs, lhs);
	acb_sqrt(lhs, lhs, PREC);
	acb_mul(lhs, lhs, eta_a, PREC);
	report_acb(
		"eta(-1/tau) = sqrt(-i tau) eta(tau)",
		acb_close(eta_b, lhs, CHECK_DIGITS),
		eta_b);

	/* j 的特殊值: j(i) = 1728, j(2i) = 287496, j(rho) = 0 */
	acb_set_d_d(tau, 0.0, 1.0);
	acb_modular_j(j_ref, tau, PREC);
	acb_set_ui(lhs, 1728);
	report_acb("j(i) = 1728", acb_close(j_ref, lhs, CHECK_DIGITS), j_ref);

	acb_set_d_d(tau, 0.0, 2.0);
	acb_modular_j(j_ref, tau, PREC);
	acb_set_ui(lhs, 287496);
	report_acb("j(2i) = 287496", acb_close(j_ref, lhs, CHECK_DIGITS), j_ref);

	arb_sqrt_ui(root, 3, PREC);
	arb_mul_2exp_si(root, root, -1);
	acb_set_d(rho, -0.5);
	arb_set(acb_imagref(rho), root);
	acb_modular_j(j_ref, rho, PREC);
	report_acb("j(rho) = 0, rho = e^(2 pi i/3)", acb_small(j_ref, CHECK_DIGITS), j_ref);

	/*
	 * 一般点上 j 与 Eisenstein 级数的关系。acb_modular_eisenstein 给出未归一化的格和
	 * G_4 与 G_6, 而
	 *
	 *   g_2 = 60 G_4,  g_3 = 140 G_6,  j = 1728 g_2^3 / (g_2^3 - 27 g_3^2)
	 */
	acb_set_d_d(tau, 0.3, 1.1);
	acb_modular_eisenstein(G, tau, 2, PREC);
	acb_mul_ui(g2, G + 0, 60, PREC);
	acb_mul_ui(g3, G + 1, 140, PREC);
	acb_pow_ui(g2cub, g2, 3, PREC);
	acb_pow_ui(g3sq, g3, 2, PREC);
	acb_mul_ui(g3sq, g3sq, 27, PREC);
	acb_sub(disc, g2cub, g3sq, PREC);
	acb_mul_ui(num, g2cub, 1728, PREC);
	acb_div(j_alt, num, disc, PREC);
	acb_modular_j(j_ref, tau, PREC);
	report_acb(
		"j = 1728 g2^3 / (g2^3 - 27 g3^2)", acb_close(j_alt, j_ref, CHECK_DIGITS), j_alt);

	/* g_2^3 - 27 g_3^2 = (2 pi)^12 Delta(tau) */
	acb_pow_ui(twopi12, cpi, 12, PREC);
	acb_mul_2exp_si(twopi12, twopi12, 12);
	acb_modular_delta(delta, tau, PREC);
	acb_mul(lhs, twopi12, delta, PREC);
	report_acb(
		"g2^3 - 27 g3^2 = (2 pi)^12 Delta", acb_close(disc, lhs, CHECK_DIGITS), disc);

	/*
	 * Delta(i) 与精确级数比较。tau = i 时 q = exp(-2 pi) 为实数, 级数为实值。
	 * Delta 的 q 展开从 q 的一次项开始, 没有常数项。
	 */
	acb_set_d_d(tau, 0.0, 1.0);
	acb_modular_delta(delta, tau, PREC);
	arb_mul_2exp_si(qabs, pi, 1);
	arb_neg(qabs, qabs);
	arb_exp(qabs, qabs, PREC);
	arb_set(qn, qabs);
	arb_zero(series);
	for (n = 1; n <= SERIES_TAKE && n <= TAU_MAX; n++) {
		arb_addmul_fmpz(series, qn, tau_coeff + n, PREC);
		arb_mul(qn, qn, qabs, PREC);
	}
	report_arb(
		"Delta(i) = sum tau(n) e^(-2 pi n)",
		arb_close(acb_realref(delta), series, CHECK_DIGITS),
		acb_realref(delta));

	/*
	 * j(tau) 与精确级数比较。级数的系数由整系数幂级数除法算出, 求值在球算术下进行,
	 * 与上面直接用 acb_modular_j 是两条途径。
	 */
	j_series_coeffs(jser);
	acb_set_d_d(tau, 0.3, 1.1);
	nome(q, tau);
	acb_inv(qinv, q, PREC);
	fmpz_poly_get_coeff_fmpz(c, jser, SERIES_TAKE);
	acb_set_fmpz(sum, c);
	for (n = SERIES_TAKE - 1; n >= 0; n--) {
		fmpz_poly_get_coeff_fmpz(c, jser, n);
		acb_mul(sum, sum, q, PREC);
		acb_add_fmpz(sum, sum, c, PREC);
	}
	acb_mul(sum, sum, qinv, PREC);
	acb_modular_j(j_ref, tau, PREC);
	report_acb(
		"j(tau) = q^-1 sum c_n q^n, c_n exact", acb_close(sum, j_ref, CHECK_DIGITS), sum);

	/* 上式两侧之差, 打印出来供直接读出余量 */
	acb_sub(lhs, sum, j_ref, PREC);
	acb_abs(diff, lhs, PREC);
	printf("  j series vs acb_modular_j difference |.| = ");
	arb_printn(diff, 3, ARB_STR_MORE);
	printf("\n");

	_acb_vec_clear(G, 2);
	fmpz_clear(c);
	fmpz_poly_clear(jser);
	arb_clear(diff);
	arb_clear(series);
	arb_clear(qn);
	arb_clear(qabs);
	arb_clear(ref);
	arb_clear(root);
	arb_clear(quarter);
	arb_clear(pi);
	acb_clear(rho);
	acb_clear(twopi12);
	acb_clear(delta);
	acb_clear(j_ref);
	acb_clear(j_alt);
	acb_clear(num);
	acb_clear(disc);
	acb_clear(g3sq);
	acb_clear(g2cub);
	acb_clear(g3);
	acb_clear(g2);
	acb_clear(sum);
	acb_clear(qinv);
	acb_clear(q);
	acb_clear(tmp);
	acb_clear(cpi);
	acb_clear(factor);
	acb_clear(lhs);
	acb_clear(eta_b);
	acb_clear(eta_a);
	acb_clear(tau1);
	acb_clear(tau);
}

/* ---------- Euler 乘积 ---------- */

/*
 * 素数 p 的局部因子
 *
 *   sum_{k>=0} tau(p^k) p^(-k s),  k 取到 p^k <= limit
 *
 * tau(p^k) 由 Hecke 递推得到, 递推只需级数给出的 tau(p)。级数按 p^(-s) 的幂逐项相加,
 * 每一项由上一项乘以 p^(-s) 得到。
 */
static void euler_local_factor(arb_t out, ulong p, const fmpz *tau, ulong limit) {
	arb_t pinv, qk;
	fmpz_t prev, cur, next, scaled, p11;
	ulong pk;

	arb_init(pinv);
	arb_init(qk);
	fmpz_init(prev);
	fmpz_init(cur);
	fmpz_init(next);
	fmpz_init(scaled);
	fmpz_init(p11);

	arb_set_ui(pinv, p);
	arb_pow_ui(pinv, pinv, EULER_S, PREC);
	arb_inv(pinv, pinv, PREC);

	fmpz_ui_pow_ui(p11, p, 11);
	arb_one(out);
	arb_one(qk);
	fmpz_one(prev);
	fmpz_set(cur, tau + p);
	pk = p;
	while (pk <= limit) {
		arb_mul(qk, qk, pinv, PREC);
		arb_addmul_fmpz(out, qk, cur, PREC);
		fmpz_mul(next, tau + p, cur);
		fmpz_mul(scaled, p11, prev);
		fmpz_sub(next, next, scaled);
		fmpz_set(prev, cur);
		fmpz_set(cur, next);
		if (pk > limit / p) {
			break;
		}
		pk *= p;
	}

	fmpz_clear(p11);
	fmpz_clear(scaled);
	fmpz_clear(next);
	fmpz_clear(cur);
	fmpz_clear(prev);
	arb_clear(qk);
	arb_clear(pinv);
}

static void check_euler_product(const fmpz *tau) {
	arb_t lhs, rhs, factor, ninv, diff, tol;
	ulong p;
	slong n;
	int ok;

	arb_init(lhs);
	arb_init(rhs);
	arb_init(factor);
	arb_init(ninv);
	arb_init(diff);
	arb_init(tol);

	/* 左侧: 截断的 Dirichlet 级数 sum_{n<=N} tau(n) n^-s */
	arb_zero(lhs);
	for (n = 1; n <= TAU_MAX; n++) {
		arb_set_ui(ninv, (ulong)n);
		arb_pow_ui(ninv, ninv, EULER_S, PREC);
		arb_inv(ninv, ninv, PREC);
		arb_addmul_fmpz(lhs, ninv, tau + n, PREC);
	}

	/* 右侧: 各素数局部因子之积 */
	arb_one(rhs);
	for (p = 2; p <= (ulong)TAU_MAX; p++) {
		if (!n_is_prime(p)) {
			continue;
		}
		euler_local_factor(factor, p, tau, (ulong)TAU_MAX);
		arb_mul(rhs, rhs, factor, PREC);
	}

	arb_distance(diff, lhs, rhs);
	tolerance(tol, EULER_DIGITS);
	ok = arb_lt(diff, tol);
	report_diff("sum tau(n) n^-13 = prod_p Euler factors", ok, diff);

	arb_clear(tol);
	arb_clear(diff);
	arb_clear(ninv);
	arb_clear(factor);
	arb_clear(rhs);
	arb_clear(lhs);
}

int main(void) {
	fmpz *tau;

	printf(
		"modular-form: eta, j, Delta and Ramanujan tau, FLINT arb/acb and exact"
		" q-series, prec = %d bits\n",
		(int)PREC);

	tau = _fmpz_vec_init(SERIES_TERMS);
	tau_from_eta_product(tau);

	printf("exact integer q-series, terms up to q^%d\n", (int)(SERIES_TERMS - 1));
	check_exact_series(tau);

	printf("modular functions and special values\n");
	check_modular_functions(tau);

	printf("Euler product\n");
	check_euler_product(tau);

	_fmpz_vec_clear(tau, SERIES_TERMS);

	printf(
		"modular-form: %d checks, %d failed\n",
		checks_passed + checks_failed,
		checks_failed);

	/* 释放 FLINT 与 MPFR 的全局缓存, 否则这些内存由库持有到进程结束 */
	flint_cleanup_master();
	return checks_failed == 0 ? 0 : 1;
}
