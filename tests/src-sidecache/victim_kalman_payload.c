/* victim_kalman_payload.c - 可信应用: 1-D Kalman 滤波器
 *
 * 模拟可信导航/制导系统中的 Kalman 状态估计:
 *   状态向量: [位置, 速度]
 *   观测模型: 直接观测位置, 带高斯噪声
 *   执行 10^5 步预测-更新迭代, 记录中间状态估计
 *
 * 所有中间数据可从相同初始条件 + LCG 种子离线重算验证。
 *
 * 编译: 见 Makefile (musl 飞地编译链)
 * SPDX-License-Identifier: GPL-2.0
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>

#define WORK_VA  0x20000000UL
#define WORK_SZ  (1UL << 20) /* 1 MiB */
#define N_STEPS  100000

#define LCG_A    1103515245u
#define LCG_C    12345u

int main(void) {
	volatile uint8_t *buf = (volatile uint8_t *)mmap(
		(void *)WORK_VA, WORK_SZ,
		PROT_READ | PROT_WRITE,
		MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED,
		-1, 0);
	if (buf == MAP_FAILED) {
		printf("[victim-kalman] mmap failed\n");
		return 1;
	}
	memset((void *)buf, 0, WORK_SZ);

	/* 状态 [位置, 速度] */
	double x_pos = 0.0, x_vel = 1.0;
	/* 协方差 (2x2) */
	double P00 = 1.0, P01 = 0.0, P10 = 0.0, P11 = 1.0;

	const double Q  = 0.01;   /* 过程噪声 */
	const double R  = 0.5;    /* 观测噪声 */
	const double dt = 0.1;    /* 时间步长 */

	uint32_t lcg = 42u;

	/* 中间状态日志 (在工作区内) */
	volatile double *x_log = (volatile double *)(buf + 0x1000);

	for (int k = 0; k < N_STEPS; k++) {
		/* 生成测量值 z = 真实位置 + 噪声 */
		lcg = lcg * LCG_A + LCG_C;
		double noise = ((double)(lcg & 0xFFFF) / 65536.0 - 0.5) * 2.0;
		double z = x_pos + noise * R;

		/* 预测步 */
		double xp_pos = x_pos + dt * x_vel;
		double xp_vel = x_vel;
		double Pp00 = P00 + dt * (P10 + P01) + dt * dt * P11 + Q;
		double Pp01 = P01 + dt * P11;
		double Pp10 = P10 + dt * P11;
		double Pp11 = P11 + Q;

		/* 更新步 */
		double y   = z - xp_pos;          /* 残差 */
		double S   = Pp00 + R;            /* 残差协方差 */
		double Si  = 1.0 / S;
		double K0  = Pp00 * Si;           /* Kalman gain [0] */
		double K1  = Pp10 * Si;           /* Kalman gain [1] */

		x_pos = xp_pos + K0 * y;
		x_vel = xp_vel + K1 * y;
		P00 = (1.0 - K0) * Pp00;
		P01 = (1.0 - K0) * Pp01;
		P10 = -K1 * Pp00 + Pp10;
		P11 = -K1 * Pp01 + Pp11;

		x_log[k] = x_pos;
	}

	printf("[victim-kalman] %d steps, pos=%.6f vel=%.6f P00=%.6f\n",
		N_STEPS, x_pos, x_vel, P00);

	/* 末尾 16 个状态的 hex 表示 */
	printf("[victim-kalman] tail=");
	for (int i = N_STEPS - 16; i < N_STEPS; i++) {
		uint64_t bits;
		double v = x_log[i];
		memcpy(&bits, &v, 8);
		printf("%016lx", (unsigned long)bits);
	}
	printf("\n");

	munmap((void *)buf, WORK_SZ);
	return 0;
}
