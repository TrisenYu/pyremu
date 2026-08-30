/* victim_cache_payload.c - 可信应用: 流密码密钥派生
 *
 * 一个正常的密码学计算: 基于 ChaCha quarter-round 的密钥派生函数。
 * 从初始状态派生 10^5 字节的确定性密钥流, 写入工作缓冲区。
 *
 * 缓存访问是计算的副产品, 没有任何刻意的侧信道行为。
 * 中间数据可用相同初始状态离线验算。
 *
 * 编译: 见 Makefile (musl 飞地编译链)
 * SPDX-License-Identifier: GPL-2.0
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>

#define WORK_AREA_VA  0x20000000UL
#define WORK_AREA_SZ  (1UL << 20) /* 1 MiB 工作区 */
#define STREAM_LEN    100000

static inline uint32_t rotl(uint32_t x, int n) {
	return (x << n) | (x >> (32 - n));
}

static inline void quarter_round(
	uint32_t *a, uint32_t *b, uint32_t *c, uint32_t *d)
{
	*a += *b; *d ^= *a; *d = rotl(*d, 16);
	*c += *d; *b ^= *c; *b = rotl(*b, 12);
	*a += *b; *d ^= *a; *d = rotl(*d, 8);
	*c += *d; *b ^= *c; *b = rotl(*b, 7);
}

int main(void) {
	volatile uint8_t *buf = (volatile uint8_t *)mmap(
		(void *)WORK_AREA_VA, WORK_AREA_SZ,
		PROT_READ | PROT_WRITE,
		MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED,
		-1, 0);
	if (buf == MAP_FAILED) {
		printf("[victim] mmap failed\n");
		return 1;
	}
	memset((void *)buf, 0, WORK_AREA_SZ);

	/* ChaCha 初始矩阵 */
	uint32_t state[4] = {
		0x61707865u, 0x3320646Eu,
		0x79622D32u, 0x6B206574u,
	};

	/* 派生密钥流, 写入工作缓冲区 */
	for (int i = 0; i < STREAM_LEN; i++) {
		state[0] += (uint32_t)i;
		quarter_round(&state[0], &state[1], &state[2], &state[3]);
		buf[i % WORK_AREA_SZ] = (uint8_t)(state[0] & 0xFF);
	}

	printf("[victim] KDF done: %d bytes, state=0x%08x\n",
		STREAM_LEN, state[0]);

	/* 输出末尾 32 字节供离线对照 */
	printf("[victim] tail=");
	for (int i = STREAM_LEN - 32; i < STREAM_LEN; i++) {
		printf("%02x", buf[i % WORK_AREA_SZ]);
	}
	printf("\n");

	munmap((void *)buf, WORK_AREA_SZ);
	return 0;
}
