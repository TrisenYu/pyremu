/* stress_payload.c - 飞地压力测试载荷 (musl 静态链接)
 *
 * 每个飞地内执行固定的 CPU + 内存压力操作，参数硬编码。
 * 通过大量创建飞地 (2/20/200/2000/20000) 测试:
 *   - 槽位分配/释放的正确性
 *   - 上下文切换 (host ↔ enclave) 的稳定性
 *   - PMP 重配置 (activate_lpmp) 的并发安全
 *   - brk/mmap ecall 路径的可靠性
 *   - mfence.did TLB/L2 按域刷新的隔离性
 *
 * 编译: make stress_payload
 *
 * SPDX-License-Identifier: GPL-2.0
 */

#include "stress_common.h"

/* ---- 硬编码参数 ---- */

#define FIB_N		 80000
#define PRIME_MAX	 30000
#define MALLOC_COUNT 300
#define MMAP_PAGES	 6
#define MIXED_ROUNDS 3

/* ================================================================
 *  main — 按固定顺序执行全部操作, 参数硬编码
 * ================================================================ */

int main(void) {
	int rc = 0;

	for (int round = 0; round < MIXED_ROUNDS; round++) {
		printf(
			"[stress] round %d: fib(%u)=%lu\n",
			round,
			FIB_N,
			stress_fib(FIB_N));

		printf(
			"[stress] round %d: primes(<=%u)=%lu\n",
			round,
			PRIME_MAX,
			stress_prime_sieve(PRIME_MAX));

		stress_malloc_cycle(MALLOC_COUNT);
		printf("[stress] round %d: malloc(%d) done\n", round, (unsigned)MALLOC_COUNT);

		int mmap_rc = stress_mmap_rw(MMAP_PAGES);
		if (mmap_rc) {
			printf("[stress] round %d: mmap FAILED\n", round);
			rc = 1;
		} else {
			printf("[stress] round %d: mmap(%d) OK\n", round, (unsigned)MMAP_PAGES);
		}
	}

	printf("[stress] all rounds done, rc=%d\n", rc);
	return rc;
}
