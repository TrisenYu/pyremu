/* stress_payload.c - 飞地压力测试载荷 (musl 静态链接)
 *
 * 编译: make stress_payload
 *
 * SPDX-License-Identifier: MIT
 */

#include "stress_common.h"

/* ---- 硬编码参数 ---- */

#define FIB_N		 80000
#define PRIME_MAX	 30000
#define MALLOC_COUNT 300
#define MMAP_PAGES	 6
#define MIXED_ROUNDS 3

int main(void) {
	int rc = 0;

	for (int round = 0; round < MIXED_ROUNDS; round++) {
		printf(
			"[stress] round %d: fib(%u)=%lu\n"
			"[stress] round %d: primes(<=%u)=%lu\n",
			round,
			FIB_N,
			stress_fib(FIB_N),
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
