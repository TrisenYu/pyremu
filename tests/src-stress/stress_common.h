/* stress_common.h - 飞地压力测试公共函数 (musl 静态链接)
 *
 * 提供 CPU 密集型和内存压力操作的公共函数，供各 payload 复用。
 * 全部声明为 static inline，避免多目标链接冲突。
 *
 * SPDX-License-Identifier: GPL-2.0
 */

#ifndef STRESS_COMMON_H
#define STRESS_COMMON_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- CPU: 斐波那契 (迭代, 模 2^64) ---- */

static inline uint64_t stress_fib(uint64_t n) {
    if (n <= 1) return n;
    uint64_t a = 0, b = 1;
    for (uint64_t i = 2; i <= n; i++) {
        uint64_t t = b;
        b = a + b;
        a = t;
    }
    return b;
}

/* ---- CPU: 素数筛 (Eratosthenes) ---- */

static inline uint64_t stress_prime_sieve(uint64_t max) {
    if (max < 2) return 0;
    char *sieve = (char *)calloc(max + 1, 1);
    if (!sieve) return 0;
    uint64_t count = 0;
    for (uint64_t i = 2; i <= max; i++) {
        if (!sieve[i]) {
            count++;
            for (uint64_t j = i * i; j <= max; j += i) sieve[j] = 1;
        }
    }
    free(sieve);
    return count;
}

/* ---- 内存: malloc/free 压力循环 ---- */

static inline void stress_malloc_cycle(int count) {
    void **ptrs = (void **)malloc((size_t)count * sizeof(void *));
    if (!ptrs) return;
    for (int i = 0; i < count; i++) {
        size_t sz = (size_t)(16 + (i % 512) * 8);
        ptrs[i] = malloc(sz);
        if (ptrs[i]) memset(ptrs[i], 0xAB + (i & 0xF), sz);
    }
    for (int i = count - 1; i >= 0; i--) free(ptrs[i]);
    free(ptrs);
}

/* ---- 内存: mmap 匿名映射读写验证 ---- */

static inline int stress_mmap_rw(int pages) {
    size_t sz = (size_t)pages * 4096;
    void *addr = mmap(NULL, sz, PROT_READ | PROT_WRITE,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (addr == MAP_FAILED) return -1;
    for (int i = 0; i < pages; i++) ((volatile char *)addr)[i * 4096] = (char)i;
    int ok = 1;
    for (int i = 0; i < pages; i++) {
        if (((volatile char *)addr)[i * 4096] != (char)i) ok = 0;
    }
    munmap(addr, sz);
    return ok ? 0 : -1;
}

/* ---- TLB: sfence.vma 全刷新 (RISC-V) ---- */

static inline void stress_tlb_flush_all(void) {
    __asm__ volatile("sfence.vma zero, zero" ::: "memory");
}

#ifdef __cplusplus
}
#endif

#endif /* STRESS_COMMON_H */
