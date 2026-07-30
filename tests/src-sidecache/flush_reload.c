/* flush_reload.c — Flush+Reload 缓存侧信道攻击演示 (Linux userspace)
 *
 * 同进程内模拟 sender/receiver 两方通过 L2 缓存状态传递 64-bit secret.
 * flush 不依赖 CBO 指令, 使用同组地址填充逐出 (eviction-based):
 *   对目标行所在缓存组, 访问 W+1 个不同 tag 的同组行, 挤出所有路.
 *
 * 缓存几何 (pyremu L2): 256 KiB, 64 B/line, 4-way, 1024 sets
 * 同组地址间距 = sets * line_size = 1024 * 64 = 65536 (64 KiB)
 *
 * 在真实硬件上 rdcycle 可分辨 hit (~10c) vs miss (~100c).
 * 在当前模拟器上 rdcycle 不反映访存延迟 → BER 预期 ~50%.
 * 方法论正确性不受影响, 交叉编译到真实 RISC-V 硬件即可验证.
 *
 * 编译 (在模拟器内 Linux 上):
 *   gcc -O3 -static flush_reload.c -o flush_reload -lm
 *
 * 或交叉编译:
 *   riscv64-linux-gnu-gcc -O3 -static flush_reload.c -o flush_reload
 *
 * SPDX-License-Identifier: GPL-2.0
 */

#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* ---- 缓存几何参数 (pyremu L2, 可调) ---- */
#define CACHE_LINE   64
#define L2_KB        256
#define L2_WAYS      4
#define L2_SETS      ((L2_KB * 1024) / (CACHE_LINE * L2_WAYS))  /* 1024 */
#define SET_STRIDE   (L2_SETS * CACHE_LINE)                      /* 65536 */
#define N_BITS       64
#define EVICT_WAYS   8          /* 逐出填充路数 (> WAYS 确保逐出) */
#define CALIBRATE_N  50         /* 校准采样数 */

/* ---- 硬件原语 ---- */

static inline uint64_t rdcycle(void) {
    uint64_t cyc;
    __asm__ volatile("rdcycle %0" : "=r"(cyc));
    return cyc;
}

static inline void fence_all(void) {
    __asm__ volatile("fence iorw, iorw" ::: "memory");
}

/* ---- 缓存逐出 (eviction-based flush) ---- */

static void flush_addr(unsigned char *probe_base, size_t base_sz,
                       unsigned char *target) {
    uintptr_t t = (uintptr_t)target;
    for (int w = 0; w < EVICT_WAYS; w++) {
        uintptr_t evict = t + (uintptr_t)((w + 1) * SET_STRIDE);
        if (evict >= (uintptr_t)probe_base
            && evict < (uintptr_t)(probe_base + base_sz)) {
            __asm__ volatile("" :: "r"(*(volatile unsigned char *)evict));
        }
    }
    fence_all();
}

/* ---- 单次加载计时 ---- */

static uint64_t time_load(volatile unsigned char *addr) {
    fence_all();
    uint64_t start = rdcycle();
    __asm__ volatile("lb %0, 0(%1)"
                     : "=r"(*(volatile unsigned char *)addr)
                     : "r"(addr));
    fence_all();
    return rdcycle() - start;
}

/* ---- 阈值校准 ---- */

static uint64_t calibrate(unsigned char *arr, size_t arr_sz) {
    uint64_t hit_sum = 0, miss_sum = 0;

    /* 命中: 反复访问同一行 */
    for (int i = 0; i < CALIBRATE_N; i++) {
        fence_all();
        (void)*(volatile unsigned char *)arr;  /* prime */
        fence_all();
        hit_sum += time_load(arr);
    }

    /* 缺失: 先逐出再测 */
    for (int i = 0; i < CALIBRATE_N; i++) {
        flush_addr(arr, arr_sz, arr);
        miss_sum += time_load(arr);
    }

    uint64_t hit_avg  = hit_sum  / CALIBRATE_N;
    uint64_t miss_avg = miss_sum / CALIBRATE_N;
    printf("  hit  avg: %6llu cycles\n", (unsigned long long)hit_avg);
    printf("  miss avg: %6llu cycles\n", (unsigned long long)miss_avg);
    printf("  diff:     %6llu cycles (%.1fx)\n",
           (unsigned long long)(miss_avg > hit_avg ? miss_avg - hit_avg : 0),
           hit_avg ? (double)miss_avg / (double)hit_avg : 0.0);
    return (hit_avg + miss_avg) / 2;
}

/* ---- Receiver: 从缓存定时恢复 secret ---- */

static uint64_t receiver_probe(unsigned char *arr, int n_bits,
                               uint64_t threshold) {
    uint64_t recovered = 0;
    for (int i = 0; i < n_bits; i++) {
        if (time_load(&arr[i * SET_STRIDE]) < threshold)
            recovered |= (1ULL << i);
    }
    return recovered;
}

/* ---- main ---- */

int main(void) {
    printf("=== Flush+Reload Cache Side-Channel ===\n");
    printf("L2: %d KiB  line: %d B  ways: %d  sets: %d  stride: %d KiB\n",
           L2_KB, CACHE_LINE, L2_WAYS, L2_SETS, SET_STRIDE / 1024);

    /* 分配探测数组: STRIDE * N_BITS + 额外空间供逐出 */
    size_t arr_sz = SET_STRIDE * (N_BITS + EVICT_WAYS + 4);
    unsigned char *arr = (unsigned char *)malloc(arr_sz);
    if (!arr) { printf("[FAIL] malloc(%zu)\n", arr_sz); return 1; }
    memset(arr, 0xCC, arr_sz);
    printf("probe array: %zu bytes at %p\n\n", arr_sz, (void *)arr);

    /* ---- 1. 阈值校准 ---- */
    printf("--- Calibration ---\n");
    uint64_t threshold = calibrate(arr, arr_sz);
    printf("threshold = %llu\n\n", (unsigned long long)threshold);

    /* ---- 2. 攻击演示 ---- */
    printf("--- Covert Channel ---\n");

    static const uint64_t secrets[] = {
        0x0000000000000001ULL,
        0x8000000000000000ULL,
        0xAAAAAAAAAAAAAAAAULL,
        0x5555555555555555ULL,
        0xFFFFFFFFFFFFFFFFULL,
        0x0000000000000000ULL,
        0xDEADBEEFCAFEBABEULL,
    };
    int n_secrets = sizeof(secrets) / sizeof(secrets[0]);
    int ok_count = 0;
    int total_bit_errs = 0;

    for (int s = 0; s < n_secrets; s++) {
        uint64_t secret = secrets[s];

        /* Receiver: flush 全部探测行 */
        for (int i = 0; i < N_BITS; i++)
            flush_addr(arr, arr_sz, &arr[i * SET_STRIDE]);

        /* Sender: 按 secret 编码 — 访问对应行将其带入缓存 */
        for (int i = 0; i < N_BITS; i++)
            if (secret & (1ULL << i))
                __asm__ volatile("" :: "r"(*(volatile unsigned char *)
                                            &arr[i * SET_STRIDE]));
        fence_all();

        /* Receiver: 探测 */
        uint64_t recovered = receiver_probe(arr, N_BITS, threshold);

        uint64_t errs   = secret ^ recovered;
        int bit_errs    = __builtin_popcountll(errs);
        int ok          = (bit_errs == 0);
        ok_count       += ok;
        total_bit_errs += bit_errs;

        printf("  [%d] secret=0x%016llX recovered=0x%016llX bit_errs=%d %s\n",
               s, (unsigned long long)secret, (unsigned long long)recovered,
               bit_errs, ok ? "OK" : "FAIL");
    }

    int total_bits = n_secrets * N_BITS;
    printf("\n=== Result ===\n");
    printf("Secrets recovered: %d/%d\n", ok_count, n_secrets);
    printf("Bit errors:        %d/%d (%.1f%%)\n",
           total_bit_errs, total_bits,
           100.0 * (double)total_bit_errs / (double)total_bits);

    /* 如果 bit 错误率显著低于 50%, 说明信道有信号 */
    double ber = (double)total_bit_errs / (double)total_bits;
    if (ok_count == n_secrets)
        printf("Verdict: PASS — all secrets recovered perfectly\n");
    else if (ber < 0.35)
        printf("Verdict: SIGNAL — BER=%.1f%% < 50%% (timing channel detected)\n",
               100.0 * ber);
    else
        printf("Verdict: NOISE — BER=%.1f%% (no observable timing channel)\n",
               100.0 * ber);

    free(arr);
    return (ok_count == n_secrets) ? 0 : 1;
}
