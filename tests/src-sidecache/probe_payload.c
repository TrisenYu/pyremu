/* probe_payload.c - 恶意探测飞地载荷 (运行在飞地 U-mode)
 *
 * 模拟攻击者获批的"可信"飞地应用, 实际执行 TLB 残留探测:
 *   1. 在与 victim 相同的 VA 分配一页
 *   2. 写入公开标记 "PUBLIC-DATA"
 *   3. 读回 VA 的数据
 *   4. 若读到 "SECRET-KEY-..." → TLB 泄漏 (命中了 victim 的 PPN)
 *   5. 若读到 "PUBLIC-DATA" → 无泄漏 (mfence.did 生效或 TLB 未残留)
 *   6. 输出探测结论
 *
 * 探测原理:
 *   同一 hart 上先后运行的飞地共享 TLB.  若域切换时 M-mode 未调
 *   mfence.did(prev_mdid), 上一飞地的 TLB 条目 (VPN→PPN) 仍有效.
 *   本飞地访问同 VA 时 TLB 命中 → 读到上一飞地的物理页而非自己的.
 *
 * 编译: riscv64-linux-gnu-gcc -static -O2 probe_payload.c -o probe_payload
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

#define PUBLIC_DATA	   "PUBLIC-DATA-00000000"
#define SENSITIVE_DATA "SECRET-KEY-12345678"
#define PROBE_VA	   ((volatile char *)0x20000000UL)
#define __unused	   __attribute__((unused))

int main(int __unused argc, char __unused **argv) {
	size_t page_sz = 4096;

	/* 与 victim 完全相同的固定 VA */
	void *addr = mmap(
		(void *)PROBE_VA,
		page_sz,
		PROT_READ | PROT_WRITE,
		MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED,
		-1,
		0);
	if (addr == MAP_FAILED) {
		perror("mmap failed");
		return 1;
	}

	/* 写入自己的公开数据 */
	strncpy((char *)addr, PUBLIC_DATA, page_sz - 1);
	printf("[probe]  wrote '%s' at VA=%p\n", PUBLIC_DATA, (void *)addr);

	/* ---- 第一次读取 (无强制 TLB flush) ---- */
	volatile char *p = (volatile char *)addr;
	char buf_first[64];
	memcpy(buf_first, (void *)p, sizeof(buf_first) - 1);
	buf_first[sizeof(buf_first) - 1] = '\0';
	printf("[probe]  1st read='%s'\n", buf_first);

	/* ---- 全 TLB flush 后再次读取 ---- */
	__asm__ volatile("sfence.vma zero, zero" ::: "memory");

	char buf_after[64];
	memcpy(buf_after, (void *)p, sizeof(buf_after) - 1);
	buf_after[sizeof(buf_after) - 1] = '\0';
	printf("[probe]  2nd read (after sfence.vma)='%s'\n", buf_after);

	/* ---- 判定 ---- */
	int leak_first	  = (strncmp(buf_first, SENSITIVE_DATA, strlen(SENSITIVE_DATA)) == 0);
	int leak_after	  = (strncmp(buf_after, SENSITIVE_DATA, strlen(SENSITIVE_DATA)) == 0);
	int correct_first = (strncmp(buf_first, PUBLIC_DATA, strlen(PUBLIC_DATA)) == 0);
	int correct_after = (strncmp(buf_after, PUBLIC_DATA, strlen(PUBLIC_DATA)) == 0);

	puts("\n[probe] === Result ===");
	printf(
		"[probe]  Before flush: %s\n",
		leak_first		? "LEAKED  (victim's data visible!)"
		: correct_first ? "CORRECT (own data)"
						: "UNKNOWN (data mismatch)");
	printf(
		"[probe]  After  flush: %s\n",
		leak_after		? "LEAKED  (mfence.did failed)"
		: correct_after ? "CORRECT (mfence.did effective)"
						: "UNKNOWN (data mismatch)");

	int exit_code;
	if (leak_first && correct_after) {
		puts("[probe] Verdict: TLB leak CONFIRMED, sfence.vma BLOCKED it");
		exit_code = 42; /* 泄漏确认码 */
	} else if (correct_first) {
		puts("[probe] Verdict: No leak - TLB properly isolated");
		exit_code = 0;
	} else {
		puts("[probe] Verdict: Unexpected state");
		exit_code = 1;
	}

	munmap(addr, page_sz);
	return exit_code;
}
