/* victim_payload.c - 受害飞地载荷 (运行在飞地 U-mode)
 *
 * 模拟处理敏感数据的合法飞地应用:
 *   1. 分配内存, 写入敏感标记 "SECRET-KEY"
 *   2. 反复访问以确保 TLB 填充 (mdid=当前飞地 ID)
 *   3. 输出标记供外部验证
 *   4. 退出
 *
 * 若无 mfence.did 清 TLB, 后续飞地可能通过同 VA 的 TLB 命中
 * 读到本飞地的物理页, 造成数据泄漏.
 *
 * 编译: riscv64-linux-gnu-gcc -static -O2 victim_payload.c -o victim_payload
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

#define SENSITIVE_DATA "SECRET-KEY-12345678"
#define PROBE_VA	   ((volatile char *)0x20000000UL) /* 固定 VA 便于探测 */

int main(int argc __attribute__((unused)), char **argv __attribute__((unused))) {
	/* 在固定 VA 分配一页 */
	size_t page_sz = 4096;
	void *addr	   = mmap(
		(void *)PROBE_VA,
		page_sz,
		PROT_READ | PROT_WRITE,
		MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED,
		-1,
		0);
	if (addr == MAP_FAILED) {
		perror("mmap");
		return 1;
	}

	/* 写入敏感标记 */
	strncpy((char *)addr, SENSITIVE_DATA, page_sz - 1);

	/* 反复访问以填充 TLB */
	volatile char *p = (volatile char *)addr;
	char sum		 = 0;
	for (int i = 0; i < 100; i++) {
		for (size_t j = 0; j < strlen(SENSITIVE_DATA); j++) {
			sum ^= p[j];
		}
	}

	printf(
		"[victim] VA=%p  data='%s'  checksum=0x%02x\n",
		(void *)addr,
		(char *)addr,
		sum & 0xFF);
	printf("[victim] TLB populated - sensitive data accessible via VA\n");

	munmap(addr, page_sz);
	return 0;
}
