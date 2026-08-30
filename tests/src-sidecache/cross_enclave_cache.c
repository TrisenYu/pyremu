/* cross_enclave_cache.c - 跨飞地 Flush+Reload 缓存侧信道测试
 *
 * 并发场景: victim 飞地先创建并保持在线 (SUSPEND),
 * attacker 飞地后创建, 在 victim 仍存活时探测 L2 缓存残留。
 *
 * 通过内核 ioctl -> SBI ecall 操作飞地生命周期:
 *   1. CREATE victim -> ENTER (写入 secret + 填充 L2, return 0 触发 SUSPEND)
 *   2. CREATE attacker -> ENTER (Flush+Reload 探测, 计算 BER)
 *   3. SHUTDOWN attacker
 *   4. SHUTDOWN victim (mfence.did + 内存清零 + 分区释放)
 *
 * victim 在 SUSPEND 后仍持有分区, L2 中 victim 域的缓存行未被刷除。
 * attacker 在 victim 存活期间运行, 尝试通过 rdcycle 计时检测缓存泄漏。
 *
 * 用法: ./cross_enclave_cache <victim_payload> <attacker_payload>
 * 编译: riscv64-linux-gnu-gcc -static -O2 cross_enclave_cache.c -o bin/cross_enclave_cache
 *
 * SPDX-License-Identifier: GPL-2.0
 */

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include "tee_enclave.h"

/* ---- helpers ---- */

static uint8_t *read_file(const char *path, size_t *out_sz) {
	FILE *fp = fopen(path, "rb");
	if (!fp) {
		fprintf(stderr, "open %s: %s\n", path, strerror(errno));
		return NULL;
	}
	fseek(fp, 0, SEEK_END);
	long sz = ftell(fp);
	fseek(fp, 0, SEEK_SET);
	if (sz <= 0) {
		fclose(fp);
		return NULL;
	}
	uint8_t *buf = malloc((size_t)sz);
	if (!buf) {
		fclose(fp);
		return NULL;
	}
	if (fread(buf, 1, (size_t)sz, fp) != (size_t)sz) {
		free(buf);
		fclose(fp);
		return NULL;
	}
	fclose(fp);
	*out_sz = (size_t)sz;
	return buf;
}

/* ---- main ---- */

int main(int argc, char **argv) {
	if (argc < 3) {
		fprintf(stderr, "Usage: %s <victim_payload> <attacker_payload>\n", argv[0]);
		return 1;
	}

	const char *victim_path	  = argv[1];
	const char *attacker_path = argv[2];

	puts("=== Cross-Enclave Flush+Reload Cache Side-Channel Test ===");
	puts("Scenario: victim stays alive, attacker probes while victim active.\n");

	int fd = open(TEE_DEVICE_PATH, O_RDWR);
	if (fd < 0) {
		fprintf(stderr, "open %s: %s\n", TEE_DEVICE_PATH, strerror(errno));
		return 1;
	}

	/* 查询初始内存 */
	struct tee_mem_info mem;
	if (ioctl(fd, TEE_IOC_GET_MEM, &mem) == 0) {
		printf(
			"[host] initial: free=%lu partitions (%lu MiB)\n\n",
			(unsigned long)mem.free_total,
			(unsigned long)(mem.free_total * 2));
	}

	/* 加载载荷文件 */
	size_t victim_sz = 0, attacker_sz = 0;
	uint8_t *victim_payload = read_file(victim_path, &victim_sz);
	if (!victim_payload) {
		fprintf(stderr, "load victim '%s' failed\n", victim_path);
		close(fd);
		return 1;
	}
	uint8_t *attacker_payload = read_file(attacker_path, &attacker_sz);
	if (!attacker_payload) {
		fprintf(stderr, "load attacker '%s' failed\n", attacker_path);
		free(victim_payload);
		close(fd);
		return 1;
	}
	printf("[host] victim:   '%s' (%zu bytes)\n", victim_path, victim_sz);
	printf("[host] attacker: '%s' (%zu bytes)\n\n", attacker_path, attacker_sz);

	/* ==== phase 1: 创建 victim 飞地, 写入 secret 并填充 L2 ==== */
	puts("--- Phase 1: Create victim, write secret + prime L2 ---");
	uint64_t victim_id = 0;
	int rc			   = ioctl(fd, TEE_IOC_CREATE, &victim_id);
	if (rc < 0) {
		fprintf(stderr, "CREATE victim: %s\n", strerror(errno));
		goto cleanup;
	}
	printf("[host] victim created (id=%lu)\n", (unsigned long)victim_id);

	struct tee_enter_args victim_args = {
		.enclave_id	  = victim_id,
		.payload_ptr  = (uint64_t)victim_payload,
		.payload_size = victim_sz,
	};
	rc = ioctl(fd, TEE_IOC_ENTER, &victim_args);
	printf("[host] victim ENTER returned (rc=%d) -- victim SUSPENDED, still alive\n", rc);

	/* victim 此时 SUSPEND, 分区和 L2 缓存行保留 */
	if (ioctl(fd, TEE_IOC_GET_MEM, &mem) == 0) {
		printf(
			"[host] victim alive: free=%lu partitions\n\n",
			(unsigned long)mem.free_total);
	}

	/* ==== phase 2: 创建 attacker 飞地, 在 victim 存活时探测 ==== */
	puts("--- Phase 2: Create attacker, probe while victim alive ---");
	uint64_t attacker_id = 0;
	rc					 = ioctl(fd, TEE_IOC_CREATE, &attacker_id);
	if (rc < 0) {
		fprintf(stderr, "CREATE attacker: %s\n", strerror(errno));
		goto shutdown_victim;
	}
	printf("[host] attacker created (id=%lu)\n", (unsigned long)attacker_id);

	struct tee_enter_args attacker_args = {
		.enclave_id	  = attacker_id,
		.payload_ptr  = (uint64_t)attacker_payload,
		.payload_size = attacker_sz,
	};
	rc = ioctl(fd, TEE_IOC_ENTER, &attacker_args);
	printf("[host] attacker ENTER returned (rc=%d)\n", rc);

	/* 先关停 attacker */
	rc = ioctl(fd, TEE_IOC_SHUTDOWN, &attacker_id);
	printf("[host] attacker SHUTDOWN (rc=%d)\n", rc);

	/* ==== phase 3: 清理 victim ==== */
	puts("\n--- Phase 3: Cleanup ---");

shutdown_victim:
	rc = ioctl(fd, TEE_IOC_SHUTDOWN, &victim_id);
	printf("[host] victim SHUTDOWN (rc=%d)\n", rc);

	if (ioctl(fd, TEE_IOC_GET_MEM, &mem) == 0) {
		printf(
			"[host] after cleanup: free=%lu partitions\n", (unsigned long)mem.free_total);
	}

	/* ==== 汇总 ==== */
	puts("\n=== Summary ===");
	puts("Check [attacker] output above for per-line timing and BER.");
	puts("  BER > 0.35 : NOISE    -> L2 isolation effective (PASS)");
	puts("  BER < 0.25 : LEAKAGE  -> cache side-channel detected (FAIL)");
	puts("  0.25-0.35  : MARGINAL -> rerun for statistical confidence");

cleanup:
	free(victim_payload);
	free(attacker_payload);
	close(fd);
	return 0;
}
