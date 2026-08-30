/* tee_max_online.c - TEE 飞地最大在线数量压力测试
 *
 * 考察给定内存下最多可同时在线 (创建并进入但不退出) 的飞地数量。
 * 每个飞地加载 stress_payload 执行真实的 CPU + 内存压力操作,
 * 飞地执行完毕后保持在线 (不 SHUTDOWN)。
 *
 * 停止条件: 连续 CREATE/ENTER 失败 32 次 (容忍瞬时错误),
 * 记录停止前的最大在线数。
 *
 * 编译: riscv64-linux-gnu-gcc -static -O2 -pthread tee_max_online.c -o bin/tee_max_online
 * 用法: ./tee_max_online <stress_payload_path>
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

/* 连续失败该次数后停止 */
#define MAX_CONSECUTIVE_FAILURES 32

/* 在线飞地 id 的存储上限。飞地数量受内存槽硬限制
 * (MAX_ENCLAVE_SLOTS=257, 含 host), 实际不会超过此值,
 * 这里取一个足够大的安全上限。 */
#define MAX_STORED_IDS 4096

/* ---- helpers ---- */

static uint8_t *read_file(const char *path, size_t *out_sz) {
	FILE *fp = fopen(path, "rb");
	if (!fp) {
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
	if (argc < 2) {
		fprintf(stderr, "Usage: %s <stress_payload_path>\n", argv[0]);
		return 1;
	}

	puts("=== TEE Enclave Max Online Enclaves Test ===");
	puts("CREATE + ENTER without SHUTDOWN, stop after 32 consecutive failures.\n");

	int fd = open(TEE_DEVICE_PATH, O_RDWR);
	if (fd < 0) {
		fprintf(stderr, "open %s: %s\n", TEE_DEVICE_PATH, strerror(errno));
		return 1;
	}

	/* 查询初始内存 */
	struct tee_mem_info mem_init;
	if (ioctl(fd, TEE_IOC_GET_MEM, &mem_init) != 0) {
		fprintf(stderr, "GET_MEM: %s\n", strerror(errno));
		close(fd);
		return 1;
	}
	printf("[info] initial: free=%lu partitions (%lu MiB)\n\n",
		(unsigned long)mem_init.free_total,
		(unsigned long)(mem_init.free_total * 2));

	/* 加载 stress_payload */
	size_t payload_sz = 0;
	uint8_t *payload = read_file(argv[1], &payload_sz);
	if (!payload) {
		fprintf(stderr, "read payload '%s' failed\n", argv[1]);
		close(fd);
		return 1;
	}
	printf("[info] payload '%s' loaded (%zu bytes)\n\n",
		argv[1], payload_sz);

	uint64_t online_ids[MAX_STORED_IDS];
	int count = 0;
	int consecutive_failures = 0;

	while (consecutive_failures < MAX_CONSECUTIVE_FAILURES) {
		/* 创建飞地 */
		uint64_t enclave_id = 0;
		int rc = ioctl(fd, TEE_IOC_CREATE, &enclave_id);
		if (rc < 0) {
			consecutive_failures++;
			printf("[info] CREATE #%d failed: errno=%d (%s), consecutive=%d/%d\n",
				count + 1, errno, strerror(errno),
				consecutive_failures, MAX_CONSECUTIVE_FAILURES);
			continue;
		}

		/* 进入飞地, 执行 stress_payload (完成后保持在线) */
		struct tee_enter_args args = {
			.enclave_id   = enclave_id,
			.payload_ptr  = (uint64_t)payload,
			.payload_size = payload_sz,
			.argc         = 0,
			.argv_ptr     = 0,
		};
		rc = ioctl(fd, TEE_IOC_ENTER, &args);
		if (rc < 0) {
			consecutive_failures++;
			printf("[error] ENTER #%d failed: errno=%d (%s), consecutive=%d/%d\n",
				count + 1, errno, strerror(errno),
				consecutive_failures, MAX_CONSECUTIVE_FAILURES);
			/* 回退: 关停该飞地 */
			ioctl(fd, TEE_IOC_SHUTDOWN, &enclave_id);
			continue;
		}

		/* 成功: 记录并重置连续失败计数 */
		if (count >= MAX_STORED_IDS) {
			fprintf(stderr, "exceeded MAX_STORED_IDS=%d, abort\n",
				MAX_STORED_IDS);
			break;
		}
		online_ids[count] = enclave_id;
		count++;
		consecutive_failures = 0;

		if (count % 10 == 0) {
			struct tee_mem_info mem_now;
			if (ioctl(fd, TEE_IOC_GET_MEM, &mem_now) == 0) {
				printf("  online=%3d  remaining=%lu partitions\n",
					count, (unsigned long)mem_now.free_total);
			}
		}
	}

	/* ==== 结果 ==== */
	struct tee_mem_info mem_end;
	ioctl(fd, TEE_IOC_GET_MEM, &mem_end);

	printf("\n=== RESULT ===\n");
	printf("max online enclaves: %d\n", count);
	printf("initial free: %lu partitions, final free: %lu partitions\n",
		(unsigned long)mem_init.free_total,
		(unsigned long)mem_end.free_total);
	if (count > 0) {
		uint64_t consumed = mem_init.free_total - mem_end.free_total;
		printf("memory per enclave: %.2f MiB (%lu partitions / %d enclaves)\n",
			(double)consumed * 2.0 / (double)count,
			(unsigned long)consumed, count);
	}

	/* 清理: 关停所有在线飞地 */
	puts("\n=== Cleanup ===");
	for (int i = 0; i < count; i++) {
		ioctl(fd, TEE_IOC_SHUTDOWN, &online_ids[i]);
	}
	printf("cleanup: %d enclaves shut down.\n", count);

	free(payload);
	close(fd);
	return 0;
}
