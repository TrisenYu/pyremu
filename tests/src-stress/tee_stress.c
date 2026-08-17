/* tee_stress.c - TEE 飞地并发压力测试
 *
 * 按固定批次 (2 / 20 / 200 / 2000 / 20000) 并发创建飞地,
 * 每个飞地加载 stress_payload 执行 CPU + 内存压力操作。
 *
 * 每批次: 启动 N 个线程, 各线程独立完成 CREATE → ENTER → SHUTDOWN。
 * 结果通过原子变量汇总。
 *
 * 编译: riscv64-linux-gnu-gcc -static -O2 -pthread tee_stress.c -o bin/tee_stress
 *
 * SPDX-License-Identifier: GPL-2.0
 */

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <unistd.h>

#include "tee_enclave.h"

/* ---- 全局共享 (只读) ---- */

static const char    *g_payload_path = NULL;
static uint8_t       *g_payload      = NULL;
static size_t         g_payload_size = 0;
static atomic_int     g_passed;
static atomic_int     g_failed;
static atomic_int     g_mem_exhausted;
static atomic_bool    g_stop;

/* ---- helpers ---- */

static void die(const char *msg) {
	fprintf(stderr, "tee_stress: %s (errno=%d)\n", msg, errno);
	exit(EXIT_FAILURE);
}

static uint8_t *read_file(const char *path, size_t *size) {
	FILE *fp = fopen(path, "rb");
	if (!fp) return NULL;
	fseek(fp, 0, SEEK_END);
	long sz = ftell(fp);
	fseek(fp, 0, SEEK_SET);
	if (sz <= 0) { fclose(fp); return NULL; }
	uint8_t *buf = malloc((size_t)sz);
	if (!buf) { fclose(fp); return NULL; }
	if (fread(buf, 1, (size_t)sz, fp) != (size_t)sz) {
		free(buf); fclose(fp); return NULL;
	}
	fclose(fp);
	*size = (size_t)sz;
	return buf;
}

/* ---- 单线程 enclave 生命周期 ---- */

static void *worker(void *arg) {
	(void)arg;

	if (atomic_load(&g_stop)) return NULL;

	int fd = open(TEE_DEVICE_PATH, O_RDWR);
	if (fd < 0) {
		atomic_fetch_add(&g_failed, 1);
		return NULL;
	}

	uint64_t enclave_id = 0;
	int rc = ioctl(fd, TEE_IOC_CREATE, &enclave_id);
	if (rc < 0) {
		if (errno == ENOMEM) {
			atomic_fetch_add(&g_mem_exhausted, 1);
			atomic_store(&g_stop, true);
		} else {
			atomic_fetch_add(&g_failed, 1);
		}
		close(fd);
		return NULL;
	}

	struct tee_enter_args args = {
		.enclave_id   = enclave_id,
		.payload_ptr  = (uint64_t)g_payload,
		.payload_size = g_payload_size,
		.argc         = 0,
		.argv_ptr     = 0,
	};
	rc = ioctl(fd, TEE_IOC_ENTER, &args);

	/* SHUTDOWN — 即使 ENTER 失败也尝试 */
	ioctl(fd, TEE_IOC_SHUTDOWN, &enclave_id);

	if (rc < 0) {
		atomic_fetch_add(&g_failed, 1);
	} else {
		atomic_fetch_add(&g_passed, 1);
	}

	close(fd);
	return NULL;
}

/* ---- 单批次 (并发) ---- */

static void run_batch(int count) {
	atomic_store(&g_passed, 0);
	atomic_store(&g_failed, 0);
	atomic_store(&g_mem_exhausted, 0);
	atomic_store(&g_stop, false);

	pthread_t *threads = calloc((size_t)count, sizeof(pthread_t));
	if (!threads) die("calloc threads");

	for (int i = 0; i < count; i++) {
		int rc = pthread_create(&threads[i], NULL, worker, NULL);
		if (rc != 0) {
			fprintf(stderr, "  [%d] pthread_create failed at %d/%d (errno=%d)\n",
			        count, i + 1, count, rc);
			/* 继续等待已创建的线程 */
		}
	}

	for (int i = 0; i < count; i++) {
		if (threads[i]) pthread_join(threads[i], NULL);
	}
	free(threads);

	int p = atomic_load(&g_passed);
	int f = atomic_load(&g_failed);
	int m = atomic_load(&g_mem_exhausted);

	printf("  result: passed=%d failed=%d", p, f);
	if (m) printf(" mem_exhausted=%d", m);
	puts("\n");
}

/* ---- main ---- */

int main(int argc, char **argv) {
	if (argc < 2) {
		fprintf(stderr, "Usage: %s <stress_payload_path>\n", argv[0]);
		return 1;
	}
	g_payload_path = argv[1];

	const int batches[]    = {2, 20, 200, 2000, 20000};
	const int num_batches  = sizeof(batches) / sizeof(batches[0]);

	/* 查询初始内存 (单次 open) */
	int info_fd = open(TEE_DEVICE_PATH, O_RDWR);
	if (info_fd < 0) die("open " TEE_DEVICE_PATH);
	struct tee_mem_info mem;
	if (ioctl(info_fd, TEE_IOC_GET_MEM, &mem) == 0) {
		puts("=== TEE Enclave Concurrent Stress Test ===\n");
		printf("[info] initial memory: free=%lu max_contiguous=%lu (2MiB units)\n\n",
		       mem.free_total, mem.max_contiguous);
	}
	close(info_fd);

	/* 加载 payload (只读, 线程共享) */
	g_payload = read_file(g_payload_path, &g_payload_size);
	if (!g_payload) die("read payload");
	printf("[info] payload '%s' loaded (%zu bytes)\n\n", g_payload_path, g_payload_size);

	int total_passed = 0, total_failed = 0;

	for (int b = 0; b < num_batches; b++) {
		int count = batches[b];
		printf("--- Batch %d: %d concurrent enclaves ---\n", b + 1, count);

		run_batch(count);

		total_passed += atomic_load(&g_passed);
		total_failed += atomic_load(&g_failed);

		if (atomic_load(&g_mem_exhausted) && count >= 200) {
			printf("[info] memory exhausted at batch size %d, stopping.\n", count);
			break;
		}
	}

	printf("=== Total: passed=%d failed=%d ===\n", total_passed, total_failed);

	free(g_payload);
	return total_failed ? 1 : 0;
}
