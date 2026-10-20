/* tee_ext_mod_svc.c - 宿主侧的模块请求服务, 见 tee_ext_mod_svc.h.
 *
 * SPDX-License-Identifier: MIT
 */

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "tee_ext_mod_svc.h"

/* 三个模块命令经 syscall() 直调 ioctl: 内核把交付结果与其它出参打包在结构体中,
 * 而 libc 的 ioctl 原型返回 int. 本文件的命令只取出参, 不经返回值传数据, 故这里
 * 与回归工具保持一致, 一律用同一条路径调用. */
static long module_ioctl(int fd, unsigned long cmd, struct tee_enclave_args *args) {
	long rc = syscall(SYS_ioctl, fd, cmd, args);
	if (rc < 0) {
		return -errno;
	}
	return rc;
}

/* 在表中查找模块. 未登记时返回 NULL. */
static const struct tee_ext_mod_entry *module_lookup(
	const struct tee_ext_mod_table *table, uint32_t module_id) {
	for (int i = 0; i < table->count; i++) {
		if (table->entries[i].module_id == module_id) {
			return &table->entries[i];
		}
	}
	return NULL;
}

/* 模块映像在模块目录中的名称. 完整路径由 tee_ext_mod_table_init 拼接. */
struct module_image {
	uint32_t module_id;
	const char *name;
};

/* 模块表, 各项由 ext_mod_table.inc 给出. */
static const struct module_image MODULE_IMAGES[] = {
#include "ext_mod_table.inc"
};

_Static_assert(
	sizeof(MODULE_IMAGES) / sizeof(MODULE_IMAGES[0]) <= TEE_EXT_MOD_TABLE_MAX,
	"module count exceeds TEE_EXT_MOD_TABLE_MAX");

int tee_ext_mod_table_init(struct tee_ext_mod_table *table, const char *dir) {
	const size_t count = sizeof(MODULE_IMAGES) / sizeof(MODULE_IMAGES[0]);

	table->count = 0;
	for (size_t i = 0; i < count; i++) {
		const char *name				= MODULE_IMAGES[i].name;
		struct tee_ext_mod_entry *entry = &table->entries[i];
		int n = snprintf(entry->path, sizeof(entry->path), "%s/%s", dir, name);
		if (n < 0 || (size_t)n >= sizeof(entry->path)) {
			fprintf(stderr, "module image path too long: '%s/%s'\n", dir, name);
			table->count = 0;
			return -ENAMETOOLONG;
		}

		entry->module_id = MODULE_IMAGES[i].module_id;
		table->count++;
	}

	return table->count;
}

static uint64_t image_size(const char *path) {
	struct stat st;
	if (stat(path, &st) != 0 || !S_ISREG(st.st_mode) || st.st_size <= 0) {
		return 0;
	}
	return (uint64_t)st.st_size;
}

/* 报告映像的字节数. 模块不存在时报告 0, 由 M 模式记为不存在的交付结果. */
static int report_size(int fd, uint64_t enclave_id, uint64_t token, uint64_t size) {
	struct tee_enclave_args args;
	memset(&args, 0, sizeof(args));
	args.enclave_id = enclave_id;
	args.token		= token;
	args.size.size	= size;

	return (int)module_ioctl(fd, TEE_IOC_MODULE_SIZE, &args);
}

/* 交付映像的字节. 缓冲区在调用期间保持有效, 驱动经内核缓冲区中转后即不再引用它. */
static int deliver_image(
	int fd, uint64_t enclave_id, uint64_t token, const void *image, uint64_t size) {
	struct tee_enclave_args args;
	memset(&args, 0, sizeof(args));
	args.enclave_id	   = enclave_id;
	args.token		   = token;
	args.image.buf_ptr = (uint64_t)image;
	args.image.size	   = size;

	return (int)module_ioctl(fd, TEE_IOC_MODULE_IMG, &args);
}

int tee_ext_mod_serve(
	int fd, const struct tee_ext_mod_table *table, uint64_t enclave_id, uint64_t token) {
	struct tee_enclave_args args;
	memset(&args, 0, sizeof(args));
	args.enclave_id = enclave_id;
	args.token		= token;

	long rc = module_ioctl(fd, TEE_IOC_MODULE_REQ, &args);
	if (rc < 0) {
		fprintf(stderr, "req failed (errno=%ld)\n", -rc);
		return (int)rc;
	}

	uint32_t module_id = (uint32_t)args.req.module_id;
	if (module_id == (uint32_t)TEE_MODULE_ID_NONE) {
		return 0;
	}

	const struct tee_ext_mod_entry *entry = module_lookup(table, module_id);
	uint64_t kind						  = args.req.kind;

	if (kind == TEE_MODULE_KIND_SIZE) {
		uint64_t size = entry ? image_size(entry->path) : 0;
		printf("req enclave=%lu id=%u kind=size -> %lu\n", enclave_id, module_id, size);
		return report_size(fd, enclave_id, token, size) < 0 ? -1 : 1;
	}

	if (kind == TEE_MODULE_KIND_LOAD) {
		if (!entry) {
			printf(
				"req enclave=%llu id=%u kind=load -> absent\n",
				(unsigned long long)enclave_id,
				module_id);
			return report_size(fd, enclave_id, token, 0) < 0 ? -1 : 1;
		}

		uint64_t size = image_size(entry->path);
		FILE *fp	  = size ? fopen(entry->path, "rb") : NULL;
		if (!fp) {
			printf(
				"req enclave=%llu id=%u kind=load -> unreadable\n",
				(unsigned long long)enclave_id,
				module_id);
			return report_size(fd, enclave_id, token, 0) < 0 ? -1 : 1;
		}

		uint8_t *image = malloc((size_t)size);
		if (!image) {
			fprintf(stderr, "out of memory for %llu bytes\n", (unsigned long long)size);
			fclose(fp);
			return -1;
		}
		size_t got = fread(image, 1, (size_t)size, fp);
		fclose(fp);

		/* 交付的字节数取实际读出的字节数: 它与报告过的字节数不一致时由 M 模式
		 * 拒绝并记为写入失败, 使映像文件的改动在此暴露. */
		int ret = deliver_image(fd, enclave_id, token, image, got);
		free(image);
		if (ret < 0) {
			fprintf(stderr, "deliver id=%u failed (errno=%d)\n", module_id, -ret);
			return -1;
		}
		printf(
			"req enclave=%llu id=%u kind=load -> %zu bytes from %s\n",
			(unsigned long long)enclave_id,
			module_id,
			got,
			entry->path);
		return 1;
	}

	fprintf(stderr, "unknown kind %llu for id=%u\n", (unsigned long long)kind, module_id);
	return report_size(fd, enclave_id, token, 0) < 0 ? -1 : 1;
}
