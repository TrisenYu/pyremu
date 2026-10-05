/* ext_mod_loader.c - 最小扩展模块装载器
 *
 * 把一个载荷创建并转入飞地, 在载荷让出期间应答它的模块请求, 直到载荷终止. 与回归
 * 工具 tee_ecall_regress 的区别在于本程序只做这一件事: 不做生命周期与令牌校验用例,
 * 也不扫描载荷名录, 故飞地的模块取入通路可以在此单点验证. 模块表与应答流程取自
 * ../ext-mod-svc, 与回归工具共用同一份实现.
 *
 * 用法: ./ext_mod_loader <payload> <modules_dir>
 * 编译: riscv64-linux-gnu-gcc -static -O2 -I../cache-probe-exploit -I../ext-mod-svc \
 *           ext_mod_loader.c ../ext-mod-svc/tee_ext_mod_svc.c -o bin/ext_mod_loader
 *
 * 退出码: 0 载荷自行退出且退出码为 0; 1 载荷以其它形态结束或执行期出错; 2 用法错误.
 *
 * SPDX-License-Identifier: MIT
 */

#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "tee_enclave.h"
#include "tee_ext_mod_svc.h"

/* 四个生命周期命令经 syscall() 直调 ioctl: 内核把运行状态与退出码打包在同一 64 位
 * 返回值中, 而 libc 的 ioctl 原型返回 int, 会把高 32 位截掉. 返回负 errno 表示
 * 调用期失败, 其余取值为 TEE_IOC_ENTER 与 TEE_IOC_RESUME 的运行状态. */
static long loader_ioctl(int fd, unsigned long cmd, void *arg) {
	long rc = syscall(SYS_ioctl, fd, cmd, arg);
	if (rc < 0) {
		return -errno;
	}
	return rc;
}

/* 读入整个文件. 失败返回 NULL. */
static uint8_t *read_whole_file(const char *path, size_t *out_size) {
	FILE *fp = fopen(path, "rb");
	if (!fp) {
		fprintf(stderr, "open '%s' failed (errno=%d)\n", path, errno);
		return NULL;
	}
	fseek(fp, 0, SEEK_END);
	long sz = ftell(fp);
	fseek(fp, 0, SEEK_SET);
	if (sz <= 0) {
		fprintf(stderr, "'%s' is empty\n", path);
		fclose(fp);
		return NULL;
	}
	uint8_t *buf = malloc((size_t)sz);
	if (!buf) {
		fprintf(stderr, "out of memory for %ld bytes\n", sz);
		fclose(fp);
		return NULL;
	}
	if (fread(buf, 1, (size_t)sz, fp) != (size_t)sz) {
		fprintf(stderr, "read '%s' failed\n", path);
		free(buf);
		fclose(fp);
		return NULL;
	}
	fclose(fp);
	*out_size = (size_t)sz;
	return buf;
}

int main(int argc, char **argv) {
	if (argc < 3) {
		fprintf(stderr, "usage: %s <payload> <modules_dir>\n", argv[0]);
		return 2;
	}
	const char *payload_path = argv[1];
	const char *modules_dir	 = argv[2];

	struct tee_ext_mod_table modules;
	int n = tee_ext_mod_table_load(&modules, modules_dir);
	if (n < 0) {
		fprintf(stderr, "load module table from '%s' failed (%d)\n", modules_dir, n);
		return 1;
	}
	printf("[ext-mod-loader] %d modules from %s (images are read on request)\n",
		   n, modules_dir);
	for (int i = 0; i < modules.count; i++) {
		printf("[ext-mod-loader]   id=%u path=%s\n",
			   modules.entries[i].module_id, modules.entries[i].path);
	}

	size_t payload_size = 0;
	uint8_t *payload	= read_whole_file(payload_path, &payload_size);
	if (!payload) {
		return 1;
	}
	printf("[ext-mod-loader] payload %s (%zu bytes)\n", payload_path, payload_size);

	int fd = open(TEE_DEVICE_PATH, O_RDWR);
	if (fd < 0) {
		fprintf(stderr, "open " TEE_DEVICE_PATH " failed (errno=%d)\n", errno);
		free(payload);
		return 1;
	}

	uint64_t enclave_id = 0;
	if (loader_ioctl(fd, TEE_IOC_CREATE, &enclave_id) < 0) {
		fprintf(stderr, "CREATE failed (errno=%d)\n", errno);
		free(payload);
		close(fd);
		return 1;
	}
	printf("[ext-mod-loader] enclave=%llu\n", (unsigned long long)enclave_id);

	/* CREATE 登记的管理令牌恒为 0, 故后续 RESUME 与模块应答均以令牌 0 调用. */
	struct tee_enclave_args args;
	memset(&args, 0, sizeof(args));
	args.enclave_id			= enclave_id;
	args.enter.payload_ptr	= (uint64_t)payload;
	args.enter.payload_size = payload_size;
	args.enter.argc			= 0;
	args.enter.argv_ptr		= 0;

	long rc		 = loader_ioctl(fd, TEE_IOC_ENTER, &args);
	int serviced = 0;
	while (rc == TEE_RUN_SUSPENDED) {
		int served = tee_ext_mod_serve(fd, &modules, enclave_id, 0);
		if (served > 0) {
			serviced++;
		}
		struct tee_enclave_token_args resume_args = {
			.enclave_id = enclave_id,
			.token		= 0,
		};
		rc = loader_ioctl(fd, TEE_IOC_RESUME, &resume_args);
	}

	free(payload);
	/* 载荷交还宿主时仍未终止: 回收它占用的槽位与内存. */
	if (rc == TEE_RUN_SUSPENDED || rc < 0) {
		struct tee_enclave_token_args shutdown_args = {
			.enclave_id = enclave_id,
			.token		= 0,
		};
		(void)loader_ioctl(fd, TEE_IOC_SHUTDOWN, &shutdown_args);
	}
	close(fd);

	if (rc < 0) {
		printf("[ext-mod-loader] enter/resume failed (errno=%ld)\n", -rc);
		printf("[ext-mod-loader] result: enter-failed\n");
		return 1;
	}

	printf("[ext-mod-loader] serviced=%d\n", serviced);
	printf("[ext-mod-loader] result: state=%d exit=%llu\n",
		   TEE_RUN_STATE(rc),
		   (unsigned long long)TEE_EXIT_CODE(rc));
	return (TEE_RUN_STATE(rc) == TEE_RUN_EXITED && TEE_EXIT_CODE(rc) == 0) ? 0 : 1;
}
