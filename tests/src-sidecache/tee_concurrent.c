// tee_concurrent.c - 多进程并发创建/进入/销毁飞地
//
// 用法: tee_concurrent <payload> [num_procs]
//
// 每个子进程独立打开 /dev/tee_enclave, 查询 mdid → CREATE → ENTER → SHUTDOWN.
// 多 hart 系统上 CREATE 由 OpenSBI create_lock 串行化 ID 分配,
// 但上下文切换 (alter_hart_ctx_for_enclave) 在不同 hart 上并发执行.
//
// Built with: /opt/custom-llvm/bin/clang -static -O2 -o tee_concurrent tee_concurrent.c

#include <errno.h>
#include <fcntl.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/wait.h>
#include <unistd.h>

#include "tee_enclave.h"

static void die(const char *msg) {
	fprintf(stderr, "[child %d] %s (errno=%d)\n", getpid(), msg, errno);
	exit(1);
}

static int do_create(int fd, unsigned long long *out_id) {
	unsigned long long args[4] = {0xdeadbeefcafebabeULL, 0, 0, 0};
	int rc					   = ioctl(fd, TEE_IOC_CREATE, args);
	if (rc < 0) {
		return rc;
	}
	*out_id = args[0];
	return 0;
}

static int do_enter(int fd, unsigned long long enclave_id, const char *payload_path) {
	struct tee_enter_args args;
	memset(&args, 0, sizeof(args));
	args.enclave_id = enclave_id;

	if (payload_path) {
		FILE *fp = fopen(payload_path, "rb");
		if (!fp) {
			return -1;
		}
		fseek(fp, 0, SEEK_END);
		long sz = ftell(fp);
		fseek(fp, 0, SEEK_SET);
		unsigned char *buf = malloc(sz);
		if (!buf) {
			fclose(fp);
			return -1;
		}
		fread(buf, 1, sz, fp);
		fclose(fp);
		args.payload_ptr  = (unsigned long long)buf;
		args.payload_size = (unsigned long long)sz;
	}

	int rc = ioctl(fd, TEE_IOC_ENTER, &args);
	if (payload_path) {
		free((void *)(unsigned long)args.payload_ptr);
	}
	return rc;
}

static void do_shutdown(int fd, unsigned long long enclave_id) {
	unsigned long long args[1] = {enclave_id};
	ioctl(fd, TEE_IOC_SHUTDOWN, args);
}

static void child_main(const char *payload_path, int proc_id) {
	int fd = open(TEE_DEVICE_PATH, O_RDWR);
	if (fd < 0) {
		die("open " TEE_DEVICE_PATH);
	}

	// 1. 查询当前 mdid
	unsigned long long mdid = 0;
	int rc					= ioctl(fd, TEE_IOC_GET_ID, &mdid);
	printf(
		"[child %d pid=%d] GET_ID: mdid=%llu rc=%d\n",
		proc_id,
		getpid(),
		(unsigned long long)mdid,
		rc);

	// 2. CREATE
	unsigned long long enclave_id = 0;
	rc							  = do_create(fd, &enclave_id);
	if (rc < 0) {
		printf("[child %d pid=%d] CREATE: failed rc=%d\n", proc_id, getpid(), rc);
		close(fd);
		return;
	}
	printf("[child %d pid=%d] CREATE: enclave_id=%llu\n", proc_id, getpid(), enclave_id);

	// 3. ENTER (if payload available)
	if (payload_path) {
		rc = do_enter(fd, enclave_id, payload_path);
		if (rc == 0) {
			printf("[child %d pid=%d] ENTER: success, back to host\n", proc_id, getpid());
		} else {
			printf("[child %d pid=%d] ENTER: failed rc=%d\n", proc_id, getpid(), rc);
		}
	}

	// 4. SHUTDOWN
	do_shutdown(fd, enclave_id);
	printf("[child %d pid=%d] SHUTDOWN: done\n", proc_id, getpid());

	close(fd);
}

int main(int argc, char **argv) {
	const char *payload = NULL;
	int num_procs		= 2;

	if (argc > 1) {
		payload = argv[1];
	}
	if (argc > 2) {
		num_procs = atoi(argv[2]);
	}
	if (num_procs < 1) {
		num_procs = 1;
	}
	if (num_procs > 64) {
		num_procs = 64;
	}

	printf("=== Concurrent Enclave Test (%d processes) ===\n\n", num_procs);

	pid_t *pids = calloc(num_procs, sizeof(pid_t));
	if (!pids) {
		perror("calloc");
		return 1;
	}

	for (int i = 0; i < num_procs; i++) {
		pid_t p = fork();
		if (p < 0) {
			perror("fork");
			// Kill already-forked children
			for (int j = 0; j < i; j++) {
				kill(pids[j], SIGTERM);
			}
			free(pids);
			return 1;
		}
		if (p == 0) {
			// Child
			free(pids);
			child_main(payload, i);
			return 0;
		}
		pids[i] = p;
	}

	// Wait for all children
	int failed = 0;
	for (int i = 0; i < num_procs; i++) {
		int status;
		waitpid(pids[i], &status, 0);
		if (WIFEXITED(status)) {
			printf("[parent] child %d exited with %d\n", i, WEXITSTATUS(status));
			if (WEXITSTATUS(status) != 0) {
				failed++;
			}
		} else {
			printf("[parent] child %d terminated abnormally\n", i);
			failed++;
		}
	}

	free(pids);
	printf("\n=== %d/%d processes succeeded ===\n", num_procs - failed, num_procs);
	return failed ? 1 : 0;
}
