/* tee_concurrent_enter.c - 同一飞地并发进入的互斥性测试
 *
 * 考察的不变量: 同一飞地至多在一个 hart 上运行. 载入飞地 (把它的上下文恢复到
 * 本 hart) 之前必须先在 M 模式占用它, 否则两个 hart 会各自把同一飞地的上下文
 * 恢复到自己的现场, 使同一份栈与内存被两个 hart 并发执行.
 *
 * 载入飞地有两条路径: ENTER 与 RESUME. 两者的占用检查分别在 enter.c 与 resume.c
 * 内实现, 是两处独立的判定, 故两条路径都要探测: 只探测 ENTER 无法说明 RESUME
 * 的占用检查生效, 反之亦然.
 *
 * 用例构造 (无需新载荷, 复用长时运行的飞地载荷):
 *   1. 主进程 CREATE_EX 一个飞地 (登记一枚非 0 的管理令牌), 飞地停在引导挂起态;
 *   2. 主进程绑定到 hart A, fork 出的子进程绑定到 hart B (A 与 B 不同), 由子进程对该
 *      飞地发起 ENTER (携带正确令牌) 并阻塞在 M 模式: 飞地正在 B 上运行, 其
 *      running_hart 已指向 B;
 *   3. 主进程确认子进程已进入 ENTER 后, 在 hart A 上对同一飞地连续发起若干次
 *      ENTER 与 RESUME, 两者都应被拒 (SBI_ERR_DENIED, 用户态见 EACCES). 令牌用正确
 *      的值, 使拒绝只可能出自占用检查而非令牌校验;
 *   4. 子进程的 ENTER 最终正常交还 host (载荷运行完或让出), 该次进入应成功.
 *
 * 断言 (四项, 各输出一行 regress-check):
 *   concurrent-enter-two-harts        两个进程确实运行在不同 hart 上;
 *   concurrent-enter-single-owner     子进程的 ENTER 成功, 且并发 ENTER 全部被拒;
 *   concurrent-enter-resume-rejected  同一占用窗口内的并发 RESUME 全部被拒;
 *   concurrent-enter-no-leak          飞地终止后内存池回到运行前的空闲分区数.
 *
 * 判定前提: 两次 ENTER 必须由不同 hart 发起. 单个 hart 上第二次 ENTER 只能在第一次
 * 返回之后才开始, 无从构成并发占用, 那样的通过毫无意义. 故本程序先把两个进程分别
 * 绑定到两个在线 hart (处理器亲和性), 并把实际 hart 编号记入共享控制块核对. 本 guest
 * 的 CPU 编号与 hart 编号一一对应 (设备树 cpu@N 即 hart N).
 *
 * 载荷需长于 ENTER_WINDOW_US, 否则飞地在主进程发起并发 ENTER 前就结束.
 *
 * 编译: riscv64-linux-gnu-gcc -static -O2 tee_concurrent_enter.c -o bin/tee_concurrent_enter
 * 用法: ./tee_concurrent_enter <payload_path> [concurrent_attempts]
 *
 * SPDX-License-Identifier: MIT
 */

#define _GNU_SOURCE

#include <errno.h>
#include <fcntl.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>

#include "tee_enclave.h"

/* 主进程默认发起的并发 ENTER 次数 (每次 ENTER 后紧跟一次 RESUME) */
#define DEFAULT_ATTEMPTS 5

/* 可绑定的 hart 数量上限 (本 guest 远小于此, 仅作数组维度) */
#define MAX_HARTS 64

/* 子进程进入 ENTER 后到主进程发起首次并发 ENTER 之间的等待 (微秒).
 * 该窗口须小于载荷的运行时长, 使并发 ENTER 发生在飞地运行期间. */
#define ENTER_WINDOW_US 20000

/* 管理令牌: 非 0, 与纯 TEE_IOC_CREATE 登记的 0 区分开, 使本次进入确实经过令牌校验. */
#define MGMT_TOKEN 0x5A5A1234ULL

/* 父子进程共享的控制块 (fork 后各自持有同一份映射) */
struct shared_ctl {
	volatile int in_enter; /* 子进程已就绪, 即将发起 ENTER */
	long child_rc;		   /* 子进程 ENTER 的返回值 */
	int child_errno;	   /* 子进程 ENTER 失败时的 errno */
	int child_pin_hart;	   /* 子进程绑定到的 hart */
	int child_obs_hart;	   /* 子进程实际观测到的 hart (不可读时为 -1) */
	int parent_pin_hart;   /* 主进程绑定到的 hart */
	int parent_obs_hart;   /* 主进程实际观测到的 hart (不可读时为 -1) */
};

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

/* 直调 ioctl: 返回值可达 64 位 (ENTER/RESUME 把退出码打包在高 32 位), 避免 libc
 * ioctl 包装的 int 截断. */
static long tee_ioctl(int fd, unsigned long req, void *arg) {
	return syscall(SYS_ioctl, fd, req, arg);
}

/* 取本进程可运行的 hart 集合, 返回个数 (最多 max 个), 编号写入 harts[]; 失败返回 -1. */
static int online_harts(int *harts, int max) {
	cpu_set_t set;

	if (sched_getaffinity(0, sizeof(set), &set) != 0) {
		return -1;
	}
	int count = 0;
	for (int cpu = 0; cpu < CPU_SETSIZE && count < max; cpu++) {
		if (CPU_ISSET(cpu, &set)) {
			harts[count++] = cpu;
		}
	}
	return count;
}

/* 把本进程绑定到一个 hart 上, 成功返回 0. */
static int pin_to_hart(int hart) {
	cpu_set_t set;

	CPU_ZERO(&set);
	CPU_SET(hart, &set);
	return sched_setaffinity(0, sizeof(set), &set);
}

/* 绑定本进程并等到它确实运行在目标 hart 上, 返回实测 hart (不可读时为 -1, 绑定失败
 * 时为 -2). 绑定后本进程可能仍留在原 hart 上, 迁移要到下一个调度点才完成, 故不能
 * 绑定后立即读取实测值 -- 否则读到的仍是旧 hart, 使两进程的实测 hart 同为旧值. 内核
 * 保证任务只在其掩码内的 hart 上运行, 故等待上限取 1 秒, 正常几个调度点内即稳定. */
static int pin_and_settle(int hart) {
	if (pin_to_hart(hart) != 0) {
		return -2;
	}
	for (int i = 0; i < 1000; i++) {
		int cpu = sched_getcpu();
		if (cpu == hart || cpu < 0) {
			return cpu;
		}
		sched_yield();
		usleep(1000);
	}
	return sched_getcpu();
}

static void report(const char *name, int ok) {
	printf(">> regress-check %s: %s\n", name, ok ? "PASS" : "FAIL");
}

/* 把某一步的错误码写成可读文本: 0 表示成功, 其余为 errno.
 * 文本写入调用方提供的缓冲区, 故同一行若打印多个错误码需各备一个缓冲区. */
static const char *errno_text(int err, char *buf, size_t size) {
	if (err == 0) {
		snprintf(buf, size, "ok");
	} else {
		snprintf(buf, size, "%d (%s)", err, strerror(err));
	}
	return buf;
}

/* 飞地交还 Host 时的状态, 取自 enum tee_run_state. */
static const char *run_state_name(int state) {
	switch (state) {
	case TEE_RUN_SUSPENDED:
		return "suspended (resumable)";
	case TEE_RUN_EXITED:
		return "self-exited, code 0";
	case TEE_RUN_EXITED_ERR:
		return "self-exited, code non-zero";
	case TEE_RUN_ABORTED:
		return "aborted by M-mode (faulting access)";
	case TEE_RUN_HOST_KILL:
		return "killed by host";
	default:
		return "unknown state";
	}
}

/* 两个进程是否确实运行在不同 hart 上. 绑定到的目标 hart 不同即为保证 (内核只让任务在其
 * 掩码内的 hart 上运行); 实测值可读时再核对一次 -- 两者实测相同说明绑定未生效, 那样的
 * "全部被拒"不能证明占用检查有效. 实测值不可读 (-1) 时不作要求.
 * 子进程绑定失败 (-2) 或两个进程的目标 hart 相同, 则本用例的前提不成立. */
static int two_distinct_harts(const struct shared_ctl *ctl) {
	if (ctl->child_obs_hart == -2 || ctl->child_pin_hart == ctl->parent_pin_hart) {
		return 0;
	}
	if (ctl->child_obs_hart >= 0 && ctl->parent_obs_hart >= 0) {
		return ctl->child_obs_hart != ctl->parent_obs_hart;
	}
	return 1;
}

/* ---- main ---- */

int main(int argc, char **argv) {
	if (argc < 2) {
		fprintf(stderr, "Usage: %s <payload_path> [concurrent_attempts]\n", argv[0]);
		return 2;
	}

	int attempts = (argc > 2) ? atoi(argv[2]) : DEFAULT_ATTEMPTS;
	if (attempts < 1) {
		attempts = 1;
	}

	puts("=== concurrent ENTER/RESUME on one enclave ===");
	puts("An enclave runs on at most one hart at any time.\n");

	/* 判定前提: 至少两个 hart, 否则两次 ENTER 无法并发. */
	int harts[MAX_HARTS];
	int hart_count = online_harts(harts, MAX_HARTS);
	if (hart_count < 2) {
		printf("[info] only %d hart(s) online, cannot issue the two ENTER calls from "
			   "different harts, the precondition of this case does not hold\n",
			   hart_count);
		report("concurrent-enter-two-harts", 0);
		report("concurrent-enter-single-owner", 0);
		report("concurrent-enter-resume-rejected", 0);
		report("concurrent-enter-no-leak", 0);
		return 1;
	}

	/* 主进程绑定到 harts[0]; 子进程随后改为绑定 harts[1]. */
	int parent_obs = pin_and_settle(harts[0]);
	if (parent_obs == -2) {
		fprintf(stderr, "sched_setaffinity(hart %d): %s\n", harts[0], strerror(errno));
		return 2;
	}

	int fd = open(TEE_DEVICE_PATH, O_RDWR);
	if (fd < 0) {
		fprintf(stderr, "open " TEE_DEVICE_PATH ": %s\n", strerror(errno));
		return 1;
	}

	size_t payload_sz = 0;
	uint8_t *payload  = read_file(argv[1], &payload_sz);
	if (!payload) {
		fprintf(stderr, "read payload '%s' failed\n", argv[1]);
		close(fd);
		return 1;
	}
	printf("[info] payload '%s' loaded (%zu bytes)\n", argv[1], payload_sz);

	/* 池基线: 飞地终止后内存应回到此值. */
	struct tee_mem_info mem_before = {0}, mem_after = {0};
	int baseline_ok = (tee_ioctl(fd, TEE_IOC_GET_MEM, &mem_before) == 0);

	struct shared_ctl *ctl = mmap(
		NULL, sizeof(*ctl), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
	if (ctl == MAP_FAILED) {
		fprintf(stderr, "mmap shared ctl: %s\n", strerror(errno));
		free(payload);
		close(fd);
		return 1;
	}
	memset(ctl, 0, sizeof(*ctl));

	uint64_t enclave_id						  = 0;
	struct tee_enclave_token_args create_args = {
		.enclave_id = 0,
		.token		= MGMT_TOKEN,
	};
	if (tee_ioctl(fd, TEE_IOC_CREATE_EX, &create_args) < 0) {
		fprintf(stderr, "CREATE_EX: %s\n", strerror(errno));
		free(payload);
		close(fd);
		return 1;
	}
	enclave_id = create_args.enclave_id;
	printf("[info] enclave %lu created\n\n", (unsigned long)enclave_id);

	/* 并发 ENTER 与子进程的 ENTER 都携带正确令牌, 使拒绝只可能出自占用检查. */
	struct tee_enclave_args args = {
		.enclave_id	  = enclave_id,
		.enter.payload_ptr  = (uint64_t)(uintptr_t)payload,
		.enter.payload_size = payload_sz,
		.enter.argc		  = 0,
		.enter.argv_ptr	  = 0,
		.token		  = MGMT_TOKEN,
	};

	/* 并发 RESUME 同样携带正确令牌, 使拒绝只可能出自 resume.c 的占用检查. */
	struct tee_enclave_token_args resume_args = {
		.enclave_id = enclave_id,
		.token		= MGMT_TOKEN,
	};

	ctl->parent_pin_hart = harts[0];
	ctl->parent_obs_hart = parent_obs;

	pid_t child = fork();
	if (child < 0) {
		fprintf(stderr, "fork: %s\n", strerror(errno));
		free(payload);
		close(fd);
		return 1;
	}
	if (child == 0) {
		/* 子进程: 先绑定到另一个 hart, 再发起 ENTER. 该调用在飞地运行期间不会返回,
		 * 故本进程阻塞在此, 飞地正处于运行态且为 harts[1] 所占用. */
		ctl->child_pin_hart = harts[1];
		ctl->child_obs_hart = pin_and_settle(harts[1]);
		__atomic_store_n(&ctl->in_enter, 1, __ATOMIC_SEQ_CST);
		long rc			 = tee_ioctl(fd, TEE_IOC_ENTER, &args);
		ctl->child_rc	 = rc;
		ctl->child_errno = errno;
		_exit(0);
	}

	/* 主进程: 等子进程进入 ENTER, 再对同一飞地并发发起 ENTER 与 RESUME */
	while (!__atomic_load_n(&ctl->in_enter, __ATOMIC_SEQ_CST)) {
		usleep(200);
	}
	usleep(ENTER_WINDOW_US);

	int denied = 0, entered = 0, other = 0;
	int resume_denied = 0, resume_entered = 0, resume_other = 0;
	for (int i = 0; i < attempts; i++) {
		long rc = tee_ioctl(fd, TEE_IOC_ENTER, &args);
		if (rc >= 0) {
			entered++;
			printf(
				"  [try %d] unexpected ENTER: %s\n",
				i + 1,
				run_state_name(TEE_RUN_STATE(rc)));
		} else if (errno == EACCES) {
			/* 驱动把 M 模式的 SBI_ERR_DENIED 折算为 -EACCES. */
			denied++;
		} else {
			other++;
			printf(
				"  [try %d] ENTER failed with unexpected errno=%d (%s)\n",
				i + 1,
				errno,
				strerror(errno));
		}

		/* RESUME 的占用检查在 resume.c 内, 与 ENTER 的检查是两处独立判定, 故同一
		 * 占用窗口内再探一次. 飞地此刻仍由子进程持有, 本次 RESUME 必须被拒; 若它
		 * 真的恢复成功, 同一飞地就同时跑在两个 hart 上. */
		rc = tee_ioctl(fd, TEE_IOC_RESUME, &resume_args);
		if (rc >= 0) {
			resume_entered++;
			printf(
				"  [try %d] unexpected RESUME: %s\n",
				i + 1,
				run_state_name(TEE_RUN_STATE(rc)));
		} else if (errno == EACCES) {
			resume_denied++;
		} else {
			resume_other++;
			printf(
				"  [try %d] RESUME failed with unexpected errno=%d (%s)\n",
				i + 1,
				errno,
				strerror(errno));
		}
	}

	int child_status = 0;
	waitpid(child, &child_status, 0);

	char buf[64] = {0};

	printf(
		"\n[info] child ENTER: %s\n",
		(ctl->child_rc >= 0) ? run_state_name(TEE_RUN_STATE(ctl->child_rc))
							 : errno_text(ctl->child_errno, buf, sizeof(buf)));
	printf(
		"[info] concurrent ENTER x%d: denied=%d, entered=%d, other=%d\n",
		attempts,
		denied,
		entered,
		other);
	printf(
		"[info] concurrent RESUME x%d: denied=%d, entered=%d, other=%d\n",
		attempts,
		resume_denied,
		resume_entered,
		resume_other);
	printf(
		"[info] hart placement: parent pinned %d (observed %d), child pinned %d "
		"(observed %d)\n",
		ctl->parent_pin_hart,
		ctl->parent_obs_hart,
		ctl->child_pin_hart,
		ctl->child_obs_hart);

	int two_harts = two_distinct_harts(ctl);
	report("concurrent-enter-two-harts", two_harts);

	/* 判定: 子进程的进入成功, 且并发 ENTER 全部被拒 (无一进入, 无其它错误). */
	int pass =
		(ctl->child_rc >= 0 && entered == 0 && other == 0 && denied == attempts
		 && two_harts);
	report("concurrent-enter-single-owner", pass);

	/* 判定: 同一占用窗口内的并发 RESUME 同样全部被拒. */
	int resume_rejected = (ctl->child_rc >= 0 && resume_entered == 0 && resume_other == 0
						   && resume_denied == attempts);
	report("concurrent-enter-resume-rejected", resume_rejected);

	/* 清理: 终止飞地 (子进程的进入已交还 Host). TEE_IOC_SHUTDOWN 为 _IOW, 驱动
	 * copy_from_user 读取 struct tee_enclave_token_args (目标飞地编号 + 管理令牌);
	 * 本飞地由 TEE_IOC_CREATE_EX 创建, 令牌即 MGMT_TOKEN. 载荷已自行退出时槽位随之
	 * 释放, 此处只能因查不到编号而返回 EIO, 属正常情况. */
	struct tee_enclave_token_args sd_args = {
		.enclave_id = enclave_id,
		.token		= MGMT_TOKEN,
	};
	if (tee_ioctl(fd, TEE_IOC_SHUTDOWN, &sd_args) < 0 && errno != EIO) {
		fprintf(stderr, "[info] SHUTDOWN: %s\n", strerror(errno));
	}

	/* 无泄漏: 飞地占用的分区 (CREATE 1 个 + ENTER 按载荷扩展) 必须全部归还. */
	int no_leak = baseline_ok && (tee_ioctl(fd, TEE_IOC_GET_MEM, &mem_after) == 0)
				  && mem_before.free_total == mem_after.free_total
				  && mem_before.max_contiguous == mem_after.max_contiguous;
	printf(
		"[info] memory pool free partitions: before %lu (max contiguous %lu), after %lu "
		"(max contiguous %lu)\n",
		(unsigned long)mem_before.free_total,
		(unsigned long)mem_before.max_contiguous,
		(unsigned long)mem_after.free_total,
		(unsigned long)mem_after.max_contiguous);
	report("concurrent-enter-no-leak", no_leak);

	free(payload);
	close(fd);
	return (pass && resume_rejected && no_leak) ? 0 : 1;
}
