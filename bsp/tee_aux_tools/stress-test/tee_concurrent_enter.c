/* tee_concurrent_enter.c - 同一飞地的并发请求: 排队等待与按到达顺序服务
 *
 * 考察的不变量: 同一飞地至多在一个 hart 上运行. 载入飞地 (把它的上下文恢复到本 hart)
 * 之前必须先在 M 模式占用它, 否则两个 hart 会各自把同一飞地的上下文恢复到自己的现场,
 * 使同一份栈与内存被两个 hart 并发执行.
 *
 * 载入飞地有两条路径: ENTER 与 RESUME. 两者的占用判定分别在 enter.c 与 resume.c 内实现,
 * 是两处独立的判定, 故两条路径都要探测: 只探测 ENTER 无法说明 RESUME 的占用判定生效,
 * 反之亦然.
 *
 * 请求当前无法被服务时的处置: 请求方 hart 不返回宿主, 留在 M 模式内入队并等待, 条件成立
 * 再把这次请求做完. 本程序断言并发到达者取得的是执行结果本身而不是一个拒绝码.
 *
 * 用例构造:
 *   1. 主进程 CREATE_EX 一个飞地, 飞地停在引导挂起态;
 *   2. fork 出的第 i 个请求方绑定到第 i 个在线 hart, 各请求方因此运行在互不相同的 hart
 *      上; 主进程不单独占用 hart, 绑定到最后一个在线 hart, 与最后一个请求方共用该 hart;
 *      请求方 0 先行发起 ENTER 并占据飞地, 其余请求方随后发起;
 *   3. 后到达者此刻无法被服务, 于是阻塞在 M 模式, 飞地仍由先到者承载;
 *   4. 先到者的 ENTER 交还宿主后, 后到达者按到达顺序逐个取得飞地并返回;
 *   5. RESUME 路径对同一飞地重复第 2 至第 4 步.
 *
 * 断言:
 *   concurrent-enter-two-harts       各请求方确实运行在不同 hart 上;
 *   concurrent-enter-queued-served   后到达的 ENTER 未被拒绝, 且取得执行结果;
 *   concurrent-enter-resume-queued   后到达的 RESUME 未被拒绝, 且取得执行结果;
 *   concurrent-enter-serve-order     先到者先返回, 后到达者未与它并发执行;
 *   concurrent-enter-no-leak         飞地终止后内存池回到运行前的空闲分区数.
 *
 * 判定前提: 并发请求必须由不同 hart 发起. 单个 hart 上第二次 ENTER 只能在第一次返回之后
 * 才开始, 无从构成并发占用, 那样的通过毫无意义. 故本程序先把各进程分别绑定到不同的在线
 * hart, 并把实际 hart 编号记入共享控制块核对. 在线 hart 不少于两个时, 两个请求方即可分处
 * 两个 hart; 主进程在请求窗口内只做放行与等待, 不需要一个单独的 hart. 本 guest 的 CPU
 * 编号与 hart 编号一一对应.
 *
 * 载荷需长于 HOLD_WINDOW_US, 否则飞地在后到达者发起请求前就已让出.
 *
 * 未覆盖的判据:
 *   - 请求队列存满需要同时在队的请求数超过队列深度, 而本 guest 的 hart 数远小于队列深度,
 *     并发请求方至多等于 hart 数, 故本程序无从构造该情形.
 *   - "等待结束后队列为空" 同样不在本程序内: 队列位于 M 模式, 宿主侧不可读. 其结论可由本
 *     程序间接得到 -- 每个等待者的请求返回即该条目已出队并被服务, 故全部等待者返回即队列
 *     已空.
 *   - 终止使等待有界、终止清空队列、请求方返回宿主后本 hart 的 mtimecmp 与 mie.MTIE 已
 *     复位, 三者都在 M 模式内判定, 宿主侧无从观测.
 *
 * 编译: riscv64-linux-gnu-gcc -static -O2 tee_concurrent_enter.c -o bin/tee_concurrent_enter
 * 用法: ./tee_concurrent_enter <payload_path> [concurrent_requests]
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
#include <time.h>
#include <unistd.h>

#include "tee_enclave.h"

/* 主进程默认放行的并发请求方个数 */
#define DEFAULT_REQUESTS 2

/* 可绑定的 hart 数量上限 */
#define MAX_HARTS 64

/* 请求方个数上限: 每个请求方各占一个 hart, 主进程不单独占用 hart */
#define MAX_REQUESTERS MAX_HARTS

/* 先到者发起请求之后到放行其余请求方之间的等待.
 * 该窗口须小于载荷的运行时长, 使后到达者落在飞地运行期间. */
#define HOLD_WINDOW_US 20000

/* 相邻两个请求方放行的间隔. 到达次序是服务次序的判据, 故各请求方的到达时刻必须
 * 可分: 间隔需大于一次调度往返, 使先放行者的到达不会落到后放行者之后. */
#define RELEASE_STAGGER_US 20000

/* 管理令牌: 非 0, 与纯 TEE_IOC_CREATE 登记的 0 区分开, 使本次进入确实经过令牌校验. */
#define MGMT_TOKEN 0x5A5A1234ULL

/* 父子进程共享的控制块. 索引 0 为先到者, 其余为后到达者. */
struct shared_ctl {
	volatile int started[MAX_REQUESTERS]; /* 该请求方已绑定完成, 即将发起请求 */
	volatile int release[MAX_REQUESTERS]; /* 该请求方可以发起请求 */
	long rc[MAX_REQUESTERS];			  /* 请求的返回值 */
	int err[MAX_REQUESTERS];			  /* 请求失败时的 errno */
	int pin_hart[MAX_REQUESTERS];		  /* 绑定到的 hart */
	int obs_hart[MAX_REQUESTERS];		  /* 实测 hart (不可读时为 -1, 绑定失败时为 -2) */
	long long req_ns[MAX_REQUESTERS];	  /* 发起请求的时刻 */
	long long done_ns[MAX_REQUESTERS];	  /* 请求返回的时刻 */
};

/* ---- helpers ---- */

static uint8_t *read_file(
	const char *path,
	size_t *out_sz) {
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
static long tee_ioctl(
	int fd,
	unsigned long req,
	void *arg) {
	return syscall(SYS_ioctl, fd, req, arg);
}

/* 取本进程可运行的 hart 集合, 返回个数 (最多 max 个), 编号写入 harts[]; 失败返回 -1. */
static int online_harts(
	int *harts,
	int max) {
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

/* 绑定本进程并等到它确实运行在目标 hart 上, 返回实测 hart.
 * 绑定后本进程可能仍留在原 hart 上, 迁移要到下一个调度点才完成, 故不能
 * 绑定后立即读取实测值 -- 否则读到的仍是旧 hart, 使两个进程的实测 hart 同为旧值. 内核
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

/* 当前时刻, 单位纳秒. CLOCK_MONOTONIC 在全部进程之间取值一致, 故各请求方的发起时刻与
 * 返回时刻可以直接比较. */
static long long now_ns(void) {
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (long long)ts.tv_sec * 1000000000LL + (long long)ts.tv_nsec;
}

static void report(
	const char *name,
	int ok) {
	printf(">> regress-check %s: %s\n", name, ok ? "PASS" : "FAIL");
}

/* 把某一步的错误码写成可读文本: 0 表示成功, 其余为 errno.
 * 文本写入调用方提供的缓冲区, 故同一行若打印多个错误码需各备一个缓冲区. */
static const char *errno_text(
	int err,
	char *buf,
	size_t size) {
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
		return "aborted";
	case TEE_RUN_HOST_KILL:
		return "killed by host";
	default:
		return "unknown state";
	}
}

/* 各请求方是否确实运行在不同 hart 上. 绑定到的目标 hart 互不相同即为保证 (内核只让任务在
 * 其掩码内的 hart 上运行); 实测值可读时再核对一次 -- 两个实测相同说明绑定未生效, 那样的
 * "按到达顺序服务"不能证明占用判定有效. 实测值不可读 (-1) 时不作要求.
 * 有请求方绑定失败 (-2) 或两个请求方的目标 hart 相同, 则本用例的前提不成立. */
static int distinct_harts(
	const struct shared_ctl *ctl,
	int cnt) {
	for (int i = 0; i < cnt; i++) {
		if (ctl->obs_hart[i] == -2) {
			return 0;
		}
		for (int j = i + 1; j < cnt; j++) {
			if (ctl->pin_hart[i] == ctl->pin_hart[j]
				|| (ctl->obs_hart[i] >= 0 && ctl->obs_hart[j] >= 0
					&& ctl->obs_hart[i] == ctl->obs_hart[j])) {
				return 0;
			}
		}
	}
	return 1;
}

/* 全部请求方是否都取得了执行结果: 无一以错误码返回.
 * 这是本用例的核心判据 -- 并发到达者被拒绝时 (用户态 EACCES) 此处为假. */
static int all_served(
	const struct shared_ctl *ctl,
	int cnt) {
	for (int i = 0; i < cnt; i++) {
		if (ctl->rc[i] < 0) {
			return 0;
		}
	}
	return 1;
}

/* 返回次序是否与到达次序一致: 按记录到的发起时刻给出到达次序, 两两核对返回时刻按同一次序
 * 递增. 请求方个数不超过 hart 数, 两两比较即可覆盖全部次序, 不需要另建排序副本. */
static int serve_order_matches(
	const struct shared_ctl *ctl,
	int cnt) {
	for (int i = 0; i < cnt; i++) {
		for (int j = i + 1; j < cnt; j++) {
			int early = (ctl->req_ns[i] <= ctl->req_ns[j]) ? i : j;
			int late  = (early == i) ? j : i;

			if (ctl->done_ns[early] > ctl->done_ns[late]) {
				return 0;
			}
		}
	}
	return 1;
}

/* 把请求方编号按到达时刻或返回时刻排序打印, 供人核对 (判据由 serve_order_matches 给出).
 * by_arrival 非 0 时按到达时刻排序. */
static void print_order(
	const struct shared_ctl *ctl,
	int cnt,
	int by_arrival) {
	int order[MAX_REQUESTERS];

	for (int i = 0; i < cnt; i++) {
		order[i] = i;
	}
	for (int i = 0; i < cnt; i++) {
		for (int j = i + 1; j < cnt; j++) {
			long long a = by_arrival ? ctl->req_ns[order[j]] : ctl->done_ns[order[j]];
			long long b = by_arrival ? ctl->req_ns[order[i]] : ctl->done_ns[order[i]];

			if (a < b) {
				int t	 = order[i];
				order[i] = order[j];
				order[j] = t;
			}
		}
	}
	printf(" [");
	for (int i = 0; i < cnt; i++) {
		printf("%s%d", i ? " " : "", order[i]);
	}
	printf("]");
}

/* 打印一轮中各请求方的绑定位置、等待时长与结果, 以及到达次序与返回次序. */
static void print_phase(
	const char *label,
	const struct shared_ctl *ctl,
	int cnt) {
	char buf[64];

	for (int i = 0; i < cnt; i++) {
		long long wait_ms = (ctl->done_ns[i] - ctl->req_ns[i]) / 1000000;

		printf(
			"  [%s %d] hart %d (pinned %d), waited %lld ms, ",
			label,
			i,
			ctl->obs_hart[i],
			ctl->pin_hart[i],
			wait_ms);
		if (ctl->rc[i] >= 0) {
			printf("returned %s\n", run_state_name(TEE_RUN_STATE(ctl->rc[i])));
		} else {
			printf("failed with errno %s\n", errno_text(ctl->err[i], buf, sizeof(buf)));
		}
	}
	printf("[info] %s order: arrival", label);
	print_order(ctl, cnt, 1);
	printf(", return");
	print_order(ctl, cnt, 0);
	puts("");
}

/* 发起一轮并发请求, 成功返回 0, fork 失败返回 -1.
 *
 * 请求方 0 先行发起并占据飞地; 其余请求方在 HOLD_WINDOW_US 之后按 RELEASE_STAGGER_US 的
 * 间隔逐个放行, 各自由此落在飞地运行期间. 请求方当前无法被服务时不返回宿主, 而在 M 模式内
 * 入队等待, 故本函数返回之时各请求方都已取得结果. */
static int run_phase(
	struct shared_ctl *ctl,
	int fd,
	unsigned long req,
	void *arg,
	const int *hart_of,
	int cnt) {
	pid_t pids[MAX_REQUESTERS];
	int i;

	for (i = 0; i < cnt; i++) {
		ctl->started[i]	 = 0;
		ctl->release[i]	 = 0;
		ctl->rc[i]		 = 0;
		ctl->err[i]		 = 0;
		ctl->pin_hart[i] = -1;
		ctl->obs_hart[i] = -1;
		ctl->req_ns[i]	 = 0;
		ctl->done_ns[i]	 = 0;
	}

	for (i = 0; i < cnt; i++) {
		pid_t pid = fork();

		if (pid < 0) {
			perror("fork");
			return -1;
		}
		if (pid != 0) {
			pids[i] = pid;
			continue;
		}
		ctl->pin_hart[i] = hart_of[i];
		ctl->obs_hart[i] = pin_and_settle(hart_of[i]);
		__atomic_store_n(&ctl->started[i], 1, __ATOMIC_SEQ_CST);
		/* 先到者立即发起; 其余请求方等主进程放行 */
		if (i != 0) {
			while (!__atomic_load_n(&ctl->release[i], __ATOMIC_SEQ_CST)) {
				usleep(200);
			}
		}
		ctl->req_ns[i]	= now_ns();
		long rc			= tee_ioctl(fd, req, arg);
		ctl->err[i]		= errno;
		ctl->rc[i]		= rc;
		ctl->done_ns[i] = now_ns();
		_exit(0);
	}

	/* 等全部请求方完成绑定, 再等窗口使飞地确定已被先到者占据 */
	for (i = 0; i < cnt; i++) {
		while (!__atomic_load_n(&ctl->started[i], __ATOMIC_SEQ_CST)) {
			usleep(200);
		}
	}
	usleep(HOLD_WINDOW_US);
	for (i = 1; i < cnt; i++) {
		__atomic_store_n(&ctl->release[i], 1, __ATOMIC_SEQ_CST);
		if (i + 1 < cnt) {
			usleep(RELEASE_STAGGER_US);
		}
	}
	for (i = 0; i < cnt; i++) {
		waitpid(pids[i], NULL, 0);
	}
	return 0;
}

/* ---- main ---- */

int main(
	int argc,
	char **argv) {
	if (argc < 2) {
		fprintf(stderr, "Usage: %s <payload_path> [concurrent_requests]\n", argv[0]);
		return 2;
	}

	int want = (argc > 2) ? atoi(argv[2]) : DEFAULT_REQUESTS;
	if (want < 1) {
		want = 1;
	}

	puts("=== concurrent ENTER/RESUME on one enclave ===");
	puts("An enclave runs on at most one hart at a time.");
	puts("A request that cannot be served now waits in M-mode until it can.\n");

	/* 判定前提: 每个请求方各占一个 hart, 请求方至少两个才构成并发, 故至少需要两个
	 * hart. 主进程不单独占用 hart, 它与最后一个请求方共用该请求方的 hart, 见下方对主
	 * 进程的绑定. */
	int harts[MAX_HARTS];
	int hart_count = online_harts(harts, MAX_HARTS);
	if (hart_count < 0) {
		fprintf(stderr, "sched_getaffinity: %s\n", strerror(errno));
		return 2;
	}
	int requester_cnt = hart_count;
	if (requester_cnt > want) {
		requester_cnt = want;
	}
	if (requester_cnt > MAX_REQUESTERS) {
		requester_cnt = MAX_REQUESTERS;
	}
	if (requester_cnt < 2) {
		printf(
			"[info] %d hart(s) online allow %d concurrent request(s), two concurrent "
			"requests need at least 2 harts, the precondition of this case does not "
			"hold\n",
			hart_count,
			requester_cnt);
		report("concurrent-enter-two-harts", 0);
		report("concurrent-enter-queued-served", 0);
		report("concurrent-enter-resume-queued", 0);
		report("concurrent-enter-serve-order", 0);
		report("concurrent-enter-no-leak", 0);
		return 1;
	}
	printf(
		"[info] %d hart(s) online, %d concurrent request(s)\n",
		hart_count,
		requester_cnt);

	/* 主进程绑定到最后一个在线 hart, 各请求方绑定在线 hart 的前 requester_cnt 个. 请求方
	 * 个数不超过 hart 数, 故与主进程共用的那个 hart 上至多有一个请求方, 即最后一个请求
	 * 方; 该请求方在放行之前于用户态轮询, 放行之后阻塞在 M 模式, 故主进程在放行完成之前
	 * 总能被调度. 全部放行之后主进程只剩 waitpid, 不再需要被调度. */
	int parent_hart = harts[hart_count - 1];
	if (pin_and_settle(parent_hart) == -2) {
		fprintf(stderr, "sched_setaffinity(hart %d): %s\n", parent_hart, strerror(errno));
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

	/* 各请求方都携带正确令牌, 故失败只可能出自占用判定的处置, 而非令牌校验. */
	struct tee_enclave_args args = {
		.enclave_id			= enclave_id,
		.enter.payload_ptr	= (uint64_t)(uintptr_t)payload,
		.enter.payload_size = payload_sz,
		.enter.argc			= 0,
		.enter.argv_ptr		= 0,
		.token				= MGMT_TOKEN,
	};
	struct tee_enclave_token_args resume_args = {
		.enclave_id = enclave_id,
		.token		= MGMT_TOKEN,
	};

	int req_harts[MAX_REQUESTERS];
	for (int i = 0; i < requester_cnt; i++) {
		req_harts[i] = harts[i];
	}

	/* ---- ENTER 路径 ---- */
	if (run_phase(ctl, fd, TEE_IOC_ENTER, &args, req_harts, requester_cnt) != 0) {
		free(payload);
		close(fd);
		return 1;
	}
	printf("[info] ENTER phase\n");
	print_phase("ENTER", ctl, requester_cnt);

	int enter_two_harts = distinct_harts(ctl, requester_cnt);
	int enter_served	= all_served(ctl, requester_cnt);
	int enter_order		= serve_order_matches(ctl, requester_cnt);

	/* ---- RESUME 路径 ---- */
	int resume_two_harts = 0, resume_served = 0, resume_order = 0, resume_ran = 0;
	if (enter_served && TEE_RUN_STATE(ctl->rc[0]) == TEE_RUN_SUSPENDED) {
		if (run_phase(ctl, fd, TEE_IOC_RESUME, &resume_args, req_harts, requester_cnt)
			!= 0) {
			free(payload);
			close(fd);
			return 1;
		}
		printf("[info] RESUME phase\n");
		print_phase("RESUME", ctl, requester_cnt);
		resume_ran		 = 1;
		resume_two_harts = distinct_harts(ctl, requester_cnt);
		resume_served	 = all_served(ctl, requester_cnt);
		resume_order	 = serve_order_matches(ctl, requester_cnt);
	} else if (enter_served) {
		printf(
			"[info] the enclave returned %s after the ENTER phase, the RESUME case "
			"needs a payload that suspends\n",
			run_state_name(TEE_RUN_STATE(ctl->rc[0])));
	} else {
		printf(
			"[info] the ENTER phase did not obtain an execution result, the RESUME "
			"case cannot be constructed\n");
	}

	/* 清理: 终止飞地. TEE_IOC_SHUTDOWN 为 _IOW, 驱动 copy_from_user 读取
	 * struct tee_enclave_token_args (目标飞地编号 + 管理令牌); 本飞地由
	 * TEE_IOC_CREATE_EX 创建, 令牌即 MGMT_TOKEN. 载荷已自行退出时槽位随之释放, 此处
	 * 只能因查不到编号而返回 EIO, 属正常情况. */
	struct tee_enclave_token_args sd_args = {
		.enclave_id = enclave_id,
		.token		= MGMT_TOKEN,
	};
	if (tee_ioctl(fd, TEE_IOC_SHUTDOWN, &sd_args) < 0 && errno != EIO) {
		fprintf(stderr, "[info] SHUTDOWN: %s\n", strerror(errno));
	}

	/* 无泄漏: 飞地占用的分区必须全部归还. */
	int no_leak = baseline_ok && (tee_ioctl(fd, TEE_IOC_GET_MEM, &mem_after) == 0)
				  && mem_before.free_total == mem_after.free_total
				  && mem_before.max_contiguous == mem_after.max_contiguous;
	printf(
		"[info] memory pool free partitions: before %lu (max contiguous %lu), after "
		"%lu (max contiguous %lu)\n",
		(unsigned long)mem_before.free_total,
		(unsigned long)mem_before.max_contiguous,
		(unsigned long)mem_after.free_total,
		(unsigned long)mem_after.max_contiguous);

	report(
		"concurrent-enter-two-harts",
		enter_two_harts && (resume_ran ? resume_two_harts : 1));
	report("concurrent-enter-queued-served", enter_served);
	report("concurrent-enter-resume-queued", resume_served);
	report(
		"concurrent-enter-serve-order", enter_order && (resume_ran ? resume_order : 0));
	report("concurrent-enter-no-leak", no_leak);

	free(payload);
	close(fd);
	return (enter_two_harts && enter_served && resume_served && enter_order
			&& (resume_ran ? resume_order : 0) && no_leak)
			   ? 0
			   : 1;
}
