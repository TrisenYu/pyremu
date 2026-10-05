/* tee_enclave_drv.c — Linux enclave management driver
 *
 * Character device /dev/tee_enclave:
 *   Uses SBI ecall (ext_id=0x20221222) to communicate with the
 *   OpenSBI M-mode enclave extension (custom-opensbi-rs).
 *   Userspace ioctls manage the full enclave lifecycle.
 *
 * SBI register convention (from ext_ecall.c):
 *   a7 = 0x20221222 (ext_id)
 *   a6 = func_id
 *   a0..a4 = arguments (function-specific)
 *   Returns: a0 = error (0 = success), a1 = value
 *
 * ENTER / RESUME 把控制交还 host 属正常返回 (a0 >= 0): 飞地此刻的情形编码在 a1
 * (sbiret.value) 的运行状态枚举, 数值与用户态 tee_enclave.h 的 enum tee_run_state
 * 以及 M 侧 ext_ecall.c 的 ENCLAVE_RUN_* 一一对应. 本驱动的 ENTER/RESUME ioctl
 * 直接以该枚举作为返回值: rc >= 0 表示控制已交还 host, rc 即枚举值 (见
 * enum tee_run_state); rc < 0 为调用期失败 (errno), 载荷未被转入运行.
 * 其余 ioctl (CREATE/SUSPEND/SHUTDOWN) 保持 rc < 0 为错误、否则 0 的既有语义.
 *
 * SPDX-License-Identifier: MIT
 */

#include <asm/sbi.h>
#include <linux/context_tracking.h>
#include <linux/fs.h>
#include <linux/irqflags.h>
#include <linux/kernel.h>
#include <linux/miscdevice.h>
#include <linux/module.h>
#include <linux/preempt.h>
#include <linux/slab.h>
#include <linux/uaccess.h>
#include <linux/vmalloc.h>

#define DEVICE_NAME "tee_enclave"

/* SBI enclave extension */
#define SBI_ENCLAVE_EXT_ID	 0x20221222ULL
#define SBI_ENCLAVE_CREATE	 400
#define SBI_ENCLAVE_ENTER	 401
#define SBI_ENCLAVE_SHUTDOWN 403
#define SBI_ENCLAVE_SUSPEND	 404
#define SBI_ENCLAVE_RESUME	 405
#define SBI_ENCLAVE_GET_ID	 407
#define SBI_ENCLAVE_GET_MEM	 409
/* 410 需两个参数 (enclave_id, mgmt_token), 见 request_shutdown.c. */
#define SBI_ENCLAVE_REQUEST_SHUTDOWN 410
/* 503 至 505: 宿主应答飞地挂起的模块请求, 三者均以 "飞地编号 + 管理令牌" 鉴权
 * (见 enclave_api/ext_mod/). */
#define SBI_ENCLAVE_MODULE_REQ	 503
#define SBI_ENCLAVE_MODULE_SIZE	 504
#define SBI_ENCLAVE_MODULE_IMG 505

/* ---- ioctl command codes (must match userspace tee_enclave.h) ---- */

#define TEE_IOC_MAGIC			 'T'
#define TEE_IOC_CREATE			 _IO(TEE_IOC_MAGIC, 0)
#define TEE_IOC_ENTER			 _IOW(TEE_IOC_MAGIC, 1, struct tee_enclave_args)
#define TEE_IOC_GET_ID			 _IOR(TEE_IOC_MAGIC, 2, unsigned long long)
#define TEE_IOC_GET_MEM			 _IOR(TEE_IOC_MAGIC, 3, struct tee_mem_info)
#define TEE_IOC_SUSPEND			 _IO(TEE_IOC_MAGIC, 4)
#define TEE_IOC_RESUME			 _IOW(TEE_IOC_MAGIC, 5, struct tee_enclave_token_args)
#define TEE_IOC_SHUTDOWN		 _IOW(TEE_IOC_MAGIC, 6, struct tee_enclave_token_args)
#define TEE_IOC_REQUEST_SHUTDOWN _IOWR(TEE_IOC_MAGIC, 7, struct tee_enclave_token_args)
#define TEE_IOC_CREATE_EX		 _IOWR(TEE_IOC_MAGIC, 8, struct tee_enclave_token_args)
#define TEE_IOC_MODULE_REQ		 _IOWR(TEE_IOC_MAGIC, 9, struct tee_enclave_args)
#define TEE_IOC_MODULE_SIZE		 _IOWR(TEE_IOC_MAGIC, 10, struct tee_enclave_args)
#define TEE_IOC_MODULE_IMG	 _IOWR(TEE_IOC_MAGIC, 11, struct tee_enclave_args)

/* argv 序列化上限: 防御恶意超大 argc 或超长字符串导致内核过量分配. */
#define TEE_MAX_ARGC	  128
#define TEE_MAX_ARGSTRLEN 4096

struct tee_mem_info {
	unsigned long long free_total;
	unsigned long long max_contiguous;
};

/* 以管理令牌鉴权的 host 侧命令参数: "飞地编号 + 创建时登记的管理令牌" 这一对
 * 参数为 CREATE_EX(400)、SHUTDOWN(403)、RESUME(405)、REQUEST_SHUTDOWN(410)
 * 四者共用, 令牌由 M 模式在锁内比对后才执行 (见 create.c / shutdown.c /
 * resume.c / request_shutdown.c).
 *
 * 四处方向有别: CREATE_EX 的 enclave_id 是出参 (令牌入参), 其余三者两者皆入参;
 * 沿用 ABI 的 TEE_IOC_CREATE 只能登记令牌 0, 故需要真令牌时用 TEE_IOC_CREATE_EX. */
struct tee_enclave_token_args {
	unsigned long long enclave_id;
	unsigned long long token;
};

/* 模块请求的类别, 与 M 模式 ENCLAVE_MODULE_KIND_* 及用户态 tee_module_kind
 * 一一对应. 宿主按类别决定是报告映像字节数还是交付映像. */
#define TEE_MODULE_KIND_NONE 0
#define TEE_MODULE_KIND_SIZE 1
#define TEE_MODULE_KIND_LOAD 2

/* 模块编号取该值表示无挂起请求. */
#define TEE_MODULE_ID_NONE 0xFFFFFFFFULL

/* ENTER 与模块三个命令 (MODULE_REQ / MODULE_SIZE / MODULE_IMG) 共用的参数结构体.
 * enclave_id 与 token 为四者共有的入参: token 是创建该飞地时登记的管理令牌,
 * M 模式在锁内比对后才载入飞地 (见 enclave_api/enter.c 的 claim_enclave_for_hart);
 * 各命令的专属字段在匿名 union 中按命令取用. 505 号调用交付的字节写入该飞地的
 * S 模式运行时经 509 号调用登记的接收缓冲区, 其物理地址按该运行时自己的页表查询
 * 得到, 宿主不指定. */
struct tee_enclave_args {
	unsigned long long enclave_id;
	unsigned long long token;
	union {
		/* ENTER: 载荷指针与长度, 以及交给载荷的 argv. */
		struct {
			unsigned long long payload_ptr;
			unsigned long long payload_size;
			unsigned long long argc;
			unsigned long long argv_ptr;
		} enter;
		/* MODULE_REQ: 模块编号与请求类别均为出参. */
		struct {
			unsigned long long module_id;
			unsigned long long kind;
		} req;
		/* MODULE_SIZE: size 为入参, 取 0 表示宿主侧不存在该模块; status 为出参. */
		struct {
			unsigned long long size;
			unsigned long long status;
		} size;
		/* MODULE_IMG: buf_ptr 与 size 为入参 (用户缓冲区), 其余为出参.
		 * 505 号调用还以 a2 回传实际写入的字节数, 而内核的 sbi_ecall 只取回
		 * a0 与 a1, 故该值不回传用户态; 成功时它等于入参 size (见
		 * ext_mod/load.c), 飞地侧另经 508 号调用读取它. */
		struct {
			unsigned long long buf_ptr;
			unsigned long long size;
			unsigned long long status;
			unsigned long long pa;
		} image;
	};
};

/* ---- 转入飞地期间的 RCU 静默态 ----
 *
 * CREATE / CREATE_EX / ENTER / RESUME 成功时把控制交给飞地, 直到飞地 SUSPEND
 * 或结束才回到本驱动, 期间该 hart 一直在 M 模式驻留, 而内核对它的视角仍是
 * "在线且在内核上下文". 于是 RCU 在 stall 阈值过半后开始等待本 hart 报静止态,
 * 而它永远不报 (dmesg 出现 rcu self-detected stall).
 *
 * 按 KVM 处理客户机态的做法, 在这四个 ecall 前后成对标记: tee_guest_enter 让 RCU
 * 不再等待本 hart (进入 RCU 扩展静默态), tee_guest_exit 在控制交还后恢复.
 * 两者要求关中断 (context_tracking.c 的 __ct_user_enter 带
 * lockdep_assert_irqs_disabled) 且 current->mm 非空 (ioctl 调用方满足).
 * 关中断同时使这期间到达的宿主 IPI 推迟到控制交还之后处理, M 模式侧另有延迟
 * 投递保证它不丢, 见 enclave_ext/ext_ipi.c 的 enclave_ext_defer_smode_ipi.
 *
 * 用 local_irq_save/restore 而非 disable/enable: 交还与进入时的中断状态一致,
 * 调用方若本就关着中断, 出飞地后不会被本驱动擅自打开. */
static inline unsigned long tee_guest_enter(void) {
	unsigned long flags;

	preempt_disable();
	local_irq_save(flags);
	context_tracking_guest_enter();
	return flags;
}

static inline void tee_guest_exit(unsigned long flags) {
	context_tracking_guest_exit();
	local_irq_restore(flags);
	preempt_enable();
}

/* ---- SBI ecall helpers (use kernel's standard sbi_ecall) ---- */

static inline struct sbiret sbi_enclave_ecall_5(
	unsigned long func_id,
	unsigned long a0_val,
	unsigned long a1_val,
	unsigned long a2_val,
	unsigned long a3_val,
	unsigned long a4_val) {
	return sbi_ecall(
		SBI_ENCLAVE_EXT_ID, func_id, a0_val, a1_val, a2_val, a3_val, a4_val, 0);
}

/* ENTER 需六个参数: a0..a4 为 enclave_id/argc/argv/payload/载荷长度,
 * a5 为管理令牌 (见 enclave_api/enter.c). */
static inline struct sbiret sbi_enclave_ecall_6(
	unsigned long func_id,
	unsigned long a0_val,
	unsigned long a1_val,
	unsigned long a2_val,
	unsigned long a3_val,
	unsigned long a4_val,
	unsigned long a5_val) {
	return sbi_ecall(
		SBI_ENCLAVE_EXT_ID, func_id, a0_val, a1_val, a2_val, a3_val, a4_val, a5_val);
}

static inline struct sbiret sbi_enclave_ecall_2(
	unsigned long func_id, unsigned long a0_val, unsigned long a1_val) {
	return sbi_ecall(SBI_ENCLAVE_EXT_ID, func_id, a0_val, a1_val, 0, 0, 0, 0);
}

/* MODULE_SIZE: a0..a2 为 enclave_id/token/映像字节数 (见 ext_mod/load.c). */
static inline struct sbiret sbi_enclave_ecall_3(
	unsigned long func_id,
	unsigned long a0_val,
	unsigned long a1_val,
	unsigned long a2_val) {
	return sbi_ecall(SBI_ENCLAVE_EXT_ID, func_id, a0_val, a1_val, a2_val, 0, 0, 0);
}

/* MODULE_IMG: a0..a3 为 enclave_id/token/宿主缓冲区地址/映像字节数. */
static inline struct sbiret sbi_enclave_ecall_4(
	unsigned long func_id,
	unsigned long a0_val,
	unsigned long a1_val,
	unsigned long a2_val,
	unsigned long a3_val) {
	return sbi_ecall(SBI_ENCLAVE_EXT_ID, func_id, a0_val, a1_val, a2_val, a3_val, 0, 0);
}

static inline struct sbiret sbi_enclave_ecall_0(unsigned long func_id) {
	return sbi_ecall(SBI_ENCLAVE_EXT_ID, func_id, 0, 0, 0, 0, 0, 0);
}

static inline struct sbiret sbi_enclave_ecall_1(
	unsigned long func_id, unsigned long a0_val) {
	return sbi_ecall(SBI_ENCLAVE_EXT_ID, func_id, a0_val, 0, 0, 0, 0, 0);
}

/* Serialize a userspace argv (pointer array + every string) into one contiguous
 * kernel vmalloc buffer, laid out as [argc u64 slots][NUL-terminated strings]
 * with each slot rewritten to a kernel VA inside the same buffer.  M-mode reads
 * argv via sbi_load_u64/u8 under MPRV=S, so it must only see kernel (U=0) pages.
 *
 * Returns 0 and sets *out_buf on success (caller vfrees it), or a negative
 * errno on failure (in which case *out_buf is left NULL). */
static int serialize_argv(
	unsigned long long argv_ptr, unsigned long long argc, void **out_buf) {
	unsigned long *uargv = NULL;
	size_t *lens		 = NULL;
	void *buf			 = NULL;
	int rc				 = 0;

	*out_buf = NULL;
	if (argc == 0 || argv_ptr == 0) {
		return 0;
	}
	if (argc > TEE_MAX_ARGC) {
		return -E2BIG;
	}

	size_t nargc = (size_t)argc;

	/* Copy the pointer array once, so userspace cannot rewrite argv_ptr between
	 * the measure pass and the copy pass. */
	uargv = kmalloc_array(nargc, sizeof(unsigned long), GFP_KERNEL);
	if (!uargv) {
		return -ENOMEM;
	}
	if (copy_from_user(
			uargv,
			(void __user *)(unsigned long)argv_ptr,
			nargc * sizeof(unsigned long))) {
		rc = -EFAULT;
		goto out;
	}

	/* strnlen_user returns the byte count including the NUL terminator,
	 * 0 on fault, and count+1 when the string is too long. */
	lens = kmalloc_array(nargc, sizeof(size_t), GFP_KERNEL);
	if (!lens) {
		rc = -ENOMEM;
		goto out;
	}
	size_t total = nargc * sizeof(unsigned long);
	for (size_t i = 0; i < nargc; i++) {
		long n = strnlen_user((char __user *)uargv[i], TEE_MAX_ARGSTRLEN);
		if (n == 0) {
			rc = -EFAULT;
			goto out;
		}
		if (n > TEE_MAX_ARGSTRLEN) {
			rc = -E2BIG;
			goto out;
		}
		lens[i] = (size_t)n;
		total += (size_t)n;
	}

	buf = vmalloc(total);
	if (!buf) {
		rc = -ENOMEM;
		goto out;
	}

	/* Lay out [nargc u64 slots][NUL-terminated strings]; each slot holds a kernel
	 * VA pointing into this same buffer. */
	unsigned long *slots = buf;
	char *strs			 = (char *)(slots + nargc);
	size_t off			 = 0;
	for (size_t i = 0; i < nargc; i++) {
		slots[i] = (unsigned long)(strs + off);
		if (copy_from_user(strs + off, (char __user *)uargv[i], lens[i])) {
			vfree(buf);
			buf = NULL;
			rc	= -EFAULT;
			goto out;
		}
		off += lens[i];
	}

	*out_buf = buf;
	buf		 = NULL;

out:
	kfree(lens);
	kfree(uargv);
	return rc;
}

/* SBI 错误码 (a0, 负值) 与 errno 是两套独立编号: 直接把 SBI 码当 ioctl 返回值,
 * 用户态看到的 errno 会是 -SBI码 这个巧合值 (如 DENIED(-4) 被读成 EINTR), 与真实
 * 失败原因无关. 统一在此换算, 用户态 errno 才与原因对应; 未知码按 EIO 兜底,
 * 与原 `?: -EIO` 的兜底一致 (dmesg 里仍打印原始 SBI 码便于定位). */
static int sbi_err_to_errno(long err) {
	switch (err) {
	case SBI_ERR_FAILURE:
		return -EIO;
	case SBI_ERR_NOT_SUPPORTED:
		return -EOPNOTSUPP;
	case SBI_ERR_INVALID_PARAM:
	case SBI_ERR_INVALID_STATE:
		return -EINVAL;
	case SBI_ERR_DENIED:
	case SBI_ERR_DENIED_LOCKED:
		return -EACCES;
	case SBI_ERR_INVALID_ADDRESS:
		return -EFAULT;
	case SBI_ERR_ALREADY_AVAILABLE:
	case SBI_ERR_ALREADY_STARTED:
	case SBI_ERR_ALREADY_STOPPED:
		return -EALREADY;
	case SBI_ERR_NO_SHMEM:
		return -ENOMEM;
	case SBI_ERR_BAD_RANGE:
		return -ERANGE;
	case SBI_ERR_TIMEOUT:
		return -ETIMEDOUT;
	case SBI_ERR_IO:
	default:
		return -EIO;
	}
}

/* ---- ioctl dispatch ---- */

static long tee_ioctl(struct file *filp, unsigned int cmd, unsigned long arg) {
	struct sbiret ret;
	long rc = 0;

	switch (cmd) {

	/* ---- CREATE: spawn a new enclave (transfers control into it) ---- */
	case TEE_IOC_CREATE: {
		unsigned long ct_flags = tee_guest_enter();
		ret					   = sbi_enclave_ecall_0(SBI_ENCLAVE_CREATE);
		tee_guest_exit(ct_flags);
		/* CREATE enters the enclave; SUSPEND returns with a0 = enclave_id.
         * Negative a0 = SBI error, positive a0 = enclave_id on success. */
		if ((long)ret.error < 0) {
			pr_err("tee_enclave: CREATE(400) failed, error=%ld\n", ret.error);
			return sbi_err_to_errno((long)ret.error);
		}
		/* Write enclave_id back to userspace args[0]. */
		if (copy_to_user((void __user *)arg, &ret.error, sizeof(unsigned long long))) {
			return -EFAULT;
		}
		return 0;
	}

	/* ---- ENTER: load payload into existing enclave and enter it ---- */
	case TEE_IOC_ENTER: {
		struct tee_enclave_args kargs;
		void *payload_buf = NULL;
		void *argv_buf	  = NULL;

		if (copy_from_user(&kargs, (void __user *)arg, sizeof(kargs))) {
			return -EFAULT;
		}

		if (kargs.enter.payload_size == 0 || kargs.enter.payload_size > (128UL << 20)) {
			return -EINVAL;
		}

		/* Copy payload from userspace into kernel buffer.
         * The SBI ENTER handler calls copy_from_user() M-mode side
         * using the host virtual address; we pass the kernel buffer VA. */
		payload_buf = vmalloc(kargs.enter.payload_size);
		if (!payload_buf) {
			return -ENOMEM;
		}

		if (copy_from_user(
				payload_buf,
				(void __user *)(unsigned long)kargs.enter.payload_ptr,
				kargs.enter.payload_size)) {
			rc = -EFAULT;
			goto out_free_payload;
		}

		/* Serialize argv into a kernel buffer (u64 slots + strings) so M-mode
		 * reads only kernel pages; see serialize_argv() above. */
		rc = serialize_argv(kargs.enter.argv_ptr, kargs.enter.argc, &argv_buf);
		if (rc < 0) {
			goto out_free_payload;
		}

		/* SBI ENTER — on success, transfers into enclave (does not return). */
		unsigned long ct_flags = tee_guest_enter();
		ret					   = sbi_enclave_ecall_6(
			   SBI_ENCLAVE_ENTER,
			   kargs.enclave_id,
			   kargs.enter.argc,
			   (unsigned long)argv_buf,
			   (unsigned long)payload_buf,
			   kargs.enter.payload_size,
			   kargs.token);
		tee_guest_exit(ct_flags);

		if ((long)ret.error < 0) {
			pr_err("tee_enclave: ENTER(401) failed, error=%ld\n", ret.error);
			rc = sbi_err_to_errno((long)ret.error);
		} else {
			/* 控制已交还 host (a0 >= 0, 无 SBI 错误): a1 即运行状态枚举
			 * tee_run_state, 作为 ioctl 返回值交给用户态分辨飞地情形. */
			rc = (long)ret.value;
		}

	out_free_payload:
		vfree(argv_buf);
		vfree(payload_buf);
		return rc;
	}

	/* ---- GET_ID: query current mdid (0 = host) ---- */
	case TEE_IOC_GET_ID: {
		ret = sbi_enclave_ecall_0(SBI_ENCLAVE_GET_ID);
		/* a0 = current enclave id */
		if (copy_to_user((void __user *)arg, &ret.error, sizeof(unsigned long long))) {
			return -EFAULT;
		}
		return 0;
	}

	/* ---- GET_MEM: query free memory pool (2 MiB units) ---- */
	case TEE_IOC_GET_MEM: {
		struct tee_mem_info info;
		ret = sbi_enclave_ecall_0(SBI_ENCLAVE_GET_MEM);
		/* a0 = free_total, a1 = max_contiguous */
		info.free_total		= ret.error;
		info.max_contiguous = ret.value;
		if (copy_to_user((void __user *)arg, &info, sizeof(info))) {
			return -EFAULT;
		}
		return 0;
	}

	/* ---- SUSPEND: yield from enclave back to host ---- */
	case TEE_IOC_SUSPEND: {
		ret = sbi_enclave_ecall_0(SBI_ENCLAVE_SUSPEND);
		/* SUSPEND returns a0 = enclave_id on success (>0),
         * a0 < 0 on failure.  skip_regs_update=1 so mepc is already advanced. */
		if ((long)ret.error < 0) {
			pr_err("tee_enclave: SUSPEND(404) failed, error=%ld\n", ret.error);
			return sbi_err_to_errno((long)ret.error);
		}
		return 0;
	}

	/* ---- RESUME: resume a suspended enclave by id ---- */
	case TEE_IOC_RESUME: {
		struct tee_enclave_token_args kargs;

		if (copy_from_user(&kargs, (void __user *)arg, sizeof(kargs))) {
			return -EFAULT;
		}
		unsigned long ct_flags = tee_guest_enter();
		ret = sbi_enclave_ecall_2(SBI_ENCLAVE_RESUME, kargs.enclave_id, kargs.token);
		tee_guest_exit(ct_flags);
		/* On success, transfers into enclave (does not return). */
		if ((long)ret.error < 0) {
			pr_err(
				"tee_enclave: RESUME(405) id=%llu failed, error=%ld\n",
				kargs.enclave_id,
				ret.error);
			rc = sbi_err_to_errno((long)ret.error);
		} else {
			/* 同 ENTER: 控制已交还 host, a1 即运行状态枚举 tee_run_state. */
			rc = (long)ret.value;
		}
		return rc;
	}

	/* ---- SHUTDOWN: host-initiated enclave destruction ---- */
	case TEE_IOC_SHUTDOWN: {
		struct tee_enclave_token_args kargs;

		if (copy_from_user(&kargs, (void __user *)arg, sizeof(kargs))) {
			return -EFAULT;
		}
		ret = sbi_enclave_ecall_2(SBI_ENCLAVE_SHUTDOWN, kargs.enclave_id, kargs.token);
		if ((long)ret.error < 0) {
			pr_err(
				"tee_enclave: SHUTDOWN(403) id=%llu failed, error=%ld\n",
				kargs.enclave_id,
				ret.error);
			return sbi_err_to_errno((long)ret.error);
		}
		return 0;
	}

	/* ---- CREATE_EX: 同 CREATE, 但由 host 指定管理令牌 ---- */
	case TEE_IOC_CREATE_EX: {
		struct tee_enclave_token_args kargs;

		if (copy_from_user(&kargs, (void __user *)arg, sizeof(kargs))) {
			return -EFAULT;
		}

		/* a0 = 管理令牌; M 模式在进入飞地前读取并存进 meta. */
		unsigned long ct_flags = tee_guest_enter();
		ret					   = sbi_enclave_ecall_1(SBI_ENCLAVE_CREATE, kargs.token);
		tee_guest_exit(ct_flags);
		if ((long)ret.error < 0) {
			pr_err("tee_enclave: CREATE_EX(400) failed, error=%ld\n", ret.error);
			return sbi_err_to_errno((long)ret.error);
		}
		/* 同 CREATE: 成功时 a0 为 enclave_id (飞地 SUSPEND 交还后才走到这里). */
		kargs.enclave_id = (unsigned long long)ret.error;
		if (copy_to_user((void __user *)arg, &kargs, sizeof(kargs))) {
			return -EFAULT;
		}
		return 0;
	}

	/* ---- REQUEST_SHUTDOWN: 请求持有飞地的 hart 自行终止它 ----
	 *
	 * 与 SHUTDOWN 的分工: SHUTDOWN 由调用方所在 hart 直接终止, 飞地正跑在别的
	 * hart 上时 M 模式会拒绝同步终止并转为本请求 (返回 -EACCES), 结果由那条
	 * hart 上阻塞中的 ENTER/RESUME 回传. 本命令则是直接发起该请求: 不返回终止
	 * 结果, 只表示请求已受理并经 IPI 送达, 令牌在 M 模式锁内校验后才置位. */
	case TEE_IOC_REQUEST_SHUTDOWN: {
		struct tee_enclave_token_args kargs;

		if (copy_from_user(&kargs, (void __user *)arg, sizeof(kargs))) {
			return -EFAULT;
		}

		ret = sbi_enclave_ecall_2(
			SBI_ENCLAVE_REQUEST_SHUTDOWN, kargs.enclave_id, kargs.token);
		if ((long)ret.error < 0) {
			pr_err(
				"tee_enclave: REQUEST_SHUTDOWN(410) id=%llu failed, error=%ld\n",
				kargs.enclave_id,
				ret.error);
			return sbi_err_to_errno((long)ret.error);
		}
		return 0;
	}

	/* ---- MODULE_REQ: 取回飞地挂起的模块请求 (503) ----
	 *
	 * 读取不改动请求, 故宿主取映像文件失败后可在下次 RESUME 循环中重试 (见
	 * ext_mod/req.c). 无挂起请求时 module_id 取 TEE_MODULE_ID_NONE. */
	case TEE_IOC_MODULE_REQ: {
		struct tee_enclave_args kargs;

		if (copy_from_user(&kargs, (void __user *)arg, sizeof(kargs))) {
			return -EFAULT;
		}

		ret = sbi_enclave_ecall_2(SBI_ENCLAVE_MODULE_REQ, kargs.enclave_id, kargs.token);
		if ((long)ret.error < 0) {
			pr_err(
				"tee_enclave: MODULE_REQ(503) id=%llu failed, error=%ld\n",
				kargs.enclave_id,
				ret.error);
			return sbi_err_to_errno((long)ret.error);
		}
		kargs.req.module_id = (unsigned long long)ret.error;
		kargs.req.kind		= (unsigned long long)ret.value;
		if (copy_to_user((void __user *)arg, &kargs, sizeof(kargs))) {
			return -EFAULT;
		}
		return 0;
	}

	/* ---- MODULE_SIZE: 报告映像字节数 (504) ----
	 *
	 * size 取 0 表示宿主侧不存在该模块, 由 M 模式记入交付结果供飞地读取, 故此处
	 * 放行. 字节数不由本驱动设上界: 505 号调用交付的字节数须与它相等, 且不大于该
	 * 飞地登记的接收缓冲区, 故可交付的字节数以该飞地拥有的内存为限. */
	case TEE_IOC_MODULE_SIZE: {
		struct tee_enclave_args kargs;

		if (copy_from_user(&kargs, (void __user *)arg, sizeof(kargs))) {
			return -EFAULT;
		}

		ret = sbi_enclave_ecall_3(
			SBI_ENCLAVE_MODULE_SIZE, kargs.enclave_id, kargs.token, kargs.size.size);
		if ((long)ret.error < 0) {
			pr_err(
				"tee_enclave: MODULE_SIZE(504) id=%llu size=%llu failed, error=%ld\n",
				kargs.enclave_id,
				kargs.size.size,
				ret.error);
			return sbi_err_to_errno((long)ret.error);
		}
		kargs.size.status = (unsigned long long)ret.error;
		if (copy_to_user((void __user *)arg, &kargs, sizeof(kargs))) {
			return -EFAULT;
		}
		return 0;
	}

	/* ---- MODULE_IMG: 交付映像字节 (505) ----
	 *
	 * M 模式经 sbi_load_* 在置位 MPRV 的条件下读取该缓冲区, 该系列只访问 U=0
	 * 的页, 故用户缓冲区的页不可直接交给它, 先经内核缓冲区中转 (同 ENTER 的
	 * 载荷). 写入的目标物理地址按该飞地的 S 模式运行时自己的页表查询得到, 用户态
	 * 不指定. */
	case TEE_IOC_MODULE_IMG: {
		struct tee_enclave_args kargs;
		void *image_buf = NULL;

		if (copy_from_user(&kargs, (void __user *)arg, sizeof(kargs))) {
			return -EFAULT;
		}
		if (kargs.image.size == 0) {
			return -EINVAL;
		}

		image_buf = vmalloc(kargs.image.size);
		if (!image_buf) {
			return -ENOMEM;
		}
		if (copy_from_user(
				image_buf,
				(void __user *)(unsigned long)kargs.image.buf_ptr,
				kargs.image.size)) {
			rc = -EFAULT;
			goto out_free_image;
		}

		ret = sbi_enclave_ecall_4(
			SBI_ENCLAVE_MODULE_IMG,
			kargs.enclave_id,
			kargs.token,
			(unsigned long)image_buf,
			kargs.image.size);
		if ((long)ret.error < 0) {
			pr_err(
				"tee_enclave: MODULE_IMG(505) id=%llu size=%llu failed, error=%ld\n",
				kargs.enclave_id,
				kargs.image.size,
				ret.error);
			rc = sbi_err_to_errno((long)ret.error);
			goto out_free_image;
		}
		kargs.image.status = (unsigned long long)ret.error;
		kargs.image.pa	   = (unsigned long long)ret.value;
		rc				   = 0;
		if (copy_to_user((void __user *)arg, &kargs, sizeof(kargs))) {
			rc = -EFAULT;
		}
	out_free_image:
		vfree(image_buf);
		return rc;
	}

	default:
		return -ENOTTY;
	}

	return rc;
}

/* ---- file_operations ---- */

static const struct file_operations tee_fops = {
	.owner			= THIS_MODULE,
	.unlocked_ioctl = tee_ioctl,
};

static struct miscdevice tee_miscdev = {
	.minor = MISC_DYNAMIC_MINOR,
	.name  = DEVICE_NAME,
	.fops  = &tee_fops,
};

/* ---- module lifecycle ---- */

static int __init tee_init(void) {
	int rc = misc_register(&tee_miscdev);
	if (rc) {
		pr_err("tee_enclave: misc_register failed: %d\n", rc);
	} else {
		pr_info("tee_enclave: /dev/%s registered\n", DEVICE_NAME);
	}
	return rc;
}

static void __exit tee_exit(void) {
	misc_deregister(&tee_miscdev);
	pr_info("tee_enclave: unloaded\n");
}

module_init(tee_init);
module_exit(tee_exit);

MODULE_LICENSE("Dual MIT/GPL");
MODULE_AUTHOR("pyremu project");
MODULE_DESCRIPTION("TEE enclave lifecycle driver (bridge to OpenSBI via SBI ecall)");
