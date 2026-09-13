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
#include <linux/fs.h>
#include <linux/kernel.h>
#include <linux/miscdevice.h>
#include <linux/module.h>
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

/* ---- ioctl command codes (must match userspace tee_enclave.h) ---- */

#define TEE_IOC_MAGIC	 'T'
#define TEE_IOC_CREATE	 _IO(TEE_IOC_MAGIC, 0)
#define TEE_IOC_ENTER	 _IOW(TEE_IOC_MAGIC, 1, struct tee_enter_args)
#define TEE_IOC_GET_ID	 _IOR(TEE_IOC_MAGIC, 2, unsigned long long)
#define TEE_IOC_GET_MEM	 _IOR(TEE_IOC_MAGIC, 3, struct tee_mem_info)
#define TEE_IOC_SUSPEND	 _IO(TEE_IOC_MAGIC, 4)
#define TEE_IOC_RESUME	 _IOW(TEE_IOC_MAGIC, 5, unsigned long long)
#define TEE_IOC_SHUTDOWN _IO(TEE_IOC_MAGIC, 6)

/* argv 序列化上限: 防御恶意超大 argc 或超长字符串导致内核过量分配. */
#define TEE_MAX_ARGC	 128
#define TEE_MAX_ARGSTRLEN 4096

struct tee_enter_args {
	unsigned long long enclave_id;
	unsigned long long payload_ptr;
	unsigned long long payload_size;
	unsigned long long argc;
	unsigned long long argv_ptr;
};

struct tee_mem_info {
	unsigned long long free_total;
	unsigned long long max_contiguous;
};

/* ---- SBI ecall helpers (use kernel's standard sbi_ecall) ---- */

static inline struct sbiret sbi_enclave_ecall_5(
	unsigned long func_id,
	unsigned long a0_val, unsigned long a1_val,
	unsigned long a2_val, unsigned long a3_val, unsigned long a4_val)
{
	return sbi_ecall(SBI_ENCLAVE_EXT_ID, func_id,
			 a0_val, a1_val, a2_val, a3_val, a4_val, 0);
}

static inline struct sbiret sbi_enclave_ecall_0(unsigned long func_id)
{
	return sbi_ecall(SBI_ENCLAVE_EXT_ID, func_id, 0, 0, 0, 0, 0, 0);
}

static inline struct sbiret sbi_enclave_ecall_1(
	unsigned long func_id, unsigned long a0_val)
{
	return sbi_ecall(SBI_ENCLAVE_EXT_ID, func_id, a0_val, 0, 0, 0, 0, 0);
}

/* Serialize a userspace argv (pointer array + every string) into one contiguous
 * kernel vmalloc buffer, laid out as [argc u64 slots][NUL-terminated strings]
 * with each slot rewritten to a kernel VA inside the same buffer.  M-mode reads
 * argv via sbi_load_u64/u8 under MPRV=S, so it must only see kernel (U=0) pages.
 *
 * Returns 0 and sets *out_buf on success (caller vfrees it), or a negative
 * errno on failure (in which case *out_buf is left NULL). */
static int serialize_argv(unsigned long long argv_ptr, unsigned long long argc,
			  void **out_buf)
{
	unsigned long *uargv = NULL;
	size_t *lens = NULL;
	void *buf = NULL;
	int rc = 0;

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
	if (copy_from_user(uargv, (void __user *)(unsigned long)argv_ptr,
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
	char *strs = (char *)(slots + nargc);
	size_t off = 0;
	for (size_t i = 0; i < nargc; i++) {
		slots[i] = (unsigned long)(strs + off);
		if (copy_from_user(strs + off, (char __user *)uargv[i], lens[i])) {
			vfree(buf);
			buf = NULL;
			rc = -EFAULT;
			goto out;
		}
		off += lens[i];
	}

	*out_buf = buf;
	buf = NULL;

out:
	kfree(lens);
	kfree(uargv);
	return rc;
}

/* ---- ioctl dispatch ---- */

static long tee_ioctl(struct file *filp, unsigned int cmd, unsigned long arg) {
	struct sbiret ret;
	long rc = 0;

	switch (cmd) {

	/* ---- CREATE: spawn a new enclave (transfers control into it) ---- */
	case TEE_IOC_CREATE: {
		ret = sbi_enclave_ecall_0(SBI_ENCLAVE_CREATE);
		/* CREATE enters the enclave; SUSPEND returns with a0 = enclave_id.
         * Negative a0 = SBI error, positive a0 = enclave_id on success. */
		if ((long)ret.error < 0) {
			pr_err("tee_enclave: CREATE(400) failed, error=%ld\n", ret.error);
			return (long)ret.error ?: -EIO;
		}
		/* Write enclave_id back to userspace args[0]. */
		if (copy_to_user((void __user *)arg, &ret.error, sizeof(unsigned long long))) {
			return -EFAULT;
		}
		return 0;
	}

	/* ---- ENTER: load payload into existing enclave and enter it ---- */
	case TEE_IOC_ENTER: {
		struct tee_enter_args kargs;
		void *payload_buf = NULL;
		void *argv_buf	  = NULL;

		if (copy_from_user(&kargs, (void __user *)arg, sizeof(kargs))) {
			return -EFAULT;
		}

		if (kargs.payload_size == 0 || kargs.payload_size > (128UL << 20)) {
			return -EINVAL;
		}

		/* Copy payload from userspace into kernel buffer.
         * The SBI ENTER handler calls copy_from_user() M-mode side
         * using the host virtual address; we pass the kernel buffer VA. */
		payload_buf = vmalloc(kargs.payload_size);
		if (!payload_buf) {
			return -ENOMEM;
		}

		if (copy_from_user(
				payload_buf,
				(void __user *)(unsigned long)kargs.payload_ptr,
				kargs.payload_size)) {
			rc = -EFAULT;
			goto out_free_payload;
		}

		/* Serialize argv into a kernel buffer (u64 slots + strings) so M-mode
		 * reads only kernel pages; see serialize_argv() above. */
		rc = serialize_argv(kargs.argv_ptr, kargs.argc, &argv_buf);
		if (rc < 0) {
			goto out_free_payload;
		}

		/* SBI ENTER — on success, transfers into enclave (does not return). */
		ret = sbi_enclave_ecall_5(
			SBI_ENCLAVE_ENTER,
			kargs.enclave_id,
			kargs.argc,
			(unsigned long)argv_buf,
			(unsigned long)payload_buf,
			kargs.payload_size);

		if ((long)ret.error < 0) {
			pr_err("tee_enclave: ENTER(401) failed, error=%ld\n", ret.error);
			rc = (long)ret.error ?: -EIO;
		} else {
			/* 控制已交还 host (a0 >= 0, 无 SBI 错误): a1 即运行状态枚举
			 * tee_run_state, 作为 ioctl 返回值交给用户态分辨飞地情形. */
			rc = (long)ret.value;
		}

	out_free_argv:
		vfree(argv_buf);
	out_free_payload:
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
			return (long)ret.error ?: -EIO;
		}
		return 0;
	}

	/* ---- RESUME: resume a suspended enclave by id ---- */
	case TEE_IOC_RESUME: {
		unsigned long long enclave_id;
		if (copy_from_user(&enclave_id, (void __user *)arg, sizeof(enclave_id))) {
			return -EFAULT;
		}
		ret = sbi_enclave_ecall_1(SBI_ENCLAVE_RESUME, enclave_id);
		/* On success, transfers into enclave (does not return). */
		if ((long)ret.error < 0) {
			pr_err(
				"tee_enclave: RESUME(405) id=%llu failed, error=%ld\n",
				enclave_id,
				ret.error);
			rc = (long)ret.error ?: -EIO;
		} else {
			/* 同 ENTER: 控制已交还 host, a1 即运行状态枚举 tee_run_state. */
			rc = (long)ret.value;
		}
		return rc;
	}

	/* ---- SHUTDOWN: host-initiated enclave destruction ---- */
	case TEE_IOC_SHUTDOWN: {
		unsigned long long enclave_id;
		if (copy_from_user(&enclave_id, (void __user *)arg, sizeof(enclave_id))) {
			return -EFAULT;
		}
		ret = sbi_enclave_ecall_1(SBI_ENCLAVE_SHUTDOWN, enclave_id);
		if ((long)ret.error < 0) {
			pr_err(
				"tee_enclave: SHUTDOWN(403) id=%llu failed, error=%ld\n",
				enclave_id,
				ret.error);
			return (long)ret.error ?: -EIO;
		}
		return 0;
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

MODULE_LICENSE("MIT");
MODULE_AUTHOR("pyremu project");
MODULE_DESCRIPTION("TEE enclave lifecycle driver (SBI ecall bridge)");
