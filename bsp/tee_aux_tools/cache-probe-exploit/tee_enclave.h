/* tee_enclave.h - Linux TEE enclave driver userspace interface
 *
 * /dev/tee_enclave: ioctl-based enclave lifecycle management.
 * SBI ecall (ext_id=0x20221222) bridges to OpenSBI M-mode enclave extension.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef TEE_ENCLAVE_H
#define TEE_ENCLAVE_H

#include <stdint.h>
#include <sys/ioctl.h>

#define TEE_DEVICE_PATH "/dev/tee_enclave"

/* ---- SBI function IDs (from custom-opensbi-rs) ---- */

#define SBI_ENCLAVE_CREATE			  400
#define SBI_ENCLAVE_ENTER			  401
#define SBI_ENCLAVE_SHUTDOWN		  403
#define SBI_ENCLAVE_SUSPEND			  404
#define SBI_ENCLAVE_RESUME			  405
#define SBI_ENCLAVE_GET_ID			  407
#define SBI_ENCLAVE_GET_HARTID		  408
#define SBI_ENCLAVE_GET_AVAILABLE_MEM 409
#define SBI_ENCLAVE_REQUEST_SHUTDOWN  410
#define SBI_ENCLAVE_MEM_ALLOC		  500
/* 503 至 505: 宿主应答飞地挂起的模块请求, 三者均以 "飞地编号 + 管理令牌" 鉴权.
 * 508 为飞地侧调用: 取回本模块的交付结果. */
#define SBI_ENCLAVE_MODULE_REQ		  503
#define SBI_ENCLAVE_MODULE_SIZE		  504
#define SBI_ENCLAVE_MODULE_IMG	  505
#define SBI_ENCLAVE_MODULE_RESULT	  508

/* ---- ioctl command codes ---- */

#define TEE_IOC_MAGIC 'T'

/* Create a new enclave.  Returns enclave_id; does NOT return if successful
 * (execution transfers into enclave).  Caller sees return only on error. */
#define TEE_IOC_CREATE _IO(TEE_IOC_MAGIC, 0)

/* Enter an existing enclave with a payload.
 * Returns only when the enclave hands control back to the host (suspended /
 * exited / aborted / host-requested teardown), or on a call-time error. */
#define TEE_IOC_ENTER _IOW(TEE_IOC_MAGIC, 1, struct tee_enclave_args)

/* Query current mdid (0 = host). */
#define TEE_IOC_GET_ID _IOR(TEE_IOC_MAGIC, 2, uint64_t)

/* Query available memory pool (2 MiB units). */
#define TEE_IOC_GET_MEM _IOR(TEE_IOC_MAGIC, 3, struct tee_mem_info)

/* Suspend current enclave, returning to host.  Only valid inside enclave. */
#define TEE_IOC_SUSPEND _IO(TEE_IOC_MAGIC, 4)

/* Resume a previously suspended enclave by id.
 * Returns under the same rules as TEE_IOC_ENTER (see tee_run_state).
 * 入参为 struct tee_enclave_token_args: 目标飞地编号 + 管理令牌. */
#define TEE_IOC_RESUME _IOW(TEE_IOC_MAGIC, 5, struct tee_enclave_token_args)

/* Shutdown current enclave (clears memory, frees slot, mfence.did).
 * 入参为 struct tee_enclave_token_args: 目标飞地编号 + 管理令牌.
 *
 * 目标飞地正跑在别的 hart 上时不做同步终止: M 模式会代为置位终止请求并向该
 * hart 发定向 IPI, 本调用返回 -EACCES, 终止结果由那条 hart 上阻塞中的
 * ENTER/RESUME 回传. 需要主动发起该请求而不等待终止结果时用
 * TEE_IOC_REQUEST_SHUTDOWN. */
#define TEE_IOC_SHUTDOWN _IOW(TEE_IOC_MAGIC, 6, struct tee_enclave_token_args)

/* 请求持有该飞地的 hart 自行终止它. 与 TEE_IOC_SHUTDOWN 的分工: 本命令只表示
 * 请求已受理并经 IPI 送达, 不返回终止结果; 令牌在 M 模式锁内校验后才置位.
 * 飞地自己经 SBI 411 读到该请求后走自退出路径. */
#define TEE_IOC_REQUEST_SHUTDOWN _IOWR(TEE_IOC_MAGIC, 7, struct tee_enclave_token_args)

/* 同 TEE_IOC_CREATE, 但由 host 指定管理令牌 (入参 token, 出参 enclave_id).
 * TEE_IOC_CREATE 登记的令牌恒为 0, 故需要真令牌时改用本命令. */
#define TEE_IOC_CREATE_EX _IOWR(TEE_IOC_MAGIC, 8, struct tee_enclave_token_args)

/* 模块三个命令共用 struct tee_enclave_args, 各自的专属字段在其中的匿名 union 中
 * 按命令取用. 505 的映像缓冲区经内核缓冲区中转 (M 模式只读 U=0 的页). */
#define TEE_IOC_MODULE_REQ	 _IOWR(TEE_IOC_MAGIC, 9, struct tee_enclave_args)
#define TEE_IOC_MODULE_SIZE	 _IOWR(TEE_IOC_MAGIC, 10, struct tee_enclave_args)
#define TEE_IOC_MODULE_IMG _IOWR(TEE_IOC_MAGIC, 11, struct tee_enclave_args)

/* ---- ENTER / RESUME 的 ioctl 返回值 (运行状态) ----
 *
 * M 模式把控制交还 host 时, 按 SBI 返回规范把本次运行的状态枚举填入 a1
 * (sbiret.value), a0 仅在调用期出错时为负错误码。驱动据此把 ENTER/RESUME 的
 * ioctl 返回值定义为下面的 tee_run_state 枚举值:
 *
 *   rc >= 0 : 控制已交还 host, rc 即枚举值, 含义见 enum tee_run_state;
 *   rc <  0 : 调用期失败 (errno), 载荷未被转入运行, 无状态可读.
 *
 * 该枚举描述"飞地此刻处于什么情形"(发起方 + 退出码), 而不是成功/失败结论,
 * 调用方据此决定继续 RESUME、判成功或判失败.
 *
 *   数值须与 bsp/custom-opensbi/include/enclave_ext/enclave_api.h 的
 *   ENCLAVE_RUN_* 枚举保持一致, 修改需两侧同步. */
enum tee_run_state {
	TEE_RUN_SUSPENDED  = 0, /* 时间片配额让出: 载荷存活挂起, 应 RESUME 继续 (非终止) */
	TEE_RUN_EXITED	   = 1, /* 载荷自行结束 (SHUTDOWN), 退出码 0: 自然跑完 */
	TEE_RUN_EXITED_ERR = 2, /* 载荷自行结束 (SHUTDOWN), 退出码非 0: 载荷自报出错 */
	TEE_RUN_ABORTED	   = 3, /* 载荷违规访问, 被 M 模式强制终止, 非自行退出 */
	TEE_RUN_HOST_KILL  = 4, /* host 请求终止, 载荷未运行到退出 */
};

/* ---- ENTER / RESUME 返回值的退出码解包 ----
 *
 * 载荷自行结束 (EXITED / EXITED_ERR) 时, M 模式把退出码打包进 a1 的高 32 位,
 * 低 32 位仍为 enum tee_run_state 枚举值. host 侧以如下宏拆分 ioctl 返回值:
 *   TEE_RUN_STATE(rc) -> 低 32 位运行状态 (与 enum tee_run_state 比较);
 *   TEE_EXIT_CODE(rc) -> 高 32 位退出码 (仅 EXITED_ERR 时非零).
 * 其余状态 (SUSPENDED / ABORTED / HOST_KILL) 高 32 位恒 0. 调用期失败 rc<0 时
 * 无打包语义, 不应调用解包宏. */
#define TEE_RUN_STATE(rc) ((int)((rc) & 0xFFFFFFFFULL))
#define TEE_EXIT_CODE(rc) ((uint64_t)((rc) >> 32))

/* ---- parameter structures ---- */

struct tee_mem_info {
	uint64_t free_total;	 /* free 2 MiB partitions */
	uint64_t max_contiguous; /* largest contiguous run */
};

/* 以管理令牌鉴权的 host 侧命令参数: "飞地编号 + 创建时登记的管理令牌" 这一对
 * 参数为 CREATE_EX(400)、SHUTDOWN(403)、RESUME(405)、REQUEST_SHUTDOWN(410)
 * 四者共用. 其中 CREATE_EX 的 enclave_id 是出参 (令牌入参), 其余三者两者皆入参.
 * 纯 TEE_IOC_CREATE 登记的令牌恒为 0, 故需要真令牌时用 TEE_IOC_CREATE_EX. */
struct tee_enclave_token_args {
	uint64_t enclave_id; /* 目标飞地 id (CREATE_EX 时为出参: 新飞地 id) */
	uint64_t token;		 /* 创建时登记的管理令牌 */
};

/* 模块请求的类别, 与 M 模式的 ENCLAVE_MODULE_KIND_* 一致. 宿主按类别决定是
 * 报告映像字节数还是交付映像. */
#define TEE_MODULE_KIND_NONE 0
#define TEE_MODULE_KIND_SIZE 1
#define TEE_MODULE_KIND_LOAD 2

/* 交付款项的结果, 与 M 模式的 ENCLAVE_MODULE_STATUS_* 一致. */
#define TEE_MODULE_STATUS_NONE	 0 /* 尚无结果 */
#define TEE_MODULE_STATUS_OK	 1 /* 映像已写入接收缓冲区 */
#define TEE_MODULE_STATUS_NOENT	 2 /* 模块不存在 (宿主报告字节数为 0) */
#define TEE_MODULE_STATUS_WRFAIL 3 /* 交付的字节数与报告的不符, 或写入失败 */

/* 模块编号取该值表示无挂起请求. */
#define TEE_MODULE_ID_NONE 0xFFFFFFFFULL

/* ENTER 与模块三个命令 (MODULE_REQ / MODULE_SIZE / MODULE_IMG) 共用的参数结构体.
 * enclave_id 与 token 为四者共有的入参: token 是创建该飞地时登记的管理令牌,
 * M 模式在锁内比对后才载入飞地; 各命令的专属字段在匿名 union 中按命令取用.
 * 505 号调用交付的字节写入该飞地的 S 模式运行时经 509 号调用登记的接收缓冲区,
 * 其物理地址按该运行时自己的页表查询得到, 宿主不指定. */
struct tee_enclave_args {
	uint64_t enclave_id;
	uint64_t token;
	union {
		/* ENTER: 载荷指针与长度, 以及交给载荷的 argv. */
		struct {
			uint64_t payload_ptr;
			uint64_t payload_size;
			uint64_t argc;
			uint64_t argv_ptr;
		} enter;
		/* MODULE_REQ: 模块编号与请求类别均为出参. */
		struct {
			uint64_t module_id;
			uint64_t kind;
		} req;
		/* MODULE_SIZE: size 为入参, 取 0 表示宿主侧不存在该模块; status 为出参. */
		struct {
			uint64_t size;
			uint64_t status;
		} size;
		/* MODULE_IMG: buf_ptr 与 size 为入参 (用户缓冲区), 其余为出参.
		 * 505 号调用还以 a2 回传实际写入的字节数, 而内核的 sbi_ecall 只取回
		 * a0 与 a1, 故该值不回传用户态; 成功时它等于入参 size, 飞地侧另经
		 * 508 号调用读取它. */
		struct {
			uint64_t buf_ptr;
			uint64_t size;
			uint64_t status;
			uint64_t pa;
		} image;
	};
};

#endif /* TEE_ENCLAVE_H */
