//! 平台可调常量由 Makefile 从 config.mk 生成 configs_gen.rs 注入。

#![allow(dead_code)]

use core::num::NonZeroU64;

// ---- 由 Makefile 从 config.mk 生成 ----
include!("configs_gen.rs");

// ---------------------------------------------------------------
//  配置取值的检查
//
// 以下取值由 config.mk 注入, 不合法时编译失败, 不进入运行期。
// ---------------------------------------------------------------

/// 每秒的时钟源计数值。取值为 0 时任何计数值都换算不出秒, 故由 NonZeroU64 在编译期
/// 排除该取值。
pub const TIMER_FREQ_NONZERO: NonZeroU64 = match NonZeroU64::new(TIMER_FREQ) {
	Some(v) => v,
	None => panic!("TIMER_FREQ in config.mk must be greater than zero"),
};

/// 一次调度 tick 所用的时钟源计数: 调度定时器按该间隔请求中断。
pub const TICKS_PER_SCHED_TICK: u64 = TIMER_FREQ / SCHED_FREQ;

/// 调度频率取 0 时一次 tick 的计数长度没有定义。
const _: () = assert!(
	SCHED_FREQ > 0,
	"SCHED_FREQ in config.mk must be greater than zero"
);

/// 时钟源频率低于 1000 Hz 时 1 ms 内不足一个计数, 时间片无法按毫秒计。
const _: () = assert!(
	TIMER_FREQ >= 1_000,
	"TIMER_FREQ in config.mk must be at least 1000"
);

/// 计数长度须整除; 不整除时实际的中断频率偏离 SCHED_FREQ。
const _: () = assert!(
	TIMER_FREQ % SCHED_FREQ == 0,
	"TIMER_FREQ must be a multiple of SCHED_FREQ"
);

/// 调度频率的区间: 167 ≤ SCHED_FREQ < TIMER_FREQ。
///
/// 下界取 Linux CFS 目标调度周期对应的频率: 该周期默认为 6 ms, 可运行任务数不超过 8 时
/// 任务按该周期轮转; 周期 6 ms 对应的频率约为 166.7 Hz, 向上取整为 167 Hz. tick 慢于
/// 该频率时, 时间片的消耗要等到下一个 tick 才能得到检查。
/// 上界取时钟源频率: 一次 tick 至少包含两个时钟源计数, 否则截止时刻在中断返回时
/// 已经到期, 定时器中断连续触发, 被中断的代码得不到执行。
const _: () = assert!(
	167 <= SCHED_FREQ && SCHED_FREQ < TIMER_FREQ,
	"SCHED_FREQ in config.mk must be at least 167 and less than TIMER_FREQ"
);

// ---------------------------------------------------------------
//  PTE 标志位
// ---------------------------------------------------------------

pub const PTE_V: u64 = 1 << 0;
pub const PTE_R: u64 = 1 << 1;
pub const PTE_W: u64 = 1 << 2;
pub const PTE_X: u64 = 1 << 3;
pub const PTE_U: u64 = 1 << 4;
pub const PTE_G: u64 = 1 << 5;
pub const PTE_A: u64 = 1 << 6;
pub const PTE_D: u64 = 1 << 7;

/// RSW 的位 0, 即页表项位 8。本运行时用它标记 MAP_SHARED 匿名映射的叶子项:
/// fork 复制地址空间时带该位的用户叶子共用同一物理页, 不带该位的按私有页复制。
pub const PTE_RSW_SHARED: u64 = 1 << 8;

/// RSW 两位 (位 8 与位 9) 的掩码。改写叶子项的标志位时以它保留 RSW,
/// 使 mprotect 与拆分超级页都不改变映射的共享属性。
pub const PTE_RSW_MASK: u64 = 0b11 << 8;

// ---------------------------------------------------------------
//  Sv39 层级
// ---------------------------------------------------------------

pub const LEVEL_GIGA: u8 = 0; // 1 GiB 超级页
pub const LEVEL_MEGA: u8 = 1; // 2 MiB 超级页
pub const LEVEL_PAGE: u8 = 2; // 4 KiB 普通页

pub const SV39_VPN_LEN: u8 = 9;

// ---------------------------------------------------------------
//  陷态原因
// ---------------------------------------------------------------

/// U 模式的 ecall: 可信应用发起系统调用.
pub const SCAUSE_ECALL_FROM_UMODE: u64 = 0x8;
/// 取指访问故障: 取指的物理地址未被任何 PMP 条目覆盖.
pub const SCAUSE_INSTR_ACCESS: u64 = 0x1;
/// 读访问故障.
pub const SCAUSE_LOAD_ACCESS: u64 = 0x5;
/// 写访问故障.
pub const SCAUSE_STORE_ACCESS: u64 = 0x7;
/// 取指页错误.
pub const SCAUSE_INSTR_PAGE_FAULT: u64 = 0xc;
/// 读页错误.
pub const SCAUSE_LOAD_PAGE_FAULT: u64 = 0xd;
/// 写页错误.
pub const SCAUSE_STORE_PAGE_FAULT: u64 = 0xf;

/// 该陷态原因是否由 PMP 条目未覆盖访问的物理地址引起.
///
/// 取指与读与写三类访问故障都属此列, 由 S 模式统一转发 M 模式按需加载该地址所属的区域;
/// M 模式加载成功才重试原访问, 加载无法使该次访问成立时按故障的来源处置
/// 页错误的取值为 [`SCAUSE_INSTR_PAGE_FAULT`] 与 [`SCAUSE_LOAD_PAGE_FAULT`] 与
/// [`SCAUSE_STORE_PAGE_FAULT`], 由页表项的权限位判定给出, 与 PMP 条目无关, 不在此列.
pub const fn is_unmatched_access_fault(scause: u64) -> bool {
	matches!(
		scause,
		SCAUSE_INSTR_ACCESS | SCAUSE_LOAD_ACCESS | SCAUSE_STORE_ACCESS
	)
}

// ---------------------------------------------------------------
//  飞地扩展 ID 与函数号
// ---------------------------------------------------------------
pub const ENCLAVE_EXT_ID: u64 = 0x2022_1222;

pub const ENCLAVE_CALL_SUSPEND: u64 = 404;
pub const ENCLAVE_CALL_SHUTDOWN: u64 = 403;
/// 申请内存: a1 = 分区数, a2 = 类别;
/// 返回 (已分配分区数, 物理起始地址).
pub const ENCLAVE_CALL_MEM_ALLOC: u64 = 500;
/// 设置一段内存的类别: a0 = 分区起始物理地址, a1 = 分区数, a2 = 新类别;
/// 返回 0 表示成功, 其余取值为 SBI 错误码.
pub const ENCLAVE_CALL_SET_MEM_CLASS: u64 = 510;
pub const ENCLAVE_CALL_GET_ID: u64 = 407;
pub const ENCLAVE_CALL_GET_HARTID: u64 = 408;
pub const ENCLAVE_CALL_GET_AVAILABLE_MEM: u64 = 409;
/// 转发访问故障: a0 = 故障虚拟地址, a1 = 故障的 scause.
/// 返回 0 表示 M 模式已加载该地址所属的区域, 重试原访问即可;
/// 返回 M 模式 enclave_types.h 的 SBI_ERR_ENCLAVE_ACCESS_UNRECOVERABLE, 取值为 -17,
/// 表示该次访问不可恢复, 重试不会成功.
pub const ENCLAVE_CALL_UNMATCHED_ACC_FAULT: u64 = 506;
/// 取随机字节: a0 = 目标物理地址, a1 = 字节数; 返回实际写入的字节数.
pub const ENCLAVE_CALL_GET_RAND_NUM: u64 = 507;
pub const ENCLAVE_CALL_REQUEST_SHUTDOWN: u64 = 410;
pub const ENCLAVE_CALL_QUERY_REQUESTS: u64 = 411;

/// 取回本飞地模块请求的结果: a0 = 交付结果,
/// a1 = 宿主报告的映像字节数, a2 = 映像起始处的物理地址,
/// a3 = 宿主实际写入的字节数. 读取后 M 模式把交付结果复位为 ENCLAVE_MODULE_STATUS_NONE.
pub const ENCLAVE_CALL_MODULE_RESULT: u64 = 508;

/// 登记模块映像接收缓冲区: a0 = 缓冲区的起始虚拟地址, a1 = 缓冲区的字节数;
/// 返回 0 表示登记成功, 非 0 为 SBI 错误码.
pub const ENCLAVE_CALL_MODULE_IMG_RECV_BUF: u64 = 509;

/// 2 MiB 分区的类别: 该分区只承载数据访问.
/// 数值与 M 模式 enclave_types.h 的 enum mem_meta_attr 一一对应, 修改需两侧同步.
pub const ENCLAVE_MEM_ATTR_DATA: u64 = 0;

/// 2 MiB 分区的类别: 该分区会被本飞地作为指令取指.
pub const ENCLAVE_MEM_ATTR_INSTR: u64 = 1;

/// SUSPEND 让出原因: 自愿让出 (引导期交接 / 等待宿主服务), 必须交还宿主.
pub const ENCLAVE_SUSPEND_VOLUNTARY: u64 = 0;
/// SUSPEND 让出原因: 时间片耗尽, 由 M 模式调度器决定切换或就地续期.
/// 数值与 M 模式 enclave_types.h 的 ENCLAVE_SUSPEND_* 一一对应, 修改需两侧同步.
pub const ENCLAVE_SUSPEND_QUOTA: u64 = 1;
/// SUSPEND 让出原因: 请求宿主报告模块映像的字节数. 让出原因占 a0 的低 32 位,
/// 高 32 位为模块编号.
pub const ENCLAVE_SUSPEND_MODULE_SIZE: u64 = 2;
/// SUSPEND 让出原因: 请求宿主交付模块映像的字节. 让出原因占 a0 的低 32 位,
/// 高 32 位为模块编号.
pub const ENCLAVE_SUSPEND_MODULE_LOAD: u64 = 3;
/// SUSPEND 让出原因: 本飞地内已无可运行线程, 交还宿主运行. 飞地停驻在让出的 hart 上,
/// 网卡中断到达时由 M 模式把该 hart 交回, 飞地在此让出处恢复执行.
pub const ENCLAVE_SUSPEND_IDLE: u64 = 4;

/// 模块请求的交付结果, 由 508 号调用取回.
/// 数值与 M 模式 enclave_types.h 的 ENCLAVE_MODULE_STATUS_* 一一对应, 修改需两侧同步.
pub const ENCLAVE_MODULE_STATUS_NONE: u64 = 0;
pub const ENCLAVE_MODULE_STATUS_OK: u64 = 1;
/// 宿主侧不存在该模块, 即宿主经 504 号调用报告的字节数为 0.
pub const ENCLAVE_MODULE_STATUS_NOENT: u64 = 2;
/// 交付的字节数与报告的不符, 或写入接收缓冲区失败.
pub const ENCLAVE_MODULE_STATUS_WRFAIL: u64 = 3;

/// QUERY_REQUESTS 返回: host 已申请终止此飞地.
pub const ENCLAVE_REQ_SHUTDOWN: u64 = 1 << 0;
/// QUERY_REQUESTS 返回: 飞地侧网卡有事件待处理.
/// 数值与 M 模式 enclave_types.h 的 ENCLAVE_REQ_NET_EVENT 一致, 修改需两侧同步.
pub const ENCLAVE_REQ_NET_EVENT: u64 = 1 << 1;

// ---------------------------------------------------------------
//  标准 SBI
// ---------------------------------------------------------------

pub const SBI_LEGACY_PUTCHAR_EXT: u64 = 0x01;
pub const SBI_DBCN_EXT: u64 = 0x4442434E;
pub const SBI_DBCN_CONSOLE_WRITE: u64 = 0;
pub const SBI_TIMER_EXT: u64 = 0x5449_4D45;
pub const SBI_SET_TIMER_FUNC: u64 = 0x00;

// ---------------------------------------------------------------
//  VA 布局
// ---------------------------------------------------------------

// VA 布局的全部常量由 Makefile 从 config.mk 生成注入 configs_gen.rs, 在文件开头
// 经 include! 展开。扩展模块虚拟地址空间基址须高于管理器虚拟地址空间基址:
// ecall_aux::va_to_pa 按该顺序判定分支, 二者的次序颠倒会使模块地址被按管理器虚拟地址
// 空间的偏移换算。

// ---------------------------------------------------------------
//  大小常量
// ---------------------------------------------------------------

pub const PAGE_SIZE: u64 = 0x1000;
pub const PAGE_SHIFT: u64 = 12;

/// M-mode 内存分配 / PMP 保护的最小粒度 = 2 MiB。
pub const CHUNK_2M_SIZE: u64 = 0x20_0000;
pub const CHUNK_2M_SHIFT: u64 = 21;

/// 一个 2 MiB 块可切分成的 4 KiB 页数。
pub const CHUNK_2M_PAGES: u64 = CHUNK_2M_SIZE / PAGE_SIZE;

/// 单次向 M-mode 申请的分区块数。
///
/// M-mode 的 mem_alloc 只在所请求的 n 个分区全部可用时授予它们, 否则一个都不授予并
/// 返回 0 个分区, 故请求数即所需的连续空闲区间长度。取 1: 申请只需池中存在任一空闲
/// 分区, 各次授予的物理地址不需相邻, VA 一侧由调用方按 2 MiB 块逐个推进。取剩余
/// 跨度会使一次增长的成功与否取决于池中存在等长连续区间, 池一旦碎片化即整段被拒,
/// 而 512 MiB 与 1 GiB 与 2 GiB 的内存压力档位分别要求 256 与 512 与 1024 个连续分区。
pub const MEM_ALLOC_BLOCKS_PER_CALL: u64 = 1;

pub const UMODE_STACK_SIZE_TOTAL: u64 = 0x10_0000;

/// Sv39 用户空间的地址上界。VA 小于该值时 VPN[2] 落在根表低半区的索引 0..256, 即全部
/// 用户空间; 不小于该值的 VA 由运行时自身的线性窗口与管理器 VA 占用。
///
/// 用户态的 VA 由硬件按 VPN 的 9 位字段查表, 高于该上界的 VA 与低半区的某个 VA 共用同一
/// 组页表项, 故越过上界的映射会静默改写别的 VA 的页表项。内存类系统调用据此拒绝越界区间。
pub const UMODE_VA_LIMIT: u64 = 1 << 38;

// ---------------------------------------------------------------
//  ELF 常量
// ---------------------------------------------------------------

pub const ELF_MAGIC: [u8; 4] = [0x7F, b'E', b'L', b'F'];
pub const ELFCLASS64: u8 = 2;
pub const ET_EXEC: u16 = 2;
pub const EM_RISCV: u16 = 243;
pub const PT_LOAD: u32 = 1;
pub const PF_R: u32 = 4;
pub const PF_W: u32 = 2;
pub const PF_X: u32 = 1;

// ---------------------------------------------------------------
//  Attestation (secp256r1 ECDSA + SHA-256)
// ---------------------------------------------------------------

/// secp256r1 标量字节长度 (私钥 / 坐标)。
pub const ECC_BYTES: usize = 32;
/// 压缩公钥长度 = 0x02/0x03 前缀 + X 坐标。
pub const PUB_KEY_LEN: usize = ECC_BYTES + 1; // 33
/// ECDSA 签名长度 = r‖s。
pub const SIG_LEN: usize = ECC_BYTES * 2; // 64
/// SHA-256 摘要长度。
pub const SHA256_DIGEST: usize = 32;

// ATTEST_PUB_KEY: [u8; 33] 由 Makefile 从 config.mk 生成注入 configs_gen.rs。

// ---------------------------------------------------------------
//  测试, 仅在 host 端编译时可用
// ---------------------------------------------------------------
#[cfg(test)]
mod tests {
	use crate::constants::*;
	#[test]
	fn test_pte_v_flag_is_lsb() {
		assert_eq!(PTE_V, 1);
	}

	#[test]
	fn test_pte_flags_are_distinct() {
		let flags = [PTE_V, PTE_R, PTE_W, PTE_X, PTE_U, PTE_G, PTE_A, PTE_D];
		for i in 0..flags.len() {
			for j in (i + 1)..flags.len() {
				assert_ne!(
					flags[i], flags[j],
					"PTE flags at position {i} and {j} overlap"
				);
			}
		}
	}

	#[test]
	fn test_chunk_2m_is_sv39_mega_page() {
		// CHUNK_2M_SIZE 必须等于一个 Sv39 mega page
		assert_eq!(CHUNK_2M_SIZE, 1 << (PAGE_SHIFT + SV39_VPN_LEN as u64));
	}

	#[test]
	fn test_page_constants_consistent() {
		assert_eq!(PAGE_SIZE, 1 << PAGE_SHIFT);
		assert_eq!(CHUNK_2M_SIZE, 1 << CHUNK_2M_SHIFT);
	}

	/// 取指访问故障也属转发范围。
	///
	/// 修复前只转发读与写两类访问故障, 取指访问故障落到用户态不可恢复异常路径:
	/// 飞地的指令类分区被淘汰后再次取指, 该次取指按段错误终止该进程
	#[test]
	fn test_instr_access_fault_is_forwarded() {
		assert!(is_unmatched_access_fault(SCAUSE_INSTR_ACCESS));
		assert!(is_unmatched_access_fault(SCAUSE_LOAD_ACCESS));
		assert!(is_unmatched_access_fault(SCAUSE_STORE_ACCESS));
	}

	/// 三类页错误由页表权限位判定给出，与 PMP 条目无关。
	#[test]
	fn test_page_faults_are_not_forwarded() {
		assert!(!is_unmatched_access_fault(SCAUSE_INSTR_PAGE_FAULT));
		assert!(!is_unmatched_access_fault(SCAUSE_LOAD_PAGE_FAULT));
		assert!(!is_unmatched_access_fault(SCAUSE_STORE_PAGE_FAULT));
	}

	/// 地址错位、非法指令、断点
	#[test]
	fn test_neighbouring_causes_are_not_forwarded() {
		for cause in [
			0x0u64,
			0x2,
			0x3,
			0x4,
			0x6,
			SCAUSE_ECALL_FROM_UMODE,
			0x9,
			0xb,
			0xe,
		] {
			assert!(
				!is_unmatched_access_fault(cause),
				"cause 0x{cause:x} must not be forwarded"
			);
		}
	}

	/// 单次内存申请的块数不随所需跨度增长。
	///
	/// 修复前该处取剩余跨度, stress_ng 的 512 MiB 与 1 GiB 与 2 GiB 三档分别向
	/// M-mode 索要 256 与 512 与 1024 个连续分区, 池一旦碎片化即整档被拒。
	#[test]
	fn test_mem_alloc_request_is_independent_of_span() {
		assert_eq!((512u64 << 20) / CHUNK_2M_SIZE, 256);
		assert_eq!((1u64 << 30) / CHUNK_2M_SIZE, 512);
		assert_eq!((2u64 << 30) / CHUNK_2M_SIZE, 1024);

		assert_eq!(MEM_ALLOC_BLOCKS_PER_CALL, 1);
	}
}
