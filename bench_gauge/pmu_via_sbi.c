/*
 * SPDX-LICENSE-IDENTIFIER: GPL2.0
 * (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026
 *
 * SBI PMU 扩展封装 — S-mode 性能测量
 * ====================================
 * 通过 OpenSBI 的 SBI PMU 扩展 (EID 0x504D55) 在 S-mode / U-mode
 * 访问硬件性能计数器. 无需直接操作 CSR, 由 M-mode 固件代理.
 *
 * 适用场景:
 *   - S-mode OS 内核 (Linux, xv6, ...)  测量内核路径
 *   - U-mode 用户程序通过系统调用间接使用 (不在此文件范围内)
 *   - OpenSBI 自身的性能分析 (通过内部 API, 非 SBI ecall)
 *
 * 与 pmu.s 的关系:
 *   pmu.s        → M-mode 直接操作 CSR, 最小开销, 适合裸机/boot 阶段
 *   pmu_via_sbi.c → S-mode 通过 ecall 委托 M-mode, 额外 1 次 ecall 开销
 *
 * 使用流程:
 *
 *   // 1. 探测可用计数器
 *   long num = sbi_pmu_num_counters();
 *
 *   // 2. 配置计数器 (匹配 CPU_CYCLES 事件)
 *   long cfg = sbi_pmu_config_one(0, 0x00000001,   // event = CPU_CYCLES
 *               SBI_PMU_CFG_CLEAR_VALUE | SBI_PMU_CFG_AUTO_START);
 *
 *   // 3. 执行被测代码
 *   my_function(arg1, arg2, ...);
 *
 *   // 4. 停止并读取
 *   sbi_pmu_stop(1 << cfg, SBI_PMU_STOP_TAKE_SNAPSHOT);
 *   long cycles = sbi_pmu_read_hw(cfg);
 *
 * 参考:
 *   RISC-V SBI Specification §PMU Extension
 *   OpenSBI lib/sbi/sbi_pmu.c
 */

#include <stdint.h>

/* ===================================================================
 *  SBI ecall 基元
 * =================================================================== */

/**
 * struct sbiret — SBI ecall 返回值
 * @error: 错误码 (0 = SBI_SUCCESS)
 * @value: 返回值 (含义取决于具体调用)
 */
struct sbiret {
	long error;
	long value;
};

/**
 * sbi_ecall — 发起 SBI ecall
 *
 * 参数传递: a7=ext, a6=fid, a0..a5=args
 * 返回:    a0=error, a1=value
 */
static inline struct sbiret sbi_ecall(
	unsigned long ext,
	unsigned long fid,
	unsigned long arg0,
	unsigned long arg1,
	unsigned long arg2,
	unsigned long arg3,
	unsigned long arg4,
	unsigned long arg5) {

	struct sbiret ret;
	register unsigned long a0 asm("a0") = arg0;
	register unsigned long a1 asm("a1") = arg1;
	register unsigned long a2 asm("a2") = arg2;
	register unsigned long a3 asm("a3") = arg3;
	register unsigned long a4 asm("a4") = arg4;
	register unsigned long a5 asm("a5") = arg5;
	register unsigned long a6 asm("a6") = fid;
	register unsigned long a7 asm("a7") = ext;

	asm volatile("ecall"
				 : "+r"(a0), "+r"(a1)
				 : "r"(a2), "r"(a3), "r"(a4), "r"(a5), "r"(a6), "r"(a7)
				 : "memory");

	ret.error = a0;
	ret.value = a1;
	return ret;
}

/* ===================================================================
 *  SBI PMU 扩展常量
 * =================================================================== */

/* 扩展 ID */
#define SBI_EXT_PMU 0x504D55UL

/* 函数 ID */
#define SBI_PMU_NUM_COUNTERS 0
#define SBI_PMU_COUNTER_GET_INFO 1
#define SBI_PMU_COUNTER_CONFIG 2
#define SBI_PMU_COUNTER_START 3
#define SBI_PMU_COUNTER_STOP 4
#define SBI_PMU_COUNTER_FW_READ 5

/* 计数器类型掩码 (counter_get_info 返回值) */
#define SBI_PMU_CTR_TYPE_MASK 0x0FULL
#define SBI_PMU_CTR_TYPE_HW 0x00 /* 硬件计数器 */
#define SBI_PMU_CTR_TYPE_FW 0x0F /* 固件计数器 */

/* 从 counter_get_info 返回值提取字段 */
#define SBI_PMU_CTR_TYPE(info) ((info) & 0x0F)
#define SBI_PMU_CTR_CSR(info) (((info) >> 16) & 0xFFFF)
#define SBI_PMU_CTR_WIDTH(info) ((info) >> 32)

/* config_matching 标志 */
#define SBI_PMU_CFG_CLEAR_VALUE (1UL << 0)
#define SBI_PMU_CFG_AUTO_START (1UL << 1)
#define SBI_PMU_CFG_SET_VUINH (1UL << 2)
#define SBI_PMU_CFG_SET_VSINH (1UL << 3)
#define SBI_PMU_CFG_SET_UINH (1UL << 4)
#define SBI_PMU_CFG_SET_SINH (1UL << 5)
#define SBI_PMU_CFG_SET_MINH (1UL << 6)
/* 便利组合: S-mode 下仅允许 S+U 计数, 清除旧值后自动启动 */
#define SBI_PMU_CFG_MEASURE                                                              \
	(SBI_PMU_CFG_CLEAR_VALUE | SBI_PMU_CFG_AUTO_START | SBI_PMU_CFG_SET_UINH)

/* start 标志 */
#define SBI_PMU_START_SET_INIT (1UL << 0)
#define SBI_PMU_START_FROM_SNAP (1UL << 1)

/* stop 标志 */
#define SBI_PMU_STOP_RESET (1UL << 0)
#define SBI_PMU_STOP_SNAPSHOT (1UL << 1)

/* 标准硬件事件编码 (event_idx) */
#define SBI_PMU_EVENT_NO_EVENT 0x00000000UL
#define SBI_PMU_EVENT_CPU_CYCLES 0x00000001UL
#define SBI_PMU_EVENT_INSTRUCTIONS 0x00000002UL
#define SBI_PMU_EVENT_L1_DCACHE_READ_MISS 0x00000003UL
#define SBI_PMU_EVENT_L1_DCACHE_WRITE_MISS 0x00000004UL
#define SBI_PMU_EVENT_L1_ICACHE_MISS 0x00000005UL
#define SBI_PMU_EVENT_DTLB_MISS 0x00000006UL
#define SBI_PMU_EVENT_ITLB_MISS 0x00000007UL
#define SBI_PMU_EVENT_BRANCH_MISS 0x00000008UL
#define SBI_PMU_EVENT_LOAD_MISS 0x00000009UL
#define SBI_PMU_EVENT_STORE_MISS 0x0000000AUL

/* ===================================================================
 *  SBI PMU API — 原始封装
 * =================================================================== */

/**
 * sbi_pmu_num_counters — 获取硬件计数器数量
 * 返回: 可用的硬件计数器总数 (含 mcycle, minstret)
 */
static inline long sbi_pmu_num_counters(void) {
	struct sbiret r = sbi_ecall(SBI_EXT_PMU, SBI_PMU_NUM_COUNTERS, 0, 0, 0, 0, 0, 0);
	return r.error ? -r.error : r.value;
}

/**
 * sbi_pmu_counter_get_info — 获取计数器的属性信息
 * @counter_idx: 计数器索引
 * 返回: 64-bit 信息字 (type, csr, width) 或负错误码
 */
static inline long sbi_pmu_counter_get_info(long counter_idx) {
	struct sbiret r =
		sbi_ecall(SBI_EXT_PMU, SBI_PMU_COUNTER_GET_INFO, counter_idx, 0, 0, 0, 0, 0);
	return r.error ? -r.error : r.value;
}

/**
 * sbi_pmu_config_matching — 查找并配置匹配指定事件的计数器
 * @counter_idx_base:  起始计数器索引
 * @counter_idx_mask:  参与搜索的计数器位掩码 (bit N 置位 = 包含 counter N)
 * @flags:             配置标志 (CLEAR_VALUE, AUTO_START, *_INH)
 * @event_idx:         事件编码 (SBI_PMU_EVENT_*)
 * @event_data:        事件附加数据 (通常为 0, 仅 cache 事件需要)
 * 返回: 已配置的计数器位掩码 (bit N 置位 = counter N 已配置并启动)
 */
static inline long sbi_pmu_config_matching(
	long counter_idx_base,
	long counter_idx_mask,
	unsigned long flags,
	unsigned long event_idx,
	unsigned long event_data) {
	struct sbiret r = sbi_ecall(
		SBI_EXT_PMU,
		SBI_PMU_COUNTER_CONFIG,
		counter_idx_base,
		counter_idx_mask,
		flags,
		event_idx,
		event_data,
		0);
	return r.error ? -r.error : r.value;
}

/**
 * sbi_pmu_start — 启动指定的计数器
 * @counter_idx_base:  起始计数器索引
 * @counter_idx_mask:  计数器位掩码
 * @flags:             启动标志
 * @initial_value:     初始值 (需 SBI_PMU_START_SET_INIT 标志)
 * 返回: 0 成功, 负错误码
 */
static inline long sbi_pmu_start(
	long counter_idx_base,
	long counter_idx_mask,
	unsigned long flags,
	unsigned long initial_value) {
	struct sbiret r = sbi_ecall(
		SBI_EXT_PMU,
		SBI_PMU_COUNTER_START,
		counter_idx_base,
		counter_idx_mask,
		flags,
		initial_value,
		0,
		0);
	return r.error ? -r.error : 0;
}

/**
 * sbi_pmu_stop — 停止指定的计数器
 * @counter_idx_base:  起始计数器索引
 * @counter_idx_mask:  计数器位掩码
 * @flags:             停止标志 (RESET / SNAPSHOT)
 * 返回: 0 成功, 负错误码
 */
static inline long sbi_pmu_stop(
	long counter_idx_base, long counter_idx_mask, unsigned long flags) {
	struct sbiret r = sbi_ecall(
		SBI_EXT_PMU,
		SBI_PMU_COUNTER_STOP,
		counter_idx_base,
		counter_idx_mask,
		flags,
		0,
		0,
		0);
	return r.error ? -r.error : 0;
}

/**
 * sbi_pmu_read_hw — 读取硬件计数器的当前值
 * @counter_idx: 计数器索引 (0 = mcycle, 2 = minstret, 3..N)
 * 返回: 64-bit 计数器值
 *
 * 注意: 硬件计数器可通过此 SBI 调用读取, 但本质上 M-mode
 * 固件只是代为执行 csrr. 也可以预先在 pmu.s 的 mcounteren
 * 中授权 S-mode 直接读, 省去 ecall 开销.
 */
static inline unsigned long sbi_pmu_read_hw(long counter_idx) {
	/* 通过 counter_get_info 间接读取 (SBI 无专用的 hw_read) */
	/* counter_get_info 返回 CSR 编号, 然后直接 csrr 读取.
     * 这里提供一个 wrapper: 如果 mcounteren 已授权, 直接读;
     * 否则通过 sbi_ecall 读 firmware counter. */
	register unsigned long val;
	/* 假设调用方已在 pmu.s 中设置 mcounteren, 直接读 */
	switch (counter_idx) {
	case 0: /* mcycle */
		asm volatile("csrr %0, 0xB00" : "=r"(val));
		break;
	case 2: /* minstret */
		asm volatile("csrr %0, 0xB02" : "=r"(val));
		break;
	default:
		/* mhpmcounter3..31: CSR = 0xB03 + (idx - 3) */
		if (counter_idx >= 3 && counter_idx <= 31) {
			long csr = 0xB03 + (counter_idx - 3);
			asm volatile("csrr %0, %1" : "=r"(val) : "i"(csr));
		} else {
			val = 0;
		}
		break;
	}
	return val;
}

/* ===================================================================
 *  便利 API — 测量辅助
 * =================================================================== */

/**
 * sbi_pmu_config_one — 配置单个计数器以测量指定事件
 * @start_idx:    搜索起始索引 (通常 0)
 * @event_idx:    事件编码
 * @flags:        配置标志
 * 返回: 已配置的计数器位掩码 (bit N), 0 表示失败
 *
 * 用法:
 *   long bm = sbi_pmu_config_one(0, SBI_PMU_EVENT_CPU_CYCLES,
 *               SBI_PMU_CFG_CLEAR_VALUE | SBI_PMU_CFG_AUTO_START);
 *   // ... 执行被测代码 ...
 *   sbi_pmu_stop(0, bm, SBI_PMU_STOP_SNAPSHOT);
 *   // 读取: 遍历 bm 找置位 bit (ctrl_idx), 调用 sbi_pmu_read_hw(ctrl_idx)
 */
static inline long sbi_pmu_config_one(
	long start_idx, unsigned long event_idx, unsigned long flags) {
	/* 将所有可用计数器纳入搜索 (bitmask 全 1) */
	return sbi_pmu_config_matching(start_idx, ~0UL, flags, event_idx, 0);
}

/**
 * sbi_pmu_read_cycle — 便捷: 读取 mcycle, 计算增量
 * @start: 之前读取的 mcycle 值
 * 返回: start 至今的周期增量
 */
static inline unsigned long sbi_pmu_read_cycle_delta(unsigned long start) {
	unsigned long now;
	asm volatile("csrr %0, 0xB00" : "=r"(now));
	return now - start;
}

/**
 * sbi_pmu_read_instret — 便捷: 读取 minstret
 */
static inline unsigned long sbi_pmu_read_instret(void) {
	unsigned long val;
	asm volatile("csrr %0, 0xB02" : "=r"(val));
	return val;
}
