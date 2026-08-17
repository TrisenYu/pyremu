/*
 * SPDX-LICENSE-IDENTIFIER: GPL2.0
 * (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026
 *
 * PMU C 调用接口
 * ==============
 * 声明 pmu.s 中的 M-mode 底层函数, 供 C 代码直接调用.
 * 编译器自动处理 RISC-V 调用约定, 调用方无需手工介入.
 *
 * 典型用法 (M-mode C 代码):
 *
 *   #include "pmu.h"
 *
 *   void demo(void) {
 *       pmu_init();
 *
 *       struct pmu_result r = PMU_MEASURE(compute_hash(data, len));
 *       printf("%llu cycles, %llu instrs\n", r.delta_cycle, r.delta_instr);
 *   }
 */

#ifndef BENCH_GAUGE_PMU_H
#define BENCH_GAUGE_PMU_H

#include <stdint.h>

/* ---- 计数器索引 ---- */
#define PMU_MCYCLE 0
#define PMU_MINSTRET 2
#define PMU_HPMCOUNTER3 3
#define PMU_HPMCOUNTER4 4
#define PMU_HPMCOUNTER5 5
#define PMU_HPMCOUNTER6 6
#define PMU_HPMCOUNTER7 7

/* ---- 汇编函数声明 (定义于 pmu.s) ---- */

extern void m_mode_pmu_init(void);
extern void m_mode_pmu_start(void);
extern void m_mode_pmu_stop(void);
extern uint64_t m_mode_pmu_read(int idx);

/* C 命名风格的便捷别名 */
static inline void pmu_init(void) {
	m_mode_pmu_init();
}
static inline void pmu_start(void) {
	m_mode_pmu_start();
}
static inline void pmu_stop(void) {
	m_mode_pmu_stop();
}
static inline uint64_t pmu_read(int idx) {
	return m_mode_pmu_read(idx);
}

/* ---- 测量结果结构体 ---- */

struct pmu_result {
	uint64_t st_cycle;	  /* 被测代码执行前的 mcycle 值 */
	uint64_t ed_cycle;	  /* 被测代码执行后的 mcycle 值 */
	uint64_t st_instr;	  /* 被测代码执行前的 minstret 值 */
	uint64_t ed_instr;	  /* 被测代码执行后的 minstret 值 */
	uint64_t delta_cycle; /* ed_cycle - st_cycle */
	uint64_t delta_instr; /* ed_instr - st_instr */
};

/* ---- 测量宏 ---- */

/**
 * PMU_MEASURE(stmt) — 测量任意 C 语句的执行开销
 *
 * @stmt  被测代码, 可以是函数调用、复合语句、甚至空语句 (测量开销本身)
 *
 * 用法:
 *   // 带参数的函数调用
 *   struct pmu_result r = PMU_MEASURE(sha256(data_buf, data_len));
 *
 *   // 复合语句 (多条代码一起测量)
 *   struct pmu_result r = PMU_MEASURE({
 *       write_reg(0x1000, 0x5A);
 *       write_reg(0x1004, 0xA5);
 *       asm volatile ("fence w, o" ::: "memory");
 *   });
 *
 *   // 测量空调用开销 (baseline)
 *   struct pmu_result r = PMU_MEASURE({});
 */
#define PMU_MEASURE(stmt)                                                                \
	({                                                                                   \
		struct pmu_result __ret = {0};                                                   \
		pmu_start();                                                                     \
		__ret.st_cycle = pmu_read(PMU_MCYCLE);                                           \
		__ret.st_instr = pmu_read(PMU_MINSTRET);                                         \
                                                                                         \
		stmt;                                                                            \
                                                                                         \
		__ret.ed_cycle = pmu_read(PMU_MCYCLE);                                           \
		__ret.ed_instr = pmu_read(PMU_MINSTRET);                                         \
		pmu_stop();                                                                      \
		__ret.delta_cycle = __ret.ed_cycle - __ret.st_cycle;                             \
		__ret.delta_instr = __ret.ed_instr - __ret.st_instr;                             \
		__ret;                                                                           \
	})

#endif /* BENCH_GAUGE_PMU_H */
