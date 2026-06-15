# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026
#
# RISC-V M-mode PMU (Performance Monitoring Unit) 底层原语
# ========================================================
# 提供最小粒度的计数器启/停/读操作, 不参与被测函数的参数传递或调用约定.
# 调用方自行在 start / stop 之间执行被测逻辑, 然后读取计数器求差.
#
# 权限模型:
#   - M-mode:   可访问全部 PMU CSR. 可通过 mcounteren 授权 S-mode 读取.
#   - S-mode:   默认无法访问 mcycle/minstret/mhpmevent (需 mcounteren 授权),
#               更不可写 mcountinhibit/mhpmevent. **S-mode 应通过 SBI PMU 扩展
#               (见 pmu_via_sbi.c) 获取计数器访问**, 或由 M-mode 固件代为操作.
#   - U-mode:   默认完全无法访问 PMU CSR. 需 scounteren (S→U) 授权.
#
# 典型用法 (M-mode, 如 OpenSBI):
#
#   call m_mode_pmu_init           # 一次性初始化: 配事件, 清零, 开 S-mode 读权限
#
#   call m_mode_pmu_start          # 启动计数器
#   <被测代码: 函数调用, 一段指令序列, 甚至嵌套 SBI 调用>
#   call m_mode_pmu_stop           # 停止计数器
#
#   li   a0, 0                     # 0 = mcycle
#   call m_mode_pmu_read           # a0 = end_cycle, 自行求差
#   li   a0, 2
#   call m_mode_pmu_read           # a0 = end_instret
#
# 注意:
#   - 测量期间不替换 mtvec, 被测函数若触发 trap 则行为由已有 handler 决定.
#   - 若仅测周期/指令数, 可不配 mhpmevent; 直接用 mcycle/minstret 即可.
# ============================================================

.section .text

# ------------------------------------------------------------
#  PMU CSR 地址 (RISC-V Privileged Spec)
# ------------------------------------------------------------
.equ CSR_MCYCLE,          0xB00       # 周期计数器 (RO, 不受 inhibit)
.equ CSR_MINSTRET,        0xB02       # 退休指令计数器
.equ CSR_MCYCLEH,         0xB80       # RV32 高 32 位
.equ CSR_MINSTRETH,       0xB82       # RV32 高 32 位

# 可编程计数器 (最多 29 个: 3..31, 取决于硬件实现)
.equ CSR_MHPMCOUNTER3,    0xB03
.equ CSR_MHPMCOUNTER4,    0xB04
.equ CSR_MHPMCOUNTER5,    0xB05
.equ CSR_MHPMCOUNTER6,    0xB06
.equ CSR_MHPMCOUNTER7,    0xB07
# ... 扩展到 mhpmcounter31 = 0xB1F

# 事件选择器: 每个可编程计数器对应一个 mhpmeventN (N=3..31)
.equ CSR_MHPMEVENT3,      0x323
.equ CSR_MHPMEVENT4,      0x324
.equ CSR_MHPMEVENT5,      0x325
.equ CSR_MHPMEVENT6,      0x326
.equ CSR_MHPMEVENT7,      0x327
# ... 扩展到 mhpmevent31 = 0x33F

# 计数器控制
.equ CSR_MCOUNTINHIBIT,   0x320       # 每 bit 抑制对应计数器
.equ CSR_MCOUNTEREN,      0x306       # M-mode → S-mode 计数器读权限

# ------------------------------------------------------------
#  标准硬件事件编码 (写入 mhpmeventN)
#  mhpmevent 寄存器结构 (RV64):
#    bits 0-19:  event selector
#    高位:       event class / config (平台特定)
#  若平台不支持某事件, 计数器保持 0.
# ------------------------------------------------------------
.equ EVENT_CYCLE,               0x01  # CPU 周期
.equ EVENT_INSTRET,             0x02  # 退休指令数
.equ EVENT_L1_DCACHE_READ_MISS, 0x03  # L1 数据缓存读缺失
.equ EVENT_L1_DCACHE_WRITE_MISS,0x04  # L1 数据缓存写缺失
.equ EVENT_L1_ICACHE_MISS,      0x05  # L1 指令缓存缺失
.equ EVENT_DTLB_MISS,           0x06  # 数据 TLB 缺失
.equ EVENT_ITLB_MISS,           0x07  # 指令 TLB 缺失
.equ EVENT_BRANCH_MISS,         0x08  # 分支预测错误
.equ EVENT_LOAD_MISS,           0x09  # 加载缓存缺失
.equ EVENT_STORE_MISS,          0x0A  # 存储缓存缺失


# ============================================================
#  m_mode_pmu_init — 初始化 PMU (M-mode, 一次性调用)
#
#  破坏: t0, t1
#  副作用:
#    - 停止并清零 mcycle, minstret, hpmcounter3..7
#    - 配置 mhpmevent3..7 映射常用事件
#    - 配置 mcounteren, 使 S-mode 可读上述计数器
# ============================================================
.global m_mode_pmu_init
m_mode_pmu_init:
    # -- 停止全部计数器 --
    li    t0, -1
    csrw  CSR_MCOUNTINHIBIT, t0

    # -- 固定计数器清零 --
    csrw  CSR_MCYCLE, zero
    csrw  CSR_MINSTRET, zero

    # -- 可编程计数器清零 (3..7) --
    csrw  CSR_MHPMCOUNTER3, zero
    csrw  CSR_MHPMCOUNTER4, zero
    csrw  CSR_MHPMCOUNTER5, zero
    csrw  CSR_MHPMCOUNTER6, zero
    csrw  CSR_MHPMCOUNTER7, zero

    # -- 事件选择 (平台特定, 不支持则计数器保持 0) --
    li    t0, EVENT_L1_DCACHE_READ_MISS
    csrw  CSR_MHPMEVENT3,  t0
    li    t0, EVENT_L1_DCACHE_WRITE_MISS
    csrw  CSR_MHPMEVENT4,  t0
    li    t0, EVENT_BRANCH_MISS
    csrw  CSR_MHPMEVENT5,  t0
    li    t0, EVENT_INSTRET
    csrw  CSR_MHPMEVENT6,  t0
    li    t0, EVENT_DTLB_MISS
    csrw  CSR_MHPMEVENT7,  t0

    # -- S-mode 可读权限: bit0=cycle,  bit2=instret,  bitN=hpmcounterN --
    li    t0, (1 << 0) | (1 << 2) | (1 << 3) \
            | (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7)
    csrw  CSR_MCOUNTEREN, t0

    ret


# ============================================================
#  m_mode_pmu_start — 启动全部计数器
# ============================================================
.global m_mode_pmu_start
m_mode_pmu_start:
    li    t0, ~((1 << 0) | (1 << 2) | (1 << 3) \
              | (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7))
    csrrc x0, CSR_MCOUNTINHIBIT, t0
    ret


# ============================================================
#  m_mode_pmu_stop — 停止全部计数器
# ============================================================
.global m_mode_pmu_stop
m_mode_pmu_stop:
    li    t0, (1 << 0) | (1 << 2) | (1 << 3) \
            | (1 << 4) | (1 << 5) | (1 << 6) | (1 << 7)
    csrrs x0, CSR_MCOUNTINHIBIT, t0
    ret


# ============================================================
#  m_mode_pmu_read — 按索引读取计数器值
#
#   输入: a0 = 计数器索引
#             0 → mcycle
#             2 → minstret
#             3..7 → mhpmcounter3..7
#   返回: a0 = 64-bit 计数值
# ============================================================
.global m_mode_pmu_read
m_mode_pmu_read:
    li    t0, 0
    beq   a0, t0, 1f
    li    t0, 2
    beq   a0, t0, 2f
    li    t0, 3
    beq   a0, t0, 3f
    li    t0, 4
    beq   a0, t0, 4f
    li    t0, 5
    beq   a0, t0, 5f
    li    t0, 6
    beq   a0, t0, 6f
    li    t0, 7
    beq   a0, t0, 7f
    li    a0, 0
    ret
1:
    csrr a0, CSR_MCYCLE
    ret
2:
    csrr a0, CSR_MINSTRET
    ret
3:
    csrr a0, CSR_MHPMCOUNTER3
    ret
4:
    csrr a0, CSR_MHPMCOUNTER4
    ret
5:
    csrr a0, CSR_MHPMCOUNTER5
    ret
6:
    csrr a0, CSR_MHPMCOUNTER6
    ret
7:
    csrr a0, CSR_MHPMCOUNTER7
    ret
