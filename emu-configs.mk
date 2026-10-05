# ===================================================================
# emu-configs.mk — 模拟器硬件特性配置 (供 configs.mk include)
# ===================================================================
# 控制模拟器运行时行为: 中断模式、缓存大小、诊断开关等。
# 可覆盖项 (?=) 支持 make 命令行 / 环境变量覆盖。

# ---- 保留内存区域 ----
# 供特定固件代码实现使用, 需与 custom-opensbi Kconfig (POOL_BASE/POOL_SIZE) 保持同步.
# DTB /reserved-memory no-map 节点据此生成, 确保 Linux 内核线性映射排除该区域,
# 避免内核分配器与固件在同一物理区间内产生访问冲突.
RESERVED_MEM_BASE    ?= 0x83000000
# 2.25 GiB. 须容纳 stress_ng 的 --vm-bytes 2G 档位: 该档位自身占用 2 GiB, 另需
# 载荷镜像、argv 与 M 模式按 2 MiB 分区取整所需的余量. 池尾为 0x113000000
# (~4.30 GiB), 故 ram 须不小于 6G, 以留出内核与根文件系统的空间.
RESERVED_MEM_SIZE    ?= 0x90000000

# ---- 固件映像保留区 ----
# 固件 (custom-opensbi) 载入 ram_base, 其映像尾部内嵌 .sittim 飞地运行时
# (链接地址 0x100000, 即物理地址 0x80100000). 自 ram_base 起直至内核载入地址
# FW_JUMP_ADDR (0x80200000) 的整个区间都必须保留: 否则内核分配器会把页面分到
# 固件映像上, 设备 DMA (如 virtio-blk) 会直接覆写 .sittim, 使 CREATE 复制进
# 飞地池的载荷变成垃圾数据, 飞地入口立即取指故障并被 M 模式静默挂起.
FW_RESERVED_MEM_BASE ?= 0x80000000
# 2 MiB
FW_RESERVED_MEM_SIZE ?= 0x200000

# ---- 模拟器编译期配置 (Python / Rust 共享) ----
# 通过 makefile 生成 pyremu/configs_gen.py, 替代各处硬编码与 os.environ.get.
# TLB 条目数
TLB_ENTRIES          ?= 256
# mtime/mcycle 随宿主机时间推进
CPU_FREQ_HZ          ?= 1000000000
# 调试器 run/continue 的时钟源超时 (秒), 0 = 禁用
DEFAULT_TIMEOUT      ?= 6000


# ---- 运行时调优
# WFI 空转单次睡眠上限 (秒), 降低宿主机 CPU 占用
WFI_MAX_SLEEP        ?= 0.2
# virtio 单次 _process_queue 最多处理描述符数, 拆小批避免长时间占用主线程
VIRTIO_PROCESS_BATCH ?= 16
# 段加载阈值 (字节): 不超过阈值的段走 L2 缓存, 更大段直写 RAM
FAST_LOAD_THRESHOLD  ?= 0x200000
# mtime 指令计数源的每指令纳秒数
NS_PER_INSTR         ?= 20

# WFI 全 idle 无定时器时固定时长轮询间隔，单位ms
WFI_WATCHDOG_MS      ?= 5


## ---- 模拟硬件特性
# 1 = AIA (IMSIC+APLIC); 0 = legacy PLIC
PYREMU_AIA           ?= 0
# 1 = H-extension (HS/VS modes + hgatp + virtual interrupts)
PYREMU_H_EXT         ?= 0

## ---- 中断控制器相关
# IMSIC M-file MMIO 基址
IMSIC_M_BASE         ?= 0x24000000
# IMSIC S-file MMIO 基址
IMSIC_S_BASE         ?= 0x28000000
# APLIC S 域 MMIO 基址
APLIC_S_BASE         ?= 0x0C000000
# APLIC M 域 MMIO 基址
APLIC_M_BASE         ?= 0x0C008000


# ---- 诊断用相关配置
PYREMU_DIAG_LOG      ?= /tmp/pyremu_diag.log
PYREMU_DIAG_VERBOSE  ?= 0
PYREMU_TRACE_SRET    ?= 1 # 启用sittim诊断

# 1 = 追踪 SRET 到 U-mode
PYREMU_TRACE_TRAPS   ?= 0
# 1 = 追踪全部 trap 投递
PYREMU_TRACE_PMP     ?= 0
# 1 = 追踪 PMP 匹配
PYREMU_AIA_DIAG_INTERVAL ?= 100
# AIA 诊断打印间隔 (批次), 0 = 禁用
PYREMU_NO_L2         ?= 0
# 1 = 禁用 L2 缓存
# 1 = 停用 TLB: 每次地址翻译都完整遍历 Sv39 页表, 不做映射缓存.
PYREMU_NO_TLB        ?= 0

export
