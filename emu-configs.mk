# ===================================================================
# emu-configs.mk — 模拟器硬件特性配置 (供 configs.mk include)
# ===================================================================
# 控制模拟器运行时行为: 中断模式、缓存大小、诊断开关等。
# 可覆盖项 (?=) 支持 make 命令行 / 环境变量覆盖。

# 二进制加速
# 0 = 纯 Python，1 = 启用宿主机二进制指令加速
PYREMU_NATIVE_SPEEDUP ?= 1

# ---- 保留内存区域 ----
# 供特定固件代码实现使用, 需与 custom-opensbi Kconfig (POOL_BASE/POOL_SIZE) 保持同步.
# DTB /reserved-memory no-map 节点据此生成, 确保 Linux 内核线性映射排除该区域,
# 避免内核分配器与固件在同一物理区间内产生访问冲突.
RESERVED_MEM_BASE    ?= 0x83000000
# 256 MiB
RESERVED_MEM_SIZE    ?= 0x10000000

# ---- 模拟器编译期配置 (Python / Rust 共享) ----
# 通过 makefile 生成 pyremu/configs_gen.py, 替代各处硬编码与 os.environ.get.
# TLB 条目数
TLB_ENTRIES          ?= 256

# mtime/mcycle 随宿主机时间推进
CPU_FREQ_HZ          ?= 1000000000

# 调试器 run/continue 的时钟源超时 (秒), 0 = 禁用
DEFAULT_TIMEOUT      ?= 3600

# ---- 运行时调优 ----
# WFI 空转单次睡眠上限 (秒), 降低宿主机 CPU 占用
WFI_MAX_SLEEP        ?= 0.2
# virtio 单次 _process_queue 最多处理描述符数, 拆小批避免长时间占用主线程
VIRTIO_PROCESS_BATCH ?= 16
# 段加载阈值 (字节): 不超过阈值的段走 L2 缓存, 更大段直写 RAM
FAST_LOAD_THRESHOLD  ?= 0x200000
# mtime 指令计数源的每指令纳秒数
NS_PER_INSTR         ?= 20

# ---- 中断子系统 ----
# 1 = AIA (IMSIC+APLIC); 0 = legacy PLIC
PYREMU_AIA           ?= 0
# 1 = H-extension (HS/VS modes + hgatp + virtual interrupts)
PYREMU_H_EXT         ?= 0
# IMSIC M-file MMIO 基址
IMSIC_M_BASE         ?= 0x24000000
# IMSIC S-file MMIO 基址 (MSI doorbell for devices)
IMSIC_S_BASE         ?= 0x28000000
# APLIC MMIO 基址
APLIC_BASE           ?= 0x0C000000
# WFI 全 idle 无定时器时的保底轮询间隔 (ms)
WFI_WATCHDOG_MS      ?= 5

# ---- 诊断开关 (可通过环境变量在运行时覆盖) ----
PYREMU_DIAG_LOG      ?= /tmp/pyremu_diag.log
PYREMU_DIAG_VERBOSE  ?= 0
# 1 = 打印 virqueue 请求等详细诊断
PYREMU_TRACE_SRET    ?= 0
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
