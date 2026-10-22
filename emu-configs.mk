# emu-configs.mk — 模拟器硬件特性配置

# 供特定固件代码实现使用, 需与 custom-opensbi Kconfig (POOL_BASE/POOL_SIZE) 保持同步.
# DTB /reserved-memory no-map 节点据此生成, 确保 Linux 内核线性映射排除该区域,
# 避免内核分配器与固件在同一物理区间内产生访问冲突.
RESERVED_MEM_BASE    ?= 0x83000000
# 2.25 GiB. 须容纳 stress_ng 的 --vm-bytes 2G 档位: 该档位自身占用 2 GiB, 另需
# 载荷镜像、argv 与 M 模式按 2 MiB 分区取整所需的余量. 池尾为 0x113000000
# (~4.30 GiB), 故 ram 须不小于 6G, 以留出内核与根文件系统的空间.
RESERVED_MEM_SIZE    ?= 0x90000000

# 固件 (custom-opensbi) 载入 ram_base, 其映像尾部内嵌 .sittim 飞地运行时
# (物理地址 0x80180000, 长 0x202F68 字节: 低 2 MiB 放代码与只读数据, 高 2 MiB 放
# 可写数据). 自 ram_base 起直至内核载入地址 FW_JUMP_ADDR (0x80400000) 的整个区间
# 都必须保留: 否则内核分配器会把页面分到固件映像上, 设备 DMA (如 virtio-blk) 会
# 直接覆写 .sittim, 使 CREATE 复制进飞地池的载荷变成垃圾数据, 飞地入口立即取指
# 故障并被 M 模式静默挂起.
FW_RESERVED_MEM_BASE ?= 0x80000000
# 4 MiB: 固件映像及其后 .sittim 的末尾止于 0x80382F68, 上界取下一个 2 MiB 边界.
FW_RESERVED_MEM_SIZE ?= 0x400000

# 生成到 pyremu/configs_gen.py, 作为 PlatformConfig 与 PeripheralConfig 的字段默认值.
# 调试器命令行给出的 --harts / --ram / --ram-base 等参数优先于这些默认值.

# hart 数量
NUM_HARTS            ?= 1
# 物理 RAM 起始地址与大小 (字节)
RAM_BASE             ?= 0x80000000
RAM_SIZE             ?= 0x08000000
# 上电复位入口地址
PROG_CNT             ?= 0x80000000
# L2 缓存大小 (字节)
L2_SIZE              ?= 0x40000
# ISA 字符串
ISA                  ?= rv64imacfd_sstc_zicsr_zifencei
# 通过 makefile 生成 pyremu/configs_gen.py, 替代各处硬编码与 os.environ.get.
# TLB 条目数
TLB_ENTRIES          ?= 256
# PMP 条目数. 固件在启动时按可访问的 pmpaddr / pmpcfg 探测该数目，同时可能影响固件的PMP_COUNT
PMP_ENTRIES          ?= 16
# mtime/mcycle 随宿主机时间推进
CPU_FREQ_HZ          ?= 1000000000
# 时钟源 (time CSR 与 CLINT mtime) 的频率 (Hz). 模拟器按指令数推进 mtime 的换算、
# DTB 的 timebase-frequency 与飞地运行时的定时换算共用该取值.
TIMER_FREQ           ?= 100000
# WFI 空转单次睡眠上限，单位s
WFI_MAX_SLEEP        ?= 0.2

# 外设 MMIO 基址 (0 = 禁用)
UART_BASE            ?= 0x10000000
SPI_BASE             ?= 0 # 0x10001000
I2C_BASE             ?= 0 # 0x10002000
GPIO_BASE            ?= 0 # 0x10003000
CLINT_BASE           ?= 0x02000000
PLIC_BASE            ?= 0x0C000000
WATCHDOG_BASE        ?= 0x10004000
CRNG_BASE            ?= 0 # 0x10006000
VIRTIO_BLK_BASE      ?= 0
VIRTIO_NET_S_BASE    ?= 0
VIRTIO_NET_M_BASE    ?= 0


# 调试器 run/continue 的时钟源超时 (秒), 0 = 禁用
DEFAULT_TIMEOUT      ?= 36000



# virtio 单次 _process_queue 最多处理描述符数, 拆小批避免长时间占用主线程
VIRTIO_PROCESS_BATCH ?= 16
# 段加载阈值 (字节): 不超过阈值的段走 L2 缓存, 更大段直写 RAM
FAST_LOAD_THRESHOLD  ?= 0x200000

# mtime 指令计数源的每指令纳秒数
NS_PER_INSTR         ?= 20

# WFI 全 idle 无定时器时固定时长轮询间隔，单位ms
WFI_WATCHDOG_MS      ?= 5



## ---- 中断控制器相关
# IMSIC M-file MMIO 基址
IMSIC_M_BASE         ?= 0x24000000
# IMSIC S-file MMIO 基址
IMSIC_S_BASE         ?= 0x28000000
# APLIC S 域 MMIO 基址
APLIC_S_BASE         ?= 0x0C000000
# APLIC M 域 MMIO 基址
APLIC_M_BASE         ?= 0x0C008000


# 1 = AIA (IMSIC+APLIC); 0 = legacy PLIC
PYREMU_AIA           ?= 0
# 1 = H-extension (HS/VS modes + hgatp + virtual interrupts)
PYREMU_H_EXT         ?= 0
# 1 = 关闭 L2 缓存: RAM 访问全部绕过 L2 缓存
PYREMU_NO_L2         ?= 0
# 1 = 关闭 TLB: 查询与插入一并跳过, 每次地址翻译都完整遍历 Sv39 页表
PYREMU_NO_TLB        ?= 0
# 1 = 带 marker 的 ebreak 由模拟器内联服务, 固件输出写标准错误
# 0 = 不拦截该序列, ebreak 按普通断点投递, 固件控制台回落到 UART 并写入每 hart 的日志
#     以 SYS_EXIT 停机序列收尾的用例随之无法结束连续执行
PYREMU_HAS_SEMIHOSTING ?= 0

export
