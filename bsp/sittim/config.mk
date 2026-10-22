# Platform configuration for sittim
# Override these in the Makefile or via environment, e.g.
#   make SCHED_FREQ=500 UART_BASE=0x10011000

# 时钟源频率 (Hz). 模拟器按该频率推进 mtime, DTB 的 timebase-frequency 取同一值.
TIMER_FREQ ?= 100000
# 调度定时器的中断频率 (Hz), 即调度所依赖的最小触发频率. 运行时每 TIMER_FREQ / SCHED_FREQ
# 个时钟源计数请求一次中断; 定时等待的截止时刻更早时, 中断随该时刻提前到来.
SCHED_FREQ ?= 1000


# 可信应用时间片.
# 	非零默认: 每次 timer 中断计数, 消耗到达时间片后会经 SUSPEND 让出 CPU并退出到宿主环境;
# 			当宿主重启飞地后会重新设置新时间片. 从而防止单个载荷独占 hart 并长时间阻塞.
# 	0: 表示不受限制
TIME_QUOTA ?= 1000

## UART 配置
# MMIO 基址, 与注册 UART 设备的地址相同.
#   UART_BASE  — 在该地址注册 UART 设备.
UART_BASE ?= 0x10000000
UART_TXDATA ?= 0x00          # TX data register offset
UART_RXDATA ?= 0x04          # RX data register offset
UART_TXCTRL ?= 0x08          # TX control register offset
UART_TXEN ?= 0x1             # TX enabled bit

## 网卡配置
# 飞地侧网卡的 MMIO 基地址与寄存器窗口长度. 该设备由 PMP 直接授予飞地, 飞地的
# S 模式驱动在此窗口内收发帧; M 模式只按同一基址读该设备的中断状态寄存器与
# 中断确认寄存器, 故两侧必须指向同一设备.
# 基地址须按窗口长度对齐; 取 0 表示平台不提供该设备, 此时不建立设备窗口映射.
NET_BASE ?= 0 # 0x10008000
NET_SIZE ?= 0 # 0x200

## 栈内存最大容量
SMODE_STACK_SIZE ?= 0x10000  # S-mode stack in bytes, 64 KiB default
## Linker base
LINK_BASE ?= 0x0             # PIE: 链接基址为 0, 运行时重定位修正

## VA mem layout
# 飞地运行时的虚拟地址空间分区. 各项之间存在约束, 改动须整组核对:
#   - 模块窗口基址高于管理器窗口基址. ecall_aux::va_to_pa 按该顺序判定分支,
#     模块地址若低于管理器窗口基址, 会被按管理器窗口的偏移换算而得到错误的物理地址;
#   - 线性窗口与上述两个窗口互不重叠. 它把物理区间
#     [LINEAR_MAP_START, LINEAR_MAP_START + LINEAR_MAP_SIZE) 映射到 LINEAR_MAP_OFFSET
#     起的同一段 VA 区间;
#   - U 模式区自 1 GiB 起向高地址排列: 堆自 UMODE_HEAP_START_ALIGNED 向上增长,
#     栈顶取 UMODE_STACK_TOP_VA 并向下增长, mmap 自 UMODE_MMAP_BASE 向上增长.
ENCLAVE_MAN_VA_START        ?= 0xFFFF_FFE0_0000_0000
ENCLAVE_MODULE_LOAD_VA_INIT ?= 0xFFFF_FFF0_0000_0000

# 模块窗口可交付的 VA 上界与基址之差. 该值约束窗口内能容纳多少个 2 MiB 分区,
ENCLAVE_MODULE_WINDOW_SIZE  ?= 0x400_0000

LINEAR_MAP_START  ?= 0x1_4000_0000
LINEAR_MAP_SIZE   ?= 0x3_4000_0000
LINEAR_MAP_OFFSET ?= 0xFFFF_FFC0_0000_0000

UMODE_HEAP_START_ALIGNED ?= 0x1_0000_0000
UMODE_STACK_TOP_VA ?= 0x1_4000_0000

# mmap 匿名映射起始 VA, 自此处向高地址增长, 避开堆与栈.
UMODE_MMAP_BASE ?= 0x2_0000_0000


## Attestation
# 是否校验扩展模块映像的签名.
# 	false = 跳过
# 	true = 取入模块映像时须先验签通过, 不通过即取入失败.
ATTEST_ENABLE ?= true

# secp256r1 压缩公钥.
# 模块映像的签名须用此密钥对的私钥签发.
# 需改写成你自己的公钥
ATTEST_PUB_KEY ?= 0x02,0xf9,0x1e,0x35,0xe8,0x6e,0xae,0x2b,0x8b,0x07,0x29,0x83,0x27,0x8c,0x3e,0x82,0x08,0x7c,0xde,0x39,0x07,0x78,0x09,0x66,0x48,0x77,0x48,0xe9,0x8f,0x16,0xc0,0xa0,0xa6
# 上述公钥对应的私钥的路径, 相对 bsp/sittim. 供签名扩展模块映像的 sign-mod
ATTEST_PRIV_KEY_PATH ?= ../attest-keys/attest_priv.pem


## 诊断输出
# 编译时启用诊断输出
# 非 0 即开启, 作用到编译选项 cargo --features diagnostic.
DIAGNOSTIC ?= 0
