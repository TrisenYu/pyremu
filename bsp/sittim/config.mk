# Platform configuration for sittim
# Override these in the Makefile or via environment, e.g.
#   make TIMER_INTERVAL=50000 UART_BASE=0x10011000

## 定时器配置
TIMER_INTERVAL ?= 10000 # 10000 for QEMU, 50000 for VisionFive2
TIMER_FREQ ?= 10000000  # mtime 递增频率 (Hz): 10 MHz for QEMU, 1 MHz for SiFive
# 可信应用的时间片.
# 	非零默认: 每次 timer 中断计数, 消耗到达时间片后会经 SUSPEND 让出 CPU并退出到宿主环境;
# 			 当宿主重启飞地后会重新设置新时间片. 从而防止单个载荷独占 hart 并长时间阻塞.
# 	0: 表示不受限制
TIME_QUOTA ?= 1000

## UART 配置
UART_BASE ?= 0x10000000      # MMIO base address
UART_TXDATA ?= 0x00          # TX data register offset
UART_RXDATA ?= 0x04          # RX data register offset
UART_TXCTRL ?= 0x08          # TX control register offset
UART_TXEN ?= 0x1             # TX enabled bit

## 网卡 (飞地侧) 配置
# 飞地侧网卡的 MMIO 基地址与寄存器窗口长度. 该设备由 PMP 直接授予飞地, 飞地的
# S 模式驱动在此窗口内收发帧; M 模式只按同一基址读该设备的中断状态寄存器与
# 中断确认寄存器, 故两侧必须指向同一设备.
# 基地址须按窗口长度对齐; 取 0 表示平台不提供该设备, 此时不建立设备窗口映射.
NET_BASE ?= 0x10008000
NET_SIZE ?= 0x200

## Stack
SMODE_STACK_SIZE ?= 0x10000  # S-mode stack in bytes (64 KiB default)

## Linker base
LINK_BASE ?= 0x0              # PIE: 链接基址为 0, 运行时重定位修正

## VA 布局
# 飞地运行时的虚拟地址空间分区. 各项之间存在约束, 改动须整组核对:
#   - 模块窗口基址高于管理器窗口基址. ecall_aux::va_to_pa 按该顺序判定分支,
#     模块地址若低于管理器窗口基址, 会被按管理器窗口的偏移换算而得到错误的物理地址;
#   - 线性窗口与上述两个窗口互不重叠. 它把物理区间
#     [LINEAR_MAP_START, LINEAR_MAP_START + LINEAR_MAP_SIZE) 映射到 LINEAR_MAP_OFFSET
#     起的同一段 VA 区间;
#   - U 模式区自 1 GiB 起向高地址排列: 堆自 UMODE_HEAP_START_ALIGNED 向上增长,
#     栈顶取 UMODE_STACK_TOP_VA 并向下增长, mmap 自 UMODE_MMAP_BASE 向上增长.
ENCLAVE_MAN_VA_START ?= 0xFFFF_FFE0_0000_0000
ENCLAVE_MODULE_LOAD_VA_INIT ?= 0xFFFF_FFF0_0000_0000
# 模块窗口可交付的 VA 上界与基址之差. 该值约束窗口内能容纳多少个 2 MiB 分区,
# 不预留物理内存: 物理内存仍按需向 M 模式申请. 窗口耗尽即取入失败, 不复用已交付的地址.
ENCLAVE_MODULE_WINDOW_SIZE ?= 0x400_0000

LINEAR_MAP_START ?= 0x1_4000_0000
LINEAR_MAP_SIZE ?= 0x3_4000_0000
LINEAR_MAP_OFFSET ?= 0xFFFF_FFC0_0000_0000

UMODE_HEAP_START_ALIGNED ?= 0x1_0000_0000
UMODE_STACK_TOP_VA ?= 0x1_4000_0000
# mmap 匿名映射起始 VA, 自此处向高地址增长, 避开堆与栈.
UMODE_MMAP_BASE ?= 0x2_0000_0000


## Attestation
# 是否强制校验载荷签名. false = 跳过 (未接入签名流水线时保持启动畅通);
# true = 载荷尾部须带有效 ECDSA 签名, 否则飞地拒绝执行并挂起.
ATTEST_ENABLE ?= false

# secp256r1 压缩公钥 (33 字节, 0x02/0x03 前缀 + X 坐标).
# 载荷签名须对 SHA-256 摘要用此密钥对的私钥签发.
# 需改写成你自己的公钥
ATTEST_PUB_KEY ?= 0x02,0xf9,0x1e,0x35,0xe8,0x6e,0xae,0x2b,0x8b,0x07,0x29,0x83,0x27,0x8c,0x3e,0x82,0x08,0x7c,0xde,0x39,0x07,0x78,0x09,0x66,0x48,0x77,0x48,0xe9,0x8f,0x16,0xc0,0xa0,0xa6


## 诊断输出
# 编译时启用诊断输出 (如 [map_sec]/[map_pool]/page_fault/syscall).
# 非 0 即开启, 等价于 cargo --features diagnostic.
DIAGNOSTIC ?= 0
