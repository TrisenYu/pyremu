# Platform configuration for rust_smode_entry
# Override these in the Makefile or via environment, e.g.
#   make TIMER_INTERVAL=50000 UART_BASE=0x10011000

## Timer 定时器配置
TIMER_INTERVAL ?= 10000      # 10000 for QEMU, 50000 for VisionFive2
TIMER_FREQ ?= 10000000       # mtime 递增频率 (Hz): 10 MHz for QEMU, 1 MHz for SiFive
TIME_QUOTA ?= 0              # 飞地时间配额 (timer interrupt 次数), 0=不限. 1000≈1s @10MHz

## UART 配置
UART_BASE ?= 0x10000000      # MMIO 基地址
UART_TXDATA ?= 0x00          # TX data register offset
UART_RXDATA ?= 0x04          # RX data register offset
UART_TXCTRL ?= 0x08          # TX control register offset
UART_TXEN ?= 0x1             # TX 使能位

## Stack
SMODE_STACK_SIZE ?= 0x10000  # S-mode stack in bytes (64 KiB default)

## Linker base
LINK_BASE ?= 0x0              # PIE: 链接基址为 0, 运行时重定位修正


## Attestation
# 是否强制校验载荷签名. false = 跳过 (未接入签名流水线时保持启动畅通);
# true = 载荷尾部须带有效 ECDSA 签名, 否则飞地拒绝执行并挂起.
ATTEST_ENABLE ?= false

# secp256r1 压缩公钥 (33 字节, 0x02/0x03 前缀 + X 坐标).
# 载荷签名须对 SHA-256 摘要用此密钥对的私钥签发.
# 需要改写成你自己的公钥
ATTEST_PUB_KEY ?= 0x02,0xf9,0x1e,0x35,0xe8,0x6e,0xae,0x2b,0x8b,0x07,0x29,0x83,0x27,0x8c,0x3e,0x82,0x08,0x7c,0xde,0x39,0x07,0x78,0x09,0x66,0x48,0x77,0x48,0xe9,0x8f,0x16,0xc0,0xa0,0xa6
