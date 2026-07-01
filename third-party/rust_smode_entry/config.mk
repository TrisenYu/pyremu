# Platform configuration for rust_smode_entry
# Override these in the Makefile or via environment, e.g.
#   make TIMER_INTERVAL=50000 UART_BASE=0x10011000

# ---- Timer ----
TIMER_INTERVAL ?= 10000      # 10000 for QEMU, 50000 for VisionFive2
TIMER_FREQ ?= 10000000       # mtime 递增频率 (Hz): 10 MHz for QEMU, 1 MHz for SiFive

# ---- UART (兼容NS16550 / SiFive) ----
UART_BASE ?= 0x10000000      # MMIO 基地址
UART_TXDATA ?= 0x00          # TX data register offset
UART_RXDATA ?= 0x04          # RX data register offset
UART_TXCTRL ?= 0x08          # TX control register offset
UART_TXEN ?= 0x1             # TX 使能位

# ---- Stack ----
SMODE_STACK_SIZE ?= 0x10000  # S-mode stack in bytes (64 KiB default)

# ---- Linker base ----
LINK_BASE ?= 0x0              # PIE: 链接基址为 0, 运行时重定位修正
