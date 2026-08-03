# SPDX-LICENSE-IDENTIFIER: GPL-2.0
# Author: <kisfg@hotmail.com> 2026
#
# Rust S-mode Runtime
# Uses custom LLVM toolchain at /opt/custom-llvm/bin/
#
# 配置: 编辑 config.mk 或者通过命令行传递定义, e.g.
#   make TIMER_INTERVAL=50000 UART_BASE=0x10011000

CUSTOM_LLVM := /opt/custom-llvm/bin
OBJCOPY    := $(CUSTOM_LLVM)/llvm-objcopy
CLANG      := $(CUSTOM_LLVM)/clang

TARGET     := riscv64imac-unknown-none-elf
BUILD_DIR  := target/$(TARGET)/release
OUTPUT     := rust_smode_entry.bin

include config.mk

phony = all

all: build

# 由 link.ld.S 和 config.mk 经 C 预处理器生成链接脚本.
link.ld: link.ld.S config.mk Makefile
	$(CLANG) -E -P -x c -DLINK_BASE=$(LINK_BASE) $< -o $@

phony += link.ld

# 由config.mk生成常量定义代码 (Kbuild-style).
# 如果config.mk 或者 Makefile 本身有所改变，就自动尝试生成常量
src/config_gen.rs: config.mk Makefile
	@echo '// generated from config.mk by Makefile — do not edit' > $@
	@echo 'pub const TIMER_INTERVAL: u64 = $(TIMER_INTERVAL);' >> $@
	@echo 'pub const TIMER_FREQ: u64 = $(TIMER_FREQ);' >> $@
	@echo 'pub const TIME_QUOTA: u64 = $(TIME_QUOTA);' >> $@
	@echo 'pub const UART_BASE: u64 = $(UART_BASE);' >> $@
	@echo 'pub const UART_TXDATA: u64 = $(UART_TXDATA);' >> $@
	@echo 'pub const UART_RXDATA: u64 = $(UART_RXDATA);' >> $@
	@echo 'pub const UART_TXCTRL: u64 = $(UART_TXCTRL);' >> $@
	@echo 'pub const UART_TXEN: u32 = $(UART_TXEN);' >> $@
	@echo 'pub const SMODE_STACK_SIZE: u64 = $(SMODE_STACK_SIZE);' >> $@
	@echo 'pub const LINK_BASE: u64 = $(LINK_BASE);' >> $@
	@echo 'pub const ATTEST_ENABLE: bool = $(ATTEST_ENABLE);' >> $@
	@echo 'pub const ATTEST_PUB_KEY: [u8; 33] = [$(ATTEST_PUB_KEY)];' >> $@
phony += src/config_gen.rs


gen-config: src/config_gen.rs
phony += gen-config


check: gen-config link.ld
	cargo check --target $(TARGET)
phony += check

build: check
	cargo build --release --target $(TARGET)
	$(OBJCOPY) -O binary $(BUILD_DIR)/rust_smode_entry $(OUTPUT)
	@echo "[$(OUTPUT) built successfully]"
phony += build

clean:
	cargo clean
	rm -f $(OUTPUT) src/config_gen.rs link.ld
phony += clean


.PHONY: $(phony) # all build check clean gen-config
