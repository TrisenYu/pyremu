# SPDX-LICENSE-IDENTIFIER: MIT
# Author: <kisfg@hotmail.com> 2026
#
# Rust S-mode Runtime
# Uses custom LLVM toolchain at /opt/custom-llvm/bin/
#
# configuration: edit config.mk or pass arguments via command line, e.g.
#   make TIMER_INTERVAL=50000 UART_BASE=0x10011000

CUSTOM_LLVM := /opt/custom-llvm/bin
OBJCOPY    := $(CUSTOM_LLVM)/llvm-objcopy
CLANG      := $(CUSTOM_LLVM)/clang

TARGET      := riscv64imac-unknown-none-elf
HOST_TARGET := x86_64-unknown-linux-gnu
BUILD_DIR  := target/$(TARGET)/release
OUTPUT     := sittim.bin

include config.mk

# DIAGNOSTIC -- diagnostic features.
CARGO_FLAGS ?=
ifneq ($(filter-out 0 false,$(DIAGNOSTIC)),)
CARGO_FLAGS += --features diagnostic
endif

FIRM_FLAGS := --features firm

phony = all

all: build

link.ld: link.ld.S config.mk Makefile
	$(CLANG) -E -P -x c -DLINK_BASE=$(LINK_BASE) $< -o $@

phony += link.ld

# When there is any change in config.mk or Makefile, regenerate constant here.
src/configs_gen.rs: config.mk Makefile
	@echo '// generated from config.mk by Makefile — do not edit' > $@
	@echo 'pub const TIMER_INTERVAL: u64 = $(TIMER_INTERVAL);' >> $@
	@echo 'pub const TIMER_FREQ: u64 = $(TIMER_FREQ);' >> $@
	@echo 'pub const TIME_QUOTA: u64 = $(TIME_QUOTA);' >> $@
	@echo 'pub const UART_BASE: u64 = $(UART_BASE);' >> $@
	@echo 'pub const UART_TXDATA: u64 = $(UART_TXDATA);' >> $@
	@echo 'pub const UART_RXDATA: u64 = $(UART_RXDATA);' >> $@
	@echo 'pub const UART_TXCTRL: u64 = $(UART_TXCTRL);' >> $@
	@echo 'pub const UART_TXEN: u32 = $(UART_TXEN);' >> $@
	@echo 'pub const NET_BASE: u64 = $(NET_BASE);' >> $@
	@echo 'pub const NET_SIZE: u64 = $(NET_SIZE);' >> $@
	@echo 'pub const SMODE_STACK_SIZE: u64 = $(SMODE_STACK_SIZE);' >> $@
	@echo 'pub const LINK_BASE: u64 = $(LINK_BASE);' >> $@
	@echo 'pub const ENCLAVE_MAN_VA_START: u64 = $(ENCLAVE_MAN_VA_START);' >> $@
	@echo 'pub const ENCLAVE_MODULE_LOAD_VA_INIT: u64 = $(ENCLAVE_MODULE_LOAD_VA_INIT);' >> $@
	@echo 'pub const ENCLAVE_MODULE_WINDOW_SIZE: u64 = $(ENCLAVE_MODULE_WINDOW_SIZE);' >> $@
	@echo 'pub const LINEAR_MAP_START: u64 = $(LINEAR_MAP_START);' >> $@
	@echo 'pub const LINEAR_MAP_SIZE: u64 = $(LINEAR_MAP_SIZE);' >> $@
	@echo 'pub const LINEAR_MAP_OFFSET: u64 = $(LINEAR_MAP_OFFSET);' >> $@
	@echo 'pub const UMODE_HEAP_START_ALIGNED: u64 = $(UMODE_HEAP_START_ALIGNED);' >> $@
	@echo 'pub const UMODE_STACK_TOP_VA: u64 = $(UMODE_STACK_TOP_VA);' >> $@
	@echo 'pub const UMODE_MMAP_BASE: u64 = $(UMODE_MMAP_BASE);' >> $@
	@echo 'pub const ATTEST_ENABLE: bool = $(ATTEST_ENABLE);' >> $@
	@echo 'pub const ATTEST_PUB_KEY: [u8; 33] = [$(ATTEST_PUB_KEY)];' >> $@
phony += src/configs_gen.rs


gen-config: src/configs_gen.rs
phony += gen-config


check: gen-config link.ld
	cargo check --target $(TARGET) $(FIRM_FLAGS) $(CARGO_FLAGS)
phony += check

build: check
	cargo build --release --target $(TARGET) $(FIRM_FLAGS) $(CARGO_FLAGS)
	$(OBJCOPY) -O binary $(BUILD_DIR)/sittim $(OUTPUT)
	@echo "[$(OUTPUT) built successfully]"
phony += build

# 扩展模块的映像与签名, 由 modules/ 下的独立工作区构建。
modules:
	$(MAKE) -C modules
phony += modules

# 测试须先由 gen-config 生成 src/configs_gen.rs, 该文件被 constants.rs 以 include!
# 包含, 而 clean 会删除它。
test: gen-config
	cargo test --target $(HOST_TARGET) $(CARGO_FLAGS)
phony += test

clean:
	cargo clean
	rm -f $(OUTPUT) src/configs_gen.rs link.ld
	$(MAKE) -C modules clean
phony += clean

.PHONY: $(phony) # all build check clean gen-config
