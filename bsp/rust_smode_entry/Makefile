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

TARGET     := riscv64imac-unknown-none-elf
BUILD_DIR  := target/$(TARGET)/release
OUTPUT     := rust_smode_entry.bin

include config.mk

# DIAGNOSTIC -- diagnostic features.
CARGO_FLAGS ?=
ifneq ($(filter-out 0 false,$(DIAGNOSTIC)),)
CARGO_FLAGS += --features diagnostic
endif

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
	@echo 'pub const SMODE_STACK_SIZE: u64 = $(SMODE_STACK_SIZE);' >> $@
	@echo 'pub const LINK_BASE: u64 = $(LINK_BASE);' >> $@
	@echo 'pub const ATTEST_ENABLE: bool = $(ATTEST_ENABLE);' >> $@
	@echo 'pub const ATTEST_PUB_KEY: [u8; 33] = [$(ATTEST_PUB_KEY)];' >> $@
phony += src/configs_gen.rs


gen-config: src/configs_gen.rs
phony += gen-config


check: gen-config link.ld
	cargo check --target $(TARGET) $(CARGO_FLAGS)
phony += check

build: check
	cargo build --release --target $(TARGET) $(CARGO_FLAGS)
	$(OBJCOPY) -O binary $(BUILD_DIR)/rust_smode_entry $(OUTPUT)
	@echo "[$(OUTPUT) built successfully]"
phony += build

clean:
	cargo clean
	rm -f $(OUTPUT) src/configs_gen.rs link.ld
phony += clean


.PHONY: $(phony) # all build check clean gen-config
