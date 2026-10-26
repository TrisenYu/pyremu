# SPDX-LICENSE-IDENTIFIER: MIT
# Author: <kisfg@hotmail.com> 2026
#
# Rust S-mode Runtime
# Uses custom LLVM toolchain at /opt/custom-llvm/bin/

CUSTOM_LLVM ?= /opt/custom-llvm/bin
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

# unconditional refresh
FORCE:
phony += FORCE

link.ld: FORCE
	$(CLANG) -E -P -x c -DLINK_BASE=$(LINK_BASE) link.ld.S -o $@.tmp
	@cmp -s $@.tmp $@ 2>/dev/null && rm -f $@.tmp || mv -f $@.tmp $@

phony += link.ld

src/configs_gen.rs: FORCE
	@rm -f $@.tmp
	@echo '// generated from config.mk by Makefile — do not edit' > $@.tmp
	@echo 'pub const SCHED_FREQ: u64 = $(SCHED_FREQ);' >> $@.tmp
	@echo 'pub const TIMER_FREQ: u64 = $(TIMER_FREQ);' >> $@.tmp
	@echo 'pub const TIME_QUOTA: u64 = $(TIME_QUOTA);' >> $@.tmp
	@echo 'pub const UART_BASE: u64 = $(UART_BASE);' >> $@.tmp
	@echo 'pub const UART_TXDATA: u64 = $(UART_TXDATA);' >> $@.tmp
	@echo 'pub const UART_RXDATA: u64 = $(UART_RXDATA);' >> $@.tmp
	@echo 'pub const UART_TXCTRL: u64 = $(UART_TXCTRL);' >> $@.tmp
	@echo 'pub const UART_TXEN: u32 = $(UART_TXEN);' >> $@.tmp
	@echo 'pub const NET_BASE: u64 = $(NET_BASE);' >> $@.tmp
	@echo 'pub const NET_SIZE: u64 = $(NET_SIZE);' >> $@.tmp
	@echo 'pub const SMODE_STACK_SIZE: u64 = $(SMODE_STACK_SIZE);' >> $@.tmp
	@echo 'pub const LINK_BASE: u64 = $(LINK_BASE);' >> $@.tmp
	@echo 'pub const ENCLAVE_MAN_VA_START: u64 = $(ENCLAVE_MAN_VA_START);' >> $@.tmp
	@echo 'pub const ENCLAVE_MODULE_LOAD_VA_INIT: u64 = $(ENCLAVE_MODULE_LOAD_VA_INIT);' >> $@.tmp
	@echo 'pub const EXT_MODS_VAS_SIZE: u64 = $(EXT_MODS_VAS_SIZE);' >> $@.tmp
	@echo 'pub const LINEAR_MAP_START: u64 = $(LINEAR_MAP_START);' >> $@.tmp
	@echo 'pub const LINEAR_MAP_SIZE: u64 = $(LINEAR_MAP_SIZE);' >> $@.tmp
	@echo 'pub const LINEAR_MAP_OFFSET: u64 = $(LINEAR_MAP_OFFSET);' >> $@.tmp
	@echo 'pub const UMODE_HEAP_START_ALIGNED: u64 = $(UMODE_HEAP_START_ALIGNED);' >> $@.tmp
	@echo 'pub const UMODE_STACK_TOP_VA: u64 = $(UMODE_STACK_TOP_VA);' >> $@.tmp
	@echo 'pub const UMODE_MMAP_BASE: u64 = $(UMODE_MMAP_BASE);' >> $@.tmp
	@echo 'pub const ATTEST_ENABLE: bool = $(ATTEST_ENABLE);' >> $@.tmp
	@echo 'pub const ATTEST_PUB_KEY: [u8; 33] = [$(ATTEST_PUB_KEY)];' >> $@.tmp
	@cmp -s $@.tmp $@ 2>/dev/null && rm -f $@.tmp || mv -f $@.tmp $@
phony += src/configs_gen.rs


gen-config: src/configs_gen.rs
phony += gen-config

tests/configs_gen.rs: FORCE
	@python3 ../../tools/gen_attest_key.py $(ATTEST_PRIV_KEY_PATH) > $@.tmp
	@cmp -s $@.tmp $@ 2>/dev/null && rm -f $@.tmp || mv -f $@.tmp $@
phony += tests/configs_gen.rs


check: gen-config link.ld
	cargo check --target $(TARGET) $(FIRM_FLAGS) $(CARGO_FLAGS)
phony += check

build: check
	cargo build --release --target $(TARGET) $(FIRM_FLAGS) $(CARGO_FLAGS)
	$(OBJCOPY) -O binary $(BUILD_DIR)/sittim $(OUTPUT)
	@echo "[$(OUTPUT) built successfully]"
phony += build

modules:
	$(MAKE) -C modules
phony += modules

# use gen-config to first generate src/configs_gen.rs,
# which will be included by constants.rs via include!,
# while `make clean` can delete it.
test: gen-config tests/configs_gen.rs
	cargo test --target $(HOST_TARGET) $(CARGO_FLAGS)
phony += test

clean:
	cargo clean
	rm -f $(OUTPUT) src/configs_gen.rs tests/configs_gen.rs link.ld
	$(MAKE) -C modules clean
phony += clean

.PHONY: $(phony)
