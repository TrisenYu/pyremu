# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026
# Top-level makefile — dispatches to subdirectory builds.

custom_bin_dir=/opt/custom-llvm/bin
elf_dir=../bins/elf
firmbin_dir=../bins/firm-bin

phony = all
all: build-multi

# 多程序内核 (kernel + fib + nqueen + A/B counter) — 委托给 multi-proc/
build-multi:
	$(MAKE) -C multi-proc
phony += build-multi


# ZSBL+FSBL preload stub (warm + cold)
#
# 地址配置: 从固件 ELF 符号表提取 coldboot 相关地址, 通过 -D 传入汇编器.
# 类似于 rust_smode_entry/config.mk 的配置化方式, 避免每次固件重编译后
# 手动更新硬编码地址.
CLANG = $(custom_bin_dir)/clang
OBJCOPY = $(custom_bin_dir)/llvm-objcopy
READELF = $(custom_bin_dir)/llvm-readelf

# ---- 可覆盖的配置变量 ----
RAM_BASE ?= 0x80000000

# 固件 ELF 路径 (用于提取符号地址)
FW_ELF ?= $(firmbin_dir)/../elf/custom_opensbi_fw_payload.elf

# 从固件 ELF 提取 coldboot_done 偏移量, 计算物理地址
# 使用 Python 处理 hex 算术, 避免 shell 对大值/前导零的兼容问题
COLD_BOOT_DONE_VADDR := $(shell $(READELF) -s $(FW_ELF) 2>/dev/null | \
	grep 'coldboot_done$$' | awk '{print $$2}')
ifneq ($(COLD_BOOT_DONE_VADDR),)
COLD_BOOT_DONE_ADDR := $(shell python3 -c \
	"print(hex($(RAM_BASE) + 0x$(COLD_BOOT_DONE_VADDR)))")
else
COLD_BOOT_DONE_ADDR := 0x800842F8  # 无法提取符号时的后备值
endif

# 从固件 ELF 提取 coldboot_lottery 偏移量
COLD_BOOT_LOTTERY_VADDR := $(shell $(READELF) -s $(FW_ELF) 2>/dev/null | \
	grep 'coldboot_lottery$$' | awk '{print $$2}')
ifneq ($(COLD_BOOT_LOTTERY_VADDR),)
COLD_BOOT_LOTTERY_ADDR := $(shell python3 -c \
	"print(hex($(RAM_BASE) + 0x$(COLD_BOOT_LOTTERY_VADDR)))")
else
COLD_BOOT_LOTTERY_ADDR := 0x800842F0  # 后备值
endif

ZSBL_DEFS = -DRAM_BASE=$(RAM_BASE) \
	-DCOLD_BOOT_DONE_ADDR=$(COLD_BOOT_DONE_ADDR) \
	-DCOLD_BOOT_LOTTERY_ADDR=$(COLD_BOOT_LOTTERY_ADDR)

ZSBL_FLAGS = -target riscv64 -march=rv64ima -mabi=lp64 -fPIE -nostdlib -fuse-ld=lld

zsbl_fsbl_stub:
	$(CLANG) $(ZSBL_FLAGS) $(ZSBL_DEFS) zsbl_fsbl_stub.S -T zsbl_fsbl.lds -o zsbl_fsbl_stub.elf
	$(OBJCOPY) -O binary zsbl_fsbl_stub.elf $(firmbin_dir)/zsbl_fsbl_stub_asm.bin
	$(CLANG) $(ZSBL_FLAGS) $(ZSBL_DEFS) -DCOLD_BOOT zsbl_fsbl_stub.S -T zsbl_fsbl.lds -o zsbl_fsbl_stub_cold.elf
	$(OBJCOPY) -O binary zsbl_fsbl_stub_cold.elf $(firmbin_dir)/zsbl_fsbl_stub_cold_asm.bin
phony += zsbl_fsbl_stub

# Clean
clean:
	-rm -f $(elf_dir)/kernel.elf $(elf_dir)/kernel_ab.elf
	-rm -f zsbl_fsbl_stub.elf zsbl_fsbl_stub_cold.elf
	$(MAKE) -C multi-proc clean
phony += clean

.PHONY: $(phony)
