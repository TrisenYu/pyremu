# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026
# Pyremu 冷/热启动引导程序 kei.sav

CLANG   ?= clang
OBJCOPY ?= llvm-objcopy
READELF ?= llvm-readelf
# 输出目录 (由上层 makefile 传入)
firmbin_dir ?= ../../build/firm-bin
# 固件 ELF 路径 (用于提取符号地址)
FW_ELF  ?= ../../build/elf/custom_opensbi_fw_payload.elf
RAM_BASE ?= 0x80000000

# 从固件 ELF 提取符号地址
COLD_BOOT_DONE_VADDR := $(shell $(READELF) -s $(FW_ELF) 2>/dev/null | \
	grep 'coldboot_done$$' | awk '{print $$2}')
ifneq ($(COLD_BOOT_DONE_VADDR),)
COLD_BOOT_DONE_ADDR := $(shell python3 -c \
	"print(hex($(RAM_BASE) + 0x$(COLD_BOOT_DONE_VADDR)))")
else # 没办法算就只能硬编码
COLD_BOOT_DONE_ADDR := 0x800842F8
endif

COLD_BOOT_LOTTERY_VADDR := $(shell $(READELF) -s $(FW_ELF) 2>/dev/null | \
	grep 'coldboot_lottery$$' | awk '{print $$2}')
ifneq ($(COLD_BOOT_LOTTERY_VADDR),)
COLD_BOOT_LOTTERY_ADDR := $(shell python3 -c \
	"print(hex($(RAM_BASE) + 0x$(COLD_BOOT_LOTTERY_VADDR)))")
else # 没办法算就只能硬编码
COLD_BOOT_LOTTERY_ADDR := 0x800842F0
endif

ZSBL_DEFS = -DRAM_BASE=$(RAM_BASE) \
	-DCOLD_BOOT_DONE_ADDR=$(COLD_BOOT_DONE_ADDR) \
	-DCOLD_BOOT_LOTTERY_ADDR=$(COLD_BOOT_LOTTERY_ADDR)
ZSBL_FLAGS = -target riscv64 -march=rv64ima -mabi=lp64 \
	-fPIE -nostdlib -fuse-ld=lld

phony = kei-sav
kei-sav:
	@mkdir -p $(firmbin_dir)
	$(CLANG) $(ZSBL_FLAGS) $(ZSBL_DEFS) kei.sav.S -T kei.sav.lds -o kei.sav.elf
	$(OBJCOPY) -O binary kei.sav.elf $(firmbin_dir)/kei.sav_asm.bin
	$(CLANG) $(ZSBL_FLAGS) $(ZSBL_DEFS) -DCOLD_BOOT kei.sav.S -T kei.sav.lds -o kei.sav_cold.elf
	$(OBJCOPY) -O binary kei.sav_cold.elf $(firmbin_dir)/kei.sav.bin
	@echo "kei.sav 已按固件符号表更新"

clean:
	-rm -f kei.sav.elf kei.sav_cold.elf
phony += clean

.PHONY: $(phony)
