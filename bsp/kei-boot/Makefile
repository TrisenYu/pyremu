# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved.
# 	Author: <jajune257@gmail.com> in 2026
# 遍く奇跡の始発点

CLANG   ?= clang
OBJCOPY ?= llvm-objcopy
firmbin_dir ?= ../../build/firm-bin
RAM_BASE ?= 0x80000000

# kei.sav.S require macro RAM_BASE for basic memory access.
# Further the MMIO base address of uart/GPIO will be passed via -D as well.
CONF_DEFS = -DRAM_BASE=$(RAM_BASE)
ZSBL_FLAGS = -target riscv64 -march=rv64ima -mabi=lp64 \
	-fPIE -nostdlib -fuse-ld=lld

phony = kei-sav
kei-sav:
	@mkdir -p $(firmbin_dir)
	$(CLANG) $(ZSBL_FLAGS) $(CONF_DEFS) \
		kei.sav.S                       \
		-T kei.sav.lds                  \
		-o kei.sav.elf
	$(OBJCOPY) kei.sav.elf \
		-O binary          \
		$(firmbin_dir)/kei.sav
	@echo "kei.sav.elf is stripped. Stored under $(firmbin_dir)/kei.sav"

clean:
	-rm -f kei.sav.elf
phony += clean

.PHONY: $(phony)
