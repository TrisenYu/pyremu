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
CLANG = $(custom_bin_dir)/clang
OBJCOPY = $(custom_bin_dir)/llvm-objcopy
ZSBL_FLAGS = -target riscv64 -march=rv64ima -mabi=lp64 -fPIE -nostdlib -fuse-ld=lld

zsbl_fsbl_stub:
	$(CLANG) $(ZSBL_FLAGS) zsbl_fsbl_stub.S -T zsbl_fsbl.lds -o zsbl_fsbl_stub.elf
	$(OBJCOPY) -O binary zsbl_fsbl_stub.elf $(firmbin_dir)/zsbl_fsbl_stub_asm.bin
	$(CLANG) $(ZSBL_FLAGS) zsbl_fsbl_stub.S -DCOLD_BOOT -T zsbl_fsbl.lds -o zsbl_fsbl_stub_cold.elf
	$(OBJCOPY) -O binary zsbl_fsbl_stub_cold.elf $(firmbin_dir)/zsbl_fsbl_stub_cold_asm.bin
phony += zsbl_fsbl_stub

# Clean
clean:
	-rm -f $(elf_dir)/kernel.elf $(elf_dir)/kernel_ab.elf
	-rm -f zsbl_fsbl_stub.elf zsbl_fsbl_stub_cold.elf
	$(MAKE) -C multi-proc clean
phony += clean

.PHONY: $(phony)
