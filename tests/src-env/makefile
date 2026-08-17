# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026
# 测试汇编构建 — dispatches to subdirectory builds.

custom_bin_dir=/opt/custom-llvm/bin
elf_dir=../../build/elf

phony = all
all: build-multi

# 多程序内核 (kernel + fib + nqueen + A/B counter) — 委托给 multi-proc/
build-multi:
	$(MAKE) -C multi-proc
phony += build-multi

clean:
	-rm -f $(elf_dir)/kernel.elf $(elf_dir)/kernel_ab.elf
	$(MAKE) -C multi-proc clean
phony += clean

.PHONY: $(phony)
