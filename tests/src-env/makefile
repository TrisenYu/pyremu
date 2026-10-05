# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026
# 测试汇编构建 — dispatches to subdirectory builds.

elf_dir=../../build/elf

phony = all
all: build-multi build-s2u

# 多程序内核 (kernel + fib + nqueen + A/B counter) — 委托给 multi-proc/
build-multi:
	$(MAKE) -C multi-proc
phony += build-multi

# M -> S -> U 移交固件 (Sv39 栈保护页 + UART 输入 + fib) — 委托给 s2u/
build-s2u:
	$(MAKE) -C s2u
phony += build-s2u

clean:
	-rm -f $(elf_dir)/kernel.elf $(elf_dir)/kernel_ab.elf
	$(MAKE) -C multi-proc clean
	$(MAKE) -C s2u clean
phony += clean

.PHONY: $(phony)
