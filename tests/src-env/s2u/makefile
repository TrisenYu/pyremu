# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026
# M -> S -> U 测试固件构建 — 用于 U 模式栈保护页测试

elf_dir=../../build/elf

llvm_mc=llvm-mc
lld=ld.lld

.PHONY: all clean

all: u_mode_run_fib.elf

u_mode_run_fib.elf:
	@mkdir -p $(elf_dir)
	@$(llvm_mc) --arch=riscv64 --triple=riscv64 --mattr=+m,+a \
		u_mode_run_fib.s --filetype=obj -o u_mode_run_fib.tmp
	@$(lld) -T s2u.ld u_mode_run_fib.tmp \
		-o $(elf_dir)/u_mode_run_fib.elf
	-rm u_mode_run_fib.tmp

clean:
	-rm -f *.tmp $(elf_dir)/u_mode_run_fib.elf
