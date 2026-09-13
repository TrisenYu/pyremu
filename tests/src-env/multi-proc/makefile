# SPDX-LICENSE-IDENTIFIER: MIT
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026
# Multi-program kernel build — 用于多进程抢占测试

elf_dir=../../../build/elf
bin_dir=../../../build/firm-bin

llvm_mc=llvm-mc
lld=ld.lld

.PHONY: all clean

all: kernel.elf kernel_ab.elf

kernel.elf:
	@mkdir -p $(elf_dir)
	@$(llvm_mc) --arch=riscv64 --triple=riscv64 --mattr=+m,+a \
		kernel.s --filetype=obj -o kernel.tmp
	@$(llvm_mc) --arch=riscv64 --triple=riscv64 --mattr=+m,+a \
		prog_fib.s --filetype=obj -o fib.tmp
	@$(llvm_mc) --arch=riscv64 --triple=riscv64 --mattr=+m,+a \
		prog_nqueen.s --filetype=obj -o nqueen.tmp
	@$(llvm_mc) --arch=riscv64 --triple=riscv64 --mattr=+m,+a \
		prog_a.s --filetype=obj -o a.tmp
	@$(llvm_mc) --arch=riscv64 --triple=riscv64 --mattr=+m,+a \
		prog_b.s --filetype=obj -o b.tmp
	@$(lld) -T multi.ld kernel.tmp fib.tmp nqueen.tmp a.tmp b.tmp \
		-o $(elf_dir)/kernel.elf
	-rm kernel.tmp fib.tmp nqueen.tmp a.tmp b.tmp

kernel_ab.elf:
	@$(llvm_mc) --arch=riscv64 --triple=riscv64 --mattr=+m,+a \
		kernel.s --filetype=obj -o kernel.tmp
	@$(llvm_mc) --arch=riscv64 --triple=riscv64 --mattr=+m,+a \
		prog_a.s --filetype=obj -o a.tmp
	@$(llvm_mc) --arch=riscv64 --triple=riscv64 --mattr=+m,+a \
		prog_b.s --filetype=obj -o b.tmp
	@$(lld) -T multi.ld kernel.tmp a.tmp b.tmp \
		-o $(elf_dir)/kernel_ab.elf
	-rm kernel.tmp a.tmp b.tmp

clean:
	-rm -f *.tmp $(elf_dir)/kernel.elf
