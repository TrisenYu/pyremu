# SPDX-LICENSE-IDENTIFIER: GPL2.0
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026
# Multi-program kernel build — 用于多进程抢占测试

custom_bin_dir=/opt/custom-llvm/bin
elf_dir=../../bins/elf
bin_dir=../../bins/firm-bin

custom_llvm_mc=$(custom_bin_dir)/llvm-mc
custom_lld=$(custom_bin_dir)/ld.lld

.PHONY: all clean

all: kernel.elf kernel_ab.elf

kernel.elf:
	$(custom_llvm_mc) --arch=riscv64 --triple=riscv64 --mattr=+m,+a \
		kernel.s --filetype=obj -o kernel.tmp
	$(custom_llvm_mc) --arch=riscv64 --triple=riscv64 --mattr=+m,+a \
		prog_fib.s --filetype=obj -o fib.tmp
	$(custom_llvm_mc) --arch=riscv64 --triple=riscv64 --mattr=+m,+a \
		prog_nqueen.s --filetype=obj -o nqueen.tmp
	$(custom_llvm_mc) --arch=riscv64 --triple=riscv64 --mattr=+m,+a \
		prog_a.s --filetype=obj -o a.tmp
	$(custom_llvm_mc) --arch=riscv64 --triple=riscv64 --mattr=+m,+a \
		prog_b.s --filetype=obj -o b.tmp
	$(custom_lld) -T multi.ld kernel.tmp fib.tmp nqueen.tmp a.tmp b.tmp \
		-o $(elf_dir)/kernel.elf
	-rm kernel.tmp fib.tmp nqueen.tmp a.tmp b.tmp

kernel_ab.elf:
	$(custom_llvm_mc) --arch=riscv64 --triple=riscv64 --mattr=+m,+a \
		kernel.s --filetype=obj -o kernel.tmp
	$(custom_llvm_mc) --arch=riscv64 --triple=riscv64 --mattr=+m,+a \
		prog_a.s --filetype=obj -o a.tmp
	$(custom_llvm_mc) --arch=riscv64 --triple=riscv64 --mattr=+m,+a \
		prog_b.s --filetype=obj -o b.tmp
	$(custom_lld) -T multi.ld kernel.tmp a.tmp b.tmp \
		-o $(elf_dir)/kernel_ab.elf
	-rm kernel.tmp a.tmp b.tmp

clean:
	-rm -f *.tmp $(elf_dir)/kernel.elf