# SPDX-LICENSE-IDENTIFIER: GPL2.0
#
# (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026
# Created at 2026/06/09 星期二 13:29:56
# Last modified at 2026/06/09 星期二 13:41:39

custom_bin_dir=/opt/custom-llvm/bin
elf_dir=../bins/elf
bin_dir=../bins/firm-bin

custom_clang=$(custom_bin_dir)/clang
custom_llvm_mc=$(custom_bin_dir)/llvm-mc
custom_lld=$(custom_bin_dir)/ld.lld
custom_objcopy=$(custom_bin_dir)/llvm-objcopy
custom_objdump=$(custom_bin_dir)/llvm-objdump
custom_readelf=$(custom_bin_dir)/llvm-readelf

generic_flag=-march=rv64gv_ztee_zknh


build-prog:
	dtc -I dts -O dtb -o ux00_zsbl.dtb ux00_zsbl.dts
	$(custom_clang) -c sifive-zsbl-start.s -o $(elf_dir)/zsbl.o
	-rm ux00_zsbl.dtb
	$(custom_objcopy) -O binary $(elf_dir)/zsbl.o $(bin_dir)/zsbl-rom.bin
	$(custom_clang) $(generic_flag) nonsense.c -O2 -o $(elf_dir)/nonsense.o
	$(custom_llvm_mc) --arch=riscv64 --triple=riscv64 --mattr=+m,+a,+zknh,+ztee \
	env_setup.s --filetype=obj -o bare_env_setup.tmp
	$(custom_lld) -T env_setup.ld bare_env_setup.tmp -o $(elf_dir)/bare_env_setup.elf
	-rm bare_env_setup.tmp

inspect:
	# 检查
	$(custom_objdump) -D --section=".text.init" $(elf_dir)/zsbl.o | head -n 32
