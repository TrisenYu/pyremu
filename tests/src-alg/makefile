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


all:
	$(custom_clang) $(generic_flag) nonsense.c -O2 -o $(elf_dir)/nonsense.o


# inspect:
# 	# 检查
# 	$(custom_objdump) -D --section=".text.init" $(elf_dir)/zsbl.o | head -n 32
