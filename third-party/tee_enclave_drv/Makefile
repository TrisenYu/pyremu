# SPDX-License-Identifier: GPL-2.0
# tee_enclave_drv Kbuild Makefile
#
# 编译: make -C /path/to/linux M=$PWD modules
# 需要 riscv64 交叉工具链 + 已编译的 Linux 内核树.
#
# 使用:
#   KDIR=$(pwd)/third-party/linux
#   make ARCH=riscv CROSS_COMPILE=riscv64-unknown-linux-gnu- -C $(KDIR) M=$PWD modules

obj-m := tee_enclave_drv.o
