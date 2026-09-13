# SPDX-License-Identifier: MIT
# tee_enclave_drv Kbuild Makefile
#
# 外部模块编译: 工具链必须与编译内核树 (bsp/linux) 所用完全一致,
# 否则 ABI 不匹配会导致 insmod 时符号解析失败或运行时崩溃.
#
# 当前内核树编译命令 (参考 bsp/build-linux.sh, 默认 gcc):
#   make HOSTCC=gcc ARCH=riscv CROSS_COMPILE=riscv64-linux-gnu- -j8


obj-m := tee_enclave_drv.o
