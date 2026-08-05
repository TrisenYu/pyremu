# SPDX-License-Identifier: GPL-2.0
# tee_enclave_drv Kbuild Makefile
#
# 外部模块编译: 工具链必须与编译内核树 (bsp/linux) 所用完全一致,
# 否则 ABI 不匹配会导致 insmod 时符号解析失败或运行时崩溃.
#
# 当前内核树编译命令 (参考):
#   make LLVM=1 HOSTCC=gcc ARCH=riscv \
#     LLVM_PREFIX=/opt/custom-llvm/bin/ PATH="/opt/custom-llvm/bin:$$PATH" \
#     CROSS_COMPILE=riscv64-unknown-linux-gnu- -j8
#
# 模块编译 (必须与上方一致):
#   make -C ../linux M=$$PWD ARCH=riscv LLVM=1 \
#     LLVM_PREFIX=/opt/custom-llvm/bin/ PATH="/opt/custom-llvm/bin:$$PATH" \
#     CROSS_COMPILE=riscv64-unknown-linux-gnu- modules

obj-m := tee_enclave_drv.o
