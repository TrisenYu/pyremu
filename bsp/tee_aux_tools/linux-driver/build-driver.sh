#!/usr/bin/env sh
# SPDX-LICENSE-IDENTIFIER: MIT
# Author: <jajune257@gmail.com> 2026

# 模块编译 **要求**：编译工具链必须与编译源码时一致
# 见 build-linux.sh 中 gcc/ clang / custom-llvm的部分

# 此处默认使用gcc作为编译工具
make -C ../../linux M=$(pwd) ARCH=riscv CROSS_COMPILE=riscv64-linux-gnu- modules
