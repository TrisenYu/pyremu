# fn_apps/config.mk — 各飞地载荷共享的 musl 交叉编译配置
#
# 用法: 在 fn_apps/<name>/Makefile 顶部 `include ../config.mk`。
# 本文件只放变量, 不放规则/目标。
#
# 工具链构成:
#   - 源码:  vendor/musl <https://git.musl-libc.org/git/musl>
#           `git submodule update --init vendor/musl`将拉取至本地
#   - 产物:  bsp/musl-gc-sysroot
#   - 编译器clang: riscv64-linux-musl为编译目标

# config.mk 所在目录 = fn_apps/; 仓库根 = fn_apps 的上级目录
_FNAPPS_DIR := $(dir $(lastword $(MAKEFILE_LIST)))
REPO        := $(abspath $(_FNAPPS_DIR)..)

CLANG := clang
QEMU  := qemu-riscv64

TARGET := riscv64-linux-musl
ARCH   := -march=rv64gc -mabi=lp64d

# musl 源码与构建产物musl-gc-sysroot
MUSL_SRC  := $(REPO)/vendor/musl
MUSL_SYSROOT := $(REPO)/bsp/musl-gc-sysroot
MUSL_LIB  := $(MUSL_SYSROOT)/lib

CFLAGS := $(ARCH) -static -nostdinc -O2 -Wall \
  -isystem $(MUSL_SRC)/include \
  -isystem $(MUSL_SRC)/arch/riscv64 \
  -isystem $(MUSL_SRC)/arch/generic \
  -isystem $(MUSL_SRC)/obj/include
LDFLAGS := $(ARCH) -static -fuse-ld=lld \
  -B$(MUSL_LIB) -L$(MUSL_LIB)
