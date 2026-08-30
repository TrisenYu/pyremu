# fn_apps/config.mk — 各飞地载荷共享的 musl 交叉编译配置
#
# 用法: 在 fn_apps/<name>/Makefile 顶部 `include ../config.mk`。
# 本文件只放变量, 不放规则/目标。
#
# 工具链构成:
#   - 源码:  vendor/musl (git submodule, 官方仓库 https://git.musl-libc.org/git/musl,
#            全新 clone 需 `git submodule update --init vendor/musl`)
#   - 产物:  bsp/musl-gc-sysroot (由 bsp/build-musl-sysroot.sh 从源码生成, 不进 git)
#   - 编译器: 标准 clang (riscv64-linux-musl 目标), 载荷不含厂商私有指令, 无需 custom-llvm
#
# 关键约束: 必须 -march=rv64gc -mabi=lp64d (双精度硬浮点), 与模拟器 ISA
# (rv64imacfd) 及 Rust prebuilt std (lp64d) 一致。软浮点 lp64 会错位浮点传参。

# config.mk 所在目录 = fn_apps/; 仓库根 = fn_apps 的上级目录 (回退一级)
_FNAPPS_DIR := $(dir $(lastword $(MAKEFILE_LIST)))
REPO        := $(abspath $(_FNAPPS_DIR)..)

CLANG := clang
QEMU  := qemu-riscv64

TARGET := riscv64-linux-musl
ARCH   := -march=rv64gc -mabi=lp64d

# musl 源码 (submodule) 与构建产物 (sysroot)
MUSL_SRC  := $(REPO)/vendor/musl
MUSL_SYSROOT := $(REPO)/bsp/musl-gc-sysroot
MUSL_LIB  := $(MUSL_SYSROOT)/lib

# 编译参数: -nostdinc 屏蔽宿主 glibc 头, 改用 -isystem 引入 musl 头 (含 arch bits 合并)
CFLAGS := $(ARCH) -static -nostdinc -O2 -Wall \
  -isystem $(MUSL_SRC)/include \
  -isystem $(MUSL_SRC)/arch/riscv64 \
  -isystem $(MUSL_SRC)/arch/generic \
  -isystem $(MUSL_SRC)/obj/include

# 链接参数: -B 找 crt*.o, -L 找 libc.a/libm.a/libgcc.a
LDFLAGS := $(ARCH) -static -fuse-ld=lld \
  -B$(MUSL_LIB) -L$(MUSL_LIB)
