#!/usr/bin/env bash
# 从 vendor/musl 重建 bsp/musl-gc-sysroot (riscv64gc/lp64d 静态 musl sysroot)。
#
# 该 sysroot 是 fn_apps/*/Makefile 交叉编译飞地载荷时链接的目标:
#   - libc.a / libm.a / crt1.o / crti.o / crtn.o / Scrt1.o  来自 musl 构建
#   - libgcc.a / crtbeginT.o / crtend.o                      来自 glibc 交叉 GCC 14
#   - libgcc_s.a                                            来自裸机 riscv64-unknown-elf GCC
#     (rustc 对 musl 目标追加 -lgcc_s 以解析 _Unwind_*, glibc 版 libgcc_eh.a 依赖
#      glibc 专有符号 _dl_find_object, 故用裸机 libgcc.a 充当, 详见脚本内注释)
#
# 关键约束: 必须用 -march=rv64gc -mabi=lp64d (双精度浮点 ABI), 与载荷的 printf("%f")
# 传参约定一致。若退化为软浮点 lp64, 浮点参数的寄存器约定会错位, 导致输出错误。
#
# 该脚本从 vendor/musl 生成 sysroot;
# 全新 clone 需先 `git submodule update --init vendor/musl` 取源码。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MUSL_SRC="$ROOT/vendor/musl"
SYSROOT="$ROOT/bsp/musl-gc-sysroot"

# 标准 clang / llvm-ar / llvm-ranlib
CLANG="clang"
LLVM_AR="llvm-ar"
LLVM_RANLIB="llvm-ranlib"

# 交叉 GCC 运行时目录: 版本号由 -print-file-name 得出
# libgcc.a 的实际位置 (跨机器 / 跨版本 13/14/15... 均成立)。
GCC="riscv64-linux-gnu-gcc"
GCC_DIR="$(dirname "$("$GCC" -print-file-name=libgcc.a)")"

# 裸机 riscv64-unknown-elf GCC 的 rv64iafd/lp64d multilib 变体, 其 libgcc.a 同时含
# 基础算术与 _Unwind_* unwind 符号, 且不依赖 glibc 专有符号 (_dl_find_object),
# 供 musl 静态链接。用 -march/-mabi 精确选中该 multilib 后再定位 libgcc.a 目录
# (multilib 目录名随工具链构建参数变化, 同样不可硬编码)。
ELF_GCC="riscv64-unknown-elf-gcc"
ELF_GCC_DIR="$(dirname "$("$ELF_GCC" -march=rv64iafd -mabi=lp64d -print-file-name=libgcc.a)")"

MUSL_CFLAGS="--target=riscv64-linux-musl -march=rv64gc -mabi=lp64d -fuse-ld=lld"

# 前置检查: 工具链缺一即报错 (而不是等到 -print-file-name 时 cryptic 地 command not found)
for cc in "$CLANG" "$LLVM_AR" "$LLVM_RANLIB" "$GCC" "$ELF_GCC"; do
    if ! command -v "$cc" >/dev/null 2>&1; then
        echo "错误: 缺少工具链 $cc" >&2
        echo "  安装: apt install clang lld gcc-riscv64-linux-gnu gcc-riscv64-unknown-elf" >&2
        exit 1
    fi
done

cd "$MUSL_SRC"

# 清掉上次配置与构建产物, 避免残留旧 ABI 的对象
make distclean >/dev/null 2>&1 || true

./configure \
    --target=riscv64 \
    --prefix=/ \
    --disable-shared \
    CC="$CLANG" \
    CFLAGS="$MUSL_CFLAGS" \
    AR="$LLVM_AR" \
    RANLIB="$LLVM_RANLIB"

make -j"$(nproc)"

# 安装 musl 静态库与 crt 到 sysroot (lib/libc.a, lib/libm.a, lib/crt*.o, lib/Scrt1.o)
rm -rf "$SYSROOT"
make install DESTDIR="$SYSROOT"

# 头文件沿用源码树 (vendor/musl/include + arch bits), sysroot 内的 include 冗余
rm -rf "$SYSROOT/include"

# 复制 GCC 运行时, 满足 clang -static 的隐式 -lgcc_eh / crtbeginT / crtend
mkdir -p "$SYSROOT/lib"
cp "$GCC_DIR/libgcc.a"    "$SYSROOT/lib/libgcc.a"
# rustc 对 musl 目标默认追加 -lgcc_s, 而 glibc 交叉 GCC 的 libgcc_s.so 仅为 GNU ld
# 脚本 (GROUP libgcc_s.so.1 -lgcc), 引用不存在的 .so.1 导致静态链接失败; 且其
# libgcc_eh.a 依赖 glibc 专有符号 _dl_find_object (musl 无此符号)。改用裸机
# riscv64-unknown-elf GCC 的 libgcc.a (含 _Unwind_* 且不依赖 glibc) 充当 libgcc_s.a。
cp "$ELF_GCC_DIR/libgcc.a" "$SYSROOT/lib/libgcc_s.a"
cp "$GCC_DIR/crtbeginT.o" "$SYSROOT/lib/crtbeginT.o"
cp "$GCC_DIR/crtend.o"    "$SYSROOT/lib/crtend.o"

echo "已生成 sysroot: $SYSROOT/lib"
ls -1 "$SYSROOT/lib"
