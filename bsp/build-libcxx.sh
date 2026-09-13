#!/usr/bin/env sh
# 从 LLVM 源码树交叉编译 libc++/libc++abi/libunwind 三件套, 装入 bsp/musl-gc-sysroot,
# 补齐 riscv64-linux-musl 静态 C++ 标准库。
#
# 前置: 先运行 bsp/build-musl-sysroot.sh 生成 C 运行库 (libc.a/libgcc.a/crt*),
#       本脚本只追加 C++ 运行库与头, 不重建 C 库。
# 用途: fn_apps/dune (DUNE FEM, C++20) 等 C++ 载荷交叉编译时的 stdlib。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# 仓库内固定路径与工具
MUSL_SRC="$ROOT/vendor/musl"
SYSROOT="$ROOT/bsp/musl-gc-sysroot"
CLANG="clang"
CLANGXX="clang++"
LLVM_AR="llvm-ar"
LLVM_RANLIB="llvm-ranlib"

# 必须由命令行参数提供的项 (无默认值)
LLVM_SRC=""

usage() {
    cat >&2 <<EOF
用法: $0 -s <llvm-src> [-h]

    -s <path>   LLVM 源码树路径  (必填, 例: $ROOT/vendor/riscv-llvm-toolchain)
    -h          显示本帮助
EOF
    exit 1
}

while getopts 's:h' opt; do
    case "$opt" in
        s) LLVM_SRC="$OPTARG" ;;
        h) usage ;;
        *) usage ;;
    esac
done

if [ -z "$LLVM_SRC" ]; then
    usage
fi

for tool in "$CLANG" "$CLANGXX" cmake ninja "$LLVM_AR" "$LLVM_RANLIB"; do
    if ! command -v "$tool" >/dev/null 2>&1; then
        echo "错误: 缺少工具 $tool" >&2
        exit 1
    fi
done

# cmake 对 CMAKE_AR/CMAKE_RANLIB 不做 PATH 查找, 传裸名会按相对路径解析,
# 故此处解析为绝对路径供 cmake 使用。
LLVM_AR="$(command -v llvm-ar)"
LLVM_RANLIB="$(command -v llvm-ranlib)"

if [ ! -f "$SYSROOT/lib/libc.a" ]; then
    echo "错误: 缺少 $SYSROOT/lib/libc.a, 请先运行 bsp/build-musl-sysroot.sh" >&2
    exit 1
fi

MUSL_ISYSTEM="-isystem $MUSL_SRC/include \
-isystem $MUSL_SRC/arch/riscv64 \
-isystem $MUSL_SRC/arch/generic \
-isystem $MUSL_SRC/obj/include"

LINUX_UAPI="-isystem /usr/riscv64-linux-gnu/include"
ARCH_FLAGS="-march=rv64gc -mabi=lp64d"
C_FLAGS="$ARCH_FLAGS -nostdinc $MUSL_ISYSTEM $LINUX_UAPI"

BUILD_DIR="$LLVM_SRC/build-libcxx-musl"
rm -rf "$BUILD_DIR"

# libunwind 的 UnwindRegistersSave/Restore.S 属预处理汇编 (.S), cmake 走
# CMAKE_ASM_* 而非 CMAKE_C/CXX_*。若不单独指定 ASM 目标, clang 默认按 host
# (x86-64) 汇编, `__riscv` 未定义 -> .S 落到 x86-64 分支 -> 产物与
# elf64lriscv 不兼容, 链接报 "incompatible with elf64lriscv"。
cmake -G Ninja \
    -S "$LLVM_SRC/runtimes" \
    -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="$CLANG" \
    -DCMAKE_CXX_COMPILER="$CLANGXX" \
    -DCMAKE_C_COMPILER_TARGET=riscv64-linux-musl \
    -DCMAKE_CXX_COMPILER_TARGET=riscv64-linux-musl \
    -DCMAKE_ASM_COMPILER_TARGET=riscv64-linux-musl \
    -DCMAKE_ASM_COMPILER="$CLANG" \
    -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_SYSTEM_PROCESSOR=riscv64 \
    -DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY \
    -DCMAKE_C_FLAGS="$C_FLAGS" \
    -DCMAKE_CXX_FLAGS="$C_FLAGS" \
    -DCMAKE_ASM_FLAGS="$ARCH_FLAGS" \
    -DCMAKE_AR="$LLVM_AR" \
    -DCMAKE_RANLIB="$LLVM_RANLIB" \
    -DLLVM_ENABLE_RUNTIMES="libcxx;libcxxabi;libunwind" \
    -DLIBCXX_ENABLE_SHARED=OFF \
    -DLIBCXX_ENABLE_STATIC=ON \
    -DLIBCXX_ENABLE_STATIC_ABI_LIBRARY=ON \
    -DLIBCXX_ENABLE_ABI_LINKER_SCRIPT=OFF \
    -DLIBCXXABI_ENABLE_SHARED=OFF \
    -DLIBCXXABI_ENABLE_STATIC=ON \
    -DLIBUNWIND_ENABLE_SHARED=OFF \
    -DLIBUNWIND_ENABLE_STATIC=ON \
    -DLIBCXX_CXX_ABI=libcxxabi \
    -DLIBCXXABI_USE_LLVM_UNWINDER=ON \
    -DLIBUNWIND_IS_BAREMETAL=OFF \
    -DLIBCXX_HAS_MUSL_LIBC=ON \
    -DLIBCXX_INCLUDE_BENCHMARKS=OFF \
    -DLLVM_INCLUDE_TESTS=OFF \
    -DLIBCXX_INCLUDE_TESTS=OFF \
    -DLIBCXXABI_INCLUDE_TESTS=OFF \
    -DLIBUNWIND_INCLUDE_TESTS=OFF \
    -DCMAKE_INSTALL_PREFIX="$SYSROOT"

ninja -C "$BUILD_DIR" -j8
ninja -C "$BUILD_DIR" install

echo "已装入 C++ 标准库到: $SYSROOT"
ls -1 "$SYSROOT/lib"
