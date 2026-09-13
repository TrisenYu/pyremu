#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# — 生成全部 fn_apps 载荷的宿主参考输出 (golden reference)
#
# 载荷为 riscv64gc-musl 交叉编译产物, 无法在 x86-64 直接运行。本脚本以同源
# 原生目标重编译每个载荷, 运行并保存其完整 stdout 为参考文件:
#   build/host_ref_fn_apps/<name>.out
#
# 载荷内部采用固定随机种子 (ising 0x1234_5678_9ABC_DEF0 / dsp 0x5eed_2026_08_30),
# 其余载荷为确定性数值计算, 因此参考输出可复现, 作为模拟器 F/D 指令正确性的
# 黄金对照 (日后以模拟器跑同一载荷, diff 参考输出即可定位 FP 指令偏差)。
#
# 原生构建产物与参考输出统一落在项目 build/ 目录 (configs.mk 的 bins_dir),
# 不污染 cargo 默认的 fn_apps/target/ 与仓库根的 output/ 串口日志目录:
#   build/host_ref_fn_apps/native/          原生 x86-64 构建产物 (CARGO_TARGET_DIR + knots)
#   build/host_ref_fn_apps/<name>.out       各载荷参考输出
#
# 原生目标不触发 .cargo/config.toml 的 riscv64-musl 自定义链接器, 全程使用宿主
# 标准 gcc, 与厂商自定义指令集 (custom-llvm) 无关。
#
# 用法: ./[name] [载荷名 ...]   # 默认全部 7 个
#       载荷名: orbit chem graphene ising fem dsp knots
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
FNAPP="$REPO/fn_apps"
[ -d "$FNAPP" ] || { echo "错误: 找不到 fn_apps/ ($FNAPP)" >&2; exit 1; }
cd "$FNAPP"

RUST_PAYLOADS=(orbit chem graphene ising fem dsp)
C_PAYLOADS=(knots)
ALL=("${RUST_PAYLOADS[@]}" "${C_PAYLOADS[@]}")

BUILD_DIR="$REPO/build"
REF_DIR="$BUILD_DIR/host_ref_fn_apps"
NATIVE_DIR="$REF_DIR/native"

NATIVE_TARGET=x86_64-unknown-linux-gnu
RUST_BIN="$NATIVE_DIR/$NATIVE_TARGET/debug"
KNOTS_BIN="$NATIVE_DIR/knots_native"

# 目录名 -> cargo 包名 / 二进制名 (orbit 的包名与 bin 名为 orbit_payload, 其余同名)
pkg_of() {
    case "$1" in
        orbit) echo "orbit_payload" ;;
        *) echo "$1" ;;
    esac
}

# 选择载荷 (命令行参数覆盖默认全部)
if (($# > 0)); then
    SELECTED=("$@")
else
    SELECTED=("${ALL[@]}")
fi

# 1) 原生编译选中的 Rust 载荷 (CARGO_TARGET_DIR 重定向到 build/, 与交叉产物分目录)
cargo_args=()
for p in "${SELECTED[@]}"; do
    case " ${RUST_PAYLOADS[*]} " in
        *" $p "*) cargo_args+=(-p "$(pkg_of "$p")") ;;
    esac
done
if ((${#cargo_args[@]} > 0)); then
    echo ">> native build (Rust): ${cargo_args[*]}"
    CARGO_TARGET_DIR="$NATIVE_DIR" \
        cargo build --target "$NATIVE_TARGET" "${cargo_args[@]}" || exit 1
fi

# 2) 原生编译 knots (C + libhomfly)
if [[ " ${SELECTED[*]} " == *" knots "* ]]; then
    echo ">> native build (C): knots"
    mkdir -p "$(dirname "$KNOTS_BIN")"
    gcc -O2 -std=gnu11 -Iknots/deps/libhomfly \
        knots/src/main.c knots/deps/libhomfly/*.c -o "$KNOTS_BIN" || exit 1
fi

# 3) 逐个运行, 保存 stdout 到 build/host_ref_fn_apps/<name>.out
mkdir -p "$REF_DIR"
fail=0
for p in "${SELECTED[@]}"; do
    case "$p" in
        knots) bin="$KNOTS_BIN" ;;
        *) bin="$RUST_BIN/$(pkg_of "$p")" ;;
    esac
    ref="$REF_DIR/$p.out"
    echo ">> run: $p -> $ref"
    if "$bin" > "$ref" 2>&1; then
        rc=0
    else
        rc=$?
    fi
    if ((rc == 0)) && ! grep -q 'FAIL' "$ref"; then
        echo "   [PASS] $p ($(wc -c < "$ref") bytes)"
    else
        echo "   [FAIL] $p (rc=$rc 或输出含 FAIL)"
        fail=1
    fi
done

if ((fail)); then
    echo ">> 存在失败载荷, 参考文件不可信, 请检查 build/host_ref_fn_apps/native 与 cargo 报错"
    exit 1
fi
echo ">> 全部参考输出已保存至 $REF_DIR"
