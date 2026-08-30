#!/usr/bin/env bash
# fn_apps/valgrind.sh — 宿主侧 (x86-64) valgrind memcheck
#
# 载荷为 riscv64gc-musl 交叉编译产物, 无法直接在 x86-64 上运行 valgrind。
# 本脚本以同源原生目标重编译每个载荷, 再用 valgrind memcheck 检查内存错误:
#   非法读写 (invalid read/write) / 使用已释放内存 (use-after-free) /
#   未初始化值读取 (uninitialised value) / 确定泄漏 (definitely lost)。
#
# Rust 载荷无 unsafe, 预期天然干净; 价值重点在 C 载荷 (knots/libhomfly) 的手工内存管理。
#
# 用法: ./valgrind.sh [载荷名 ...]    # 默认全部
set -uo pipefail
cd "$(dirname "$0")"

RUST_PAYLOADS=(orbit chem graphene ising fem dsp)
C_PAYLOADS=(knots)

NATIVE_TARGET=x86_64-unknown-linux-gnu
RUST_BIN=target/$NATIVE_TARGET/debug
KNOTS_BIN=target/valgrind/knots_native
LOG_DIR=target/valgrind

# 目录名 -> cargo 包名 / 二进制名 (多数一致; orbit 的包名与 bin 名为 orbit_payload)
pkg_of() {
    case "$1" in
        orbit) echo "orbit_payload" ;;
        *) echo "$1" ;;
    esac
}

# 公共 valgrind 参数: 泄漏视为错误, 追踪未初始化值来源
VG_COMMON=(--leak-check=full --show-leak-kinds=definite,indirect
    --errors-for-leak-kinds=definite --track-origins=yes --error-exitcode=1)

# 选择载荷 (命令行参数覆盖默认全部)
if (($# > 0)); then
    SELECTED=("$@")
else
    SELECTED=("${RUST_PAYLOADS[@]}" "${C_PAYLOADS[@]}")
fi

# 1) 原生编译选中的 Rust 载荷 (与交叉产物分目录, 不互相覆盖)
cargo_args=()
for p in "${SELECTED[@]}"; do
    case " ${RUST_PAYLOADS[*]} " in
        *" $p "*) cargo_args+=(-p "$(pkg_of "$p")") ;;
    esac
done
if ((${#cargo_args[@]} > 0)); then
    echo ">> native build (Rust): ${cargo_args[*]}"
    cargo build --target "$NATIVE_TARGET" "${cargo_args[@]}" || exit 1
fi

# 2) 原生编译 knots (C + libhomfly)
if [[ " ${SELECTED[*]} " == *" knots "* ]]; then
    echo ">> native build (C): knots"
    mkdir -p "$(dirname "$KNOTS_BIN")"
    gcc -O2 -g -std=gnu11 -Iknots/deps/libhomfly \
        knots/src/main.c knots/deps/libhomfly/*.c -o "$KNOTS_BIN" || exit 1
fi

# 3) 逐个 valgrind memcheck
mkdir -p "$LOG_DIR"
fail=0
for p in "${SELECTED[@]}"; do
    case "$p" in
        knots) bin="$KNOTS_BIN" ;;
        *) bin="$RUST_BIN/$(pkg_of "$p")" ;;
    esac
    log="$LOG_DIR/$p.valgrind.log"
    echo ">> valgrind: $p"
    if valgrind --log-file="$log" "${VG_COMMON[@]}" "$bin" >/dev/null 2>&1; then
        echo "   [PASS] $p"
        rm -f "$log"
    else
        echo "   [FAIL] $p  (report: $log)"
        fail=1
    fi
done

if ((fail)); then
    echo ">> 存在内存问题, 详见 target/valgrind/*.valgrind.log"
    exit 1
fi
echo ">> 全部载荷通过 valgrind memcheck"
