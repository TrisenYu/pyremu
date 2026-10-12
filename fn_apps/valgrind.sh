#!/usr/bin/env bash
# fn_apps/valgrind.sh - valgrind memcheck in x86-64
#
# 用法: ./valgrind.sh [载荷名 ...]
# 无参数执行则默认取所有载荷作为待测试项。
set -uo pipefail
cd "$(dirname "$0")"

RUST_PAYLOADS=(orbit chem graphene ising fem dsp linalg)
C_PAYLOADS=(knots json_calc)
CPP_PAYLOADS=(json_cpp)

NATIVE_TARGET=x86_64-unknown-linux-gnu
RUST_BIN=target/$NATIVE_TARGET/debug
KNOTS_BIN=target/valgrind/knots_native
JSON_BIN=target/valgrind/json_calc_native
JSON_CPP_BIN=target/valgrind/json_cpp_native
JSON_INC_DIR=target/valgrind/json_calc_inc
LOG_DIR=target/valgrind

pkg_of() {
    case "$1" in
	orbit)
		echo "orbit_payload";;
	*)
		echo "$1" ;;
    esac
}

VG_COMMON=(
	--leak-check=full
	--show-leak-kinds=definite,indirect
    --errors-for-leak-kinds=definite
	--track-origins=yes
	--error-exitcode=1
)

if (($# > 0)); then
    SELECTED=("$@")
else
    SELECTED=("${RUST_PAYLOADS[@]}" "${C_PAYLOADS[@]}" "${CPP_PAYLOADS[@]}")
fi

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

if [[ " ${SELECTED[*]} " == *" knots "* ]]; then
    echo ">> native build (C): knots"
    mkdir -p "$(dirname "$KNOTS_BIN")"
    gcc -O2 -g -std=gnu11 -Iknots/deps/libhomfly \
        knots/src/main.c knots/deps/libhomfly/*.c -o "$KNOTS_BIN" || exit 1
fi

# json_calc 与 json_cpp 共用内嵌输入json_config.inc, 故按需生成一次
if [[ " ${SELECTED[*]} " == *" json_calc "* || " ${SELECTED[*]} " == *" json_cpp "* ]]; then
    mkdir -p "$JSON_INC_DIR"
    python3 json_calc/scripts/embed_json.py \
        "$JSON_INC_DIR/json_config.inc" \
        json_calc/input/sqlparse_data.json || exit 1
fi
if [[ " ${SELECTED[*]} " == *" json_calc "* ]]; then
    echo ">> native build (C): json_calc"
    mkdir -p "$(dirname "$JSON_BIN")"
    gcc -O2 -g -std=gnu11 \
        -Ijson_calc/cjson -I"$JSON_INC_DIR" \
        json_calc/src/main.c json_calc/cjson/cJSON.c -lm -o "$JSON_BIN" || exit 1
fi
if [[ " ${SELECTED[*]} " == *" json_cpp "* ]]; then
    echo ">> native build (C++): json_cpp"
    mkdir -p "$(dirname "$JSON_CPP_BIN")"
    g++ -O2 -g -std=c++17 \
        -Ijson_calc -I"$JSON_INC_DIR" \
        json_calc/src/main_cpp.cpp -o "$JSON_CPP_BIN" || exit 1
fi

# 3) 逐个 valgrind memcheck
mkdir -p "$LOG_DIR"
fail=0
for p in "${SELECTED[@]}"; do
    case "$p" in
        knots) bin="$KNOTS_BIN" ;;
        json_calc) bin="$JSON_BIN" ;;
        json_cpp) bin="$JSON_CPP_BIN" ;;
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
    echo ">> something went wrong, see details in target/valgrind/*.valgrind.log"
    exit 1
fi
echo ">> valgrind memcheck pass"
