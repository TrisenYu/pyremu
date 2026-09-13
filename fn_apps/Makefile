# fn_apps/Makefile — 统一构建所有飞地载荷 (可信应用 payload)
#
# 职责:
#   1. 检测 bsp/musl-gc-sysroot, 缺失时自动构建 (依赖 vendor/musl 子模块)
#   2. 逐个构建 Rust 载荷 (cargo, riscv64gc-unknown-linux-musl 静态)、C 载荷
#      (knots/cfrac/chibicc/json_calc) 与 C++ 载荷 (dune/json_cpp)
#   3. 收集最终静态 ELF 到 bin/, 供可信应用 (飞地) 加载
#
# 用法:
#   make              # 构建全部载荷到 bin/
#   make <name>       # 仅构建单个 (orbit/chem/graphene/ising/fem/dsp/linalg/knots/cfrac/chibicc/json_calc/dune/json_cpp)
#   make check        # qemu-riscv64 下逐个运行自检, 输出含 FAIL 即报错
#   make clean        # 清理 cargo 产物与 bin/ (保留 sysroot)
#
# 依赖: vendor/musl 子模块 + clang/llvm-ar/llvm-ranlib (见 bsp/build-musl-sysroot.sh)

include config.mk

CARGO   := cargo
TARGET  := riscv64gc-unknown-linux-musl
PROFILE ?= release

# cargo 的 --release 开关 (debug 无对应开关, 故置空)
ifeq ($(PROFILE),release)
CARGO_RELEASE := --release
else
CARGO_RELEASE :=
endif

# 目录名 -> cargo 包名 (orbit 的包名为 orbit_payload)
pkg_of = $(if $(filter orbit,$(1)),orbit_payload,$(1))

RUST_PAYLOADS := orbit chem graphene ising fem dsp linalg
# malloc_churn (t-test1) 为多线程 C 载荷, check 目标需传并发参数, 故单列一条规则
# (见下); 归类上仍属 C 载荷.
C_PAYLOADS   := knots cfrac chibicc json_calc malloc_churn
# json_cpp 与 json_calc 同一源码目录 (共用输入集与内嵌 json_config.inc), 仅实现语言不同
CPP_PAYLOADS := dune json_cpp
ALL_PAYLOADS := $(RUST_PAYLOADS) $(C_PAYLOADS) $(CPP_PAYLOADS)

BIN := $(abspath bin)

.PHONY: all check clean $(ALL_PAYLOADS)

all: $(ALL_PAYLOADS)

# ---- sysroot: 缺失时自动构建 ----
$(MUSL_LIB)/libc.a:
	@test -f "$(MUSL_SRC)/Makefile" || { \
		echo "错误: 缺少 vendor/musl 子模块, 请先执行: git submodule update --init vendor/musl"; \
		exit 1; }
	@echo ">> 缺少 musl-gc-sysroot, 开始构建 ..."
	"$(REPO)/bsp/build-musl-sysroot.sh"

# ---- Rust 载荷: cargo 构建 + 收集到 bin/ ----
define rust_rule
$(1): $(MUSL_LIB)/libc.a
	$(CARGO) build $(CARGO_RELEASE) -p $(call pkg_of,$(1))
	@mkdir -p $(BIN)
	cp target/$(TARGET)/$(PROFILE)/$(call pkg_of,$(1)) $(BIN)/$(1)
	@echo "  -> $(BIN)/$(1)"
endef
$(foreach p,$(RUST_PAYLOADS),$(eval $(call rust_rule,$(p))))

# ---- C 载荷: 委托各子目录 Makefile ----
knots: $(MUSL_LIB)/libc.a
	$(MAKE) -C knots
	@mkdir -p $(BIN)
	cp knots/build/knots_payload $(BIN)/knots
	@echo "  -> $(BIN)/knots"

cfrac: $(MUSL_LIB)/libc.a
	$(MAKE) -C cfrac
	@mkdir -p $(BIN)
	cp cfrac/build/cfrac_payload $(BIN)/cfrac
	@echo "  -> $(BIN)/cfrac"

chibicc: $(MUSL_LIB)/libc.a
	$(MAKE) -C chibicc
	@mkdir -p $(BIN)
	cp chibicc/build/chibicc_payload $(BIN)/chibicc
	@echo "  -> $(BIN)/chibicc"

# 两个载荷各自独立: 只构建所用实现, 避免一方编译失败牵连另一方
json_calc: $(MUSL_LIB)/libc.a
	$(MAKE) -C json_calc build/json_calc_payload
	@mkdir -p $(BIN)
	cp json_calc/build/json_calc_payload $(BIN)/json_calc
	@echo "  -> $(BIN)/json_calc"

# json_cpp: 同目录的 nlohmann/json 实现, 与 json_calc 对拍同一输入集
json_cpp: $(MUSL_LIB)/libc.a
	$(MAKE) -C json_calc build/json_cpp_payload
	@mkdir -p $(BIN)
	cp json_calc/build/json_cpp_payload $(BIN)/json_cpp
	@echo "  -> $(BIN)/json_cpp"

# malloc_churn: Wolfram Gloger 多线程 malloc 压力测试 (t-test1, 单 hart 抢占式
# clone/futex 的验证载荷). 源码目录名为 malloc-churn, 产物名与载荷名一致.
malloc_churn: $(MUSL_LIB)/libc.a
	$(MAKE) -C malloc-churn
	@mkdir -p $(BIN)
	cp malloc-churn/build/malloc_churn $(BIN)/malloc_churn
	@echo "  -> $(BIN)/malloc_churn"

# C++ 载荷: DUNE FEM (依赖 sysroot 内 libdune*.a, 见 dune/Makefile)
dune: $(MUSL_LIB)/libc.a
	$(MAKE) -C dune
	@mkdir -p $(BIN)
	cp dune/build/dune_payload $(BIN)/dune
	@echo "  -> $(BIN)/dune"

# ---- 自检: qemu 逐个运行, 输出含 FAIL 即失败 ----
check: $(ALL_PAYLOADS)
	@fail=0; \
	for p in $(ALL_PAYLOADS); do \
		echo "当前测试项 $$p"; \
		ok=1; \
		if [ "$$p" = "chibicc" ]; then \
			$(QEMU) $(BIN)/$$p -cc1 -x c -cc1-input - -cc1-output - \
				< chibicc/test/inp_src.c 2>&1 | grep -q '.globl main' || ok=0; \
		elif [ "$$p" = "malloc_churn" ]; then \
			$(QEMU) $(BIN)/$$p 6 2 1000 1000 32 2>&1 | grep -q 'Done\.' || ok=0; \
		else \
			$(QEMU) $(BIN)/$$p 2>&1 | grep -q 'FAIL' && ok=0; \
		fi; \
		if [ $$ok -eq 1 ]; then \
			echo "   [PASS] $$p"; \
		else \
			echo "   [FAIL] $$p"; fail=1; \
		fi; \
	done; \
	exit $$fail

clean:
	$(CARGO) clean
	$(MAKE) -C knots clean
	$(MAKE) -C json_calc clean
	$(MAKE) -C malloc-churn clean
	rm -rf $(BIN)
