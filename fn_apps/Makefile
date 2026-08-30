# fn_apps/Makefile — 统一构建所有飞地载荷 (可信应用 payload)
#
# 职责:
#   1. 检测 bsp/musl-gc-sysroot, 缺失时自动构建 (依赖 vendor/musl 子模块)
#   2. 逐个构建 Rust 载荷 (cargo, riscv64gc-unknown-linux-musl 静态) 与 C 载荷 (knots)
#   3. 收集最终静态 ELF 到 bin/, 供可信应用 (飞地) 加载
#
# 用法:
#   make              # 构建全部载荷到 bin/
#   make <name>       # 仅构建单个 (orbit/chem/graphene/ising/fem/dsp/knots)
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

RUST_PAYLOADS := orbit chem graphene ising fem dsp
C_PAYLOADS   := knots
ALL_PAYLOADS := $(RUST_PAYLOADS) $(C_PAYLOADS)

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

# ---- C 载荷: 委托 knots/Makefile ----
knots: $(MUSL_LIB)/libc.a
	$(MAKE) -C knots
	@mkdir -p $(BIN)
	cp knots/build/knots_payload $(BIN)/knots
	@echo "  -> $(BIN)/knots"

# ---- 自检: qemu 逐个运行, 输出含 FAIL 即失败 ----
check: $(ALL_PAYLOADS)
	@fail=0; \
	for p in $(ALL_PAYLOADS); do \
		echo ">> 自检 $$p"; \
		if $(QEMU) $(BIN)/$$p 2>&1 | grep -q 'FAIL'; then \
			echo "   [FAIL] $$p"; fail=1; \
		else \
			echo "   [PASS] $$p"; \
		fi; \
	done; \
	exit $$fail

clean:
	$(CARGO) clean
	$(MAKE) -C knots clean
	rm -rf $(BIN)
