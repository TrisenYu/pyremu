# L 函数载荷 (Lfunc) — 交叉编译为 riscv64gc musl 静态 ELF
#
#   依赖 — FLINT (含 arb/acb 球算术(ball arithmetic)与 Dirichlet L 函数例程)、
#          MPFR、GMP, 由 ../build-deps.sh 从 vendor/{flint,mpfr,gmp} 子模块交叉编译,
#          产物位于 ../build/sysroot。
#
# musl 交叉工具链配置 (标准 clang + riscv64gc/lp64d 硬浮点) 见 ../../config.mk。

include ../../config.mk

# 依赖 sysroot; 可用 make DEPS_SYSROOT=/path 指向其它位置的构建产物
DEPS_SYSROOT ?= $(abspath ../build/sysroot)

BUILD := build

OUT := $(BUILD)/Lfunc_payload

# 追加依赖头目录与库目录。链接顺序: 用到符号者在前, FLINT 依赖 MPFR 与 GMP。
CFLAGS  += -isystem $(DEPS_SYSROOT)/include
LDFLAGS += -L$(DEPS_SYSROOT)/lib -L$(BUILD)/lib -lflint -lmpfr -lgmp -lm
# FLINT 的调试信息对载荷无用, 去掉以缩小载荷体积
LDFLAGS += -Wl,--strip-debug

all: $(OUT)

# clang 驱动隐式追加 -lgcc_eh, 从 musl sysroot 复制 libgcc.a 改名为 libgcc_eh.a 满足之
$(BUILD)/lib/libgcc_eh.a: $(MUSL_LIB)/libgcc.a
	@mkdir -p $(@D)
	cp $< $@

$(OUT): src/main.c $(BUILD)/lib/libgcc_eh.a
	@mkdir -p $(@D)
	$(CLANG) --target=$(TARGET) $(CFLAGS) src/main.c $(LDFLAGS) -o $@

run: $(OUT)
	$(QEMU) $(OUT)

clean:
	rm -rf $(BUILD)

.PHONY: all run clean
