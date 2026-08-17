# tests/src-stress/Makefile
# TEE 飞地压力测试 - 批量创建/执行/销毁飞地
#
# 两套编译链:
#   Linux glibc (riscv64-linux-gnu-gcc) - tee_stress (测试控制程序)
#   Enclave musl (clang + musl libc.a)   - stress_payload (飞地内载荷)
#
# 目标:
#   make all              全部编译
#   make clean            清理

phony =

# 取消 GNU Make 内置的 %: %.c 隐式模式规则
%: %.c

# ---- 交叉编译器 (Linux glibc) ----
CROSS_CC  ?= $(shell command -v riscv64-linux-gnu-gcc 2>/dev/null || echo riscv64-unknown-linux-gnu-gcc)
CFLAGS     = -static -O2 -Wall

# ---- musl 飞地编译链 ----
MUSL_DIR    = $(abspath ../../bsp/musl-1.2.6)
MUSL_CC     = /opt/custom-llvm/bin/clang
MUSL_TARGET = riscv64-linux-musl
MUSL_ARCH   = rv64imac
LIBGCC      = /usr/lib/gcc/riscv64-unknown-elf/14.2.0/rv64imac/lp64/libgcc.a

MUSL_CFLAGS = --target=$(MUSL_TARGET) -march=$(MUSL_ARCH) \
              -static -nostdlib -nostdinc -O2 -Wall \
              -isystem $(MUSL_DIR)/arch/riscv64 \
              -isystem $(MUSL_DIR)/arch/generic \
              -isystem $(MUSL_DIR)/obj/include \
              -isystem $(MUSL_DIR)/include

MUSL_LDLIBS = $(MUSL_DIR)/lib/crt1.o \
              $(MUSL_DIR)/lib/crti.o \
              $(MUSL_DIR)/lib/crtn.o \
              $(MUSL_DIR)/lib/libc.a \
              $(LIBGCC)

OUT_DIR    = bin
PROGRAMS   = tee_stress
PAYLOADS   = stress_payload

BINS       = $(addprefix $(OUT_DIR)/, $(PROGRAMS))
PAY_BINS   = $(addprefix $(OUT_DIR)/, $(PAYLOADS))

# ---- 默认目标 ----
all: $(BINS) $(PAY_BINS)
phony += all

# ---- Linux glibc 编译 ----
$(OUT_DIR)/%: %.c tee_enclave.h
	@mkdir -p $(@D)
	$(CROSS_CC) $(CFLAGS) $< -o $@

# ---- musl enclave payload 编译 ----
$(OUT_DIR)/stress_payload: stress_payload.c stress_common.h
	@mkdir -p $(@D)
	$(MUSL_CC) $(MUSL_CFLAGS) -fuse-ld=lld $(MUSL_LDLIBS) $< -o $@

# ---- 清理 ----
clean:
	rm -rf $(OUT_DIR)
phony += clean

.PHONY: $(phony)
