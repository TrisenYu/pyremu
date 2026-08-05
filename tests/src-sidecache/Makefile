# tests/src-sidecache/Makefile
# 缓存侧信道攻击 PoC 套件 - 交叉编译为 RISC-V 静态 ELF
#
# 两套编译链:
#   Linux U-mode   (riscv64-linux-gnu-gcc)  - flush_reload
#   Enclave musl   (clang + musl libc.a)    - hello_musl, victim_musl, probe_musl
#
# 目标:
#   make all                 全部编译
#   make flush_reload        仅编译 Flush+Reload 用户态演示
#   make musl                仅编译 musl 飞地 payload
#   make clean               清理

phony =

# 取消 GNU Make 内置的 %: %.c 隐式模式规则:
# 防止在源码目录直接生成无 bin/ 前缀的二进制 (如 tee_concurrent).
%: %.c

# ---- 交叉编译器 (Linux glibc) ----
CROSS_CC ?= $(shell command -v riscv64-linux-gnu-gcc 2>/dev/null || echo riscv64-unknown-linux-gnu-gcc)
CFLAGS  = -static -O2 -Wall

# ---- musl 飞地编译链 ----
MUSL_DIR   = $(abspath ../../bsp/musl-1.2.6)
MUSL_CC    = /opt/custom-llvm/bin/clang
MUSL_TARGET = riscv64-linux-musl
MUSL_ARCH   = rv64imac
LIBGCC      = /usr/lib/gcc/riscv64-unknown-elf/14.2.0/rv64imac/lp64/libgcc.a

MUSL_CFLAGS = --target=$(MUSL_TARGET) -march=$(MUSL_ARCH) \
              -static -nostdlib -nostdinc -O2 -Wall \
              -isystem $(MUSL_DIR)/include \
              -isystem $(MUSL_DIR)/obj/include

MUSL_LDLIBS = $(MUSL_DIR)/lib/crt1.o \
              $(MUSL_DIR)/lib/crti.o \
              $(MUSL_DIR)/lib/crtn.o \
              $(MUSL_DIR)/lib/libc.a \
              $(LIBGCC)

OUT_DIR    = bin
PROGRAMS   = flush_reload tee_test tee_concurrent
MUSL_PROGS = hello_payload victim_musl probe_musl tlb_leak_musl

BINS      = $(addprefix $(OUT_DIR)/, $(PROGRAMS))
MUSL_BINS = $(addprefix $(OUT_DIR)/, $(MUSL_PROGS))

# ---- 默认目标 (仅 Linux glibc 程序, musl payload 单独 make musl) ----
all: $(BINS)
phony += all

musl: $(MUSL_BINS)
phony += musl

# ---- Linux glibc 编译 ----
$(OUT_DIR)/flush_reload: flush_reload.c
	@mkdir -p $(@D)
	$(CROSS_CC) $(CFLAGS) $< -o $@

$(OUT_DIR)/%: %.c
	@mkdir -p $(@D)
	$(CROSS_CC) $(CFLAGS) $< -o $@

# ---- musl enclave payload 编译 ----
$(OUT_DIR)/hello_payload: hello_payload.c
	@mkdir -p $(@D)
	$(MUSL_CC) $(MUSL_CFLAGS) -fuse-ld=lld $(MUSL_LDLIBS) $< -o $@

$(OUT_DIR)/%_musl: %_payload.c
	@mkdir -p $(@D)
	$(MUSL_CC) $(MUSL_CFLAGS) -fuse-ld=lld $(MUSL_LDLIBS) $< -o $@

# ---- 清理 ----
clean:
	rm -rf $(OUT_DIR)
phony += clean

# ---- 帮助 ----
help:
	@echo "tests/src-sidecache"
	@echo "  make          build all"
	@echo "  make musl     build musl enclave payloads"
	@echo "  make clean    remove bin/"
	@echo "  Linux: $(PROGRAMS)"
	@echo "  Musl:  $(MUSL_PROGS)"
phony += help

.PHONY: $(phony)
