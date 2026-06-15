# rv8-bench — RISC-V (及 host) 基准测试套件
# https://github.com/larkmjc/rv8-bench
#
# 交叉编译 (默认, RISC-V):
#   make
#
# 本地编译 (x86/ARM host, 功能验证):
#   make TARGET=host
#   make TARGET=host CC=clang CXX=clang++ STRIP=llvm-strip

phony =

CFLAGS  = -fPIE -g -MD -MP
LDFLAGS = -static

# ---- 目标选择 ----
# TARGET=riscv64 → 交叉编译, 输出到 bin/riscv64/
# TARGET=host    → 本地编译, 输出到 bin/native-isa/
TARGET ?= riscv64

ifeq ($(TARGET),host)
	OUT_DIR = bin/native-isa
	CC      = gcc
	CXX     = g++
	STRIP   = strip
else
	OUT_DIR = bin/riscv64
	CC      = /opt/custom-llvm/bin/clang
	CXX     = /opt/custom-llvm/bin/clang++
	STRIP   = /opt/custom-llvm/bin/llvm-strip
endif

PROGRAMS   = aes bigint dhrystone miniz norx primes qsort sha512
OPT_LEVELS = O3 O2 Os

PROGS = $(addprefix $(OUT_DIR)/, $(PROGRAMS))

# 所有目标: <out>/<prog>_O3.o  <out>/<prog>_O3_stripped.o  ...
O_PROGS = $(foreach opt,$(OPT_LEVELS), \
            $(addsuffix _$(opt).o, $(PROGS)) \
            $(addsuffix _$(opt)_stripped.o, $(PROGS)))

# 自动依赖: -MD -MP 为每次编译生成 .d 文件, 追踪 #include 的头文件
D_FILES = $(foreach opt,$(OPT_LEVELS),$(addsuffix _$(opt).d, $(PROGS)))
-include $(D_FILES)

all: $(O_PROGS)
phony += all

npm:
	npm install
phony += npm

help:
	@echo "rv8-bench — RISC-V benchmark suite"
	@echo ""
	@echo "  make                    = make all     (cross-compile → bin/riscv64/)"
	@echo "  make all TARGET=host                   (native compile → bin/native-isa/)"
	@echo "  make TARGET=host CC=clang CXX=clang++  (native with clang)"
	@echo "  make bin/riscv64/aes_O3.o              (single program)"
	@echo "  make help                              (this help)"
	@echo ""
	@echo "  targets:  $(PROGRAMS)"
	@echo "  opt:      $(OPT_LEVELS)"
phony += help

clean:
	rm -f $(O_PROGS) $(D_FILES)
phony += clean

# ---- 编译 + strip 规则: $(eval) 展开 O3/O2/Os × .c/.cc (1 模板 → 每个 opt 3 规则) ----
define compile_opt
$(OUT_DIR)/%_$(1).o: %.c
	@mkdir -p $$(@D)
	$$(CC) $$(LDFLAGS) -$(1) $$(CFLAGS) $$< -o $$@

$(OUT_DIR)/%_$(1).o: %.cc
	@mkdir -p $$(@D)
	$$(CXX) $$(LDFLAGS) -$(1) $$(CFLAGS) $$< -o $$@

$(OUT_DIR)/%_$(1)_stripped.o: $(OUT_DIR)/%_$(1).o
	@mkdir -p $$(@D)
	$$(STRIP) --strip-all $$< -o $$@
endef

$(foreach opt,$(OPT_LEVELS),$(eval $(call compile_opt,$(opt))))

.PHONY: $(phony)
