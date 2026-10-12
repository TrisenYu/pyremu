# dune — DUNE 有限元求解二维 Poisson 方程
include ../config.mk

CLANGXX := clang++

BUILD := build
SRC   := poisson.cc
OUT   := $(BUILD)/dune_payload

# C++ 编译参数:
# 	-nostdinc++ 屏蔽x86编译环境默认提供的 libstdc++
#   并使用 sysroot 内 libc++
CXXFLAGS := $(ARCH) -std=c++20 -O2 -Wall \
  -nostdinc++ \
  -isystem $(MUSL_SYSROOT)/include/c++/v1 \
  -isystem $(MUSL_SYSROOT)/include \
  -isystem $(MUSL_SRC)/include \
  -isystem $(MUSL_SRC)/arch/riscv64 \
  -isystem $(MUSL_SRC)/arch/generic \
  -isystem $(MUSL_SRC)/obj/include

# DUNE 静态库, 依赖顺序为 uggrid、grid、geometry、common。
DUNE_LIBS := -lduneuggrid -ldunegrid -ldunegeometry -ldunecommon

all: $(OUT)

$(OUT): $(SRC)
	@mkdir -p $(@D)
	$(CLANGXX) --target=$(TARGET) \
		$(CXXFLAGS) $< $(LDFLAGS) -nodefaultlibs \
		$(DUNE_LIBS) -lc++ -lunwind -lc -lgcc -o $@

run: $(OUT)
	$(QEMU) $(OUT)

check: $(OUT)
	@$(QEMU) $(OUT) 2>&1 | grep -q 'PASS' \
	  && echo "PASS" || echo "FAIL"

clean:
	rm -rf $(BUILD)

.PHONY: all run check clean
