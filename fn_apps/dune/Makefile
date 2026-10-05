# dune — DUNE 有限元求解二维 Poisson 方程 (数值偏微分) 载荷
#
# libc++ 载荷: 用 DUNE 2.11.0 的 YaspGrid + Q1 Lagrange 基 + ISTL CG
# 求解 -∇²u = 2π² sin(πx)sin(πy), 边界 u=0。自检 Q1 二阶收敛 (N=8 与 N=16
# 节点均方根误差之比 ≈ 4)
#
# 依赖: 8 个 DUNE 模块已交叉编译进 bsp/musl-gc-sysroot (libdune*.a + 头文件 +
# dune-*-config.hh), 见本目录下各 dune-*/ 源码树。本 Makefile 仅编译载荷本体,
# 不重建 DUNE (静态库已随 sysroot 落盘)。
#
# musl 交叉工具链配置见 ../config.mk; C++ 侧补充 libc++ 头与 DUNE 头搜索路径。

include ../config.mk

CLANGXX := clang++

BUILD := build
SRC   := poisson.cc
OUT   := $(BUILD)/dune_payload

# C++ 编译参数: -nostdinc++ 屏蔽宿主 libstdc++ 头, 改引 sysroot 内 libc++ 头;
# DUNE 头与配置头 dune-*-config.hh 均位于 $(MUSL_SYSROOT)/include。
CXXFLAGS := $(ARCH) -std=c++20 -O2 -Wall \
  -nostdinc++ \
  -isystem $(MUSL_SYSROOT)/include/c++/v1 \
  -isystem $(MUSL_SYSROOT)/include \
  -isystem $(MUSL_SRC)/include \
  -isystem $(MUSL_SRC)/arch/riscv64 \
  -isystem $(MUSL_SRC)/arch/generic \
  -isystem $(MUSL_SRC)/obj/include

# DUNE 静态库, 依赖序为 uggrid、grid、geometry、common。
DUNE_LIBS := -lduneuggrid -ldunegrid -ldunegeometry -ldunecommon

all: $(OUT)

# -nodefaultlibs 屏蔽 clang 隐式追加 -lgcc_eh; 显式列 DUNE 库与 C++ 运行库。
# 链接顺序须为 DUNE 库、libc++、libunwind、libc、libgcc, 因为 libc++ 引用后三者符号。
$(OUT): $(SRC)
	@mkdir -p $(@D)
	$(CLANGXX) --target=$(TARGET) $(CXXFLAGS) $< $(LDFLAGS) -nodefaultlibs \
	  $(DUNE_LIBS) -lc++ -lunwind -lc -lgcc -o $@

run: $(OUT)
	$(QEMU) $(OUT)

check: $(OUT)
	@$(QEMU) $(OUT) 2>&1 | grep -q '\[regress\] dune-poisson: PASS' \
	  && echo "dune poisson: PASS" || echo "dune poisson: FAIL"

clean:
	rm -rf $(BUILD)

.PHONY: all run check clean
