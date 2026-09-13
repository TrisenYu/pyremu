# cfrac — Dave Barrett 的连分数分解程序 (continued fraction factorization)
#
# 编译器类内存负载: 大量短生命周期小对象 malloc/free。mimalloc-bench 官方基准,
# 因子分解 17545186520507317056371138836327483792789528 约产生 4300 万次
# malloc + free (100% <= 64B)。默认 (无参数) 分解一个较小的内置合数, 用于快速自检;
# 重度压测可改用更大的数。
#
# 源码为 1980 年代 K&R C (老式函数定义, 如 `int main(argc, argv) char *argv[];`),
# 需 -std=gnu89; 已排除 pfactor.c (自带 main 的独立 trial-division demo) 与
# getopt.c/getopt.h (改用 musl 自带 getopt, cfrac.c 直接 extern 声明)。
#
# musl 交叉工具链配置 (标准 clang + riscv64gc/lp64d 硬浮点) 见 ../config.mk。

include ../config.mk

# K&R C: gnu89 支持老式函数定义; -w 屏蔽 deprecated-non-prototype 等海量告警
# (config.mk 默认 -Wall 对这类 80 年代代码毫无意义)。
CFLAGS += -std=gnu89 -w

BUILD := build
SRCS  := $(wildcard src/*.c)
OBJS  := $(addprefix $(BUILD)/, $(notdir $(SRCS:.c=.o)))
OUT   := $(BUILD)/cfrac_payload

all: $(OUT)

$(BUILD)/%.o: src/%.c
	@mkdir -p $(@D)
	$(CLANG) --target=$(TARGET) $(CFLAGS) -c $< -o $@

# -nodefaultlibs 屏蔽 clang 对 musl 目标隐式追加的 -lgcc_eh (sysroot 未提供该库),
# 显式列运行库; -lm 供 findk/pomeranceLpow 的 log/exp/sqrt。
$(OUT): $(OBJS)
	@mkdir -p $(@D)
	$(CLANG) --target=$(TARGET) $(CFLAGS) $(OBJS) $(LDFLAGS) \
	  -nodefaultlibs -lc -lm -lgcc -o $@

run: $(OUT)
	$(QEMU) $(OUT)

clean:
	rm -rf $(BUILD)

.PHONY: all run clean
