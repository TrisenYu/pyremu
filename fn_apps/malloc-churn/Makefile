# malloc-churn — 频繁交织 malloc/free 的内存分配压力载荷
#
#   源码: src/t-test1.c — Wolfram Gloger (1996-2004) 的多线程 malloc 压力测试,
#   逐字节取自 samhain HIDS 源码仓库 (trac.la-samhna.de, trunk/src/t-test1.c,
#   主干修订版), 仅含一处 samhain 为链接其日志器而添加的 safe_fatal 空桩 (未被调用)。
#
#   工作负载: 每个工作线程维护 bins 个保留分配池, 以随机尺寸 (1..size-max) 在
#   malloc / calloc / realloc / memalign 四种策略间随机切换, 并以随机 (非 LIFO)
#   顺序释放; 每 N_TOTAL_PRINT 个线程打印一轮 malloc/memalign/realloc/calloc 累计计数。
#
# musl 交叉工具链配置 (标准 clang + riscv64gc/lp64d 硬浮点) 见 ../config.mk。

include ../config.mk

BUILD := build

OUT := $(BUILD)/malloc_churn

# 追加本地 lib 目录 (供 libgcc_eh.a 符号解析)
LDFLAGS += -L$(BUILD)/lib

all: $(OUT)

# clang 驱动隐式追加 -lgcc_eh, 从 musl sysroot 复制 libgcc.a 改名为 libgcc_eh.a 满足之
$(BUILD)/lib/libgcc_eh.a: $(MUSL_LIB)/libgcc.a
	@mkdir -p $(@D)
	cp $< $@

$(OUT): src/t-test1.c $(BUILD)/lib/libgcc_eh.a
	@mkdir -p $(@D)
	$(CLANG) --target=$(TARGET) $(CFLAGS) src/t-test1.c $(LDFLAGS) -o $@

run: $(OUT)
	$(QEMU) $(OUT) 100 2 10000 10000 32

clean:
	rm -rf $(BUILD)

.PHONY: all run clean
