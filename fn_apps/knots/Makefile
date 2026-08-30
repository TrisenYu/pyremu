# 拓扑扭结载荷 (knots) — 交叉编译为 riscv64gc musl 静态 ELF
#
#   libhomfly — 公有领域 HOMFLY 实现
#
# musl 交叉工具链配置 (标准 clang + riscv64gc/lp64d 硬浮点) 见 ../config.mk。

include ../config.mk

HOMFLY_DIR := deps/libhomfly

BUILD := build

HOMFLY_SRCS := $(wildcard $(HOMFLY_DIR)/*.c)
HOMFLY_OBJS := $(addprefix $(BUILD)/, $(notdir $(HOMFLY_SRCS:.c=.o)))
HOMFLY_A    := $(BUILD)/libhomfly.a

OUT := $(BUILD)/knots_payload

# 追加 libhomfly 头目录
CFLAGS += -isystem $(HOMFLY_DIR)

# 追加本地 lib 目录 (供 libgcc_eh.a 符号解析)
LDFLAGS += -L$(BUILD)/lib

all: $(OUT)

# clang 驱动隐式追加 -lgcc_eh, 从 musl sysroot 复制 libgcc.a 改名为 libgcc_eh.a 满足之
$(BUILD)/lib/libgcc_eh.a: $(MUSL_LIB)/libgcc.a
	@mkdir -p $(@D)
	cp $< $@

$(BUILD)/%.o: $(HOMFLY_DIR)/%.c
	@mkdir -p $(@D)
	$(CLANG) --target=$(TARGET) $(CFLAGS) -c $< -o $@

$(HOMFLY_A): $(HOMFLY_OBJS)
	@mkdir -p $(@D)
	ar rcs $@ $(HOMFLY_OBJS)

$(OUT): src/main.c $(HOMFLY_A) $(BUILD)/lib/libgcc_eh.a
	@mkdir -p $(@D)
	$(CLANG) --target=$(TARGET) $(CFLAGS) src/main.c $(HOMFLY_A) $(LDFLAGS) -o $@

run: $(OUT)
	$(QEMU) $(OUT)

clean:
	rm -rf $(BUILD)

.PHONY: all run clean
