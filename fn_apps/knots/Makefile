# 拓扑扭结载荷
include ../config.mk

HOMFLY_DIR := deps/libhomfly
BUILD := build

HOMFLY_SRCS := $(wildcard $(HOMFLY_DIR)/*.c)
HOMFLY_OBJS := $(addprefix $(BUILD)/, $(notdir $(HOMFLY_SRCS:.c=.o)))
HOMFLY_A    := $(BUILD)/libhomfly.a

OUT := $(BUILD)/knots_payload

# 补充头文件及动态链接库定义
CFLAGS += -isystem $(HOMFLY_DIR)
LDFLAGS += -L$(BUILD)/lib

all: $(OUT)

# clang 驱动隐式追加 -lgcc_eh,
# 从 musl sysroot 复制 libgcc.a 改名为 libgcc_eh.a
$(BUILD)/lib/libgcc_eh.a: $(MUSL_LIB)/libgcc.a
	@mkdir -p $(@D)
	cp $< $@

$(BUILD)/%.o: $(HOMFLY_DIR)/%.c
	@mkdir -p $(@D)
	$(CLANG) \
		--target=$(TARGET) \
		$(CFLAGS) -c $< \
		-o $@

$(HOMFLY_A): $(HOMFLY_OBJS)
	@mkdir -p $(@D)
	ar rcs $@ $(HOMFLY_OBJS)

$(OUT): src/main.c $(HOMFLY_A) $(BUILD)/lib/libgcc_eh.a
	@mkdir -p $(@D)
	$(CLANG) \
		--target=$(TARGET) \
		$(CFLAGS) src/main.c $(HOMFLY_A) \
		$(LDFLAGS) \
		-o $@

run: $(OUT)
	$(QEMU) $(OUT)

clean:
	rm -rf $(BUILD)

.PHONY: all run clean
