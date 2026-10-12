include ../config.mk

# CFLAGS 去掉 -g; -fno-common 保持 C11 语义; -Wno-switch 屏蔽
# codegen.c 大量"enum 未穷举"的故意告警。
CFLAGS += -std=c11 -fno-common -Wno-switch

BUILD := build
SRCS  := $(wildcard src/*.c)
OBJS  := $(addprefix $(BUILD)/, $(notdir $(SRCS:.c=.o)))
OUT   := $(BUILD)/chibicc_payload

all: $(OUT)

$(BUILD)/%.o: src/%.c
	@mkdir -p $(@D)
	$(CLANG) --target=$(TARGET) $(CFLAGS) -c $< -o $@

# 无需 -lm: 编译器本体不调用 libm
$(OUT): $(OBJS)
	@mkdir -p $(@D)
	$(CLANG) --target=$(TARGET) $(CFLAGS) $(OBJS) $(LDFLAGS) \
	  -nodefaultlibs -lc -lgcc -o $@

# -cc1 直接编译; stdin 注入自检源, stdout 输出 x86-64 汇编
run: $(OUT)
	$(QEMU) $(OUT) -cc1 -x c -cc1-input - -cc1-output - - < test/inp_src.c

check: $(OUT)
	@$(QEMU) $(OUT) -cc1 -x c -cc1-input - -cc1-output - < test/inp_src.c 2>&1 \
	  | grep -q '.globl main' && echo "chibicc selfhost: PASS" || echo "chibicc selfhost: FAIL"

clean:
	rm -rf $(BUILD)

.PHONY: all run check clean
