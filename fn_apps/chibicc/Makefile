# chibicc — Rui Ueyama 的自举 C 编译器 (递归下降解析器)
#
# 编译器类内存负载: tokenize/preprocess/parse 阶段为每个 token/AST 节点/类型调用
# calloc(1, sizeof(...)), 生成大量小对象且长期存活 (chibicc 设计上不释放, 靠进程退出
# 回收)。与 cfrac (短生命周期 malloc/free 翻腾) 互补, 提供"多小对象长期驻留"的
# 堆增长负载, 适合压测飞地 brk 按需扩容。
#
# 关键适配:
#   - chibicc 是 gcc 式"驱动 + -cc1 子进程"架构, 驱动默认 fork/exec 自调用 -cc1,
#     飞地无进程模型不可用。改为直接 `-cc1` 模式 (main.c 的 cc1() 无子进程),
#     经 stdin 读源、stdout 写 x86-64 汇编。
#   - 自检源 test/inp_src.c 不含 #include 依赖: 飞地内经 compound payload 把该源注入
#     ramfs (见 tee_ecall_regress.c 的 test_chibicc), 只有源文件本身可用, 无系统头文件。
#   - 输出为 x86-64 汇编 (chibicc 默认目标), 仅用于验证解析链路, 不实际汇编执行。
#
# musl 交叉工具链配置 (标准 clang + riscv64gc/lp64d 硬浮点) 见 ../config.mk。

include ../config.mk

# chibicc 上游 CFLAGS 去掉 -g; -fno-common 保持 C11 语义; -Wno-switch 屏蔽
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

# 无需 -lm: 编译器本体不调用 libm (浮点常量解析用 strtod, 属 libc)。
$(OUT): $(OBJS)
	@mkdir -p $(@D)
	$(CLANG) --target=$(TARGET) $(CFLAGS) $(OBJS) $(LDFLAGS) \
	  -nodefaultlibs -lc -lgcc -o $@

# -cc1 直接编译 (免 fork/exec); stdin 注入自检源, stdout 输出 x86-64 汇编
run: $(OUT)
	$(QEMU) $(OUT) -cc1 -x c -cc1-input - -cc1-output - - < test/inp_src.c

check: $(OUT)
	@$(QEMU) $(OUT) -cc1 -x c -cc1-input - -cc1-output - < test/inp_src.c 2>&1 \
	  | grep -q '.globl main' && echo "chibicc selfhost: PASS" || echo "chibicc selfhost: FAIL"

clean:
	rm -rf $(BUILD)

.PHONY: all run check clean
