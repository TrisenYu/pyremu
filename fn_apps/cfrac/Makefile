# cfrac — Dave Barrett 的连分数分解程序 (continued fraction factorization)
#
# 编译器类内存负载: 大量短生命周期小对象 malloc/free。mimalloc-bench 官方基准,
# 因子分解 17545186520507317056371138836327483792789528 约产生 4300 万次
# malloc + free (100% <= 64B)。默认 (无参数) 分解一个较小的内置合数, 用于快速自检;
# 重度压测可改用更大的数。
#
# 源码为 1980 年代 K&R C: 函数定义把参数声明写在参数表之后 (如 `int main(argc, argv)
# int argc; char *argv[]; {`), 该写法自 C99 起不再合法, 故需 -std=gnu89。已排除
# pfactor.c (自带 main 的独立 trial-division demo) 与 getopt.c/getopt.h (改用 musl
# 自带 getopt, cfrac.c 直接 extern 声明)。
#
# musl 交叉工具链配置 (标准 clang + riscv64gc/lp64d 硬浮点) 见 ../config.mk。

include ../config.mk

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

# ---- 无参数启动的自检 ----
# 复现载荷以 argc=0, argv=NULL 启动的情形 (宿主经 TEE_IOC_ENTER 可不传参数).
# 载荷的 main 在此按 -Dmain=cfrac_main 改名, 由 test/zero_argv_main.c 调用.
# 断言: 打印内置合数的分解式, 退出码为 0, 不出现 usage 行.
# 修复前这一组入参落入 usage 分支, 并因 progName 取自 *argv 而打印 "(null)".
ARGV0_OUT  := $(BUILD)/cfrac_argv0
ARGV0_OBJS := $(addprefix $(BUILD)/argv0_, $(notdir $(SRCS:.c=.o))) \
              $(BUILD)/argv0_zero_argv_main.o

$(BUILD)/argv0_%.o: src/%.c
	@mkdir -p $(@D)
	$(CLANG) --target=$(TARGET) $(CFLAGS) -Dmain=cfrac_main -c $< -o $@

$(BUILD)/argv0_zero_argv_main.o: test/zero_argv_main.c
	@mkdir -p $(@D)
	$(CLANG) --target=$(TARGET) $(CFLAGS) -c $< -o $@

$(ARGV0_OUT): $(ARGV0_OBJS)
	@mkdir -p $(@D)
	$(CLANG) --target=$(TARGET) $(CFLAGS) $(ARGV0_OBJS) $(LDFLAGS) \
	  -nodefaultlibs -lc -lm -lgcc -o $@

check-argv0: $(ARGV0_OUT)
	@out=$$($(QEMU) $(ARGV0_OUT) 2>&1); rc=$$?; \
	if [ $$rc -ne 0 ]; then \
		echo "[FAIL] cfrac 无参数: 退出码 $$rc"; echo "$$out"; exit 1; \
	fi; \
	case "$$out" in \
	*"17545186520507317056371138836327483792736 = "*) ;; \
	*) echo "[FAIL] cfrac 无参数: 输出不是内置合数的分解式"; \
	   echo "$$out"; exit 1 ;; \
	esac; \
	echo "   [PASS] cfrac 无参数 (分解内置合数)"

clean:
	rm -rf $(BUILD)

.PHONY: all run check-argv0 clean
