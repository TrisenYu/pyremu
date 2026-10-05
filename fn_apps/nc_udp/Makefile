# nc_udp — 以 busybox 的 nc applet 在 UDP 模式下自收自发的套接字通路载荷
#
# 载荷由两部分构成: vendor/busybox 的 nc (上游源码未改) 与 src/main.c 的启动外壳,
# 外壳的作用见 src/main.c 的文件头。
#
# busybox 在源码树外构建 (O=$(BB_BUILD)), 子模块的工作区保持干净。搬入载荷时另有三处
# 处理:
#   1. libbb/lib.a 中 appletlib.o 的 main 改名为 bb_main。外壳要定义 main, 而 busybox
#      的入口与它同名; 改名后两者各归其名, nc 仍由 busybox 自己的 applet 分发进入。
#   2. 补齐 libgcc_eh.a。clang 为目标链接隐式追加 -lgcc_eh, musl sysroot 只提供
#      libgcc.a, 以同名副本补齐, 与其余载荷 Makefile 的做法一致。
#   3. 以 llvm-strip 收尾。busybox 构建末尾对 busybox_unstripped 执行 strip, 主机的
#      strip 是 x86_64 版, 认不出 riscv64 目标文件。
#
# busybox 的配置种子为同目录 busybox.config。
#
# musl 交叉工具链配置 (标准 clang + riscv64gc/lp64d 硬浮点) 见 ../config.mk。

include ../config.mk

BB_SRC   := $(REPO)/vendor/busybox
BB_BUILD := $(abspath build/busybox)
BUILD    := build
SHIM     := $(abspath $(BUILD)/lib)

JOBS ?= $(shell nproc)

OUT := $(BUILD)/nc_udp_payload

# 自检所用端口。取 39555 而非载荷默认的 5555, 避免与主机上已占用的端口相撞。
PORT ?= 39555

# busybox 的目标编译器。busybox 自行追加 -Oz、-static 等参数, 故此处只给目标三元组、
# 体系结构、链接器与头文件/库的搜索路径。
BB_INCLUDES := -isystem $(MUSL_SRC)/include \
	-isystem $(MUSL_SRC)/arch/riscv64 \
	-isystem $(MUSL_SRC)/arch/generic \
	-isystem $(MUSL_SRC)/obj/include
BB_CC := $(CLANG) --target=$(TARGET) $(ARCH) -fuse-ld=lld -nostdinc \
	$(BB_INCLUDES) -B$(MUSL_LIB) -L$(SHIM) -L$(MUSL_LIB)

# busybox 的收尾步骤。busybox 的 Makefile 内定为 $(CROSS_COMPILE)strip, 即主机的 strip。
STRIP ?= llvm-strip

# busybox 各子目录的静态库与 built-in.o。声明为递归变量, 使通配在 busybox 构建完成之后
# 展开; 库中已含 nc.o, 由 busybox 的入口经 nc_main 引用。libbb 的另一份副本
# (main 已改名) 取代原库。
BB_ARCHIVES = $(filter-out $(BB_BUILD)/libbb/lib.a, \
		$(wildcard $(BB_BUILD)/*/lib.a) $(wildcard $(BB_BUILD)/*/*/lib.a)) \
	$(wildcard $(BB_BUILD)/*/built-in.o) $(wildcard $(BB_BUILD)/*/*/built-in.o) \
	$(SHIM)/libbb_bb_main.a

all: $(OUT)

# clang 驱动隐式追加 -lgcc_eh, 从 musl sysroot 复制 libgcc.a 改名为 libgcc_eh.a 满足之
$(SHIM)/libgcc_eh.a: $(MUSL_LIB)/libgcc.a
	@mkdir -p $(@D)
	cp $< $@

# ---- busybox: 源码树外构建 ----
# 该构建以最终的 busybox_unstripped 收尾, 故先备好 libgcc_eh.a。
$(BB_BUILD)/.config: busybox.config
	@test -f "$(BB_SRC)/Makefile" || { \
		echo "错误: 缺少 vendor/busybox 子模块, 请先执行: git submodule update --init vendor/busybox"; \
		exit 1; }
	@mkdir -p $(@D)
	cp $< $@

$(BB_BUILD)/libbb/lib.a: $(BB_BUILD)/.config $(SHIM)/libgcc_eh.a
	$(MAKE) -C $(BB_SRC) O=$(BB_BUILD) CC='$(BB_CC)' STRIP='$(STRIP)' -j$(JOBS)

# busybox 入口的改名副本
$(SHIM)/libbb_bb_main.a: $(BB_BUILD)/libbb/lib.a
	@mkdir -p $(@D)
	llvm-objcopy --redefine-sym main=bb_main $< $@

# ---- 载荷: 外壳 + busybox 全部归档 ----
$(OUT): src/main.c $(SHIM)/libbb_bb_main.a $(SHIM)/libgcc_eh.a
	@mkdir -p $(@D)
	$(CLANG) --target=$(TARGET) $(CFLAGS) src/main.c $(LDFLAGS) -L$(SHIM) \
	  -Wl,--start-group $(BB_ARCHIVES) -Wl,--end-group -o $@

run: $(OUT)
	$(QEMU) $(OUT) 127.0.0.1 $(PORT)

# 自检: 主机上的往返。标记只能经 nc 的接收方向写出, 它出现即表示发送、回环与接收三程
# 都成立。载荷按相对路径读写标记文件, 故在构建目录中运行。
check: $(OUT)
	@rm -f $(BUILD)/udp_msg
	@out=$$(cd $(BUILD) && $(QEMU) $(abspath $(OUT)) 127.0.0.1 $(PORT) 2>&1); \
	echo "$$out"; \
	echo "$$out" | grep -q 'udp round trip ok' \
	  || { echo "   [FAIL] nc_udp: 自检标记未经套接字返回"; exit 1; }; \
	echo "   [PASS] nc_udp"

clean:
	rm -rf $(BUILD)

.PHONY: all run check clean
