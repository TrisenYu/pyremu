# fn_apps/Makefile — 统一构建所有飞地载荷 (可信应用 payload)
#
# 职责:
#   1. 检测 bsp/musl-gc-sysroot, 缺失时自动构建 (依赖 vendor/musl 子模块)
#   2. 逐个构建 Rust 载荷 (cargo, riscv64gc-unknown-linux-musl 静态)、C 载荷与 C++ 载荷,
#      名单见下方三个载荷名单变量
#   3. 收集最终静态 ELF 到 bin/, 供可信应用 (飞地) 加载
#
# 用法:
#   make              # 构建全部载荷到 bin/
#   make <name>       # 仅构建单个载荷, 可取的名字见下方三个载荷名单变量
#   make check        # qemu-riscv64 下逐个运行自检, 输出含 FAIL 即报错
#   make clean        # 清理 cargo 产物、stress-ng 的构建产物与 bin/ (保留 sysroot)
#
# 依赖: vendor/musl 与 vendor/busybox 子模块 + clang/llvm-ar/llvm-ranlib
# (见 bsp/build-musl-sysroot.sh)

include config.mk

CARGO   := cargo
TARGET  := riscv64gc-unknown-linux-musl
PROFILE ?= release

# 各子目录 Makefile 与上游构建的并行度
JOBS ?= $(shell nproc)

# cargo 的 --release 开关 (debug 无对应开关, 故置空)
ifeq ($(PROFILE),release)
CARGO_RELEASE := --release
else
CARGO_RELEASE :=
endif

# 目录名 -> cargo 包名 (orbit 的包名为 orbit_payload)
pkg_of = $(if $(filter orbit,$(1)),orbit_payload,$(1))

RUST_PAYLOADS := orbit chem graphene ising fem dsp linalg
# malloc_churn (t-test1) 为多线程 C 载荷, check 目标需传并发参数, 故单列一条规则
# (见下); 归类上仍属 C 载荷.
C_PAYLOADS   := knots cfrac chibicc json_calc malloc_churn stress_ng nc_udp
# json_cpp 与 json_calc 同一源码目录 (共用输入集与内嵌 json_config.inc), 仅实现语言不同
CPP_PAYLOADS := dune json_cpp
ALL_PAYLOADS := $(RUST_PAYLOADS) $(C_PAYLOADS) $(CPP_PAYLOADS)

BIN := $(abspath bin)

.PHONY: all check clean langlands $(ALL_PAYLOADS)

all: $(ALL_PAYLOADS)

# ---- sysroot: 缺失时自动构建 ----
$(MUSL_LIB)/libc.a:
	@test -f "$(MUSL_SRC)/Makefile" || { \
		echo "错误: 缺少 vendor/musl 子模块, 请先执行: git submodule update --init vendor/musl"; \
		exit 1; }
	@echo ">> 缺少 musl-gc-sysroot, 开始构建 ..."
	"$(REPO)/bsp/build-musl-sysroot.sh"

# ---- Rust 载荷: cargo 构建 + 收集到 bin/ ----
define rust_rule
$(1): $(MUSL_LIB)/libc.a
	$(CARGO) build $(CARGO_RELEASE) -p $(call pkg_of,$(1))
	@mkdir -p $(BIN)
	cp target/$(TARGET)/$(PROFILE)/$(call pkg_of,$(1)) $(BIN)/$(1)
	@echo "  -> $(BIN)/$(1)"
endef
$(foreach p,$(RUST_PAYLOADS),$(eval $(call rust_rule,$(p))))

# ---- C 载荷: 委托各子目录 Makefile ----
knots: $(MUSL_LIB)/libc.a
	$(MAKE) -C knots
	@mkdir -p $(BIN)
	cp knots/build/knots_payload $(BIN)/knots
	@echo "  -> $(BIN)/knots"

cfrac: $(MUSL_LIB)/libc.a
	$(MAKE) -C cfrac
	@mkdir -p $(BIN)
	cp cfrac/build/cfrac_payload $(BIN)/cfrac
	@echo "  -> $(BIN)/cfrac"

chibicc: $(MUSL_LIB)/libc.a
	$(MAKE) -C chibicc
	@mkdir -p $(BIN)
	cp chibicc/build/chibicc_payload $(BIN)/chibicc
	@echo "  -> $(BIN)/chibicc"

# 两个载荷各自独立: 只构建所用实现, 避免一方编译失败牵连另一方
json_calc: $(MUSL_LIB)/libc.a
	$(MAKE) -C json_calc build/json_calc_payload
	@mkdir -p $(BIN)
	cp json_calc/build/json_calc_payload $(BIN)/json_calc
	@echo "  -> $(BIN)/json_calc"

# json_cpp: 同目录的 nlohmann/json 实现, 与 json_calc 对拍同一输入集
json_cpp: $(MUSL_LIB)/libc.a
	$(MAKE) -C json_calc build/json_cpp_payload
	@mkdir -p $(BIN)
	cp json_calc/build/json_cpp_payload $(BIN)/json_cpp
	@echo "  -> $(BIN)/json_cpp"

# malloc_churn: Wolfram Gloger 多线程 malloc 压力测试 (t-test1, 单 hart 抢占式
# clone/futex 的验证载荷). 源码目录名为 malloc-churn, 产物名与载荷名一致.
malloc_churn: $(MUSL_LIB)/libc.a
	$(MAKE) -C malloc-churn
	@mkdir -p $(BIN)
	cp malloc-churn/build/malloc_churn $(BIN)/malloc_churn
	@echo "  -> $(BIN)/malloc_churn"

# nc_udp: 以 vendor/busybox 的 nc applet 在 UDP 模式下自收自发的套接字通路载荷。
# 该载荷在源码树外单独构建一份 busybox (配置只启用 nc), 见 nc_udp/Makefile。
nc_udp: $(MUSL_LIB)/libc.a
	$(MAKE) -C nc_udp
	@mkdir -p $(BIN)
	cp nc_udp/build/nc_udp_payload $(BIN)/nc_udp
	@echo "  -> $(BIN)/nc_udp"

# ---- stress-ng: 上游整套压力测试工具, 交叉编译为静态 musl 载荷 ----
#
# 源码是子模块 stress-ng。上游 Makefile 在源码树内构建, 其 .gitignore 已忽略全部生成物
# (config、config.h、configs/、*.o 与最终二进制), 故直接在其中构建, 子模块工作区保持干净。
#
# 交叉编译另有四处处理:
#   1. 内核头文件。stress-ng 引用 linux/ 与 asm/ 下的 UAPI 头 (150 余处), 而 musl 源码
#      树与 sysroot 都不提供; 故从 Debian 的 riscv64 交叉包取 linux、asm、asm-generic
#      等子树, 以符号链接汇入 build/stress-ng/khdr。只链接 UAPI 子树而不整个 include
#      目录, 是为了不把该目录下 glibc 的头文件 (bits、sys、gnu 等) 混进 musl 的头文件
#      搜索路径。
#   2. libgcc_eh。clang 为目标链接隐式追加 -lgcc_eh, 而 musl sysroot 只提供 libgcc.a,
#      以同名副本补齐, 与其余载荷 Makefile 的做法一致。
#   3. 编译器。上游 Makefile 用 $(CC) 同时做特性探测与正式编译, 故目标参数 (--target、
#      march、abi、头文件与库搜索路径) 必须全部写在 CC 与 CXX 内。
#   4. CFLAGS。上游自行追加 -O2、-std=gnu99、-Wall -Wextra 等参数, 本处不传 CFLAGS。
#   5. apparmor-data.h。该头文件由 usr.bin.pulseaudio.eg 经 apparmor_parser 生成, 不在
#      .gitignore 覆盖范围内, 而上游 Makefile 让 stress-apparmor.o 无条件依赖它, 故每次构建
#      都会在源码树内生成。它只被 stress-apparmor.c 的 HAVE_APPARMOR 分支包含, 而 musl
#      sysroot 不提供 libapparmor, 探测不通过, 本构建无任何源文件引用它。故向 make 传
#      -o apparmor-data.h (声明该文件无需重新生成), 使该规则不执行。
SNG_SRC   := $(abspath stress-ng)
# 内核头文件符号链接与 libgcc_eh 副本的存放目录 (载荷自己的构建目录, 不在源码树内)
SNG_BUILD := $(abspath build/stress-ng)
# riscv64 内核 UAPI 头的来源目录, 由 linux-libc-dev-riscv64-cross 提供
SNG_KHDR_SRC  := /usr/riscv64-linux-gnu/include
SNG_KHDR_DIRS := linux asm asm-generic drm misc mtd rdma scsi sound video xen
SNG_TRIPLE    := riscv64-linux-musl
SNG_INCLUDES  := -isystem $(shell $(CLANG) -print-resource-dir)/include \
	-isystem $(MUSL_SRC)/include -isystem $(MUSL_SRC)/arch/riscv64 \
	-isystem $(MUSL_SRC)/arch/generic -isystem $(MUSL_SRC)/obj/include
SNG_CC  := $(CLANG) --target=$(SNG_TRIPLE) $(ARCH) -static -fuse-ld=lld -nostdinc \
	$(SNG_INCLUDES) -isystem $(SNG_BUILD)/khdr -B$(MUSL_LIB) -L$(MUSL_LIB) -L$(SNG_BUILD)/lib
SNG_CXX := clang++ --target=$(SNG_TRIPLE) $(ARCH) -static -fuse-ld=lld -nostdinc \
	$(SNG_INCLUDES) -isystem $(MUSL_SYSROOT)/include/c++/v1 \
	-isystem $(SNG_BUILD)/khdr -B$(MUSL_LIB) -L$(MUSL_LIB) -L$(SNG_BUILD)/lib

stress_ng: $(MUSL_LIB)/libc.a
	@test -f "$(SNG_SRC)/Makefile" || { \
		echo "错误: 缺少 stress-ng 子模块, 请先执行: git submodule update --init fn_apps/stress-ng"; \
		exit 1; }
	@test -d "$(SNG_KHDR_SRC)/linux" || { \
		echo "错误: 缺少 riscv64 内核头文件 $(SNG_KHDR_SRC)/linux"; \
		echo "  安装: apt install gcc-riscv64-linux-gnu (随附 linux-libc-dev-riscv64-cross)"; \
		exit 1; }
	set -e; \
	mkdir -p $(SNG_BUILD)/khdr $(SNG_BUILD)/lib; \
	for d in $(SNG_KHDR_DIRS); do ln -sfn "$(SNG_KHDR_SRC)/$$d" "$(SNG_BUILD)/khdr/$$d"; done; \
	cp $(MUSL_LIB)/libgcc.a $(SNG_BUILD)/lib/libgcc_eh.a
	$(MAKE) -C $(SNG_SRC) CC='$(SNG_CC)' CXX='$(SNG_CXX)' STATIC=1 \
		-o apparmor-data.h -j$(JOBS)
	@mkdir -p $(BIN)
	cp $(SNG_SRC)/stress-ng $(BIN)/stress_ng
	@echo "  -> $(BIN)/stress_ng"

# C++ 载荷: DUNE FEM (依赖 sysroot 内 libdune*.a, 见 dune/Makefile)
dune: $(MUSL_LIB)/libc.a
	$(MAKE) -C dune
	@mkdir -p $(BIN)
	cp dune/build/dune_payload $(BIN)/dune
	@echo "  -> $(BIN)/dune"

# ---- Langlands: 依赖 FLINT 的数论载荷 (Lfunc / modular-form) ----
#
# 这两个载荷的依赖 (GMP/MPFR/FLINT 的 riscv64 静态库) 不由本 Makefile 构建, 须先执行
# Langlands/build-deps.sh 从 vendor/{gmp,mpfr,flint} 子模块产出 Langlands/build/sysroot。
# 故不并入 ALL_PAYLOADS: 缺该 sysroot 时本目标报错退出, 其余载荷的构建与收集不受影响。
# 两份产物各约 7.5 MiB, 安装到 bin/ 后由 rootfs 构建脚本装入 /eval 的载荷目录。
LANGLANDS_DIR := $(abspath Langlands)

langlands:
	@test -f $(LANGLANDS_DIR)/build/sysroot/lib/libflint.a || { \
		echo "错误: 缺少 FLINT 依赖 sysroot, 请先执行: $(LANGLANDS_DIR)/build-deps.sh"; \
		exit 1; }
	$(MAKE) -C $(LANGLANDS_DIR)/Lfunc
	$(MAKE) -C $(LANGLANDS_DIR)/modular-form
	@mkdir -p $(BIN)
	cp $(LANGLANDS_DIR)/Lfunc/build/Lfunc_payload $(BIN)/Lfunc
	cp $(LANGLANDS_DIR)/modular-form/build/modular-form_payload $(BIN)/modular_form
	@echo "  -> $(BIN)/Lfunc $(BIN)/modular_form"

# ---- 自检: qemu 逐个运行, 输出含 FAIL 即失败 ----
check: $(ALL_PAYLOADS)
	@fail=0; \
	for p in $(ALL_PAYLOADS); do \
		echo "当前测试项 $$p"; \
		ok=1; \
		if [ "$$p" = "chibicc" ]; then \
			$(QEMU) $(BIN)/$$p -cc1 -x c -cc1-input - -cc1-output - \
				< chibicc/test/inp_src.c 2>&1 | grep -q '.globl main' || ok=0; \
		elif [ "$$p" = "malloc_churn" ]; then \
			$(QEMU) $(BIN)/$$p 6 2 1000 1000 32 2>&1 | grep -q 'Done\.' || ok=0; \
		elif [ "$$p" = "stress_ng" ]; then \
			$(QEMU) $(BIN)/$$p --vm 1 --vm-bytes 2M --timeout 3s 2>&1 \
				| grep -q 'successful run completed' || ok=0; \
		elif [ "$$p" = "nc_udp" ]; then \
			$(MAKE) -C nc_udp check >/dev/null 2>&1 || ok=0; \
		else \
			$(QEMU) $(BIN)/$$p 2>&1 | grep -q 'FAIL' && ok=0; \
		fi; \
		if [ $$ok -eq 1 ]; then \
			echo "   [PASS] $$p"; \
		else \
			echo "   [FAIL] $$p"; fail=1; \
		fi; \
	done; \
	$(MAKE) -C cfrac check-argv0 || fail=1; \
	exit $$fail

clean:
	$(CARGO) clean
	$(MAKE) -C knots clean
	$(MAKE) -C json_calc clean
	$(MAKE) -C malloc-churn clean
	$(MAKE) -C nc_udp clean
	$(MAKE) -C stress-ng clean
	$(MAKE) -C $(LANGLANDS_DIR)/Lfunc clean
	$(MAKE) -C $(LANGLANDS_DIR)/modular-form clean
	rm -rf $(SNG_BUILD)
	rm -rf $(BIN)
