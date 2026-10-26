phony =

# 全部用于编译配置的变量定义集中在mk/*.mk中。
# 并通过统一导出文件mods.mk而导出。
include mk/mods.mk

ifeq ($(OS),Windows_NT)
	uv_check := $(shell where uv 2>nul)
	py_check  := $(shell dir .venv 2>nul)
	activate  := .venv/Scripts/activate
else
	uv_check := $(shell command -v uv 2>/dev/null)
	py_check  := $(shell ls .venv/pyvenv.cfg 2>/dev/null)
	activate  := . $(or $(shell echo $$VIRTUAL_ENV),.venv/bin/activate)
endif

__uv-check:
ifndef uv_check
	$(error can not find uv, which could be installed by: curl -LsSf https://astral.sh/uv/install.sh | sh)
endif
phony += __uv-check

__pyenv-check: __uv-check
ifndef py_check
	@uv venv && uv sync
else
	@uv sync
endif
phony += __pyenv-check


# CARGO_FLAGS 变化时强制重建 native .so
CONFIG_GEN_RS = $(NATIVE_DIR)/cpu/src/configs_gen.rs

# 由 mk/ 下的配置分片生成 Rust 常量定义
$(CONFIG_GEN_RS): FORCE
	@rm -f $@.tmp
	@echo '// generated from mk/*.mk by Makefile — do not edit' >> $@.tmp
	@echo 'pub const TLB_ENTRIES: usize = $(TLB_ENTRIES);' >> $@.tmp
	@echo 'pub const CFG_NO_TLB: bool = $(if $(filter 1,$(PYREMU_NO_TLB)),true,false);' >> $@.tmp
	@echo 'pub const CFG_NO_L2: bool = $(if $(filter 1,$(PYREMU_NO_L2)),true,false);' >> $@.tmp
	@echo 'pub const CFG_AIA: bool = $(if $(filter 1,$(PYREMU_AIA)),true,false);' >> $@.tmp
	@echo 'pub const CFG_H_EXT: bool = $(if $(filter 1,$(PYREMU_H_EXT)),true,false);' >> $@.tmp
	@echo 'pub const CFG_HAS_SEMIHOSTING: bool = $(if $(filter 1,$(PYREMU_HAS_SEMIHOSTING)),true,false);' >> $@.tmp
	@echo 'pub const IMSIC_M_BASE: u64 = $(IMSIC_M_BASE);' >> $@.tmp
	@echo 'pub const IMSIC_S_BASE: u64 = $(IMSIC_S_BASE);' >> $@.tmp
	@echo 'pub const CPU_FREQ_HZ: u64 = $(CPU_FREQ_HZ);' >> $@.tmp
	@cmp -s $@.tmp $@ 2>/dev/null && rm -f $@.tmp || mv -f $@.tmp $@

$(NATIVE_SO) $(TERMIO_SO): $(NATIVE_SRC) $(CONFIG_GEN_RS)
	@$(MAKE) -C pyremu
	cargo build --release --manifest-path $(NATIVE_DIR)/Cargo.toml --workspace $(CARGO_FLAGS)
	install -m 755 $(NATIVE_DIR)/target/release/libdecode.so $(NATIVE_SO)
	install -m 755 $(NATIVE_DIR)/target/release/libtermio.so $(TERMIO_SO)
	@echo "libdecode.so and libtermio.so have built"

# 加速执行库
build-native: $(NATIVE_SO) $(TERMIO_SO)
phony += build-native

clean-native:
	rm -f $(NATIVE_SO) $(TERMIO_SO)
	rm -rf $(NATIVE_DIR)/target
phony += clean-native

# 清理 sittim、pyremu/_native 与 custom-opensbi 的编译产物.
# custom-opensbi 目录不存在时跳过该步.
clean: clean-native
	$(MAKE) -C $(SITTIM_DIR) clean
ifneq ($(wildcard $(FW_SRC_DIR)),)
	$(MAKE) -C $(FW_SRC_DIR) distclean
endif
phony += clean

FORCE:
phony += FORCE


# 编译 sittim.bin, 供 opensbi 嵌入 .sittim 段.
#
# 依赖分两类:
#   源文件类: src 下全部 .rs; 生成 configs_gen.rs 的输入 config.mk 与 Makefile;
#     生成 link.ld 的输入 link.ld.S; 以及 Cargo 清单. 任一文件更新即重建.
#   编译控制类: config.mk 各项以 ?= 给出默认值, 命令行上的同名变量经递归 make 传给
#     子 make 并覆盖默认值. 这一来源不改变任何文件的时间戳, 故以一份取值戳记判定,
#     写法与 FW_CFG_STAMP 相同; 不在 config.mk 内、但经命令行覆盖即改变编译结果的
#     变量一并列入戳记.
#
# 取值得在父 make 中可见才能写入戳记, 故一并读入该配置文件; 各项以 ?= 给出默认值,
# 命令行覆盖优先生效.
include $(SITTIM_DIR)/config.mk
SITTIM_CFG_VARS := $(shell sed -n 's/^\([A-Za-z_][A-Za-z0-9_]*\)[[:space:]]*?=.*/\1/p' $(SITTIM_DIR)/config.mk)
SITTIM_CFG_VARS += CARGO_FLAGS
SITTIM_CFG_STAMP = $(bins_dir)/sittim_cfg.stamp
$(SITTIM_CFG_STAMP): FORCE
	@mkdir -p $(bins_dir)
	@printf '%s\n' $(foreach v,$(SITTIM_CFG_VARS),$(v)=$($(v))) > $@.tmp
	@cmp -s $@.tmp $@ 2>/dev/null && rm -f $@.tmp || mv -f $@.tmp $@

SITTIM_SRC_DEPS := $(shell find $(SITTIM_DIR)/src -name '*.rs' 2>/dev/null)
SITTIM_SRC_DEPS += $(SITTIM_DIR)/config.mk $(SITTIM_DIR)/Makefile $(SITTIM_DIR)/link.ld.S
SITTIM_SRC_DEPS += $(SITTIM_DIR)/Cargo.toml $(SITTIM_DIR)/Cargo.lock

# sittim 的 config.mk 不包含根目录的配置文件, 平台级取值在此经命令行传入: 经本文件构建
# 时采用 mk/ 下配置分片的取值, 单独在 bsp/sittim 下构建时采用其 config.mk 的默认值.
$(SITTIM_BIN): $(SITTIM_SRC_DEPS) $(SITTIM_CFG_STAMP)
	$(MAKE) -C $(SITTIM_DIR) build TIMER_FREQ=$(TIMER_FREQ) UART_BASE=$(UART_BASE)

build-rust: $(SITTIM_BIN)
phony += build-rust


# fw_basic_flag 变化时强制重建
FW_CFG_STAMP = $(bins_dir)/fw_cfg.stamp
$(FW_CFG_STAMP): FORCE
	@mkdir -p $(bins_dir)
	@printf '%s\n' $(fw_basic_flag) > $@.tmp
	@cmp -s $@.tmp $@ 2>/dev/null && rm -f $@.tmp || mv -f $@.tmp $@

$(FW_BUILD_DIR)/fw_payload.elf: $(SITTIM_BIN) $(FW_SRC_DEPS) $(FW_CFG_STAMP)
	$(MAKE) -C $(FW_SRC_DIR) distclean
	$(MAKE) -C $(FW_SRC_DIR) $(FW_MAKE_FLAGS) -j$$(nproc)

# fw_jump 同样内嵌 sittim.bin (firmware/objects.mk _ENCLAVE_DEFAULT),
$(FW_BUILD_DIR)/fw_jump.elf: $(SITTIM_BIN) $(FW_SRC_DEPS) $(FW_CFG_STAMP)
	$(MAKE) -C $(FW_SRC_DIR) distclean
	$(MAKE) -C $(FW_SRC_DIR) $(FW_JUMP_FLAGS) -j$$(nproc)


# 将 opensbi 构建产物复制到 tests/ 目录, 供模拟器加载
# 当构建产物比 tests/ 中的副本更新时自动触发
$(fw_payload): $(FW_BUILD_DIR)/fw_payload.elf
	@mkdir -p $(elf_dir)
	cp $(FW_BUILD_DIR)/fw_payload.elf $(fw_payload)
	cp $(FW_BUILD_DIR)/fw_jump.elf $(fw_jump)
	cp $(FW_BUILD_DIR)/fw_dynamic.elf $(fw_dynamic)
	@echo "固件已拷贝: $(fw_payload)"

$(fw_jump): $(FW_BUILD_DIR)/fw_jump.elf
	@mkdir -p $(elf_dir)
	cp $(FW_BUILD_DIR)/fw_jump.elf $(fw_jump)
	@echo "固件已拷贝: $(fw_jump)"

$(fw_payload_linux): $(SITTIM_BIN) $(FW_SRC_DEPS) $(FW_CFG_STAMP)
	@echo "构建opensbi与Linux内核..."
	$(MAKE) -C $(FW_SRC_DIR) distclean $(FW_MAKE_FLAGS) \
		FW_PAYLOAD_PATH=$(realpath $(LINUX_IMG))
	$(MAKE) -C $(FW_SRC_DIR) $(FW_MAKE_FLAGS) \
		FW_PAYLOAD_PATH=$(realpath $(LINUX_IMG)) -j$$(nproc)
	@mkdir -p $(elf_dir)
	cp $(FW_BUILD_DIR)/fw_payload.elf $(fw_payload_linux)
	cp $(FW_BUILD_DIR)/fw_payload.bin $(fw_payload_linux:.elf=.bin)
	@echo "copy $(fw_payload_linux) as bootstrap payload"


build-fw: $(FW_BUILD_DIR)/fw_payload.elf
phony += build-fw

build-fw-jump: $(fw_jump)
phony += build-fw-jump

# 依赖固件产物, 保证启动模拟器前固件已构建完成
kei-sav: $(fw_payload)
	$(MAKE) -C bsp/kei-boot kei-sav


build-fw-linux: $(fw_payload_linux)
phony += build-fw-linux

# ---- 符号导出: 把带符号表的 ELF 汇集到 $(sym_dir) ----
#
# 只汇集已构建完成的产物, 不触发任何编译; 尚未构建的产物跳过并报告。
# 目录中每项是指向构建产物的相对符号链接, 产物重新链接后内容随之更新。
# 文件名按来源加前缀, 供 sym ls 按类归组:
#   fw-*      固件
#   kernel-*  测试固件
#   app-*     飞地载荷, 取自 fn_apps/bin
#   kei-sav   零阶段启动器 (bsp/kei-boot/kei.sav.elf)
#   sittim    Rust S 模式可信管理程序
#   vmlinux   Linux 内核
SYM_EXPORT_SPECS = \
	$(fw_payload):fw-payload \
	$(fw_jump):fw-jump \
	$(fw_dynamic):fw-dynamic \
	$(fw_payload_linux):fw-linux \
	$(elf_dir)/kernel.elf:kernel \
	$(elf_dir)/kernel_ab.elf:kernel-ab \
	$(elf_dir)/lottery_boot.elf:kernel-lottery \
	$(elf_dir)/u_mode_run_fib.elf:kernel-u-mode-fib \
	bsp/kei-boot/kei.sav.elf:kei-sav \
	$(SITTIM_DIR)/target/riscv64imac-unknown-none-elf/release/sittim:sittim \
	$(LINUX_VMLINUX):vmlinux

sym-export:
	@tools/export_symbols.sh $(sym_dir) $(SYM_EXPORT_SPECS) fn_apps/bin:app-
phony += sym-export

# ---- initramfs 打包 debootstrap 目录下生成cpio与gzip ----
$(INITRAMFS):
	@tools/build_initramfs.sh $(ROOTFS_DIR) $(INITRAMFS)

build-initramfs: $(INITRAMFS)
phony += build-initramfs


# 用 tools/dump_dtb.py 构造平台预设dtb二进制文件，
# 再用 dtc 还原为人类可读的 build/emu.dts。
gen-dtb: __pyenv-check
	@mkdir -p $(bins_dir)
	PYTHONPATH=$(CURDIR) uv run python tools/dump_dtb.py $(bins_dir)/emu.dtb
	dtc -I dtb -O dts $(bins_dir)/emu.dtb -o $(bins_dir)/emu.dts
	@echo "  -> $(bins_dir)/emu.dtb + $(bins_dir)/emu.dts"
phony += gen-dtb

# 项目打包
pkg-up: __pyenv-check
	python3 -m build --sdist --sdist-extract-dir pyremu


# ---- 主入口: 启动模拟器 ----
# 完整依赖链: Rust 编译、固件编译、拷贝、kei.sav、模拟器
emu: kei-sav __pyenv-check
	@$(MAKE) --no-print-directory sym-export
	python -m pyremu.debugger --hart-logs=output/ $(pyargs)
phony += emu

# fw_payload 模式 (Linux 内嵌 OpenSBI payload)
# 注意，嵌入的是vmlinux，会缺乏调试符号
emu-linux-payload: kei-sav $(fw_payload_linux) __pyenv-check
	@$(MAKE) -C pyremu
	@$(MAKE) --no-print-directory sym-export
	python -m pyremu.debugger \
		--hart-logs=output/ --ram-base=0x80000000 --ram 512M $(zsbl_args) \
		--harts=$(NUM_HARTS) $(fw_payload_linux)
phony += emu-linux-payload

# 启动linux并调试整个调试系统与试验环境
emu-linux-sh: kei-sav $(fw_jump) build-native __pyenv-check
# 调试输出地址
# 内存基地址
# 模拟内存大小
# 零阶段启动器
# 内核镜像
# 内核符号所在目录
# 设备树生成
# ext4参数
# 启动参数
# hart数量
# 调试对象
	@$(MAKE) -C pyremu
	@$(MAKE) --no-print-directory sym-export
	PYTHON_GIL=0 python -m pyremu.debugger \
		--hart-logs=output/ \
		--ram-base=0x80000000 \
		--ram $(ram) \
		$(zsbl_args) \
		--kernel=$(LINUX_IMG) \
		--sym-dir=$(abspath $(sym_dir)) \
		--fdt -1 \
		$(disk_args) \
		$(initrd_args) \
		$(bootargs_arg) \
		--harts=$(NUM_HARTS) \
		$(fw_jump)
phony += emu-linux-sh


# ---- 测试 ----
# 限制: 4 GiB 虚拟内存, --ignore 排除 ordering-dependent 失败.
test-build:
	$(MAKE) -C tests all
	@$(MAKE) --no-print-directory sym-export
phony += test-build

test: __pyenv-check test-build
	$(MAKE) -C bsp/sittim test
	ulimit -v 4194304 && PYTHONUNBUFFERED=1 PYTHON_GIL=0 uv run pytest -v -x
phony += test

cov-test: __pyenv-check test-build
	find pyremu tests examples -type d -name __pycache__ -prune -exec rm -rf {} +
	ulimit -v 4194304 && PYTHONUNBUFFERED=1 PYTHON_GIL=0 uv run pytest -v -x \
		--cov=pyremu --cov-report=term --full-trace
phony += cov-test

.PHONY: $(phony)

# 由 mk/ 下的配置分片生成 Python 常量
# include pyremu/makefile
