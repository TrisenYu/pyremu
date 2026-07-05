phony =

# ===================================================================
# Pyremu 顶层 makefile
# ===================================================================
# 主要入口: make emu — 自动构建固件 -> 更新 ZSBL/FSBL -> 启动调试器
#
# 构建依赖链:
#   rust_smode_entry.bin -> custom-opensbi (distclean + 全量编译)
#       -> 拷贝 fw_*.elf 到 tests/ -> ZSBL/FSBL 符号更新 -> 启动调试器
#
# make build-native — 编译 Rust cdylib 加速库 (pyremu/_native/libdecode.so)
# make clean-native  — 删除编译产物
#

# ---- Native 加速库 (Rust cdylib -> ctypes) ----
NATIVE_DIR     = pyremu/_native
NATIVE_SO      = $(NATIVE_DIR)/libdecode.so
NATIVE_SRC     = $(shell find $(NATIVE_DIR)/src -type f -name '*.rs') $(NATIVE_DIR)/Cargo.toml

$(NATIVE_SO): $(NATIVE_SRC)
	cargo build --release --manifest-path $(NATIVE_DIR)/Cargo.toml
	cp $(NATIVE_DIR)/target/release/libdecode.so $(NATIVE_SO)
	@echo "[native] libdecode.so 已构建"

build-native: $(NATIVE_SO)
phony += build-native

clean-native:
	rm -f $(NATIVE_SO)
	rm -rf $(NATIVE_DIR)/target
phony += clean-native

# ---- 路径配置 ----
hart_num       = 1
zsbl_fsbl      = tests/bins/firm-bin/zsbl_fsbl_stub_cold_asm.bin
fw_payload     = tests/bins/elf/custom_opensbi_fw_payload.elf
fw_jump        = tests/bins/elf/custom_opensbi_fw_jump.elf
fw_dynamic     = tests/bins/elf/custom_opensbi_fw_dynamic.elf

# custom-opensbi
FW_SRC_DIR     = third-party/custom-opensbi
FW_BUILD_DIR   = $(FW_SRC_DIR)/build/platform/generic/firmware

# Rust S-mode 可信管理程序 (嵌入到固件 .coffer_enclave_man 段)
RUST_SMODE_DIR = third-party/rust_smode_entry
RUST_SMODE_BIN = $(RUST_SMODE_DIR)/rust_smode_entry.bin

# 调试器参数
pyargs         = --ram-base=0x80000000 --preload=$(zsbl_fsbl) --hart=$(hart_num) $(fw_payload)

# ---- Rust S-mode Runtime ----
# 编译 rust_smode_entry.bin, 供 custom-opensbi 嵌入 .coffer_enclave_man 段
$(RUST_SMODE_BIN): $(wildcard $(RUST_SMODE_DIR)/src/*.rs) $(wildcard $(RUST_SMODE_DIR)/src/**/*.rs)
	$(MAKE) -C $(RUST_SMODE_DIR) build

build-rust: $(RUST_SMODE_BIN)
phony += build-rust

# ---- 固件构建 ----
# 全量编译 custom-opensbi (generic 平台, 跳过 BSS 清零).
# 依赖:
#   - Rust S-mode runtime (嵌入 .coffer_enclave_man 段)
#   - custom-opensbi 自身全部源码 (firmware/lib/include/platform/Kconfig/scripts)
# 当任一依赖更新或产物缺失时自动触发 distclean + 全量重编译.
FW_SRC_DEPS := $(FW_SRC_DIR)/Makefile
FW_SRC_DEPS += $(shell find $(FW_SRC_DIR)/firmware $(FW_SRC_DIR)/lib -type f 2>/dev/null)
FW_SRC_DEPS += $(shell find $(FW_SRC_DIR)/include $(FW_SRC_DIR)/platform -type f 2>/dev/null)
FW_SRC_DEPS += $(shell find $(FW_SRC_DIR)/Kconfig $(FW_SRC_DIR)/scripts -type f 2>/dev/null)

# 参数覆盖 custom-opensbi 默认值, 确保确定性构建
FW_MAKE_FLAGS := PLATFORM=generic
FW_MAKE_FLAGS += PLATFORM_RISCV_XLEN=64
FW_MAKE_FLAGS += PLATFORM_RISCV_ABI=lp64
FW_MAKE_FLAGS += FW_SKIP_BSS_ZERO=1
FW_MAKE_FLAGS += FW_PAYLOAD=y

$(FW_BUILD_DIR)/fw_payload.elf: $(RUST_SMODE_BIN) $(FW_SRC_DEPS)
	$(MAKE) -C $(FW_SRC_DIR) distclean $(FW_MAKE_FLAGS)
	$(MAKE) -C $(FW_SRC_DIR) $(FW_MAKE_FLAGS) -j$$(nproc)

# 便捷别名: 显式请求固件重新编译 (distclean + 全量)
build-fw: $(FW_BUILD_DIR)/fw_payload.elf
phony += build-fw

# ---- fw_jump 构建 (无内嵌 payload, mret 后跳转到 FW_JUMP_ADDR) ----
FW_JUMP_FLAGS := PLATFORM=generic
FW_JUMP_FLAGS += PLATFORM_RISCV_XLEN=64
FW_JUMP_FLAGS += PLATFORM_RISCV_ABI=lp64
FW_JUMP_FLAGS += FW_SKIP_BSS_ZERO=1
FW_JUMP_FLAGS += FW_JUMP=y
FW_JUMP_FLAGS += FW_JUMP_ADDR=0x80200000

$(FW_BUILD_DIR)/fw_jump.elf: $(FW_SRC_DEPS)
	$(MAKE) -C $(FW_SRC_DIR) distclean $(FW_JUMP_FLAGS)
	$(MAKE) -C $(FW_SRC_DIR) $(FW_JUMP_FLAGS) -j$$(nproc)

build-fw-jump: $(FW_BUILD_DIR)/fw_jump.elf
phony += build-fw-jump

# ---- 固件产物拷贝 ----
# 将 custom-opensbi 构建产物复制到 tests/ 目录, 供模拟器加载
# 当构建产物比 tests/ 中的副本更新时自动触发
$(fw_payload): $(FW_BUILD_DIR)/fw_payload.elf
	cp $(FW_BUILD_DIR)/fw_payload.elf $(fw_payload)
	cp $(FW_BUILD_DIR)/fw_jump.elf $(fw_jump)
	cp $(FW_BUILD_DIR)/fw_dynamic.elf $(fw_dynamic)
	@echo "固件已拷贝: $(fw_payload)"

# ---- ZSBL/FSBL 构建 ----
# 依赖固件 ELF (提取符号地址), 自动更新 coldboot_done 等硬编码地址
# 当固件副本更新时自动触发
$(zsbl_fsbl): $(fw_payload)
	$(MAKE) -C tests/src-env zsbl_fsbl_stub
	@echo "ZSBL/FSBL 已根据固件符号表更新"

# ---- 主入口: 启动模拟器 ----
# 完整依赖链: Rust -> 固件编译 -> 拷贝 -> ZSBL -> 模拟器
emu: $(zsbl_fsbl)
	PYREMU_TRACE_TRAPS=1 PYREMU_TRACE_PMP=1 \
	python -m pyremu.debugger $(pyargs)
phony += emu

# ---- Linux 内核启动 ----
# 使用 fw_payload 模式将 Linux Image 嵌入 OpenSBI, 经 M->S 移交直接启动.
# ZSBL 复用于设置 coldboot_done=1 (地址在两次编译中相同).
LINUX_IMG     = third-party/linux/arch/riscv/boot/Image
fw_payload_linux = tests/bins/elf/custom_opensbi_fw_payload_linux.elf

$(fw_payload_linux): $(RUST_SMODE_BIN) $(FW_SRC_DEPS)
	@echo "构建 custom-opensbi + Linux 内核 payload..."
	$(MAKE) -C $(FW_SRC_DIR) distclean $(FW_MAKE_FLAGS) \
		FW_PAYLOAD_PATH=$(realpath $(LINUX_IMG))
	$(MAKE) -C $(FW_SRC_DIR) $(FW_MAKE_FLAGS) \
		FW_PAYLOAD_PATH=$(realpath $(LINUX_IMG)) -j$$(nproc)
	cp $(FW_BUILD_DIR)/fw_payload.elf $(fw_payload_linux)
	cp $(FW_BUILD_DIR)/fw_payload.bin $(fw_payload_linux:.elf=.bin)
	@echo "Linux 固件已拷贝: $(fw_payload_linux)"

build-fw-linux: $(fw_payload_linux)
phony += build-fw-linux

emu-linux: $(zsbl_fsbl) $(fw_payload_linux)
	python -m pyremu.debugger \
		--ram-base=0x80000000 --ram 512M --preload=$(zsbl_fsbl) \
		--bootargs="earlycon=sbi" \
		--hart=1 $(fw_payload_linux)
phony += emu-linux

# ---- Linux 内核 fw_jump 模式 ----
# OpenSBI (fw_jump.elf) 启动后 mret -> 0x80200000,
# 内核 Image 预载到该地址, vmlinux 符号供调试.
LINUX_IMG    = third-party/linux/arch/riscv/boot/Image
LINUX_VMLINUX = third-party/linux/vmlinux
fw_jump_elf  = tests/bins/elf/custom_opensbi_fw_jump.elf

emu-linux-jump: $(zsbl_fsbl) $(fw_jump_elf)
	python -m pyremu.debugger \
		--ram-base=0x80000000 --ram 512M --preload=$(zsbl_fsbl) \
		--kernel=$(LINUX_IMG) --sym=$(LINUX_VMLINUX) \
		--bootargs="earlycon=sbi" \
		--hart=1 $(fw_jump_elf)
phony += emu-linux-jump

# ---- 测试 ----
test:
	pytest -x
phony += test

cov-test:
	pytest -x --cov=. --cov-report=term --full-trace
phony += cov-test

# ---- 工具链参考 (供手动使用) ----
opt-cc      = /opt/custom-llvm/bin/clang
opt-src-dir = $(PWD)/tests/src/
opt-bin-dir = $(PWD)/tests/bins/

.PHONY: $(phony)
