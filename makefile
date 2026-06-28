phony =

# ===================================================================
# Pyremu 顶层 makefile
# ===================================================================
# 主要入口: make emu — 自动构建固件 → 更新 ZSBL/FSBL → 启动调试器
#
# 构建依赖链:
#   rust_smode_entry.bin → custom-opensbi (distclean + 全量编译)
#       → 拷贝 fw_*.elf 到 tests/ → ZSBL/FSBL 符号更新 → 启动调试器
#

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
# distclean + 全量编译 custom-opensbi (generic 平台, 跳过 BSS 清零)
# 依赖 Rust S-mode runtime 先编译完成
# 作为真实 target (非 PHONY), 当 Rust 二进制更新或产物缺失时自动触发
$(FW_BUILD_DIR)/fw_payload.elf: $(RUST_SMODE_BIN)
	$(MAKE) -C $(FW_SRC_DIR) distclean
	$(MAKE) -C $(FW_SRC_DIR) PLATFORM=generic FW_SKIP_BSS_ZERO=1 FW_PAYLOAD=y -j$$(nproc)

# 便捷别名: 显式请求固件重新编译 (distclean + 全量)
build-fw: $(FW_BUILD_DIR)/fw_payload.elf
phony += build-fw

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
# 完整依赖链: Rust → 固件编译 → 拷贝 → ZSBL → 模拟器
emu: $(zsbl_fsbl)
	PYREMU_TRACE_TRAPS=1 PYREMU_TRACE_PMP=1 \
	python -m pyremu.debugger $(pyargs)
phony += emu

# ---- 测试 ----
test:
	pytest
phony += test

cov-test:
	pytest --cov=. --cov-report=term --full-trace
phony += cov-test

# ---- 工具链参考 (供手动使用) ----
opt-cc      = /opt/custom-llvm/bin/clang
opt-src-dir = $(PWD)/tests/src/
opt-bin-dir = $(PWD)/tests/bins/

.PHONY: $(phony)
