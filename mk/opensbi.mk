# sbi.mk — opensbi 的编译与链接参数

# 源码目录与构建输出目录; custom-opensbi 缺失时取 bsp/opensbi.
FW_SRC_DIR     = $(firstword $(wildcard $(bsp_dir)/custom-opensbi) $(bsp_dir)/opensbi)
FW_BUILD_DIR   = $(FW_SRC_DIR)/build/platform/generic/firmware

# 拷贝到 $(elf_dir) 的固件产物
fw_payload     = $(elf_dir)/custom_opensbi_fw_payload.elf
fw_jump        = $(elf_dir)/custom_opensbi_fw_jump.elf
fw_dynamic     = $(elf_dir)/custom_opensbi_fw_dynamic.elf
# 内嵌 Linux 内核的 fw_payload
fw_payload_linux = $(elf_dir)/custom_opensbi_fw_payload_linux.elf

# 全量编译 opensbi.
# 依赖:
#   - Rust S-mode runtime (嵌入 .sittim 段)
#   - opensbi 自身全部源码 (firmware/lib/include/platform/Kconfig/scripts)
# 当任一依赖更新或产物缺失时自动触发 distclean + 全量重编译.
FW_SRC_DEPS  = $(FW_SRC_DIR)/Makefile
FW_SRC_DEPS += $(shell find $(FW_SRC_DIR)/firmware $(FW_SRC_DIR)/lib -type f 2>/dev/null)
FW_SRC_DEPS += $(shell find $(FW_SRC_DIR)/include $(FW_SRC_DIR)/platform -type f 2>/dev/null)
FW_SRC_DEPS += $(shell find $(FW_SRC_DIR)/Kconfig $(FW_SRC_DIR)/scripts -type f 2>/dev/null)

# 参数覆盖 opensbi 默认值, 确保确定性构建
fw_basic_flag  = PLATFORM=generic
fw_basic_flag += PLATFORM_RISCV_XLEN=64
fw_basic_flag += PLATFORM_RISCV_ABI=lp64
# fw_basic_flag += FW_SKIP_BSS_ZERO=1

# 以下取值并入固件的 .config
# 池的起始与大小在 pyremu 侧读同一对变量
fw_basic_flag += CONFIG_POOL_BASE=$(RESERVED_MEM_BASE)
fw_basic_flag += CONFIG_POOL_SIZE=$(RESERVED_MEM_SIZE)

## ---- fw_payload 构建 (内嵌 payload) ----
FW_MAKE_FLAGS  = $(fw_basic_flag)
FW_MAKE_FLAGS += FW_PAYLOAD=y

## ---- fw_jump 构建 (无内嵌 payload, mret 后跳转到 FW_JUMP_ADDR) ----
FW_JUMP_FLAGS  = $(fw_basic_flag)
FW_JUMP_FLAGS += FW_JUMP=y
# 该地址等于 FW_RESERVED_MEM_BASE + FW_RESERVED_MEM_SIZE,
# 即固件映像保留区的上界.
FW_JUMP_FLAGS += FW_JUMP_ADDR=0x80400000
