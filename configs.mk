# ===================================================================
# configs.mk — Pyremu 顶层 makefile 的全部变量定义
# ===================================================================
# 由 makefile 通过 `include configs.mk` 导入。此文件只放变量, 不放规则/目标。
# 可覆盖项 (?=): hart_num / ram / rdinit / DISK / INITRD / ROOTFS_DIR。

# ---- 公共基础路径 (供下方各定义复用, 避免重复前缀) ----
third_party = third-party
bins_dir    = tests/bins
elf_dir     = $(bins_dir)/elf
firm_dir    = $(bins_dir)/firm-bin
linux_dir   = $(third_party)/linux

# ---- Native 加速库 (Rust cdylib -> ctypes) ----
NATIVE_DIR     = pyremu/_native
NATIVE_SO      = $(NATIVE_DIR)/libdecode.so
TERMIO_SO      = $(NATIVE_DIR)/libtermio.so
# 额外 cargo flags。
# PYREMU_DIAG_LOG / PYREMU_TRACE_SRET 已指定时自动启用 diagnostic feature。
CARGO_FLAGS    ?=
ifneq ($(or $(PYREMU_DIAG_LOG),$(PYREMU_TRACE_SRET)),)
  CARGO_FLAGS += --features diagnostic
endif
NATIVE_SRC     = $(shell find $(NATIVE_DIR)/cpu $(NATIVE_DIR)/termio -type f -name '*.rs') \
                 $(NATIVE_DIR)/Cargo.toml $(NATIVE_DIR)/cpu/Cargo.toml $(NATIVE_DIR)/termio/Cargo.toml
# ---- 路径配置 ----
hart_num       = 2
zsbl_fsbl      = $(firm_dir)/zsbl_fsbl_stub_cold_asm.bin
fw_payload     = $(elf_dir)/custom_opensbi_fw_payload.elf
fw_jump        = $(elf_dir)/custom_opensbi_fw_jump.elf
fw_dynamic     = $(elf_dir)/custom_opensbi_fw_dynamic.elf

# custom-opensbi
# Rust S-mode 可信管理程序 (嵌入到固件 .coffer_enclave_man 段)
FW_SRC_DIR     = $(third_party)/custom-opensbi
RUST_SMODE_DIR = $(third_party)/rust_smode_entry
FW_BUILD_DIR   = $(FW_SRC_DIR)/build/platform/generic/firmware
RUST_SMODE_BIN = $(RUST_SMODE_DIR)/rust_smode_entry.bin

# 调试器参数
pyargs = --ram-base=0x80000000 \
	--preload=$(zsbl_fsbl) \
	--hart=$(hart_num) \
	$(fw_payload)

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
fw_basic_flag := PLATFORM=generic
fw_basic_flag += PLATFORM_RISCV_XLEN=64
fw_basic_flag += PLATFORM_RISCV_ABI=lp64
fw_basic_flag += FW_SKIP_BSS_ZERO=1

## ---- fw_payload 构建 (内嵌 payload) ----
FW_MAKE_FLAGS := $(fw_basic_flag)
FW_MAKE_FLAGS += FW_PAYLOAD=y

## ---- fw_jump 构建 (无内嵌 payload, mret 后跳转到 FW_JUMP_ADDR) ----
FW_JUMP_FLAGS := $(fw_basic_flag)
FW_JUMP_FLAGS += FW_JUMP=y
FW_JUMP_FLAGS += FW_JUMP_ADDR=0x80200000

# ---- Linux 内核启动 ----
# fw_payload 模式: Linux Image 嵌入 OpenSBI, 经 M->S 移交直接启动 (ZSBL 设 coldboot_done=1)。
# fw_jump 模式: OpenSBI mret -> 0x80200000, 内核 Image 预载该地址, vmlinux 供调试符号。
LINUX_IMG        = $(linux_dir)/arch/riscv/boot/Image
LINUX_VMLINUX    = $(linux_dir)/vmlinux
fw_payload_linux = $(elf_dir)/custom_opensbi_fw_payload_linux.elf
fw_jump_elf      = $(elf_dir)/custom_opensbi_fw_jump.elf

# ---- initramfs / rootfs (debootstrap) ----
# ROOTFS_DIR: debootstrap 输出目录 (打包为 cpio.gz)。
# INITRD: 传给 emu-linux 的 initramfs 路径 (为空则不挂载 rootfs)。
#   用法: make emu-linux INITRD=$(INITRAMFS) hart_num=4 ram=2G
ROOTFS_DIR ?= $(third_party)/rootfs
INITRAMFS   = $(bins_dir)/initramfs.cpio.gz
INITRD     ?=
# DISK: virtio-blk 磁盘镜像 (ext4/raw), 挂载为 /dev/vda。默认指向 debootstrap 生成的
#   rootfs; 文件不存在时自动跳过 (回退到无根文件系统, 即 VFS panic)。root 所有的镜像
#   自动只读打开。覆盖: make emu-linux DISK=/path/to/other.ext4
DISK       ?= $(third_party)/setup-rootfs/debootstrap/riscv-sd.ext4
# ram / rdinit 可覆盖; 挂载 initramfs 时建议加大 RAM (全量驻留内存)。
ram        ?= 2G
rdinit     ?= /bin/bash
# Rust native 引擎编译期配置
TLB_ENTRIES ?= 256
# ---- Kernel bootargs ----
# dyndbg: dynamic debug 控制 (内核 pr_debug/dev_dbg), 可覆盖.
#   +p 启用全部, func NAME +p 按函数, file PATH +p 按文件, 留空关闭.
dyndbg         ?=
# func run_init_process +p
init           ?= /bin/zsh
bootargs_extra ?=
# norandmaps
# nokaslr norandmaps
_bootargs   = earlycon=sbi console=ttySIF0 nokaslr norandmaps
# _bootargs  += dyndbg="$(dyndbg)"
_bootargs  += root=/dev/vda rw
_bootargs  += $(if $(init),init=$(init))
_bootargs  += $(if $(INITRD),rdinit=$(rdinit)) $(bootargs_extra)
bootargs_arg    = --bootargs='$(_bootargs)'
# INITRD 非空时追加 --initrd (rdinit 已在 bootargs 中).
initrd_args     = $(if $(INITRD),--initrd=$(INITRD))
# DISK 指向的镜像存在时追加 --disk.
disk_args       = $(if $(wildcard $(DISK)),--disk=$(DISK))

# ---- 工具链参考 (供手动使用) ----
opt-cc      = /opt/custom-llvm/bin/clang
opt-src-dir = $(PWD)/tests/src/
opt-bin-dir = $(PWD)/$(bins_dir)/
