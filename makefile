# Last modified at 2026/07/14 星期二 11:02:21
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

# 全部变量定义集中在 configs.mk (路径/固件 flags/DISK/INITRD/ram 等)。
include configs.mk


# CARGO_FLAGS 变化时强制重建 native .so (需求: PYREMU_TRACE_SRET / PYREMU_DIAG_LOG
# 设为非零/非空值时启用 --features diagnostic, 否则不启用).
# 若仅依赖 $(NATIVE_SRC) 时间戳, 修改环境变量后 make 不会重建 → 旧的
# (不含 diagnostic) .so 被继续使用 → 诊断日志静默缺失.
NATIVE_CONFIG_GEN = $(NATIVE_DIR)/cpu/src/config_gen.rs

# 由 configs.mk 生成 Rust 常量定义 — 照搬 rust_smode_entry/Makefile 模式.
$(NATIVE_CONFIG_GEN): configs.mk makefile
	@echo '// generated from configs.mk by Makefile — do not edit' > $@
	@echo 'pub const TLB_ENTRIES: usize = $(TLB_ENTRIES);' >> $@

$(NATIVE_SO) $(TERMIO_SO): $(NATIVE_SRC) $(NATIVE_CONFIG_GEN)
	cargo build --release --manifest-path $(NATIVE_DIR)/Cargo.toml --workspace $(CARGO_FLAGS)
	install -m 755 $(NATIVE_DIR)/target/release/libdecode.so $(NATIVE_SO)
	install -m 755 $(NATIVE_DIR)/target/release/libtermio.so $(TERMIO_SO)
	@echo "libdecode.so + libtermio.so 已构建"

build-native: $(NATIVE_CONFIG_GEN)
	cargo build --release --manifest-path $(NATIVE_DIR)/Cargo.toml --workspace $(CARGO_FLAGS)
	install -m 755 $(NATIVE_DIR)/target/release/libdecode.so $(NATIVE_SO)
	install -m 755 $(NATIVE_DIR)/target/release/libtermio.so $(TERMIO_SO)
	@echo "libdecode.so + libtermio.so 已构建"
phony += build-native

clean-native:
	rm -f $(NATIVE_SO) $(TERMIO_SO)
	rm -rf $(NATIVE_DIR)/target
phony += clean-native

FORCE:
phony += FORCE


# ---- Rust S-mode Runtime ----
# 编译 rust_smode_entry.bin, 供 custom-opensbi 嵌入 .coffer_enclave_man 段
$(RUST_SMODE_BIN): $(wildcard $(RUST_SMODE_DIR)/src/*.rs) $(wildcard $(RUST_SMODE_DIR)/src/**/*.rs)
	$(MAKE) -C $(RUST_SMODE_DIR) build

build-rust: $(RUST_SMODE_BIN)
phony += build-rust


$(FW_BUILD_DIR)/fw_payload.elf: $(RUST_SMODE_BIN) $(FW_SRC_DEPS)
	$(MAKE) -C $(FW_SRC_DIR) distclean
	$(MAKE) -C $(FW_SRC_DIR) $(FW_MAKE_FLAGS) -j$$(nproc)

# 便捷别名: 显式请求固件重新编译 (distclean + 全量)
build-fw: $(FW_BUILD_DIR)/fw_payload.elf
phony += build-fw


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
	python -m pyremu.debugger --hart-logs=output/ $(pyargs)
phony += emu

$(fw_payload_linux): $(RUST_SMODE_BIN) $(FW_SRC_DEPS)
	@echo "构建 custom-opensbi + Linux 内核 payload..."
	$(MAKE) -C $(FW_SRC_DIR) distclean $(FW_MAKE_FLAGS) \
		FW_PAYLOAD_PATH=$(realpath $(LINUX_IMG))
	$(MAKE) -C $(FW_SRC_DIR) $(FW_MAKE_FLAGS) \
		FW_PAYLOAD_PATH=$(realpath $(LINUX_IMG)) -j$$(nproc)
	cp $(FW_BUILD_DIR)/fw_payload.elf $(fw_payload_linux)
	cp $(FW_BUILD_DIR)/fw_payload.bin $(fw_payload_linux:.elf=.bin)
	@echo "凭linux作为载荷的固件已拷贝: $(fw_payload_linux)"

build-fw-linux: $(fw_payload_linux)
phony += build-fw-linux

# fw_payload 模式 (Linux 内嵌 OpenSBI payload) — 备用; 主用 emu-linux (fw_jump)。
# 嵌入的是vmlinux，会缺乏调试符号
emu-linux-payload: $(zsbl_fsbl) $(fw_payload_linux)
	python -m pyremu.debugger \
		--hart-logs=output/ --ram-base=0x80000000 --ram 512M --preload=$(zsbl_fsbl) \
		--harts=$(hart_num) $(fw_payload_linux)
phony += emu-linux-payload


# 主入口: fw_jump 模式启动 Linux。默认挂载 DISK 指向的 rootfs (存在时)。
#   用法: make emu-linux hart_num=4 ram=2G [DISK=... | INITRD=...]
emu-linux: $(zsbl_fsbl) $(fw_jump_elf) build-native $(if $(INITRD),$(INITRAMFS)) $(PLATFORM_CONFIG_GEN)
	python -m pyremu.debugger \
		--hart-logs=output/ --ram-base=0x80000000 --ram $(ram) --preload=$(zsbl_fsbl) \
		--kernel=$(LINUX_IMG) --sym=$(LINUX_VMLINUX) \
		$(initrd_args) $(disk_args) $(bootargs_arg) \
		--harts=$(hart_num) $(fw_jump_elf)
phony += emu-linux

# 最小 init 验证内核能否走到执行 init 阶段
emu-linux-sh: $(zsbl_fsbl) $(fw_jump_elf) build-native $(PLATFORM_CONFIG_GEN)
	PYTHON_GIL=0 python -m pyremu.debugger \
		--hart-logs=output/ --ram-base=0x80000000 --ram $(ram) --preload=$(zsbl_fsbl) \
		--kernel=$(LINUX_IMG) --sym=$(LINUX_VMLINUX) --fdt -1 \
		$(disk_args) $(initrd_args) $(bootargs_arg) \
		--harts=$(hart_num) $(fw_jump_elf)
phony += emu-linux-sh

# ---- initramfs 打包 (debootstrap 目录 -> newc cpio + gzip) ----
# 用 fakeroot (若可用) 保留属主与 /dev 节点。
$(INITRAMFS):
	@test -d $(ROOTFS_DIR) || { echo "缺少 rootfs 目录: $(ROOTFS_DIR) (先运行 debootstrap)"; exit 1; }
	cd $(ROOTFS_DIR) && \
	  if command -v fakeroot >/dev/null 2>&1; then FAKE=fakeroot; else FAKE=; echo "[warn] 无 fakeroot, 属主/设备节点可能丢失"; fi; \
	  find . -print0 | $$FAKE cpio --null -o --format=newc 2>/dev/null | gzip -9 > $(CURDIR)/$(INITRAMFS)
	@echo "initramfs 已生成: $(INITRAMFS) ($$(du -h $(INITRAMFS) | cut -f1))"

build-initramfs: $(INITRAMFS)
phony += build-initramfs

# ---- 测试 ----
# 限制: 4 GiB 虚拟内存 (防止 batch 测试内存膨胀), --ignore 排除 ordering-dependent 失败.
# 详见 memory/test-constraints.md 与 memory/ordering-dependent-native-batch-failures.md.
test:
	ulimit -v 4194304 && PYTHON_GIL=0 pytest -x --ignore=tests/test_multihart_diff.py
phony += test

cov-test:
	ulimit -v 4194304 && timeout=300 PYTHON_GIL=0 pytest -x \
	--cov=. --cov-report=term --full-trace
phony += cov-test

.PHONY: $(phony)

# 由 configs.mk 生成 Python 常量 — 集中管理模拟器全部编译期配置.
PLATFORM_CONFIG_GEN = pyremu/configs_gen.py
$(PLATFORM_CONFIG_GEN): configs.mk makefile
	@echo '# generated from configs.mk by Makefile — do not edit' > $@
	@echo '# mem layout' >> $@
	@echo 'RESERVED_MEM_BASE = $(RESERVED_MEM_BASE)' >> $@
	@echo 'RESERVED_MEM_SIZE = $(RESERVED_MEM_SIZE)' >> $@
	@echo '' >> $@
	@echo '# arguments for emulator' >> $@
	@echo 'TLB_ENTRIES = $(TLB_ENTRIES)' >> $@
	@echo 'NATIVE_MAX_INSTRS = $(NATIVE_MAX_INSTRS)' >> $@
	@echo 'TRAP_LOOP_THRESHOLD = $(TRAP_LOOP_THRESHOLD)' >> $@
	@echo '' >> $@
	@echo '# diagnostic variables' >> $@
	@echo 'PYREMU_NATIVE_BATCH = $(PYREMU_NATIVE_BATCH)' >> $@
	@echo 'PYREMU_DIAG_LOG = "$(PYREMU_DIAG_LOG)"' >> $@
	@echo 'PYREMU_DIAG_VERBOSE = $(PYREMU_DIAG_VERBOSE)' >> $@
	@echo 'PYREMU_TRACE_SRET = $(PYREMU_TRACE_SRET)' >> $@
	@echo 'PYREMU_TRACE_TRAPS = $(PYREMU_TRACE_TRAPS)' >> $@
	@echo 'PYREMU_TRACE_PMP = $(PYREMU_TRACE_PMP)' >> $@
