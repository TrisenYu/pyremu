# emu-speedup.mk — 加速执行库
#
# Rust 侧编译为 cdylib, 生成 libdecode.so 与 libtermio.so, 由 Python 经 ctypes 加载.

NATIVE_DIR     = pyremu/_native
NATIVE_SO      = $(NATIVE_DIR)/libdecode.so
TERMIO_SO      = $(NATIVE_DIR)/libtermio.so
CARGO_FLAGS    ?=
NATIVE_SRC = $(shell find $(NATIVE_DIR)/cpu $(NATIVE_DIR)/termio -type f -name '*.rs') \
             $(NATIVE_DIR)/Cargo.toml $(NATIVE_DIR)/cpu/Cargo.toml $(NATIVE_DIR)/termio/Cargo.toml
