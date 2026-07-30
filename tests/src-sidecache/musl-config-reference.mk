# config.mak — pyremu enclave musl cross-build for riscv64
ARCH = riscv64
CC = /opt/custom-llvm/bin/clang
CFLAGS = --target=riscv64-linux-musl -march=rv64imac -fuse-ld=lld
CROSS_COMPILE =
AR = /opt/custom-llvm/bin/llvm-ar
RANLIB = /opt/custom-llvm/bin/llvm-ranlib
