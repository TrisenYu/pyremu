# config.mak - pyremu enclave musl cross-build for riscv64
ARCH = riscv64
CC = clang
CFLAGS = --target=riscv64-linux-musl -march=rv64imac -fuse-ld=lld
CROSS_COMPILE =
AR = llvm-ar
RANLIB = llvm-ranlib
