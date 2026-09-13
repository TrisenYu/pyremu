# examples/easy_kern/Makefile — 构建 S 模式裸机 host 示例 (easy_kern)
#
# easy_kern 运行于 OpenSBI fw_jump 之上, 无需 Linux: 直接以 SBI ecall
# 演示飞地 CREATE/ENTER/RESUME 生命周期, 用于在仿真器上快速验证载荷
# 进入运行时与 main().
#
# 构建流程:
#   1. 把 $(PAYLOAD) (默认 fn_apps/bin/cfrac 静态 ELF) 拷为 payload.bin
#   2. clang 交叉编译 easy_kern.c / entry.S / payload_embed.S (.incbin 嵌入)
#   3. ld.lld 按 easy_kern.ld 链接到 0x80200000
#   4. llvm-objcopy 转为 raw 二进制 easy_kern.bin
#
# 产物 easy_kern.bin 由仿真器 --preload 到 0x80200000, 冷启动
# build/elf/custom_opensbi_fw_jump.elf (PIE 自 0x80000000 展开) 后跳转进入.
#
# 用法:
#   make                 # 构建 (默认内嵌 ../../fn_apps/bin/cfrac)
#   make PAYLOAD=<elf>   # 指定其他载荷 (需是运行时可装载的静态 ELF)
#   make clean
#
# 运行 (仓库根目录):
#   timeout 300 env PYTHONPATH=. uv run python examples/demo_easy_kern.py

PAYLOAD ?= ../../fn_apps/bin/cfrac

BUILD := build
CC    := clang
LD    := ld.lld
OBJCP := llvm-objcopy

# 与仓库其余 RISC-V 交叉目标一致 (riscv64-linux-musl). easy_kern 为整型裸机
# 代码, 用 rv64imac/lp64; 载荷仅作数据内嵌, 不参与本目标编译. -mcmodel=medany
# 使链接地址 0x80200000 附近的访问转成 PC 相对, 配合 -fno-pic.
TARGET := riscv64-linux-musl
ARCH   := -march=rv64imac -mabi=lp64
CFLAGS := $(ARCH) -mcmodel=medany -ffreestanding -fno-builtin \
          -fno-stack-protector -fno-pic -O2

OBJS := $(BUILD)/entry.o $(BUILD)/easy_kern.o $(BUILD)/payload_embed.o

.PHONY: all clean

all: easy_kern.bin

# .incbin 相对当前目录展开, 故把载荷拷到本目录下
payload.bin: $(PAYLOAD)
	cp $(PAYLOAD) payload.bin

easy_kern.elf: $(OBJS) easy_kern.ld
	$(LD) -T easy_kern.ld -nostdlib $(OBJS) -o $@

# raw 二进制自 0x80200000 起连续展开 (.bss 与栈不在文件中, 由 entry.S 处理)
easy_kern.bin: easy_kern.elf
	$(OBJCP) -O binary easy_kern.elf $@
	@echo "  -> $@  (payload: $(PAYLOAD))"

$(BUILD)/entry.o: entry.S
	@mkdir -p $(@D)
	$(CC) --target=$(TARGET) $(ARCH) -c $< -o $@

$(BUILD)/easy_kern.o: easy_kern.c
	@mkdir -p $(@D)
	$(CC) --target=$(TARGET) $(CFLAGS) -c $< -o $@

$(BUILD)/payload_embed.o: payload_embed.S payload.bin
	@mkdir -p $(@D)
	$(CC) --target=$(TARGET) $(ARCH) -c $< -o $@

clean:
	rm -rf $(BUILD) payload.bin easy_kern.elf easy_kern.bin
