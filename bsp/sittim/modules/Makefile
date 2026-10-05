# SPDX-LICENSE-IDENTIFIER: MIT
# Author: <kisfg@hotmail.com> 2026
#
# 扩展模块映像的构建与签名。
#
# 模块映像是一份连续的二进制文件: 没有 ELF 头, 没有重定位项, 模块入口在文件偏移 0,
# 末尾 64 字节是 ECDSA (secp256r1) 签名 r‖s。签名者为 bsp/tee_aux_tools/sign-mod。
#
# 位置无关性由编译参数保证, 并由 check-pic 目标逐次核对:
#   relocation-model=pic 使取址按当前 PC 计算, 大型 switch 的分支表 (.LJTI) 随之取
#     32 位相对偏移;
#   code-model=medium 使取址按跨越范围超出单条跳转指令立即数所能表示的情形构造,
#     该范围的目标地址由 auipc 与后续指令两条构成, 同样按当前 PC 计算。
# 取 pic 而非 static: static 把符号按本模块内可见处理, 分支表内的目标由链接器就地
# 写成绝对地址, 该地址随加载基址失效, 且映像里不留重定位项。
#
# 编译参数写在本文件里, 不放进 .cargo/config.toml: 本目录在 bsp/sittim 之下, 而
# bsp/sittim/.cargo/config.toml 为同一个目标三元组设了 relocation-model=pie, cargo
# 对数组型 rustflags 是跨配置层合并而非覆盖。本文件经 RUSTFLAGS 传参, 该变量一经设置
# cargo 即不再读取配置里的 rustflags。本目录不含 .cargo/config.toml, 直接调用 cargo
# 会因继承的目标三元组不存在而报错, 不会静默产出一份参数不对的静态库。

CUSTOM_LLVM := /opt/custom-llvm/bin
LD          := /usr/bin/ld.lld
OBJCOPY     := $(CUSTOM_LLVM)/llvm-objcopy
READELF     := $(CUSTOM_LLVM)/llvm-readelf

# cargo 可执行文件; 调用方以绝对路径覆盖, 供 PATH 中无 cargo 的构建环境 (如经由 sudo
# 运行的 rootfs 构建) 使用.
CARGO       ?= cargo

TARGET      := riscv64imac-unknown-none-elf
HOST_TARGET := x86_64-unknown-linux-gnu
CARGO_OUT   := target/$(TARGET)/release

BUILD_DIR := build

# 部署目录: 各映像扁平放入, 并附上清单 modules.list. 清单所列的文件名相对该目录给出,
# 与映像在此目录中的名称一致. rootfs 侧的安装直接取本目录的内容.
DIST_DIR := dist

# 模块清单。新增一个模块即在 Cargo.toml 的 members 中登记包名, 并在此追加同名项。
MODULES := attest net fs

# 签名用私钥。公钥由签名者从私钥推出并打印, 其取值与 sittim 的 config.mk 中
# ATTEST_PUB_KEY 相同。
KEY       := ../../attest-keys/attest_priv.pem
SIGN_DIR  := ../../tee_aux_tools/sign-mod
SIGN_TOOL := $(SIGN_DIR)/target/$(HOST_TARGET)/release/sign-mod

MODULE_RUSTFLAGS := -C relocation-model=pic -C code-model=medium

# 位置无关性检查所用的第二个基址。检查时同一批目标文件分别以它与地址 0 链接。
PIC_CHECK_BASE := 0x80000000
PIC_CHECK_DIR  := $(BUILD_DIR)/pic-check

phony = all

# .elf 与 .bin 只作为中间产物供签名使用, 但留着可随时反汇编核对映像布局,
# 故不作删除。
.SECONDARY:

all: $(MODULES:%=$(BUILD_DIR)/%.signed.bin) check-pic
phony += all

# 位置无关性检查: 同一批目标文件以两个基址各链接一次, 两次的映像必须逐字节相同,
# 且不含重定位项。
check-pic: $(MODULES:%=$(BUILD_DIR)/%.bin)
	@mkdir -p $(PIC_CHECK_DIR)
	@for m in $(MODULES); do \
		$(LD) -T module.ld --image-base=$(PIC_CHECK_BASE) $(CARGO_OUT)/lib$$m.a -o $(PIC_CHECK_DIR)/$$m.elf; \
		$(OBJCOPY) -O binary --set-section-flags .bss=alloc,load,contents $(PIC_CHECK_DIR)/$$m.elf $(PIC_CHECK_DIR)/$$m.bin; \
		if ! cmp -s $(BUILD_DIR)/$$m.bin $(PIC_CHECK_DIR)/$$m.bin; then \
			echo "$$m: different binary file"; exit 1; \
		fi; \
		if $(READELF) -r $(PIC_CHECK_DIR)/$$m.elf | grep -q R_RISCV; then \
			echo "$$m: can not accept reloc program"; exit 1; \
		fi; \
	done
phony += check-pic

# 模块的静态库。增量由 cargo 自身保证, 该规则每次都调用它。
$(CARGO_OUT)/lib%.a: FORCE
	RUSTFLAGS='$(MODULE_RUSTFLAGS)' \
		$(CARGO) build --release --target $(TARGET) -p $*

$(BUILD_DIR)/%.elf: $(CARGO_OUT)/lib%.a module.ld
	@mkdir -p $(BUILD_DIR)
	$(LD) -T module.ld $< -o $@

$(BUILD_DIR)/%.bin: $(BUILD_DIR)/%.elf
	$(OBJCOPY) -O binary --set-section-flags .bss=alloc,load,contents $< $@

$(BUILD_DIR)/%.signed.bin: $(BUILD_DIR)/%.bin $(SIGN_TOOL)
	$(SIGN_TOOL) $(KEY) $< $@

dist: $(MODULES:%=$(BUILD_DIR)/%.signed.bin) modules.list check-pic
	rm -rf $(DIST_DIR)
	mkdir -p $(DIST_DIR)
	cp $(MODULES:%=$(BUILD_DIR)/%.signed.bin) modules.list $(DIST_DIR)/
phony += dist

# 签名工具是宿主程序, 在它自己的目录下构建: cargo 按当前目录向上寻找配置文件,
# 从本目录调用会读到 sittim 的目标三元组与模块的编译参数。
$(SIGN_TOOL): FORCE
	cd $(SIGN_DIR) && $(CARGO) build --release --target $(HOST_TARGET)

clean:
	rm -rf $(BUILD_DIR) $(DIST_DIR)
	cargo clean
phony += clean

.PHONY: $(phony) FORCE
