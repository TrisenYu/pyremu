# SPDX-LICENSE-IDENTIFIER: MIT
# Author: <kisfg@hotmail.com> 2026
#
# 扩展模块映像的构建与签名。
#
# 模块映像是一份连续的二进制文件: 没有 ELF 头, 没有重定位项, 末尾 64 字节是 ECDSA
# (secp256r1) 签名 r‖s。签名者为 bsp/tee_aux_tools/sign-mod。
#
# 映像偏移 0 处是 64 字节的映像描述块, 由 module.ld 的段内数据命令生成: 它声明模块入口在
# 映像内的偏移、映像本体的字节数与可写静态数据的起始偏移。取入方按声明的偏移调用入口, 并在
# 调用之前逐项核对描述块, 不符即拒绝, 见 src/ext_mod/img_descriptor.rs。入口偏移自此不为 0,
# 故按旧约定构建的映像与本批映像互不兼容, modules/ 与固件的重建须同批进行。
#
# 映像中初值非零的部分即描述块、.text 与 .rodata, 由 objcopy 连续写出, 映像本体的末尾由
# module.ld 的 _image_body_end 给出。可写的静态数据须全部落在 BSS 段, 且该段的起始地址须等于
# 映像本体按页取整的结果: 取入方自映像起点至该边界按指令类映射为可读可执行, 槽位其余部分按
# 数据类映射为可读写, 故落在可执行部分之内的可写数据在装入后不可写。BSS 段不含初值, 因而不
# 写入映像文件。这两项约束由 module.ld 强制: 该脚本把 BSS 段的起始地址对齐到页边界, 并在
# .data 段非空时中断链接。
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

# 私钥路径与其余平台量取自 bsp/sittim 的配置; 其中的路径以 bsp/sittim 为基准, 本文件
# 按目录关系加一级前缀, 见 KEY。
include ../config.mk

TARGET      := riscv64imac-unknown-none-elf
HOST_TARGET := x86_64-unknown-linux-gnu
CARGO_OUT   := target/$(TARGET)/release

BUILD_DIR := build

# 部署目录: 各映像放入, rootfs 侧的安装直接取本目录的内容. 映像文件名相对该目录给出,
# 与宿主侧模块表所登记的名称一致, 见 MOD_TBL_INC.
DIST_DIR := dist

# 模块清单。新增一个模块即在 Cargo.toml 的 members 中登记包名, 并在此追加同名项; 宿主侧
# 模块表由 mod-tbl 目标按部署目录中的映像生成, 不需要在别处登记, 但每个模块须在其
# src/lib.rs 中声明恰一处 `const MODULE_ID: u32 = <十进制数>;`。
MODULES := attest net fs

# 宿主侧模块表。宿主工具的 C 源以引号包含直接取用, 故生成物落在该工具的源码目录, 见
# bsp/tee_aux_tools/ext-mod-svc/tee_ext_mod_svc.c。
MOD_TBL_INC := ../../tee_aux_tools/ext-mod-svc/ext_mod_table.inc

# 签名用私钥。公钥由签名者从私钥推出并打印, 其取值与 sittim 的 config.mk 中
# ATTEST_PUB_KEY 相同。
KEY       := ../$(ATTEST_PRIV_KEY_PATH)
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
		$(OBJCOPY) -O binary $(PIC_CHECK_DIR)/$$m.elf $(PIC_CHECK_DIR)/$$m.bin; \
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
	$(OBJCOPY) -O binary $< $@

$(BUILD_DIR)/%.signed.bin: $(BUILD_DIR)/%.bin $(SIGN_TOOL)
	$(SIGN_TOOL) $(KEY) $< $@

dist: $(MODULES:%=$(BUILD_DIR)/%.signed.bin) check-pic
	rm -rf $(DIST_DIR)
	mkdir -p $(DIST_DIR)
	cp $(MODULES:%=$(BUILD_DIR)/%.signed.bin) $(DIST_DIR)/
phony += dist

# 模块表各项的名字与次序: 取部署目录中的映像文件名。该目录由 dist 目标重建, 其中只有本次
# 部署的映像, 故按 .signed.bin 取即可; 按名排序使同一批映像生成同一份表。
DIST_MODS = $(sort $(patsubst $(DIST_DIR)/%.signed.bin,%,$(wildcard $(DIST_DIR)/*.signed.bin)))

# 模块编号: 取该模块源码中形如 `const MODULE_ID: u32 = <十进制数>;` 的声明。
mod_id = $(shell sed -n 's/^const MODULE_ID: u32 = \([0-9][0-9]*\);$$/\1/p' $1/src/lib.rs)

# 各模块的编号, 次序同 DIST_MODS。
DIST_MOD_IDS = $(foreach m,$(DIST_MODS),$(call mod_id,$(m)))

# 校验一个模块的编号声明: 须恰有一处。
check_mod_id = $(if $(filter 1,$(words $(call mod_id,$1))),,$(error $1/src/lib.rs: MODULE_ID declaration count is $(words $(call mod_id,$1)), want exactly 1))

# 模块表的一项。
mod_row = { $(call mod_id,$1), "$1.signed.bin" },

# 模块表的首注释。
define MOD_TBL_HDR
/* ext_mod_table.inc - 扩展模块表, 由 tee_ext_mod_svc.c 在编译期包含.
 *
 * 本文件由 bsp/sittim/modules/Makefile 的 mod-tbl 目标生成, 不手工编辑: 各项取部署目录 dist
 * 中的映像文件, 按模块名排序, 模块编号取自该模块源码中 MODULE_ID 的声明.
 *
 * 每项为一对 "<模块编号>, \"<映像文件名>\"". 映像文件名相对模块目录给出, 与
 * bsp/sittim/modules 的 dist 目标所产出的名称一致; 完整路径由取入方把模块目录与该
 * 文件名拼接得到, 故本文件不含目录.
 *
 * 模块编号与映像声明的编号不一致时飞地的取入以编号不符失败, 见 sittim 的
 * ext_mod::LoadError::IdMismatch; 宿主侧的 tee_ext_mod_serve 按请求中的编号查表, 查不到时
 * 以字节数 0 应答, 飞地据此判该模块不存在.
 */
endef

# 宿主侧模块表: 按部署目录中的映像生成, 故以部署为前置。写文件之前先校验各模块的编号声明,
# 不符即中止, 原有的表保持不变。
mod-tbl: dist
	$(foreach m,$(DIST_MODS),$(call check_mod_id,$(m)))
	$(if $(filter $(words $(DIST_MODS)),$(words $(sort $(DIST_MOD_IDS)))),,$(error duplicate module id: $(DIST_MOD_IDS)))
	$(file >$(MOD_TBL_INC),$(MOD_TBL_HDR))
	$(foreach m,$(DIST_MODS),$(file >>$(MOD_TBL_INC),$(call mod_row,$(m))))
phony += mod-tbl

# 签名工具是宿主程序, 在它自己的目录下构建: cargo 按当前目录向上寻找配置文件,
# 从本目录调用会读到 sittim 的目标三元组与模块的编译参数。
$(SIGN_TOOL): FORCE
	cd $(SIGN_DIR) && $(CARGO) build --release --target $(HOST_TARGET)

clean:
	rm -rf $(BUILD_DIR) $(DIST_DIR)
	cargo clean
phony += clean

.PHONY: $(phony) FORCE
