# pyremu-args.mk — 传给模拟器的启动参数

# 零阶段引导程序: 裸二进制装入引导只读存储器的基址, 设备树随之放在基址加
# ROM_DTB_OFFSET, 见 mk/boot-rom.mk. 各启动目标共用本组参数.
zsbl_args = --preload=$(kei-sav) --preload-addr=$(ROM_BASE)

# make emu 传给 pyremu.debugger 的参数: 内存基址、零阶段引导程序与固件映像.
# hart 数量取自 NUM_HARTS, 见 mk/processor.mk.
pyargs = --ram-base=$(RAM_BASE) \
	$(zsbl_args) \
	--harts=$(NUM_HARTS) \
	$(fw_payload)
