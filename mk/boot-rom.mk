# boot-rom.mk — 引导只读存储器配置

# 生成到 pyremu/configs_gen.py, 作为 PlatformConfig 的字段默认值.

# 基址存放零阶段引导程序; 设备树位于基址加 ROM_DTB_OFFSET, 与零阶段引导程序
# 按 _start + 0x10000 算出的交付地址一致. PlatformConfig.rom_base 取 0 时不注册该设备.
ROM_BASE        ?= 0x40000000
ROM_SIZE        ?= 0x20000
ROM_DTB_OFFSET  ?= 0x10000
