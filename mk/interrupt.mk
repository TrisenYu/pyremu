# interrupt.mk — 中断控制器配置

# 生成到 pyremu/configs_gen.py, 作为 PlatformConfig 与 PeripheralConfig 的字段默认值.

## ---- 中断控制器相关
# IMSIC M-file MMIO 基址
IMSIC_M_BASE         ?= 0x24000000
# IMSIC S-file MMIO 基址
IMSIC_S_BASE         ?= 0x28000000
# APLIC S 域 MMIO 基址
APLIC_S_BASE         ?= 0x0C000000
# APLIC M 域 MMIO 基址
APLIC_M_BASE         ?= 0x0C008000

CLINT_BASE           ?= 0x02000000
PLIC_BASE            ?= 0x0C000000
