# peripherals.mk — 外设 MMIO 基址
#
# 经 mk/mods.mk 导入, 最终生成到 pyremu/configs_gen.py, 作为 PeripheralConfig 的
# 字段默认值. 基址取 0 表示该设备不注册.

UART_BASE            ?= 0x10000000
SPI_BASE             ?= 0 # 0x10001000
I2C_BASE             ?= 0 # 0x10002000
GPIO_BASE            ?= 0 # 0x10003000
WATCHDOG_BASE        ?= 0x10004000
CRNG_BASE            ?= 0 # 0x10006000
VIRTIO_BLK_BASE      ?= 0
VIRTIO_NET_S_BASE    ?= 0
VIRTIO_NET_M_BASE    ?= 0
