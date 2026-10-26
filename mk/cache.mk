# cache.mk — L2 缓存配置
# 生成到 pyremu/configs_gen.py, 作为 PlatformConfig 的字段默认值.

# L2 缓存大小，单位是字节
L2_SIZE              ?= 0x40000

# 不超过阈值的，由模拟器经过 L2 缓存载入, 否则直接写入 RAM
FAST_LOAD_THRESHOLD  ?= 0x200000
