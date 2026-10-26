# memory.mk — 主存布局

# 生成到 pyremu/configs_gen.py, 作为 PlatformConfig 的字段默认值.

# 物理 RAM 起始地址与大小 (字节)
RAM_BASE             ?= 0x80000000
RAM_SIZE             ?= 0x08000000
# 上电复位入口地址
PROG_CNT             ?= 0x80000000

# 供特定固件代码实现使用, 需与 custom-opensbi Kconfig (POOL_BASE/POOL_SIZE) 保持同步.
# DTB /reserved-memory no-map 节点据此生成, 确保 Linux 内核线性映射排除该区域,
# 避免内核分配器与固件在同一物理区间内产生访问冲突.
RESERVED_MEM_BASE    ?= 0x83000000
# 2.25 GiB. 须容纳 stress_ng 的 --vm-bytes 2G 档位: 该档位自身占用 2 GiB, 另需
# 载荷镜像、argv 与 M 模式按 2 MiB 分区取整所需的余量. 池尾为 0x113000000
# (~4.30 GiB), 故 ram 须不小于 6G, 以留出内核与根文件系统的空间.
RESERVED_MEM_SIZE    ?= 0x90000000

# 固件 (custom-opensbi) 载入 ram_base, 其映像尾部内嵌 .sittim 飞地运行时
# (物理地址 0x80180000, 长 0x202F68 字节: 低 2 MiB 放代码与只读数据, 高 2 MiB 放
# 可写数据). 自 ram_base 起直至内核载入地址 FW_JUMP_ADDR (0x80400000) 的整个区间
# 都必须保留: 否则内核分配器会把页面分到固件映像上, 设备 DMA (如 virtio-blk) 会
# 直接覆写 .sittim, 使 CREATE 复制进飞地池的载荷变成垃圾数据, 飞地入口立即取指
# 故障并被 M 模式静默挂起.
FW_RESERVED_MEM_BASE ?= 0x80000000
# 4 MiB: 固件映像及其后 .sittim 的末尾止于 0x80382F68, 上界取下一个 2 MiB 边界.
FW_RESERVED_MEM_SIZE ?= 0x400000
