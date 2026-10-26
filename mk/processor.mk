# processor.mk — 处理器配置

# 生成到 pyremu/configs_gen.py,
# 作为 PlatformConfig 与 PeripheralConfig 的字段默认值.
# 调试器命令行给出的 --harts / --ram / --ram-base 等参数优先于这些默认值.

## 处理器配置
# ISA 字符串
ISA                  ?= rv64imacfd_sstc_zicsr_zifencei
# hart 数量
NUM_HARTS            ?= 3
# 通过 makefile 生成 pyremu/configs_gen.py, 替代各处硬编码与 os.environ.get.
# TLB 条目数
TLB_ENTRIES          ?= 256
# PMP 条目数. 固件在启动时按可访问的 pmpaddr / pmpcfg 探测该数目，同时可能影响固件的PMP_COUNT
PMP_ENTRIES          ?= 16
# mtime/mcycle 随宿主机时间推进
CPU_FREQ_HZ          ?= 1000000000

# 时钟源的频率，单位Hz.
# 由time CSR 与 CLINT mtime所使用，模拟器按指令数推进对 mtime 的换算、
# DTB 的 timebase-frequency 与飞地运行时的定时换算共用该取值.
TIMER_FREQ           ?= 100000

## TODO: 需要去除这一定义上非常奇怪的变量
# WFI 空转单次睡眠上限，单位s
WFI_MAX_SLEEP        ?= 0.2
# WFI 全 idle 无定时器时固定时长轮询间隔，单位ms
WFI_WATCHDOG_MS      ?= 5

# mtime 指令计数源的每指令纳秒数
NS_PER_INSTR         ?= 20
