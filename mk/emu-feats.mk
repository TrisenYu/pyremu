# emu-feats.mk — 模拟器特性与运行参数

# 生成到 pyremu/configs_gen.py, 作为 PlatformConfig 的字段默认值.

# 调试器 run/continue 的时钟源超时 (秒), 0 = 禁用
DEFAULT_TIMEOUT      ?= 36000

# virtio 单次 _process_queue 最多处理描述符数, 拆小批避免长时间占用主线程
VIRTIO_PROCESS_BATCH ?= 16

## 特性选项配置
# 1 = AIA (IMSIC+APLIC); 0 = legacy PLIC
PYREMU_AIA           ?= 0
# 1 = H-extension (HS/VS modes + hgatp + virtual interrupts)
PYREMU_H_EXT         ?= 0
# 1 = 关闭 L2 缓存: RAM 访问全部绕过 L2 缓存
PYREMU_NO_L2         ?= 0
# 1 = 关闭 TLB: 查询与插入一并跳过, 每次地址翻译都完整遍历 Sv39 页表
PYREMU_NO_TLB        ?= 0
# 1 = 带 marker 的 ebreak 由模拟器内联服务, 固件输出写标准错误
# 0 = 不拦截该序列, ebreak 按普通断点投递, 固件控制台回落到 UART 并写入每 hart 的日志
#     以 SYS_EXIT 停机序列收尾的用例随之无法结束连续执行
PYREMU_HAS_SEMIHOSTING ?= 0
