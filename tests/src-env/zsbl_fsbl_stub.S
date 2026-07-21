# =============================================================================
# ZSBL (Zero Stage Boot Loader) + FSBL (First Stage Boot Loader) 微型实现
# =============================================================================
#
# 目的: 在 OpenSBI 固件镜像之前运行, 完成最少的硬件初始化后跳转到
# OpenSBI 的 _start_warm 入口。ZSBL 是处理器上电后执行的第一段代码,
# FSBL 负责设置 mtvec、传递 hart ID 与 FDT 指针。
#
# 内存布局假设 (与 pyremu emulator 约定一致):
#   RAM_BASE            由 makefile 通过 -D 传入 (默认 0x8000_0000)
#   FDT blob            放置在 _start + 0x10000 (PC-relative, 与注入位置无关)
#
# 地址来源:
#   RAM_BASE             — 由 -DRAM_BASE=$(RAM_BASE) 传入
#   FDT                  — lla 伪指令 PC-relative, 与注入位置自动适配
#
# 流程:
#   硬件上电 -> ZSBL (_start) -> 设置 a1=FDT, 跳转 FSBL ->
#   设置 mtvec -> 读取 mhartid -> unimp 触发 trap ->
#   mtvec 指向固件 _start -> AMOSWAP lottery -> 正常固件启动流程
#
# 关键设计决策: ZSBL 不再预设 coldboot_done=1.
#   多核场景下 coldboot_done 必须由固件的 init_coldboot() 在完成
#   sbi_domain_init() 后设置, 否则 warm boot hart 会在 domain 就绪前
#   进入 init_warmboot -> sbi_hsm_hart_get_state -> sbi_hart_hang.
#   单核场景下 AMOSWAP lottery 仅有一个竞争者, 同样正确.
# =============================================================================

.section .text, "ax", @progbits
.globl _start

# ---------------------------------------------------------------------------
# ZSBL — 入口。仅设置 FDT 指针后移交 FSBL。
# a1 = FDT 物理地址。OpenSBI 在 fw_platform_init() 中通过 a1 读取设备树,
# 约定 FDT 放置于 _start + 64 KiB 处。
# ---------------------------------------------------------------------------
_start:
	lla	a1, _start + 0x10000    # a1 <- FDT blob 物理地址
	j	fsbl_start              # 移交 FSBL

# ---------------------------------------------------------------------------
# FSBL — 最小初始化序列。
# 1. 将 mtvec 指向固件入口, 确保 trap 可处理
# 2. 读取 mhartid -> a0 (OpenSBI 期望 a0 = hart ID)
# 3. 重新加载 FDT 地址 -> a1
# ---------------------------------------------------------------------------
fsbl_start:
	# --- step 1: 设置 M 模式 trap 向量 ---
	# OpenSBI 在 _start 中将 mtvec 设为 _start_warm, 但 ZSBL
	# 直接从此处开始执行, 故需手动设置 mtvec 指向固件基址。
	# csrrw x0, mtvec, t0 即 mtvec <- t0 (旧值丢弃到 x0)。
	li	t0, RAM_BASE             # t0 <- ram_base (makefile 注入)
	csrrw	x0, mtvec, t0        # mtvec <- ram_base

	# --- step 2: 读取 hart ID ---
	# OpenSBI sbi_init() 期望 a0 = hart ID (mhartid CSR)。
	# csrrs a0, mhartid, x0 即 a0 <- mhartid (原子 read-set, 写回 0=no-op)。
	csrrs	a0, mhartid, x0        # a0 <- mhartid

	# --- step 3: 重新加载 FDT 地址 ---
	lla	a1, _start + 0x10000    # a1 <- FDT 地址 (再次确保)

# ---------------------------------------------------------------------------
# COLD_BOOT 宏 — 调试用自旋栅栏 (已弃用)
#
# COLD_BOOT 宏已移除 — 固件地址在重编译后发生变化, 硬编码 patch
# 地址不再可靠。单 hart 模式下 AMOSWAP lottery 始终由 hart 0 胜出,
# 多 hart 模式下固件的 _wait_for_boot_hart 自旋 + init_coldboot 设置
# coldboot_done 的流程已正确处理同步, 不再需要 ZSBL 预设此标志.
# ---------------------------------------------------------------------------
	unimp                         # 不应到达; 若到达则触发 Illegal Instruction trap
