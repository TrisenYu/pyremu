# =============================================================================
# ZSBL (Zero Stage Boot Loader) + FSBL (First Stage Boot Loader) 微型实现
# =============================================================================
#
# 目的: 在 OpenSBI 固件镜像之前运行, 完成最少的硬件初始化后跳转到
# OpenSBI 的 _start_warm 入口。ZSBL 是处理器上电后执行的第一段代码,
# FSBL 负责设置 mtvec、传递 hart ID 与 FDT 指针并标记冷启动已完成。
#
# 内存布局假设 (与 pyremu emulator 约定一致):
#   RAM_BASE            由 makefile 通过 -D 传入 (默认 0x8000_0000)
#   COLD_BOOT_DONE_ADDR 由 makefile 从固件 ELF 符号表提取后经 -D 传入
#   FDT blob            放置在 _start + 0x10000 (PC-relative, 与注入位置无关)
#
# 地址来源:
#   RAM_BASE             — 由 -DRAM_BASE=$(RAM_BASE) 传入
#   COLD_BOOT_DONE_ADDR  — 由 makefile 通过 llvm-readelf -s 提取并加 RAM_BASE
#   FDT                  — lla 伪指令 PC-relative, 与注入位置自动适配
#   (COLD_BOOT 宏已弃用 — 单 hart 模式下无需修补 sbi_init)
#
# 流程:
#   硬件上电 -> ZSBL (_start) -> 设置 a1=FDT, 跳转 FSBL ->
#   标记 coldboot_done=1 -> 设置 mtvec -> 读取 mhartid -> 修补 sbi_init
#   (可选, COLD_BOOT) -> 最终由外部跳转到 OpenSBI _start_warm
# =============================================================================

.section .text, "ax", @progbits
.globl _start

# ---------------------------------------------------------------------------
# ZSBL — 入口。仅设置 FDT 指针后移交 FSBL。
# a1 = FDT 物理地址。OpenSBI 在 fw_platform_init() 中通过 a1 读取设备树,
# 约定 FDT 放置于 _start + 64 KiB 处。
# ---------------------------------------------------------------------------
_start:
	lla	a1, _start + 0x10000    # a1 ← FDT blob 物理地址
	j	fsbl_start              # 移交 FSBL

# ---------------------------------------------------------------------------
# FSBL — 最小初始化序列。
# 1. 设置 coldboot_done=1, 通知 OpenSBI 跳过 hart lottery
# 2. 将 mtvec 指向固件入口, 确保 trap 可处理
# 3. 读取 mhartid -> a0 (OpenSBI 期望 a0 = hart ID)
# 4. 重新加载 FDT 地址 -> a1
# ---------------------------------------------------------------------------
fsbl_start:
	# --- step 1: 标记冷启动完成 ---
	# coldboot_done 是 OpenSBI fw_base.S 中定义的 1 字节 flag,
	# 地址由 makefile 从固件 ELF 符号表提取并通过 -DCOLD_BOOT_DONE_ADDR 传入。
	# 写入 1 后, OpenSBI _start_warm 中的 lottery 检查会跳过
	# AMOSWAP 竞争, 直接进入 sbi_init() 的 warm boot 路径。
	li	t0, COLD_BOOT_DONE_ADDR   # t0 ← &coldboot_done (makefile 注入)
	li	t1, 1                    # t1 ← 1
	sw	t1, 0(t0)                # coldboot_done = 1
	fence	w, w                  # 确保 store 对所有 observer 可见

	# --- step 2: 设置 M 模式 trap 向量 ---
	# OpenSBI 在 _start 中将 mtvec 设为 _start_warm, 但 ZSBL
	# 直接从此处开始执行, 故需手动设置 mtvec 指向固件基址。
	# csrrw x0, mtvec, t0 即 mtvec ← t0 (旧值丢弃到 x0)。
	li	t0, RAM_BASE             # t0 ← ram_base (makefile 注入)
	csrrw	x0, mtvec, t0        # mtvec ← ram_base

	# --- step 3: 读取 hart ID ---
	# OpenSBI sbi_init() 期望 a0 = hart ID (mhartid CSR)。
	# csrrs a0, mhartid, x0 即 a0 ← mhartid (原子 read-set, 写回 0=no-op)。
	csrrs	a0, mhartid, x0        # a0 ← mhartid

	# --- step 4: 重新加载 FDT 地址 ---
	lla	a1, _start + 0x10000    # a1 ← FDT 地址 (再次确保)

# ---------------------------------------------------------------------------
# COLD_BOOT 宏 — 调试用自旋栅栏 (可选)
#
# 当定义 COLD_BOOT 时, 将 sbi_init() 中 hart=1 的分支指令替换为
# 自旋死循环 "j ." (编码 0x0080_006F)。效果: 除 hart 0 以外的所有
# hart 在进入 sbi_init() 后会陷入自旋, 不再继续执行。
#
# 目标地址由固件 ELF 符号表提取:
#   原始指令:  beq a0, a1, <offset>   (a0=mhartid, a1=1)
#             -> hart 1 跳到目标
#   修补后:    j .                   (hart 1 在此自旋)
#
# 这用于早期多 hart 同步—确保只有 hart 0 完成整套 init_sm() 流程。
# ---------------------------------------------------------------------------
# COLD_BOOT 宏已移除 — 固件地址在重编译后发生变化, 硬编码 patch
# 地址不再可靠。单 hart 模式下 AMOSWAP lottery 始终由 hart 0 胜出,
# 不修补分支也不影响启动正确性。
	unimp                         # 不应到达; 若到达则触发 Illegal Instruction trap
