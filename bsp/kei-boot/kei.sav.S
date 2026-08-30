/*
	----------------<< Dedicated to AL-1S >>----------------
	kei.sav: RISC-V Zero/First Stage Boot Loader (ZSBL/FSBL)

	涉及参数:
		RAM_BASE    默认 0x8000_0000，由 makefile 通过 -D 传入
		FDT blob    默认放置在 _start + 0x10000
		后续串口/GPIO 的 MMIO 基址同样应通过与 RAM_BASE 相同的方式 (-D) 传入,
		而非在源文件内硬编码。

	流程: 设置早期mtvec后，将hartid写入到a0并设置设备树数据所在地址到a1
	最后通过跳转到固件代码来移交执行与控制权
	约定: a0 = hart ID (mhartid), a1 = FDT 物理地址。

	上电自检 (POST) 顺序:
		[*] CPU 基本算术 / 逻辑 / 移位 / 分支 / 跳转指令
		[*] 内存读取 (读 RAM_BASE; 地址无效则触发访存异常自动进入 mtvec)
		[ ] 串口通道 (MMIO 基址依赖提供给设备树的对应, 经 -D 传入, 暂未实现)
		[ ] Cache / 看门狗 / RTC / I2C / SPI / USB / GPU 自检测试 (暂未实现)

	自检失败时，后续将经 GPIO 输出引脚点亮对应的自检指示 LED
	同样，对应指示自检阶段的GPIO 基地址会通过makefile的 -D 传递给kei.sav
*/
.section .text, "ax", @progbits

/**
 * 固件入口
**/
.global _start
_start:
	/* 设置mtvec为 AL1S_HANG */
	lla		t0, AL1S_HANG
	csrrw	x0, mtvec, t0
	/*
		清零 BSS 段 (链接脚本保证 _bss_start/_bss_end 均按 8 字节对齐)
		未显式初始化的全局/静态变量由 ABI 约定必须为零,
		上电后 RAM 内容不确定, 移交固件前先行清零,
		避免后续阶段 (如 kei_aux.c) 读到残留数据
	*/
	lla		t0, _bss_start
	lla		t1, _bss_end
	bgeu	t0, t1, AL1S_BSS_CLEAR_DONE // 跳过空段
AL1S_BSS_CLEAR:
	sd		x0, 0(t0)
	addi	t0, t0, 8
	bltu	t0, t1, AL1S_BSS_CLEAR

/**
 * CPU 自检: 基本算术 / 逻辑 / 移位 / 分支 / 跳转指令.
 *
 * 逐项计算预期值并比较, 任一不符即跳 AL1S_CPU_FAIL.
 * 全部通过后落入内存读取自检 (AL1S_MEM_TEST).
**/
AL1S_BSS_CLEAR_DONE:
AL1S_CPU_ALG_TEST:
	/* --- 加减法 --- */
	li		t0, 5
	li		t1, 7
	add		t2, t0, t1          // 5 + 7 = 12
	li		t3, 12
	bne		t2, t3, AL1S_CPU_FAIL
	sub		t2, t1, t0          // 7 - 5 = 2
	li		t3, 2
	bne		t2, t3, AL1S_CPU_FAIL

	/* --- 乘除法 (M 扩展) --- */
	mul		t2, t0, t1          // 5 * 7 = 35
	li		t3, 35
	bne		t2, t3, AL1S_CPU_FAIL
	div		t2, t2, t1          // 35 / 7 = 5
	li		t3, 5
	bne		t2, t3, AL1S_CPU_FAIL
	li		t0, 37
	rem		t2, t0, t1          // 37 % 7 = 2
	li		t3, 2
	bne		t2, t3, AL1S_CPU_FAIL

	/* --- 逻辑运算 --- */
	li		t0, 0xF0
	li		t1, 0x0F
	and		t2, t0, t1          // 0xF0 & 0x0F = 0
	bnez	t2, AL1S_CPU_FAIL
	or		t2, t0, t1          // 0xF0 | 0x0F = 0xFF
	li		t3, 0xFF
	bne		t2, t3, AL1S_CPU_FAIL
	xor		t2, t0, t1          // 0xF0 ^ 0x0F = 0xFF
	li		t3, 0xFF
	bne		t2, t3, AL1S_CPU_FAIL

	/* --- 移位 --- */
	li		t0, 1
	slli	t1, t0, 4           // 1 << 4 = 16
	li		t3, 16
	bne		t1, t3, AL1S_CPU_FAIL
	srli	t1, t1, 4           // 16 >> 4 = 1
	bne		t1, t0, AL1S_CPU_FAIL
	li		t0, -8              // 0xFFFF_FFFF_FFFF_FFF8
	srai	t1, t0, 1           // 算术右移 = -4
	li		t3, -4
	bne		t1, t3, AL1S_CPU_FAIL

	/* --- 条件分支 (beq / bne / blt / bge) --- */
	li		t0, 1
	li		t1, 1
	beq		t0, t1, _AL1S_CPU_BEQ_TAKEN
	j		AL1S_CPU_FAIL
_AL1S_CPU_BEQ_TAKEN:
	bne		t0, t1, AL1S_CPU_FAIL   // 相等

	li		t0, -5
	li		t1, 3
	blt		t0, t1, _AL1S_CPU_BLT_TAKEN  // -5 < 3
	j		AL1S_CPU_FAIL
_AL1S_CPU_BLT_TAKEN:
	bge		t0, t1, AL1S_CPU_FAIL   // -5 >= 3 为假, 不应失败

	/* --- 跳转 (jal / jalr) --- */
	jal		_AL1S_CPU_JAL_TARGET
	j		AL1S_CPU_FAIL
_AL1S_CPU_JAL_TARGET:
	lla		t0, _AL1S_CPU_JALR_TARGET
	jalr	x0, 0(t0)
	j		AL1S_CPU_FAIL
_AL1S_CPU_JALR_TARGET:
	/* CPU 自检通过, 落入内存读取自检 */


/**
 * 内存读取自检: 读取 RAM_BASE 处内容.
 *
 * 仅检测该地址是否可读: 若地址无效 (访存异常), 硬件自动陷入 mtvec
 * (AL1S_HANG) 从而挂起; 若读取成功 (无异常) 即视为通过, 不校验数据值,
 * 也不做写入 (写入可能破坏已加载的固件镜像, 暂不考虑).
*/
AL1S_MEM_TEST:
	li		t0, RAM_BASE
	ld		t1, 0(t0)           // 读 8 字节; 无效地址 -> 访存异常 -> mtvec
	/* 读取成功即通过, 移交固件 */
	j		AL1S_POST_DONE


/**
 * CPU 自检失败处理 (软件检测到算术/逻辑/移位/分支/跳转结果不符).
 *
 * TODO: 失败时经 GPIO OUTPUT_VAL 点亮 CPU 自检 LED (GPIO 基址经 -D 传入).
 * 当前无 LED 表示, 仅挂起。
*/
AL1S_CPU_FAIL:
	/* TODO: 拉高CPU自检LED的电平 */
	j		AL1S_HANG
AL1S_UART_FAIL:
	/* TODO: 拉高串口自检LED的电平 */
	j       AL1S_HANG
AL1S_HARDDISK_FAIL:
	/* TODO: 拉高磁盘自检LED的电平 */
	j       AL1S_HANG
AL1S_DISPLAY_FAIL:
	/* TODO: 拉高显示器自检LED的电平 */
	j       AL1S_HANG

/**
 * 自检通过后的固件移交点
*/
.global AL1S_POST_DONE
AL1S_POST_DONE:
	/*
		移交前清零临时寄存器 t1..t6 (仅 t0 保留作跳转目标),
		避免固件读到自检过程残留的中间值。
	*/
	li		t1, 0
	li		t2, 0
	li		t3, 0
	li		t4, 0
	li		t5, 0
	li		t6, 0
	/* 跳转到下一阶段的程序 */
	csrrs	a0, mhartid, x0      // 读取当前 hart ID 到 a0
	lla		a1, _start + 0x10000 // 设置 默认放置于 _start + 64 KiB 的FDT 地址
	li		t0, RAM_BASE         // 固件的 _start 会设置于 RAM_BASE。此处略去加载环节
	jr		t0                   // Alice, startup!


/**
 * 简易hang
*/
.global AL1S_HANG
AL1S_HANG:
	wfi
	j       AL1S_HANG
	unimp
