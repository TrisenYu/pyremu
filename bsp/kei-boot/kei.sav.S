/*
	kei.sav: RISC-V Zero/First Stage Boot Loader (ZSBL/FSBL)

	涉及参数:
		RAM_BASE    默认 0x8000_0000，由 makefile 通过 -D 传入
		FDT blob    默认放置在 _start + 0x10000

	流程: 设置早期mtvec后，将hartid写入到a0并设置设备树数据所在地址到a1
	最后通过跳转到固件代码来移交执行与控制权
	约定: a0 = hart ID (mhartid), a1 = FDT 物理地址。

	尚未对设备做任何自检工作
	所以后续可能会加入进来，使得kei.sav.S在编译后能达到4KB左右的大小
	o) CPU test
	o) Cache test
	o) Memory test
	o) Serial channels test
	o) Watchdog timer test
	o) RTC test
	o) I2C test
	o) SPI test
	o) USB test
	o) GPU test
	o) Ethernet test

	以及所有test失效时应当执行的相关操作
	失败时启用相应的自检LED
*/
.section .text, "ax", @progbits

/**
 * 固件入口
*/
.globl _start
_start:
	/* 设置mtvec为_AL1S_HANG */
	lla		t0, _AL1S_HANG
	csrrw	x0, mtvec, t0

	/*
		清零 BSS 段 (链接脚本保证 _bss_start/_bss_end 均按 8 字节对齐)
		未显式初始化的全局/静态变量由 ABI 约定必须为零,
		上电后 RAM 内容不确定, 移交固件前先行清零,
		避免后续阶段 (如 kei_aux.c) 读到残留数据
	*/
	lla		t0, _bss_start
	lla		t1, _bss_end
	bgeu	t0, t1, _AL1S_BSS_DONE // 空段直接跳过
_AL1S_BSS_CLEAR:
	sd		x0, 0(t0)
	addi	t0, t0, 8
	bltu	t0, t1, _AL1S_BSS_CLEAR
_AL1S_BSS_DONE:
	/* 跳转到下一阶段的程序 */
	csrrs	a0, mhartid, x0      // 读取当前 hart ID 到 a0
	lla		a1, _start + 0x10000 // 设置 默认放置于 _start + 64 KiB 的FDT 地址
	li		t0, RAM_BASE         // 固件的 _start 会设置于 RAM_BASE。此处略去加载环节
	jr		t0                   // Alice, startup!


/**
 * 简易hang
*/
_AL1S_HANG:
	wfi
	j       _AL1S_HANG
	unimp

/** TODO:
_AL1S_ALU_TEST:
_AL1S_BRANCH_TEST:
_AL1S_CACHE_TEST:
_AL1S_MEM_TEST:
_AL1S_UART_TEST:
_AL1S_TIMER_TEST:
_AL1S_WATCHDOG_TEST:
_AL1S_GPU_TEST:
*/
