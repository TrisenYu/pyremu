/*
	----------------<< Dedicated to AL-1S >>----------------
	kei.sav: RISC-V Zero/First Stage Boot Loader (ZSBL/FSBL)

	Related Arguments:
		- RAM_BASE    default set to 0x8000_0000, passed from outer Makefile via -D
		- FDT blob    default set to _start + 0x10000

		Instead hard-encoded in source code, the MMIO base address of uart/GPIO
		should be passed in the same way of RAM_BASE.

	After setting early mtvec, hartid will be written into GPR a0,
	and the fdt-blob will be set to GPR a1,
	then jump to the firmware entry to hand over execution and control.
	calling-convention: a0 = hart ID (mhartid), a1 = physical address of FDT

	POST order:
		[*] CPU basic arithmetic / logic / bit-shifting / branch / jump instructions
		[*] memory read (read RAM_BASE; an invalid address raises an access
		    fault and traps to mtvec)
		[ ] uart channel (MMIO base taken from the device tree and passed in
		    via -D, not implemented yet)
		[ ] cache / watchdog / RTC / I2C / SPI / USB / GPU POST (not implemented yet)

	A failed POST will later light the matching indicator LED through a GPIO
	output pin; the GPIO base address of that POST stage is passed to kei.sav
	via -D from the outer Makefile, in the same way as RAM_BASE.
*/
.section .text, "ax", @progbits

/**
 * the Entry of zero stage bootloader
**/
.global _start
_start:
	/* set mtvec to AL1S_HANG */
	lla		t0, AL1S_HANG
	csrrw	x0, mtvec, t0
	/*
		zero BSS. (the linker script keeps _bss_start/_bss_end 8-byte aligned)
		The ABI requires globals/statics without an explicit initializer to be
		zero while RAM contents are undefined at power-on, so clear them before
		handing over to the firmware, otherwise later stages (e.g. kei_aux.c)
		would read stale data.
	*/
	lla		t0, _bss_start
	lla		t1, _bss_end
	bgeu	t0, t1, AL1S_BSS_CLEAR_DONE
AL1S_BSS_CLEAR:
	sd		x0, 0(t0)
	addi	t0, t0, 8
	bltu	t0, t1, AL1S_BSS_CLEAR

/**
 * CPU POST: basic arithmetic / logic / shift / branch / jump instructions.
 *
 * Each item computes an expected value and compares; any mismatch raises
 * AL1S_CPU_FAIL. Once all items pass, control falls through to the memory
 * read POST (AL1S_MEM_TEST).
**/
AL1S_BSS_CLEAR_DONE:
AL1S_CPU_ALG_TEST:
	/* --- add and sub --- */
	li		t0, 5
	li		t1, 7
	add		t2, t0, t1          // 5 + 7 = 12
	li		t3, 12
	bne		t2, t3, AL1S_CPU_FAIL
	sub		t2, t1, t0          // 7 - 5 = 2
	li		t3, 2
	bne		t2, t3, AL1S_CPU_FAIL

	/* --- M extension for multiply and division --- */
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

	/* --- and or not --- */
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

	/* --- left shift or right shift --- */
	li		t0, 1
	slli	t1, t0, 4           // 1 << 4 = 16
	li		t3, 16
	bne		t1, t3, AL1S_CPU_FAIL
	srli	t1, t1, 4           // 16 >> 4 = 1
	bne		t1, t0, AL1S_CPU_FAIL
	li		t0, -8              // 0xFFFF_FFFF_FFFF_FFF8
	srai	t1, t0, 1           // -4
	li		t3, -4
	bne		t1, t3, AL1S_CPU_FAIL

	/* --- beq / bne / blt / bge --- */
	li		t0, 1
	li		t1, 1
	beq		t0, t1, _AL1S_CPU_BEQ_TAKEN
	j		AL1S_CPU_FAIL
_AL1S_CPU_BEQ_TAKEN:
	bne		t0, t1, AL1S_CPU_FAIL

	li		t0, -5
	li		t1, 3
	blt		t0, t1, _AL1S_CPU_BLT_TAKEN  // -5 < 3
	j		AL1S_CPU_FAIL
_AL1S_CPU_BLT_TAKEN:
	bge		t0, t1, AL1S_CPU_FAIL   // -5 >= 3, should be false and not fail

	/* --- jal / jalr --- */
	jal		_AL1S_CPU_JAL_TARGET
	j		AL1S_CPU_FAIL
_AL1S_CPU_JAL_TARGET:
	lla		t0, _AL1S_CPU_JALR_TARGET
	jalr	x0, 0(t0)
	j		AL1S_CPU_FAIL
_AL1S_CPU_JALR_TARGET:
	/* pass easy POST */


/**
 * Memory read POST: read the contents at RAM_BASE.
 *
 * Only readability is checked: an invalid address raises an access fault, so
 * the hardware traps to mtvec (AL1S_HANG) and hangs; a successful read (no
 * exception) counts as a pass. The value is not verified and no write is
 * performed (a write could corrupt the loaded firmware image; out of scope).
*/
AL1S_MEM_TEST:
	li		t0, RAM_BASE
	ld		t1, 0(t0)
	/*
		Read Only. 8 bytes
		invalid address generate an access fault,
		which will force PC changing to mtvec.
		a successful read is a pass:
		hand over to the firmware
	*/
	j		AL1S_POST_DONE


/**
 * CPU POST failure handler (software detected a mismatch in an arithmetic /
 * logic / shift / branch / jump result).
 *
 * TODO: on failure drive the CPU POST LED through GPIO OUTPUT_VAL (GPIO base
 * passed in via -D). No LED indication exists yet; hang instead.
*/
AL1S_CPU_FAIL:
	/* TODO: raise the CPU POST LED */
	j		AL1S_HANG
AL1S_UART_FAIL:
	/* TODO: raise the uart POST LED */
	j       AL1S_HANG
AL1S_HARDDISK_FAIL:
	/* TODO: raise the hard disk POST LED */
	j       AL1S_HANG
AL1S_DISPLAY_FAIL:
	/* TODO: raise the display POST LED */
	j       AL1S_HANG

/**
 * Firmware hand-over point after a successful POST
*/
.global AL1S_POST_DONE
AL1S_POST_DONE:
	/*
		Clear the temporary registers t1..t6 before the hand-over (only t0 is
		kept as the jump target), so that the firmware never reads leftover
		intermediate values from the POST.
	*/
	li		t1, 0
	li		t2, 0
	li		t3, 0
	li		t4, 0
	li		t5, 0
	li		t6, 0
	/* jump to the next stage */
	csrrs	a0, mhartid, x0      // read the current hart ID into a0
	lla		a1, _start + 0x10000 // set the FDT address, placed at _start + 64 KiB by default
	li		t0, RAM_BASE         // the firmware _start is placed at RAM_BASE; loading is skipped here
	jr		t0                   // Alice, startup!


/**
 * easy and plain hang
*/
.global AL1S_HANG
AL1S_HANG:
	wfi
	j       AL1S_HANG
	unimp
