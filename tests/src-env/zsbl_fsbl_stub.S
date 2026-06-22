# 极简 ZSBL + FSBL

.section .text, "ax", @progbits
.globl _start

_start:
	lla	a1, _start + 0x10000
	j	fsbl_start

fsbl_start:
	li	t0, 0x800842F8
	li	t1, 1
	sw	t1, 0(t0)
	fence	w,w

	li	t0, 0x80000000
	csrrw	x0, mtvec, t0 # mtvec <- 0x80000000
	csrrs	a0, mhartid, x0
	lla	a1, _start + 0x10000

#ifdef COLD_BOOT
	li	t0, 0x8000e0d0
	li	t1, 0x0080006f
	sw	t1, 0(t0)
	fence	w,w
#endif
	unimp
