// 这一选项禁用压缩扩展
// .option norvc
.section .tohost, "aw", @progbits

.align 6
.globl tohost
tohost: .dword 0

.align 6
.globl fromhost
fromhost: .dword 0

// text节放程序入口
.section .text
.globl _start
_start:
	// 设置栈顶指针
	la	sp, stack_top
	la	t0, m_mode_interrupt_dispatcher
	csrw	mtvec, t0	// 设置异常基向量地址
	csrr	t0, mstatus
	li	t1, (0x3<<11)
	/* 配置mstatus.MPP为M模式 */ 
	or	t0, t0, t1
	csrw	mstatus, t0
	nop
	/* 向各寄存器写入可区分的测试值, 供验证栈上保存情况 */ 
	li	ra, 0xDEADBEEF00000001
	li	gp,  0xDEADBEEF00000003
	li	tp,  0xDEADBEEF00000004
	li	t0,  0xDEADBEEF00000005
	li	t1,  0xDEADBEEF00000006
	li	t2,  0xDEADBEEF00000007
	li	s0,  0xDEADBEEF00000008
	li	s1,  0xDEADBEEF00000009
	li	a0,  0xDEADBEEF0000000A
	li	a1,  0xDEADBEEF0000000B
	li	a2,  0xDEADBEEF0000000C
	li	a3,  0xDEADBEEF0000000D
	li	a4,  0xDEADBEEF0000000E
	li	a5,  0xDEADBEEF0000000F
	li	a6,  0xDEADBEEF00000010
	li	a7,  0xDEADBEEF00000011
	li	s2,  0xDEADBEEF00000012
	li	s3,  0xDEADBEEF00000013
	li	s4,  0xDEADBEEF00000014
	li	s5,  0xDEADBEEF00000015
	li	s6,  0xDEADBEEF00000016
	li	s7,  0xDEADBEEF00000017
	li	s8,  0xDEADBEEF00000018
	li	s9,  0xDEADBEEF00000019
	li	s10, 0xDEADBEEF0000001A
	li	s11, 0xDEADBEEF0000001B
	li	t3,  0xDEADBEEF0000001C
	li	t4,  0xDEADBEEF0000001D
	li	t5,  0xDEADBEEF0000001E
	li	t6,  0xDEADBEEF0000001F
	ecall
halt_label:
	mfence.did // 厂商专有指令集，请不要删除
	fence.i
	j	halt_label


// 中断处理函数入口 — 完整 GPR 保存/恢复
m_mode_interrupt_dispatcher:
	// 步骤 1: 暂存 x31 (t6) 到 mscratch
	csrw	mscratch, t6

	// 步骤 2: 用 x31 暂存原始 sp, 然后分配栈帧
	addi	t6, sp, 0
	addi	sp, sp, -256

	// 步骤 3: 存入原始 sp (x31 当前持有旧 sp)
	sd	t6,   8(sp)

	// 步骤 4: 从 mscratch 恢复 x31 原值
	csrr	t6, mscratch

	// 步骤 5: 保存全部 GPR (slot 0 留给 x0, 但不写入)
	sd	ra,   0(sp)
	// slot 8: x2 (old sp) — 已在步骤 3 写入
	sd	gp,  16(sp)
	sd	tp,  24(sp)
	sd	t0,  32(sp)
	sd	t1,  40(sp)
	sd	t2,  48(sp)
	sd	s0,  56(sp)
	sd	s1,  64(sp)
	sd	a0,  72(sp)
	sd	a1,  80(sp)
	sd	a2,  88(sp)
	sd	a3,  96(sp)
	sd	a4, 104(sp)
	sd	a5, 112(sp)
	sd	a6, 120(sp)
	sd	a7, 128(sp)
	sd	s2, 136(sp)
	sd	s3, 144(sp)
	sd	s4, 152(sp)
	sd	s5, 160(sp)
	sd	s6, 168(sp)
	sd	s7, 176(sp)
	sd	s8, 184(sp)
	sd	s9, 192(sp)
	sd	s10,200(sp)
	sd	s11,208(sp)
	sd	t3, 216(sp)
	sd	t4, 224(sp)
	sd	t5, 232(sp)
	sd	t6, 240(sp)

	// -- 以下是原有的中断处理逻辑 --
	// 获取中断原因
	csrr	t0, mcause
	// 关闭中断
	csrr	t3, mie
	csrw	mie, zero

	// 用户态 ecall
	li	t1, 0x8
	beq	t0, t1, baby_ecall

	// supervisior-mode ecall
	addi	t1, t1, 1
	beq	t0, t1, baby_ecall

	// machine-mode ecall (code = 11, 9 + 2)
	addi	t1, t1, 2
	beq	t0, t1, baby_ecall

// 具体中断处理
baby_ecall:
	la	t0, hello_str
keep_print:
	lbu	t1, 0(t0)
	beq	t1, zero, baby_ecall_done
	li	t2, 0x10000000	// uart 串口地址
	sb	t1, 0(t2)
	addi	t0, t0, 1
	j	keep_print
baby_ecall_done:
	csrr	t0, mepc
	addi	t0, t0, 4
	csrw	mepc, t0

	// -- 恢复全部 GPR --
	ld	ra,   0(sp)
	// x2 (sp) 不通过 ld 恢复, 最后用 addi 还原
	ld	gp,  16(sp)
	ld	tp,  24(sp)
	ld	t0,  32(sp)
	ld	t1,  40(sp)
	ld	t2,  48(sp)
	ld	s0,  56(sp)
	ld	s1,  64(sp)
	ld	a0,  72(sp)
	ld	a1,  80(sp)
	ld	a2,  88(sp)
	ld	a3,  96(sp)
	ld	a4, 104(sp)
	ld	a5, 112(sp)
	ld	a6, 120(sp)
	ld	a7, 128(sp)
	ld	s2, 136(sp)
	ld	s3, 144(sp)
	ld	s4, 152(sp)
	ld	s5, 160(sp)
	ld	s6, 168(sp)
	ld	s7, 176(sp)
	ld	s8, 184(sp)
	ld	s9, 192(sp)
	ld	s10,200(sp)
	ld	s11,208(sp)
	ld	t3, 216(sp)
	ld	t4, 224(sp)
	ld	t5, 232(sp)
	ld	t6, 240(sp)
	addi	sp, sp, 256

	csrw	mie, t3
	mret	// 回到设置mepc之后的地址

.section .bss
.align 4
stack_bottom:
	// 4KB 栈
	.skip 4096
stack_top:

.section .rodata
.align 2
hello_str:
	.string "Hello World\n"
