
  .section .text.init
  .globl _prog_start
_prog_start:
  .globl _start
_start:
    la t0, trap_entry
    csrw mtvec, t0
    li s2, 0x8
    csrw mie, s2
    li s1, 0
    csrr s2, mhartid
    bne s1, s2, 42f
    li t0, 0x3000000 + 0x80000
    li t1, 1
    sw t1, 0(t0)
    li t1, 0x1e0000
    sd t1, 8(t0)
    li t1, 0x8000000
    sd t1, 16(t0)
    li t1, 0xa000000
    sd t1, 24(t0)
    li t1, 0xff000000
    sw t1, 4(t0)
    li t1, 3
    sw t1, 0(t0)
1:
    lw t1, 0(t0)
    andi t1, t1, 2
    bnez t1, 1b
    sw zero, 0(t0)
    la t0, _data_lma
    la t1, _data
    beq t0, t1, 2f
    la t2, _edata
    bgeu t1, t2, 2f
1:
    lw t3, 0(t0)
    sw t3, 0(t1)
    addi t0, t0, (1 << 2)
    addi t1, t1, (1 << 2)
    bltu t1, t2, 1b
2:
    li s1, 0x2000000 
41:
    li s2, 1
    sw s2, 0(s1)
    addi s1, s1, 4
    li s2, 0x2000000 + (32*4)
    blt s1, s2, 41b
42:
    wfi
    csrr s2, mip
    andi s2, s2, 0x8
    beqz s2, 42b
    li s1, 0x2000000
    csrr s2, mhartid
    slli s2, s2, 2
    add s2, s2, s1
    sw zero, 0(s2)
41:
    lw s2, 0(s1)
    bnez s2, 41b
    addi s1, s1, 4
    li s2, 0x2000000 + (32*4)
    blt s1, s2, 41b
    csrr t0, mhartid
    slli t0, t0, 12
    la sp, _sp
    sub sp, sp, t0
    call main
    li t0, 0x8000000
    csrr a0, mhartid
    la a1, _dtb
    jr t0
.align 2
trap_entry:
  call handle_trap
  .section .rodata
_dtb:
    .incbin "ux00_zsbl.dtb"
    .attribute	4, 16
    .attribute	5, "rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_zicsr2p0_zmmul1p0_zaamo1p0_zalrsc1p0_zca1p0_zcd1p0"
    .file	"sifive-zsbl-main.i"
    .text
    .globl	handle_trap                     # -- Begin function handle_trap
    .p2align	1
    .type	handle_trap,@function
handle_trap:                            # @handle_trap
	.cfi_startproc
# %bb.0:
	addi	sp, sp, -32
	.cfi_def_cfa_offset 32
	sd	ra, 24(sp)                      # 8-byte Folded Spill
	sd	s0, 16(sp)                      # 8-byte Folded Spill
	.cfi_offset ra, -8
	.cfi_offset s0, -16
	addi	s0, sp, 32
	.cfi_def_cfa s0, 0
	#APP
	csrr	a0, mcause
	#NO_APP
	sd	a0, -24(s0)
	ld	a0, -24(s0)
	sd	a0, -32(s0)
	ld	a0, -32(s0)
	li	a1, 1
	call	ux00boot_fail
	.cfi_def_cfa sp, 32
	ld	ra, 24(sp)                      # 8-byte Folded Reload
	ld	s0, 16(sp)                      # 8-byte Folded Reload
	.cfi_restore ra
	.cfi_restore s0
	addi	sp, sp, 32
	.cfi_def_cfa_offset 0
	ret
.Lfunc_end0:
	.size	handle_trap, .Lfunc_end0-handle_trap
	.cfi_endproc
                                        # -- End function
	.globl	init_uart                   # -- Begin function init_uart
	.p2align	1
	.type	init_uart,@function
init_uart:                              # @init_uart
	.cfi_startproc
# %bb.0:
	addi	sp, sp, -32
	.cfi_def_cfa_offset 32
	sd	ra, 24(sp)                      # 8-byte Folded Spill
	sd	s0, 16(sp)                      # 8-byte Folded Spill
	.cfi_offset ra, -8
	.cfi_offset s0, -16
	addi	s0, sp, 32
	.cfi_def_cfa s0, 0
                                        # kill: def $x11 killed $x10
	sw	a0, -20(s0)
	lui	a0, 28
	addi	a0, a0, 512
	sd	a0, -32(s0)
	lwu	a0, -20(s0)
	li	a1, 1000
	mul	a0, a0, a1
	ld	a1, -32(s0)
	call	uart_min_clk_divisor
	lui	a1, 65552
	sw	a0, 24(a1)
	.cfi_def_cfa sp, 32
	ld	ra, 24(sp)                      # 8-byte Folded Reload
	ld	s0, 16(sp)                      # 8-byte Folded Reload
	.cfi_restore ra
	.cfi_restore s0
	addi	sp, sp, 32
	.cfi_def_cfa_offset 0
	ret
.Lfunc_end1:
	.size	init_uart, .Lfunc_end1-init_uart
	.cfi_endproc
                                        # -- End function
	.p2align	1                               # -- Begin function uart_min_clk_divisor
	.type	uart_min_clk_divisor,@function
uart_min_clk_divisor:                   # @uart_min_clk_divisor
	.cfi_startproc
# %bb.0:
	addi	sp, sp, -48
	.cfi_def_cfa_offset 48
	sd	ra, 40(sp)                      # 8-byte Folded Spill
	sd	s0, 32(sp)                      # 8-byte Folded Spill
	.cfi_offset ra, -8
	.cfi_offset s0, -16
	addi	s0, sp, 48
	.cfi_def_cfa s0, 0
	sd	a0, -32(s0)
	sd	a1, -40(s0)
	ld	a0, -32(s0)
	ld	a1, -40(s0)
	add	a0, a0, a1
	addi	a0, a0, -1
	divu	a0, a0, a1
	sd	a0, -48(s0)
	ld	a0, -48(s0)
	bnez	a0, .LBB2_2
	j	.LBB2_1
.LBB2_1:
	li	a0, 0
	sw	a0, -20(s0)
	j	.LBB2_3
.LBB2_2:
	ld	a0, -48(s0)
	addiw	a0, a0, -1
	sw	a0, -20(s0)
	j	.LBB2_3
.LBB2_3:
	lw	a0, -20(s0)
	.cfi_def_cfa sp, 48
	ld	ra, 40(sp)                      # 8-byte Folded Reload
	ld	s0, 32(sp)                      # 8-byte Folded Reload
	.cfi_restore ra
	.cfi_restore s0
	addi	sp, sp, 48
	.cfi_def_cfa_offset 0
	ret
.Lfunc_end2:
	.size	uart_min_clk_divisor, .Lfunc_end2-uart_min_clk_divisor
	.cfi_endproc
                                        # -- End function
	.globl	puts                        # -- Begin function puts
	.p2align	1
	.type	puts,@function
puts:                                   # @puts
	.cfi_startproc
# %bb.0:
	addi	sp, sp, -32
	.cfi_def_cfa_offset 32
	sd	ra, 24(sp)                      # 8-byte Folded Spill
	sd	s0, 16(sp)                      # 8-byte Folded Spill
	.cfi_offset ra, -8
	.cfi_offset s0, -16
	addi	s0, sp, 32
	.cfi_def_cfa s0, 0
	sd	a0, -24(s0)
	li	a0, 1
	.cfi_def_cfa sp, 32
	ld	ra, 24(sp)                      # 8-byte Folded Reload
	ld	s0, 16(sp)                      # 8-byte Folded Reload
	.cfi_restore ra
	.cfi_restore s0
	addi	sp, sp, 32
	.cfi_def_cfa_offset 0
	ret
.Lfunc_end3:
	.size	puts, .Lfunc_end3-puts
	.cfi_endproc
                                        # -- End function
	.globl	main                            # -- Begin function main
	.p2align	1
	.type	main,@function
main:                                   # @main
	.cfi_startproc
# %bb.0:
	addi	sp, sp, -48
	.cfi_def_cfa_offset 48
	sd	ra, 40(sp)                      # 8-byte Folded Spill
	sd	s0, 32(sp)                      # 8-byte Folded Spill
	.cfi_offset ra, -8
	.cfi_offset s0, -16
	addi	s0, sp, 48
	.cfi_def_cfa s0, 0
	li	a0, 0
	sw	a0, -20(s0)
	#APP
	csrr	a0, mhartid
	#NO_APP
	sd	a0, -32(s0)
	ld	a0, -32(s0)
	sd	a0, -40(s0)
	ld	a0, -40(s0)
	bnez	a0, .LBB4_5
	j	.LBB4_1
.LBB4_1:
	lui	a0, 65536
	lwu	a0, 44(a0)
	andi	a0, a0, 2
	beqz	a0, .LBB4_3
	j	.LBB4_2
.LBB4_2:
	lui	a0, 8
	addi	a0, a0, 232
	sw	a0, -44(s0)
	j	.LBB4_4
.LBB4_3:
	lui	a0, 4
	addi	a0, a0, 116
	sw	a0, -44(s0)
	j	.LBB4_4
.LBB4_4:
	lw	a0, -44(s0)
	call	init_uart
	lw	a2, -44(s0)
.Lpcrel_hi0:
	auipc	a0, %got_pcrel_hi(gpt_guid_sifive_fsbl)
	ld	a1, %pcrel_lo(.Lpcrel_hi0)(a0)
	lui	a0, 32768
	call	ux00boot_load_gpt_partition
	j	.LBB4_5
.LBB4_5:
.Lpcrel_hi1:
	auipc	a0, %pcrel_hi(barrier)
	addi	a0, a0, %pcrel_lo(.Lpcrel_hi1)
	li	a1, 5
	call	Barrier_Wait
	li	a0, 0
	.cfi_def_cfa sp, 48
	ld	ra, 40(sp)                      # 8-byte Folded Reload
	ld	s0, 32(sp)                      # 8-byte Folded Reload
	.cfi_restore ra
	.cfi_restore s0
	addi	sp, sp, 48
	.cfi_def_cfa_offset 0
	ret
.Lfunc_end4:
	.size	main, .Lfunc_end4-main
	.cfi_endproc
                                        # -- End function
	.p2align	1                               # -- Begin function Barrier_Wait
	.type	Barrier_Wait,@function
Barrier_Wait:                           # @Barrier_Wait
	.cfi_startproc
# %bb.0:
	addi	sp, sp, -64
	.cfi_def_cfa_offset 64
	sd	ra, 56(sp)                      # 8-byte Folded Spill
	sd	s0, 48(sp)                      # 8-byte Folded Spill
	.cfi_offset ra, -8
	.cfi_offset s0, -16
	addi	s0, sp, 64
	.cfi_def_cfa s0, 0
                                        # kill: def $x12 killed $x11
	sd	a0, -24(s0)
	sw	a1, -28(s0)
	lw	a0, -28(s0)
	li	a1, 1
	bne	a0, a1, .LBB5_2
	j	.LBB5_1
.LBB5_1:
	j	.LBB5_12
.LBB5_2:
	ld	a0, -24(s0)
	fence	rw, rw
	lw	a0, 16(a0)
	fence	r, rw
	sw	a0, -32(s0)
	ld	a0, -24(s0)
	lw	a1, -32(s0)
	slli	a1, a1, 2
	add	a1, a1, a0
	li	a0, 1
	sw	a0, -40(s0)
	lw	a0, -40(s0)
	amoadd.w.aqrl	a0, a0, (a1)
	sw	a0, -44(s0)
	lw	a0, -44(s0)
	addiw	a0, a0, 1
	sw	a0, -36(s0)
	lw	a0, -36(s0)
	lw	a1, -28(s0)
	bne	a0, a1, .LBB5_6
	j	.LBB5_3
.LBB5_3:
	lw	a1, -32(s0)
	li	a0, 1
	subw	a0, a0, a1
	ld	a1, -24(s0)
	fence	rw, w
	sw	a0, 16(a1)
	fence	rw, rw
	ld	a0, -24(s0)
	lw	a1, -32(s0)
	slli	a1, a1, 2
	add	a1, a1, a0
	li	a0, -1
	sw	a0, -48(s0)
	lw	a0, -48(s0)
	amoadd.w.aqrl	a0, a0, (a1)
	sw	a0, -52(s0)
	lw	a0, -36(s0)
	li	a1, 2
	blt	a0, a1, .LBB5_5
	j	.LBB5_4
.LBB5_4:
	ld	a0, -24(s0)
	lw	a1, -32(s0)
	slli	a1, a1, 2
	add	a1, a1, a0
	fence	rw, w
	li	a0, 1
	sw	a0, 8(a1)
	fence	rw, rw
	j	.LBB5_5
.LBB5_5:
	j	.LBB5_12
.LBB5_6:
	j	.LBB5_7
.LBB5_7:                                # =>This Inner Loop Header: Depth=1
	ld	a0, -24(s0)
	lw	a1, -32(s0)
	slli	a1, a1, 2
	add	a0, a0, a1
	fence	rw, rw
	lw	a0, 8(a0)
	fence	r, rw
	bnez	a0, .LBB5_9
	j	.LBB5_8
.LBB5_8:                                #   in Loop: Header=BB5_7 Depth=1
	j	.LBB5_7
.LBB5_9:
	ld	a0, -24(s0)
	lw	a1, -32(s0)
	slli	a1, a1, 2
	add	a1, a1, a0
	li	a0, -1
	sw	a0, -56(s0)
	lw	a0, -56(s0)
	amoadd.w.aqrl	a0, a0, (a1)
	sw	a0, -60(s0)
	lw	a0, -60(s0)
	li	a1, 1
	bne	a0, a1, .LBB5_11
	j	.LBB5_10
.LBB5_10:
	ld	a0, -24(s0)
	lw	a1, -32(s0)
	slli	a1, a1, 2
	add	a1, a1, a0
	fence	rw, w
	li	a0, 0
	sw	a0, 8(a1)
	fence	rw, rw
	j	.LBB5_11
.LBB5_11:
	j	.LBB5_12
.LBB5_12:
	.cfi_def_cfa sp, 64
	ld	ra, 56(sp)                      # 8-byte Folded Reload
	ld	s0, 48(sp)                      # 8-byte Folded Reload
	.cfi_restore ra
	.cfi_restore s0
	addi	sp, sp, 64
	.cfi_def_cfa_offset 0
	ret
.Lfunc_end5:
	.size	Barrier_Wait, .Lfunc_end5-Barrier_Wait
	.cfi_endproc
                                        # -- End function
	.type	barrier,@object                 # @barrier
	.local	barrier
	.comm	barrier,20,4
	.ident	"clang version 22.0.0git (git@github.com:TrisenYu/llvm-4-ext-riscv-isa.git 419a827f578b4e8bd295dcacf946e996e1b93f04)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym ux00boot_fail
	.addrsig_sym init_uart
	.addrsig_sym uart_min_clk_divisor
	.addrsig_sym ux00boot_load_gpt_partition
	.addrsig_sym Barrier_Wait
	.addrsig_sym gpt_guid_sifive_fsbl
	.addrsig_sym barrier
