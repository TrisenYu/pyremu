# 飞地 S-mode 入口与陷态向量。
# 紧跟 smode_entry/entry.s + ref-emod entry.S 的结构。

# ================================================================
#  SAVE / RESTORE 宏（264 字节帧：32 GPR + usp 槽）
#  必须在使用前定义（LLVM 集成汇编器要求前置声明）。
# ================================================================

.macro SAVE_CONTEXT
    addi sp, sp, -264
    sd   x0,  0(sp)
    sd   ra,  8(sp)
    sd   gp,  24(sp)
    sd   tp,  32(sp)
    sd   t0,  40(sp)
    sd   t1,  48(sp)
    sd   t2,  56(sp)
    sd   s0,  64(sp)
    sd   s1,  72(sp)
    sd   a0,  80(sp)
    sd   a1,  88(sp)
    sd   a2,  96(sp)
    sd   a3,  104(sp)
    sd   a4,  112(sp)
    sd   a5,  120(sp)
    sd   a6,  128(sp)
    sd   a7,  136(sp)
    sd   s2,  144(sp)
    sd   s3,  152(sp)
    sd   s4,  160(sp)
    sd   s5,  168(sp)
    sd   s6,  176(sp)
    sd   s7,  184(sp)
    sd   s8,  192(sp)
    sd   s9,  200(sp)
    sd   s10, 208(sp)
    sd   s11, 216(sp)
    sd   t3,  224(sp)
    sd   t4,  232(sp)
    sd   t5,  240(sp)
    sd   t6,  248(sp)
.endm

.macro RESTORE_CONTEXT
    ld   ra,  8(sp)
    ld   gp,  24(sp)
    ld   tp,  32(sp)
    ld   t0,  40(sp)
    ld   t1,  48(sp)
    ld   t2,  56(sp)
    ld   s0,  64(sp)
    ld   s1,  72(sp)
    ld   a0,  80(sp)
    ld   a1,  88(sp)
    ld   a2,  96(sp)
    ld   a3,  104(sp)
    ld   a4,  112(sp)
    ld   a5,  120(sp)
    ld   a6,  128(sp)
    ld   a7,  136(sp)
    ld   s2,  144(sp)
    ld   s3,  152(sp)
    ld   s4,  160(sp)
    ld   s5,  168(sp)
    ld   s6,  176(sp)
    ld   s7,  184(sp)
    ld   s8,  192(sp)
    ld   s9,  200(sp)
    ld   s10, 208(sp)
    ld   s11, 216(sp)
    ld   t3,  224(sp)
    ld   t4,  232(sp)
    ld   t5,  240(sp)
    ld   t6,  248(sp)
    addi sp, sp, 264
.endm

# ================================================================
#  入口
# ================================================================
#
#  PIE 寻址约定: 全部使用显式 %pcrel_hi / %pcrel_lo, 绝不通过 GOT.
#  二进制链接在 0x0, 运行时可在任意物理地址正常工作.

.align 4
.section ".text.init"
.global _start
_start:
    # ---- 保存 M-mode 传入的参数（callee-saved 寄存器）----
    # M-mode 在 CREATE ecall 返回前设置:
    #   a0 = enclave_id, a1 = base_pa, a2 = payload_size
    mv   s0, a0                 # s0 = enclave_id
    mv   s1, a1                 # s1 = base_pa
    mv   s2, a2                 # s2 = payload_size

    # ---- 临时栈 (PC 相对) ----
.L0_tmp_stack:
    auipc sp, %pcrel_hi(tmp_stack_top)
    addi  sp, sp, %pcrel_lo(.L0_tmp_stack)
    # ---- 为 BootInfo 分配栈空间 (3 × u64 = 24 字节, 对齐到 32) ----
    addi sp, sp, -32
    # ---- 设置 rust_main_before_mmu 参数 ----
    # fn(out: *mut BootInfo, enclave_id: u64, base_pa: u64, payload_size: u64)
    mv   a0, sp                 # a0 = &BootInfo (out 指针)
    mv   a1, s0                 # a1 = enclave_id
    mv   a2, s1                 # a2 = base_pa
    mv   a3, s2                 # a3 = payload_size
    # ---- 注册早期陷态处理 (PC 相对) ----
.L0_early:
    auipc t0, %pcrel_hi(early_trap)
    addi  t0, t0, %pcrel_lo(.L0_early)
    csrw stvec, t0
    # Bare 模式调用阶段一 (PC 相对 call)
    call rust_main_before_mmu
    # 从栈上加载返回的 BootInfo 字段
    ld   a0, 0(sp)              # a0 = satp
    ld   a1, 8(sp)              # a1 = smode_sp
    ld   a2, 16(sp)             # a2 = va_offset
    addi sp, sp, 32             # 恢复临时栈指针
    # ---- 切换到虚拟地址并启用 MMU ----
.L0_temp_stvec:
    auipc s7, %pcrel_hi(temp_stvec)
    addi  s7, s7, %pcrel_lo(.L0_temp_stvec)
    add  s7, a2, s7             # s7 = VA(temp_stvec)
    csrw stvec, s7
    mv   sp, a1                 # 切换到虚拟 S-mode 栈
    sfence.vma
    csrw satp, a0               # 启用 Sv39

.align 2
temp_stvec:
    sfence.vma
    # MMU 开启后调用阶段二重新设置 trap 处理 (PC 相对)
.L1_trap_vec:
    auipc s7, %pcrel_hi(trap_vector)
    addi  s7, s7, %pcrel_lo(.L1_trap_vec)
    add  s7, s7, a2             # PA → VA (a2 = va_offset 仍有效)
    csrw stvec, s7
    # 以及其余初始化操作
    call rust_main_after_mmu
    # sret 切换到 U-mode
    csrrw sp, sscratch, sp
    sret

# ================================================================
#  Bare 模式下的占位陷态向量 — 仅作死循环，Bare 阶段不应触发陷阱
# ================================================================

.align 2
early_trap:
    wfi
    j   early_trap

# ================================================================
#  陷态入口
# ================================================================

.align 4
.section ".text"
.global trap_vector
trap_vector:
    # sscratch 交换：sp ↔ sscratch
    csrrw sp, sscratch, sp
    bnez  sp, .L_user_trap      # sp != 0 → 用户态陷态

    # ---- 内核陷态：sscratch == 0 ----
    csrr  sp, sstatus
    andi  sp, sp, 0x100         # SPP bit
    bnez  sp, .L_kernel_trap    # SPP=1 → 内核陷态

    # ---- 新线程首次陷态：SPP=0，需分配内核栈 ----
    csrrw sp, sscratch, sp      # 恢复 sp
    SAVE_CONTEXT
    jal   alloc_smode_stack     # 分配内核栈 → a0
    csrw  sscratch, a0
    RESTORE_CONTEXT
    csrrw sp, sscratch, sp
    j     .L_user_trap

.L_kernel_trap:
    # 内核陷态：清零 sscratch
    mv    sp, zero
    csrrw sp, sscratch, sp

.L_user_trap:
    # 保存全部寄存器
    SAVE_CONTEXT
    # 取出用户 sp 并保存
    csrrw t1, sscratch, x0
    sd   t1, 256(sp)

    # 准备 C ABI 参数 → trap_dispatch(gprs, sepc, scause, stval)
    mv    a0, sp                # a0 = &TrapGprs
    csrr  a1, sepc
    csrr  a2, scause
    csrr  a3, stval
    call  trap_dispatch

    # 恢复用户 sp
    ld   t1, 256(sp)
    csrw  sscratch, t1
    RESTORE_CONTEXT

    # 恢复 sp
    csrrw sp, sscratch, sp
    bnez  sp, .L_trap_sret
    csrrw sp, sscratch, sp
.L_trap_sret:
    sret

# ================================================================
#  临时栈（4 KiB BSS，MMU 使能前使用）
# ================================================================

.align 12
.section ".bss"
.globl tmp_stack
tmp_stack:
    .zero 0x1000
tmp_stack_top:
