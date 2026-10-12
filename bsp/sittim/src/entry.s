# 飞地 S-mode 入口与陷态向量。

# ================================================================
#  SAVE / RESTORE 宏, 264 字节帧：32 GPR + 陷入前的 sp 槽
#  必须在使用前定义, LLVM 集成汇编器要求前置声明。
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
    # ---- 保存 M-mode 传入的参数, 即 callee-saved 寄存器 ----
    # M-mode 在 CREATE ecall 返回前设置:
    #   a0 = enclave_id, a1 = base_pa, a2 = payload_size,
    #   a3 = 授予的分区总数
    mv   s0, a0                 # s0 = enclave_id
    mv   s1, a1                 # s1 = base_pa
    mv   s2, a2                 # s2 = payload_size
    mv   s3, a3                 # s3 = 授予的分区总数

    # ---- 设置临时栈 ----
_setup_tmp_stk:
    auipc sp, %pcrel_hi(tmp_stack_top)
    addi  sp, sp, %pcrel_lo(_setup_tmp_stk)
    # ---- 为 BootInfo 分配栈空间 (3 × u64 = 24 字节, 对齐到 32) ----
    addi sp, sp, -32

    # ---- 应用 R_RISCV_RELATIVE 重定位 (PIE) ----
    # s1 = base_pa (运行时加载地址), link base = 0.
    # 条目表的两端 _rela_dyn_start / _rela_dyn_end 由链接脚本给出,
    # 循环逐个处理该区间内的全部 R_RISCV_RELATIVE (type=3) 条目.
.L0_reloc_s:
    auipc t0, %pcrel_hi(_rela_dyn_start)
    addi  t0, t0, %pcrel_lo(.L0_reloc_s)
    # auipc+addi 已通过 PC 相对寻址得到运行时地址 (PC 含 base_pa),
    # 无需再加 s1, 否则会 double-count base_pa.
.L0_reloc_e:
    auipc t1, %pcrel_hi(_rela_dyn_end)
    addi  t1, t1, %pcrel_lo(.L0_reloc_e)
    beq   t0, t1, .L_reloc_done
.L_reloc_loop:
    ld    t2, 0(t0)                # r_offset
    ld    t3, 16(t0)               # r_addend (跳过 8 字节 r_info)
    add   t2, s1, t2               # 目标地址 = base_pa + r_offset
    add   t3, s1, t3               # 重定位值 = base_pa + r_addend
    sd    t3, 0(t2)                # *(base_pa + r_offset) = base_pa + r_addend
    addi  t0, t0, 24               # sizeof(Elf64_Rela) = 24
    bltu  t0, t1, .L_reloc_loop
.L_reloc_done:

    # MMU使能前让 BSS 清零
    # PIE 重定位后, _bss_start / _bss_end 的地址为运行时 PA.
    # EnclaveContext (含 page_table_root) 位于 BSS 内, 必须初始化为零.
.L0_bss_s:
    auipc t0, %pcrel_hi(_bss_start)
    addi  t0, t0, %pcrel_lo(.L0_bss_s)
.L0_bss_e:
    auipc t1, %pcrel_hi(_bss_end)
    addi  t1, t1, %pcrel_lo(.L0_bss_e)
    beq   t0, t1, _bss_reset_done
.L_bss_loop:
    sd    zero, 0(t0)
    addi  t0, t0, 8
    bltu  t0, t1, .L_bss_loop
_bss_reset_done:
    # 设置 main_before_mmu 参数
    # 	out: *mut BootInfo,
	#   enclave_id: u64,
	#   base_pa: u64,
	#   payload_size: u64,
    #   partition_num: u64
    mv   a0, sp                 # a0 = &BootInfo
    mv   a1, s0                 # a1 = enclave_id
    mv   a2, s1                 # a2 = base_pa
    mv   a3, s2                 # a3 = payload_size
    mv   a4, s3                 # a4 = partition_num
    # 注册相对于pc的早期陷态处理
setup_early_stvec:
    auipc t0, %pcrel_hi(early_trap)
    addi  t0, t0, %pcrel_lo(setup_early_stvec)
    csrw stvec, t0
    # Bare 模式调用阶段一 (PC 相对 call)
    call main_before_mmu
    # 从栈上加载返回的 BootInfo 字段
    ld   a0, 0(sp)              # a0 = satp
    ld   a1, 8(sp)              # a1 = smode_sp
    ld   a2, 16(sp)             # a2 = va_offset
    addi sp, sp, 32             # 恢复临时栈指针
.L0_temp_stvec: # 切换到虚拟地址并启用 MMU
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
    add  s7, s7, a2             # PA -> VA (a2 = va_offset 仍有效)
    csrw stvec, s7
    # 以及其余初始化操作
    call main_after_mmu
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
    bnez  sp, save_user_ctx      # sp != 0 -> 用户态陷态

    # 内核陷态：sscratch == 0
    csrr  sp, sstatus
    andi  sp, sp, 0x100         # SPP bit
    bnez  sp, save_kern_ctx     # SPP=1 -> 内核陷态

    # 新线程首次陷态：SPP=0，需分配内核栈
    csrrw sp, sscratch, sp      # 恢复 sp
    SAVE_CONTEXT
    jal   alloc_smode_stack     # 分配内核栈 -> a0
    csrw  sscratch, a0
    RESTORE_CONTEXT
    csrrw sp, sscratch, sp
    j     save_user_ctx

save_kern_ctx:
    # 内核陷态的帧存放于临时栈之内: 帧的起始地址不低于 tmp_stack, 末端不高于 tmp_stack_top,
    # 整帧因而落在临时栈之内, 与陷入前的 sp 的取值无关.
    #
    # 帧的末端按下列取值: 陷入前的 sp 落在 [tmp_stack + 264, tmp_stack_top) 之内时取该值,
    # 帧自该值向低地址生长; 该值不在此区间之内时取 tmp_stack_top, 帧自栈顶向低地址生长.
    #
    # 进入本段时 sp 为 0, 陷入前的 sp 存放在 sscratch 内. 本段借用 t1 存放它: 与 sscratch 交换
    # 后 t1 取得陷入前的 sp, 其原值暂存于 sscratch, 在 SAVE_CONTEXT 之前换回, 故帧内记录的仍是
    # 本段之前的 t1 取值. 其余寄存器一律不改动.
    csrrw t1, sscratch, t1      # t1 = 陷入前的 sp, sscratch = t1 的原值
.L_kern_min_sp:
    auipc sp, %pcrel_hi(tmp_stack)
    addi  sp, sp, %pcrel_lo(.L_kern_min_sp)
    addi  sp, sp, 264           # 帧完整落在临时栈内时陷入前的 sp 的最小取值
    bgeu  t1, sp, .L_kern_check_top
.L_kern_end_at_top:
    auipc sp, %pcrel_hi(tmp_stack_top)
    addi  sp, sp, %pcrel_lo(.L_kern_end_at_top)
    j     .L_kern_frame_ready
.L_kern_check_top:
    auipc sp, %pcrel_hi(tmp_stack_top)
    addi  sp, sp, %pcrel_lo(.L_kern_check_top)
    bgeu  t1, sp, .L_kern_frame_ready
    mv    sp, t1
.L_kern_frame_ready:
    csrrw t1, sscratch, t1      # t1 取回原值, sscratch 取回陷入前的 sp
    SAVE_CONTEXT
    # 记下陷入前的 sp: 由 sscratch 取出后存入陷入前的 sp 槽位, sscratch 同时清零
    csrrw t1, sscratch, x0
    sd    t1, 256(sp)
    j     trap_common

save_user_ctx:
    # 保存全部寄存器
    SAVE_CONTEXT
    # 记下陷入前的 sp: 用户 sp 由 sscratch 取出后存入陷入前的 sp 槽位
    csrrw t1, sscratch, x0
    sd   t1, 256(sp)

trap_common:
    # 设置trap_dispatch(gprs, sepc, scause, stval)的C ABI参数
    mv    a0, sp                # a0 = &TrapGprs
    csrr  a1, sepc
    csrr  a2, scause
    csrr  a3, stval
    call  trap_dispatch

    # a0 = trap_dispatch 返回: 0 表示不切换; 非 0 为目标线程陷态栈帧基址.
    # 切换即 mv sp, a0 (sp 由当前帧换到目标帧), 随后统一走恢复路径.
    beqz  a0, not_alter_thread
    mv    sp, a0
not_alter_thread:
    # sstatus.SPP 给出 sret 的返回目标: 取 1 时返回 S 模式, 取 0 时返回 U 模式
    # sstatus 在 RESTORE_CONTEXT 之前读取: 该宏会把帧内 t0 的取值装入 t0
    csrr  t0, sstatus
    andi  t0, t0, 0x100
    bnez  t0, ret_to_smode

    # 返回 U 模式: sscratch 取帧内陷入前的 sp
    ld   t1, 256(sp)
    csrw  sscratch, t1
    RESTORE_CONTEXT
    csrrw sp, sscratch, sp      # sp = 陷入前的 sp, sscratch = 帧的末端
    j     ret_to_user

ret_to_smode:
    # 返回 S 模式: sscratch 在 S 模式之内保持 0, sp 取帧内陷入前的 sp
    RESTORE_CONTEXT
    addi  sp, sp, -264          # sp = 帧基址
    ld    sp, 256(sp)           # sp = 陷入前的 sp
ret_to_user:
    sret

# ================================================================
#  临时栈, 16 KiB, 位于 .bss
#  MMU 使能之前由 _start 使用; MMU 使能之后由陷态入口存放内核陷态的陷态栈帧,
#  陷态分发所用的栈也取自本栈. 本栈尺寸与线程内核栈相同.
# ================================================================

.align 12
.section ".bss"
.globl tmp_stack
tmp_stack:
    .zero 0x4000
tmp_stack_top:
