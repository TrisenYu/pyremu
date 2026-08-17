// SPDX-LICENSE-IDENTIFIER: GPL2.0
// (C) All rights reserved. Author: <kisfg@hotmail.com> in 2026

// M+S 模式内核: 进程调度 + trap 处理 + UART 驱动.
//
// 特性:
//   - Round-robin 调度 (CLINT MTI -> S 模式 STI)
//   - 最多 MAX_PROCS 个 U 模式进程
//   - 每进程独立栈 + 保护页 (Sv39)
//   - ECALL 系统调用分发 (report/exit/uart_putc/uart_puts/uart_getc/report_nq)
//   - 页错误 -> 终止进程, 调度下一个
//
// 链接: 与 prog_fib.o + prog_nqueen.o 共同链接.
// 符号: process_table, num_processes, current_pid 供 Python Loader 访问.

.section .text
.globl _start
.globl process_table
.globl num_processes
.globl current_pid
.globl uart_putc
.globl uart_puts
.globl uart_getc
.globl uart_putdec

// ============================================================
//  常量
// ============================================================
.equ UART_BASE,     0x10000000
.equ UART_TX,       0x00
.equ UART_RXDATA,   0x04
.equ UART_TXCTRL,   0x08
.equ UART_IP,       0x14

.equ CLINT_BASE,    0x02000000
.equ CLINT_MTIMECMP,0x4000
.equ CLINT_MTIME,   0xBFF8

.equ MAX_PROCS,     4
.equ PCB_SIZE,      296       // 48 + 31*8 GPR save area
.equ PCB_STATE_OFF,      0
.equ PCB_ENTRY_OFF,      8    // +4 padding after state for 8B alignment
.equ PCB_STACK_OFF,      16
.equ PCB_SEPC_OFF,       24
.equ PCB_SP_OFF,         32
.equ PCB_EXIT_OFF,       40
.equ PCB_GPR_OFF,       48   // GPR save area (x1-x31)

.equ PS_EMPTY,      0
.equ PS_READY,      1
.equ PS_RUNNING,    2

.equ TIMESLICE,     2000      // 每次调度分片 (cycles)

.equ RBUF_SIZE,     64
.equ RBUF_MASK,     63
.equ LINE_MAX,      20

.equ FRAME_SIZE,    336       // 88 kernel + 248 GPR save
.equ GPR_OFF,        88        // GPR area offset within frame
.equ N_GPRS,         31
.equ STACK_BASE_U,  0x80100000  // 第一个 U 栈基址

// Sv39
.equ PTE_V,    (1 << 0)
.equ PTE_RWXU, PTE_V | (1<<1) | (1<<2) | (1<<3) | (1<<4)
.equ PTE_RWU,  PTE_V | (1<<1) | (1<<2) | (1<<4)


// ============================================================
//  M 模式入口
// ============================================================
_start:
    la   sp, stack_top_m

    // UART TX 使能
    li   t0, UART_BASE + UART_TXCTRL
    li   t1, 1
    sw   t1, 0(t0)

    // 初始化 mtimecmp 防 boot 期间误触发定时器
    li   t0, CLINT_BASE + CLINT_MTIMECMP
    li   t1, -1
    sd   t1, 0(t0)

    // PMP: 全地址空间 R+W+X
    li   t0, 0x1F
    csrw pmpcfg0, t0
    li   t0, -1
    srli t0, t0, 10
    csrw pmpaddr0, t0

    // M trap 向量
    la   t0, m_trap_handler
    csrw mtvec, t0

    // 委派异常: ECALL + page faults -> S
    li   t0, 0xB100           // bits 8,12,13,15
    csrw medeleg, t0
    // 委派中断: MTI -> S
    li   t0, (1 << 7)
    csrw mideleg, t0
    // 使能 MTIE
    csrw mie, t0

    // mstatus: MPP=S, MPIE=1
    csrr t0, mstatus
    li   t1, ~(3 << 11)
    and  t0, t0, t1
    li   t1, (1 << 11) | (1 << 7)
    or   t0, t0, t1
    csrw mstatus, t0

    // -> S 模式入口
    la   t0, s_mode_boot
    csrw mepc, t0
    mret


m_trap_handler:
    // 处理未委派的 trap (简单跳过)
    csrrw t0, mscratch, t0
    csrr t0, mepc
    addi t0, t0, 4
    csrw mepc, t0
    csrrw t0, mscratch, t0
    mret


// ============================================================
//  S 模式入口
// ============================================================
s_mode_boot:
    la   sp, stack_top_s
    addi fp, sp, 0

    // sscratch ← S 栈顶
    la   t0, stack_top_s
    csrw sscratch, t0

    // S trap 向量
    la   t0, s_trap_handler
    csrw stvec, t0

    // sstatus: SPIE=1, SIE=1, SPP=U
    li   t0, (1 << 5)
    csrw sstatus, t0
    li   t0, (1 << 1)
    csrs sstatus, t0
    li   t0, (1 << 8)
    csrc sstatus, t0
    // SUM=1: 允许 S 模式访存 U 模式页 (读 faulting 指令、用户栈、进程切换)
    li   t0, (1 << 18)
    csrs sstatus, t0

    // 调试: 在 Sv39 前打印
    la   a0, str_boot
    call uart_puts

    // Sv39 页表
    call setup_sv39

    // 打印启动信息
    la   a0, str_boot
    call uart_puts

    // 首次调度 -> U 模式
    call schedule_next
    // schedule_next 已设置 sepc 和 sscratch (进程的 U sp)
    csrrw sp, sscratch, sp     // sp = U 栈, sscratch = S 栈 (schedule_next 的栈帧)
    la   t0, stack_top_s
    csrw sscratch, t0          // sscratch = S 栈顶 ← 关键: 统一复位到栈顶
    sret



// ============================================================
//  save_gprs_to_frame — 保存用户 x1-x31 到帧 GPR 区域 (sp 偏移 GPR_OFF)
//    破坏: t0 (值取自内核帧, 函数不依赖 t0 初始值)
// ============================================================
save_gprs_to_frame:
    ld   t0,  0(sp);  sd   t0, GPR_OFF + 0*8(sp)   // x1/ra
    csrr t0, sscratch;  sd   t0, GPR_OFF + 1*8(sp)  // x2/sp
    sd   x3,  GPR_OFF + 2*8(sp)
    sd   x4,  GPR_OFF + 3*8(sp)
    ld   t0, 32(sp);  sd   t0, GPR_OFF + 4*8(sp)   // x5/t0
    ld   t0, 40(sp);  sd   t0, GPR_OFF + 5*8(sp)   // x6/t1
    sd   x7,  GPR_OFF + 6*8(sp)
    ld   t0,  8(sp);  sd   t0, GPR_OFF + 7*8(sp)   // x8/s0/fp
    sd   x9,  GPR_OFF + 8*8(sp)
    ld   t0, 16(sp);  sd   t0, GPR_OFF + 9*8(sp)   // x10/a0
    ld   t0, 24(sp);  sd   t0, GPR_OFF +10*8(sp)   // x11/a1
    sd   x12, GPR_OFF +11*8(sp)
    sd   x13, GPR_OFF +12*8(sp)
    sd   x14, GPR_OFF +13*8(sp)
    sd   x15, GPR_OFF +14*8(sp)
    sd   x16, GPR_OFF +15*8(sp)
    sd   x17, GPR_OFF +16*8(sp)
    ld   t0, 56(sp);  sd   t0, GPR_OFF +17*8(sp)   // x18/s2
    ld   t0, 64(sp);  sd   t0, GPR_OFF +18*8(sp)   // x19/s3
    sd   x20, GPR_OFF +19*8(sp)
    sd   x21, GPR_OFF +20*8(sp)
    sd   x22, GPR_OFF +21*8(sp)
    sd   x23, GPR_OFF +22*8(sp)
    sd   x24, GPR_OFF +23*8(sp)
    sd   x25, GPR_OFF +24*8(sp)
    sd   x26, GPR_OFF +25*8(sp)
    sd   x27, GPR_OFF +26*8(sp)
    sd   x28, GPR_OFF +27*8(sp)
    sd   x29, GPR_OFF +28*8(sp)
    sd   x30, GPR_OFF +29*8(sp)
    sd   x31, GPR_OFF +30*8(sp)
    ret


// ============================================================
//  restore_gprs_from_frame — 从帧 GPR 区域恢复 x1-x31
//    内核 ra 暂存于 72(sp), 函数末尾恢复
// ============================================================
restore_gprs_from_frame:
    sd   ra, 72(sp)               // 暂存内核返回地址

    ld   x1,  GPR_OFF + 0*8(sp)
    ld   t0,  GPR_OFF + 1*8(sp);  csrw sscratch, t0  // x2/sp
    ld   x3,  GPR_OFF + 2*8(sp)
    ld   x4,  GPR_OFF + 3*8(sp)
    ld   x5,  GPR_OFF + 4*8(sp)
    ld   x6,  GPR_OFF + 5*8(sp)
    ld   x7,  GPR_OFF + 6*8(sp)
    ld   x8,  GPR_OFF + 7*8(sp)
    ld   x9,  GPR_OFF + 8*8(sp)
    ld   x10, GPR_OFF + 9*8(sp)
    ld   x11, GPR_OFF +10*8(sp)
    ld   x12, GPR_OFF +11*8(sp)
    ld   x13, GPR_OFF +12*8(sp)
    ld   x14, GPR_OFF +13*8(sp)
    ld   x15, GPR_OFF +14*8(sp)
    ld   x16, GPR_OFF +15*8(sp)
    ld   x17, GPR_OFF +16*8(sp)
    ld   x18, GPR_OFF +17*8(sp)
    ld   x19, GPR_OFF +18*8(sp)
    ld   x20, GPR_OFF +19*8(sp)
    ld   x21, GPR_OFF +20*8(sp)
    ld   x22, GPR_OFF +21*8(sp)
    ld   x23, GPR_OFF +22*8(sp)
    ld   x24, GPR_OFF +23*8(sp)
    ld   x25, GPR_OFF +24*8(sp)
    ld   x26, GPR_OFF +25*8(sp)
    ld   x27, GPR_OFF +26*8(sp)
    ld   x28, GPR_OFF +27*8(sp)
    ld   x29, GPR_OFF +28*8(sp)
    ld   x30, GPR_OFF +29*8(sp)
    ld   x31, GPR_OFF +30*8(sp)

    ld   ra, 72(sp)               // 恢复内核返回地址
    ret


// ============================================================
//  copy_gprs_to_pcb — 帧 GPR 区域 -> PCB[current_pid]
//    输入: t2 = PCB 基址  (调用前已计算)
//    使用 fp 寻址帧 (fp 在 trap 入口设定后不变)
// ============================================================
copy_gprs_to_pcb:
    addi t2, t2, PCB_GPR_OFF
    addi t1, fp, GPR_OFF - FRAME_SIZE   // fp - 248 = GPR area start
    li   t0, N_GPRS
1:  ld   t3, 0(t1);  sd   t3, 0(t2)
    addi t1, t1, 8;  addi t2, t2, 8
    addi t0, t0, -1;  bnez t0, 1b
    ret


// ============================================================
//  copy_gprs_from_pcb — PCB[pid] -> 帧 GPR 区域
//    输入: t3 = PCB 基址
// ============================================================
copy_gprs_from_pcb:
    addi t3, t3, PCB_GPR_OFF
    addi t1, fp, GPR_OFF - FRAME_SIZE
    li   t0, N_GPRS
1:  ld   t2, 0(t3);  sd   t2, 0(t1)
    addi t3, t3, 8;  addi t1, t1, 8
    addi t0, t0, -1;  bnez t0, 1b
    ret

// ============================================================
//  setup_sv39 — identity 映射 (代码/数据/UART + 多进程 U 栈)
//
//  页表层级均以 page_tables 符号的实际地址为基准计算 PPN,
//  不再硬编码地址, 适应不同链接布局.
// ============================================================
setup_sv39:
    addi sp, sp, -32
    sd   ra,  0(sp)
    sd   s0,  8(sp)
    sd   s1, 16(sp)

    la   s0, page_tables

    // s1 = PPN of page_tables (symbol address >> 12)
    srli s1, s0, 12

    // --- 构建指针 PTE 的辅助宏 (手动展开) ---
    //   RISC-V 标准连续编码: PTE = 0x20000001 | (((base + k) & 0x3FF) << 10)
    //   注: 当前布局下 PPN[1]=0, PPN[2]=1, 故可简化为上述公式.

    // L1[2] -> L2_hi (PPN = base + 1)
    addi t0, s1, 1
    andi t0, t0, 0x3FF
    slli t0, t0, 10
    li   t1, 0x20000001
    or   t0, t0, t1
    sd   t0, 16(s0)

    // L1[0] -> L2_lo (PPN = base + 2)
    addi t0, s1, 2
    andi t0, t0, 0x3FF
    slli t0, t0, 10
    li   t1, 0x20000001
    or   t0, t0, t1
    sd   t0, 0(s0)

    // L2_hi[0] -> L3_main (PPN = base + 3)
    li   t2, 0x1000
    add  t2, s0, t2             // t2 = &L2_hi[0] (写目标)
    addi t0, s1, 3
    andi t0, t0, 0x3FF
    slli t0, t0, 10
    li   t1, 0x20000001
    or   t0, t0, t1
    sd   t0, 0(t2)

    // L2_lo[128] -> L3_uart (PPN = base + 4)
    li   t2, 0x2000
    add  t2, s0, t2             // t2 = &L2_lo
    addi t0, s1, 4
    andi t0, t0, 0x3FF
    slli t0, t0, 10
    li   t1, 0x20000001
    or   t0, t0, t1
    sd   t0, 1024(t2)

    // L2_lo[16] -> L3_clint (PPN = base + 5)
    addi t0, s1, 5
    andi t0, t0, 0x3FF
    slli t0, t0, 10
    li   t1, 0x20000001
    or   t0, t0, t1
    sd   t0, 128(t2)

    // L3_main: 16 页 identity (代码/数据/BSS)
    li   t0, 0x3000
    add  s1, s0, t0
    li   t0, 0x000000002000001f
    li   t1, 16
1:
    sd   t0, 0(s1)
    addi s1, s1, 8
    li   t2, (1 << 10)
    add  t0, t0, t2
    addi t1, t1, -1
    bnez t1, 1b

    // 每进程 U 栈: pid 0..MAX_PROCS-1 各 1 页 (保护页在下方, 不映射)
    //   栈地址 = STACK_BASE_U + pid * 0x2000
    //   L3_main 索引 = (STACK_BASE_U >> 12) + pid * 2
    //   STACK_BASE_U = 0x80100000 -> VPN[0] = 0x100
    li   t0, 0x3000
    add  s1, s0, t0            // s1 = L3_main
    li   t3, MAX_PROCS         // 循环计数

    // 栈 PTE 基: PPN = STACK_BASE_U >> 12 = 0x80100, 每进程 +2
    li   t0, 0x0000000020040017  // PPN=0x80100, R+W+U
    li   t4, 0x100             // 第一个 L3 索引 (VA 0x80100000 的 VPN[0])
    slli t4, t4, 3             // 索引 -> 字节偏移 (×8)
    add  s1, s1, t4            // s1 = &L3_main[0x100]

2:
    sd   t0, 0(s1)             // 写栈页 PTE
    addi s1, s1, 16            // 下一进程 (+2 页: 跳过 1 保护页 + 1 栈页)
    li   t2, 0x800             // PPN += 2 页
    add  t0, t0, t2
    addi t3, t3, -1
    bnez t3, 2b

    // L3_uart[0]
    li   t0, 0x4000
    add  s1, s0, t0
    li   t0, 0x0000000004000017
    sd   t0, 0(s1)

    // L3_clint: 16 页 identity 映射 (CLINT @ 0x02000000, 64 KiB)
    li   t0, 0x5000
    add  s1, s0, t0
    li   t0, 0x0000000000800017  // PPN=0x02000, R+W+U
    li   t1, 16
1:
    sd   t0, 0(s1)
    addi s1, s1, 8
    li   t2, (1 << 10)
    add  t0, t0, t2
    addi t1, t1, -1
    bnez t1, 1b

    // satp: MODE=Sv39 (8), PPN 从 page_tables 运行时地址重新计算
    // (注: s1 已被后续循环覆盖, 此处基于 s0 重新导出 PPN)
    li   t0, 8
    slli t0, t0, 60
    srli t1, s0, 12             // t1 = PPN of page_tables
    li   t2, (1 << 44) - 1
    and  t1, t1, t2
    or   t0, t0, t1
    csrw satp, t0
    sfence.vma zero, zero

    ld   ra,  0(sp)
    ld   s0,  8(sp)
    ld   s1, 16(sp)
    addi sp, sp, 32
    ret


// ============================================================
//  timer_set_next — 设置下次定时器中断
// ============================================================
timer_set_next:
    li   t0, CLINT_BASE
    li   t2, CLINT_MTIME
    add  t2, t0, t2            // t2 = CLINT_BASE + MTIME
    ld   t1, 0(t2)             // 读当前 mtime
    li   t3, TIMESLICE
    add  t1, t1, t3
    li   t2, CLINT_MTIMECMP
    add  t2, t0, t2            // t2 = CLINT_BASE + MTIMECMP
    sd   t1, 0(t2)             // mtimecmp = mtime + TIMESLICE
    ret


// ============================================================
//  schedule_next — 调度下一个 READY 进程
//
//   若 current_pid ≥ 0: 保存当前进程 sepc/sp 到 PCB
//   查找下一个 READY 进程, 恢复其 sepc/sp
//   设置 mtimecmp, SRET 到 U 模式
//
//   从 s_trap_handler 调用, 最终会执行 sret
// ============================================================
schedule_next:
    addi sp, sp, -16
    sd   ra, 0(sp)

    // -- 保存当前进程 --
    la   t0, current_pid
    lw   t1, 0(t0)
    bltz t1, sched_find_next   // current_pid < 0 -> 首次调度

    // PCB 地址 = process_table + current_pid * PCB_SIZE
    la   t2, process_table
    li   t3, PCB_SIZE
    mul  t3, t1, t3
    add  t2, t2, t3

    // state ← READY
    li   t3, PS_READY
    sw   t3, PCB_STATE_OFF(t2)

    // saved_sepc ← sepc
    csrr t3, sepc
    sd   t3, PCB_SEPC_OFF(t2)

    // saved_sp ← (U sp 保存在 sscratch 中)
    //   注: s_trap_handler 入口 csrrw sp, sscratch, sp 将 U sp 存入了 sscratch
    csrr t3, sscratch
    sd   t3, PCB_SP_OFF(t2)

sched_find_next:
    // 从 current_pid+1 开始找下一个 READY
    la   t0, current_pid
    lw   t1, 0(t0)
    addi t1, t1, 1             // start = current_pid + 1
    li   t4, MAX_PROCS

    li   t5, 0                  // 扫描计数器 (最多扫描 MAX_PROCS 次)

sched_scan:
    // wrap: t2 = t1 % MAX_PROCS
    mv   t2, t1
    blt  t2, zero, sched_wrap_neg
    li   t0, MAX_PROCS
    blt  t2, t0, sched_no_wrap
sched_wrap_neg:
    li   t0, MAX_PROCS
    rem  t2, t2, t0            // RV64 R-type rem: rem rd, rs1, rs2 (with register divisor)
    bltz t2, sched_add_mod
    j    sched_no_wrap
sched_add_mod:
    add  t2, t2, t0
sched_no_wrap:

    // PCB[t2].state
    la   t3, process_table
    li   t6, PCB_SIZE
    mul  t6, t2, t6
    add  t3, t3, t6
    lw   t6, PCB_STATE_OFF(t3)

    li   t0, PS_READY
    beq  t6, t0, sched_found

    addi t1, t1, 1
    addi t5, t5, 1
    li   t0, MAX_PROCS
    blt  t5, t0, sched_scan    // 未扫满一轮, 继续

    // 无 READY 进程 -> 停机
    // (stop_machine 在文件末尾定义)
    j    stop_machine

sched_found:
    // current_pid ← t2
    la   t0, current_pid
    sw   t2, 0(t0)

    // PCB[t2].state ← RUNNING
    li   t5, PS_RUNNING
    sw   t5, PCB_STATE_OFF(t3)

    // 恢复 sepc ← PCB[t2].saved_sepc (若首次则为 entry_pc)
    ld   t5, PCB_SEPC_OFF(t3)
    beqz t5, 1f                // sepc==0 -> 首次, 用 entry_pc
    csrw sepc, t5
    j    2f
1:
    ld   t5, PCB_ENTRY_OFF(t3)
    csrw sepc, t5
2:

    // 恢复 U sp -> sscratch (s_trap_done 会交换回 sp)
    ld   t5, PCB_SP_OFF(t3)
    bnez t5, 3f                // sp 有值 -> 恢复
    ld   t5, PCB_STACK_OFF(t3) // 首次 -> 用 stack_top
3:
    // 写入 sscratch, 供 s_trap_done 的 csrrw 交换
    csrw sscratch, t5

    // 重设定时器
    call timer_set_next

    ld   ra, 0(sp)
    addi sp, sp, 16
    ret


// ============================================================
//  terminate_current — 终止当前进程
//    a0 = exit code
// ============================================================
terminate_current:
    la   t0, current_pid
    lw   t1, 0(t0)
    bltz t1, sched_find_next   // 无当前进程 -> 直接调度

    la   t2, process_table
    li   t3, PCB_SIZE
    mul  t3, t1, t3
    add  t2, t2, t3

    li   t3, PS_EMPTY
    sw   t3, PCB_STATE_OFF(t2)
    sw   a0, PCB_EXIT_OFF(t2)

    la   a0, str_exit
    call uart_puts
    lw   a0, PCB_EXIT_OFF(t2)
    call uart_putdec
    li   a0, '\n'
    call uart_putc

    // 标记 current_pid = -1: 阻止 schedule_next 重新保存已终止的进程
    la   t0, current_pid
    li   t3, -1
    sw   t3, 0(t0)

    // 设置 ra = s_trap_done: schedule_next 的 ret 将直接跳转到
    // s_trap_done (恢复 trap 帧 -> sret 到新进程),
    // 而非 sys_exit (会落入 s_trap_fault)
    la   ra, s_trap_done
    j    schedule_next


// ============================================================
//  S 模式 trap handler
// ============================================================
s_trap_handler:
    csrrw sp, sscratch, sp     // sp = S 栈, sscratch = 用户 sp

    addi sp, sp, -FRAME_SIZE   // 336B 帧
    sd   ra,  0(sp)
    sd   fp,  8(sp)
    sd   a0, 16(sp)
    sd   a1, 24(sp)
    sd   t0, 32(sp)
    sd   t1, 40(sp)
    sd   s2, 56(sp)
    sd   s3, 64(sp)
    addi fp, sp, FRAME_SIZE

    // 保存全部用户 GPR 到帧
    call save_gprs_to_frame

    // 帧 GPR -> PCB (若有当前进程)
    la   t2, current_pid
    lw   t1, 0(t2)
    bltz t1, 1f
    la   t2, process_table
    li   t0, PCB_SIZE
    mul  t1, t1, t0
    add  t2, t2, t1             // t2 = &PCB[current_pid]
    call copy_gprs_to_pcb
1:

    csrr t0, scause

    // 中断? (bit 63 置位 -> 负数)
    bltz t0, s_trap_interrupt

    // ---- 异常 ----
    // ECALL from U-mode? (code 8)
    andi t1, t0, 0xFF
    li   t2, 8
    bne  t1, t2, s_trap_fault

    // ECALL 分发
    li   t1, 0                  // report fib
    beq  a7, t1, sys_report_fib
    li   t1, 1                  // exit
    beq  a7, t1, sys_exit
    li   t1, 2                  // uart_putc
    beq  a7, t1, sys_uart_putc
    li   t1, 3                  // uart_puts
    beq  a7, t1, sys_uart_puts
    li   t1, 4                  // uart_getc
    beq  a7, t1, sys_uart_getc
    li   t1, 5                  // report nqueen
    beq  a7, t1, sys_report_nq

    // 未知 syscall -> exit(255)
    li   a0, 255
    j    sys_exit

// -- syscall handlers --

// ---- ECALL 后推进 sepc ----
s_trap_advance:
    csrr t0, sepc
    addi t0, t0, 4
    csrw sepc, t0
    j    s_trap_done

// ---- syscall handlers ----

sys_report_fib:
    // a0=fib result, a1=n
    ld   s2, 16(sp)
    ld   s3, 24(sp)
    la   a0, str_fib
    call uart_puts
    mv   a0, s3
    call uart_putdec
    la   a0, str_eq
    call uart_puts
    mv   a0, s2
    call uart_putdec
    li   a0, '\n'
    call uart_putc
    j    s_trap_advance

sys_report_nq:
    // a0=n, a1=solutions
    ld   s2, 16(sp)
    ld   s3, 24(sp)
    la   a0, str_nq
    call uart_puts
    mv   a0, s2
    call uart_putdec
    la   a0, str_eq
    call uart_puts
    mv   a0, s3
    call uart_putdec
    li   a0, '\n'
    call uart_putc
    j    s_trap_advance

sys_uart_putc:
    ld   a0, 16(sp)
    call uart_putc
    j    s_trap_advance

sys_uart_puts:
    ld   a0, 16(sp)
    call uart_puts
    j    s_trap_advance

sys_uart_getc:
    call uart_getc
    sd   a0, 16(sp)
    j    s_trap_advance

sys_exit:
    call terminate_current      // terminate_current 直接跳转到 schedule_next
    // 不会返回此处

s_trap_fault:
    // 页错误等: 打印诊断并终止
    csrr s2, scause
    la   a0, str_fault
    call uart_puts
    mv   a0, s2
    call uart_putdec
    li   a0, '\n'
    call uart_putc
    mv   a0, s2                // exit code = scause
    call terminate_current
    // 不会返回

// ---- 中断 ----
s_trap_interrupt:
    // 检查是否为 STI (委托的 MTI: scause = 0x8000000000000007)
    csrr t0, scause
    li   t1, 0x8000000000000007
    bne  t0, t1, 1f

    // 保存当前进程的 sepc/sp
    la   t0, current_pid
    lw   t1, 0(t0)
    bltz t1, 2f               // 无当前进程 -> 直接调度

    la   t2, process_table
    li   t3, PCB_SIZE
    mul  t3, t1, t3
    add  t2, t2, t3

    csrr t3, sepc
    sd   t3, PCB_SEPC_OFF(t2)
    csrr t3, sscratch
    sd   t3, PCB_SP_OFF(t2)

    j    2f

1:
    // 其他中断: 忽略

2:
    call schedule_next
    // schedule_next 会跳转到 s_trap_done 或直接到新进程

s_trap_done:
    // PCB GPR -> 帧 (恢复当前进程的寄存器)
    la   t0, current_pid
    lw   t1, 0(t0)
    bltz t1, 1f
    la   t3, process_table
    li   t2, PCB_SIZE
    mul  t2, t1, t2
    add  t3, t3, t2             // t3 = &PCB[current_pid]
    call copy_gprs_from_pcb
1:
    // 从帧恢复全部用户寄存器
    call restore_gprs_from_frame
    // 内核帧恢复 (覆盖可能被 ECALL 修改的 a0 等)
    ld   ra,  0(sp)
    ld   fp,  8(sp)
    ld   a0, 16(sp)
    ld   a1, 24(sp)
    ld   t0, 32(sp)
    ld   t1, 40(sp)
    ld   s2, 56(sp)
    ld   s3, 64(sp)
    addi sp, sp, FRAME_SIZE
    csrrw sp, sscratch, sp     // sp = 用户 sp, sscratch = S 栈顶
    sret


	// ============================================================
	//  S 模式 trap handler (嵌套中断变体)
	// ============================================================
	s_trap_handler_nested_enabled:
	    csrrw sp, sscratch, sp
	    addi sp, sp, -88
	    sd   ra,  0(sp)
	    sd   fp,  8(sp)
	    sd   a0, 16(sp)
	    sd   a1, 24(sp)
	    sd   t0, 32(sp)
	    sd   t1, 40(sp)
	    sd   s2, 56(sp)
	    sd   s3, 64(sp)
	    addi fp, sp, 88
	    csrr t0, sscratch
	    sd   t0, 72(sp)
	    csrw sscratch, sp
	    li   t0, (1 << 1)
	    csrs sstatus, t0
	    csrr t0, scause
	    bltz t0, s_trap_interrupt
	    andi t1, t0, 0xFF
	    li   t2, 8
	    bne  t1, t2, s_trap_fault
	    li   t1, 0
	    beq  a7, t1, sys_report_fib
	    li   t1, 1
	    beq  a7, t1, sys_exit
	    li   t1, 2
	    beq  a7, t1, sys_uart_putc
	    li   t1, 3
	    beq  a7, t1, sys_uart_puts
	    li   t1, 4
	    beq  a7, t1, sys_uart_getc
	    li   t1, 5
	    beq  a7, t1, sys_report_nq
	    li   a0, 255
	    j    sys_exit


	s_trap_done_nested_enabled:
	    li   t0, (1 << 1)
	    csrc sstatus, t0
	    ld   t0, 72(sp)
	    csrw sscratch, t0
	    ld   ra,  0(sp)
	    ld   fp,  8(sp)
	    ld   a0, 16(sp)
	    ld   a1, 24(sp)
	    ld   t0, 32(sp)
	    ld   t1, 40(sp)
	    ld   s2, 56(sp)
	    ld   s3, 64(sp)
	    addi sp, sp, 88
	    csrrw sp, sscratch, sp
	    sret


// ============================================================
//  UART 发送
// ============================================================
uart_putc:
    li   t0, UART_BASE + UART_TX
    li   t2, 0x80000000
1:
    lw   t1, UART_TXCTRL(t0)
    and  t1, t1, t2
    bnez t1, 1b
    sb   a0, 0(t0)
    ret


// ============================================================
//  UART 接收 (ring buffer)
// ============================================================
uart_poll:
    addi sp, sp, -16
    sd   ra, 0(sp)
    li   a0, 0
    li   t2, 16
1:
    beqz t2, 2f
    li   t0, UART_BASE
    lw   t1, UART_IP(t0)
    andi t1, t1, 2
    beqz t1, 2f
    la   t0, rbuf_count
    lw   t1, 0(t0)
    li   t0, RBUF_SIZE
    bge  t1, t0, 2f
    li   t0, UART_BASE
    lbu  t1, UART_RXDATA(t0)
    mv   a1, t1
    call rbuf_put
    addi a0, a0, 1
    addi t2, t2, -1
    j    1b
2:
    ld   ra, 0(sp)
    addi sp, sp, 16
    ret

rbuf_put:
    la   t0, rbuf_count; lw t1, 0(t0); li t2, RBUF_SIZE
    bge t1, t2, rbuf_put_full
    la   t0, rbuf_head; lw t2, 0(t0); andi t2, t2, RBUF_MASK
    la   t0, rbuf; add t2, t0, t2; sb a1, 0(t2)
    la   t0, rbuf_head; lw t2, 0(t0); addi t2, t2, 1; sw t2, 0(t0)
    la   t0, rbuf_count; lw t2, 0(t0); addi t2, t2, 1; sw t2, 0(t0)
    li   a0, 0; ret
rbuf_put_full: li a0, 1; ret

rbuf_get:
    la   t0, rbuf_count; lw t1, 0(t0); beqz t1, rbuf_get_empty
    la   t0, rbuf_tail; lw t2, 0(t0); andi t2, t2, RBUF_MASK
    la   t0, rbuf; add t2, t0, t2; lbu a0, 0(t2)
    la   t0, rbuf_tail; lw t2, 0(t0); addi t2, t2, 1; sw t2, 0(t0)
    la   t0, rbuf_count; lw t2, 0(t0); addi t2, t2, -1; sw t2, 0(t0)
    li   a1, 0; ret
rbuf_get_empty: li a0, 0; li a1, 1; ret

uart_getc:
    addi sp, sp, -16
    sd   ra, 0(sp)
1:
    call rbuf_get
    beqz a1, 2f
    call uart_poll
    call rbuf_get
    beqz a1, 2f
    // 无数据: 短暂重试, 上限 1000 次避免死循环阻塞定时器
    //   但为了防止嵌套 trap 问题, 直接返回 0
    li   a0, 0
    j    2f
2:
    ld   ra, 0(sp)
    addi sp, sp, 16
    ret


// ============================================================
//  uart_puts / uart_putdec
// ============================================================
uart_puts:
    addi sp, sp, -16; sd ra, 0(sp); sd s0, 8(sp); mv s0, a0
1:  lbu a0, 0(s0); beqz a0, 2f
    call uart_putc; addi s0, s0, 1; j 1b
2:  ld ra, 0(sp); ld s0, 8(sp); addi sp, sp, 16; ret

uart_putdec:
    addi sp, sp, -48; sd ra, 0(sp); sd s0, 8(sp); sd s1, 16(sp)
    mv s0, a0; addi s1, sp, 40; sb zero, 0(s1); addi s1, s1, -1
    bnez s0, putdec_loop
    li t0, '0'; sb t0, 0(s1); addi s1, s1, -1; j putdec_out
putdec_loop:
    li t0, 10; divu t1, s0, t0; remu t2, s0, t0
    addi t2, t2, '0'; sb t2, 0(s1); addi s1, s1, -1; mv s0, t1
    bnez s0, putdec_loop
putdec_out:
    addi a0, s1, 1; call uart_puts
    ld ra, 0(sp); ld s0, 8(sp); ld s1, 16(sp); addi sp, sp, 48; ret


// ============================================================
//  数据 / rodata
// ============================================================
.section .data
.align 2
.globl current_pid
current_pid:
    .word -1

.section .rodata
.align 2
str_boot:  .asciz "kernel: S-mode boot (multi-proc)\n"
str_fault: .asciz "fault: scause="
str_exit:  .asciz "exit: code="
str_fib:   .asciz "fib("
str_nq:    .asciz "nq("
str_eq:    .asciz ")="
str_dbg1:  .asciz "[rpt]"
str_dbg2:  .asciz "[sc="
str_dbg3:  .asciz " a7="

// 占位符 — ECALL handler 中使用
a0_save: .dword 0
a1_reg:  .dword 0
s2_reg:  .dword 0
a1_val:  .dword 0
str_fib_ptr: .dword 0


// ============================================================
//  BSS
// ============================================================
.section .bss

.globl process_table
.globl num_processes

.align 4
process_table:
    .skip MAX_PROCS * PCB_SIZE   // 4 * 296 = 1184 bytes
num_processes:
    .skip 4

.align 12
page_tables:
    .skip 6 * 4096

rbuf:       .skip RBUF_SIZE
rbuf_head:  .skip 4
rbuf_tail:  .skip 4
rbuf_count: .skip 4

.align 4
    .skip 4096
stack_top_m:
    .skip 4096
stack_top_s:

// ---- 停机序列 (back to .text) ----
// 全部进程完成后执行: semihosting SYS_EXIT (QEMU 裸机测试同款约定),
// 一个 hart 执行即停止整个引擎 (状态保留), run() 返回 / 调试器接管.
.section .text
stop_machine:
    li   a0, 0x18
    slli x0, x0, 0x1f
    ebreak
    srai x0, x0, 7
    // 兜底: 纯 Python 路径无 semihosting, WFI 自旋
stop_machine_idle:
    wfi
    j    stop_machine_idle
