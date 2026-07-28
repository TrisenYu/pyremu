//! Batch instruction execution loop — full RISC-V dispatch (Phases A–E).
//!
//! ``run_batch`` is the single FFI entry point that replaces the Python
//! fetch-decode-execute loop.  It runs up to *max_instrs* instructions
//! across all non-halted harts, handling all RV64IMAC instructions inline.

use std::cell::Cell;

use crate::decode::decode_fields;
use crate::handlers::{
    handle_amo, handle_compressed, handle_fp_load, handle_fp_store, handle_load, handle_store,
    handle_system, pmp_ok, ClintCtx, DevCtx, PmpCtx, EXIT_SENTINEL,
};
use crate::op_dispatcher::{
    handle_alu, handle_auipc, handle_br, handle_fence, handle_fp_fma, handle_fp_op, handle_jal,
    handle_jalr, handle_lui, handle_op32, handle_op_imm, handle_op_imm32,
};
use crate::state::{
    exit_reason, riscv_mode, BatchResult, FfiClintCtx, FfiDevCtx, FfiPmpCtx, FfiVirtIoCtx,
    HartState, MemCtx,
};
use crate::translate::{translate_va, TranslateFault, WalkCtx};
use crate::trap::{deliver_illegal_instruction, deliver_trap, exc_code, mcause_val};

// ============================================================
//  Instruction fetch
// ============================================================

#[inline]
fn fetch_instr(ram: *const u8, ram_size: u64, ram_base: u64, pa: u64) -> Option<u32> {
    let offset = pa.wrapping_sub(ram_base);
    if offset > ram_size.saturating_sub(4) {
        return None;
    }
    let ptr = unsafe { ram.add(offset as usize) };
    let b0 = unsafe { *ptr } as u32;
    let b1 = unsafe { *ptr.add(1) } as u32;
    let b2 = unsafe { *ptr.add(2) } as u32;
    let b3 = unsafe { *ptr.add(3) } as u32;
    Some(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24))
}

// ============================================================
//  Interrupt checking (Phase E)
// ============================================================

/// Interrupt priority order (RISC-V Privileged Spec §3.1.9):
/// MEI > MSI > MTI > SEI > SSI > STI
const MIE_MEIE: u64 = 1 << 11; // M-mode external
const MIE_MSIE: u64 = 1 << 3; // M-mode software
const MIE_MTIE: u64 = 1 << 7; // M-mode timer
const MIE_SEIE: u64 = 1 << 9; // S-mode external
const MIE_SSIE: u64 = 1 << 1; // S-mode software
const MIE_STIE: u64 = 1 << 5; // S-mode timer

/// Check if any interrupt is pending and enabled. Returns (cause_code, is_m_mode_interrupt)
/// or None.  Respects delegation via ``mideleg``.
fn check_pending_interrupts(state: &HartState) -> Option<(u64, bool)> {
    let pending = state.mip & state.mie;
    if pending == 0 {
        return None;
    }

    // Priority-ordered check (each returns immediately on first hit)
    let checks: [(u64, u64, u64); 6] = [
        (MIE_MEIE, 11, 0), // MEI
        (MIE_MSIE, 3, 0),  // MSI
        (MIE_MTIE, 7, 0),  // MTI
        (MIE_SEIE, 9, 1),  // SEI (delegatable)
        (MIE_SSIE, 1, 1),  // SSI (delegatable)
        (MIE_STIE, 5, 1),  // STI (delegatable)
    ];

    for (mask, cause, delegatable) in &checks {
        if pending & mask == 0 {
            continue;
        }

        // --- delegatable interrupt (SEI / SSI / STI) ---
        if *delegatable != 0 {
            let delegated = state.mideleg & mask != 0;
            if delegated && state.mode < riscv_mode::M {
                // S-mode global interrupt enable
                if state.mstatus & (1 << 1) == 0 {
                    continue; // SIE=0 -> skip, try next priority
                }
                return Some((*cause, false)); // deliver to S-mode
            }
            // not delegated, or currently in M-mode -> handle as M-mode
        }

        // --- M-mode interrupt ---
        // In S/U mode, M-level interrupts always preempt
        if state.mode < riscv_mode::M {
            return Some((*cause, true));
        }
        // In M-mode: need MIE=1
        if state.mstatus & (1 << 3) != 0 {
            return Some((*cause, true));
        }
        // MIE=0 in M-mode -> no interrupts taken
        return None;
    }
    None
}

/// Check for interrupts at instruction boundary.
/// Returns true if an interrupt was delivered (hart will continue executing
/// the trap handler inline — PC already redirected by ``deliver_trap``).
///
/// We deliberately do NOT set ``exit_reason = TRAP`` here.  Setting it would
/// cause the entire batch to exit on every interrupt, which is catastrophic for
/// multi-hart performance: after the sending hart yields its slice via the
/// cross-hart IPI mechanism, the waking hart takes the interrupt and immediately
/// ends the batch.  In the *next* batch the sending hart is still spin-waiting
/// (``tlb_sync``) but writes no more MSIP, so the yield mechanism never fires
/// again — the sender burns its full 512-instr slice every round while the
/// target never gets enough CPU time to acknowledge.
///
/// By keeping interrupt delivery fully inline the entire acknowledge cycle
/// (trap handler -> clear MSIP -> MRET -> resume) completes within the same batch.
fn check_and_deliver_interrupt(
    state: &mut HartState,
    result: &mut BatchResult,
    _clint: &ClintCtx,
) -> bool {
    if let Some((cause, is_m_mode)) = check_pending_interrupts(state) {
        let code = mcause_val(cause, true);
        deliver_trap(state, code, 0, result);
        result.trap_cause = code as u32;
        result.trap_is_interrupt = 1;
        result.trap_delegated = if is_m_mode { 0 } else { 1 };
        result.trap_tval = 0;

        // NOTE: do NOT clear CLINT MSIP here.
        // The firmware trap handler (sbi_ipi_process -> sbi_ipi_raw_clear)
        // is responsible for clearing its own interrupt source.  Clearing it
        // before the handler runs risks that sbi_trap_handler fails to call
        // sbi_ipi_process (e.g. due to a stale mip read), leaving ipi_type
        // permanently pending -> cross-hart TLB shootdown deadlock (observed
        // during Linux SMP boot).
        //
        // If the firmware fails to clear MSIP, the next check_pending_interrupts
        // will re-deliver.  A storm is better than a silent deadlock; the
        // firmware *always* clears MSIP in sbi_ipi_raw_clear, so the storm is
        // only theoretical.
        true
    } else {
        false
    }
}

// ============================================================
//  Full dispatch table (Phases A-D)
// ============================================================

/// Dispatch a 32-bit instruction. Returns advance (0/4) or EXIT_SENTINEL.
fn dispatch(
    state: &mut HartState,
    f: &crate::decode::DecodedFields,
    instr: u32,
    result: &mut BatchResult,
    ctx: &WalkCtx,
    hart_id: u8,
    pmp: &PmpCtx,
    dev: &DevCtx,
    clint: &ClintCtx,
) -> u64 {
    match f.opcode {
        0b01100_11 => handle_alu(state, f, instr, result),
        0b00100_11 => handle_op_imm(state, f, instr, result),
        0b00110_11 => handle_op_imm32(state, f, instr, result),
        0b01110_11 => handle_op32(state, f, instr, result),
        0b01101_11 => handle_lui(state, f),
        0b00101_11 => handle_auipc(state, f),
        0b11011_11 => handle_jal(state, f),
        0b11001_11 => handle_jalr(state, f),
        0b11000_11 => handle_br(state, f, instr, result),
        0b00011_11 => handle_fence(state, f, instr, result),

        // Phase B: Loads / Stores
        0b00000_11 => handle_load(state, f, instr, result, ctx, pmp, dev, clint),
        0b01000_11 => handle_store(state, f, instr, result, ctx, pmp, dev, clint),

        // Phase G: FP loads / stores (FLW/FLD/FSW/FSD)
        0b00001_11 => handle_fp_load(state, f, instr, result, ctx, pmp, dev, clint),
        0b01001_11 => handle_fp_store(state, f, instr, result, ctx, pmp, dev, clint),

        // Phase C: System
        0b11100_11 => handle_system(state, f, instr, result, ctx, hart_id, clint, pmp),

        // Phase D: AMO
        0b01011_11 => handle_amo(state, f, instr, result, ctx, pmp, dev),

        // Phase G: F/D floating point (compute — OP-FP + FMA)
        0b10100_11 => handle_fp_op(state, f, instr, result),
        0b10000_11 | 0b10001_11 | 0b10010_11 | 0b10011_11 => {
            handle_fp_fma(state, f, instr, result)
        }

        _ => {
            // Unrecognised opcode — deliver IllInstr inline rather than
            // exiting to Python.  This ensures the consecutive-trap loop
            // detector can halt the hart for dead code (e.g. all-zeros)
            // instead of entering an infinite exit/restart cycle.
            deliver_illegal_instruction(state, instr as u64, result);
            0
        }
    }
}

/// Translate PC through MMU for instruction fetch.
/// Returns ``Some(pa)`` on success, ``None`` if a trap was delivered.
#[inline]
fn translate_fetch_pc(
    state: &mut HartState,
    ctx: &WalkCtx,
    pc: u64,
    result: &mut BatchResult,
    hart_id: u8,
) -> Option<u64> {
    // Bare mode or M-mode (RISC-V spec §4.1.12: M-mode always uses
    // physical addresses regardless of satp.MODE, unless MPRV=1 with
    // MPP=S/U).  The ``translate_va`` call below also handles this,
    // but short-circuiting here avoids an unnecessary TLB lookup.
    if state.mmu_mode == 0 || state.mode == riscv_mode::M {
        return Some(pc); // Bare mode: VA == PA
    }
    match translate_va(state, ctx, pc, false, true) {
        Ok(t) => Some(t.pa),
        Err(e) => {
            let cause = match e {
                TranslateFault::PageFault(c) => c,
                TranslateFault::AccessFault => exc_code::INSTR_ACCESS_FAULT,
            };
            deliver_trap(state, mcause_val(cause, false), pc, result);
            result.exit_reason = exit_reason::TRAP;
            result.exit_hart_id = hart_id;
            result.exit_pc = pc;
            result.exit_instr = 0;
            None
        }
    }
}

/// Fetch an instruction word from physical RAM.
/// Returns ``Some(instr)`` on success, ``None`` if the address is out of range.
#[inline]
fn fetch_instr_safe(
    state: &mut HartState,
    mem: &MemCtx,
    pa: u64,
    pc_before: u64,
    result: &mut BatchResult,
    hart_id: u8,
) -> Option<u32> {
    match fetch_instr(mem.ram, mem.ram_size, mem.ram_base, pa) {
        Some(w) => Some(w),
        None => {
            deliver_trap(
                state,
                mcause_val(exc_code::INSTR_ACCESS_FAULT, false),
                pc_before,
                result,
            );
            result.exit_reason = exit_reason::TRAP;
            result.exit_hart_id = hart_id;
            result.exit_pc = pc_before;
            result.exit_instr = 0;
            None
        }
    }
}

const TRAP_LOOP_THRESHOLD: u8 = 3;

/// Sync MTIP / STIP in ``state.mip`` from the current CLINT mtime value
/// and the per‑hart ``stimecmp`` CSR (Sstc extension).
///
/// MTIP (bit 7) is gated on CLINT ``mtimecmp``.
/// STIP (bit 5) is gated on **either** CLINT ``mtimecmp`` **or** the Sstc
/// ``stimecmp`` CSR, whichever expires first.
///
/// Called before WFI checks and interrupt delivery so that timer interrupts
/// are visible within the same batch when mtime has advanced past the deadline.
#[inline]
fn sync_mtip(state: &mut HartState, clint: &ClintCtx) {
    let hart_id = state.mhartid as usize;
    let cur_mtime = unsafe { *clint.mtime };
    let cmp = unsafe { *clint.mtimecmp.add(hart_id) };
    let sstc_cmp = state.stimecmp;

    // MTIP: only from CLINT mtimecmp
    if cmp > 0 && cur_mtime >= cmp {
        state.mip |= 1 << 7; // MTIP
    } else {
        state.mip &= !(1 << 7);
    }

    // STIP: from CLINT mtimecmp *or* Sstc stimecmp
    let st_pending = (cmp > 0 && cur_mtime >= cmp) || (sstc_cmp > 0 && cur_mtime >= sstc_cmp);
    if st_pending {
        state.mip |= 1 << 5; // STIP
    } else {
        state.mip &= !(1 << 5);
    }
}

#[inline]
fn check_bp_hit(pc: u64, breakpoints: &[u64], fetch_pa: Option<u64>) -> bool {
    if breakpoints.is_empty() {
        return false;
    }
    // Direct VA match — covers breakpoints set on virtual addresses.
    if breakpoints.iter().any(|&bp| bp == pc) {
        return true;
    }
    // PA match — covers breakpoints set on physical addresses when
    // the hart is running with MMU enabled (pc is VA, bp value is PA).
    if let Some(pa) = fetch_pa {
        if pa != pc {
            return breakpoints.iter().any(|&bp| bp == pa);
        }
    }
    false
}

fn run_hart_slice(
    state: &mut HartState,
    hart_id: u8,
    mem: &MemCtx,
    max_instrs: u64,
    result: &mut BatchResult,
    pmp: &PmpCtx,
    dev: &DevCtx,
    clint: &ClintCtx,
    breakpoints: &[u64],
) -> u64 {
    let mut count: u64 = 0;
    let ctx = WalkCtx {
        ram: mem.ram,
        ram_size: mem.ram_size,
        ram_base: mem.ram_base,
        shadow_base: mem.shadow_base,
        shadow_size: mem.shadow_size,
        tlb_gen: std::ptr::null(),
        itlb_hand: Cell::new(0),
        dtlb_hand: Cell::new(0),
        lr_reserved: std::ptr::null_mut(),
        num_harts: 1,
    };

    // The round-robin loop above already splits the batch budget fairly
    // among active harts (see ``num_active``).  No per-hart throttling
    // needed here — large budgets + IPI yield ensure both fairness and
    // critical-section atomicity.

    while count < max_instrs {
        // ---- Cross-hart IPI yield ----
        // If a previous instruction wrote MSIP=1 to a *different* hart,
        // yield this slice early so the target hart can respond within
        // the same batch round-robin round.
        if clint.yield_for_ipi.get() {
            clint.yield_for_ipi.set(false);
            return count;
        }

        // ---- Halted check ----
        if state.halted != 0 {
            return count;
        }

        // ---- WFI waiting ----
        // WFI 唤醒条件分两路:
        //
        // 1. CLINT MSIP (软件中断): 直接读取 CLINT MSIP 寄存器,
        //    绕过 mie.MSIE.  真实硬件上 CLINT 将 MSIP 位断言为
        //    物理中断线, CPU 从 WFI 苏醒 *不依赖* mie.MSIE.
        //    mie.MSIE 仅控制该中断是否被 *投递* (下面的
        //    check_and_deliver_interrupt 决定), 不影响唤醒.
        //
        // 2. 定时器中断 (MTIP/STIP): 仍使用 mip & mie, 因 mtime
        //    通过 sync_mtip 同步到 mip, 且定时器使能位 (MTIE/STIE)
        //    直接反映在 mie 中.
        //
        // 解释为何必须绕过 mie.MSIE: 多核 TLB shootdown 场景中,
        // 发送核 (Hart 1) 通过 CLINT MSIP 向接收核 (Hart 0) 发送
        // IPI 后进入 tlb_sync 自旋等待.  若接收核的 mie.MSIE 因
        // 固件代码路径被意外清零, 仅依赖 mip & mie 将永远无法唤醒
        // -> 死锁.  唤醒后若 MSIE=0, 下面会临时置位以确保中断投递
        // (M 模式 handler 处理 TLB 请求, 清除 MSIP, MRET 回 S 模式).
        //
        // 首先将 MTIP/STIP 同步到 mip, 确保在本 hart 执行期间
        // mtime 的推进 (由其他 hart 或本 hart 指令递增) 能立即
        // 反映到 mip, 使得定时器中断可以在同一 batch 内唤醒 WFI.
        if state.waiting != 0 {
            sync_mtip(state, clint);

            // Check CLINT MSIP directly — hardware interrupt line,
            // not gated by mie.MSIE.
            let msip_active: bool = {
                let hid = hart_id as usize;
                hid < clint.num_harts as usize
                    && unsafe { *clint.msip.add(hid) } != 0
            };

            let pending = state.mip & state.mie;
            if pending != 0 || msip_active {
                state.waiting = 0;
                // 置位 wfi_woken 以告知后续的 WFI handler:
                // 此 hart 刚从 WFI 被中断唤醒, 若 while 循环
                // 分支回到 WFI, 应将该 WFI 视为 NOP 推进 PC.
                state.wfi_woken = 1;

                // MSIP 活跃但 mie.MSIE=0: 临时置位 MSIE, 确保
                // 下面的 check_and_deliver_interrupt 可投递该中断
                // 到 M 模式 trap handler (处理 TLB 请求 / 清除 MSIP).
                // 真实硬件上中断线断言后 CPU 总是苏醒; 这里额外
                // 保证中断可投递, 避免苏醒后因 MSIE=0 而跳过 IPI
                // 处理, 导致发送核 tlb_sync 永远自旋.
                if msip_active && (state.mie & (1 << 3)) == 0 {
                    state.mie |= 1 << 3;
                }
            } else {
                result.exit_reason = exit_reason::WFI_WAIT;
                result.exit_hart_id = hart_id;
                result.exit_pc = state.pc;
                result.exit_instr = 0;
                return count;
            }
        }

        // ---- Interrupt check at instruction boundary ----
        // Sync MTIP/STIP before checking, so timer interrupts that
        // matured during this batch are visible.
        sync_mtip(state, clint);
        if check_and_deliver_interrupt(state, result, clint) {
            // Interrupt delivered inline — PC already at the trap handler
            // (mtvec / stvec).  Continue executing from there within the
            // same slice so the full acknowledge cycle (trap handler ->
            // clear MSIP -> MRET -> resume) completes without a batch exit.
            continue;
        }

        // ---- Instruction fetch with MMU translation ----
        let pc_before = state.pc;
        let fetch_pa = match translate_fetch_pc(state, &ctx, pc_before, result, hart_id) {
            Some(pa) => pa,
            None => return count,
        };

        // ---- PMP execute check on the fetch address ----
        // RISC-V spec: instruction fetch requires X permission on the
        // physical address.  Python's ``check_instruction_fetch`` does
        // this; the Rust batch engine must match that behaviour.
        if !pmp_ok(state, fetch_pa, 4, false, true, pmp) {
            deliver_trap(
                state,
                mcause_val(exc_code::INSTR_ACCESS_FAULT, false),
                pc_before,
                result,
            );
            result.exit_reason = exit_reason::TRAP;
            result.exit_hart_id = hart_id;
            result.exit_pc = pc_before;
            result.exit_instr = 0;
            return count;
        }

        let instr_word = match fetch_instr_safe(
            state, mem, fetch_pa, pc_before, result, hart_id,
        ) {
            Some(w) => w,
            None => return count,
        };

        // ---- Breakpoint (pre-execution) ----
        // Check BEFORE executing the instruction so that breakpoints set
        // on JALR / JAL / branch instructions fire correctly.  Without this,
        // only the post-execution PC was checked, which for a JALR at addr X
        // is the jump target (not X).
        if check_bp_hit(pc_before, breakpoints, Some(fetch_pa)) {
            result.exit_reason = exit_reason::BREAKPOINT;
            result.exit_hart_id = hart_id;
            result.exit_pc = pc_before;
            result.exit_instr = instr_word;
            return count;
        }

        // ---- Decode ----
        let f = decode_fields(instr_word);

        // ---- Compressed instruction ----
        if f.is_compressed != 0 {
            let half = (instr_word & 0xFFFF) as u16;
            let advance = handle_compressed(state, half, instr_word, result, &ctx, pmp, dev, clint);
            if advance == EXIT_SENTINEL {
                // Preserve handler-set exit reasons (MMIO, etc.); only
                // default to ECALL when the handler didn't set one.
                if result.exit_reason == exit_reason::NORMAL {
                    result.exit_reason = exit_reason::ECALL;
                }
                result.exit_hart_id = hart_id;
                result.exit_pc = pc_before;
                result.exit_instr = instr_word;
                return count;
            }
            if advance == 0 && state.pc == pc_before {
                // Trap or jump; PC already updated by handler
                count += 1;
                check_consecutive_trap(state, result, hart_id, &mut count);
                if check_bp_hit(state.pc, breakpoints, None) {
                    result.exit_reason = exit_reason::BREAKPOINT;
                    result.exit_hart_id = hart_id;
                    result.exit_pc = state.pc;
                    return count;
                }
                continue;
            }
            if advance != 0 && state.pc == pc_before {
                state.pc = state.pc.wrapping_add(advance);
                state.consecutive_traps = 0;
            }
            count += 1;
            if check_bp_hit(state.pc, breakpoints, None) {
                result.exit_reason = exit_reason::BREAKPOINT;
                result.exit_hart_id = hart_id;
                result.exit_pc = state.pc;
                return count;
            }
            continue;
        }

        // ---- Dispatch ----
        let advance = dispatch(
            state, &f, instr_word, result, &ctx, hart_id, pmp, dev, clint,
        );

        if advance == EXIT_SENTINEL {
            // Preserve handler-set exit reasons (EBREAK, MMIO, etc.); only
            // default to ECALL when the handler didn't set one.
            if result.exit_reason == exit_reason::NORMAL {
                result.exit_reason = exit_reason::ECALL;
            }
            result.exit_hart_id = hart_id;
            result.exit_pc = pc_before;
            result.exit_instr = instr_word;
            return count;
        }

        // ---- PC update ----
        if advance != 0 && state.pc == pc_before {
            state.pc = state.pc.wrapping_add(advance);
            state.consecutive_traps = 0;
        }

        count += 1;

        if check_bp_hit(state.pc, breakpoints, None) {
            result.exit_reason = exit_reason::BREAKPOINT;
            result.exit_hart_id = hart_id;
            result.exit_pc = state.pc;
            return count;
        }

        // ---- Consecutive trap check ----
        if state.consecutive_traps >= TRAP_LOOP_THRESHOLD {
            state.halted = 1;
            result.exit_reason = exit_reason::TRAP;
            result.exit_hart_id = hart_id;
            result.exit_pc = state.pc;
            result.exit_instr = 0;
            return count;
        }
    }

    count
}

#[inline]
fn check_consecutive_trap(
    state: &mut HartState,
    result: &mut BatchResult,
    hart_id: u8,
    _count: &mut u64,
) {
    // Count handled above in run_hart_slice main loop
    if state.consecutive_traps >= TRAP_LOOP_THRESHOLD {
        state.halted = 1;
        result.exit_reason = exit_reason::TRAP;
        result.exit_hart_id = hart_id;
        result.exit_pc = state.pc;
        result.exit_instr = 0;
    }
}

// ============================================================
//  Public FFI entry point
// ============================================================

/// **DEPRECATED**: 顺序 round-robin 批量执行引擎。
///
/// 已被 [`crate::concurrent::run_parallel`] (thread-per-hart 并发模型) 取代。
/// 串行 batch 在指令片边界截断跨核临界区 (如 TLB shootdown / IPI 协议),
/// 导致 SMP 死锁; 并发模型无此问题。保留仅为向后兼容与差分对照
/// (`PYREMU_NATIVE_SERIAL=1`), 不再演进; 相关单元测试已 `#[ignore]`。
#[no_mangle]
#[deprecated(note = "use run_parallel (concurrent model); run_batch serial path is frozen")]
pub unsafe extern "C" fn run_batch(
    states: *mut HartState,
    num_harts: u32,
    max_instrs: u64,
    result: *mut BatchResult,
    mem: *const MemCtx,
    pmp: *const FfiPmpCtx,
    clint: *const FfiClintCtx,
    dev: *const FfiDevCtx,
    virtio: *const FfiVirtIoCtx,
    bp_addrs: *const u64,
    bp_count: u32,
) {
    // Zero-initialise the result
    unsafe {
        (*result).total_instrs = 0;
        (*result).exit_reason = exit_reason::NORMAL;
        (*result).exit_hart_id = 0;
        (*result).exit_pc = 0;
        (*result).exit_instr = 0;
        (*result).trap_cause = 0;
        (*result).trap_tval = 0;
        (*result).trap_is_interrupt = 0;
        (*result).trap_delegated = 0;
    }

    // Read FFI structs once (safe — Python holds them alive for the call).
    let mem = unsafe { &*mem };
    let pmp_raw = unsafe { &*pmp };
    let clint_raw = unsafe { &*clint };
    let dev_raw = unsafe { &*dev };
    let virtio_raw: *mut FfiVirtIoCtx = if virtio.is_null() { std::ptr::null_mut() }
                     else { (unsafe { &*virtio }) as *const _ as *mut _ };

    // Bundle internal context structs.
    // 注意: PMP cfg/addr 缓冲现为 per-hart 布局 (num_harts * 64 项),
    // 每个 hart 在下方 round-robin 循环中按 hid*64 偏移取自己的切片。
    let dev_ctx = DevCtx {
        bases: dev_raw.bases,
        ends: dev_raw.ends,
        num: dev_raw.num,
        virtio_base: if virtio_raw.is_null() { 0 } else { unsafe { (*virtio_raw).base } },
        virtio_raw,
    };
    let clint_ctx = ClintCtx {
        base: clint_raw.base,
        mtime: clint_raw.mtime,
        mtimecmp: clint_raw.mtimecmp,
        msip: clint_raw.msip,
        states,
        num_harts,
        yield_for_ipi: Cell::new(false),
        ipi_sender_hart: Cell::new(0),
        ipi_sender_rounds: Cell::new(0),
    };

    // Breakpoint addresses (empty slice when bp_count == 0).
    let breakpoints: &[u64] = if bp_count > 0 && !bp_addrs.is_null() {
        unsafe { std::slice::from_raw_parts(bp_addrs, bp_count as usize) }
    } else {
        &[]
    };

    // Round-robin scheduling: give each hart a fixed slice per round
    // so that all harts make progress within a single batch.  This is
    // critical for correctness of multi-hart firmware that relies on
    // tight interleaving (e.g. lottery locks during cold boot).
    // Cooperative round-robin: each hart runs until it hits a natural
    // boundary (WFI, IPI yield, trap/ECALL/MMIO).  No fixed slice size —
    // critical sections like ticket-lock acquire->enqueue->release always
    // complete atomically within one turn.  See CHANGELOG.md §2026-07-11.
    let mut grand_total: u64 = 0;
    let mut any_hart_ran: bool;
    let _last_runner: i32 = -1; // track which hart ran most recently

    // Pre-build mip values for all harts.  Since mtime is now a mutable
    // pointer, we read it once for the initial build; per-instruction
    // updates are handled by ``sync_mtip`` in ``run_hart_slice``.
    {
        let cur_mtime = unsafe { *clint_raw.mtime };
        for hid in 0..num_harts {
            let state = unsafe { &mut *states.add(hid as usize) };
            let hart_mip = build_mip(
                state, cur_mtime, clint_raw.mtimecmp, clint_raw.msip,
            );
            state.mip = hart_mip;
        }
    }

    loop {
        any_hart_ran = false;

        // Fairness: if both harts are active (not waiting), split budget.
        // Otherwise the single active hart gets the full remaining budget.
        // This prevents a busy hart from starving its peer while still
        // allowing a single active hart to run critical sections atomically.
        let num_active = (0..num_harts).filter(|&hid| {
            let s = unsafe { &*states.add(hid as usize) };
            s.halted == 0 && s.waiting == 0
        }).count();

        for hid in 0..num_harts {
            if grand_total >= max_instrs {
                return;
            }

            let state = unsafe { &mut *states.add(hid as usize) };

            if state.halted != 0 {
                continue;
            }

            // Each active hart gets an equal share of the remaining budget.
            // When only one hart is active (the other is in WFI), it gets
            // the full remaining budget -> critical sections never split.
            let remaining = max_instrs.saturating_sub(grand_total);
            let share = if num_active > 0 {
                remaining / (num_active as u64).max(1)
            } else {
                remaining
            };
            // Minimum 512 instrs per turn so WFI wakeups always get a
            // meaningful slice; below that we exit the batch.
            let slice_budget = share.max(512);
            if remaining < 512 {
                return;
            }

            // Reset cross-hart IPI yield flag before each hart's slice
            // so that a stale flag from a previous hart doesn't cause
            // this hart to yield prematurely.
            clint_ctx.yield_for_ipi.set(false);

            // Per-hart PMP 切片 (cfg/addr 缓冲按 hid*64 偏移)。
            let (pcfg, paddr) = if pmp_raw.num > 0 {
                (
                    unsafe { pmp_raw.cfg.add(hid as usize * 64) },
                    unsafe { pmp_raw.addr.add(hid as usize * 64) },
                )
            } else {
                (pmp_raw.cfg, pmp_raw.addr)
            };
            let pmp_ctx_h = PmpCtx { cfg: pcfg, addr: paddr, num: pmp_raw.num };

            let executed = run_hart_slice(
                state,
                hid as u8,
                mem,
                slice_budget,
                unsafe { &mut *result },
                &pmp_ctx_h,
                &dev_ctx,
                &clint_ctx,
                breakpoints,
            );
            grand_total += executed;

            // Decrement short-slice round counter for the IPI sender
            // after each call, regardless of how the slice ended
            // (normal, yield_for_ipi, exit).  This ensures the counter
            // tracks round-robin rounds, not individual instructions.
            if clint_ctx.ipi_sender_rounds.get() > 0
                && hid as u8 == clint_ctx.ipi_sender_hart.get()
            {
                clint_ctx.ipi_sender_rounds.set(
                    clint_ctx.ipi_sender_rounds.get() - 1,
                );
            }

            // Per-hart instruction counter — increment even when
            // WFI_WAIT (executed == 0) so the hart's total is accurate.
            state.total_instrs = state.total_instrs.wrapping_add(executed);

            // Advance mtime by instructions this hart actually executed,
            // so the next hart's WFI check sees the timer progress.
            if executed > 0 {
                unsafe {
                    *clint_raw.mtime = (*clint_raw.mtime).wrapping_add(executed);
                }
                any_hart_ran = true;
            }

            unsafe {
                (*result).total_instrs = grand_total;
            }

            // Exit to Python on trap / sys-exit / ebreak (not on WFI_WAIT
            // — WFI_WAIT means only *this* hart is idle; keep going for
            // other harts).
            let reason = unsafe { (*result).exit_reason };
            if reason != exit_reason::NORMAL && reason != exit_reason::WFI_WAIT {
                return;
            }
            // Reset exit_reason for the next hart's slice (WFI_WAIT is
            // per-hart, not per-batch).
            unsafe {
                (*result).exit_reason = exit_reason::NORMAL;
            }
        }

        // If no hart made progress (all halted or waiting indefinitely),
        // try to fast-forward mtime to the next timer deadline first.
        // If a timer is set, advance mtime, wake up the corresponding
        // hart(s) inline, and continue the round-robin loop without
        // ever exiting to Python.  This eliminates the dominant batch-
        // exit cause for single-core Linux idle loops and significantly
        // reduces FFI overhead.
        if !any_hart_ran {
            if !wfi_fast_forward_mtime(states, num_harts, &clint_ctx) {
                return;
            }
            // At least one hart was woken — fall through to next round.
        }
    }
}

/// Fast-forward ``mtime`` to the nearest timer deadline when all harts
/// are idle in WFI.  Wakes up any hart whose timer has expired after
/// the fast-forward.
///
/// Returns ``true`` if at least one hart was woken up (caller should
/// continue the round-robin loop).  Returns ``false`` if no timer is
/// configured — all harts are truly idle and the caller should exit the
/// batch so Python can ``time.sleep()`` until the next event.
fn wfi_fast_forward_mtime(
    states: *mut HartState,
    num_harts: u32,
    clint: &ClintCtx,
) -> bool {
    let cur_mtime = unsafe { *clint.mtime };
    let mut next_wake: u64 = u64::MAX;

    // Find the earliest CLINT mtimecmp across all non-halted harts.
    for hid in 0..(num_harts as usize) {
        let state = unsafe { &*states.add(hid) };
        if state.halted != 0 {
            continue;
        }
        // CLINT mtimecmp
        let cmp = unsafe { *clint.mtimecmp.add(hid) };
        if cmp > 0 && cmp > cur_mtime && cmp < next_wake {
            next_wake = cmp;
        }
        // Sstc stimecmp
        if state.stimecmp > 0
            && state.stimecmp > cur_mtime
            && state.stimecmp < next_wake
        {
            next_wake = state.stimecmp;
        }
    }

    if next_wake == u64::MAX {
        return false; // No timer armed — truly idle
    }

    // Fast-forward mtime to the earliest deadline.
    unsafe { *clint.mtime = next_wake; }

    // Re-sync MTIP/STIP for all harts and wake up any whose
    // timer has now expired.
    let mut any_woken = false;
    for hid in 0..(num_harts as usize) {
        let state = unsafe { &mut *states.add(hid) };
        let cmp = unsafe { *clint.mtimecmp.add(hid) };

        // MTIP (bit 7): CLINT timer
        if cmp > 0 && next_wake >= cmp {
            state.mip |= 1 << 7;
        } else {
            state.mip &= !(1 << 7);
        }
        // STIP (bit 5): Sstc timer
        if state.stimecmp > 0 && next_wake >= state.stimecmp {
            state.mip |= 1 << 5;
        } else {
            state.mip &= !(1 << 5);
        }

        // Wake up WFI-waiting harts whose timer just fired.
        if state.waiting != 0 && (state.mip & state.mie) != 0 {
            state.waiting = 0;
            state.wfi_woken = 1;
            any_woken = true;
        }
    }

    any_woken
}

/// Build the mip CSR value for a hart based on CLINT state.
fn build_mip(state: &HartState, mtime: u64, mtimecmp: *mut u64, msip: *mut u8) -> u64 {
    let hart_id = state.mhartid as usize;
    let mut mip = state.mip;

    // MTIP: timer interrupt.  cmp == 0 means timer is disabled, matching
    // the Python CLINT behaviour (``mtimecmp > 0`` guard).
    let cmp = unsafe { *mtimecmp.add(hart_id) };
    if cmp > 0 && mtime >= cmp {
        mip |= 1 << 7; // MTIP
    } else {
        mip &= !(1 << 7);
    }

    // MSIP: software interrupt
    let sip_val = unsafe { *msip.add(hart_id) };
    if sip_val != 0 {
        mip |= 1 << 3; // MSIP
    } else {
        mip &= !(1 << 3);
    }

    // STIP: S-mode timer — gated on CLINT mtimecmp *or* Sstc stimecmp,
    // whichever expires first.  The kernel may use either mechanism (or both).
    // ``stimecmp`` is the Sstc CSR (0x14D); it is synced from Python's
    // ``csrs["stimecmp"].val`` during ``marshal_hart``.
    let sstc_cmp = state.stimecmp;
    let st_pending = (cmp > 0 && mtime >= cmp) || (sstc_cmp > 0 && mtime >= sstc_cmp);
    if st_pending {
        mip |= 1 << 5; // STIP
    } else {
        mip &= !(1 << 5);
    }

    mip
}

#[cfg(test)]
mod tests {
    // run_batch 已废弃 (见函数上方 #[deprecated]); 测试仍编译以作差分对照,
    // 但均标记 #[ignore]。允许调用已废弃项以避免 warning。
    #![allow(deprecated)]
    use super::*;
    use crate::state::riscv_mode;
    use std::mem;

    fn make_state(pc: u64, hart_id: u64) -> HartState {
        let mut s: HartState = unsafe { std::mem::zeroed() };
        s.mode = riscv_mode::M;
        s.mtvec = 0x8000_0100;
        s.pc = pc;
        s.mhartid = hart_id;
        s
    }

    fn write_u32_le(buf: &mut [u8], offset: usize, val: u32) {
        buf[offset] = val as u8;
        buf[offset + 1] = (val >> 8) as u8;
        buf[offset + 2] = (val >> 16) as u8;
        buf[offset + 3] = (val >> 24) as u8;
    }

    /// Build default zero-length CLINT / PMP / device arrays and call ``run_batch``.
    unsafe fn run_batch_defaults(
        states: *mut HartState,
        num: u32,
        ram: *mut u8,
        ram_sz: u64,
        ram_base: u64,
        shadow_base: u64,
        shadow_size: u64,
        max_instrs: u64,
        result: *mut BatchResult,
    ) {
        let mem = MemCtx {
            ram,
            ram_size: ram_sz,
            ram_base,
            shadow_base,
            shadow_size,
        };
        let dev_bases: [u64; 0] = [];
        let dev_ends: [u64; 0] = [];
        let dev = FfiDevCtx {
            bases: dev_bases.as_ptr(),
            ends: dev_ends.as_ptr(),
            num: 0,
        };
        let mut pmp_cfg: [u8; 0] = [];
        let mut pmp_addr: [u64; 0] = [];
        let pmp = FfiPmpCtx {
            cfg: pmp_cfg.as_mut_ptr(),
            addr: pmp_addr.as_mut_ptr(),
            num: 0,
            pmpsplit: 0,
        };
        let mut mtimecmp: [u64; 1] = [u64::MAX];
        let mut msip: [u8; 1] = [0];
        let mut mtime: u64 = 0;
        let clint = FfiClintCtx {
            mtime: &mut mtime as *mut u64,
            mtimecmp: mtimecmp.as_mut_ptr(),
            msip: msip.as_mut_ptr(),
            base: 0,
        };
        run_batch(
            states, num, max_instrs, result,
            &mem as *const MemCtx,
            &pmp as *const FfiPmpCtx,
            &clint as *const FfiClintCtx,
            &dev as *const FfiDevCtx,
            std::ptr::null_mut() as *mut FfiVirtIoCtx, // virtio
            std::ptr::null(),
            0,
        );
    }

    #[test]
    #[ignore = "run_batch deprecated — superseded by run_parallel concurrent model"]
    fn run_batch_addi_sequence() {
        let mut ram = vec![0u8; 64];
        write_u32_le(&mut ram, 0, 0x00A0_0293);
        write_u32_le(&mut ram, 4, 0x0052_8293);
        write_u32_le(&mut ram, 8, 0xFFD2_8293);
        write_u32_le(&mut ram, 12, 0x0FF0_000F);

        let mut state = make_state(0, 0);
        let mut result: BatchResult = unsafe { mem::zeroed() };
        unsafe {
            run_batch_defaults(
                &mut state as *mut HartState,
                1,
                ram.as_mut_ptr(),
                ram.len() as u64,
                0,
                0,
                0,
                3,
                &mut result as *mut BatchResult,
            );
        }

        assert_eq!(result.exit_reason, exit_reason::NORMAL);
        assert_eq!(result.total_instrs, 3);
        assert_eq!(state.gprs[5], 12);
        assert_eq!(state.pc, 12);
    }

    #[test]
    #[ignore = "run_batch deprecated — superseded by run_parallel concurrent model"]
    fn run_batch_branch_loop() {
        let mut ram = vec![0u8; 64];
        write_u32_le(&mut ram, 0, 0x0030_0293);
        write_u32_le(&mut ram, 4, 0xFFF2_8293);
        write_u32_le(&mut ram, 8, 0xFE02_9EE3);
        write_u32_le(&mut ram, 12, 0x0010_0313);

        let mut state = make_state(0, 0);
        let mut result: BatchResult = unsafe { mem::zeroed() };
        unsafe {
            run_batch_defaults(
                &mut state as *mut HartState,
                1,
                ram.as_mut_ptr(),
                ram.len() as u64,
                0,
                0,
                0,
                8,
                &mut result as *mut BatchResult,
            );
        }

        assert_eq!(result.exit_reason, exit_reason::NORMAL);
        assert_eq!(result.total_instrs, 8);
        assert_eq!(state.gprs[5], 0);
        assert_eq!(state.gprs[6], 1);
        assert_eq!(state.pc, 16);
    }

    #[test]
    #[ignore = "run_batch deprecated — superseded by run_parallel concurrent model"]
    fn run_batch_max_instrs_limit() {
        let mut ram = vec![0u8; 128];
        for i in 0..10 {
            write_u32_le(&mut ram, (i * 4) as usize, 0x0012_8293);
        }

        let mut state = make_state(0, 0);
        let mut result: BatchResult = unsafe { mem::zeroed() };
        unsafe {
            run_batch_defaults(
                &mut state as *mut HartState,
                1,
                ram.as_mut_ptr(),
                ram.len() as u64,
                0,
                0,
                0,
                5,
                &mut result as *mut BatchResult,
            );
        }

        assert_eq!(result.exit_reason, exit_reason::NORMAL);
        assert_eq!(result.total_instrs, 5);
        assert_eq!(state.gprs[5], 5);
    }

    #[test]
    #[ignore = "run_batch deprecated — superseded by run_parallel concurrent model"]
    fn run_batch_load_store() {
        let mut ram = vec![0u8; 256];
        ram[0x40] = 0x42;

        write_u32_le(&mut ram, 0, 0x00050283); // lb x5, 0(x10)
        write_u32_le(&mut ram, 4, 0x00550223); // sb x5, 4(x10)
        write_u32_le(&mut ram, 8, 0x0FF0_000F); // FENCE nop

        let mut state = make_state(0, 0);
        state.gprs[10] = 0x40;

        let mut result: BatchResult = unsafe { mem::zeroed() };
        unsafe {
            run_batch_defaults(
                &mut state as *mut HartState,
                1,
                ram.as_mut_ptr(),
                ram.len() as u64,
                0,
                0,
                0,
                3,
                &mut result as *mut BatchResult,
            );
        }

        assert_eq!(result.exit_reason, exit_reason::NORMAL);
        assert_eq!(state.gprs[5], 0x42);
        assert_eq!(ram[0x44], 0x42);
    }

    #[test]
    #[ignore = "run_batch deprecated — superseded by run_parallel concurrent model"]
    fn fetch_out_of_bounds_traps() {
        let ram = vec![0u8; 64];
        let mut state = make_state(ram.len() as u64 + 100, 0);
        state.mtvec = 0x8000_0100;
        let mut result: BatchResult = unsafe { mem::zeroed() };
        unsafe {
            run_batch_defaults(
                &mut state as *mut HartState,
                1,
                ram.as_ptr() as *mut u8,
                ram.len() as u64,
                0,
                0,
                0,
                10,
                &mut result as *mut BatchResult,
            );
        }

        assert_eq!(result.exit_reason, exit_reason::TRAP);
    }

    #[test]
    #[ignore = "run_batch deprecated — superseded by run_parallel concurrent model"]
    fn run_batch_ecall_inline_no_exit() {
        // ECALL is now handled inline: the trap is delivered within the batch,
        // PC jumps to mtvec, and execution continues from there without exiting
        // to Python.  This test verifies the inline behaviour.
        let mut ram = vec![0u8; 256];
        // ECALL at offset 0
        write_u32_le(&mut ram, 0, 0x0000_0073);

        let mut state = make_state(0, 0);
        state.mode = riscv_mode::S; // ECALL from S-mode
        state.mtvec = 0x80;

        let mut result: BatchResult = unsafe { mem::zeroed() };
        unsafe {
            run_batch_defaults(
                &mut state as *mut HartState,
                1,
                ram.as_mut_ptr(),
                ram.len() as u64,
                0,
                0,
                0,
                1, // only 1 instruction — ECALL itself
                &mut result as *mut BatchResult,
            );
        }

        // ECALL delivers trap inline: PC -> mtvec, mode -> M.
        // No batch exit with ECALL reason.
        assert_ne!(result.exit_reason, exit_reason::ECALL,
            "ECALL must NOT cause an ECALL exit (trap is delivered inline)");
        assert_eq!(result.total_instrs, 1,
            "ECALL instruction should be counted");
        assert_eq!(state.mode, riscv_mode::M,
            "mode should be M after ECALL trap delivery");
        assert_eq!(state.pc, 0x80,
            "PC should jump to mtvec after ECALL trap");
        // Verify the trap context was saved correctly
        assert_eq!(state.mepc, 0, "mepc should be the ECALL address");
        assert_eq!(state.mcause & 0x7FFF_FFFF_FFFF_FFFF, exc_code::ECALL_SMODE,
            "mcause should be ECALL from S-mode");
    }

    #[test]
    #[ignore = "run_batch deprecated — superseded by run_parallel concurrent model"]
    fn run_batch_mret() {
        let mut ram = vec![0u8; 64];
        write_u32_le(&mut ram, 0, 0x3020_0073); // MRET

        let mut state = make_state(0, 0);
        state.mode = riscv_mode::M; // M-mode so PMP doesn't block the fetch
        state.mstatus = (1 << 7) | (1 << 11) | (1 << 3); // MPIE | MPP=S | MIE
        state.mepc = 0x8000_1000;

        let mut result: BatchResult = unsafe { mem::zeroed() };
        unsafe {
            run_batch_defaults(
                &mut state as *mut HartState,
                1,
                ram.as_mut_ptr(),
                ram.len() as u64,
                0,
                0,
                0,
                1,
                &mut result as *mut BatchResult,
            );
        }

        assert_eq!(result.total_instrs, 1);
        assert_eq!(state.mode, riscv_mode::S);
        assert_eq!(state.pc, 0x8000_1000);
    }

    #[test]
    #[ignore = "run_batch deprecated — superseded by run_parallel concurrent model"]
    fn run_batch_compressed_addi() {
        let mut ram = vec![0u8; 64];
        write_u32_le(&mut ram, 0, 0x0000_0515); // c.addi x10, 5
        write_u32_le(&mut ram, 4, 0x0FF0_000F); // FENCE

        let mut state = make_state(0, 0);
        let mut result: BatchResult = unsafe { mem::zeroed() };
        unsafe {
            run_batch_defaults(
                &mut state as *mut HartState,
                1,
                ram.as_mut_ptr(),
                ram.len() as u64,
                0,
                0,
                0,
                2,
                &mut result as *mut BatchResult,
            );
        }

        assert_eq!(result.total_instrs, 2);
        assert_eq!(state.gprs[10], 5);
    }

    #[test]
    #[ignore = "run_batch deprecated — superseded by run_parallel concurrent model"]
    fn wfi_nop_when_wfi_woken_set() {
        // Verify that when wfi_woken=1, WFI acts as NOP (clears flag,
        // advances PC, does NOT enter waiting).  This is critical for
        // while (…) wfi() polling loops: after an interrupt wakes the
        // hart, the loop branches back to WFI, and the first WFI must
        // be a NOP to allow re-checking the exit condition.
        let mut ram = vec![0u8; 64];
        // WFI instruction at offset 0
        write_u32_le(&mut ram, 0, 0x1050_0073);

        let mut state = make_state(0, 0);
        state.mode = riscv_mode::M;
        state.wfi_woken = 1;
        // Ensure no interrupt is pending (otherwise WFI NOPs for a
        // different reason — the mip&mie check — which would mask
        // a missing wfi_woken check).
        state.mie = 0;
        state.mip = 0;

        let mut result: BatchResult = unsafe { mem::zeroed() };
        unsafe {
            run_batch_defaults(
                &mut state as *mut HartState,
                1,
                ram.as_mut_ptr(),
                ram.len() as u64,
                0,
                0,
                0,
                1,
                &mut result as *mut BatchResult,
            );
        }

        assert_eq!(result.total_instrs, 1);
        assert_eq!(state.wfi_woken, 0, "wfi_woken should be cleared by WFI NOP");
        assert_eq!(
            state.waiting, 0,
            "WFI should NOT enter waiting when wfi_woken=1"
        );
        assert_eq!(state.pc, 4, "PC should advance past WFI");
    }

    #[test]
    #[ignore = "run_batch deprecated — superseded by run_parallel concurrent model"]
    fn wfi_waiting_when_wfi_woken_cleared() {
        // Complement to wfi_nop_when_wfi_woken_set: with wfi_woken=0
        // and no pending interrupt, WFI enters waiting state.
        let mut ram = vec![0u8; 64];
        write_u32_le(&mut ram, 0, 0x1050_0073);

        let mut state = make_state(0, 0);
        state.mode = riscv_mode::M;
        state.wfi_woken = 0;
        state.mie = 0;
        state.mip = 0;

        let mut result: BatchResult = unsafe { mem::zeroed() };
        unsafe {
            run_batch_defaults(
                &mut state as *mut HartState,
                1,
                ram.as_mut_ptr(),
                ram.len() as u64,
                0,
                0,
                0,
                1,
                &mut result as *mut BatchResult,
            );
        }

        assert_eq!(
            result.total_instrs, 1,
            "WFI 指令应被计数 (PC 在进入等待前推进)"
        );
        assert_eq!(state.wfi_woken, 0);
        assert_eq!(
            state.waiting, 1,
            "WFI should enter waiting when wfi_woken=0"
        );
    }

    #[test]
    #[ignore = "run_batch deprecated — superseded by run_parallel concurrent model"]
    fn wfi_woken_persists_through_marshal_roundtrip() {
        // Simulate the Python->Rust marshal cycle: Python sets wfi_woken=1
        // after deliver_trap wakes a hart from WFI.  The next batch must
        // honour this flag so that the branch-back WFI acts as NOP.
        let mut ram = vec![0u8; 64];
        write_u32_le(&mut ram, 0, 0x1050_0073);

        let mut state = make_state(0, 0);
        state.mode = riscv_mode::M;
        // Python-side deliver_trap sets these:
        state.waiting = 0; // just woken from WFI
        state.wfi_woken = 1; // tells WFI to NOP on next encounter
        state.mie = 0;
        state.mip = 0;

        let mut result: BatchResult = unsafe { mem::zeroed() };
        unsafe {
            run_batch_defaults(
                &mut state as *mut HartState,
                1,
                ram.as_mut_ptr(),
                ram.len() as u64,
                0,
                0,
                0,
                2,
                &mut result as *mut BatchResult,
            );
        }

        // First WFI: wfi_woken=1 -> NOP, clears flag, advances PC.
        // After that, there's no more instruction (only 4 bytes loaded),
        // but fetch out-of-bounds triggers InstrAccessFault.
        // Key assertion: wfi_woken MUST be consumed.
        assert_eq!(
            state.wfi_woken, 0,
            "wfi_woken must be consumed by the first WFI after marshal roundtrip"
        );
    }

    #[test]
    #[ignore = "run_batch deprecated — superseded by run_parallel concurrent model"]
    fn wfi_woken_set_on_wake_in_slice() {
        // When a hart is in WFI (waiting=1) and an interrupt becomes
        // pending, run_hart_slice must set wfi_woken=1 alongside
        // clearing waiting.  This ensures the WFI handler sees the
        // flag when the while loop branches back to WFI.
        let mut ram = vec![0u8; 64];
        // Put WFI instruction at offset 0
        write_u32_le(&mut ram, 0, 0x1050_0073);

        let mut state = make_state(0, 0);
        state.mode = riscv_mode::M;
        // MIE must be 1 for M-mode to take interrupts
        state.mstatus = 1 << 3; // MIE=1
                                // Simulate: hart already executed WFI and is now waiting
        state.waiting = 1;
        state.wfi_woken = 0;
        state.pc = 4; // PC was advanced past WFI by the WFI handler
                      // Enable MSIP — must be set via the CLINT msip array, not just
                      // state.mip, because build_mip() at batch start overwrites mip.
        state.mie = 1 << 3;
        state.mip = 0; // will be rebuilt by build_mip
        state.mtvec = 0x8000_0100;

        let mut result: BatchResult = unsafe { mem::zeroed() };
        let mem = MemCtx {
            ram: ram.as_mut_ptr(),
            ram_size: ram.len() as u64,
            ram_base: 0,
            shadow_base: 0,
            shadow_size: 0,
        };
        let dev_bases: [u64; 0] = [];
        let dev_ends: [u64; 0] = [];
        let dev = FfiDevCtx { bases: dev_bases.as_ptr(), ends: dev_ends.as_ptr(), num: 0 };
        let mut pmp_cfg: [u8; 0] = [];
        let mut pmp_addr: [u64; 0] = [];
        let pmp = FfiPmpCtx { cfg: pmp_cfg.as_mut_ptr(), addr: pmp_addr.as_mut_ptr(), num: 0, pmpsplit: 0 };
        let mut mtimecmp: [u64; 1] = [u64::MAX];
        let mut msip: [u8; 1] = [1]; // MSIP set -> build_mip keeps MSIP in mip
        let mut mtime_val: u64 = 0;
        let clint = FfiClintCtx { mtime: &mut mtime_val as *mut u64, mtimecmp: mtimecmp.as_mut_ptr(), msip: msip.as_mut_ptr(), base: 0 };
        unsafe {
            run_batch(
                &mut state as *mut HartState, 1, 1, &mut result as *mut BatchResult,
                &mem as *const MemCtx, &pmp as *const FfiPmpCtx,
                &clint as *const FfiClintCtx, &dev as *const FfiDevCtx,
            std::ptr::null_mut() as *mut FfiVirtIoCtx, // virtio
                std::ptr::null(), 0,
            );
        }

        // The WFI check at the top of run_hart_slice should:
        // 1. Detect the pending MSIP -> clear waiting, SET wfi_woken
        // 2. Deliver the interrupt trap -> PC = mtvec
        assert_eq!(
            state.wfi_woken, 1,
            "wfi_woken must be set when waking from WFI due to interrupt"
        );
        assert_eq!(state.waiting, 0, "waiting must be cleared on wake");
        assert_eq!(state.pc, 0x8000_0100, "PC should jump to mtvec");
        assert_eq!(result.exit_reason, exit_reason::TRAP);
    }

    #[test]
    #[ignore = "run_batch deprecated — superseded by run_parallel concurrent model"]
    fn wfi_wake_on_msip_even_when_mie_zero() {
        // Regression: WFI wake-up must only check mip & mie (source-level),
        // NOT mstatus.MIE (global enable).  On real hardware a hart can
        // resume from WFI when an interrupt is pending at the source level;
        // the interrupt is only *delivered* when both source + global
        // enables are set.  If we require mstatus.MIE for wake-up, a
        // secondary hart parked in WFI with MIE=0 will never respond to
        // MSIP from the cold-boot hart, deadlocking multi-hart boot.
        let mut ram = vec![0u8; 64];
        write_u32_le(&mut ram, 0, 0x1050_0073); // WFI at offset 0

        let mut state = make_state(0, 0);
        state.mode = riscv_mode::M;
        // MIE=0: global M-mode interrupt enable is OFF.
        // This is the exact scenario that the warm-boot WFI loop hits:
        // sbi_hsm_hart_wait sets mie.MSIE=1 (source enable) but never
        // writes mstatus, so mstatus.MIE stays at its reset value of 0.
        state.mstatus = 0; // MIE=0
        state.waiting = 1; // already asleep from a prior WFI
        state.wfi_woken = 0;
        state.pc = 4; // PC advanced past WFI
        state.mie = 1 << 3; // MSIE=1 (source enable)
        state.mip = 0; // rebuilt by build_mip
        state.mtvec = 0x8000_0100;

        let mut result: BatchResult = unsafe { mem::zeroed() };
        let mem = MemCtx {
            ram: ram.as_mut_ptr(),
            ram_size: ram.len() as u64,
            ram_base: 0,
            shadow_base: 0,
            shadow_size: 0,
        };
        let dev_bases: [u64; 0] = [];
        let dev_ends: [u64; 0] = [];
        let dev = FfiDevCtx { bases: dev_bases.as_ptr(), ends: dev_ends.as_ptr(), num: 0 };
        let mut pmp_cfg: [u8; 0] = [];
        let mut pmp_addr: [u64; 0] = [];
        let pmp = FfiPmpCtx { cfg: pmp_cfg.as_mut_ptr(), addr: pmp_addr.as_mut_ptr(), num: 0, pmpsplit: 0 };
        let mut mtimecmp: [u64; 1] = [u64::MAX];
        let mut msip: [u8; 1] = [1]; // MSIP from another hart
        let mut mtime_val: u64 = 0;
        let clint = FfiClintCtx { mtime: &mut mtime_val as *mut u64, mtimecmp: mtimecmp.as_mut_ptr(), msip: msip.as_mut_ptr(), base: 0 };
        unsafe {
            run_batch(
                &mut state as *mut HartState, 1, 1, &mut result as *mut BatchResult,
                &mem as *const MemCtx, &pmp as *const FfiPmpCtx,
                &clint as *const FfiClintCtx, &dev as *const FfiDevCtx,
            std::ptr::null_mut() as *mut FfiVirtIoCtx, // virtio
                std::ptr::null(), 0,
            );
        }

        // The hart MUST wake from WFI because mip & mie != 0, even though
        // mstatus.MIE == 0.  The interrupt itself won't be delivered
        // (check_and_deliver_interrupt requires MIE), so the hart resumes
        // execution at WFI+4.  One instruction (lb x0, 0(x0) at offset 4)
        // executes before the slice budget (max_instrs=1) is exhausted.
        assert_eq!(
            state.waiting, 0,
            "waiting must be cleared when mip & mie != 0, even with MIE=0"
        );
        assert_eq!(
            state.wfi_woken, 1,
            "wfi_woken must be set on wake-up regardless of mstatus.MIE"
        );
        assert_eq!(
            result.exit_reason,
            exit_reason::NORMAL,
            "no trap delivered (MIE=0), batch should continue normally"
        );
    }

    /// WFI must wake even when ``mie.MSIE == 0``, as long as the CLINT
    /// MSIP register is asserted by another hart.  This matches real
    /// hardware: the physical interrupt line from the CLINT wakes the CPU
    /// regardless of the CSR enable bit.  Without this, a secondary hart
    /// whose ``mie.MSIE`` was accidentally cleared will never respond to
    /// cross-hart IPI, deadlocking multi-hart TLB shootdown (OpenSBI's
    /// ``tlb_sync`` spin-wait).
    ///
    /// After waking, the fix force-sets ``mie.MSIE`` so the interrupt
    /// can be delivered to M-mode for processing.
    #[test]
    #[ignore = "run_batch deprecated — superseded by run_parallel concurrent model"]
    fn wfi_wakes_on_clint_msip_even_when_msie_zero() {
        let mut ram = vec![0u8; 64];
        // Empty RAM at offset 4+ — the hart will attempt to fetch here
        // after waking.  We don't care about the fetch result; we only
        // care about the WFI wake-up and interrupt delivery.

        let mut state = make_state(0, 0);
        state.mode = riscv_mode::S; // S-mode, as in the Linux idle-loop deadlock
        state.mstatus = 1 << 1;     // SIE=1 (S-mode global interrupt enable)
        state.waiting = 1;          // already asleep from a prior WFI
        state.wfi_woken = 0;
        state.pc = 4;               // PC advanced past WFI
        state.mie = 0;              // MSIE=0 — THE BUG CONDITION
        // STIE set so mideleg-based timer interrupts can fire.
        state.mideleg = 1 << 5;     // STIP delegated to S-mode
        state.mip = 0;              // rebuilt by build_mip
        state.mtvec = 0x8000_0100;
        state.stvec = 0x8000_0400;

        let mut result: BatchResult = unsafe { mem::zeroed() };
        let mem = MemCtx {
            ram: ram.as_mut_ptr(),
            ram_size: ram.len() as u64,
            ram_base: 0,
            shadow_base: 0,
            shadow_size: 0,
        };
        let dev_bases: [u64; 0] = [];
        let dev_ends: [u64; 0] = [];
        let dev = FfiDevCtx { bases: dev_bases.as_ptr(), ends: dev_ends.as_ptr(), num: 0 };
        let mut pmp_cfg: [u8; 0] = [];
        let mut pmp_addr: [u64; 0] = [];
        let pmp = FfiPmpCtx { cfg: pmp_cfg.as_mut_ptr(), addr: pmp_addr.as_mut_ptr(), num: 0, pmpsplit: 0 };
        let mut mtimecmp: [u64; 1] = [u64::MAX];
        let mut msip: [u8; 1] = [1]; // MSIP from another hart
        let mut mtime_val: u64 = 0;
        // CLINT base must be non-zero for inline handling
        let clint = FfiClintCtx { mtime: &mut mtime_val as *mut u64, mtimecmp: mtimecmp.as_mut_ptr(), msip: msip.as_mut_ptr(), base: 0x0200_0000 };
        unsafe {
            run_batch(
                &mut state as *mut HartState, 1, 3, &mut result as *mut BatchResult,
                &mem as *const MemCtx, &pmp as *const FfiPmpCtx,
                &clint as *const FfiClintCtx, &dev as *const FfiDevCtx,
            std::ptr::null_mut() as *mut FfiVirtIoCtx, // virtio
                std::ptr::null(), 0,
            );
        }

        // 1. Hart MUST wake from WFI despite mie.MSIE=0 — MSIP is
        //    asserted at the CLINT level, which is the physical wake signal.
        assert_eq!(
            state.waiting, 0,
            "waiting must be cleared even when mie.MSIE=0 (CLINT MSIP wakes the hart)"
        );
        assert_eq!(
            state.wfi_woken, 1,
            "wfi_woken must be set on CLINT MSIP wake-up"
        );

        // 2. After waking, MSIE must be force-set so the interrupt can
        //    be delivered.  Without this the M-mode trap handler never
        //    runs, and the TLB sync counter is never decremented.
        assert!(
            state.mie & (1 << 3) != 0,
            "mie.MSIE must be force-set after CLINT MSIP wake-up (got mie={:#x})",
            state.mie
        );

        // 3. The MSIP interrupt must be delivered to M-mode since it's
        //    not delegatable.  PC should jump to mtvec.
        assert_eq!(
            state.mode, riscv_mode::M,
            "MSIP must cause transition to M-mode (got mode={})", state.mode
        );
        assert_eq!(
            state.pc, 0x8000_0100,
            "PC must jump to mtvec after MSIP trap delivery (got {:#x})", state.pc
        );
    }

    #[test]
    #[ignore = "run_batch deprecated — superseded by run_parallel concurrent model"]
    fn store_to_mmio_device_exits_batch() {
        // Regression: ndev was always 0 so ``is_device_addr`` never
        // detected MMIO — stores silently fell through to ram_write.
        // Verify a regular SW to a device address triggers an MMIO exit.
        let mut ram = vec![0u8; 64];
        // sw x5, 0(x10) -> 0x00552023
        write_u32_le(&mut ram, 0, 0x00552023);

        let mut state = make_state(0, 0);
        state.mode = riscv_mode::M;
        state.gprs[5] = 0xDEAD_BEEF; // rs2: value to store
        state.gprs[10] = 0x1000_0000; // rs1: device base

        let mut result: BatchResult = unsafe { mem::zeroed() };
        let mem = MemCtx {
            ram: ram.as_mut_ptr(),
            ram_size: ram.len() as u64,
            ram_base: 0,
            shadow_base: 0,
            shadow_size: 0,
        };
        let dev_bases: [u64; 1] = [0x1000_0000];
        let dev_ends: [u64; 1] = [0x1000_1000];
        let dev = FfiDevCtx { bases: dev_bases.as_ptr(), ends: dev_ends.as_ptr(), num: 1 };
        let mut pmp_cfg: [u8; 0] = [];
        let mut pmp_addr: [u64; 0] = [];
        let pmp = FfiPmpCtx { cfg: pmp_cfg.as_mut_ptr(), addr: pmp_addr.as_mut_ptr(), num: 0, pmpsplit: 0 };
        let mut mtimecmp: [u64; 1] = [u64::MAX];
        let mut msip: [u8; 1] = [0];
        let mut mtime_val: u64 = 0;
        let clint = FfiClintCtx { mtime: &mut mtime_val as *mut u64, mtimecmp: mtimecmp.as_mut_ptr(), msip: msip.as_mut_ptr(), base: 0 };
        unsafe {
            run_batch(
                &mut state as *mut HartState, 1, 1, &mut result as *mut BatchResult,
                &mem as *const MemCtx, &pmp as *const FfiPmpCtx,
                &clint as *const FfiClintCtx, &dev as *const FfiDevCtx,
            std::ptr::null_mut() as *mut FfiVirtIoCtx, // virtio
                std::ptr::null(), 0,
            );
        }

        assert_eq!(
            result.exit_reason,
            exit_reason::MMIO,
            "store to device MMIO should exit with MMIO reason, got {}",
            result.exit_reason
        );
        assert_eq!(
            result.exit_instr, 0x00552023,
            "exit_instr should be the SW instruction"
        );
        assert_eq!(state.pc, 0, "PC should NOT advance on MMIO exit");
    }

    #[test]
    #[ignore = "run_batch deprecated — superseded by run_parallel concurrent model"]
    fn compressed_store_to_mmio_device_exits_batch() {
        // Regression: handle_c0 / handle_c2 ignored the EXIT_SENTINEL
        // return from store_mem_compressed and unconditionally returned 2,
        // so compressed stores to MMIO silently advanced PC without ever
        // reaching Python for device handling.
        let mut ram = vec![0u8; 64];
        // c.sw x13, 0(x8) -> half=0xC014, full word=0x0000C014
        write_u32_le(&mut ram, 0, 0x0000_C014);

        let mut state = make_state(0, 0);
        state.mode = riscv_mode::M;
        state.gprs[8] = 0x1000_0000; // rs1': device base
        state.gprs[13] = 0xCAFE_BABE; // rs2': value to store

        let mut result: BatchResult = unsafe { mem::zeroed() };
        let mem = MemCtx {
            ram: ram.as_mut_ptr(),
            ram_size: ram.len() as u64,
            ram_base: 0,
            shadow_base: 0,
            shadow_size: 0,
        };
        let dev_bases: [u64; 1] = [0x1000_0000];
        let dev_ends: [u64; 1] = [0x1000_1000];
        let dev = FfiDevCtx { bases: dev_bases.as_ptr(), ends: dev_ends.as_ptr(), num: 1 };
        let mut pmp_cfg: [u8; 0] = [];
        let mut pmp_addr: [u64; 0] = [];
        let pmp = FfiPmpCtx { cfg: pmp_cfg.as_mut_ptr(), addr: pmp_addr.as_mut_ptr(), num: 0, pmpsplit: 0 };
        let mut mtimecmp: [u64; 1] = [u64::MAX];
        let mut msip: [u8; 1] = [0];
        let mut mtime_val: u64 = 0;
        let clint = FfiClintCtx { mtime: &mut mtime_val as *mut u64, mtimecmp: mtimecmp.as_mut_ptr(), msip: msip.as_mut_ptr(), base: 0 };
        unsafe {
            run_batch(
                &mut state as *mut HartState, 1, 1, &mut result as *mut BatchResult,
                &mem as *const MemCtx, &pmp as *const FfiPmpCtx,
                &clint as *const FfiClintCtx, &dev as *const FfiDevCtx,
            std::ptr::null_mut() as *mut FfiVirtIoCtx, // virtio
                std::ptr::null(), 0,
            );
        }

        assert_eq!(
            result.exit_reason,
            exit_reason::MMIO,
            "C.SW to device MMIO should exit with MMIO reason, got {}",
            result.exit_reason
        );
        assert_eq!(
            result.exit_instr, 0x0000_C014,
            "exit_instr should be the full compressed instruction word"
        );
        assert_eq!(state.pc, 0, "PC should NOT advance on MMIO exit");
    }

    #[test]
    #[ignore = "run_batch deprecated — superseded by run_parallel concurrent model"]
    fn wfi_wakes_on_timer_interrupt() {
        // Regression: mtime was a frozen u64, so timer interrupts (MTIP/STIP)
        // never matured during a batch.  After making mtime a *mut u64 that
        // Rust increments per-slice and adding sync_mtip before the WFI check,
        // a WFI hart must wake when mtime advances past mtimecmp.
        //
        // Strategy: 2 harts.  Hart 0 executes enough instructions to push
        // mtime past hart 1's mtimecmp.  Hart 1 is parked in WFI waiting.
        // After hart 0's slice, mtime advances; in the next round-robin
        // round, hart 1's sync_mtip sees the timer has expired and wakes it.
        let mut ram = vec![0u8; 64];
        // Hart 0 code @ offset 0: ADDI x5, x0, 0  × 9 + FENCE (10 instrs)
        for i in 0..9 {
            write_u32_le(&mut ram, (i * 4) as usize, 0x0000_0293); // addi x5, x0, 0
        }
        write_u32_le(&mut ram, 36, 0x0FF0_000F); // FENCE (NOP)

        // Hart 1 code @ offset 40: WFI (enters waiting, then timer wakes it)
        write_u32_le(&mut ram, 40, 0x1050_0073);

        // --- Hart 0: busy worker ---
        let mut state0 = make_state(0, 0);
        state0.mode = riscv_mode::M;
        state0.mstatus = 1 << 3; // MIE=1
        state0.mie = 1 << 7;     // MTIE=1 (timer interrupt enabled)
        state0.mtvec = 0x8000_0100;

        // --- Hart 1: parked in WFI waiting ---
        let mut state1 = make_state(40, 1);
        state1.mode = riscv_mode::M;
        state1.mstatus = 1 << 3; // MIE=1
        state1.mie = 1 << 7;     // MTIE=1
        state1.mtvec = 0x8000_0100;
        state1.waiting = 1;      // already asleep from a prior WFI
        state1.wfi_woken = 0;
        state1.pc = 44;          // PC advanced past WFI

        let mut states = [state0, state1];

        let mem = MemCtx {
            ram: ram.as_mut_ptr(),
            ram_size: ram.len() as u64,
            ram_base: 0,
            shadow_base: 0,
            shadow_size: 0,
        };
        let dev_bases: [u64; 0] = [];
        let dev_ends: [u64; 0] = [];
        let dev = FfiDevCtx { bases: dev_bases.as_ptr(), ends: dev_ends.as_ptr(), num: 0 };
        let mut pmp_cfg: [u8; 0] = [];
        let mut pmp_addr: [u64; 0] = [];
        let pmp = FfiPmpCtx { cfg: pmp_cfg.as_mut_ptr(), addr: pmp_addr.as_mut_ptr(), num: 0, pmpsplit: 0 };

        // mtime starts at 0; hart 1's mtimecmp is set to 5 ticks in the
        // future.  Hart 0 executes 10 instructions -> mtime advances past 5.
        let mut mtime_val: u64 = 0;
        let mut mtimecmp: [u64; 2] = [u64::MAX, 5]; // hart 0: disabled, hart 1: expire at 5
        let mut msip: [u8; 2] = [0; 2];
        let clint = FfiClintCtx {
            mtime: &mut mtime_val as *mut u64,
            mtimecmp: mtimecmp.as_mut_ptr(),
            msip: msip.as_mut_ptr(),
            base: 0,
        };

        let mut result: BatchResult = unsafe { mem::zeroed() };
        unsafe {
            run_batch(
                states.as_mut_ptr(), 2, 100, &mut result as *mut BatchResult,
                &mem as *const MemCtx, &pmp as *const FfiPmpCtx,
                &clint as *const FfiClintCtx, &dev as *const FfiDevCtx,
            std::ptr::null_mut() as *mut FfiVirtIoCtx, // virtio
                std::ptr::null(), 0,
            );
        }

        // After the batch:
        // - Hart 0 executed 10 instructions (mtime advanced to ≥10)
        // - Hart 1's WFI check should see mtime >= mtimecmp[1]=5 -> wake up
        // - Hart 1 should deliver timer interrupt -> PC = mtvec
        assert_eq!(
            states[1].waiting, 0,
            "hart 1 must be woken from WFI by timer interrupt"
        );
        assert_eq!(
            states[1].wfi_woken, 1,
            "wfi_woken must be set when timer interrupt wakes WFI hart"
        );
        assert!(
            mtime_val >= 5,
            "mtime must have advanced past mtimecmp (mtime={}, mtimecmp=5)", mtime_val
        );
    }

    /// SSTC ``stimecmp``-only timer wakeup (no CLINT ``mtimecmp``).
    ///
    /// Regression: ``sync_mtip`` and ``build_mip`` only checked CLINT
    /// ``mtimecmp``.  When the kernel uses the Sstc extension (writing the
    /// ``stimecmp`` CSR directly, bypassing the CLINT ``mtimecmp``), timer
    /// interrupts were never signaled (STIP always 0), so WFI harts
    /// never woke up.
    ///
    /// This test simulates the exact scenario: ``mtimecmp`` is 0 (disabled),
    /// ``stimecmp`` is set to a future deadline, MTIE is enabled.  After
    /// another hart advances mtime past the deadline, ``sync_mtip`` must
    /// see the expired ``stimecmp``, set STIP, and wake the sleeping hart.
    #[test]
    #[ignore = "run_batch deprecated — superseded by run_parallel concurrent model"]
    fn wfi_wakes_on_stimecmp_sstc_timer() {
        let mut ram = vec![0u8; 128];
        // Hart 0 code @ offset 0: ADDI × 19 + FENCE (20 instrs -> mtime ≥ 20)
        for i in 0..19 {
            write_u32_le(&mut ram, (i * 4) as usize, 0x0000_0293); // addi x5, x0, 0
        }
        write_u32_le(&mut ram, 76, 0x0FF0_000F); // FENCE (NOP)

        // Hart 1 code @ offset 80: WFI
        write_u32_le(&mut ram, 80, 0x1050_0073);

        // --- Hart 0: busy worker ---
        let mut state0 = make_state(0, 0);
        state0.mode = riscv_mode::M;
        state0.mstatus = 1 << 3; // MIE=1
        state0.mie = 1 << 7;     // MTIE=1
        state0.mtvec = 0x8000_0100;

        // --- Hart 1: S-mode, SSTC timer armed, in WFI ---
        let mut state1 = make_state(80, 1);
        state1.mode = riscv_mode::S;
        state1.mstatus = 1 << 1; // SIE=1 (S-mode global interrupt enable)
        state1.mie = 1 << 5;     // STIE enabled (bit 5; needed for STIP wakeup)
        state1.mideleg = 1 << 5; // STI delegated to S-mode
        state1.mtvec = 0x8000_0100;
        state1.stvec = 0x8000_0400;
        state1.stimecmp = 10;    // SSTC: timer expires at mtime=10
        state1.waiting = 1;      // already in WFI from a prior instruction
        state1.wfi_woken = 0;
        state1.pc = 84;          // PC advanced past WFI

        let mut states = [state0, state1];

        let mem = MemCtx {
            ram: ram.as_mut_ptr(),
            ram_size: ram.len() as u64,
            ram_base: 0,
            shadow_base: 0,
            shadow_size: 0,
        };
        let dev_bases: [u64; 0] = [];
        let dev_ends: [u64; 0] = [];
        let dev = FfiDevCtx { bases: dev_bases.as_ptr(), ends: dev_ends.as_ptr(), num: 0 };
        let mut pmp_cfg: [u8; 0] = [];
        let mut pmp_addr: [u64; 0] = [];
        let pmp = FfiPmpCtx { cfg: pmp_cfg.as_mut_ptr(), addr: pmp_addr.as_mut_ptr(), num: 0, pmpsplit: 0 };

        // CLINT: mtimecmp = 0 for both harts (disabled — kernel uses SSTC only).
        // mtime starts at 0; hart 0 executes 20+ instrs -> mtime ≥ 20 > stimecmp(=10).
        let mut mtime_val: u64 = 0;
        let mut mtimecmp: [u64; 2] = [0, 0];
        let mut msip: [u8; 2] = [0; 2];
        let clint = FfiClintCtx {
            mtime: &mut mtime_val as *mut u64,
            mtimecmp: mtimecmp.as_mut_ptr(),
            msip: msip.as_mut_ptr(),
            base: 0x0200_0000,  // non-zero -> CLINT inlining active
        };

        let mut result: BatchResult = unsafe { mem::zeroed() };
        unsafe {
            run_batch(
                states.as_mut_ptr(), 2, 100, &mut result as *mut BatchResult,
                &mem as *const MemCtx, &pmp as *const FfiPmpCtx,
                &clint as *const FfiClintCtx, &dev as *const FfiDevCtx,
            std::ptr::null_mut() as *mut FfiVirtIoCtx, // virtio
                std::ptr::null(), 0,
            );
        }

        assert_eq!(
            states[1].waiting, 0,
            "hart 1 must be woken from WFI by SSTC stimecmp timer (mtime={}, stimecmp=10, mtimecmp=0)",
            mtime_val
        );
        assert_eq!(
            states[1].wfi_woken, 1,
            "wfi_woken must be set when SSTC stimecmp wakes WFI hart"
        );
        assert!(
            mtime_val >= 10,
            "mtime must have advanced past stimecmp (mtime={}, stimecmp=10)", mtime_val
        );
    }

    /// ``build_mip`` sets STIP from ``stimecmp`` even when ``mtimecmp == 0``.
    ///
    /// Regression: ``build_mip`` was clearing STIP when CLINT ``mtimecmp`` was
    /// zero, ignoring the Sstc ``stimecmp`` CSR.  After the fix, STIP must be
    /// set when *either* timer source has expired.
    #[test]
    fn build_mip_sets_stip_from_stimecmp_alone() {
        let mut state: HartState = unsafe { mem::zeroed() };
        state.mode = riscv_mode::S;
        state.stimecmp = 100; // SSTC armed

        let mtime: u64 = 200; // past the deadline
        let mtimecmp: [u64; 1] = [0]; // CLINT timer disabled
        let msip: [u8; 1] = [0];

        let mip = build_mip(&state, mtime, mtimecmp.as_ptr() as *mut u64, msip.as_ptr() as *mut u8);

        assert!(
            mip & (1 << 5) != 0,
            "STIP must be set when stimecmp has expired (mtime={}, stimecmp=100, mtimecmp=0), got mip={:#x}",
            mtime, mip
        );
        assert!(
            mip & (1 << 7) == 0,
            "MTIP must NOT be set when mtimecmp is zero (only stimecmp drives STIP)"
        );
    }

    /// ``build_mip`` clears STIP when *neither* ``mtimecmp`` nor ``stimecmp``
    /// has expired yet.
    #[test]
    fn build_mip_clears_stip_when_neither_expired() {
        let mut state: HartState = unsafe { mem::zeroed() };
        state.mode = riscv_mode::S;
        state.mip = 1 << 5; // STIP was previously set
        state.stimecmp = 500; // far future

        let mtime: u64 = 100; // not there yet
        let mtimecmp: [u64; 1] = [0]; // CLINT timer disabled
        let msip: [u8; 1] = [0];

        let mip = build_mip(&state, mtime, mtimecmp.as_ptr() as *mut u64, msip.as_ptr() as *mut u8);

        assert!(
            mip & (1 << 5) == 0,
            "STIP must be cleared when neither mtimecmp nor stimecmp has expired"
        );
    }

    /// ``build_mip`` sets STIP from CLINT ``mtimecmp`` when ``stimecmp``
    /// has *not* expired — the two sources are OR'd, not mutually exclusive.
    #[test]
    fn build_mip_sets_stip_from_mtimecmp_when_stimecmp_not_expired() {
        let mut state: HartState = unsafe { mem::zeroed() };
        state.mode = riscv_mode::S;
        state.stimecmp = 500; // far future — not expired

        let mtime: u64 = 200;
        let mtimecmp: [u64; 1] = [100]; // CLINT timer expired
        let msip: [u8; 1] = [0];

        let mip = build_mip(&state, mtime, mtimecmp.as_ptr() as *mut u64, msip.as_ptr() as *mut u8);

        assert!(
            mip & (1 << 5) != 0,
            "STIP must be set from mtimecmp even when stimecmp hasn't expired yet"
        );
        assert!(
            mip & (1 << 7) != 0,
            "MTIP must be set when mtimecmp has expired"
        );
    }

    #[test]
    #[ignore = "run_batch deprecated — superseded by run_parallel concurrent model"]
    fn sie_write_propagates_to_mie_via_run_batch() {
        // Verify that csrrw x0, sie, rs1 updates mie through the full
        // run_batch pipeline (instruction fetch -> dispatch -> csr_write).
        let mut ram = vec![0u8; 64];
        // addi x5, x0, 32  -> x5 = 0x20 (STIE bit)
        write_u32_le(&mut ram, 0, 0x0200_0293);
        // csrrw x0, sie, x5  -> write sie = 0x20
        write_u32_le(&mut ram, 4, 0x1042_9073);
        // wfi -> stop batch cleanly
        write_u32_le(&mut ram, 8, 0x1050_0073);

        let mut state = make_state(0, 0);
        state.mode = riscv_mode::M;
        state.mideleg = 1 << 5; // STIE delegated
        state.mie = 0;
        state.mip = 0;

        let mut result: BatchResult = unsafe { mem::zeroed() };
        unsafe {
            run_batch_defaults(
                &mut state as *mut HartState,
                1,
                ram.as_mut_ptr(),
                ram.len() as u64,
                0,
                0,
                0,
                3,
                &mut result as *mut BatchResult,
            );
        }

        assert_eq!(state.mie & (1 << 5), 1 << 5,
            "csrrw sie,STIE must set mie.STIE via run_batch pipeline (got mie=0x{:x})",
            state.mie);
    }

    // ============================================================
    //  Pre-execution breakpoint tests
    // ============================================================

    /// Breakpoint set on a JALR instruction must fire BEFORE execution,
    /// i.e. on the JALR address itself, not on the jump target.
    #[test]
    #[ignore = "run_batch deprecated — superseded by run_parallel concurrent model"]
    fn bp_fires_on_jalr_pre_execution() {
        let mut ram = vec![0u8; 64];
        // addi x1, x0, 16  (load target address into x1)
        write_u32_le(&mut ram, 0, 0x0100_0093);
        // jalr x0, 0(x1)   (jump to x1; breakpoint here)
        write_u32_le(&mut ram, 4, 0x0000_8067);
        // fence (at target — should NOT be reached before breakpoint fires)
        write_u32_le(&mut ram, 16, 0x0FF0_000F);

        let mut state = make_state(0, 0);
        let mut result: BatchResult = unsafe { mem::zeroed() };
        let bp_addrs: [u64; 1] = [4];
        let mem = MemCtx {
            ram: ram.as_mut_ptr(), ram_size: ram.len() as u64,
            ram_base: 0, shadow_base: 0, shadow_size: 0,
        };
        let (mut mtimecmp, mut msip, mut mtime) = ([u64::MAX; 1], [0u8; 1], 0u64);
        let (mut pmp_cfg, mut pmp_addr): ([u8; 0], [u64; 0]) = ([], []);
        let (dev_bases, dev_ends): ([u64; 0], [u64; 0]) = ([], []);
        let pmp = FfiPmpCtx {
            cfg: pmp_cfg.as_mut_ptr(), addr: pmp_addr.as_mut_ptr(),
            num: 0, pmpsplit: 0,
        };
        let clint = FfiClintCtx {
            mtime: &mut mtime as *mut u64,
            mtimecmp: mtimecmp.as_mut_ptr(), msip: msip.as_mut_ptr(), base: 0,
        };
        let dev = FfiDevCtx { bases: dev_bases.as_ptr(), ends: dev_ends.as_ptr(), num: 0 };
        unsafe {
            run_batch(
                &mut state as *mut HartState, 1, 16,
                &mut result as *mut BatchResult,
                &mem as *const MemCtx, &pmp as *const FfiPmpCtx,
                &clint as *const FfiClintCtx, &dev as *const FfiDevCtx,
            std::ptr::null_mut() as *mut FfiVirtIoCtx, // virtio
                bp_addrs.as_ptr(), 1,
            );
        }
        assert_eq!(result.exit_reason, exit_reason::BREAKPOINT,
            "breakpoint on JALR must fire (got reason={})", result.exit_reason);
        assert_eq!(result.exit_pc, 4,
            "exit PC must be the JALR address 0x4 (got 0x{:x})", result.exit_pc);
    }

    /// Breakpoint on a JAL instruction must fire BEFORE execution.
    #[test]
    #[ignore = "run_batch deprecated — superseded by run_parallel concurrent model"]
    fn bp_fires_on_jal_pre_execution() {
        let mut ram = vec![0u8; 64];
        // jal x0, 12   (jump to PC+12; breakpoint at 0x0)
        write_u32_le(&mut ram, 0, 0x00C0_006F);
        // fence (at 0x4 — should NOT be reached if breakpoint fires)
        write_u32_le(&mut ram, 4, 0x0FF0_000F);

        let mut state = make_state(0, 0);
        let mut result: BatchResult = unsafe { mem::zeroed() };
        let bp_addrs: [u64; 1] = [0];
        let mem = MemCtx {
            ram: ram.as_mut_ptr(), ram_size: ram.len() as u64,
            ram_base: 0, shadow_base: 0, shadow_size: 0,
        };
        let (mut mtimecmp, mut msip, mut mtime) = ([u64::MAX; 1], [0u8; 1], 0u64);
        let (mut pmp_cfg, mut pmp_addr): ([u8; 0], [u64; 0]) = ([], []);
        let (dev_bases, dev_ends): ([u64; 0], [u64; 0]) = ([], []);
        let pmp = FfiPmpCtx {
            cfg: pmp_cfg.as_mut_ptr(), addr: pmp_addr.as_mut_ptr(),
            num: 0, pmpsplit: 0,
        };
        let clint = FfiClintCtx {
            mtime: &mut mtime as *mut u64,
            mtimecmp: mtimecmp.as_mut_ptr(), msip: msip.as_mut_ptr(), base: 0,
        };
        let dev = FfiDevCtx { bases: dev_bases.as_ptr(), ends: dev_ends.as_ptr(), num: 0 };
        unsafe {
            run_batch(
                &mut state as *mut HartState, 1, 16,
                &mut result as *mut BatchResult,
                &mem as *const MemCtx, &pmp as *const FfiPmpCtx,
                &clint as *const FfiClintCtx, &dev as *const FfiDevCtx,
            std::ptr::null_mut() as *mut FfiVirtIoCtx, // virtio
                bp_addrs.as_ptr(), 1,
            );
        }
        assert_eq!(result.exit_reason, exit_reason::BREAKPOINT,
            "breakpoint on JAL must fire (got reason={})", result.exit_reason);
        assert_eq!(result.exit_pc, 0,
            "exit PC must be the JAL address 0x0 (got 0x{:x})", result.exit_pc);
    }

    /// Post-execution breakpoint check still works for sequential code.
    #[test]
    #[ignore = "run_batch deprecated — superseded by run_parallel concurrent model"]
    fn bp_fires_on_next_sequential_pc() {
        let mut ram = vec![0u8; 64];
        // addi x1, x0, 5
        write_u32_le(&mut ram, 0, 0x0050_0093);
        // addi x2, x0, 3  (breakpoint here after first addi executes)
        write_u32_le(&mut ram, 4, 0x0030_0113);
        // fence
        write_u32_le(&mut ram, 8, 0x0FF0_000F);

        let mut state = make_state(0, 0);
        let mut result: BatchResult = unsafe { mem::zeroed() };
        let bp_addrs: [u64; 1] = [4];
        let mem = MemCtx {
            ram: ram.as_mut_ptr(), ram_size: ram.len() as u64,
            ram_base: 0, shadow_base: 0, shadow_size: 0,
        };
        let (mut mtimecmp, mut msip, mut mtime) = ([u64::MAX; 1], [0u8; 1], 0u64);
        let (mut pmp_cfg, mut pmp_addr): ([u8; 0], [u64; 0]) = ([], []);
        let (dev_bases, dev_ends): ([u64; 0], [u64; 0]) = ([], []);
        let pmp = FfiPmpCtx {
            cfg: pmp_cfg.as_mut_ptr(), addr: pmp_addr.as_mut_ptr(),
            num: 0, pmpsplit: 0,
        };
        let clint = FfiClintCtx {
            mtime: &mut mtime as *mut u64,
            mtimecmp: mtimecmp.as_mut_ptr(), msip: msip.as_mut_ptr(), base: 0,
        };
        let dev = FfiDevCtx { bases: dev_bases.as_ptr(), ends: dev_ends.as_ptr(), num: 0 };
        unsafe {
            run_batch(
                &mut state as *mut HartState, 1, 16,
                &mut result as *mut BatchResult,
                &mem as *const MemCtx, &pmp as *const FfiPmpCtx,
                &clint as *const FfiClintCtx, &dev as *const FfiDevCtx,
            std::ptr::null_mut() as *mut FfiVirtIoCtx, // virtio
                bp_addrs.as_ptr(), 1,
            );
        }
        assert_eq!(result.exit_reason, exit_reason::BREAKPOINT,
            "sequential breakpoint must fire (got reason={})", result.exit_reason);
        assert_eq!(result.exit_pc, 4,
            "exit PC must be 0x4 (got 0x{:x})", result.exit_pc);
    }

    // ----------------------------------------------------------
    //  Cross-hart MSIP / IPI yield tests
    // ----------------------------------------------------------

    fn make_clint_n(
        base: u64, mtime: &mut u64, mtimecmp: &mut [u64], msip: &mut [u8],
    ) -> FfiClintCtx {
        FfiClintCtx {
            mtime: mtime as *mut u64,
            mtimecmp: mtimecmp.as_mut_ptr(),
            msip: msip.as_mut_ptr(),
            base,
        }
    }

    /// Hart 0 stores 1 to MSIP[1] -> target hart's msip/mip updated.
    /// Uses CLINT_BASE=0x1000 which lies OUTSIDE the RAM range so the
    /// store MUST be intercepted by the CLINT handler (not ram_write).
    #[test]
    #[ignore = "run_batch deprecated — superseded by run_parallel concurrent model"]
    fn cross_hart_msip_write_updates_target() {
        const CLINT_BASE: u64 = 0x1000;
        let mut ram = vec![0u8; 64];
        // WFI at PC=0 so hart 0 is running a harmless instruction first
        // sw x2, 4(x1) at PC=4 (advance past initial harmless instr if any)
        // Actually: just put SW at PC=0 and nothing at PC=4
        write_u32_le(&mut ram, 0, 0x0020_A223); // sw x2, 4(x1) -> CLINT_BASE+4=0x1004

        let mut s0 = make_state(0, 0);
        s0.gprs[1] = CLINT_BASE;  // x1 = 0x1000
        s0.gprs[2] = 1;           // x2 = 1
        let s1 = make_state(0, 1); // target hart

        let mut result: BatchResult = unsafe { mem::zeroed() };
        let mem = MemCtx { ram: ram.as_mut_ptr(), ram_size: ram.len() as u64,
            ram_base: 0, shadow_base: 0, shadow_size: 0 };
        let dev = FfiDevCtx { bases: [].as_ptr(), ends: [].as_ptr(), num: 0 };
        let mut pmp_cfg: [u8; 0] = [];
        let mut pmp_addr: [u64; 0] = [];
        let pmp = FfiPmpCtx { cfg: pmp_cfg.as_mut_ptr(), addr: pmp_addr.as_mut_ptr(),
            num: 0, pmpsplit: 0 };
        let mut mtimecmp = [u64::MAX; 2];
        let mut msip = [0u8; 2];
        let mut mtime: u64 = 0;
        let clint = make_clint_n(CLINT_BASE, &mut mtime, &mut mtimecmp, &mut msip);

        let mut states = [s0, s1];
        unsafe {
            run_batch(states.as_mut_ptr(), 2, 1, &mut result as *mut BatchResult,
                &mem as *const MemCtx, &pmp as *const FfiPmpCtx,
                &clint as *const FfiClintCtx, &dev as *const FfiDevCtx,
            std::ptr::null_mut() as *mut FfiVirtIoCtx, // virtio
                std::ptr::null(), 0);
        }

        // CLINT_BASE=0x1000, offset 4 -> MSIP[1]
        let store_pa = states[0].diag.clint_mtc_wr; // we stashed PA here
        assert_eq!(msip[1], 1,
            "MSIP[1] must be 1 (reason={} total={} pc0={:#x} msip_set={} store_pa={:#x})",
            result.exit_reason, result.total_instrs, states[0].pc,
            states[0].diag.clint_msip_set, store_pa);
        assert!(states[1].mip & (1 << 3) != 0, "hart 1 mip must have MSIP");
        assert_eq!(states[0].mip & (1 << 3), 0, "hart 0 mip must be clean");
        assert_eq!(states[0].pc, 4, "hart 0 PC advanced past SW");
    }

    /// Self MSIP: hart 0 stores 1 to its own MSIP[0]; hart 1 untouched.
    #[test]
    #[ignore = "run_batch deprecated — superseded by run_parallel concurrent model"]
    fn self_msip_write_only_affects_self() {
        const CLINT_BASE: u64 = 0x200_0000;
        let mut ram = vec![0u8; 32];
        // sw x2, 0(x1)  ->  MSIP[0]
        write_u32_le(&mut ram, 0, 0x0020_A023);

        let mut s0 = make_state(0, 0);
        s0.gprs[1] = CLINT_BASE;
        s0.gprs[2] = 1;
        let s1 = make_state(0, 1);

        let mut result: BatchResult = unsafe { mem::zeroed() };
        let mem = MemCtx { ram: ram.as_mut_ptr(), ram_size: ram.len() as u64,
            ram_base: 0, shadow_base: 0, shadow_size: 0 };
        let dev = FfiDevCtx { bases: [].as_ptr(), ends: [].as_ptr(), num: 0 };
        let mut pmp_cfg: [u8; 0] = [];
        let mut pmp_addr: [u64; 0] = [];
        let pmp = FfiPmpCtx { cfg: pmp_cfg.as_mut_ptr(), addr: pmp_addr.as_mut_ptr(),
            num: 0, pmpsplit: 0 };
        let mut mtimecmp = [u64::MAX; 2];
        let mut msip = [0u8; 2];
        let mut mtime: u64 = 0;
        let clint = make_clint_n(CLINT_BASE, &mut mtime, &mut mtimecmp, &mut msip);

        let mut states = [s0, s1];
        unsafe {
            run_batch(states.as_mut_ptr(), 2, 1, &mut result as *mut BatchResult,
                &mem as *const MemCtx, &pmp as *const FfiPmpCtx,
                &clint as *const FfiClintCtx, &dev as *const FfiDevCtx,
            std::ptr::null_mut() as *mut FfiVirtIoCtx, // virtio
                std::ptr::null(), 0);
        }

        assert_eq!(msip[0], 1, "MSIP[0] must be 1 after self-IPI");
        assert!(states[0].mip & (1 << 3) != 0, "hart 0 mip must have MSIP");
        assert_eq!(msip[1], 0, "MSIP[1] must stay 0");
        assert_eq!(states[1].mip & (1 << 3), 0, "hart 1 mip must be clean");
    }

    /// SBI IPI fast path (mask_base==0) for cross-hart IPI sets target MSIP.
    #[test]
    #[ignore = "run_batch deprecated — superseded by run_parallel concurrent model"]
    fn sbi_ipi_fast_path_cross_hart() {
        let mut ram = vec![0u8; 64];
        write_u32_le(&mut ram, 0, 0x0000_0073); // ECALL at offset 0

        let mut s0 = make_state(0, 0);
        s0.mode = riscv_mode::S;
        s0.gprs[17] = 0x735049; // a7 = SBI_IPI EID (9)
        s0.gprs[16] = 0;        // a6 = SEND_IPI FID (0)
        s0.gprs[10] = 2;        // a0 = hart_mask (bit 1 -> hart 1)
        s0.gprs[11] = 0;        // a1 = mask_base (0 -> fast path)
        s0.mtvec = 0x80;

        let s1 = make_state(0, 1);

        let mut result: BatchResult = unsafe { mem::zeroed() };
        let mem = MemCtx { ram: ram.as_mut_ptr(), ram_size: ram.len() as u64,
            ram_base: 0, shadow_base: 0, shadow_size: 0 };
        let dev = FfiDevCtx { bases: [].as_ptr(), ends: [].as_ptr(), num: 0 };
        let mut pmp_cfg: [u8; 0] = [];
        let mut pmp_addr: [u64; 0] = [];
        let pmp = FfiPmpCtx { cfg: pmp_cfg.as_mut_ptr(), addr: pmp_addr.as_mut_ptr(),
            num: 0, pmpsplit: 0 };
        let mut mtimecmp = [u64::MAX; 2];
        let mut msip = [0u8; 2];
        let mut mtime: u64 = 0;
        let clint = make_clint_n(0x200_0000, &mut mtime, &mut mtimecmp, &mut msip);

        let mut states = [s0, s1];
        unsafe {
            run_batch(states.as_mut_ptr(), 2, 2, &mut result as *mut BatchResult,
                &mem as *const MemCtx, &pmp as *const FfiPmpCtx,
                &clint as *const FfiClintCtx, &dev as *const FfiDevCtx,
            std::ptr::null_mut() as *mut FfiVirtIoCtx, // virtio
                std::ptr::null(), 0);
        }

        assert_eq!(msip[1], 1, "SBI fast path: MSIP[1] must be 1");
        assert!(states[1].mip & (1 << 3) != 0, "SBI fast path: hart 1 mip must have MSIP");
        assert_eq!(states[0].gprs[10], 0, "a0 must be SBI_SUCCESS (0)");
    }

    /// 3-hart broadcast: hart 0 writes MSIP to harts 1 and 2.
    /// Verifies both targets receive the IPI and the yield mechanism
    /// doesn't lose any MSIP writes.
    #[test]
    #[ignore = "run_batch deprecated — superseded by run_parallel concurrent model"]
    fn cross_hart_msip_multi_target_3harts() {
        const CLINT_BASE: u64 = 0x200_0000;
        let mut ram = vec![0u8; 64];
        // Hart 0 instructions: SW to MSIP[1], then SW to MSIP[2]
        // sw x2, 4(x1)  ->  MSIP[1]  at PC=0
        write_u32_le(&mut ram, 0, 0x0020_A223);
        // sw x2, 8(x1)  ->  MSIP[2]  at PC=4
        write_u32_le(&mut ram, 4, 0x0020_A423);

        let mut s0 = make_state(0, 0);
        s0.gprs[1] = CLINT_BASE;
        s0.gprs[2] = 1;

        // Hart 1 and 2 are halted (dummy targets — skip immediately without
        // consuming instruction budget).  Previously we used waiting=1, but
        // with CLINT-directed WFI wake-up, a hart with MSIP asserted will
        // now wake even when mie.MSIE==0, consuming budget meant for the
        // sender.  Halted harts are unconditionally skipped.
        let mut s1 = make_state(0, 1);
        s1.halted = 1;
        let mut s2 = make_state(0, 2);
        s2.halted = 1;

        let mut result: BatchResult = unsafe { mem::zeroed() };
        let mem = MemCtx { ram: ram.as_mut_ptr(), ram_size: ram.len() as u64,
            ram_base: 0, shadow_base: 0, shadow_size: 0 };
        let dev = FfiDevCtx { bases: [].as_ptr(), ends: [].as_ptr(), num: 0 };
        let mut pmp_cfg: [u8; 0] = [];
        let mut pmp_addr: [u64; 0] = [];
        let pmp = FfiPmpCtx { cfg: pmp_cfg.as_mut_ptr(), addr: pmp_addr.as_mut_ptr(),
            num: 0, pmpsplit: 0 };
        let mut mtimecmp = [u64::MAX; 3];
        let mut msip = [0u8; 3];
        let mut mtime: u64 = 0;
        let clint = make_clint_n(CLINT_BASE, &mut mtime, &mut mtimecmp, &mut msip);

        let mut states = [s0, s1, s2];
        unsafe {
            // max_instrs=2: enough for both SWs (each round hart 0 yields
            // after 1 SW, then other two WFI harts run 0 instrs each).
            run_batch(states.as_mut_ptr(), 3, 2, &mut result as *mut BatchResult,
                &mem as *const MemCtx, &pmp as *const FfiPmpCtx,
                &clint as *const FfiClintCtx, &dev as *const FfiDevCtx,
            std::ptr::null_mut() as *mut FfiVirtIoCtx, // virtio
                std::ptr::null(), 0);
        }

        // Both targets received the IPI.
        assert_eq!(msip[1], 1, "MSIP[1] must be 1 (first target)");
        assert_eq!(msip[2], 1, "MSIP[2] must be 1 (second target)");
        assert_eq!(msip[0], 0, "MSIP[0] must be 0 (sender, not targeted)");
        assert!(states[1].mip & (1 << 3) != 0, "hart 1 mip must have MSIP");
        assert!(states[2].mip & (1 << 3) != 0, "hart 2 mip must have MSIP");
        assert_eq!(states[0].mip & (1 << 3), 0, "hart 0 mip must be clean");
    }
}
