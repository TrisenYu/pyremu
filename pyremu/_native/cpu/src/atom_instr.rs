use crate::concurrent::{ModuleState, StopInfo};
use crate::handlers::{
    lr_check, lr_clear_all, lr_set, pmp_ok, try_handle_virtio, DevCtx, PmpCtx, EXIT_SENTINEL,
};
use crate::hart_sched::{ram_offset, read_gpr};
use crate::peripheral::is_device_addr;
use crate::state::{exit_reason, HartState};
use crate::translate::{translate_va, TranslateFault, WalkCtx};
use crate::trap::{deliver_illegal_instruction, deliver_trap, exc_code, mcause_val};
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

pub(crate) fn atomic_minmax_u32(atomic: &AtomicU32, funct5: u8, rs2: u32) -> u64 {
    loop {
        let cur = atomic.load(Ordering::Relaxed);
        let next = match funct5 {
            0b10000 => (cur as i32).min(rs2 as i32) as u32, // AMOMIN
            0b10100 => (cur as i32).max(rs2 as i32) as u32, // AMOMAX
            0b11000 => cur.min(rs2),                        // AMOMINU
            0b11100 => cur.max(rs2),                        // AMOMAXU
            _ => return 0,
        };
        match atomic.compare_exchange_weak(cur, next, Ordering::AcqRel, Ordering::Relaxed) {
            Ok(v) => return v as u64,
            Err(_) => continue,
        }
    }
}

pub(crate) fn atomic_minmax_u64(atomic: &AtomicU64, funct5: u8, rs2: u64) -> u64 {
    loop {
        let cur = atomic.load(Ordering::Relaxed);
        let next = match funct5 {
            0b10000 => (cur as i64).min(rs2 as i64) as u64, // AMOMIN
            0b10100 => (cur as i64).max(rs2 as i64) as u64, // AMOMAX
            0b11000 => cur.min(rs2),                        // AMOMINU
            0b11100 => cur.max(rs2),                        // AMOMAXU
            _ => return 0,
        };
        match atomic.compare_exchange_weak(cur, next, Ordering::AcqRel, Ordering::Relaxed) {
            Ok(v) => return v,
            Err(_) => continue,
        }
    }
}

pub(crate) fn handle_amo_concurrent(
    state: &mut HartState,
    funct5: u8,
    funct3: u8,
    rs2_val: u64,
    rd: u8,
    pa: u64,
    ctx: &WalkCtx,
    pmp: &PmpCtx,
    dev: &DevCtx,
) -> u64 {
    let width: u8 = match funct3 {
        0b010 => 4,
        0b011 => 8,
        _ => return 0, // illegal — caller handles
    };

    // Alignment check (RISC-V AMO requires natural alignment)
    if pa & (width as u64 - 1) != 0 {
        return 0; // caller delivers misaligned trap
    }

    // PMP check
    if !pmp_ok(state, pa, width as u32, true, false, pmp) {
        return 0;
    }

    // MMIO check
    if is_device_addr(pa, dev) {
        return EXIT_SENTINEL; // exit to Python for MMIO handling
    }

    let off = match ram_offset(
        pa,
        width as u32,
        ctx.ram_base,
        ctx.ram_size,
        ctx.shadow_base,
        ctx.shadow_size,
    ) {
        Some(o) => o as usize,
        None => return 0,
    };

    match funct5 {
        0b00010 => {
            // LR.W / LR.D
            // RISC-V spec §8.2: LR.W sign-extends the loaded 32-bit value to
            // 64 bits on RV64.
            let loaded: u64 = if width == 4 {
                let atomic = unsafe { &*(ctx.ram.add(off) as *const AtomicU32) };
                let raw = atomic.load(Ordering::Acquire);
                ((raw as i32) as i64) as u64
            } else {
                let atomic = unsafe { &*(ctx.ram.add(off) as *const AtomicU64) };
                atomic.load(Ordering::Acquire)
            };
            state.reservation_valid = 1;
            state.reservation_addr = pa;
            state.reservation_value = loaded;
            // Register this reservation so other harts' stores will clear it.
            lr_set(ctx, state.mhartid as u8, pa);
            state.gprs[rd as usize] = if rd != 0 { loaded } else { 0 };
            4
        }
        0b00011 => {
            // SC.W / SC.D
            let shared_ok = lr_check(ctx, state.mhartid as u8, pa);
            if !shared_ok || state.reservation_valid == 0 || state.reservation_addr != pa {
                state.reservation_valid = 0;
                if rd != 0 {
                    state.gprs[rd as usize] = 1;
                }
                return 4;
            }
            // Clear ALL reservations BEFORE the CAS, with SeqCst fence,
            // so other harts see the clear before (or with) the new value.
            lr_clear_all(ctx);
            std::sync::atomic::fence(Ordering::SeqCst);
            // CAS with the LR-loaded value as expected.
            let success = if width == 4 {
                let atomic = unsafe { &*(ctx.ram.add(off) as *const AtomicU32) };
                atomic
                    .compare_exchange(
                        state.reservation_value as u32,
                        rs2_val as u32,
                        Ordering::Release,
                        Ordering::Relaxed,
                    )
                    .is_ok()
            } else {
                let atomic = unsafe { &*(ctx.ram.add(off) as *const AtomicU64) };
                atomic
                    .compare_exchange(
                        state.reservation_value,
                        rs2_val,
                        Ordering::Release,
                        Ordering::Relaxed,
                    )
                    .is_ok()
            };
            state.reservation_valid = 0;
            if rd != 0 {
                state.gprs[rd as usize] = if success { 0 } else { 1 };
            }
            4
        }
        // AMOSWAP / AMOADD / AMOXOR / AMOAND / AMOOR / AMOMIN / AMOMAX / AMOMINU / AMOMAXU
        0b00001 | 0b00000 | 0b00100 | 0b01100 | 0b01000 | 0b10000 | 0b10100 | 0b11000 | 0b11100 => {
            // Clear reservations BEFORE the atomic write (ABA fix).
            lr_clear_all(ctx);
            std::sync::atomic::fence(Ordering::SeqCst);
            let old: u64 = if width == 4 {
                let atomic = unsafe { &*(ctx.ram.add(off) as *const AtomicU32) };
                let rv = rs2_val as u32;
                match funct5 {
                    0b00001 => atomic.swap(rv, Ordering::AcqRel) as u64,
                    0b00000 => atomic.fetch_add(rv, Ordering::AcqRel) as u64,
                    0b00100 => atomic.fetch_xor(rv, Ordering::AcqRel) as u64,
                    0b01100 => atomic.fetch_and(rv, Ordering::AcqRel) as u64,
                    0b01000 => atomic.fetch_or(rv, Ordering::AcqRel) as u64,
                    _ => atomic_minmax_u32(atomic, funct5, rv),
                }
            } else {
                let atomic = unsafe { &*(ctx.ram.add(off) as *const AtomicU64) };
                let old_val = match funct5 {
                    0b00001 => atomic.swap(rs2_val, Ordering::AcqRel),
                    0b00000 => atomic.fetch_add(rs2_val, Ordering::AcqRel),
                    0b00100 => atomic.fetch_xor(rs2_val, Ordering::AcqRel),
                    0b01100 => atomic.fetch_and(rs2_val, Ordering::AcqRel),
                    0b01000 => atomic.fetch_or(rs2_val, Ordering::AcqRel),
                    _ => atomic_minmax_u64(atomic, funct5, rs2_val),
                };
                old_val
            };
            if rd != 0 {
                state.gprs[rd as usize] = if width == 4 {
                    ((old as i32) as i64) as u64
                } else {
                    old
                };
            }
            #[cfg(feature = "diagnostic")]
            if funct5 == 0b00001 || funct5 == 0b00100 {
                // AMOSWAP / AMOXOR — rare in normal userspace
                let name = if funct5 == 0b00001 {
                    "AMOSWAP"
                } else {
                    "AMOXOR"
                };
                crate::diag::log_line(&format!(
                    "[{}] pc={:#018x} pa={:#018x} w={} rd=x{} rs2={:#018x} old={:#018x}",
                    name, state.pc, pa, width, rd, rs2_val, old,
                ));
            }
            4
        }
        _ => 0, // illegal — caller handles
    }
}

pub(crate) fn handle_amo_concurrent_dispatch(
    state: &mut HartState,
    f: &crate::decode::DecodedFields,
    instr: u32,
    ctx: &WalkCtx,
    pmp: &PmpCtx,
    dev: &DevCtx,
    module: &ModuleState,
) -> u64 {
    let funct5 = f.func7 >> 2;
    let width: u8 = match f.func3 {
        0b010 => 4,
        0b011 => 8,
        _ => {
            let mut dummy = unsafe { std::mem::zeroed() };
            deliver_illegal_instruction(state, instr as u64, &mut dummy);
            return 0;
        }
    };

    let base = read_gpr(state, f.rs1);
    let va = base;

    if va & (width as u64 - 1) != 0 {
        let mut dummy = unsafe { std::mem::zeroed() };
        deliver_trap(
            state,
            mcause_val(exc_code::LD_MISALIGNED, false),
            va,
            &mut dummy,
        );
        return 0;
    }

    let tr = match translate_va(state, ctx, va, true, false) {
        Ok(t) => t,
        Err(TranslateFault::PageFault(cause)) => {
            let mut dummy = unsafe { std::mem::zeroed() };
            deliver_trap(state, mcause_val(cause, false), va, &mut dummy);
            return 0;
        }
        Err(_) => {
            let mut dummy = unsafe { std::mem::zeroed() };
            deliver_trap(
                state,
                mcause_val(exc_code::LD_ACCESS_FAULT, false),
                va,
                &mut dummy,
            );
            return 0;
        }
    };

    if !pmp_ok(state, tr.pa, width as u32, true, false, pmp) {
        let mut dummy = unsafe { std::mem::zeroed() };
        deliver_trap(
            state,
            mcause_val(exc_code::LD_ACCESS_FAULT, false),
            va,
            &mut dummy,
        );
        return 0;
    }

    let rs2_val = read_gpr(state, f.rs2);

    // virtio-blk inline check before generic MMIO exit.
    if let Some(_) = try_handle_virtio(tr.pa, true, rs2_val, width, dev) {
        return 4;
    }

    if is_device_addr(tr.pa, dev) {
        let info = StopInfo {
            reason: exit_reason::MMIO,
            hart_id: state.mhartid as u8,
            pc: state.pc,
            instr,
            ..StopInfo::empty()
        };
        module.request_stop(info);
        return EXIT_SENTINEL;
    }

    let advance =
        handle_amo_concurrent(state, funct5, f.func3, rs2_val, f.rd, tr.pa, ctx, pmp, dev);

    if advance == EXIT_SENTINEL {
        let info = StopInfo {
            reason: exit_reason::MMIO,
            hart_id: state.mhartid as u8,
            pc: state.pc,
            instr,
            ..StopInfo::empty()
        };
        module.request_stop(info);
    }
    advance
}
