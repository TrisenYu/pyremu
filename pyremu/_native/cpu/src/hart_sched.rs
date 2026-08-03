use std::cell::Cell;
// use std::time::Instant;
use crate::atom_instr::handle_amo_concurrent_dispatch;
use crate::concurrent::{
    ConcurrentClintCtx, FfiExtIrqCtx, ModuleState, SharedDevCtx, SharedMemCtx, SharedPmpCtx, StopInfo,
};
use crate::decode::decode_fields;
use crate::fpu::{handle_fp_load_concurrent, handle_fp_store_concurrent};
use crate::handlers::{
    handle_compressed, lr_clear_all, pmp_ok, try_handle_virtio, ClintCtx, DevCtx, PmpCtx,
    EXIT_SENTINEL,
};
use crate::interrupt::{
    check_and_deliver_interrupt_concurrent,
    clint::{sync_msip, sync_mtip, try_handle_clint_concurrent},
    wfi::{wfi_spin, TRAP_LOOP_THRESHOLD},
};
use crate::op_dispatcher::{
    handle_alu, handle_auipc, handle_br, handle_fence, handle_fp_fma, handle_fp_op, handle_jal,
    handle_jalr, handle_lui, handle_op32, handle_op_imm, handle_op_imm32,
};
use crate::peripheral::{
	is_device_addr,
	uart::try_handle_uart_concurrent,
};
use crate::state::{exit_reason, riscv_mode, BatchResult, FfiUartCtx, HartState, MemCtx};
use crate::translate::{tlb_flush_all, tlb_mark_all_dirty, translate_va, TranslateFault, WalkCtx};
use crate::trap::{
    deliver_illegal_instruction, deliver_trap, exc_code, mcause_val, priv_ecall_concurrent,
    priv_mret_concurrent, priv_sret_concurrent, priv_wfi_concurrent,
};
use std::sync::atomic::{self, AtomicU32, AtomicU64, Ordering};

pub(crate) fn ram_offset(
    pa: u64,
    size: u32,
    ram_base: u64,
    ram_size: u64,
    shadow_base: u64,
    shadow_size: u64,
) -> Option<u64> {
    let end = pa + size as u64;
    if pa >= ram_base && end <= ram_base + ram_size {
        return Some(pa - ram_base);
    }
    if shadow_size > 0 {
        let sh_end = shadow_base + shadow_size;
        if pa >= shadow_base && end <= sh_end {
            return Some(pa - shadow_base);
        }
    }
    None
}

pub(crate) fn fetch_instr(ram: *const u8, ram_size: u64, ram_base: u64, pa: u64) -> Option<u32> {
    let offset = pa.wrapping_sub(ram_base);
    if offset > ram_size.saturating_sub(4) {
        return None;
    }
    let ptr = unsafe { ram.add(offset as usize) };
    // Acquire fence pairs with Release stores in ram_write_raw so this hart
    // observes instruction bytes written by another hart (e.g. store instruction
    // writing to a page that is later made executable).
    atomic::fence(Ordering::Acquire);
    let b0 = unsafe { *ptr } as u32;
    let b1 = unsafe { *ptr.add(1) } as u32;
    let b2 = unsafe { *ptr.add(2) } as u32;
    let b3 = unsafe { *ptr.add(3) } as u32;
    Some(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24))
}

pub(crate) fn fetch_instr_safe(
    state: &mut HartState,
    mem: &MemCtx,
    pa: u64,
    pc_before: u64,
    _hart_id: u8,
) -> Option<u32> {
    match fetch_instr(mem.ram, mem.ram_size, mem.ram_base, pa) {
        Some(w) => Some(w),
        None => {
            let mut dummy = unsafe { std::mem::zeroed() };
            deliver_trap(
                state,
                mcause_val(exc_code::INSTR_ACCESS_FAULT, false),
                pc_before,
                &mut dummy,
            );
            None
        }
    }
}

pub(crate) fn sext(val: u64, bits: u32) -> u64 {
    let half = 1u64 << (bits - 1);
    (val & (half - 1)).wrapping_sub(val & half) & u64::MAX
}

pub(crate) fn read_gpr(state: &HartState, rs: u8) -> u64 {
    if rs == 0 {
        0
    } else {
        state.gprs[rs as usize]
    }
}

pub(crate) fn write_gpr(state: &mut HartState, rd: u8, val: u64) {
    if rd != 0 {
        state.gprs[rd as usize] = val;
    }
}

pub(crate) fn check_bp_hit(pc: u64, breakpoints: &[u64], fetch_pa: Option<u64>) -> bool {
    if breakpoints.is_empty() {
        return false;
    }
    // Direct VA match — covers breakpoints set on virtual addresses.
    if breakpoints.iter().any(|&bp| bp == pc) {
        return true;
    }
    // PA match — covers breakpoints set on physical addresses when
    // the hart is running with MMU enabled (pc is VA, bp value is PA).
    // Matches Python ``_bp_match_pc`` (breakpoint.py:110-118).
    if let Some(pa) = fetch_pa {
        if pa != pc {
            return breakpoints.iter().any(|&bp| bp == pa);
        }
    }
    false
}

pub(crate) fn dispatch_concurrent(
    state: &mut HartState,
    f: &crate::decode::DecodedFields,
    instr: u32,
    ctx: &WalkCtx,
    hart_id: u8,
    pmp: &PmpCtx,
    dev: &DevCtx,
    clint: &ConcurrentClintCtx,
    uart: &FfiUartCtx,
    module: &ModuleState,
) -> u64 {
    match f.opcode {
        0b01100_11 => {
            // R-type ALU — pure compute, no shared state
            let mut dummy = unsafe { std::mem::zeroed() };
            handle_alu(state, f, instr, &mut dummy)
        }
        0b00100_11 => {
            let mut dummy = unsafe { std::mem::zeroed() };
            handle_op_imm(state, f, instr, &mut dummy)
        }
        0b00110_11 => {
            let mut dummy = unsafe { std::mem::zeroed() };
            handle_op_imm32(state, f, instr, &mut dummy)
        }
        0b01110_11 => {
            let mut dummy = unsafe { std::mem::zeroed() };
            handle_op32(state, f, instr, &mut dummy)
        }
        0b01101_11 => handle_lui(state, f),
        0b00101_11 => handle_auipc(state, f),
        0b11011_11 => handle_jal(state, f),
        0b11001_11 => handle_jalr(state, f),
        0b11000_11 => {
            let mut dummy = unsafe { std::mem::zeroed() };
            handle_br(state, f, instr, &mut dummy)
        }
        0b00011_11 => {
            let mut dummy = unsafe { std::mem::zeroed() };
            handle_fence(state, f, instr, &mut dummy)
        }

        // Loads / Stores — need CLINT inline handling for concurrent path
        0b00000_11 => handle_load_concurrent(state, f, instr, ctx, pmp, dev, clint, uart, module),
        0b01000_11 => handle_store_concurrent(state, f, instr, ctx, pmp, dev, clint, uart, module),

        // FP loads / stores (FLW/FLD/FSW/FSD)
        0b00001_11 => handle_fp_load_concurrent(state, f, instr, ctx, pmp, dev, module),
        0b01001_11 => handle_fp_store_concurrent(state, f, instr, ctx, pmp, dev, module),

        // System
        0b11100_11 => handle_system_concurrent(state, f, instr, ctx, hart_id, clint, pmp, module),

        // AMO — concurrent atomic path
        0b01011_11 => handle_amo_concurrent_dispatch(state, f, instr, ctx, pmp, dev, module),

        // F/D floating point (compute — OP-FP + FMA); pure compute like ALU
        0b10100_11 => {
            let mut dummy = unsafe { std::mem::zeroed() };
            handle_fp_op(state, f, instr, &mut dummy)
        }
        0b10000_11 | 0b10001_11 | 0b10010_11 | 0b10011_11 => {
            let mut dummy = unsafe { std::mem::zeroed() };
            handle_fp_fma(state, f, instr, &mut dummy)
        }

        _ => {
            let mut dummy = unsafe { std::mem::zeroed() };
            deliver_illegal_instruction(state, instr as u64, &mut dummy);
            0
        }
    }
}

pub(crate) fn read_ram_cross_page(
    state: &mut HartState,
    ctx: &WalkCtx,
    va: u64,
    pa_first: u64,
    size: u8,
    pmp: &PmpCtx,
) -> (u64, bool) {
    let page_off = va & 0xFFF;
    let bytes_first = (0x1000 - page_off) as u8;

    // Read bytes on the first page.
    let mut result: u64 = 0;
    for b in 0..bytes_first {
        let byte = ram_read_raw(ctx, pa_first + b as u64, 1);
        result |= byte << (b * 8);
    }

    // Translate the second VA page.
    let va2 = va.wrapping_add(bytes_first as u64);
    let tr2 = match translate_va(state, ctx, va2, false, false) {
        Ok(t) => t,
        Err(TranslateFault::PageFault(cause)) => {
            let mut dummy = unsafe { std::mem::zeroed() };
            deliver_trap(state, mcause_val(cause, false), va2, &mut dummy);
            return (0, false);
        }
        Err(TranslateFault::AccessFault) => {
            let mut dummy = unsafe { std::mem::zeroed() };
            deliver_trap(
                state,
                mcause_val(exc_code::LD_ACCESS_FAULT, false),
                va2,
                &mut dummy,
            );
            return (0, false);
        }
    };

    // PMP check on the second page.
    let second_sz = (size - bytes_first) as u32;
    if !pmp_ok(state, tr2.pa, second_sz, false, false, pmp) {
        let mut dummy = unsafe { std::mem::zeroed() };
        deliver_trap(
            state,
            mcause_val(exc_code::LD_ACCESS_FAULT, false),
            va,
            &mut dummy,
        );
        return (0, false);
    }

    // Read bytes on the second page.
    for b in bytes_first..size {
        let byte = ram_read_raw(ctx, tr2.pa.wrapping_add((b - bytes_first) as u64), 1);
        result |= byte << (b * 8);
    }

    (result, true)
}

pub(crate) fn write_ram_cross_page(
    state: &mut HartState,
    ctx: &WalkCtx,
    va: u64,
    pa_first: u64,
    val: u64,
    size: u8,
    pmp: &PmpCtx,
) -> bool {
    let page_off = va & 0xFFF;
    let bytes_first = (0x1000 - page_off) as u8;

    // Write bytes on the first page.
    for b in 0..bytes_first {
        let byte = (val >> (b * 8)) as u8;
        ram_write_raw(ctx, pa_first + b as u64, byte as u64, 1);
    }

    // Translate the second VA page.
    let va2 = va.wrapping_add(bytes_first as u64);
    let tr2 = match translate_va(state, ctx, va2, true, false) {
        Ok(t) => t,
        Err(TranslateFault::PageFault(cause)) => {
            let mut dummy = unsafe { std::mem::zeroed() };
            deliver_trap(state, mcause_val(cause, false), va2, &mut dummy);
            return false;
        }
        Err(TranslateFault::AccessFault) => {
            let mut dummy = unsafe { std::mem::zeroed() };
            deliver_trap(
                state,
                mcause_val(exc_code::ST_ACCESS_FAULT, false),
                va2,
                &mut dummy,
            );
            return false;
        }
    };

    // PMP check on the second page.
    let second_sz = (size - bytes_first) as u32;
    if !pmp_ok(state, tr2.pa, second_sz, true, false, pmp) {
        let mut dummy = unsafe { std::mem::zeroed() };
        deliver_trap(
            state,
            mcause_val(exc_code::ST_ACCESS_FAULT, false),
            va,
            &mut dummy,
        );
        return false;
    }

    // Write bytes on the second page.
    for b in bytes_first..size {
        let byte = (val >> (b * 8)) as u8;
        ram_write_raw(
            ctx,
            tr2.pa.wrapping_add((b - bytes_first) as u64),
            byte as u64,
            1,
        );
    }

    true
}

pub(crate) fn handle_load_concurrent(
    state: &mut HartState,
    f: &crate::decode::DecodedFields,
    instr: u32,
    ctx: &WalkCtx,
    pmp: &PmpCtx,
    dev: &DevCtx,
    clint: &ConcurrentClintCtx,
    uart: &FfiUartCtx,
    module: &ModuleState,
) -> u64 {
    let base = read_gpr(state, f.rs1);
    let va = base.wrapping_add(f.imm12_se);
    let (size, signed) = match f.func3 {
        0b000 => (1u8, true),
        0b001 => (2, true),
        0b010 => (4, true),
        0b011 => (8, false),
        0b100 => (1, false),
        0b101 => (2, false),
        0b110 => (4, false),
        _ => {
            let mut dummy = unsafe { std::mem::zeroed() };
            deliver_illegal_instruction(state, instr as u64, &mut dummy);
            return 0;
        }
    };

    let aligned = va & (size as u64 - 1) == 0;

    let tr = match translate_va(state, ctx, va, false, false) {
        Ok(t) => t,
        Err(TranslateFault::PageFault(cause)) => {
            let mut dummy = unsafe { std::mem::zeroed() };
            deliver_trap(state, mcause_val(cause, false), va, &mut dummy);
            return 0;
        }
        Err(TranslateFault::AccessFault) => {
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

    if !pmp_ok(state, tr.pa, size as u32, false, false, pmp) {
        let mut dummy = unsafe { std::mem::zeroed() };
        deliver_trap(
            state,
            mcause_val(exc_code::LD_ACCESS_FAULT, false),
            va,
            &mut dummy,
        );
        return 0;
    }

    // CLINT inline
    if let Some(data) = try_handle_clint_concurrent(tr.pa, false, 0, state, clint, module) {
        let result_val = if signed {
            match size {
                1 => sext(data, 8),
                2 => sext(data, 16),
                4 => sext(data, 32),
                _ => data,
            }
        } else {
            data
        };
        write_gpr(state, f.rd, result_val);
        return 4;
    }

    // virtio-blk inline
    if let Some(data) = try_handle_virtio(tr.pa, false, 0, size, dev) {
        let result_val = if signed {
            match size {
                1 => sext(data, 8),
                2 => sext(data, 16),
                4 => sext(data, 32),
                _ => data,
            }
        } else {
            data
        };
        write_gpr(state, f.rd, result_val);
        return 4;
    }

    // UART inline read — prevent TX FIFO polling from killing the batch
    if let Some(data) = try_handle_uart_concurrent(tr.pa, false, 0, state.mhartid as u8, uart) {
        let result_val = if signed {
            match size {
                1 => sext(data, 8),
                2 => sext(data, 16),
                4 => sext(data, 32),
                _ => data,
            }
        } else {
            data
        };
        write_gpr(state, f.rd, result_val);
        return 4;
    }

    // MMIO
    if is_device_addr(tr.pa, dev) {
        module.request_stop(StopInfo {
            reason: exit_reason::MMIO,
            hart_id: state.mhartid as u8,
            pc: state.pc,
            instr,
            ..StopInfo::empty()
        });
        return EXIT_SENTINEL;
    }

    // Bounds check: ram_offset returns None for PA outside [ram_base, ram_base+ram_size).
    if ram_offset(
        tr.pa,
        size as u32,
        ctx.ram_base,
        ctx.ram_size,
        ctx.shadow_base,
        ctx.shadow_size,
    )
    .is_none()
    {
        let mut dummy = unsafe { std::mem::zeroed() };
        deliver_trap(
            state,
            mcause_val(exc_code::LD_ACCESS_FAULT, false),
            va,
            &mut dummy,
        );
        return 0;
    }

    // Cross-page check must come BEFORE alignment check: an 8-byte access
    // at VA 0x1FF8 is naturally aligned but crosses a 4 KiB page boundary.
    // The two VA pages may map to non-consecutive PA pages, so we must
    // translate the second page independently and splice bytes.
    let val = if (va & 0xFFF) + size as u64 > 0x1000 {
        let (v, ok) = read_ram_cross_page(state, ctx, va, tr.pa, size, pmp);
        if !ok {
            return 0;
        }
        v
    } else if aligned {
        ram_read_raw(ctx, tr.pa, size)
    } else {
        // Misaligned within same physical page: simple byte-by-byte.
        let mut result: u64 = 0;
        for b in 0..size {
            let byte = ram_read_raw(ctx, tr.pa + b as u64, 1);
            result |= byte << (b * 8);
        }
        result
    };
    let result_val = if signed {
        match size {
            1 => sext(val, 8),
            2 => sext(val, 16),
            4 => sext(val, 32),
            _ => val,
        }
    } else {
        val
    };
    write_gpr(state, f.rd, result_val);
    4
}

pub(crate) fn handle_store_concurrent(
    state: &mut HartState,
    f: &crate::decode::DecodedFields,
    instr: u32,
    ctx: &WalkCtx,
    pmp: &PmpCtx,
    dev: &DevCtx,
    clint: &ConcurrentClintCtx,
    uart: &FfiUartCtx,
    module: &ModuleState,
) -> u64 {
    let base = read_gpr(state, f.rs1);
    let va = base.wrapping_add(f.imm_s);
    let size: u8 = match f.func3 {
        0b000 => 1,
        0b001 => 2,
        0b010 => 4,
        0b011 => 8,
        _ => {
            let mut dummy = unsafe { std::mem::zeroed() };
            deliver_illegal_instruction(state, instr as u64, &mut dummy);
            return 0;
        }
    };
    // Diagnostic: log when a small value (1..0xFF) is stored — catches
    // tag values like DT_RELA=7 being written to data structures.

    let aligned = va & (size as u64 - 1) == 0;

    let tr = match translate_va(state, ctx, va, true, false) {
        Ok(t) => t,
        Err(TranslateFault::PageFault(cause)) => {
            let mut dummy = unsafe { std::mem::zeroed() };
            deliver_trap(state, mcause_val(cause, false), va, &mut dummy);
            return 0;
        }
        Err(TranslateFault::AccessFault) => {
            let mut dummy = unsafe { std::mem::zeroed() };
            deliver_trap(
                state,
                mcause_val(exc_code::ST_ACCESS_FAULT, false),
                va,
                &mut dummy,
            );
            return 0;
        }
    };

    if !pmp_ok(state, tr.pa, size as u32, true, false, pmp) {
        let mut dummy = unsafe { std::mem::zeroed() };
        deliver_trap(
            state,
            mcause_val(exc_code::ST_ACCESS_FAULT, false),
            va,
            &mut dummy,
        );
        return 0;
    }

    let val = read_gpr(state, f.rs2);

    // CLINT inline
    if let Some(_) = try_handle_clint_concurrent(tr.pa, true, val, state, clint, module) {
        return 4;
    }

    // UART inline — prevent sbi_printf from killing the batch
    if let Some(_) = try_handle_uart_concurrent(tr.pa, true, val, state.mhartid as u8, uart) {
        return 4;
    }

    // virtio-blk inline
    if let Some(_) = try_handle_virtio(tr.pa, true, val, size, dev) {
        return 4;
    }

    // MMIO
    if is_device_addr(tr.pa, dev) {
        module.request_stop(StopInfo {
            reason: exit_reason::MMIO,
            hart_id: state.mhartid as u8,
            pc: state.pc,
            instr,
            ..StopInfo::empty()
        });
        return EXIT_SENTINEL;
    }

    // Bounds check
    if ram_offset(
        tr.pa,
        size as u32,
        ctx.ram_base,
        ctx.ram_size,
        ctx.shadow_base,
        ctx.shadow_size,
    )
    .is_none()
    {
        let mut dummy = unsafe { std::mem::zeroed() };
        deliver_trap(
            state,
            mcause_val(exc_code::ST_ACCESS_FAULT, false),
            va,
            &mut dummy,
        );
        return 0;
    }

    // Cross-page check must come BEFORE alignment check (see handle_load_concurrent).
    if (va & 0xFFF) + size as u64 > 0x1000 {
        if !write_ram_cross_page(state, ctx, va, tr.pa, val, size, pmp) {
            return 0;
        }
    } else if aligned {
        ram_write_raw(ctx, tr.pa, val, size);
    } else {
        // Misaligned within same physical page: simple byte-by-byte.
        for b in 0..size {
            let byte = (val >> (b * 8)) as u8;
            ram_write_raw(ctx, tr.pa + b as u64, byte as u64, 1);
        }
    }
    4
}

pub(crate) fn ram_read_raw(ctx: &WalkCtx, pa: u64, size: u8) -> u64 {
    let off = match ram_offset(
        pa,
        size as u32,
        ctx.ram_base,
        ctx.ram_size,
        ctx.shadow_base,
        ctx.shadow_size,
    ) {
        Some(o) => o as usize,
        None => return 0,
    };
    let ptr = ctx.ram as *mut u8;
    match size {
        1 => {
            // Acquire fence before raw byte read: pairs with Release
            // fence (or atomic store) on other hart threads, ensuring
            // the byte value is visible across harts (real-hardware TSO).
            atomic::fence(Ordering::Acquire);
            unsafe { *ptr.add(off) as u64 }
        }
        2 => {
            atomic::fence(Ordering::Acquire);
            let b0 = unsafe { *ptr.add(off) } as u64;
            let b1 = unsafe { *ptr.add(off + 1) } as u64;
            b0 | (b1 << 8)
        }
        4 if (pa & 3) == 0 => {
            let a = unsafe { &*(ptr.add(off) as *const AtomicU32) };
            a.load(Ordering::Acquire) as u64
        }
        4 => {
            atomic::fence(Ordering::Acquire);
            let b0 = unsafe { *ptr.add(off) } as u64;
            let b1 = unsafe { *ptr.add(off + 1) } as u64;
            let b2 = unsafe { *ptr.add(off + 2) } as u64;
            let b3 = unsafe { *ptr.add(off + 3) } as u64;
            b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
        }
        8 if (pa & 7) == 0 => {
            let a = unsafe { &*(ptr.add(off) as *const AtomicU64) };
            let val = a.load(Ordering::Acquire);
            val
        }
        8 => {
            atomic::fence(Ordering::Acquire);
            let b0 = unsafe { *ptr.add(off) } as u64;
            let b1 = unsafe { *ptr.add(off + 1) } as u64;
            let b2 = unsafe { *ptr.add(off + 2) } as u64;
            let b3 = unsafe { *ptr.add(off + 3) } as u64;
            let b4 = unsafe { *ptr.add(off + 4) } as u64;
            let b5 = unsafe { *ptr.add(off + 5) } as u64;
            let b6 = unsafe { *ptr.add(off + 6) } as u64;
            let b7 = unsafe { *ptr.add(off + 7) } as u64;
            let val = b0 | (b1 << 8)
                | (b2 << 16)
                | (b3 << 24)
                | (b4 << 32)
                | (b5 << 40)
                | (b6 << 48)
                | (b7 << 56);
            val
        }
        _ => 0,
    }
}

pub(crate) fn ram_write_raw(ctx: &WalkCtx, pa: u64, val: u64, size: u8) {
    let off = match ram_offset(
        pa,
        size as u32,
        ctx.ram_base,
        ctx.ram_size,
        ctx.shadow_base,
        ctx.shadow_size,
    ) {
        Some(o) => o as usize,
        None => return,
    };
    // RISC-V spec §8.2: invalidate all LR reservations BEFORE the
    // store becomes visible.  With SeqCst ordering, observers see
    // the reservation clear before (or at the same time as) the new
    // value, closing the ABA window where reservation B sees old
    // value V -> store writes new value U -> store from third hart
    // writes back V -> SC on B succeeds (should have failed).
    lr_clear_all(ctx);
    atomic::fence(Ordering::SeqCst);

    let ptr = ctx.ram as *mut u8;
    match size {
        1 => {
            unsafe { *ptr.add(off) = val as u8 };
            atomic::fence(Ordering::Release);
        }
        2 => {
            unsafe { *ptr.add(off) = val as u8 };
            unsafe { *ptr.add(off + 1) = (val >> 8) as u8 };
            atomic::fence(Ordering::Release);
        }
        4 if (pa & 3) == 0 => {
            let a = unsafe { &*(ptr.add(off) as *const AtomicU32) };
            a.store(val as u32, Ordering::Release);
        }
        4 => {
            unsafe { *ptr.add(off) = val as u8 };
            unsafe { *ptr.add(off + 1) = (val >> 8) as u8 };
            unsafe { *ptr.add(off + 2) = (val >> 16) as u8 };
            unsafe { *ptr.add(off + 3) = (val >> 24) as u8 };
            atomic::fence(Ordering::Release);
        }
        8 if (pa & 7) == 0 => {
            let a = unsafe { &*(ptr.add(off) as *const AtomicU64) };
            a.store(val, Ordering::Release);
        }
        8 => {
            unsafe { *ptr.add(off) = val as u8 };
            unsafe { *ptr.add(off + 1) = (val >> 8) as u8 };
            unsafe { *ptr.add(off + 2) = (val >> 16) as u8 };
            unsafe { *ptr.add(off + 3) = (val >> 24) as u8 };
            unsafe { *ptr.add(off + 4) = (val >> 32) as u8 };
            unsafe { *ptr.add(off + 5) = (val >> 40) as u8 };
            unsafe { *ptr.add(off + 6) = (val >> 48) as u8 };
            unsafe { *ptr.add(off + 7) = (val >> 56) as u8 };
            atomic::fence(Ordering::Release);
        }
        _ => {}
    }
}

/// Handle CSR instructions (func3 ≠ 0b000) within the concurrent batch engine.
///
/// Builds a temporary serial ``ClintCtx`` from the concurrent one so that the
/// shared ``csr::handle_csr`` path remains unchanged.  Also syncs stimecmp to
/// CLINT mtimecmp after CSR writes for Sstc correctness.
fn handle_csr_concurrent(
    state: &mut HartState,
    f: &crate::decode::DecodedFields,
    instr: u32,
    hart_id: u8,
    clint: &ConcurrentClintCtx,
    pmp: &PmpCtx,
) -> u64 {
    let cur_mtime = unsafe { &*clint.mtime }.load(Ordering::Relaxed);
    let hid = state.mhartid as usize;

    // Build a temporary serial ClintCtx for the CSR handler
    let _serial_clint = ClintCtx {
        base: clint.base,
        mtime: clint.mtime as *mut u64,
        mtimecmp: clint.mtimecmp as *mut u64,
        msip: clint.msip as *mut u8,
        states: std::ptr::null_mut(),
        num_harts: clint.num_harts,
        yield_for_ipi: Cell::new(false),
        ipi_sender_hart: Cell::new(0),
        ipi_sender_rounds: Cell::new(0),
    };

    let mut dummy = unsafe { std::mem::zeroed() };
    let advance = crate::csr::handle_csr(
        state, f.rd, f.rs1, f.func12, f.func3, instr, &mut dummy, hart_id, cur_mtime, pmp,
    );

    // Sync stimecmp -> CLINT mtimecmp after CSR write to stimecmp (Sstc).
    if f.func12 == 0x14D && hid < clint.num_harts as usize {
        unsafe { &*clint.mtimecmp.add(hid) }.store(state.stimecmp, Ordering::Release);
    }

    advance
}

pub(crate) fn handle_system_concurrent(
    state: &mut HartState,
    f: &crate::decode::DecodedFields,
    instr: u32,
    _ctx: &WalkCtx,
    hart_id: u8,
    clint: &ConcurrentClintCtx,
    pmp: &PmpCtx,
    module: &ModuleState,
) -> u64 {
    if f.func3 != 0b000 {
        return handle_csr_concurrent(state, f, instr, hart_id, clint, pmp);
    }

    // func3 == 0b000: privileged instructions
    match f.func12 {
        0 => return priv_ecall_concurrent(state, instr, clint, module),
        1 => {
            // EBREAK — semihosting or NOP.
            if let Some(advance) = crate::handlers::try_semihosting(state, _ctx, state.pc) {
                return advance;
            }
            return 4;
        }
        0x302 => return priv_mret_concurrent(state, instr),
        0x102 => return priv_sret_concurrent(state, instr, _ctx),
        0x105 => return priv_wfi_concurrent(state, instr),
        f_val if (f_val >> 5) == 0x09 => {
            // SFENCE.VMA — flush local TLB + broadcast to all harts.
            //
            // RISC-V SFENCE.VMA only flushes the local hart's TLB; remote
            // shootdown requires an IPI.  We implement broadcast semantics
            // via a global generation counter: every hart that detects a
            // generation mismatch flushes its own TLB at its next instruction
            // boundary.  This is legal because over-invalidation never breaks
            // correctness, and it closes the coherency window where hart A
            // writes a PTE + SFENCE.VMA but hart B's TLB still has the stale
            // entry.
            //
            // Ordering: flush the local TLB first, then increment the global
            // generation (Release).  On x86-64, ``fetch_add`` with Release is
            // ``lock xadd`` — a full hardware barrier that makes all prior
            // PTE stores globally visible before the generation change is
            // observed by other harts.  Paired with the Acquire load in the
            // Mark this hart's own TLB entries dirty; other harts
            // will detect the gen change and mark their own entries
            // dirty at the next instruction boundary.
            tlb_flush_all(&mut state.itlb);
            tlb_flush_all(&mut state.dtlb);
            let new_gen = module
                .tlb_gen
                .fetch_add(1, std::sync::atomic::Ordering::Release)
                .wrapping_add(1);
            module.tlb_gen_per_hart[hart_id as usize]
                .store(new_gen, std::sync::atomic::Ordering::Relaxed);
            return 4;
        }
        0x5A0 => {
            // MFENCE.DID
            for e in state.itlb.iter_mut() {
                if e.mdid == state.mdid {
                    e.valid = 0;
                }
            }
            for e in state.dtlb.iter_mut() {
                if e.mdid == state.mdid {
                    e.valid = 0;
                }
            }
            return 4;
        }
        _ => {
            let mut dummy = unsafe { std::mem::zeroed() };
            deliver_illegal_instruction(state, instr as u64, &mut dummy);
            return 0;
        }
    }
}

pub(crate) fn exec_compressed_concurrent(
    state: &mut HartState,
    instr_word: u32,
    pc_before: u64,
    hart_id: u8,
    ctx: &WalkCtx,
    pmp: &PmpCtx,
    dev: &DevCtx,
    clint: &ConcurrentClintCtx,
    module: &ModuleState,
    breakpoints: &[u64],
) -> bool {
    let half = (instr_word & 0xFFFF) as u16;
    let _serial_clint = ClintCtx {
        base: clint.base,
        mtime: clint.mtime as *mut u64,
        mtimecmp: clint.mtimecmp as *mut u64,
        msip: clint.msip as *mut u8,
        states: std::ptr::null_mut(),
        num_harts: clint.num_harts,
        yield_for_ipi: Cell::new(false),
        ipi_sender_hart: Cell::new(0),
        ipi_sender_rounds: Cell::new(0),
    };
    let mut dummy = unsafe { std::mem::zeroed() };
    let advance = handle_compressed(
        state,
        half,
        instr_word,
        &mut dummy,
        ctx,
        pmp,
        dev,
        &_serial_clint,
    );

    if advance == EXIT_SENTINEL {
        let reason = if dummy.exit_reason != exit_reason::NORMAL {
            dummy.exit_reason
        } else {
            exit_reason::ECALL
        };
        module.request_stop(StopInfo {
            reason,
            hart_id,
            pc: pc_before,
            instr: instr_word,
            ..StopInfo::empty()
        });
        return false;
    }

    if advance == 0 && state.pc == pc_before {
        state.consecutive_traps = state.consecutive_traps.saturating_add(1);
    } else if advance != 0 && state.pc == pc_before {
        state.pc = state.pc.wrapping_add(advance);
        state.consecutive_traps = 0;
    }

    post_instr_checks(state, hart_id, clint, module, breakpoints)
}

pub(crate) fn post_instr_checks(
    state: &mut HartState,
    hart_id: u8,
    clint: &ConcurrentClintCtx,
    module: &ModuleState,
    breakpoints: &[u64],
) -> bool {
    // Per-instruction mtime increment causes timer-interrupt storms in
    // multi-core mode: N harts each increment the shared mtime on every
    // instruction ->mtime advances N× too fast ->the kernel's next-tick
    // deadline is already in the past by the time the handler returns ->
    // immediate re-trigger ->hart spends 1B+ instructions spinning in
    // the timer handler (riscv_clocksource_rdtime ->ktime_get ->
    // tick_nohz_handler …).
    //
    // Fix: throttle mtime to advance every MTIME_DIVISOR instructions
    // AND divide the increment by the number of harts.  With 4 harts
    // each adding 64 ticks / 256 instrs, the *total* mtime rate stays
    // at 1 tick / instr regardless of hart count.
    const MTIME_DIVISOR: u64 = 256;
    const HZ_RATIO: u64 = 1; // CPU freq ≈ HZ_RATIO × timer freq (10 MHz)
                             // Reduced from 100: the emulator runs at ~1.4 MHz, not 1 GHz.
                             // HZ_RATIO=100 made mtime advance at 0.1% real speed, stretching
                             // a 10s deferred_probe_timeout to ~3 hours.  1 gives ~14% speed.
    state.total_instrs = state.total_instrs.wrapping_add(1);
    if state.total_instrs & (MTIME_DIVISOR - 1) == 0 {
        let n = core::cmp::max(clint.num_harts as u64, 1) * HZ_RATIO;
        let inc = core::cmp::max(MTIME_DIVISOR / n, 1);
        unsafe { &*clint.mtime }.fetch_add(inc, Ordering::Relaxed);
    }

    if check_bp_hit(state.pc, breakpoints, None) {
        module.request_stop(StopInfo {
            reason: exit_reason::BREAKPOINT,
            hart_id,
            pc: state.pc,
            ..StopInfo::empty()
        });
        return false;
    }

    if state.consecutive_traps >= TRAP_LOOP_THRESHOLD {
        state.halted = 1;
        module.request_stop(StopInfo {
            reason: exit_reason::TRAP,
            hart_id,
            pc: state.pc,
            ..StopInfo::empty()
        });
        return false;
    }

    true
}

pub(crate) fn translate_fetch_pc_concurrent(
    state: &mut HartState,
    ctx: &WalkCtx,
    pc: u64,
) -> Option<u64> {
    if state.mmu_mode == 0 || state.mode == riscv_mode::M || state.mode == riscv_mode::D {
        return Some(pc); // Bare mode / M-mode / D-mode: VA == PA
    }
    match translate_va(state, ctx, pc, false, true) {
        Ok(t) => Some(t.pa),
        Err(TranslateFault::PageFault(c)) => {
            let mut dummy: BatchResult = unsafe { std::mem::zeroed() };
            deliver_trap(state, mcause_val(c, false), pc, &mut dummy);
            None
        }
        Err(TranslateFault::AccessFault) => {
            let mut dummy: BatchResult = unsafe { std::mem::zeroed() };
            deliver_trap(
                state,
                mcause_val(exc_code::INSTR_ACCESS_FAULT, false),
                pc,
                &mut dummy,
            );
            None
        }
    }
}

/// Fetch a 32-bit instruction word from physical memory.
///
/// When PC is within the last 2 bytes of a 4 KiB page (`page_offset >= 0xFFE`),
/// the 4-byte fetch crosses into the next virtual page.  The two pages may be
/// mapped to *non-consecutive* physical pages, so we translate the second
/// page independently and splice 2 bytes from each.
///
/// Returns `Some(word)` on success, or `None` if a trap was delivered (the
/// caller must handle the post-instruction boilerplate).
#[inline]
fn fetch_instr_word(
    state: &mut HartState,
    ctx: &WalkCtx,
    mem: &MemCtx,
    pc_before: u64,
    page_offset: u64,
    fetch_pa: u64,
    pmp: &PmpCtx,
    hart_id: u8,
) -> Option<u32> {
    if page_offset < 0xFFE {
        return fetch_instr_safe(state, mem, fetch_pa, pc_before, hart_id);
    }
    // 2 bytes on first page, 2 bytes on second page.
    let fetch_pa2 = translate_fetch_pc_concurrent(state, ctx, pc_before.wrapping_add(2))?;
    // PMP check on the second page as well.
    if !pmp_ok(state, fetch_pa2, 4, false, true, pmp) {
        let mut dummy = unsafe { std::mem::zeroed() };
        deliver_trap(
            state,
            mcause_val(exc_code::INSTR_ACCESS_FAULT, false),
            pc_before,
            &mut dummy,
        );
        return None;
    }
    // Read 2 bytes from each physical page and combine (little-endian).
    let lo = ram_read_raw(ctx, fetch_pa, 2) as u32;
    let hi = ram_read_raw(ctx, fetch_pa2, 2) as u32;
    return Some(lo | (hi << 16));
}

// ============================================================
//  hart_worker pipeline helpers
// ============================================================

/// Main-loop control flow sentinel.
enum Step {
    /// Advance to the next pipeline stage.
    Next,
    /// Jump back to the top of the main loop.
    Continue,
    /// Exit the worker function immediately.
    Exit,
}

/// Reconstruct local context structs from the `Send`-safe FFI wrappers.
fn make_contexts(
    mem: SharedMemCtx,
    pmp: SharedPmpCtx,
    dev: SharedDevCtx,
    module: &ModuleState,
) -> (MemCtx, PmpCtx, DevCtx, WalkCtx) {
    let mem_val = MemCtx {
        ram: mem.ram,
        ram_size: mem.ram_size,
        ram_base: mem.ram_base,
        shadow_base: mem.shadow_base,
        shadow_size: mem.shadow_size,
    };
    let pmp_val = PmpCtx {
        cfg: pmp.cfg,
        addr: pmp.addr,
        num: pmp.num,
    };
    let dev_val = DevCtx {
        bases: dev.bases,
        ends: dev.ends,
        num: dev.num,
        virtio_base: dev.virtio_base,
        virtio_raw: dev.virtio_raw,
    };
    let walk = WalkCtx {
        ram: mem_val.ram,
        ram_size: mem_val.ram_size,
        ram_base: mem_val.ram_base,
        shadow_base: mem_val.shadow_base,
        shadow_size: mem_val.shadow_size,
        tlb_gen: &module.tlb_gen as *const AtomicU64,
        itlb_hand: Cell::new(0),
        dtlb_hand: Cell::new(0),
        lr_reserved: mem.lr_reserved,
        num_harts: module.wfi_flags.len() as u32,
    };
    (mem_val, pmp_val, dev_val, walk)
}

/// Mark this hart's TLB entries dirty if another hart executed SFENCE.VMA.
/// Dirty entries are re-walked on next access rather than flushed — this
/// preserves cached translations that are still valid.
fn mark_tlb_dirty_if_stale(state: &mut HartState, module: &ModuleState, hart_id: u8) {
    let global_gen = module.tlb_gen.load(Ordering::Acquire);
    let my_gen = module.tlb_gen_per_hart[hart_id as usize].load(Ordering::Relaxed);
    if my_gen == global_gen {
        return;
    }
    tlb_mark_all_dirty(&mut state.itlb);
    tlb_mark_all_dirty(&mut state.dtlb);
    module.tlb_gen_per_hart[hart_id as usize].store(global_gen, Ordering::Relaxed);
}

/// Handle WFI wait / wake.  Returns ``(Step, just_woke)``.
fn handle_wfi_state(
    state: &mut HartState,
    hart_id: u8,
    clint: &ConcurrentClintCtx,
    dev: &DevCtx,
    module: &ModuleState,
    stop_flag: *const u8,
    uart: &FfiUartCtx,
    ext_irq: *mut FfiExtIrqCtx,
) -> (Step, bool) {
    if state.waiting == 0 {
        return (Step::Next, false);
    }
    // Deferred Python-side work (e.g. virtio QueueNotify): exit batch so
    // Python gets a chance to process I/O before we re-enter WFI spin.
    if dev.has_pending_python_work() {
        module.request_stop(StopInfo {
            reason: exit_reason::WFI_WAIT,
            hart_id,
            pc: state.pc,
            ..StopInfo::empty()
        });
        return (Step::Exit, false);
    }
    if !wfi_spin(state, hart_id as usize, clint, module, stop_flag, uart.rx_notify, ext_irq) {
        // Batch exit from WFI: flush TLB in case another hart did
        // SFENCE.VMA while we were spinning.
        mark_tlb_dirty_if_stale(state, module, hart_id);
        return (Step::Exit, false);
    }
    state.consecutive_traps = 0;
    (Step::Next, true)
}

/// Synchronise hardware interrupt lines and deliver the highest-priority
/// pending interrupt.  Returns ``true`` when a trap was delivered (caller
/// should ``continue`` the main loop).
fn step_interrupts(
    state: &mut HartState,
    hart_id: u8,
    clint: &ConcurrentClintCtx,
    _just_woke_from_wfi: bool,
) -> bool {
    let _ = hart_id; // used only in #[cfg(feature = "diagnostic")] block
    sync_mtip(state, clint);
    // Always sync MSIP — even when just woke from WFI.
    // The WFI wake's wfi_sync_and_check already called sync_msip and
    // auto-cleared the CLINT level bit, but the initiating hart may
    // send additional MSIPs in a tight loop (e.g. tlb_sync re-sending
    // while spinning).  Skipping sync_msip here would miss those
    // subsequent edges and leave the target hart parked in WFI despite
    // pending IPIs -> TLB-shootdown deadlock.
    sync_msip(state, clint);
    let msip_was_pending = (state.mip & (1 << 3)) != 0;
    if msip_was_pending && (state.mie & (1 << 3)) == 0 {
        state.mie |= 1 << 3;
    }
    if check_and_deliver_interrupt_concurrent(state, clint) {
        return true;
    }
    false
}

/// Instruction fetch pipeline: translate VA, PMP execute check, read RAM,
/// breakpoint match.  Returns ``Ok(pc_before, instr_word)`` on success or
/// ``Err(step)`` when a trap was delivered / bp hit.
fn step_fetch_instr(
    state: &mut HartState,
    hart_id: u8,
    ctx: &WalkCtx,
    pmp: &PmpCtx,
    mem: &MemCtx,
    clint: &ConcurrentClintCtx,
    module: &ModuleState,
    breakpoints: &[u64],
    instr_count: &mut u64,
) -> Result<(u64, u32), Step> {
    let pc_before = state.pc;
    let page_offset = pc_before & 0xFFF;

    // -- VA -> PA translation --
    let fetch_pa: u64 = match translate_fetch_pc_concurrent(state, ctx, pc_before) {
        Some(pa) => pa,
        None => {
            if state.pc == pc_before {
                state.consecutive_traps = state.consecutive_traps.saturating_add(1);
            }
            *instr_count += 1;
            let step = if post_instr_checks(state, hart_id, clint, module, breakpoints) {
                Step::Continue
            } else {
                Step::Exit
            };
            return Err(step);
        }
    };

    // -- PMP execute check --
    if !pmp_ok(state, fetch_pa, 4, false, true, pmp) {
        let mut dummy = unsafe { std::mem::zeroed() };
        deliver_trap(
            state,
            mcause_val(exc_code::INSTR_ACCESS_FAULT, false),
            pc_before,
            &mut dummy,
        );
        if state.pc == pc_before {
            state.consecutive_traps = state.consecutive_traps.saturating_add(1);
        }
        *instr_count += 1;
        let step = if post_instr_checks(state, hart_id, clint, module, breakpoints) {
            Step::Continue
        } else {
            Step::Exit
        };
        return Err(step);
    }

    // -- Read instruction word (cross-page aware) --
    let instr_word = match fetch_instr_word(
        state,
        ctx,
        mem,
        pc_before,
        page_offset,
        fetch_pa,
        pmp,
        hart_id,
    ) {
        Some(w) => w,
        None => {
            if state.pc == pc_before {
                state.consecutive_traps = state.consecutive_traps.saturating_add(1);
            }
            *instr_count += 1;
            let step = if post_instr_checks(state, hart_id, clint, module, breakpoints) {
                Step::Continue
            } else {
                Step::Exit
            };
            return Err(step);
        }
    };

    // -- Breakpoint --
    if check_bp_hit(pc_before, breakpoints, Some(fetch_pa)) {
        module.request_stop(StopInfo {
            reason: exit_reason::BREAKPOINT,
            hart_id,
            pc: pc_before,
            instr: instr_word,
            ..StopInfo::empty()
        });
        return Err(Step::Exit);
    }

    Ok((pc_before, instr_word))
}

// ============================================================
//  Main per-hart worker
// ============================================================

pub(crate) fn hart_worker(
    state: &mut HartState,
    hart_id: u8,
    mem: SharedMemCtx,
    pmp: SharedPmpCtx,
    dev: SharedDevCtx,
    clint: &ConcurrentClintCtx,
    uart: &FfiUartCtx,
    module: &ModuleState,
    breakpoints: &[u64],
    _max_instrs_per_hart: u64,
    stop_flag: *const u8,
    ext_irq: *mut FfiExtIrqCtx,
) {
    let (_mem_val, _pmp_val, _dev_val, ctx) = make_contexts(mem, pmp, dev, module);
    let mem = &_mem_val;
    let pmp = &_pmp_val;
    let dev = &_dev_val;

    // HartState persists across FFI calls — stale TLB entries from a
    // previous batch with epoch=0 would match the fresh ModuleState's
    // tlb_gen=0, producing incorrect VA->PA hits and memory corruption
    // (garbage inode metadata -> "Permission denied" / ENOTDIR in ext4).
    tlb_flush_all(&mut state.itlb);
    tlb_flush_all(&mut state.dtlb);

    let mut instr_count: u64 = 0;
    loop {
        // ---- Guards ----
        if module.stop_flag.load(Ordering::Acquire) {
            return;
        }
        if !stop_flag.is_null() && unsafe { *stop_flag != 0 } {
            return;
        }
        // External interrupt (UART, VirtIO, …): daemon set pending after
        // injecting data into UART RX FIFO.  Raise SEIP/MEIP inline so the
        // guest's trap handler processes the interrupt within this batch.
        // The batch exits naturally when the guest's PLIC driver reads
        // claim/complete MMIO registers — no forced exit needed.
        if !ext_irq.is_null() && unsafe { (*ext_irq).pending != 0 } {
            // Level-triggered: Python sets pending=1 when ring buffer or
            // UART FIFO has data, clears to 0 only when both are empty.
            // We do NOT clear pending here — Python manages the lifecycle.
            if state.mie & (1 << 9) != 0 {
                state.mip |= 1 << 9;  // SEIP
            }
            if state.mie & (1 << 11) != 0 {
                state.mip |= 1 << 11; // MEIP
            }
        }
        // TermIO RX notification: stdin bytes arrived in ring buffer.
        // Force immediate batch exit -> Python drain_rx() moves data
        // from ring buffer into UART FIFO -> _native_sync_plic_mip
        // raises SEIP -> guest reads data on next batch without
        // waiting for PLIC round-trip or batch completion.
        if !uart.rx_notify.is_null() && unsafe { *uart.rx_notify != 0 } {
            module.stop_flag.store(true, Ordering::Release);
            return;
        }
        if state.halted != 0 {
            std::hint::spin_loop();
            continue;
        }

        // ---- WFI ----
        let (wfi_step, just_woke_from_wfi) =
            handle_wfi_state(state, hart_id, clint, dev, module, stop_flag, uart, ext_irq);
        match wfi_step {
            Step::Exit => return,
            Step::Continue => continue,
            Step::Next => {}
        }

        // ---- Interrupts ----
        if step_interrupts(state, hart_id, clint, just_woke_from_wfi) {
            continue;
        }

        // ---- TLB coherency (broadcast SFENCE.VMA) ----
        mark_tlb_dirty_if_stale(state, module, hart_id);

        // ---- Fetch ----
        let (pc_before, instr_word) = match step_fetch_instr(
            state,
            hart_id,
            &ctx,
            pmp,
            mem,
            clint,
            module,
            breakpoints,
            &mut instr_count,
        ) {
            Ok(v) => v,
            Err(Step::Continue) => continue,
            Err(Step::Exit) => return,
            Err(Step::Next) => unreachable!(),
        };

        // ---- Decode & execute ----
        let f = decode_fields(instr_word);
        if f.is_compressed != 0 {
            if !exec_compressed_concurrent(
                state,
                instr_word,
                pc_before,
                hart_id,
                &ctx,
                pmp,
                dev,
                clint,
                module,
                breakpoints,
            ) {
                return;
            }
            instr_count += 1;
            continue;
        }
        let advance = dispatch_concurrent(
            state, &f, instr_word, &ctx, hart_id, pmp, dev, clint, uart, module,
        );
        if advance == EXIT_SENTINEL {
            if module.stop_flag.load(Ordering::Acquire) {
               return;
            }
			module.request_stop(StopInfo {
				reason: exit_reason::MMIO,
				hart_id,
				pc: pc_before,
				instr: instr_word,
				..StopInfo::empty()
			});
            return;
        }

        if advance == 0 && state.pc == pc_before {
            state.consecutive_traps = state.consecutive_traps.saturating_add(1);
        } else if advance != 0 && state.pc == pc_before {
            state.pc = state.pc.wrapping_add(advance);
            state.consecutive_traps = 0;
        }

        if !post_instr_checks(state, hart_id, clint, module, breakpoints) {
            return;
        }
        instr_count += 1;
    }
}

// ============================================================
//  Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::concurrent::{ConcurrentClintCtx, ModuleState, StopInfo};
    use crate::handlers::{DevCtx, PmpCtx};
    use crate::state::{riscv_mode, FfiUartCtx, HartState, TlbEntry};
    use crate::translate::WalkCtx;

    /// Build a permissive PMP context: single TOR entry covering all of
    /// memory with R/W/X.  Needed because pmp_ok per RISC-V spec §3.7.1
    /// denies S/U access when num_entries==0.
    fn make_permissive_pmp(cfg: &mut [u8], addr: &mut [u64]) -> PmpCtx {
        // TOR entry: covers [0, addr[0]) = [0, u64::MAX) — entire address space.
        // PMP_R(1) | PMP_W(2) | PMP_X(4) | PMP_A_TOR(8)
        cfg[0] = 0x0F;
        addr[0] = u64::MAX;
        PmpCtx { cfg: cfg.as_mut_ptr(), addr: addr.as_mut_ptr(), num: 1 }
    }

    /// Build a minimal 4K-page Sv39 page table inside RAM.
    /// Places L2 at PA 0x5000, L1 at 0x6000, L0 at 0x7000.
    /// Returns the `satp` value (Sv39 mode, root PPN = 5).
    fn setup_two_pages(
        ram: &mut [u8],
        va_a: u64,
        pa_a: u64,
        va_b: u64,
        pa_b: u64,
        perm: u64,
    ) -> u64 {
        let vpn_a = crate::mmu::sv39_decompose_va(va_a);
        let vpn_b = crate::mmu::sv39_decompose_va(va_b);
        // Both pages must share the same L2/L1 path (same 2 MiB region).
        assert_eq!(vpn_a.vpn2, vpn_b.vpn2);
        assert_eq!(vpn_a.vpn1, vpn_b.vpn1);

        let satp = 8u64 << 60 | 5; // Sv39, root PPN = 5

        // L2 (PA 0x5000): pointer to L1 at PPN=6
        let l2_off = 0x5000usize + (vpn_a.vpn2 * 8) as usize;
        let l2_val = (6u64 << 10) | crate::translate::PTE_V;
        ram[l2_off..l2_off + 8].copy_from_slice(&l2_val.to_le_bytes());

        // L1 (PA 0x6000): pointer to L0 at PPN=7
        let l1_off = 0x6000usize + (vpn_a.vpn1 * 8) as usize;
        let l1_val = (7u64 << 10) | crate::translate::PTE_V;
        ram[l1_off..l1_off + 8].copy_from_slice(&l1_val.to_le_bytes());

        // L0 (PA 0x7000): two leaf entries
        let ppn_a = pa_a >> 12;
        let l0_off_a = 0x7000usize + (vpn_a.vpn0 * 8) as usize;
        let l0_val_a = (ppn_a << 10) | perm | crate::translate::PTE_V;
        ram[l0_off_a..l0_off_a + 8].copy_from_slice(&l0_val_a.to_le_bytes());

        let ppn_b = pa_b >> 12;
        let l0_off_b = 0x7000usize + (vpn_b.vpn0 * 8) as usize;
        let l0_val_b = (ppn_b << 10) | perm | crate::translate::PTE_V;
        ram[l0_off_b..l0_off_b + 8].copy_from_slice(&l0_val_b.to_le_bytes());

        satp
    }

    /// Regression: 8-byte read crossing a 4 KiB page boundary must splice
    /// bytes from two independently-translated PA pages, even when the
    /// VA is naturally aligned (the aligned path must NOT be taken for
    /// cross-page accesses).
    ///
    /// Maps VA 0x1000->PA 0x1000 and VA 0x2000->PA 0x3000 (non-consecutive,
    /// skipping PA 0x2000).  A misaligned 8-byte read at VA 0x1FFC (aligned
    /// to 4 but not to 8, crossing into VA 0x2000) must return the bytes from
    /// PA 0x1FFC-0x1FFF concatenated with PA 0x3000-0x3003.
    #[test]
    fn cross_page_read_splices_non_consecutive_pa() {
        const RAM_SIZE: usize = 0x8000;
        let mut ram = vec![0u8; RAM_SIZE];

        // Page A (VA 0x1000): PA 0x1000
        // Page B (VA 0x2000): PA 0x3000 (non-consecutive — gap at PA 0x2000)
        let satp = setup_two_pages(
            &mut ram,
            0x1000,
            0x1000,
            0x2000,
            0x3000,
            crate::translate::PTE_R | crate::translate::PTE_W | crate::translate::PTE_U,
        );

        // Write pattern: 0xAA bytes at end of first PA page, 0xBB at start of second.
        // VA 0x1FFC ->PA 0x1FFC (bytes_first = 4, on page A)
        // VA 0x2000 ->PA 0x3000 (bytes 4-7 of the 8-byte read, on page B)
        for i in 0x1FFC..0x2000 {
            ram[i] = 0xAA;
        }
        for i in 0x3000..0x3004 {
            ram[i] = 0xBB;
        }

        let mut state: HartState = unsafe { std::mem::zeroed() };
        state.mode = riscv_mode::U; // U-mode ->page walk + PMP bypass (num=0)
        state.mmu_mode = 8; // Sv39
        state.satp = satp;
        // Init dtlb as invalid to force page walk.
        for e in state.dtlb.iter_mut() {
            *e = TlbEntry::empty();
        }

        let ctx = WalkCtx {
            ram: ram.as_mut_ptr(),
            ram_size: RAM_SIZE as u64,
            ram_base: 0,
            shadow_base: 0,
            shadow_size: 0,
            tlb_gen: std::ptr::null(),
            itlb_hand: Cell::new(0),
            dtlb_hand: Cell::new(0),
            lr_reserved: std::ptr::null_mut(),
            num_harts: 1,
        };
        // PMP with zero entries — no restrictions (pmp_ok short-circuits on num==0).
        let mut _pmp_cfg = vec![0u8; 64];
        let mut _pmp_addr = vec![0u64; 64];
        let pmp = make_permissive_pmp(&mut _pmp_cfg, &mut _pmp_addr);

        // 8-byte read at VA 0x1FFC — crosses page boundary.
        let (val, ok) = read_ram_cross_page(&mut state, &ctx, 0x1FFC, 0x1FFC, 8, &pmp);
        assert!(ok, "cross-page read must succeed");

        // Expected: little-endian assembly of 4 bytes 0xAA + 4 bytes 0xBB.
        let expected: u64 = 0xBBBB_BBBB_AAAA_AAAA;
        assert_eq!(
            val, expected,
            "cross-page read: expected 0x{expected:016X}, got 0x{val:016X}\
             \n  (4 bytes from PA 0x1FFC + 4 bytes from PA 0x3000)"
        );
    }

    /// Store variant: 8-byte write crossing a page boundary must scatter
    /// bytes to two independently-translated PA pages.
    #[test]
    fn cross_page_write_scatters_non_consecutive_pa() {
        const RAM_SIZE: usize = 0x8000;
        let mut ram = vec![0u8; RAM_SIZE];

        let satp = setup_two_pages(
            &mut ram,
            0x1000,
            0x1000,
            0x2000,
            0x3000,
            crate::translate::PTE_R | crate::translate::PTE_W | crate::translate::PTE_U,
        );

        let mut state: HartState = unsafe { std::mem::zeroed() };
        state.mode = riscv_mode::U;
        state.mmu_mode = 8;
        state.satp = satp;
        for e in state.dtlb.iter_mut() {
            *e = TlbEntry::empty();
        }

        let ctx = WalkCtx {
            ram: ram.as_mut_ptr(),
            ram_size: RAM_SIZE as u64,
            ram_base: 0,
            shadow_base: 0,
            shadow_size: 0,
            tlb_gen: std::ptr::null(),
            itlb_hand: Cell::new(0),
            dtlb_hand: Cell::new(0),
            lr_reserved: std::ptr::null_mut(),
            num_harts: 1,
        };
        let mut _pmp_cfg = vec![0u8; 64];
        let mut _pmp_addr = vec![0u64; 64];
        let pmp = make_permissive_pmp(&mut _pmp_cfg, &mut _pmp_addr);

        // Write 0xCCCCCCCC_DDDDDDDD at VA 0x1FFC (4 bytes to each page).
        let val: u64 = 0xCCCC_CCCC_DDDD_DDDD;
        let ok = write_ram_cross_page(&mut state, &ctx, 0x1FFC, 0x1FFC, val, 8, &pmp);
        assert!(ok, "cross-page write must succeed");

        // Lower 4 bytes (0xDDDDDDDD) go to PA 0x1FFC-0x1FFF (page A).
        assert_eq!(&ram[0x1FFC..0x2000], &[0xDD, 0xDD, 0xDD, 0xDD]);
        // Upper 4 bytes (0xCCCCCCCC) go to PA 0x3000-0x3003 (page B, non-consecutive).
        assert_eq!(&ram[0x3000..0x3004], &[0xCC, 0xCC, 0xCC, 0xCC]);
        // PA 0x2000-0x2FFF must remain untouched.
        assert!(
            ram[0x2000..0x3000].iter().all(|&b| b == 0),
            "gap between mapped pages must not be written"
        );
    }

    // ============================================================
    //  rd == rs1  regression tests
    // ============================================================

    /// A load where rd==rs1 must read the base register BEFORE writing the
    /// loaded value.  Regression: ``lhu a5, 0x336(a5)`` crashes at VA 0x33D
    /// when a5=7 because the old value is lost before address computation.
    #[test]
    fn load_rd_equals_rs1_uses_old_value_for_address() {
        let mut state: HartState = unsafe { std::mem::zeroed() };
        state.mode = riscv_mode::M;
        state.gprs[15] = 0x100; // a5 = low address within RAM
        state.mmu_mode = 0; // Bare mode — VA==PA

        let mut ram = vec![0xCDu8; 0x1000];
        let ctx = WalkCtx {
            ram: ram.as_mut_ptr(),
            ram_size: 0x1000,
            ram_base: 0,
            shadow_base: 0,
            shadow_size: 0,
            tlb_gen: std::ptr::null(),
            itlb_hand: Cell::new(0),
            dtlb_hand: Cell::new(0),
            lr_reserved: std::ptr::null_mut(),
            num_harts: 1,
        };
        let mut _pmp_cfg = vec![0u8; 1];
        let mut _pmp_addr = vec![0u64; 1];
        let pmp = make_permissive_pmp(&mut _pmp_cfg, &mut _pmp_addr);
        let dev = DevCtx {
            bases: std::ptr::null(),
            ends: std::ptr::null(),
            num: 0,
            virtio_base: 0,
            virtio_raw: std::ptr::null_mut(),
        };
        // Build minimal inline CLINT/UART contexts for the load handler.
        let _mtime = std::sync::atomic::AtomicU64::new(0);
        let _mtimecmp = std::sync::atomic::AtomicU64::new(0);
        let _msip = std::sync::atomic::AtomicU8::new(0);
        let clint = ConcurrentClintCtx {
            base: 0,
            mtime: &_mtime,
            mtimecmp: &_mtimecmp,
            msip: &_msip,
            num_harts: 1,
            msip_pending: Cell::new(std::ptr::null()),
            hart_threads: Cell::new(std::ptr::null()),
        };
        let _tx_buf = [0u8; 16];
        let _tx_wr = std::sync::atomic::AtomicU32::new(0);
        let uart = FfiUartCtx {
            base: 0,
            tx_buf: _tx_buf.as_ptr() as *mut u8,
            tx_cap: 16,
            tx_wr: &_tx_wr as *const std::sync::atomic::AtomicU32 as *mut u32,
            ie: 0,
            txctrl: 0,
            rxctrl: 0,
            rx_fifo_len: 0,
            tx_notify_fd: -1,
            no_stdout: 0, rx_notify: std::ptr::null_mut(),
        };
        let _wfi_flags: Vec<std::sync::atomic::AtomicU8> = (0..1)
            .map(|_| std::sync::atomic::AtomicU8::new(0))
            .collect();
                let _tlb_gen: Vec<std::sync::atomic::AtomicU64> = (0..1)
            .map(|_| std::sync::atomic::AtomicU64::new(0))
            .collect();
        let _msip_pending: Vec<std::sync::atomic::AtomicU64> = (0..1)
            .map(|_| std::sync::atomic::AtomicU64::new(0))
            .collect();
        let module = ModuleState {
            stop_flag: std::sync::atomic::AtomicBool::new(false),
            stop_info: std::sync::Mutex::new(StopInfo::empty()),
            wfi_count: std::sync::atomic::AtomicU32::new(0),
            wfi_flags: _wfi_flags.into_boxed_slice(),
                        active_hart_num: 1,
            tlb_gen: std::sync::atomic::AtomicU64::new(0),
            tlb_gen_per_hart: _tlb_gen.into_boxed_slice(),
            lr_reserved: Box::new([]),
            msip_pending: _msip_pending.into_boxed_slice(),
        };
        // opcode=0000011, rd=15, func3=101(LHU), rs1=15, imm=0x336
        let instr: u32 = 0x3367d783u32; // lhu a5, 0x336(a5)
        let f = crate::decode::decode_fields(instr);

        // Write known data at the target address (0x100 + 0x336 = 0x436)
        let target_pa = 0x100u64 + 0x336u64;
        ram[target_pa as usize] = 0x42;
        ram[target_pa as usize + 1] = 0x13;

        let adv = handle_load_concurrent(
            &mut state, &f, instr, &ctx, &pmp, &dev, &clint, &uart, &module,
        );
        assert_eq!(adv, 4);
        // After lhu: a5 should be 0x1342 (little-endian: 0x42 | 0x13<<8)
        assert_eq!(
            state.gprs[15], 0x1342,
            "LHU rd==rs1: loaded value must be 0x1342, not the old address"
        );
    }

    /// When a load with rd==rs1 triggers a page fault, the destination
    /// register must NOT be overwritten.  The kernel's trap handler relies
    /// on seeing the original register state.
    #[test]
    fn load_rd_equals_rs1_preserves_rd_on_pagefault() {
        let mut state: HartState = unsafe { std::mem::zeroed() };
        state.mode = riscv_mode::U; // U-mode ->Sv39 active
        state.mmu_mode = 8; // Sv39
        state.satp = 8u64 << 60; // Sv39, root PPN=0
        state.stvec = 0x80000400; // S-mode trap vector
        state.medeleg = 1 << 13; // Delegate LdPageFault to S-mode
        state.gprs[15] = 7; // a5 = 7 (the DT_RELA crash value)
                            // Empty RAM ->any page walk will fail (PTE.V=0)
        let mut ram = vec![0u8; 0x1000];
        let ctx = WalkCtx {
            ram: ram.as_mut_ptr(),
            ram_size: 0x1000,
            ram_base: 0,
            shadow_base: 0,
            shadow_size: 0,
            tlb_gen: std::ptr::null(),
            itlb_hand: Cell::new(0),
            dtlb_hand: Cell::new(0),
            lr_reserved: std::ptr::null_mut(),
            num_harts: 1,
        };
        let mut _pmp_cfg = vec![0u8; 1];
        let mut _pmp_addr = vec![0u64; 1];
        let pmp = make_permissive_pmp(&mut _pmp_cfg, &mut _pmp_addr);
        let dev = DevCtx {
            bases: std::ptr::null(),
            ends: std::ptr::null(),
            num: 0,
            virtio_base: 0,
            virtio_raw: std::ptr::null_mut(),
        };
        let _mtime = std::sync::atomic::AtomicU64::new(0);
        let _mtimecmp = std::sync::atomic::AtomicU64::new(0);
        let _msip = std::sync::atomic::AtomicU8::new(0);
        let clint = ConcurrentClintCtx {
            base: 0,
            mtime: &_mtime,
            mtimecmp: &_mtimecmp,
            msip: &_msip,
            num_harts: 1,
            msip_pending: Cell::new(std::ptr::null()),
            hart_threads: Cell::new(std::ptr::null()),
        };
        let _tx_buf = [0u8; 16];
        let _tx_wr = std::sync::atomic::AtomicU32::new(0);
        let uart = FfiUartCtx {
            base: 0,
            tx_buf: _tx_buf.as_ptr() as *mut u8,
            tx_cap: 16,
            tx_wr: &_tx_wr as *const std::sync::atomic::AtomicU32 as *mut u32,
            ie: 0,
            txctrl: 0,
            rxctrl: 0,
            rx_fifo_len: 0,
            tx_notify_fd: -1,
            no_stdout: 0, rx_notify: std::ptr::null_mut(),
        };
        let _wfi_flags: Vec<std::sync::atomic::AtomicU8> = (0..1)
            .map(|_| std::sync::atomic::AtomicU8::new(0))
            .collect();
                let _tlb_gen: Vec<std::sync::atomic::AtomicU64> = (0..1)
            .map(|_| std::sync::atomic::AtomicU64::new(0))
            .collect();
        let _msip_pending: Vec<std::sync::atomic::AtomicU64> = (0..1)
            .map(|_| std::sync::atomic::AtomicU64::new(0))
            .collect();
        let module = ModuleState {
            stop_flag: std::sync::atomic::AtomicBool::new(false),
            stop_info: std::sync::Mutex::new(StopInfo::empty()),
            wfi_count: std::sync::atomic::AtomicU32::new(0),
            wfi_flags: _wfi_flags.into_boxed_slice(),
                        active_hart_num: 1,
            tlb_gen: std::sync::atomic::AtomicU64::new(0),
            tlb_gen_per_hart: _tlb_gen.into_boxed_slice(),
            lr_reserved: Box::new([]),
            msip_pending: _msip_pending.into_boxed_slice(),
        };

        let instr: u32 = 0x3367d783u32; // lhu a5, 0x336(a5)
        let f = crate::decode::decode_fields(instr);

        let adv = handle_load_concurrent(
            &mut state, &f, instr, &ctx, &pmp, &dev, &clint, &uart, &module,
        );
        // Page fault delivered, PC redirected ->advance = 0
        assert_eq!(adv, 0, "page fault must return 0 (PC already redirected)");
        // a5 MUST retain its original value (7), NOT be overwritten
        assert_eq!(
            state.gprs[15], 7,
            "rd==rs1 on page fault: a5 must stay 7, not be corrupted to 0 or loaded data"
        );
        // PC must have changed (to stvec)
        assert_ne!(
            state.pc, 0,
            "PC must be redirected to stvec after page fault"
        );
    }

    /// CSR with rd==rs1: old CSR value read before rs1, then written to rd.
    /// Test CSRRW where rd=rs1 — the old CSR value is swapped into rd.
    #[test]
    fn csrrw_rd_equals_rs1_swaps_correctly() {
        let mut state: HartState = unsafe { std::mem::zeroed() };
        state.mode = riscv_mode::M;
        state.gprs[15] = 0xDEADBEEF; // a5 = value to write to CSR
        state.stvec = 0x80001000; // stvec = old value

        let mut result = unsafe { std::mem::zeroed() };
        let pmp = PmpCtx {
            cfg: std::ptr::null_mut(),
            addr: std::ptr::null_mut(),
            num: 0,
        };
        // CSRRW a5, stvec, a5  -> funct3=001, rd=15, rs1=15, csr=0x105
        let adv = crate::csr::handle_csr(&mut state, 15, 15, 0x105, 1, 0, &mut result, 0, 0, &pmp);
        assert_eq!(adv, 4);
        // a5 should now hold old stvec (0x80001000), not 0xDEADBEEF
        assert_eq!(
            state.gprs[15], 0x80001000,
            "CSRRW rd==rs1: a5 must swap to old CSR value"
        );
        // stvec should now hold the old a5 value (0xDEADBEEF)
        assert_eq!(
            state.stvec, 0xDEADBEEF,
            "CSRRW rd==rs1: CSR must get old a5 value"
        );
    }

    /// CSRRS with rd==rs1: rs1 bits are set in CSR, old CSR value ->rd.
    #[test]
    fn csrrs_rd_equals_rs1_sets_bits_correctly() {
        let mut state: HartState = unsafe { std::mem::zeroed() };
        state.mode = riscv_mode::M;
        state.gprs[10] = 0x0000000F; // a0 = mask to set
        state.mie = 0x00000088; // MIE currently has MTIE+MSIE set

        let mut result = unsafe { std::mem::zeroed() };
        let pmp = PmpCtx {
            cfg: std::ptr::null_mut(),
            addr: std::ptr::null_mut(),
            num: 0,
        };
        // CSRRS a0, mie, a0  -> funct3=010, rd=10, rs1=10, csr=0x304
        let adv = crate::csr::handle_csr(&mut state, 10, 10, 0x304, 2, 0, &mut result, 0, 0, &pmp);
        assert_eq!(adv, 4);
        // a0 should hold old MIE value (0x88), not the mask
        assert_eq!(
            state.gprs[10], 0x88,
            "CSRRS rd==rs1: a0 must hold old MIE, not 0x0F"
        );
        // MIE should now be 0x88 | 0x0F = 0x8F
        assert_eq!(
            state.mie, 0x8F,
            "CSRRS rd==rs1: MIE must have 0x0F bits set"
        );
    }
}
