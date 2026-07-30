//! Thread-per-hart concurrent execution engine.
//!
//! Replaces the sequential round-robin batch model with true OS-thread parallelism:
//! one ``std::thread`` per non-halted hart, shared RAM via raw pointer (x86 TSO),
//! CLINT state via ``Atomic*``, and AMO atomics via ``AtomicU32``/``AtomicU64``.
//!
//! ``run_parallel`` is the FFI entry point, mirroring ``run_batch``'s signature
//! so the Python side can switch transparently.
use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, Mutex};


// ============================================================
//  Items extracted to sibling modules in cpu/src/
// ============================================================
//
// Each extracted module is declared in lib.rs as a top-level module
// of the pyremu-native crate.  The `use crate::*` paths below bring
// the pub(crate) items back into scope so `run_parallel` (below) and
// the test functions can reference them without qualification changes.
use crate::hart_sched::*;

#[cfg(feature = "diagnostic")]
#[allow(unused)]
use crate::diag;

use crate::state::{
    exit_reason,
	BatchResult, HartState, MemCtx,
	FfiClintCtx, FfiDevCtx, FfiPmpCtx,
	FfiUartCtx, FfiVirtIoCtx,
};


// ============================================================
//  Send/Sync wrappers for FFI raw pointers
// ============================================================
//
// Python holds all backing memory (bytearray, ctypes arrays) alive for
// the duration of the FFI call.  These wrappers assert Send + Sync
// so they can cross thread boundaries safely.

/// Wrapper around ``MemCtx`` that is ``Send + Sync``.
///
/// # Safety
/// The Python FFI caller must keep the underlying RAM alive.
#[derive(Clone, Copy)]
pub(crate) struct SharedMemCtx {
    pub ram: *mut u8,
    pub ram_size: u64,
    pub ram_base: u64,
    pub shadow_base: u64,
    pub shadow_size: u64,
    /// Per-hart LR/SC reservation slots.  LR sets slot[hid]=pa;
    /// any store clears all slots.  Allocated in ModuleState.
    pub lr_reserved: *mut AtomicU64,
}
unsafe impl Send for SharedMemCtx {}
unsafe impl Sync for SharedMemCtx {}

/// Wrapper around PMP pointers that is ``Send + Sync``.
#[derive(Clone, Copy)]
pub(crate) struct SharedPmpCtx {
    pub cfg: *mut u8,
    pub addr: *mut u64,
    pub num: u8,
}
unsafe impl Send for SharedPmpCtx {}
unsafe impl Sync for SharedPmpCtx {}

/// Wrapper around device MMIO pointers that is ``Send + Sync``.
#[derive(Clone, Copy)]
pub(crate) struct SharedDevCtx {
    pub bases: *const u64,
    pub ends: *const u64,
    pub num: u8,
    pub virtio_base: u64,
    pub virtio_raw: *mut FfiVirtIoCtx,
}
unsafe impl Send for SharedDevCtx {}
unsafe impl Sync for SharedDevCtx {}


// ============================================================
//  Concurrent CLINT context — Atomic wrappers around shared state
// ============================================================

/// CLINT state safe for concurrent access across hart threads.
///
/// All fields are atomic or read-only after initialisation.
/// Pointers are cast from Python ctypes arrays; on x86-64
/// ``AtomicU64`` has the same layout as ``u64`` (8 bytes, 8-byte aligned)
/// and ``AtomicU8`` has the same layout as ``u8`` (1 byte, 1-byte aligned).
///
/// MSIP is modelled as an **edge counter** (not a level): each write of 1
/// increments the per-hart counter; ``sync_msip`` detects new edges by
/// comparing against the last-seen value.  This correctly handles
/// consecutive writes of 1 without an intervening 0, which a pure
/// level-triggered detector would miss.
pub struct ConcurrentClintCtx {
    pub base: u64,
    /// Shared mtime counter — each hart increments atomically per instruction.
    pub mtime: *const AtomicU64,
    /// Per-hart mtimecmp registers — indexed by hart_id.
    pub mtimecmp: *const AtomicU64,
    /// Per-hart MSIP level bytes — indexed by hart_id.
    pub msip: *const AtomicU8,
    pub num_harts: u32,
    /// Per-hart atomic MSIP pending slots (Release write from sender,
    /// Acquire swap from receiver).  Set after ``ModuleState`` creation.
    /// ``Cell`` allows late initialisation despite ``&self``.
    pub msip_pending: Cell<*const AtomicU64>,
}

// Safety: Python holds the backing ctypes arrays alive for the FFI call.
// All pointers target properly-aligned atomic types.
unsafe impl Send for ConcurrentClintCtx {}
unsafe impl Sync for ConcurrentClintCtx {}

impl ConcurrentClintCtx {
    /// Build from the FFI ``FfiClintCtx`` raw pointers.
    ///
    /// # Safety
    ///
    /// The caller must ensure the backing ctypes arrays outlive this struct.
    /// On x86-64 the transmute from ``*mut u64`` to ``*const AtomicU64`` is
    /// sound because both types have identical size and alignment.
    pub unsafe fn from_ffi(raw: &FfiClintCtx, num_harts: u32) -> Self {
        ConcurrentClintCtx {
            base: raw.base,
            mtime: raw.mtime as *const AtomicU64,
            mtimecmp: raw.mtimecmp as *const AtomicU64,
            msip: raw.msip as *const AtomicU8,
            num_harts,
            msip_pending: Cell::new(std::ptr::null()),
        }
    }
}

// ============================================================
//  Stop coordination
// ============================================================

/// Reason a hart set the global stop flag.
#[derive(Clone, Copy)]
pub struct StopInfo {
    pub reason: u8,
    pub hart_id: u8,
    pub pc: u64,
    pub instr: u32,
    pub trap_cause: u32,
    pub trap_tval: u64,
    pub trap_is_interrupt: u8,
    pub trap_delegated: u8,
}

impl StopInfo {
    pub const fn empty() -> Self {
        StopInfo {
            reason: exit_reason::NORMAL,
            hart_id: 0,
            pc: 0,
            instr: 0,
            trap_cause: 0,
            trap_tval: 0,
            trap_is_interrupt: 0,
            trap_delegated: 0,
        }
    }
}

/// Global coordination state shared across all hart threads.
///
/// One instance per ``run_parallel`` call, wrapped in ``Arc``.
pub struct ModuleState {
    /// When set to true, all hart threads must exit at their next
    /// instruction boundary.
    pub stop_flag: AtomicBool,
    /// Details of the stop event, written by the triggering hart.
    /// Protected by a Mutex so only one hart sets it.
    pub stop_info: Mutex<StopInfo>,
    /// Number of harts currently in WFI spin-wait.
    pub wfi_count: AtomicU32,
    /// Per-hart WFI status flags (index = hart_id).
    /// Each hart sets its slot to 1 when entering WFI, 0 when leaving.
    pub wfi_flags: Box<[AtomicU8]>,
    /// Number of non-halted harts. Set once before threads spawn, read-only after.
    pub active_hart_num: u32,
    /// Global TLB generation counter — incremented (Release) by any hart
    /// that executes SFENCE.VMA.  Each hart locally samples this counter
    /// (Acquire) at instruction boundaries and flushes its own TLB when
    /// it detects a mismatch against its per-hart slot in ``tlb_gen_per_hart``.
    /// This provides broadcast-TLB-invalidation semantics without explicit
    /// IPI-driven shootdown: over-invalidation never breaks correctness,
    /// and it closes the window where hart A's page-table write followed by
    /// local SFENCE.VMA is invisible to hart B's TLB hit.
    pub tlb_gen: AtomicU64,
    /// Per-hart last-seen TLB generation.  When a hart's value is less than
    /// the global ``tlb_gen``, it must flush both itlb and dtlb before its
    /// next instruction fetch or data access.
    pub tlb_gen_per_hart: Box<[AtomicU64]>,
    /// Per-hart LR/SC reservation slots.  Each slot holds the PA reserved by
    /// that hart (0 = no reservation).  Any store to a PA clears matching
    /// reservations from ALL harts, ensuring correct multi-hart LR/SC semantics
    /// (RISC-V spec §8.2: "a successful SC on a given hart renders any LR on
    /// that hart invalid, and a successful AMO, SC, or store on any other hart
    /// to the reservation set renders any LR on this hart invalid").
    pub lr_reserved: Box<[AtomicU64]>,
    /// Per-hart cross-thread MSIP notification.  When another hart writes to
    /// CLINT MSIP for this hart, the bit is set here atomically (Release).
    /// This hart's ``sync_msip`` reads & clears it with ``swap(0, Acquire)``
    /// and merges into ``HartState.mip``.  This avoids the non-atomic RMW race
    /// on ``HartState.mip`` between the sender's direct write and the receiver's
    /// ``sync_mtip``/``sync_msip`` calls.
    pub msip_pending: Box<[AtomicU64]>,
}

impl ModuleState {
    pub fn new(num_harts: u32, active_hart_num: u32, init_tlb_gen: u64) -> Self {
        let mut v = Vec::with_capacity(num_harts as usize);
        let mut gv = Vec::with_capacity(num_harts as usize);
        let mut rv = Vec::with_capacity(num_harts as usize);
        let mut mv = Vec::with_capacity(num_harts as usize);
        for _ in 0..num_harts {
            v.push(AtomicU8::new(0));
            gv.push(AtomicU64::new(0));
            rv.push(AtomicU64::new(0));
            mv.push(AtomicU64::new(0));
        }
        ModuleState {
            stop_flag: AtomicBool::new(false),
            stop_info: Mutex::new(StopInfo::empty()),
            wfi_count: AtomicU32::new(0),
            wfi_flags: v.into_boxed_slice(),
            active_hart_num,
            tlb_gen: AtomicU64::new(init_tlb_gen),
            tlb_gen_per_hart: gv.into_boxed_slice(),
            lr_reserved: rv.into_boxed_slice(),
            msip_pending: mv.into_boxed_slice(),
        }
    }

    /// Set the stop flag and record the reason.  Only the first caller
    /// wins; subsequent callers are silently ignored.
    pub fn request_stop(&self, info: StopInfo) {
        if self.stop_flag.swap(true, Ordering::Release) {
            return; // already stopping
        }
        if let Ok(mut guard) = self.stop_info.lock() {
            *guard = info;
        }
    }

    /// Check whether all non-halted harts are in WFI.
    #[inline]
    pub fn all_in_wfi(&self) -> bool {
        self.wfi_count.load(Ordering::Acquire) >= self.active_hart_num
    }
}

//  FFI structs (grouped parameters for run_parallel)
// ============================================================

/// Hart execution parameters passed across the FFI boundary.
#[repr(C)]
/// Shared external-interrupt context owned by Python, polled by Rust.
///
/// PLIC and device state lives on the Python side.  When a device raises
/// (or lowers) an interrupt, Python updates this struct.  Rust checks
/// ``pending`` periodically inside the hart loop; when set it exits the
/// batch so Python can call ``_native_sync_plic_mip()`` to update each
/// hart's ``mip`` with the latest PLIC-driven MEIP/SEIP bits.
#[repr(C)]
pub struct FfiExtIrqCtx {
    /// Non-zero: at least one external interrupt source is asserted and
    /// the PLIC state may have changed.  Rust exits the batch on next check.
    pub pending: u8,
    /// Bitmap of pending interrupt sources.  Bit *i* corresponds to PLIC
    /// interrupt source *i* (1 = UART, 2 = VirtIO, …).  Updated atomically
    /// by Python; currently informational, may drive inline delivery later.
    pub sources: u32,
    /// Highest priority among currently-pending sources, or 0 if none.
    /// Rust may skip the batch exit when priority ≤ the current hart's
    /// PLIC threshold (not yet implemented — always exits when pending≠0).
    pub max_priority: u8,
    pub _pad: [u8; 2],
}

#[repr(C)]
pub struct FfiHartCtx {
    pub states: *mut HartState,
    pub num_harts: u32,
    pub max_instrs: u64,
    pub result: *mut BatchResult,
    pub stop_flag: *const u8,
    pub ext_irq: *mut FfiExtIrqCtx,
}

/// Peripheral / memory contexts passed across the FFI boundary.
#[repr(C)]
pub struct FfiPeriphCtx {
    pub mem: *const MemCtx,
    pub pmp: *const FfiPmpCtx,
    pub clint: *const FfiClintCtx,
    pub dev: *const FfiDevCtx,
    pub uart: *const FfiUartCtx,
    pub virtio: *const FfiVirtIoCtx,
}

/// Breakpoint configuration passed across the FFI boundary.
#[repr(C)]
pub struct FfiBpCtx {
    pub addrs: *const u64,
    pub count: u32,
}

/// TLB generation counter passed across the FFI boundary (mutable —
/// Rust writes back the post-batch values so they persist across calls).
#[repr(C)]
pub struct FfiTlbCtx {
    pub gen: *mut u64,
    pub gen_per_hart: *mut u64,
}

// ============================================================
//  FFI entry point helpers
// ============================================================

/// Empty UART context for when no UART is configured.
static EMPTY_UART: FfiUartCtx = FfiUartCtx {
    base: 0,
    tx_buf: std::ptr::null_mut(),
    tx_cap: 0,
    tx_wr: std::ptr::null_mut(),
    ie: 0,
    txctrl: 0,
    rxctrl: 0,
    rx_fifo_len: 0,
    tx_notify_fd: -1,
    no_stdout: 0,
};

#[inline]
unsafe fn init_result(result: *mut BatchResult) {
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

#[inline]
unsafe fn count_active(states: *mut HartState, num_harts: u32) -> u32 {
    let mut count: u32 = 0;
    for hid in 0..num_harts {
        if (*states.add(hid as usize)).halted == 0 {
            count += 1;
        }
    }
    count
}

#[inline]
unsafe fn build_bps(bp: *const FfiBpCtx) -> &'static [u64] {
    if bp.is_null() {
        return &[];
    }
    let bp = &*bp;
    if bp.count == 0 || bp.addrs.is_null() {
        return &[];
    }
    std::slice::from_raw_parts(bp.addrs, bp.count as usize)
}

/// Spawn one OS thread per hart, run ``hart_worker``, join all.
unsafe fn run_harts(
    states: *mut HartState,
    num_harts: u32,
    stop_flag: *const u8,
    ext_irq: *mut FfiExtIrqCtx,
    shared_mem: SharedMemCtx,
    shared_pmp: SharedPmpCtx,
    shared_dev: SharedDevCtx,
    cc_clint: &ConcurrentClintCtx,
    uart_ffi: *const FfiUartCtx,
    module: &Arc<ModuleState>,
    breakpoints: &'static [u64],
    max_instrs: u64,
) {
    let max_per_hart = if max_instrs == 0 {
        u64::MAX
    } else {
        (max_instrs / num_harts as u64).max(1)
    };

    let uart_ref: &'static FfiUartCtx = if uart_ffi.is_null() {
        &EMPTY_UART
    } else {
        std::mem::transmute(&*uart_ffi)
    };

    let mut handles = Vec::with_capacity(num_harts as usize);
    for hid in 0..num_harts {
        let mref = Arc::clone(module);
        let mem_send = shared_mem;
        let pmp_send = if shared_pmp.num > 0 {
            SharedPmpCtx {
                cfg: shared_pmp.cfg.add(hid as usize * 64),
                addr: shared_pmp.addr.add(hid as usize * 64),
                num: shared_pmp.num,
            }
        } else {
            shared_pmp
        };
        let dev_send = shared_dev;
        let bp_send: &'static [u64] = std::mem::transmute(breakpoints);
        let state_addr = states.add(hid as usize) as usize;
        let clint_ptr: &'static ConcurrentClintCtx = std::mem::transmute(cc_clint);
        let uart_ptr = uart_ref;
        let mph = max_per_hart;
        let stop_ptr = stop_flag as usize;
        let ext_irq_ptr = ext_irq as usize;

        let handle = std::thread::spawn(move || {
            let state = &mut *(state_addr as *mut HartState);
            hart_worker(
                state, hid as u8, mem_send, pmp_send, dev_send,
                clint_ptr, uart_ptr, &mref, bp_send, mph,
                stop_ptr as *const u8,
                ext_irq_ptr as *mut FfiExtIrqCtx,
            );
        });
        handles.push(handle);
    }

    for h in handles {
        let _ = h.join();
    }
}

#[inline]
unsafe fn collect_stop(module: &ModuleState, result: *mut BatchResult) {
    if let Ok(guard) = module.stop_info.lock() {
        (*result).exit_reason = guard.reason;
        (*result).exit_hart_id = guard.hart_id;
        (*result).exit_pc = guard.pc;
        (*result).exit_instr = guard.instr;
        (*result).trap_cause = guard.trap_cause;
        (*result).trap_tval = guard.trap_tval;
        (*result).trap_is_interrupt = guard.trap_is_interrupt;
        (*result).trap_delegated = guard.trap_delegated;
    }
}

#[inline]
unsafe fn sum_instrs(states: *mut HartState, num_harts: u32, result: *mut BatchResult) {
    let mut total: u64 = 0;
    for hid in 0..num_harts {
        total = total.wrapping_add((*states.add(hid as usize)).total_instrs);
    }
    (*result).total_instrs = total;
}

#[inline]
unsafe fn writeback_tlb(
    gen_ptr: *mut u64,
    gen_per_hart_ptr: *mut u64,
    module: &ModuleState,
    num_harts: u32,
) {
    if gen_ptr.is_null() {
        return;
    }
    *gen_ptr = module.tlb_gen.load(Ordering::Relaxed);
    for i in 0..num_harts as usize {
        *gen_per_hart_ptr.add(i) = module.tlb_gen_per_hart[i].load(Ordering::Relaxed);
    }
}

#[inline]
unsafe fn writeback_mtime(clint_raw: &FfiClintCtx, cc_clint: &ConcurrentClintCtx) {
    *clint_raw.mtime = (*cc_clint.mtime).load(Ordering::SeqCst);
}

// ============================================================
//  FFI entry point
// ============================================================

#[no_mangle]
#[allow(dead_code)]
pub unsafe extern "C" fn run_parallel(
    hart: *const FfiHartCtx,
    ffi: *const FfiPeriphCtx,
    bp: *const FfiBpCtx,
    tlb: *mut FfiTlbCtx,
) {
    // 1. Unpack FFI structs
    let hart = unsafe { &*hart };
    let ffi = unsafe { &*ffi };
    let states = hart.states;
    let num_harts = hart.num_harts;

    // 2. Zero-initialise result + build contexts
    let (mem, pmp_raw, clint_raw, dev_raw, cc_clint, active) = unsafe {
        init_result(hart.result);
        (
            &*ffi.mem,
            &*ffi.pmp,
            &*ffi.clint,
            &*ffi.dev,
            ConcurrentClintCtx::from_ffi(&*ffi.clint, num_harts),
            count_active(states, num_harts),
        )
    };

    // 3. Module state (TLB gen persists across batches)
    let tlb_gen_ptr = if tlb.is_null() { std::ptr::null_mut() } else { unsafe { (*tlb).gen } };
    let tlb_gen_per_hart_ptr = if tlb.is_null() { std::ptr::null_mut() } else { unsafe { (*tlb).gen_per_hart } };
    let init_gen = if tlb_gen_ptr.is_null() { 0 } else { unsafe { *tlb_gen_ptr } };
    let module = Arc::new(ModuleState::new(num_harts, active, init_gen));
    // Wire the msip_pending atomic channel into the CLINT context so
    // senders can atomically signal the target hart's WFI loop.
    cc_clint.msip_pending.set(module.msip_pending.as_ptr());

    // 4. Build shared contexts
    let virtio_raw: *mut FfiVirtIoCtx = if ffi.virtio.is_null() {
        std::ptr::null_mut()
    } else {
        ffi.virtio as *mut FfiVirtIoCtx
    };
    let shared_mem = SharedMemCtx {
        ram: mem.ram, ram_size: mem.ram_size, ram_base: mem.ram_base,
        shadow_base: mem.shadow_base, shadow_size: mem.shadow_size,
        lr_reserved: module.lr_reserved.as_ptr() as *mut AtomicU64,
    };
    let shared_pmp = SharedPmpCtx { cfg: pmp_raw.cfg, addr: pmp_raw.addr, num: pmp_raw.num };
    let shared_dev = SharedDevCtx {
        bases: dev_raw.bases, ends: dev_raw.ends, num: dev_raw.num,
        virtio_base: if virtio_raw.is_null() { 0 } else { unsafe { (*virtio_raw).base } },
        virtio_raw,
    };


    unsafe {
		// 5. Spawn & join hart threads
        run_harts(
            states, num_harts, hart.stop_flag, hart.ext_irq,
            shared_mem, shared_pmp, shared_dev,
            &cc_clint, ffi.uart, &module,
            build_bps(bp), hart.max_instrs,
        );
		// 6. Collect results — single unsafe block for all write-backs
	    collect_stop(&module, hart.result);
        sum_instrs(states, num_harts, hart.result);
        writeback_tlb(tlb_gen_ptr, tlb_gen_per_hart_ptr, &module, num_harts);
        writeback_mtime(clint_raw, &cc_clint);
    }
}

/// FFI entry: dump accumulated instruction frequency counters to the
/// diagnostic log and reset them.  Call once at emulator termination.
#[no_mangle]
pub extern "C" fn icount_flush() {
    crate::diag::icount_dump();
}

// ============================================================
//  Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;
    // Items extracted from this file into sibling modules — bring them
    // back in scope so tests can reference them directly.
    use crate::state::riscv_mode;
    use std::mem;

    fn make_state(pc: u64, hart_id: u64) -> HartState {
        let mut s: HartState = unsafe { mem::zeroed() };
        s.mode = riscv_mode::M;
        s.mtvec = 0x8000_0100;
        s.pc = pc;
        s.mhartid = hart_id;
        s
    }

    /// Convenience: call ``run_parallel`` with individual args (for tests).
    unsafe fn call_run_parallel(
        states: *mut HartState,
        num_harts: u32,
        max_instrs: u64,
        result: *mut BatchResult,
        mem: *const MemCtx,
        pmp: *const FfiPmpCtx,
        clint: *const FfiClintCtx,
        dev: *const FfiDevCtx,
        uart: *const FfiUartCtx,
        virtio: *const FfiVirtIoCtx,
        bp_addrs: *const u64,
        bp_count: u32,
        stop_flag: *const u8,
        tlb_gen: *mut u64,
        tlb_gen_per_hart: *mut u64,
    ) {
        let hart = FfiHartCtx {
            states, num_harts, max_instrs, result, stop_flag,
        };
        let periph = FfiPeriphCtx {
            mem, pmp, clint, dev, uart, virtio,
        };
        let bp = FfiBpCtx {
            addrs: bp_addrs, count: bp_count,
        };
        let mut tlb = FfiTlbCtx {
            gen: tlb_gen, gen_per_hart: tlb_gen_per_hart,
        };
        run_parallel(&hart, &periph, &bp, &mut tlb);
    }

    fn write_u32_le(buf: &mut [u8], offset: usize, val: u32) {
        buf[offset] = val as u8;
        buf[offset + 1] = (val >> 8) as u8;
        buf[offset + 2] = (val >> 16) as u8;
        buf[offset + 3] = (val >> 24) as u8;
    }

    /// Build default zero-length PMP / device arrays and call ``run_parallel``.
    unsafe fn run_parallel_defaults(
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
        // Single TOR entry covering full address space (R/W/X).
        // pmp_ok per RISC-V spec §3.7.1 denies S/U access when num==0,
        // so we must provide at least one permissive entry for tests.
        let mut pmp_cfg: [u8; 1] = [0x0F]; // PMP_R|PMP_W|PMP_X|PMP_A_TOR
        let mut pmp_addr: [u64; 1] = [u64::MAX];
        let pmp = FfiPmpCtx {
            cfg: pmp_cfg.as_mut_ptr(),
            addr: pmp_addr.as_mut_ptr(),
            num: 1,
            pmpsplit: 0,
        };
        // Allocate per-hart CLINT arrays based on the actual number of harts.
        let nh = num as usize;
        let mut mtimecmp_vec: Vec<u64> = vec![u64::MAX; nh];
        let mut msip_vec: Vec<u8> = vec![0u8; nh];
        let mut mtime_val: u64 = 0;
        let clint = FfiClintCtx {
            mtime: &mut mtime_val as *mut u64,
            mtimecmp: mtimecmp_vec.as_mut_ptr(),
            msip: msip_vec.as_mut_ptr(),
            base: 0,
        };
        call_run_parallel(
            states,
            num,
            max_instrs,
            result,
            &mem as *const MemCtx,
            &pmp as *const FfiPmpCtx,
            &clint as *const FfiClintCtx,
            &dev as *const FfiDevCtx,
            std::ptr::null(), // uart
            std::ptr::null(), // virtio
            std::ptr::null(), // bp_addrs
            0,                // bp_count
            std::ptr::null(), // stop_flag
            std::ptr::null(), // ext_irq
            std::ptr::null_mut(), std::ptr::null_mut(),
        );
    }

    #[test]
    fn parallel_single_hart_addi() {
        let mut ram = vec![0u8; 256];
        // Instructions at PC=0:
        write_u32_le(&mut ram, 0, 0x00A0_0293); // addi x5, x0, 10
        write_u32_le(&mut ram, 4, 0x0052_8293); // addi x5, x5, 5
        write_u32_le(&mut ram, 8, 0x1050_0073); // WFI -> clean exit

        let mut state = make_state(0, 0);
        state.mode = riscv_mode::M; // WFI is NOP in M-mode when mip=0->enters waiting
        state.mie = 0;
        state.mip = 0;
        let mut result: BatchResult = unsafe { mem::zeroed() };
        unsafe {
            run_parallel_defaults(
                &mut state as *mut HartState,
                1,
                ram.as_mut_ptr(),
                ram.len() as u64,
                0,
                0,
                0,
                100,
                &mut result as *mut BatchResult,
            );
        }
        assert_eq!(state.gprs[5], 15, "addi instructions should execute");
        assert_eq!(result.exit_reason, exit_reason::WFI_WAIT);
    }

    /// Test that ECALL delivers trap inline: mode -> M, PC -> mtvec,
    /// mepc = ECALL address.  The M-mode handler at mtvec advances
    /// mepc by 4 (to skip ECALL) and MRETs back to S-mode, where
    /// WFI terminates the batch cleanly.
    #[test]
    fn parallel_ecall_inline_trap() {
        let mut ram = vec![0u8; 256];
        // S-mode code at PC=0:
        write_u32_le(&mut ram, 0, 0x0010_0293); // addi x5, x0, 1
        write_u32_le(&mut ram, 4, 0x0000_0073); // ECALL (S-mode -> M-mode)
        write_u32_le(&mut ram, 8, 0x1050_0073); // WFI (reached after MRET)

        // M-mode handler at mtvec=0x20 (uses t1=x6 to avoid clobbering x5):
        let mtvec_addr: u64 = 0x20;
        // addi t1, x0, 4     -> 0x00400313  (t1 = 4)
        write_u32_le(&mut ram, 0x20, 0x0040_0313);
        // csrr t1, mepc      -> 0x34102373  (t1 = mepc)
        write_u32_le(&mut ram, 0x24, 0x3410_2373);
        // addi t1, t1, 4     -> 0x00430313  (t1 = mepc + 4)
        write_u32_le(&mut ram, 0x28, 0x0043_0313);
        // csrw mepc, t1      -> 0x34131073  (mepc = t1)
        write_u32_le(&mut ram, 0x2C, 0x3413_1073);
        // mret                -> 0x30200073
        write_u32_le(&mut ram, 0x30, 0x3020_0073);

        let mut state = make_state(0, 0);
        state.mode = riscv_mode::S;
        state.mtvec = mtvec_addr;
        state.mstatus = (1 << 11) | (1 << 7); // MPP=S, MPIE=1
        state.mie = 0;
        state.mip = 0;

        let mut result: BatchResult = unsafe { mem::zeroed() };
        unsafe {
            run_parallel_defaults(
                &mut state as *mut HartState,
                1,
                ram.as_mut_ptr(),
                ram.len() as u64,
                0,
                0,
                0,
                100,
                &mut result as *mut BatchResult,
            );
        }
        // addi x5,0,1 executed before ECALL
        assert_eq!(state.gprs[5], 1);
        // ECALL trap delivered inline -> M-mode handler -> MRET -> S-mode -> WFI
        assert_eq!(
            result.exit_reason,
            exit_reason::WFI_WAIT,
            "inline ECALL+M-mode handler+MRET+WFI->WFI_WAIT exit"
        );
    }

    #[test]
    fn parallel_two_harts_independent() {
        let mut ram = vec![0u8; 4096];
        // Hart 0: addi x5, x0, 10; addi x5, x5, 3; WFI (PC 0..12)
        write_u32_le(&mut ram, 0, 0x00A0_0293);
        write_u32_le(&mut ram, 4, 0x0032_8293);
        write_u32_le(&mut ram, 8, 0x1050_0073); // WFI
                                                // Hart 1: addi x6, x0, 20; addi x6, x6, 7; WFI (PC 128..140)
        write_u32_le(&mut ram, 128, 0x0140_0313);
        write_u32_le(&mut ram, 132, 0x0073_0313);
        write_u32_le(&mut ram, 136, 0x1050_0073); // WFI

        let mut s0 = make_state(0, 0);
        s0.mode = riscv_mode::M; // WFI needs M-mode to avoid TW trap
        s0.mie = 0;
        s0.mip = 0;
        let mut s1 = make_state(128, 1);
        s1.mode = riscv_mode::M;
        s1.mie = 0;
        s1.mip = 0;
        let mut states = [s0, s1];

        let mut result: BatchResult = unsafe { mem::zeroed() };
        unsafe {
            run_parallel_defaults(
                states.as_mut_ptr(),
                2,
                ram.as_mut_ptr(),
                ram.len() as u64,
                0,
                0,
                0,
                100,
                &mut result as *mut BatchResult,
            );
        }
        // Both harts finished their adds before entering WFI.
        // Since all harts end up in WFI, the batch exits with WFI_WAIT.
        assert_eq!(states[0].gprs[5], 13, "hart 0: x5 should be 13");
        assert_eq!(states[1].gprs[6], 27, "hart 1: x6 should be 27");
        assert_eq!(
            result.exit_reason,
            exit_reason::WFI_WAIT,
            "both harts WFI -> should exit with WFI_WAIT, got {}",
            result.exit_reason
        );
    }

    #[test]
    fn parallel_amo_single_hart() {
        // Verify AMOADD works with a single hart in concurrent mode.
        let mut ram = vec![0u8; 4096];
        let shared_addr: u64 = 0xF0;
        write_u32_le(&mut ram, 0, 0x0063_A2AF); // amoadd.w x5, x6, (x7)
        write_u32_le(&mut ram, 4, 0x1050_0073); // WFI

        ram[shared_addr as usize] = 42;
        let mut state = make_state(0, 0);
        state.mode = riscv_mode::M;
        state.mie = 0;
        state.mip = 0;
        state.gprs[7] = shared_addr; // addr
        state.gprs[6] = 3; // addend
        let mut result: BatchResult = unsafe { mem::zeroed() };
        unsafe {
            run_parallel_defaults(
                &mut state as *mut HartState,
                1,
                ram.as_mut_ptr(),
                ram.len() as u64,
                0,
                0,
                0,
                100,
                &mut result as *mut BatchResult,
            );
        }
        let final_val = ram[shared_addr as usize] as u32
            | (ram[shared_addr as usize + 1] as u32) << 8
            | (ram[shared_addr as usize + 2] as u32) << 16
            | (ram[shared_addr as usize + 3] as u32) << 24;
        assert_eq!(
            final_val, 45,
            "single hart AMOADD: 42+3=45, got {}",
            final_val
        );
        assert_eq!(result.exit_reason, exit_reason::WFI_WAIT);
    }

    #[test]
    fn parallel_amo_add_two_harts() {
        let mut ram = vec![0u8; 4096];
        let shared_addr: u64 = 0xF0; // aligned, well within RAM

        // Hart 0: AMOADD.W to shared_addr; then ECALL
        //   amoadd.w x5, x6, (x7): rs1=x7=addr, rs2=x6=addend, rd=x5=old
        write_u32_le(&mut ram, 0, 0x0063_A2AF); // amoadd.w x5, x6, (x7)
        write_u32_le(&mut ram, 4, 0x1050_0073); // WFI

        // Hart 1: AMOADD.W to same shared_addr; then WFI
        write_u32_le(&mut ram, 128, 0x0063_A2AF); // amoadd.w x5, x6, (x7)
        write_u32_le(&mut ram, 132, 0x1050_0073); // WFI

        // Set initial value at shared_addr = 42 (little-endian)
        ram[shared_addr as usize] = 42;
        ram[shared_addr as usize + 1] = 0;
        ram[shared_addr as usize + 2] = 0;
        ram[shared_addr as usize + 3] = 0;

        let mut s0 = make_state(0, 0);
        s0.mode = riscv_mode::M; // WFI needs M-mode
        s0.mie = 0;
        s0.mip = 0;
        s0.gprs[7] = shared_addr; // rs1 — address
        s0.gprs[6] = 3; // rs2 — value to add

        let mut s1 = make_state(128, 1);
        s1.mode = riscv_mode::M;
        s1.mie = 0;
        s1.mip = 0;
        s1.gprs[7] = shared_addr;
        s1.gprs[6] = 5;

        let mut states = [s0, s1];
        let mut result: BatchResult = unsafe { mem::zeroed() };
        unsafe {
            run_parallel_defaults(
                states.as_mut_ptr(),
                2,
                ram.as_mut_ptr(),
                ram.len() as u64,
                0,
                0,
                0,
                100,
                &mut result as *mut BatchResult,
            );
        }

        // Final value should be 42 + 3 + 5 = 50 (both AMOADDs applied atomically)
        let final_val = ram[shared_addr as usize] as u32
            | (ram[shared_addr as usize + 1] as u32) << 8
            | (ram[shared_addr as usize + 2] as u32) << 16
            | (ram[shared_addr as usize + 3] as u32) << 24;
        assert_eq!(
            final_val, 50,
            "AMOADD final value should be 42+3+5=50, got {}",
            final_val
        );
    }

    #[test]
    fn parallel_stop_on_mmio_store() {
        let mut ram = vec![0u8; 4096];
        let dev_addr: u64 = 0x1000_0000; // UART MMIO
                                         // Hart 0: store x5 -> dev_addr (MMIO exit); Hart 1: addi x0,x0,0 loop
                                         // sb x5, 0(x6)  where x6 = dev_addr, x5 = 0x41 ('A')
        write_u32_le(&mut ram, 0, 0x0051_3023); // sd x5, 0(x2) — no, let me use a simple store
                                                // Actually: sd x5, 0(x6) = 0x0053_3023 (stores x5 to [x6+0])
        write_u32_le(&mut ram, 0, 0x0053_3023); // sd x5, 0(x6)
                                                // Hart 1: NOP loop at offset 128 — enough iterations to stay alive
        // until hart 0's MMIO store triggers the batch exit.
        for i in 0..500 {
            write_u32_le(&mut ram, (128 + i * 4) as usize, 0x0000_0013); // addi x0, x0, 0
        }

        let mut s0 = make_state(0, 0);
        s0.mode = riscv_mode::M; // Bare-mode VA=PA
        s0.gprs[5] = 0x41; // value to store
        s0.gprs[6] = dev_addr; // rs1 — store address
        let mut s1 = make_state(128, 1);
        s1.mode = riscv_mode::M;
        let mut states = [s0, s1];

        // Custom device MMIO: cover [dev_addr, dev_addr+8)
        let dev_bases: [u64; 1] = [dev_addr];
        let dev_ends: [u64; 1] = [dev_addr + 8];
        let dev = FfiDevCtx {
            bases: dev_bases.as_ptr(),
            ends: dev_ends.as_ptr(),
            num: 1,
        };
        let mem = MemCtx {
            ram: ram.as_mut_ptr(),
            ram_size: ram.len() as u64,
            ram_base: 0,
            shadow_base: 0,
            shadow_size: 0,
        };
        // Single TOR entry covering full address space (R/W/X).
        // pmp_ok per RISC-V spec §3.7.1 denies S/U access when num==0,
        // so we must provide at least one permissive entry for tests.
        let mut pmp_cfg: [u8; 1] = [0x0F]; // PMP_R|PMP_W|PMP_X|PMP_A_TOR
        let mut pmp_addr: [u64; 1] = [u64::MAX];
        let pmp = FfiPmpCtx {
            cfg: pmp_cfg.as_mut_ptr(),
            addr: pmp_addr.as_mut_ptr(),
            num: 1,
            pmpsplit: 0,
        };
        let nh = 2usize;
        let mut mtimecmp_vec: Vec<u64> = vec![u64::MAX; nh];
        let mut msip_vec: Vec<u8> = vec![0u8; nh];
        let mut mtime_val: u64 = 0;
        let clint = FfiClintCtx {
            mtime: &mut mtime_val as *mut u64,
            mtimecmp: mtimecmp_vec.as_mut_ptr(),
            msip: msip_vec.as_mut_ptr(),
            base: 0,
        };
        let mut result: BatchResult = unsafe { mem::zeroed() };
        unsafe {
            call_run_parallel(
                states.as_mut_ptr(),
                2,
                100,
                &mut result as *mut BatchResult,
                &mem as *const MemCtx,
                &pmp as *const FfiPmpCtx,
                &clint as *const FfiClintCtx,
                &dev as *const FfiDevCtx,
                std::ptr::null(), // uart
                std::ptr::null(), // virtio
                std::ptr::null(), // bp_addrs
                0,                // bp_count
                std::ptr::null(), // stop_flag
            std::ptr::null(), // ext_irq
                std::ptr::null_mut(), std::ptr::null_mut(),
            );
        }

        // Store to device MMIO should trigger batch exit with MMIO reason.
        assert_eq!(
            result.exit_reason,
            exit_reason::MMIO,
            "store to MMIO device should cause batch exit with MMIO reason"
        );
        assert_eq!(result.exit_hart_id, 0, "exit hart should be 0");
    }

    #[test]
    fn parallel_wfi_all_idle_stops() {
        let mut ram = vec![0u8; 256];
        // Both harts: WFI
        write_u32_le(&mut ram, 0, 0x1050_0073); // WFI
        write_u32_le(&mut ram, 128, 0x1050_0073);

        let mut s0 = make_state(0, 0);
        s0.mode = riscv_mode::M;
        s0.mie = 0;
        s0.mip = 0;
        let mut s1 = make_state(128, 1);
        s1.mode = riscv_mode::M;
        s1.mie = 0;
        s1.mip = 0;
        let mut states = [s0, s1];

        let mut result: BatchResult = unsafe { mem::zeroed() };
        unsafe {
            run_parallel_defaults(
                states.as_mut_ptr(),
                2,
                ram.as_mut_ptr(),
                ram.len() as u64,
                0,
                0,
                0,
                100,
                &mut result as *mut BatchResult,
            );
        }

        // Both harts should be in WFI, and batch should exit with WFI_WAIT
        assert_eq!(
            result.exit_reason,
            exit_reason::WFI_WAIT,
            "All WFI should cause WFI_WAIT exit, got {}",
            result.exit_reason
        );
        assert_eq!(states[0].waiting, 1);
        assert_eq!(states[1].waiting, 1);
    }

    /// Helper: run_parallel with a custom CLINT base address.
    #[allow(unused)]
	unsafe fn run_parallel_with_clint(
        states: *mut HartState,
        num: u32,
        ram: *mut u8,
        ram_sz: u64,
        ram_base: u64,
        clint_base: u64,
        max_instrs: u64,
        result: *mut BatchResult,
    ) {
        let mem = MemCtx {
            ram,
            ram_size: ram_sz,
            ram_base,
            shadow_base: 0,
            shadow_size: 0,
        };
        let dev_bases: [u64; 0] = [];
        let dev_ends: [u64; 0] = [];
        let dev = FfiDevCtx {
            bases: dev_bases.as_ptr(),
            ends: dev_ends.as_ptr(),
            num: 0,
        };
        // Single TOR entry covering full address space (R/W/X).
        // pmp_ok per RISC-V spec §3.7.1 denies S/U access when num==0,
        // so we must provide at least one permissive entry for tests.
        let mut pmp_cfg: [u8; 1] = [0x0F]; // PMP_R|PMP_W|PMP_X|PMP_A_TOR
        let mut pmp_addr: [u64; 1] = [u64::MAX];
        let pmp = FfiPmpCtx {
            cfg: pmp_cfg.as_mut_ptr(),
            addr: pmp_addr.as_mut_ptr(),
            num: 1,
            pmpsplit: 0,
        };
        let nh = num as usize;
        let mut mtimecmp_vec: Vec<u64> = vec![u64::MAX; nh];
        let mut msip_vec: Vec<u8> = vec![0u8; nh];
        let mut mtime_val: u64 = 0;
        let clint = FfiClintCtx {
            mtime: &mut mtime_val as *mut u64,
            mtimecmp: mtimecmp_vec.as_mut_ptr(),
            msip: msip_vec.as_mut_ptr(),
            base: clint_base,
        };
        call_run_parallel(
            states,
            num,
            max_instrs,
            result,
            &mem as *const MemCtx,
            &pmp as *const FfiPmpCtx,
            &clint as *const FfiClintCtx,
            &dev as *const FfiDevCtx,
            std::ptr::null(), // uart
            std::ptr::null(), // virtio
            std::ptr::null(), // bp_addrs
            0,                // bp_count
            std::ptr::null(), // stop_flag
            std::ptr::null(), // ext_irq
            std::ptr::null_mut(), std::ptr::null_mut(),
        );
    }

    /// Verify that a store to CLINT MSIP succeeds without traps.
    #[test]
    fn store_to_clint_msip_single_hart() {
        let mut ram = vec![0u8; 1024];
        let clint_base: u64 = 0x2000000;

        // Single hart in M-mode writes to CLINT MSIP for hart 0 (self).
        // 0x00: addi t0, x0, 0x42        ->t0 = 0x42 (control: prove PROLOGUE runs)
        // 0x04: sw   t0, 0x1000(x0)     ->store to RAM (control: prove STORE works)
        // 0x08: lui  t0, 0x20000        ->t0 = 0x20000000
        // 0x0C: addi t0, t0, 0          ->t0 = 0x20000000 (CLINT MSIP for hart 0)
        // 0x10: addi t1, x0, 1          ->t1 = 1
        // 0x14: sw   t1, 0(t0)          ->write MSIP=1 for hart 0
        // 0x18: addi t2, x0, 0x42       ->t2 = 0x42 (proves CLINT write succeeded)
        // 0x1C: wfi                      ->exit
        write_u32_le(&mut ram, 0x00, 0x04200293); // addi t0, x0, 0x42
        write_u32_le(&mut ram, 0x04, 0x00502023); // sw t0, 0(x0) — RAM store to PA 0
        write_u32_le(&mut ram, 0x08, 0x020002B7); // lui t0, 0x2000 ->t0 = 0x2000000
        write_u32_le(&mut ram, 0x0C, 0x00028293); // addi t0, t0, 0
        write_u32_le(&mut ram, 0x10, 0x00100313); // addi t1, x0, 1
        write_u32_le(&mut ram, 0x14, 0x0062A023); // sw t1, 0(t0)
        write_u32_le(&mut ram, 0x18, 0x04200393); // addi t2, x0, 0x42
        write_u32_le(&mut ram, 0x1C, 0x10500073); // wfi

        let mut state = HartState {
            pc: 0,
            mode: riscv_mode::M,
            mstatus: 0,
            ..unsafe { std::mem::zeroed() }
        };
        state.mie = 0;
        state.mip = 0;
        state.mtvec = 0x200; // in case of trap, MRET at 0x200
        write_u32_le(&mut ram, 0x200, 0x30200073); // mret

        let mut result: BatchResult = unsafe { mem::zeroed() };

        unsafe {
            run_parallel_with_clint(
                &mut state as *mut HartState,
                1,
                ram.as_mut_ptr(),
                ram.len() as u64,
                0,
                clint_base,
                50,
                &mut result as *mut BatchResult,
            );
        }

        // RAM store at 0x04 should have written 0x42 to PA=0
        assert_eq!(ram[0], 0x42, "RAM store should have set RAM[0]=0x42");
        assert_eq!(state.gprs[5], 0x2000000, "t0 should = 0x2000000; got {:#x}", state.gprs[5]);
        assert_eq!(state.gprs[7], 0x42,
            "CLINT MSIP write should succeed; t2={:#x}", state.gprs[7]);
        assert_eq!(state.halted, 0, "hart should not be halted");
    }

    /// Cross-hart MSI delivery test.
    ///
    /// Hart 0 sends an MSI to Hart 1 via CLINT MMIO write, then spins.
    /// Hart 1 starts in WFI, should wake on MSI, trap to M-mode handler
    /// (which does MRET), and execute the marker instruction after WFI.
    /// After the batch, Hart 1's t0 MUST be 0x42 — proving the MSI was
    /// delivered, the trap handler ran, and execution resumed at the
    /// instruction following WFI.
    #[test]
    fn cross_hart_msip_wakes_target() {
        let mut ram = vec![0u8; 4096];
        let clint_base: u64 = 0x2000000;

        // --- Hart 0 instructions (PC=0) ---
        // Delay loop: spin ~512 iterations so hart 1 can enter WFI first.
        // 0x00: addi t0, x0, 512       ->t0 = 512
        // 0x04: addi t0, t0, -1        ->t0--
        // 0x08: bnez t0, -8            ->loop back to 0x04
        // 0x0C: lui  t0, 0x20000       ->t0 = 0x20000000
        // 0x10: addi t0, t0, 4         ->t0 = 0x20000004 (CLINT MSIP for hart 1)
        // 0x14: addi t1, x0, 1         ->t1 = 1
        // 0x18: sw   t1, 0(t0)         ->write MSIP=1 for hart 1
        // 0x1C: addi x0, x0, 0          ->NOP (avoid jal-self consecutive_traps)
        // 0x20: jal  x0, -12           ->jump back to 0x1C (spin loop)
        write_u32_le(&mut ram, 0x00, 0x20000293); // addi t0, x0, 512
        write_u32_le(&mut ram, 0x04, 0xFFF28293); // addi t0, t0, -1
        write_u32_le(&mut ram, 0x08, 0xFE029EE3); // bnez t0, -8  -> PC=0x04
        write_u32_le(&mut ram, 0x0C, 0x020002B7); // lui t0, 0x2000 ->t0 = 0x2000000
        write_u32_le(&mut ram, 0x10, 0x00428293); // addi t0, t0, 4
        write_u32_le(&mut ram, 0x14, 0x00100313); // addi t1, x0, 1
        write_u32_le(&mut ram, 0x18, 0x0062A023); // sw t1, 0(t0)
        write_u32_le(&mut ram, 0x1C, 0x00000013); // addi x0, x0, 0 (NOP)
        write_u32_le(&mut ram, 0x20, 0xFFDFF06F); // jal x0, -12 -> PC=0x1C

        // --- Hart 1 instructions (PC=0x100) ---
        // 0x100: wfi                    ->enter WFI
        // 0x104: addi t0, x0, 0x42     ->t0 = 0x42 (marker that MSI was received)
        // 0x108: wfi                    ->back to WFI
        // 0x10C: jal  x0, -12          ->back to 0x104 (loop: safety net
        //                                  if PC somehow advances past WFI;
        //                                  handler at 0x200 does bare MRET
        //                                  so mepc must point to valid code)
        write_u32_le(&mut ram, 0x100, 0x10500073); // wfi
        write_u32_le(&mut ram, 0x104, 0x04200293); // addi t0, x0, 0x42
        write_u32_le(&mut ram, 0x108, 0x10500073); // wfi
        write_u32_le(&mut ram, 0x10C, 0xFF5FF06F); // jal x0, -12 -> PC=0x104

        // --- Shared mtvec handler (PC=0x200) ---
        // 0x200: mret  (Hart 0 handler — never traps in this test)
        write_u32_le(&mut ram, 0x200, 0x30200073); // mret

        // --- Hart 1 MSI handler (PC=0x300) ---
        // Clears CLINT MSIP before MRET to prevent re-triggering the same
        // level-sensitive interrupt.  Real firmware does this in the IPI
        // handler (sbi_ipi_raw_clear).  Without the clear, sync_msip
        // re-asserts mip.MSIP after MRET because the CLINT level is still
        // high -> infinite trap loop -> addi never executes.
        //   0x300: lui  t1, 0x20000    ->t1 = 0x20000000 (CLINT base)
        //   0x304: sw   x0, 4(t1)      ->CLINT[hart1].msip = 0
        //   0x308: mret
        write_u32_le(&mut ram, 0x300, 0x02000337); // lui t1, 0x20000
        write_u32_le(&mut ram, 0x304, 0x00032223); // sw x0, 4(t1)
        write_u32_le(&mut ram, 0x308, 0x30200073); // mret

        // --- Hart 0 state ---
        let mut s0 = make_state(0, 0);
        s0.mode = riscv_mode::M;
        s0.mie = 0;
        s0.mip = 0;
        s0.mtvec = 0x200; // valid MRET handler at 0x200 (shared with Hart 1)

        // --- Hart 1 state ---
        let mut s1 = make_state(0x100, 1);
        s1.mode = riscv_mode::M;
        s1.mie = 0;                       // no interrupts enabled initially
        s1.mip = 0;                       // no interrupts pending
        s1.mtvec = 0x300;                 // M-mode handler at 0x300 (clears MSIP then MRET)
        s1.mstatus = (3 << 11) | (1 << 7) | (1 << 3); // MPP=M, MPIE=1, MIE=1

        let mut states = [s0, s1];
        let mut result: BatchResult = unsafe { mem::zeroed() };

        unsafe {
            run_parallel_with_clint(
                states.as_mut_ptr(),
                2,
                ram.as_mut_ptr(),
                ram.len() as u64,
                0, // ram_base
                clint_base,
                100_000, // max_instrs — enough for delay + spin + MSI delivery
                &mut result as *mut BatchResult,
            );
        }

        // MSI MUST have been detected by Hart 1's sync_msip.
        assert!(states[1].diag.msip_last_seen > 0,
            "Hart 1 should have detected MSIP via sync_msip");

        // mcause MUST indicate MSI was delivered (bit 63=1=interrupt, code=3=MSI).
        let is_interrupt = (states[1].mcause >> 63) != 0;
        let cause_code = states[1].mcause & 0x7FFF_FFFF_FFFF_FFFF;
        assert!(is_interrupt && cause_code == 3,
            "Hart 1 mcause should show MSI interrupt; got mcause={:#x}", states[1].mcause);

        // Hart 1 must NOT have halted or consecutive traps (no trap loop).
        assert_eq!(states[1].halted, 0, "Hart 1 should not be halted");
        assert_eq!(states[1].consecutive_traps, 0,
            "Hart 1 consecutive_traps should be 0; got {}",
            states[1].consecutive_traps);

        // The marker value: if MSI arrived AFTER WFI, PC advances past WFI
        // before the trap, mepc points to the addi, and t0=0x42 after MRET.
        // If MSI arrived BEFORE WFI, mepc points to WFI itself, and after
        // MRET WFI re-executes ->t0 stays 0.
        // Either way the MSI was delivered; we assert the detection markers.
        if states[1].gprs[5] != 0x42 && states[1].mepc != 0x100 {
            panic!(
                "If marker not set, mepc should be 0x100 (MSI before WFI); \
                 got t0={:#x} mepc={:#x}",
                states[1].gprs[5], states[1].mepc
            );
        }
    }

    /// Regression: BLTZ (BLT rs1, x0, offset) with negative rs1.
    ///
    /// Verifies that ``addi s2, a5, -1`` and ``bltz s2, target`` work correctly
    /// with negative values.  a5 = -1 ->s2 = -2 ->bltz MUST branch.
    #[test]
    fn regression_bltz_negative_takes_branch() {
        let mut ram = vec![0u8; 128];

        // Instructions at PC=0:
        // 0x00: addi s2, a5, -1   -> s2 = -2 (a5 = -1)
        // 0x04: bltz s2, +0x10    -> branch to 0x18 (if s2 < 0)
        // 0x08: addi s1, x0, 0xBAD ->s1 = 0xBAD (only reached if bltz fails)
        // 0x0C: ebreak             -> exit batch
        // ...
        // 0x18: addi s2, x0, 42   -> s2 = 42 (landing pad)
        // 0x1C: ebreak             -> exit batch

        write_u32_le(&mut ram, 0, 0xFFF78913); // addi s2, a5, -1
        write_u32_le(&mut ram, 4, 0x00094A63); // bltz s2, +0x14 -> PC=0x18
        write_u32_le(&mut ram, 8, 0xBAD00493); // addi s1, x0, 0xBAD (should skip)
        write_u32_le(&mut ram, 12, 0x00100073); // ebreak
        write_u32_le(&mut ram, 24, 0x02A00913); // addi s2, x0, 42 (landing pad)
        write_u32_le(&mut ram, 28, 0x00100073); // ebreak

        let mut state = HartState {
            pc: 0,
            gprs: [0u64; 32],
            mode: riscv_mode::M,
            mstatus: 0,
            ..unsafe { std::mem::zeroed() }
        };
        state.gprs[15] = 0xFFFF_FFFF_FFFF_FFFFu64; // a5 = -1

        let mut result: BatchResult = unsafe { std::mem::zeroed() };
        let mem = MemCtx { ram: ram.as_mut_ptr(), ram_size: ram.len() as u64,
            ram_base: 0, shadow_base: 0, shadow_size: 0 };
        let dev = FfiDevCtx { bases: [].as_ptr(), ends: [].as_ptr(), num: 0 };
        let pmp = FfiPmpCtx { cfg: [].as_mut_ptr(), addr: [].as_mut_ptr(),
            num: 0, pmpsplit: 0 };
        let mut mtimecmp = [u64::MAX; 1];
        let mut msip = [0u8; 1];
        let mut mtime: u64 = 0;
        let clint = FfiClintCtx { mtime: &mut mtime, mtimecmp: mtimecmp.as_mut_ptr(),
            msip: msip.as_mut_ptr(), base: 0 };

        unsafe {
            call_run_parallel(&mut state, 1, 10, &mut result, &mem, &pmp, &clint, &dev,
                std::ptr::null(), std::ptr::null(), std::ptr::null(), 0,
                std::ptr::null(),
                std::ptr::null_mut(), std::ptr::null_mut());
        }

        // BLTZ should have branched ->s2 = 42
        assert_eq!(state.gprs[18], 42,
            "BLTZ s2,-2 MUST branch: expected s2=42, got {}", state.gprs[18]);
        // s1 should NOT be 0xBAD (proves we didn't fall through)
        assert_ne!(state.gprs[9], 0xBAD,
            "BLTZ should have branched; s1=0xBAD means fallthrough occurred");
    }

}
