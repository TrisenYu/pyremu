//! Instruction handlers for Phases B–D: Load/Store, System, AMO, Compressed.
//!
//! Each handler receives ``&mut HartState``, decoded fields, and memory access
//! context; it returns the PC advance (0, 2, or 4) or ``EXIT_SENTINEL`` to
//! signal that Python must take over.

use std::cell::Cell;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicU8, Ordering};
use std::sync::Mutex;

#[cfg(feature = "diagnostic")]
use crate::diag;


use crate::csr;
use crate::decode::{decode_compressed, CompressedFields, DecodedFields};
use crate::state::{exit_reason, riscv_mode, BatchResult, HartState};
use crate::translate::{translate_va, TranslateFault, WalkCtx};
use crate::trap::{deliver_illegal_instruction, deliver_trap, exc_code, mcause_val};

// ============================================================
//  CLINT inline context (passed through the dispatch chain)
// ============================================================

// ============================================================
//  Context structs — keep handler signatures compact
// ============================================================

/// PMP (Physical Memory Protection) configuration for the batch.
pub struct PmpCtx {
    pub cfg: *mut u8,
    pub addr: *mut u64,
    pub num: u8,
}

/// Device MMIO address ranges (base + end per device).
pub struct DevCtx {
    pub bases: *const u64,
    pub ends: *const u64,
    pub num: u8,
    /// virtio-blk MMIO base address (0 = no virtio device).
    pub virtio_base: u64,
    /// Mutable pointer to the FFI virtio-blk state that Rust updates inline.
    /// Valid for the duration of one ``run_batch`` / ``run_parallel`` call.
    pub virtio_raw: *mut crate::state::FfiVirtIoCtx,
}

impl DevCtx {
    /// 检查是否有设备将工作延迟到 Python 侧处理 (如 virtio QueueNotify
    /// 仅设 notify_pending=1, 实际 virtqueue 处理在 batch 退出后进行)。
    ///
    /// 返回 true 时, hart 调度器应在进入 WFI 自旋前主动退出 batch,
    /// 否则 ``wfi_check_all_idle`` 的 timer fast-forward 会反复唤醒
    /// hart 处理定时器, virtqueue 永远得不到处理 → 客机 I/O 永久挂起。
    pub fn has_pending_python_work(&self) -> bool {
        if !self.virtio_raw.is_null() {
            let notify = unsafe { (*self.virtio_raw).notify_pending };
            if notify != 0 {
                return true;
            }
        }
        false
    }
}

/// Bundled CLINT state for inline MMIO handling.
///
/// Carries mutable pointers to the per-hart ``msip`` and ``mtimecmp`` arrays
/// shared with Python, plus a pointer to all ``HartState`` s so that MSIP
/// writes targeting a different hart can immediately update that hart's
/// ``mip`` field.
pub struct ClintCtx {
    pub base: u64,
    pub mtime: *mut u64,
    pub mtimecmp: *mut u64,
    pub msip: *mut u8,
    pub states: *mut HartState,
    pub num_harts: u32,
    /// Set when a hart writes MSIP=1 to a *different* hart.  The dispatch
    /// loop checks this flag and yields the current hart's slice early so
    /// the target hart can respond to the IPI within the same batch round.
    ///
    /// For multi-target broadcasts (``sbi_ipi_send_many`` with N>1 targets),
    /// the sender yields after the first MSIP write, then resumes in the
    /// next round-robin round to send the remaining MSIPs.  This is correct
    /// because OpenSBI's hartmask is progressively cleared — no target is
    /// lost.  N targets may take N rounds, but each round consumes ~1
    /// instruction on the sender instead of the full 512-instr spin-wait.
    pub yield_for_ipi: Cell<bool>,
    /// Hart ID of the most recent cross-hart MSIP sender; this hart receives
    /// short slices so the receiver can complete IPI-triggered work (TLB
    /// flush, sync counter decrement) before the sender's spin-wait resumes.
    pub ipi_sender_hart: Cell<u8>,
    /// Remaining rounds of short slices for *ipi_sender_hart*.  Decremented
    /// once per round-robin round until zero, then the sender resumes full
    /// slices.  Reset to a fresh count each time a new MSIP is sent.
    pub ipi_sender_rounds: Cell<u8>,
}

// Re-export from csr.rs
pub use crate::csr::EXIT_SENTINEL;

// ============================================================
//  SBI extension / function IDs (RISC-V SBI spec v2.0)
// ============================================================

#[allow(dead_code)]
mod sbi_eid {
    pub const TIME: u64 = 0x54494D45; // "TIME" — timer extension
    pub const IPI: u64 = 0x735049; // "IPI"  — inter-processor interrupt
    pub const RFNC: u64 = 0x52464E43; // "RFNC" — remote fence (TLB shootdown)
    pub const HSM: u64 = 0x48534D; // "HSM"  — hart state management
}

mod sbi_fid_time {
    pub const SET_TIMER: u64 = 0;
}

mod sbi_fid_ipi {
    pub const SEND_IPI: u64 = 0;
}

// ============================================================
//  GPR read/write helpers
// ============================================================

#[inline]
fn read_gpr(state: &HartState, rs: u8) -> u64 {
    if rs == 0 {
        0
    } else {
        state.gprs[rs as usize]
    }
}

#[inline]
fn write_gpr(state: &mut HartState, rd: u8, val: u64) {
    if rd != 0 {
        state.gprs[rd as usize] = val;
    }
}

/// Sign-extend 32-bit -> 64-bit.
#[inline]
fn sext32(val: u64) -> u64 {
    ((val as i32) as i64) as u64
}

/// Sign-extend from *bits* to unsigned 64-bit.
#[inline]
fn sext(val: u64, bits: u32) -> u64 {
    let half = 1u64 << (bits - 1);
    (val & (half - 1)).wrapping_sub(val & half) & u64::MAX
}

/// C1 ALU register-register operations (sf=0b11): SUB/XOR/OR/AND/SUBW/ADDW.
/// Returns ``None`` for illegal encodings (caller delivers IllInstr).
#[inline]
fn c1_alu_reg_op(v1: u64, v2: u64, bit12: u8, bit65: u8) -> Option<u64> {
    if bit12 == 0 {
        Some(match bit65 {
            0b00 => v1.wrapping_sub(v2),
            0b01 => v1 ^ v2,
            0b10 => v1 | v2,
            _ => v1 & v2,
        })
    } else {
        match bit65 {
            0b00 => Some(sext32((v1.wrapping_sub(v2)) & 0xFFFF_FFFF)),
            0b01 => Some(sext32((v1.wrapping_add(v2)) & 0xFFFF_FFFF)),
            _ => None,
        }
    }
}

// ============================================================
//  Memory read/write helpers
// ============================================================

/// Read *size* bytes from physical address *pa* in RAM. Little-endian.
#[inline]
fn ram_read(ctx: &WalkCtx, pa: u64, size: u8) -> u64 {
    let off = super::mem::ram_offset_inline(
        pa,
        size as u32,
        ctx.ram_base,
        ctx.ram_size,
        ctx.shadow_base,
        ctx.shadow_size,
    );
    if off.is_none() {
        return 0;
    }
    let ptr = ctx.ram as *mut u8;
    let ptr = unsafe { ptr.add(off.unwrap() as usize) };
    match size {
        1 => unsafe { *ptr as u64 },
        2 => {
            let b0 = unsafe { *ptr } as u64;
            let b1 = unsafe { *ptr.add(1) } as u64;
            b0 | (b1 << 8)
        }
        4 if (pa & 3) == 0 => {
            // Aligned 4-byte: atomic Acquire load pairs with Release stores
            // from other hart threads (ram_write / ram_write_raw).
            let a = unsafe { &*(ptr as *const AtomicU32) };
            a.load(Ordering::Acquire) as u64
        }
        4 => {
            let b0 = unsafe { *ptr } as u64;
            let b1 = unsafe { *ptr.add(1) } as u64;
            let b2 = unsafe { *ptr.add(2) } as u64;
            let b3 = unsafe { *ptr.add(3) } as u64;
            b0 | (b1 << 8) | (b2 << 16) | (b3 << 24)
        }
        8 if (pa & 7) == 0 => {
            // Aligned 8-byte: atomic Acquire load pairs with Release stores
            // from other hart threads.
            let a = unsafe { &*(ptr as *const AtomicU64) };
            a.load(Ordering::Acquire)
        }
        8 => {
            let b0 = unsafe { *ptr } as u64;
            let b1 = unsafe { *ptr.add(1) } as u64;
            let b2 = unsafe { *ptr.add(2) } as u64;
            let b3 = unsafe { *ptr.add(3) } as u64;
            let b4 = unsafe { *ptr.add(4) } as u64;
            let b5 = unsafe { *ptr.add(5) } as u64;
            let b6 = unsafe { *ptr.add(6) } as u64;
            let b7 = unsafe { *ptr.add(7) } as u64;
            b0 | (b1 << 8)
                | (b2 << 16)
                | (b3 << 24)
                | (b4 << 32)
                | (b5 << 40)
                | (b6 << 48)
                | (b7 << 56)
        }
        _ => 0,
    }
}

/// Write *size* bytes of *val* to physical address *pa* in RAM. Little-endian.
///
/// Aligned 4- and 8-byte writes use atomic stores with Release ordering so
/// that loads on other hart threads (which use Acquire loads via
/// ``ram_read`` / ``ram_read_raw``) are guaranteed to see the full write.
/// Without this, concurrent compressed-store + regular-load pairs on the
/// same address can observe torn writes.
#[inline]
fn ram_write(ctx: &WalkCtx, pa: u64, val: u64, size: u8) -> bool {
    let off = super::mem::ram_offset_inline(
        pa,
        size as u32,
        ctx.ram_base,
        ctx.ram_size,
        ctx.shadow_base,
        ctx.shadow_size,
    );
    if off.is_none() {
        return false;
    }
    let ptr = ctx.ram as *mut u8;
    let ptr = unsafe { ptr.add(off.unwrap() as usize) };
    match size {
        1 => unsafe {
            *ptr = val as u8;
        },
        2 => {
            unsafe {
                *ptr = val as u8;
            }
            unsafe {
                *ptr.add(1) = (val >> 8) as u8;
            }
        }
        4 if (pa & 3) == 0 => {
            // Aligned 4-byte: atomic Release store pairs with Acquire
            // loads from other hart threads.
            let a = unsafe { &*(ptr as *const AtomicU32) };
            a.store(val as u32, Ordering::Release);
        }
        4 => {
            unsafe {
                *ptr = val as u8;
            }
            unsafe {
                *ptr.add(1) = (val >> 8) as u8;
            }
            unsafe {
                *ptr.add(2) = (val >> 16) as u8;
            }
            unsafe {
                *ptr.add(3) = (val >> 24) as u8;
            }
        }
        8 if (pa & 7) == 0 => {
            // Aligned 8-byte: atomic Release store pairs with Acquire
            // loads from other hart threads.
            let a = unsafe { &*(ptr as *const AtomicU64) };
            a.store(val, Ordering::Release);
        }
        8 => {
            unsafe {
                *ptr = val as u8;
            }
            unsafe {
                *ptr.add(1) = (val >> 8) as u8;
            }
            unsafe {
                *ptr.add(2) = (val >> 16) as u8;
            }
            unsafe {
                *ptr.add(3) = (val >> 24) as u8;
            }
            unsafe {
                *ptr.add(4) = (val >> 32) as u8;
            }
            unsafe {
                *ptr.add(5) = (val >> 40) as u8;
            }
            unsafe {
                *ptr.add(6) = (val >> 48) as u8;
            }
            unsafe {
                *ptr.add(7) = (val >> 56) as u8;
            }
        }
        _ => return false,
    }
    true
}

/// Check if an address falls within a device MMIO range.
/// Returns true if the address should be handled by Python.
#[inline]
fn is_device_addr(pa: u64, dev: &DevCtx) -> bool {
    for i in 0..dev.num as usize {
        let base = unsafe { *dev.bases.add(i) };
        let end = unsafe { *dev.ends.add(i) };
        if pa >= base && pa < end {
            return true;
        }
    }
    false
}

// ============================================================
//  CLINT inline MMIO handler
// ============================================================

/// Descriptor for a memory access that may be handled by CLINT inline.
///
/// Descriptor for a memory access that may be handled by CLINT inline.
///
/// All fields are ``u64`` to avoid byte-alignment or padding ambiguity.
/// ``is_write``: 0 = read, 1 = write.
#[repr(C)]
#[allow(dead_code)]
struct MemAccess {
    pa: u64,
    is_write: u64,
    write_data: u64,
    size: u64,
}

impl MemAccess {
    fn read(pa: u64, size: u8) -> Self {
        Self {
            pa,
            is_write: 0,
            write_data: 0,
            size: size as u64,
        }
    }
    fn write(pa: u64, size: u8, data: u64) -> Self {
        Self {
            pa,
            is_write: 1,
            write_data: data,
            size: size as u64,
        }
    }
}

/// Write MSIP for a target hart, updating its MIP and setting the
/// cross-hart yield flag when the target is a different hart.
///
/// Shared by ``try_handle_clint`` and the SBI IPI fast path so that
/// yield-for-IPI logic is in one place.
#[inline]
fn clint_write_msip(clint: &ClintCtx, target: usize, current: usize, val: u8) {
    if target >= clint.num_harts as usize {
        return;
    }
    // In concurrent mode the msip pointer was transmuted from *const AtomicU8.
    // Use an atomic store so release semantics are visible to the target
    // hart's sync_msip acquire load.
    if clint.states.is_null() {
        unsafe {
            let atomic_msip = clint.msip as *const AtomicU8;
            (*atomic_msip.add(target)).store(val, Ordering::Release);
        }
        return;
    }
    unsafe {
        *clint.msip.add(target) = val;
    }
    let ts = unsafe { &mut *clint.states.add(target) };
    if val == 0 {
        ts.mip &= !(1 << 3);
        ts.diag.clint_msip_clr = ts.diag.clint_msip_clr.wrapping_add(1);
		return;
    }
	ts.mip |= 1 << 3;
	ts.diag.clint_msip_set = ts.diag.clint_msip_set.wrapping_add(1);
	// Cross-hart IPI: request early slice yield so the target can
	// respond within the same round-robin round instead of forcing the
	// sender to spin-wait (e.g. tlb_sync) for the full 512-instr slice.
	if target != current {
		clint.yield_for_ipi.set(true);
		// Give the sender short slices for several rounds so the
		// receiver gets proportionally more CPU time to complete
		// the IPI-triggered work (e.g., TLB flush + sync decrement)
		// before the sender resumes its spin-wait.
		clint.ipi_sender_hart.set(current as u8);
		clint.ipi_sender_rounds.set(16);
	}
}

/// Try to handle a CLINT MMIO access inline inside the native batch engine.
///
/// Returns:
/// - ``Some(data)`` for a successful read (caller writes *data* to the
///   destination register).
/// - ``Some(0)`` for a successful write.
/// - ``None`` if *pa* is not within the CLINT address range (caller falls
///   through to the normal ``is_device_addr`` / MMIO-exit path).
fn try_handle_clint(access: &MemAccess, state: &HartState, clint: &ClintCtx) -> Option<u64> {
    // Snapshot fields to locals before any operation — defensive copy
    // in case writes through clint.states could alias with the caller's
    // stack temporary.
    let pa = access.pa;
    let is_write = access.is_write;
    let write_data = access.write_data;

    if clint.base == 0 {
        return None;
    }
    let offset = pa.wrapping_sub(clint.base);

    if offset < 0x4000 {
        let hart_id = (offset / 4) as usize;
        if hart_id >= clint.num_harts as usize {
            return Some(0);
        }
        if is_write == 0 {
            let val = unsafe { *clint.msip.add(hart_id) } as u64 & 1;
            return Some(val);
        }
        clint_write_msip(
            clint,
            hart_id,
            state.mhartid as usize,
            (write_data & 1) as u8,
        );
        return Some(0);
    } else if offset < 0xBFF8 {
        let hart_id = usize::try_from((offset - 0x4000) / 8).unwrap_or(usize::MAX);
        if hart_id >= clint.num_harts as usize {
            return Some(0);
        }
        if is_write == 0 {
            return Some(unsafe { *clint.mtimecmp.add(hart_id) });
        }
        unsafe {
            *clint.mtimecmp.add(hart_id) = write_data;
        }
        if hart_id < clint.num_harts as usize && !clint.states.is_null() {
            let t = unsafe { &mut *clint.states.add(hart_id) };
            t.diag.clint_mtc_wr = t.diag.clint_mtc_wr.wrapping_add(1);
        }
        return Some(0);
    } else if offset < 0xC000 && is_write == 0 {
        return Some(unsafe { *clint.mtime });
    }
    None
}

// ============================================================
//  virtio-blk inline MMIO handler
// ============================================================

/// Device features advertised by the virtio-blk device (64-bit).
///
/// VIRTIO_F_RING_EVENT_IDX (bit 29) and VIRTIO_F_RING_INDIRECT_DESC (bit 28)
/// are deliberately NOT advertised: the Python-side virtqueue processor
/// (_process_descriptor_chain) does not implement avail_event writes or
/// indirect-descriptor-table traversal.  Advertising either feature causes
/// the guest driver to take code paths that break on our device.
/// Without them the driver falls back to flags-based notification
/// (VRING_USED_F_NO_NOTIFY) and direct descriptor chains — both of which
/// work correctly.
const VIRTIO_DEVICE_FEATURES: u64 = 1u64 << 32; // VIRTIO_F_VERSION_1

/// Try to handle a virtio-blk MMIO access inline inside the native batch engine.
///
/// The virtio base address and mutable state pointer are embedded in ``DevCtx``
/// so that no additional parameters need to be threaded through the handler
/// call chain.  Returns:
/// - ``Some(data)`` for a successful read (caller writes *data* to the
///   destination register).
/// - ``Some(0)`` for a successful write.
/// - ``None`` if *pa* is not within the virtio MMIO range, or if the access
///   requires Python-side handling (``QueueNotify`` → ``notify_pending=1``,
///   processed at the next natural batch boundary).
pub fn try_handle_virtio(
    pa: u64,
    is_write: bool,
    write_data: u64,
    _size: u8,
    dev: &DevCtx,
) -> Option<u64> {
    if dev.virtio_base == 0 {
        return None;
    }
    let offset = pa.wrapping_sub(dev.virtio_base);
    if offset >= 0x200 {
        return None;
    }
    if is_write {
        virtio_write(offset, write_data, dev.virtio_raw)
    } else {
        virtio_read(offset, dev.virtio_raw)
    }
}

/// Handle virtio-blk MMIO writes inline.  See ``try_handle_virtio`` for the
/// return-value contract.
fn virtio_write(offset: u64, write_data: u64, raw: *mut crate::state::FfiVirtIoCtx) -> Option<u64> {
    match offset {
        // DeviceFeaturesSel (0x014) — page selector
        0x014 => unsafe {
            (*raw).device_features_sel = write_data as u32;
            Some(0)
        },
        // DriverFeatures (0x020) — guest features for selected page
        0x020 => unsafe {
            let sel = (*raw).driver_features_sel;
            let mask = (write_data as u64 & 0xFFFF_FFFF) << (sel * 32);
            (*raw).driver_features =
                ((*raw).driver_features & !(0xFFFF_FFFFu64 << (sel * 32))) | mask;
            Some(0)
        },
        // DriverFeaturesSel (0x024) — page selector
        0x024 => unsafe {
            (*raw).driver_features_sel = write_data as u32;
            Some(0)
        },
        // QueueSel (0x030)
        0x030 => unsafe {
            (*raw).queue_sel = write_data as u32;
            Some(0)
        },
        // QueueNum (0x038) — capped at queue_num_max
        0x038 => unsafe {
            (*raw).queue_num = core::cmp::min(write_data as u32, (*raw).queue_num_max);
            Some(0)
        },
        // QueueReady (0x044)
        0x044 => unsafe {
            (*raw).queue_ready = if write_data != 0 { 1 } else { 0 };
            Some(0)
        },
        // QueueNotify (0x050) — set notify_pending so Python processes
        // the queue at the next *natural* batch boundary (max-instrs
        // or all-idle WFI) instead of forcing an immediate MMIO exit.
        // The virtqueue descriptors live in guest RAM and are still
        // valid when Python eventually reads them.
        0x050 => {
            unsafe {
                (*raw).notify_pending = 1;
            }
            Some(0) // inline-handled: do NOT exit batch
        }
        // InterruptStatus (0x060) — writing sets bits (unusual, but handle inline)
        0x060 => unsafe {
            (*raw).interrupt_status |= write_data as u32;
            Some(0)
        },
        // InterruptACK (0x064) — clear bits; flag PLIC lowering if all zero
        0x064 => unsafe {
            let old = (*raw).interrupt_status;
            (*raw).interrupt_status = old & !(write_data as u32);
            if (*raw).interrupt_status == 0 && old != 0 {
                (*raw).irq_maybe_lower = 1;
            }
            Some(0)
        },
        // Status (0x070) — writing 0 resets the device
        0x070 => {
            if write_data != 0 {
                unsafe {
                    (*raw).status = write_data as u32;
                }
                return Some(0);
            }
            unsafe {
                (*raw).status = 0;
                (*raw).device_features_sel = 0;
                (*raw).driver_features_sel = 0;
                (*raw).driver_features = 0;
                (*raw).queue_sel = 0;
                (*raw).queue_ready = 0;
                (*raw).interrupt_status = 0;
                (*raw).queue_desc = 0;
                (*raw).queue_driver = 0;
                (*raw).queue_device = 0;
            }
            Some(0)
        }
        // QueueDescLow / High (0x080 / 0x084)
        0x080 => unsafe {
            (*raw).queue_desc =
                ((*raw).queue_desc & 0xFFFF_FFFF_0000_0000) | (write_data as u64 & 0xFFFF_FFFF);
            Some(0)
        },
        0x084 => unsafe {
            (*raw).queue_desc =
                ((*raw).queue_desc & 0xFFFF_FFFF) | ((write_data as u64 & 0xFFFF_FFFF) << 32);
            Some(0)
        },
        // QueueDriverLow / High (0x090 / 0x094)
        0x090 => unsafe {
            (*raw).queue_driver =
                ((*raw).queue_driver & 0xFFFF_FFFF_0000_0000) | (write_data as u64 & 0xFFFF_FFFF);
            Some(0)
        },
        0x094 => unsafe {
            (*raw).queue_driver =
                ((*raw).queue_driver & 0xFFFF_FFFF) | ((write_data as u64 & 0xFFFF_FFFF) << 32);
            Some(0)
        },
        // QueueDeviceLow / High (0x0A0 / 0x0A4)
        0x0A0 => unsafe {
            (*raw).queue_device =
                ((*raw).queue_device & 0xFFFF_FFFF_0000_0000) | (write_data as u64 & 0xFFFF_FFFF);
            Some(0)
        },
        0x0A4 => unsafe {
            (*raw).queue_device =
                ((*raw).queue_device & 0xFFFF_FFFF) | ((write_data as u64 & 0xFFFF_FFFF) << 32);
            Some(0)
        },
        // Config space (0x100+) — read-only in hardware; writes are ignored.
        // Other undefined offsets — ignored (writes have no effect).
        _ => Some(0),
    }
}

/// Handle virtio-blk MMIO reads inline.  See ``try_handle_virtio`` for the
/// return-value contract.
fn virtio_read(offset: u64, raw: *mut crate::state::FfiVirtIoCtx) -> Option<u64> {
    match offset {
        // MagicValue (0x000)
        0x000 => Some(0x74726976),
        // Version (0x004)
        0x004 => Some(0x2),
        // DeviceID (0x008) — 2 = block device
        0x008 => Some(0x2),
        // VendorID (0x00C)
        0x00C => Some(0x0),
        // DeviceFeatures (0x010) — page-selected
        0x010 => {
            let sel = unsafe { (*raw).device_features_sel };
            Some((VIRTIO_DEVICE_FEATURES >> (sel * 32)) & 0xFFFF_FFFF)
        }
        // QueueNumMax (0x034)
        0x034 => unsafe { Some((*raw).queue_num_max as u64) },
        // InterruptStatus (0x060)
        0x060 => unsafe { Some((*raw).interrupt_status as u64) },
        // Status (0x070)
        0x070 => unsafe { Some((*raw).status as u64) },
        // ConfigGeneration (0x0FC)
        0x0FC => Some(0),
        // Config space (0x100+)
        off if off >= 0x100 => {
            let local = off - 0x100;
            match local {
                // Capacity (u64, low 32 at 0x000, high 32 at 0x004)
                0x000 => unsafe { Some((*raw).capacity & 0xFFFF_FFFF) },
                0x004 => unsafe { Some(((*raw).capacity >> 32) & 0xFFFF_FFFF) },
                _ => Some(0),
            }
        }
        _ => Some(0),
    }
}

// ============================================================
//  PMP check stub
// ============================================================

/// Check PMP for a physical address access.
/// Returns true if access is allowed.
#[inline]
pub(crate) fn pmp_ok(
    state: &HartState,
    pa: u64,
    size: u32,
    is_write: bool,
    is_execute: bool,
    pmp: &PmpCtx,
) -> bool {
    // If no PMP entries, any mode can access any physical address.
    // RISC-V spec: PMP with zero entries imposes no restrictions.
    if pmp.num == 0 {
        return true;
    }
    // Quick check: M-mode with MPRV=0 bypasses PMP
    if state.mode == riscv_mode::M {
        let mprv = (state.mstatus >> 17) & 1;
        if mprv == 0 {
            return true;
        }
    }
    // Delegate to the full PMP check function
    crate::pmp::pmp_check(
        pmp.cfg,
        pmp.addr,
        pmp.num,
        pa,
        size,
        if is_write { 1 } else { 0 },
        if is_execute { 1 } else { 0 },
        state.mode,
        state.mstatus,
        state.pmpsplit,
        state.mdid,
    ) != 0
}

// ============================================================
//  Load handlers
// ============================================================

pub fn handle_load(
    state: &mut HartState,
    f: &DecodedFields,
    instr: u32,
    result: &mut BatchResult,
    ctx: &WalkCtx,
    pmp: &PmpCtx,
    dev: &DevCtx,
    clint: &ClintCtx,
) -> u64 {
    let base = read_gpr(state, f.rs1);
    let va = base.wrapping_add(f.imm12_se);
    let (size, signed) = match f.func3 {
        0b000 => (1, true),  // LB
        0b001 => (2, true),  // LH
        0b010 => (4, true),  // LW
        0b011 => (8, false), // LD
        0b100 => (1, false), // LBU
        0b101 => (2, false), // LHU
        0b110 => (4, false), // LWU
        _ => {
            deliver_illegal_instruction(state, instr as u64, result);
            return 0;
        }
    };

    // Alignment check: misaligned loads are not supported
    if va & (size as u64 - 1) != 0 {
        deliver_trap(
            state,
            mcause_val(exc_code::LD_MISALIGNED, false),
            va,
            result,
        );
        return 0;
    }

    // Translate VA -> PA
    let tr = match translate_va(state, ctx, va, false, false) {
        Ok(t) => t,
        Err(TranslateFault::PageFault(cause)) => {
            deliver_trap(state, mcause_val(cause, false), va, result);
            return 0;
        }
        Err(TranslateFault::AccessFault) => {
            deliver_trap(
                state,
                mcause_val(exc_code::LD_ACCESS_FAULT, false),
                va,
                result,
            );
            return 0;
        }
    };

    // PMP check
    if !pmp_ok(state, tr.pa, size as u32, false, false, pmp) {
        deliver_trap(
            state,
            mcause_val(exc_code::LD_ACCESS_FAULT, false),
            va,
            result,
        );
        return 0;
    }

    // CLINT inline check — handle MSIP/MTIMECMP/MTIME reads directly
    // to avoid expensive MMIO exits during SMP bringup.
    if let Some(data) = try_handle_clint(&MemAccess::read(tr.pa, size), state, clint) {
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

    // virtio-blk inline check — handle all MMIO registers except QueueNotify
    // to avoid expensive batch exits during device probe.
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

    // MMIO check (non-CLINT, non-virtio devices)
    if is_device_addr(tr.pa, dev) {
        result.exit_reason = exit_reason::MMIO;
        result.exit_instr = instr;
        return EXIT_SENTINEL;
    }

    // Read from RAM
    let val = ram_read(ctx, tr.pa, size);

    // Sign-extend if needed
    let result_val = if signed {
        match size {
            1 => sext(val, 8),
            2 => sext(val, 16),
            4 => sext(val, 32),
            _ => val,
        }
    } else {
        // Zero-extend for unsigned loads (already zero-extended by ram_read)
        val
    };

    write_gpr(state, f.rd, result_val);
    4
}

// ============================================================
//  Store handlers
// ============================================================

pub fn handle_store(
    state: &mut HartState,
    f: &DecodedFields,
    instr: u32,
    result: &mut BatchResult,
    ctx: &WalkCtx,
    pmp: &PmpCtx,
    dev: &DevCtx,
    clint: &ClintCtx,
) -> u64 {
    let base = read_gpr(state, f.rs1);
    let va = base.wrapping_add(f.imm_s);
    let size: u8 = match f.func3 {
        0b000 => 1, // SB
        0b001 => 2, // SH
        0b010 => 4, // SW
        0b011 => 8, // SD
        _ => {
            deliver_illegal_instruction(state, instr as u64, result);
            return 0;
        }
    };

    // Alignment check
    if va & (size as u64 - 1) != 0 {
        deliver_trap(
            state,
            mcause_val(exc_code::ST_MISALIGNED, false),
            va,
            result,
        );
        return 0;
    }

    // Translate VA -> PA
    let tr = match translate_va(state, ctx, va, true, false) {
        Ok(t) => t,
        Err(TranslateFault::PageFault(cause)) => {
            deliver_trap(state, mcause_val(cause, false), va, result);
            return 0;
        }
        Err(TranslateFault::AccessFault) => {
            deliver_trap(
                state,
                mcause_val(exc_code::ST_ACCESS_FAULT, false),
                va,
                result,
            );
            return 0;
        }
    };

    // PMP check
    if !pmp_ok(state, tr.pa, size as u32, true, false, pmp) {
        deliver_trap(
            state,
            mcause_val(exc_code::ST_ACCESS_FAULT, false),
            va,
            result,
        );
        return 0;
    }

    let val = read_gpr(state, f.rs2);

    // CLINT inline check — handle MSIP/MTIMECMP writes directly
    // to avoid expensive MMIO exits during SMP bringup.
    if let Some(_) = try_handle_clint(&MemAccess::write(tr.pa, size, val), state, clint) {
        return 4;
    }

    // virtio-blk inline check — handle all MMIO registers except QueueNotify
    // to avoid expensive batch exits during device probe.
    if let Some(_) = try_handle_virtio(tr.pa, true, val, size, dev) {
        return 4;
    }

    // MMIO check (non-CLINT, non-virtio devices)
    if is_device_addr(tr.pa, dev) {
        result.exit_reason = exit_reason::MMIO;
        result.exit_instr = instr;
        return EXIT_SENTINEL;
    }

    ram_write(ctx, tr.pa, val, size);

    4
}

// ============================================================
//  F/D floating-point load / store
// ============================================================

/// mstatus.FS 掩码 (bits[14:13])。
const MSTATUS_FS_LS: u64 = 0b11 << 13;
/// mstatus.SD (bit 63)。
const MSTATUS_SD_LS: u64 = 1 << 63;
/// 单精度 NaN-boxing 掩码。
const NANBOX_S_LS: u64 = 0xFFFF_FFFF_0000_0000;

/// FLW / FLD (opcode 0x07): 从内存加载到 FPR。
/// FLW 结果 NaN-boxed; FLD 加载完整 64 位。
pub fn handle_fp_load(
    state: &mut HartState,
    f: &DecodedFields,
    instr: u32,
    result: &mut BatchResult,
    ctx: &WalkCtx,
    pmp: &PmpCtx,
    dev: &DevCtx,
    clint: &ClintCtx,
) -> u64 {
    if (state.mstatus & MSTATUS_FS_LS) == 0 {
        deliver_illegal_instruction(state, instr as u64, result);
        return 0;
    }
    let base = read_gpr(state, f.rs1);
    let va = base.wrapping_add(f.imm12_se);
    let size: u8 = match f.func3 {
        0b010 => 4, // FLW
        0b011 => 8, // FLD
        _ => {
            deliver_illegal_instruction(state, instr as u64, result);
            return 0;
        }
    };

    if va & (size as u64 - 1) != 0 {
        deliver_trap(
            state,
            mcause_val(exc_code::LD_MISALIGNED, false),
            va,
            result,
        );
        return 0;
    }

    let tr = match translate_va(state, ctx, va, false, false) {
        Ok(t) => t,
        Err(TranslateFault::PageFault(cause)) => {
            deliver_trap(state, mcause_val(cause, false), va, result);
            return 0;
        }
        Err(TranslateFault::AccessFault) => {
            deliver_trap(
                state,
                mcause_val(exc_code::LD_ACCESS_FAULT, false),
                va,
                result,
            );
            return 0;
        }
    };

    if !pmp_ok(state, tr.pa, size as u32, false, false, pmp) {
        deliver_trap(
            state,
            mcause_val(exc_code::LD_ACCESS_FAULT, false),
            va,
            result,
        );
        return 0;
    }

    // virtio-blk inline check before generic MMIO exit.
    if let Some(data) = try_handle_virtio(tr.pa, false, 0, size, dev) {
        let boxed = if size == 4 { NANBOX_S_LS | data } else { data };
        state.fprs[f.rd as usize] = boxed;
        state.mstatus |= MSTATUS_FS_LS | MSTATUS_SD_LS;
        return 4;
    }

    // 浮点数据极少落在 CLINT; 设备地址退回 Python MMIO 处理。
    if is_device_addr(tr.pa, dev) {
        result.exit_reason = exit_reason::MMIO;
        result.exit_instr = instr;
        return EXIT_SENTINEL;
    }
    // CLINT 命中也退回 Python (FP 从 MMIO 加载罕见, 不做 inline)。
    let _ = clint;

    let val = ram_read(ctx, tr.pa, size);
    let boxed = if size == 4 { NANBOX_S_LS | val } else { val };
    state.fprs[f.rd as usize] = boxed;
    state.mstatus |= MSTATUS_FS_LS | MSTATUS_SD_LS;
    4
}

/// FSW / FSD (opcode 0x27): 将 FPR 存入内存。
/// FSW 存低 32 位; FSD 存完整 64 位。
pub fn handle_fp_store(
    state: &mut HartState,
    f: &DecodedFields,
    instr: u32,
    result: &mut BatchResult,
    ctx: &WalkCtx,
    pmp: &PmpCtx,
    dev: &DevCtx,
    clint: &ClintCtx,
) -> u64 {
    if (state.mstatus & MSTATUS_FS_LS) == 0 {
        deliver_illegal_instruction(state, instr as u64, result);
        return 0;
    }
    let base = read_gpr(state, f.rs1);
    let va = base.wrapping_add(f.imm_s);
    let size: u8 = match f.func3 {
        0b010 => 4, // FSW
        0b011 => 8, // FSD
        _ => {
            deliver_illegal_instruction(state, instr as u64, result);
            return 0;
        }
    };

    if va & (size as u64 - 1) != 0 {
        deliver_trap(
            state,
            mcause_val(exc_code::ST_MISALIGNED, false),
            va,
            result,
        );
        return 0;
    }

    let tr = match translate_va(state, ctx, va, true, false) {
        Ok(t) => t,
        Err(TranslateFault::PageFault(cause)) => {
            deliver_trap(state, mcause_val(cause, false), va, result);
            return 0;
        }
        Err(TranslateFault::AccessFault) => {
            deliver_trap(
                state,
                mcause_val(exc_code::ST_ACCESS_FAULT, false),
                va,
                result,
            );
            return 0;
        }
    };

    if !pmp_ok(state, tr.pa, size as u32, true, false, pmp) {
        deliver_trap(
            state,
            mcause_val(exc_code::ST_ACCESS_FAULT, false),
            va,
            result,
        );
        return 0;
    }

    // FSW 存低 32 位 (无需 unbox — 直接取寄存器低位)。
    let val = state.fprs[f.rs2 as usize];

    // virtio-blk inline check before generic MMIO exit.
    if let Some(_) = try_handle_virtio(tr.pa, true, val, size, dev) {
        return 4;
    }

    if is_device_addr(tr.pa, dev) {
        result.exit_reason = exit_reason::MMIO;
        result.exit_instr = instr;
        return EXIT_SENTINEL;
    }
    let _ = clint;
    ram_write(ctx, tr.pa, val, size);
    4
}

// ============================================================
//  System instruction handler
// ============================================================

/// Dispatch privileged instructions: ECALL, EBREAK, MRET, SRET, WFI, SFENCE.VMA.
/// Called from ``handle_system`` when func3 == 0b000.
#[inline]
fn dispatch_privileged(
    state: &mut HartState,
    func12: u16,
    instr: u32,
    result: &mut BatchResult,
    clint: &ClintCtx,
    ctx: &WalkCtx,
) -> u64 {
    match func12 {
        0 => {
            // ---- SBI fast paths (avoid ECALL -> Python round-trip) ----
            // SBI calling convention: a7=x17=EID, a6=x16=FID.
            let a7 = read_gpr(state, 17);
            let a6 = read_gpr(state, 16);

            // SBI_TIME set_timer: write stime_value -> mtimecmp.
            if a7 == sbi_eid::TIME && a6 == sbi_fid_time::SET_TIMER {
                let stime_val = read_gpr(state, 10); // a0
                let hid = state.mhartid as usize;
                if hid < clint.num_harts as usize {
                    unsafe {
                        *clint.mtimecmp.add(hid) = stime_val;
                    }
                }
                state.stimecmp = stime_val;
                state.gprs[10] = 0; // SBI_SUCCESS
                return 4;
            }

            // SBI_IPI send_ipi: write MSIP to each target hart in hart_mask.
            if a7 == sbi_eid::IPI && a6 == sbi_fid_ipi::SEND_IPI {
                let hart_mask = read_gpr(state, 10); // a0
                let mask_base = read_gpr(state, 11); // a1
                if mask_base != 0 {
                    // mask_base != 0: fall through to Python.
                    result.exit_reason = exit_reason::ECALL;
                    result.exit_instr = instr;
                    return EXIT_SENTINEL;
                }
                // mask_base == 0 is the common case (Linux uses simple bitmap).
                let cur = state.mhartid as usize;
                for t in 0..clint.num_harts as u64 {
                    if hart_mask & (1u64 << t) == 0 {
                        continue;
                    }
                    clint_write_msip(clint, t as usize, cur, 1);
                }
                state.gprs[10] = 0; // SBI_SUCCESS
                return 4;
            }

            // ---- Generic ECALL — deliver trap inline, stay in batch ----
            // Instead of exiting to Python (one FFI round-trip per SBI call),
            // deliver the trap inline and let the M-mode handler execute within
            // the same batch.  CLINT is already inlined, and MRET/SRET return
            // the hart to the original mode.  This eliminates the dominant batch-
            // exit cause during Linux boot, where hart 0 makes thousands of SBI
            // ecalls (TIME, IPI, RFENCE, DBCN, etc.).
            //
            // Set PYREMU_INLINE_ECALL=0 to restore the old exit-to-Python
            // behaviour for differential testing.
            let ecall_cause = match state.mode {
                riscv_mode::U => mcause_val(exc_code::ECALL_UMODE, false),
                riscv_mode::S => mcause_val(exc_code::ECALL_SMODE, false),
                _ => mcause_val(exc_code::ECALL_MMODE, false),
            };
            deliver_trap(state, ecall_cause, 0, result);
            // deliver_trap sets PC -> mtvec/stvec and mode -> M/S.
            // Return 0 so the dispatch loop continues execution from the
            // trap handler entry point, all within the same batch.
            0
        }
        1 => {
            // EBREAK — if this is a semihosting sequence, handle it inline;
            // otherwise treat as NOP (no external debugger attached).
            if let Some(advance) = try_semihosting(state, ctx, state.pc) {
                return advance;
            }
            4
        }
        0x302 => {
            // MRET
            let mpp = (state.mstatus >> 11) & 0x3;
            let mpie = (state.mstatus >> 7) & 1;
            state.mode = match mpp {
                0 => riscv_mode::U,
                1 => riscv_mode::S,
                3 => riscv_mode::M,
                _ => riscv_mode::M,
            };
            // MIE ← MPIE, MPIE ← 1
            state.mstatus &= !(1 << 3); // clear MIE
            if mpie != 0 {
                state.mstatus |= 1 << 3;
            } // MIE = MPIE
            state.mstatus |= 1 << 7; // MPIE = 1
            state.mstatus &= !(0b11 << 11); // MPP ← U (0)
            state.pc = state.mepc;
            state.waiting = 0;
            state.consecutive_traps = 0; // successful trap completion
            0
        }
        0x102 => {
            // SRET — check privilege
            if state.mode < riscv_mode::S {
                deliver_illegal_instruction(state, instr as u64, result);
                return 0;
            }
            let spp = (state.mstatus >> 8) & 1;
            let spie = (state.mstatus >> 5) & 1;
            state.mode = if spp == 0 {
                riscv_mode::U
            } else {
                riscv_mode::S
            };
            // SIE ← SPIE, SPIE ← 1
            state.mstatus &= !(1 << 1); // clear SIE
            if spie != 0 {
                state.mstatus |= 1 << 1;
            } // SIE = SPIE
            state.mstatus |= 1 << 5; // SPIE = 1
            state.mstatus &= !(1 << 8); // SPP ← U (0)
            state.pc = state.sepc;
            state.waiting = 0;
            state.consecutive_traps = 0; // successful trap completion
            0
        }
        0x105 => {
            // WFI — wait for interrupt.
            // The top-of-loop check will exit the hart slice with
            // WFI_WAIT when no interrupt is pending.
            let tw = (state.mstatus >> 21) & 1;
            if tw != 0 && state.mode != riscv_mode::M {
                deliver_illegal_instruction(state, instr as u64, result);
                return 0;
            }
            // 刚从 WFI 被中断唤醒 (MRET 返回到 WFI 之后的 while 循环,
            // 循环条件仍未满足, 分支回到此处): 强制将 WFI 视为 NOP 以推进
            // PC, 允许 while (…) wfi() 轮询循环重新检查状态条件.
            if state.wfi_woken != 0 {
                state.wfi_woken = 0;
                return 4; // NOP — PC 推进至下一条指令
            }
            let pending = state.mip & state.mie;
            if pending != 0 {
                return 4;
            } // interrupt already pending -> NOP
            state.waiting = 1;
            4 // advance PC past WFI; wait handled at top of loop
        }
        // SFENCE.VMA: funct7=0b0001001 (bits [31:25]), rs2 in bits [24:20].
        // funct12 = (funct7 << 5) | rs2 = 0x120 | rs2.
        // Match any rs2 value (0x120..0x13F) so that ASID-specific flushes
        // (sfence.vma x0, rs2) are handled instead of raising IllInstr.
        f if (f >> 5) == 0x09 => {
            // SFENCE.VMA — flush all TLBs (both itlb and dtlb).
            // Full flush is conservative but always correct; ASID / VA
            // filtering can be added later for performance.
            for e in state.itlb.iter_mut() {
                e.valid = 0;
            }
            for e in state.dtlb.iter_mut() {
                e.valid = 0;
            }
            4
        }
        0x5A0 => {
            // MFENCE.DID — TEE memory-domain fence: flush TLB entries
            // matching the current hart's mdid from all harts' TLBs,
            // and invalidate the L2 cache.  The L2 invalidation is
            // handled by Python on exit; here we flush the local TLBs.
            // This matches the Python ``handle_system`` / ``decoder.py``
            // mfence.did implementation.
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
            4
        }
        _ => {
            deliver_illegal_instruction(state, instr as u64, result);
            0
        }
    }
}

pub fn handle_system(
    state: &mut HartState,
    f: &DecodedFields,
    instr: u32,
    result: &mut BatchResult,
    ctx: &WalkCtx,
    hart_id: u8,
    clint: &ClintCtx,
    pmp: &PmpCtx,
) -> u64 {
    match f.func3 {
        0b000 => dispatch_privileged(state, f.func12, instr, result, clint, ctx),
        0b001 | 0b010 | 0b011 | 0b101 | 0b110 | 0b111 => {
            // CSR instructions: funct3 = CSRRW(001), CSRRS(010), CSRRC(011),
            //                       CSRRWI(101), CSRRSI(110), CSRRCI(111)
            let cur_mtime = unsafe { *clint.mtime };
            let advance = csr::handle_csr(
                state, f.rd, f.rs1, f.func12, f.func3, instr, result, hart_id, cur_mtime, pmp,
            );
            // Sync MIP.MSIP -> CLINT msip: if software cleared MSIP via CSR
            // write, also clear the CLINT MSIP register so that ``build_mip``
            // doesn't re-assert it on the next batch entry.
            if f.func12 != 0x344 /* MIP */ || f.func12 /* SIP */ == 0x144 {
                return advance;
            }
            let msip_in_mip = (state.mip >> 3) & 1;
            let msip_in_clint = unsafe { *clint.msip.add(hart_id as usize) } as u64 & 1;
            if msip_in_mip == 0 && msip_in_clint != 0 {
                unsafe {
                    *clint.msip.add(hart_id as usize) = 0;
                }
                state.diag.clint_msip_clr = state.diag.clint_msip_clr.wrapping_add(1);
            } else if msip_in_mip != 0 && msip_in_clint == 0 {
                // CLINT was cleared directly (e.g. sw) but state.mip is
                // stale — sync state.mip down.  Do NOT set CLINT.
                state.mip &= !(1 << 3);
                state.diag.clint_msip_clr = state.diag.clint_msip_clr.wrapping_add(1);
            }
            advance
        }
        _ => {
            deliver_illegal_instruction(state, instr as u64, result);
            0
        }
    }
}

// All trap codes now imported directly from crate::trap::exc_code

// ============================================================
//  AMO handler (Phase D)
// ============================================================

pub fn handle_amo(
    state: &mut HartState,
    f: &DecodedFields,
    instr: u32,
    result: &mut BatchResult,
    ctx: &WalkCtx,
    pmp: &PmpCtx,
    dev: &DevCtx,
) -> u64 {
    let _aq = (instr >> 26) & 1;
    let _rl = (instr >> 25) & 1;
    let funct5 = f.func7 >> 2; // bits [31:27]
    let width: u8 = match f.func3 {
        0b010 => 4, // .W
        0b011 => 8, // .D
        _ => {
            deliver_illegal_instruction(state, instr as u64, result);
            return 0;
        }
    };

    let base = read_gpr(state, f.rs1);
    let va = base; // AMO: rs1 is the address, rs2 is the operand

    // Alignment check
    if va & (width as u64 - 1) != 0 {
        deliver_trap(
            state,
            mcause_val(exc_code::LD_MISALIGNED, false),
            va,
            result,
        );
        return 0;
    }

    // Translate VA -> PA
    let tr = match translate_va(state, ctx, va, true, false) {
        Ok(t) => t,
        Err(TranslateFault::PageFault(cause)) => {
            deliver_trap(state, mcause_val(cause, false), va, result);
            return 0;
        }
        Err(_) => {
            deliver_trap(
                state,
                mcause_val(exc_code::LD_ACCESS_FAULT, false),
                va,
                result,
            );
            return 0;
        }
    };

    // PMP check
    if !pmp_ok(state, tr.pa, width as u32, true, false, pmp) {
        deliver_trap(
            state,
            mcause_val(exc_code::LD_ACCESS_FAULT, false),
            va,
            result,
        );
        return 0;
    }

    // virtio-blk inline check — AMO to virtio MMIO is nonsensical
    // but handle it inline to avoid unnecessary exits.
    let rd = f.rd;
    let rs2_val = read_gpr(state, f.rs2);

    if let Some(_) = try_handle_virtio(tr.pa, true, rs2_val, width, dev) {
        return 4;
    }

    // MMIO check - AMO to MMIO exits to Python
    if is_device_addr(tr.pa, dev) {
        result.exit_reason = exit_reason::MMIO;
        result.exit_instr = instr;
        return EXIT_SENTINEL;
    }

    match funct5 {
        0b00010 => {
            // LR.W / LR.D
            let loaded = ram_read(ctx, tr.pa, width);
            state.reservation_valid = 1;
            state.reservation_addr = tr.pa;
            write_gpr(state, rd, if width == 4 { sext32(loaded) } else { loaded });
            4
        }
        0b00011 => {
            // SC.W / SC.D
            if state.reservation_valid == 0 || state.reservation_addr != tr.pa {
                write_gpr(state, rd, 1);
            } else {
                ram_write(ctx, tr.pa, rs2_val, width);
                state.reservation_valid = 0;
                write_gpr(state, rd, 0);
            }
            4
        }
        0b00001 | 0b00000 | 0b00100 | 0b01100 | 0b01000 | 0b10000 | 0b10100 | 0b11000 | 0b11100 => {
            // AMOSWAP/AMOADD/AMOXOR/AMOAND/AMOOR/AMOMIN/AMOMAX/AMOMINU/AMOMAXU
            let loaded = ram_read(ctx, tr.pa, width);
            let signed_ld = if width == 4 {
                sext32(loaded) as i64
            } else {
                loaded as i64
            };
            let signed_rs2 = if width == 4 {
                sext32(rs2_val) as i64
            } else {
                rs2_val as i64
            };

            let result_val: u64 = match funct5 {
                0b00001 => rs2_val,                            // AMOSWAP
                0b00000 => loaded.wrapping_add(rs2_val),       // AMOADD
                0b00100 => loaded ^ rs2_val,                   // AMOXOR
                0b01100 => loaded & rs2_val,                   // AMOAND
                0b01000 => loaded | rs2_val,                   // AMOOR
                0b10000 => (signed_ld.min(signed_rs2)) as u64, // AMOMIN
                0b10100 => (signed_ld.max(signed_rs2)) as u64, // AMOMAX
                0b11000 => loaded.min(rs2_val),                // AMOMINU
                0b11100 => loaded.max(rs2_val),                // AMOMAXU
                _ => {
                    deliver_illegal_instruction(state, instr as u64, result);
                    return 0;
                }
            };

            ram_write(ctx, tr.pa, result_val, width);
            write_gpr(state, rd, if width == 4 { sext32(loaded) } else { loaded });
            4
        }
        _ => {
            deliver_illegal_instruction(state, instr as u64, result);
            0
        }
    }
}

// ============================================================
//  Compressed instruction handlers (Phase D)
// ============================================================

pub fn handle_compressed(
    state: &mut HartState,
    half: u16,
    instr_word: u32,
    result: &mut BatchResult,
    ctx: &WalkCtx,
    pmp: &PmpCtx,
    dev: &DevCtx,
    clint: &ClintCtx,
) -> u64 {
    let cf = decode_compressed(half);

    match cf.quadrant {
        0 => handle_c0(state, &cf, instr_word, result, ctx, pmp, dev, clint),
        1 => handle_c1(state, &cf, instr_word, result),
        2 => handle_c2(state, &cf, instr_word, result, ctx, pmp, dev, clint),
        _ => {
            deliver_illegal_instruction(state, half as u64, result);
            0
        }
    }
}

/// C0: C.ADDI4SPN, C.LW, C.LD, C.SW, C.SD
fn handle_c0(
    state: &mut HartState,
    cf: &CompressedFields,
    instr_word: u32,
    result: &mut BatchResult,
    ctx: &WalkCtx,
    pmp: &PmpCtx,
    dev: &DevCtx,
    clint: &ClintCtx,
) -> u64 {
    match cf.funct3 {
        0b000 => {
            // C.ADDI4SPN: rd' = sp + nzuimm
            if cf.imm == 0 {
                deliver_illegal_instruction(state, instr_word as u64, result);
                return 0;
            }
            let sp = state.gprs[2]; // x2 = sp
            write_gpr(state, cf.rd, sp.wrapping_add(cf.imm));
            2
        }
        0b001 => {
            // C.FLD: fpr[rd'] = mem[rs1' + uimm] (RV64DC)
            let addr = read_gpr(state, cf.rs1p).wrapping_add(cf.imm);
            let prev_mode = state.mode;
            let val = load_mem_compressed(
                state, ctx, addr, 8, false, instr_word, result, pmp, dev, clint,
            );
            if state.mode != prev_mode {
                return 0;
            } // trap delivered
            if val == EXIT_SENTINEL {
                return EXIT_SENTINEL;
            }
            state.fprs[cf.rdp as usize] = val;
            state.mstatus |= MSTATUS_FS_LS | MSTATUS_SD_LS;
            2
        }
        0b010 => {
            // C.LW: rd' = mem[rs1' + uimm]
            let addr = read_gpr(state, cf.rs1p).wrapping_add(cf.imm);
            crate::diag::diag_small_addr_access(state, addr, read_gpr(state, cf.rs1p), cf.rs1p, instr_word, false, 4);
            let prev_mode = state.mode;
            let val = load_mem_compressed(
                state, ctx, addr, 4, false, instr_word, result, pmp, dev, clint,
            );
            if state.mode != prev_mode {
                return 0;
            } // trap delivered
            if val == EXIT_SENTINEL {
                return EXIT_SENTINEL;
            }
            let final_val = sext32(val);
            write_gpr(state, cf.rd, final_val);
            2
        }
        0b011 => {
            // C.LD: rd' = mem[rs1' + uimm]
            let addr = read_gpr(state, cf.rs1p).wrapping_add(cf.imm);
            crate::diag::diag_small_addr_access(state, addr, read_gpr(state, cf.rs1p), cf.rs1p, instr_word, false, 8);
            let prev_mode = state.mode;
            let val = load_mem_compressed(
                state, ctx, addr, 8, false, instr_word, result, pmp, dev, clint,
            );
            if state.mode != prev_mode {
                return 0;
            } // trap delivered
            if val == EXIT_SENTINEL {
                return EXIT_SENTINEL;
            }
            write_gpr(state, cf.rd, val);
            2
        }
        0b101 => {
            // C.FSD: mem[rs1' + uimm] = fpr[rd'] (RV64DC)
            let addr = read_gpr(state, cf.rs1p).wrapping_add(cf.imm);
            let fpr_val = state.fprs[cf.rdp as usize];
            let ret = store_mem_compressed(
                state, ctx, addr, fpr_val, 8, instr_word, result, pmp, dev, clint,
            );
            if ret == EXIT_SENTINEL {
                return EXIT_SENTINEL;
            }
            state.mstatus |= MSTATUS_FS_LS | MSTATUS_SD_LS;
            2
        }
        0b110 => {
            // C.SW: mem[rs1' + uimm] = rs2'
            let addr = read_gpr(state, cf.rs1p).wrapping_add(cf.imm);
            let rs2_val = read_gpr(state, cf.rdp) & 0xFFFF_FFFF;
            let ret = store_mem_compressed(
                state, ctx, addr, rs2_val, 4, instr_word, result, pmp, dev, clint,
            );
            if ret == EXIT_SENTINEL {
                return EXIT_SENTINEL;
            }
            2
        }
        0b111 => {
            // C.SD: mem[rs1' + uimm] = rs2'
            let addr = read_gpr(state, cf.rs1p).wrapping_add(cf.imm);
            let rs2_val = read_gpr(state, cf.rdp);
            let ret = store_mem_compressed(
                state, ctx, addr, rs2_val, 8, instr_word, result, pmp, dev, clint,
            );
            if ret == EXIT_SENTINEL {
                return EXIT_SENTINEL;
            }
            2
        }
        _ => {
            deliver_illegal_instruction(state, instr_word as u64, result);
            0
        }
    }
}

/// C1: C.ADDI, C.ADDIW, C.LI, C.LUI, C.ADDI16SP, C.SRLI/C.SRAI/C.ANDI,
///      C.SUB/C.XOR/C.OR/C.AND, C.SUBW/C.ADDW, C.J, C.BEQZ/C.BNEZ
fn handle_c1(
    state: &mut HartState,
    cf: &CompressedFields,
    instr_word: u32,
    result: &mut BatchResult,
) -> u64 {
    match cf.funct3 {
        0b000..=0b010 => {
            // C.ADDI / C.ADDIW / C.LI
            let imm = cf.imm;
            if cf.funct3 == 0b000 {
                // C.ADDI: rd += imm
                let v = read_gpr(state, cf.rd).wrapping_add(imm);
                write_gpr(state, cf.rd, v);
            } else if cf.funct3 == 0b001 {
                // C.ADDIW: rd = sext32(rd + imm)
                let v = (read_gpr(state, cf.rd).wrapping_add(imm)) & 0xFFFF_FFFF;
                write_gpr(state, cf.rd, sext32(v));
            } else {
                // C.LI: rd = imm
                write_gpr(state, cf.rd, imm);
            }
            2
        }
        0b011 => {
            if cf.rd == 2 {
                // C.ADDI16SP: sp += nzimm
                if cf.imm == 0 {
                    return EXIT_SENTINEL; // illegal
                }
                let sp = state.gprs[2];
                state.gprs[2] = sp.wrapping_add(cf.imm);
            } else {
                // C.LUI: rd = nzimm << 12
                if cf.imm == 0 {
                    return EXIT_SENTINEL;
                }
                write_gpr(state, cf.rd, (cf.imm << 12) & 0xFFFF_FFFF_FFFF_FFFF);
            }
            2
        }
        0b100 => {
            // C1 ALU: SRLI / SRAI / ANDI / SUB/XOR/OR/AND / SUBW/ADDW
            let rd_rs1 = cf.rs1p; // creg-mapped rd/rs1
            let v1 = read_gpr(state, rd_rs1);
            let sf = cf.sf;
            let r: u64 = match sf {
                0b00 => {
                    let shamt = ((cf.bit12 as u64) << 5) | (cf.rs2 as u64 & 0x1F);
                    v1 >> shamt
                }
                0b01 => {
                    let shamt = ((cf.bit12 as u64) << 5) | (cf.rs2 as u64 & 0x1F);
                    ((v1 as i64) >> shamt) as u64
                }
                0b10 => {
                    let imm = sext(((cf.bit12 as u64) << 5) | (cf.rs2 as u64 & 0x1F), 6);
                    v1 & imm
                }
                0b11 => {
                    let v2 = read_gpr(state, cf.rdp);
                    match c1_alu_reg_op(v1, v2, cf.bit12, cf.bit65) {
                        Some(val) => val,
                        None => {
                            deliver_illegal_instruction(state, instr_word as u64, result);
                            return 0;
                        }
                    }
                }
                _ => {
                    deliver_illegal_instruction(state, instr_word as u64, result);
                    return 0;
                }
            };
            write_gpr(state, rd_rs1, r);
            2
        }
        0b101 => {
            // C.J: pc += offset
            state.pc = state.pc.wrapping_add(cf.imm);
            0
        }
        0b110 | 0b111 => {
            // C.BEQZ / C.BNEZ
            let rs1 = cf.rs1p; // creg-mapped from bits[9:7]
            let v = read_gpr(state, rs1);
            let taken = if cf.funct3 == 0b110 { v == 0 } else { v != 0 };
            if taken {
                state.pc = state.pc.wrapping_add(cf.imm);
                0
            } else {
                2
            }
        }
        _ => {
            deliver_illegal_instruction(state, instr_word as u64, result);
            0
        }
    }
}

/// C2: C.SLLI, C.LWSP, C.LDSP, C.JR/C.MV/C.EBREAK/C.JALR/C.ADD, C.SWSP, C.SDSP
fn handle_c2(
    state: &mut HartState,
    cf: &CompressedFields,
    instr_word: u32,
    result: &mut BatchResult,
    ctx: &WalkCtx,
    pmp: &PmpCtx,
    dev: &DevCtx,
    clint: &ClintCtx,
) -> u64 {
    match cf.funct3 {
        0b000 => {
            // C.SLLI: rd = rd << shamt
            let v = read_gpr(state, cf.rd) << cf.imm;
            write_gpr(state, cf.rd, v);
            2
        }
        0b001 => {
            // C.FLDSP: fpr[rd] = mem[sp + uimm] (RV64DC)
            let addr = state.gprs[2].wrapping_add(cf.imm);
            let prev_mode = state.mode;
            let val = load_mem_compressed(
                state, ctx, addr, 8, false, instr_word, result, pmp, dev, clint,
            );
            if state.mode != prev_mode {
                return 0;
            } // trap delivered
            if val == EXIT_SENTINEL {
                return EXIT_SENTINEL;
            }
            state.fprs[cf.rd as usize] = val;
            state.mstatus |= MSTATUS_FS_LS | MSTATUS_SD_LS;
            2
        }
        0b010 => {
            // C.LWSP: rd = mem[sp + uimm]
            let addr = state.gprs[2].wrapping_add(cf.imm);
            crate::diag::diag_small_addr_access(state, addr, state.gprs[2], 2, instr_word, false, 4);
            let prev_mode = state.mode;
            let val = load_mem_compressed(
                state, ctx, addr, 4, false, instr_word, result, pmp, dev, clint,
            );
            if state.mode != prev_mode {
                return 0;
            } // trap delivered
            if val == EXIT_SENTINEL {
                return EXIT_SENTINEL;
            }
            let final_val = sext32(val);
            write_gpr(state, cf.rd, final_val);
            2
        }
        0b011 => {
            // C.LDSP: rd = mem[sp + uimm]
            let addr = state.gprs[2].wrapping_add(cf.imm);
            crate::diag::diag_small_addr_access(state, addr, state.gprs[2], 2, instr_word, false, 8);
            let prev_mode = state.mode;
            let val = load_mem_compressed(
                state, ctx, addr, 8, false, instr_word, result, pmp, dev, clint,
            );
            if state.mode != prev_mode {
                return 0;
            } // trap delivered
            if val == EXIT_SENTINEL {
                return EXIT_SENTINEL;
            }
            write_gpr(state, cf.rd, val);
            2
        }
        0b100 => {
            let rd_rs1 = cf.rd;
            let rs2 = cf.rs2;
            if rd_rs1 == 0 && rs2 == 0 {
                // C.EBREAK — treated as NOP (see EBREAK above)
                return 2;
            } else if rs2 == 0 {
                let target = read_gpr(state, rd_rs1);
                if cf.bit12 != 0 {
                    // C.JALR: ra = pc + 2
                    state.gprs[1] = state.pc.wrapping_add(2);
                }
                state.pc = target & !1;
                0
            } else if cf.bit12 != 0 {
                // C.ADD: rd += rs2
                let result_val = read_gpr(state, rd_rs1).wrapping_add(read_gpr(state, rs2));
                write_gpr(state, rd_rs1, result_val);
                2
            } else {
                // C.MV: rd = rs2
                let v = read_gpr(state, rs2);
                write_gpr(state, rd_rs1, v);
                2
            }
        }
        0b101 => {
            // C.FSDSP: mem[sp + uimm] = fpr[rs2] (RV64DC)
            if (state.mstatus & MSTATUS_FS_LS) == 0 {
                deliver_illegal_instruction(state, instr_word as u64, result);
                return 0;
            }
            let addr = state.gprs[2].wrapping_add(cf.imm2);
            let fpr_val = state.fprs[cf.rs2 as usize];
            let ret = store_mem_compressed(
                state, ctx, addr, fpr_val, 8, instr_word, result, pmp, dev, clint,
            );
            if ret == EXIT_SENTINEL {
                return EXIT_SENTINEL;
            }
            state.mstatus |= MSTATUS_FS_LS | MSTATUS_SD_LS;
            2
        }
        0b110 => {
            // C.SWSP: mem[sp + uimm] = rs2
            let addr = state.gprs[2].wrapping_add(cf.imm2);
            let val = read_gpr(state, cf.rs2) & 0xFFFF_FFFF;
            let ret = store_mem_compressed(
                state, ctx, addr, val, 4, instr_word, result, pmp, dev, clint,
            );
            if ret == EXIT_SENTINEL {
                return EXIT_SENTINEL;
            }
            2
        }
        0b111 => {
            // C.SDSP: mem[sp + uimm] = rs2
            let addr = state.gprs[2].wrapping_add(cf.imm2);
            let val = read_gpr(state, cf.rs2);
            let ret = store_mem_compressed(
                state, ctx, addr, val, 8, instr_word, result, pmp, dev, clint,
            );
            if ret == EXIT_SENTINEL {
                return EXIT_SENTINEL;
            }
            2
        }
        _ => {
            deliver_illegal_instruction(state, instr_word as u64, result);
            0
        }
    }
}

// ============================================================
//  Compressed load/store helpers
// ============================================================

#[inline]
fn load_mem_compressed(
    state: &mut HartState,
    ctx: &WalkCtx,
    va: u64,
    size: u8,
    _signed: bool,
    instr_word: u32,
    result: &mut BatchResult,
    pmp: &PmpCtx,
    dev: &DevCtx,
    clint: &ClintCtx,
) -> u64 {
    let tr = match translate_va(state, ctx, va, false, false) {
        Ok(t) => t,
        Err(TranslateFault::PageFault(cause)) => {
            deliver_trap(state, mcause_val(cause, false), va, result);
            return 0; // trap delivered inline; caller must check state.mode
        }
        Err(_) => {
            deliver_trap(
                state,
                mcause_val(exc_code::LD_ACCESS_FAULT, false),
                va,
                result,
            );
            return 0; // trap delivered inline; caller must check state.mode
        }
    };

    if !pmp_ok(state, tr.pa, size as u32, false, false, pmp) {
        deliver_trap(
            state,
            mcause_val(exc_code::LD_ACCESS_FAULT, false),
            va,
            result,
        );
        return 0; // trap delivered inline; caller must check state.mode
    }

    if let Some(data) = try_handle_clint(&MemAccess::read(tr.pa, size), state, clint) {
        return data;
    }

    if let Some(data) = try_handle_virtio(tr.pa, false, 0, size, dev) {
        return data;
    }

    if is_device_addr(tr.pa, dev) {
        result.exit_reason = exit_reason::MMIO;
        result.exit_instr = instr_word;
        return EXIT_SENTINEL;
    }

    ram_read(ctx, tr.pa, size)
}

#[inline]
fn store_mem_compressed(
    state: &mut HartState,
    ctx: &WalkCtx,
    va: u64,
    val: u64,
    size: u8,
    instr_word: u32,
    result: &mut BatchResult,
    pmp: &PmpCtx,
    dev: &DevCtx,
    clint: &ClintCtx,
) -> u64 {
    let tr = match translate_va(state, ctx, va, true, false) {
        Ok(t) => t,
        Err(TranslateFault::PageFault(cause)) => {
            deliver_trap(state, mcause_val(cause, false), va, result);
            return 0;
        }
        Err(_) => {
            deliver_trap(
                state,
                mcause_val(exc_code::ST_ACCESS_FAULT, false),
                va,
                result,
            );
            return 0;
        }
    };

    if !pmp_ok(state, tr.pa, size as u32, true, false, pmp) {
        deliver_trap(
            state,
            mcause_val(exc_code::ST_ACCESS_FAULT, false),
            va,
            result,
        );
        return 0;
    }

    if let Some(_) = try_handle_clint(&MemAccess::write(tr.pa, size, val), state, clint) {
        return 0;
    }

    if let Some(_) = try_handle_virtio(tr.pa, true, val, size, dev) {
        return 0;
    }

    if is_device_addr(tr.pa, dev) {
        result.exit_reason = exit_reason::MMIO;
        result.exit_instr = instr_word;
        return EXIT_SENTINEL;
    }

    ram_write(ctx, tr.pa, val, size);
    0 // caller returns 2
}

// ============================================================
//  Tests
// ============================================================

// ============================================================
//  Semihosting support
// ============================================================
//
// Semihosting is a debug mechanism where firmware communicates with
// a host debugger/emulator through a special trap sequence.  The RISC-V
// semihosting call sequence is:
//
//   slli zero, zero, 0x1f    0x01f01013      // entry marker
//   ebreak                   0x00100073      // trap
//   srai zero, zero, 0x7     0x40705013      // exit marker
//
// OpenSBI uses semihosting for its early console when built with
// CONFIG_SEMIHOSTING=y.  The emulator intercepts this sequence and
// dispatches the semihosting operation (SYS_WRITE, SYS_WRITEC, …).

/// Semihosting entry marker — ``slli zero, zero, 0x1f``.
const SEMIHOSTING_PRE: u32 = 0x01f01013;
/// Semihosting exit marker — ``srai zero, zero, 0x7``.
const SEMIHOSTING_POST: u32 = 0x40705013;

/// Semihosting operation codes.
const SH_SYS_OPEN: u64 = 0x01;
const SH_SYS_WRITEC: u64 = 0x03;
const SH_SYS_WRITE: u64 = 0x05;
const SH_SYS_ISTTY: u64 = 0x09;

/// Read a u64 from RAM at physical address *pa* (Bare-Mode, VA = PA).
#[inline]
unsafe fn read_ram_u64(ctx: &WalkCtx, pa: u64) -> u64 {
    if pa + 8 > ctx.ram_base + ctx.ram_size || pa < ctx.ram_base {
        return 0;
    }
    let off = (pa - ctx.ram_base) as isize;
    (ctx.ram.offset(off) as *const u64).read_unaligned()
}

/// Read a u8 from RAM at physical address *pa*.
#[inline]
unsafe fn read_ram_u8(ctx: &WalkCtx, pa: u64) -> u8 {
    if pa >= ctx.ram_base + ctx.ram_size || pa < ctx.ram_base {
        return 0;
    }
    let off = (pa - ctx.ram_base) as isize;
    *ctx.ram.offset(off)
}

/// Global state tracking which hart last received a ``[hart X]`` tag.
/// Only prepend the tag when the outputting hart changes — NOT at every
/// ``\n``.  This avoids noisy mid-line interleaving when firmware prints
/// sub-fields on separate lines.
static SH_LAST_TAGGED_HART: Mutex<Option<u64>> = Mutex::new(None);

/// Write ``data`` to stderr, prepending the ``[hart X]`` label only
/// when the outputting hart changes (not at every newline).
fn write_labeled_stderr(data: &[u8], hart_id: u64) {
    extern "C" {
        fn write(fd: i32, buf: *const u8, count: usize) -> isize;
    }

    // Only prepend the tag when the hart changes.
    {
        let mut last = SH_LAST_TAGGED_HART.lock().unwrap();
        if *last != Some(hart_id) {
            // Bold-blue ANSI prefix matching the debugger TUI styling.
            let prefix = format!("\x1b[1;34m[hart {}]\x1b[0m ", hart_id);
            let prefix_bytes = prefix.as_bytes();
            let mut off = 0;
            while off < prefix_bytes.len() {
                let n = unsafe { write(2, prefix_bytes[off..].as_ptr(), prefix_bytes.len() - off) };
                if n <= 0 {
                    break;
                }
                off += n as usize;
            }
            *last = Some(hart_id);
        }
    }

    // Write data to stderr — no per-\n tagging.
    let mut off = 0;
    while off < data.len() {
        let n = unsafe { write(2, data[off..].as_ptr(), data.len() - off) };
        if n <= 0 {
            break;
        }
        off += n as usize;
    }
}

/// Raw write to a file descriptor — bypasses Rust stdio locks to avoid
/// contention with Python's rich/prompt_toolkit terminal control on stdout.
/// Writes to stderr (fd 2) so semihosting output doesn't corrupt the TUI.
#[allow(unused)]
fn raw_write_stderr(data: &[u8]) {
    extern "C" {
        fn write(fd: i32, buf: *const u8, count: usize) -> isize;
    }
    let mut off = 0usize;
    while off < data.len() {
        let remaining = &data[off..];
        let n = unsafe { write(2, remaining.as_ptr(), remaining.len()) };
        if n <= 0 {
            break;
        }
        off += n as usize;
    }
}

/// Execute a semihosting operation.  Returns the value to place in `a0`.
fn semihosting_dispatch(ctx: &WalkCtx, op: u64, param_block: u64, hart_id: u64) -> u64 {
    match op {
        SH_SYS_WRITEC => {
            let ch = unsafe { read_ram_u8(ctx, param_block) };
            write_labeled_stderr(&[ch], hart_id);
            0
        }
        SH_SYS_WRITE => {
            let fd = unsafe { read_ram_u64(ctx, param_block) };
            let buf_addr = unsafe { read_ram_u64(ctx, param_block + 8) };
            let len = unsafe { read_ram_u64(ctx, param_block + 16) };
            if len == 0 {
                return 0;
            }
            // Route all output to stderr (fd 2) — stdout is owned by
            // the debugger's rich/prompt_toolkit TUI.
            if fd != 1 && fd != 2 {
                return -1i64 as u64;
            }
            if buf_addr >= ctx.ram_base && buf_addr + len <= ctx.ram_base + ctx.ram_size {
                let off = (buf_addr - ctx.ram_base) as isize;
                let data = unsafe { std::slice::from_raw_parts(ctx.ram.offset(off), len as usize) };
                write_labeled_stderr(data, hart_id);
                return 0;
            }
            return -1i64 as u64;
        }
        SH_SYS_OPEN => {
            // Return a valid fd for the probe to succeed.
            1
        }
        SH_SYS_ISTTY => {
            let fd = unsafe { read_ram_u64(ctx, param_block) };
            u64::from(fd <= 2)
        }
        _ => -1i64 as u64,
    }
}

/// Check whether *ebreak_pc* is part of a semihosting call sequence.
///
/// If the instructions before and after ``ebreak`` match the semihosting
/// marker pattern this function handles the operation, updates ``a0``,
/// and returns ``Some(8)`` (skip the 4‑byte ebreak + 4‑byte exit marker).
/// Returns ``None`` for a normal ebreak.
pub fn try_semihosting(state: &mut HartState, ctx: &WalkCtx, ebreak_pc: u64) -> Option<u64> {
    // Semihosting only runs with VA = PA (M‑mode, Bare translation).
    if ebreak_pc < ctx.ram_base || ebreak_pc + 4 > ctx.ram_base + ctx.ram_size {
        return None;
    }
    let ram_off = (ebreak_pc - ctx.ram_base) as isize;
    if ram_off < 4 {
        return None;
    }

    // Marker before ebreak
    let prev: u32 = unsafe { (ctx.ram.offset(ram_off - 4) as *const u32).read_unaligned() };
    if prev != u32::from_le(SEMIHOSTING_PRE) {
        return None;
    }

    // Marker after ebreak
    let next: u32 = unsafe { (ctx.ram.offset(ram_off + 4) as *const u32).read_unaligned() };
    if next != u32::from_le(SEMIHOSTING_POST) {
        return None;
    }

    let op = state.gprs[10]; // a0
    let param_block = state.gprs[11]; // a1
    let hart_id = state.mhartid;
    state.gprs[10] = semihosting_dispatch(ctx, op, param_block, hart_id);

    // Skip ebreak (4) + post-marker (4) — the remaining instructions
    // (sext.w / bltz / sub / j) handle the return value naturally.
    Some(8)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem;

    #[test]
    fn sext32_sign_extends() {
        assert_eq!(sext32(0x7FFFFFFF), 0x7FFFFFFF);
        assert_eq!(sext32(0x80000000), 0xFFFF_FFFF_8000_0000u64);
        assert_eq!(sext32(0xFFFFFFFF), 0xFFFF_FFFF_FFFF_FFFFu64);
    }

    #[test]
    fn sext_various_widths() {
        assert_eq!(sext(0xFF, 8), 0xFFFF_FFFF_FFFF_FFFFu64);
        assert_eq!(sext(0x7F, 8), 0x7F);
        assert_eq!(sext(0x8000, 16), 0xFFFF_FFFF_FFFF_8000u64);
        assert_eq!(sext(0x7FFF, 16), 0x7FFF);
    }

    /// Direct test: MemAccess::write + try_handle_clint sets MSIP.
    #[test]
    fn try_handle_clint_msip_write_direct() {
        let mut s0: HartState = unsafe { mem::zeroed() };
        s0.mhartid = 0;
        let mut s1: HartState = unsafe { mem::zeroed() };
        s1.mhartid = 1;

        let mut mtime: u64 = 0;
        let mut mtimecmp: [u64; 2] = [u64::MAX; 2];
        let mut msip: [u8; 2] = [0; 2];

        let mut states = [s0, s1];
        let ctx = ClintCtx {
            base: 0x200_0000,
            mtime: &mut mtime as *mut u64,
            mtimecmp: mtimecmp.as_mut_ptr(),
            msip: msip.as_mut_ptr(),
            states: states.as_mut_ptr(),
            num_harts: 2,
            yield_for_ipi: Cell::new(false),
            ipi_sender_hart: Cell::new(0),
            ipi_sender_rounds: Cell::new(0),
        };

        // Write MSIP[1] (cross-hart)
        let access = MemAccess::write(0x200_0004, 4, 1);
        let result = try_handle_clint(&access, &states[0], &ctx);
        assert_eq!(result, Some(0));
        assert_eq!(msip[1], 1, "direct: MSIP[1] must be 1");
        assert!(
            states[1].mip & (1 << 3) != 0,
            "direct: hart 1 mip must have MSIP"
        );
        assert!(
            ctx.yield_for_ipi.get(),
            "direct: cross-hart must set yield flag"
        );
        assert_eq!(
            ctx.ipi_sender_hart.get(),
            0,
            "direct: sender hart must be tracked for short-slice"
        );
        assert_eq!(
            ctx.ipi_sender_rounds.get(),
            16,
            "direct: short-slice rounds must be initialised"
        );

        // Write MSIP[0] (self-IPI should NOT set yield or sender tracking)
        ctx.yield_for_ipi.set(false);
        ctx.ipi_sender_rounds.set(0);
        let access_self = MemAccess::write(0x200_0000, 4, 1);
        let result2 = try_handle_clint(&access_self, &states[0], &ctx);
        assert_eq!(result2, Some(0));
        assert_eq!(msip[0], 1, "direct: MSIP[0] must be 1 after self-IPI");
        assert!(
            !ctx.yield_for_ipi.get(),
            "direct: self-IPI must NOT set yield flag"
        );
        assert_eq!(
            ctx.ipi_sender_rounds.get(),
            0,
            "direct: self-IPI must NOT set short-slice rounds"
        );
    }

    /// MSIP clear also goes through ``clint_write_msip`` and clears the
    /// target hart's mip bit.
    #[test]
    fn try_handle_clint_msip_clear_updates_mip() {
        let mut s0: HartState = unsafe { mem::zeroed() };
        s0.mhartid = 0;
        s0.mip = 1 << 3; // MSIP pending

        let mut mtime: u64 = 0;
        let mut mtimecmp: [u64; 2] = [u64::MAX; 2];
        let mut msip: [u8; 2] = [1, 1]; // both MSIPs set

        let mut states = [s0, unsafe { mem::zeroed() }];
        let ctx = ClintCtx {
            base: 0x200_0000,
            mtime: &mut mtime as *mut u64,
            mtimecmp: mtimecmp.as_mut_ptr(),
            msip: msip.as_mut_ptr(),
            states: states.as_mut_ptr(),
            num_harts: 2,
            yield_for_ipi: Cell::new(false),
            ipi_sender_hart: Cell::new(0),
            ipi_sender_rounds: Cell::new(0),
        };

        // Clear MSIP[0] (self-clear: target == current == 0)
        let access = MemAccess::write(0x200_0000, 4, 0);
        let result = try_handle_clint(&access, &states[0], &ctx);
        assert_eq!(result, Some(0));
        assert_eq!(msip[0], 0, "MSIP[0] should be cleared");
        assert_eq!(
            states[0].mip & (1 << 3),
            0,
            "mip.MSIP must be 0 after clear"
        );
        // Self-clear should NOT set yield or sender tracking
        assert!(!ctx.yield_for_ipi.get(), "MSIP clear should not set yield");
        assert_eq!(
            ctx.ipi_sender_rounds.get(),
            0,
            "MSIP clear should not set short-slice rounds"
        );
    }

    // ============================================================
    //  Semihosting tests
    // ============================================================

    /// Build the semihosting trap sequence in RAM at *offset*.
    /// Returns the address (ram_base + offset) of the ebreak instruction.
    fn write_semihosting_seq(ram: &mut [u8], ram_base: u64, offset: usize) -> u64 {
        let pre: [u8; 4] = 0x01f01013u32.to_le_bytes(); // slli zero, zero, 0x1f
        let ebr: [u8; 4] = 0x00100073u32.to_le_bytes(); // ebreak
        let post: [u8; 4] = 0x40705013u32.to_le_bytes(); // srai zero, zero, 0x7

        ram[offset - 4..offset].copy_from_slice(&pre);
        ram[offset..offset + 4].copy_from_slice(&ebr);
        ram[offset + 4..offset + 8].copy_from_slice(&post);
        ram_base + offset as u64
    }

    /// Write a u64 to RAM at *offset*.
    fn write_ram_u64(ram: &mut [u8], offset: usize, val: u64) {
        ram[offset..offset + 8].copy_from_slice(&val.to_le_bytes());
    }

    #[test]
    fn semihosting_writec_outputs_char() {
        use super::try_semihosting;
        use crate::state::HartState;
        use crate::translate::WalkCtx;

        let mut ram: Vec<u8> = vec![0u8; 0x200];
        let ram_base: u64 = 0x8000_0000;

        // Place a single character at offset 0x100
        ram[0x100] = b'X';

        // Place the semihosting sequence with ebreak at offset 0x80
        let ebreak_pc = write_semihosting_seq(&mut ram, ram_base, 0x80);

        let mut state: HartState = unsafe { std::mem::zeroed() };
        state.gprs[10] = super::SH_SYS_WRITEC; // a0 = WRITEC
        state.gprs[11] = ram_base + 0x100; // a1 = ptr to char 'X'
        state.pc = ebreak_pc;
        state.mode = crate::state::riscv_mode::M;

        let ctx = WalkCtx {
            ram: ram.as_mut_ptr(),
            ram_size: ram.len() as u64,
            ram_base,
            shadow_base: 0,
            shadow_size: 0,
            tlb_gen: std::ptr::null(),
        };

        let result = try_semihosting(&mut state, &ctx, ebreak_pc);
        assert_eq!(result, Some(8), "should return skip-8 for semihosting");
        assert_eq!(state.gprs[10], 0, "WRITEC should return 0 on success");
    }

    #[test]
    fn semihosting_write_outputs_buffer() {
        use super::try_semihosting;
        use crate::state::HartState;
        use crate::translate::WalkCtx;

        let mut ram: Vec<u8> = vec![0u8; 0x200];
        let ram_base: u64 = 0x8000_0000;

        let msg = b"Hello";
        let msg_off = 0x100;
        ram[msg_off..msg_off + 5].copy_from_slice(msg);

        // Parameter block at offset 0x80:
        // [0] = fd (1 = stdout)
        // [8] = buffer address
        // [16] = length
        write_ram_u64(&mut ram, 0x80, 1); // fd=1
        write_ram_u64(&mut ram, 0x88, ram_base + msg_off as u64); // buf addr
        write_ram_u64(&mut ram, 0x90, 5); // len

        // Place the semihosting sequence with ebreak at offset 0xa0
        let ebreak_pc = write_semihosting_seq(&mut ram, ram_base, 0xa0);

        let mut state: HartState = unsafe { std::mem::zeroed() };
        state.gprs[10] = super::SH_SYS_WRITE; // a0 = WRITE
        state.gprs[11] = ram_base + 0x80; // a1 = param block
        state.pc = ebreak_pc;
        state.mode = crate::state::riscv_mode::M;

        let ctx = WalkCtx {
            ram: ram.as_mut_ptr(),
            ram_size: ram.len() as u64,
            ram_base,
            shadow_base: 0,
            shadow_size: 0,
            tlb_gen: std::ptr::null(),
        };

        let result = try_semihosting(&mut state, &ctx, ebreak_pc);
        assert_eq!(result, Some(8), "should return skip-8 for semihosting");
        assert_eq!(state.gprs[10], 0, "WRITE should return 0 on success");
    }

    #[test]
    fn semihosting_regular_ebreak_not_handled() {
        use super::try_semihosting;
        use crate::state::HartState;
        use crate::translate::WalkCtx;

        let mut ram: Vec<u8> = vec![0u8; 0x200];
        let ram_base: u64 = 0x8000_0000;

        // Place ebreak WITHOUT semihosting markers
        let ebr_addr = ram_base + 0x80;
        let ebr: [u8; 4] = 0x00100073u32.to_le_bytes();
        ram[0x80..0x84].copy_from_slice(&ebr);

        let mut state: HartState = unsafe { std::mem::zeroed() };
        state.pc = ebr_addr;
        state.mode = crate::state::riscv_mode::M;

        let ctx = WalkCtx {
            ram: ram.as_mut_ptr(),
            ram_size: ram.len() as u64,
            ram_base,
            shadow_base: 0,
            shadow_size: 0,
            tlb_gen: std::ptr::null(),
        };

        let result = try_semihosting(&mut state, &ctx, ebr_addr);
        assert_eq!(result, None, "plain ebreak should NOT be semihosting");
    }
}
