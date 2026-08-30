//! CSR read/write for the execution engine (Phase C).
//!
//! Handles inline read/write for CSRs whose values are stored directly in
//! ``HartState``.  Unknown or write-sensitive CSRs trigger an exit to Python
//! so the full ``registers.py`` machinery can process them.

use std::sync::atomic::Ordering;

use crate::concurrent::ConcurrentClintCtx;
use crate::handlers::PmpCtx;
use crate::interrupt::clint::write_stimecmp;
use crate::interrupt::{
	compute_mtopi, compute_stopi, imsic_reg_read, imsic_reg_write, imsic_topei_claim_iid,
	imsic_topei_peek, sync_imsic_one, IID_M_IPI, IID_S_IPI,
};
use crate::state::{exit_reason, riscv_mode, HartState, InstrToBeExec, PYREMU_AIA};
use crate::trap::deliver_illegal_instruction;

// ============================================================
//  CsrContext — bundled parameters for csr_read / csr_write
// ============================================================

/// All state needed to read or write a CSR during execution.
///
/// ``clint`` provides live ``mtime`` (instead of a frozen snapshot) and
/// ``mtimecmp`` for MTIP claim bumping.
/// ``extern_hart_id`` is the caller-supplied index (0 / 1 / …); it may
/// differ from the CSR ``mhartid`` value stored in ``state``.
pub struct CsrContext<'a> {
	pub state: &'a mut HartState,
	pub extern_hart_id: u32,
	pub pmp: &'a PmpCtx,
	pub clint: &'a ConcurrentClintCtx,
}

impl<'a> CsrContext<'a> {
	#[inline]
	pub fn new(
		state: &'a mut HartState,
		extern_hart_id: u32,
		pmp: &'a PmpCtx,
		clint: &'a ConcurrentClintCtx,
	) -> Self {
		CsrContext {
			state,
			extern_hart_id,
			pmp,
			clint,
		}
	}

	/// Live ``mtime`` from the shared CLINT counter (atomic load).
	#[inline]
	pub fn mtime(&self) -> u64 {
		unsafe { &*self.clint.mtime }.load(Ordering::Relaxed)
	}

	/// Bump ``mtimecmp[extern_hart_id]`` to ``mtime + delta`` so the
	/// timer interrupt does not immediately re-fire after a claim.
	#[inline]
	pub fn bump_mtimecmp(&self, delta: u64) {
		let t = self.mtime();
		let target = t.wrapping_add(delta);
		unsafe {
			(&*self.clint.mtimecmp.add(self.extern_hart_id as usize))
				.store(target, Ordering::Release);
		}
	}
}

// ============================================================
//  CSR address constants
// ============================================================

// Machine-level
pub const MSTATUS: u16 = 0x300;
pub const MISA: u16 = 0x301;
pub const MEDELEG: u16 = 0x302;
pub const MIDELEG: u16 = 0x303;
pub const MIE: u16 = 0x304;
pub const MTVEC: u16 = 0x305;
pub const MCOUNTEREN: u16 = 0x306;
pub const MSCRATCH: u16 = 0x340;
pub const MEPC: u16 = 0x341;
pub const MCAUSE: u16 = 0x342;
pub const MTVAL: u16 = 0x343;
pub const MIP: u16 = 0x344;

// Machine info
pub const MVENDORID: u16 = 0xF11;
pub const MARCHID: u16 = 0xF12;
pub const MIMPID: u16 = 0xF13;
pub const MHARTID: u16 = 0xF14;

// Supervisor-level
pub const SSTATUS: u16 = 0x100;
pub const SIE: u16 = 0x104;
pub const STVEC: u16 = 0x105;
pub const SCOUNTEREN: u16 = 0x106;
pub const SSCRATCH: u16 = 0x140;
pub const SEPC: u16 = 0x141;
pub const SCAUSE: u16 = 0x142;
pub const STVAL: u16 = 0x143;
pub const SIP: u16 = 0x144;
pub const SATP: u16 = 0x180; // Sstc extension
pub const STIMECMP: u16 = 0x14D;

// AIA IMSIC CSRs
pub const MISELECT: u16 = 0x350;
pub const MIREG: u16 = 0x351;
pub const SISELECT: u16 = 0x150;
pub const SIREG: u16 = 0x151;
pub const MTOPEI: u16 = 0x35C;
pub const STOPEI: u16 = 0x15C;

// User-level floating-point CSRs (F/D extension)
pub const FFLAGS: u16 = 0x001;
pub const FRM: u16 = 0x002;
pub const FCSR: u16 = 0x003;

// ============================================================
//  mstatus / sstatus field masks
// ============================================================

// mstatus / sstatus field masks (must match Python hart.py bit definitions)
const MSTATUS_SIE: u64 = 1 << 1; // Supervisor interrupt enable
const MSTATUS_SPIE: u64 = 1 << 5; // Supervisor previous interrupt enable
const MSTATUS_UBE: u64 = 1 << 6; // User big-endian
const MSTATUS_SPP: u64 = 1 << 8; // Supervisor previous privilege
const MSTATUS_VS: u64 = 0b11 << 9; // Virtualisation state (H-ext)
const MSTATUS_FS: u64 = 0b11 << 13; // Floating-point unit state
const MSTATUS_XS: u64 = 0b11 << 15; // User extension state
const MSTATUS_MPRV: u64 = 1 << 17; // Modify privilege
const MSTATUS_SUM: u64 = 1 << 18; // Permit Supervisor User Memory access
const MSTATUS_MXR: u64 = 1 << 19; // Make eXecutable Readable
const MSTATUS_SD: u64 = 1 << 63; // State dirty (read-only)

// sstatus is a restricted view of mstatus (RISC-V Privileged Spec §4.1.1).
// Reading sstatus returns only the subset of mstatus bits accessible from S-mode.
const SSTATUS_READ_MASK: u64 = MSTATUS_SIE
	| MSTATUS_SPIE
	| MSTATUS_UBE
	| MSTATUS_SPP
	| MSTATUS_VS
	| MSTATUS_FS
	| MSTATUS_XS
	| MSTATUS_MPRV
	| MSTATUS_SUM
	| MSTATUS_MXR
	| MSTATUS_SD;

// Writing sstatus updates only the writable subset (VS and SD are read-only,
// matching Python's _SSTATUS_WRITABLE_MASK).
const SSTATUS_WRITE_MASK: u64 = SSTATUS_READ_MASK & !(MSTATUS_VS | MSTATUS_SD);

// ============================================================
//  Public API
// ============================================================

/// Result: 0 = success (handler returns advance 4), 1 = IllInstr trap,
/// 2 = exit to Python (unknown CSR or write requiring Python side-effects).
pub const CSR_OK: u8 = 0;
pub const CSR_ILL: u8 = 1;
pub const CSR_EXIT: u8 = 2;

/// Read a CSR value. Returns ``(value, status)``.
/// status == CSR_OK: value is valid, caller writes to rd.
/// status == CSR_ILL: deliver IllInstr.
/// status == CSR_EXIT: exit to Python.
pub fn csr_read(ctx: &mut CsrContext, addr: u16) -> (u64, u8) {
	// Check privilege
	let priv_req = (addr >> 8) as u8 & 0x3; // bits [9:8]
	if !csr_priv_ok(ctx.state.mode, priv_req) {
		return (0, CSR_ILL);
	}

	// Counter enable checks for S-mode reading M-mode counters
	if priv_req == 0 && ctx.state.mode == riscv_mode::S {
		match addr {
			0xC00 | 0xC80 => {
				if ctx.state.mcounteren & 1 == 0 {
					return (0, CSR_ILL);
				}
			}
			0xC01 | 0xC81 => {
				if ctx.state.mcounteren & 2 == 0 {
					return (0, CSR_ILL);
				}
			}
			0xC02 | 0xC82 => {
				if ctx.state.mcounteren & 4 == 0 {
					return (0, CSR_ILL);
				}
			}
			_ => {}
		}
	}
	if priv_req == 1 && ctx.state.mode == riscv_mode::U {
		match addr {
			0xC00 | 0xC80 => {
				if ctx.state.scounteren & 1 == 0 {
					return (0, CSR_ILL);
				}
			}
			0xC01 | 0xC81 => {
				if ctx.state.scounteren & 2 == 0 {
					return (0, CSR_ILL);
				}
			}
			0xC02 | 0xC82 => {
				if ctx.state.scounteren & 4 == 0 {
					return (0, CSR_ILL);
				}
			}
			_ => {}
		}
	}

	match addr {
        // Machine-level CSRs in HartState
        MSTATUS => (ctx.state.mstatus, CSR_OK),
        MEDELEG => (ctx.state.medeleg, CSR_OK),
        MIDELEG => (ctx.state.mideleg, CSR_OK),
        MIE     => (ctx.state.mie, CSR_OK),
        MTVEC   => (ctx.state.mtvec, CSR_OK),
        MSCRATCH => (ctx.state.mscratch, CSR_OK),
        MEPC    => (ctx.state.mepc, CSR_OK),
        MCAUSE  => (ctx.state.mcause, CSR_OK),
        MTVAL   => (ctx.state.mtval, CSR_OK),
        MIP     => (ctx.state.mip.load(Ordering::Acquire), CSR_OK),
        MCOUNTEREN => (ctx.state.mcounteren, CSR_OK),

        // Supervisor-level CSRs in HartState
        SSTATUS => (ctx.state.mstatus & SSTATUS_READ_MASK, CSR_OK),
        SIE     => (ctx.state.mie & ctx.state.mideleg, CSR_OK),
        SIP     => (ctx.state.mip.load(Ordering::Acquire) & ctx.state.mideleg, CSR_OK),
        STVEC   => (ctx.state.stvec, CSR_OK),
        SSCRATCH => (ctx.state.sscratch, CSR_OK),
        SEPC    => (ctx.state.sepc, CSR_OK),
        SCAUSE  => (ctx.state.scause, CSR_OK),
        STVAL   => (ctx.state.stval, CSR_OK),
        SATP    => (ctx.state.satp, CSR_OK),
        SCOUNTEREN => (ctx.state.scounteren, CSR_OK),
        STIMECMP => (ctx.state.stimecmp, CSR_OK),

        // Floating-point CSRs (require mstatus.FS != Off)
        FFLAGS => if fp_off(ctx.state) { (0, CSR_ILL) } else { ((ctx.state.fcsr & 0x1F) as u64, CSR_OK) },
        FRM    => if fp_off(ctx.state) { (0, CSR_ILL) } else { (((ctx.state.fcsr >> 5) & 0x7) as u64, CSR_OK) },
        FCSR   => if fp_off(ctx.state) { (0, CSR_ILL) } else { ((ctx.state.fcsr & 0xFF) as u64, CSR_OK) },

        // Machine info registers
        MVENDORID => (0, CSR_OK),
        MARCHID   => (0, CSR_OK),
        MIMPID    => (0, CSR_OK),
        MHARTID   => (ctx.state.mhartid as u64, CSR_OK),

        // MISA — report RV64IMAC
        MISA => (0x800000000014112Du64, CSR_OK),

        // time (from CLINT mtime)
        0xC01 => (ctx.mtime(), CSR_OK),
        // timeh
        0xC81 => (ctx.mtime() >> 32, CSR_OK),

        // cycle/instret — we don't track these,
        // but reading them should not trap. Return 0.
        0xC00 | 0xC02 | 0xC80 | 0xC82 => (0, CSR_OK),

        // TEE CSRs (mdid=0x5C0, pmpsplit=0x5C1) — memory domain ID
        // and PMP virtualization split register.
        // mdid 是完整 64-bit: 收窄到 u8 会使 ≥256 的飞地 ID 别名回 0 (host),
        // 既绕过 PMP 隔离 (mdid!=0 判定失效), 又让 SUSPEND handler 误判 host.
        0x5C0 => (ctx.state.mdid, CSR_OK),
        0x5C1 => (ctx.state.pmpsplit as u64, CSR_OK),

        // ---- AIA IMSIC CSRs (gated behind PYREMU_AIA) ----
        MISELECT => {
            if PYREMU_AIA { (ctx.state.imsic_m.select as u64, CSR_OK) }
            else { (0, CSR_ILL) }
        }
        MIREG => {
            if PYREMU_AIA { imsic_reg_read(&ctx.state.imsic_m, ctx.state.imsic_m.select) }
            else { (0, CSR_ILL) }
        }
        SISELECT => {
            if PYREMU_AIA { (ctx.state.imsic_s.select as u64, CSR_OK) }
            else { (0, CSR_ILL) }
        }
        SIREG => {
            if PYREMU_AIA { imsic_reg_read(&ctx.state.imsic_s, ctx.state.imsic_s.select) }
            else { (0, CSR_ILL) }
        }
        MTOPEI => {
            if !PYREMU_AIA {
                return (0, CSR_ILL);
            }
            // mtopei (0x35C) reads the top IMSIC M-level external
            // interrupt and claims the eip bit.  Per RISC-V AIA spec
            // this CSR only sees interrupts in the IMSIC M-file
            // (external interrupts IID >= 6 + software IPIs IID=1/3
            // when routed through seteipnum).  Legacy MSIP/MTIP from
            // CLINT are NOT visible here — the dispatcher uses mtopi
            // (0xFB0) for the combined view.
            let (raw, _) = imsic_topei_peek(&ctx.state.imsic_m);
            if raw != 0 {
                let iid = (raw >> 16) as u32;
                imsic_topei_claim_iid(&mut ctx.state.imsic_m, iid);
                sync_imsic_one(ctx.state, true);
            }
            return (raw, CSR_OK);
        }
        STOPEI => {
            if !PYREMU_AIA {
                return (0, CSR_ILL);
            }
            // stopei (0x15C): S-mode counterpart of mtopei —
            // IMSIC S-file only (external + IPI), does NOT include
            // legacy SSIP/STIP which are dispatched through stopi
            // (0xDB0).
            let (raw, _) = imsic_topei_peek(&ctx.state.imsic_s);

            if raw != 0 {
                let iid = (raw >> 16) as u32;
                imsic_topei_claim_iid(&mut ctx.state.imsic_s, iid);
                sync_imsic_one(ctx.state, false);
            }
            return (raw, CSR_OK);
        }
        0xFB0 /* mtopi */ => {
            if PYREMU_AIA { compute_mtopi(ctx.state, ctx.mtime()) }
            else { (0, CSR_ILL) }
        }
        0xDB0 /* stopi */ => {
            if PYREMU_AIA { compute_stopi(ctx.state, ctx.mtime()) }
            else { (0, CSR_ILL) }
        }
        // Other AIA CSRs (mvien, mvip, etc.)
        0x308 /* mvien */ | 0x309 /* mvip */
        | 0x318 /* mvienh */ | 0x319 /* mviph */ => (0, CSR_OK),

        // Debug/trace trigger CSRs (0x7A0-0x7AF) — not implemented.
        // Return 0 rather than IllInstr: OpenSBI does not delegate IllInstr
        // to S-mode, so S-mode trigger probes would trap to M-mode where
        // OpenSBI's handler calls sbi_trap_error -> WFI, stalling the kernel.
        // Returning 0 lets Linux probe these CSRs without trapping.
        // Real hardware with Sdext would make them accessible in M-mode;
        // without Sdext they are unimplemented and return 0.
        0x7A0..=0x7AF => (0, CSR_OK),

        // RV32-only high-half CSRs — return 0 on RV64 (reads are legal but zero).
        0x310 /* mstatush */ | 0x312 /* medelegh */ | 0x313 /* midelegh */ => (0, CSR_OK),

        // ---- PMP CSRs (RV64) ----
        // Odd-numbered pmpcfg registers are illegal on RV64.
        a if (0x3A1..=0x3AF).contains(&a) && (a & 1) != 0 => (0, CSR_ILL),
        // Even-numbered pmpcfg: read 8 config bytes, pack as u64 LE.
        a if (0x3A0..=0x3AE).contains(&a) && (a & 1) == 0 => {
            let entry_base = ((a - 0x3A0) / 2) * 8;
            if (entry_base as u8) >= ctx.pmp.num {
                return (0, CSR_OK); // unimplemented entries read as zero
            }
            let mut val: u64 = 0;
            let base = entry_base as usize;
            // 仅读取 [base, num) 内的条目, 越界条目按 0 处理 (未实现).
            let n = ((ctx.pmp.num as usize) - base).min(8);
            for i in 0..n {
                let b = unsafe { *ctx.pmp.cfg.add(base + i) } as u64;
                val |= b << (i * 8);
            }
            return (val, CSR_OK);
        }
        // PMP address registers (0x3B0-0x3EF): one per entry, 8 bytes each.
        a if (0x3B0..=0x3EF).contains(&a) => {
            let n = (a - 0x3B0) as usize;
            if n < ctx.pmp.num as usize {
                (unsafe { *ctx.pmp.addr.add(n) }, CSR_OK)
            } else {
                (0, CSR_OK)
            }
        }

        // Everything else: exit to Python
        _ => (0, CSR_EXIT),
    }
}

/// Write a CSR value. Returns ``(status)``.
/// status == CSR_OK: write succeeded.
/// status == CSR_ILL: deliver IllInstr.
/// status == CSR_EXIT: exit to Python.
pub fn csr_write(ctx: &mut CsrContext, addr: u16, val: u64) -> u8 {
	let priv_req = (addr >> 8) as u8 & 0x3;
	if !csr_priv_ok(ctx.state.mode, priv_req) {
		return CSR_ILL;
	}

	match addr {
        MSTATUS => {
            // SD is read-only
            let sd = ctx.state.mstatus & MSTATUS_SD;
            let wr = val & !MSTATUS_SD;
            ctx.state.mstatus = wr | sd;
            CSR_OK
        }
        MEDELEG => { ctx.state.medeleg = val; CSR_OK }
        MIDELEG => { ctx.state.mideleg = val; CSR_OK }
        MIE     => { ctx.state.mie = val; CSR_OK }
        MTVEC   => { ctx.state.mtvec = val; CSR_OK }
        MSCRATCH => { ctx.state.mscratch = val; CSR_OK }
        MEPC    => { ctx.state.mepc = val; CSR_OK }
        MCAUSE  => { ctx.state.mcause = val; CSR_OK }
        MTVAL   => { ctx.state.mtval = val; CSR_OK }
        MIP     => {
            // MSIP, STIP, MTIP, SEIP, MEIP are read-only — driven by CLINT /
            // timer (SSTC) / external interrupt controller.  Preserve them;
            // only allow writes to software-writable bits (SSIP, USIP).
            // STIP/SEIP mirror hardware state and must never be clobbered by
            // a guest ``csrw mip`` — the SSTC timer and the external interrupt
            // controller are the sole owners of these bits.
            let ro_mask: u64 = (1 << 3) | (1 << 5) | (1 << 7) | (1 << 9) | (1 << 11);
            ctx.state.mip.store((ctx.state.mip.load(Ordering::Acquire) & ro_mask) | (val & !ro_mask), Ordering::Release);
            CSR_OK
        }
        MCOUNTEREN => { ctx.state.mcounteren = val; CSR_OK }

        SSTATUS => {
            let mask = SSTATUS_WRITE_MASK;
            ctx.state.mstatus = (ctx.state.mstatus & !mask) | (val & mask);
            CSR_OK
        }
        SIE => {
            // sie is a restricted view of mie; only mideleg-delegated bits are writable
            let mask = ctx.state.mideleg;
            ctx.state.mie = (ctx.state.mie & !mask) | (val & mask);
            CSR_OK
        }
        SIP => {
            // sip is a restricted view of mip; only mideleg-delegated bits are writable
            let mask = ctx.state.mideleg;
            ctx.state.mip.store((ctx.state.mip.load(Ordering::Acquire) & !mask) | (val & mask), Ordering::Release);
            CSR_OK
        }
        STVEC   => { ctx.state.stvec = val; CSR_OK }
        SSCRATCH => { ctx.state.sscratch = val; CSR_OK }
        SEPC    => { ctx.state.sepc = val; CSR_OK }
        SCAUSE  => { ctx.state.scause = val; CSR_OK }
        STVAL   => { ctx.state.stval = val; CSR_OK }
        SCOUNTEREN => { ctx.state.scounteren = val; CSR_OK }
        SATP    => {
            // ASID (bits[59:44]) is hardwired to 0 (WARL): neither the Rust
            // speedup TLB nor the Python TLB tags entries with an ASID.
            // If ASID bits were readable-back, Linux would enable its ASID
            // allocator and skip sfence.vma on context switches, letting
            // stale translations from the previous address space hit — user
            // processes then load garbage and SIGSEGV (observed in ld.so).
            let val = val & !(0xFFFFu64 << 44);
            let new_mode = (val >> 60) as u8;
            ctx.state.satp = val;
            ctx.state.mmu_mode = new_mode;
            tlb_flush_inline(ctx.state);
            CSR_OK
        }

        // Machine info registers — read-only, writes are silently ignored
        MISA | MVENDORID | MARCHID | MIMPID | MHARTID     => CSR_OK,

        // time/timeh (0xC01/0xC81) — CLINT mtime 的只读镜像. 内核的
        // rdtime 惯用法 ``csrrs rd, time, x0`` 会回写原值, 写操作必须
        // 静默忽略 (WARL). 否则落入 _ => CSR_EXIT → 每条 rdtime 都退出
        // (~1.4ms/条) → 内核吞吐崩溃 → STI 处理尾部长于 tick 周期
        // → 永久 STI 活锁, 启动阻塞在 vgaarb: loaded.
        0xC01 | 0xC81 => CSR_OK,

        // cycle/instret (0xC00/0xC02/0xC80/0xC82) — 同为只读计数器,
        // csrrs 读惯用法回写原值, 同样静默忽略.
        0xC00 | 0xC02 | 0xC80 | 0xC82 => CSR_OK,

        // Sstc stimecmp (0x14D) — S-mode timer compare value.
        // QEMU 一次性定时器语义 (riscv_aclint_mtimer_write_timecmp):
        // 过去值立即置位 STIP, 未来值清位并设定本 hart 指令计数空间的
        // deadline (sync_mtip 活跃分支据此判定, 与共享 mtime 跨 hart 膨胀解耦).
        // Writing stimecmp also syncs to the CLINT mtimecmp array via the
        // pointer stored in clint context; this happens in Python's
        // ``_csr_write_raw("stimecmp", …)`` path during unmarshal.
        STIMECMP => {
            ctx.state.stimecmp = val;
            write_stimecmp(ctx.state, ctx.mtime(), val, ctx.clint.timebase_hz);
            CSR_OK
        }

        // Floating-point CSRs — writing sets FS to Dirty + SD.
        FFLAGS => {
            if fp_off(ctx.state) { return CSR_ILL; }
            ctx.state.fcsr = (ctx.state.fcsr & !0x1F) | (val as u32 & 0x1F);
            ctx.state.mstatus |= MSTATUS_FS | MSTATUS_SD;
            CSR_OK
        }
        FRM => {
            if fp_off(ctx.state) { return CSR_ILL; }
            ctx.state.fcsr = (ctx.state.fcsr & !0xE0) | ((val as u32 & 0x7) << 5);
            ctx.state.mstatus |= MSTATUS_FS | MSTATUS_SD;
            CSR_OK
        }
        FCSR => {
            if fp_off(ctx.state) { return CSR_ILL; }
            ctx.state.fcsr = val as u32 & 0xFF;
            ctx.state.mstatus |= MSTATUS_FS | MSTATUS_SD;
            CSR_OK
        }

        // TEE CSRs (mdid / pmpsplit)
        // mdid 完整 64-bit 保存 — 见读路径注释, 收窄到 u8 是 batch-4 挂死根因.
        0x5C0 => { ctx.state.mdid = val; CSR_OK }
        0x5C1 => { ctx.state.pmpsplit = val as u8; CSR_OK }

        // Debug/trace trigger CSRs (0x7A0-0x7AF) — not implemented.
        // Writes are silently ignored (WO in some implementations, RW in others;
        // without Sdext they are WARL=0).
        0x7A0..=0x7AF => CSR_OK,

        // ---- AIA IMSIC CSRs (gated behind PYREMU_AIA) ----
        MISELECT => {
            if PYREMU_AIA { ctx.state.imsic_m.select = val as u32; CSR_OK }
            else { CSR_ILL }
        }
        MIREG => {
            if PYREMU_AIA {
                let select = ctx.state.imsic_m.select;
                let rc = imsic_reg_write(&mut ctx.state.imsic_m, select, val);
                // MIREG write may have changed eip/eie/eidelivery/eithreshold
                // → MEIP/SEIP may need updating.  Without per-instruction
                // sync_imsic, the next check_pending_interrupts would use stale mip.
                sync_imsic_one(ctx.state, true);
                rc
            } else { CSR_ILL }
        }
        SISELECT => {
            if PYREMU_AIA { ctx.state.imsic_s.select = val as u32; CSR_OK }
            else { CSR_ILL }
        }
        SIREG => {
            if PYREMU_AIA {
                let select = ctx.state.imsic_s.select;
                let rc = imsic_reg_write(&mut ctx.state.imsic_s, select, val);
                sync_imsic_one(ctx.state, false);
                rc
            } else { CSR_ILL }
        }
        MTOPEI | STOPEI => {
            if !PYREMU_AIA {
                return CSR_ILL;
            }
            // Writing to mtopei/stopei claims the IID encoded in the
            // write value per RISC-V AIA spec.
            let iid = ((val >> 16) & 0x7FF) as u32;
            if iid == 0 {
                return CSR_OK;
            }
            let file = if addr == MTOPEI { &mut ctx.state.imsic_m } else { &mut ctx.state.imsic_s };
            imsic_topei_claim_iid(file, iid);
            sync_imsic_one(ctx.state, addr == MTOPEI);
            // Clear MSIP/SSIP for IPI IIDs (same rationale as
            // compute_mtopi — sbi_ipi_raw_clear is a no-op in AIA).
            if addr == MTOPEI && iid == IID_M_IPI {
                ctx.state.mip.fetch_and(!(1 << 3), Ordering::AcqRel);
            } else if addr == STOPEI && iid == IID_S_IPI {
                ctx.state.mip.fetch_and(!(1 << 1), Ordering::AcqRel);
            }
            return CSR_OK;
        }
        0xFB0 /* mtopi */ => {
            if !PYREMU_AIA {
                return CSR_ILL;
            }
            let iid = ((val >> 16) & 0x7FF) as u32;
            if iid == 0 {
                return CSR_OK;
            } else if iid == IID_M_IPI {
                // M-mode IPI (MSIP): claim IMSIC M-file eip + clear MSIP.
                imsic_topei_claim_iid(&mut ctx.state.imsic_m, iid);
                sync_imsic_one(ctx.state, true);
                ctx.state.mip.fetch_and(!(1 << 3), Ordering::AcqRel);
            } else if iid == 7 {
                // IRQ_M_TIMER (MTIP): clear mip bit.
                // Only bump mtimecmp if it is still expired —
                // sbi_timer_process() may have already programmed
                // a future deadline.  Unconditional bump_mtimecmp(4)
                // overwrites that deadline, causing MTIP to re-fire
                // every ~4 instructions → infinite M-mode trap loop
                // → 22x slowdown → RCU stall.
                ctx.state.mip.fetch_and(!(1 << 7), Ordering::AcqRel);
                let cur = ctx.mtime();
                let cmp = unsafe {
                    &*ctx.clint.mtimecmp.add(ctx.extern_hart_id as usize)
                }.load(Ordering::Acquire);
                if cmp <= cur {
                    ctx.bump_mtimecmp(4);
                }
            } else if iid == 11 {
                // IID=11 (MEI) — major identity for machine external.
                // MTOPI 在 QEMU 中为只读 CSR (CSR_MTOPI 无 write 函数),
                // 回写必须为 no-op. 外部中断 (含 M-file IPI) 只能经 MTOPEI
                // (csr_swap) 认领, 否则 `csrr mtopi` 的 read-modify-write
                // 会提前清掉 top eip, 使 OpenSBI 的 csr_swap(MTOPEI) 读空.
            } else {
                // IMSIC minor IID: claim specific eip bit + sync MEIP.
                imsic_topei_claim_iid(&mut ctx.state.imsic_m, iid);
                sync_imsic_one(ctx.state, true);
            }
            return CSR_OK;
        }
        0xDB0 /* stopi */ => {
            if !PYREMU_AIA {
                return CSR_ILL;
            }
            let iid = ((val >> 16) & 0x7FF) as u32;
            if iid == 0 {
                return CSR_OK;
            } else if iid == IID_S_IPI {
                // S-mode IPI (SSIP): claim IMSIC S-file eip + clear SSIP.
                imsic_topei_claim_iid(&mut ctx.state.imsic_s, iid);
                sync_imsic_one(ctx.state, false);
                ctx.state.mip.fetch_and(!(1 << 1), Ordering::AcqRel);
            } else if iid == 5 {
                // IRQ_S_TIMER (STIP): clear mip bit + bump
                // stimecmp so the interrupt does not re-fire.
                ctx.state.mip.fetch_and(!(1 << 5), Ordering::AcqRel);
                let now = ctx.mtime();
                if ctx.state.stimecmp <= now {
                    ctx.state.stimecmp = now + 4;
                    // bump 后 stimecmp 为未来值: 按 QEMU 语义设定一次性 deadline.
                    // 不设定则 sync_mtip 活跃分支退化为共享 mtime 比较, 会被其他
                    // 活跃 hart 的进度膨胀提前置位 → mret 后立即再 trap 的活锁.
                    write_stimecmp(ctx.state, now, ctx.state.stimecmp, ctx.clint.timebase_hz);
                }
            } else if iid == 9 {
                // IID=9 (SEI) — major identity for supervisor external.
                // STOPI 在 QEMU 中为只读 CSR (CSR_STOPI 无 write 函数),
                // 回写原值必须是无副作用的 no-op. 外部中断 (含跨 hart IPI,
                // minor IID=1) 只能通过 STOPEI (csr_swap) 认领. 若在此处
                // 于 `csrr stopi` 的 read-modify-write 中提前认领 top eip,
                // 内核 imsic_handle_irq 的 csr_swap(STOPEI) 将读到空, 导致
                // IPI 握手永远无法完成 (发送方在 flush_icache_all 自旋).
            } else {
                // IMSIC minor IID: claim specific eip bit + sync SEIP.
                imsic_topei_claim_iid(&mut ctx.state.imsic_s, iid);
                sync_imsic_one(ctx.state, false);
            }
            return CSR_OK;
        }
        // Other AIA CSRs — no IMSIC; writes are ignored (WARL / read-only).
        0x308 /* mvien */ | 0x309 /* mvip */
        | 0x318 /* mvienh */ | 0x319 /* mviph */ => CSR_OK,

        // RV32-only high-half CSRs — writes ignored on RV64.
        0x310 /* mstatush */ | 0x312 /* medelegh */ | 0x313 /* midelegh */ => CSR_OK,

        // ---- PMP CSRs (RV64) ----
        // Odd-numbered pmpcfg registers are illegal on RV64.
        a if (0x3A1..=0x3AF).contains(&a) && (a & 1) != 0 => CSR_ILL,
        // Even-numbered pmpcfg: unpack u64 into 8 config bytes.
        a if (0x3A0..=0x3AE).contains(&a) && (a & 1) == 0 => {
            let entry_base = ((a - 0x3A0) as usize / 2) * 8;
            if (entry_base as u8) < ctx.pmp.num {
                // 仅写入 [entry_base, num) 内的条目; num < entry_base+8 时截断,
                // 避免越界写 (UB — release 下会破坏 cfg 的别名假设).
                let n = ((ctx.pmp.num as usize) - entry_base).min(8);
                for i in 0..n {
                    let b = ((val >> (i * 8)) & 0xFF) as u8;
                    unsafe { *ctx.pmp.cfg.add(entry_base + i) = b; }
                }
            }
            CSR_OK
        }
        // PMP address registers (0x3B0-0x3EF): one per entry, 8 bytes each.
        a if (0x3B0..=0x3EF).contains(&a) => {
            let n = (a - 0x3B0) as usize;
            if n < ctx.pmp.num as usize {
                unsafe { *ctx.pmp.addr.add(n) = val; }
            }
            CSR_OK
        }

        // Everything else -> exit to Python
        _ => CSR_EXIT,
    }
}

// ============================================================
//  Helpers
// ============================================================

/// Flush both TLBs without importing translate module.
#[inline]
fn tlb_flush_inline(state: &mut HartState) {
	for e in state.itlb.iter_mut() {
		e.valid = 0;
	}
	for e in state.dtlb.iter_mut() {
		e.valid = 0;
	}
}

/// 浮点单元是否被禁用 (mstatus.FS == Off) — FP CSR 访问需 FS != 0。
#[inline]
fn fp_off(state: &HartState) -> bool {
	(state.mstatus & MSTATUS_FS) == 0
}

/// Check if *mode* is privileged enough to access a CSR at *priv_req*.
#[inline]
fn csr_priv_ok(mode: u8, priv_req: u8) -> bool {
	match priv_req {
		0 => true,                  // user-readable
		1 => mode >= riscv_mode::S, // supervisor
		2 => mode >= riscv_mode::H, // hypervisor (unused)
		3 => mode >= riscv_mode::M, // machine
		_ => false,
	}
}

/// Handle a CSR instruction (CSRRW/CSRRS/CSRRC/CSRRWI/CSRRSI/CSRRCI).
pub fn handle_csr(
	ctx: &mut CsrContext,
	rd: u8,
	rs1: u8,
	csr_addr: u16,
	funct3: u8,
	instr: u32,
	instr_group: &mut InstrToBeExec,
) -> u64 {
	// Read old CSR value
	let (old_val, status) = csr_read(ctx, csr_addr);
	if status == CSR_ILL {
		deliver_illegal_instruction(ctx.state, instr as u64);
		return 0;
	}
	if status == CSR_EXIT {
		instr_group.exit_reason = exit_reason::ECALL;
		instr_group.exit_instr = instr;
		return EXIT_SENTINEL;
	}

	// Compute new value based on funct3
	let rs1_val = if funct3 >= 4 {
		// CSRRWI / CSRRSI / CSRRCI: use zero-extended rs1 (5-bit unsigned)
		rs1 as u64
	} else {
		read_gpr(ctx.state, rs1)
	};

	let new_val: u64 = match funct3 & 0x3 {
		1 => rs1_val,            // CSRRW / CSRRWI
		2 => old_val | rs1_val,  // CSRRS / CSRRSI
		3 => old_val & !rs1_val, // CSRRC / CSRRCI
		_ => {
			deliver_illegal_instruction(ctx.state, instr as u64);
			return 0;
		}
	};

	// mtopei/stopei: claim the specific IID that was peeked (not the current
	// top-priority IID).  This prevents a cross-hart race where sender B sets
	// a new eip between receiver A's peek and claim — a generic top-priority
	// scan would claim B's interrupt and lose A's acknowledgment.
	if csr_addr == MTOPEI || csr_addr == STOPEI {
		let is_mfile = csr_addr == MTOPEI;
		let file = if is_mfile {
			&mut ctx.state.imsic_m
		} else {
			&mut ctx.state.imsic_s
		};
		let iid = (old_val >> 16) as u32;
		imsic_topei_claim_iid(file, iid);
		// Clear the corresponding mip bit for software interrupts
		// (IID=3 MSIP for M-file, IID=1 SSIP for S-file).  In pure
		// AIA mode these bits are zero (IPIs route through MEIP/SEIP),
		// but hybrid CLINT+IMSIC setups need the explicit clear.
		if is_mfile && iid == 3 {
			ctx.state.mip.fetch_and(!(1 << 3), Ordering::AcqRel); // MSIP
		} else if !is_mfile && iid == 1 {
			ctx.state.mip.fetch_and(!(1 << 1), Ordering::AcqRel); // SSIP
		}
		// Timer interrupts (IID=5 STIP, IID=7 MTIP) are claimed
		// through the mtopi/stopi path, not IMSIC.  If the kernel
		// writes STOPEI/MTOPEI with a timer IID, clear the mip bit
		// and bump the comparator to prevent immediate re-fire.
		if !is_mfile && iid == 5 {
			ctx.state.mip.fetch_and(!(1 << 5), Ordering::AcqRel);
			if ctx.state.stimecmp > 0 && ctx.state.stimecmp <= ctx.mtime() {
				ctx.state.stimecmp = ctx.mtime() + 4;
			}
		}
		if is_mfile && iid == 7 {
			ctx.state.mip.fetch_and(!(1 << 7), Ordering::AcqRel);
			// Only bump mtimecmp if it is still expired (mirrors the
			// mtopi csr_write fix — prevents overwriting a future
			// deadline programmed by sbi_timer_process()).
			let cur = ctx.mtime();
			let cmp = unsafe { &*ctx.clint.mtimecmp.add(ctx.extern_hart_id as usize) }
				.load(Ordering::Acquire);
			if cmp <= cur {
				ctx.bump_mtimecmp(4);
			}
		}
		// After claim, re-sync the IMSIC file to update mip (MEIP/SEIP).
		// Without per-instruction sync_imsic, the next check_pending_interrupts
		// would see a stale MEIP/SEIP → spurious trap.
		sync_imsic_one(ctx.state, is_mfile);
		if rd != 0 {
			write_gpr(ctx.state, rd, old_val);
		}
		return 4;
	}

	// Write back to CSR (ALWAYS write, even if rd==x0 — the CSR write
	// must happen regardless; only the old-value readback is skipped when rd==0).
	let wstatus = csr_write(ctx, csr_addr, new_val);
	if wstatus == CSR_ILL {
		deliver_illegal_instruction(ctx.state, instr as u64);
		return 0;
	}
	if wstatus == CSR_EXIT {
		instr_group.exit_reason = exit_reason::ECALL;
		instr_group.exit_instr = instr;
		return EXIT_SENTINEL;
	}

	// STIMECMP 写路径的 mip.STIP 立即更新已由 csr_write 内的 write_stimecmp
	// 完成 (0→清, 过去→置位, 未来→清+设定 deadline), 与 QEMU 语义逐行一致;
	// 此处不再重复基于共享 mtime 的重新比较 — 那会与 deadline 模型矛盾
	// (跨 batch 边界的共享 mtime 膨胀会把刚设定的未来定时器重新推成"已到期").
	// sync_mtip 仍在下一指令边界运行, 但 stopi 在此之前读到的 mip.STIP
	// 已是最新值 (write_stimecmp 在写指令内同步置/清位)。

	// stopi / mtopi claim is now handled entirely within csr_write
	// (see the 0xDB0 / 0xFB0 match arms), which has access to mtime
	// and mtimecmp via CsrContext.  No duplicated claim needed here.

	// Write old value to rd (skip for x0)
	if rd != 0 {
		write_gpr(ctx.state, rd, old_val);
	}
	4
}

// ============================================================
//  GPR read/write (local copies for csr.rs independence)
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

/// Sentinel value: return this from a handler to signal exit-to-Python.
pub const EXIT_SENTINEL: u64 = u64::MAX;

// ============================================================
//  Tests
// ============================================================

#[cfg(test)]
mod tests {
	use super::*;

	fn test_state() -> HartState {
		let mut s: HartState = unsafe { std::mem::zeroed() };
		s.mode = riscv_mode::M;
		s
	}

	fn test_pmp() -> PmpCtx {
		PmpCtx {
			cfg: std::ptr::null_mut(),
			addr: std::ptr::null_mut(),
			num: 0,
		}
	}

	/// Create a dummy ``ConcurrentClintCtx`` with a real ``mtime`` counter
	/// so ``ctx.mtime()`` works in tests.  ``mtimecmp`` and ``msip`` are
	/// null — tests that need timer-claim bumping must set them up explicitly.
	fn test_clint(mtime_val: u64) -> ConcurrentClintCtx {
		let mtime_box = Box::new(std::sync::atomic::AtomicU64::new(mtime_val));
		let mtime_ptr = Box::into_raw(mtime_box) as *const std::sync::atomic::AtomicU64;
		ConcurrentClintCtx {
			base: 0,
			mtime: mtime_ptr,
			mtimecmp: std::ptr::null(),
			msip: std::ptr::null(),
			num_harts: 1,
			msip_pending: std::cell::Cell::new(std::ptr::null()),
			hart_threads: std::cell::Cell::new(std::ptr::null()),
			hart_states: std::cell::Cell::new(std::ptr::null()),
			timebase_hz: 0, // 单元测试禁用 clock-source 推进, 保持确定性
		}
	}

	fn test_csr_ctx<'a>(state: &'a mut HartState, clint: &'a ConcurrentClintCtx) -> CsrContext<'a> {
		// Leak a boxed PmpCtx so the reference has 'static lifetime — fine for tests.
		CsrContext::new(state, 0, Box::leak(Box::new(test_pmp())), clint)
	}

	fn test_csr_ctx_pmp<'a>(
		state: &'a mut HartState,
		pmp: &'a PmpCtx,
		clint: &'a ConcurrentClintCtx,
	) -> CsrContext<'a> {
		CsrContext::new(state, 0, pmp, clint)
	}

	#[test]
	fn read_mstatus() {
		let mut s = test_state();
		s.mstatus = 0x1880; // MPP=S, MPIE=1
		let clint = test_clint(0);
		let mut ctx = test_csr_ctx(&mut s, &clint);
		let (v, st) = csr_read(&mut ctx, MSTATUS);
		assert_eq!(st, CSR_OK);
		assert_eq!(v, 0x1880);
	}

	#[test]
	fn read_sstatus_view() {
		let mut s = test_state();
		s.mstatus = MSTATUS_SIE | MSTATUS_SPIE | MSTATUS_SPP;
		let clint = test_clint(0);
		let mut ctx = test_csr_ctx(&mut s, &clint);
		let (v, st) = csr_read(&mut ctx, SSTATUS);
		assert_eq!(st, CSR_OK);
		assert_eq!(
			v & SSTATUS_READ_MASK,
			MSTATUS_SIE | MSTATUS_SPIE | MSTATUS_SPP
		);
	}

	#[test]
	fn write_satp_flushes_tlb() {
		let mut s = test_state();
		s.mmu_mode = 8;
		// Pre-fill TLB
		s.itlb[0].valid = 1;
		s.dtlb[0].valid = 1;
		// Write satp
		{
			let clint = test_clint(0);
			let mut ctx = test_csr_ctx(&mut s, &clint);
			let st = csr_write(&mut ctx, SATP, 0x8000000000000000u64); // Sv39, PPN=0
			assert_eq!(st, CSR_OK);
		}
		assert_eq!(s.mmu_mode, 8);
		// TLBs should be flushed
		assert_eq!(s.itlb[0].valid, 0);
		assert_eq!(s.dtlb[0].valid, 0);
	}

	/// satp.ASID (bits[59:44]) is hardwired to 0 — the TLB has no ASID tag.
	/// A readable-back ASID makes Linux enable its ASID allocator and skip
	/// sfence.vma on context switch, so stale TLB entries from the previous
	/// address space survive (user processes crash on garbage loads).
	/// Real-world trigger value from a live session: satp=0x8000100000082f1b
	/// (Sv39, ASID=1, PPN=0x82f1b) — ASID=1 proves the allocator was active.
	#[test]
	fn write_satp_asid_hardwired_zero() {
		let mut s = test_state();
		// Linux probe: write all-ones ASID, read back to count writable bits
		let probe = 0x8000000000000000u64 | (0xFFFFu64 << 44) | 0x82f1b;
		{
			let clint = test_clint(0);
			let mut ctx = test_csr_ctx(&mut s, &clint);
			let st = csr_write(&mut ctx, SATP, probe);
			assert_eq!(st, CSR_OK);
			let (v, st) = csr_read(&mut ctx, SATP);
			assert_eq!(st, CSR_OK);
			assert_eq!(
				v, 0x8000000000082f1b,
				"ASID must read back as 0; MODE/PPN preserved"
			);
			assert_ne!(v, probe, "old behavior (raw ASID stored) must not reappear");
		}
		// MODE and PPN must be unaffected by the mask
		assert_eq!(s.mmu_mode, 8);
	}

	#[test]
	fn write_sstatus_updates_mstatus_fields() {
		let mut s = test_state();
		// Write SIE bit through sstatus
		{
			let clint = test_clint(0);
			let mut ctx = test_csr_ctx(&mut s, &clint);
			let st = csr_write(&mut ctx, SSTATUS, MSTATUS_SIE);
			assert_eq!(st, CSR_OK);
		}
		assert_eq!(s.mstatus & MSTATUS_SIE, MSTATUS_SIE);
	}

	#[test]
	fn write_unknown_csr_triggers_exit() {
		let mut s = test_state();
		let clint = test_clint(0);
		let mut ctx = test_csr_ctx(&mut s, &clint);
		let st = csr_write(&mut ctx, 0xB00, 0); // mcycle
		assert_eq!(st, CSR_EXIT);
	}

	/// sie (0x104) is a restricted view of mie (0x304).
	/// Writing sie should update mie for mideleg-delegated bits.
	#[test]
	fn write_sie_updates_mie() {
		let mut s = test_state();
		s.mideleg = 1 << 5; // STIE delegated
		s.mie = 0;
		{
			let clint = test_clint(0);
			let mut ctx = test_csr_ctx(&mut s, &clint);
			let st = csr_write(&mut ctx, SIE, 1 << 5); // write STIE through sie
			assert_eq!(st, CSR_OK);
		}
		assert_eq!(
			s.mie & (1 << 5),
			1 << 5,
			"mie.STIE must be set when sie.STIE is written"
		);
		// Non-delegated bits must not leak through
		s.mideleg = 1 << 5;
		s.mie = 0;
		{
			let clint = test_clint(0);
			let mut ctx = test_csr_ctx(&mut s, &clint);
			let st = csr_write(&mut ctx, SIE, (1 << 5) | (1 << 3)); // STIE + MSIP
			assert_eq!(st, CSR_OK);
		}
		assert_eq!(
			s.mie & (1 << 3),
			0,
			"non-delegated bits in sie write must be masked"
		);
		assert_eq!(s.mie & (1 << 5), 1 << 5, "delegated STIE must still be set");
	}

	/// sie read must return mie & mideleg (only delegated bits visible).
	#[test]
	fn read_sie_shows_delegated_bits() {
		// Need to construct state with correct mie/mideleg
		let mut s2 = test_state();
		s2.mideleg = (1 << 5) | (1 << 1); // STIE + SSIE delegated
		s2.mie = (1 << 5) | (1 << 3); // STIE + MSIE set in mie
		let clint = test_clint(0);
		let mut ctx = test_csr_ctx(&mut s2, &clint);
		let (v, st) = csr_read(&mut ctx, SIE);
		assert_eq!(st, CSR_OK);
		assert_eq!(
			v,
			1 << 5,
			"sie must only show delegated bits (STIE), not MSIE"
		);
	}

	/// sip (0x144) is a restricted view of mip (0x344).
	#[test]
	fn write_sip_updates_mip() {
		let mut s = test_state();
		s.mideleg = 1 << 1; // SSIP delegated
		s.mip.store(0, Ordering::Release);
		{
			let clint = test_clint(0);
			let mut ctx = test_csr_ctx(&mut s, &clint);
			let st = csr_write(&mut ctx, SIP, 1 << 1); // write SSIP through sip
			assert_eq!(st, CSR_OK);
		}
		assert_eq!(
			s.mip.load(Ordering::Acquire) & (1 << 1),
			1 << 1,
			"mip.SSIP must be set when sip.SSIP is written"
		);
	}

	/// mip (0x344) 的只读位 (STIP/SEIP) 不得被 guest ``csrw mip`` 覆写.
	/// 修复前 ro_mask 缺 STIP(5)/SEIP(9), 一条 ``csrw mip, 0`` 会清掉
	/// SSTC 定时器/外部中断的 pending 位, 破坏定时器状态.
	#[test]
	fn write_mip_preserves_stip_and_seip() {
		let mut s = test_state();
		s.mip
			.store((1 << 5) | (1 << 9) | (1 << 1), Ordering::Release);
		{
			let clint = test_clint(0);
			let mut ctx = test_csr_ctx(&mut s, &clint);
			let st = csr_write(&mut ctx, MIP, 0); // 尝试清除全部 pending 位
			assert_eq!(st, CSR_OK);
		}
		let mip = s.mip.load(Ordering::Acquire);
		assert_eq!(
			mip & (1 << 5),
			1 << 5,
			"STIP is read-only, must be preserved"
		);
		assert_eq!(
			mip & (1 << 9),
			1 << 9,
			"SEIP is read-only, must be preserved"
		);
		assert_eq!(
			mip & (1 << 1),
			0,
			"SSIP is software-writable, must be cleared"
		);
	}

	#[test]
	fn read_time() {
		let mut s = test_state();
		let clint = test_clint(123456789);
		let mut ctx = test_csr_ctx(&mut s, &clint);
		let (v, st) = csr_read(&mut ctx, 0xC01);
		assert_eq!(st, CSR_OK);
		assert_eq!(v, 123456789);
	}

	#[test]
	fn read_timeh() {
		let mut s = test_state();
		let clint = test_clint(0x123456789AB);
		let mut ctx = test_csr_ctx(&mut s, &clint);
		let (v, st) = csr_read(&mut ctx, 0xC81);
		assert_eq!(st, CSR_OK);
		// 0x123456789AB >> 32 -> upper bits = 0x123 = 291
		assert_eq!(v, 0x123);
	}

	#[test]
	fn csr_priv_check() {
		assert!(csr_priv_ok(riscv_mode::M, 3));
		assert!(csr_priv_ok(riscv_mode::M, 1));
		assert!(!csr_priv_ok(riscv_mode::S, 3));
		assert!(csr_priv_ok(riscv_mode::S, 1));
		assert!(csr_priv_ok(riscv_mode::U, 0));
		assert!(!csr_priv_ok(riscv_mode::U, 1));
	}

	// ---- handle_csr tests (rd==x0 regression) ----

	fn make_instr_group() -> InstrToBeExec {
		unsafe { std::mem::zeroed() }
	}

	/// ``csrw mscratch, t0`` -> ``csrrw x0, mscratch, t0``
	/// The CSR write MUST happen even though rd==x0 (only the readback is skipped).
	#[test]
	fn handle_csr_write_with_rd_x0() {
		let mut s = test_state();
		s.gprs[5] = 0xDEAD_BEEF; // t0 = value to write
		let mut r = make_instr_group();
		let (advance, mscratch_val) = {
			let clint = test_clint(0);
			let mut ctx = test_csr_ctx(&mut s, &clint);
			// CSRRW x0, mscratch, t0  ->  funct3=001, rs1=5(t0), rd=0(x0), csr=0x340(mscratch)
			let adv = handle_csr(&mut ctx, 0, 5, MSCRATCH, 0b001, 0x34051073, &mut r);
			(adv, ctx.state.mscratch)
		};
		assert_eq!(advance, 4);
		assert_eq!(
			mscratch_val, 0xDEAD_BEEF,
			"mscratch must be written even though rd==x0"
		);
	}

	/// ``csrwi mscratch, 5`` -> ``csrrwi x0, mscratch, 5``
	#[test]
	fn handle_csr_write_imm_with_rd_x0() {
		let mut s = test_state();
		let mut r = make_instr_group();
		let (advance, mscratch_val) = {
			let clint = test_clint(0);
			let mut ctx = test_csr_ctx(&mut s, &clint);
			// CSRRWI x0, mscratch, 5  ->  funct3=101, rs1=5(imm), rd=0, csr=0x340
			let adv = handle_csr(&mut ctx, 0, 5, MSCRATCH, 0b101, 0x340552f3, &mut r);
			(adv, ctx.state.mscratch)
		};
		assert_eq!(advance, 4);
		assert_eq!(
			mscratch_val, 5,
			"mscratch must be written via csrwi even though rd==x0"
		);
	}

	/// ``csrrs t0, mscratch, x0`` -> rd=t0, rs1=x0  (read-only, no write bits set)
	/// Old value must go to rd, CSR unchanged.
	#[test]
	fn handle_csr_read_to_rd() {
		let mut s = test_state();
		s.mscratch = 0xCAFE;
		let mut r = make_instr_group();
		let (advance, mscratch_val, gpr5_val) = {
			let clint = test_clint(0);
			let mut ctx = test_csr_ctx(&mut s, &clint);
			// CSRRS t0(5), mscratch, x0  ->  funct3=010, rs1=0, rd=5
			let adv = handle_csr(&mut ctx, 5, 0, MSCRATCH, 0b010, 0x3402b2f3, &mut r);
			(adv, ctx.state.mscratch, ctx.state.gprs[5])
		};
		assert_eq!(advance, 4);
		assert_eq!(mscratch_val, 0xCAFE, "mscratch unchanged on read");
		assert_eq!(gpr5_val, 0xCAFE, "old value written to rd");
	}

	// ---- PMP CSR write tests (inline in Rust, no CSR_EXIT) ----

	/// Build a PmpCtx backed by real mutable arrays so that PMP CSR
	/// writes can be verified against the underlying storage.
	fn test_pmp_real(num: u8) -> (PmpCtx, Vec<u8>, Vec<u64>) {
		let mut cfg = vec![0u8; num as usize];
		let mut addr = vec![0u64; num as usize];
		let ctx = PmpCtx {
			cfg: cfg.as_mut_ptr(),
			addr: addr.as_mut_ptr(),
			num,
		};
		(ctx, cfg, addr)
	}

	#[test]
	fn pmpcfg_write_packs_bytes() {
		let (pmp, cfg, _addr) = test_pmp_real(16);
		let mut s = test_state();
		let clint = test_clint(0);
		let mut ctx = test_csr_ctx_pmp(&mut s, &pmp, &clint);
		// pmpcfg0 (0x3A0) covers entries 0-7.
		// Write: entry 0 = 0xAB, entry 2 = 0xCD (byte positions 0 and 2).
		let val: u64 = 0xAB | (0xCDu64 << 16);
		let st = csr_write(&mut ctx, 0x3A0, val);
		assert_eq!(st, CSR_OK);
		assert_eq!(cfg[0], 0xAB);
		assert_eq!(cfg[1], 0x00);
		assert_eq!(cfg[2], 0xCD);
		// Entries beyond 7 should be untouched.
		assert_eq!(cfg[8], 0x00);
	}

	#[test]
	fn pmpcfg_write_out_of_range_is_noop() {
		// num=4, 但分配 8 字节 cfg 缓冲区, 以便显式断言越界条目 4-7 未被写入
		// (否则 pmpcfg0 的 8 字节写会越界污染堆内存 — release 下触发 UB).
		let mut cfg = vec![0u8; 8];
		let mut addr = vec![0u64; 4];
		let pmp = PmpCtx {
			cfg: cfg.as_mut_ptr(),
			addr: addr.as_mut_ptr(),
			num: 4,
		};
		let mut s = test_state();
		let clint = test_clint(0);
		let mut ctx = test_csr_ctx_pmp(&mut s, &pmp, &clint);
		// pmpcfg0 writes entries 0-7 but only 0-3 exist (num=4).
		let st = csr_write(&mut ctx, 0x3A0, 0xFF_FF_FF_FF_FF_FF_FF_FFu64);
		assert_eq!(st, CSR_OK);
		// 有效条目 0-3 被写入.
		assert_eq!(cfg[0], 0xFF);
		assert_eq!(cfg[1], 0xFF);
		assert_eq!(cfg[2], 0xFF);
		assert_eq!(cfg[3], 0xFF);
		// 越界条目 4-7 不得被写入 (回归: 旧实现写满 8 字节导致越界写).
		assert_eq!(cfg[4], 0x00);
		assert_eq!(cfg[5], 0x00);
		assert_eq!(cfg[6], 0x00);
		assert_eq!(cfg[7], 0x00);
	}

	#[test]
	fn pmpcfg_odd_returns_ill() {
		let (pmp, _cfg, _addr) = test_pmp_real(8);
		let mut s = test_state();
		let clint = test_clint(0);
		let mut ctx = test_csr_ctx_pmp(&mut s, &pmp, &clint);
		// pmpcfg1 (0x3A1) is illegal on RV64.
		let st = csr_write(&mut ctx, 0x3A1, 0);
		assert_eq!(st, CSR_ILL);
	}

	#[test]
	fn pmpaddr_write_works() {
		let (pmp, _cfg, addr) = test_pmp_real(16);
		let mut s = test_state();
		let clint = test_clint(0);
		let mut ctx = test_csr_ctx_pmp(&mut s, &pmp, &clint);
		// pmpaddr3 (0x3B3) — write entry 3.
		let st = csr_write(&mut ctx, 0x3B3, 0xDEAD_BEEF_CAFE_BABEu64);
		assert_eq!(st, CSR_OK);
		assert_eq!(addr[3], 0xDEAD_BEEF_CAFE_BABE);
		assert_eq!(addr[2], 0, "adjacent entries untouched");
		assert_eq!(addr[4], 0, "adjacent entries untouched");
	}

	#[test]
	fn pmpaddr_write_out_of_range_is_noop() {
		let (pmp, _cfg, addr) = test_pmp_real(4);
		let mut s = test_state();
		let clint = test_clint(0);
		let mut ctx = test_csr_ctx_pmp(&mut s, &pmp, &clint);
		// pmpaddr4 (0x3B4) — entry 4, but num=4 so valid indices are 0-3.
		let st = csr_write(&mut ctx, 0x3B4, 0xFFFF_FFFF_FFFF_FFFFu64);
		assert_eq!(st, CSR_OK);
		assert_eq!(addr[3], 0, "last valid entry untouched");
	}

	#[test]
	fn pmp_write_does_not_exit() {
		let (pmp, _cfg, _addr) = test_pmp_real(8);
		let mut s = test_state();
		let clint = test_clint(0);
		let mut ctx = test_csr_ctx_pmp(&mut s, &pmp, &clint);
		// All PMP writes should return CSR_OK, not CSR_EXIT.
		assert_eq!(csr_write(&mut ctx, 0x3A0, 0), CSR_OK); // pmpcfg0
		assert_eq!(csr_write(&mut ctx, 0x3A2, 0), CSR_OK); // pmpcfg2
		assert_eq!(csr_write(&mut ctx, 0x3B0, 0), CSR_OK); // pmpaddr0
		assert_eq!(csr_write(&mut ctx, 0x3B7, 0), CSR_OK); // pmpaddr7
		assert_eq!(csr_write(&mut ctx, 0x3EF, 0), CSR_OK); // pmpaddr63 (max)
	}

	// ============================================================
	//  AIA CSR 门控 — 用编译期 PYREMU_AIA 分发期望值
	// ============================================================
	//
	// PYREMU_AIA=false (默认) → 全部 AIA CSR 返回 CSR_ILL, 防止内核
	// 探测到不存在的 IMSIC 硬件后崩溃.
	// PYREMU_AIA=true  → CSR_OK, IMSIC 寄存器可正常读写.
	//
	// 用条件断言而非硬编码期望值, 确保测试在两种构建配置下都正确.

	const AIA_CSRS: [(u16, &str); 8] = [
		(MISELECT, "miselect"),
		(MIREG, "mireg"),
		(SISELECT, "siselect"),
		(SIREG, "sireg"),
		(MTOPEI, "mtopei"),
		(STOPEI, "stopei"),
		(0xFB0, "mtopi"),
		(0xDB0, "stopi"),
	];

	fn aia_expected_status() -> u8 {
		if PYREMU_AIA {
			CSR_OK
		} else {
			CSR_ILL
		}
	}

	#[test]
	fn aia_csr_read_status() {
		let mut s = test_state();
		let clint = test_clint(0);
		let mut ctx = test_csr_ctx(&mut s, &clint);
		let expected = aia_expected_status();
		for (addr, name) in &AIA_CSRS {
			let (_val, st) = csr_read(&mut ctx, *addr);
			assert_eq!(st, expected, "{name} (0x{addr:X}) read status mismatch");
		}
	}

	#[test]
	fn aia_csr_write_status() {
		let mut s = test_state();
		let clint = test_clint(0);
		let mut ctx = test_csr_ctx(&mut s, &clint);
		let expected = aia_expected_status();
		for (addr, name) in &AIA_CSRS {
			let st = csr_write(&mut ctx, *addr, 0xDEAD_BEEFu64);
			assert_eq!(st, expected, "{name} (0x{addr:X}) write status mismatch");
		}
	}

	/// stopi (0xDB0) 回写必须 no-op (QEMU 中 STOPI 为只读 CSR).
	///
	/// ``csrr stopi`` 读出 major identity SEI=9, 内核 read-modify-write 回写
	/// 同一值时, 若在此处认领 top eip, 会提前清掉外部中断, 使后续 STOPEI
	/// (csr_swap) 读到空 — 导致跨 hart IPI / 外设中断握手永远无法完成。
	/// 回归: 回写后 eip 位必须保持 pending。
	#[test]
	fn stopi_write_back_does_not_claim_external_eip() {
		if !PYREMU_AIA {
			return; // 仅 AIA 构建下有意义
		}
		let mut s = test_state();
		let clint = test_clint(0);
		let mut ctx = test_csr_ctx(&mut s, &clint);

		// 准备 S-file: eidelivery=1, eie/eip 置位 IID=10 (外部中断).
		ctx.state.imsic_s.present = 1;
		ctx.state.imsic_s.eidelivery = 1;
		ctx.state.imsic_s.eip_ext_any = 1;
		ctx.state.imsic_s.eie[0].store(1 << 10, Ordering::Relaxed);
		ctx.state.imsic_s.eip[0].store(1 << 10, Ordering::Relaxed);

		// csrr stopi → (9 << 16) | prio (SEI major identity).
		let (val, st) = csr_read(&mut ctx, 0xDB0);
		assert_eq!(st, CSR_OK);
		assert_eq!(
			(val >> 16) & 0x7FF,
			9,
			"stopi must report SEI major identity"
		);

		// 回写同一值 (read-modify-write) — 必须 no-op, eip 保持 pending.
		let wst = csr_write(&mut ctx, 0xDB0, val);
		assert_eq!(wst, CSR_OK);
		assert_ne!(
			ctx.state.imsic_s.eip[0].load(Ordering::Acquire) & (1 << 10),
			0,
			"stopi write-back must NOT claim the external eip"
		);
	}

	/// mtopi (0xFB0) 回写必须 no-op (QEMU 中 MTOPI 为只读 CSR). 对称于 stopi.
	#[test]
	fn mtopi_write_back_does_not_claim_external_eip() {
		if !PYREMU_AIA {
			return; // 仅 AIA 构建下有意义
		}
		let mut s = test_state();
		let clint = test_clint(0);
		let mut ctx = test_csr_ctx(&mut s, &clint);

		ctx.state.imsic_m.present = 1;
		ctx.state.imsic_m.eidelivery = 1;
		ctx.state.imsic_m.eip_ext_any = 1;
		ctx.state.imsic_m.eie[0].store(1 << 10, Ordering::Relaxed);
		ctx.state.imsic_m.eip[0].store(1 << 10, Ordering::Relaxed);

		let (val, st) = csr_read(&mut ctx, 0xFB0);
		assert_eq!(st, CSR_OK);
		assert_eq!(
			(val >> 16) & 0x7FF,
			11,
			"mtopi must report MEI major identity"
		);

		let wst = csr_write(&mut ctx, 0xFB0, val);
		assert_eq!(wst, CSR_OK);
		assert_ne!(
			ctx.state.imsic_m.eip[0].load(Ordering::Acquire) & (1 << 10),
			0,
			"mtopi write-back must NOT claim the external eip"
		);
	}

	/// handle_csr: AIA CSR 读 — 若 !PYREMU_AIA 则应投递 IllInstr.
	#[test]
	fn handle_csr_aia_miselect_read() {
		let mut s = test_state();
		let mut r = make_instr_group();
		let clint = test_clint(0);
		let mut ctx = test_csr_ctx(&mut s, &clint);
		// csrrs t0, miselect, x0  (funct3=CSRRS, rd=5, rs1=0, csr=0x350)
		let advance = handle_csr(&mut ctx, 5, 0, MISELECT, 0b010, 0x3502F073u32, &mut r);
		if PYREMU_AIA {
			assert_eq!(advance, 4, "AIA enabled → normal advance");
			assert_eq!(ctx.state.mcause, 0, "mcause must not be set");
		} else {
			assert_eq!(advance, 0, "AIA disabled → IllInstr redirects PC");
			assert_eq!(
				ctx.state.mcause & 0x7FFF_FFFF_FFFF_FFFF,
				2,
				"mcause must be IllInstr (2)"
			);
		}
	}

	/// handle_csr: AIA CSR 写 → !PYREMU_AIA 时 IllInstr.
	#[test]
	fn handle_csr_aia_miselect_write() {
		let mut s = test_state();
		s.gprs[10] = 0x70; // a0 = IMSIC_EITHRESHOLD select
		let mut r = make_instr_group();
		let clint = test_clint(0);
		let mut ctx = test_csr_ctx(&mut s, &clint);
		// csrrw x0, miselect, a0  (funct3=CSRRW, rd=0, rs1=10, csr=0x350)
		let advance = handle_csr(&mut ctx, 0, 10, MISELECT, 0b001, 0x35051073u32, &mut r);
		if PYREMU_AIA {
			assert_eq!(advance, 4, "AIA enabled → normal advance");
			assert_eq!(ctx.state.imsic_m.select, 0x70, "miselect stored");
		} else {
			assert_eq!(advance, 0, "AIA disabled → IllInstr redirects PC");
			assert_eq!(ctx.state.mcause & 0x7FFF_FFFF_FFFF_FFFF, 2);
		}
	}

	/// handle_csr: mtopi (0xFB0) — OpenSBI 用 csr_read_allowed(CSR_MTOPI) 探测
	/// AIA 存在性. !PYREMU_AIA 时此读必须触发 IllInstr 并设置 mcause.
	#[test]
	fn handle_csr_mtopi_read() {
		let mut s = test_state();
		s.pc = 0x8000_0000; // simulate real instruction fetch address
		let mut r = make_instr_group();
		let clint = test_clint(0);
		let mut ctx = test_csr_ctx(&mut s, &clint);
		// csrrs t1, mtopi, x0  (funct3=CSRRS, rd=6, rs1=0, csr=0xFB0)
		let advance = handle_csr(&mut ctx, 6, 0, 0xFB0, 0b010, 0xFB032F73u32, &mut r);
		if PYREMU_AIA {
			assert_eq!(advance, 4, "AIA enabled → normal advance");
		} else {
			assert_eq!(advance, 0, "AIA disabled → IllInstr redirects PC");
			assert_eq!(ctx.state.mcause & 0x7FFF_FFFF_FFFF_FFFF, 2);
			assert_ne!(ctx.state.mepc, 0, "mepc must be saved by trap delivery");
		}
	}

	/// TEE CSR (mdid 0x5C0 / pmpsplit 0x5C1) 不受 AIA 门控影响.
	#[test]
	fn tee_csrs_not_gated_by_aia() {
		let mut s = test_state();
		let clint = test_clint(0);
		let mut ctx = test_csr_ctx(&mut s, &clint);

		// mdid (0x5C0)
		let (_v, st) = csr_read(&mut ctx, 0x5C0);
		assert_eq!(st, CSR_OK, "mdid read always OK");
		assert_eq!(csr_write(&mut ctx, 0x5C0, 3), CSR_OK);
		assert_eq!(ctx.state.mdid, 3);

		// pmpsplit (0x5C1)
		assert_eq!(csr_write(&mut ctx, 0x5C1, 1), CSR_OK);
		assert_eq!(ctx.state.pmpsplit, 1);
	}

	/// mdid 必须完整 64-bit 保存 — 修复前 `val as u8` 使 csrw mdid, 300 截断为 44,
	/// csrw mdid, 256 截断为 0 (别名回 host). 这是 stress batch-4 (2000 并发飞地)
	/// 挂死的根因: SUSPEND handler 读到 curr==0 误判 host 请求 → sbi_hart_hang.
	#[test]
	fn tee_mdid_full_width_write() {
		let mut s = test_state();
		let clint = test_clint(0);
		let mut ctx = test_csr_ctx(&mut s, &clint);

		// 300 > u8::MAX: 修复前截断为 44
		assert_eq!(csr_write(&mut ctx, 0x5C0, 300), CSR_OK);
		assert_eq!(ctx.state.mdid, 300);

		// 256 ≡ 0 (mod 256): 修复前别名回 host (mdid=0), 绕过 PMP 隔离
		assert_eq!(csr_write(&mut ctx, 0x5C0, 256), CSR_OK);
		assert_eq!(ctx.state.mdid, 256);

		// 读回与写一致
		let (v, st) = csr_read(&mut ctx, 0x5C0);
		assert_eq!(st, CSR_OK);
		assert_eq!(v, 256);
	}
}
