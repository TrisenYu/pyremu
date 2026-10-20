//! Sv39 page-table walk and TLB management for the speedup execution engine.
//!
//! Replaces the Python ``sv39_walk`` + ``TLB.lookup/insert/flush`` with
//! Rust-native implementations that operate directly on RAM and inline
//! ``TlbEntry`` arrays in ``HartState``.
//!
//! Implements hardware A/D bit setting: the A (Accessed) bit is set on any
//! PTE that is read during a walk; the D (Dirty) bit is set on the leaf PTE
//! when the access is a write.  This matches RISC-V privileged spec §4.3.2.
//!
//! # SUM/MXR support (RISC-V Privileged Spec §4.1.12)
//!
//! - **SUM** (bit 18): when set, S-mode may access pages marked U=1.
//!   Without SUM, S-mode access to user pages faults.
//! - **MXR** (bit 19): when set, executable-only pages (X=1, R=0) are
//!   readable by load instructions.  Does not affect stores or instruction
//!   fetches.

use core::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use crate::mmu::{pte_parse, sv39_decompose_va};
use crate::pmp::{pmp_ok, PmpCtx};
use crate::state::{riscv_mode, HartState, TlbEntry, CFG_NO_TLB, TLB_ENTRIES};

// ============================================================
//  Page-table walk context
// ============================================================

/// Shared context for memory access, TLB management, and LR/SC reservation.
pub struct WalkCtx {
	pub ram: *mut u8,
	pub ram_size: u64,
	pub ram_base: u64,
	pub shadow_base: u64,
	pub shadow_size: u64,
	/// 已发生的 TLB 刷新次数. 各 hart 在取下一条指令之前比较该计数与自己上次
	/// 见到的取值, 不相等即把自己的条目全部标为陈旧. 条目不被丢弃, 下一次访问
	/// 重走页表, 仍然有效的翻译得以保留.
	pub tlb_flush_cnt: *const AtomicU64,
	/// Per-hart LR reservation slots (indexed by hart_id).
	/// LR sets `lr_reserved[hid] = pa`; any store clears all slots.
	/// Allocated in ``ModuleState``, pointer passed through FFI.
	pub lr_reserved: *mut AtomicU64,
	pub num_harts: u32,
}

/// Result of address translation.
#[derive(Debug)]
pub struct TranslateResult {
	pub pa: u64,
	pub perm: u8,
	pub level: u8,
}

/// Trap codes emitted by translation failures.
#[derive(Debug, PartialEq)]
pub enum TranslateFault {
	PageFault(u64), // cause code (12=Instr, 13=Load, 15=Store)
	AccessFault,
}

// ============================================================
//  Access kind: rwx
// ============================================================

/// 访问类别的位: 读权限、写权限、执行权限.
///
/// 一次访问由这三位的组合给出, 即 `translate_va` 的 `rwx` 参数. 三种访问由
/// 三个构造函数给出, 判定取 `has_read` / `has_write` / `has_exec`.
pub const RWX_READ: u8 = 0b001;
pub const RWX_WRITE: u8 = 0b010;
pub const RWX_EXEC: u8 = 0b100;

/// 数据读取访问.
#[inline]
pub fn rwx_read() -> u8 {
	RWX_READ
}

/// 数据写入访问. 写访问同时要求页面的读权限, 故取值为读位与写位.
#[inline]
pub fn rwx_write() -> u8 {
	RWX_READ | RWX_WRITE
}

/// 取指访问.
#[inline]
pub fn rwx_exec() -> u8 {
	RWX_EXEC
}

/// 本次访问是否请求读权限.
#[inline]
pub fn has_read(rwx: u8) -> bool {
	rwx & RWX_READ != 0
}

/// 本次访问是否请求写权限.
#[inline]
pub fn has_write(rwx: u8) -> bool {
	rwx & RWX_WRITE != 0
}

/// 本次访问是否请求执行权限.
#[inline]
pub fn has_exec(rwx: u8) -> bool {
	rwx & RWX_EXEC != 0
}

/// 本次访问要求页表项具备的权限位.
///
/// 取指不要求 R 位: RISC-V 允许 X-only 页 (R=0, W=0, X=1) 被取指, X 检查由
/// `translate_va` 的 `check_pte_perm` 后置完成. 数据读取在 `mstatus.MXR` 置位时
/// 同样不要求 R 位: 此时仅可执行的页可被读取, R 与 X 的判定同样后置完成. 其余
/// 情形数据读取要求 R. 数据写入要求 R 与 W, 与 MXR 无关.
#[inline]
fn pte_perm_needed(rwx: u8, is_mxr: bool) -> u64 {
	let mut need = 0u64;
	if has_read(rwx) && !has_exec(rwx) && !is_mxr {
		need |= PTE_R;
	}
	if has_write(rwx) {
		need |= PTE_R | PTE_W;
	}
	need
}

/// 记一次 TLB 翻译结果. 命中含权限检查未通过的命中: 该情形下 TLB 已给出翻译.
#[inline]
fn record_tlb_access(state: &mut HartState, is_execute: bool, is_hit: bool) {
	let counter = match (is_execute, is_hit) {
		(true, true) => &mut state.itlb_hits,
		(true, false) => &mut state.itlb_misses,
		(false, true) => &mut state.dtlb_hits,
		(false, false) => &mut state.dtlb_misses,
	};
	*counter = counter.wrapping_add(1);
}

/// 查询 TLB 并记录本次翻译结果, 供 translate_va 在未命中时继续页表遍历.
///
/// CFG_NO_TLB 为真时不做查询也不记录, 一律按未命中处理.
#[inline]
fn tlb_probe_counted(
	state: &mut HartState,
	ctx: &WalkCtx,
	va: u64,
	vpn: u64,
	asid: u16,
	eff_mode: u8,
	is_write: bool,
	is_execute: bool,
) -> TlbProbeResult {
	if CFG_NO_TLB {
		// CFG_NO_TLB 为真时 TLB 查询与插入一并跳过, 每次地址翻译都完整遍历
		// Sv39 页表, 用于隔离地址翻译相关的性能问题. 该常量由 emu-configs.mk
		// 经 Makefile 生成到 configs_gen.rs, 与 Python 侧读取的 PYREMU_NO_TLB
		// 取同一值.
		return TlbProbeResult::Walk(None);
	}
	let tlb = if is_execute {
		&mut state.itlb
	} else {
		&mut state.dtlb
	};
	let probe = tlb_probe(
		tlb,
		vpn,
		va,
		asid,
		if ctx.tlb_flush_cnt.is_null() {
			None
		} else {
			Some(unsafe { &*ctx.tlb_flush_cnt })
		},
		eff_mode,
		state.mstatus,
		is_write,
		is_execute,
	);
	let is_hit = !matches!(probe, TlbProbeResult::Walk(_));
	record_tlb_access(state, is_execute, is_hit);
	probe
}

// ============================================================
//  LR/SC reservation helpers
// ============================================================

/// Set this hart's LR reservation.  pa=0 means "no reservation".
#[inline]
pub(crate) fn lr_set(ctx: &WalkCtx, hid: u8, pa: u64) {
	if ctx.lr_reserved.is_null() {
		return;
	}
	unsafe { &*ctx.lr_reserved.add(hid as usize) }.store(pa, Ordering::Release);
}

/// Check whether this hart still holds a reservation on *pa*.
#[inline]
pub(crate) fn lr_check(ctx: &WalkCtx, hid: u8, pa: u64) -> bool {
	if ctx.lr_reserved.is_null() {
		return false;
	}
	pa != 0 && unsafe { &*ctx.lr_reserved.add(hid as usize) }.load(Ordering::Acquire) == pa
}

/// Clear ALL harts' reservations.  Called on every store / AMO write.
#[inline]
pub(crate) fn lr_clear_all(ctx: &WalkCtx) {
	if ctx.lr_reserved.is_null() {
		return;
	}
	for i in 0..ctx.num_harts as usize {
		unsafe { &*ctx.lr_reserved.add(i) }.store(0, Ordering::Release);
	}
}

// ============================================================
//  RAM read/write (with atomic Acquire/Release for multi-hart)
// ============================================================

/// Read *size* bytes from physical address *pa* in RAM. Little-endian.
/// Aligned 4- and 8-byte reads use atomic Acquire.
#[inline]
pub(crate) fn ram_read(ctx: &WalkCtx, pa: u64, size: u8) -> u64 {
	let off = crate::mem::ram_offset_inline(
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
	let ptr = unsafe { ctx.ram.add(off.unwrap() as usize) };
	match size {
		1 => unsafe { *ptr as u64 },
		2 => {
			let b0 = unsafe { *ptr } as u64;
			let b1 = unsafe { *ptr.add(1) } as u64;
			b0 | (b1 << 8)
		}
		4 if (pa & 3) == 0 => {
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
			let a = unsafe { &*(ptr as *const AtomicU64) };
			a.load(Ordering::Acquire)
		}
		8 => {
			let b = |i: usize| unsafe { *ptr.add(i) } as u64;
			b(0) | (b(1) << 8)
				| (b(2) << 16)
				| (b(3) << 24)
				| (b(4) << 32)
				| (b(5) << 40)
				| (b(6) << 48)
				| (b(7) << 56)
		}
		_ => 0,
	}
}

/// Write *size* bytes of *val* to physical address *pa* in RAM. Little-endian.
/// Aligned 4- and 8-byte writes use atomic Release.  Clears all LR reservations
/// before the store becomes visible (ABA fix).
#[inline]
pub(crate) fn ram_write(ctx: &WalkCtx, pa: u64, val: u64, size: u8) -> bool {
	let off = crate::mem::ram_offset_inline(
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
	lr_clear_all(ctx);
	std::sync::atomic::fence(Ordering::SeqCst);
	let ptr = unsafe { ctx.ram.add(off.unwrap() as usize) };
	match size {
		1 => unsafe {
			*ptr = val as u8;
		},
		2 => unsafe {
			*ptr = val as u8;
			*ptr.add(1) = (val >> 8) as u8;
		},
		4 if (pa & 3) == 0 => {
			let a = unsafe { &*(ptr as *const AtomicU32) };
			a.store(val as u32, Ordering::Release);
		}
		4 => unsafe {
			*ptr = val as u8;
			for i in 1..4 {
				*ptr.add(i) = (val >> (i << 3)) as u8;
			}
		},
		8 if (pa & 7) == 0 => {
			let a = unsafe { &*(ptr as *const AtomicU64) };
			a.store(val, Ordering::Release);
		}
		8 => unsafe {
			*ptr = val as u8;
			for i in 1..8 {
				*ptr.add(i) = (val >> (i << 3)) as u8;
			}
		},
		_ => return false,
	}
	true
}

// ============================================================
//  RAM access for page walks
// ============================================================

/// Compute RAM offset; returns None if address is out of range.
#[inline]
fn ram_offset(ctx: &WalkCtx, pa: u64, size: u64) -> Option<usize> {
	let end = pa + size;
	if pa >= ctx.ram_base && end <= ctx.ram_base + ctx.ram_size {
		return Some((pa - ctx.ram_base) as usize);
	}
	if ctx.shadow_size > 0 {
		let sh_end = ctx.shadow_base + ctx.shadow_size;
		if pa >= ctx.shadow_base && end <= sh_end {
			return Some((pa - ctx.shadow_base) as usize);
		}
	}
	None
}

/// Read a 64-bit raw PTE value from physical memory.
///
/// Uses ``AtomicU64::load(Acquire)`` to synchronise with ``Release`` stores
/// from other hart threads (regular store instructions emit ``Release`` on
/// aligned 8-byte writes via ``ram_write_raw``).  Without this ordering the
/// page-table walker may observe a stale PTE that was overwritten by another
/// hart, populate the TLB with a wrong PA, and cause loads/stores to silently
/// hit the wrong physical page — producing garbage register values that can
/// manifest as near-NULL SIGSEGVs in userspace (the dynamic linker loading a
/// pointer from a corrupted page, then dereferencing it).
#[inline]
fn read_pte(ctx: &WalkCtx, pa: u64) -> u64 {
	let off = match ram_offset(ctx, pa, 8) {
		Some(o) => o,
		None => return 0,
	};
	// PTE addresses are always 8-byte aligned in Sv39 (root_ppn << 12 + vpn * 8).
	let ptr = unsafe { ctx.ram.add(off) };
	let a = unsafe { &*(ptr as *const AtomicU64) };
	let raw = a.load(Ordering::Acquire);
	raw
}

/// Write a 64-bit raw PTE value to physical memory (A/D bit update).
///
/// Uses ``AtomicU64::store(Release)`` to pair with the ``Acquire`` load in
/// CAS-based PTE write-back for A/D bit updates.
///
/// Returns ``true`` if the CAS succeeded (PTE was unchanged since *expected*
/// was read).  On failure another hart modified the PTE concurrently — the
/// caller must re-read and re-validate to avoid overwriting a concurrent
/// unmapping/remapping with stale A/D bits.
#[inline]
fn write_pte_cas(ctx: &WalkCtx, pa: u64, expected: u64, new_val: u64) -> bool {
	if new_val == expected {
		return true;
	}
	let off = match ram_offset(ctx, pa, 8) {
		Some(o) => o,
		None => return false,
	};
	let ptr = unsafe { ctx.ram.add(off) };
	let a = unsafe { &*(ptr as *const AtomicU64) };
	a.compare_exchange(expected, new_val, Ordering::AcqRel, Ordering::Acquire)
		.is_ok()
}

// ============================================================
//  PTE flag constants
// ============================================================

#[allow(dead_code)]
pub(crate) const PTE_V: u64 = 1 << 0;
pub(crate) const PTE_R: u64 = 1 << 1;
pub(crate) const PTE_W: u64 = 1 << 2;
pub(crate) const PTE_X: u64 = 1 << 3;
pub(crate) const PTE_U: u64 = 1 << 4;
const PTE_A: u64 = 1 << 6;
const PTE_D: u64 = 1 << 7;

const SATP_MODE_SV39: u64 = 8;

// ============================================================
//  Sv39 3-level page-table walk — extracted helpers
// ============================================================

/// 读取一个页表项, 读取之前先经 PMP 检查.
///
/// hart 隐式读取页表页, 与显式访存受同一组 PMP 条目约束, 权限一律按读取判定,
/// 特权级与 mstatus 取本次遍历发生时的取值. 被拒绝时返回
/// `TranslateFault::AccessFault`, 由 `translate_va` 的调用方按原访问的类型投递
/// 访问故障: 取指为 1, 读取为 5, 写入为 7, tval 取原访问的虚拟地址.
#[inline]
fn read_pte_checked(
	state: &HartState,
	pmp: &PmpCtx,
	ctx: &WalkCtx,
	addr: u64,
) -> Result<u64, TranslateFault> {
	if !pmp_ok(state, addr, 8, false, false, pmp) {
		return Err(TranslateFault::AccessFault);
	}
	Ok(read_pte(ctx, addr))
}

/// 写入页表项之前先经 PMP 检查.
///
/// 遍历中置位页表项的 A 位与 D 位, 是 hart 在页表遍历期间对页表页发起的物理写入,
/// 与遍历读取页表页受同一组 PMP 条目约束, 权限按写入判定. 被拒绝时返回
/// `TranslateFault::AccessFault`, 由调用方按原访问的类型投递访问故障.
#[inline]
fn write_pte_cas_checked(
	state: &HartState,
	pmp: &PmpCtx,
	ctx: &WalkCtx,
	addr: u64,
	expected: u64,
	new_val: u64,
) -> Result<bool, TranslateFault> {
	if new_val == expected {
		return Ok(true);
	}
	if !pmp_ok(state, addr, 8, true, false, pmp) {
		return Err(TranslateFault::AccessFault);
	}
	Ok(write_pte_cas(ctx, addr, expected, new_val))
}

/// Read and validate the L2 (root) PTE; set A bit.
/// Returns the raw value and parsed PTE on success.
fn walk_l2(
	ctx: &WalkCtx,
	state: &HartState,
	pmp: &PmpCtx,
	root_ppn: u64,
	vpn2: u64,
) -> Result<Option<(u64, crate::mmu::PteFields)>, TranslateFault> {
	let addr = (root_ppn << 12).wrapping_add(vpn2 * 8);
	let raw = read_pte_checked(state, pmp, ctx, addr)?;
	let pte = pte_parse(raw);
	if pte.v == 0 || pte.is_leaf == 0 && pte.is_ptr == 0 {
		return Ok(None);
	}
	if raw & PTE_A != 0 {
		return Ok(Some((raw, pte)));
	}
	if !write_pte_cas_checked(state, pmp, ctx, addr, raw, raw | PTE_A)? {
		let raw2 = read_pte_checked(state, pmp, ctx, addr)?;
		let pte2 = pte_parse(raw2);
		if pte2.v == 0 || pte2.is_leaf == 0 && pte2.is_ptr == 0 {
			return Ok(None);
		}
		return Ok(Some((raw2, pte2)));
	}
	Ok(Some((raw, pte)))
}

/// From a valid L2 pointer, read L1; return `(raw, pte)` or `None`.
/// Sets A bit atomically on the L1 PTE.
fn walk_l1(
	ctx: &WalkCtx,
	state: &HartState,
	pmp: &PmpCtx,
	l2_ppn: u64,
	vpn1: u64,
) -> Result<Option<(u64, crate::mmu::PteFields)>, TranslateFault> {
	let addr = (l2_ppn << 12).wrapping_add(vpn1 * 8);
	let raw = read_pte_checked(state, pmp, ctx, addr)?;
	let pte = pte_parse(raw);
	if pte.v == 0 || pte.is_leaf == 0 && pte.is_ptr == 0 {
		return Ok(None);
	}
	if raw & PTE_A != 0 {
		return Ok(Some((raw, pte)));
	}
	if !write_pte_cas_checked(state, pmp, ctx, addr, raw, raw | PTE_A)? {
		let raw2 = read_pte_checked(state, pmp, ctx, addr)?;
		let pte2 = pte_parse(raw2);
		if pte2.v == 0 || pte2.is_leaf == 0 && pte2.is_ptr == 0 {
			return Ok(None);
		}
		return Ok(Some((raw2, pte2)));
	}
	Ok(Some((raw, pte)))
}

/// Handle an L1 leaf (2 MiB superpage): check permissions, set A+D, assemble PA.
fn walk_l1_leaf(
	ctx: &WalkCtx,
	state: &HartState,
	pmp: &PmpCtx,
	l1_addr: u64,
	l1_raw: u64,
	l1_ppn: u64,
	perm: u8,
	vpn0: u64,
	offset: u64,
	rwx: u8,
) -> Result<Option<TranslateResult>, TranslateFault> {
	// MXR 置位时数据读取可落在仅可执行的页上, 由 check_pte_perm 后置判定,
	// 此处不要求 R 位.
	let is_mxr = (state.mstatus & (1 << 19)) != 0;
	let need_perm = pte_perm_needed(rwx, is_mxr);
	if perm as u64 & need_perm != need_perm {
		return Ok(None);
	}
	let want_a = l1_raw & PTE_A == 0;
	let want_d = has_write(rwx) && l1_raw & PTE_D == 0;
	// PPN[43:9] from PTE, PPN[8:0] from VA vpn0
	let ppn = (l1_ppn & 0xFFFF_FFFF_FFFF_FE00u64) | vpn0;
	let pa = (ppn << 12) | offset;
	if !want_a && !want_d {
		return Ok(Some(TranslateResult {
			pa: pa & 0xFFFF_FFFF_FFFF_FFFF,
			perm,
			level: 1,
		}));
	}

	let mut new_raw = l1_raw;
	if want_a {
		new_raw |= PTE_A;
	}
	if want_d {
		new_raw |= PTE_D;
	}
	if !write_pte_cas_checked(state, pmp, ctx, l1_addr, l1_raw, new_raw)? {
		// CAS failed: another hart modified this PTE concurrently.
		// Re-read and re-validate to avoid using a stale mapping.
		let raw2 = read_pte_checked(state, pmp, ctx, l1_addr)?;
		let pte2 = pte_parse(raw2);
		if pte2.v == 0 || pte2.is_leaf == 0 || pte2.perm as u64 & need_perm != need_perm {
			return Ok(None);
		}
		let ppn2 = (pte2.ppn & 0xFFFF_FFFF_FFFF_FE00u64) | vpn0;
		let pa2 = (ppn2 << 12) | offset;
		return Ok(Some(TranslateResult {
			pa: pa2 & 0xFFFF_FFFF_FFFF_FFFF,
			perm: pte2.perm,
			level: 1,
		}));
	}
	Ok(Some(TranslateResult {
		pa: pa & 0xFFFF_FFFF_FFFF_FFFF,
		perm,
		level: 1,
	}))
}

/// From a valid L1 pointer, read L0; return a 4 KiB leaf result or `None`.
/// Sets A (always) and D (on write) bits on the leaf PTE.
fn walk_l0(
	ctx: &WalkCtx,
	state: &HartState,
	pmp: &PmpCtx,
	l1_ppn: u64,
	vpn0: u64,
	offset: u64,
	rwx: u8,
) -> Result<Option<TranslateResult>, TranslateFault> {
	let addr = (l1_ppn << 12).wrapping_add(vpn0 * 8);
	let raw = read_pte_checked(state, pmp, ctx, addr)?;
	let pte = pte_parse(raw);
	if pte.v == 0 || pte.is_leaf == 0 {
		return Ok(None);
	}
	// MXR 置位时数据读取可落在仅可执行的页上, 由 check_pte_perm 后置判定,
	// 此处不要求 R 位.
	let is_mxr = (state.mstatus & (1 << 19)) != 0;
	let need_perm = pte_perm_needed(rwx, is_mxr);
	if pte.perm as u64 & need_perm != need_perm {
		return Ok(None);
	}
	let want_a = raw & PTE_A == 0;
	let want_d = has_write(rwx) && raw & PTE_D == 0;
	if want_a || want_d {
		let mut new_raw = raw;
		if want_a {
			new_raw |= PTE_A;
		}
		if want_d {
			new_raw |= PTE_D;
		}
		if !write_pte_cas_checked(state, pmp, ctx, addr, raw, new_raw)? {
			let raw2 = read_pte_checked(state, pmp, ctx, addr)?;
			let pte2 = pte_parse(raw2);
			if pte2.v == 0 || pte2.is_leaf == 0 {
				return Ok(None);
			}
			if pte2.perm as u64 & need_perm != need_perm {
				return Ok(None);
			}
			let pa = (pte2.ppn << 12) | offset;
			return Ok(Some(TranslateResult {
				pa: pa & 0xFFFF_FFFF_FFFF_FFFF,
				perm: pte2.perm,
				level: 0,
			}));
		}
	}

	let pa = (pte.ppn << 12) | offset;
	Ok(Some(TranslateResult {
		pa: pa & 0xFFFF_FFFF_FFFF_FFFF,
		perm: pte.perm,
		level: 0,
	}))
}

// ============================================================
//  Sv39 page-table walk — entry point
// ============================================================

/// Perform a full Sv39 page-table walk with hardware A/D bit updates.
///
/// Returns ``Ok(Some(TranslateResult))`` on success (4 KiB or 2 MiB leaf),
/// ``Ok(None)`` if the walk encounters an invalid/non-resident PTE, and
/// ``Err(TranslateFault::AccessFault)`` when a PTE read, or the write that sets
/// the A or D bit, is denied by PMP.
pub fn sv39_walk(
	ctx: &WalkCtx,
	state: &HartState,
	pmp: &PmpCtx,
	satp: u64,
	va: u64,
	rwx: u8,
) -> Result<Option<TranslateResult>, TranslateFault> {
	let mode = satp >> 60;
	if mode != SATP_MODE_SV39 {
		return Ok(None);
	}
	let root_ppn = satp & 0xF_FFFF_FFFF;
	let vpn = sv39_decompose_va(va);

	let (_l2_raw, l2_pte) = match walk_l2(ctx, state, pmp, root_ppn, vpn.vpn2)? {
		Some(v) => v,
		None => return Ok(None),
	};

	if l2_pte.is_ptr == 0 {
		return Ok(None);
	}

	let (l1_raw, l1_pte) = match walk_l1(ctx, state, pmp, l2_pte.ppn, vpn.vpn1)? {
		Some(v) => v,
		None => return Ok(None),
	};

	if l1_pte.is_leaf != 0 {
		let l1_addr = (l2_pte.ppn << 12).wrapping_add(vpn.vpn1 * 8);
		return walk_l1_leaf(
			ctx,
			state,
			pmp,
			l1_addr,
			l1_raw,
			l1_pte.ppn,
			l1_pte.perm,
			vpn.vpn0,
			vpn.offset,
			rwx,
		);
	}

	walk_l0(ctx, state, pmp, l1_pte.ppn, vpn.vpn0, vpn.offset, rwx)
}

// ============================================================
//  TLB operations
// ============================================================

// 条目总数由 emu-configs.mk 的 TLB_ENTRIES 给出, 经 makefile 生成 configs_gen.rs,
// 与 Python 侧 configs_gen.py 取自同一处配置, 两侧容量因此始终一致.
#[inline]
const fn tlb_sz() -> usize {
	TLB_ENTRIES
}

// ============================================================
//  Set-associative organization
// ============================================================

/// 组相联的路数.  条目数组按组划分: 第 s 组的第 w 路位于 s + w * TLB_SETS.
const TLB_WAYS: usize = 4;

/// 组数.  组号由页号的乘法散列高位给出, 故必须为 2 的整数次幂.
const TLB_SETS: usize = TLB_ENTRIES / TLB_WAYS;

// 改动 emu-configs.mk 的 TLB_ENTRIES 时须同时满足: 为 2 的整数次幂, 且不小于
// 路数.  两条约束在此于编译期检查, 不满足则编译失败, 不会退化为错误的分组.
const _: () = assert!(TLB_WAYS.is_power_of_two());
const _: () = assert!(TLB_ENTRIES.is_power_of_two());
const _: () = assert!(TLB_ENTRIES >= TLB_WAYS);

#[inline]
const fn tlb_set_bits() -> u32 {
	TLB_SETS.trailing_zeros()
}

/// 由页号算出所属的组号.
///
/// 取乘法散列的高位而非页号的低位: 相邻 2 MiB 大页的页号相差 512, 低位会被
/// 512 的因子整除, 若直接取低位则内核线性映射的连续大页全部落进同一组,
/// 组相联退化为直接映射.  乘法散列把页号的高低位混合, 消除该规律性.
#[inline]
fn tlb_set(vpn: u64) -> usize {
	const GOLDEN: u64 = 0x9E37_79B9_7F4A_7C15;
	(vpn.wrapping_mul(GOLDEN) >> (64 - tlb_set_bits())) as usize
}

/// 第 s 组第 w 路的条目下标.
#[inline]
const fn tlb_slot(set: usize, way: usize) -> usize {
	set + way * TLB_SETS
}

/// 在某组内取一个可逐出的条目: 优先空槽位, 其次未被访问的槽位,
/// 全部被访问过时清除访问位后取第 0 路.
///
/// 只查看该组的 TLB_WAYS 个槽位, 与组数无关.
#[inline]
fn set_victim(tlb: &mut [TlbEntry], set: usize) -> usize {
	for w in 0..TLB_WAYS {
		let idx = tlb_slot(set, w);
		if tlb[idx].valid == 0 {
			return idx;
		}
	}
	for w in 0..TLB_WAYS {
		let idx = tlb_slot(set, w);
		if tlb[idx].accessed == 0 {
			return idx;
		}
	}
	for w in 0..TLB_WAYS {
		let idx = tlb_slot(set, w);
		tlb[idx].accessed = 0;
	}
	tlb_slot(set, 0)
}

/// Result of a TLB lookup.
pub enum TlbResult {
	/// Clean entry found — ready to use.
	Hit(usize),
	/// Dirty entry found at this index — needs re-walk, but caller should
	/// re-insert the new translation at the same index to avoid wasting a slot.
	DirtyReuse(usize),
	/// No matching entry.
	Miss,
}

/// Find a TLB entry matching *vpn*.  Returns:
/// - ``Hit(idx)`` for a clean match (accessed bit already set).
/// - ``DirtyReuse(idx)`` for a stale match — caller must re-walk and insert
///   at this index to avoid leaking the slot.
/// - ``Miss`` if no entry matches.
///
/// 只查看页号所映射的那一组内的 TLB_WAYS 个条目, 比较次数不随条目总数增长.
#[inline]
pub fn tlb_lookup(tlb: &mut [TlbEntry], vpn: u64, asid: u16) -> TlbResult {
	let set = tlb_set(vpn);
	for w in 0..TLB_WAYS {
		let i = tlb_slot(set, w);
		if tlb[i].valid == 0 || tlb[i].vpn != vpn {
			continue;
		}
		// ASID-tagged: different address space -> miss.
		// asid=0 (Bare mode) matches any entry (no tag).
		if asid != 0 && tlb[i].asid != 0 && tlb[i].asid != asid {
			continue;
		}
		if tlb[i].dirty != 0 {
			return TlbResult::DirtyReuse(i);
		}
		tlb[i].accessed = 1;
		return TlbResult::Hit(i);
	}
	TlbResult::Miss
}

/// 在某组内取一个条目下标, 供插入使用.
///
/// 页号已在该组内时返回其下标, 使重复插入就地更新而不额外占用槽位;
/// 否则返回该组内可逐出的条目.
#[inline]
fn tlb_slot_for_insert(tlb: &mut [TlbEntry], set: usize, vpn: u64) -> usize {
	for w in 0..TLB_WAYS {
		let idx = tlb_slot(set, w);
		if tlb[idx].valid != 0 && tlb[idx].vpn == vpn {
			return idx;
		}
	}
	set_victim(tlb, set)
}

/// Insert or update the translation of *vpn* into its set.
///
/// ``prefer_idx`` should be the index from a prior ``DirtyReuse``; it is used
/// only when it lies in the set that *vpn* maps to.  An index outside that set
/// cannot hold this translation, so the set's own victim is taken instead.
#[inline]
pub fn tlb_insert(
	tlb: &mut [TlbEntry],
	prefer_idx: Option<usize>,
	vpn: u64,
	ppn: u64,
	perm: u8,
	level: u8,
	mdid: u64,
	asid: u16,
) {
	let set = tlb_set(vpn);
	// 槽位 i 属于第 i % TLB_SETS 组, 故该判据即"该槽位位于本页号所映射的组内".
	// 组号由页号唯一决定, 落在别组的槽位无法被后续查询命中, 不能用于存放本条翻译.
	let idx = match prefer_idx {
		Some(i) if i < tlb_sz() && i % TLB_SETS == set => i,
		_ => tlb_slot_for_insert(tlb, set, vpn),
	};
	tlb[idx].vpn = vpn;
	tlb[idx].ppn = ppn;
	tlb[idx].perm = perm;
	tlb[idx].level = level;
	tlb[idx].valid = 1;
	tlb[idx].mdid = mdid;
	tlb[idx].dirty = 0;
	tlb[idx].accessed = 1;
	tlb[idx].asid = asid;
	tlb[idx].tlb_epoch = 0;
}

/// 把全部有效条目标为陈旧. 在 TLB 刷新计数发生变化时调用: 下一次查询重走页表,
/// 取得当前的物理页号.
#[inline]
pub fn tlb_mark_all_dirty(tlb: &mut [TlbEntry]) {
	for e in tlb.iter_mut() {
		if e.valid != 0 {
			e.dirty = 1;
		}
	}
}

/// Flush the entire TLB (invalidate all entries).
#[inline]
pub fn tlb_flush_all(tlb: &mut [TlbEntry]) {
	for e in tlb.iter_mut() {
		e.valid = 0;
		e.dirty = 0;
		e.accessed = 0;
	}
}

/// Flush a specific VPN from the TLB.
#[inline]
pub fn tlb_flush_vpn(tlb: &mut [TlbEntry], vpn: u64) {
	for e in tlb.iter_mut() {
		if e.valid != 0 && e.vpn == vpn {
			e.valid = 0;
		}
	}
}

// ============================================================
//  translate_va helpers — effective mode + permission check
// ============================================================

/// Compute the effective privilege level for memory translation.
///
/// In M-mode, `MPRV=1` overrides the effective mode to `MPP` (RISC-V
/// Privileged Spec §4.1.12).  If MPRV=0 or MPP=M, returns `None`
/// (caller should use Bare translation).
///
/// Non-M modes use their actual privilege level.
/// Effective privilege mode for MMU translation.
///
/// Returns `None` when the MMU should be bypassed (Bare translation):
/// - M-mode without MPRV, or MPRV pointing back to M-mode
/// - D-mode (debug mode) — always bypasses MMU, same as M-mode
fn effective_mode(state: &HartState) -> Option<u8> {
	// D-mode bypasses MMU just like M-mode (matches Python RiscvMode.D).
	if state.mode == riscv_mode::D {
		return None;
	}
	if state.mode != riscv_mode::M {
		return Some(state.mode);
	}
	let mprv = (state.mstatus >> 17) & 1;
	let mpp = (state.mstatus >> 11) & 3;
	if mprv == 0 || mpp == riscv_mode::M as u64 {
		return None; // Bare translation
	}
	Some(mpp as u8)
}

/// Check PTE permissions with SUM / MXR / U-bit awareness.
///
/// Returns `true` if the access is allowed.
fn check_pte_perm(perm: u8, eff_mode: u8, mstatus: u64, is_write: bool, is_execute: bool) -> bool {
	let perm = perm as u64;
	let pte_u = (perm & PTE_U) != 0;
	let pte_r = (perm & PTE_R) != 0;
	let pte_w = (perm & PTE_W) != 0;
	let pte_x = (perm & PTE_X) != 0;

	// U-bit check
	if eff_mode == riscv_mode::U {
		if !pte_u {
			return false;
		}
	} else if eff_mode == riscv_mode::S {
		// SUM=1 允许 S 模式数据访问 U=1 页; 但取指永不许可
		// (RISC-V §4.3.2 基础规则 — S 模式不能从 U 页取指, 与 SUM 无关).
		// 早期实现遗漏 is_execute 分支, 使 SUM 放行了 S 模式对 U 页的取指,
		// 与 Python mem_check_aux._check_pte_perm 分歧.
		if pte_u && (is_execute || (mstatus & (1 << 18)) == 0) {
			return false;
		}
	}

	if is_execute {
		return pte_x;
	}
	if is_write {
		return pte_r && pte_w;
	}
	// Read: MXR=1 allows executable-only pages to be readable
	if !pte_r {
		if (mstatus & (1 << 19)) != 0 && pte_x {
			return true;
		}
		return false;
	}
	true
}

/// Reconstruct a physical address from a TLB entry and virtual address.
#[inline]
fn reconstruct_tlb_pa(entry: &TlbEntry, va: u64) -> u64 {
	let offset = va & 0xFFF;
	let pa = if entry.level == 1 {
		// 2 MiB superpage: PPN[8:0] from VA vpn0
		let ppn = (entry.ppn & 0xFFFF_FFFF_FFFF_FE00) | ((va >> 12) & 0x1FF);
		(ppn << 12) | offset
	} else {
		(entry.ppn << 12) | offset
	};
	pa & 0xFFFF_FFFF_FFFF_FFFF
}

/// Return the appropriate page-fault cause code for the access type.
#[inline]
fn fault_code(is_execute: bool, is_write: bool) -> u64 {
	if is_execute {
		12
	} else if is_write {
		15
	} else {
		13
	}
}

// ============================================================
//  TLB probe with SFENCE.VMA TOCTOU protection
// ============================================================

/// TLB 查询结果.
enum TlbProbeResult {
	/// 命中, 直接返回该翻译.
	Hit(TranslateResult),
	/// 命中, 但权限检查未通过, 返回该故障.
	Fault(TranslateFault),
	/// 转页表遍历, 命中但已陈旧的条目所占用槽位可以复用.
	Walk(Option<usize>),
}

/// 查询 *vpn* 对应的条目. ``tlb_flush_cnt`` 非空时, 在命中之后再次读取该计数:
/// 两次取值不相等说明查询期间有别的 hart 改写了页表, 该条目作废并转页表遍历.
/// 为空时按普通查询处理, 用于单 hart 路径.
#[inline]
fn tlb_probe(
	tlb: &mut [TlbEntry],
	vpn: u64,
	va: u64,
	asid: u16,
	tlb_flush_cnt: Option<&AtomicU64>,
	eff_mode: u8,
	mstatus: u64,
	is_write: bool,
	is_execute: bool,
) -> TlbProbeResult {
	let flush_cnt_before = tlb_flush_cnt.map(|g| g.load(Ordering::Acquire));
	match tlb_lookup(tlb, vpn, asid) {
		TlbResult::Hit(idx) => {
			if let Some(g) = tlb_flush_cnt {
				if flush_cnt_before != Some(g.load(Ordering::Acquire)) {
					tlb[idx].valid = 0; // stale — invalidate, walk
					return TlbProbeResult::Walk(None);
				}
			}
			let e = &tlb[idx];
			if !check_pte_perm(e.perm, eff_mode, mstatus, is_write, is_execute) {
				return TlbProbeResult::Fault(TranslateFault::PageFault(fault_code(
					is_execute, is_write,
				)));
			}
			let pa = reconstruct_tlb_pa(e, va);
			TlbProbeResult::Hit(TranslateResult {
				pa,
				perm: e.perm,
				level: e.level,
			})
		}
		TlbResult::DirtyReuse(idx) => TlbProbeResult::Walk(Some(idx)),
		TlbResult::Miss => TlbProbeResult::Walk(None),
	}
}

// ============================================================
//  Main translate helper
// ============================================================

/// Translate a virtual address to a physical address.
///
/// ``rwx`` gives the access kind: 本次访问请求的读权限、写权限与执行权限的位组合,
/// 取值由 `rwx_read` / `rwx_write` / `rwx_exec` 给出. Returns
/// ``Ok(TranslateResult)`` with PA, permissions, and level, or
/// ``Err(TranslateFault)`` on failure. 页表遍历读取页表页与置位 A 位、D 位的写入
/// 都先经 PMP 检查, 被拒绝时返回 ``TranslateFault::AccessFault``.
pub fn translate_va(
	state: &mut HartState,
	ctx: &WalkCtx,
	va: u64,
	rwx: u8,
	pmp: &PmpCtx,
) -> Result<TranslateResult, TranslateFault> {
	let is_write = has_write(rwx);
	let is_execute = has_exec(rwx);

	// ---- Bare mode / M-mode bypass ----
	if state.mmu_mode != 8 {
		return Ok(TranslateResult {
			pa: va & 0xFFFF_FFFF_FFFF_FFFF,
			perm: 0xF,
			level: 0,
		});
	}

	let eff_mode = match effective_mode(state) {
		Some(m) => m,
		None => {
			return Ok(TranslateResult {
				pa: va & 0xFFFF_FFFF_FFFF_FFFF,
				perm: 0xF,
				level: 0,
			});
		}
	};

	let vpn = va >> 12;

	// ---- TLB lookup ----
	let asid = ((state.satp >> 44) & 0xFFFF) as u16;

	// 查询之前先记下刷新计数. 遍历期间别的 hart 若改写了页表, 本次取得的页表项
	// 已经失效, 该次翻译不写入 TLB, 由写入之前的那次比较判定.
	let flush_cnt_before = if ctx.tlb_flush_cnt.is_null() {
		None
	} else {
		Some(unsafe { &*ctx.tlb_flush_cnt }.load(Ordering::Acquire))
	};

	let reuse_idx =
		match tlb_probe_counted(state, ctx, va, vpn, asid, eff_mode, is_write, is_execute) {
			TlbProbeResult::Hit(result) => return Ok(result),
			TlbProbeResult::Fault(fault) => return Err(fault),
			TlbProbeResult::Walk(reuse) => reuse,
		};

	// ---- Page-table walk ----
	let result = match sv39_walk(ctx, state, pmp, state.satp, va, rwx)? {
		Some(r) => r,
		None => return Err(TranslateFault::PageFault(fault_code(is_execute, is_write))),
	};

	if !check_pte_perm(result.perm, eff_mode, state.mstatus, is_write, is_execute) {
		return Err(TranslateFault::PageFault(fault_code(is_execute, is_write)));
	}

	// ---- 遍历完成之后, 写入 TLB 之前, 再次比较刷新计数 ----
	// 两次取值不相等说明遍历期间页表被改写, 刚取得的页表项可能已失效. 此时不写入
	// TLB: 陈旧条目不进入缓存, 下一次查询按当前的页表项重新遍历.
	let is_flush_cnt_unchanged = match flush_cnt_before {
		None => true,
		Some(gb) => {
			let ga = unsafe { &*ctx.tlb_flush_cnt }.load(Ordering::Acquire);
			gb == ga
		}
	};

	// ---- 刷新计数未变且 TLB 已启用时写入 TLB ----
	if is_flush_cnt_unchanged && !CFG_NO_TLB {
		let tlb_mut = if is_execute {
			&mut state.itlb
		} else {
			&mut state.dtlb
		};
		tlb_insert(
			tlb_mut,
			reuse_idx,
			vpn,
			result.ppn_for_tlb(vpn),
			result.perm,
			result.level,
			state.mdid,
			asid,
		);
	}

	Ok(result)
}

impl TranslateResult {
	/// Reconstruct the PPN value to store in TLB.
	fn ppn_for_tlb(&self, _vpn: u64) -> u64 {
		if self.level == 1 {
			(self.pa >> 12) & 0xFFFF_FFFF_FFFF_FE00
		} else {
			self.pa >> 12
		}
	}
}

// ============================================================
//  Tests
// ============================================================

#[cfg(test)]
mod tests {
	use super::*;
	use crate::state::TlbEntry;

	/// Build a minimal page-table in a Vec and return (vec, WalkCtx).
	/// Places the L2 table at offset 0x0, L1 at 0x1000, L0 at 0x2000.
	fn setup_4k_page_table(ram: &mut [u8], va: u64, pa: u64, perm: u64) -> u64 {
		let vpn = sv39_decompose_va(va);
		let satp = 8u64 << 60; // Sv39, root PPN = 0

		// L2: pointer to L1 at PPN=1 (addr 0x1000)
		let l2_off = (vpn.vpn2 * 8) as usize;
		let l2_val = (1u64 << 10) | PTE_V;
		ram[l2_off..l2_off + 8].copy_from_slice(&l2_val.to_le_bytes());

		// L1: pointer to L0 at PPN=2 (addr 0x2000)
		let l1_off = 0x1000usize + (vpn.vpn1 * 8) as usize;
		let l1_val = (2u64 << 10) | PTE_V;
		ram[l1_off..l1_off + 8].copy_from_slice(&l1_val.to_le_bytes());

		// L0: leaf PTE mapping va->pa
		let l0_off = 0x2000usize + (vpn.vpn0 * 8) as usize;
		let l0_ppn = pa >> 12;
		let l0_val = (l0_ppn << 10) | perm | PTE_V;
		ram[l0_off..l0_off + 8].copy_from_slice(&l0_val.to_le_bytes());

		satp
	}

	/// 空 PMP 配置. 遍历发生在 M 模式且 MPRV=0, PMP 不施加判定, 页表页的读取全部放行.
	fn no_pmp() -> PmpCtx {
		PmpCtx { cfg: std::ptr::null_mut(), addr: std::ptr::null_mut(), num: 0 }
	}

	/// 以 M 模式与空 PMP 配置遍历页表.
	fn walk_with_no_pmp(
		ctx: &WalkCtx,
		satp: u64,
		va: u64,
		rwx: u8,
	) -> Option<TranslateResult> {
		let mut state: HartState = unsafe { std::mem::zeroed() };
		state.mode = riscv_mode::M;
		let pmp = no_pmp();
		sv39_walk(ctx, &state, &pmp, satp, va, rwx).unwrap()
	}

	/// 以给定的 `mstatus` 在 M 模式与空 PMP 配置下遍历页表.
	fn walk_with_mstatus(
		ctx: &WalkCtx,
		satp: u64,
		va: u64,
		rwx: u8,
		mstatus: u64,
	) -> Option<TranslateResult> {
		let mut state: HartState = unsafe { std::mem::zeroed() };
		state.mode = riscv_mode::M;
		state.mstatus = mstatus;
		let pmp = no_pmp();
		sv39_walk(ctx, &state, &pmp, satp, va, rwx).unwrap()
	}

	fn empty_tlb() -> [TlbEntry; TLB_ENTRIES] {
		[TlbEntry::empty(); TLB_ENTRIES]
	}

	/// 取第 *n* 个落在组 *set* 内的页号.
	///
	/// 扫描上界确保组号函数退化时本条以断言失败终止, 而不是无限循环.
	fn vpn_in_set(set: usize, n: u64) -> u64 {
		let mut seen = 0u64;
		for vpn in 0u64..(1 << 24) {
			if tlb_set(vpn) == set {
				if seen == n {
					return vpn;
				}
				seen += 1;
			}
		}
		panic!("set {set} holds fewer than {} page numbers", n + 1)
	}

	fn insert(tlb: &mut [TlbEntry], prefer: Option<usize>, vpn: u64, ppn: u64) {
		tlb_insert(tlb, prefer, vpn, ppn, 0xF, 0, 0, 0);
	}

	#[test]
	fn tlb_lookup_miss() {
		let mut tlb = empty_tlb();
		assert!(matches!(tlb_lookup(&mut tlb, 0x100, 0), TlbResult::Miss));
	}

	#[test]
	fn tlb_insert_and_lookup() {
		let mut tlb = empty_tlb();
		insert(&mut tlb, None, 0x1000, 0x80000);
		let idx = match tlb_lookup(&mut tlb, 0x1000, 0) {
			TlbResult::Hit(i) => i,
			_ => panic!("expected clean hit"),
		};
		let e = &tlb[idx];
		assert_eq!(e.ppn, 0x80000);
		assert_eq!(e.perm, 0xF);
		assert_eq!(e.level, 0);
		assert_eq!(e.valid, 1);
		assert_eq!(e.dirty, 0);
		// Mark dirty -> returns DirtyReuse, not Miss
		tlb[idx].dirty = 1;
		let reuse = match tlb_lookup(&mut tlb, 0x1000, 0) {
			TlbResult::DirtyReuse(i) => i,
			other => panic!(
				"expected DirtyReuse, got {:?}",
				match other {
					TlbResult::Hit(_) => "Hit",
					TlbResult::Miss => "Miss",
					TlbResult::DirtyReuse(_) => unreachable!(),
				}
			),
		};
		assert_eq!(reuse, idx);
		// Re-insert at the dirty slot
		insert(&mut tlb, Some(reuse), 0x1000, 0x90000);
		assert!(matches!(tlb_lookup(&mut tlb, 0x1000, 0), TlbResult::Hit(_)));
		assert_eq!(tlb[reuse].ppn, 0x90000);
	}

	#[test]
	fn test_tlb_flush_all() {
		let mut tlb = empty_tlb();
		insert(&mut tlb, None, 0x1000, 0x80000);
		crate::translate::tlb_flush_all(&mut tlb);
		assert!(matches!(tlb_lookup(&mut tlb, 0x1000, 0), TlbResult::Miss));
	}

	#[test]
	fn test_tlb_flush_vpn() {
		let mut tlb = empty_tlb();
		insert(&mut tlb, None, 0x1000, 0x80000);
		insert(&mut tlb, None, 0x2000, 0x90000);
		crate::translate::tlb_flush_vpn(&mut tlb, 0x1000);
		assert!(matches!(tlb_lookup(&mut tlb, 0x1000, 0), TlbResult::Miss));
		assert!(matches!(tlb_lookup(&mut tlb, 0x2000, 0), TlbResult::Hit(_)));
	}

	/// 命中项的槽位号必须落在该页号所映射的组内.
	///
	/// 查询只查看这一组的 TLB_WAYS 个槽位, 故该不变量成立即比较次数与条目总数无关.
	/// 逐组填充, 使插入次序与组号互不相关: 若槽位改回按插入次序线性分配, 命中槽位
	/// 与组号不再一致, 本条失败.
	#[test]
	fn tlb_hit_always_lies_in_own_set() {
		let mut tlb = empty_tlb();
		for set in 0..TLB_SETS {
			for way in 0..TLB_WAYS {
				let vpn = vpn_in_set(set, way as u64);
				insert(&mut tlb, None, vpn, vpn * 0x1000);
			}
		}
		for set in 0..TLB_SETS {
			for way in 0..TLB_WAYS {
				let vpn = vpn_in_set(set, way as u64);
				let idx = match tlb_lookup(&mut tlb, vpn, 0) {
					TlbResult::Hit(i) => i,
					_ => panic!("expected a hit in set {set} way {way}"),
				};
				assert_eq!(idx % TLB_SETS, set, "slot {idx} is not in set {set}");
			}
		}
	}

	/// 逐出只在目标组内发生, 且组内有效条目数不超过路数.
	#[test]
	fn tlb_evicts_within_own_set_only() {
		let mut tlb = empty_tlb();
		let target = 0usize;
		let other = 1usize;
		// 在另一组放一个条目, 逐出不得触及它.
		let keeper = vpn_in_set(other, 0);
		insert(&mut tlb, None, keeper, 0xABCDE);

		// 填满目标组.
		for way in 0..TLB_WAYS {
			let vpn = vpn_in_set(target, way as u64);
			insert(&mut tlb, None, vpn, vpn * 0x1000);
		}
		// 再插入 TLB_WAYS 个同组的新页号, 全部应落在目标组内.
		for n in TLB_WAYS..(2 * TLB_WAYS) {
			let vpn = vpn_in_set(target, n as u64);
			insert(&mut tlb, None, vpn, vpn * 0x1000);
			assert!(
				matches!(tlb_lookup(&mut tlb, vpn, 0), TlbResult::Hit(_)),
				"freshly inserted vpn {vpn:#x} is not resident"
			);
		}

		let in_target = (0..TLB_WAYS)
			.filter(|w| tlb[tlb_slot(target, *w)].valid != 0)
			.count();
		assert_eq!(in_target, TLB_WAYS, "set {target} should hold {TLB_WAYS}");
		assert!(matches!(tlb_lookup(&mut tlb, keeper, 0), TlbResult::Hit(_)));
	}

	#[test]
	fn tlb_dirty_reuse() {
		let mut tlb = empty_tlb();
		let set = 7usize;
		// 填满该组, 使后续插入必然触发逐出.
		for way in 0..TLB_WAYS {
			let vpn = vpn_in_set(set, way as u64);
			insert(&mut tlb, None, vpn, vpn * 0x1000);
		}
		let target = vpn_in_set(set, 0);
		let idx = match tlb_lookup(&mut tlb, target, 0) {
			TlbResult::Hit(i) => i,
			_ => panic!("expected a hit before marking dirty"),
		};
		tlb[idx].dirty = 1;
		let reuse = match tlb_lookup(&mut tlb, target, 0) {
			TlbResult::DirtyReuse(i) => i,
			_ => panic!("expected DirtyReuse"),
		};
		assert_eq!(reuse, idx);
		// 在脏槽位就地重填: 该组有效条目数不变, 不额外占用槽位.
		insert(&mut tlb, Some(reuse), target, 0x90000);
		assert_eq!(tlb[reuse].ppn, 0x90000);
		assert_eq!(tlb[reuse].dirty, 0);
		let in_set = (0..TLB_WAYS)
			.filter(|w| tlb[tlb_slot(set, *w)].valid != 0)
			.count();
		assert_eq!(in_set, TLB_WAYS, "refill must not add a valid entry");
	}

	/// 相邻 2 MiB 大页的页号相差 512, 组号不得仅由页号低位决定.
	///
	/// 若组号取页号低位, 内核线性映射的连续大页全部落进同一组, 组相联退化为直接
	/// 映射, 可用容量从 TLB_ENTRIES 降到 TLB_WAYS. 本条以低位方案必然失败.
	#[test]
	fn tlb_set_spreads_consecutive_megapages() {
		let mut seen = [false; TLB_SETS];
		let mut distinct = 0usize;
		for k in 0..TLB_SETS as u64 {
			let vpn = 0x40000 + k * 512;
			let set = tlb_set(vpn);
			if !seen[set] {
				seen[set] = true;
				distinct += 1;
			}
		}
		assert!(
			distinct > TLB_WAYS,
			"consecutive megapages land in only {distinct} sets"
		);
	}

	#[test]
	fn sv39_walk_sets_a_bit_on_leaf() {
		let mut ram = vec![0u8; 0x3000];
		let satp = setup_4k_page_table(&mut ram, 0x1000, 0x8000_0000, PTE_R | PTE_W | PTE_X);
		let ctx = WalkCtx {
			ram: ram.as_mut_ptr(),
			ram_size: ram.len() as u64,
			ram_base: 0,
			shadow_base: 0,
			shadow_size: 0,
			tlb_flush_cnt: std::ptr::null(),
			lr_reserved: std::ptr::null_mut(),
			num_harts: 1,
		};

		let result = walk_with_no_pmp(&ctx, satp, 0x1000, rwx_read());
		assert!(result.is_some());
		let l0_off = 0x2000usize + (sv39_decompose_va(0x1000).vpn0 * 8) as usize;
		let l0_val = u64::from_le_bytes(ram[l0_off..l0_off + 8].try_into().unwrap());
		assert_ne!(l0_val & PTE_A, 0, "A bit should be set on leaf PTE");
	}

	#[test]
	fn sv39_walk_sets_d_bit_on_write() {
		let mut ram = vec![0u8; 0x3000];
		let satp = setup_4k_page_table(&mut ram, 0x2000, 0x8000_1000, PTE_R | PTE_W);
		let ctx = WalkCtx {
			ram: ram.as_mut_ptr(),
			ram_size: ram.len() as u64,
			ram_base: 0,
			shadow_base: 0,
			shadow_size: 0,
			tlb_flush_cnt: std::ptr::null(),
			lr_reserved: std::ptr::null_mut(),
			num_harts: 1,
		};

		let result = walk_with_no_pmp(&ctx, satp, 0x2000, rwx_write());
		assert!(result.is_some());
		let l0_off = 0x2000usize + (sv39_decompose_va(0x2000).vpn0 * 8) as usize;
		let l0_val = u64::from_le_bytes(ram[l0_off..l0_off + 8].try_into().unwrap());
		assert_ne!(
			l0_val & PTE_D,
			0,
			"D bit should be set on write to leaf PTE"
		);
	}

	#[test]
	fn sv39_walk_sets_a_bit_on_intermediate_levels() {
		let mut ram = vec![0u8; 0x3000];
		let satp = setup_4k_page_table(&mut ram, 0x3000, 0x8000_2000, PTE_R | PTE_W);
		let ctx = WalkCtx {
			ram: ram.as_mut_ptr(),
			ram_size: ram.len() as u64,
			ram_base: 0,
			shadow_base: 0,
			shadow_size: 0,
			tlb_flush_cnt: std::ptr::null(),
			lr_reserved: std::ptr::null_mut(),
			num_harts: 1,
		};

		let _result = walk_with_no_pmp(&ctx, satp, 0x3000, rwx_read());
		let l2_off = (sv39_decompose_va(0x3000).vpn2 * 8) as usize;
		let l2_val = u64::from_le_bytes(ram[l2_off..l2_off + 8].try_into().unwrap());
		assert_ne!(l2_val & PTE_A, 0, "A bit should be set on L2 PTE");

		let l1_off = 0x1000usize + (sv39_decompose_va(0x3000).vpn1 * 8) as usize;
		let l1_val = u64::from_le_bytes(ram[l1_off..l1_off + 8].try_into().unwrap());
		assert_ne!(l1_val & PTE_A, 0, "A bit should be set on L1 PTE");
	}

	#[test]
	fn sv39_walk_allows_fetch_of_xonly_leaf() {
		// 回归: X-only 页 (R=0, W=0, X=1) 必须允许取指.
		// 修复前 walk_l0 对取指也强制 PTE_R, 使 runtime 的 X-only text
		// (trap_vector 高 VA 映射) 取指被误判 InstrPageFault -> Batch-20 trap loop.
		let mut ram = vec![0u8; 0x3000];
		let va = 0xffff_ffe0_0000_1000u64; // 与 enclave trap_vector 同型的高 VA
		let pa = 0x8ae0_1000u64;
		let satp = setup_4k_page_table(&mut ram, va, pa, PTE_X);
		let ctx = WalkCtx {
			ram: ram.as_mut_ptr(),
			ram_size: ram.len() as u64,
			ram_base: 0,
			shadow_base: 0,
			shadow_size: 0,
			tlb_flush_cnt: std::ptr::null(),
			lr_reserved: std::ptr::null_mut(),
			num_harts: 1,
		};

		// 取指: 必须成功, 且翻译到正确的 PA.
		let fetch = walk_with_no_pmp(&ctx, satp, va, rwx_exec());
		assert!(
			fetch.is_some(),
			"fetch from an X-only leaf must succeed (returned None before the fix)"
		);
		assert_eq!(
			fetch.unwrap().pa,
			pa,
			"fetch from an X-only leaf must translate to the mapped PA"
		);

		// 数据读: R=0 页不可读, 仍应失败.
		let load = walk_with_no_pmp(&ctx, satp, va, rwx_read());
		assert!(
			load.is_none(),
			"data load from an X-only leaf must stay denied (PTE_R required)"
		);

		// 写: R&W 均缺, 仍应失败.
		let store = walk_with_no_pmp(&ctx, satp, va, rwx_write());
		assert!(
			store.is_none(),
			"store to an X-only leaf must stay denied (PTE_R|PTE_W required)"
		);
	}

	#[test]
	fn sv39_walk_honors_mxr_for_data_read_of_xonly_leaf() {
		// 回归: OpenSBI 的 sbi_get_insn 以 MPRV|MXR 在 mepc 处读取半字, 该地址落在
		// sittim 的 X-only .text 页上. 修复前 walk_l0 在 MXR 置位时仍要求 PTE_R,
		// 数据读取返回 None, 固件把 store access fault 的模拟改报为取指页错误.
		let mut ram = vec![0u8; 0x3000];
		let va = 0xffff_ffe0_0001_7376u64;
		let pa = 0x8301_7376u64;
		let satp = setup_4k_page_table(&mut ram, va, pa, PTE_X);
		let ctx = WalkCtx {
			ram: ram.as_mut_ptr(),
			ram_size: ram.len() as u64,
			ram_base: 0,
			shadow_base: 0,
			shadow_size: 0,
			tlb_flush_cnt: std::ptr::null(),
			lr_reserved: std::ptr::null_mut(),
			num_harts: 1,
		};

		let mxr = 1u64 << 19;
		let load = walk_with_mstatus(&ctx, satp, va, rwx_read(), mxr);
		assert!(
			load.is_some(),
			"data load from an X-only leaf must succeed when MXR is set (returned None before the fix)"
		);
		assert_eq!(
			load.unwrap().pa,
			pa,
			"data load under MXR must translate to the mapped PA"
		);

		// MXR 未置位时数据读取仍然拒绝.
		assert!(
			walk_with_mstatus(&ctx, satp, va, rwx_read(), 0).is_none(),
			"data load from an X-only leaf must stay denied when MXR is clear"
		);

		// 写不受 MXR 影响.
		assert!(
			walk_with_mstatus(&ctx, satp, va, rwx_write(), mxr).is_none(),
			"store to an X-only leaf must stay denied when MXR is set"
		);

		// 成功的读取置位叶子的 A 位, 说明遍历走完了叶子项.
		let l0_off = 0x2000usize + (sv39_decompose_va(va).vpn0 * 8) as usize;
		let l0_val = u64::from_le_bytes(ram[l0_off..l0_off + 8].try_into().unwrap());
		assert_ne!(l0_val & PTE_A, 0, "A bit should be set on the MXR read");
	}

	#[test]
	fn sv39_walk_honors_mxr_for_data_read_of_xonly_superpage() {
		// 同一判定位于 walk_l1_leaf, 覆盖 2 MiB 叶子项.
		let mut ram = vec![0u8; 0x2000];
		let va = 0xffff_ffe0_0040_0000u64;
		let pa_mega = 0x8ae0_0000u64;
		let vpn = sv39_decompose_va(va);
		let satp = 8u64 << 60;

		let l2_off = (vpn.vpn2 * 8) as usize;
		let l2_val = (1u64 << 10) | PTE_V;
		ram[l2_off..l2_off + 8].copy_from_slice(&l2_val.to_le_bytes());

		let l1_off = 0x1000usize + (vpn.vpn1 * 8) as usize;
		let l1_val = ((pa_mega >> 12) << 10) | PTE_X | PTE_V;
		ram[l1_off..l1_off + 8].copy_from_slice(&l1_val.to_le_bytes());

		let ctx = WalkCtx {
			ram: ram.as_mut_ptr(),
			ram_size: ram.len() as u64,
			ram_base: 0,
			shadow_base: 0,
			shadow_size: 0,
			tlb_flush_cnt: std::ptr::null(),
			lr_reserved: std::ptr::null_mut(),
			num_harts: 1,
		};

		let mxr = 1u64 << 19;
		let load = walk_with_mstatus(&ctx, satp, va, rwx_read(), mxr);
		assert!(
			load.is_some(),
			"data load from an X-only superpage must succeed when MXR is set (returned None before the fix)"
		);
		assert_eq!(
			load.unwrap().pa,
			pa_mega,
			"data load under MXR must translate to the mapped PA"
		);
		assert!(
			walk_with_mstatus(&ctx, satp, va, rwx_read(), 0).is_none(),
			"data load from an X-only superpage must stay denied when MXR is clear"
		);
		// 写入不受 MXR 影响.
		assert!(
			walk_with_mstatus(&ctx, satp, va, rwx_write(), mxr).is_none(),
			"store to an X-only superpage must stay denied when MXR is set"
		);
	}

	/// 遍历阶段要求的权限位: 数据读取只在 MXR 清除时要求 R; 写入恒要求 R|W;
	/// 取指不要求任何位, 由 `check_pte_perm` 后置判定.
	#[test]
	fn pte_perm_needed_defers_read_requirement_under_mxr() {
		assert_eq!(
			pte_perm_needed(rwx_read(), false),
			PTE_R,
			"a data read must require PTE_R when MXR is clear"
		);
		assert_eq!(
			pte_perm_needed(rwx_read(), true),
			0,
			"a data read must not require PTE_R when MXR is set"
		);
		assert_eq!(
			pte_perm_needed(rwx_write(), false),
			PTE_R | PTE_W,
			"a store must require PTE_R and PTE_W when MXR is clear"
		);
		assert_eq!(
			pte_perm_needed(rwx_write(), true),
			PTE_R | PTE_W,
			"a store must require PTE_R and PTE_W when MXR is set"
		);
		assert_eq!(
			pte_perm_needed(rwx_exec(), false),
			0,
			"a fetch must not require PTE_R"
		);
		assert_eq!(
			pte_perm_needed(rwx_exec(), true),
			0,
			"a fetch must not require PTE_R when MXR is set"
		);
	}

	/// MXR 置位时 TLB 命中的 X-only 条目必须放行数据读取, MXR 清除后同一条目
	/// 必须报故障. 遍历阶段的放宽只对新装入的页面生效是不够的.
	#[test]
	fn tlb_probe_honors_mxr_for_xonly_entry() {
		let mut tlb = empty_tlb();
		let vpn = 0x0017_376u64;
		let ppn = 0x8_3017u64;
		tlb_insert(&mut tlb, None, vpn, ppn, PTE_X as u8, 0, 0, 0);

		let mxr = 1u64 << 19;
		let va = vpn << 12;
		match tlb_probe(&mut tlb, vpn, va, 0, None, riscv_mode::S, mxr, false, false) {
			TlbProbeResult::Hit(r) => assert_eq!(
				r.pa,
				ppn << 12,
				"data load under MXR must translate to the entry's PA"
			),
			_ => panic!("data load of an X-only TLB entry must hit when MXR is set"),
		}
		assert!(
			matches!(
				tlb_probe(&mut tlb, vpn, va, 0, None, riscv_mode::S, 0, false, false),
				TlbProbeResult::Fault(_)
			),
			"data load of an X-only TLB entry must fault when MXR is clear"
		);
		assert!(
			matches!(
				tlb_probe(&mut tlb, vpn, va, 0, None, riscv_mode::S, mxr, true, false),
				TlbProbeResult::Fault(_)
			),
			"store to an X-only TLB entry must fault when MXR is set"
		);
	}

	/// MXR 只放宽数据读取: 取指仍只认 X 位, 写入仍要求 R|W, 仅有 W 位的
	/// 页不属于可执行页, 读取同样拒绝.
	#[test]
	fn mxr_widens_only_data_reads() {
		let mxr = 1u64 << 19;
		let xonly = PTE_X as u8;
		assert!(
			check_pte_perm(xonly, riscv_mode::S, mxr, false, false),
			"MXR must allow a data read of an X-only page"
		);
		assert!(
			!check_pte_perm(xonly, riscv_mode::S, mxr, true, false),
			"MXR must not allow a store to an X-only page"
		);
		assert!(
			check_pte_perm(xonly, riscv_mode::S, mxr, false, true),
			"MXR must not affect a fetch of an X-only page"
		);
		let ronly = PTE_R as u8;
		assert!(
			!check_pte_perm(ronly, riscv_mode::S, mxr, false, true),
			"MXR must not make an R-only page fetchable"
		);
		let wonly = PTE_W as u8;
		assert!(
			!check_pte_perm(wonly, riscv_mode::S, mxr, false, false),
			"MXR must not make a W-only page readable"
		);
	}

	#[test]
	fn bare_mode_translate_is_identity() {}

	#[test]
	fn mmode_bypasses_sv39_even_when_satp_is_sv39() {
		let mut state: HartState = unsafe { std::mem::zeroed() };
		state.mode = riscv_mode::M;
		state.mmu_mode = 8;
		state.satp = 8u64 << 60;
		state.mstatus = 0;

		let mut ram = vec![0u8; 0x1000];
		let ctx = WalkCtx {
			ram: ram.as_mut_ptr(),
			ram_size: ram.len() as u64,
			ram_base: 0,
			shadow_base: 0,
			shadow_size: 0,
			tlb_flush_cnt: std::ptr::null(),
			lr_reserved: std::ptr::null_mut(),
			num_harts: 1,
		};

		let pmp = no_pmp();
		let result = translate_va(&mut state, &ctx, 0x8000_1000, rwx_read(), &pmp);
		assert!(
			result.is_ok(),
			"M-mode with MPRV=0 must use Bare translation"
		);
		let tr = result.unwrap();
		assert_eq!(tr.pa, 0x8000_1000);
		assert_eq!(tr.level, 0);
	}

	#[test]
	fn mmode_mprv1_with_mpp_s_uses_sv39() {
		let mut state: HartState = unsafe { std::mem::zeroed() };
		state.mode = riscv_mode::M;
		state.mmu_mode = 8;
		state.satp = 8u64 << 60;
		state.mstatus = (1u64 << 17) | (1u64 << 11);

		let mut ram = vec![0u8; 0x1000];
		let ctx = WalkCtx {
			ram: ram.as_mut_ptr(),
			ram_size: ram.len() as u64,
			ram_base: 0,
			shadow_base: 0,
			shadow_size: 0,
			tlb_flush_cnt: std::ptr::null(),
			lr_reserved: std::ptr::null_mut(),
			num_harts: 1,
		};

		let pmp = no_pmp();
		let result = translate_va(&mut state, &ctx, 0x8000_1000, rwx_read(), &pmp);
		assert!(
			result.is_err(),
			"MPRV=1+MPP=S must attempt Sv39 translation"
		);
	}

	/// D-mode (debug mode) must bypass Sv39 translation just like M-mode.
	/// Regression: effective_mode() previously returned `Some(D)` which
	/// caused page-walk attempts in debug mode.
	#[test]
	fn dmode_bypasses_sv39_translation() {
		let mut state: HartState = unsafe { std::mem::zeroed() };
		state.mode = riscv_mode::D;
		state.mmu_mode = 8;
		state.satp = 8u64 << 60; // Sv39, root PPN = 0

		let mut ram = vec![0u8; 0x1000];
		let ctx = WalkCtx {
			ram: ram.as_mut_ptr(),
			ram_size: ram.len() as u64,
			ram_base: 0,
			shadow_base: 0,
			shadow_size: 0,
			tlb_flush_cnt: std::ptr::null(),
			lr_reserved: std::ptr::null_mut(),
			num_harts: 1,
		};

		let pmp = no_pmp();
		let result = translate_va(&mut state, &ctx, 0x8000_1000, rwx_read(), &pmp);
		assert!(
			result.is_ok(),
			"D-mode must use Bare translation (bypass Sv39)"
		);
		let tr = result.unwrap();
		assert_eq!(tr.pa, 0x8000_1000, "D-mode VA must equal PA");
	}

	/// D-mode instruction fetch must also bypass MMU translation.
	#[test]
	fn dmode_fetch_bypasses_sv39() {
		let mut state: HartState = unsafe { std::mem::zeroed() };
		state.mode = riscv_mode::D;
		state.mmu_mode = 8;
		state.satp = 8u64 << 60;

		let mut ram = vec![0u8; 0x1000];
		let ctx = WalkCtx {
			ram: ram.as_mut_ptr(),
			ram_size: ram.len() as u64,
			ram_base: 0,
			shadow_base: 0,
			shadow_size: 0,
			tlb_flush_cnt: std::ptr::null(),
			lr_reserved: std::ptr::null_mut(),
			num_harts: 1,
		};

		let pmp = no_pmp();
		let pa = crate::hart_sched::translate_fetch_pc_concurrent(
			&mut state,
			&ctx,
			0x8000_2000,
			&pmp,
		);
		assert_eq!(
			pa,
			Some(0x8000_2000),
			"D-mode fetch must bypass Sv39 (VA==PA)"
		);
	}

	/// 页表遍历读取页表页时经 PMP 检查.
	///
	/// 遍历由 hart 隐式发起, 与显式访存受同一组 PMP 条目约束, 权限按读取判定.
	/// 被拒绝时返回 `TranslateFault::AccessFault`, 由调用方按原访问的类型投递
	/// 访问故障 (取指 1, 读取 5, 写入 7); 缺页异常只表示页表项无效.
	#[test]
	fn pte_read_denied_by_pmp_is_access_fault() {
		let mut ram = vec![0u8; 0x3000];
		let satp = setup_4k_page_table(&mut ram, 0x1000, 0x8000_0000, PTE_R | PTE_W);
		let ctx = WalkCtx {
			ram: ram.as_mut_ptr(),
			ram_size: ram.len() as u64,
			ram_base: 0,
			shadow_base: 0,
			shadow_size: 0,
			tlb_flush_cnt: std::ptr::null(),
			lr_reserved: std::ptr::null_mut(),
			num_harts: 1,
		};

		let mut state: HartState = unsafe { std::mem::zeroed() };
		state.mode = riscv_mode::S;
		state.mmu_mode = 8;
		state.satp = satp;

		// 只放开 L0 页 [0x2000, 0x4000), 根页表所在的页 0 不在任何条目之内.
		// 根页表的页表项读取因此被拒绝, 此时尚未取得任何叶子项的权限.
		let restricted_cfg = [0x1fu8]; // NAPOT | R | W | X
		let restricted_addr = [0x0000_0000_0000_0bffu64]; // NAPOT, [0x2000, 0x4000)
		let restricted_pmp = PmpCtx {
			cfg: restricted_cfg.as_ptr() as *mut u8,
			addr: restricted_addr.as_ptr() as *mut u64,
			num: 1,
		};

		let walk = sv39_walk(&ctx, &state, &restricted_pmp, satp, 0x1000, rwx_read());
		assert!(
			matches!(walk, Err(TranslateFault::AccessFault)),
			"page-table read outside every PMP entry must yield AccessFault"
		);

		// 同一条拒绝经由 translate_va 返回, 不是缺页异常. 调用方据此投递
		// 访问故障并把原访问的虚拟地址写入 tval.
		let tr = translate_va(&mut state, &ctx, 0x1000, rwx_write(), &restricted_pmp);
		assert!(
			matches!(tr, Err(TranslateFault::AccessFault)),
			"store translation must report AccessFault, not PageFault(15)"
		);

		// 对照: 根页表也落在条目之内时, 同一次遍历成功. 故障来自 PMP 条目,
		// 而不是页表本身.
		let full_cfg = [0x1fu8];
		let full_addr = [0x0000_0000_0000_07ffu64]; // NAPOT, [0, 0x4000)
		let full_pmp = PmpCtx {
			cfg: full_cfg.as_ptr() as *mut u8,
			addr: full_addr.as_ptr() as *mut u64,
			num: 1,
		};
		let walk = sv39_walk(&ctx, &state, &full_pmp, satp, 0x1000, rwx_read());
		assert!(
			matches!(walk, Ok(Some(_))),
			"the same walk must succeed once the root page is covered"
		);
	}

	/// 置位 A 位的写入被 PMP 拒绝时, 翻译返回访问故障.
	///
	/// 条目覆盖三个页表页但只有读权限: 遍历的读取放行, 置位 A 位的写入被拒绝,
	/// 该次翻译按原访问投递访问故障, 页表项保持不变.
	#[test]
	fn pte_ad_write_denied_by_pmp_is_access_fault() {
		let mut ram = vec![0u8; 0x3000];
		let satp = setup_4k_page_table(&mut ram, 0x1000, 0x8000_0000, PTE_R | PTE_W);
		let ctx = WalkCtx {
			ram: ram.as_mut_ptr(),
			ram_size: ram.len() as u64,
			ram_base: 0,
			shadow_base: 0,
			shadow_size: 0,
			tlb_flush_cnt: std::ptr::null(),
			lr_reserved: std::ptr::null_mut(),
			num_harts: 1,
		};

		let mut state: HartState = unsafe { std::mem::zeroed() };
		state.mode = riscv_mode::S;
		state.mmu_mode = 8;
		state.satp = satp;

		// NAPOT [0, 0x4000) 覆盖 L2 / L1 / L0 三个页表页, 权限只有读.
		let read_only_cfg = [0x19u8]; // NAPOT | R
		let full_addr = [0x0000_0000_0000_07ffu64];
		let read_only_pmp = PmpCtx {
			cfg: read_only_cfg.as_ptr() as *mut u8,
			addr: full_addr.as_ptr() as *mut u64,
			num: 1,
		};

		let walk = sv39_walk(&ctx, &state, &read_only_pmp, satp, 0x1000, rwx_read());
		assert!(
			matches!(walk, Err(TranslateFault::AccessFault)),
			"A-bit update denied by PMP must yield AccessFault"
		);

		// 页表项未被部分改写.
		let l2_off = (sv39_decompose_va(0x1000).vpn2 * 8) as usize;
		let l2_val = u64::from_le_bytes(ram[l2_off..l2_off + 8].try_into().unwrap());
		assert_eq!(l2_val & PTE_A, 0, "denied A-bit update must not write the PTE");

		// 对照: 同一条目加上写权限之后, 同一次遍历成功.
		let writable_cfg = [0x1bu8]; // NAPOT | R | W
		let writable_pmp = PmpCtx {
			cfg: writable_cfg.as_ptr() as *mut u8,
			addr: full_addr.as_ptr() as *mut u64,
			num: 1,
		};
		let walk = sv39_walk(&ctx, &state, &writable_pmp, satp, 0x1000, rwx_read());
		assert!(
			matches!(walk, Ok(Some(_))),
			"the same walk must succeed once the write permission is granted"
		);
	}

	/// Regression: S 模式取指永不从 U 页执行, 即使 SUM=1 也不能放行
	/// (RISC-V §4.3.2 基础规则 — SUM 只放宽数据访问, 取指不受 SUM 影响).
	/// 早期实现遗漏 is_execute 分支, 使 SUM 错误放行了 S 模式对 U 页的取指,
	/// 与 Python mem_check_aux._check_pte_perm 分歧: 内核 sret 到 U 页时
	/// native 模式取指被错误允许/拒绝不一致, 产生难以定位的执行路径分叉.
	#[test]
	fn s_mode_fetch_of_user_page_faults_even_with_sum() {
		// U=1 可执行页: SUM=1 时 S 模式取指 U 页必须 fault.
		let perm = (PTE_U | PTE_X) as u8;
		assert!(
			!check_pte_perm(perm, riscv_mode::S, 1 << 18, false, true),
			"S-mode fetch of a U page must fault regardless of SUM"
		);
		// SUM=0 时同样 fault.
		assert!(
			!check_pte_perm(perm, riscv_mode::S, 0, false, true),
			"S-mode fetch of a U page must fault when SUM=0"
		);
		// 对照: 同一 U|X 页在 U 模式取指合法 (不会走到 U-check 拒绝).
		assert!(
			check_pte_perm(perm, riscv_mode::U, 0, false, true),
			"U-mode fetch of a U|X page must be allowed"
		);
	}

	/// Regression: SUM=1 只放宽 S 模式对 U 页的*数据*访问 — 读取 (或写入) 放行,
	/// 取指仍拒绝. 与 Python mem_check_aux._check_pte_perm 保持一致.
	#[test]
	fn sum_allows_data_access_but_not_fetch() {
		// U=1 可读页: SUM=1 只放行 S 模式*数据*访问, 取指仍拒绝.
		let perm = (PTE_U | PTE_R) as u8;
		assert!(
			check_pte_perm(perm, riscv_mode::S, 1 << 18, false, false),
			"S-mode data load of a U page must be allowed when SUM=1"
		);
		// SUM=0: S 模式数据读 U 页拒绝.
		assert!(
			!check_pte_perm(perm, riscv_mode::S, 0, false, false),
			"S-mode data load of a U page must be denied when SUM=0"
		);
		// 取指 (即使读到可读页) 仍拒绝 — 取指只看 X 位.
		assert!(
			!check_pte_perm(perm, riscv_mode::S, 1 << 18, false, true),
			"S-mode fetch of a read-only U page must fault (no X bit)"
		);
	}
}
