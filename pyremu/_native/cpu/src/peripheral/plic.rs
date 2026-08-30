//! Inline PLIC MMIO handler for the concurrent (thread-per-hart) engine.
//!
//! 对照 Python ``pyremu/interrupt/plic.py`` 的寄存器布局与 claim/complete 语义:
//! - Priority 区  0x000000 + 4*(src-1)      (u32, bits [2:0] 有效)
//! - Pending 区   0x001000 + 4*word          (只读, 由硬件 set_irq 控制)
//! - Enable 区    0x002000 + 0x80*ctx + 4*word
//! - Context 区   0x200000 + 0x1000*ctx      (threshold +0x000, claim/complete +0x004)
//!
//! 状态全部放在 Python 侧分配的共享 ctypes 数组 (``FfiPlicCtx``) 中:
//! Rust 内联读写 (claim 清 pending / complete 重挂 / enable 写), Python 在
//! 加速执行边界 marshal/unmarshal, 设备 ``raise_device_irq`` 直接 write-through
//! pending/level, 保证 batch 内新设备中断对 Rust 可见 (与 CLINT/virtio 模式一致).
//!
//! 中断仲裁后同步 mip.MEIP/SEIP: 仅在 IMSIC 缺席 (纯 legacy PLIC) 时由 Rust 内联
//! 维护 (``plic_recompute_mip``); AIA 模式 (IMSIC present) 的 mip 由 IMSIC 驱动,
//! Python 侧 ``_native_sync_plic_mip`` 在边界处合并 PLIC 兜底.

use crate::state::FfiPlicCtx;
use crate::state::HartState;
use std::sync::atomic::Ordering;

// ---- PLIC 地址空间常量 (mirror plic.py) ----
pub(crate) const PLIC_PENDING_BASE: u64 = 0x001000;
pub(crate) const PLIC_ENABLE_BASE: u64 = 0x002000;
pub(crate) const PLIC_ENABLE_STRIDE: u64 = 0x80;
pub(crate) const PLIC_CONTEXT_BASE: u64 = 0x200000;
pub(crate) const PLIC_CONTEXT_STRIDE: u64 = 0x1000;
pub(crate) const PLIC_CTX_THRESHOLD_OFF: u64 = 0x000;
pub(crate) const PLIC_CTX_CLAIM_OFF: u64 = 0x004;

/// 返回 context 中优先级最高且符合条件的中断源编号, 无则返回 0.
///
/// 条件: pending[i] AND enable[ctx][i] AND priority[i] > threshold[ctx];
/// 同优先级取最小 source id (id 递增遍历自然满足).  不检查 ``claimed`` —
/// pending 是 per-source (非 per-context), 与 Python ``_find_highest`` 一致.
///
/// enable 位图位映射与 Python ``_read/_write_enable_word`` 完全一致:
/// word w bit i ↔ source (w*32 + i); source 0 保留恒为 0.
#[inline]
fn plic_find_highest(plic: &FfiPlicCtx, context: usize) -> u32 {
	if context >= plic.num_contexts as usize {
		return 0;
	}
	// Safety: Python 保证 backing ctypes 数组在 FFI 调用期间存活.
	let threshold = unsafe { *plic.threshold.add(context) } as u32;
	let num_words = ((plic.num_sources as usize) + 31) / 32;
	let mut best = 0u32;
	let mut best_prio = 0u32;
	for src in 1..=plic.num_sources as usize {
		if unsafe { *plic.pending.add(src) } == 0 {
			continue;
		}
		let word = src / 32;
		let bit = 1u32 << (src % 32);
		if unsafe { *plic.enable.add(context * num_words + word) } & bit == 0 {
			continue;
		}
		let pri = unsafe { *plic.priority.add(src) } as u32;
		if pri <= threshold {
			continue;
		}
		if pri > best_prio {
			best_prio = pri;
			best = src as u32;
		}
	}
	best
}

/// 计算某 hart 的 PLIC 驱动 MEIP/SEIP 位 (M-context = 2*hart_id, S = +1).
pub(crate) fn plic_pending_mip(plic: &FfiPlicCtx, hart_id: u32) -> u64 {
	let m_ctx = 2 * hart_id;
	let s_ctx = m_ctx + 1;
	let mut mip: u64 = 0;
	if (m_ctx as usize) < plic.num_contexts as usize && plic_find_highest(plic, m_ctx as usize) > 0 {
		mip |= 1 << 11; // MEIP
	}
	if (s_ctx as usize) < plic.num_contexts as usize && plic_find_highest(plic, s_ctx as usize) > 0 {
		mip |= 1 << 9; // SEIP
	}
	mip
}

/// 将某 hart 的 mip.MEIP/SEIP 位重新同步到内联 PLIC 仲裁结果.
///
/// 仅 IMSIC 缺席 (纯 legacy) 时有效 — AIA 模式 mip 由 IMSIC eip 驱动, PLIC 只是
/// 兜底源 (由 Python 边界处 ``_native_sync_plic_mip`` 合并), 此处不得覆盖.
#[inline]
fn plic_recompute_mip(state: &mut HartState, plic: &FfiPlicCtx) {
	if state.imsic_m.present != 0 || state.imsic_s.present != 0 {
		return;
	}
	let plic_mip = plic_pending_mip(plic, state.mhartid as u32);
	let ext_mask: u64 = (1 << 11) | (1 << 9);
	let cur = state.mip.load(Ordering::Acquire);
	state.mip.store((cur & !ext_mask) | plic_mip, Ordering::Release);
}

/// 与 M 模式无关的 MEIP/SEIP 同步 — 供 step_interrupts 在 ext_irq 置位后核对.
/// 返回 true 表示已执行 (调用方无需再走其它路径).  ``plic`` 为原始指针,
/// null 或 IMSIC 在场时视为未执行.
pub(crate) fn plic_sync_mip_if_legacy(
	state: &mut HartState,
	plic: *mut FfiPlicCtx,
) -> bool {
	if plic.is_null() {
		return false;
	}
	let plic = unsafe { &*plic };
	// 门控: 仅当 mip 外部位已置位才扫描 (零位时零开销).
	if plic.base == 0
		|| state.imsic_m.present != 0
		|| state.imsic_s.present != 0
		|| (state.mip.load(Ordering::Acquire) & ((1 << 11) | (1 << 9))) == 0
	{
		return false;
	}
	plic_recompute_mip(state, plic);
	true
}

#[inline]
fn plic_read_pending_word(word_idx: u64, plic: &FfiPlicCtx) -> Option<u64> {
	let mut result: u64 = 0;
	let base = (word_idx << 5) as usize;
	for i in 0..32u64 {
		let src = base + i as usize;
		if src >= 1 && (src as u64) <= plic.num_sources as u64 {
			if unsafe { *plic.pending.add(src) } != 0 {
				result |= 1 << i;
			}
		}
	}
	Some(result)
}

#[inline]
fn plic_read_enable_word(context: u64, word_idx: u64, plic: &FfiPlicCtx) -> Option<u64> {
	if context >= plic.num_contexts as u64 {
		return Some(0);
	}
	let num_words = ((plic.num_sources as usize) + 31) / 32;
	if (word_idx as usize) >= num_words {
		return Some(0);
	}
	let v = unsafe { *plic.enable.add((context as usize) * num_words + word_idx as usize) };
	Some(v as u64)
}

#[inline]
fn plic_write_enable_word(context: u64, word_idx: u64, val: u64, plic: &FfiPlicCtx) {
	if context >= plic.num_contexts as u64 {
		return;
	}
	let num_words = ((plic.num_sources as usize) + 31) / 32;
	if (word_idx as usize) >= num_words {
		return;
	}
	// 掩掉本 word 内超出 num_sources 的位, 与 Python _write_enable_word
	// 的 ``src > num_sources: continue`` 一致; word 0 额外清 source 0 (保留位).
	let src_base = (word_idx as usize) * 32;
	let valid_bits = plic.num_sources as usize - src_base;
	let mut mask: u32 = if valid_bits >= 32 {
		0xFFFF_FFFF
	} else {
		(1u32 << valid_bits) - 1
	};
	if word_idx == 0 {
		mask &= !1u32; // source 0 保留
	}
	unsafe { *plic.enable.add((context as usize) * num_words + word_idx as usize) = (val as u32) & mask };
}

/// Claim: 返回最高优先级待处理源, 清除 pending, 记录 claimed.
/// 与 Python ``_do_claim`` 语义一致 (claim 仅清 pending, 不碰 level).
#[inline]
fn plic_do_claim(context: usize, plic: &FfiPlicCtx) -> u32 {
	let src = plic_find_highest(plic, context);
	if src > 0 {
		unsafe { *plic.pending.add(src as usize) = 0 };
		unsafe { *plic.claimed.add(context) = src };
	}
	src
}

/// Complete: 若 claimed[ctx] == src 则清除 claimed, 且设备电平仍高时重挂 pending
/// (gateway 语义, 对照 Python ``_do_complete``).
#[inline]
fn plic_do_complete(context: usize, src: u32, plic: &FfiPlicCtx) {
	if src > 0 && src <= plic.num_sources {
		if unsafe { *plic.claimed.add(context) } == src {
			unsafe { *plic.claimed.add(context) = 0 };
			if unsafe { *plic.level.add(src as usize) } != 0 {
				unsafe { *plic.pending.add(src as usize) = 1 };
			}
		}
	}
}

/// Inline PLIC MMIO read/write.  返回 ``Some(val)`` 表示已处理; ``None`` 表示
/// 不在此 PLIC 的 MMIO 范围 (调用方继续后续 inline 链) 或 size 不受支持
/// (对齐 Python read/write 的 2/4 字节限制, 其它尺寸回落到 MMIO 退出).
/// ``plic`` 为原始指针 — null 表示无 PLIC 设备, 直接回落.
pub(crate) fn try_handle_plic_concurrent(
	pa: u64,
	is_write: bool,
	write_data: u64,
	size: u8,
	state: &mut HartState,
	plic: *mut FfiPlicCtx,
) -> Option<u64> {
	if plic.is_null() {
		return None;
	}
	let plic = unsafe { &*plic };
	if plic.base == 0 || (size != 2 && size != 4) {
		return None;
	}
	let offset = pa.wrapping_sub(plic.base);
	// 地址落在设备 region 之外 (含越界 context) -> 交由 is_device_addr 判定 MMIO 退出.
	let top = PLIC_CONTEXT_BASE + (plic.num_contexts as u64) * PLIC_CONTEXT_STRIDE;
	if offset >= top {
		return None;
	}
	if !is_write {
		let val = plic_mmio_read(offset, plic);
		// Claim 清除了 pending — 立即重算 mip, 避免延后到 step_interrupts 才清 MEIP.
		if offset % PLIC_CONTEXT_STRIDE == PLIC_CTX_CLAIM_OFF
			&& offset >= PLIC_CONTEXT_BASE
		{
			plic_recompute_mip(state, plic);
		}
		Some(val)
	} else {
		plic_mmio_write(offset, write_data, plic);
		// enable/threshold/priority/complete 都可能改变仲裁结果 — 立即重算 mip
		// (覆盖 "先挂起后使能" 的延迟投递场景, 无需等到 batch 边界).
		plic_recompute_mip(state, plic);
		Some(0)
	}
}

fn plic_mmio_read(offset: u64, plic: &FfiPlicCtx) -> u64 {
	let num_sources = plic.num_sources as u64;
	// Source priorities (0x000000 - 0x000FFC)
	if offset < PLIC_PENDING_BASE {
		let src = offset / 4;
		if src >= 1 && src <= num_sources {
			return (unsafe { *plic.priority.add(src as usize) } & 0x7) as u64;
		}
		return 0;
	}
	// Pending bits (0x001000 - 0x00107C), read-only
	if offset < PLIC_ENABLE_BASE {
		let word_idx = (offset - PLIC_PENDING_BASE) / 4;
		return plic_read_pending_word(word_idx, plic).unwrap_or(0);
	}
	// Enable bits (0x002000 - ...)
	if offset < PLIC_CONTEXT_BASE {
		let rel = offset - PLIC_ENABLE_BASE;
		let context = rel / PLIC_ENABLE_STRIDE;
		let word_idx = (rel % PLIC_ENABLE_STRIDE) / 4;
		return plic_read_enable_word(context, word_idx, plic).unwrap_or(0);
	}
	// Context registers (0x200000 - ...)
	let rel = offset - PLIC_CONTEXT_BASE;
	let context = rel / PLIC_CONTEXT_STRIDE;
	let ctx_off = rel % PLIC_CONTEXT_STRIDE;
	if context >= plic.num_contexts as u64 {
		return 0;
	}
	match ctx_off {
		PLIC_CTX_THRESHOLD_OFF => {
			(unsafe { *plic.threshold.add(context as usize) } & 0x7) as u64
		}
		PLIC_CTX_CLAIM_OFF => plic_do_claim(context as usize, plic) as u64,
		_ => 0,
	}
}

fn plic_mmio_write(offset: u64, val: u64, plic: &FfiPlicCtx) {
	let num_sources = plic.num_sources as u64;
	// Source priorities
	if offset < PLIC_PENDING_BASE {
		let src = offset / 4;
		if src >= 1 && src <= num_sources {
			unsafe { *plic.priority.add(src as usize) = (val & 0x7) as u8 };
		}
		return;
	}
	// Pending bits — read-only (由硬件 set_irq 控制)
	if offset < PLIC_ENABLE_BASE {
		return;
	}
	// Enable bits
	if offset < PLIC_CONTEXT_BASE {
		let rel = offset - PLIC_ENABLE_BASE;
		let context = rel / PLIC_ENABLE_STRIDE;
		let word_idx = (rel % PLIC_ENABLE_STRIDE) / 4;
		plic_write_enable_word(context, word_idx, val, plic);
		return;
	}
	// Context registers
	let rel = offset - PLIC_CONTEXT_BASE;
	let context = rel / PLIC_CONTEXT_STRIDE;
	let ctx_off = rel % PLIC_CONTEXT_STRIDE;
	if context >= plic.num_contexts as u64 {
		return;
	}
	match ctx_off {
		PLIC_CTX_THRESHOLD_OFF => {
			unsafe { *plic.threshold.add(context as usize) = (val & 0x7) as u8 };
		}
		PLIC_CTX_CLAIM_OFF => plic_do_complete(context as usize, val as u32, plic),
		_ => {}
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	/// 构造一个最小 PLIC FFI 上下文 (内存由测试持有).
	fn make_plic(num_sources: usize, num_contexts: usize) -> (Box<[u8]>, Box<[u8]>, Box<[u8]>, Box<[u32]>, Box<[u8]>, Box<[u32]>, FfiPlicCtx) {
		let num_words = (num_sources + 31) / 32;
		let mut priority = vec![0u8; num_sources + 1].into_boxed_slice();
		let mut pending = vec![0u8; num_sources + 1].into_boxed_slice();
		let mut level = vec![0u8; num_sources + 1].into_boxed_slice();
		let mut enable = vec![0u32; num_contexts * num_words].into_boxed_slice();
		let mut threshold = vec![0u8; num_contexts].into_boxed_slice();
		let mut claimed = vec![0u32; num_contexts].into_boxed_slice();
		let plic = FfiPlicCtx {
			base: 0x0C00_0000,
			num_sources: num_sources as u32,
			num_contexts: num_contexts as u32,
			priority: priority.as_mut_ptr(),
			pending: pending.as_mut_ptr(),
			level: level.as_mut_ptr(),
			enable: enable.as_mut_ptr(),
			threshold: threshold.as_mut_ptr(),
			claimed: claimed.as_mut_ptr(),
		};
		(priority, pending, level, enable, threshold, claimed, plic)
	}

	fn plic_ptr(plic: &mut FfiPlicCtx) -> *mut FfiPlicCtx {
		plic as *mut FfiPlicCtx
	}

	fn set_prio(plic: &mut FfiPlicCtx, src: usize, pri: u8) {
		unsafe { plic.priority.add(src).write(pri) };
	}

	fn get_claimed(plic: &FfiPlicCtx, ctx: usize) -> u32 {
		unsafe { plic.claimed.add(ctx).read() }
	}

	fn make_state() -> HartState {
		let mut s: HartState = unsafe { std::mem::zeroed() };
		s.mhartid = 0;
		s
	}

	/// Claim 清除 pending 并记录 claimed; complete 在电平仍高时重挂 pending.
	#[test]
	fn claim_complete_re_pend_when_level_high() {
		let (_p, mut pending, mut level, mut enable, _t, _c, mut plic) = make_plic(32, 2);
		level[10] = 1;
		pending[10] = 1;
		enable[0] = 1u32 << 10; // context 0, source 10 (bit 10 of word 0)
		set_prio(&mut plic, 10, 3);

		let mut state = make_state();
		let _ = try_handle_plic_concurrent(
			plic.base + PLIC_CONTEXT_BASE + PLIC_CTX_CLAIM_OFF,
			false,
			0,
			4,
			&mut state,
			plic_ptr(&mut plic),
		);

		assert_eq!(pending[10], 0, "claim 应清除 pending");
		assert_eq!(get_claimed(&plic, 0), 10);

		// complete source 10 (ctx 0), level 仍高 -> 重挂 pending
		let _ = try_handle_plic_concurrent(plic.base + PLIC_CONTEXT_BASE + PLIC_CTX_CLAIM_OFF, true, 10, 4, &mut state, plic_ptr(&mut plic));
		assert_eq!(pending[10], 1, "电平仍高时 complete 应重挂 pending");
		assert_eq!(get_claimed(&plic, 0), 0);
	}

	/// complete 时设备已降电平 (level=0) -> 不重挂 pending.
	#[test]
	fn complete_with_level_low_does_not_repend() {
		let (_p, mut pending, mut level, mut enable, _t, _c, mut plic) = make_plic(32, 2);
		level[10] = 0;
		pending[10] = 1;
		enable[0] = 1u32 << 10;
		set_prio(&mut plic, 10, 3);

		let mut state = make_state();
		let _ = try_handle_plic_concurrent(
			plic.base + PLIC_CONTEXT_BASE + PLIC_CTX_CLAIM_OFF,
			false,
			0,
			4,
			&mut state,
			plic_ptr(&mut plic),
		);
		let _ = try_handle_plic_concurrent(plic.base + PLIC_CONTEXT_BASE + PLIC_CTX_CLAIM_OFF, true, 10, 4, &mut state, plic_ptr(&mut plic));
		assert_eq!(pending[10], 0, "level=0 时 complete 不应重挂 pending");
	}

	/// 使能 + 阈值过滤: 低于阈值的 source 不可见; 同优先级取最小 id.
	#[test]
	fn find_highest_respects_threshold_and_priority() {
		let (_p, mut pending, _l, mut enable, mut threshold, _c, mut plic) = make_plic(32, 2);
		pending[5] = 1;
		pending[7] = 1;
		enable[0] = (1u32 << 5) | (1u32 << 7); // sources 5, 7
		set_prio(&mut plic, 5, 1);
		set_prio(&mut plic, 7, 1);
		threshold[0] = 2; // 屏蔽 priority <= 2

		// 阈值过滤后两个 source 都不可见
		assert_eq!(plic_find_highest(&plic, 0), 0);

		threshold[0] = 0;
		assert_eq!(plic_find_highest(&plic, 0), 5, "同优先级取最小 id");

		set_prio(&mut plic, 7, 4);
		assert_eq!(plic_find_highest(&plic, 0), 7, "最高优先级胜出");
	}

	/// mip 同步: claim 清除最后一个 pending 后 MEIP 立即清零.
	#[test]
	fn claim_clears_meip_immediately() {
		let (_p, mut pending, _l, mut enable, _t, _c, mut plic) = make_plic(32, 2);
		pending[3] = 1;
		enable[0] = 1u32 << 3; // context 0 (M-ctx of hart 0) source 3
		set_prio(&mut plic, 3, 1);

		let mut state = make_state();
		state.imsic_m.present = 0;
		state.imsic_s.present = 0;
		// 模拟 Python 边界 sync 后 MEIP 置位
		state.mip.store(1 << 11, Ordering::Release);

		// claim
		let _ = try_handle_plic_concurrent(plic.base + PLIC_CONTEXT_BASE + PLIC_CTX_CLAIM_OFF, false, 0, 4, &mut state, plic_ptr(&mut plic));
		assert_eq!(state.mip.load(Ordering::Acquire) & (1 << 11), 0, "claim 后 MEIP 应立即清除");

		// 重新挂起 source 3 (设备再次 raise) -> 下一 ext_irq 轮询置 MEIP, sync 核对保持
		pending[3] = 1;
		state.mip.store(1 << 11, Ordering::Release);
		assert!(plic_sync_mip_if_legacy(&mut state, plic_ptr(&mut plic)), "legacy 且 mip 置位时应执行核对");
		assert_ne!(state.mip.load(Ordering::Acquire) & (1 << 11), 0, "有 pending 时 MEIP 保持");
	}
}
