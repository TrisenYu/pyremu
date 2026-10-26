//! 加速并发执行引擎中对 PLIC 的响应实现方案.
//!
//! 布局与 claim、complete 的语义:
//! - 优先级区: 0x000000 + 4*(src-1), u32, bits [2:0] 有效
//! - pending 区: 0x001000 + 4*word, 只读, 由硬件 set_irq 控制
//! - 使能区: 0x002000 + 0x80*ctx + 4*word
//! - context 区: 0x200000 + 0x1000*ctx, 阈值在 +0x000, claim 与 complete 在 +0x004
//!
//! 状态全部定义在 Python 侧分配出的共享 ctypes 数组中，
//! 此处通过 FFI 获取到了对应的内存结构与具体的位置
//! 设备中断的投递目标能直接写入 pending 与 level,
//! 使得加速执行过程中新产生的设备中断对 Rust 侧可见.
//!
//! 中断仲裁结果同步到 mip 的 MEIP 与 SEIP 位

use crate::ffi::FfiPlicCtx;
use crate::state::HartState;
use std::sync::atomic::Ordering;

// PLIC 地址空间常量
pub(crate) const PLIC_PENDING_BASE: u64 = 0x001000;
pub(crate) const PLIC_ENABLE_BASE: u64 = 0x002000;
pub(crate) const PLIC_ENABLE_STRIDE: u64 = 0x80;
pub(crate) const PLIC_CONTEXT_BASE: u64 = 0x200000;
pub(crate) const PLIC_CONTEXT_STRIDE: u64 = 0x1000;
pub(crate) const PLIC_CTX_THRESHOLD_OFF: u64 = 0x000;
pub(crate) const PLIC_CTX_CLAIM_OFF: u64 = 0x004;

/// 返回该 context 中优先级最高的待投递中断源编号, 无符合条件者返回 0.
///
/// 条件: pending[i] 非零且 enable[ctx][i] 置位且 priority[i] > threshold[ctx];
/// 同优先级取编号最小的中断源, 由递增遍历自然满足. 不检查 ``claimed``:
/// pending 按中断源记录, 不按 context 记录, 与 Python 的 ``_find_highest`` 一致.
#[inline]
fn plic_find_highest(plic: &FfiPlicCtx, context: usize) -> u32 {
	if context >= plic.num_contexts as usize {
		return 0;
	}
	// ctypes 数组由 Python 分配, 在本次 FFI 调用期间有效.
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

/// 计算某 hart 由 PLIC 驱动的 mip.MEIP 与 mip.SEIP 位.
pub(crate) fn plic_pending_mip(plic: &FfiPlicCtx, hart_id: u32) -> u64 {
	let m_ctx = 2 * hart_id;
	let s_ctx = m_ctx + 1;
	let mut mip: u64 = 0;
	if (m_ctx as usize) < plic.num_contexts as usize && plic_find_highest(plic, m_ctx as usize) > 0
	{
		mip |= 1 << 11; // MEIP
	}
	if (s_ctx as usize) < plic.num_contexts as usize && plic_find_highest(plic, s_ctx as usize) > 0
	{
		mip |= 1 << 9; // SEIP
	}
	mip
}

/// 未接入 IMSIC 时，将某 hart 的 mip.MEIP 与 mip.SEIP 位同步为就地 PLIC 仲裁的结果.
#[inline]
fn plic_recompute_mip(state: &mut HartState, plic: &FfiPlicCtx) {
	if state.imsic_m.present != 0 || state.imsic_s.present != 0 {
		return;
	}
	let plic_mip = plic_pending_mip(plic, state.mhartid as u32);
	let ext_mask: u64 = (1 << 11) | (1 << 9);
	let cur = state.mip.load(Ordering::Acquire);
	state
		.mip
		.store((cur & !ext_mask) | plic_mip, Ordering::Release);
}

/// 供 step_interrupts 在 ext_irq 置位后核对 MEIP 与 SEIP 的同步.
/// 返回 true 表示已执行同步; ``plic`` 为空指针或已接入 IMSIC 时不执行.
pub(crate) fn plic_sync_mip_if_legacy(state: &mut HartState, plic: *mut FfiPlicCtx) -> bool {
	if plic.is_null() {
		return false;
	}
	let plic = unsafe { &*plic };
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
		if src < 1 || (src as u64) > plic.num_sources as u64 {
			continue;
		}
		if unsafe { *plic.pending.add(src) } != 0 {
			result |= 1 << i;
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
	let v = unsafe {
		*plic
			.enable
			.add((context as usize) * num_words + word_idx as usize)
	};
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
	let src_base = (word_idx as usize) * 32;
	let valid_bits = plic.num_sources as usize - src_base;
	let mut mask: u32 = if valid_bits >= 32 {
		0xFFFF_FFFF
	} else {
		(1u32 << valid_bits) - 1
	};
	if word_idx == 0 {
		mask &= !1u32;
	}
	unsafe {
		*plic
			.enable
			.add((context as usize) * num_words + word_idx as usize) = (val as u32) & mask
	};
}

/// 执行 claim: 返回优先级最高的待处理中断源, 清除其 pending 并记录到 claimed.
/// claim 只清除 pending, 不改动 level.
#[inline]
fn plic_do_claim(context: usize, plic: &FfiPlicCtx) -> u32 {
	let src = plic_find_highest(plic, context);
	if src > 0 {
		unsafe { *plic.pending.add(src as usize) = 0 };
		unsafe { *plic.claimed.add(context) = src };
	}
	src
}

/// 执行 complete: ``claimed[ctx]`` 等于 src 时清除 claimed, 设备电平仍高时重新置位 pending.
#[inline]
fn plic_do_complete(context: usize, src: u32, plic: &FfiPlicCtx) {
	if src <= 0 || src > plic.num_sources {
		return;
	}
	if unsafe { *plic.claimed.add(context) } == src {
		unsafe { *plic.claimed.add(context) = 0 };
		if unsafe { *plic.level.add(src as usize) } != 0 {
			unsafe { *plic.pending.add(src as usize) = 1 };
		}
	}
}

/// 返回 ``Some(val)`` 表示已处理; 返回 ``None`` 表示
/// 地址不在本 PLIC 的 MMIO 范围, 由调用方继续尝试后续就地处理, 或访问宽度不是
/// 2 与 4 字节, 改由 MMIO 退出路径处理.
/// ``plic`` 为空指针表示无 PLIC 设备.
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
	// 地址位于设备区间之外或 context 越界时, 由 is_device_addr 判定为 MMIO 退出.
	let top = PLIC_CONTEXT_BASE + (plic.num_contexts as u64) * PLIC_CONTEXT_STRIDE;
	if offset >= top {
		return None;
	}
	if !is_write {
		let val = plic_mmio_read(offset, plic);
		// claim 清除 pending 后立即重算 mip 并尽早正确地清除 MEIP.
		if offset % PLIC_CONTEXT_STRIDE == PLIC_CTX_CLAIM_OFF && offset >= PLIC_CONTEXT_BASE {
			plic_recompute_mip(state, plic);
		}
		Some(val)
	} else {
		plic_mmio_write(offset, write_data, plic);
		// 使能、阈值、优先级与 complete 都可能改变仲裁结果, 故立即重算 mip;
		// 中断先挂起后使能时也能即时投递.
		plic_recompute_mip(state, plic);
		Some(0)
	}
}

fn plic_mmio_read(offset: u64, plic: &FfiPlicCtx) -> u64 {
	let num_sources = plic.num_sources as u64;
	// 中断源优先级区: 0x000000 - 0x000FFC
	if offset < PLIC_PENDING_BASE {
		let src = offset / 4;
		if src >= 1 && src <= num_sources {
			return (unsafe { *plic.priority.add(src as usize) } & 0x7) as u64;
		}
		return 0;
	}
	// pending 位图: 0x001000 - 0x00107C, 只读
	if offset < PLIC_ENABLE_BASE {
		let word_idx = (offset - PLIC_PENDING_BASE) / 4;
		return plic_read_pending_word(word_idx, plic).unwrap_or(0);
	}
	// 使能位图: 0x002000 + 0x80*ctx + 4*word
	if offset < PLIC_CONTEXT_BASE {
		let rel = offset - PLIC_ENABLE_BASE;
		let context = rel / PLIC_ENABLE_STRIDE;
		let word_idx = (rel % PLIC_ENABLE_STRIDE) / 4;
		return plic_read_enable_word(context, word_idx, plic).unwrap_or(0);
	}
	// context 寄存器区: 0x200000 + 0x1000*ctx
	let rel = offset - PLIC_CONTEXT_BASE;
	let context = rel / PLIC_CONTEXT_STRIDE;
	let ctx_off = rel % PLIC_CONTEXT_STRIDE;
	if context >= plic.num_contexts as u64 {
		return 0;
	}
	match ctx_off {
		PLIC_CTX_THRESHOLD_OFF => (unsafe { *plic.threshold.add(context as usize) } & 0x7) as u64,
		PLIC_CTX_CLAIM_OFF => plic_do_claim(context as usize, plic) as u64,
		_ => 0,
	}
}

fn plic_mmio_write(offset: u64, val: u64, plic: &FfiPlicCtx) {
	let num_sources = plic.num_sources as u64;
	// 中断源优先级区
	if offset < PLIC_PENDING_BASE {
		let src = offset / 4;
		if src >= 1 && src <= num_sources {
			unsafe { *plic.priority.add(src as usize) = (val & 0x7) as u8 };
		}
		return;
	}
	// pending 位图只读, 由硬件的 set_irq 控制
	if offset < PLIC_ENABLE_BASE {
		return;
	}
	// 使能位图
	if offset < PLIC_CONTEXT_BASE {
		let rel = offset - PLIC_ENABLE_BASE;
		let context = rel / PLIC_ENABLE_STRIDE;
		let word_idx = (rel % PLIC_ENABLE_STRIDE) / 4;
		plic_write_enable_word(context, word_idx, val, plic);
		return;
	}
	// context 寄存器
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

	/// 构造最小的 PLIC FFI 上下文, 内存由测试持有.
	fn make_plic(
		num_sources: usize,
		num_contexts: usize,
	) -> (
		Box<[u8]>,
		Box<[u8]>,
		Box<[u8]>,
		Box<[u32]>,
		Box<[u8]>,
		Box<[u32]>,
		FfiPlicCtx,
	) {
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

	/// claim 清除 pending 并记录 claimed; complete 在电平仍高时重新置位 pending.
	#[test]
	fn claim_complete_re_pend_when_level_high() {
		let (_p, mut pending, mut level, mut enable, _t, _c, mut plic) = make_plic(32, 2);
		level[10] = 1;
		pending[10] = 1;
		enable[0] = 1u32 << 10; // 启用 context 0 的中断源 10, 即第 0 个字的 bit 10
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

		// complete 中断源 10, context 0; 电平仍高,重新设置 pending 位
		let _ = try_handle_plic_concurrent(
			plic.base + PLIC_CONTEXT_BASE + PLIC_CTX_CLAIM_OFF,
			true,
			10,
			4,
			&mut state,
			plic_ptr(&mut plic),
		);
		assert_eq!(pending[10], 1, "电平仍高时 complete 应重挂 pending");
		assert_eq!(get_claimed(&plic, 0), 0);
	}

	/// complete 时设备电平已经降低, level 为 0, 不重新设置 pending 位.
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
		let _ = try_handle_plic_concurrent(
			plic.base + PLIC_CONTEXT_BASE + PLIC_CTX_CLAIM_OFF,
			true,
			10,
			4,
			&mut state,
			plic_ptr(&mut plic),
		);
		assert_eq!(pending[10], 0, "level=0 时 complete 不应重挂 pending");
	}

	/// 使能与阈值过滤: 优先级不高于阈值的中断源不可见; 同优先级取编号最小者.
	#[test]
	fn find_highest_respects_threshold_and_priority() {
		let (_p, mut pending, _l, mut enable, mut threshold, _c, mut plic) = make_plic(32, 2);
		pending[5] = 1;
		pending[7] = 1;
		enable[0] = (1u32 << 5) | (1u32 << 7); // 中断源 5 与 7
		set_prio(&mut plic, 5, 1);
		set_prio(&mut plic, 7, 1);
		threshold[0] = 2; // 屏蔽优先级不高于 2 的中断源

		// 阈值过滤后两个中断源都不可见
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
		enable[0] = 1u32 << 3; // context 0 为 hart 0 的 M 模式 context, 启用中断源 3
		set_prio(&mut plic, 3, 1);

		let mut state = make_state();
		state.imsic_m.present = 0;
		state.imsic_s.present = 0;
		state.mip.store(1 << 11, Ordering::Release);

		// claim
		let _ = try_handle_plic_concurrent(
			plic.base + PLIC_CONTEXT_BASE + PLIC_CTX_CLAIM_OFF,
			false,
			0,
			4,
			&mut state,
			plic_ptr(&mut plic),
		);
		assert_eq!(state.mip.load(Ordering::Acquire) & (1 << 11), 0);

		pending[3] = 1;
		state.mip.store(1 << 11, Ordering::Release);
		assert!(plic_sync_mip_if_legacy(&mut state, plic_ptr(&mut plic)));
		assert_ne!(state.mip.load(Ordering::Acquire) & (1 << 11), 0);
	}
}
