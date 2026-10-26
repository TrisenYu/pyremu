//! IMSIC 中断文件在加速执行引擎中的处理.
//!
//! 本文件提供:
//! 1. 跨 hart 置位与清除 eip 位, 并更新对应文件的 eip_ext_any 与 hart 的 mip 位;
//! 2. 把 IMSIC 的 MMIO 地址解码为 hart 与文件;
//! 3. 在加速执行引擎中直接处理 seteipnum 与 clreipnum 的写入, 以及 IMSIC 的
//!    MMIO 读取;
//! 4. 经 mireg 与 sireg 访问 IMSIC 寄存器;
//! 5. 读取与清除 topei, 把 eip 与 eie 的状态同步到 mip 的 MEIP 与 SEIP.
//!
//! eip 的写入值是中断编号, 不是中断原因; 写入只作用于被寻址的文件. eidelivery 为 0
//! 时 IPI 由 CLINT MSIP 投递, eip 生命周期由 Python 侧 trap_handler.py 管理.

use crate::concurrent::ConcurrentClintCtx;
use crate::state::{HartState, ImsicFile, CFG_AIA};
use core::sync::atomic::Ordering;

use std::thread::Thread;

// IPI 使用的中断编号, 即写入 seteipnum 的值. 该编号是外部中断编号, 不是中断原因.
// OpenSBI 以 1 向目标 hart 的 M 文件发送 IPI, 接收方在 M 模式经 MTOPEI 读出后交给
// sbi_ipi_process. 常量 3 只用于 M 文件的 IPI, OpenSBI 不发出该值.
pub(crate) const IID_S_IPI: u32 = 1; // OpenSBI 的 IPI 中断编号
pub(crate) const IID_M_IPI: u32 = 3; // M 文件的 IPI 中断编号

// ============================================================
//  跨 hart 操作 IMSIC eip
// ============================================================

/// 置位目标 hart 的 IMSIC eip 位, 并置位对应文件的 eip_ext_any 与该 hart 的 mip 位.
///
/// file_off 为 0x0000 时操作 M 文件并置位 MEIP, 为其他值时操作 S 文件并置位 SEIP.
/// 指针为空时直接返回; 调用方须保证 hart 小于 hart_states 指向的数组长度; 访问的
/// 字段都是原子字段, 可从其他线程调用. eip_ext_any 为 0 时 imsic_topei_peek 不扫描
/// 外部中断而直接返回 0, 即使 eip 与 eie 都已置位.
#[inline]
pub(crate) fn imsic_eip_set(
	hart_states: *const HartState,
	hart: usize,
	file_off: u64,
	eip_num: u32,
) {
	if hart_states.is_null() {
		return;
	}
	// 共享引用: 访问的 eip、eip_ext_any、mip 均为原子字段;
	// 不对其他线程的 HartState 创建 &mut.
	let target = unsafe { &*(hart_states.add(hart) as *const HartState) };
	let (eip_array, eip_ext_any) = if file_off == 0x0000 {
		(&target.imsic_m.eip, &target.imsic_m.eip_ext_any)
	} else {
		(&target.imsic_s.eip, &target.imsic_s.eip_ext_any)
	};
	let mip_bit: u64 = if file_off == 0x0000 { 1 << 11 } else { 1 << 9 };
	let word = (eip_num >> 5) as usize;
	if word >= eip_array.len() {
		return;
	}
	let bit = eip_num & 31;
	eip_array[word].fetch_or(1u32 << bit, Ordering::Release);
	// 置位该文件的 eip 缓存. imsic_topei_peek 以它为外部中断的快速排除条件,
	// 不置位则外部中断对该 hart 不可见. 任一 eip 位非 0 即为 1, 软件中断位也
	// 计入, 与 imsic_update_eip_ext_any 的语义一致.
	eip_ext_any.fetch_or(1, Ordering::Release);
	// 按 AIA 规范, 软件中断与外部中断都经 MEIP 与 SEIP 投递;
	// IMSIC 的 IPI 不使用 MSIP 与 SSIP 机制.
	target.mip.fetch_or(mip_bit, Ordering::AcqRel);
}

/// 清除目标 hart 的 IMSIC eip 位.
///
/// 并发约束与 ``imsic_eip_set`` 相同: 只经共享引用访问原子字段, 不跨线程写
/// ``eip_ext_any``.
#[inline]
pub(crate) fn imsic_eip_clear(
	hart_states: *const HartState,
	hart: usize,
	file_off: u64,
	eip_num: u32,
) {
	if hart_states.is_null() {
		return;
	}
	// 共享引用: 访问的 eip 与 mip 均为原子字段.
	let target = unsafe { &*(hart_states.add(hart) as *const HartState) };
	let file: &ImsicFile = if file_off == 0x0000 {
		&target.imsic_m
	} else {
		&target.imsic_s
	};
	let mip_bit: u64 = if file_off == 0x0000 { 1 << 11 } else { 1 << 9 };
	let word = (eip_num >> 5) as usize;
	if word >= file.eip.len() {
		return;
	}
	let bit = eip_num & 31;
	file.eip[word].fetch_and(!(1u32 << bit), Ordering::Release);
	// 按 AIA 规范, 全部 IMSIC 中断都经 MEIP 与 SEIP 投递. 清除之后重新查询,
	// 没有待处理中断时清除 mip 位.
	// imsic_topei_peek 只接收共享引用并读取原子 eip 字段, 可从共享引用安全调用.
	let (topei_val, _) = imsic_topei_peek(file);
	if topei_val == 0 {
		target.mip.fetch_and(!mip_bit, Ordering::AcqRel);
	}
}

// ============================================================
//  地址解码
// ============================================================

/// 解码后的 IMSIC MMIO 目标.
pub(crate) struct ImsicAddr {
	pub(crate) hart: usize,
	/// 为 ``true`` 时指向 S 文件, 为 ``false`` 时指向 M 文件.
	pub(crate) is_sfile: bool,
	/// 在 4 KiB 页内的偏移: 0x0000 为 seteipnum, 0x0008 为 clreipnum.
	pub(crate) reg_off: u64,
}

/// 把物理地址解码为 IMSIC 目标.
///
///   ============ ================================ ===================
///   区间         偏移                             每个 hart 的步长
///   ============ ================================ ===================
///   M 文件       [0,          padded*0x1000)      0x1000
///   S 文件       [padded*0x1000, 2*padded*0x1000)  0x1000
///   ============ ================================ ===================
///
/// ``padded`` 是 ``num_harts`` 向上取整到 2 的整数次幂的结果, 必须与 DTB 的
/// ``padded_count`` 和 Python 侧 IMSIC 的 ``_padded_file_count`` 一致, 否则 OpenSBI 的
/// ``imsic_data_check`` 会拒绝未对齐的 ``reg``. M 文件区间与 S 文件区间都按 ``padded``
/// 计算大小. 直接使用 ``num_harts`` 会使 S 文件基址少一个 hart: hart 数不是 2 的整数
/// 次幂时 (例如 hart 取 3 或 5), S 文件 IPI 与设备 MSI 会被投递到错误的 hart.
pub(crate) fn decode_imsic_addr(pa: u64, num_harts: u64) -> Option<ImsicAddr> {
	let base = crate::state::IMSIC_M_BASE;
	let page_stride: u64 = 0x1000; // IMSIC_MMIO_PAGE_SZ
	let m_range = num_harts.next_power_of_two() * page_stride;

	if pa < base {
		return None;
	}

	let off = pa - base;
	if off < m_range {
		// M 文件区间
		let hart = (off / page_stride) as usize;
		if hart >= num_harts as usize {
			return None;
		}
		return Some(ImsicAddr {
			hart,
			is_sfile: false,
			reg_off: off % page_stride,
		});
	}
	if off < 2 * m_range {
		// S 文件区间
		let s_off = off - m_range;
		let hart = (s_off / page_stride) as usize;
		if hart >= num_harts as usize {
			return None;
		}
		return Some(ImsicAddr {
			hart,
			is_sfile: true,
			reg_off: s_off % page_stride,
		});
	}
	None
}

/// 处理对 seteipnum 的写入.
///
/// 置位被寻址文件的 eip 位, 并唤醒目标 hart 的线程. 写入的值是中断编号, 不是中断
/// 原因, 只作用于被寻址的文件, 不存在跨文件路由. OpenSBI 以编号 1 写目标 hart 的
/// M 文件发送 IPI, 目标 hart 的 M 模式经 MTOPEI 读出后交给 sbi_ipi_process. 返回 true.
#[inline]
pub(crate) fn imsic_handle_seteipnum(
	hart_states: *const HartState,
	hart_threads: *const Thread,
	num_harts: u32,
	target_hart: usize,
	file_tag: u64,
	eip_num: u32,
) -> bool {
	imsic_eip_set(hart_states, target_hart, file_tag, eip_num);
	if !hart_threads.is_null() && target_hart < num_harts as usize {
		unsafe { &*hart_threads.add(target_hart) }.unpark();
	}
	true
}

/// 处理对 clreipnum 的写入.
///
/// 清除被寻址文件的 eip 位, 不存在跨文件路由.
#[inline]
pub(crate) fn imsic_handle_clreipnum(
	hart_states: *const HartState,
	tgt_hart: usize,
	file_tag: u64,
	eip_num: u32,
) {
	imsic_eip_clear(hart_states, tgt_hart, file_tag, eip_num);
}

// ============================================================
//  MMIO 写入与读取的处理入口
// ============================================================

/// 在加速执行引擎中直接处理一次 IMSIC MMIO 写入.
///
/// 未启用 AIA 时返回 false. 地址解码成功且页内偏移为 0x0000 时按 seteipnum 处理并
/// 返回 true, 为 0x0008 时按 clreipnum 处理并返回 true, 其余偏移返回 false, 由
/// 调用方退出加速执行并按 Python 的 MMIO 路径处理. 处理 seteipnum 时同时置位目标
/// hart 的 mip 位, 使其退出 WFI.
pub(crate) fn try_handle_imsic_concurrent(
	pa: u64,
	val: u64,
	_state: &HartState,
	clint: &ConcurrentClintCtx,
) -> bool {
	if !CFG_AIA {
		return false;
	}
	let addr = match decode_imsic_addr(pa, clint.num_harts as u64) {
		Some(a) => a,
		None => return false,
	};
	let file_tag: u64 = if addr.is_sfile { 0x1000 } else { 0x0000 };
	let eip_num = val as u32;

	if addr.reg_off == 0x0000 {
		return imsic_handle_seteipnum(
			clint.hart_states.get(),
			clint.hart_threads.get(),
			clint.num_harts,
			addr.hart,
			file_tag,
			eip_num,
		);
	}
	if addr.reg_off == 0x0008 {
		imsic_handle_clreipnum(clint.hart_states.get(), addr.hart, file_tag, eip_num);
		return true;
	}
	false
}

/// 在加速执行引擎中直接处理一次 IMSIC MMIO 读取.
///
/// 未启用 AIA 或地址不在 IMSIC 区间内时返回 None; 否则返回 0. IMSIC 的可读寄存器
/// 经 CSR 访问, MMIO 侧只提供 seteipnum 与 clreipnum 的写入, 故读取一律为 0.
#[inline]
pub(crate) fn try_handle_imsic_read_concurrent(pa: u64, num_harts: u32) -> Option<u64> {
	if !CFG_AIA {
		return None;
	}
	decode_imsic_addr(pa, num_harts as u64).map(|_| 0)
}

// ============================================================
//  mip 同步
// ============================================================

/// 按 eip 数组的当前内容重设 eip_ext_any: 遇到第一个非 0 的字时置 1 并返回,
/// 全部字为 0 时置 0.
///
/// eip 被 CSR 写入或回写 topei 之后调用.
pub(crate) fn imsic_update_eip_ext_any(file: &mut ImsicFile) {
	for i in 0..file.eip.len() {
		let eip = file.eip[i].load(Ordering::Acquire);
		if eip != 0 {
			file.eip_ext_any.store(1, Ordering::Release);
			return;
		}
	}
	file.eip_ext_any.store(0, Ordering::Release);
}

/// 把 M 文件与 S 文件的状态同步到 hart 的 mip 位.
///
/// M 文件驱动 MEIP (bit 11), S 文件驱动 SEIP (bit 9). eidelivery 非 0 时按
/// imsic_topei_peek 的结果置位或清除; eidelivery 为 0 时只在 topei 非 0 时置位,
/// 从不清除, 对应的中断线可能同时由 PLIC 驱动. 判定经 imsic_topei_peek, 使
/// eithreshold 与中断编号范围的过滤与 compute_stopi、compute_mtopi 一致.
pub(crate) fn sync_imsic(state: &mut HartState) {
	// ---- M 文件驱动 MEIP (bit 11) ----
	if state.imsic_m.present != 0 {
		let (topei_val, _) = imsic_topei_peek(&state.imsic_m);
		if state.imsic_m.eidelivery != 0 {
			// 完整 IMSIC 模式, 按 topei 驱动 MEIP.
			if topei_val != 0 {
				state.mip.fetch_or(1 << 11, Ordering::AcqRel);
			} else {
				state.mip.fetch_and(!(1 << 11), Ordering::AcqRel);
			}
		} else if topei_val != 0 {
			// eidelivery 为 0 时 imsic_topei_peek 只返回 IPI 位. 有 IPI 时
			// 置位 MEIP 使 hart 退出 WFI; 此处不清除 MEIP, PLIC 可能独立驱动
			// 该位.
			state.mip.fetch_or(1 << 11, Ordering::AcqRel);
		}
	}

	// ---- S 文件驱动 SEIP (bit 9) ----
	if state.imsic_s.present != 0 {
		let (topei_val, _) = imsic_topei_peek(&state.imsic_s);
		if state.imsic_s.eidelivery != 0 {
			if topei_val != 0 {
				state.mip.fetch_or(1 << 9, Ordering::AcqRel);
			} else {
				state.mip.fetch_and(!(1 << 9), Ordering::AcqRel);
			}
		} else if topei_val != 0 {
			state.mip.fetch_or(1 << 9, Ordering::AcqRel);
		}
	}
}

/// 把单个 IMSIC 文件同步到它的 mip 位, 供 CSR 写入或跨 hart 操作之后调用.
///
/// mfile 为 true 时更新 M 文件的 MEIP (bit 11), 为 false 时更新 S 文件的 SEIP
/// (bit 9); 文件不存在时直接返回.
#[inline]
pub(crate) fn sync_imsic_one(state: &mut HartState, mfile: bool) {
	let (file, mip_bit): (&ImsicFile, u64) = if mfile {
		(&state.imsic_m, 1 << 11)
	} else {
		(&state.imsic_s, 1 << 9)
	};
	if file.present == 0 {
		return;
	}
	let (topei_val, _) = imsic_topei_peek(file);
	if file.eidelivery != 0 {
		// 完整 IMSIC 模式, 按 topei 驱动 mip 位.
		if topei_val != 0 {
			state.mip.fetch_or(mip_bit, Ordering::AcqRel);
		} else {
			state.mip.fetch_and(!mip_bit, Ordering::AcqRel);
		}
	} else if topei_val != 0 {
		// eidelivery 为 0 时只有 IPI 位可见, 只置位不清除.
		state.mip.fetch_or(mip_bit, Ordering::AcqRel);
	}
}

// ============================================================
//  topei 与最高优先级外部中断的辅助函数
// ============================================================

/// 中断编号的上界加 1, AIA 规范的最大编号为 2047.
const _MAX_IID: u32 = 2048;
/// 2048 位 eip 与 eie 数组的 u32 字数.
const _EIP_WORDS: usize = 64;

// miselect 与 mireg 访问 IMSIC 间接寄存器的选择号.
// 按 AIA 规范 v1.0: eidelivery 为 0x70, eithreshold 为 0x72, eip 为 0x80-0xBF, eie 为 0xC0-0xFF.
const _SEL_EIDELIVERY: u32 = 0x70;
const _SEL_EITHRESHOLD: u32 = 0x72;
const _SEL_EIP_BASE: u32 = 0x80;
const _SEL_EIP_END: u32 = 0xBF;
const _SEL_EIE_BASE: u32 = 0xC0;
const _SEL_EIE_END: u32 = 0xFF;
/// 经 CSR 间接访问逐位操作 eip 与 eie: siselect 为 0x58 时向 sireg 写入中断编号
/// 置位对应的 eip 位, 0x59 清除该位, 0x5A 与 0x5B 同样操作 eie.
const _SEL_SETEIPNUM: u32 = 0x58;
const _SEL_CLREIPNUM: u32 = 0x59;
const _SEL_SETEIENUM: u32 = 0x5A;
const _SEL_CLREIENUM: u32 = 0x5B;

// ============================================================
//  IMSIC CSR 间接寄存器辅助函数, mireg 与 sireg
// ============================================================

/// 读取由 miselect 或 siselect 选中的 IMSIC 寄存器.
pub(crate) fn imsic_reg_read(file: &ImsicFile, select: u32) -> (u64, u8) {
	match select {
		_SEL_EIDELIVERY => (file.eidelivery as u64, 0),
		_SEL_EITHRESHOLD => (file.eithreshold as u64, 0),
		_SEL_EIP_BASE..=_SEL_EIP_END => {
			let idx = (select - _SEL_EIP_BASE) as usize;
			let lo = file.eip[idx].load(Ordering::Acquire) as u64;
			let hi = if idx + 1 < _EIP_WORDS {
				file.eip[idx + 1].load(Ordering::Acquire) as u64
			} else {
				0
			};
			(lo | (hi << 32), 0)
		}
		_SEL_EIE_BASE..=_SEL_EIE_END => {
			let idx = (select - _SEL_EIE_BASE) as usize;
			let lo = file.eie[idx].load(Ordering::Acquire) as u64;
			let hi = if idx + 1 < _EIP_WORDS {
				file.eie[idx + 1].load(Ordering::Acquire) as u64
			} else {
				0
			};
			(lo | (hi << 32), 0)
		}
		_ => (0, 0),
	}
}

/// 写入由 miselect 或 siselect 选中的 IMSIC 寄存器, 返回 0.
pub(crate) fn imsic_reg_write(file: &mut ImsicFile, select: u32, val: u64) -> u8 {
	match select {
		_SEL_EIDELIVERY => {
			let new = (val & 1) as u8;
			file.eidelivery = new;
			0
		}
		_SEL_EITHRESHOLD => {
			file.eithreshold = (val & 0x3FF) as u8;
			0
		}
		_SEL_EIP_BASE..=_SEL_EIP_END => {
			let idx = (select - _SEL_EIP_BASE) as usize;
			file.eip[idx].store(val as u32, Ordering::Release);
			if idx + 1 < _EIP_WORDS {
				file.eip[idx + 1].store((val >> 32) as u32, Ordering::Release);
			}
			imsic_update_eip_ext_any(file);
			0
		}
		_SEL_EIE_BASE..=_SEL_EIE_END => {
			let idx = (select - _SEL_EIE_BASE) as usize;
			file.eie[idx].store(val as u32, Ordering::Release);
			if idx + 1 < _EIP_WORDS {
				file.eie[idx + 1].store((val >> 32) as u32, Ordering::Release);
			}
			0
		}
		// 按中断编号，设置或清除对应的 eip 位.
		_SEL_SETEIPNUM => {
			let eip_num = val as u32;
			if eip_num < _MAX_IID {
				let word = (eip_num >> 5) as usize;
				let bit = eip_num & 31;
				file.eip[word].fetch_or(1u32 << bit, Ordering::Release);
				imsic_update_eip_ext_any(file);
			}
			0
		}
		_SEL_CLREIPNUM => {
			let eip_num = val as u32;
			if eip_num < _MAX_IID {
				let word = (eip_num >> 5) as usize;
				let bit = eip_num & 31;
				file.eip[word].fetch_and(!(1u32 << bit), Ordering::Release);
				imsic_update_eip_ext_any(file);
			}
			0
		}
		_SEL_SETEIENUM => {
			let eip_num = val as u32;
			if eip_num < _MAX_IID {
				let word = (eip_num >> 5) as usize;
				let bit = eip_num & 31;
				file.eie[word].fetch_or(1u32 << bit, Ordering::Release);
			}
			0
		}
		_SEL_CLREIENUM => {
			let eip_num = val as u32;
			if eip_num < _MAX_IID {
				let word = (eip_num >> 5) as usize;
				let bit = eip_num & 31;
				file.eie[word].fetch_and(!(1u32 << bit), Ordering::Release);
			}
			0
		}
		_ => 0,
	}
}

/// 读取优先级最高的待处理且使能的 IMSIC 中断, 不回写 topei 清除 eip 位.
///
/// 返回 ``(value, ok)``, value 为 ``(中断编号 << 16) | 优先级 `` 或 0, 供 mtopi 与
/// mtopei 的 CSR 读取使用. 软件中断在 eip 置位时即为待处理, 不要求 eie; eie 只
/// 使能外部中断.
pub(crate) fn imsic_topei_peek(file: &ImsicFile) -> (u64, u8) {
	// eidelivery 只使能外部中断. 先查 IPI 使内核尚未写入 eidelivery 时
	// stopi 与 mtopi 也能看到 IPI, 避免多核启动初期丢失 IPI.
	let eip0 = file.eip[0].load(Ordering::Acquire);
	let ipi_mask: u32 = (1 << IID_M_IPI) | (1 << IID_S_IPI);
	let ipi_pending = eip0 & ipi_mask;
	if ipi_pending != 0 {
		let bit = 31 - ipi_pending.leading_zeros();
		let iid = bit;
		let prio = iid & 0xFF;
		return (((iid as u64) << 16) | (prio as u64), 0);
	}

	// 外部中断要求 eidelivery 非 0.
	if file.eidelivery == 0 || file.eip_ext_any.load(Ordering::Acquire) == 0 {
		return (0, 0);
	}
	let eie0 = file.eie[0].load(Ordering::Acquire);
	// 外部中断掩码 IPI 位已在上方处理.
	let ext_mask: u32 = !((1 << IID_M_IPI) | (1 << IID_S_IPI));
	for i in (0.._EIP_WORDS).rev() {
		let eip_word = file.eip[i].load(Ordering::Acquire);
		let pending_enabled = if i == 0 {
			// 第 0 个字取 eip 与 eie 的外部位, IPI 位已在上方处理.
			eip_word & eie0 & ext_mask
		} else {
			eip_word & file.eie[i].load(Ordering::Acquire)
		};
		if pending_enabled == 0 {
			continue;
		}
		let bit = 31 - pending_enabled.leading_zeros();
		let iid = (i as u32 * 32) + bit;
		if iid == 0 || iid >= _MAX_IID {
			continue;
		}
		let prio = iid & 0xFF;
		if prio > file.eithreshold as u32 {
			return (((iid as u64) << 16) | (prio as u64), 0);
		}
	}
	(0, 0)
}

/// 清除指定中断编号对应的 eip 位, 不重新扫描最高优先级中断.
///
/// 按编号直接清除, 避免在读取 topei 与清除之间有更高优先级的中断进入时清除
/// 错误的位. 软件中断不要求 eidelivery, 与 ``imsic_topei_peek`` 的 IPI 快速路径
/// 一致; 外部中断要求 eidelivery 非 0.
///
/// 缺少软件中断这条路径时, 待处理的 IPI 会被 ``compute_stopi`` 与
/// ``compute_mtopi`` 反复报告, 形成无限分派循环, 其余中断无法处理.
pub(crate) fn imsic_topei_claim_iid(file: &mut ImsicFile, iid: u32) {
	if iid == 0 || iid >= _MAX_IID {
		return;
	}
	let word = (iid >> 5) as usize;
	let bit = iid & 31;
	let is_sw = iid == IID_M_IPI || iid == IID_S_IPI;

	// 软件中断不要求 eidelivery 即可清除, 与 imsic_topei_peek 的 IPI 快速路径
	// 一致; 外部中断要求 eidelivery 非 0.
	if !is_sw && file.eidelivery == 0 {
		return;
	}

	let is_pending = if is_sw {
		(file.eip[word].load(Ordering::Acquire) & (1 << bit)) != 0
	} else {
		(file.eip[word].load(Ordering::Acquire)
			& file.eie[word].load(Ordering::Acquire)
			& (1 << bit))
			!= 0
	};
	if word < _EIP_WORDS && is_pending {
		file.eip[word].fetch_and(!(1 << bit), Ordering::Release);
		imsic_update_eip_ext_any(file);
	}
}

/// 陷态进入时清除软件中断在 IMSIC 中的 eip 位.
///
/// 调用方在对应文件的 eidelivery 为 0 时调用本函数. eidelivery 非 0 时 IPI 经
/// MEIP 与 SEIP 投递, eip 由回写 MTOPEI 与 STOPEI 清除, 陷态进入不清除 eip.
///
/// 发生 MSIP 陷态时清除 M 文件中编号 1 与编号 3 的 eip 位, 以及 S 文件中编号 1
/// 的 eip 位; 发生 SSIP 陷态时清除 S 文件中编号 1 的 eip 位. 不清除时
/// ``compute_mtopi`` 与 ``compute_stopi`` 会读到残留的中断编号, 反复投递同一个 IPI.
///
/// 清除后调用 ``imsic_update_eip_ext_any`` 重算外部中断标记, 不直接置 0, 使其余
/// eip 字中待处理的外部中断不被丢弃.
#[inline]
pub(crate) fn imsic_clear_ipi_on_trap(state: &mut HartState, cause: u64) {
	let clear_eip = |file: &mut ImsicFile, iid: u32| {
		let w = (iid >> 5) as usize;
		let b = iid & 31;
		if w < file.eip.len() {
			file.eip[w].fetch_and(!(1u32 << b), Ordering::Release);
			imsic_update_eip_ext_any(file);
		}
	};
	match cause {
		3 => {
			if state.imsic_m.present != 0 {
				clear_eip(&mut state.imsic_m, IID_M_IPI);
				clear_eip(&mut state.imsic_m, IID_S_IPI);
			}
			if state.imsic_s.present != 0 {
				clear_eip(&mut state.imsic_s, IID_S_IPI);
			}
		}
		1 => {
			if state.imsic_s.present != 0 {
				clear_eip(&mut state.imsic_s, IID_S_IPI);
			}
		}
		_ => {}
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::interrupt::{check_pending_interrupts, compute_mtopi, compute_stopi};
	use crate::state::{riscv_mode, HartState};

	fn make_state() -> HartState {
		let mut s: HartState = unsafe { std::mem::zeroed() };
		s.imsic_m.present = 1;
		s.imsic_m.eidelivery = 1;
		s.imsic_s.present = 1;
		s.imsic_s.eidelivery = 1;
		s
	}

	#[test]
	fn sync_imsic_clears_seip_when_all_below_eithreshold() {
		// 唯一待处理且使能的 IMSIC 中断优先级不高于 eithreshold 时, SEIP 不得置位.
		// 否则 sync_imsic 与 compute_stopi 的结果不一致, 形成 SEI 与定时器中断的
		// 无限循环.
		let mut state = make_state();
		state.imsic_s.eithreshold = 100;
		// 在编号 21 处挂起中断, 优先级为 21 & 0xFF = 21
		state.imsic_s.eie[0].fetch_or(1 << 21, Ordering::Relaxed);
		state.imsic_s.eip[0].fetch_or(1 << 21, Ordering::Relaxed);
		imsic_update_eip_ext_any(&mut state.imsic_s);

		sync_imsic(&mut state);
		assert_eq!(
			state.mip.load(Ordering::Acquire) & (1 << 9),
			0,
			"SEIP must be 0 when the only pending IMSIC int is below eithreshold"
		);
	}

	#[test]
	fn sync_imsic_sets_seip_when_interrupt_above_eithreshold() {
		// 阈值低于优先级时, SEIP 须置位.
		let mut state = make_state();
		state.imsic_s.eithreshold = 0;
		state.imsic_s.eie[0].fetch_or(1 << 21, Ordering::Relaxed);
		state.imsic_s.eip[0].fetch_or(1 << 21, Ordering::Relaxed);
		imsic_update_eip_ext_any(&mut state.imsic_s);

		sync_imsic(&mut state);
		assert_eq!(
			state.mip.load(Ordering::Acquire) & (1 << 9),
			1 << 9,
			"SEIP must be set when IMSIC int priority > eithreshold"
		);
	}

	#[test]
	fn sync_imsic_respects_eithreshold_for_mfile() {
		// 对 M 文件与 MEIP 做同样的检查.
		let mut state = make_state();
		state.imsic_m.eithreshold = 200;
		// 在编号 100 处挂起, 优先级为 100 & 0xFF = 100
		let word = (100 / 32) as usize;
		let bit = 100 % 32;
		state.imsic_m.eie[word].fetch_or(1 << bit, Ordering::Relaxed);
		state.imsic_m.eip[word].fetch_or(1 << bit, Ordering::Relaxed);
		imsic_update_eip_ext_any(&mut state.imsic_m);

		sync_imsic(&mut state);
		assert_eq!(
			state.mip.load(Ordering::Acquire) & (1 << 11),
			0,
			"MEIP must be 0 when IMSIC int is below eithreshold"
		);

		// 降低阈值后, MEIP 应当置位
		state.imsic_m.eithreshold = 50;
		sync_imsic(&mut state);
		assert_eq!(
			state.mip.load(Ordering::Acquire) & (1 << 11),
			1 << 11,
			"MEIP must be set after lowering eithreshold below priority"
		);
	}

	#[test]
	fn sync_imsic_clears_seip_when_no_pending() {
		// 没有任何中断挂起时, SEIP 须为 0.
		let mut state = make_state();
		state.mip.fetch_or(1 << 9, Ordering::AcqRel); // 上一次同步留下的 SEIP, 没有置位的 eip 位

		sync_imsic(&mut state);
		assert_eq!(
			state.mip.load(Ordering::Acquire) & (1 << 9),
			0,
			"SEIP must be cleared when no IMSIC pending"
		);
	}

	#[test]
	fn sync_imsic_sfile_independent_of_mfile_eidelivery() {
		// 按 AIA 规范, 每个 IMSIC 文件独立控制自己的外部中断线. M 文件的
		// eidelivery 为 0 而 S 文件的 eidelivery 为 1 时, S 文件仍须驱动 SEIP.
		let mut state: HartState = unsafe { std::mem::zeroed() };
		state.imsic_m.present = 1;
		state.imsic_m.eidelivery = 0; // M 文件不驱动 MEIP
		state.imsic_s.present = 1;
		state.imsic_s.eidelivery = 1; // S 文件驱动 SEIP

		// 在 S 文件的编号 21 处设置待处理且使能的外部中断
		state.imsic_s.eie[0].store(1 << 21, Ordering::Relaxed);
		state.imsic_s.eip[0].store(1 << 21, Ordering::Relaxed);
		imsic_update_eip_ext_any(&mut state.imsic_s);

		// 预先设置 MEIP, 模拟主循环中清空 ext_irq .
		// M 文件不驱动 MEIP, sync_imsic 不得改动该位.
		state.mip.fetch_or(1 << 11, Ordering::AcqRel);

		sync_imsic(&mut state);

		// M 文件的 eidelivery 为 0 时, MEIP 保持不变.
		assert_eq!(
			state.mip.load(Ordering::Acquire) & (1 << 11),
			1 << 11,
			"MEIP must be preserved when M-file eidelivery=0"
		);
		// S 文件的 eidelivery 为 1 且有中断挂起时, 设置 SEIP .
		assert_eq!(
			state.mip.load(Ordering::Acquire) & (1 << 9),
			1 << 9,
			"SEIP must be set when S-file has pending interrupt even if M-file eid=0"
		);
	}

	#[test]
	fn sync_imsic_sfile_clears_seip_after_claim() {
		// 内核回写 stopei 清除 S 文件的中断后, eip 位清除, sync_imsic 须撤销 SEIP.
		let mut state: HartState = unsafe { std::mem::zeroed() };
		state.imsic_m.present = 1;
		state.imsic_m.eidelivery = 0;
		state.imsic_s.present = 1;
		state.imsic_s.eidelivery = 1;

		// 中断挂起时 SEIP 置位
		state.imsic_s.eie[0].store(1 << 21, Ordering::Relaxed);
		state.imsic_s.eip[0].store(1 << 21, Ordering::Relaxed);
		imsic_update_eip_ext_any(&mut state.imsic_s);
		sync_imsic(&mut state);
		assert_eq!(
			state.mip.load(Ordering::Acquire) & (1 << 9),
			1 << 9,
			"SEIP set when pending"
		);

		// 清除 eip 位, 对应内核回写 stopei
		imsic_topei_claim_iid(&mut state.imsic_s, 21);
		sync_imsic(&mut state);
		assert_eq!(
			state.mip.load(Ordering::Acquire) & (1 << 9),
			0,
			"SEIP must be cleared after interrupt is claimed"
		);
	}

	// ============================================================
	//  与 IPI 相关的测试
	// ============================================================

	/// 创建一个 HartState, 两个 IMSIC 文件都存在且 eidelivery 为 0.
	fn make_state_no_delivery() -> HartState {
		let mut s: HartState = unsafe { std::mem::zeroed() };
		s.imsic_m.present = 1;
		s.imsic_m.eidelivery = 0;
		s.imsic_s.present = 1;
		s.imsic_s.eidelivery = 0;
		s
	}

	// ---- imsic_topei_peek 的 IPI 快速路径 ----

	#[test]
	fn topei_peek_ipi_visible_without_eidelivery() {
		// eidelivery 与 eie 都为 0 时, 编号 1 与编号 3 的 IPI 仍须经 topei_peek
		// 可见. 该路径使多核启动初期不丢失 IPI.
		let state = make_state_no_delivery();

		// 在 S 文件的 eidelivery 与 eie 都为 0 时置位 S 模式 IPI
		state.imsic_s.eip[0].store(1 << IID_S_IPI, Ordering::Relaxed);
		let (val, _) = imsic_topei_peek(&state.imsic_s);
		assert_ne!(val, 0, "S-mode IPI must be visible with eidelivery=0");
		assert_eq!((val >> 16) as u32, IID_S_IPI, "IID must be 1 (S-mode IPI)");

		// 在 M 文件的 eidelivery 与 eie 都为 0 时置位 M 模式 IPI
		state.imsic_m.eip[0].store(1 << IID_M_IPI, Ordering::Relaxed);
		let (val, _) = imsic_topei_peek(&state.imsic_m);
		assert_ne!(val, 0, "M-mode IPI must be visible with eidelivery=0");
		assert_eq!((val >> 16) as u32, IID_M_IPI, "IID must be 3 (M-mode IPI)");
	}

	#[test]
	fn topei_peek_ipi_visible_when_both_pending() {
		// 第 0 个字中编号 1 与编号 3 同时挂起时, topei_peek 返回优先级较高者,
		// 即位索引较大的编号 3.
		let state = make_state_no_delivery();
		state.imsic_m.eip[0].store((1 << IID_S_IPI) | (1 << IID_M_IPI), Ordering::Relaxed);
		let (val, _) = imsic_topei_peek(&state.imsic_m);
		assert_eq!(
			(val >> 16) as u32,
			IID_M_IPI,
			"higher IID (3) takes priority over IID=1"
		);
	}

	#[test]
	fn topei_peek_external_blocked_without_eidelivery() {
		// eidelivery 为 0 时外部中断不得可见.
		let state = make_state_no_delivery();
		state.imsic_s.eie[0].store(1 << 10, Ordering::Relaxed);
		state.imsic_s.eip[0].store(1 << 10, Ordering::Relaxed);
		let (val, _) = imsic_topei_peek(&state.imsic_s);
		assert_eq!(val, 0, "external IID=10 blocked with eidelivery=0");
	}

	#[test]
	fn topei_peek_ipi_dominates_external() {
		// 第 0 个字中 IPI 与外部中断同时挂起时, IPI 快速路径先于外部中断扫描执行,
		// 返回 IPI; 外部中断本身也被 eidelivery 为 0 屏蔽.
		let state = make_state_no_delivery();
		state.imsic_s.eie[0].store(1 << 10, Ordering::Relaxed);
		state.imsic_s.eip[0].store((1 << IID_S_IPI) | (1 << 10), Ordering::Relaxed);
		let (val, _) = imsic_topei_peek(&state.imsic_s);
		assert_eq!(
			(val >> 16) as u32,
			IID_S_IPI,
			"IPI fast-path must return IID=1 even when external IID=10 also pending"
		);
	}

	#[test]
	fn topei_peek_ipi_passes_eithreshold() {
		// IPI 快速路径不检查 eithreshold, 无条件返回 IPI.
		let mut state = make_state_no_delivery();
		state.imsic_s.eithreshold = 200; // 高于 IPI 的优先级 1 & 0xFF = 1
		state.imsic_s.eip[0].store(1 << IID_S_IPI, Ordering::Relaxed);
		let (val, _) = imsic_topei_peek(&state.imsic_s);
		assert_ne!(val, 0, "IPI bypasses eithreshold check");
	}

	// ---- imsic_topei_claim_iid 清除 IPI ----

	#[test]
	fn ipi_claimable_without_eidelivery() {
		// eidelivery 为 0 时, imsic_topei_claim_iid 仍须清除 IPI 的 eip 位.
		// 否则 compute_stopi 与 compute_mtopi 会持续报告同一个 IPI.
		let mut state = make_state_no_delivery();
		state.imsic_s.eip[0].store(1 << IID_S_IPI, Ordering::Relaxed);

		// 清除前确认 IPI 处于待处理状态
		let (before, _) = imsic_topei_peek(&state.imsic_s);
		assert_ne!(before, 0, "IPI pending before claim");

		imsic_topei_claim_iid(&mut state.imsic_s, IID_S_IPI);

		let (after, _) = imsic_topei_peek(&state.imsic_s);
		assert_eq!(after, 0, "IPI cleared after claim even with eidelivery=0");
		assert_eq!(
			state.imsic_s.eip[0].load(Ordering::Acquire) & (1 << IID_S_IPI),
			0,
			"eip bit must be cleared"
		);
	}

	#[test]
	fn ipi_mfile_claimable_without_eidelivery() {
		// 对 M 文件的编号 3 做同样的检查.
		let mut state = make_state_no_delivery();
		state.imsic_m.eip[0].store(1 << IID_M_IPI, Ordering::Relaxed);

		imsic_topei_claim_iid(&mut state.imsic_m, IID_M_IPI);

		assert_eq!(
			state.imsic_m.eip[0].load(Ordering::Acquire) & (1 << IID_M_IPI),
			0,
			"M-mode IPI eip cleared after claim with eidelivery=0"
		);
	}

	#[test]
	fn external_not_claimable_without_eidelivery() {
		// eidelivery 为 0 时外部中断不得被清除; eie 位未置位时同样不满足清除
		// 条件. 检查 eip 位保持不变.
		let mut state = make_state_no_delivery();
		state.imsic_s.eie[0].store(1 << 10, Ordering::Relaxed);
		state.imsic_s.eip[0].store(1 << 10, Ordering::Relaxed);

		imsic_topei_claim_iid(&mut state.imsic_s, 10);

		assert_eq!(
			state.imsic_s.eip[0].load(Ordering::Acquire) & (1 << 10),
			1 << 10,
			"external eip unchanged — claim blocked by eidelivery=0"
		);
	}

	// ---- sync_imsic 把 IPI 同步到 mip 位 ----

	#[test]
	fn sync_imsic_sets_seip_for_ipi_without_eidelivery() {
		// eidelivery 为 0 时, 编号 1 的 IPI 挂起仍须由 sync_imsic 置位 SEIP.
		// 该路径负责多核启动初期的 IPI 唤醒.
		let mut state = make_state_no_delivery();
		state.imsic_s.eip[0].store(1 << IID_S_IPI, Ordering::Relaxed);

		sync_imsic(&mut state);
		assert_eq!(
			state.mip.load(Ordering::Acquire) & (1 << 9),
			1 << 9,
			"SEIP must be set for pending S-mode IPI even with eidelivery=0"
		);
	}

	#[test]
	fn sync_imsic_sets_meip_for_ipi_without_eidelivery() {
		// 对 M 文件的编号 3 IPI 与 MEIP 做同样的检查.
		let mut state = make_state_no_delivery();
		state.imsic_m.eip[0].store(1 << IID_M_IPI, Ordering::Relaxed);

		sync_imsic(&mut state);
		assert_eq!(
			state.mip.load(Ordering::Acquire) & (1 << 11),
			1 << 11,
			"MEIP must be set for pending M-mode IPI even with eidelivery=0"
		);
	}

	#[test]
	fn sync_imsic_does_not_clear_seip_without_eidelivery() {
		// eidelivery 为 0 时 sync_imsic 不得清除 SEIP, 因为 PLIC 可能独立驱动该位.
		let mut state = make_state_no_delivery();
		// 预先置位 SEIP, 模拟 PLIC 中断正在处理
		state.mip.store(1 << 9, Ordering::Relaxed);
		// IMSIC 无待处理中断

		sync_imsic(&mut state);
		assert_eq!(
			state.mip.load(Ordering::Acquire) & (1 << 9),
			1 << 9,
			"SEIP preserved — sync_imsic with eidelivery=0 does not clear"
		);
	}

	#[test]
	fn sync_imsic_clears_seip_after_ipi_claim() {
		// 设置 IPI 后 sync_imsic 设置 SEIP; 清除 IPI 后 sync_imsic 撤销 SEIP.
		// 该用例覆盖 eidelivery 非 0 时 IPI 的完整过程.
		let mut state = make_state_no_delivery();
		state.imsic_s.eidelivery = 1; // 允许清除

		// 置位 IPI
		state.imsic_s.eip[0].store(1 << IID_S_IPI, Ordering::Relaxed);
		sync_imsic(&mut state);
		assert_eq!(
			state.mip.load(Ordering::Acquire) & (1 << 9),
			1 << 9,
			"SEIP set after IPI injected"
		);

		// 清除 IPI
		imsic_topei_claim_iid(&mut state.imsic_s, IID_S_IPI);
		sync_imsic(&mut state);
		assert_eq!(
			state.mip.load(Ordering::Acquire) & (1 << 9),
			0,
			"SEIP cleared after IPI claimed"
		);
	}

	// ---- compute_mtopi 与 compute_stopi 报告 IPI ----

	#[test]
	fn compute_stopi_reports_ipi_without_eidelivery() {
		// compute_stopi 经 topei_peek 的 IPI 快速路径, 把 IPI 的中断编号 1 映射为
		// 中断原因 SEI 的编号 9. IMSIC 经 SEIP 投递 IPI, 因此 stopi 须报告 9,
		// 不得报告 SSI 的编号 1, 即使 eidelivery 为 0. IPI 的中断编号只经 STOPEI
		// 读出.
		let mut state = make_state_no_delivery();
		state.imsic_s.eip[0].store(1 << IID_S_IPI, Ordering::Relaxed);
		let (val, _) = compute_stopi(&mut state, 0);
		assert_ne!(val, 0, "stopi must report IPI even with eidelivery=0");
		assert_eq!(
			(val >> 16) as u32,
			9,
			"stopi reports IID=9 (SEI) for S-file IPI"
		);

		// 读取 stopi 不清除 eip 位, 该位留给 STOPEI 清除.
		assert_eq!(
			state.imsic_s.eip[0].load(Ordering::Acquire) & (1 << IID_S_IPI),
			1 << IID_S_IPI,
			"stopi read must not claim the IPI eip bit"
		);
	}

	#[test]
	fn compute_mtopi_reports_ipi_without_eidelivery() {
		// M 模式同理: compute_mtopi 把 M 文件的 IPI 编号 3 映射为中断原因 MEI 的
		// 编号 11. IMSIC 经 MEIP 投递 IPI, 因此 mtopi 须报告 11, 不得报告 MSI 的
		// 编号 3.
		let mut state = make_state_no_delivery();
		state.imsic_m.eip[0].store(1 << IID_M_IPI, Ordering::Relaxed);

		let (val, _) = compute_mtopi(&mut state, 0);
		assert_ne!(val, 0, "mtopi must report IPI even with eidelivery=0");
		assert_eq!(
			(val >> 16) as u32,
			11,
			"mtopi reports IID=11 (MEI) for M-file IPI"
		);

		// 读取 mtopi 不清除 eip 位, 该位留给 MTOPEI 清除.
		assert_eq!(
			state.imsic_m.eip[0].load(Ordering::Acquire) & (1 << IID_M_IPI),
			1 << IID_M_IPI,
			"mtopi read must not claim the IPI eip bit"
		);
	}

	#[test]
	fn stopi_sfile_ipi_reports_sei_major_identity() {
		// S 文件的 IPI 编号 1 须由 stopi 报告为中断原因 SEI 的编号 9, 不得报告
		// SSI 的编号 1. 例如，Linux 的 riscv_intc_aia_irq() 以 `stopi >> 16` 分派到
		// intc 域; IMSIC 模式下编号 1 没有对应处理函数, IPI 被丢弃, 发送方 hart
		// 在 smp_call_function_many_cond 中等待目标 hart 而持续自旋.
		let mut state = make_state_no_delivery();
		state.imsic_s.eip[0].store(1 << IID_S_IPI, Ordering::Relaxed);
		let (val, _) = compute_stopi(&mut state, 0);
		assert_eq!(
			(val >> 16) as u32,
			9,
			"S-file IPI must be reported as SEI (9), not SSI (1)"
		);
	}

	// ---- imsic_handle_seteipnum 的 IPI 寻址 ----

	#[test]
	fn seteipnum_mfile_iid1_stays_in_mfile() {
		// 对 M 文件写入 ``seteipnum`` 且中断编号为 1 时, 须置位 M 文件中的
		// eip 位并驱动 MEIP, 不得转发到 S 文件. OpenSBI 把 ``IMSIC_IPI_ID`` 即
		// 1 写入目标 hart 的 M 文件来发送 IPI; 转发到 S 文件会使 M 模式的 IPI
		// 丢失, 多核启动时其余 hart 无法唤醒.
		let state = make_state_no_delivery();

		// 模拟跨 hart 写入 seteipnum: 目标 hart 为 0, 操作 M 文件, 编号为 1
		let states = &state as *const HartState;
		let result = imsic_handle_seteipnum(states, std::ptr::null(), 1, 0, 0x0000, IID_S_IPI);
		assert!(result, "seteipnum must return true (handled)");

		// M 文件须记录该 IPI.
		assert_eq!(
			state.imsic_m.eip[0].load(Ordering::Acquire) & (1 << IID_S_IPI),
			1 << IID_S_IPI,
			"M-file eip must have IID=1 (no cross-file routing)"
		);
		// M 文件的 eip 写入须驱动 MEIP.
		assert_eq!(
			state.mip.load(Ordering::Acquire) & (1 << 11),
			1 << 11,
			"MEIP must be set when M-file eip[1] is set"
		);
		// S 文件不得收到该 IPI.
		assert_eq!(
			state.imsic_s.eip[0].load(Ordering::Acquire) & (1 << IID_S_IPI),
			0,
			"S-file eip must NOT have IID=1 (addressed file only)"
		);
	}

	#[test]
	fn seteipnum_mfile_iid3_stays_in_mfile() {
		// 对 M 文件写入编号 3 时, 该 IPI 留在 M 文件.
		let state = make_state_no_delivery();
		let states = &state as *const HartState;
		let result = imsic_handle_seteipnum(states, std::ptr::null(), 1, 0, 0x0000, IID_M_IPI);
		assert!(result);

		assert_eq!(
			state.imsic_m.eip[0].load(Ordering::Acquire) & (1 << IID_M_IPI),
			1 << IID_M_IPI,
			"M-file must have IID=3 (no routing)"
		);
		assert_eq!(
			state.imsic_s.eip[0].load(Ordering::Acquire) & (1 << IID_M_IPI),
			0,
			"S-file must NOT have IID=3"
		);
	}

	#[test]
	fn seteipnum_sfile_iid1_stays_in_sfile() {
		// 对 S 文件写入编号 1 时, 该 IPI 留在 S 文件.
		let state = make_state_no_delivery();
		let states = &state as *const HartState;
		let result = imsic_handle_seteipnum(states, std::ptr::null(), 1, 0, 0x1000, IID_S_IPI);
		assert!(result);

		assert_eq!(
			state.imsic_s.eip[0].load(Ordering::Acquire) & (1 << IID_S_IPI),
			1 << IID_S_IPI,
			"S-file must have IID=1"
		);
		// M 文件不应收到该 IPI.
	}

	// ---- imsic_eip_set 的跨 hart 原子写入 ----

	#[test]
	fn eip_set_sfile_sets_eip_and_seip() {
		// 对 S 文件调用 imsic_eip_set 须置位 eip 位与 SEIP.
		let state = make_state_no_delivery();
		let states = &state as *const HartState;
		imsic_eip_set(states, 0, 0x1000, IID_S_IPI);

		assert_eq!(
			state.imsic_s.eip[0].load(Ordering::Acquire) & (1 << IID_S_IPI),
			1 << IID_S_IPI,
			"eip bit set"
		);
		assert_eq!(
			state.mip.load(Ordering::Acquire) & (1 << 9),
			1 << 9,
			"SEIP set by imsic_eip_set"
		);
	}

	#[test]
	fn eip_set_mfile_sets_eip_and_meip() {
		let state = make_state_no_delivery();
		let states = &state as *const HartState;
		imsic_eip_set(states, 0, 0x0000, IID_M_IPI);

		assert_eq!(
			state.imsic_m.eip[0].load(Ordering::Acquire) & (1 << IID_M_IPI),
			1 << IID_M_IPI,
			"eip bit set"
		);
		assert_eq!(
			state.mip.load(Ordering::Acquire) & (1 << 11),
			1 << 11,
			"MEIP set by imsic_eip_set"
		);
	}

	#[test]
	fn eip_set_null_pointer_is_noop() {
		// 指针为空时 imsic_eip_set 直接返回.
		imsic_eip_set(std::ptr::null(), 0, 0x0000, IID_M_IPI);
		// 未发生异常即通过.
	}

	/// 跨 hart 置位外部中断编号时, 必须同时置位 eip_ext_any.
	///
	/// 设备 MSI 以 MMIO 写 ``seteipnum`` 进入目标 hart 的中断文件, 依次经
	/// ``imsic_handle_seteipnum`` 与 ``imsic_eip_set``. ``imsic_topei_peek`` 以
	/// ``eip_ext_any`` 作为外部中断的快速排除条件, 该标记未置位时即使 eip 与 eie
	/// 都已就绪也返回 0, 于是出现 MEIP 或 SEIP 已置起而 ``mtopi`` 与 ``stopei`` 读出 0 的状态.
	#[test]
	fn eip_set_external_iid_marks_eip_ext_any() {
		// 编号 9 属于外部中断, 且不是 IPI 使用的编号 1 与 3.
		const IID_EXT: u32 = 9;
		let state = make_state();
		state.imsic_s.eie[0].store(1 << IID_EXT, Ordering::Relaxed);
		let states = &state as *const HartState;

		imsic_eip_set(states, 0, 0x1000, IID_EXT);

		assert_eq!(
			state.imsic_s.eip_ext_any.load(Ordering::Acquire),
			1,
			"注入外部 IID 后 eip_ext_any 必须为 1"
		);
		let (topei_val, _) = imsic_topei_peek(&state.imsic_s);
		assert_ne!(
			topei_val, 0,
			"注入外部 IID 后 imsic_topei_peek 必须读出 identity"
		);
		assert_eq!(
			(topei_val >> 16) & 0x7FF,
			IID_EXT as u64,
			"读出的 minor identity 应为注入的 IID"
		);
	}

	#[test]
	fn eip_clear_sfile_clears_eip_and_seip() {
		let state = make_state_no_delivery();
		state.imsic_s.eip[0].store(1 << IID_S_IPI, Ordering::Relaxed);
		state.mip.store(1 << 9, Ordering::Relaxed);

		let states = &state as *const HartState;
		imsic_eip_clear(states, 0, 0x1000, IID_S_IPI);

		assert_eq!(
			state.imsic_s.eip[0].load(Ordering::Acquire) & (1 << IID_S_IPI),
			0,
			"eip bit cleared"
		);
		assert_eq!(
			state.mip.load(Ordering::Acquire) & (1 << 9),
			0,
			"SEIP cleared by imsic_eip_clear"
		);
	}

	// ---- imsic_topei_peek: eip_ext_any 不影响 IPI ----

	#[test]
	fn topei_peek_ipi_visible_even_when_eip_ext_any_zero() {
		// eip_ext_any 是第 1 个字及之后外部中断的快速排除条件, 第 0 个字的 IPI
		// 不经过该判断. 检查 eip_ext_any 为 0 时 IPI 仍可见.
		let mut state = make_state_no_delivery();
		state.imsic_m.eidelivery = 1;
		// eip_ext_any 初值为 0, 各外部中断字均无待处理位
		state.imsic_m.eip[0].store(1 << IID_M_IPI, Ordering::Relaxed);

		let (val, _) = imsic_topei_peek(&state.imsic_m);
		assert_ne!(val, 0, "IPI visible even when eip_ext_any==0");
	}

	// ---- IPI 与 CLINT 并存 ----

	#[test]
	fn clint_msip_still_visible_alongside_imsic_ipi() {
		// IMSIC 的 eidelivery 为 0 且 CLINT 也置位 MSIP 时,
		// check_pending_interrupts 须同时看到两者. IMSIC 的 IPI 驱动 MEIP,
		// CLINT 驱动 MSIP.
		let mut state = make_state_no_delivery();
		// IMSIC M 文件的 IPI
		state.imsic_m.eip[0].store(1 << IID_M_IPI, Ordering::Relaxed);
		// CLINT 的 MSIP
		state.mip.fetch_or(1 << 3, Ordering::AcqRel);
		// 在 mie 中使能两者
		state.mie = (1 << 11) | (1 << 3); // MEIE 与 MSIE

		sync_imsic(&mut state);

		// MEIP 与 MSIP 都已置位
		let mip = state.mip.load(Ordering::Acquire);
		assert_eq!(mip & (1 << 11), 1 << 11, "MEIP from IMSIC IPI");
		assert_eq!(mip & (1 << 3), 1 << 3, "MSIP from CLINT");

		// check_pending_interrupts 优先选择 MEI
		let result = check_pending_interrupts(&state);
		assert!(result.is_some(), "interrupt must be pending");
		assert_eq!(
			result.unwrap().0,
			11,
			"MEI (cause 11) takes priority over MSI"
		);
	}

	// ============================================================
	//  多 hart 集成: 跨 hart 的 IPI 经 compute_stopi 与 compute_mtopi 报告
	// ============================================================

	fn make_hart1_state() -> HartState {
		let mut s: HartState = unsafe { std::mem::zeroed() };
		s.imsic_m.present = 1;
		s.imsic_m.eidelivery = 1;
		s.imsic_s.present = 1;
		s.imsic_s.eidelivery = 1;
		s.mhartid = 1;
		s.mie = (1 << 9) | (1 << 11); // SEIE 与 MEIE
								// mideleg: 把 SEI, SSI 与 STI 委派到 S 模式, AIA 模式下 OpenSBI 的默认设置.
		s.mideleg = (1 << 9) | (1 << 1) | (1 << 5);
		s
	}

	fn make_hart0_state() -> HartState {
		let mut s: HartState = unsafe { std::mem::zeroed() };
		s.imsic_m.present = 1;
		s.imsic_m.eidelivery = 1;
		s.imsic_s.present = 1;
		s.imsic_s.eidelivery = 1;
		s.mhartid = 0;
		s.mie = (1 << 9) | (1 << 11) | (1 << 3); // SEIE, MEIE 与 MSIE
		s.mideleg = (1 << 9) | (1 << 1) | (1 << 5);
		s
	}

	#[test]
	fn multi_hart_smode_ipi_sfile_to_stopi() {
		// 模拟 hart 0 经 IMSIC S 文件向 hart 1 发送 S 模式 IPI.
		// hart 1 的 compute_stopi 须报告编号 9, 即 SEI.
		let _h0 = make_hart0_state();
		let mut h1 = make_hart1_state();

		// 直接跨 hart 写入: imsic_eip_set 以 *const HartState 作为数组基址,
		// hart 索引 0 指向 h1.
		let h1_ptr = &h1 as *const HartState;
		imsic_eip_set(h1_ptr, 0, 0x1000, IID_S_IPI); // S 文件, hart 索引 0

		sync_imsic(&mut h1);

		// S 文件有编号 1 的 IPI 挂起, SEIP 须置位.
		assert_eq!(
			h1.mip.load(Ordering::Acquire) & (1 << 9),
			1 << 9,
			"SEIP must be set after cross-hart IPI"
		);
		assert_eq!(
			h1.imsic_s.eip[0].load(Ordering::Acquire) & (1 << IID_S_IPI),
			1 << IID_S_IPI,
			"S-file eip must have IID=1"
		);

		// hart 1 内核读取 stopi, 由 compute_stopi 处理
		let (stopi_val, _) = compute_stopi(&mut h1, 0);
		assert_ne!(stopi_val, 0, "stopi must report pending interrupt");
		assert_eq!(
			(stopi_val >> 16) as u32,
			9,
			"stopi must report IID=9 (SEI) for the S-file IPI from hart 0"
		);
	}

	#[test]
	fn multi_hart_mmode_ipi_mfile_to_mtopi() {
		// 模拟 hart 0 经 IMSIC M 文件向 hart 1 发送 M 模式 IPI.
		// hart 1 的 compute_mtopi 须报告编号 11, 即 MEI.
		let _h0 = make_hart0_state();
		let mut h1 = make_hart1_state();

		let h1_ptr = &h1 as *const HartState;
		imsic_eip_set(h1_ptr, 0, 0x0000, IID_M_IPI); // M 文件, hart 索引 0

		sync_imsic(&mut h1);

		// 须设置 MEIP.
		assert_eq!(
			h1.mip.load(Ordering::Acquire) & (1 << 11),
			1 << 11,
			"MEIP must be set after cross-hart M-mode IPI"
		);

		// hart 1 的 OpenSBI 读取 mtopi, 由 compute_mtopi 处理
		let (mtopi_val, _) = compute_mtopi(&mut h1, 0);
		assert_ne!(mtopi_val, 0, "mtopi must report pending interrupt");
		assert_eq!(
			(mtopi_val >> 16) as u32,
			11,
			"mtopi must report IID=11 (MEI) for the M-file IPI from hart 0"
		);
	}

	#[test]
	fn multi_hart_stopi_ipi_priority_over_stip() {
		// IMSIC S 文件的 IPI 与 STIP 同时挂起时, compute_stopi 须先返回 IPI,
		// 检查顺序为 IMSIC S 文件、SSIP、STIP.
		let mut h1 = make_hart1_state();

		// 设置 STIP: stimecmp 已过期, mie.STIE 已使能.
		h1.stimecmp = 50;
		h1.mie |= 1 << 5; // STIE
					// 直接写 eip 位产生 IPI, 模拟 hart 0 的 seteipnum.
		h1.imsic_s.eip[0].store(1 << IID_S_IPI, Ordering::Relaxed);
		// 由 IMSIC 同步 SEIP.
		sync_imsic(&mut h1);

		// 手工置位 STIP; 指令边界处由 sync_mtip 完成该操作.
		// stimecmp 为 50, mtime 为 100, 满足 mtime > stimecmp, STIP 挂起.
		h1.mip.fetch_or(1 << 5, Ordering::AcqRel);

		// 检查 compute_stopi 的优先级顺序.
		let (val, _) = compute_stopi(&mut h1, 100); // mtime 为 100
		assert_ne!(val, 0, "stopi must return something");
		// IMSIC S 文件先被检查, 即使 STIP 挂起也须返回编号 9, 即 SEI.
		assert_eq!(
			(val >> 16) as u32,
			9,
			"stopi must report IID=9 (SEI) before STIP (IMSIC S-file has priority)"
		);

		// compute_stopi 不清除 STIP, 该位在 mip 中仍置位.
		assert_eq!(
			h1.mip.load(Ordering::Acquire) & (1 << 5),
			1 << 5,
			"STIP still pending after first stopi (not checked yet)"
		);

		// 内核回写 stopi 清除该 IPI.
		imsic_topei_claim_iid(&mut h1.imsic_s, IID_S_IPI);
		sync_imsic_one(&mut h1, false); // mfile 为 false, 处理 S 文件与 SEIP
								  // 与 csr_write(stopi) 路径一致, 清除后撤销 SEIP.
		h1.mip.fetch_and(!(1 << 9), Ordering::AcqRel);

		// 清除并同步后, IMSIC S 文件无待处理位, SEIP 撤销.
		assert_eq!(
			h1.imsic_s.eip[0].load(Ordering::Acquire) & (1 << IID_S_IPI),
			0,
			"S-file eip cleared after claim"
		);
		assert_eq!(
			h1.mip.load(Ordering::Acquire) & (1 << 9),
			0,
			"SEIP cleared after claim + sync"
		);

		// 再次读取 stopi, 此时应报告 STIP 的编号 5.
		let (val2, _) = compute_stopi(&mut h1, 100);
		assert_ne!(val2, 0, "second stopi must return STIP");
		assert_eq!(
			(val2 >> 16) as u32,
			5,
			"second stopi must report IID=5 (STIP) after IPI is claimed"
		);
	}

	#[test]
	fn multi_hart_stopi_stip_evaluation_agrees_with_sync_mtip() {
		// stimecmp 大于 0 且 mtime 不小于 stimecmp 时, sync_mtip 置位 STIP,
		// 按 RISC-V SSTC 规范使用不小于比较. compute_stopi 须在同一条件下
		// 报告编号 5.
		let mut h1 = make_hart1_state();
		h1.stimecmp = 50;
		h1.mie |= 1 << 5; // STIE
					// IMSIC 无待处理中断, SSIP 为 0.

		// mtime 为 50, 满足 mtime >= stimecmp.
		// sync_mtip 置位 STIP, compute_stopi 须报告编号 5.
		let (val, _) = compute_stopi(&mut h1, 50);
		assert_eq!(
			(val >> 16) as u32,
			5,
			"stopi reports IID=5 when mtime == stimecmp (>= comparison)"
		);

		// mtime 为 49, 不满足 mtime >= stimecmp, STIP 不置位, stopi 返回 0.
		let (val2, _) = compute_stopi(&mut h1, 49);
		assert_eq!(val2, 0, "stopi returns 0 when mtime < stimecmp");

		// mtime 为 51, 满足 mtime >= stimecmp, 报告 STIP.
		let (val3, _) = compute_stopi(&mut h1, 51);
		assert_eq!(
			(val3 >> 16) as u32,
			5,
			"stopi reports IID=5 when mtime > stimecmp"
		);
	}

	#[test]
	fn multi_hart_stopi_ipi_in_sfile_not_visible_in_mtopi() {
		// hart 1 的 S 文件中编号 1 的 IPI 经 compute_stopi 可见, 经 compute_mtopi
		// 不可见. 两个文件互相独立, 不存在跨文件转发.
		let _h0 = make_hart0_state();
		let mut h1 = make_hart1_state();

		// 直接置位 IMSIC S 文件的 eip 位, 只作用于被寻址的文件.
		h1.imsic_s.eip[0].store(1 << IID_S_IPI, Ordering::Relaxed);
		h1.mip.fetch_or(1 << 9, Ordering::AcqRel); // SEIP
		sync_imsic(&mut h1);

		// hart 1 读取 stopi, 看到 S 文件中的 IPI.
		let (val, _) = compute_stopi(&mut h1, 0);
		assert_eq!(
			(val >> 16) as u32,
			9,
			"stopi sees IID=9 (SEI) pending in the S-file"
		);

		// compute_mtopi 只看 M 文件, 其中无待处理位, 不存在跨文件泄漏.
		let (mtopi_val, _) = compute_mtopi(&mut h1, 0);
		assert_eq!(
			mtopi_val, 0,
			"mtopi returns 0: S-file IPI is not visible in the M-file"
		);
	}

	#[test]
	fn multi_hart_stopi_ipi_claim_then_recheck() {
		// 完整过程: IPI 到达, stopi 报告编号 9, 内核回写 stopi 清除该 IPI,
		// 再次读取 stopi 返回 0.
		let _h0 = make_hart0_state();
		let mut h1 = make_hart1_state();

		// 模拟跨 hart 的 IPI: 置位 S 文件中编号 1 的 eip 位与 SEIP.
		h1.imsic_s.eip[0].store(1 << IID_S_IPI, Ordering::Relaxed);
		sync_imsic(&mut h1);

		// 第一步: 读取 stopi, 得到编号 9, 即 SEI
		let (val1, _) = compute_stopi(&mut h1, 0);
		assert_eq!((val1 >> 16) as u32, 9, "first stopi -> IID=9 (SEI)");

		// 第二步: 内核回写 stopi, 清除 S 文件中编号 1 的中断.
		imsic_topei_claim_iid(&mut h1.imsic_s, IID_S_IPI);
		sync_imsic_one(&mut h1, false); // S 文件与 SEIP
		h1.mip.fetch_and(!(1 << 9), Ordering::AcqRel);

		// 第三步: eip 位与 SEIP 已清除.
		assert_eq!(
			h1.mip.load(Ordering::Acquire) & (1 << 9),
			0,
			"SEIP must be 0 after claim + sync"
		);
		// 第四步: 再次读取 stopi 返回 0, 无待处理中断.
		let (val2, _) = compute_stopi(&mut h1, 0);
		assert_eq!(val2, 0, "stopi returns 0 after IPI is claimed");
	}

	#[test]
	fn multi_hart_stopi_ipi_two_harts_independent() {
		// 发往 hart 1 的 IPI 不影响 hart 2 的 stopi.
		let mut h1 = make_hart1_state();
		let mut h2 = make_hart1_state(); // 第二个目标, 拓扑相同

		// 只向 hart 1 发送 IPI: 置位 S 文件的 eip 位与 SEIP.
		h1.imsic_s.eip[0].store(1 << IID_S_IPI, Ordering::Relaxed);
		h1.mip.fetch_or(1 << 9, Ordering::AcqRel);
		sync_imsic(&mut h1);

		// hart 1 的 stopi 可见该 IPI.
		let (val_h1, _) = compute_stopi(&mut h1, 0);
		assert_eq!((val_h1 >> 16) as u32, 9, "hart 1 stopi must see IPI as SEI");

		// hart 2 无 IPI, stopi 须返回 0.
		let (val_h2, _) = compute_stopi(&mut h2, 0);
		assert_eq!(val_h2, 0, "hart 2 stopi must return 0 (no IPI)");
	}

	#[test]
	fn multi_hart_mtopi_ipi_priority_over_mtip() {
		// IMSIC M 文件的 IPI 与 MTIP 同时挂起时, compute_mtopi 须先返回 IPI,
		// 检查顺序为 IMSIC M 文件、MSIP、MTIP.
		let mut h1 = make_hart1_state();

		// 设置 MTIP: 手工置位 mip 的 MTIP 并使能 MTIE.
		h1.mip.fetch_or(1 << 7, Ordering::AcqRel);
		h1.mie |= 1 << 7; // MTIE

		// 同时置位 IMSIC M 文件的 IPI.
		h1.imsic_m.eip[0].store(1 << IID_M_IPI, Ordering::Relaxed);
		sync_imsic(&mut h1);

		// 检查 compute_mtopi 的优先级顺序.
		let (val, _) = compute_mtopi(&mut h1, 0);
		assert_ne!(val, 0, "mtopi must return something");
		assert_eq!(
			(val >> 16) as u32,
			11,
			"mtopi must report IID=11 (MEI, IMSIC M-file) before MTIP"
		);

		// 清除该 IPI 后, compute_mtopi 应报告 MTIP.
		imsic_topei_claim_iid(&mut h1.imsic_m, IID_M_IPI);
		sync_imsic_one(&mut h1, true); // M 文件与 MEIP
		h1.mip.fetch_and(!(1 << 11), Ordering::AcqRel); // 清除 MEIP

		let (val2, _) = compute_mtopi(&mut h1, 0);
		assert_ne!(val2, 0, "mtopi must report MTIP after IPI claimed");
		assert_eq!(
			(val2 >> 16) as u32,
			7,
			"mtopi must report IID=7 (MTIP) after IMSIC IPI claimed"
		);
	}

	#[test]
	fn multi_hart_mtopi_spurious_mei_cleanup() {
		// ext_irq 置位 MEIP 而 IMSIC M 文件无待处理中断时, step_interrupts 的
		// 清理须清除 MEIP.
		let h1 = make_hart1_state();

		// 模拟 ext_irq 排空置位 MEIP, 主循环在每轮加速执行开始时完成该操作.
		h1.mip.fetch_or(1 << 11, Ordering::AcqRel);
		// IMSIC M 文件无置位的 eip 位, topei 应返回 0.
		let (topei_val, _) = imsic_topei_peek(&h1.imsic_m);
		assert_eq!(topei_val, 0, "topei returns 0 when no IMSIC pending");

		// 执行清理, 与 step_interrupts 的逻辑相同.
		if (h1.mip.load(Ordering::Acquire) & (1 << 11)) != 0 && h1.imsic_m.present != 0 {
			if h1.imsic_m.eidelivery != 0 {
				let (val, _) = imsic_topei_peek(&h1.imsic_m);
				if val == 0 {
					h1.mip.fetch_and(!(1 << 11), Ordering::AcqRel);
				}
			} else {
				let ipi_mask: u32 = (1 << IID_S_IPI) | (1 << IID_M_IPI);
				let has_ipi = (h1.imsic_m.eip[0].load(Ordering::Acquire) & ipi_mask) != 0;
				if !has_ipi {
					h1.mip.fetch_and(!(1 << 11), Ordering::AcqRel);
				}
			}
		}

		assert_eq!(
			h1.mip.load(Ordering::Acquire) & (1 << 11),
			0,
			"MEIP cleared by cleanup when no IMSIC M-file interrupt pending"
		);

		// 清理后 check_pending_interrupts 不得投递 MEI.
		let mip = h1.mip.load(Ordering::Acquire);
		assert_eq!(mip & (1 << 11), 0, "MEIP cleared -> no MEI");
	}

	/// S 文件有 IPI 挂起且 SEI 未委派时, SEI 陷态进入 M 模式. OpenSBI 的 M 模式
	/// 处理程序读取地址 0xFB0 处的 mtopi, 由 compute_mtopi 处理. compute_mtopi
	/// 只检查 IMSIC M 文件, 看不到 S 文件中的 IPI, 因此 mtopi 返回 0. OpenSBI
	/// 随后在不撤销 SEIP 的情况下执行 MRET, 下一条指令再次触发 M 模式 SEI 陷态.
	///
	/// 本用例验证 M 文件为空而 S 文件有 IPI 挂起时 mtopi 返回 0.
	#[test]
	fn multi_hart_mtopi_zero_when_sfile_ipi_not_delegated() {
		let mut h1 = make_hart1_state();
		// SEI 未委派时, SEI 陷态进入 M 模式
		h1.mideleg &= !(1 << 9);
		// S 文件有 IPI 挂起
		h1.imsic_s.eip[0].store(1 << IID_S_IPI, Ordering::Relaxed);
		sync_imsic(&mut h1);

		// IMSIC S 文件有 IPI, SEIP 须置位
		assert_eq!(
			h1.mip.load(Ordering::Acquire) & (1 << 9),
			1 << 9,
			"SEIP set when S-file has IPI"
		);

		// hart 处于 S 模式, OpenSBI 已移交内核.
		h1.mode = riscv_mode::S;
		h1.mstatus = 0; // SIE, SPIE 与 SPP 都为 0

		// SEIP 设置而 SEIE 初值为 0 时, check_pending_interrupts 不投递中断;
		// SEIP 挂起时 step_interrupts 会强制使能 SEIE.
		h1.mie |= 1 << 9; // 强制使能 SEIE

		// SEIP 与 SEIE 已置位, mideleg[9] 为 0, SEI 未委派, 陷态进入 M 模式.
		let result = check_pending_interrupts(&h1);
		assert!(result.is_some(), "interrupt must be pending");
		let (cause, is_m_mode) = result.unwrap();
		assert_eq!(cause, 9, "cause=9 (SEI)");
		assert!(is_m_mode, "SEI delivered to M-mode when mideleg[9]=0");

		// 模拟 OpenSBI 的 M 模式处理程序读取 mtopi; compute_mtopi 只检查 M 文件,
		// 返回 0.
		let (mtopi_val, _) = compute_mtopi(&mut h1, 0);
		assert_eq!(
			mtopi_val, 0,
			"mtopi=0 when S-file IPI and mideleg[9]=0: \
             M-file has nothing, S-file IPI is invisible to mtopi"
		);

		// mtopi 返回 0 后, OpenSBI 未撤销 SEIP 就执行 MRET,
		// SEIP 仍置位, 下一条指令再次触发 SEI.
		assert_eq!(
			h1.mip.load(Ordering::Acquire) & (1 << 9),
			1 << 9,
			"SEIP still set after mtopi=0 — spurious SEI loop"
		);
	}

	/// SEI 已委派且 S 模式关闭全局中断时, check_pending_interrupts 须跳过 SEI,
	/// 不得改为向 M 模式投递. 向 M 模式投递只存在于 S 文件的 IPI 会重复
	/// multi_hart_mtopi_zero_when_sfile_ipi_not_delegated 验证的过程:
	/// mtopi 返回 0, MRET 后再触发 SEI.
	#[test]
	fn multi_hart_sei_skipped_when_sie_zero_delegated() {
		let mut h1 = make_hart1_state();
		// SEI 已委派
		h1.mideleg = (1 << 9) | (1 << 1) | (1 << 5);
		// S 文件有 IPI 挂起
		h1.imsic_s.eip[0].store(1 << IID_S_IPI, Ordering::Relaxed);
		sync_imsic(&mut h1);

		// hart 处于 S 模式且 SIE 为 0, 内核关闭了全局中断.
		h1.mode = riscv_mode::S;
		h1.mstatus = 0; // SIE 为 0
		h1.mie |= 1 << 9; // 强制使能 SEIE, step_interrupts 会完成该操作

		// SEIP 与 SEIE 已置位, mideleg[9] 为 1, SEI 已委派到 S 模式;
		// SIE 为 0, 因此跳过该中断.
		let result = check_pending_interrupts(&h1);
		assert!(
			result.is_none(),
			"SEI skipped when mideleg[9]=1 and SIE=0: \
             no fall-through to M-mode"
		);

		// SEIP 仍置位, 中断挂起但未被接受, hart 继续执行;
		// SIE 置位后该 SEI 正常投递.
		assert_eq!(
			h1.mip.load(Ordering::Acquire) & (1 << 9),
			1 << 9,
			"SEIP stays pending when SIE=0"
		);
	}

	// ---- decode_imsic_addr 的 2 的整数次幂补齐 ----

	#[test]
	fn decode_imsic_addr_pads_non_power_of_two_harts() {
		// hart 数为 3 时补齐为 4. M 文件占用 [0, 0x4000),
		// S 文件占用 [0x4000, 0x8000), 与 DTB 及 Python 侧一致.
		let base = crate::state::IMSIC_M_BASE;

		// hart 0 至 2 的 M 文件须解码到正确的 hart.
		for hart in 0..3 {
			let a = decode_imsic_addr(base + (hart * 0x1000), 3).unwrap();
			assert_eq!(a.hart, hart as usize);
			assert!(!a.is_sfile, "offset {hart:#x} is an M-file");
			assert_eq!(a.reg_off, 0);
		}

		// S 文件的基址为 base + 0x4000, 即 base 加补齐后的 hart 数乘以 0x1000.
		let s0 = decode_imsic_addr(base + 0x4000, 3).unwrap();
		assert!(s0.is_sfile, "base+0x4000 is the S-file half");
		assert_eq!(s0.hart, 0, "S-file base maps to hart 0, not hart 1");

		// hart 2 的 S 文件位于 base + 0x4000 + 0x2000 = base + 0x6000.
		let s2 = decode_imsic_addr(base + 0x6000, 3).unwrap();
		assert!(s2.is_sfile);
		assert_eq!(s2.hart, 2);

		// 补齐位置上的 hart 3 没有对应文件, 解码须拒绝该地址.
		assert!(decode_imsic_addr(base + 0x7000, 3).is_none());
	}

	#[test]
	fn decode_imsic_addr_power_of_two_harts_unchanged() {
		// hart 数为 4 时补齐后仍为 4.
		let base = crate::state::IMSIC_M_BASE;
		for hart in 0..4 {
			let a = decode_imsic_addr(base + (hart * 0x1000), 4).unwrap();
			assert_eq!(a.hart, hart as usize);
			assert!(!a.is_sfile);
		}
		// S 文件的基址为 base + 0x4000.
		let s0 = decode_imsic_addr(base + 0x4000, 4).unwrap();
		assert!(s0.is_sfile);
		assert_eq!(s0.hart, 0);
	}
}
