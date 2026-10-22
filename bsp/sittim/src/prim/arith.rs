//! 整数对齐运算与计数值换算。对齐的取整粒度由第二个参数给出, 单位为字节; 计数值
//! 的每秒计数由第二个参数给出, 单位为赫兹。

use core::num::NonZeroU64;

/// 把 x 向上取整到 y 的整数倍。
/// y 必须为 2 的幂; 取 0 时结果无定义。
#[inline]
pub const fn align_up(x: u64, y: u64) -> u64 {
	(x + y - 1) & !(y - 1)
}

/// 把 x 向下取整到 y 的整数倍。
/// y 必须为 2 的幂; 取 0 时结果无定义。
#[inline]
pub const fn align_down(x: u64, y: u64) -> u64 {
	x & !(y - 1)
}

/// 把计数值换算为秒与微秒。
pub fn ticks_to_sec_usec(ticks: u64, per_sec: NonZeroU64) -> (u64, u64) {
	let per_sec = per_sec.get() as u128;
	let ticks = ticks as u128;
	let sec = ticks / per_sec;
	let usec = (ticks % per_sec) * 1_000_000 / per_sec;
	(sec as u64, usec as u64)
}

/// 把计数值换算为秒与纳秒。
pub fn ticks_to_sec_nsec(ticks: u64, per_sec: NonZeroU64) -> (u64, u64) {
	let per_sec = per_sec.get() as u128;
	let ticks = ticks as u128;
	let sec = ticks / per_sec;
	let nsec = (ticks % per_sec) * 1_000_000_000 / per_sec;
	(sec as u64, nsec as u64)
}

#[cfg(test)]
mod tests {
	use super::*;
	use crate::constants::{CHUNK_2M_SIZE, PAGE_SIZE};

	fn per_sec(value: u64) -> NonZeroU64 {
		NonZeroU64::new(value).expect("non zero value")
	}

	#[test]
	fn test_align_up_rounds_to_the_next_multiple() {
		assert_eq!(align_up(0x1000, PAGE_SIZE), 0x1000);
		assert_eq!(align_up(0x1001, PAGE_SIZE), 0x2000);
		assert_eq!(align_up(0x1, PAGE_SIZE), 0x1000);
		assert_eq!(align_up(0xfff, PAGE_SIZE), 0x1000);
		assert_eq!(align_up(0, PAGE_SIZE), 0);
	}

	#[test]
	fn test_align_down_rounds_to_the_previous_multiple() {
		assert_eq!(align_down(0x1fff, PAGE_SIZE), 0x1000);
		assert_eq!(align_down(0x1000, PAGE_SIZE), 0x1000);
		assert_eq!(align_down(0, PAGE_SIZE), 0);
	}

	/// 粒度由第二个参数给出, 同一入参在不同粒度下取得不同的结果。
	#[test]
	fn test_align_takes_the_granularity_from_the_second_argument() {
		assert_eq!(align_up(0x20_0001, CHUNK_2M_SIZE), 0x40_0000);
		assert_eq!(align_up(0x20_0001, PAGE_SIZE), 0x20_1000);
		assert_eq!(align_up(0x20_0000, CHUNK_2M_SIZE), 0x20_0000);

		assert_eq!(align_down(0x3f_ffff, CHUNK_2M_SIZE), 0x20_0000);
		assert_eq!(align_down(0x3f_ffff, PAGE_SIZE), 0x3f_f000);
		assert_eq!(align_down(0x20_0000, CHUNK_2M_SIZE), 0x20_0000);
	}

	#[test]
	fn test_sub_second_part_with_a_tick_rate_below_one_megahertz() {
		let rate = per_sec(100_000);
		assert_eq!(ticks_to_sec_usec(0, rate), (0, 0));
		assert_eq!(ticks_to_sec_usec(50_000, rate), (0, 500_000));
		assert_eq!(ticks_to_sec_usec(150_000, rate), (1, 500_000));
		assert_eq!(ticks_to_sec_nsec(50_000, rate), (0, 500_000_000));
	}

	#[test]
	fn test_sub_second_part_with_a_rate_above_one_megahertz() {
		let rate = per_sec(10_000_000);
		assert_eq!(ticks_to_sec_usec(5_000_000, rate), (0, 500_000));
		assert_eq!(ticks_to_sec_usec(10_000_000, rate), (1, 0));
		assert_eq!(ticks_to_sec_nsec(10_000_001, rate), (1, 100));
	}

	#[test]
	fn test_sub_second_part_stays_below_one_second() {
		let rate = per_sec(100_000);
		assert_eq!(ticks_to_sec_nsec(99_999, rate), (0, 999_990_000));
	}

	#[test]
	fn test_large_tick_rate_does_not_overflow() {
		let rate = per_sec(1 << 40);
		assert_eq!(ticks_to_sec_nsec((1 << 40) - 1, rate), (0, 999_999_999));
	}
}
