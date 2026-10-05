//! 地址与长度的对齐运算。取整粒度由第二个参数给出, 单位为字节。

/// 把 x 向上取整到 y 的整数倍。
///
/// y 必须为 2 的幂; 取 0 时结果无定义。
#[inline]
pub const fn align_up(x: u64, y: u64) -> u64 {
	(x + y - 1) & !(y - 1)
}

/// 把 x 向下取整到 y 的整数倍。
///
/// y 必须为 2 的幂; 取 0 时结果无定义。
#[inline]
pub const fn align_down(x: u64, y: u64) -> u64 {
	x & !(y - 1)
}

// ---------------------------------------------------------------
//  测试
// ---------------------------------------------------------------

#[cfg(test)]
mod tests {
	use super::*;
	use crate::constants::{CHUNK_2M_SIZE, PAGE_SIZE};

	#[test]
	fn test_align_up_rounds_to_the_next_multiple() {
		assert_eq!(align_up(0x1000, PAGE_SIZE), 0x1000);
		assert_eq!(align_up(0x1001, PAGE_SIZE), 0x2000);
		assert_eq!(align_up(0x1, PAGE_SIZE), 0x1000);
		assert_eq!(align_up(0xFFF, PAGE_SIZE), 0x1000);
		assert_eq!(align_up(0, PAGE_SIZE), 0);
	}

	#[test]
	fn test_align_down_rounds_to_the_previous_multiple() {
		assert_eq!(align_down(0x1FFF, PAGE_SIZE), 0x1000);
		assert_eq!(align_down(0x1000, PAGE_SIZE), 0x1000);
		assert_eq!(align_down(0, PAGE_SIZE), 0);
	}

	/// 粒度由第二个参数给出, 同一入参在不同粒度下取得不同的结果。
	#[test]
	fn test_align_takes_the_granularity_from_the_second_argument() {
		assert_eq!(align_up(0x20_0001, CHUNK_2M_SIZE), 0x40_0000);
		assert_eq!(align_up(0x20_0001, PAGE_SIZE), 0x20_1000);
		assert_eq!(align_up(0x20_0000, CHUNK_2M_SIZE), 0x20_0000);

		assert_eq!(align_down(0x3F_FFFF, CHUNK_2M_SIZE), 0x20_0000);
		assert_eq!(align_down(0x3F_FFFF, PAGE_SIZE), 0x3F_F000);
		assert_eq!(align_down(0x20_0000, CHUNK_2M_SIZE), 0x20_0000);
	}
}
