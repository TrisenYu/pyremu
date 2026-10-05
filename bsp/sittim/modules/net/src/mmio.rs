//! 网卡 MMIO 寄存器的读写。
//!
//! 寄存器偏移与访问宽度按 virtio-mmio 规范 1.0 的现代布局 (version 2)。寄存器按 MMIO
//! 基地址加偏移访问, 该基地址由运行时经管理器给出; 访问只按寄存器宽度进行, 一律用
//! volatile。

use crate::platform;

#[cfg(not(target_arch = "riscv64"))]
use core::sync::atomic::{fence, Ordering};

/// 魔数 "virt"。
pub const MAGIC_VALUE: u32 = 0x7472_6976;
/// 现代布局的版本号。
pub const VERSION_MODERN: u32 = 2;
/// 网卡的设备号。
pub const DEVICE_ID_NET: u32 = 1;

// 寄存器偏移。
pub const R_MAGIC: u64 = 0x000;
pub const R_VERSION: u64 = 0x004;
pub const R_DEVICE_ID: u64 = 0x008;
pub const R_DEVICE_FEATURES: u64 = 0x010;
pub const R_DEVICE_FEATURES_SEL: u64 = 0x014;
pub const R_DRIVER_FEATURES: u64 = 0x020;
pub const R_DRIVER_FEATURES_SEL: u64 = 0x024;
pub const R_QUEUE_SEL: u64 = 0x030;
pub const R_QUEUE_NUM_MAX: u64 = 0x034;
pub const R_QUEUE_NUM: u64 = 0x038;
pub const R_QUEUE_READY: u64 = 0x044;
pub const R_QUEUE_NOTIFY: u64 = 0x050;
pub const R_STATUS: u64 = 0x070;
pub const R_QUEUE_DESC_LOW: u64 = 0x080;
pub const R_QUEUE_DESC_HIGH: u64 = 0x084;
pub const R_QUEUE_DRIVER_LOW: u64 = 0x090;
pub const R_QUEUE_DRIVER_HIGH: u64 = 0x094;
pub const R_QUEUE_DEVICE_LOW: u64 = 0x0A0;
pub const R_QUEUE_DEVICE_HIGH: u64 = 0x0A4;

/// 设备状态位。
pub const STATUS_ACKNOWLEDGE: u32 = 1;
pub const STATUS_DRIVER: u32 = 2;
pub const STATUS_DRIVER_OK: u32 = 4;
pub const STATUS_FEATURES_OK: u32 = 8;

/// 设备提供的 feature 位 (高 32 位)。
pub const FEATURE_VERSION_1: u64 = 1 << 32;
/// 设备在配置空间给出 MAC 地址。
pub const FEATURE_NET_MAC: u64 = 1 << 5;

/// 设备配置空间在 MMIO 区间内的偏移。MMIO 区间的前 0x100 字节是寄存器区, 自 0x100
/// 起是设备配置空间, 网卡的 MAC 在其中。
pub const DEVICE_CONFIG_OFFSET: u64 = 0x100;

/// 网卡的 MMIO 基地址。
fn base() -> u64 {
	platform::mmio_base()
}

/// 平台是否提供网卡。管理器给出的 MMIO 基地址为 0 即平台没有该网卡。
pub fn is_present() -> bool {
	platform::mmio_base() != 0
}

pub fn read32(offset: u64) -> u32 {
	unsafe { core::ptr::read_volatile((base() + offset) as *const u32) }
}

pub fn write32(offset: u64, value: u32) {
	unsafe { core::ptr::write_volatile((base() + offset) as *mut u32, value) };
}

/// 把一个 64 位地址写入低/高两个 32 位寄存器。
pub fn write64(low: u64, high: u64, value: u64) {
	write32(low, value as u32);
	write32(high, (value >> 32) as u32);
}

/// 读取设备配置空间中第 `local` 个字节。MAC 的六个字节不保证落在对齐的 32 位字上,
/// 故逐字节读取。
pub fn read_config_u8(local: u64) -> u8 {
	unsafe { core::ptr::read_volatile((base() + DEVICE_CONFIG_OFFSET + local) as *const u8) }
}

/// 等待此前的内存访问与设备访问全部完成。
///
/// RISC-V 取前驱集与后继集都含 I/O 的全量次序。其余目标没有把 I/O 单独列出的
/// 次序指令, 取 seq_cst 次序。
pub fn fence_io() {
	#[cfg(target_arch = "riscv64")]
	unsafe {
		core::arch::asm!("fence iorw, iorw");
	}
	#[cfg(not(target_arch = "riscv64"))]
	fence(Ordering::SeqCst);
}

#[cfg(test)]
mod tests {
	use super::{
		DEVICE_CONFIG_OFFSET, FEATURE_NET_MAC, FEATURE_VERSION_1, R_DEVICE_FEATURES,
		R_DEVICE_FEATURES_SEL, R_DEVICE_ID, R_DRIVER_FEATURES, R_DRIVER_FEATURES_SEL, R_MAGIC,
		R_QUEUE_DESC_HIGH, R_QUEUE_DESC_LOW, R_QUEUE_DEVICE_HIGH, R_QUEUE_DEVICE_LOW,
		R_QUEUE_DRIVER_HIGH, R_QUEUE_DRIVER_LOW, R_QUEUE_NOTIFY, R_QUEUE_NUM, R_QUEUE_NUM_MAX,
		R_QUEUE_READY, R_QUEUE_SEL, R_STATUS, R_VERSION, is_present, read32, read_config_u8,
		write32, write64,
	};
	use crate::platform;

	// 主机侧以进程内的字节数组代替 MMIO 区间, 寄存器写入可读回。各用例只取用协议栈
	// 的用例不写的寄存器: 协议栈的收发经 push_tx 与 finish_rx 写 R_QUEUE_NOTIFY,
	// 故本模块的用例不用该寄存器。

	/// 寄存器偏移两两不同, 且都落在设备配置空间之前。
	#[test]
	fn test_register_offsets_are_distinct_and_precede_the_config_space() {
		let offsets = [
			R_MAGIC,
			R_VERSION,
			R_DEVICE_ID,
			R_DEVICE_FEATURES,
			R_DEVICE_FEATURES_SEL,
			R_DRIVER_FEATURES,
			R_DRIVER_FEATURES_SEL,
			R_QUEUE_SEL,
			R_QUEUE_NUM_MAX,
			R_QUEUE_NUM,
			R_QUEUE_READY,
			R_QUEUE_NOTIFY,
			R_STATUS,
			R_QUEUE_DESC_LOW,
			R_QUEUE_DESC_HIGH,
			R_QUEUE_DRIVER_LOW,
			R_QUEUE_DRIVER_HIGH,
			R_QUEUE_DEVICE_LOW,
			R_QUEUE_DEVICE_HIGH,
		];
		for (i, a) in offsets.iter().enumerate() {
			for b in &offsets[i + 1..] {
				assert_ne!(a, b, "{a:#x} 与 {b:#x} 重合");
			}
		}
		assert!(offsets.iter().all(|&offset| offset + 4 <= DEVICE_CONFIG_OFFSET));
	}

	/// write64 依次写入低半与高半两个 32 位寄存器, 故每对寄存器相差 4 字节。
	#[test]
	fn test_the_high_halves_follow_the_low_halves() {
		assert_eq!(R_DEVICE_FEATURES_SEL, R_DEVICE_FEATURES + 4);
		assert_eq!(R_DRIVER_FEATURES_SEL, R_DRIVER_FEATURES + 4);
		assert_eq!(R_QUEUE_DESC_HIGH, R_QUEUE_DESC_LOW + 4);
		assert_eq!(R_QUEUE_DRIVER_HIGH, R_QUEUE_DRIVER_LOW + 4);
		assert_eq!(R_QUEUE_DEVICE_HIGH, R_QUEUE_DEVICE_LOW + 4);
	}

	/// 本模块访问的字节数覆盖配置空间内的六个 MAC 字节, 且不超出平台映射的 MMIO 区间。
	#[test]
	fn test_the_access_extent_covers_the_config_space() {
		/// 本模块访问的 MMIO 字节数: 设备配置空间内 MAC 的六个字节之后的第一个字节。
		const ACCESS_BYTES: u64 = DEVICE_CONFIG_OFFSET + 6;
		assert!(ACCESS_BYTES <= platform::MMIO_REGION_BYTES);
	}

	/// init 分两次写入协商结果, 故 MAC 位落在低 32 位内, 版本位落在高 32 位内。
	#[test]
	fn test_the_feature_bits_land_in_the_two_halves() {
		assert!(FEATURE_NET_MAC < 1 << 32);
		assert_eq!(FEATURE_VERSION_1 >> 32, 1);
	}

	/// 平台给出的 MMIO 基地址非零时, 平台即提供网卡。
	#[test]
	fn test_is_present_follows_the_given_base() {
		assert_ne!(platform::mmio_base(), 0);
		assert!(is_present());
	}

	/// 32 位寄存器写入后按同一偏移读回。
	#[test]
	fn test_a_register_reads_back_what_was_written() {
		write32(R_STATUS, 0xA5A5_5A5A);
		assert_eq!(read32(R_STATUS), 0xA5A5_5A5A);
	}

	/// write64 把 64 位地址分写到低半与高半两个寄存器。
	#[test]
	fn test_write64_splits_the_value_over_the_two_halves() {
		let value = 0x0123_4567_89AB_CDEF;
		write64(R_QUEUE_DESC_LOW, R_QUEUE_DESC_HIGH, value);
		assert_eq!(read32(R_QUEUE_DESC_LOW) as u64, value & 0xFFFF_FFFF);
		assert_eq!(read32(R_QUEUE_DESC_HIGH) as u64, value >> 32);
	}

	/// 配置空间内第 `local` 个字节按 MMIO 基地址加偏移读出, 即六个 MAC 字节不要求
	/// 落在对齐的 32 位字上也能逐个读出。
	#[test]
	fn test_the_config_space_bytes_are_read_by_offset() {
		let mac = [0x02u8, 0x00, 0x00, 0x11, 0x22, 0x33];
		for (local, byte) in mac.iter().enumerate() {
			let at = (platform::mmio_base() + DEVICE_CONFIG_OFFSET + local as u64) as *mut u8;
			unsafe { core::ptr::write_volatile(at, *byte) };
		}
		for (local, byte) in mac.iter().enumerate() {
			assert_eq!(read_config_u8(local as u64), *byte);
		}
	}
}
