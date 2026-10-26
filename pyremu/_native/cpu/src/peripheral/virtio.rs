//! 在加速执行引擎内处理 virtio-blk 的 MMIO 访问, 寄存器读写无需切换到 Python 侧.
//!
//! 设备特性、队列描述符与配置空间的读都由 Rust 直接处理. 偏移 0x050 的
//! ``QueueNotify`` 返回 ``None``, 由调用方强制 MMIO 退出;
//! 队列在退出链接库执行上下文后清空, 并跳过重复的执行.

use super::DevCtx;

/// virtio-blk 声明的设备特性, 共 64 位.
///
/// 不声明第29位 VIRTIO_F_RING_EVENT_IDX 与第28位 VIRTIO_F_RING_INDIRECT_DESC
/// 声明后驱动会执行本设备不支持的分支. 不声明时驱动改用
/// ``VRING_USED_F_NO_NOTIFY`` 标志通知与描述符.
const VIRTIO_DEVICE_FEATURES: u64 = 1u64 << 32; // VIRTIO_F_VERSION_1

/// 在加速执行引擎内就地处理 virtio-blk 的 MMIO 访问. 基址与状态指针都定义于
/// ``DevCtx`` 中, 处理函数的调用链无需再传额外参数. 返回值:
/// - ``Some(data)``: 读成功, 调用方把 data 写入目标寄存器.
/// - ``Some(0)``: 写成功.
/// - ``None``: pa 不在 virtio MMIO 范围内, 或该访问必须强制 MMIO 退出:
///   ``QueueNotify`` 交由 Python 重新执行存储指令并调用 ``_process_queue()``.
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

/// 就地处理 virtio-blk 的 MMIO 写, 返回值约定见 ``try_handle_virtio``.
fn virtio_write(offset: u64, write_data: u64, raw: *mut crate::ffi::FfiVirtIoCtx) -> Option<u64> {
	match offset {
		// DeviceFeaturesSel: 0x014, 选择设备特性页
		0x014 => unsafe {
			(*raw).device_features_sel = write_data as u32;
			Some(0)
		},
		// DriverFeatures: 0x020, 按 driver_features_sel 选择要写入的特性页
		0x020 => unsafe {
			let sel = (*raw).driver_features_sel;
			let mask = (write_data as u64 & 0xFFFF_FFFF) << (sel * 32);
			(*raw).driver_features =
				((*raw).driver_features & !(0xFFFF_FFFFu64 << (sel * 32))) | mask;
			Some(0)
		},
		// DriverFeaturesSel: 0x024, 选择驱动特性页
		0x024 => unsafe {
			(*raw).driver_features_sel = write_data as u32;
			Some(0)
		},
		// QueueSel: 0x030, 选择队列
		0x030 => unsafe {
			(*raw).queue_sel = write_data as u32;
			Some(0)
		},
		// QueueNum: 0x038, 取值不超过 queue_num_max
		0x038 => unsafe {
			(*raw).queue_num = core::cmp::min(write_data as u32, (*raw).queue_num_max);
			Some(0)
		},
		// QueueReady: 0x044, 队列就绪标志
		0x044 => unsafe {
			(*raw).queue_ready = if write_data != 0 { 1 } else { 0 };
			Some(0)
		},
		// QueueNotify: 0x050. 该写返回 None, 强制 MMIO 退出. 若在此处处理,
		// AIA 模式的中断在 CSR 上完成, 队列处理要等到本轮加速执行结束, 延迟可达数秒.
		// 退出前置位 notify_pending, 供退出动态链接库后清空队列
		0x050 => unsafe {
			(*raw).notify_pending = 1;
			None
		},
		// InterruptStatus: 0x060, 写入即置位中断位
		0x060 => unsafe {
			(*raw).interrupt_status |= write_data as u32;
			Some(0)
		},
		// InterruptACK: 0x064, 清空中断位. ACK 前 ISR 非零且 ACK 后归零时返回 None,
		// 强制 MMIO 退出
		0x064 => {
			let old = unsafe { (*raw).interrupt_status };
			let new = old & !(write_data as u32);
			if new == 0 && old != 0 {
				return None;
			}
			unsafe { (*raw).interrupt_status = new };
			Some(0)
		}
		// Status: 0x070, 写入 0 复位设备
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
		// QueueDescLow 与 QueueDescHigh: 0x080 与 0x084
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
		// QueueDriverLow 与 QueueDriverHigh: 0x090 与 0x094
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
		// QueueDeviceLow 与 QueueDeviceHigh 分别对应 0x0A0 与 0x0A4
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
		// 配置空间位于 0x100 起, 只读。其余未定义偏移的写入同样无效果.
		_ => Some(0),
	}
}

/// 处理 virtio-blk 的 MMIO 读, 返回值约定见 ``try_handle_virtio``.
fn virtio_read(offset: u64, raw: *mut crate::ffi::FfiVirtIoCtx) -> Option<u64> {
	match offset {
		// MagicValue: 0x000
		0x000 => Some(0x74726976),
		// Version: 0x004
		0x004 => Some(0x2),
		// DeviceID: 0x008, 2 表示块设备
		0x008 => Some(0x2),
		// VendorID: 0x00C
		0x00C => Some(0x0),
		// DeviceFeatures: 0x010, 按 device_features_sel 选择特性页
		0x010 => {
			let sel = unsafe { (*raw).device_features_sel };
			Some((VIRTIO_DEVICE_FEATURES >> (sel * 32)) & 0xFFFF_FFFF)
		}
		// QueueNumMax: 0x034
		0x034 => unsafe { Some((*raw).queue_num_max as u64) },
		// InterruptStatus: 0x060
		0x060 => unsafe { Some((*raw).interrupt_status as u64) },
		// Status: 0x070
		0x070 => unsafe { Some((*raw).status as u64) },
		// ConfigGeneration: 0x0FC
		0x0FC => Some(0),
		// 配置空间: 0x100 起
		off if off >= 0x100 => {
			let local = off - 0x100;
			match local {
				// Capacity: u64, 低 32 位在 0x000, 高 32 位在 0x004
				0x000 => unsafe { Some((*raw).capacity & 0xFFFF_FFFF) },
				0x004 => unsafe { Some(((*raw).capacity >> 32) & 0xFFFF_FFFF) },
				_ => Some(0),
			}
		}
		_ => Some(0),
	}
}

#[cfg(test)]
mod tests {
	use super::try_handle_virtio;
	use crate::ffi::FfiVirtIoCtx;
	use crate::peripheral::DevCtx;

	fn make_dev(ctx: &mut FfiVirtIoCtx) -> DevCtx {
		DevCtx {
			bases: std::ptr::null(),
			ends: std::ptr::null(),
			num: 0,
			virtio_base: 0x1000_8000,
			virtio_raw: ctx as *mut FfiVirtIoCtx,
			plic: std::ptr::null_mut(),
		}
	}

	/// ACK 清空全部中断位时必须强制 MMIO 退出, 返回 ``None``: 否则 PLIC complete
	/// 的处理会读到过时的高电平, pending 被重新置位, 产生虚假中断.
	#[test]
	fn ack_clearing_all_bits_forces_mmio_exit() {
		let mut ctx: FfiVirtIoCtx = unsafe { std::mem::zeroed() };
		ctx.interrupt_status = 1;
		let dev = make_dev(&mut ctx);
		let r = try_handle_virtio(0x1000_8000 + 0x064, true, 1, 4, &dev);
		assert_eq!(r, None, "ACK 清空全部中断位应强制 MMIO 退出以同步拉低电平");
	}

	/// ACK 只清除部分中断位且 ISR 仍非零时就地处理, 不退出加速执行, 无需拉低中断线电平.
	#[test]
	fn ack_clearing_partial_bits_stays_inline() {
		let mut ctx: FfiVirtIoCtx = unsafe { std::mem::zeroed() };
		ctx.interrupt_status = 0b11;
		let dev = make_dev(&mut ctx);
		let r = try_handle_virtio(0x1000_8000 + 0x064, true, 1, 4, &dev);
		assert_eq!(r, Some(0), "部分清除应内联处理");
		assert_eq!(ctx.interrupt_status, 0b10, "应仅清除 ACK 指定位");
	}

	/// ACK 写入 0 时不改变 ISR, 就地处理.
	#[test]
	fn ack_zero_stays_inline() {
		let mut ctx: FfiVirtIoCtx = unsafe { std::mem::zeroed() };
		ctx.interrupt_status = 1;
		let dev = make_dev(&mut ctx);
		let r = try_handle_virtio(0x1000_8000 + 0x064, true, 0, 4, &dev);
		assert_eq!(r, Some(0));
		assert_eq!(ctx.interrupt_status, 1);
	}
}
