//! Device MMIO helpers and inline handlers.

pub mod uart;
pub mod virtio;

use crate::state::FfiVirtIoCtx;

/// Device MMIO address ranges (base + end per device).
pub struct DevCtx {
	pub bases: *const u64,
	pub ends: *const u64,
	pub num: u8,
	/// virtio-blk MMIO base address (0 = no virtio device).
	pub virtio_base: u64,
	/// Mutable pointer to the FFI virtio-blk state that Rust updates inline.
	/// Valid for the duration of one ``run_parallel`` call.
	pub virtio_raw: *mut FfiVirtIoCtx,
}

impl DevCtx {
	/// 检查是否有设备将工作延迟到 Python 侧处理 (如 virtio QueueNotify
	/// 仅设 notify_pending=1, 实际 virtqueue 处理在退出后进行)。
	pub fn has_pending_python_work(&self) -> bool {
		if self.virtio_raw.is_null() {
			return false;
		}
		let notify = unsafe { (*self.virtio_raw).notify_pending };
		if notify != 0 {
			return true;
		}
		return false;
	}
}

pub fn is_device_addr(pa: u64, dev: &DevCtx) -> bool {
	for i in 0..dev.num as usize {
		let base = unsafe { *dev.bases.add(i) };
		let end = unsafe { *dev.ends.add(i) };
		if pa >= base && pa < end {
			return true;
		}
	}
	false
}
