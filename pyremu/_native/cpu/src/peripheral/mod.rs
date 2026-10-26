//! 设备 MMIO 辅助函数与对应处理函数.

pub mod plic;
pub mod uart;
pub mod virtio;

use crate::ffi::FfiPlicCtx;
use crate::ffi::FfiVirtIoCtx;

/// 设备 MMIO 地址区间。 ``bases[i]`` 与 ``ends[i]`` 记录第 i 个设备的起始与结束地址.
pub struct DevCtx {
	pub bases: *const u64,
	pub ends: *const u64,
	pub num: u8,
	/// virtio-blk 的 MMIO 基址; 为 0 时不注册 virtio 设备.
	pub virtio_base: u64,
	/// 指向 FFI virtio-blk 状态的可变指针.
	/// 用于更新虚拟块设备状态.
	pub virtio_raw: *mut FfiVirtIoCtx,
	/// FFI PLIC 状态的可变指针, 用于更新 claim、complete、
	/// 优先级、使能与阈值. 空指针表示无 PLIC 设备, 仅在单次 ``run_parallel``
	/// 调用期间有效.
	pub plic: *mut FfiPlicCtx,
}

impl DevCtx {
	/// 判定是否有设备把工作延迟到 Python 侧处理: QueueNotify 只置位
	/// ``notify_pending``, 虚拟队列的处理在 MMIO 退出后进行.
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
