//! 运行时经管理器交给本模块的平台设施。
//!
//! 三项设施都只能由运行时给出: 网卡的 MMIO 基地址由运行时的设备映射建立, 地址翻译要
//! 读飞地的页表, 时间源的计数与频率取自平台配置。模块的入口收到管理器回调集合, 在此
//! 留存, 驱动与协议栈经本模块取用。
//!
//! 主机目标上没有管理器, 三项以进程内的状态代替: MMIO 基地址指向本模块的一个字节
//! 数组, 地址翻译取恒等映射, 时间基准取 0。

#[cfg(target_arch = "riscv64")]
use core::sync::atomic::{AtomicPtr, Ordering};

#[cfg(target_arch = "riscv64")]
use crate::man::Manager;

/// 运行时的管理器回调集合。模块入口在取用任何设施之前写入一次, 此后只读。
#[cfg(target_arch = "riscv64")]
static MANAGER: AtomicPtr<Manager> = AtomicPtr::new(core::ptr::null_mut());

/// 留存运行时的管理器回调集合。
#[cfg(target_arch = "riscv64")]
pub fn attach(manager: *const Manager) {
	MANAGER.store(manager as *mut Manager, Ordering::Release);
}

/// 取管理器回调集合。调用方须先经 [`attach`] 留存。
#[cfg(target_arch = "riscv64")]
fn manager() -> &'static Manager {
	unsafe { &*MANAGER.load(Ordering::Acquire) }
}

/// 网卡的 MMIO 基地址; 平台不提供该网卡时取 0。
#[cfg(target_arch = "riscv64")]
pub fn mmio_base() -> u64 {
	manager().device_window_va
}

/// 把飞地地址空间中的虚拟地址翻译为物理地址。
#[cfg(target_arch = "riscv64")]
pub fn va_to_pa(va: u64) -> u64 {
	unsafe { (manager().va_to_pa)(va) }
}

/// 读时间源, 单位为计数。
#[cfg(target_arch = "riscv64")]
pub fn read_time() -> u64 {
	unsafe { (manager().read_time)() }
}

/// 时间源的频率, 单位为赫兹。
#[cfg(target_arch = "riscv64")]
pub fn time_freq() -> u64 {
	manager().time_freq
}

/// 主机侧代替网卡 MMIO 区间的字节数, 取运行时配置中的同名取值。
#[cfg(not(target_arch = "riscv64"))]
pub const MMIO_REGION_BYTES: u64 = 0x200;

/// 主机侧代替网卡 MMIO 区间的字节数组。
#[cfg(not(target_arch = "riscv64"))]
#[repr(C, align(16))]
struct MmioRegion([u8; MMIO_REGION_BYTES as usize]);

/// 主机侧的 MMIO 区间, 供驱动按寄存器偏移读写。
#[cfg(not(target_arch = "riscv64"))]
static mut MMIO_REGION: MmioRegion = MmioRegion([0; MMIO_REGION_BYTES as usize]);

#[cfg(not(target_arch = "riscv64"))]
pub fn mmio_base() -> u64 {
	core::ptr::addr_of!(MMIO_REGION) as u64
}

/// 主机侧的地址翻译取恒等映射: 虚拟地址与物理地址同值。
#[cfg(not(target_arch = "riscv64"))]
pub fn va_to_pa(va: u64) -> u64 {
	va
}

/// 主机上没有时钟源, 时间基准取 0。
#[cfg(not(target_arch = "riscv64"))]
pub fn read_time() -> u64 {
	0
}

/// 主机侧的时间源频率, 取运行时配置中的同名取值, 使毫秒计数非零。
#[cfg(not(target_arch = "riscv64"))]
pub fn time_freq() -> u64 {
	10_000_000
}
