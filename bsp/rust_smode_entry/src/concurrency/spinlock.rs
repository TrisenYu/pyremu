//! 自旋锁。紧跟 ref-emod lock/lock.c 的 AMOSWAP 实现。
//! 用于多 hart 场景下保护共享资源（UART、页池、上下文）。

use core::cell::UnsafeCell;
use core::ops::{Deref, DerefMut};
use core::sync::atomic::{AtomicI32, Ordering};

/// 自旋锁守卫 — unlock 在 drop 时自动完成。
pub struct SpinGuard<'a, T> {
	lock: &'a SpinLock<T>,
}

impl<T> Deref for SpinGuard<'_, T> {
	type Target = T;
	#[inline]
	fn deref(&self) -> &T {
		unsafe { &*self.lock.data.get() }
	}
}

impl<T> DerefMut for SpinGuard<'_, T> {
	#[inline]
	fn deref_mut(&mut self) -> &mut T {
		unsafe { &mut *self.lock.data.get() }
	}
}

impl<T> Drop for SpinGuard<'_, T> {
	fn drop(&mut self) {
		self.lock.lock.store(0, Ordering::Release);
	}
}

/// 自旋锁。`lock()` 返回守卫，离开作用域自动释放。
pub struct SpinLock<T> {
	lock: AtomicI32,
	data: UnsafeCell<T>,
}

// SpinLock 本身是 Send + Sync 当 T 是 Send 时。
unsafe impl<T: Send> Send for SpinLock<T> {}
unsafe impl<T: Send> Sync for SpinLock<T> {}

impl<T> SpinLock<T> {
	/// 用给定数据创建自旋锁。
	pub const fn new(data: T) -> Self {
		Self {
			lock: AtomicI32::new(0),
			data: UnsafeCell::new(data),
		}
	}

	/// 获取锁。自旋直到成功。
	pub fn lock(&self) -> SpinGuard<'_, T> {
		// TAS: while amoswap.w != 0 { spin }
		while self.lock.swap(1, Ordering::Acquire) != 0 {
			// 自旋提示（对应 ref-emod 的 spin_lock_check -> spin_trylock 循环）
			core::hint::spin_loop();
		}
		SpinGuard { lock: self }
	}

	/// 尝试获取锁，不阻塞。
	#[allow(unused)]
	pub fn try_lock(&self) -> Option<SpinGuard<'_, T>> {
		if self.lock.swap(1, Ordering::Acquire) == 0 {
			Some(SpinGuard { lock: self })
		} else {
			None
		}
	}

	/// 检查锁是否被持有（调试用）。
	#[allow(unused)]
	pub fn is_locked(&self) -> bool {
		self.lock.load(Ordering::Relaxed) != 0
	}
}
