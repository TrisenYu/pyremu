//! 内核使用的并发控制工具
//!
//! 与用户侧并发系统调用接口 (crate::syscall::concurrency) 无关: 本模块保护的是
//! 可信内核自身的共享状态, 当前唯一使用者是文件系统状态 (见 syscall::fs)。
//! 实现紧跟 ref-emod lock/lock.c 的 AMOSWAP。

use core::cell::UnsafeCell;
use core::ops::{ Deref, DerefMut };
use core::sync::atomic::{ AtomicI32, Ordering };

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
			// 自旋提示: 对应 ref-emod 的 spin_lock_check 与 spin_trylock 重试循环
			core::hint::spin_loop();
		}
		SpinGuard { lock: self }
	}

	/// 尝试获取锁，不阻塞。
	#[allow(unused)]
	pub fn try_lock(&self) -> Option<SpinGuard<'_, T>> {
		if self.lock.swap(1, Ordering::Acquire) == 0 { Some(SpinGuard { lock: self }) } else { None }
	}

	/// 检查锁是否被持有。
	#[allow(unused)]
	pub fn is_locked(&self) -> bool {
		self.lock.load(Ordering::Relaxed) != 0
	}
}

#[cfg(test)]
mod tests {
	use std::sync::Arc;
	use std::sync::mpsc;
	use std::thread;

	use super::SpinLock;

	/// 取得守卫后数据可读可写, 守卫离开作用域后锁被释放且数据保留。
	#[test]
	fn test_lock_guard_accesses_data_and_drop_releases() {
		let lock = SpinLock::new(0u32);
		assert!(!lock.is_locked());
		{
			let mut guard = lock.lock();
			assert!(lock.is_locked());
			assert_eq!(*guard, 0);
			*guard = 7;
		}
		assert!(!lock.is_locked());
		let guard = lock.try_lock().expect("锁已释放");
		assert_eq!(*guard, 7);
	}

	/// 本线程持有守卫期间 try_lock 返回 None 而不进入自旋, 释放后可再次取得。
	#[test]
	fn test_try_lock_fails_while_lock_is_held() {
		let lock = SpinLock::new(0u32);
		let guard = lock.lock();
		assert!(lock.try_lock().is_none());
		drop(guard);
		assert!(lock.try_lock().is_some());
	}

	/// 另一线程持有锁期间 is_locked 为真且 try_lock 返回 None, 该线程释放后可取得。
	#[test]
	fn test_try_lock_observes_another_thread_holding_the_lock() {
		let lock = Arc::new(SpinLock::new(0u32));
		let (held_tx, held_rx) = mpsc::channel();
		let (release_tx, release_rx) = mpsc::channel();
		let holder_lock = Arc::clone(&lock);
		let holder = thread::spawn(move || {
			let mut guard = holder_lock.lock();
			*guard = 11;
			held_tx.send(()).expect("通知已持有锁");
			release_rx.recv().expect("等待释放锁");
		});

		held_rx.recv().expect("等待另一线程持有锁");
		assert!(lock.is_locked());
		assert!(lock.try_lock().is_none());
		release_tx.send(()).expect("通知释放锁");
		holder.join().expect("持有线程结束");
		assert!(!lock.is_locked());
		let guard = lock.try_lock().expect("锁已释放");
		assert_eq!(*guard, 11);
	}
}
