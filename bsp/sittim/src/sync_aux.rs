//! 内核使用的并发控制工具
//!
//! 与用户侧并发系统调用接口 (crate::syscall::concurrency) 无关: 本模块保护的是
//! 可信内核自身的共享状态, 当前唯一使用者是文件系统状态 (见 syscall::fs)。
//! 实现紧跟 ref-emod lock/lock.c 的 AMOSWAP。

use core::cell::UnsafeCell;
use core::ops::{ Deref, DerefMut };
use core::sync::atomic::{ AtomicI32, Ordering };

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

/// 自旋锁。离开作用域自动释放。
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

	pub fn lock(&self) -> SpinGuard<'_, T> {
		// TAS: while amoswap.w != 0 { spin }
		while self.lock.swap(1, Ordering::Acquire) != 0 {
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
		let guard = lock.try_lock().expect("lock has been released");
		assert_eq!(*guard, 7);
	}

	#[test]
	fn test_try_lock_fails_while_lock_is_held() {
		let lock = SpinLock::new(0u32);
		let guard = lock.lock();
		assert!(lock.try_lock().is_none());
		drop(guard);
		assert!(lock.try_lock().is_some());
	}

	#[test]
	fn test_try_lock_observes_another_thread_holding_the_lock() {
		let lock = Arc::new(SpinLock::new(0u32));
		let (held_tx, held_rx) = mpsc::channel();
		let (release_tx, release_rx) = mpsc::channel();
		let holder_lock = Arc::clone(&lock);
		let holder = thread::spawn(move || {
			let mut guard = holder_lock.lock();
			*guard = 11;
			held_tx.send(()).expect("lock has been hold");
			release_rx.recv().expect("wait for releasing lock");
		});

		held_rx.recv().expect("");
		assert!(lock.is_locked());
		assert!(lock.try_lock().is_none());
		release_tx.send(()).expect("notifying for releasing the lock");
		holder.join().expect("end of the lock's lifetime");
		assert!(!lock.is_locked());
		let guard = lock.try_lock().expect("lock has been released");
		assert_eq!(*guard, 11);
	}
}
