//! 一条 virtqueue 的描述符表、可用环形缓冲区与已用环形缓冲区的驱动侧读写。
//!
//! 三处结构由驱动以静态存储提供, 设备按物理地址访问。未协商
//! VIRTIO_F_RING_EVENT_IDX 与 VIRTIO_F_RING_INDIRECT_DESC, 故可用环形缓冲区无
//! used_event 字段, 描述符表也不含间接表项; 未协商 VIRTIO_F_RING_AVAIL_F_NO_INTERRUPT,
//! 故已用环形缓冲区无 avail_event 字段。
//!
//! 描述符表与可用环形缓冲区由驱动写、设备读, 已用环形缓冲区由设备写、驱动读。驱动向
//! 可用环形缓冲区写入一项后调用 mmio::fence_io 再写可用环形缓冲区的索引; 读已用环形
//! 缓冲区的项时先读该环形缓冲区的索引, 调用 mmio::fence_io 再读该项。

use crate::mmio;
use crate::platform;

/// 描述符表、可用环形缓冲区与已用环形缓冲区的项数。
pub const QNUM: u16 = 8;

/// 描述符表项: 缓冲区物理地址 (8 字节), 长度 (4 字节), 标志 (2 字节), 链中下一项 (2 字节)。
pub const DESC_SIZE: usize = 16;
/// 缓冲区由设备写入。
pub const DESC_F_WRITE: u16 = 2;

/// 可用环形缓冲区占用的字节数: 标志 (2 字节), 索引 (2 字节), 其后每项 2 字节。
const AVAIL_BYTES: usize = 4 + QNUM as usize * 2;
/// 已用环形缓冲区占用的字节数: 标志 (2 字节), 索引 (2 字节), 其后每项 8 字节。
const USED_BYTES: usize = 4 + QNUM as usize * 8;

/// 一条队列在内存中的三处结构: 描述符表、可用环形缓冲区与已用环形缓冲区。描述符表须
/// 16 字节对齐, 可用环形缓冲区与已用环形缓冲区须 2 字节对齐, 整体按 16 字节对齐即同时满足。
#[repr(C, align(16))]
pub struct QueueStorage {
	desc: [u8; QNUM as usize * DESC_SIZE],
	avail: [u8; AVAIL_BYTES],
	used: [u8; USED_BYTES],
}

impl QueueStorage {
	pub const fn new() -> Self {
		Self {
			desc: [0; QNUM as usize * DESC_SIZE],
			avail: [0; AVAIL_BYTES],
			used: [0; USED_BYTES],
		}
	}
}

/// 已用环形缓冲区的项: 描述符索引与设备写入的字节数。
#[derive(Clone, Copy)]
pub struct UsedEntry {
	pub id: u32,
	pub len: u32,
}

/// 一条队列的驱动侧视图。
#[derive(Clone, Copy)]
pub struct Queue {
	desc: u64,
	avail: u64,
	used: u64,
	desc_pa: u64,
	avail_pa: u64,
	used_pa: u64,
	next_avail: u16,
	last_used: u16,
}

impl Queue {
	pub const fn new() -> Self {
		Self {
			desc: 0,
			avail: 0,
			used: 0,
			desc_pa: 0,
			avail_pa: 0,
			used_pa: 0,
			next_avail: 0,
			last_used: 0,
		}
	}

	/// 把本队列接到静态存储上, 并算出描述符表、可用环形缓冲区与已用环形缓冲区的物理地址。
	pub fn attach(&mut self, storage: *mut QueueStorage) {
		unsafe {
			self.desc = core::ptr::addr_of_mut!((*storage).desc) as u64;
			self.avail = core::ptr::addr_of_mut!((*storage).avail) as u64;
			self.used = core::ptr::addr_of_mut!((*storage).used) as u64;
		}
		self.desc_pa = platform::va_to_pa(self.desc);
		self.avail_pa = platform::va_to_pa(self.avail);
		self.used_pa = platform::va_to_pa(self.used);
		self.next_avail = 0;
		self.last_used = 0;
	}

	pub fn desc_pa(&self) -> u64 {
		self.desc_pa
	}

	pub fn avail_pa(&self) -> u64 {
		self.avail_pa
	}

	pub fn used_pa(&self) -> u64 {
		self.used_pa
	}

	/// 下一个待写入可用环形缓冲区的槽位。
	pub fn next_slot(&self) -> u16 {
		self.next_avail % QNUM
	}

	/// 可用环形缓冲区中设备尚未取走的槽位数。
	pub fn space_left(&self) -> u16 {
		QNUM - self.next_avail.wrapping_sub(self.last_used).min(QNUM)
	}

	/// 写入一项描述符表项。
	pub fn write_desc(&mut self, idx: u16, addr_pa: u64, len: u32, flags: u16) {
		let base = self.desc + idx as u64 * DESC_SIZE as u64;
		unsafe {
			core::ptr::write_volatile(base as *mut u64, addr_pa);
			core::ptr::write_volatile((base + 8) as *mut u32, len);
			core::ptr::write_volatile((base + 12) as *mut u16, flags);
			core::ptr::write_volatile((base + 14) as *mut u16, 0);
		}
	}

	/// 向可用环形缓冲区的槽位写入描述符索引, 并写入推进后的可用环形缓冲区的索引。
	pub fn push_avail(&mut self, idx: u16) {
		let slot = self.next_slot() as u64;
		unsafe {
			core::ptr::write_volatile((self.avail + 4 + slot * 2) as *mut u16, idx);
		}
		self.next_avail = self.next_avail.wrapping_add(1);
		mmio::fence_io();
		unsafe {
			core::ptr::write_volatile((self.avail + 2) as *mut u16, self.next_avail);
		}
	}

	/// 读取一个已用环形缓冲区的项; 该环形缓冲区的索引未推进时返回 None。本调用不推进
	/// 该索引, 由 commit_used 推进。
	pub fn peek_used(&self) -> Option<UsedEntry> {
		let idx = unsafe { core::ptr::read_volatile((self.used + 2) as *const u16) };
		if idx == self.last_used {
			return None;
		}
		mmio::fence_io();
		let slot = (self.last_used % QNUM) as u64;
		let base = self.used + 4 + slot * 8;
		Some(UsedEntry {
			id: unsafe { core::ptr::read_volatile(base as *const u32) },
			len: unsafe { core::ptr::read_volatile((base + 4) as *const u32) },
		})
	}

	/// 推进已用环形缓冲区的索引, 丢弃 peek_used 读到的那一项。
	pub fn commit_used(&mut self) {
		self.last_used = self.last_used.wrapping_add(1);
	}
}

#[cfg(test)]
mod tests {
	use super::{AVAIL_BYTES, DESC_F_WRITE, DESC_SIZE, QNUM, Queue, QueueStorage, USED_BYTES};

	/// 描述符表、可用环形缓冲区与已用环形缓冲区在存储中的起始偏移。
	const DESC_OFF: usize = 0;
	const AVAIL_OFF: usize = QNUM as usize * DESC_SIZE;
	const USED_OFF: usize = AVAIL_OFF + AVAIL_BYTES;

	fn u16_at(bytes: &[u8], at: usize) -> u16 {
		u16::from_le_bytes([bytes[at], bytes[at + 1]])
	}

	fn u32_at(bytes: &[u8], at: usize) -> u32 {
		u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
	}

	fn u64_at(bytes: &[u8], at: usize) -> u64 {
		let mut raw = [0u8; 8];
		raw.copy_from_slice(&bytes[at..at + 8]);
		u64::from_le_bytes(raw)
	}

	/// 一条接好静态存储的队列, 以及对该存储三处区域的读写途径。
	struct Store {
		storage: Box<QueueStorage>,
		queue: Queue,
	}

	impl Store {
		fn new() -> Self {
			let mut storage = Box::new(QueueStorage::new());
			let mut queue = Queue::new();
			queue.attach(&mut *storage);
			Self { storage, queue }
		}

		fn base(&self) -> u64 {
			core::ptr::addr_of!(*self.storage) as u64
		}

		fn desc_bytes(&self) -> &[u8] {
			&self.storage.desc
		}

		fn avail_bytes(&self) -> &[u8] {
			&self.storage.avail
		}

		/// 可用环形缓冲区的索引。
		fn avail_index(&self) -> u16 {
			u16_at(self.avail_bytes(), 2)
		}

		/// 可用环形缓冲区第 `slot` 个槽位内的描述符索引。
		fn avail_slot(&self, slot: usize) -> u16 {
			u16_at(self.avail_bytes(), 4 + slot * 2)
		}

		fn used_index(&self) -> u16 {
			u16_at(&self.storage.used, 2)
		}

		/// 设备侧写入一个已用环形缓冲区的项, 并把该环形缓冲区的索引推进一项。
		fn device_push_used(&mut self, id: u32, len: u32) {
			let idx = self.used_index();
			let at = 4 + (idx % QNUM) as usize * 8;
			self.storage.used[at..at + 4].copy_from_slice(&id.to_le_bytes());
			self.storage.used[at + 4..at + 8].copy_from_slice(&len.to_le_bytes());
			self.storage.used[2..4].copy_from_slice(&idx.wrapping_add(1).to_le_bytes());
		}
	}

	/// 存储按 16 字节对齐, 三处区域的地址依次相差描述符表与可用环形缓冲区的字节数。
	#[test]
	fn test_attach_records_the_three_region_addresses() {
		let s = Store::new();
		assert_eq!(core::mem::align_of::<QueueStorage>(), 16);
		assert_eq!(s.base() % 16, 0);
		assert_eq!(s.queue.desc_pa(), s.base() + DESC_OFF as u64);
		assert_eq!(s.queue.avail_pa(), s.base() + AVAIL_OFF as u64);
		assert_eq!(s.queue.used_pa(), s.base() + USED_OFF as u64);
		assert_eq!(s.queue.next_slot(), 0);
		assert_eq!(s.queue.space_left(), QNUM);
	}

	/// 三处区域之间没有间隙; 存储末尾的对齐填充小于 16 字节。
	#[test]
	fn test_storage_regions_are_contiguous() {
		let size = core::mem::size_of::<QueueStorage>();
		assert!(size >= USED_OFF + USED_BYTES);
		assert!(size - (USED_OFF + USED_BYTES) < 16);
	}

	/// 每发布一项, 可用槽位数减一, 槽位下标以 QNUM 作为取模对象推进。
	#[test]
	fn test_space_left_falls_as_the_driver_publishes() {
		let mut s = Store::new();
		for i in 0..QNUM {
			assert_eq!(s.queue.space_left(), QNUM - i);
			assert_eq!(s.queue.next_slot(), i);
			s.queue.push_avail(i);
		}
		assert_eq!(s.queue.space_left(), 0);
		assert_eq!(s.queue.next_slot(), 0);

		// 环满之后继续发布时, 可用槽位数不会取到负值, 槽位下标回到环的起点。
		s.queue.push_avail(0);
		assert_eq!(s.queue.space_left(), 0);
		assert_eq!(s.queue.next_slot(), 1);
	}

	/// 设备取走一项并由驱动提交之后, 该槽位回到可用状态。
	#[test]
	fn test_commit_used_returns_the_avail_slot() {
		let mut s = Store::new();
		s.queue.push_avail(0);
		s.queue.push_avail(1);
		assert_eq!(s.queue.space_left(), QNUM - 2);

		s.device_push_used(0, 64);
		assert_eq!(s.queue.peek_used().unwrap().id, 0);
		s.queue.commit_used();
		assert_eq!(s.queue.space_left(), QNUM - 1);

		s.device_push_used(1, 64);
		assert_eq!(s.queue.peek_used().unwrap().id, 1);
		s.queue.commit_used();
		assert_eq!(s.queue.space_left(), QNUM);
	}

	/// 发布一项时先写槽位再写索引。
	#[test]
	fn test_push_avail_writes_the_slot_then_the_index() {
		let mut s = Store::new();
		assert_eq!(u16_at(s.avail_bytes(), 0), 0);
		assert_eq!(s.avail_index(), 0);

		s.queue.push_avail(5);
		assert_eq!(s.avail_slot(0), 5);
		assert_eq!(s.avail_index(), 1);
	}

	/// 发布 QNUM 项之后环被填满, 槽位下标以 QNUM 作为取模对象回到起点。
	#[test]
	fn test_push_avail_reuses_the_slot_after_a_full_ring() {
		let mut s = Store::new();
		for i in 0..QNUM {
			s.queue.push_avail(i);
		}
		for slot in 0..QNUM as usize {
			assert_eq!(s.avail_slot(slot), slot as u16);
		}
		assert_eq!(s.avail_index(), QNUM);

		s.queue.push_avail(0x55);
		assert_eq!(s.avail_slot(0), 0x55);
		assert_eq!(s.avail_index(), QNUM + 1);
	}

	/// 描述符表项的四个字段各自的偏移与字节序。
	#[test]
	fn test_write_desc_lays_out_the_four_fields() {
		let mut s = Store::new();
		let addr = 0x1122_3344_5566_7788_u64;
		let len = 0x99AA_BBCC_u32;
		s.queue.write_desc(3, addr, len, DESC_F_WRITE);

		let at = 3 * DESC_SIZE;
		let d = s.desc_bytes();
		assert_eq!(u64_at(d, at), addr);
		assert_eq!(u32_at(d, at + 8), len);
		assert_eq!(u16_at(d, at + 12), DESC_F_WRITE);
		assert_eq!(u16_at(d, at + 14), 0);
	}

	/// 写入一项描述符表项时, 其余项保持原值。
	#[test]
	fn test_write_desc_leaves_the_other_entries_untouched() {
		let mut s = Store::new();
		s.queue.write_desc(QNUM - 1, 0xAAAA_BBBB_CCCC_DDDD, 0x1234_5678, DESC_F_WRITE);

		let at = (QNUM as usize - 1) * DESC_SIZE;
		assert!(s.desc_bytes()[..at].iter().all(|&b| b == 0));
		assert_eq!(u64_at(s.desc_bytes(), at), 0xAAAA_BBBB_CCCC_DDDD);
	}

	/// 已用环形缓冲区的索引未推进时读到 None; 推进之后同一项可重复读到。
	#[test]
	fn test_peek_used_waits_for_the_used_index_to_advance() {
		let mut s = Store::new();
		assert!(s.queue.peek_used().is_none());

		s.device_push_used(2, 68);
		let first = s.queue.peek_used().unwrap();
		assert_eq!((first.id, first.len), (2, 68));

		// 本调用不推进已用环形缓冲区的索引。
		let again = s.queue.peek_used().unwrap();
		assert_eq!((again.id, again.len), (2, 68));

		s.queue.commit_used();
		assert!(s.queue.peek_used().is_none());
	}

	/// 已用环形缓冲区的项数超过项数上限时, 槽位下标以 QNUM 作为取模对象。
	#[test]
	fn test_peek_used_reads_the_slot_by_the_used_index() {
		let mut s = Store::new();
		for i in 0..QNUM + 3 {
			let (id, len) = (i as u32, i as u32 * 4);
			s.queue.push_avail(i as u16);
			s.device_push_used(id, len);
			let entry = s.queue.peek_used().unwrap();
			assert_eq!((entry.id, entry.len), (id, len));
			s.queue.commit_used();
		}
		assert!(s.queue.peek_used().is_none());
		assert_eq!(s.queue.space_left(), QNUM);
	}
}
