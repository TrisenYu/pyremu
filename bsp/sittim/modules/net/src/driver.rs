//! virtio-net 网卡驱动。
//!
//! 一条接收队列与一条发送队列。队列寄存器经 mmio 模块读写, 描述符表、可用环形缓冲区
//! 与已用环形缓冲区由 virt_queue 模块读写。网卡的 MMIO 基地址由运行时经管理器给出。

use core::ptr::{addr_of, addr_of_mut};

use crate::mmio;
use crate::platform;
use crate::virt_queue::{DESC_F_WRITE, QNUM, Queue, QueueStorage};

/// 接收队列的索引。
const QUEUE_RX: u32 = 0;
/// 发送队列的索引。
const QUEUE_TX: u32 = 1;

/// 每个缓冲区起始处的 virtio-net 首部长度。
pub(crate) const VNET_HDR: usize = 10;
/// 以太帧的最大长度: 14 字节首部与 1500 字节载荷之和。
pub(crate) const ETHERNET_FRAME_MAX: usize = 1514;
/// 单个缓冲区的长度。
const BUF_SIZE: usize = VNET_HDR + ETHERNET_FRAME_MAX;

/// 一条队列的缓冲区集合。
pub(crate) type Buffers = [[u8; BUF_SIZE]; QNUM as usize];

/// 驱动状态。
pub(crate) struct State {
	rx_queue: Queue,
	tx_queue: Queue,
	rx_storage: QueueStorage,
	tx_storage: QueueStorage,
	rx_bufs: Buffers,
	tx_bufs: Buffers,
	pub(crate) mac: [u8; 6],
	pub(crate) is_ready: bool,
}

impl State {
	const fn new() -> Self {
		Self {
			rx_queue: Queue::new(),
			tx_queue: Queue::new(),
			rx_storage: QueueStorage::new(),
			tx_storage: QueueStorage::new(),
			rx_bufs: [[0; BUF_SIZE]; QNUM as usize],
			tx_bufs: [[0; BUF_SIZE]; QNUM as usize],
			mac: [0; 6],
			is_ready: false,
		}
	}
}

/// 驱动状态。只在 S 模式内核态访问, 设备事件路径与系统调用路径都在 SIE = 0 下运行。
static mut STATE: State = State::new();

pub(crate) fn state() -> &'static mut State {
	unsafe { &mut *addr_of_mut!(STATE) }
}

/// 设备配置空间给出的 MAC 地址。未协商 VIRTIO_NET_F_MAC 时全零。
pub fn mac() -> [u8; 6] {
	state().mac
}

/// 按 virtio-mmio 的状态序列初始化设备并挂出全部接收缓冲区。返回是否初始化成功。
///
/// 设备已就绪时直接返回真, 不重复握手: 状态序列只在设备处于未初始化状态时有效。
pub fn init() -> bool {
	if state().is_ready {
		return true;
	}
	if !mmio::is_present() {
		return false;
	}
	if mmio::read32(mmio::R_MAGIC) != mmio::MAGIC_VALUE
		|| mmio::read32(mmio::R_VERSION) != mmio::VERSION_MODERN
		|| mmio::read32(mmio::R_DEVICE_ID) != mmio::DEVICE_ID_NET
	{
		return false;
	}

	mmio::write32(mmio::R_STATUS, mmio::STATUS_ACKNOWLEDGE);
	mmio::write32(mmio::R_STATUS, mmio::STATUS_ACKNOWLEDGE | mmio::STATUS_DRIVER);

	mmio::write32(mmio::R_DEVICE_FEATURES_SEL, 0);
	let low = mmio::read32(mmio::R_DEVICE_FEATURES) as u64;
	mmio::write32(mmio::R_DEVICE_FEATURES_SEL, 1);
	let features = low | ((mmio::read32(mmio::R_DEVICE_FEATURES) as u64) << 32);
	if features & mmio::FEATURE_VERSION_1 == 0 {
		return false;
	}
	let negotiated = mmio::FEATURE_VERSION_1 | (features & mmio::FEATURE_NET_MAC);
	mmio::write32(mmio::R_DRIVER_FEATURES_SEL, 0);
	mmio::write32(mmio::R_DRIVER_FEATURES, negotiated as u32);
	mmio::write32(mmio::R_DRIVER_FEATURES_SEL, 1);
	mmio::write32(mmio::R_DRIVER_FEATURES, (negotiated >> 32) as u32);

	mmio::write32(
		mmio::R_STATUS,
		mmio::STATUS_ACKNOWLEDGE | mmio::STATUS_DRIVER | mmio::STATUS_FEATURES_OK,
	);
	if mmio::read32(mmio::R_STATUS) & mmio::STATUS_FEATURES_OK == 0 {
		return false;
	}

	let st = state();
	let State { rx_queue, rx_storage, tx_queue, tx_storage, .. } = st;
	if !setup_queue(QUEUE_RX, rx_queue, rx_storage) {
		return false;
	}
	if !setup_queue(QUEUE_TX, tx_queue, tx_storage) {
		return false;
	}

	if features & mmio::FEATURE_NET_MAC != 0 {
		for i in 0..6 {
			st.mac[i] = mmio::read_config_u8(i as u64);
		}
	}

	mmio::write32(
		mmio::R_STATUS,
		mmio::STATUS_ACKNOWLEDGE
			| mmio::STATUS_DRIVER
			| mmio::STATUS_FEATURES_OK
			| mmio::STATUS_DRIVER_OK,
	);
	// 设备在 DRIVER_OK 之后才接受可用环形缓冲区的通知。
	let State { rx_queue, rx_bufs, .. } = st;
	for slot in 0..QNUM {
		post_rx_buffer(rx_queue, rx_bufs, slot);
	}
	mmio::write32(mmio::R_QUEUE_NOTIFY, QUEUE_RX);
	st.is_ready = true;
	true
}

/// 配置一条队列的项数与三处结构的物理地址, 并置其就绪位。
fn setup_queue(index: u32, queue: &mut Queue, storage: &mut QueueStorage) -> bool {
	mmio::write32(mmio::R_QUEUE_SEL, index);
	if mmio::read32(mmio::R_QUEUE_NUM_MAX) < QNUM as u32 {
		return false;
	}
	mmio::write32(mmio::R_QUEUE_NUM, QNUM as u32);
	queue.attach(storage);
	mmio::write64(mmio::R_QUEUE_DESC_LOW, mmio::R_QUEUE_DESC_HIGH, queue.desc_pa());
	mmio::write64(mmio::R_QUEUE_DRIVER_LOW, mmio::R_QUEUE_DRIVER_HIGH, queue.avail_pa());
	mmio::write64(mmio::R_QUEUE_DEVICE_LOW, mmio::R_QUEUE_DEVICE_HIGH, queue.used_pa());
	mmio::write32(mmio::R_QUEUE_READY, 1);
	true
}

/// 把一个接收缓冲区挂到接收队列的可用环形缓冲区。
fn post_rx_buffer(queue: &mut Queue, bufs: &Buffers, slot: u16) {
	let pa = platform::va_to_pa(addr_of!(bufs[slot as usize]) as u64);
	queue.write_desc(slot, pa, BUF_SIZE as u32, DESC_F_WRITE);
	queue.push_avail(slot);
}

/// 已用环形缓冲区中尚未取用的一项: 描述符索引与设备写入的字节数。
#[derive(Clone, Copy)]
pub struct RxSlot {
	pub(crate) slot: u16,
	pub(crate) len: usize,
}

/// 把该接收项的缓冲区交还接收队列, 使设备可再次向该缓冲区写入帧, 再向
/// `R_QUEUE_NOTIFY` 写入接收队列的下标, 随后推进已用环形缓冲区的索引。
pub(crate) fn finish_rx(queue: &mut Queue, bufs: &Buffers, item: RxSlot) {
	post_rx_buffer(queue, bufs, item.slot);
	mmio::write32(mmio::R_QUEUE_NOTIFY, QUEUE_RX);
	queue.commit_used();
}

/// 把槽位 `slot` 上长度为 `len` 的以太帧写入描述符并交还发送队列, 使设备可取出该帧
/// 并发出, 再向 `R_QUEUE_NOTIFY` 写入发送队列的下标。
pub(crate) fn push_tx(queue: &mut Queue, bufs: &mut Buffers, slot: u16, len: usize) {
	let pa = platform::va_to_pa(addr_of_mut!(bufs[slot as usize]) as u64);
	queue.write_desc(slot, pa, (VNET_HDR + len) as u32, 0);
	queue.push_avail(slot);
	mmio::write32(mmio::R_QUEUE_NOTIFY, QUEUE_TX);
}

/// 驱动对接收与发送两侧静态存储的借用。取用期间不得再读驱动状态。
pub struct Driver {
	pub(crate) rx_queue: &'static mut Queue,
	pub(crate) tx_queue: &'static mut Queue,
	pub(crate) rx_bufs: &'static mut Buffers,
	pub(crate) tx_bufs: &'static mut Buffers,
}

impl Driver {
	/// 取用驱动状态。驱动未就绪时返回 None。
	pub fn acquire() -> Option<Self> {
		let st = state();
		if !st.is_ready {
			return None;
		}
		let State { rx_queue, tx_queue, rx_bufs, tx_bufs, .. } = st;
		Some(Self { rx_queue, tx_queue, rx_bufs, tx_bufs })
	}

	/// 读取一个接收项, 不推进已用环形缓冲区的索引。已用环形缓冲区为空时返回 None。
	/// 设备给出的描述符索引越出缓冲区集合时丢弃该项; 该项不含以太帧时重挂该缓冲区后
	/// 丢弃该项。两种情况都继续读下一项。
	pub fn peek_rx(&mut self) -> Option<RxSlot> {
		loop {
			let entry = self.rx_queue.peek_used()?;
			let slot = entry.id as usize;
			let len = (entry.len as usize).min(BUF_SIZE);
			if slot >= QNUM as usize {
				// 描述符索引越出缓冲区集合, 无法重挂, 只丢弃该项。
				self.rx_queue.commit_used();
				continue;
			}
			if len <= VNET_HDR {
				// 该项不含以太帧, 重挂该缓冲区后丢弃该项。
				finish_rx(self.rx_queue, self.rx_bufs, RxSlot { slot: slot as u16, len });
				continue;
			}
			return Some(RxSlot { slot: slot as u16, len });
		}
	}

	/// 取用一个发送槽位。发送队列没有空闲槽位时返回 None。
	pub fn reserve_tx(&mut self) -> Option<u16> {
		// 回收设备已取走的槽位。
		while self.tx_queue.peek_used().is_some() {
			self.tx_queue.commit_used();
		}
		if self.tx_queue.space_left() == 0 {
			return None;
		}
		Some(self.tx_queue.next_slot())
	}
}

#[cfg(test)]
mod tests {
	use core::ptr::addr_of;

	use super::{BUF_SIZE, Buffers, DESC_F_WRITE, QNUM, Queue, QueueStorage, post_rx_buffer};
	use crate::virt_queue::DESC_SIZE;

	fn u16_at(at: usize) -> u16 {
		unsafe { core::ptr::read_unaligned((at) as *const u16) }
	}

	fn u32_at(at: usize) -> u32 {
		unsafe { core::ptr::read_unaligned((at) as *const u32) }
	}

	fn u64_at(at: usize) -> u64 {
		unsafe { core::ptr::read_unaligned((at) as *const u64) }
	}

	/// 挂在堆上的队列存储与缓冲区集合, 即描述符表项所指的地址在用例内保持稳定。
	struct Setup {
		storage: Box<QueueStorage>,
		bufs: Box<Buffers>,
		queue: Queue,
	}

	impl Setup {
		fn new() -> Self {
			Self {
				storage: Box::new(QueueStorage::new()),
				bufs: Box::new([[0u8; BUF_SIZE]; QNUM as usize]),
				queue: Queue::new(),
			}
		}

		fn attach(&mut self) -> &mut Self {
			self.queue.attach(&mut *self.storage);
			self
		}

		/// 可用环形缓冲区的索引。
		fn avail_index(&self) -> u16 {
			u16_at((self.queue.avail_pa() + 2) as usize)
		}

		/// 第 `slot` 项描述符表项内的四个字段。
		fn desc(&self, slot: usize) -> (u64, u32, u16, u16) {
			let at = (self.queue.desc_pa() + slot as u64 * DESC_SIZE as u64) as usize;
			(u64_at(at), u32_at(at + 8), u16_at(at + 12), u16_at(at + 14))
		}
	}

	/// 挂出一个接收缓冲区: 描述符指向该缓冲区的起始地址, 长度为整个缓冲区, 且标为
	/// 设备写入。
	#[test]
	fn test_post_rx_buffer_describes_the_buffer_and_publishes_it() {
		let mut s = Setup::new();
		s.attach();

		post_rx_buffer(&mut s.queue, &s.bufs, 3);

		assert_eq!(
			s.desc(3),
			(addr_of!(s.bufs[3]) as u64, BUF_SIZE as u32, DESC_F_WRITE, 0)
		);
		assert_eq!(s.avail_index(), 1);
		assert_eq!(u16_at((s.queue.avail_pa() + 4) as usize), 3);
	}

	/// 逐项挂出全部接收缓冲区之后, 可用环形缓冲区被填满, 各项指向各自的缓冲区。
	#[test]
	fn test_post_rx_buffer_fills_the_avail_ring() {
		let mut s = Setup::new();
		s.attach();

		for slot in 0..QNUM {
			post_rx_buffer(&mut s.queue, &s.bufs, slot);
		}

		assert_eq!(s.queue.space_left(), 0);
		assert_eq!(s.avail_index(), QNUM);
		for slot in 0..QNUM as usize {
			assert_eq!(s.desc(slot).0, addr_of!(s.bufs[slot]) as u64);
			assert_eq!(u16_at((s.queue.avail_pa() + 4 + slot as u64 * 2) as usize), slot as u16);
		}
	}

	/// 缓冲区集合占用的字节数等于单个缓冲区的长度乘以项数。
	#[test]
	fn test_the_buffer_set_holds_one_buffer_per_slot() {
		assert_eq!(core::mem::size_of::<Buffers>(), BUF_SIZE * QNUM as usize);
		assert_eq!(core::mem::align_of::<Buffers>(), 1);
	}

	/// 缓冲区未挂出时, 可用环形缓冲区为空。
	#[test]
	fn test_a_fresh_queue_has_nothing_published() {
		let mut s = Setup::new();
		s.attach();
		assert_eq!(s.avail_index(), 0);
		assert_eq!(s.queue.space_left(), QNUM);
	}
}
