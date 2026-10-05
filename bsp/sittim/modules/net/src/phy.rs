//! 协议栈与网卡之间的设备适配层, 即 phy::Device 的实现。
//!
//! Device::receive 同时给出 RxToken 与 TxToken: 前者借用设备写入的接收缓冲区, 后者
//! 借用待交还发送队列的发送缓冲区, 两个借用分别在自己的 consume 中归还。

use smoltcp::phy::{Device, DeviceCapabilities, Medium, RxToken, TxToken};
use smoltcp::time::Instant;

use crate::driver::{
	Buffers, Driver, ETHERNET_FRAME_MAX, RxSlot, VNET_HDR, finish_rx, push_tx,
};
use crate::virt_queue::{QNUM, Queue};

/// 网卡的 phy::Device 实现。
pub struct NetDevice {
	drv: Driver,
}

impl NetDevice {
	pub fn new(drv: Driver) -> Self {
		Self { drv }
	}
}

impl Device for NetDevice {
	type RxToken<'b>
		= Rx<'b>
	where
		Self: 'b;
	type TxToken<'b>
		= Tx<'b>
	where
		Self: 'b;

	fn receive(&mut self, _timestamp: Instant) -> Option<(Rx<'_>, Tx<'_>)> {
		let drv = &mut self.drv;
		let item = drv.peek_rx()?;
		let slot = drv.reserve_tx()?;
		let Driver { rx_queue, tx_queue, rx_bufs, tx_bufs } = drv;
		Some((
			Rx { queue: &mut **rx_queue, bufs: &**rx_bufs, item },
			Tx { queue: &mut **tx_queue, bufs: &mut **tx_bufs, slot },
		))
	}

	fn transmit(&mut self, _timestamp: Instant) -> Option<Tx<'_>> {
		let drv = &mut self.drv;
		let slot = drv.reserve_tx()?;
		let Driver { tx_queue, tx_bufs, .. } = drv;
		Some(Tx { queue: &mut **tx_queue, bufs: &mut **tx_bufs, slot })
	}

	fn capabilities(&self) -> DeviceCapabilities {
		let mut caps = DeviceCapabilities::default();
		caps.medium = Medium::Ethernet;
		caps.max_transmission_unit = ETHERNET_FRAME_MAX;
		caps.max_burst_size = Some(QNUM as usize);
		caps
	}
}

/// RxToken 的实现: consume 把设备写入的以太帧交给回调, 随后把该缓冲区交还接收队列。
pub struct Rx<'a> {
	queue: &'a mut Queue,
	bufs: &'a Buffers,
	item: RxSlot,
}

impl RxToken for Rx<'_> {
	fn consume<R, F>(self, f: F) -> R
	where
		F: FnOnce(&[u8]) -> R,
	{
		let Rx { queue, bufs, item } = self;
		let frame = &bufs[item.slot as usize][VNET_HDR..item.len];
		let out = f(frame);
		finish_rx(queue, bufs, item);
		out
	}
}

/// TxToken 的实现: consume 把回调写入的以太帧交还发送队列。
pub struct Tx<'a> {
	queue: &'a mut Queue,
	bufs: &'a mut Buffers,
	slot: u16,
}

impl TxToken for Tx<'_> {
	fn consume<R, F>(self, len: usize, f: F) -> R
	where
		F: FnOnce(&mut [u8]) -> R,
	{
		// 接口的 MTU 由 capabilities 声明为 ETHERNET_FRAME_MAX, 故 len 不超过该值。
		let n = len.min(ETHERNET_FRAME_MAX);
		let Tx { queue, bufs, slot } = self;
		let buf = &mut bufs[slot as usize];
		buf[..VNET_HDR].fill(0);
		let out = f(&mut buf[VNET_HDR..VNET_HDR + n]);
		push_tx(queue, bufs, slot, n);
		out
	}
}
