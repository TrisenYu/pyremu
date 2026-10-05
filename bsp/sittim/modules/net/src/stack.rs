//! 飞地内的 IPv4/UDP 协议栈。
//!
//! 接口与套接字集合以本模块的静态存储建立, 收发经 phy::NetDevice 接到网卡的描述符
//! 环。
//!
//! UDP 套接字按下标管理, 下标与静态存储中的缓冲区组一一对应。建栈时一次建满
//! UDP_SOCKETS 个套接字并记下各自的句柄, 关闭只是解绑并释放下标, 不增删集合成员,
//! 故下标、句柄与缓冲区三者的对应关系在栈的生命周期内不变。

use core::ptr::addr_of_mut;

use smoltcp::iface::{Config, Interface, SocketHandle, SocketSet, SocketStorage};
use smoltcp::socket::udp;
use smoltcp::time::Instant;
use smoltcp::wire::{
	EthernetAddress, HardwareAddress, IpAddress, IpCidr, IpEndpoint, IpListenEndpoint,
	Ipv4Address,
};

use crate::driver;
use crate::phy::NetDevice;
use crate::platform;

/// UDP 套接字的下标个数。
pub const UDP_SOCKETS: usize = 8;
/// 单个套接字单个方向上的报文元数据条数。
const META_SLOTS: usize = 4;
/// 单个套接字单个方向上的载荷字节数, 即该方向的缓冲区容量。
pub const PAYLOAD_BYTES: usize = META_SLOTS * 1600;

/// 本机 IPv4 地址。
pub const LOCAL_IP: Ipv4Address = Ipv4Address::new(10, 0, 0, 2);
/// 本机 IPv4 地址的前缀长度。
pub const LOCAL_PREFIX_LEN: u8 = 24;
/// 时间源每毫秒的计数。
fn ticks_per_ms() -> u64 {
	platform::time_freq() / 1000
}

/// 自动绑定本地端口时的起始端口, 取用过的端口自此向上搜索。
///
/// 取 0xC000 即 49152: 端口号按 65536 取模划分时, [0xC000, 0xFFFF] 恰是 IANA 划给
/// 动态分配的区间 (系统端口 0~0x3FF, 用户端口 0x400~0xBFFF 不在其内), 故自动绑定的
/// 端口不会落在载荷显式绑定的服务端口上。区间长度 0x4000 即 16384 个端口。
const EPHEMERAL_PORT_BASE: u16 = 0xC000;

/// 一个 UDP 套接字所占的缓冲区。
struct BufferSet {
	meta_rx: [udp::PacketMetadata; META_SLOTS],
	payload_rx: [u8; PAYLOAD_BYTES],
	meta_tx: [udp::PacketMetadata; META_SLOTS],
	payload_tx: [u8; PAYLOAD_BYTES],
}

impl BufferSet {
	const fn new() -> Self {
		Self {
			meta_rx: [udp::PacketMetadata::EMPTY; META_SLOTS],
			payload_rx: [0; PAYLOAD_BYTES],
			meta_tx: [udp::PacketMetadata::EMPTY; META_SLOTS],
			payload_tx: [0; PAYLOAD_BYTES],
		}
	}
}

/// 协议栈的静态存储。只在 init 内取用一次, 各字段的借用随套接字与套接字集合长期存在。
struct Storage {
	buf_sets: [BufferSet; UDP_SOCKETS],
	socket_storage: [SocketStorage<'static>; UDP_SOCKETS],
}

impl Storage {
	const fn new() -> Self {
		Self {
			buf_sets: [const { BufferSet::new() }; UDP_SOCKETS],
			socket_storage: [SocketStorage::EMPTY; UDP_SOCKETS],
		}
	}
}

/// 接口、套接字集合与各下标的取用状态。
struct Stack {
	iface: Interface,
	sockets: SocketSet<'static>,
	handles: [SocketHandle; UDP_SOCKETS],
	is_used: [bool; UDP_SOCKETS],
	/// 各下标经 connect 记录的对端地址。
	peer: [Option<IpEndpoint>; UDP_SOCKETS],
	/// 下一次自动绑定使用的临时端口。
	next_ephemeral_port: u16,
}

static mut STORAGE: Storage = Storage::new();
static mut STACK: Option<Stack> = None;

fn stack() -> Option<&'static mut Stack> {
	unsafe { (*addr_of_mut!(STACK)).as_mut() }
}

/// 协议栈是否已建立。
pub fn is_ready() -> bool {
	stack().is_some()
}

/// 读取当前时刻, 毫秒为单位。
fn now() -> Instant {
	Instant::from_millis((platform::read_time() / ticks_per_ms()) as i64)
}

/// 取得下标对应的套接字。
fn socket(st: &mut Stack, index: usize) -> &mut udp::Socket<'static> {
	st.sockets.get_mut::<udp::Socket>(st.handles[index])
}

/// 建立接口与套接字集合。网卡未就绪, 或设备未给出 MAC 地址时返回 false。
pub fn init() -> bool {
	if is_ready() {
		return true;
	}
	// 取设备寄存器之前先完成 virtio-mmio 握手: 驱动的接收与发送两侧都要求设备已就绪。
	if !driver::init() {
		return false;
	}
	let mac = driver::mac();
	if mac == [0; 6] {
		return false;
	}
	let Some(drv) = driver::Driver::acquire() else {
		return false;
	};
	let mut device = NetDevice::new(drv);

	let mut config = Config::new(HardwareAddress::Ethernet(EthernetAddress(mac)));
	config.random_seed = 0x5EED_1234;
	let mut iface = Interface::new(config, &mut device, now());
	iface.update_ip_addrs(|addrs| {
		let _ = addrs.push(IpCidr::new(IpAddress::Ipv4(LOCAL_IP), LOCAL_PREFIX_LEN));
	});

	let storage: &'static mut Storage = unsafe { &mut *addr_of_mut!(STORAGE) };
	let Storage { buf_sets, socket_storage } = storage;
	let mut sockets = SocketSet::new(&mut socket_storage[..]);
	let mut handles = [SocketHandle::default(); UDP_SOCKETS];
	for (bufs, handle) in buf_sets.iter_mut().zip(handles.iter_mut()) {
		let sock = udp::Socket::new(
			udp::PacketBuffer::new(&mut bufs.meta_rx[..], &mut bufs.payload_rx[..]),
			udp::PacketBuffer::new(&mut bufs.meta_tx[..], &mut bufs.payload_tx[..]),
		);
		*handle = sockets.add(sock);
	}

	let st = Stack {
		iface,
		sockets,
		handles,
		is_used: [false; UDP_SOCKETS],
		peer: [None; UDP_SOCKETS],
		next_ephemeral_port: EPHEMERAL_PORT_BASE,
	};
	unsafe { *addr_of_mut!(STACK) = Some(st) };
	true
}

/// 推进协议栈一次: 收取已到达的帧, 发出已排队的帧。
pub fn poll() {
	let Some(st) = stack() else {
		return;
	};
	let Some(drv) = driver::Driver::acquire() else {
		return;
	};
	let mut device = NetDevice::new(drv);
	st.iface.poll(now(), &mut device, &mut st.sockets);
}

/// 取用一个 UDP 套接字下标。无空闲下标时返回 None。
pub fn open() -> Option<usize> {
	let st = stack()?;
	let index = st.is_used.iter().position(|is_used| !is_used)?;
	st.is_used[index] = true;
	st.peer[index] = None;
	Some(index)
}

/// 释放一个 UDP 套接字下标, 并解绑该套接字的本地端口。
pub fn close(index: usize) {
	let Some(st) = stack() else {
		return;
	};
	st.is_used[index] = false;
	st.peer[index] = None;
	socket(st, index).close();
}

/// 该下标是否已取用。
pub fn is_used(index: usize) -> bool {
	stack().is_some_and(|st| st.is_used[index])
}

/// 绑定本地端口的结果。
pub enum BindError {
	/// 协议栈未建立。
	NotReady,
	/// 该套接字已绑定。
	AlreadyBound,
	/// 端口已被其它已取用的套接字占用。
	PortInUse,
	/// 自动分配端口时无空闲端口。
	NoFreePort,
}

/// 绑定本地端点。*addr* 为 None 时本地地址取未指定, 即接受发往本机任一地址的报文;
/// *port* 为 0 时绑定一个临时端口。
pub fn bind(index: usize, addr: Option<Ipv4Address>, port: u16) -> Result<(), BindError> {
	let Some(st) = stack() else {
		return Err(BindError::NotReady);
	};
	if socket(st, index).is_open() {
		return Err(BindError::AlreadyBound);
	}
	if port == 0 {
		let port = alloc_ephemeral_port(st).ok_or(BindError::NoFreePort)?;
		return bind_endpoint(st, index, addr, port).map_err(|_| BindError::NoFreePort);
	}
	if is_port_in_use(st, index, port) {
		return Err(BindError::PortInUse);
	}
	bind_endpoint(st, index, addr, port).map_err(|_| BindError::PortInUse)
}

/// 把本地端点写入套接字。*addr* 为 None 时本地地址取未指定。
fn bind_endpoint(
	st: &mut Stack, index: usize, addr: Option<Ipv4Address>, port: u16,
) -> Result<(), BindError> {
	let local = IpListenEndpoint { addr: addr.map(IpAddress::Ipv4), port };
	socket(st, index).bind(local).map_err(|_| BindError::PortInUse)
}

/// 记录对端地址, 供后续的发送与 getpeername 使用。
/// 套接字尚未绑定本地端口时先绑定一个临时端口。
pub fn connect(index: usize, remote: IpEndpoint) -> bool {
	let Some(st) = stack() else {
		return false;
	};
	if !socket(st, index).is_open() && bind_ephemeral(st, index).is_err() {
		return false;
	}
	st.peer[index] = Some(remote);
	true
}

/// 经 connect 记录的对端地址。未记录时为 None。
pub fn peer(index: usize) -> Option<IpEndpoint> {
	let st = stack()?;
	st.peer[index]
}

/// 清除经 connect 记录的对端地址。
pub fn clear_peer(index: usize) {
	if let Some(st) = stack() {
		st.peer[index] = None;
	}
}

/// 为套接字绑定一个临时本地端口。无可用端口时返回错误。
fn bind_ephemeral(st: &mut Stack, index: usize) -> Result<(), BindError> {
	let port = alloc_ephemeral_port(st).ok_or(BindError::NoFreePort)?;
	let local = IpEndpoint::new(IpAddress::Ipv4(LOCAL_IP), port);
	socket(st, index).bind(local).map_err(|_| BindError::NoFreePort)
}

/// 端口是否已被下标 *index* 之外的某个已取用套接字绑定。
fn is_port_in_use(st: &Stack, index: usize, port: u16) -> bool {
	(0..UDP_SOCKETS).any(|other| {
		let endpoint = st.sockets.get::<udp::Socket>(st.handles[other]).endpoint();
		other != index && st.is_used[other] && endpoint.port == port
	})
}

/// 取一个未被其它已取用套接字占用的临时端口。
fn alloc_ephemeral_port(st: &mut Stack) -> Option<u16> {
	let start = st.next_ephemeral_port;
	let mut port = start;
	loop {
		if port < EPHEMERAL_PORT_BASE {
			port = EPHEMERAL_PORT_BASE;
		}
		let mut is_taken = false;
		for index in 0..UDP_SOCKETS {
			if st.is_used[index] && socket(st, index).endpoint().port == port {
				is_taken = true;
				break;
			}
		}
		if !is_taken {
			st.next_ephemeral_port = port.wrapping_add(1);
			return Some(port);
		}
		port = port.wrapping_add(1);
		if port == start {
			return None;
		}
	}
}

/// 该套接字绑定的本地端点。未绑定时地址为未指定, 端口为 0。
pub fn local_endpoint(index: usize) -> IpListenEndpoint {
	let Some(st) = stack() else {
		return IpListenEndpoint { addr: None, port: 0 };
	};
	socket(st, index).endpoint()
}

/// 向 `remote` 发送一个报文。套接字不可用或发送缓冲区无空间时返回 false。
/// 套接字尚未绑定本地端口时先绑定一个临时端口。
pub fn send(index: usize, remote: IpEndpoint, data: &[u8]) -> bool {
	let Some(st) = stack() else {
		return false;
	};
	if !socket(st, index).is_open() && bind_ephemeral(st, index).is_err() {
		return false;
	}
	socket(st, index).send_slice(data, remote).is_ok()
}

/// 该套接字是否有已到达的报文。
pub fn can_recv(index: usize) -> bool {
	let Some(st) = stack() else {
		return false;
	};
	socket(st, index).can_recv()
}

/// 该套接字的发送缓冲区是否有空间。
pub fn can_send(index: usize) -> bool {
	let Some(st) = stack() else {
		return false;
	};
	socket(st, index).can_send()
}

/// 取走一个报文, 把载荷与对端地址交给 `sink`。
pub fn recv<R>(index: usize, sink: impl FnOnce(&[u8], IpEndpoint) -> R) -> Option<R> {
	let st = stack()?;
	let (data, meta) = socket(st, index).recv().ok()?;
	Some(sink(data, meta.endpoint))
}

/// 读取一个报文, 把载荷与对端地址交给 `sink`。该报文留在接收队列中, 可由后续的
/// 接收再次取用。
pub fn peek<R>(index: usize, sink: impl FnOnce(&[u8], IpEndpoint) -> R) -> Option<R> {
	let st = stack()?;
	let (data, meta) = socket(st, index).peek().ok()?;
	Some(sink(data, meta.endpoint))
}

#[cfg(test)]
mod tests {
	use std::sync::{Mutex, MutexGuard};

	use smoltcp::wire::{IpAddress, IpEndpoint, Ipv4Address};

	use super::{
		BindError, EPHEMERAL_PORT_BASE, LOCAL_IP, META_SLOTS, PAYLOAD_BYTES, UDP_SOCKETS, bind,
		can_recv, can_send, clear_peer, close, connect, init, is_ready, is_used, local_endpoint,
		open, peer, recv, send,
	};

	/// 协议栈与网卡驱动状态都是全局量, 用例依次取用。
	static LOCK: Mutex<()> = Mutex::new(());
	/// 设备配置空间给出的 MAC 地址。
	const MAC: [u8; 6] = [0x02, 0x00, 0x00, 0x00, 0x00, 0x01];

	/// 建立协议栈, 并释放上一个用例占用的下标与临时端口。
	///
	/// 主机上没有设备可握手, 故直接把驱动状态置为就绪并给出 MAC 地址; 协议栈据此建立,
	/// 不再走 virtio-mmio 状态序列。
	fn setup() -> MutexGuard<'static, ()> {
		let guard = LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
		let driver = crate::driver::state();
		driver.mac = MAC;
		driver.is_ready = true;
		if !is_ready() {
			assert!(init(), "协议栈建立失败");
		}
		for index in 0..UDP_SOCKETS {
			close(index);
		}
		super::stack().unwrap().next_ephemeral_port = EPHEMERAL_PORT_BASE;
		guard
	}

	fn remote(port: u16) -> IpEndpoint {
		IpEndpoint::new(IpAddress::Ipv4(Ipv4Address::new(10, 0, 0, 1)), port)
	}

	/// 取用下标时取编号最小的空闲下标, 下标用尽后返回 None, 释放后该下标重新可用。
	#[test]
	fn test_open_takes_the_lowest_free_index() {
		let _guard = setup();
		for index in 0..UDP_SOCKETS {
			assert_eq!(open(), Some(index));
			assert!(is_used(index));
		}
		assert_eq!(open(), None);

		close(3);
		assert!(!is_used(3));
		assert_eq!(open(), Some(3));
	}

	/// close 释放下标并清除该下标记录的对端地址。
	#[test]
	fn test_close_releases_the_index_and_the_peer() {
		let _guard = setup();
		let index = open().unwrap();
		assert!(connect(index, remote(5000)));
		assert_eq!(peer(index), Some(remote(5000)));

		close(index);
		assert!(!is_used(index));
		assert_eq!(peer(index), None);
	}

	/// 端口取 0 时绑定第一个临时端口, 本地地址取未指定。
	#[test]
	fn test_bind_zero_port_takes_the_first_ephemeral_port() {
		let _guard = setup();
		let index = open().unwrap();
		assert!(matches!(bind(index, None, 0), Ok(())));
		assert_eq!(local_endpoint(index).port, EPHEMERAL_PORT_BASE);
		assert_eq!(local_endpoint(index).addr, None);
	}

	/// 已绑定的套接字再次绑定时报已绑定, 与端口是否被占用无关。
	#[test]
	fn test_bind_twice_reports_already_bound() {
		let _guard = setup();
		let index = open().unwrap();
		assert!(matches!(bind(index, None, 0x1234), Ok(())));
		assert!(matches!(bind(index, None, 0x1234), Err(BindError::AlreadyBound)));
		assert!(matches!(bind(index, None, 0), Err(BindError::AlreadyBound)));
	}

	/// 端口已被另一个已取用的套接字绑定时报端口占用。
	#[test]
	fn test_bind_a_port_held_by_another_socket_reports_port_in_use() {
		let _guard = setup();
		let first = open().unwrap();
		let second = open().unwrap();
		assert!(matches!(bind(first, None, 0x1234), Ok(())));
		assert!(matches!(bind(second, None, 0x1234), Err(BindError::PortInUse)));
	}

	/// 同一端口在未被取用的下标上可以绑定。
	#[test]
	fn test_bind_a_port_held_by_a_released_socket_succeeds() {
		let _guard = setup();
		let first = open().unwrap();
		assert!(matches!(bind(first, None, 0x1234), Ok(())));
		close(first);

		let second = open().unwrap();
		assert_eq!(second, first);
		assert!(matches!(bind(second, None, 0x1234), Ok(())));
		assert_eq!(local_endpoint(second).port, 0x1234);
	}

	/// 自动取端口时跳过已被占用的端口。
	#[test]
	fn test_ephemeral_port_search_skips_a_taken_port() {
		let _guard = setup();
		let first = open().unwrap();
		let second = open().unwrap();
		assert!(matches!(bind(first, None, EPHEMERAL_PORT_BASE), Ok(())));

		assert!(matches!(bind(second, None, 0), Ok(())));
		assert_eq!(local_endpoint(second).port, EPHEMERAL_PORT_BASE + 1);
	}

	/// 释放端口之后不从该端口重新取, 而是接着上一次的位置向上搜索。
	#[test]
	fn test_a_released_port_is_not_taken_again() {
		let _guard = setup();
		let index = open().unwrap();
		assert!(matches!(bind(index, None, 0), Ok(())));
		let first = local_endpoint(index).port;
		close(index);

		assert!(matches!(bind(index, None, 0), Ok(())));
		assert_eq!(local_endpoint(index).port, first + 1);
	}

	/// 显式给出的本地地址与端口被记下。
	#[test]
	fn test_bind_records_the_given_address_and_port() {
		let _guard = setup();
		let index = open().unwrap();
		assert!(matches!(bind(index, Some(LOCAL_IP), 0x2000), Ok(())));
		assert_eq!(local_endpoint(index).addr, Some(IpAddress::Ipv4(LOCAL_IP)));
		assert_eq!(local_endpoint(index).port, 0x2000);
	}

	/// connect 记下对端地址, 并在此之前绑定本机地址与一个临时端口。
	#[test]
	fn test_connect_records_the_peer_and_binds_the_local_endpoint() {
		let _guard = setup();
		let index = open().unwrap();
		assert!(connect(index, remote(6000)));

		assert_eq!(peer(index), Some(remote(6000)));
		assert_eq!(local_endpoint(index).addr, Some(IpAddress::Ipv4(LOCAL_IP)));
		assert_eq!(local_endpoint(index).port, EPHEMERAL_PORT_BASE);
	}

	/// clear_peer 只清除对端地址, 本地端点与下标保持已取用。
	#[test]
	fn test_clear_peer_keeps_the_local_endpoint() {
		let _guard = setup();
		let index = open().unwrap();
		assert!(connect(index, remote(6000)));

		clear_peer(index);
		assert_eq!(peer(index), None);
		assert!(is_used(index));
		assert_eq!(local_endpoint(index).addr, Some(IpAddress::Ipv4(LOCAL_IP)));
	}

	/// 无报文到达时接收侧恒为空; 发送缓冲区被填满后发送被拒, 且不再报告可发送。
	#[test]
	fn test_send_and_receive_status_follow_the_socket() {
		let _guard = setup();
		let index = open().unwrap();
		assert!(!can_recv(index));
		assert!(recv(index, |data, _| data.len()).is_none());

		let payload = [0u8; PAYLOAD_BYTES / META_SLOTS];
		let mut sent = 0;
		while sent < META_SLOTS && send(index, remote(7000), &payload) {
			sent += 1;
		}
		assert_eq!(sent, META_SLOTS);
		assert!(!can_send(index));
		assert!(!send(index, remote(7000), &payload));
		assert!(!can_recv(index));
	}

	/// 报文长于发送缓冲区容量时发送失败。
	#[test]
	fn test_send_reports_failure_when_the_payload_exceeds_the_buffer() {
		let _guard = setup();
		let index = open().unwrap();
		assert!(!send(index, remote(7000), &[0u8; PAYLOAD_BYTES + 1]));
		assert!(send(index, remote(7000), &[0u8; PAYLOAD_BYTES]));
	}
}
