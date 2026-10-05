//! 网络模块导出的操作表。
//!
//! 运行时与网络模块共用本文件: 运行时以 `mod net_ops` 引入, 模块以 `#[path]` 引入同一
//! 份定义, 两侧不可能漂移。
//!
//! 表内各项不是系统调用, 而是运行时自身的调用点 —— fd 层的套接字读写、调度器对协议栈
//! 的推进、运行时对套接字槽位状态的查询。形态与 ref-impl/emod 的 `emod_net_api_t`
//! 一致: 模块在接口中给出一张操作表, 而不是把这些能力混进系统调用处理函数表。
//!
//! 跨边界的数据只有三类: 计数、端点与缓冲区指针。端点的地址按 [`NetEndpoint`] 的
//! 定长缓冲区承载, 其宽度按最长的地址族给出, 故本表不随地址族变化; 缓冲区指针取运行时
//! 与模块共同的虚拟地址空间, 二者同处一个地址空间, 故本表不出现协议栈自身的类型。

/// 端点结构体中地址缓冲区的字节数, 按最长的地址族给出。
pub const NET_ADDR_MAX: usize = 16;

/// 地址族未指定。端点取该族时地址缓冲区无意义, 表示本地地址不作限定。
pub const NET_AF_UNSPEC: u16 = 0;
/// 地址族为 IPv4, 取 AF_INET 的编号, 与载荷侧的取值一致, 无需折算。
pub const NET_AF_INET: u16 = 2;

/// 绑定成功。
pub const NET_BIND_OK: u64 = 0;
/// 协议栈尚未建立。
pub const NET_BIND_NOT_READY: u64 = 1;
/// 套接字已绑定。
pub const NET_BIND_ALREADY_BOUND: u64 = 2;
/// 请求的端口已被占用。
pub const NET_BIND_PORT_IN_USE: u64 = 3;
/// 没有空闲的临时端口可分配。
pub const NET_BIND_NO_FREE_PORT: u64 = 4;

/// 无空闲套接字槽位时 [`NetOps::open`] 的返回值。
pub const NET_NO_SLOT: u64 = u64::MAX;

/// 一个网络端点: 地址族, 地址字节与端口。
///
/// 地址缓冲区的宽度按最长的地址族给出, 使用哪一地址族由 `family` 决定, 超出该族宽度的
/// 字节由填写方置 0、读取方忽略。端口按主机字节序给出。
#[repr(C)]
#[derive(Clone, Copy)]
pub struct NetEndpoint {
	pub family: u16,
	pub addr: [u8; NET_ADDR_MAX],
	pub port: u16,
}

impl NetEndpoint {
	/// 取 IPv4 地址的四个字节; 地址族不是 IPv4 时返回 None。
	pub fn octets_v4(&self) -> Option<[u8; 4]> {
		if self.family != NET_AF_INET {
			return None;
		}
		let mut octets = [0u8; 4];
		octets.copy_from_slice(&self.addr[..4]);
		Some(octets)
	}

	/// 由 IPv4 地址的四个字节与端口建立端点。
	pub fn from_v4(octets: [u8; 4], port: u16) -> Self {
		let mut addr = [0u8; NET_ADDR_MAX];
		addr[..4].copy_from_slice(&octets);
		Self {
			family: NET_AF_INET,
			addr,
			port,
		}
	}
}

/// 网络模块导出的操作表。
///
/// 表由模块在取入时填写, 且必须在模块的入口内填写而不能作为静态初始化项: 静态初始化
/// 会把函数地址作为绝对常量写进映像, 而映像在链接时不知道自己的加载基址。
#[repr(C)]
pub struct NetOps {
	/// 协议栈的套接字槽位数。运行时的接收等待地址按本项逐槽位折算。
	pub socket_slots: u64,
	/// 单个槽位一次可收发的最大字节数。
	pub payload_bytes: u64,
	/// 协议栈的本机地址, 按 [`NetEndpoint`] 承载。
	pub local_addr: NetEndpoint,
	/// 本机地址的掩码前缀长度, 单位为位。
	pub local_prefix_len: u8,

	/// 协议栈是否已建立。
	pub is_ready: unsafe extern "C" fn() -> bool,
	/// 建立协议栈。已建立时返回真。
	pub init: unsafe extern "C" fn() -> bool,
	/// 推进协议栈一次。
	pub poll: unsafe extern "C" fn(),
	/// 取一个空闲槽位; 无空闲槽位时返回 [`NET_NO_SLOT`]。
	pub open: unsafe extern "C" fn() -> u64,
	/// 释放一个槽位。
	pub close: unsafe extern "C" fn(u64),
	/// 该槽位是否已被占用。
	pub is_used: unsafe extern "C" fn(u64) -> bool,
	/// 绑定本地地址与端口。端点的地址族取 `NET_AF_UNSPEC` 表示本地地址不作限定; 端口
	/// 取 0 表示由协议栈分配一个临时端口。返回值为 `NET_BIND_*` 之一。
	pub bind: unsafe extern "C" fn(u64, *const NetEndpoint) -> u64,
	/// 记录对端地址; 失败返回假。
	pub connect: unsafe extern "C" fn(u64, *const NetEndpoint) -> bool,
	/// 取已记录的对端地址。返回假时输出参数不被写入。
	pub peer: unsafe extern "C" fn(u64, *mut NetEndpoint) -> bool,
	/// 清除已记录的对端地址。
	pub clear_peer: unsafe extern "C" fn(u64),
	/// 取本地地址与端口; 未绑定时给出 `NET_AF_UNSPEC` 的端点。
	pub local_endpoint: unsafe extern "C" fn(u64, *mut NetEndpoint),
	/// 把 [buf, buf + len) 作为一个报文发往给定端点; 发送缓冲区无空间时返回假。
	pub send: unsafe extern "C" fn(u64, *const NetEndpoint, *const u8, u64) -> bool,
	/// 是否有报文可接收。
	pub can_recv: unsafe extern "C" fn(u64) -> bool,
	/// 是否可以发送报文。
	pub can_send: unsafe extern "C" fn(u64) -> bool,
	/// 取一个报文写入 [buf, buf + len), 返回写入的字节数并写出报文来源; 无报文可接收
	/// 时返回 -1。报文长于缓冲区时按缓冲区长度截断, 其余字节丢弃; 来源指针可以取空,
	/// 表示调用方不需要来源。
	pub recv: unsafe extern "C" fn(u64, *mut u8, u64, *mut NetEndpoint) -> i64,
	/// 与 [`NetOps::recv`] 相同, 但不把报文从接收队列中移除。
	pub peek: unsafe extern "C" fn(u64, *mut u8, u64, *mut NetEndpoint) -> i64,
}

// ---------------------------------------------------------------
//  测试
// ---------------------------------------------------------------

#[cfg(test)]
mod tests {
	use super::*;

	/// 操作表的对齐由其中的函数指针决定, 加载器按该对齐检查表的落点。
	#[test]
	fn test_net_ops_is_pointer_aligned() {
		assert_eq!(core::mem::align_of::<NetOps>(), 8);
		assert_eq!(core::mem::size_of::<NetOps>() % 8, 0);
	}

	/// 地址缓冲区的宽度不短于 IPv4 地址, 端点结构的地址字段按该宽度定长。
	#[test]
	fn test_endpoint_address_buffer_holds_the_longest_family() {
		assert_eq!(NET_ADDR_MAX, 16);
		assert!(NET_ADDR_MAX >= core::mem::size_of::<u32>());
		assert_eq!(core::mem::size_of::<NetEndpoint>(), 20);
	}

	/// IPv4 端点的建立与读取往返, 其余字节置 0; 地址族不匹配时读取失败。
	#[test]
	fn test_endpoint_round_trips_ipv4_octets() {
		let endpoint = NetEndpoint::from_v4([10, 0, 0, 2], 0x1234);
		assert_eq!(endpoint.family, NET_AF_INET);
		assert_eq!(endpoint.octets_v4(), Some([10, 0, 0, 2]));
		assert_eq!(&endpoint.addr[4..], &[0u8; NET_ADDR_MAX - 4]);

		let mut unspecified = endpoint;
		unspecified.family = NET_AF_UNSPEC;
		assert_eq!(unspecified.octets_v4(), None);
	}
}
