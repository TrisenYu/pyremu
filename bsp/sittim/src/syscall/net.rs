//! AF_INET 的 SOCK_DGRAM 套接字系统调用。
//!
//! 套接字状态由网络模块的协议栈按套接字下标持有, 本模块做参数校验、sockaddr 的读写与
//! 接收的阻塞语义; 涉及协议栈的动作经该模块导出的操作表调用, 见
//! [net_ops](crate::ext_mod::net_ops) 与 [runtime](crate::ext_mod::runtime)。fd 表项
//! 只记该下标, 见 [fs](super::fs) 的 FdTarget。
//!
//! 接收的阻塞地址由套接字下标折算, 与 futex 的等待地址、wait4 的阻塞地址同属一类:
//! 只参与唤醒匹配, 内核从不解引用。报文到达时由 M 模式转发的接收事件唤醒, 见
//! [sched](super::concurrency::sched)。
//!
//! 网络模块未接入时 fd 表内不可能有套接字表项 (socket 调用本身就要求该模块), 故各处理
//! 函数在该情形下返回 ENODEV。

use core::cmp;
use core::ptr;

use crate::ext_mod::net_ops::{
	NetEndpoint,
	NetOps,
	NET_ADDR_MAX,
	NET_AF_UNSPEC,
	NET_BIND_ALREADY_BOUND,
	NET_BIND_NO_FREE_PORT,
	NET_BIND_NOT_READY,
	NET_BIND_OK,
	NET_BIND_PORT_IN_USE,
	NET_NO_SLOT,
};
use crate::ext_mod::runtime;

use super::concurrency::thread;
use super::fdtable::{ FdTarget, alloc_socket_fd, fd_entry };
use super::fs::O_NONBLOCK;
use super::types::{ MsgHdr, SockaddrIn };
use super::{
	EADDRINUSE,
	EADDRNOTAVAIL,
	EAFNOSUPPORT,
	EAGAIN,
	EBADF,
	EDESTADDRREQ,
	EFAULT,
	EINVAL,
	EMFILE,
	EMSGSIZE,
	ENETUNREACH,
	ENODEV,
	ENOPROTOOPT,
	ENOTCONN,
	ENOTSOCK,
	EOPNOTSUPP,
	EPROTONOSUPPORT,
};

/// AF_INET 的地址族编号。
const AF_INET: u16 = 2;
/// SOCK_DGRAM 的类型编号, 以及类型字段高位的两个标志编号。
const SOCK_DGRAM: u64 = 2;
const SOCK_NONBLOCK: u64 = 0o4000;
const SOCK_CLOEXEC: u64 = 0o2000000;
/// IPPROTO_UDP 的协议编号。
const IPPROTO_UDP: u64 = 17;
/// 收发标志中被接受的 MSG_DONTWAIT 与 MSG_PEEK 编号。
const MSG_DONTWAIT: u64 = 0x40;
const MSG_PEEK: u64 = 0x2;
/// 报文首部的 msg_flags 字段中表示报文被截断的 MSG_TRUNC 编号。
const MSG_TRUNC: i32 = 0x20;
/// struct sockaddr_in 的字节数。
const SOCKADDR_IN_SIZE: usize = 16;
/// 一个 UDP 报文载荷的最大字节数 = 以太帧最大长度 1514 减去以太网首部 14、
/// IPv4 首部 20 与 UDP 首部 8。
pub const UDP_PAYLOAD_MAX: usize = 1472;
/// iovec 数组的最大项数, 与 Linux 的 UIO_MAXIOV 一致。
const IOV_MAX: u64 = 1024;

// setsockopt 与 getsockopt 的层与选项编号。
const SOL_SOCKET: u64 = 1;
const SO_TYPE: u64 = 3;
const SO_ERROR: u64 = 4;
const SO_SNDBUF: u64 = 7;
const SO_RCVBUF: u64 = 8;
const SO_DOMAIN: u64 = 39;

// shutdown 的方向编号。
const SHUT_RD: u64 = 0;
const SHUT_RDWR: u64 = 2;

/// 接收的阻塞地址基址。每个套接字下标一个, 折算方式为基址加上下标乘以 8。
const RECV_WAIT_ADDR_BASE: u64 = 0x7fff_1000_0000;

/// 网络模块的操作表; 该模块未接入时返回 None。
///
/// 首次调用时取入该模块, 宿主未提供它时返回 None。
fn ops() -> Option<&'static NetOps> {
	runtime::acquire_net_ops()
}

/// 地址族未指定的端点: 绑定时表示本地地址不作限定, 接收时作为来源的输出初值。
fn unspecified_endpoint(port: u16) -> NetEndpoint {
	NetEndpoint {
		family: NET_AF_UNSPEC,
		addr: [0; NET_ADDR_MAX],
		port,
	}
}

/// 下标 *index* 的套接字的接收阻塞地址。
fn recv_wait_addr(index: u64) -> u64 {
	RECV_WAIT_ADDR_BASE + index * 8
}

/// 推进协议栈一次。
///
/// 发送之后与设备事件到达时调用, 见 [sched](super::concurrency::sched)。
pub fn poll_stack() {
	if let Some(ops) = ops() {
		unsafe { (ops.poll)() }
	}
}

/// 唤醒全部有报文可接收的套接字上阻塞的线程。
pub fn wake_receivers() {
	let Some(ops) = ops() else {
		return;
	};
	for index in 0..ops.socket_slots {
		if unsafe { (ops.is_used)(index) && (ops.can_recv)(index) } {
			thread::wake_blocked(recv_wait_addr(index), u64::MAX);
		}
	}
}

/// 该套接字槽位是否有报文可接收; 模块未接入时返回假。
///
/// ppoll 把套接字的就绪状态并入等待判据, 见 [fs](super::fs)。
pub fn can_recv(index: u64) -> bool {
	match ops() {
		Some(ops) => unsafe { (ops.can_recv)(index) }
		None => false,
	}
}

/// 该套接字槽位是否可以发送报文; 模块未接入时返回假。
pub fn can_send(index: u64) -> bool {
	match ops() {
		Some(ops) => unsafe { (ops.can_send)(index) }
		None => false,
	}
}

/// 释放一个套接字槽位。fd 表释放套接字表项时调用, 见 [fs](super::fs)。
pub fn close_socket(index: u64) {
	if let Some(ops) = ops() {
		unsafe { (ops.close)(index) }
	}
}

/// 取 fd 绑定的套接字下标与打开标志。fd 绑定控制台或文件系统节点时返回 ENOTSOCK,
/// 越界 fd 与空闲槽位返回 EBADF。
fn socket_index(fd: u64) -> Result<(u64, u32), u64> {
	match fd_entry(fd) {
		Some((FdTarget::Socket(index), flags)) => Ok((index, flags)),
		Some((FdTarget::Console, _)) | Some((FdTarget::Vfs(..), _)) => Err(ENOTSOCK),
		None => Err(EBADF),
	}
}

/// 取第 *index* 个套接字经 connect 记录的对端地址; 未记录时返回 None。
fn peer_of(ops: &NetOps, index: u64) -> Option<NetEndpoint> {
	let mut endpoint = unspecified_endpoint(0);
	if unsafe { (ops.peer)(index, &mut endpoint) } {
		Some(endpoint)
	} else {
		None
	}
}

/// 取第 *index* 个套接字的本机地址与端口。
fn local_endpoint_of(ops: &NetOps, index: u64) -> NetEndpoint {
	let mut endpoint = unspecified_endpoint(0);
	unsafe {
		(ops.local_endpoint)(index, &mut endpoint);
	}
	endpoint
}

/// 从载荷地址空间读出 sockaddr_in, 返回端点。
///
/// 地址为 0 时返回 EFAULT, 长度小于 struct sockaddr_in 时返回 EINVAL, 地址族不是
/// AF_INET 时返回 EAFNOSUPPORT。
fn read_sockaddr(addr: u64, addrlen: u64) -> Result<NetEndpoint, u64> {
	if addr == 0 {
		return Err(EFAULT);
	}
	if addrlen < (SOCKADDR_IN_SIZE as u64) {
		return Err(EINVAL);
	}
	let sa = unsafe { ptr::read_unaligned(addr as *const SockaddrIn) };
	if sa.sin_family != AF_INET {
		return Err(EAFNOSUPPORT);
	}
	// 载荷的 sockaddr_in 按网络字节序承载端口, 端点的端口按主机字节序给出, 换算在此
	// 完成, 操作表的两侧都不再换算。
	Ok(NetEndpoint::from_v4(sa.sin_addr, u16::from_be(sa.sin_port)))
}

/// 校验写入 sockaddr_in 的目标与容量。*addrlen* 指向载荷地址空间中的容量值。
///
/// 地址或容量指针为 0 时返回 EFAULT, 容量小于 struct sockaddr_in 时返回 EINVAL。
fn check_sockaddr_out(addr: u64, addrlen: u64) -> Result<(), u64> {
	if addr == 0 || addrlen == 0 {
		return Err(EFAULT);
	}
	let capacity = unsafe { ptr::read_unaligned(addrlen as *const u32) as u64 };
	if capacity < (SOCKADDR_IN_SIZE as u64) {
		return Err(EINVAL);
	}
	Ok(())
}

/// 端点对应的 sockaddr_in。地址族不是 IPv4 时地址取未指定地址, 即全 0。
fn sockaddr_in_of(endpoint: &NetEndpoint) -> SockaddrIn {
	SockaddrIn {
		sin_family: AF_INET,
		sin_port: endpoint.port.to_be(),
		sin_addr: endpoint.octets_v4().unwrap_or([0; 4]),
		sin_zero: [0; 8],
	}
}

/// 把 *endpoint* 写入载荷地址空间的 sockaddr_in, 并把容量值改写为
/// struct sockaddr_in 的字节数。调用前须经 check_sockaddr_out。
fn write_sockaddr(addr: u64, addrlen: u64, endpoint: &NetEndpoint) {
	unsafe {
		ptr::write_unaligned(addr as *mut SockaddrIn, sockaddr_in_of(endpoint));
		ptr::write_unaligned(addrlen as *mut u32, SOCKADDR_IN_SIZE as u32);
	}
}

/// 目的地址是否在本机所在的子网内。
///
/// 协议栈只配置了本机地址一条路由, 没有网关, 子网外的目的地址没有送达路径。本机地址与
/// 前缀长度都由操作表给出, 故按位比较: 掩码取前缀位为 1、其余位为 0, 目的地址与本机
/// 地址各自与掩码按位与之后比较。前缀长度按 32 截断, 更长的前缀等同于全部 32 位。
fn is_local_subnet(ops: &NetOps, endpoint: &NetEndpoint) -> bool {
	let (Some(dst), Some(local)) = (endpoint.octets_v4(), ops.local_addr.octets_v4()) else {
		return false;
	};
	let prefix_bits = cmp::min(ops.local_prefix_len as u32, 32);
	let mask = if prefix_bits == 0 { 0 } else { u32::MAX << (32 - prefix_bits) };
	(u32::from_be_bytes(dst) & mask) == (u32::from_be_bytes(local) & mask)
}

/// 绑定失败对应的错误码; 绑定成功取 0。
fn bind_error(code: u64) -> u64 {
	match code {
		NET_BIND_OK => 0,
		NET_BIND_NOT_READY => ENODEV,
		NET_BIND_ALREADY_BOUND => EINVAL,
		NET_BIND_PORT_IN_USE | NET_BIND_NO_FREE_PORT => EADDRINUSE,
		_ => ENODEV,
	}
}

/// socket(198): 建立一个套接字。域只支持 AF_INET, 类型只支持 SOCK_DGRAM。
pub fn socket_handler(domain: u64, sock_type: u64, protocol: u64) -> u64 {
	if domain != (AF_INET as u64) {
		return EAFNOSUPPORT;
	}
	if (sock_type & !(SOCK_NONBLOCK | SOCK_CLOEXEC)) != SOCK_DGRAM {
		return EPROTONOSUPPORT;
	}
	if protocol != 0 && protocol != IPPROTO_UDP {
		return EPROTONOSUPPORT;
	}
	let Some(ops) = ops() else {
		return ENODEV;
	};
	let index = unsafe { (ops.open)() };
	if index == NET_NO_SLOT {
		return EMFILE;
	}
	// SOCK_NONBLOCK 属描述的状态标志; SOCK_CLOEXEC 与 open 的 O_CLOEXEC 取同一个位值,
	// 是描述符自身的标志, 按该位值原样转交。
	let flags = if (sock_type & SOCK_NONBLOCK) != 0 { O_NONBLOCK } else { 0 };
	let desc_flags = (sock_type & SOCK_CLOEXEC) as u32;
	let fd = alloc_socket_fd(index, flags, desc_flags);
	if fd == EMFILE {
		unsafe {
			(ops.close)(index);
		}
	}
	fd
}

/// bind(200): 绑定本地地址与端口。地址取未指定地址或本机地址, 取其余地址时返回
/// EADDRNOTAVAIL; 端口为 0 时由协议栈分配一个临时端口。
pub fn bind_handler(fd: u64, addr: u64, addrlen: u64) -> u64 {
	let index = match socket_index(fd) {
		Ok((index, _)) => index,
		Err(e) => {
			return e;
		}
	};
	let endpoint = match read_sockaddr(addr, addrlen) {
		Ok(endpoint) => endpoint,
		Err(e) => {
			return e;
		}
	};
	let Some(ops) = ops() else {
		return ENODEV;
	};
	// 未指定地址表示本地地址取任意, 本机地址表示只接受发往本机地址的报文; 其余地址
	// 没有接收路径, 协议栈只认本机地址的入站报文。
	let local = if endpoint.octets_v4() == Some([0, 0, 0, 0]) {
		unspecified_endpoint(endpoint.port)
	} else if endpoint.octets_v4() == ops.local_addr.octets_v4() {
		endpoint
	} else {
		return EADDRNOTAVAIL;
	};
	bind_error(unsafe { (ops.bind)(index, &local) })
}

/// connect(203): 记录对端地址。UDP 没有连接建立过程, 本调用只记录地址, 供未给出
/// 目的地址的发送与 getpeername 使用。
pub fn connect_handler(fd: u64, addr: u64, addrlen: u64) -> u64 {
	let index = match socket_index(fd) {
		Ok((index, _)) => index,
		Err(e) => {
			return e;
		}
	};
	let endpoint = match read_sockaddr(addr, addrlen) {
		Ok(endpoint) => endpoint,
		Err(e) => {
			return e;
		}
	};
	if endpoint.port == 0 {
		return EDESTADDRREQ;
	}
	let Some(ops) = ops() else {
		return ENODEV;
	};
	if !is_local_subnet(ops, &endpoint) {
		return ENETUNREACH;
	}
	// 记录对端前须绑定本地端口; 失败只可能是临时端口已用尽。
	if !(unsafe { (ops.connect)(index, &endpoint) }) {
		return EADDRINUSE;
	}
	0
}

/// getsockname(204): 读取本地地址与端口。套接字未绑定时地址为不确定地址, 端口为 0。
pub fn getsockname_handler(fd: u64, addr: u64, addrlen: u64) -> u64 {
	let index = match socket_index(fd) {
		Ok((index, _)) => index,
		Err(e) => {
			return e;
		}
	};
	if let Err(e) = check_sockaddr_out(addr, addrlen) {
		return e;
	}
	let Some(ops) = ops() else {
		return ENODEV;
	};
	write_sockaddr(addr, addrlen, &local_endpoint_of(ops, index));
	0
}

/// getpeername(205): 读取经 connect 记录的对端地址。未记录时返回 ENOTCONN。
pub fn getpeername_handler(fd: u64, addr: u64, addrlen: u64) -> u64 {
	let index = match socket_index(fd) {
		Ok((index, _)) => index,
		Err(e) => {
			return e;
		}
	};
	if let Err(e) = check_sockaddr_out(addr, addrlen) {
		return e;
	}
	let Some(ops) = ops() else {
		return ENODEV;
	};
	let Some(endpoint) = peer_of(ops, index) else {
		return ENOTCONN;
	};
	write_sockaddr(addr, addrlen, &endpoint);
	0
}

/// 取发送的目的端点。*dest_addr* 非 0 时读取其中的地址与端口, 为 0 时取经 connect
/// 记录的对端地址。
///
/// *dest_addr* 为 0 且未记录对端地址时返回 EDESTADDRREQ; 目的端口为 0 时返回
/// EDESTADDRREQ; 目的地址在本机子网之外时返回 ENETUNREACH。
fn send_endpoint(ops: &NetOps, index: u64, dest_addr: u64, addrlen: u64) -> Result<NetEndpoint, u64> {
	if dest_addr == 0 {
		return match peer_of(ops, index) {
			Some(endpoint) => Ok(endpoint),
			None => Err(EDESTADDRREQ),
		};
	}
	let endpoint = read_sockaddr(dest_addr, addrlen)?;
	if endpoint.port == 0 {
		return Err(EDESTADDRREQ);
	}
	if !is_local_subnet(ops, &endpoint) {
		return Err(ENETUNREACH);
	}
	Ok(endpoint)
}

/// 把 *data* 作为一个报文发往 *endpoint*, 返回发送的字节数。发送缓冲区无空间时
/// 返回 EAGAIN。
fn send_bytes(ops: &NetOps, index: u64, endpoint: &NetEndpoint, data: &[u8]) -> u64 {
	if !(unsafe { (ops.send)(index, endpoint, data.as_ptr(), data.len() as u64) }) {
		return EAGAIN;
	}
	// 推进协议栈一次, 使本报文与随之而起的地址解析立即开始; 其后的推进由设备
	// 中断驱动的接收事件完成。
	unsafe {
		(ops.poll)();
	}
	data.len() as u64
}

/// 发送的核心: 把 [buf, buf+len) 作为一个报文发出, 返回发送的字节数。
///
/// 缓冲区地址为 0 而长度为非零时返回 EFAULT; 报文长于 UDP 载荷上限时返回 EMSGSIZE;
/// 目的端点的错误码由 send_endpoint 给出。
fn send_common(ops: &NetOps, index: u64, buf: u64, len: u64, dest_addr: u64, addrlen: u64) -> u64 {
	if len > 0 && buf == 0 {
		return EFAULT;
	}
	if len > (UDP_PAYLOAD_MAX as u64) {
		return EMSGSIZE;
	}
	let endpoint = match send_endpoint(ops, index, dest_addr, addrlen) {
		Ok(endpoint) => endpoint,
		Err(e) => {
			return e;
		}
	};
	let data = if len == 0 {
		&[][..]
	} else {
		unsafe { core::slice::from_raw_parts(buf as *const u8, len as usize) }
	};
	send_bytes(ops, index, &endpoint, data)
}

/// 一次接收的结果: 取到报文时给出写入缓冲区的字节数与报文来源, 需要退回 ecall
/// 重新执行本次调用时为 Restart。
enum RecvOutcome {
	Received(usize, NetEndpoint),
	Restart,
}

/// 接收的核心: 取一个报文写入 [buf, buf+len), 返回写入的字节数与报文来源。
/// *is_peek* 为真时只读取报文而不将其从接收队列中移除。
///
/// 报文长于缓冲区时按缓冲区长度截断, 其余字节丢弃; 缓冲区容量为 0 时返回 0。
/// 无报文可接收时, *is_nonblocking* 为真则返回 EAGAIN, 否则阻塞当前线程, 待有报文
/// 到达或被唤醒后重新检查。
fn recv_common(
	ops: &NetOps,
	index: u64,
	is_nonblocking: bool,
	buf: u64,
	len: u64,
	is_peek: bool
) -> Result<RecvOutcome, u64> {
	loop {
		let mut src = unspecified_endpoint(0);
		let taken = if is_peek {
			unsafe { (ops.peek)(index, buf as *mut u8, len, &mut src) }
		} else {
			unsafe { (ops.recv)(index, buf as *mut u8, len, &mut src) }
		};
		if taken >= 0 {
			return Ok(RecvOutcome::Received(taken as usize, src));
		}
		if is_nonblocking {
			return Err(EAGAIN);
		}
		thread::block_current(recv_wait_addr(index));
		// 有切换目标时退回 ecall 之前重新执行本次调用, 使阻塞期间让出的线程在别处
		// 运行; 无切换目标说明是被唤醒而仍无报文, 重新检查。
		if thread::switch_pending() {
			thread::request_restart();
			return Ok(RecvOutcome::Restart);
		}
	}
}

/// sendto(206): 发送一个 UDP 报文。*dest_addr* 为 0 时使用经 connect 记录的对端
/// 地址。
pub fn sendto_handler(fd: u64, buf: u64, len: u64, flags: u64, dest_addr: u64, addrlen: u64) -> u64 {
	let index = match socket_index(fd) {
		Ok((index, _)) => index,
		Err(e) => {
			return e;
		}
	};
	if (flags & !MSG_DONTWAIT) != 0 {
		return EOPNOTSUPP;
	}
	let Some(ops) = ops() else {
		return ENODEV;
	};
	send_common(ops, index, buf, len, dest_addr, addrlen)
}

/// recvfrom(207): 接收一个 UDP 报文。*src_addr* 非 0 时一并写入报文来源的地址与
/// 端口。缓冲区容量为 0 时返回 0, 缓冲区地址不作要求。
pub fn recvfrom_handler(fd: u64, buf: u64, len: u64, flags: u64, src_addr: u64, addrlen: u64) -> u64 {
	let (index, open_flags) = match socket_index(fd) {
		Ok(entry) => entry,
		Err(e) => {
			return e;
		}
	};
	if (flags & !(MSG_DONTWAIT | MSG_PEEK)) != 0 {
		return EOPNOTSUPP;
	}
	if buf == 0 && len != 0 {
		return EFAULT;
	}
	if src_addr != 0 {
		if let Err(e) = check_sockaddr_out(src_addr, addrlen) {
			return e;
		}
	}
	let Some(ops) = ops() else {
		return ENODEV;
	};
	let is_nonblocking = (flags & MSG_DONTWAIT) != 0 || (open_flags & O_NONBLOCK) != 0;
	match recv_common(ops, index, is_nonblocking, buf, len, (flags & MSG_PEEK) != 0) {
		Ok(RecvOutcome::Received(n, src)) => {
			if src_addr != 0 {
				write_sockaddr(src_addr, addrlen, &src);
			}
			n as u64
		}
		Ok(RecvOutcome::Restart) => 0,
		Err(e) => e,
	}
}

/// sendmsg(211): 发送一个 UDP 报文。目的地址取报文首部的 msg_name, 为 0 时使用经
/// connect 记录的对端地址; 载荷为 msg_iov 各项按次序汇集的结果。报文首部中的控制
/// 信息不予处理。
pub fn sendmsg_handler(fd: u64, msg: u64, flags: u64) -> u64 {
	let index = match socket_index(fd) {
		Ok((index, _)) => index,
		Err(e) => {
			return e;
		}
	};
	if (flags & !MSG_DONTWAIT) != 0 {
		return EOPNOTSUPP;
	}
	if msg == 0 {
		return EFAULT;
	}
	let Some(ops) = ops() else {
		return ENODEV;
	};
	let hdr = unsafe { ptr::read_unaligned(msg as *const MsgHdr) };
	let endpoint = match send_endpoint(ops, index, hdr.msg_name, hdr.msg_namelen as u64) {
		Ok(endpoint) => endpoint,
		Err(e) => {
			return e;
		}
	};
	let mut payload = [0u8; UDP_PAYLOAD_MAX];
	let total = match gather_iovecs(hdr.msg_iov, hdr.msg_iovlen as u64, &mut payload) {
		Ok(total) => total,
		Err(e) => {
			return e;
		}
	};
	send_bytes(ops, index, &endpoint, &payload[..total])
}

/// recvmsg(212): 接收一个 UDP 报文。报文来源写入报文首部的 msg_name, 载荷按
/// msg_iov 的次序分散写入。各缓冲区容量之和小于报文长度时在报文首部的 msg_flags
/// 中置 MSG_TRUNC 位, 返回值为各缓冲区容量之和。
///
/// 报文首部的控制信息缓冲区不产出内容, 其长度按无控制信息写回。
pub fn recvmsg_handler(fd: u64, msg: u64, flags: u64) -> u64 {
	let (index, open_flags) = match socket_index(fd) {
		Ok(entry) => entry,
		Err(e) => {
			return e;
		}
	};
	if (flags & !(MSG_DONTWAIT | MSG_PEEK)) != 0 {
		return EOPNOTSUPP;
	}
	if msg == 0 {
		return EFAULT;
	}
	let hdr = unsafe { ptr::read_unaligned(msg as *const MsgHdr) };
	if hdr.msg_name != 0 && (hdr.msg_namelen as usize) < SOCKADDR_IN_SIZE {
		return EINVAL;
	}
	if let Err(e) = check_iovecs(hdr.msg_iov, hdr.msg_iovlen as u64) {
		return e;
	}
	let Some(ops) = ops() else {
		return ENODEV;
	};
	let mut payload = [0u8; UDP_PAYLOAD_MAX];
	let is_nonblocking = (flags & MSG_DONTWAIT) != 0 || (open_flags & O_NONBLOCK) != 0;
	let (n, src) = match
		recv_common(
			ops,
			index,
			is_nonblocking,
			payload.as_mut_ptr() as u64,
			UDP_PAYLOAD_MAX as u64,
			(flags & MSG_PEEK) != 0
		)
	{
		Ok(RecvOutcome::Received(n, src)) => (n, src),
		Ok(RecvOutcome::Restart) => {
			return 0;
		}
		Err(e) => {
			return e;
		}
	};
	// 以载荷上限为接收容量读入本地缓冲区, 故写入各 iovec 的字节数即报文长度与
	// 各缓冲区容量之和中的较小者。
	let written = scatter_iovecs(hdr.msg_iov, hdr.msg_iovlen as u64, &payload[..n]);
	let mut out = hdr;
	out.msg_controllen = 0;
	if (written as usize) < n {
		out.msg_flags |= MSG_TRUNC;
	}
	if hdr.msg_name != 0 {
		unsafe {
			ptr::write_unaligned(hdr.msg_name as *mut SockaddrIn, sockaddr_in_of(&src));
		}
		out.msg_namelen = SOCKADDR_IN_SIZE as u32;
	}
	unsafe {
		ptr::write_unaligned(msg as *mut MsgHdr, out);
	}
	written as u64
}

/// 取第 *index* 个 iovec 的缓冲区地址与长度, 每个 iovec 占 16 字节。
fn iovec_at(io_vec_arr: u64, index: u64) -> (u64, u64) {
	let slot = io_vec_arr.wrapping_add(index.wrapping_mul(16));
	let base = unsafe { (slot as *const u64).read_volatile() };
	let len = unsafe { (slot.wrapping_add(8) as *const u64).read_volatile() };
	(base, len)
}

/// 校验 iovec 数组, 返回各项缓冲区容量之和。
///
/// 项数大于 IOV_MAX 时返回 EINVAL; 数组地址为 0 而项数非零时返回 EFAULT; 长度为
/// 非零而地址为 0 的缓冲区返回 EFAULT。
fn check_iovecs(io_vec_arr: u64, io_vec_size: u64) -> Result<usize, u64> {
	if io_vec_size > IOV_MAX {
		return Err(EINVAL);
	}
	if io_vec_arr == 0 {
		return if io_vec_size == 0 { Ok(0) } else { Err(EFAULT) };
	}
	let mut total = 0usize;
	for i in 0..io_vec_size {
		let (base, len) = iovec_at(io_vec_arr, i);
		if len != 0 && base == 0 {
			return Err(EFAULT);
		}
		total = total.saturating_add(len as usize);
	}
	Ok(total)
}

/// 按 iovec 的次序把 *payload* 分散写入各缓冲区, 返回写入的字节数。
fn scatter_iovecs(io_vec_arr: u64, io_vec_size: u64, payload: &[u8]) -> u64 {
	let mut offset = 0usize;
	for i in 0..io_vec_size {
		if offset >= payload.len() {
			break;
		}
		let (base, len) = iovec_at(io_vec_arr, i);
		let count = cmp::min(len as usize, payload.len() - offset);
		if count == 0 {
			continue;
		}
		unsafe {
			ptr::copy_nonoverlapping(payload.as_ptr().add(offset), base as *mut u8, count);
		}
		offset += count;
	}
	offset as u64
}

/// 按 iovec 的次序把各缓冲区汇集到 *payload*, 返回汇集后的字节数。
/// 各缓冲区容量之和大于 *payload* 的长度时返回 EMSGSIZE。
fn gather_iovecs(io_vec_arr: u64, io_vec_size: u64, payload: &mut [u8]) -> Result<usize, u64> {
	let total = check_iovecs(io_vec_arr, io_vec_size)?;
	if total > payload.len() {
		return Err(EMSGSIZE);
	}
	let mut offset = 0usize;
	for i in 0..io_vec_size {
		let (base, len) = iovec_at(io_vec_arr, i);
		let count = len as usize;
		if count == 0 {
			continue;
		}
		unsafe {
			ptr::copy_nonoverlapping(base as *const u8, payload.as_mut_ptr().add(offset), count);
		}
		offset += count;
	}
	Ok(offset)
}

/// read 在套接字上的接收动作。等同于不给出来源地址的 recvfrom, 阻塞判据只取 fd 的
/// 打开标志。
pub fn read_socket(index: u64, open_flags: u32, buf: *mut u8, len: u64) -> u64 {
	if buf.is_null() && len != 0 {
		return EFAULT;
	}
	let Some(ops) = ops() else {
		return ENODEV;
	};
	let is_nonblocking = (open_flags & O_NONBLOCK) != 0;
	match recv_common(ops, index, is_nonblocking, buf as u64, len, false) {
		Ok(RecvOutcome::Received(n, _)) => n as u64,
		Ok(RecvOutcome::Restart) => 0,
		Err(e) => e,
	}
}

/// write 在套接字上的发送动作。等同于不给出目的地址的 sendto, 即发往经 connect
/// 记录的对端地址。
pub fn write_socket(index: u64, buf: *const u8, len: u64) -> u64 {
	let Some(ops) = ops() else {
		return ENODEV;
	};
	send_common(ops, index, buf as u64, len, 0, 0)
}

/// readv 在套接字上的接收动作: 取一个报文, 按 iovec 的次序逐一写入, 返回写入 iovec
/// 的字节数。各缓冲区容量之和不足报文长度时丢弃其余字节。
pub fn readv_socket(index: u64, open_flags: u32, io_vec_arr: u64, io_vec_size: u64) -> u64 {
	if let Err(e) = check_iovecs(io_vec_arr, io_vec_size) {
		return e;
	}
	let Some(ops) = ops() else {
		return ENODEV;
	};
	let mut payload = [0u8; UDP_PAYLOAD_MAX];
	let is_nonblocking = (open_flags & O_NONBLOCK) != 0;
	let n = match
		recv_common(ops, index, is_nonblocking, payload.as_mut_ptr() as u64, UDP_PAYLOAD_MAX as u64, false)
	{
		Ok(RecvOutcome::Received(n, _)) => n,
		Ok(RecvOutcome::Restart) => {
			return 0;
		}
		Err(e) => {
			return e;
		}
	};
	scatter_iovecs(io_vec_arr, io_vec_size, &payload[..n])
}

/// writev 在套接字上的发送动作: 按 iovec 的次序汇集为一个报文后发出, 各缓冲区
/// 容量之和超出载荷上限时返回 EMSGSIZE。
pub fn writev_socket(index: u64, io_vec_arr: u64, io_vec_size: u64) -> u64 {
	let mut payload = [0u8; UDP_PAYLOAD_MAX];
	let total = match gather_iovecs(io_vec_arr, io_vec_size, &mut payload) {
		Ok(total) => total,
		Err(e) => {
			return e;
		}
	};
	write_socket(index, payload.as_ptr(), total as u64)
}

/// setsockopt(208): 设置套接字选项。
///
/// 只接受 SOL_SOCKET 层的收发缓冲区容量两项: 容量由协议栈的静态缓冲区固定, 本调用
/// 返回成功但不改变容量。其余层与选项未实现, 返回 ENOPROTOOPT。
pub fn setsockopt_handler(fd: u64, level: u64, optname: u64, optval: u64, optlen: u64) -> u64 {
	if let Err(e) = socket_index(fd) {
		return e;
	}
	if level != SOL_SOCKET {
		return ENOPROTOOPT;
	}
	match optname {
		SO_SNDBUF | SO_RCVBUF => {
			if optval == 0 {
				return EFAULT;
			}
			if optlen < 4 {
				return EINVAL;
			}
			0
		}
		_ => ENOPROTOOPT,
	}
}

/// getsockopt(209): 读取套接字选项。
///
/// 支持 SOL_SOCKET 层的 SO_TYPE、SO_DOMAIN、SO_ERROR 与收发缓冲区容量, 其余层与
/// 选项返回 ENOPROTOOPT。
pub fn getsockopt_handler(fd: u64, level: u64, optname: u64, optval: u64, optlen: u64) -> u64 {
	if let Err(e) = socket_index(fd) {
		return e;
	}
	if level != SOL_SOCKET {
		return ENOPROTOOPT;
	}
	if optval == 0 || optlen == 0 {
		return EFAULT;
	}
	let buf_bytes = match ops() {
		Some(ops) => ops.payload_bytes as u32,
		// 套接字槽位容量由协议栈的静态缓冲区决定, 该模块未接入时取 0。
		None => 0,
	};
	let value: u32 = match optname {
		SO_TYPE => SOCK_DGRAM as u32,
		SO_DOMAIN => AF_INET as u32,
		SO_ERROR => 0,
		SO_SNDBUF | SO_RCVBUF => buf_bytes,
		_ => {
			return ENOPROTOOPT;
		}
	};
	unsafe {
		ptr::write_unaligned(optval as *mut u32, value);
		ptr::write_unaligned(optlen as *mut u32, 4);
	}
	0
}

/// shutdown(210): 关闭套接字的读方向或写方向。
///
/// UDP 没有连接拆除过程: 写方向关闭时清除经 connect 记录的对端地址, 此后未给出目的
/// 地址的发送返回 EDESTADDRREQ, 读方向关闭不改变任何状态。未记录对端地址时返回
/// ENOTCONN。
pub fn shutdown_handler(fd: u64, how: u64) -> u64 {
	let index = match socket_index(fd) {
		Ok((index, _)) => index,
		Err(e) => {
			return e;
		}
	};
	if how > SHUT_RDWR {
		return EINVAL;
	}
	let Some(ops) = ops() else {
		return ENODEV;
	};
	if peer_of(ops, index).is_none() {
		return ENOTCONN;
	}
	if how != SHUT_RD {
		unsafe {
			(ops.clear_peer)(index);
		}
	}
	0
}
