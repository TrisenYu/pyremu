//! 网络模块的入口。
//!
//! 网卡驱动与协议栈整体在本模块内。载荷发起的套接字系统调用仍由运行时处理 (其参数与
//! 返回值的编解码即载荷侧的接口), 运行时处理时涉及协议栈的部分经本模块导出的操作表
//! 调用。故本模块的系统调用处理函数表全为空槽位, 运行时自身的调用点见 [`OPS`]。
//!
//! 映像的入口在偏移 0, 无 ELF 头, 无重定位项, 见 ../module.ld 与 ../Makefile。
//!
//! 接口结构体、管理器回调集合与操作表按 `#[path]` 取自 sittim 的 src/ext_mod, 与运行时
//! 共用同一份定义。

#![no_std]

/// 与 sittim 共用的接口定义。该文件中的一部分条目只有 sittim 侧使用, 本模块不引用
/// 它们, 故在本模块内不作未使用判定。
#[allow(dead_code)]
#[path = "../../../src/ext_mod/abi.rs"]
mod abi;

/// 与 sittim 共用的管理器回调集合。
#[allow(dead_code)]
#[path = "../../../src/ext_mod/man.rs"]
mod man;

/// 与 sittim 共用的操作表定义, 本模块负责填写它。
#[allow(dead_code)]
#[path = "../../../src/ext_mod/net_ops.rs"]
mod net_ops;

mod driver;
mod mmio;
mod phy;
mod platform;
mod stack;
mod virt_queue;

use core::mem::MaybeUninit;

use smoltcp::wire::{IpAddress, IpEndpoint, Ipv4Address};

use abi::{
	GetterFn, InitFn, ModuleDesc, ModuleInterface, SyscallTable, MODULE_NAME_LEN,
	SYSCALL_TABLE_LEN,
};
use man::Manager;
use net_ops::{
	NetEndpoint, NetOps, NET_ADDR_MAX, NET_AF_UNSPEC, NET_BIND_ALREADY_BOUND,
	NET_BIND_NO_FREE_PORT, NET_BIND_NOT_READY, NET_BIND_OK, NET_BIND_PORT_IN_USE, NET_NO_SLOT,
};

/// 模块编号, 与 ref-impl/emod 的 `EMODULE_ID_NET` 取同一值。
const MODULE_ID: u32 = 4;

/// 模块名, 不足 [`MODULE_NAME_LEN`] 的部分以 0 填充。
const MODULE_NAME: [u8; MODULE_NAME_LEN] = {
	let mut name = [0u8; MODULE_NAME_LEN];
	let src = b"net";
	let mut i = 0;
	while i < src.len() {
		name[i] = src[i];
		i += 1;
	}
	name
};

/// 本模块导出的系统调用处理函数表。套接字的系统调用处理函数留在运行时, 故本表全为
/// 空槽位。
static SYSCALLS: SyscallTable = SyscallTable { handlers: [None; SYSCALL_TABLE_LEN] };

/// 本模块导出的操作表。
///
/// 各函数的地址在入口内写入, 不以静态初值给出: 静态初值会把该地址作为常量放进映像,
/// 而映像在链接时不知道自己将被放在哪个基址。
static mut OPS: MaybeUninit<NetOps> = MaybeUninit::uninit();

/// 模块入口, 位于映像偏移 0。
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.init")]
pub unsafe extern "C" fn module_init(manager: *const Manager) -> GetterFn {
	platform::attach(manager);
	unsafe { core::ptr::addr_of_mut!(OPS).cast::<NetOps>().write(fill_ops()) };
	get_interface
}

/// 入口的签名与运行时侧的 [`InitFn`] 一致。
const _: InitFn = module_init;

/// 取用器, 交付本模块的标识与两张表。可重复调用, 每次返回相同接口。
unsafe extern "C" fn get_interface(_manager: *const Manager) -> ModuleInterface {
	ModuleInterface {
		desc: ModuleDesc {
			module_id: MODULE_ID,
			name: MODULE_NAME,
			signature: 0,
		},
		syscalls: &raw const SYSCALLS,
		ops: core::ptr::addr_of!(OPS).cast::<u8>(),
	}
}

/// 建立本模块导出的操作表。
fn fill_ops() -> NetOps {
	NetOps {
		socket_slots: stack::UDP_SOCKETS as u64,
		payload_bytes: stack::PAYLOAD_BYTES as u64,
		local_addr: NetEndpoint::from_v4(stack::LOCAL_IP.octets(), 0),
		local_prefix_len: stack::LOCAL_PREFIX_LEN,

		is_ready: ops_is_ready,
		init: ops_init,
		poll: ops_poll,
		open: ops_open,
		close: ops_close,
		is_used: ops_is_used,
		bind: ops_bind,
		connect: ops_connect,
		peer: ops_peer,
		clear_peer: ops_clear_peer,
		local_endpoint: ops_local_endpoint,
		send: ops_send,
		can_recv: ops_can_recv,
		can_send: ops_can_send,
		recv: ops_recv,
		peek: ops_peek,
	}
}

// ---------------------------------------------------------------
//  端点的转换
// ---------------------------------------------------------------

/// 由协议栈的端点建立操作表的端点。
///
/// 协议栈未启用 IPv6, 地址类型的唯一变体即 IPv4。
fn endpoint_of(endpoint: IpEndpoint) -> NetEndpoint {
	match endpoint.addr {
		IpAddress::Ipv4(ip) => NetEndpoint::from_v4(ip.octets(), endpoint.port),
	}
}

/// 由操作表的端点建立协议栈的端点。地址族不是 IPv4 时返回 None。
fn ip_endpoint_of(endpoint: &NetEndpoint) -> Option<IpEndpoint> {
	let octets = endpoint.octets_v4()?;
	Some(IpEndpoint::new(IpAddress::Ipv4(Ipv4Address::from(octets)), endpoint.port))
}

/// 由操作表的端点取本地地址与端口。地址族不是 IPv4 时本地地址取未指定。
fn local_addr_of(endpoint: &NetEndpoint) -> (Option<Ipv4Address>, u16) {
	match endpoint.octets_v4() {
		Some(octets) => (Some(Ipv4Address::from(octets)), endpoint.port),
		None => (None, endpoint.port),
	}
}

// ---------------------------------------------------------------
//  操作表的各项
// ---------------------------------------------------------------

unsafe extern "C" fn ops_is_ready() -> bool {
	stack::is_ready()
}

unsafe extern "C" fn ops_init() -> bool {
	stack::init()
}

unsafe extern "C" fn ops_poll() {
	stack::poll();
}

unsafe extern "C" fn ops_open() -> u64 {
	match stack::open() {
		Some(index) => index as u64,
		None => NET_NO_SLOT,
	}
}

unsafe extern "C" fn ops_close(index: u64) {
	stack::close(index as usize);
}

unsafe extern "C" fn ops_is_used(index: u64) -> bool {
	stack::is_used(index as usize)
}

unsafe extern "C" fn ops_bind(index: u64, endpoint: *const NetEndpoint) -> u64 {
	let (addr, port) = match unsafe { endpoint.as_ref() } {
		Some(endpoint) => local_addr_of(endpoint),
		None => (None, 0),
	};
	match stack::bind(index as usize, addr, port) {
		Ok(()) => NET_BIND_OK,
		Err(err) => bind_code(err),
	}
}

/// 绑定失败对应的返回值。
fn bind_code(err: stack::BindError) -> u64 {
	match err {
		stack::BindError::NotReady => NET_BIND_NOT_READY,
		stack::BindError::AlreadyBound => NET_BIND_ALREADY_BOUND,
		stack::BindError::PortInUse => NET_BIND_PORT_IN_USE,
		stack::BindError::NoFreePort => NET_BIND_NO_FREE_PORT,
	}
}

unsafe extern "C" fn ops_connect(index: u64, endpoint: *const NetEndpoint) -> bool {
	let Some(remote) = (unsafe { endpoint.as_ref() }).and_then(ip_endpoint_of) else {
		return false;
	};
	stack::connect(index as usize, remote)
}

unsafe extern "C" fn ops_peer(index: u64, out: *mut NetEndpoint) -> bool {
	let Some(remote) = stack::peer(index as usize) else {
		return false;
	};
	if !out.is_null() {
		unsafe { out.write(endpoint_of(remote)) };
	}
	true
}

unsafe extern "C" fn ops_clear_peer(index: u64) {
	stack::clear_peer(index as usize);
}

unsafe extern "C" fn ops_local_endpoint(index: u64, out: *mut NetEndpoint) {
	let local = stack::local_endpoint(index as usize);
	let endpoint = match local.addr {
		Some(IpAddress::Ipv4(ip)) => NetEndpoint::from_v4(ip.octets(), local.port),
		None => NetEndpoint {
			family: NET_AF_UNSPEC,
			addr: [0; NET_ADDR_MAX],
			port: local.port,
		},
	};
	if !out.is_null() {
		unsafe { out.write(endpoint) };
	}
}

unsafe extern "C" fn ops_send(
	index: u64,
	endpoint: *const NetEndpoint,
	buf: *const u8,
	len: u64,
) -> bool {
	let Some(remote) = (unsafe { endpoint.as_ref() }).and_then(ip_endpoint_of) else {
		return false;
	};
	if len != 0 && buf.is_null() {
		return false;
	}
	let data: &[u8] = if len == 0 {
		&[]
	} else {
		unsafe { core::slice::from_raw_parts(buf, len as usize) }
	};
	stack::send(index as usize, remote, data)
}

unsafe extern "C" fn ops_can_recv(index: u64) -> bool {
	stack::can_recv(index as usize)
}

unsafe extern "C" fn ops_can_send(index: u64) -> bool {
	stack::can_send(index as usize)
}

unsafe extern "C" fn ops_recv(
	index: u64,
	buf: *mut u8,
	len: u64,
	src: *mut NetEndpoint,
) -> i64 {
	unsafe { take_report(index, buf, len, src, false) }
}

unsafe extern "C" fn ops_peek(
	index: u64,
	buf: *mut u8,
	len: u64,
	src: *mut NetEndpoint,
) -> i64 {
	unsafe { take_report(index, buf, len, src, true) }
}

/// 取一个报文写入调用方的缓冲区, 返回写入的字节数; 无报文可接收时返回 -1。
///
/// 报文长于 `len` 时按 `len` 截断, 其余字节丢弃。`src` 与 `buf` 为空指针时相应的输出
/// 不作写入; `len` 非零而 `buf` 为空时按无报文可接收处理。
unsafe fn take_report(
	index: u64,
	buf: *mut u8,
	len: u64,
	src: *mut NetEndpoint,
	is_peek: bool,
) -> i64 {
	if len != 0 && buf.is_null() {
		return -1;
	}
	let sink = |data: &[u8], endpoint: IpEndpoint| {
		let n = core::cmp::min(data.len(), len as usize);
		if n != 0 {
			unsafe { core::ptr::copy_nonoverlapping(data.as_ptr(), buf, n) };
		}
		if !src.is_null() {
			unsafe { src.write(endpoint_of(endpoint)) };
		}
		n as i64
	};
	let taken = if is_peek {
		stack::peek(index as usize, sink)
	} else {
		stack::recv(index as usize, sink)
	};
	taken.unwrap_or(-1)
}

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo<'_>) -> ! {
	unsafe { core::arch::asm!("unimp", options(noreturn)) }
}
