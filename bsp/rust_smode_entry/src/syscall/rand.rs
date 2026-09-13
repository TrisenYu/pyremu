//! 宿主随机数通道。
//!
//! 随机字节优先取自宿主随机数设备 (pyremu,crng)。飞地不能直接访问宿主外设的
//! MMIO (PMP 只放行本飞地的内存区域), 故经 M-mode 转发: M-mode 读取设备后写入
//! 本飞地的物理页, 与 syscall/io.rs 的输入侧和输出侧同属一条代理链路。
//!
//! 设备或该 ecall 不可用时退化为运行时内的软件伪随机数。getrandom 取不到随机数
//! 时 Rust std 会直接 panic, 故本接口必须恒有输出; 退化路径的强度弱于设备路径,
//! 仅用于在缺少随机数设备的平台上维持可用。

use crate::ecall_aux;
use crate::syscall::{EFAULT, EINVAL};

/// 每次 ecall 搬运的字节数, 也是运行时栈上缓冲的大小。
const RAND_CHUNK: usize = 256;

// Linux 的 GRND_* 标志位 (include/uapi/linux/random.h)。
const GRND_NONBLOCK: u64 = 0x0001;
const GRND_RANDOM: u64 = 0x0002;
const GRND_INSECURE: u64 = 0x0004;

// ---------------------------------------------------------------
//  设备路径
// ---------------------------------------------------------------

/// 请 M-mode 用宿主随机数设备填满 *out*, 设备不可用时返回 0。
fn fill_from_device(out: &mut [u8]) -> u64 {
	let got = ecall_aux::enclave_call_get_rand_num(out);
	if got == out.len() as u64 {
		return got;
	}
	0
}

// ---------------------------------------------------------------
//  退化路径: 软件伪随机数
// ---------------------------------------------------------------

/// xorshift64* 的状态, 首次使用时播种。
static mut FALLBACK_STATE: u64 = 0;
static mut FALLBACK_SEEDED: bool = false;

/// 读取时间计数器。
fn time_ticks() -> u64 {
	let t: u64;
	unsafe { core::arch::asm!("csrr {0}, 0xC01", out(reg) t) };
	t
}

/// 推进状态并返回一个伪随机 64 位值, 首次调用时用时间计数器播种。
fn fallback_word() -> u64 {
	unsafe {
		if !FALLBACK_SEEDED {
			let seed = time_ticks();
			// xorshift 的状态不能为 0, 否则永远输出 0。
			FALLBACK_STATE = if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed };
			FALLBACK_SEEDED = true;
		}
		let mut x = FALLBACK_STATE;
		x ^= x >> 12;
		x ^= x << 25;
		x ^= x >> 27;
		FALLBACK_STATE = x;
		x.wrapping_mul(0x2545_F491_4F6C_DD1D)
	}
}

/// 以软件伪随机数填满 *out*。
fn fill_fallback(out: &mut [u8]) {
	let mut off = 0;
	while off < out.len() {
		let word = fallback_word().to_le_bytes();
		let n = core::cmp::min(8, out.len() - off);
		out[off..off + n].copy_from_slice(&word[..n]);
		off += n;
	}
}

// ---------------------------------------------------------------
//  getrandom (278)
// ---------------------------------------------------------------

/// getrandom: 参数 (Linux rv64) 为 buf = a0, buflen = a1, flags = a2。
/// 返回写入的字节数。
///
/// 载荷的缓冲区是只在其自身页表中有效的 U 模式虚拟地址, M-mode 无法折算, 故逐块
/// 拷到运行时栈上再交给 M-mode 按物理地址写入 (与 io.rs 的控制台输出同理)。
/// 三个标志位都不改变行为, 未识别的标志位按 Linux 语义返回 EINVAL。
pub fn getrandom_handler(buf: *mut u8, buflen: u64, flags: u64) -> u64 {
	if buf.is_null() && buflen != 0 {
		return EFAULT;
	}
	if flags & !(GRND_NONBLOCK | GRND_RANDOM | GRND_INSECURE) != 0 {
		return EINVAL;
	}

	let mut chunk = [0u8; RAND_CHUNK];
	let mut off: u64 = 0;
	// 设备一旦探测失败, 本次调用余下的字节全部走退化路径, 不再重复发起 ecall。
	let mut device_usable = true;

	while off < buflen {
		let n = core::cmp::min(RAND_CHUNK as u64, buflen - off) as usize;
		let from_device = device_usable && fill_from_device(&mut chunk[..n]) != 0;
		if !from_device {
			device_usable = false;
			fill_fallback(&mut chunk[..n]);
		}
		for i in 0..n {
			unsafe { buf.add((off + i as u64) as usize).write_volatile(chunk[i]) };
		}
		off += n as u64;
	}
	buflen
}
