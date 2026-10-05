//! 载荷加载在主机目标上的单元测试入口。
//!
//! 模块名为 `elf_src` 而非 `elf`: elf.rs 内部以 `elf::ElfBytes` 引用 elf crate,
//! 同名的本地模块会遮蔽该 crate。
//!
//! 主机上没有 SBI, `enclave_call_exit` 以 panic 中断执行并记下退出码, 用例经
//! catch_unwind 观察它, 从而断言载荷无法加载时飞地以哪个退出码终止。

#![allow(dead_code)]

#[path = "../src/constants.rs"]
mod constants;

/// 主机上没有进程上下文, 页表根取 0。
mod context {
	pub fn root_pa() -> u64 {
		0
	}
}

/// 主机上没有 satp 寄存器, 读回 0。
mod csr {
	pub fn read_satp() -> u64 {
		0
	}
}

/// 主机上没有 S 模式与 U 模式页池。S 模式页以进程内一块 4 KiB 对齐的静态内存
/// 充当, U 模式页不交付。
mod mem {
	use std::sync::atomic::{ AtomicUsize, Ordering };

	const POOL_PAGES: usize = 64;

	#[derive(Clone, Copy)]
	#[repr(align(4096))]
	struct Page([u8; 4096]);

	static mut POOL: [Page; POOL_PAGES] = [Page([0; 4096]); POOL_PAGES];
	static CURSOR: AtomicUsize = AtomicUsize::new(0);

	pub fn try_alloc_smode_page(n: u64) -> u64 {
		let first = CURSOR.fetch_add(n as usize, Ordering::SeqCst);
		if first + n as usize > POOL_PAGES {
			return !0u64;
		}
		core::ptr::addr_of_mut!(POOL) as u64 + (first as u64) * 4096
	}

	pub fn alloc_umode_page(_n: u64) -> u64 {
		!0u64
	}

	pub fn alloc_mmap_chunk_pa() -> u64 {
		!0u64
	}
}

/// SBI 桩: 控制台写入丢弃, 内存申请不交付, 退出经 panic 上报。
mod ecall_aux {
	use std::sync::atomic::{ AtomicU64, Ordering };

	/// 最近一次交付的退出码; 尚未交付时为 `u64::MAX`。
	pub static EXIT_CODE: AtomicU64 = AtomicU64::new(u64::MAX);

	/// 飞地终止的 panic 载荷, 供用例识别。
	#[derive(Debug)]
	pub struct EnclaveExited;

	pub fn sbi_console_write(_bytes: &[u8]) {}

	pub fn enclave_call_mem_alloc(_n: u64) -> (u64, u64) {
		(0, !0u64)
	}

	pub fn enclave_call_exit(code: u64) -> ! {
		EXIT_CODE.store(code, Ordering::SeqCst);
		std::panic::panic_any(EnclaveExited)
	}
}

/// 主机上没有 SBI 控制台。FmtBuf 与目标侧同形, 写入经 sbi_console_write 丢弃。
mod println {
	use core::fmt;

	pub struct FmtBuf {
		buf: [u8; 512],
		pos: usize,
	}

	impl FmtBuf {
		pub fn new() -> Self {
			Self { buf: [0u8; 512], pos: 0 }
		}

		pub fn as_written(&self) -> &[u8] {
			&self.buf[..self.pos]
		}
	}

	impl fmt::Write for FmtBuf {
		fn write_str(&mut self, s: &str) -> fmt::Result {
			let bytes = s.as_bytes();
			let n = bytes.len().min(self.buf.len() - self.pos);
			self.buf[self.pos..self.pos + n].copy_from_slice(&bytes[..n]);
			self.pos += n;
			Ok(())
		}
	}
}

#[macro_export]
macro_rules! println {
	($($arg:tt)*) => {{
		let mut __buf = $crate::println::FmtBuf::new();
		let _ = core::fmt::Write::write_fmt(&mut __buf, format_args!($($arg)*));
		$crate::ecall_aux::sbi_console_write(__buf.as_written());
	}};
}

#[path = "../src/mem_prim/mod.rs"]
mod mem_prim;

#[path = "../src/paging.rs"]
mod paging;

#[path = "../src/elf.rs"]
mod elf_src;

use std::panic::{ catch_unwind, AssertUnwindSafe };
use std::sync::atomic::Ordering;

use ecall_aux::{ EnclaveExited, EXIT_CODE };

/// 载荷被拒绝时的退出码, 与 elf.rs 的 PAYLOAD_REJECT_EXIT_CODE 一致。
const PAYLOAD_REJECT_EXIT_CODE: u64 = 125;

/// 以给定映像调用 load_elf, 返回其交付的退出码; load_elf 正常返回时返回 None。
///
/// 映像的物理地址按 elf.rs 的换算反推, 使 `elf_pa + LINEAR_MAP_OFFSET` 恰好等于
/// 缓冲区在进程内的地址: elf.rs 经该别名读取映像, 而主机上只有进程内地址可读。
fn release_exit_code(image: &'static [u8]) -> Option<u64> {
	EXIT_CODE.store(u64::MAX, Ordering::SeqCst);
	let elf_pa = (image.as_ptr() as u64).wrapping_sub(constants::LINEAR_MAP_OFFSET);
	let result = catch_unwind(AssertUnwindSafe(|| {
		elf_src::load_elf(elf_pa, image.len() as u64);
	}));
	let code = EXIT_CODE.load(Ordering::SeqCst);
	match result {
		Ok(()) => None,
		Err(payload) => {
			assert!(
				payload.downcast_ref::<EnclaveExited>().is_some(),
				"load_elf 未以飞地终止收尾, 而是 panic 于其它原因"
			);
			Some(code)
		}
	}
}

/// 非 ELF 映像 (如误入载荷目录的重定向输出文件) 使飞地以载荷拒绝码终止,
/// 而不是留在冻结挂起状态 —— 后者会使宿主一直阻塞在 ENTER 或 RESUME 上。
#[test]
fn non_elf_image_terminates_enclave() {
	let image: &'static [u8] = b"nc_udp: udp round trip ok\n";
	assert_eq!(release_exit_code(image), Some(PAYLOAD_REJECT_EXIT_CODE));
}

/// 带 ELF 魔数但不足以容纳 ELF64 头的映像同样以载荷拒绝码终止。
#[test]
fn truncated_elf_image_terminates_enclave() {
	let image: &'static [u8] = b"\x7fELF\x02\x01\x01";
	assert_eq!(release_exit_code(image), Some(PAYLOAD_REJECT_EXIT_CODE));
}
