//! 载荷加载在主机目标上的单元测试入口。
#![allow(dead_code)]
#![allow(unused_imports)]

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

/// 主机上没有 S 模式页池。S 模式的页表页以进程内一块 4 KiB 对齐的静态内存充当;
/// U 模式的物理内存由 M 模式按分区交付, 主机上没有该通路。
mod mem {
	use std::sync::atomic::{ AtomicU64, AtomicUsize, Ordering };

	const POOL_PAGES: usize = 64;

	#[derive(Clone, Copy)]
	#[repr(align(4096))]
	struct Page([u8; 4096]);

	static mut POOL: [Page; POOL_PAGES] = [Page([0; 4096]); POOL_PAGES];
	static PAGES_USED: AtomicUsize = AtomicUsize::new(0);

	pub fn try_alloc_smode_page(n: u64) -> u64 {
		let first = PAGES_USED.fetch_add(n as usize, Ordering::SeqCst);
		if first + n as usize > POOL_PAGES {
			return !0u64;
		}
		core::ptr::addr_of_mut!(POOL) as u64 + (first as u64) * 4096
	}

	/// 分区申请的次数。
	pub static ALLOC_CALLS: AtomicU64 = AtomicU64::new(0);

	/// 主机上不申请 M 模式分区: 载荷段的物理内存由 M 模式交付, 主机上没有该通路, 故
	/// 段装入一律在申请失败处终止。段的虚拟区间与映像内容的核对不经过该处。
	pub fn alloc_adjacent_pages(_pages: u64, _attr: u64) -> u64 {
		ALLOC_CALLS.fetch_add(1, Ordering::SeqCst);
		!0u64
	}

	/// 主机上不向 M 模式发起类别改判: 类别改判只经 510 号调用对飞地内存生效。
	pub fn set_partition_attr(_pa: u64, _bytes: u64, _attr: u64) -> bool {
		false
	}

	/// 主机上两个类别的来源都不交付, 类别只用于选择来源, 故一律取数据类。
	pub fn attr_of_pte_flags(_flags: u64) -> u64 {
		crate::constants::ENCLAVE_MEM_ATTR_DATA
	}
}

/// SBI 桩: 控制台写入丢弃, 内存申请不交付, 退出经 panic 上报。
mod ecall_aux {
	/// 飞地终止的 panic 载荷, 携带退出码, 供用例识别。
	#[derive(Debug)]
	pub struct EnclaveExited(pub u64);

	pub fn sbi_console_write(_bytes: &[u8]) {}

	pub fn enclave_call_mem_alloc(_n: u64, _attr: u64) -> (u64, u64) {
		(0, !0u64)
	}

	pub fn enclave_call_exit(code: u64) -> ! {
		std::panic::panic_any(EnclaveExited(code))
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

use ecall_aux::EnclaveExited;

/// 载荷被拒绝时的退出码, 与 elf.rs 的 PAYLOAD_REJECT_EXIT_CODE 一致。
const PAYLOAD_REJECT_EXIT_CODE: u64 = 125;

/// 以给定映像调用 load_elf, 返回其交付的退出码; load_elf 正常返回时返回 None。
fn release_exit_code(image: &'static [u8]) -> Option<u64> {
	mem::ALLOC_CALLS.store(0, Ordering::SeqCst);
	let elf_pa = (image.as_ptr() as u64).wrapping_sub(constants::LINEAR_MAP_OFFSET);
	let result = catch_unwind(AssertUnwindSafe(|| {
		elf_src::load_elf(elf_pa, image.len() as u64);
	}));
	match result {
		Ok(()) => None,
		Err(payload) => Some(payload.downcast::<EnclaveExited>().unwrap().0),
	}
}

/// 本次装入申请的分区数。
fn alloc_calls() -> u64 {
	mem::ALLOC_CALLS.load(Ordering::SeqCst)
}

#[test]
fn non_elf_image_terminates_enclave() {
	let image: &'static [u8] = b"nc_udp: udp round trip ok\n";
	assert_eq!(release_exit_code(image), Some(PAYLOAD_REJECT_EXIT_CODE));
}

#[test]
fn truncated_elf_image_terminates_enclave() {
	let image: &'static [u8] = b"\x7fELF\x02\x01\x01";
	assert_eq!(release_exit_code(image), Some(PAYLOAD_REJECT_EXIT_CODE));
}

/// 一个 PT_LOAD 段在映像中的取值, 对应程序头的 p_offset、p_vaddr、p_filesz 与
/// p_memsz。其余字段取固定值: p_type 为 PT_LOAD, p_flags 为可读可执行, p_align
/// 为页大小。
struct LoadSegment {
	offset: u64,
	vaddr: u64,
	filesz: u64,
	memsz: u64,
}

/// 构造一个 ELF64 映像: 文件头与程序头表合法, 含两个 PT_LOAD 段, 段内容为 0。
fn image_with_two_load_segments(first: LoadSegment, second: LoadSegment) -> &'static [u8] {
	let mut image = Vec::new();
	// e_ident: 魔数、ELFCLASS64、小端、EV_CURRENT, 其余字节为 0。
	image.extend_from_slice(b"\x7fELF\x02\x01\x01");
	image.resize(16, 0);
	image.extend_from_slice(&constants::ET_EXEC.to_le_bytes());
	image.extend_from_slice(&constants::EM_RISCV.to_le_bytes());
	image.extend_from_slice(&1u32.to_le_bytes()); // e_version
	image.extend_from_slice(&first.vaddr.to_le_bytes()); // e_entry
	image.extend_from_slice(&64u64.to_le_bytes()); // e_phoff, 紧随文件头
	image.extend_from_slice(&0u64.to_le_bytes()); // e_shoff, 无段头表
	image.extend_from_slice(&0u32.to_le_bytes()); // e_flags
	image.extend_from_slice(&64u16.to_le_bytes()); // e_ehsize
	image.extend_from_slice(&56u16.to_le_bytes()); // e_phentsize
	image.extend_from_slice(&2u16.to_le_bytes()); // e_phnum
	image.extend_from_slice(&[0u8; 6]); // e_shentsize, e_shnum, e_shstrndx
	for segment in [first, second] {
		image.extend_from_slice(&1u32.to_le_bytes()); // p_type = PT_LOAD
		image.extend_from_slice(&5u32.to_le_bytes()); // p_flags = PF_R | PF_X
		image.extend_from_slice(&segment.offset.to_le_bytes());
		image.extend_from_slice(&segment.vaddr.to_le_bytes());
		image.extend_from_slice(&0u64.to_le_bytes()); // p_paddr
		image.extend_from_slice(&segment.filesz.to_le_bytes());
		image.extend_from_slice(&segment.memsz.to_le_bytes());
		image.extend_from_slice(&0x1000u64.to_le_bytes()); // p_align
	}
	assert_eq!(image.len(), 64 + 2 * 56);
	image.resize(0x1000, 0);
	Box::leak(image.into_boxed_slice())
}

/// 两个 PT_LOAD 段的页区间相交时, 映像在申请任何分区之前即被拒。
///
/// 这是段页区间核对的回归用例: 段各自申请分区并映射自己的页区间, 区间相交的布局会
/// 使后装入的段改写同一张页的叶子项。
#[test]
fn overlapping_segment_pages_terminate_enclave() {
	// 前者占页 [0x1000, 0x3000), 后者占页 [0x2000, 0x3000)。
	let image = image_with_two_load_segments(
		LoadSegment {
			offset: 0x1000,
			vaddr: 0x1000,
			filesz: 0x200,
			memsz: 0x1400,
		},
		LoadSegment {
			offset: 0x2000,
			vaddr: 0x2000,
			filesz: 0x100,
			memsz: 0x100,
		}
	);

	assert_eq!(release_exit_code(image), Some(PAYLOAD_REJECT_EXIT_CODE));
	assert_eq!(alloc_calls(), 0);
}

/// 两个 PT_LOAD 段的页区间相接于页边界时不算相交, 映像进入段装入流程。
#[test]
fn touching_segment_pages_enter_the_load_path() {
	// 前者占页 [0x1000, 0x2000), 后者占页 [0x2000, 0x3000)。
	let image = image_with_two_load_segments(
		LoadSegment {
			offset: 0x1000,
			vaddr: 0x1000,
			filesz: 0x200,
			memsz: 0x1000,
		},
		LoadSegment {
			offset: 0x2000,
			vaddr: 0x2000,
			filesz: 0x200,
			memsz: 0x1000,
		}
	);

	assert_eq!(release_exit_code(image), Some(PAYLOAD_REJECT_EXIT_CODE));
	// 首个段随即申请分区; 主机上分区不交付, 装入在此终止。
	assert_eq!(alloc_calls(), 1);
}
