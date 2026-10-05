//! ELF64 加载

use crate::constants::*;
use crate::ecall_aux;
use crate::mem;
use crate::mem_prim::align::{align_down, align_up};
use crate::paging::{self, SetPageConfig};
use crate::println;

/// 载荷被拒绝时的退出码。
///
/// 低于 128: 被信号终止的进程退出码从 128 起算, 故本码与它们不相混。
const PAYLOAD_REJECT_EXIT_CODE: u64 = 125;

/// 载荷无法执行: 向宿主报告原因并终止本飞地。
///
/// 本路径在 U 模式入口之前执行, 飞地内没有可执行的载荷, 也没有恢复路径。宿主经
/// ENTER 或 RESUME 的返回值取到运行状态 exited-err 与本退出码, 据此收尾本次载荷
/// 并继续下一个。
pub fn reject_payload(reason: &str) -> ! {
	println!("{reason}\n");
	ecall_aux::enclave_call_exit(PAYLOAD_REJECT_EXIT_CODE)
}

/// 检查 ELF 格式 —— 对应 smode_entry 的 check_elf_format。
fn check_elf_format(
	elf_header: &elf::file::FileHeader<elf::endian::LittleEndian>,
	elf_size: u64,
) -> bool {
	if elf_header.class != elf::file::Class::ELF64
		|| elf_header.e_type != ET_EXEC
		|| elf_header.e_machine != EM_RISCV
	{
		return true;
	}

	let prog_head_end =
		elf_header.e_phoff + elf_header.e_phentsize as u64 * elf_header.e_phnum as u64;
	if elf_size < prog_head_end || prog_head_end < elf_header.e_phoff {
		return true;
	}

	let sect_head_end =
		elf_header.e_shoff + elf_header.e_shentsize as u64 * elf_header.e_shnum as u64;
	if elf_size < sect_head_end || sect_head_end < elf_header.e_shoff {
		return true;
	}

	false
}

/// Elf64_Word -> PTE flags —— 对应 set_pte_flag_via_elf64word。
fn set_pte_flag_via_elf64word(word: u32) -> u64 {
	let mut ret = PTE_V | PTE_U;
	if PF_R & word != 0 {
		ret |= PTE_R;
	}
	if PF_W & word != 0 {
		ret |= PTE_W;
	}
	if PF_X & word != 0 {
		ret |= PTE_X;
	}
	ret
}

/// 映射一页载荷段落。物理页申请失败或映射失败即无法继续加载本载荷。
fn map_or_reject(va: u64, pa: u64, flags: u64, level: u8) {
	if pa == !0u64 || paging::map_page(va, pa, flags, level).is_err() {
		reject_payload("elf: out of memory");
	}
}

/// 按 config 连续映射 `n_pages` 页, 未全部建成即无法继续加载本载荷。
fn map_range_or_reject(cfg: &SetPageConfig, n_pages: u64) {
	if paging::map_page_in_range(cfg, n_pages) < n_pages {
		reject_payload("elf: out of memory");
	}
}

/// 映射单个 PT_LOAD 段 + BSS 扩展。
unsafe fn map_one_segment(elf_paddr: u64, prog_header: elf::segment::ProgramHeader) {
	if prog_header.p_type != PT_LOAD {
		return;
	}

	let flags = set_pte_flag_via_elf64word(prog_header.p_flags);
	let ed_va = align_up(prog_header.p_vaddr + prog_header.p_filesz, PAGE_SIZE);

	// 文件内容部分
	let mut cfg = SetPageConfig {
		vbase: align_down(prog_header.p_vaddr, PAGE_SIZE),
		pbase: align_down(prog_header.p_offset + elf_paddr, PAGE_SIZE),
		flags,
		level: LEVEL_PAGE,
	};
	map_range_or_reject(&cfg, (ed_va - cfg.vbase) >> PAGE_SHIFT);

	// filesz >= memsz -> 无 BSS
	if prog_header.p_filesz >= prog_header.p_memsz {
		return;
	}

	// filesz 未进行页对齐时, bss 尾 [p_vaddr+p_filesz, align_up(..., PAGE_SIZE))
	// 可能会与末张文件页同页。
	// 该文件页映射自 M-mode 移交的扁平拷贝 (map 只映射、不复制文件内容), 后续
	// LOAD 段可能会复用同一张物理页。若就地清零会覆写共享扁平页。
	// 因此需要先对末尾的内存页做分配、然后复制本段文件前缀;
	// 从而确保后续的 bss 清零不会清空文件内容。
	let seg_data_end = prog_header.p_vaddr + prog_header.p_filesz;
	if seg_data_end & (PAGE_SIZE - 1) != 0 {
		let u_page = seg_data_end & !(PAGE_SIZE - 1);
		let file_page_pa = cfg.pbase + (u_page - cfg.vbase);
		let priv_pa = mem::alloc_umode_page(1);
		map_or_reject(u_page, priv_pa, flags, LEVEL_PAGE);
		unsafe {
			core::ptr::copy_nonoverlapping(
				(file_page_pa + LINEAR_MAP_OFFSET) as *const u8,
				u_page as *mut u8,
				(seg_data_end - u_page) as usize,
			);
		}
	}

	let bss_va = prog_header.p_vaddr + prog_header.p_filesz;

	// bss 的页跨度按虚拟区间 [bss_va, p_vaddr + p_memsz) 计算, 而不是
	// align_up(memsz) - align_up(filesz) 这个"长度差": 后者在 filesz 与 memsz
	// 同处一页时恒为 0, 会漏掉跨页的 bss 尾部 —— 该页从未分配, 随后的清零
	// 便落在未映射页上, 在 S 模式触发 store page fault 并冻结飞地.
	let diff =
		align_up(prog_header.p_vaddr + prog_header.p_memsz, PAGE_SIZE) - align_up(bss_va, PAGE_SIZE);

	// 小段：从 U-mode 页池分配 -> memset -> return
	if diff < CHUNK_2M_SIZE {
		let n = diff >> PAGE_SHIFT;
		cfg.vbase = ed_va;
		cfg.pbase = mem::alloc_umode_page(n);
		map_range_or_reject(&cfg, n);
		unsafe {
			core::ptr::write_bytes(
				bss_va as *mut u8,
				0,
				(prog_header.p_memsz - prog_header.p_filesz) as usize,
			);
		}
		return;
	}

	// 大段：向 M-mode 申请 CHUNK_2M
	let ed_va_align = align_up(ed_va, CHUNK_2M_SIZE);
	let gap = ed_va_align - ed_va;

	if gap > 0 {
		let (_, gap_pa) = ecall_aux::enclave_call_mem_alloc(1);
		cfg.vbase = ed_va;
		cfg.pbase = gap_pa + (ed_va % CHUNK_2M_SIZE);
		map_range_or_reject(&cfg, gap >> PAGE_SHIFT);
	}

	let mut left = diff >> CHUNK_2M_SHIFT;
	let mut va = ed_va_align;

	while left > 0 {
		let (n, pa) = ecall_aux::enclave_call_mem_alloc(left);
		if n == 0 {
			break;
		}
		map_range_or_reject(
			&SetPageConfig {
				vbase: va,
				pbase: pa,
				flags,
				level: LEVEL_MEGA,
			},
			n,
		);
		left -= n;
		va += n * CHUNK_2M_SIZE;
	}

	unsafe {
		core::ptr::write_bytes(
			bss_va as *mut u8,
			0,
			(prog_header.p_memsz - prog_header.p_filesz) as usize,
		);
	}
}

/// 载荷加载结果。除入口地址外, 还携带程序头表在载荷地址空间中的位置,
/// 供调用方填写 auxv 的 AT_PHDR / AT_PHENT / AT_PHNUM。
///
/// musl 静态链接版的 dl_iterate_phdr (vendor/musl/src/ldso/dl_iterate_phdr.c)
/// 直接以 auxv 的三项为唯一数据源遍历程序头; libunwind 又经它定位
/// PT_GNU_EH_FRAME (.eh_frame_hdr) 以取得展开表。auxv 缺这三项时遍历条目数为 0,
/// libunwind 找不到任何 FDE, _Unwind_RaiseException 判定已到栈底,
/// __cxa_throw 随即调用 std::terminate —— 载荷内任何一个 C++ 异常都无法被捕获。
pub struct PayloadInfo {
	pub entry: u64,
	/// 程序头表首地址 (AT_PHDR), 为载荷地址空间中的 VA。
	pub phdr_va: u64,
	/// 单个程序头条目字节数 (AT_PHENT)。
	pub phentsize: u64,
	/// 程序头条目数 (AT_PHNUM)。
	pub phnum: u64,
}

/// 求程序头表自身的 VA。程序头表位于文件的 [e_phoff, e_phoff + phnum*phentsize)
/// 区间, 该区间必含于某个 PT_LOAD 段内; 按该段的 p_vaddr 与 p_offset 平移即得
/// 它在载荷地址空间中的地址。载荷按链接地址原样加载且未做重定位, 故该平移成立。
/// 未找到覆盖段时返回 0, 调用方据此跳过 AT_PHDR 的填写。
fn phdr_va_of(elf_bytes: &elf::ElfBytes<elf::endian::LittleEndian>) -> u64 {
	let e_phoff = elf_bytes.ehdr.e_phoff;
	let Some(segments) = elf_bytes.segments() else {
		return 0;
	};
	for prog_header in segments.iter() {
		if prog_header.p_type != PT_LOAD || e_phoff < prog_header.p_offset {
			continue;
		}
		if e_phoff >= prog_header.p_offset + prog_header.p_filesz {
			continue;
		}
		return prog_header.p_vaddr + (e_phoff - prog_header.p_offset);
	}
	0
}

/// 加载 ELF 到 U-mode。成功返回入口地址与程序头表信息。
pub fn load_elf(elf_pa: u64, elf_size: u64) -> PayloadInfo {
	let elf_va = elf_pa.wrapping_add(LINEAR_MAP_OFFSET);
	let data = unsafe { core::slice::from_raw_parts(elf_va as *const u8, elf_size as usize) };

	let elf_bytes = elf::ElfBytes::<elf::endian::LittleEndian>::minimal_parse(data)
		.unwrap_or_else(|_| reject_payload("elf: parse failed"));

	if check_elf_format(&elf_bytes.ehdr, elf_size) {
		reject_payload("elf: check failed");
	}

	if let Some(segments) = elf_bytes.segments() {
		for prog_header in segments.iter() {
			unsafe { map_one_segment(elf_pa, prog_header) };
		}
	}

	PayloadInfo {
		entry: elf_bytes.ehdr.e_entry,
		phdr_va: phdr_va_of(&elf_bytes),
		phentsize: elf_bytes.ehdr.e_phentsize as u64,
		phnum: elf_bytes.ehdr.e_phnum as u64,
	}
}
