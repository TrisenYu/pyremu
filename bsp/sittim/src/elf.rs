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

/// 按 config 连续映射 `n_pages` 页, 未全部建成即无法继续加载本载荷。
fn map_range_or_reject(cfg: &SetPageConfig, n_pages: u64) {
	if paging::map_page_in_range(cfg, n_pages) < n_pages {
		reject_payload("elf: out of memory");
	}
}

/// 段的虚拟区间按页取整后的跨度, 即该段在地址空间内占用的字节数。
fn segment_mapped_size(prog_header: &elf::segment::ProgramHeader) -> u64 {
	let va_start = align_down(prog_header.p_vaddr, PAGE_SIZE);
	align_up(prog_header.p_vaddr + prog_header.p_memsz, PAGE_SIZE) - va_start
}

/// 核对各 PT_LOAD 段按页取整后的虚拟区间两两不相交。
///
/// 每段的内存取自各自申请的分区, 段的页区间相交时, 后装入的段会改写同一张页的叶子
/// 项, 先装入的段落在该页内的字节随即失去映射: 两个段各有自己的一份文件内容, 一张
/// 页只承载其中一个。故相交的布局在装入前即拒收, 不留下列入内存却无映射的字节。
///
/// 段的声明对齐只保证 p_vaddr 与 p_offset 对 p_align 同余, p_align 大于页大小时
/// 段与段可以相接于页内, 故此处按页取整后比较。
fn check_segments_page_disjoint(
	segments: elf::segment::SegmentTable<'_, elf::endian::LittleEndian>,
) -> bool {
	for (i, first) in segments.iter().enumerate() {
		if first.p_type != PT_LOAD || first.p_memsz == 0 {
			continue;
		}
		let first_start = align_down(first.p_vaddr, PAGE_SIZE);
		let first_end = first_start + segment_mapped_size(&first);
		for second in segments.iter().skip(i + 1) {
			if second.p_type != PT_LOAD || second.p_memsz == 0 {
				continue;
			}
			let second_start = align_down(second.p_vaddr, PAGE_SIZE);
			let second_end = second_start + segment_mapped_size(&second);
			if first_start < second_end && second_start < first_end {
				return false;
			}
		}
	}
	true
}

/// 加载单个 PT_LOAD 段: 申请覆盖该段全部内存的连续分区, 拷入文件内容, 清零 BSS
/// 部分, 并映射到段的虚拟地址。
///
/// 分区类别取段标志: 会被取指的段 (PF_X) 归指令类, 其余归数据类。一个 2 MiB 分区
/// 整体只承载一个类别, 故本段内存 (含 BSS) 同属一类。
///
/// 类别即权限, 指令类的分区只有读与执行权限, S 模式无法写入。故两类一律先按数据类
/// 取得, 文件内容与 BSS 写入完毕、整块改判为指令类之后, 再按段标志位收回写权限。
///
/// `elf_va_base` 是载荷映像经线性别名 (PA + `LINEAR_MAP_OFFSET`) 的起始地址,
/// 文件内容自该处读取。
unsafe fn map_one_segment(elf_va_base: u64, prog_header: elf::segment::ProgramHeader) {
	if prog_header.p_type != PT_LOAD || prog_header.p_memsz == 0 {
		return;
	}

	let is_instr_class = prog_header.p_flags & PF_X != 0;
	let flags = set_pte_flag_via_elf64word(prog_header.p_flags);
	// 指令类的分区只登记读与执行权限, 故可取指的段在改判之后不保留写权限。
	let final_flags = if is_instr_class { flags & !PTE_W } else { flags };

	// 段的虚拟区间按页取整后整体申请: filesz 未按页对齐时, 末张文件页的剩余部分
	// 由 BSS 占据, 与文件内容同属本段, 故一并落在本段的分区内。
	let va_start = align_down(prog_header.p_vaddr, PAGE_SIZE);
	let pages = segment_mapped_size(&prog_header) >> PAGE_SHIFT;
	let pa = mem::alloc_adjacent_pages(pages, ENCLAVE_MEM_ATTR_DATA);
	if pa == !0u64 {
		reject_payload("elf: out of memory");
	}
	// 写入期间按读与写权限映射: 段标志位不含写权限时, 按段标志位映射的页无法承接
	// 文件内容与 BSS 的写入, 而写访问要求页表项的读位与写位同时置位。
	map_range_or_reject(
		&SetPageConfig {
			vbase: va_start,
			pbase: pa,
			flags: flags | PTE_R | PTE_W,
			level: LEVEL_PAGE,
		},
		pages,
	);

	// 文件内容按整页拷入: 段的文件数据自 p_offset 起算, 与 p_vaddr 对页大小同余
	// (链接器保证), 故两侧同按页对齐后逐字节对应。
	let file_end_va = align_up(prog_header.p_vaddr + prog_header.p_filesz, PAGE_SIZE);
	if file_end_va > va_start {
		let src_va = elf_va_base.wrapping_add(align_down(prog_header.p_offset, PAGE_SIZE));
		unsafe {
			core::ptr::copy_nonoverlapping(
				src_va as *const u8,
				va_start as *mut u8,
				(file_end_va - va_start) as usize,
			);
		}
	}

	// BSS 部分清零。本段的分区由本段独占, 清零不会改动其它段或文件内容。
	if prog_header.p_memsz > prog_header.p_filesz {
		unsafe {
			core::ptr::write_bytes(
				(prog_header.p_vaddr + prog_header.p_filesz) as *mut u8,
				0,
				(prog_header.p_memsz - prog_header.p_filesz) as usize,
			);
		}
	}

	// 写入完毕后整块改判为指令类: 改判一经返回, 该分区只登记读与执行权限。
	if is_instr_class
		&& !mem::set_partition_attr(pa, pages << PAGE_SHIFT, ENCLAVE_MEM_ATTR_INSTR)
	{
		reject_payload("elf: classify segment failed");
	}
	// 页表项按段标志位收回写权限, 与该分区的权限位一致。
	for i in 0..pages {
		if !paging::set_leaf_flags(va_start + (i << PAGE_SHIFT), final_flags) {
			reject_payload("elf: protect segment failed");
		}
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
		if !check_segments_page_disjoint(segments) {
			reject_payload("elf: segments share a page");
		}
		// 各段的页区间互不相交, 装入顺序不影响结果: 每段映射自己的分区。
		for prog_header in segments {
			unsafe { map_one_segment(elf_va, prog_header) };
		}
	}

	PayloadInfo {
		entry: elf_bytes.ehdr.e_entry,
		phdr_va: phdr_va_of(&elf_bytes),
		phentsize: elf_bytes.ehdr.e_phentsize as u64,
		phnum: elf_bytes.ehdr.e_phnum as u64,
	}
}
