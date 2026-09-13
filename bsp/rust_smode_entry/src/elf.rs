//! ELF64 加载

use crate::constants::*;
use crate::ecall_aux;
use crate::hang;
use crate::mem;
use crate::paging::{self, SetPageConfig};

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
fn set_pte_flag_via_elf64word(word: u32) -> u8 {
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

/// 载荷段落映射。失败即无法继续加载 (载荷超出可用内存, 或页池/页表耗尽),
/// 以 fail-stop 收场并放开中断, 使宿主仍可下发终止请求拆掉该飞地。
fn map_or_halt(va: u64, pa: u64, flags: u8, level: u8) {
	if pa == !0u64 || paging::map_page(va, pa, flags, level).is_err() {
		hang::fault_halt("load_elf: out of memory\n");
	}
}

/// 按 config 连续映射 `n_pages` 页, 未全部建成即 fail-stop。
fn map_range_or_halt(cfg: &SetPageConfig, n_pages: u64) {
	if paging::map_page_in_range(cfg, n_pages) < n_pages {
		hang::fault_halt("load_elf: out of memory\n");
	}
}

/// 映射单个 PT_LOAD 段 + BSS 扩展。
unsafe fn map_one_segment(elf_paddr: u64, prog_header: elf::segment::ProgramHeader) {
	if prog_header.p_type != PT_LOAD {
		return;
	}

	let flags = set_pte_flag_via_elf64word(prog_header.p_flags);
	let ed_va = mem::page_up(prog_header.p_vaddr + prog_header.p_filesz);

	// 文件内容部分
	let mut cfg = SetPageConfig {
		vbase: mem::page_down(prog_header.p_vaddr),
		pbase: mem::page_down(prog_header.p_offset + elf_paddr),
		flags,
		level: LEVEL_PAGE,
	};
	map_range_or_halt(&cfg, (ed_va - cfg.vbase) >> PAGE_SHIFT);

	// filesz >= memsz -> 无 BSS
	if prog_header.p_filesz >= prog_header.p_memsz {
		return;
	}

	// filesz 未进行页对齐时, bss 尾 [p_vaddr+p_filesz, page_up(...))
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
		map_or_halt(u_page, priv_pa, flags, LEVEL_PAGE);
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
	// page_up(memsz) - page_up(filesz) 这个"长度差": 后者在 filesz 与 memsz
	// 同处一页时恒为 0, 会漏掉跨页的 bss 尾部 —— 该页从未分配, 随后的清零
	// 便落在未映射页上, 在 S 模式触发 store page fault 并冻结飞地.
	let diff =
		mem::page_up(prog_header.p_vaddr + prog_header.p_memsz) - mem::page_up(bss_va);

	// 小段：从 U-mode 页池分配 -> memset -> return
	if diff < CHUNK_2M_SIZE {
		let n = diff >> PAGE_SHIFT;
		cfg.vbase = ed_va;
		cfg.pbase = mem::alloc_umode_page(n);
		map_range_or_halt(&cfg, n);
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
	let ed_va_align = mem::chunk_2m_up(ed_va);
	let gap = ed_va_align - ed_va;

	if gap > 0 {
		let (_, gap_pa) = ecall_aux::enclave_call_mem_alloc(1);
		cfg.vbase = ed_va;
		cfg.pbase = gap_pa + (ed_va % CHUNK_2M_SIZE);
		map_range_or_halt(&cfg, gap >> PAGE_SHIFT);
	}

	let mut left = diff >> CHUNK_2M_SHIFT;
	let mut va = ed_va_align;

	while left > 0 {
		let (n, pa) = ecall_aux::enclave_call_mem_alloc(left);
		if n == 0 {
			break;
		}
		map_range_or_halt(
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

/// 加载 ELF 到 U-mode。成功返回入口 VA。
pub fn load_elf(elf_pa: u64, elf_size: u64) -> u64 {
	let elf_va = elf_pa.wrapping_add(LINEAR_MAP_OFFSET);
	let data = unsafe { core::slice::from_raw_parts(elf_va as *const u8, elf_size as usize) };

	let elf_bytes = elf::ElfBytes::<elf::endian::LittleEndian>::minimal_parse(data)
		.unwrap_or_else(|_| hang::fault_halt("elf: parse failed\n"));

	if check_elf_format(&elf_bytes.ehdr, elf_size) {
		hang::fault_halt("elf: check failed\n");
	}

	if let Some(segments) = elf_bytes.segments() {
		for prog_header in segments.iter() {
			unsafe { map_one_segment(elf_pa, prog_header) };
		}
	}

	elf_bytes.ehdr.e_entry
}
