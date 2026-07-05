//! ELF64 加载。紧跟 third-party/smode_entry/load_elf.c 的结构。

use crate::constants::*;
use crate::hang;
use crate::memory;
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

/// 映射单个 PT_LOAD 段 + BSS 扩展。
unsafe fn map_one_segment(elf_paddr: u64, prog_header: elf::segment::ProgramHeader) {
    if prog_header.p_type != PT_LOAD {
        return;
    }

    let flags = set_pte_flag_via_elf64word(prog_header.p_flags);
    let ed_va = memory::page_up(prog_header.p_vaddr + prog_header.p_filesz);

    // 文件内容部分
    let mut cfg = SetPageConfig {
        vbase: memory::page_down(prog_header.p_vaddr),
        pbase: memory::page_down(prog_header.p_offset + elf_paddr),
        flags,
        level: LEVEL_PAGE,
    };
    paging::map_page_in_range(&cfg, (ed_va - cfg.vbase) >> PAGE_SHIFT);

    // filesz >= memsz -> 无 BSS
    if prog_header.p_filesz >= prog_header.p_memsz {
        return;
    }

    let diff = memory::page_up(prog_header.p_memsz) - memory::page_up(prog_header.p_filesz);

    // 小段：从 U-mode 页池分配 -> memset -> return
    if diff < CHUNK_2M_SIZE {
        let n = diff >> PAGE_SHIFT;
        cfg.vbase = ed_va;
        cfg.pbase = memory::alloc_umode_page(n);
        paging::map_page_in_range(&cfg, n);
        unsafe {
            core::ptr::write_bytes(
                (prog_header.p_vaddr + prog_header.p_filesz) as *mut u8,
                0,
                (prog_header.p_memsz - prog_header.p_filesz) as usize,
            );
        }
        return;
    }

    // 大段：向 M-mode 申请 CHUNK_2M
    let ed_va_align = memory::chunk_2m_up(ed_va);
    let gap = ed_va_align - ed_va;

    if gap > 0 {
        let (_, gap_pa) = crate::call::enclave_call_mem_alloc(1);
        cfg.vbase = ed_va;
        cfg.pbase = gap_pa + (ed_va % CHUNK_2M_SIZE);
        paging::map_page_in_range(&cfg, gap >> PAGE_SHIFT);
    }

    let mut left = diff >> CHUNK_2M_SHIFT;
    let mut va = ed_va_align;

    while left > 0 {
        let (n, pa) = crate::call::enclave_call_mem_alloc(left);
        if n == 0 {
            break;
        }
        paging::map_page_in_range(
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
            (prog_header.p_vaddr + prog_header.p_filesz) as *mut u8,
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
        .unwrap_or_else(|_| hang::hang_with_msg("elf: parse failed\n"));

    if check_elf_format(&elf_bytes.ehdr, elf_size) {
        hang::hang_with_msg("elf: check failed\n");
    }

    if let Some(segments) = elf_bytes.segments() {
        for prog_header in segments.iter() {
            unsafe { map_one_segment(elf_pa, prog_header) };
        }
    }

    elf_bytes.ehdr.e_entry
}
