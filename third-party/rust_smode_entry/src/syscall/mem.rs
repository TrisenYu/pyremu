//! 内存类系统调用: mmap(222)。
//! brk(214) 逻辑仍在 crate::memory::sys_brk_handler，此处仅 re-export。

use crate::constants::{LEVEL_PAGE, PAGE_SHIFT, PAGE_SIZE, PTE_R, PTE_U, PTE_V, PTE_W, PTE_X, UMODE_MMAP_BASE};
use crate::context;
use crate::memory;
use crate::paging;

use super::ENOSYS;

// ---------------------------------------------------------------
//  mmap 常量 (Linux rv64 ABI)
// ---------------------------------------------------------------

const MAP_FIXED: u64 = 0x10;
const MAP_ANONYMOUS: u64 = 0x20;

const PROT_READ: u64 = 0x1;
const PROT_WRITE: u64 = 0x2;
const PROT_EXEC: u64 = 0x4;

// ---------------------------------------------------------------
//  mmap handler (222)
// ---------------------------------------------------------------

pub fn mmap_handler(addr: u64, len: u64, prot: u64, flags: u64, _fd: u64, _off: u64) -> u64 {
    // 仅支持匿名映射
    if (flags & MAP_ANONYMOUS) == 0 {
        return ENOSYS;
    }

    let pages = memory::page_up(len) >> PAGE_SHIFT;
    if pages == 0 {
        return ENOSYS;
    }

    let pa = memory::alloc_umode_page(pages);

    let va = if (flags & MAP_FIXED) != 0 {
        memory::page_down(addr)
    } else {
        let ctx = context::ctx_mut();
        if ctx.umode_mmap_base == 0 {
            ctx.umode_mmap_base = UMODE_MMAP_BASE;
        }
        let v = ctx.umode_mmap_base;
        ctx.umode_mmap_base = v + pages * PAGE_SIZE;
        v
    };

    let mut pte_flags: u8 = (PTE_U | PTE_V) as u8;
    if (prot & PROT_READ) != 0 {
        pte_flags |= PTE_R;
    }
    if (prot & PROT_WRITE) != 0 {
        pte_flags |= PTE_W;
    }
    if (prot & PROT_EXEC) != 0 {
        pte_flags |= PTE_X;
    }

    for i in 0..pages {
        paging::map_page(va + i * PAGE_SIZE, pa + i * PAGE_SIZE, pte_flags, LEVEL_PAGE);
    }

    va
}
