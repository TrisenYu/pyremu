//! 内存类系统调用: mmap(222) / munmap(215) / mremap(216)。
//! brk(214) 逻辑仍在 crate::mem::sys_brk_handler，此处仅 re-export。

use crate::constants::{
    CHUNK_2M_SIZE, LEVEL_GIGA, LEVEL_MEGA, LEVEL_PAGE, PAGE_SHIFT, PAGE_SIZE, PTE_R, PTE_U,
    PTE_V, PTE_W, PTE_X,
};
use crate::mem;
use crate::paging;

use super::{EINVAL, ENOMEM, ENOSYS};

// ---------------------------------------------------------------
//  mmap 常量 (Linux rv64 ABI)
// ---------------------------------------------------------------

const MAP_FIXED: u64 = 0x10;
const MAP_ANONYMOUS: u64 = 0x20;

const PROT_READ: u64 = 0x1;
const PROT_WRITE: u64 = 0x2;
const PROT_EXEC: u64 = 0x4;

const MREMAP_MAYMOVE: u64 = 0x1;
const MREMAP_FIXED: u64 = 0x2;

/// prot 位图 -> PTE 标志。PROT_NONE 得到一个 R/W/X 全 0 的叶子项:
/// 硬件在末级遇到此编码即报页错误, 正是 PROT_NONE 的语义。
fn prot_to_pte(prot: u64) -> u8 {
    let mut flags: u8 = (PTE_U | PTE_V) as u8;
    if (prot & PROT_READ) != 0 {
        flags |= PTE_R;
    }
    if (prot & PROT_WRITE) != 0 {
        flags |= PTE_W;
    }
    if (prot & PROT_EXEC) != 0 {
        flags |= PTE_X;
    }
    flags
}

// ---------------------------------------------------------------
//  mmap handler (222)
// ---------------------------------------------------------------

pub fn mmap_handler(addr: u64, len: u64, prot: u64, flags: u64, _fd: u64, _off: u64) -> u64 {
    // 仅支持匿名映射
    if (flags & MAP_ANONYMOUS) == 0 {
        return ENOSYS;
    }

    let pages = mem::page_up(len) >> PAGE_SHIFT;
    if pages == 0 {
        return EINVAL;
    }
    let total_bytes = pages * PAGE_SIZE;
    let pte_flags = prot_to_pte(prot);

    // MAP_FIXED: 调用方指定地址, 按 4 KiB 粒度就地映射。
    if (flags & MAP_FIXED) != 0 {
        let va = mem::page_down(addr);
        if !mem::map_umode_pages(va, pages, pte_flags) {
            return ENOMEM;
        }
        return va;
    }

    // 常规路径: 由映射区的页计数决定放置与粒度 (见 mem::take_mmap_region)。
    let va = mem::take_mmap_region(total_bytes, pte_flags);
    if va == 0 {
        return ENOMEM;
    }
    va
}

// ---------------------------------------------------------------
//  munmap handler (215)
// ---------------------------------------------------------------

/// 解除 [addr, addr + len) 的 VA 映射, 使后续访问落入 access fault 由 S/M 侧裁决。
///
/// 物理页不在本函数归还: umode 页池是 bump 分配器 (无单页归还能力), 2 MiB 池内存
/// 统一由 M-mode 在飞地 shutdown 时经 clear_enclave_mem_region 整块回收。此处只负责
/// 清空页表项并刷新翻译缓存, 与运行时的「不信任应用 free、shutdown 整体回收」约定一致。
pub fn munmap_handler(addr: u64, len: u64) -> u64 {
    // 参数合法性: Linux munmap 要求地址 4 KiB 对齐且长度非零, 违者 EINVAL。
    if addr & (PAGE_SIZE - 1) != 0 || len == 0 {
        return EINVAL;
    }

    let start = mem::page_down(addr);
    let end = mem::page_up(addr + len);

    // 逐段解除映射, 粒度跟随页表叶子层级:
    //   4 KiB 普通页单页解除; 2 MiB / 1 GiB 超页整块解除并跳过整块,
    //   避免在超页内部再次解引用 (超页之下不存在叶子项)。
    let giga_size = 0x4000_0000_u64;
    let mut va = start;
    while va < end {
        match paging::va_real_page_level(va) {
            None => va += PAGE_SIZE, // 本就未映射, 直接跳过
            Some(LEVEL_PAGE) => {
                paging::unmap_page(va, LEVEL_PAGE);
                va += PAGE_SIZE;
            }
            Some(LEVEL_MEGA) => {
                paging::unmap_page(va, LEVEL_MEGA);
                va += CHUNK_2M_SIZE;
            }
            Some(LEVEL_GIGA) => {
                paging::unmap_page(va, LEVEL_GIGA);
                va += giga_size;
            }
            Some(_) => va += PAGE_SIZE, // 未知层级不应出现, 保守跳过
        }
    }

    0
}

// ---------------------------------------------------------------
//  mremap handler (216)
// ---------------------------------------------------------------

/// mremap(216)。musl 的 realloc 对 mmap 组用 MREMAP_MAYMOVE 扩容。
///
/// 旧实现把它并入「跳过 (无副作用)」一律返回 0, 而 0 在指针语义下是 NULL,
/// realloc 据此判失败 —— 增长型缓冲的使用方 (如 cJSON_Print 的打印缓冲)
/// 直接拿到 NULL。
pub fn mremap_handler(old_addr: u64, old_size: u64, new_size: u64, flags: u64, new_addr: u64) -> u64 {
    if old_size == 0 || new_size == 0 {
        return EINVAL;
    }

    // 原地扩容/收缩: 判断条件与页映射都在 memory 侧 (需要映射区的页计数与后备
    // 块物理基址)。
    if (flags & MREMAP_FIXED) == 0
        && mem::grow_mmap_in_place(old_addr, old_size, new_size)
    {
        return old_addr;
    }

    if (flags & MREMAP_MAYMOVE) == 0 {
        return ENOMEM;
    }

    // 另择地址: 复制仍有效的字节数, 再解除旧映射 (物理页不归还)。
    let dst = if (flags & MREMAP_FIXED) != 0 {
        let va = mem::page_down(new_addr);
        let pages = mem::page_up(new_size) >> PAGE_SHIFT;
        if !mem::map_umode_pages(va, pages, (PTE_U | PTE_V | PTE_R | PTE_W) as u8) {
            return ENOMEM;
        }
        va
    } else {
        let va = mem::take_mmap_region(new_size, (PTE_U | PTE_V | PTE_R | PTE_W) as u8);
        if va == 0 {
            return ENOMEM;
        }
        va
    };

    let copy_len = if old_size < new_size { old_size } else { new_size };
    unsafe {
        core::ptr::copy_nonoverlapping(
            old_addr as *const u8,
            dst as *mut u8,
            copy_len as usize,
        );
    }

    // MREMAP_FIXED 的目标由调用方指定, 旧区间保留由调用方自理。
    if (flags & MREMAP_FIXED) == 0 {
        munmap_handler(old_addr, old_size);
    }

    dst
}
