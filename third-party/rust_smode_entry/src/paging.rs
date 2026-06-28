//! Sv39 页表管理。逻辑紧跟 ref-emod/emod_manager/memory/page_table.c，
//! 但用卫语句替代原 C 的深层嵌套。
//!
//! 关键约定：
//! - MMU 开启后（satp != 0），中间页表指针通过 LINEAR_MAP_OFFSET 换算。
//! - A/D 位始终置 1（无 swap）。
//! - map / unmap 后 sfence.vma 刷新对应 VA。

use crate::constants::*;
use crate::context;
use crate::csr;
use crate::hang;
use crate::memory;

// ---------------------------------------------------------------
//  Sv39 PTE
// ---------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct Pte(pub u64);

impl Pte {
    pub const fn empty() -> Self {
        Self(0)
    }

    /// R | W | X 任一置位即为叶子节点。
    pub fn is_leaf(&self) -> bool {
        self.0 & ((PTE_R | PTE_W | PTE_X) as u64) != 0
    }

    /// 提取 PPN（bits [53:10]）。
    pub fn ppn(&self) -> u64 {
        (self.0 >> 10) & 0xF_FFFF_FFFF
    }

    /// 写入 PPN，保留低 10 位标志。
    pub fn set_ppn(&mut self, ppn: u64) {
        self.0 = (self.0 & 0x3FF) | ((ppn & 0xF_FFFF_FFFF) << 10);
    }

    /// 写入标志位，自动置 A/D。
    pub fn set_flags(&mut self, flags: u8) {
        self.0 |= flags as u64;
        self.0 |= (PTE_A | PTE_D) as u64;
    }

    pub fn clear(&mut self) {
        self.0 = 0;
    }
}

// ---------------------------------------------------------------
//  VA 辅助
// ---------------------------------------------------------------

/// level 0 → VPN[2] (bits 38:30), 1 → VPN[1] (29:21), 2 → VPN[0] (20:12)
#[inline]
fn get_vpn(va: u64, level: u8) -> usize {
    match level {
        0 => ((va >> 30) & 0x1FF) as usize,
        1 => ((va >> 21) & 0x1FF) as usize,
        2 => ((va >> 12) & 0x1FF) as usize,
        _ => 0,
    }
}

/// 物理地址 → PTE 指针。satp 有效时叠加 LINEAR_MAP_OFFSET。
#[inline]
unsafe fn pte_ptr(pa: u64) -> *mut Pte {
    if csr::read_satp() != 0 {
        pa.wrapping_add(LINEAR_MAP_OFFSET) as *mut Pte
    } else {
        pa as *mut Pte
    }
}

// ---------------------------------------------------------------
//  页表遍历
// ---------------------------------------------------------------

/// 定位给定 VA 在指定 Sv39 层级的叶子 PTE。
/// `alloc=true` 时缺失的中间页表从 S-mode page pool 分配。
fn get_leaf_pte(vaddr: u64, level: u8, alloc: bool) -> &'static mut Pte {
    let root_pa = context::ctx().page_table_root.as_ptr() as u64;
    let mut table = unsafe { pte_ptr(root_pa) };

    // 从 GIGA 向下遍历到比目标高一级
    for i in (0..=2).rev() {
        if i <= level {
            continue; // 已到目标层级，退出后命中目标
        }

        let pte = unsafe { &mut *table.add(get_vpn(vaddr, i)) };

        if pte.0 & (PTE_V as u64) == 0 {
            if !alloc {
                hang::hang_with_msg("get_leaf_pte: missing intermediate table\n");
            }
            let next_pa = memory::alloc_smode_page(1);
            pte.set_ppn(next_pa >> PAGE_SHIFT);
            pte.0 |= PTE_V as u64;
            table = unsafe { pte_ptr(next_pa) };
            continue;
        }

        if pte.is_leaf() {
            hang::hang_with_msg("get_leaf_pte: unexpected super-page\n");
        }

        let next_pa = pte.ppn() << PAGE_SHIFT;
        table = unsafe { pte_ptr(next_pa) };
    }

    // 命中目标层级
    let pte = unsafe { &mut *table.add(get_vpn(vaddr, level)) };
    if pte.0 & (PTE_V as u64) == 0 && !alloc {
        hang::hang_with_msg("get_leaf_pte: target PTE not found\n");
    }
    pte
}

// ---------------------------------------------------------------
//  map / unmap
// ---------------------------------------------------------------

pub fn map_page(vaddr: u64, paddr: u64, flags: u8, level: u8) {
    let ppn = paddr >> PAGE_SHIFT;
    let pte = get_leaf_pte(vaddr, level, true);

    pte.clear();
    pte.set_ppn(ppn);
    pte.set_flags(flags);
    pte.0 |= PTE_V as u64;

    unsafe { core::arch::asm!("sfence.vma {0}, zero", in(reg) vaddr) };
}

#[allow(dead_code)]
pub fn unmap_page(vaddr: u64, level: u8) {
    get_leaf_pte(vaddr, level, false).clear();
    unsafe { core::arch::asm!("sfence.vma {0}, zero", in(reg) vaddr) };
}

// ---------------------------------------------------------------
//  VA → PA 翻译
// ---------------------------------------------------------------

struct WalkResult {
    level: i8,
    pte: Pte,
}

/// 三层 Sv39 页表遍历。未命中时 level = -1。
fn walk_page_table(va: u64) -> WalkResult {
    let idxs: [usize; 3] = [
        ((va & 0x7F_C000_0000) >> 30) as usize,
        ((va & 0x00_3FE0_0000) >> 21) as usize,
        ((va & 0x00_001F_F000) >> 12) as usize,
    ];

    let root_pa = context::ctx().page_table_root.as_ptr() as u64;
    let mut table = unsafe { pte_ptr(root_pa) };

    for i in 0..3 {
        let pte = unsafe { &*table.add(idxs[i]) };

        if pte.0 & (PTE_V as u64) == 0 {
            return WalkResult {
                level: -1,
                pte: Pte::empty(),
            };
        }

        if pte.is_leaf() {
            return WalkResult {
                level: i as i8,
                pte: *pte,
            };
        }

        let next_pa = pte.ppn() << PAGE_SHIFT;
        table = unsafe { pte_ptr(next_pa) };
    }

    WalkResult {
        level: -1,
        pte: Pte::empty(),
    }
}

/// VA → PA。未映射返回 None。
pub fn get_pa(va: u64) -> Option<u64> {
    let r = walk_page_table(va);
    if r.level < 0 {
        return None;
    }

    let base = r.pte.ppn() << PAGE_SHIFT;
    let offset_bits = PAGE_SHIFT as usize + (2 - r.level as usize) * SV39_VPN_LEN as usize;
    Some(base | ((1_u64 << offset_bits) - 1) & va)
}

// ---------------------------------------------------------------
//  初始化
// ---------------------------------------------------------------

/// 为 MMU 启用的瞬间建立恒等映射 (PA=VA)。
///
/// `csrw satp` 后 PC 仍在低物理地址，必须有一条 PA→PA 的映射
/// 让 CPU 能继续取指，直到代码通过高 VA 访问 trampoline。
pub fn identity_map_trampoline(pa: u64) {
    let base = crate::memory::chunk_2m_down(pa);
    map_page(base, base, PTE_R | PTE_W | PTE_X, LEVEL_MEGA);
}

/// 构建 satp 值（Sv39 模式，ASID=0）。
pub fn init_satp(root_pa: u64) -> u64 {
    let ppn = root_pa >> PAGE_SHIFT;
    (ppn & 0xF_FFFF_FFFF) | (8_u64 << 60)
}

/// 1 GiB 超级页线性映射。PA [LINEAR_MAP_START, +SIZE) → VA (PA + OFFSET)。
pub fn setup_linear_map() {
    let giga = 0x4000_0000_u64;
    let pages = LINEAR_MAP_SIZE.div_ceil(giga);
    let mut pa = LINEAR_MAP_START;

    for _ in 0..pages {
        let va = pa.wrapping_add(LINEAR_MAP_OFFSET);
        map_page(va, pa, PTE_R | PTE_W, LEVEL_GIGA);
        pa += giga;
    }

    unsafe { core::arch::asm!("sfence.vma") };
}

// ---------------------------------------------------------------
//  批量映射辅助（对应 smode_entry 的 set_page_config + map_page_in_range）
// ---------------------------------------------------------------

/// 页映射参数集合。将 VA 基址、PA 基址、标志、层级打包为单一参数，
/// 避免在调用方展开冗长的 for 循环。
pub struct SetPageConfig {
    pub vbase: u64,
    pub pbase: u64,
    pub flags: u8,
    pub level: u8,
}

/// 按 config 连续映射 `n_pages` 页。
pub fn map_page_in_range(cfg: &SetPageConfig, n_pages: u64) {
    for i in 0..n_pages {
        map_page(
            cfg.vbase + i * PAGE_SIZE,
            cfg.pbase + i * PAGE_SIZE,
            cfg.flags,
            cfg.level,
        );
    }
}
