//! 全局 EnclaveContext —— 替代 ref-emod 中分散在 8+ 个 .c 文件里的 ~30 个 file-static 全局变量。
//! 启动阶段一次写入，此后任意读取。

use core::cell::UnsafeCell;
use core::mem::MaybeUninit;

use crate::constants::PAGE_SIZE;
use crate::paging::Pte;

// ---------------------------------------------------------------
//  PoolDesc
// ---------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub struct PoolDesc {
    pub offset: u64,
    pub size: u64,
    pub used_pages: u64,
}

impl PoolDesc {
    #[allow(dead_code)]
    pub const fn empty() -> Self {
        Self {
            offset: 0,
            size: 0,
            used_pages: 0,
        }
    }

    pub fn avail_bytes(&self) -> u64 {
        self.size.saturating_sub(self.used_pages * PAGE_SIZE)
    }
}

// ---------------------------------------------------------------
//  SharedBuf
// ---------------------------------------------------------------

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub struct SharedBuf {
    pub pa: u64,
    pub va: u64,
}

// ---------------------------------------------------------------
//  页表根 — 独立 static 强制 4 KiB 对齐
// ---------------------------------------------------------------

/// 页表根（512 项，必须 4 KiB 对齐）。
///
/// 独立 wrapper 结构体以 `#[repr(align(4096))]` 强制对齐,
/// **不依赖 BSS 布局** (BSS 中其他零初始化静态可能破坏
/// `EnclaveContext` 内嵌字段的 4 KiB 对齐约束, 导致
/// satp.PPN 指向错误物理页 → MMU 开启后全部取指页错误
/// → trap loop → hart halt).
#[repr(align(4096))]
#[allow(dead_code)]
pub struct PageTableRoot([Pte; 512]);

static mut PAGE_TABLE_ROOT: PageTableRoot = PageTableRoot([Pte(0); 512]);

// ---------------------------------------------------------------
//  EnclaveContext
// ---------------------------------------------------------------

pub struct EnclaveContext {
    /// 飞地管理器物理起始地址。
    pub manager_pa_start: u64,
    /// 下一个飞地模块的加载 VA。
    #[allow(dead_code)]
    pub enclave_module_load_va: u64,
    /// U-mode 堆顶。
    pub umode_heap_top: u64,
    /// mmap 匿名映射当前 VA 上限 (从 UMODE_MMAP_BASE 向上增长)。
    pub umode_mmap_base: u64,
    pub umode_pool: PoolDesc,
    pub smode_pool: PoolDesc,
    #[allow(dead_code)]
    pub shared_buffer: Option<SharedBuf>,
    pub umode_pool_pa_aligned: u64,
    /// 飞地时间配额 (timer interrupt 次数), 0=不限.
    pub time_quota: u64,
    /// 当前时间片内已消耗的 timer interrupt 次数.
    pub ticks_consumed: u64,
}

// ---------------------------------------------------------------
//  OnceCell —— 单次写入，多次读取
// ---------------------------------------------------------------

struct OnceCell<T> {
    value: UnsafeCell<MaybeUninit<T>>,
    ready: UnsafeCell<bool>,
}

unsafe impl<T> Sync for OnceCell<T> {}

impl<T> OnceCell<T> {
    const fn new() -> Self {
        Self {
            value: UnsafeCell::new(MaybeUninit::uninit()),
            ready: UnsafeCell::new(false),
        }
    }

    /// 逐字段初始化 — 完全避免大结构体的 memcpy。
    ///
    /// `f` 接收一个指向未初始化内存的 `*mut T` 裸指针,
    /// 调用方负责逐字段写入。适用于 CTX 已在 BSS 中全零、
    /// 仅需设置少量非零字段的场景。
    fn init_direct(&self, f: impl FnOnce(*mut T)) {
        let ready = unsafe { &mut *self.ready.get() };
        if *ready {
            panic!("EnclaveContext::init_direct called twice");
        }
        f(self.value.get() as *mut T);
        *ready = true;
    }

    #[allow(dead_code)]
    fn set(&self, val: T) {
        self.init_direct(|ptr| unsafe {
            // ptr::write 会逐字段移动, 对 [Pte; 512] 零值数组
            // 编译器应生成简单的循环而非 memcpy.
            core::ptr::write(ptr, val);
        });
    }

    fn get(&self) -> &T {
        assert!(
            *unsafe { &*self.ready.get() },
            "EnclaveContext not initialised"
        );
        unsafe { (*self.value.get()).assume_init_ref() }
    }

    #[allow(clippy::mut_from_ref)]
    fn get_mut(&self) -> &mut T {
        assert!(
            *unsafe { &*self.ready.get() },
            "EnclaveContext not initialised"
        );
        unsafe { (*self.value.get()).assume_init_mut() }
    }
}

static CTX: OnceCell<EnclaveContext> = OnceCell::new();

// ---------------------------------------------------------------
//  Public API
// ---------------------------------------------------------------

/// 初始化 CTX — 逐字段写入, 不依赖任何 memcpy.
/// CTX 位于 BSS 段, 启动时已全零, 故仅需设置非零字段.
/// PAGE_TABLE_ROOT 独立于 CTX 以 `#[repr(align(4096))]` 强制对齐.
pub fn init_context(man_pa_start: u64, enclave_module_load_va: u64) {
    CTX.init_direct(|ptr| unsafe {
        (*ptr).manager_pa_start = man_pa_start;
        (*ptr).enclave_module_load_va = enclave_module_load_va;
        (*ptr).time_quota = crate::constants::TIME_QUOTA;
        // 其余字段 = 0 (umode_heap_top, pools, ticks_consumed, shared_buffer, ...)
        // BSS 已保证全零, 无需显式赋值
    });
}

#[inline]
pub fn ctx() -> &'static EnclaveContext {
    CTX.get()
}

/// 返回页表根的物理地址 (始终 4 KiB 对齐).
///
/// `PAGE_TABLE_ROOT` 作为独立 `#[repr(align(4096))]` static,
/// 不依赖 BSS 内 `EnclaveContext` 的布局, 确保 MMU 使能后
/// satp.PPN 指向正确的物理页.
#[inline]
pub fn root_pa() -> u64 {
    &raw const PAGE_TABLE_ROOT as u64
}

#[inline]
pub fn ctx_mut() -> &'static mut EnclaveContext {
    CTX.get_mut()
}
