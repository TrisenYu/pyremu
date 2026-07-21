//! 飞地 S-mode 运行时入口。两阶段启动，以 MMU 是否使能为界。

#![no_std]
#![no_main]

mod call;
mod constants;
mod context;
mod attest;
mod csr;
mod elf;
mod hang;
mod memory;
mod paging;
mod println;
mod string;
mod syscall;
mod trap;
mod uart;

use core::panic::PanicInfo;

use crate::constants::*;

unsafe extern "C" {
    static _end: u8;
}

core::arch::global_asm!(include_str!("entry.s"));

// ---------------------------------------------------------------
//  阶段一：MMU 使能前 —— 构建页表
// ---------------------------------------------------------------

/// 阶段一返回给 entry.s 的启动信息。`repr(C)` 确保与汇编的
/// 寄存器约定一致（satp->a0, smode_sp->a1, va_offset->a2）。
#[repr(C)]
pub struct BootInfo {
    pub satp: u64,
    pub smode_sp: u64,
    pub va_offset: u64,
}

/// 阶段一入口 — 由 entry.s 调用。
///
/// 参数通过寄存器传入 (entry.s 已保存 M-mode 的 a0/a1/a2 并重新排列):
///   a0 = ret_boot_info:  *mut BootInfo  (返回值写入此处)
///   a1 = enclave_id
///   a2 = man_pa_start                 (M-mode 分配的池物理地址)
///   a3 = man_size                      (Rust 二进制大小)
///
/// 不直接返回 BootInfo 结构体 (>16 字节会触发 RISC-V psABI
/// 隐式指针传递，破坏调用约定)。
#[unsafe(no_mangle)]
pub unsafe extern "C" fn rust_main_before_mmu(
    ret_boot_info: *mut BootInfo,
    enclave_id: u64,
    man_pa_start: u64,
    man_size: u64,
) {
    // uart::uart_init();
    println!("[enclave] before MMU: id={enclave_id} pa=0x{man_pa_start:x} size=0x{man_size:x}\n");
    // _end 符号已由 entry.s 中的 PIE 重定位调整至运行时地址 (base_pa + link_addr),
    // 无需再加 load_offset, 否则会 double-count base_pa.
    let end_pa = &raw const _end as u64;

	context::init_context(man_pa_start, ENCLAVE_MODULE_LOAD_VA_INIT);

    let pool_offset = end_pa - man_pa_start;
    let pool_size = memory::page_down(memory::chunk_2m_up(end_pa) - end_pa);
    memory::init_smode_pool(pool_offset, pool_size);
    memory::map_smode_page_pool(pool_offset, pool_size);
    memory::map_sections();
    paging::setup_linear_map();
    paging::identity_map_trampoline(man_pa_start);

    let root_pa = context::root_pa();
    let satp_val = paging::init_satp(root_pa);
    let smode_sp = unsafe { memory::alloc_smode_stack() };
    let va_offset = ENCLAVE_MAN_VA_START.wrapping_sub(man_pa_start);

    unsafe {
        ret_boot_info.write(BootInfo { satp: satp_val, smode_sp, va_offset });
    }
}

// ---------------------------------------------------------------
//  阶段二：MMU 使能后 —— 加载 ELF -> sret U-mode
// ---------------------------------------------------------------

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rust_main_after_mmu() {
    let (payload_pa, payload_size, argc) = call::enclave_call_suspend(0);

    // 完整性证明: 执行载荷前先验证其尾部 ECDSA 签名。
    // 载荷布局 [ bare | 64 字节签名 ]; 对 bare 做 SHA-256 后验签。
    // ATTEST_ENABLE=false 时跳过 (签名流水线未就绪, 保持启动畅通)。
    if ATTEST_ENABLE {
        if !attest::attest_payload(payload_pa, payload_size) {
            hang::hang_with_msg("[enclave] attestation failed — refusing to run payload");
        }
        println!("[enclave] attestation passed\n");
    }

    let argv_pa = payload_pa + memory::page_up(payload_size) + PAGE_SIZE;
    let pool_start = memory::chunk_2m_down(argv_pa);
    context::ctx_mut().umode_pool_pa_aligned = pool_start;
    memory::init_umode_pool(
        memory::page_up(argv_pa) - pool_start,
        memory::page_down(
            memory::chunk_2m_up(argv_pa + PAGE_SIZE) + CHUNK_2M_SIZE - (argv_pa + PAGE_SIZE),
        ),
    );

    let umode_sp = memory::alloc_map_umode_stack();
    memory::map_user_argv(argv_pa, argc);
    let entry = elf::load_elf(payload_pa, payload_size);

    context::ctx_mut().umode_heap_top = UMODE_HEAP_START_ALIGNED - memory::umode_pool_avail();

    let mut sstatus = csr::read_sstatus();
    sstatus |= csr::SSTATUS_SUM;
    sstatus &= !csr::SSTATUS_SPP;

    csr::write_sstatus(sstatus);
    csr::write_sepc(entry);
    csr::write_sscratch(umode_sp);
    csr::write_sie(csr::STI);

    let now: u64;
    unsafe { core::arch::asm!("csrr {0}, time", out(reg) now) };
    call::sbi_set_timer(now + TIMER_INTERVAL);

    println!("[enclave] entry=0x{entry:x} -> sret\n");
}

// ---------------------------------------------------------------
//  panic
// ---------------------------------------------------------------

#[panic_handler]
fn panic_handler(info: &PanicInfo) -> ! {
    if let Some(loc) = info.location() {
        println!(
            "[panic] {}:{} — {}\n",
            loc.file(), loc.line(),
            info.message()
        );
    } else {
        println!("[panic] {}\n", info.message());
    }
    hang::hang()
}
