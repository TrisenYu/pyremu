//! 飞地 S-mode 运行时启动入口。以 MMU 是否使能为界分为两阶段。
#![no_std]
#![no_main]

mod attest;
mod constants;
mod context;
mod csr;
mod diag;
mod ecall_aux;
mod elf;
mod ext_mod;
mod hang;
mod mem;
mod mem_prim;
mod paging;
mod println;
mod sync_aux;
mod syscall;
mod trap;

use core::panic::PanicInfo;

use crate::constants::*;
use crate::mem_prim::align::{align_down, align_up};
use crate::syscall::concurrency::proc;
use crate::syscall::concurrency::thread;

unsafe extern "C" {
	static _end: u8;
}

// entry.s 是 RISC-V 汇编, 只有固件构建编译它。
core::arch::global_asm!(include_str!("entry.s"));

// ---------------------------------------------------------------
//  阶段一：MMU 使能前 —— 构建页表
// ---------------------------------------------------------------

/// 阶段一返回给 entry.s 的启动信息。`repr(C)` 确保与汇编的
/// 寄存器约定一致: satp 映射到 a0, smode_sp 映射到 a1, va_offset 映射到 a2。
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
	_enclave_id: u64,
	man_pa_start: u64,
	_man_size: u64,
) {
	// _end 符号已由 entry.s 中的 PIE 重定位调整至运行时地址 (base_pa + link_addr),
	// 无需再加 load_offset, 否则会 double-count base_pa.
	let end_pa = &raw const _end as u64;

	context::init_context(man_pa_start, ENCLAVE_MODULE_LOAD_VA_INIT);

	let pool_offset = end_pa - man_pa_start;
	let pool_size = align_down(align_up(end_pa, CHUNK_2M_SIZE) - end_pa, PAGE_SIZE);
	mem::init_smode_pool(pool_offset, pool_size);
	mem::map_smode_page_pool(pool_offset, pool_size);
	mem::map_sections();
	// 设备窗口在任何 clone 之前建立: 该映射落在根表高半区, 各进程地址空间共用。
	if !paging::map_device_region(NET_BASE, NET_SIZE) {
		hang::fault_halt("map_device_region: page table\n");
	}
	if !paging::setup_linear_map() {
		hang::fault_halt("setup_linear_map: page table\n");
	}
	if let Err(e) = paging::identity_map_trampoline(man_pa_start) {
		hang::fault_halt(e.name());
	}

	let root_pa = context::root_pa();
	let satp_val = paging::init_satp(root_pa);
	let smode_sp = unsafe { mem::alloc_smode_stack() };
	// 记录主线程 thread 0 的内核栈顶, 供 MMU 使能后 init_main_thread 使用.
	thread::note_boot_kstack_top(smode_sp);
	let va_offset = ENCLAVE_MAN_VA_START.wrapping_sub(man_pa_start);

	unsafe {
		ret_boot_info.write(BootInfo {
			satp: satp_val,
			smode_sp,
			va_offset,
		});
	}
}

// ---------------------------------------------------------------
//  阶段二：MMU 使能后 —— 加载 ELF -> sret U-mode
// ---------------------------------------------------------------

#[unsafe(no_mangle)]
pub unsafe extern "C" fn rust_main_after_mmu() {
	// 引导期交接: 必须交还宿主, 由宿主在 ENTER 返回路径上传入载荷信息.
	let (payload_pa, payload_size, argc) =
		ecall_aux::enclave_call_suspend(ENCLAVE_SUSPEND_VOLUNTARY);
	#[cfg(feature = "diagnostic")]
	println!(
		"Sittim: payload pa=0x{payload_pa:x} size=0x{payload_size:x} argc={argc}\n"
	);

	// 完整性证明: 执行载荷前先验证其尾部 ECDSA 签名。
	// 载荷布局 [ bare | 64 字节签名 ]; 对 bare 做 SHA-256 后验签。
	// ATTEST_ENABLE=false 时跳过 (签名流水线未就绪, 保持启动畅通)。
	// 验签不通过即本载荷不可执行, 与载荷无法加载同属准入失败, 故走同一终止路径。
	if ATTEST_ENABLE {
		if !attest::attest_payload(payload_pa, payload_size) {
			elf::reject_payload("attestation failed, refusing to run payload");
		}
	}

	// argv 块基址必须与 M-mode enter_enclave_handler 的投递目标一致:
	// ext_ecall.c 中 payload_argv_paddr = ROUNDUP(payload_base_pa + payload_size, PAGE_SIZE),
	// 即载荷末页之后第一个页. 多偏移一页会指向 argv 块之外, 使 map_user_argv
	// 读到空白页 (argv 内容在其前一页).
	let argv_pa = payload_pa + align_up(payload_size, PAGE_SIZE);
	let pool_start = align_down(argv_pa, CHUNK_2M_SIZE);
	let pool_offset = align_up(argv_pa, PAGE_SIZE) - pool_start;
	let pool_size = align_down(
		align_up(argv_pa + PAGE_SIZE, CHUNK_2M_SIZE) + CHUNK_2M_SIZE - (argv_pa + PAGE_SIZE),
		PAGE_SIZE,
	);
	context::ctx_mut().umode_pool_pa_aligned = pool_start;
	mem::init_umode_pool(pool_offset, pool_size);

	// M-mode 移交的载荷/argv 物理区 (载荷后紧跟 argv) 落在池的低物理地址,
	// 不在 setup_linear_map 的 [5 GiB, 18 GiB) 全局线性窗口内, 故在读取前
	// 把 [payload_pa, pool_start + pool_offset + pool_size) 补建到
	// LINEAR_MAP_OFFSET 别名, 供 elf::load_elf / map_user_argv /
	// setup_musl_stack 经 PA+OFFSET 读取.
	//
	// 上界取整个池的末尾, 不取 argv 块末尾: U-mode 池占用 argv 块之后的同一段
	// 物理区, 池内的页同样以 PA+OFFSET 别名被 S 模式访问, 只映射到 argv 块末尾
	// 会使池内首次访问触发取数页错误。
	if !paging::map_linear_range(payload_pa, pool_start + pool_offset + pool_size) {
		hang::fault_halt("map_linear_range: page table\n");
	}

	// 建立飞地文件系统 (挂载表 + 根文件系统 + 内核自带的伪文件系统) 与 fd 表, 供载荷的
	// open/read/close 等系统调用使用。文件内容经 fs::vfs_inject_file 在载荷运行前注入。
	syscall::fs::mount_init();
	syscall::fs::vfs_init();
	// 从载荷镜像末尾解析宿主预置的文件清单 (compound payload trailer), 逐条注入
	// 文件系统。chibicc 等需 fopen 的载荷由此获得自检源文件; 无清单的载荷返回 0。
	syscall::inject_manifest(payload_pa, payload_size);

	let umode_sp = mem::alloc_map_umode_stack();
	mem::map_user_argv(argv_pa, argc);
	let payload = elf::load_elf(payload_pa, payload_size);

	// musl _start 要求 sp 指向 argc 的栈布局 (argc/argv/envp/auxv)
	let umode_sp = setup_musl_stack(umode_sp, argv_pa, argc, &payload);

	// 堆从 UMODE_HEAP_START_ALIGNED 向上增长, 与 mmap 区 (UMODE_MMAP_BASE)
	// 及栈区 (UMODE_STACK_TOP_VA 下方 1 MiB) 各自独立, 互不重叠。
	// 已映射上界从同一起点开始, 由 brk 按 2 MiB 块推进。
	context::ctx_mut().umode_heap_top = UMODE_HEAP_START_ALIGNED;
	context::ctx_mut().umode_heap_mapped_end = UMODE_HEAP_START_ALIGNED;

	let mut sstatus = csr::read_sstatus();
	// 首次 sret 进入 U-mode 时, 硬件以 sstatus.SPIE 恢复 SIE. 这里必须把
	// SPIE 置位, 否则载荷全程运行在全局中断关闭状态, S 定时器 (配额抢占)
	// 与 host 经 IPI 注入的软件中断 (终止请求) 都被屏蔽: 未自行退出的长
	// 任务会永久独占当前 hart, 宿主 Linux 表现为 RCU stall 且 NMI 无响应
	// (与 hang.rs fault_halt 注释描述的是同一问题).
	sstatus |= csr::SSTATUS_SPIE;
	sstatus |= csr::SSTATUS_SUM;
	sstatus &= !csr::SSTATUS_SPP;
	// 关键: 关闭全局中断使能 (SIE). 待进入 U 模式的现场 (sepc / sstatus /
	// sscratch) 必须在本函数末尾写入后直达 sret, 中间不可被 S 级中断打断.
	// 此前首次 arm 定时器 (now+TIMER_INTERVAL) 在 ELF 加载前就已写入, 到本函数
	// 末尾 deadline 早已过期, STIP 已挂起; 若 sstatus.SIE 仍为 1, 一旦末尾
	// csrw sie 打开 STIE, 该定时器中断立即在之后的指令处被取走. 陷态入口的
	// 硬件行为: sepc <- 被中断的物理 PC (运行时自身地址 0x8300445a), SPP <- 1;
	// 而 handler 的 sret 返回后按 xRET 规则把 SPP 清 0. 于是首次 sret 便以
	// U 模式在运行时自身的物理地址取指, 恒等映射页无 U 权限, 立即触发取指页
	// 错误 (scause=0xc, sepc=stval=物理地址), 载荷一条指令未执行即自毁为
	// EXITED_ERR (退出码 139). 清 SIE 后现场写入与 sret 之间不可被 S 级中断
	// 打断, 且 SIE 将经 sret 由 SPIE=1 恢复, 载荷运行时的中断不受影响.
	sstatus &= !csr::SSTATUS_SIE;
	let now: u64;
	unsafe { core::arch::asm!("csrr {0}, time", out(reg) now) };
	// 与 trap.rs 定时器中断路径一致: 直接改写 stimecmp (Sstc) 而非经 SBI TIME
	// ecall.
	csr::write_stimecmp(now + TIMER_INTERVAL);

	#[cfg(feature = "diagnostic")]
	println!(
		"Sittim entry=0x{:x} phdr=0x{:x} phnum={} sp=0x{umode_sp:x} argv_pa=0x{argv_pa:x} argc={argc} -> sret\n",
		payload.entry, payload.phdr_va, payload.phnum
	);

	// 登记启动进程 (槽位 0) 与主线程并建立调度状态. 用户现场经参数传入 (见
	// thread::init_main_thread 的说明), 不依赖 CSR 的写入时机.
	proc::init_main_process();
	thread::init_main_thread(payload.entry, sstatus);

	// 末尾写入 U 模式现场. 此时 sstatus.SIE 已清 0 (见上), 因此从 csrw sstatus
	// 到 sret 之间不会再有 S 级中断取走, sepc / sscratch / SPP 得以原样送达 sret.
	// csrw sie 仅设置中断使能掩码, 在 SIE=0 时不触发任何中断.
	csr::write_sstatus(sstatus);
	csr::write_sepc(payload.entry);
	csr::write_sscratch(umode_sp);
	csr::write_sie(csr::STI | csr::SSI);
}

// ---------------------------------------------------------------
//  musl 栈布局: sp 指向 argc | argv[] | NULL | envp[] | NULL | auxv[]
// ---------------------------------------------------------------

const AT_NULL: u64 = 0;
const AT_PHDR: u64 = 3;
const AT_PHENT: u64 = 4;
const AT_PHNUM: u64 = 5;
const AT_PAGESZ: u64 = 6;
const AT_BASE: u64 = 7;
const AT_ENTRY: u64 = 9;
const AT_UID: u64 = 11;
const AT_GID: u64 = 13;

unsafe fn push_u64(sp: &mut u64, val: u64) {
	*sp = sp.wrapping_sub(8);
	unsafe { core::ptr::write_volatile(*sp as *mut u64, val); }
}

/// 向 U-mode 栈写入 musl _start 期望的 argc/argv/envp/auxv 布局。
/// argv[i] 取 map_user_argv 改写后的 U-mode VA (线性映射视图), 与 M-mode
/// 拷贝的字符串一一对应。返回新的 sp (指向 argc)。
///
/// auxv 必须完整给出程序头表的三项 (AT_PHDR / AT_PHENT / AT_PHNUM): musl 静态
/// 链接版的 dl_iterate_phdr 以此为唯一数据源, libunwind 经它定位 .eh_frame_hdr
/// 才能取到展开表。缺失时 C++ 异常一律退化为 std::terminate (详见 elf::PayloadInfo)。
fn setup_musl_stack(sp_top: u64, argv_pa: u64, argc: u64, payload: &elf::PayloadInfo) -> u64 {
	let mut sp = sp_top;

	// argv 终止 NULL 先于 argv 数组压入 (位于 argv[argc-1] 更高地址);
	// argv[argc-1] .. argv[0] 逆序压入使 argc 最终落在最低地址.
	// 此处 sstatus.SUM 尚未置位, 不能经 U-mode VA 读 argv 页; 改从 argv_pa 的
	// S-mode 线性映射别名读取 map_user_argv 已改写为 U-mode VA 的指针.
	let argv_arr = argv_pa.wrapping_add(LINEAR_MAP_OFFSET) as *const u64;

	// auxv 先写入, 位于栈底. 内存中自低向高按类型与值成对排布,
	// 故每对先压值、后压类型。
	unsafe {
		push_u64(&mut sp, 0); // AT_NULL value
		push_u64(&mut sp, AT_NULL); // AT_NULL type
		push_u64(&mut sp, 0); // AT_GID = 0
		push_u64(&mut sp, AT_GID); // AT_GID type
		push_u64(&mut sp, 0); // AT_UID = 0
		push_u64(&mut sp, AT_UID); // AT_UID type
		push_u64(&mut sp, 0); // AT_BASE: 静态可执行文件无解释器
		push_u64(&mut sp, AT_BASE); // AT_BASE type
		push_u64(&mut sp, payload.entry); // AT_ENTRY
		push_u64(&mut sp, AT_ENTRY); // AT_ENTRY type
		push_u64(&mut sp, payload.phnum); // AT_PHNUM
		push_u64(&mut sp, AT_PHNUM); // AT_PHNUM type
		push_u64(&mut sp, payload.phentsize); // AT_PHENT
		push_u64(&mut sp, AT_PHENT); // AT_PHENT type
		push_u64(&mut sp, payload.phdr_va); // AT_PHDR
		push_u64(&mut sp, AT_PHDR); // AT_PHDR type
		push_u64(&mut sp, 0x1000); // AT_PAGESZ = 4096
		push_u64(&mut sp, AT_PAGESZ); // AT_PAGESZ type

		// envp (空)
		push_u64(&mut sp, 0); // envp[0] = NULL

		// argv 终止 NULL
		push_u64(&mut sp, 0);
		// argv[argc-1] .. argv[0]
		for i in (0..argc as usize).rev() {
			let arg_ptr = argv_arr.add(i).read_volatile();
			push_u64(&mut sp, arg_ptr);
		}

		// argc
		push_u64(&mut sp, argc);
	}
	sp
}

// ---------------------------------------------------------------
//  panic
// ---------------------------------------------------------------

// 测试构建链接 std 与 libtest, 二者各自提供 panic 入口。
#[panic_handler]
fn panic_handler(info: &PanicInfo) -> ! {
	if let Some(loc) = info.location() {
		println!(
			"panic at {}:{}, {}\n",
			loc.file(),
			loc.line(),
			info.message()
		);
	} else {
		println!("panic: {}\n", info.message());
	}
	hang::fault_halt("Sittim S-mode runtime panic, waiting for shutdown")
}
