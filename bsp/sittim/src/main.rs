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
mod trap_frame;

use core::panic::PanicInfo;

use crate::constants::*;
use crate::mem_prim::align::align_up;
use crate::syscall::concurrency::proc;
use crate::syscall::concurrency::thread;

unsafe extern "C" {
	static _end: u8;
	static _instr_class_end: u8;
}

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
/// 所有参数按RISC-V调用规范通过a0~a4寄存器传入
#[unsafe(no_mangle)]
pub unsafe extern "C" fn main_before_mmu(
	ret_boot_info: *mut BootInfo,
	_enclave_id: u64,
	man_pa_start: u64,
	_man_size: u64,
	partition_num: u64
) {
	// _end 与 _instr_class_end 符号已由 entry.s 中的 PIE 重定位调整至运行时地址
	// (base_pa + link_addr), 无需再加 load_offset, 否则会 double-count base_pa.
	let end_pa = &raw const _end as u64;
	let instr_class_end = &raw const _instr_class_end as u64;
	let man_size = end_pa - man_pa_start;

	// 授予的区间以 2 MiB 分区为单位划分类别, 故镜像必须落在整数个分区内, 判据见
	// [`mem::check_img_layout`]。
	if
		let Err(msg) = mem::check_img_layout(
			man_pa_start,
			instr_class_end,
			end_pa,
			partition_num
		)
	{
		hang::fault_halt(msg);
	}

	context::init_context(man_pa_start, ENCLAVE_MODULE_LOAD_VA_INIT);

	// 页池自镜像末尾起, 覆盖授予的其余分区: 前若干分区由镜像占用, 其余归页池。
	let pool_offset = man_size;
	let pool_size = partition_num * CHUNK_2M_SIZE - man_size;
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
	if let Err(e) = paging::identity_map_trampoline(man_pa_start, partition_num) {
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
pub unsafe extern "C" fn main_after_mmu() {
	// 引导期交接: 必须交还宿主, 由宿主在 ENTER 返回路径上传入载荷信息.
	let (payload_pa, payload_size, argc) = ecall_aux::enclave_call_suspend(
		ENCLAVE_SUSPEND_VOLUNTARY
	);

	// argv 块基址必须与 M-mode enter_enclave_handler 的投递目标一致:
	// enter.c 中 payload_argv_paddr = ROUNDUP(payload_base_pa + payload_size, PARTITION_SIZE),
	// 即载荷末字节之后的第一个 2 MiB 分区边界. 该边界同时是指令类与数据类的分界:
	// 载荷所在分区归指令类, argv 所在分区归数据类, 一个分区只承载一个类别.
	// 故两侧算式必须一致: 少偏移会与载荷的末个分区重叠, 多偏移会使 argv 块落在
	// 分区之外, map_user_argv 读到空白页.
	let argv_pa = payload_pa + align_up(payload_size, CHUNK_2M_SIZE);
	// M-mode 为 argv 与 U 模式栈另授一个分区, 即 enter.c 的 argv 分区: argv 数据落在
	// 该分区的首地址, U 模式栈落在该分区的末尾, 见 mem::alloc_map_umode_stack.
	// 堆与 mmap 区不再出自任何页池, 由 grow_umode_va / take_mmap_region 经 500 号
	// 调用逐个申请数据类分区.

	// M-mode 移交的载荷物理区与紧随其后的 argv 物理区不在 setup_linear_map 的
	// [5 GiB, 18 GiB) 全局线性窗口内, 故把 [payload_pa, argv_pa + CHUNK_2M_SIZE)
	// 补建到 LINEAR_MAP_OFFSET 别名.
	//
	// 该别名必须在读取本区间的任何代码之前建立, 读取方为 syscall::inject_manifest,
	// elf::load_elf, map_user_argv 与 setup_musl_stack, 四处一律经 PA + OFFSET 读取.
	//
	// 上界取 argv 分区的末尾: argv 数据的字节数取决于各字符串的长度, 此处只知道
	// argc; 取分区末尾即覆盖 argv 指针数组所在的页与该分区内的其余页.
	if !paging::map_linear_range(payload_pa, argv_pa + CHUNK_2M_SIZE) {
		hang::fault_halt("map_linear_range: page table\n");
	}

	/*
		建立飞地文件系统与 fd 表, 供载荷的
		open/read/close 等系统调用使用。
		文件内容经 fs::vfs_inject_file 在载荷运行前注入。
		从载荷镜像末尾解析宿主预置的文件清单, 逐条注入文件系统。
	*/
	syscall::fs::mount_init();
	syscall::fs::vfs_init();
	syscall::inject_manifest(payload_pa, payload_size);

	let umode_sp = mem::alloc_map_umode_stack(argv_pa);
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
	// SPIE 置位, 否则载荷全程运行在全局中断关闭状态, S 定时器
	// 与 host 经 IPI 注入的软件终止消息中断都被屏蔽:
	// 未自行退出的长任务将独占当前 hart, 宿主 Linux 表现为 RCU stall 且 NMI 无响应
	sstatus |= csr::SSTATUS_SPIE;
	sstatus |= csr::SSTATUS_SUM;
	sstatus &= !csr::SSTATUS_SPP;
	// 解决方式: 关闭全局中断使能SIE.
	// 待进入 U 模式的现场必须在本函数末尾写入后直达 sret, 中间不可被 S 级中断打断.
	// 若 sstatus.SIE 仍为 1, 一旦末尾
	// csrw sie 打开 STIE, 该定时器中断立即在之后的指令处被取走. 陷态入口的
	// 而 handler 的 sret 返回后按 xRET 规则把 SPP 清 0. 于是首次 sret 便以
	// U 模式在运行时自身的物理地址取指, 恒等映射页无 U 权限, 立即触发取指页
	// 错误. 清 SIE 后现场写入与 sret 之间不可被 S 级中断
	// 打断, 且 SIE 将经 sret 由 SPIE=1 恢复, 载荷运行时的中断不受影响.
	sstatus &= !csr::SSTATUS_SIE;
	let now: u64;
	unsafe {
		core::arch::asm!("csrr {0}, time", out(reg) now);
	}
	// 与 trap.rs 定时器中断路径一致: 直接改写 stimecmp (Sstc) 而非经 SBI TIME
	// ecall.
	csr::write_stimecmp(now + TIMER_INTERVAL);

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

/*
    musl 栈布局:
	sp ->  argc
	      ------
		  argv[]
		  ------
		   NULL
		  ------
		  envp[]
		  ------
		   NULL
		  ------
		  auvx[]
		  ------

*/
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
	unsafe {
		core::ptr::write_volatile(*sp as *mut u64, val);
	}
}

/// 向 U-mode 栈写入 musl _start 期望的 argc/argv/envp/auxv 布局。
/// argv[i] 取 map_user_argv 改写后的 U-mode VA (线性映射视图), 与 M-mode
/// 拷贝的字符串一一对应。返回新的 sp (指向 argc)。
///
/// auxv 必须完整给出程序头表的三项 (AT_PHDR / AT_PHENT / AT_PHNUM): musl 静态
/// 链接版的 dl_iterate_phdr 以此为唯一数据源, libunwind 经它定位 .eh_frame_hdr
/// 才能取到展开表。缺失时 C++ 异常一律退化为 std::terminate (详见 elf::PayloadInfo)。
fn setup_musl_stack(
	sp_top: u64,
	argv_pa: u64,
	argc: u64,
	payload: &elf::PayloadInfo
) -> u64 {
	let mut sp = sp_top;

	// argv 终止 NULL 先于 argv 数组压入 (位于 argv[argc-1] 更高地址);
	// argv[argc-1] .. argv[0] 逆序压入使 argc 最终落在最低地址.
	// 此处 sstatus.SUM 尚未置位, 不能经 U-mode VA 读 argv 页; 改从 argv_pa 的
	// S-mode 线性映射别名读取 map_user_argv 已改写为 U-mode VA 的指针.
	let argv_arr = argv_pa.wrapping_add(LINEAR_MAP_OFFSET) as *const u64;

	// auxv 先写入, 位于栈底. 内存中自低向高按类型与值成对排布, 故先压值后压类型。
	unsafe {
		push_u64(&mut sp, 0);
		push_u64(&mut sp, AT_NULL);
		push_u64(&mut sp, 0);
		push_u64(&mut sp, AT_GID);
		push_u64(&mut sp, 0);
		push_u64(&mut sp, AT_UID);
		push_u64(&mut sp, 0);
		push_u64(&mut sp, AT_BASE);
		push_u64(&mut sp, payload.entry);
		push_u64(&mut sp, AT_ENTRY);
		push_u64(&mut sp, payload.phnum);
		push_u64(&mut sp, AT_PHNUM);
		push_u64(&mut sp, payload.phentsize);
		push_u64(&mut sp, AT_PHENT);
		push_u64(&mut sp, payload.phdr_va);
		push_u64(&mut sp, AT_PHDR);
		push_u64(&mut sp, 0x1000);
		push_u64(&mut sp, AT_PAGESZ);
		// envp
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
		println!("panic at {}:{}, {}\n", loc.file(), loc.line(), info.message());
	} else {
		println!("panic: {}\n", info.message());
	}
	hang::fault_halt("Sittim S-mode runtime panic, waiting for shutdown")
}
