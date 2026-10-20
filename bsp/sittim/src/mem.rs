//! 页池分配器与内存管理。 memory.c + page_pool.c 逻辑。
//!
//! 设计要点：
//! - 用相对偏移量而非绝对地址，方便日后迁移。
//! - S-mode 页池 (页表与内核栈) 落在管理器镜像末尾起的分区内。
//! - U 模式的物理内存不出自页池: argv 与 U 模式栈落在 M 模式随载荷一并授予的分区内,
//!   堆与 mmap 区经 `enclave_call_mem_alloc` 逐个分区申请。

use crate::constants::*;
use crate::context::{ self, PoolDesc };
use crate::mem_prim::align::{ align_down, align_up };
use crate::ecall_aux;
use crate::hang;
use crate::paging;

// ---------------------------------------------------------------
//  页池初始化
// ---------------------------------------------------------------

#[inline]
pub fn init_smode_pool(offset: u64, size: u64) {
	context::ctx_mut().smode_pool = PoolDesc {
		offset,
		size,
		used_pages: 0,
	};
}

// ---------------------------------------------------------------
//  页池分配
// ---------------------------------------------------------------

/// 从 S-mode 页池分配 `n` 个物理页，返回物理地址；池耗尽返回 `!0u64`。
///
/// 不再在此处 hang：由调用方决定是向上层报 OOM 还是视为致命错误 (S-mode 页表/栈,
/// 见 `alloc_smode_page`)。
fn pool_alloc(n: u64) -> u64 {
	let ctx = context::ctx();
	let expected = ctx.smode_pool.used_pages + n;
	if expected * PAGE_SIZE > ctx.smode_pool.size {
		return !0u64;
	}
	let pa =
		ctx.manager_pa_start +
		ctx.smode_pool.offset +
		ctx.smode_pool.used_pages * PAGE_SIZE;
	context::ctx_mut().smode_pool.used_pages = expected;
	pa
}

/// S-mode 页池分配 (页表/栈): 池耗尽属启动期致命错误, fail-stop.
pub fn alloc_smode_page(n: u64) -> u64 {
	let pa = pool_alloc(n);
	if pa == !0u64 {
		hang::fault_halt("alloc_smode_page: out of pages\n");
	}
	pa
}

/// S-mode 页池分配: 池耗尽返回 `!0u64`, 由调用方决定向上报错还是视为致命。
///
/// 载荷可控的路径 (brk / mmap 触发的页表扩容) 必须走本函数: 这些请求不得让
/// 可信管理程序自身停止服务, 否则载荷一次越界的内存请求即可使整个飞地失联,
/// 对宿主表现为, 与载荷仍在计算无法区分。
#[inline]
pub fn try_alloc_smode_page(n: u64) -> u64 {
	pool_alloc(n)
}

/// 向 M 模式申请一段可容纳 `pages` 个 4 KiB 页的连续分区, 分区类别取 `attr`
/// 正常返回该段分区的物理基址, 池中无足够的连续分区时
/// 否则返回 `!0u64`, 此时不占用任何分区。
///
/// 供需要把整段虚拟区间落在连续物理区的调用方使用: 一次申请即保证区间内各处连续,
/// 不必逐页申请再拼接。分区数向上取整到 2 MiB, 故所得区间不小于请求的字节数。
pub fn alloc_adjacent_pages(pages: u64, attr: u64) -> u64 {
	let partitions = align_up(pages * PAGE_SIZE, CHUNK_2M_SIZE) >> CHUNK_2M_SHIFT;
	if partitions == 0 {
		return !0u64;
	}
	let (granted, pa) = ecall_aux::enclave_call_mem_alloc(partitions, attr);
	if granted < partitions {
		return !0u64;
	}
	pa
}

/// `pa` 所在的分区是否不与飞地管理器镜像共用, 即可由单个映射整块改判。
///
/// 管理器镜像所在的分区由运行时自身的段与 S 模式页池占用, 不归属任何单个映射,
/// 改判其类别会一并改掉运行时自身页的权限。argv 与 U 模式栈所在的分区由本飞地独占
/// 且只被这两处映射引用, 载荷对其改判只影响自身的映射, 故不在此列。
pub fn is_dedicated_partition(pa: u64) -> bool {
	let ctx = context::ctx();
	let paddr = pa & !(CHUNK_2M_SIZE - 1);
	let image_end = ctx.manager_pa_start + ctx.smode_pool.offset + ctx.smode_pool.size;
	!(paddr >= ctx.manager_pa_start && paddr < image_end)
}

/// 把覆盖 [pa, pa + bytes) 的整数个 2 MiB 分区改判为 `attr`, 成功返回 true。
///
/// 类别是分区的属性, 故 `pa` 须按分区对齐, 区间涉及的每个分区整块改判。该区间由
/// 调用方独占, 区间之外的分区不受影响; 与飞地管理器镜像共用的分区一律拒绝, 判据见
/// [`is_dedicated_partition`]。由数据类改判为指令类之后, 分区只登记读与执行权限,
/// 写入将被拒绝。
pub fn set_partition_attr(pa: u64, bytes: u64, attr: u64) -> bool {
	if (pa & (CHUNK_2M_SIZE - 1)) != 0 {
		return false;
	}
	let partitions = align_up(bytes, CHUNK_2M_SIZE) >> CHUNK_2M_SHIFT;
	if partitions == 0 {
		return false;
	}
	for i in 0..partitions {
		if !is_dedicated_partition(pa + i * CHUNK_2M_SIZE) {
			return false;
		}
	}
	ecall_aux::enclave_call_set_mem_attr(pa, partitions, attr) == 0
}

// ---------------------------------------------------------------
//  段映射
// ---------------------------------------------------------------

/// 用链接器符号将 enclave-man 自身各段映射到 VA 空间。
/// 链接器符号已由 entry.s 中的 PIE 重定位调整至运行时地址,
/// 无需再加 load_offset, 否则会 double-count base_pa.
///
/// 只读段组的末尾取 `_instr_class_end`, 不取 `_rodata_end`: 该组在 `.rodata` 之后
/// 还有 `.eh_frame`、`.dynsym`、`.dynstr`、哈希表与 `.rela.dyn`, 运行时都要读取。
/// 可写段组的起点取 `_data_class_start`, 不取 `_data_start`: 紧随类别分界的还有
/// `.dynamic` 与 `.got`, 二者由启动期的重定位写入。
pub fn map_sections() {
	let ctx = context::ctx();
	let start_pa = ctx.manager_pa_start;
	let va_ofs = ENCLAVE_MAN_VA_START.wrapping_sub(start_pa);

	unsafe extern "C" {
		static _text_start: u8;
		static _text_end: u8;
		static _rodata_start: u8;
		static _instr_class_end: u8;
		static _data_class_start: u8;
		static _data_end: u8;
		static _bss_start: u8;
		static _bss_end: u8;
	}

	// 逐个映射各段，避免 Rust 2024 下 macro token pasting 的限制
	unsafe {
		map_one_section(
			&raw const _text_start as u64,
			&raw const _text_end as u64,
			va_ofs,
			PTE_X
		);
		map_one_section(
			&raw const _rodata_start as u64,
			&raw const _instr_class_end as u64,
			va_ofs,
			PTE_R
		);
		map_one_section(
			&raw const _data_class_start as u64,
			&raw const _data_end as u64,
			va_ofs,
			PTE_R | PTE_W
		);
		map_one_section(
			&raw const _bss_start as u64,
			&raw const _bss_end as u64,
			va_ofs,
			PTE_R | PTE_W
		);
	}
}

/// 映射单个段：start_pa / end_pa -> VA = PA + va_offset。
unsafe fn map_one_section(sec_start: u64, sec_end: u64, va_offset: u64, flags: u64) {
	if sec_start >= sec_end {
		return;
	}
	let size = sec_end.wrapping_sub(sec_start);
	// VA = PA + va_offset (sec_start 是 PIE 重定位后的运行时 PA)
	let va = sec_start.wrapping_add(va_offset);
	for i in 0..align_up(size, PAGE_SIZE) >> PAGE_SHIFT {
		map_or_halt(
			va + i * PAGE_SIZE,
			align_down(sec_start, PAGE_SIZE) + i * PAGE_SIZE,
			flags,
			LEVEL_PAGE,
			"map_sections: page table\n"
		);
	}
}

/// 将 S-mode 页池映射到 VA 空间。
pub fn map_smode_page_pool(pool_ofs: u64, pool_size: u64) {
	let ctx = context::ctx();
	let start_pa = ctx.manager_pa_start + pool_ofs;
	let va_ofs = ENCLAVE_MAN_VA_START.wrapping_sub(ctx.manager_pa_start);

	for i in 0..pool_size >> PAGE_SHIFT {
		let pa = start_pa + i * PAGE_SIZE;
		map_or_halt(
			pa.wrapping_add(va_ofs),
			pa,
			PTE_R | PTE_W,
			LEVEL_PAGE,
			"map_smode_page_pool: page table\n"
		);
	}
}

/// 启动期建立映射。失败则打印消息, 重新打开全局中断后挂起当前 hart, 不再返回。
fn map_or_halt(va: u64, pa: u64, flags: u64, level: u8, msg: &str) {
	if paging::map_page(va, pa, flags, level).is_err() {
		hang::fault_halt(msg);
	}
}

// ---------------------------------------------------------------
//  栈分配
// ---------------------------------------------------------------

#[unsafe(no_mangle)]
pub unsafe extern "C" fn alloc_smode_stack() -> u64 {
	let ctx = context::ctx();
	let va_ofs = ENCLAVE_MAN_VA_START.wrapping_sub(ctx.manager_pa_start);
	alloc_smode_page(SMODE_STACK_SIZE >> PAGE_SHIFT) + SMODE_STACK_SIZE + va_ofs
}

/// 从 S-mode 页池分配一块线程内核栈 (不挂起), 返回栈顶 VA; 池耗尽返回 0。
///
/// 页面由启动期的 map_smode_page_pool 预映射, 本分配只向前推进 used_pages。
/// 调用方 (thread::clone_thread) 在池耗尽时返回 ENOMEM 而非挂死内核。
pub fn try_alloc_thread_kstack(kstack_size: u64) -> u64 {
	let ctx = context::ctx();
	let va_ofs = ENCLAVE_MAN_VA_START.wrapping_sub(ctx.manager_pa_start);
	let pa = pool_alloc(kstack_size >> PAGE_SHIFT);
	if pa == !0u64 {
		return 0;
	}
	pa + kstack_size + va_ofs
}

/// 把 U 模式栈映射到 argv 分区末尾的 `UMODE_STACK_SIZE_TOTAL` 字节。
///
/// 该分区由 M 模式随载荷一并授予, 承载 argv 数据 (自分区首地址起) 与 U 模式栈
/// (分区末尾), 二者分处分区两端, 不相互覆盖。栈底页与 argv 页因此是不同的物理页。
pub fn alloc_map_umode_stack(argv_pa: u64) -> u64 {
	let pages = UMODE_STACK_SIZE_TOTAL >> PAGE_SHIFT;
	let bottom_pa = argv_pa + CHUNK_2M_SIZE - UMODE_STACK_SIZE_TOTAL;
	let bottom_va = UMODE_STACK_TOP_VA - UMODE_STACK_SIZE_TOTAL;

	for i in 0..pages {
		map_or_halt(
			bottom_va + i * PAGE_SIZE,
			bottom_pa + i * PAGE_SIZE,
			PTE_U | PTE_R | PTE_W,
			LEVEL_PAGE,
			"alloc_map_umode_stack: page table\n"
		);
	}
	UMODE_STACK_TOP_VA
}

/// 将用户 argv 从 PA 映射到 U-mode VA。
///
/// argv 数据位于 argv 分区的首个 4 KiB 页内, 故只
/// 映射这一个页; M 模式在 ENTER 时已核对总字节数不超过 `PAGE_SIZE`。各指针改写为
/// 页内偏移与栈顶 VA 之和, 与线性映射视图中的字符串一一对应。
pub fn map_user_argv(argv_pa: u64, argc: u64) {
	let va_argv = UMODE_STACK_TOP_VA;
	let argv_ptr = argv_pa.wrapping_add(LINEAR_MAP_OFFSET) as *mut u64;

	for i in 0..argc as usize {
		unsafe {
			let val = argv_ptr.add(i).read_volatile();
			let off = val % PAGE_SIZE;
			argv_ptr.add(i).write_volatile(va_argv + off);
		}
	}
	map_or_halt(
		va_argv,
		argv_pa,
		PTE_U | PTE_R | PTE_W,
		LEVEL_PAGE,
		"map_user_argv: page table\n"
	);
}

// ---------------------------------------------------------------
//  brk, 即 U-mode 堆扩展
// ---------------------------------------------------------------

/// 按 2 MiB 块把 U-mode VA 区间 [from, to) 建成映射, 返回实际映射到的 VA 上界。
///
/// 每个块经一次 M-mode 分区申请取得, 整块由本次映射占用。块内按 2 MiB 超页覆盖:
/// 超页之下不存在下一级页表, 块内不能再落 4 KiB 页, 故同一 VA 块只能由单一粒度
/// 覆盖, 同一 VA 一经映射即不再二次映射。
///
/// 取得的分区类别由 `flags` 的可执行位决定 (见 [`attr_of_pte_flags`])。
pub fn grow_umode_va(from: u64, to: u64, flags: u64) -> u64 {
	// 用户空间的映射止于 UMODE_VA_LIMIT: 越界的块与低半区共用根表项, 建立映射会改写
	// 另一个 VA 的页表项。上界之下按块推进, 故返回的上界可能略低于请求的上界。
	let to = if to > UMODE_VA_LIMIT { UMODE_VA_LIMIT } else { to };
	let mut va = from;
	let attr = attr_of_pte_flags(flags);

	while va < to {
		let (granted, pa) = ecall_aux::enclave_call_mem_alloc(
			MEM_ALLOC_BLOCKS_PER_CALL,
			attr
		);
		if granted == 0 {
			return va;
		}
		for i in 0..granted {
			let r = paging::map_page(
				va + i * CHUNK_2M_SIZE,
				pa + i * CHUNK_2M_SIZE,
				flags,
				LEVEL_MEGA
			);
			if r.is_err() {
				return va + i * CHUNK_2M_SIZE;
			}
		}
		va += granted * CHUNK_2M_SIZE;
	}

	va
}

/// 一段映射为 4 KiB 页时所归属的分区类别: 带可执行位 (PTE_X) 的映射从指令类
/// 取得内存, 其余从数据类取得。
///
/// 类别是整块 2 MiB 分区的属性, 故由映射的可执行位唯一决定: 同一段映射的各页
/// 同属一类, 一个分区也因此不会被同一次分配拆给两类映射。
pub fn attr_of_pte_flags(flags: u64) -> u64 {
	if (flags & PTE_X) != 0 { ENCLAVE_MEM_ATTR_INSTR } else { ENCLAVE_MEM_ATTR_DATA }
}

/// 在指定 VA 处按 4 KiB 粒度映射 `pages` 页, 供 MAP_FIXED 使用。
///
/// 该区间不经过映射区的页计数, 故不做块独占: 每次调用申请一段连续分区, 段内未
/// 用到的页不建立映射。申请失败或该处已被超页覆盖时返回 false, 由调用方回 ENOMEM。
///
/// 指令类映射的物理区先按数据类申请并映射为可写, 供调用方写入内容; 写入完成后由
/// 调用方把分区改判为指令类, 并把叶子项恢复为 `flags`。
pub fn map_umode_pages(va: u64, pages: u64, flags: u64) -> bool {
	if !is_umode_page_range(va, pages) {
		return false;
	}
	let pa = alloc_adjacent_pages(pages, ENCLAVE_MEM_ATTR_DATA);
	if pa == !0u64 {
		return false;
	}
	let leaf_flags = if attr_of_pte_flags(flags) == ENCLAVE_MEM_ATTR_INSTR {
		flags | PTE_R | PTE_W
	} else {
		flags
	};
	map_pages_at(va, pa, pages, leaf_flags)
}

/// 让 [dst_va, dst_va + pages * PAGE_SIZE) 指向 [src_va, ...) 已映射的同一批物理页。
///
/// 供 mremap 搬移共享映射使用: 共享映射的目的区间必须与源区间共用物理页, 逐字节
/// 复制会使两个地址各自持有副本, 与共享语义相反。源区间有未映射的页时返回 false。
pub fn alias_umode_pages(src_va: u64, dst_va: u64, pages: u64, flags: u64) -> bool {
	if !is_umode_page_range(dst_va, pages) {
		return false;
	}
	for i in 0..pages {
		let Some(pa) = paging::get_pa(src_va + i * PAGE_SIZE) else {
			return false;
		};
		if paging::map_page(dst_va + i * PAGE_SIZE, pa, flags, LEVEL_PAGE).is_err() {
			return false;
		}
	}
	true
}

// ---------------------------------------------------------------
//  mmap 匿名映射区
// ---------------------------------------------------------------

/// 在 [va, va + pages * PAGE_SIZE) 建立 4 KiB 映射, 物理页自 `pa` 起连续。
///
/// 只用于后备物理内存由调用方保管的场景 (按块切分的 mmap 区): 页的物理地址由
/// 块的物理基址加块内偏移算出, 故不能走 map_umode_pages。
pub fn map_pages_at(va: u64, pa: u64, pages: u64, flags: u64) -> bool {
	for i in 0..pages {
		if
			paging
				::map_page(va + i * PAGE_SIZE, pa + i * PAGE_SIZE, flags, LEVEL_PAGE)
				.is_err()
		{
			return false;
		}
	}
	true
}

/// 映射区中第 *page_index* 页的 VA。
fn mmap_page_va(page_index: u64) -> u64 {
	UMODE_MMAP_BASE + page_index * PAGE_SIZE
}

/// VA *va* 在映射区中的页号。
fn mmap_page_index_of(va: u64) -> u64 {
	(va - UMODE_MMAP_BASE) >> PAGE_SHIFT
}

/// 映射区已交付部分的末端 VA, 即下一次交付的起始地址。
fn mmap_region_end() -> u64 {
	mmap_page_va(context::ctx().umode_mmap_pages_used)
}

/// mmap 匿名映射区可交付的页数: 自 UMODE_MMAP_BASE 起至用户空间上界。
const MMAP_REGION_PAGES: u64 = (UMODE_VA_LIMIT - UMODE_MMAP_BASE) >> PAGE_SHIFT;

/// 虚拟地址区间 [start, start + bytes) 是否整体落在用户空间内。
///
/// Sv39 按 VPN 的 9 位字段查表, 不小于 `UMODE_VA_LIMIT` 的 VA 与低半区的某个 VA 共用
/// 同一组页表项, 在该区间建立映射会改写另一个 VA 的页表项。上方溢出同样视为越界。
pub fn is_umode_va_range(start: u64, bytes: u64) -> bool {
	match start.checked_add(bytes) {
		Some(end) => end <= UMODE_VA_LIMIT,
		None => false,
	}
}

/// 自 *st_va* 起 *page_num* 个 4 KiB 页是否整体落在用户空间内。
fn is_umode_page_range(st_va: u64, page_num: u64) -> bool {
	let Some(bytes) = page_num.checked_mul(PAGE_SIZE) else {
		return false;
	};
	page_num != 0 && is_umode_va_range(st_va, bytes)
}

/// 自页计数 *st_page* 起 *page_num* 个 4 KiB 页是否仍在 mmap 映射区内。
fn is_mmap_pages_in_region(st_page: u64, page_num: u64) -> bool {
	match st_page.checked_add(page_num) {
		Some(end) => end <= MMAP_REGION_PAGES,
		None => false,
	}
}

/// 从 mmap 匿名映射区取一段 VA 并建成映射, 返回起始 VA; 失败返回 0。
///
/// 映射区按 4 KiB 页计数推进 (umode_mmap_pages_used), 映射 VA 与块边界都由该
/// 计数导出。请求大小只决定交付多少页, 不改变块的划分: 长度非 4 KiB 对齐时向上
/// 取整到页 (与 Linux 一致), 任意长度的请求都可能跨块, 跨块时为下一个块取后备
/// 内存后继续。
///
/// 数据类的请求可与其他数据类请求共用同一块的后备物理内存: 一个 2 MiB 块最多拆成
/// 512 次 4 KiB 页使用。指令类的请求整块独占, 不参与块内切分, 见下。
pub fn take_mmap_region(bytes: u64, pte_flags: u64) -> u64 {
	// 长度不小于用户空间上界时, 按页与按块的取整都会向上溢出而出小值, 后续算式因此
	// 接受一个远大于映射区的区间。提前拒绝使算式只在无溢出的区间内求值。
	if bytes == 0 || bytes > UMODE_VA_LIMIT {
		return 0;
	}
	let pages = align_up(bytes, PAGE_SIZE) >> PAGE_SHIFT;

	// 指令类的请求整块独占, 不参与块内切分: 交付的物理区随后整块改判为指令类,
	// 块内若有其它映射会随之失去原有权限。物理区按数据类取得并映射为可写, 供调用方
	// 写入内容; 写入完成后由调用方改判并把叶子项恢复为请求的权限。
	if attr_of_pte_flags(pte_flags) == ENCLAVE_MEM_ATTR_INSTR {
		let aligned = align_up(context::ctx().umode_mmap_pages_used, CHUNK_2M_PAGES);
		let blocks = align_up(pages * PAGE_SIZE, CHUNK_2M_SIZE) >> CHUNK_2M_SHIFT;
		if !is_mmap_pages_in_region(aligned, blocks * CHUNK_2M_PAGES) {
			return 0;
		}
		let va = mmap_page_va(aligned);
		let pa = alloc_adjacent_pages(blocks * CHUNK_2M_PAGES, ENCLAVE_MEM_ATTR_DATA);
		if pa == !0u64 {
			return 0;
		}
		if !map_pages_at(va, pa, pages, pte_flags | PTE_R | PTE_W) {
			return 0;
		}
		// 已交付页计数越过块边界, 原正在切分的块不再交付, 其后备物理内存随之作废。
		context::ctx_mut().umode_curr_mmap_pa = 0;
		context::ctx_mut().umode_mmap_pages_used = aligned + blocks * CHUNK_2M_PAGES;
		return va;
	}

	if pages >= CHUNK_2M_PAGES {
		// 达到一整块及以上: 用 2 MiB 超页覆盖。超页叶子之下不存在下一级页表,
		// 块内不能再落 4 KiB 页, 故页计数不在块边界时先推进到块边界, 当前块的余量
		// 作废 (只损失地址空间, 无物理代价)。
		let used = context::ctx().umode_mmap_pages_used;
		let aligned = align_up(used, CHUNK_2M_PAGES);
		let blocks = align_up(pages * PAGE_SIZE, CHUNK_2M_SIZE) >> CHUNK_2M_SHIFT;
		if !is_mmap_pages_in_region(aligned, blocks * CHUNK_2M_PAGES) {
			return 0;
		}
		let va = mmap_page_va(aligned);
		let end = va + blocks * CHUNK_2M_SIZE;
		if grow_umode_va(va, end, pte_flags) < end {
			return 0;
		}
		// 交付之后的页计数落在块边界, 正在切分的块不再被交付, 其后的不足一块的请求
		// 另行取得后备内存。
		context::ctx_mut().umode_curr_mmap_pa = 0;
		context::ctx_mut().umode_mmap_pages_used = aligned + blocks * CHUNK_2M_PAGES;
		return va;
	}

	// 不足一整块: 在块内按 4 KiB 页交付, 页计数每跨过块边界就为新的块取后备内存。
	if !is_mmap_pages_in_region(context::ctx().umode_mmap_pages_used, pages) {
		return 0;
	}
	let va = mmap_region_end();
	let mut done = 0u64;
	// 正在切分的块的后备物理内存基址。交付失败时该值不写回, 使它始终描述末页所在的
	// 块, 供 mremap 的原地长度变化判定页计数是否仍落在同一个块内。
	let mut split_pa = context::ctx().umode_curr_mmap_pa;
	while done < pages {
		let in_block = context::ctx().umode_mmap_pages_used & (CHUNK_2M_PAGES - 1);
		if in_block == 0 {
			let pa = alloc_adjacent_pages(CHUNK_2M_PAGES, ENCLAVE_MEM_ATTR_DATA);
			if pa == !0u64 {
				break;
			}
			split_pa = pa;
		}
		let room = CHUNK_2M_PAGES - in_block;
		let n = if pages - done < room { pages - done } else { room };
		let pa = split_pa + in_block * PAGE_SIZE;
		if !map_pages_at(va + done * PAGE_SIZE, pa, n, pte_flags) {
			break;
		}
		context::ctx_mut().umode_curr_mmap_pa = split_pa;
		context::ctx_mut().umode_mmap_pages_used += n;
		done += n;
	}

	// 未交付满: 已交付的部分保留 (页计数按实交付页数推进, 不会被二次发放),
	// 返回 0 由调用方回 ENOMEM。
	if done < pages {
		return 0;
	}
	va
}

/// 正在切分的块所描述的 2 MiB 块的起始 VA; 后备物理内存基址
/// (umode_curr_mmap_pa) 为 0, 或页计数为 0 时为 None。
///
/// 该值由页计数导出, 页计数落在块边界时是刚填满的块的起点。
fn mmap_split_block_start() -> Option<u64> {
	let pages = context::ctx().umode_mmap_pages_used;
	if context::ctx().umode_curr_mmap_pa == 0 || pages == 0 {
		return None;
	}
	let last_page = mmap_page_va(pages - 1);
	Some(align_down(last_page, CHUNK_2M_SIZE))
}

/// 收缩能否原地完成: 该映射须是映射区的最后一段, 且收缩后的末页须与原末页落在同一个
/// 2 MiB 块内。
fn is_mmap_shrink_in_place_allowed(old_end: u64, new_end: u64) -> bool {
	// 该映射须是映射区的最后一段。
	if mmap_region_end() != old_end {
		return false;
	}
	// 收缩到块边界时块内偏移为 0, 下一次不足一块的请求重新取得后备物理内存, 原块不再
	// 被交付。
	if new_end % CHUNK_2M_SIZE == 0 {
		return true;
	}
	// 收缩后仍落在块内时, 块内交付按后备物理内存基址加块内偏移算出物理地址, 故收缩后的
	// 末页须与收缩前的末页同块 —— 跨越块边界的收缩会让两个不同的块共用同一片后备物理
	// 内存。
	mmap_split_block_start() == Some(align_down(new_end - PAGE_SIZE, CHUNK_2M_SIZE))
}

/// 扩容能否原地完成: 该映射须是映射区的最后一段, 且新增的末页须与原末页落在同一个
/// 2 MiB 块内。
fn is_mmap_grow_in_place_allowed(old_end: u64, new_end: u64) -> bool {
	// 该映射须是映射区的最后一段。
	if mmap_region_end() != old_end {
		return false;
	}
	// 新增页的物理地址按后备物理内存基址加块内偏移算出, 故新增的末页须与原末页同块 ——
	// 换块须先取得新的后备内存, 那属于 mremap 的搬移路径。
	match mmap_split_block_start() {
		Some(block_start) =>
			align_down(new_end - PAGE_SIZE, CHUNK_2M_SIZE) == block_start,
		None => false,
	}
}

/// mremap 的原地扩容/收缩: 成功返回 true, 调用方沿用原地址。
///
/// 仅当该映射是映射区的最后一段, 即其末页是映射区已交付的最后一页时才成立: 块内切分
/// 之后, 同一块内的后继映射也满足「末页已映射」这一较弱的条件, 按它扩容会覆盖后继
/// 映射。mremap 不带 prot, 新增页沿用原映射末页的权限位。
pub fn grow_mmap_in_place(addr: u64, old_size: u64, new_size: u64) -> bool {
	// 末址越过用户空间上界, 或两者的加法向上溢出时, 页计数与块边界都不再成立。
	let (Some(old_raw), Some(new_raw)) = (addr.checked_add(old_size), addr.checked_add(new_size))
	else {
		return false;
	};
	if new_raw > UMODE_VA_LIMIT {
		return false;
	}
	let old_end = align_up(old_raw, PAGE_SIZE);
	let new_end = align_up(new_raw, PAGE_SIZE);

	// 收缩: 只回退页计数, 页表项与物理内存保留 (与 munmap 一致, 不回收物理页)。
	if new_end <= old_end {
		if !is_mmap_shrink_in_place_allowed(old_end, new_end) {
			return false;
		}
		// 收缩后的页计数落在块边界时, 正在切分的块不再是末页所在的块, 其后的不足一块的
		// 请求另行取得后备内存。
		if new_end % CHUNK_2M_SIZE == 0 {
			context::ctx_mut().umode_curr_mmap_pa = 0;
		}
		context::ctx_mut().umode_mmap_pages_used = mmap_page_index_of(new_end);
		return true;
	}

	if !is_mmap_grow_in_place_allowed(old_end, new_end) {
		return false;
	}
	let flags = match paging::leaf_pte_flags(old_end - PAGE_SIZE) {
		Some(f) => f,
		None => {
			return false;
		}
	};
	let pa =
		context::ctx().umode_curr_mmap_pa +
		(old_end - align_down(old_end, CHUNK_2M_SIZE));
	let pages = (new_end - old_end) >> PAGE_SHIFT;
	if !map_pages_at(old_end, pa, pages, flags) {
		return false;
	}
	context::ctx_mut().umode_mmap_pages_used = mmap_page_index_of(new_end);
	true
}

pub fn sys_brk_handler(new_brk: u64) -> u64 {
	let old_top = context::ctx().umode_heap_top;
	if new_brk == 0 {
		return old_top; // 仅查询
	}
	// 越过用户空间上界的堆顶无法建立映射, 按增长失败返回旧值。
	if new_brk > UMODE_VA_LIMIT {
		return old_top;
	}

	if new_brk <= old_top {
		// 收缩或不变: 只回退 brk。已映射的物理页不回收 —— 堆的分区只增不减,
		// 由 M-mode 在飞地终止时整块回收。
		context::ctx_mut().umode_heap_top = new_brk;
		return new_brk;
	}

	// 向上扩展: 堆的已映射上界按 2 MiB 块推进, 与 brk 值解耦。
	let need_end = align_up(align_up(new_brk, PAGE_SIZE), CHUNK_2M_SIZE);
	let mapped_end = context::ctx().umode_heap_mapped_end;

	if need_end > mapped_end {
		let got = grow_umode_va(mapped_end, need_end, PTE_U | PTE_R | PTE_W);
		context::ctx_mut().umode_heap_mapped_end = got;
		if got < need_end {
			// 增长未完全满足 (M-mode 拒绝或池耗尽): 返回旧 brk, 让 musl 判定失败
			// (malloc 返回 NULL), 而不是返回未映射的 new_brk 造成后续缺页。
			return old_top;
		}
	}

	context::ctx_mut().umode_heap_top = new_brk;
	new_brk
}

// ---------------------------------------------------------------
//  镜像布局核对
// ---------------------------------------------------------------

/// 核对飞地管理器镜像的布局, 返回第一个不满足的不变式。
///
/// 授予的区间以 2 MiB 分区为单位划分类别, 故镜像必须落在整数个分区内:
/// - 只读段组之后的字节仍属分区 0, 代码与只读数据越过分区 0 会侵入按数据类登记的分区,
///   该分区只登记读与写权限, 其上的取指被拒绝;
/// - 镜像末尾不落在分区边界上时, 页池的起始物理地址不按分区对齐, 池内的页会横跨两个
///   类别的分区;
/// - 授予的分区数不足时镜像被截断, `.bss` 之后的段不可访问。
///
/// `man_pa_start` 是镜像的物理起始地址, `instr_class_end` 是只读段组的末尾, `end_pa` 是
/// 镜像的末尾 (含 `.bss`), `partition_num` 是 CREATE 授予本飞地的分区数。
pub fn check_img_layout(
	man_pa_start: u64,
	instr_class_end: u64,
	end_pa: u64,
	partition_num: u64
) -> Result<(), &'static str> {
	if instr_class_end - man_pa_start > CHUNK_2M_SIZE {
		return Err("sittim: RO sections overflow partition 0\n");
	}
	let man_size = end_pa - man_pa_start;
	if man_size % CHUNK_2M_SIZE != 0 {
		return Err("sittim: image ed-addr is not aligned\n");
	}
	if partition_num * CHUNK_2M_SIZE < man_size {
		return Err("sittim: granted partitions do not cover the image\n");
	}

	Ok(())
}



#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_chunk_page_ratio() {
		assert_eq!(CHUNK_2M_SIZE % PAGE_SIZE, 0);
		assert_eq!(CHUNK_2M_SIZE >> PAGE_SHIFT, 512);
	}
}
