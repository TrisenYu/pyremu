//! 文件系统模块的入口。
//!
//! ramfs 的目录树与文件数据整体在本模块内。fd 表由运行时持有, 载荷发起的文件系统系统
//! 调用也仍由运行时处理 (其参数与返回值的编解码即载荷侧的接口); 运行时处理时涉及目录树
//! 的部分经本模块导出的操作表调用, 且只传节点下标。故本模块的系统调用处理函数表全为空
//! 槽位, 运行时自身的调用点见 [`OPS`]。
//!
//! 映像的入口在偏移 0, 无 ELF 头, 无重定位项, 见 ../module.ld 与 ../Makefile。
//!
//! 接口结构体、管理器回调集合、操作表与自旋锁按 `#[path]` 取自 sittim 的 src, 与运行时
//! 共用同一份定义。

#![no_std]

/// 与 sittim 共用的接口定义。该文件中的一部分条目只有 sittim 侧使用, 本模块不引用
/// 它们, 故在本模块内不作未使用判定。
#[allow(dead_code)]
#[path = "../../../src/ext_mod/abi.rs"]
mod abi;

/// 与 sittim 共用的管理器回调集合。
#[allow(dead_code)]
#[path = "../../../src/ext_mod/man.rs"]
mod man;

/// 与 sittim 共用的自旋锁。
#[path = "../../../src/sync_aux.rs"]
mod sync_aux;

/// 与 sittim 共用的操作表定义, 本模块负责填写它。
#[allow(dead_code)]
#[path = "../../../src/ext_mod/vfs_ops.rs"]
mod vfs_ops;

mod ramfs;

/// 操作表的各项, 与主机侧的测试入口共用。
mod table;

use core::mem::MaybeUninit;

use abi::{
	GetterFn, InitFn, ModuleDesc, ModuleInterface, SyscallTable, MODULE_NAME_LEN,
	SYSCALL_TABLE_LEN,
};
use man::Manager;
use vfs_ops::VfsOps;

/// 模块编号, 与 ref-impl/emod 的 `EMODULE_ID_VFS` 取同一值。
const MODULE_ID: u32 = 3;

/// 模块名, 不足 [`MODULE_NAME_LEN`] 的部分以 0 填充。
const MODULE_NAME: [u8; MODULE_NAME_LEN] = {
	let mut name = [0u8; MODULE_NAME_LEN];
	let src = b"fs";
	let mut i = 0;
	while i < src.len() {
		name[i] = src[i];
		i += 1;
	}
	name
};

/// 本模块导出的系统调用处理函数表。文件系统的系统调用处理函数留在运行时, 故本表全为
/// 空槽位。
static SYSCALLS: SyscallTable = SyscallTable { handlers: [None; SYSCALL_TABLE_LEN] };

/// 本模块导出的操作表。
///
/// 各函数的地址在入口内写入, 不以静态初值给出: 静态初值会把该地址作为常量放进映像,
/// 而映像在链接时不知道自己将被放在哪个基址。
static mut OPS: MaybeUninit<VfsOps> = MaybeUninit::uninit();

/// 模块入口, 位于映像偏移 0。
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.init")]
pub unsafe extern "C" fn module_init(_manager: *const Manager) -> GetterFn {
	unsafe { core::ptr::addr_of_mut!(OPS).cast::<VfsOps>().write(table::fill_ops()) };
	get_interface
}

/// 入口的签名与运行时侧的 [`InitFn`] 一致。
const _: InitFn = module_init;

/// 取用器, 交付本模块的标识与两张表。可重复调用, 每次返回相同接口。
unsafe extern "C" fn get_interface(_manager: *const Manager) -> ModuleInterface {
	ModuleInterface {
		desc: ModuleDesc {
			module_id: MODULE_ID,
			name: MODULE_NAME,
			signature: 0,
		},
		syscalls: &raw const SYSCALLS,
		ops: core::ptr::addr_of!(OPS).cast::<u8>(),
	}
}

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo<'_>) -> ! {
	unsafe { core::arch::asm!("unimp", options(noreturn)) }
}
