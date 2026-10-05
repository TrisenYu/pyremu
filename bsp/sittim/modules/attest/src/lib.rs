//! 示例模块: 导出一个恒返回固定值的系统调用处理函数, 用于贯通取入、验签与调用通路。
//!
//! 映像的入口在偏移 0, 无 ELF 头, 无重定位项, 见 ../module.ld 与 ../Makefile。
//!
//! 接口结构体与管理器回调集合按 `#[path]` 取自 sittim 的 src/ext_mod, 与运行时共用
//! 同一份定义。

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

use abi::{
	GetterFn, InitFn, ModuleDesc, ModuleInterface, SyscallTable, MODULE_NAME_LEN,
	SYSCALL_TABLE_LEN,
};
use man::Manager;

/// 模块编号, 与 ref-impl/emod 的 `EMODULE_ID_DUMMY` 取同一值。
const MODULE_ID: u32 = 7;

/// 模块名, 不足 [`MODULE_NAME_LEN`] 的部分以 0 填充。
const MODULE_NAME: [u8; MODULE_NAME_LEN] = {
	let mut name = [0u8; MODULE_NAME_LEN];
	let src = b"attest";
	let mut i = 0;
	while i < src.len() {
		name[i] = src[i];
		i += 1;
	}
	name
};

/// 本模块导出的系统调用编号。
const EXPORTED_SYSCALL: usize = 255;

/// 导出的处理函数的返回值, 用于确认调用进入了本模块。
const HANDLER_RESULT: u64 = 0x5A5A_0000_0000_0001;

/// 本模块导出的系统调用处理函数表。
///
/// 处理函数的地址在入口内写入, 不以静态初值给出: 静态初值会把该地址作为常量放进
/// 映像, 而映像在链接时不知道自己将被放在哪个基址。
static mut SYSCALLS: SyscallTable = SyscallTable { handlers: [None; SYSCALL_TABLE_LEN] };

/// 模块入口, 位于映像偏移 0。
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.init")]
pub unsafe extern "C" fn module_init(_manager: *const Manager) -> GetterFn {
	unsafe {
		SYSCALLS.handlers[EXPORTED_SYSCALL] = Some(handle_exported_syscall);
	}
	get_interface
}

/// 入口的签名与运行时侧的 [`InitFn`] 一致。
const _: InitFn = module_init;

/// 取用器, 交付本模块的标识与系统调用处理函数表。可重复调用, 每次返回相同接口。
///
/// 本模块不导出操作表, 故 `ops` 取空指针。
unsafe extern "C" fn get_interface(_manager: *const Manager) -> ModuleInterface {
	ModuleInterface {
		desc: ModuleDesc {
			module_id: MODULE_ID,
			name: MODULE_NAME,
			signature: 0,
		},
		syscalls: &raw const SYSCALLS,
		ops: core::ptr::null(),
	}
}

/// 导出的系统调用处理函数, 六个参数依次对应 a0 至 a5, 恒返回 [`HANDLER_RESULT`]。
unsafe extern "C" fn handle_exported_syscall(
	_a0: u64,
	_a1: u64,
	_a2: u64,
	_a3: u64,
	_a4: u64,
	_a5: u64,
) -> u64 {
	HANDLER_RESULT
}

#[panic_handler]
fn panic(_info: &core::panic::PanicInfo<'_>) -> ! {
	unsafe { core::arch::asm!("unimp", options(noreturn)) }
}
