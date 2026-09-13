//! 诊断输出入口。
//!
//! 运行时各处的诊断行统一经本模块输出, 而非直接调用 `println!`:
//! 这样诊断输出的去向 (当前为 SBI 控制台) 与开关只由一处决定, 调用点新增
//! 诊断行时不必各自重复同一套输出逻辑。
//!
//! 本模块整体以 `diagnostic` 特性门控: 调用点以
//! `#[cfg(feature = "diagnostic")] use crate::diag;` 导入并同样门控每条
//! 诊断语句, 未启用该特性时本模块为空, 发布构建不含任何诊断代码。

/// 输出一条诊断行。
///
/// 诊断行用于定位时序与状态问题, 例如判定一次时间片让出之后 host 的
/// RESUME 是否真的让载荷继续推进。
#[cfg(feature = "diagnostic")]
#[inline(always)]
pub fn log(args: core::fmt::Arguments) {
	crate::println!("{}", args);
}

/// 读取 mtime 时钟源 (CSR 0xC01), 供诊断行标注受调试程序内已经经过的时间。
#[cfg(feature = "diagnostic")]
#[inline(always)]
pub fn read_mtime() -> u64 {
	let t: u64;
	unsafe {
		core::arch::asm!("csrr {0}, 0xC01", out(reg) t);
	}
	t
}
