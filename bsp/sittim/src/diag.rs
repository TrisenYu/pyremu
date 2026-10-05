//! 诊断输出入口。
//!
//! 运行时各处的诊断行统一经本模块输出, 而非直接调用 `println!`:
//! 这样诊断输出目标 (当前为 SBI 控制台) 与开关只由一处决定, 调用点新增
//! 诊断行时不必各自重复同一套输出逻辑。
//!
//! 本模块整体以 `diagnostic` 特性门控: 调用点以
//! `#[cfg(feature = "diagnostic")] use crate::diag;` 导入并同样门控每条
//! 诊断语句, 未启用该特性时本模块为空, 发布构建不含任何诊断代码。

// 本模块的内容只与飞地的处理器相关设施绑定, 只在固件构建中编译。

/// 输出一条诊断行。
///
/// 例如载荷发生不可恢复的用户态异常时, 输出该次异常的原因与现场。
#[cfg(feature = "diagnostic")]
#[inline(always)]
pub fn log(args: core::fmt::Arguments) {
	crate::println!("{}", args);
}
