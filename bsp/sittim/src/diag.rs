//! 诊断输出入口。
//! 本模块整体以 `diagnostic` 决定是否使能: 调用点以
//! `#[cfg(feature = "diagnostic")] use crate::diag;` 导入并同样门控每条
//! 诊断语句, 未启用该特性时本模块为空, 发布构建不含任何诊断代码。

// 本模块的内容只与飞地的处理器相关设施绑定, 只在固件构建中编译。

/// 输出一条诊断行。
///
/// 例如载荷发生不可恢复的用户态异常时, 输出该次异常的原因与现场。
#[inline(always)]
#[allow(unused)]
pub fn log(args: core::fmt::Arguments) {
	#[cfg(feature = "diagnostic")]
	{
		crate::println!("{}", args);
	}
}
