//! 飞地时间片调度与执行状态管理。
//!
//! 职责:
//!   - 时间片记账与配额检查 (每次 timer interrupt 触发)
//!   - 飞地执行状态追踪 (已消耗时间 / 剩余配额 / 违规计数)
//!   - SUSPEND/RESUME 调度决策
//!
//! M-mode 提供机制 (mdid, PMP, 定时器中断, SUSPEND/RESUME),
//! 本模块提供策略 — 配额耗尽时主动让出 CPU, PMP 违规时决定处置方式.

use crate::context;
use crate::ecall_aux;

/// 时间片消耗记账与配额检查。
///
/// 每次 timer interrupt 调用一次。
/// 若 `time_quota > 0` 且已消耗 ticks >= quota, 则通过 SUSPEND ecall
/// 让出 CPU 给 host。host 调用 RESUME 后从此函数返回, 计数器自动归零
/// 进入下一时间片。
///
/// # 调用约定
///
/// 此函数可能在 SUSPEND 处阻塞 (直至 host RESUME), 因此调用方必须
/// 确保当前 U-mode 上下文已可被安全地保存和恢复。在 S-mode trap handler
/// 中调用是安全的 — M-mode 的 `alter_hart_ctx_for_enclave` 会完整保存
/// S-mode 寄存器状态 (含 sepc/sstatus, 即 U-mode 的 trap frame)。
pub fn tick_and_check_quota() {
	let expired = {
		let ctx = context::ctx_mut();
		ctx.ticks_consumed = ctx.ticks_consumed.wrapping_add(1);
		ctx.time_quota > 0 && ctx.ticks_consumed >= ctx.time_quota
	};

	if expired {
		// 让出 CPU: M-mode 保存当前飞地上下文 -> 切回 host.
		// host RESUME 后从此处继续执行.
		ecall_aux::enclave_call_suspend(0);
		// 新时间片开始
		context::ctx_mut().ticks_consumed = 0;
	}
}

/// 返回当前时间片的剩余配额 (timer interrupt 次数)。
/// `time_quota == 0` 表示无限制, 返回 `u64::MAX`。
#[allow(dead_code)]
pub fn remaining_quota() -> u64 {
	let ctx = context::ctx();
	if ctx.time_quota == 0 {
		u64::MAX
	} else {
		ctx.time_quota.saturating_sub(ctx.ticks_consumed)
	}
}

/// 返回当前时间片已消耗的 timer interrupt 次数。
#[allow(dead_code)]
pub fn ticks_consumed() -> u64 {
	context::ctx().ticks_consumed
}

/// 查询 M-mode 是否有 host 发来的待处理请求并执行对应操作。
///
/// 当前支持的请求:
///   - `ENCLAVE_REQ_SHUTDOWN`: host 申请终止飞地.
///     token 已在 M-mode 验证, S-mode 信任 M-mode 的判断直接执行 SHUTDOWN.
///     Phase B 将替换为: S-mode 自行验证 host attestation quote.
///
/// 在 timer interrupt 或 software interrupt 上下文中调用。
/// SHUTDOWN 不返回 (通过 `enclave_call_exit` -> M-mode 清理 -> 切回 host).
pub fn check_pending_requests() {
	let flags = ecall_aux::enclave_call_query_requests();

	if flags & crate::constants::ENCLAVE_REQ_SHUTDOWN != 0 {
		// host 已通过 token 验证, 执行终止.
		// `enclave_call_exit` 触发 SHUTDOWN ecall -> M-mode 清理飞地
		// -> alter_hart_ctx_for_enclave 切回 host -> 不返回.
		ecall_aux::enclave_call_exit(0);
	}
}
