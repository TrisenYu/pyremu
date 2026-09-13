//! 飞地内的并发原语与调度。
//!
//! 本包位于 trap 层与 syscall 层之下: S 模式 trap 入口 (crate::trap) 直接依赖
//! 线程表与抢占调度, 系统调用层只做参数搬运。若把这些模块放到 syscall 之下,
//! trap 就必须反向依赖 syscall 层。
//!
//! 细分：
//! - `thread`    — 线程表、上下文切换与线程生命周期
//! - `sched`     — 时间片抢占调度
//! - `futex`     — futex 等待/唤醒语义
//! - `spinlock`  — 自旋锁

pub mod futex;
pub mod sched;
pub mod spinlock;
pub mod thread;
