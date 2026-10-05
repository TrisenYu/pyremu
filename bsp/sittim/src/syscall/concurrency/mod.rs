//! 用户侧并发编程的系统调用接口。
//!
//! 用户态并发语义集中在本包: 线程表、进程表、信号、futex 与时间片调度都由本包提供。
//! syscall 层的处理函数只做寄存器参数到本包原语的搬运, 而 S 模式陷阱入口
//! (crate::trap) 在返回 U 模式之前也要经本包完成线程切换与信号投递, 故本包同时
//! 被 syscall 层与 trap 层依赖。
//!
//! 细分：
//! - `thread` — 线程表、上下文切换、线程生命周期与线程类系统调用
//! - `proc`   — 进程表、地址空间、fork/wait4、信号与间隔定时器
//! - `sig`    — 信号状态、信号帧与投递处置
//! - `sched`  — 时间片记账与配额检查
//! - `futex`  — futex 等待/唤醒语义
//! - `timer`  — 等待定时器设施 (定时等待的截止时刻与到期唤醒)

pub mod futex;
pub mod proc;
pub mod sched;
pub mod sig;
pub mod thread;
pub mod timer;
