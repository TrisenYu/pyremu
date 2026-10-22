//! 用户侧并发编程的系统调用接口。
//!
//! 用户态并发语义集中在本包: 线程表、进程表、信号、futex 与时间片调度都由本包提供。
//!
//! 细分：
//! - `thread` — 线程表、上下文切换、线程生命周期与线程类系统调用
//! - `proc`   — 进程表、地址空间、fork/wait4、信号与间隔定时器
//! - `sig`    — 信号状态、信号帧与投递处置
//! - `sched`  — 时间片更新及到时检查
//! - `futex`  — futex 等待/唤醒语义
//! - `timer`  — 等待定时器

pub mod futex;
pub mod proc;
pub mod sched;
pub mod sig;
pub mod thread;
pub mod timer;
