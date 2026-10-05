//! 文件系统模块在主机目标上的单元测试入口。
//!
//! 文件系统模块是独立的包, 其目标与 sittim 相同: 静态库只能在 RISC-V 裸机目标上构建。
//! 本入口按源码包含该模块的节点层, 由主机上的测试工具链编译。各 `mod` 的名称即模块内
//! 所用的模块名, 故文件名与模块内所用的 `crate::` 路径一致。
//!
//! 模块的入口 lib.rs 不在此列: 它带 `#[panic_handler]` 与映像入口, 与主机的测试工具链
//! 冲突。被包含的节点层不访问设备与 CSR, 其存储是模块内的静态数组, 故其用例在主机上
//! 执行的结果与在飞地内一致。

#![allow(dead_code)]

#[path = "../src/sync_aux.rs"]
mod sync_aux;

#[path = "../src/ext_mod/vfs_ops.rs"]
mod vfs_ops;

#[path = "../modules/fs/src/ramfs.rs"]
mod ramfs;
