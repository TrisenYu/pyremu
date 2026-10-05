//! 网络模块各文件在主机目标上的单元测试入口。
//!
//! 网络模块是独立的包, 其目标与 sittim 相同: 静态库只能在 RISC-V 裸机目标上构建。
//! 本入口按源码包含该模块的各文件, 由主机上的测试工具链编译。各 `mod` 的名称即模块
//! 内各文件所用的模块名, 故文件名与模块内所用的 `crate::` 路径一致。
//!
//! 模块内的平台设施由 platform.rs 自己分成两支: 主机侧以进程内的状态代替管理器,
//! MMIO 基地址指向本模块的字节数组, 地址翻译取恒等映射, 时间基准取固定值。故寄存器
//! 的读写在本入口内可执行, 无设备响应。

#![allow(dead_code)]

#[path = "../modules/net/src/platform.rs"]
mod platform;

#[path = "../modules/net/src/mmio.rs"]
mod mmio;

#[path = "../modules/net/src/virt_queue.rs"]
mod virt_queue;

#[path = "../modules/net/src/driver.rs"]
mod driver;

#[path = "../modules/net/src/phy.rs"]
mod phy;

#[path = "../modules/net/src/stack.rs"]
mod stack;
