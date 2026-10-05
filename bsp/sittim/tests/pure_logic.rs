//! 主机目标上的纯逻辑单元测试入口。
//!
//! 本入口只包含不访问 CSR、不访问设备寄存器、不依赖飞地上下文的模块, 故其用例
//! 在主机上执行的结果与在飞地内执行的结果一致。

#[path = "../src/constants.rs"]
mod constants;

#[path = "../src/mem_prim/mod.rs"]
mod mem_prim;

#[path = "../src/sync_aux.rs"]
mod sync_aux;
