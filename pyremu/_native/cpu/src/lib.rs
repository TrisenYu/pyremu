//! Pyremu 加速执行引擎的 Rust 库, 提供 RISC-V 指令分发,
//! TLB, PMP 匹配, Sv39 页表遍历与 ALU 运算.
//!
//! 全部公开符号以 `#[no_mangle] pub extern "C"` 导出, 供 ctypes 调用.

mod alu;
mod atom_instr;
mod concurrent;
mod consts;
pub mod csr;
mod decode;
mod ffi;
mod fpu;
pub mod handlers;
mod hart_sched;
mod interrupt;
mod mem;
mod mmu;
mod op_dispatcher;
mod peripheral;
mod pmp;
mod state;
pub mod translate;
mod trap;

pub use alu::*;
pub use consts::*;
pub use decode::*;
pub use ffi::*;
pub use mem::*;
pub use mmu::*;
pub use pmp::*;
pub use state::*;
