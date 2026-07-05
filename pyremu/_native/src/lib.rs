//! Pyremu native acceleration library — RISC-V full instruction dispatch,
//! TLB, PMP matching, Sv39 page-table walk, and ALU compute.
//!
//! All public symbols use `#[no_mangle] pub extern "C"` for ctypes interop.

mod alu;
pub mod csr;
mod decode;
mod exec;
pub mod handlers;
mod mem;
mod mmu;
mod op_dispatcher;
mod pmp;
mod state;
pub mod translate;
mod trap;

pub use alu::*;
pub use decode::*;
pub use exec::*;
pub use mem::*;
pub use mmu::*;
pub use pmp::*;
pub use state::*;
