//! Pyremu native acceleration library — RISC-V full instruction dispatch,
//! TLB, PMP matching, Sv39 page-table walk, and ALU compute.
//!
//! All public symbols use `#[no_mangle] pub extern "C"` for ctypes interop.

mod alu;
mod atom_instr;
mod concurrent;
mod decode;
mod diag;
mod ffi;
mod fpu;
mod hart_sched;
mod interrupt;
mod mem;
mod mmu;
mod op_dispatcher;
mod peripheral;
mod pmp;
mod state;
mod trap;
pub mod csr;
pub mod handlers;
pub mod translate;

pub use alu::*;
pub use decode::*;
pub use ffi::*;
pub use mem::*;
pub use mmu::*;
pub use pmp::*;
pub use state::*;
