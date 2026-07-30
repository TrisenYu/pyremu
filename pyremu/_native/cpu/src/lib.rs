//! Pyremu native acceleration library — RISC-V full instruction dispatch,
//! TLB, PMP matching, Sv39 page-table walk, and ALU compute.
//!
//! All public symbols use `#[no_mangle] pub extern "C"` for ctypes interop.

mod alu;
mod atom_instr;
mod concurrent;
pub mod csr;
mod decode;
mod diag;
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
pub use decode::*;
pub use mem::*;
pub use mmu::*;
pub use pmp::*;
pub use state::*;
