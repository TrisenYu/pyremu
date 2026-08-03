//! Safe-ish wrappers around RISC-V CSR instructions.
//! Uses inline `asm!()` (Rust 2024 stabilised asm).
//!
//! `read_csr!(sstatus)` 等宏通过 `stringify!` 将 CSR 寄存器名直接拼入
//! asm 模板，与 C 的 `read_csr(sstatus)` -> `"csrr %0, sstatus"` 等效。
//!
//! 此文件为完整的 S-mode CSR 参考定义，未使用的常量与包装函数有意保留。

#![allow(dead_code)]

// ---------------------------------------------------------------
//  S-mode CSR 地址常量（文档 / 引用用；宏使用小写寄存器名）
// ---------------------------------------------------------------

pub const SSTATUS: u16 = 0x100;
pub const SIE: u16 = 0x104;
pub const STVEC: u16 = 0x105;
pub const SIP: u16 = 0x144;
pub const SATP: u16 = 0x180;
pub const SSCRATCH: u16 = 0x140;
pub const SEPC: u16 = 0x141;
pub const SCAUSE: u16 = 0x142;
pub const STVAL: u16 = 0x143;

// ---------------------------------------------------------------
//  Bit masks  (per smode_entry/csr_aux.h)
// ---------------------------------------------------------------

// sstatus 字段
pub const SSTATUS_UIE: u64 = 0x01;
pub const SSTATUS_SIE: u64 = 0x02;
pub const SSTATUS_SPIE: u64 = 0x20;
pub const SSTATUS_SPP: u64 = 0x100;
pub const SSTATUS_SUM: u64 = 0x40000;

// S 模式不区分使能与中断暂停，统一命名
pub const SSI: u64 = 0x2;
pub const STI: u64 = 0x20;
pub const SEI: u64 = 0x200;

// ---------------------------------------------------------------
//  CSR 读写宏
//  用法: read_csr!(sstatus), write_csr!(sie, val), clear_csr!(sip, mask)
//  `$csr:ident` -> `stringify!` -> 汇编器接收小写 CSR 寄存器名。
// ---------------------------------------------------------------

macro_rules! read_csr {
    ($csr:ident) => {{
        let val: u64;
        unsafe {
            core::arch::asm!(
                concat!("csrr {0}, ", stringify!($csr)),
                out(reg) val,
            );
        }
        val
    }};
}

macro_rules! write_csr {
    ($csr:ident, $val:expr) => {{
        unsafe {
            core::arch::asm!(
                concat!("csrw ", stringify!($csr), ", {0}"),
                in(reg) $val,
            );
        }
    }};
}

#[allow(unused_macros)]
macro_rules! set_csr {
    ($csr:ident, $mask:expr) => {{
        unsafe {
            core::arch::asm!(
                concat!("csrs ", stringify!($csr), ", {0}"),
                in(reg) $mask,
            );
        }
    }};
}

macro_rules! clear_csr {
    ($csr:ident, $mask:expr) => {{
        unsafe {
            core::arch::asm!(
                concat!("csrc ", stringify!($csr), ", {0}"),
                in(reg) $mask,
            );
        }
    }};
}

pub(crate) use clear_csr;
// pub(crate) use read_csr;
// pub(crate) use set_csr;
// pub(crate) use write_csr;

// ---------------------------------------------------------------
//  Named convenience wrappers
// ---------------------------------------------------------------

#[inline]
pub fn read_sstatus() -> u64 {
    read_csr!(sstatus)
}
#[inline]
pub fn write_sstatus(v: u64) {
    write_csr!(sstatus, v)
}
#[inline]
pub fn read_sie() -> u64 {
    read_csr!(sie)
}
#[inline]
pub fn write_sie(v: u64) {
    write_csr!(sie, v)
}
#[inline]
pub fn read_sip() -> u64 {
    read_csr!(sip)
}
#[inline]
pub fn write_sip(v: u64) {
    write_csr!(sip, v)
}
#[inline]
pub fn read_satp() -> u64 {
    read_csr!(satp)
}
#[inline]
pub fn write_satp(v: u64) {
    write_csr!(satp, v)
}
#[inline]
pub fn read_sepc() -> u64 {
    read_csr!(sepc)
}
#[inline]
pub fn write_sepc(v: u64) {
    write_csr!(sepc, v)
}
#[inline]
pub fn read_stvec() -> u64 {
    read_csr!(stvec)
}
#[inline]
pub fn write_stvec(v: u64) {
    write_csr!(stvec, v)
}
#[inline]
pub fn read_scause() -> u64 {
    read_csr!(scause)
}
#[inline]
pub fn read_stval() -> u64 {
    read_csr!(stval)
}
#[inline]
pub fn read_sscratch() -> u64 {
    read_csr!(sscratch)
}
#[inline]
pub fn write_sscratch(v: u64) {
    write_csr!(sscratch, v)
}
