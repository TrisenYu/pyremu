//! 所有魔数集中定义。
//!
//! 平台可调常量由 Makefile 从 config.mk 生成 config_gen.rs 注入（Kbuild 风格）。
//!
//! 参照：
//!   ref-emod/emod_manager/config.mk           VA 布局
//!   smode_entry/ecall_types.h                 ENCLAVE_EXT_ID = 0x20221222
//!   custom-opensbi/include/enclave_ext/enclave_types.h  函数 ID
//!
//! 此文件为完整常量参考，未使用的项目有意保留。

#![allow(dead_code)]

// ---- 由 Makefile 从 config.mk 生成 ----
include!("config_gen.rs");

// ---------------------------------------------------------------
//  PTE 标志位
// ---------------------------------------------------------------

pub const PTE_V: u8 = 1 << 0;
pub const PTE_R: u8 = 1 << 1;
pub const PTE_W: u8 = 1 << 2;
pub const PTE_X: u8 = 1 << 3;
pub const PTE_U: u8 = 1 << 4;
pub const PTE_G: u8 = 1 << 5;
pub const PTE_A: u8 = 1 << 6;
pub const PTE_D: u8 = 1 << 7;

// ---------------------------------------------------------------
//  Sv39 层级
// ---------------------------------------------------------------

pub const LEVEL_GIGA: u8 = 0; // 1 GiB 超级页
pub const LEVEL_MEGA: u8 = 1; // 2 MiB 超级页
pub const LEVEL_PAGE: u8 = 2; // 4 KiB 普通页

pub const SV39_VPN_LEN: u8 = 9;

// ---------------------------------------------------------------
//  飞地扩展 ID 与函数号
// ---------------------------------------------------------------

pub const ENCLAVE_EXT_ID: u64 = 0x2022_1222;

pub const ENCLAVE_CALL_SUSPEND: u64 = 404;
pub const ENCLAVE_CALL_SHUTDOWN: u64 = 403;
pub const ENCLAVE_CALL_MEM_ALLOC: u64 = 500;
pub const ENCLAVE_CALL_GET_ID: u64 = 407;
pub const ENCLAVE_CALL_GET_HARTID: u64 = 408;
pub const ENCLAVE_CALL_GET_AVAILABLE_MEM: u64 = 409;
pub const ENCLAVE_CALL_UNMATCHED_ACC_FAULT: u64 = 506;

// ---------------------------------------------------------------
//  标准 SBI
// ---------------------------------------------------------------

pub const SBI_LEGACY_PUTCHAR_EXT: u64 = 0x01;
pub const SBI_TIMER_EXT: u64 = 0x5449_4D45;
pub const SBI_SET_TIMER_FUNC: u64 = 0x00;

// ---------------------------------------------------------------
//  VA 布局（来自 ref-emod/config.mk）
// ---------------------------------------------------------------

pub const ENCLAVE_MAN_VA_START: u64 = 0xFFFF_FFE0_0000_0000;
pub const ENCLAVE_MODULE_LOAD_VA_INIT: u64 = 0xFFFF_FFF0_0000_0000;

pub const LINEAR_MAP_START: u64 = 0x1_4000_0000;
pub const LINEAR_MAP_SIZE: u64 = 0x3_4000_0000;
pub const LINEAR_MAP_OFFSET: u64 = 0xFFFF_FFC0_0000_0000;

pub const UMODE_HEAP_START_ALIGNED: u64 = 0x1_0000_0000;
pub const UMODE_STACK_TOP_VA: u64 = 0x1_4000_0000;

/// mmap 匿名映射起始 VA, 从高地址向下增长, 避开 heap (0x1_0000_0000) 和 stack.
pub const UMODE_MMAP_BASE: u64 = 0x2_0000_0000;

// ---------------------------------------------------------------
//  大小常量
// ---------------------------------------------------------------

pub const PAGE_SIZE: u64 = 0x1000;
pub const PAGE_SHIFT: u64 = 12;

/// M-mode 内存分配 / PMP 保护的最小粒度 = 2 MiB。
pub const CHUNK_2M_SIZE: u64 = 0x20_0000;
pub const CHUNK_2M_SHIFT: u64 = 21;

pub const UMODE_STACK_SIZE_TOTAL: u64 = 0x10_0000;

// ---------------------------------------------------------------
//  ELF 常量
// ---------------------------------------------------------------

pub const ELF_MAGIC: [u8; 4] = [0x7F, b'E', b'L', b'F'];
pub const ELFCLASS64: u8 = 2;
pub const ET_EXEC: u16 = 2;
pub const EM_RISCV: u16 = 243;
pub const PT_LOAD: u32 = 1;
pub const PF_R: u32 = 4;
pub const PF_W: u32 = 2;
pub const PF_X: u32 = 1;

// ---------------------------------------------------------------
//  Attestation (secp256r1 ECDSA + SHA-256)
// ---------------------------------------------------------------

/// secp256r1 标量字节长度 (私钥 / 坐标)。
pub const ECC_BYTES: usize = 32;
/// 压缩公钥长度 = 0x02/0x03 前缀 + X 坐标。
pub const PUB_KEY_LEN: usize = ECC_BYTES + 1; // 33
/// ECDSA 签名长度 = r‖s。
pub const SIG_LEN: usize = ECC_BYTES * 2; // 64
/// SHA-256 摘要长度。
pub const SHA256_DIGEST: usize = 32;

// ATTEST_PUB_KEY: [u8; 33] 由 Makefile 从 config.mk 生成注入 config_gen.rs。

// ---------------------------------------------------------------
//  测试（仅 host 端编译时可用）
// ---------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pte_v_flag_is_lsb() {
        assert_eq!(PTE_V, 1);
    }

    #[test]
    fn test_pte_flags_are_distinct() {
        let flags = [PTE_V, PTE_R, PTE_W, PTE_X, PTE_U, PTE_G, PTE_A, PTE_D];
        for i in 0..flags.len() {
            for j in (i + 1)..flags.len() {
                assert_ne!(
                    flags[i], flags[j],
                    "PTE flags at position {i} and {j} overlap"
                );
            }
        }
    }

    #[test]
    fn test_chunk_2m_is_sv39_mega_page() {
        // CHUNK_2M_SIZE 必须等于一个 Sv39 mega page
        assert_eq!(CHUNK_2M_SIZE, 1 << (PAGE_SHIFT + SV39_VPN_LEN as u64));
    }

    #[test]
    fn test_page_constants_consistent() {
        assert_eq!(PAGE_SIZE, 1 << PAGE_SHIFT);
        assert_eq!(CHUNK_2M_SIZE, 1 << CHUNK_2M_SHIFT);
    }
}
