//! 飞地 payload / 内核完整性证明 (attestation)。
//!
//! 载荷尾部追加 64 字节 ECDSA (secp256r1) 签名 (r‖s)，对前面的 bare
//! 载荷计算 SHA-256，用内置公钥验证签名。纯 Rust (`sha2` + `p256`)，
//! no_std / no-alloc / 整数运算 (兼容 riscv64imac 无 F/D)。
//!
//! 与 ref-emod `attest_emodule` 的差异: 使用 SHA-256 (真正 32 字节摘要),
//! 不做 MD2 的 16->32 字节复制填充。
//!
//! 后续可用密码学加速指令替换 SHA-256 / ECDSA 计算, 但本模块的
//! `attest_payload` 接口保持不变。

use p256::ecdsa::signature::hazmat::PrehashVerifier;
use p256::ecdsa::{Signature, VerifyingKey};
use sha2::{Digest, Sha256};

use crate::constants::{ATTEST_PUB_KEY, LINEAR_MAP_OFFSET, SIG_LEN};

/// 对 `[pa, pa+size)` 的载荷做完整性证明。
///
/// 载荷布局:
/// ```text
/// [ bare 载荷 (size-64 字节) | ECDSA 签名 r‖s (64 字节) ]
/// ```
///
/// 步骤:
/// 1. 经 linear-map 读取载荷字节 (与 [`crate::elf::load_elf`] 相同的翻译)
/// 2. 对 bare 载荷计算 SHA-256 摘要
/// 3. 用内置公钥 [`ATTEST_PUB_KEY`] 验证尾部 64 字节签名
///
/// 返回 `true` 表示验证通过。`size <= 64` 时无有效载荷, 返回 `false`。
pub fn attest_payload(pa: u64, size: u64) -> bool {
    let size = size as usize;
    if size <= SIG_LEN {
        return false;
    }

    // 经 linear-map 翻译读取载荷 (elf.rs 相同模式): PA + LINEAR_MAP_OFFSET。
    let va = pa.wrapping_add(LINEAR_MAP_OFFSET);
    let data = unsafe { core::slice::from_raw_parts(va as *const u8, size) };

    // 拆分 bare 载荷与尾部签名。
    let bare_len = size - SIG_LEN;
    let (bare, sig_bytes) = data.split_at(bare_len);

    // SHA-256 摘要 (32 字节)。
    let digest = Sha256::digest(bare);

    // 解析压缩公钥 (33 字节 SEC1) 与签名 (64 字节 r‖s)。
    let vk = match VerifyingKey::from_sec1_bytes(&ATTEST_PUB_KEY) {
        Ok(k) => k,
        Err(_) => return false,
    };
    let sig = match Signature::from_slice(sig_bytes) {
        Ok(s) => s,
        Err(_) => return false,
    };

    // 预哈希验证 (摘要已算好, 直接对 32 字节验证)。
    vk.verify_prehash(digest.as_slice(), &sig).is_ok()
}
