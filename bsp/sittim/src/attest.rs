//! 飞地载荷与模块映像的完整性证明 (attestation)。
//!
//! 交付物尾部追加 64 字节 ECDSA (secp256r1) 签名 (r‖s), 对签名之前的映像字节
//! 计算摘要, 用内置公钥验证签名。纯 Rust (`sha2` + `p256`),
//! no_std / no-alloc / 整数运算。
//!
//! 摘要算法经 [`digest`] 单点隔离。更换 SM3 或私有算法时只改该函数, 但签名
//! 工具必须同步更换; 两者不同步时验签必然失败。

use p256::ecdsa::signature::hazmat::PrehashVerifier;
use p256::ecdsa::{Signature, VerifyingKey};
use sha2::{Digest, Sha256};

use crate::constants::{ATTEST_PUB_KEY, LINEAR_MAP_OFFSET, SHA256_DIGEST, SIG_LEN};

/// 计算消息的摘要, 长度为 [`SHA256_DIGEST`] 字节。
///
/// 全项目唯一的摘要计算点。飞地载荷与模块映像共用它, 故两者的摘要算法
/// 必然一致。
pub fn digest(msg: &[u8]) -> [u8; SHA256_DIGEST] {
	let mut out = [0u8; SHA256_DIGEST];
	out.copy_from_slice(&Sha256::digest(msg));
	out
}

/// 对模块窗口内 `[va, va+size)` 的模块映像做完整性证明。
///
/// 映像已经由 `ext_mod::loader` 映射到模块窗口, 故直接按 VA 读取, 无需地址翻译。
/// 布局与 [`attest_payload`] 相同: `[未签名的映像 (size-64 字节) | 签名 (64 字节)]`。
///
/// 返回 `true` 表示验证通过。`size <= 64` 时无有效映像, 返回 `false`。
pub fn attest_image(va: u64, size: u64) -> bool {
	let size = size as usize;
	if size <= SIG_LEN {
		return false;
	}

	let data = unsafe { core::slice::from_raw_parts(va as *const u8, size) };
	verify_trailer(data)
}

/// 对 `[pa, pa+size)` 的飞地载荷做完整性证明。
///
/// 载荷布局:
/// ```text
/// [ 未签名的载荷 (size-64 字节) | ECDSA 签名 r‖s (64 字节) ]
/// ```
///
/// 步骤:
/// 1. 经 linear-map 读取载荷字节 (与 [`crate::elf::load_elf`] 相同的翻译)
/// 2. 对未签名的载荷计算摘要 (见 [`digest`])
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
	verify_trailer(data)
}

/// 拆分未签名的映像与尾部签名, 计算摘要并验证签名。
///
/// `data` 的长度必须大于 [`SIG_LEN`], 由两个调用方各自在入口处保证。
fn verify_trailer(data: &[u8]) -> bool {
	let bare_len = data.len() - SIG_LEN;
	let (bare, sig_bytes) = data.split_at(bare_len);

	let digest = digest(bare);

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
	vk.verify_prehash(&digest, &sig).is_ok()
}
