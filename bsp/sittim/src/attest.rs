//! 扩展模块映像的完整性证明。
//!
//! 映像尾部追加 64 字节 ECDSA (secp256r1) 签名, 签名对象是签名之前的映像字节的摘要,
//! 用内置公钥验证。验证在映像的入口被调用之前完成, 见 [`crate::ext_mod::loader`]。
//!
//! 摘要算法经 [`digest`] 单点隔离。更换 SM3 或私有算法时只改该函数, 但签名
//! 工具必须同步更换

use p256::ecdsa::signature::hazmat::PrehashVerifier;
use p256::ecdsa::{ Signature, VerifyingKey };
use sha2::{ Digest, Sha256 };

use crate::constants::{ ATTEST_PUB_KEY, SHA256_DIGEST, SIG_LEN };

/// 计算消息的摘要, 长度为 [`SHA256_DIGEST`] 字节。
pub fn digest(msg: &[u8]) -> [u8; SHA256_DIGEST] {
	let mut out = [0u8; SHA256_DIGEST];
	out.copy_from_slice(&Sha256::digest(msg));
	out
}

/// 对模块窗口内 `[va, va+size)` 的模块映像做完整性证明。
///
/// 映像已经由 `ext_mod::loader` 映射到模块窗口, 故直接按 VA 读取, 无需地址翻译。
/// 映像的布局是 `[未签名的映像 (size-64 字节) | 签名 (64 字节)]`。
///
/// 返回 `true` 表示验证通过。`size <= 64` 时无有效映像, 返回 `false`。
pub fn attest_image(va: u64, size: u64) -> bool {
	let size = size as usize;
	if size <= SIG_LEN {
		return false;
	}

	let data = unsafe { core::slice::from_raw_parts(va as *const u8, size) };
	verify_trail_sign(data)
}

/// 拆分未签名的映像与尾部签名, 计算摘要并验证签名。
///
/// `data` 的长度必须大于 [`SIG_LEN`]。
fn verify_trail_sign(data: &[u8]) -> bool {
	let bare_len = data.len() - SIG_LEN;
	let (bare, sig_bytes) = data.split_at(bare_len);

	let digest = digest(bare);

	// 解析 33 字节 SEC1 压缩公钥与 64 字节 r‖s 签名。
	let vk = match VerifyingKey::from_sec1_bytes(&ATTEST_PUB_KEY) {
		Ok(k) => k,
		Err(_) => {
			return false;
		}
	};
	let sig = match Signature::from_slice(sig_bytes) {
		Ok(s) => s,
		Err(_) => {
			return false;
		}
	};

	vk.verify_prehash(&digest, &sig).is_ok()
}
