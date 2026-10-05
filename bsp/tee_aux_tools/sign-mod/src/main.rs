//! 为模块文件追加签名。
//!
//! 用法: sign-mod <私钥.pem> <输入文件> <输出文件>
//!
//! 输出文件为输入文件后接 64 字节 ECDSA (secp256r1) 签名 r‖s。摘要算法与 sittim 的
//! `attest::digest` 同为 SHA-256, 两侧不同步时验签必然失败。
//!
//! 私钥为 SEC1 格式的 PEM, 公钥由私钥推出并打印, 供与 sittim 的 `config.mk` 中
//! ATTEST_PUB_KEY 比对。

use std::fs;
use std::process::ExitCode;

use p256::ecdsa::signature::hazmat::PrehashSigner;
use p256::ecdsa::{Signature, SigningKey};
use p256::elliptic_curve::sec1::ToSec1Point;
use p256::SecretKey;
use sha2::{Digest, Sha256};

/// 签名的字节数, 即 r 与 s 两个 32 字节标量。
const SIG_LEN: usize = 64;

fn main() -> ExitCode {
	match run() {
		Ok(()) => ExitCode::SUCCESS,
		Err(message) => {
			eprintln!("sign-mod: {message}");
			ExitCode::FAILURE
		}
	}
}

fn run() -> Result<(), String> {
	let args: Vec<String> = std::env::args().collect();
	if args.len() != 4 {
		return Err("usage: sign-mod <private_key.pem> <input_file> <output_file>".to_string());
	}
	let (key_path, input_path, out_path) = (&args[1], &args[2], &args[3]);

	let pem = fs::read_to_string(key_path).map_err(|err| format!("cannot read {key_path}: {err}"))?;
	let secret = SecretKey::from_sec1_pem(&pem)
		.map_err(|err| format!("cannot parse the private key: {err}"))?;

	let input = fs::read(input_path).map_err(|err| format!("cannot read {input_path}: {err}"))?;

	// 签名类型须显式给出: `SigningKey` 对定长 r‖s 与 DER 两种签名都实现了
	// `PrehashSigner`, 仅凭返回值的后续用法无法定下其类型。
	let digest = Sha256::digest(&input);
	let signature: Signature = SigningKey::from(&secret)
		.sign_prehash(&digest)
		.map_err(|err| format!("cannot sign: {err}"))?;

	let signature_bytes = signature.to_bytes();
	if signature_bytes.len() != SIG_LEN {
		return Err(format!(
			"signature is {} bytes, expected {SIG_LEN}",
			signature_bytes.len()
		));
	}

	let mut out = input;
	out.extend_from_slice(&signature_bytes);
	fs::write(out_path, &out).map_err(|err| format!("cannot write {out_path}: {err}"))?;

	// 打印推得的公钥, 供与 sittim 的 ATTEST_PUB_KEY 比对。
	let public = secret.public_key().to_sec1_point(true);
	println!("public key: {}", hex(public.as_bytes()));

	Ok(())
}

/// 把字节序列写作十六进制字符串。
fn hex(bytes: &[u8]) -> String {
	let mut out = String::with_capacity(bytes.len() * 2);
	for byte in bytes {
		out.push_str(&format!("{byte:02x}"));
	}
	out
}
