//! OFDM 基带收发仿真: QPSK 调制 -> IFFT -> 循环前缀 -> AWGN 信道 -> FFT -> 解调。
//!
//! 简化模型: 无导频/保护带、无信道估计与均衡, 假设理想同步与平坦无衰落信道。
//! IFFT/FFT 复用 fft 模块的归一化薄封装 (逆变换除以 N, 前向不归一化)。

use std::f64::consts::PI;

use rustfft::num_complex::Complex;

use crate::fft;

/// 确定性伪随机数发生器 (xorshift64), 供 OFDM 比特流与 AWGN 噪声, 保证结果可复现。
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// 均匀分布 [0, 1)
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// 标准正态分布 N(0, 1) (Box-Muller 变换)
    fn next_gauss(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-12); // 避免 ln(0)
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }
}

/// QPSK 星座 (Gray 编码): bit1 = 实部符号, bit0 = 虚部符号 (0 -> +, 1 -> -),
/// 归一化使符号能量 |s|^2 = 1。00 -> (+1+i)/√2, 01 -> (+1-i)/√2, 11 -> (-1-i)/√2, 10 -> (-1+i)/√2。
fn qpsk_mod(b2: u8) -> Complex<f64> {
    let re = if b2 & 0b10 == 0 { 1.0 } else { -1.0 };
    let im = if b2 & 0b01 == 0 { 1.0 } else { -1.0 };
    let s = 1.0 / std::f64::consts::SQRT_2;
    Complex::new(re * s, im * s)
}

/// QPSK 硬判决 (最近邻): 按实部/虚部符号各恢复 1 bit。
fn qpsk_demod(c: Complex<f64>) -> u8 {
    let b1 = if c.re < 0.0 { 0b10 } else { 0 };
    let b0 = if c.im < 0.0 { 0b01 } else { 0 };
    b1 | b0
}

/// 由符号信噪比 Es/N0 (dB) 反推 AWGN 时域每维标准差。
///
/// 推导: QPSK 符号能量 Es=1, 归一化 IFFT 使时域每样本功率 = Es/N = 1/N;
/// 未归一化 FFT 把时域噪声功率 (每维 σ^2) 放大 N 倍到频域, 故频域每维噪声方差 = N σ^2,
/// 得 Es/N0 = 1 / (2 N σ^2), 反解 σ = 1 / sqrt(2 N · 10^{Es/N0 / 10})。
/// 无噪用 +inf -> sigma = 0。
pub fn awgn_sigma(snr_db: f64, n_fft: usize) -> f64 {
    if snr_db.is_infinite() {
        return 0.0;
    }
    (1.0 / (2.0 * n_fft as f64 * 10.0_f64.powf(snr_db / 10.0))).sqrt()
}

/// 单个 OFDM 链路仿真: QPSK -> IFFT -> 加循环前缀 -> AWGN -> 去前缀 -> FFT -> 解调,
/// 返回误比特率 (BER) 与总比特数。
pub fn simulate(n_fft: usize, n_cp: usize, n_sym: usize, sigma: f64, rng: &mut Rng) -> (f64, usize) {
    let total_bits = n_fft * n_sym * 2;
    let tx_bits: Vec<u8> = (0..total_bits).map(|_| (rng.next_u64() & 1) as u8).collect();

    let mut err = 0usize;
    let mut bit_idx = 0usize;
    for _ in 0..n_sym {
        // 调制: 每子载波 2 bit -> QPSK 符号
        let mut freq = vec![Complex::new(0.0, 0.0); n_fft];
        let mut sym = vec![0u8; n_fft];
        for (k, v) in freq.iter_mut().enumerate() {
            let b2 = (tx_bits[bit_idx] << 1) | tx_bits[bit_idx + 1];
            sym[k] = b2;
            *v = qpsk_mod(b2);
            bit_idx += 2;
        }
        // IFFT (归一化) -> 时域 OFDM 符号
        fft::inverse(&mut freq);
        // 加循环前缀: 时域 = [尾 n_cp 个样本] + [n_fft 个样本]
        let mut time = Vec::with_capacity(n_fft + n_cp);
        time.extend_from_slice(&freq[n_fft - n_cp..]);
        time.extend_from_slice(&freq);
        // AWGN 信道
        for v in time.iter_mut() {
            v.re += sigma * rng.next_gauss();
            v.im += sigma * rng.next_gauss();
        }
        // 去循环前缀 + FFT -> 频域符号
        let mut rx = time[n_cp..].to_vec();
        fft::forward(&mut rx);
        // 解调 + 逐 bit 计数
        for k in 0..n_fft {
            let diff = sym[k] ^ qpsk_demod(rx[k]);
            err += ((diff & 0b10) != 0) as usize + ((diff & 0b01) != 0) as usize;
        }
    }
    (err as f64 / total_bits as f64, total_bits)
}
