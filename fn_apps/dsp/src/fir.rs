//! FIR 低通滤波器设计 (Hamming 窗 sinc) 与频率响应。

use std::f64::consts::PI;

use rustfft::num_complex::Complex;

/// 频率响应 H(f) = Σ h[n] exp(-i 2π f n) (f 归一化到采样率)。
pub fn freq_response(h: &[f64], f: f64) -> Complex<f64> {
    let mut acc = Complex::new(0.0, 0.0);
    for (n, &v) in h.iter().enumerate() {
        acc += Complex::new(0.0, -2.0 * PI * f * n as f64).exp() * v;
    }
    acc
}

/// 设计 Hamming 窗 sinc 低通 FIR (理想低通冲激响应乘 Hamming 窗, 再归一化 DC 增益)。
/// 返回 m+1 个系数 (m 为阶数), 截止频率 fc 归一化到采样率。
pub fn lowpass(m: usize, fc: f64) -> Vec<f64> {
    let mid = m as f64 / 2.0;
    let mut h = vec![0.0f64; m + 1];
    let mut sum = 0.0;
    for (n, v) in h.iter_mut().enumerate() {
        let x = n as f64 - mid;
        // 理想低通冲激响应 2 fc sinc(2 fc x) = sin(2π fc x) / (π x)
        let ideal = if x == 0.0 { 2.0 * fc } else { (2.0 * PI * fc * x).sin() / (PI * x) };
        let win = 0.54 - 0.46 * (2.0 * PI * n as f64 / m as f64).cos(); // Hamming
        *v = ideal * win;
        sum += *v;
    }
    for v in h.iter_mut() {
        *v /= sum; // 归一化 -> 频率响应 H(0) = 1
    }
    h
}
