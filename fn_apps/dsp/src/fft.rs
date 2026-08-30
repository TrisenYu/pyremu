//! 复数 FFT/IFFT 薄封装 (rustfft), 统一前向未归一化 / 逆变换归一化的语义。

use rustfft::{num_complex::Complex, FftPlanner};

/// 前向 FFT (未归一化, 与 rustfft 一致): X[k] = Σ x[n] e^{-i 2π kn/N}。
pub fn forward(x: &mut [Complex<f64>]) {
    let mut planner = FftPlanner::<f64>::new();
    planner.plan_fft_forward(x.len()).process(x);
}

/// 逆 FFT 并除以 N, 使 IFFT(FFT(x)) == x (rustfft 逆变换本身未归一化)。
pub fn inverse(x: &mut [Complex<f64>]) {
    let mut planner = FftPlanner::<f64>::new();
    planner.plan_fft_inverse(x.len()).process(x);
    let n = x.len() as f64;
    for v in x.iter_mut() {
        *v /= n;
    }
}
