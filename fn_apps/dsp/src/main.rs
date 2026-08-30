//! dsp — 数字信号处理 (FFT 谱分析 + FIR 低通滤波 + OFDM 基带收发 + DTMF 音频检测)
//!
//! 四部分, 各部分的算法定义拆分到独立模块, main.rs 负责自检编排:
//!   - fft:   复数 FFT/IFFT 薄封装 (rustfft, 前向未归一化 / 逆变换归一化)。
//!   - fir:   Hamming 窗 sinc 低通滤波器设计 + 频率响应。
//!   - ofdm:  QPSK + IFFT/FFT + 循环前缀 + AWGN 的基带收发仿真。
//!   - dtmf:  电话拨号音 (双音多频) 的 Goertzel 检测。
//!
//! 自检量 (均可解析/数值精确验证):
//!   - Parseval 定理: 时域总能量 = 频域总能量 / N (DFT 能量守恒)。
//!   - 单一正弦: sin(2π k0 n / N) 的频谱仅在 bin k0 处有峰, 幅为 N/2。
//!   - 往返一致: IFFT(FFT(x)) == x (舍入精度内)。
//!   - FIR: 通带 (含 DC) 增益 ≈ 1, 阻带增益 ≈ 0。
//!   - OFDM: 无噪信道误比特率 (BER) == 0。
//!   - DTMF: 16 个拨号数字 (697..941 Hz 低频组 + 1209..1633 Hz 高频组) 全部识别正确。
//!
//! 单位: 频率归一化到采样率 (f = 0.5 即 Nyquist), 截止频率 fc 无量纲;
//!       信噪比用符号信噪比 Es/N0 (dB)。

use std::f64::consts::PI;

use rustfft::num_complex::Complex;

mod dtmf;
mod fft;
mod fir;
mod ofdm;

fn main() {
    println!("=== DSP: FFT spectrum + FIR low-pass + OFDM transceiver + DTMF audio ===");

    // ---- 第 1 部分: FFT 谱分析 ----
    let n = 256usize;
    let k0 = 16usize;
    println!();
    println!("[dsp] FFT: N={n}, test signal x[n] = sin(2*pi*{k0}*n/N)");

    let x: Vec<Complex<f64>> = (0..n)
        .map(|i| Complex::new((2.0 * PI * k0 as f64 * i as f64 / n as f64).sin(), 0.0))
        .collect();

    // 自检 1: Parseval 能量守恒 (rustfft 前向未归一化, 故频域能量除以 N)
    let e_time: f64 = x.iter().map(|c| c.norm_sqr()).sum();
    let mut xf = x.clone();
    fft::forward(&mut xf);
    let e_freq: f64 = xf.iter().map(|c| c.norm_sqr()).sum::<f64>() / n as f64;
    let parseval_ok = (e_time - e_freq).abs() < 1e-9 * e_time.max(1.0);
    println!("[dsp]   time energy = {e_time:.9}, freq energy/N = {e_freq:.9}");
    println!(
        "[{}] Parseval: time energy == freq energy / N",
        if parseval_ok { "PASS" } else { "FAIL" }
    );

    // 自检 2: 频谱峰位于 bin k0 (幅 N/2), 其余 bin 幅 ≈ 0
    let mag_peak = xf[k0].norm();
    let mag_mirror = xf[n - k0].norm();
    let others: f64 = xf
        .iter()
        .enumerate()
        .filter(|&(k, _)| k != k0 && k != n - k0)
        .map(|(_, c)| c.norm())
        .sum();
    let peak_ok = (mag_peak - n as f64 / 2.0).abs() < 1e-9 && others < 1e-9;
    println!(
        "[dsp]   |X[{k0}]| = {mag_peak:.6}, |X[{}]| = {mag_mirror:.6} (expect {:.0})",
        n - k0,
        n as f64 / 2.0
    );
    println!("[dsp]   sum of non-peak |X[k]| = {others:.3e} (expect 0)");
    println!(
        "[{}] spectral peak at bin k0 with magnitude N/2",
        if peak_ok { "PASS" } else { "FAIL" }
    );

    // 自检 3: IFFT(FFT(x)) == x 往返一致 (fft::inverse 已归一化)
    let mut y = x.clone();
    fft::forward(&mut y);
    fft::inverse(&mut y);
    let mut roundtrip_err = 0.0f64;
    for (i, &c) in y.iter().enumerate() {
        roundtrip_err = roundtrip_err.max((c - x[i]).norm());
    }
    let roundtrip_ok = roundtrip_err < 1e-12;
    println!("[dsp]   max |IFFT(FFT(x)) - x| = {roundtrip_err:.3e}");
    println!(
        "[{}] IFFT(FFT(x)) == x (round-trip)",
        if roundtrip_ok { "PASS" } else { "FAIL" }
    );

    // ---- 第 2 部分: FIR 低通滤波 ----
    let m = 63usize; // 阶数 (64 个系数)
    let fc = 0.25; // 归一化截止频率
    let h = fir::lowpass(m, fc);
    println!();
    println!("[dsp] FIR low-pass: order={m}, fc={fc} (normalized), Hamming window");

    // 自检 4: 频率响应 通带 ≈ 1, 阻带 ≈ 0
    let h_dc = fir::freq_response(&h, 0.0).norm();
    let h_pass = fir::freq_response(&h, 0.05).norm();
    let h_stop = fir::freq_response(&h, 0.40).norm();
    let fir_ok = (h_dc - 1.0).abs() < 1e-12 && (h_pass - 1.0).abs() < 0.01 && h_stop < 0.01;
    println!("[dsp]   |H(0.00)| = {h_dc:.6}  (DC, expect 1)");
    println!("[dsp]   |H(0.05)| = {h_pass:.6}  (passband, expect ~1)");
    println!("[dsp]   |H(0.40)| = {h_stop:.6}  (stopband, expect ~0)");
    println!(
        "[{}] FIR frequency response (passband ~1, stopband ~0)",
        if fir_ok { "PASS" } else { "FAIL" }
    );

    // ---- 第 3 部分: OFDM 基带收发 ----
    let n_fft = 64usize; // 数据子载波数 (简化, 无导频/保护带)
    let n_cp = 16usize; // 循环前缀长度
    let n_sym = 16usize; // OFDM 符号数
    println!();
    println!("[dsp] OFDM: {n_fft} subcarriers, CP={n_cp}, {n_sym} symbols, QPSK");

    // 自检 5: 无噪信道 BER == 0 (IFFT/FFT + 循环前缀往返无损)
    let mut rng = ofdm::Rng::new(0x5eed_2026_08_30); // 固定种子, 结果可复现
    let (ber_clean, total) = ofdm::simulate(n_fft, n_cp, n_sym, 0.0, &mut rng);
    let clean_ok = ber_clean == 0.0;
    println!("[dsp]   clean channel: BER = {ber_clean:.6} over {total} bits");
    println!(
        "[{}] OFDM clean-channel BER == 0 (lossless IFFT/FFT round-trip)",
        if clean_ok { "PASS" } else { "FAIL" }
    );

    // 演示: AWGN 信道下 BER 随 SNR 单调下降 (仅打印, 不作硬阈值)
    println!("[dsp]   AWGN channel (Es/N0 vs BER):");
    for snr_db in [0.0_f64, 5.0, 10.0] {
        let (ber, _) = ofdm::simulate(n_fft, n_cp, n_sym, ofdm::awgn_sigma(snr_db, n_fft), &mut rng);
        println!("[dsp]     Es/N0 = {snr_db:>4.1} dB  ->  BER = {ber:.6}");
    }

    // ---- 第 4 部分: DTMF 音频检测 (电话拨号音) ----
    let fs = 8000.0; // 电话采样率
    let n_audio = 320; // 40ms 音长 (DTMF 规范最短时长)
    println!();
    println!("[dsp] DTMF (dual-tone multi-frequency) audio detection, fs={} Hz", fs as i32);

    let mut dtmf_ok = true;
    for row in 0..4 {
        for col in 0..4 {
            let expected = dtmf::KEYS[row][col];
            let x = dtmf::synth(dtmf::LOW[row], dtmf::HIGH[col], fs, n_audio);
            let (dr, dc) = dtmf::detect(&x, fs);
            let ok = dr == row && dc == col;
            dtmf_ok &= ok;
            println!(
                "[dsp]   digit '{}' ({:.0}+{:.0} Hz) -> detected '{}' [{}]",
                expected,
                dtmf::LOW[row],
                dtmf::HIGH[col],
                dtmf::KEYS[dr][dc],
                if ok { "PASS" } else { "FAIL" }
            );
        }
    }
    println!(
        "[{}] DTMF decoder identifies all 16 digits",
        if dtmf_ok { "PASS" } else { "FAIL" }
    );

    println!();
    println!("=== dsp: done ===");
}
