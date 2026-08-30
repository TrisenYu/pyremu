//! DTMF (双音多频, 电话拨号音) 音频检测。
//!
//! DTMF 是电话键盘拨号音, 每个按键由两个频率叠加而成: 一个来自低频组
//! (697/770/852/941 Hz), 一个来自高频组 (1209/1336/1477/1633 Hz), 标准见 ITU-T Q.23。
//! 检测用 Goertzel 算法在 8 个标准频率处各测一次 DFT 幅度, 再取每组最大值定位按键,
//! 相比全 FFT 只对单个频率做 O(N) 递推, 无频谱泄漏。

use std::f64::consts::PI;

/// DTMF 低频组标准频率 (Hz, ITU-T Q.23)
pub const LOW: [f64; 4] = [697.0, 770.0, 852.0, 941.0];
/// DTMF 高频组标准频率 (Hz, ITU-T Q.23)
pub const HIGH: [f64; 4] = [1209.0, 1336.0, 1477.0, 1633.0];
/// 键盘布局: 行 = 低频组下标, 列 = 高频组下标
pub const KEYS: [[char; 4]; 4] = [
    ['1', '2', '3', 'A'],
    ['4', '5', '6', 'B'],
    ['7', '8', '9', 'C'],
    ['*', '0', '#', 'D'],
];

/// Goertzel 算法: 计算信号 x 在单一目标频率 freq 处的 DFT 幅度 (采样率 fs)。
fn goertzel_mag(x: &[f64], fs: f64, freq: f64) -> f64 {
    let coeff = 2.0 * (2.0 * PI * freq / fs).cos();
    let mut s1 = 0.0f64;
    let mut s2 = 0.0f64;
    for &sample in x {
        let s = sample + coeff * s1 - s2;
        s2 = s1;
        s1 = s;
    }
    (s1 * s1 + s2 * s2 - coeff * s1 * s2).sqrt()
}

/// 生成某 DTMF 数字的时域信号: 低频组频率与高频组频率等幅叠加。
pub fn synth(low: f64, high: f64, fs: f64, n: usize) -> Vec<f64> {
    let amp = 0.5; // 两音各 0.5 幅度, 避免叠加溢出
    (0..n)
        .map(|i| {
            let t = i as f64 / fs;
            amp * (2.0 * PI * low * t).sin() + amp * (2.0 * PI * high * t).sin()
        })
        .collect()
}

/// DTMF 检测: 在低频组/高频组内各取 Goertzel 幅度最大者, 返回 (行, 列)。
pub fn detect(x: &[f64], fs: f64) -> (usize, usize) {
    let low: Vec<f64> = LOW.iter().map(|&f| goertzel_mag(x, fs, f)).collect();
    let high: Vec<f64> = HIGH.iter().map(|&f| goertzel_mag(x, fs, f)).collect();
    let row = low.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
    let col = high.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
    (row, col)
}
