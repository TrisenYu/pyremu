//! ising — 2D 方格子 Ising 模型 Monte Carlo (统计物理)
//!
//! 哈密顿量 H = -J Σ_{<i,j>} s_i s_j, 自旋 s_i = ±1, J=1, 周期性边界条件。
//! Metropolis 算法: 随机翻转一个自旋, 能量变化 ΔE = 2 J s_i Σ_邻居 s_j,
//! 以概率 min(1, exp(-ΔE/kT)) 接受。随机数使用 rand crate (StdRng, 固定种子)。
//!
//! 自检: 2D Ising 有 Onsager (1944) 精确解, 临界温度
//!     Tc = 2 / ln(1 + √2) ≈ 2.269185
//! 有限尺寸下比热峰位置略高于 Tc 并随 L 增大逼近; 判定峰处于 Tc ± 0.15 内。
//! 另验证低温 (T=1.0) 铁磁有序 (m->1) 与高温 (T=3.5) 顺磁无序 (m->0)。
//! 附加 Binder 累积量 U4 = 1 - <m^4> / (3 <m^2>^2) (连续相变定位辅助量)。

use rand::{rngs::StdRng, Rng, SeedableRng};

/// 格点边长 (方格子 L×L)
const L: usize = 16;
/// 总自旋数
const N: usize = L * L;
/// 耦合常数 J (归一化 1)
const J: f64 = 1.0;
const NTHERM: usize = 1000;
const NSAMP: usize = 3000;

/// 单个自旋翻转 (Metropolis), beta = 1/(kT)
fn metropolis_step(spins: &mut [i8], rng: &mut StdRng, beta: f64) {
    let site = rng.gen_range(0..N);
    let (i, j) = (site / L, site % L);
    // 周期边界的四个邻居
    let up = ((i + L - 1) % L) * L + j;
    let down = ((i + 1) % L) * L + j;
    let left = i * L + (j + L - 1) % L;
    let right = i * L + (j + 1) % L;
    let sum = spins[up] + spins[down] + spins[left] + spins[right];
    let de = 2.0 * J * (spins[site] as f64) * (sum as f64);
    if de <= 0.0 || rng.gen::<f64>() < (-de * beta).exp() {
        spins[site] = -spins[site];
    }
}

/// 总能量 E = -J Σ_{<i,j>} s_i s_j (每对键只计一次: 右邻 + 下邻)
fn total_energy(spins: &[i8]) -> f64 {
    let mut e = 0.0;
    for i in 0..L {
        for j in 0..L {
            let s = spins[i * L + j] as f64;
            let right = spins[i * L + (j + 1) % L] as f64;
            let down = spins[((i + 1) % L) * L + j] as f64;
            e += -J * s * (right + down);
        }
    }
    e
}

/// 总磁化 M = Σ s_i
fn total_magnetization(spins: &[i8]) -> f64 {
    spins.iter().map(|&s| s as f64).sum()
}

/// 在温度 T 下做 Monte Carlo, 返回 (每自旋能量, 每自旋 |m|, 比热 C, 磁化率 χ, Binder U4)
fn measure(spins: &mut [i8], rng: &mut StdRng, t: f64) -> (f64, f64, f64, f64, f64) {
    let beta = 1.0 / t;
    // 平衡
    for _ in 0..NTHERM {
        for _ in 0..N {
            metropolis_step(spins, rng, beta);
        }
    }
    // 测量
    let (mut se, mut se2, mut sm, mut sm2, mut sm4) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for _ in 0..NSAMP {
        for _ in 0..N {
            metropolis_step(spins, rng, beta);
        }
        let e = total_energy(spins);
        let m = total_magnetization(spins);
        let m2 = m * m;
        se += e;
        se2 += e * e;
        sm += m.abs();
        sm2 += m2;
        sm4 += m2 * m2;
    }
    let ns = NSAMP as f64;
    let nn = N as f64;
    let e_avg = se / ns;
    let m_avg = sm / ns;
    // 比热 C = (<E²> - <E>²) / (N T²)
    let c = (se2 / ns - e_avg * e_avg) / (nn * t * t);
    // 磁化率 χ = (<M²> - <M>²) / (N T)
    let chi = (sm2 / ns - m_avg * m_avg) / (nn * t);
    // Binder 累积量 U4 = 1 - <m^4> / (3 <m^2>²)
    let u4 = 1.0 - (sm4 / ns) / (3.0 * (sm2 / ns).powi(2));
    (e_avg / nn, m_avg / nn, c, chi, u4)
}

/// 起始高温无序配置 (随机 ±1)
fn init_random(spins: &mut [i8], rng: &mut StdRng) {
    for s in spins.iter_mut() {
        *s = if rng.gen::<f64>() < 0.5 { 1 } else { -1 };
    }
}

fn main() {
    println!("=== 2D Ising model Monte Carlo (Metropolis, rand::StdRng, periodic BC) ===");
    println!("[ising] L = {L}, N = {N} spins, J = {J}");

    let tc_exact = 2.0 / (1.0 + 2.0_f64.sqrt()).ln(); // Onsager 精确解
    println!("[ising] Onsager exact Tc = {tc_exact:.6}");

    let mut rng = StdRng::seed_from_u64(0x1234_5678_9ABC_DEF0);

    // 自检 1/2: 低温有序, 高温无序
    let mut spins = vec![0i8; N];
    init_random(&mut spins, &mut rng);
    let (_, m_low, _, _, _) = measure(&mut spins, &mut rng, 1.0);
    let (_, m_high, _, _, _) = measure(&mut spins, &mut rng, 3.5);
    println!("\n[ising] magnetization |m|: T=1.0 -> {m_low:.4} (expect ~1, ferromagnetic)");
    println!("[ising] magnetization |m|: T=3.5 -> {m_high:.4} (expect ~0, paramagnetic)");
    println!(
        "[{}] ordered at low T, disordered at high T",
        if m_low > 0.9 && m_high < 0.15 { "PASS" } else { "FAIL" }
    );

    // 温度扫描, 找比热峰定位 Tc
    println!();
    println!("[ising] temperature sweep (T, E/N, |m|, C, chi, U4):");
    let mut tc_peak = 0.0;
    let mut c_max = -1.0;
    let mut t = 1.8;
    while t <= 2.8001 {
        let (e, m, c, chi, u4) = measure(&mut spins, &mut rng, t);
        if c > c_max {
            c_max = c;
            tc_peak = t;
        }
        println!(
            "[ising]   T={t:.2}  E/N={e:.3}  |m|={m:.3}  C={c:.4}  chi={chi:.4}  U4={u4:.4}"
        );
        t += 0.05;
    }
    println!(
        "\n[ising] specific-heat peak at T = {tc_peak:.2} (exact Tc = {tc_exact:.6})"
    );
    println!(
        "[{}] Tc peak within 0.15 of Onsager value (finite-size shift accounted)",
        if (tc_peak - tc_exact).abs() < 0.15 { "PASS" } else { "FAIL" }
    );

    println!("\n=== ising: done ===");
}
