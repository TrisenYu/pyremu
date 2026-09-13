//! graphene — 石墨烯紧束缚能带与态密度 (凝聚态电子结构)
//!
//! 物理模型: 石墨烯蜂窝晶格, 每个原胞两个碳原子 (A/B 子晶格), 每个原子一个
//! 2p_z 轨道, 仅考虑最近邻跳跃 t。布洛赫基下紧束缚哈密顿量为 2x2 Hermitian:
//!
//!     H(k) = [ 0      f(k)  ]
//!            [ f*(k)  0     ]
//!
//! 其中 f(k) = -t (e^{i k·δ1} + e^{i k·δ2} + e^{i k·δ3}), δ 为三个最近邻矢量。
//! 能带解析解 E±(k) = ±|f(k)|。此处用 nalgebra 的复数 Matrix2 表示 H(k) 并做
//! Hermitian 对角化 (SymmetricEigen), 复数类型复用 num-complex; 态密度在倒空间
//! 布里渊区网格上采样对角化本征值。
//!
//! 自检量 (均可解析/数值精确验证):
//!   1. Dirac 点: K = (4π/(3√3), 0) 处 |f(K)|=0, 能带无带隙 (半金属)。
//!   2. 费米速度: E(K+q) 线性色散, 斜率 vF = 3 t a / 2 = 1.5 (t=a=1, ℏ=1)。
//!   3. 高对称点能量: Γ 点 E=3t (带顶), M 点 E=t (鞍点)。
//!   4. 态密度: 归一化 ∫ρ(E)dE = 2 (两条能带), van Hove 奇点峰位 E ≈ t。
//!
//! 单位: 键长 a=1, 跳跃 t=1 (对应真实石墨烯 t≈2.8 eV)。

use nalgebra::{Matrix2, SymmetricEigen};
use num_complex::Complex64;
use std::f64::consts::PI;

/// 最近邻跳跃积分 t (归一化为 1)
const T: f64 = 1.0;

/// 三个最近邻矢量 (键长 a=1): δ1=(√3/2, 1/2), δ2=(-√3/2, 1/2), δ3=(0,-1)
fn nn_vectors() -> [[f64; 2]; 3] {
    [
        [3.0_f64.sqrt() / 2.0, 0.5],
        [-3.0_f64.sqrt() / 2.0, 0.5],
        [0.0, -1.0],
    ]
}

/// f(k) = -t Σ e^{i k·δ}
fn fk(k: [f64; 2]) -> Complex64 {
    let mut f = Complex64::new(0.0, 0.0);
    for d in nn_vectors() {
        let phase = k[0] * d[0] + k[1] * d[1];
        f += Complex64::new(phase.cos(), phase.sin());
    }
    -T * f
}

/// 2x2 Hermitian 哈密顿量 H(k)
fn hamiltonian(k: [f64; 2]) -> Matrix2<Complex64> {
    let f = fk(k);
    Matrix2::new(
        Complex64::new(0.0, 0.0),
        f,
        f.conj(),
        Complex64::new(0.0, 0.0),
    )
}

/// Hermitian 对角化, 返回升序本征值 [E-, E+] (本征值为实数)
///
/// nalgebra SymmetricEigen 对本征值的排列约定未声明升序 (实测为降序),
/// 此处显式排序以保证 E- <= E+。
fn band_energies(k: [f64; 2]) -> [f64; 2] {
    let h = hamiltonian(k);
    let eig = SymmetricEigen::new(h);
    let mut ev = [eig.eigenvalues[0], eig.eigenvalues[1]];
    if ev[0] > ev[1] {
        ev.swap(0, 1);
    }
    ev
}

/// 态密度: 在布里渊区平行四边形 (b1, b2) 上均匀采样 nk×nk 个 k 点, 每个 k 点
/// 两条能带落入直方图并归一化。返回 (能量中心, 态密度, 正能带 van Hove 峰位)。
fn dos(nk: usize) -> (Vec<f64>, Vec<f64>, f64) {
    let nbins = 200;
    let emin = -3.5 * T;
    let emax = 3.5 * T;
    let de = (emax - emin) / nbins as f64;
    let mut hist = vec![0.0f64; nbins];

    // 倒格基矢 (键长 a=1, 晶格常数 a0=√3)
    let b1 = [2.0 * PI / 3.0, 2.0 * PI / 3.0_f64.sqrt()];
    let b2 = [2.0 * PI / 3.0, -2.0 * PI / 3.0_f64.sqrt()];

    for i in 0..nk {
        for j in 0..nk {
            let u = i as f64 / nk as f64;
            let v = j as f64 / nk as f64;
            let k = [u * b1[0] + v * b2[0], u * b1[1] + v * b2[1]];
            for e in band_energies(k) {
                let bin = ((e - emin) / de) as usize;
                if bin < nbins {
                    hist[bin] += 1.0;
                }
            }
        }
    }

    // 归一化使 ∫ρ dE = 2 (两条能带): 均匀采样 nk² 个 k 点、每点 2 条带,
    // 直方图计数除以 k 点数再除以能量 bin 宽, 而非除以总计数 (那会恒等于 1)。
    let nkpts = (nk * nk) as f64;
    let centers: Vec<f64> = (0..nbins).map(|b| emin + (b as f64 + 0.5) * de).collect();
    let density: Vec<f64> = hist.iter().map(|&h| h / (nkpts * de)).collect();

    // van Hove 峰: 正能带 (E>0) 态密度极大值位置
    let mut peak_e = 0.0;
    let mut peak_val = -1.0;
    for (c, &d) in centers.iter().zip(density.iter()) {
        if *c > 0.0 && d > peak_val {
            peak_val = d;
            peak_e = *c;
        }
    }
    (centers, density, peak_e)
}

fn main() {
    println!("=== graphene tight-binding band structure + DOS (nalgebra, num-complex) ===");
    println!("[graphene] hopping t = {T}, bond length a = 1 (normalized units)");

    // 高对称点 (倒空间坐标)
    let gamma = [0.0, 0.0];
    let kdirac = [4.0 * PI / (3.0 * 3.0_f64.sqrt()), 0.0]; // K = (4π/(3√3), 0)
    let mpt = [PI / 3.0_f64.sqrt(), -PI / 3.0]; // M = (π/√3, -π/3)

    println!("\n[graphene] diagonalized 2x2 Hermitian H(k) at high-symmetry points:");
    let e_gamma = band_energies(gamma);
    let e_k = band_energies(kdirac);
    let e_m = band_energies(mpt);
    println!(
        "[graphene]   Gamma (0,0)   E- = {:.6}, E+ = {:.6} (expect -3/+3)",
        e_gamma[0], e_gamma[1]
    );
    println!(
        "[graphene]   K (Dirac)     E- = {:.2e}, E+ = {:.2e} (expect 0)",
        e_k[0], e_k[1]
    );
    println!(
        "[graphene]   M (saddle)    E- = {:.6}, E+ = {:.6} (expect -1/+1)",
        e_m[0], e_m[1]
    );

    // 自检 1: Dirac 点无带隙
    println!(
        "[{}] E(K) = 0 (Dirac point, semimetal)",
        if e_k[1].abs() < 1e-12 { "PASS" } else { "FAIL" }
    );

    // 自检 2: 费米速度 vF = 3ta/2 = 1.5 (沿 x/y 两方向各取小 q 数值求导)
    let theory_vf = 1.5 * T;
    let mut pass_vf = true;
    for eps in [1e-5_f64, 1e-6] {
        let vx = band_energies([kdirac[0] + eps, 0.0])[1] / eps;
        let vy = band_energies([kdirac[0], eps])[1] / eps;
        println!(
            "[graphene]   eps={eps:.0e}: vF_x={vx:.6}, vF_y={vy:.6} (isotropic, theory {theory_vf})"
        );
        if (vx - theory_vf).abs() > 1e-3 || (vy - theory_vf).abs() > 1e-3 {
            pass_vf = false;
        }
    }
    println!(
        "[{}] Fermi velocity vF = 3ta/2 = {} (linear dispersion at K)",
        if pass_vf { "PASS" } else { "FAIL" },
        theory_vf
    );

    // 自检 3: 态密度归一化 + van Hove 峰位
    let (centers, density, peak_e) = dos(120);
    let de = centers[1] - centers[0];
    let integral: f64 = density.iter().map(|&d| d * de).sum();
    println!(
        "\n[graphene] DOS over BZ grid (nk=120, {} energy bins):",
        centers.len()
    );
    println!("[graphene]   integral rho(E) dE = {integral:.6} (expect 2, two bands)");
    println!("[graphene]   van Hove peak at E = {peak_e:.4} (expect ~1, M saddle point)");
    println!(
        "[{}] DOS normalization + van Hove peak near E=t",
        if (integral - 2.0).abs() < 0.05 && (peak_e - 1.0).abs() < 0.15 {
            "PASS"
        } else {
            "FAIL"
        }
    );

    println!("\n=== graphene: done ===");
}
