//! graphene — 石墨烯紧束缚能带 (凝聚态电子结构)
//!
//! 物理模型: 石墨烯蜂窝晶格, 每个原胞两个碳原子 (A/B 子晶格), 每个原子一个
//! 2p_z 轨道, 仅考虑最近邻跳跃 t。布洛赫基下紧束缚哈密顿量为 2x2:
//!
//!     H(k) = [ 0      f(k)  ]
//!            [ f*(k)  0     ]
//!
//! 其中 f(k) = -t (e^{i k·δ1} + e^{i k·δ2} + e^{i k·δ3}), δ 为三个最近邻矢量。
//! 能带解析解 E±(k) = ±|f(k)|。为避免复数, 用 |f|^2 的实形式直接计算:
//!
//!     |f(k)|^2 = t^2 [ 3 + 2cos(k·(δ1-δ2)) + 2cos(k·(δ1-δ3)) + 2cos(k·(δ2-δ3)) ]
//!
//! 自检量 (均可解析/数值精确验证):
//!   1. Dirac 点: K = (4π/(3√3), 0) 处 |f(K)|=0, 能带无带隙 (半金属)。
//!   2. 费米速度: E(K+q) 线性色散, 斜率 vF = 3 t a / 2 = 1.5 (t=a=1, ℏ=1)。
//!   3. 高对称点能量: Γ 点 E=3t (带顶), M 点 E=t (鞍点)。
//!
//! 单位: 键长 a=1, 跳跃 t=1 (对应真实石墨烯 t≈2.8 eV)。

use std::f64::consts::PI;

/// 最近邻跳跃积分 t (归一化为 1)
const T: f64 = 1.0;
/// 键长 a (归一化为 1)
const A: f64 = 1.0;

/// 导带能量 E(k) = |f(k)|, 用 |f|^2 实形式 (避免复数)
fn eband(k: [f64; 2]) -> f64 {
    // 三个最近邻矢量之差 = 晶格矢量 (键长 a=1):
    //   δ1=(√3/2, 1/2), δ2=(-√3/2, 1/2), δ3=(0,-1)
    let v1 = [3.0_f64.sqrt() * A, 0.0]; // δ1 - δ2
    let v2 = [3.0_f64.sqrt() * A / 2.0, 1.5 * A]; // δ1 - δ3
    let v3 = [-3.0_f64.sqrt() * A / 2.0, 1.5 * A]; // δ2 - δ3
    let dot = |v: [f64; 2]| k[0] * v[0] + k[1] * v[1];
    let s = 3.0 + 2.0 * dot(v1).cos() + 2.0 * dot(v2).cos() + 2.0 * dot(v3).cos();
    // s 为平方模, 应 >= 0; 极小负值由浮点舍入产生, 截断到 0
    T * s.max(0.0).sqrt()
}

/// 二维线性插值 a + t(b - a)
fn lerp2(a: [f64; 2], b: [f64; 2], t: f64) -> [f64; 2] {
    [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]
}

/// 沿 Γ->K->M->Γ 路径采样能带, 供自检高对称点能量
fn main() {
    println!("=== graphene tight-binding band structure (nearest-neighbor) ===");
    println!("[graphene] hopping t = {T}, bond length a = {A} (normalized units)");

    // 高对称点 (倒空间坐标)
    let gamma = [0.0, 0.0];
    let kdirac = [4.0 * PI / (3.0 * 3.0_f64.sqrt()), 0.0]; // K = (4π/(3√3), 0)
    let mpt = [PI / 3.0_f64.sqrt(), -PI / 3.0]; // M = (π/√3, -π/3)

    println!();
    println!("[graphene] band energies at high-symmetry points (E = |f(k)|):");
    println!("[graphene]   Gamma (0,0)   E = {:.6}  (expect 3t = 3)", eband(gamma));
    println!("[graphene]   K (Dirac)     E = {:.2e}  (expect 0, gapless)", eband(kdirac));
    println!("[graphene]   M (saddle)    E = {:.6}  (expect t = 1)", eband(mpt));

    // 自检 1: Dirac 点无带隙
    let ek = eband(kdirac);
    println!();
    println!(
        "[{}] E(K) = 0 (Dirac point, semimetal)",
        if ek.abs() < 1e-12 { "PASS" } else { "FAIL" }
    );

    // 自检 2: 费米速度 vF = 3ta/2 = 1.5 (沿 x/y 两方向各取小 q 数值求导)
    // eps 下限受 E(K+q) 的 sqrt 相消舍入约束: 过小 (1e-7) 时 s ≈ (1.5 eps)^2 已接近
    // 机器精度相对误差, vF 数值导数失真, 故仅取 1e-5 / 1e-6。
    let theory_vf = 1.5 * T * A;
    let mut pass_vf = true;
    for eps in [1e-5_f64, 1e-6] {
        let vx = eband([kdirac[0] + eps, 0.0]) / eps;
        let vy = eband([kdirac[0], eps]) / eps;
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

    // 能带路径 Γ->K->M->Γ (弧长参数 s 从 0 到 3, 每段 8 个采样点)
    println!();
    println!("[graphene] band path Gamma -> K -> M -> Gamma (E in units of t):");
    let seg = 8;
    for i in 0..=3 * seg {
        let s = i as f64 / seg as f64; // s ∈ [0, 3]
        let k = if s <= 1.0 {
            lerp2(gamma, kdirac, s)
        } else if s <= 2.0 {
            lerp2(kdirac, mpt, s - 1.0)
        } else {
            lerp2(mpt, gamma, s - 2.0)
        };
        println!("[graphene]   s={s:.2}  E={:+.6}", eband(k));
    }

    println!();
    println!("=== graphene: done ===");
}
