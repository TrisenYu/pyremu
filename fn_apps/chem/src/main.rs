//! hf_scf — Hartree-Fock 自洽场 (量子化学)
//!
//! 体系: H2 分子, STO-3G 最小基组 (每个 H 原子 1 个 1s 收缩函数, 由 3 个高斯
//! 原语线性组合)。限制性闭壳层 Hartree-Fock (RHF), 2 个电子占据 1 个成键轨道。
//!
//! 积分: 全 s 型高斯原语, 重叠/动能/核吸引/双电子积分均有闭式解 (高斯乘积
//! 定理 + Boys 函数 F0)。基组采用 Pople-Hehre-Stewart 1969 的 STO-3G 标准值
//! (原语已归一化, Σ c_i c_j S_ij = 1)。
//!
//! 自检量 (参考值见 Szabo & Ostlund "Modern Quantum Chemistry"):
//!   1. H 原子 STO-3G 基态能量 = -0.4666 Ha (单电子, E = <φ|h|φ>)
//!   2. H2 分子 STO-3G HF 总能量 = -1.1167 Ha (键长 R = 1.4 Bohr)
//!   3. 平衡键长 R_e ≈ 1.35 Bohr, 结合能 ≈ 0.18 Ha
//!
//! 单位: 原子单位 (Hartree, Bohr), 库仑积分 1/R 为核-核排斥。

use std::f64::consts::PI;

/// 三维矢量
type V3 = [f64; 3];

/// STO-3G H 1s 收缩: 高斯原语指数
const H_ALPHA: [f64; 3] = [3.425250914, 0.6239137298, 0.1688554040];
/// STO-3G H 1s 收缩: 收缩系数 (原语已归一化)
const H_COEFF: [f64; 3] = [0.1543289673, 0.5353281423, 0.4446345422];

/// 高斯原语归一化常数 (2α/π)^{3/4}
fn gnorm(a: f64) -> f64 {
    (2.0 * a / PI).powf(0.75)
}

/// 误差函数 erf (Abramowitz & Stegun 7.1.26, 相对精度 ~1.5e-7)
fn erf(x: f64) -> f64 {
    let neg = x < 0.0;
    let x = x.abs();
    const P: f64 = 0.3275911;
    const A1: f64 = 0.254829592;
    const A2: f64 = -0.284496736;
    const A3: f64 = 1.421413741;
    const A4: f64 = -1.453152027;
    const A5: f64 = 1.061405429;
    let t = 1.0 / (1.0 + P * x);
    let y = 1.0 - ((((A5 * t + A4) * t + A3) * t + A2) * t + A1) * t * (-x * x).exp();
    if neg {
        -y
    } else {
        y
    }
}

/// Boys 函数 F0(x) = ∫_0^1 exp(-x t^2) dt = (√π/(2√x)) erf(√x)
fn boys_f0(x: f64) -> f64 {
    if x < 1e-7 {
        // 小 x 泰勒展开 (避免 0/0 奇异)
        return 1.0 - x / 3.0 + x * x / 10.0 - x * x * x / 42.0;
    }
    let sx = x.sqrt();
    (PI.sqrt() / (2.0 * sx)) * erf(sx)
}

/// 两点距离平方
fn dist2(a: V3, b: V3) -> f64 {
    (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)
}

/// 两个归一化 s 型高斯原语的重叠积分 S = <a|b>
fn overlap(a: f64, ca: V3, b: f64, cb: V3) -> f64 {
    let p = a + b;
    (2.0 * (a * b).sqrt() / p).powf(1.5) * (-a * b * dist2(ca, cb) / p).exp()
}

/// 动能积分 T = <a| -∇²/2 |b>
fn kinetic(a: f64, ca: V3, b: f64, cb: V3) -> f64 {
    let p = a + b;
    let r2 = dist2(ca, cb);
    let s = overlap(a, ca, b, cb);
    s * (a * b / p) * (3.0 - 2.0 * a * b * r2 / p)
}

/// 核吸引积分 V = <a| -1/|r-C| |b>, C 为核位置
fn nuclear(a: f64, ca: V3, b: f64, cb: V3, cc: V3) -> f64 {
    let p = a + b;
    let s = overlap(a, ca, b, cb);
    // 高斯乘积中心 P = (αA + βB)/(α+β)
    let pc = [
        (a * ca[0] + b * cb[0]) / p,
        (a * ca[1] + b * cb[1]) / p,
        (a * ca[2] + b * cb[2]) / p,
    ];
    -s * (2.0 / PI.sqrt()) * p.sqrt() * boys_f0(p * dist2(pc, cc))
}

/// 双电子积分 (ab|cd) = <ab| 1/r12 |cd> (全 s 型归一化原语)
fn eri(a: f64, ca: V3, b: f64, cb: V3, c: f64, cc: V3, d: f64, cd: V3) -> f64 {
    let p = a + b;
    let q = c + d;
    let r_ab2 = dist2(ca, cb);
    let r_cd2 = dist2(cc, cd);
    let pc = [
        (a * ca[0] + b * cb[0]) / p,
        (a * ca[1] + b * cb[1]) / p,
        (a * ca[2] + b * cb[2]) / p,
    ];
    let qc = [
        (c * cc[0] + d * cd[0]) / q,
        (c * cc[1] + d * cd[1]) / q,
        (c * cc[2] + d * cd[2]) / q,
    ];
    let kab = gnorm(a) * gnorm(b);
    let kcd = gnorm(c) * gnorm(d);
    let pref = 2.0 * PI.powf(2.5) / (p * q * (p + q).sqrt());
    let arg = p * q / (p + q) * dist2(pc, qc);
    kab * kcd * pref * (-a * b * r_ab2 / p).exp() * (-c * d * r_cd2 / q).exp() * boys_f0(arg)
}

/// 基函数: 3 个高斯原语同心的收缩 s 函数
struct Basis {
    center: V3,
    alpha: [f64; 3],
    coeff: [f64; 3],
}

impl Basis {
    /// H 原子 STO-3G 1s, 中心在 c
    fn hydrogen(c: V3) -> Self {
        Basis {
            center: c,
            alpha: H_ALPHA,
            coeff: H_COEFF,
        }
    }
}

/// 收缩重叠积分 S_μν = Σ_ij c_μi c_νj <μi|νj>
fn c_overlap(mu: &Basis, nu: &Basis) -> f64 {
    let mut s = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            s += mu.coeff[i] * nu.coeff[j] * overlap(mu.alpha[i], mu.center, nu.alpha[j], nu.center);
        }
    }
    s
}

/// 收缩动能积分
fn c_kinetic(mu: &Basis, nu: &Basis) -> f64 {
    let mut s = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            s += mu.coeff[i] * nu.coeff[j] * kinetic(mu.alpha[i], mu.center, nu.alpha[j], nu.center);
        }
    }
    s
}

/// 收缩核吸引积分 (核在 c)
fn c_nuclear(mu: &Basis, nu: &Basis, c: V3) -> f64 {
    let mut s = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            s += mu.coeff[i] * nu.coeff[j] * nuclear(mu.alpha[i], mu.center, nu.alpha[j], nu.center, c);
        }
    }
    s
}

/// 收缩双电子积分 (μν|λσ)
fn c_eri(mu: &Basis, nu: &Basis, la: &Basis, si: &Basis) -> f64 {
    let mut s = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                for l in 0..3 {
                    s += mu.coeff[i] * nu.coeff[j] * la.coeff[k] * si.coeff[l]
                        * eri(
                            mu.alpha[i], mu.center,
                            nu.alpha[j], nu.center,
                            la.alpha[k], la.center,
                            si.alpha[l], si.center,
                        );
                }
            }
        }
    }
    s
}

/// Fock 矩阵 F = h + Σ P_λσ [(μν|λσ) - ½(μλ|νσ)]
fn fock(
    h: &[[f64; 2]; 2],
    g: &[[[[f64; 2]; 2]; 2]; 2],
    p: &[[f64; 2]; 2],
) -> [[f64; 2]; 2] {
    let mut f = [[0.0; 2]; 2];
    for m in 0..2 {
        for n in 0..2 {
            let mut gsum = 0.0;
            for l in 0..2 {
                for s in 0..2 {
                    gsum += p[l][s] * (g[m][n][l][s] - 0.5 * g[m][l][n][s]);
                }
            }
            f[m][n] = h[m][n] + gsum;
        }
    }
    f
}

/// 广义本征问题 FC = SCE 的最低能级解: 对称正交化 X=S^{-1/2} -> 对角化 X^T F X
fn diag2(f: &[[f64; 2]; 2], x: &[[f64; 2]; 2]) -> (f64, [f64; 2]) {
    // F' = X^T F X
    let mut ft = [[0.0; 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            let mut acc = 0.0;
            for k in 0..2 {
                for l in 0..2 {
                    acc += x[k][i] * f[k][l] * x[l][j];
                }
            }
            ft[i][j] = acc;
        }
    }
    let (aa, bb, cc) = (ft[0][0], ft[0][1], ft[1][1]);
    let mid = (aa + cc) / 2.0;
    let disc = (((aa - cc) / 2.0).powi(2) + bb * bb).sqrt();
    let e0 = mid - disc; // 最低能级
    // 对应特征向量 (bb, e0 - aa), 归一化
    let mut v = [bb, e0 - aa];
    let n = (v[0] * v[0] + v[1] * v[1]).sqrt();
    if n < 1e-12 {
        v = [1.0, 0.0];
    } else {
        v[0] /= n;
        v[1] /= n;
    }
    // 回到 AO 基: C = X v
    let c = [x[0][0] * v[0] + x[0][1] * v[1], x[1][0] * v[0] + x[1][1] * v[1]];
    (e0, c)
}

/// H2 分子 RHF SCF, 键长 r (Bohr), 返回 (总能量, 占据轨道能)
fn run_scf(r: f64) -> (f64, f64) {
    let ca = [0.0, 0.0, 0.0];
    let cb = [0.0, 0.0, r];
    let basis = [Basis::hydrogen(ca), Basis::hydrogen(cb)];

    // 单电子积分 (2x2): 重叠 S, 动能 T, 核吸引 V_A/V_B
    let mut s = [[0.0; 2]; 2];
    let mut t = [[0.0; 2]; 2];
    let mut va = [[0.0; 2]; 2];
    let mut vb = [[0.0; 2]; 2];
    for m in 0..2 {
        for n in 0..2 {
            s[m][n] = c_overlap(&basis[m], &basis[n]);
            t[m][n] = c_kinetic(&basis[m], &basis[n]);
            va[m][n] = c_nuclear(&basis[m], &basis[n], ca);
            vb[m][n] = c_nuclear(&basis[m], &basis[n], cb);
        }
    }
    let mut h = [[0.0; 2]; 2];
    for m in 0..2 {
        for n in 0..2 {
            h[m][n] = t[m][n] + va[m][n] + vb[m][n];
        }
    }

    // 双电子积分张量 (μν|λσ)
    let mut g = [[[[0.0; 2]; 2]; 2]; 2];
    for m in 0..2 {
        for n in 0..2 {
            for l in 0..2 {
                for sg in 0..2 {
                    g[m][n][l][sg] = c_eri(&basis[m], &basis[n], &basis[l], &basis[sg]);
                }
            }
        }
    }

    // 对称正交化 X = S^{-1/2} (同核双原子 S 为对称 2x2, S00=S11)
    let s00 = s[0][0];
    let s01 = s[0][1];
    let isp = 1.0 / (s00 + s01).sqrt();
    let ism = 1.0 / (s00 - s01).sqrt();
    let x = [
        [0.5 * (isp + ism), 0.5 * (isp - ism)],
        [0.5 * (isp - ism), 0.5 * (isp + ism)],
    ];

    // 初始密度: 对角化 core Hamiltonian
    let (mut e0, c) = diag2(&h, &x);
    let mut p = [[0.0; 2]; 2];
    for m in 0..2 {
        for n in 0..2 {
            p[m][n] = 2.0 * c[m] * c[n];
        }
    }

    // SCF 迭代 (密度收敛判据)
    for _ in 0..50 {
        let f = fock(&h, &g, &p);
        let (e_new, c_new) = diag2(&f, &x);
        let mut p_new = [[0.0; 2]; 2];
        for m in 0..2 {
            for n in 0..2 {
                p_new[m][n] = 2.0 * c_new[m] * c_new[n];
            }
        }
        let mut diff = 0.0;
        for m in 0..2 {
            for n in 0..2 {
                let d = (p_new[m][n] - p[m][n]).abs();
                if d > diff {
                    diff = d;
                }
            }
        }
        p = p_new;
        e0 = e_new;
        if diff < 1e-12 {
            break;
        }
    }

    // 总能量 E = Σ P h + ½ Σ P P [(μν|λσ) - ½(μλ|νσ)] + 1/R
    let mut e = 0.0;
    for m in 0..2 {
        for n in 0..2 {
            e += p[m][n] * h[m][n];
        }
    }
    let mut ee = 0.0;
    for m in 0..2 {
        for n in 0..2 {
            for l in 0..2 {
                for sg in 0..2 {
                    ee += p[m][n] * p[l][sg] * (g[m][n][l][sg] - 0.5 * g[m][l][n][sg]);
                }
            }
        }
    }
    e += 0.5 * ee + 1.0 / r;
    (e, e0)
}

/// H 原子 STO-3G 基态能量 (单电子, 无 SCF)
fn h_atom_energy() -> f64 {
    let o = [0.0, 0.0, 0.0];
    let b = Basis::hydrogen(o);
    c_kinetic(&b, &b) + c_nuclear(&b, &b, o)
}

fn main() {
    println!("=== Hartree-Fock SCF (restricted, STO-3G) ===");

    // 自检 1: H 原子基态能量
    let eh = h_atom_energy();
    println!();
    println!("[hf_scf] H atom STO-3G energy = {:.6} Ha", eh);
    println!("[hf_scf] reference (analytic STO-3G) = -0.4666 Ha");
    println!(
        "[{}] H atom energy matches reference (|err| < 1e-4)",
        if (eh + 0.4666).abs() < 1e-4 { "PASS" } else { "FAIL" }
    );

    // 自检 2: H2 分子在 R=1.4 Bohr 的 HF 总能量
    let r_ref = 1.4;
    let (e_h2, e_orb) = run_scf(r_ref);
    println!();
    println!("[hf_scf] H2 STO-3G HF energy at R={:.1} Bohr = {:.6} Ha", r_ref, e_h2);
    println!("[hf_scf] reference (Szabo & Ostlund) = -1.1167 Ha");
    println!("[hf_scf] occupied orbital energy = {:.6} Ha", e_orb);
    println!(
        "[{}] H2 energy matches reference (|err| < 1e-4)",
        if (e_h2 + 1.1167).abs() < 1e-4 { "PASS" } else { "FAIL" }
    );

    // 自检 3: 势能面扫描, 定位平衡键长与结合能
    println!();
    println!("[hf_scf] potential energy scan (R in Bohr, E in Hartree):");
    let mut e_min = f64::INFINITY;
    let mut r_min = 0.0;
    for i in 10..=24 {
        let r = i as f64 / 10.0; // 1.0 .. 2.4
        let (e, _) = run_scf(r);
        if e < e_min {
            e_min = e;
            r_min = r;
        }
        println!("[hf_scf]   R={r:.1}  E={e:.6}");
    }
    // 离解极限: 2 个 H 原子 = -0.9332 Ha, 结合能 = E_min - (-0.9332)
    let e_dissoc = 2.0 * h_atom_energy();
    let binding = e_dissoc - e_min;
    println!(
        "[hf_scf] equilibrium: R_e = {r_min:.1} Bohr, E_min = {e_min:.6} Ha"
    );
    println!(
        "[hf_scf] dissociation limit = {e_dissoc:.6} Ha, binding energy = {binding:.4} Ha"
    );
    println!(
        "[{}] equilibrium bond length R_e ~ 1.35 Bohr, binding ~ 0.18 Ha",
        if (r_min - 1.35).abs() < 0.15 && (binding - 0.18).abs() < 0.05 {
            "PASS"
        } else {
            "FAIL"
        }
    );

    println!();
    println!("=== hf_scf: done ===");
}
