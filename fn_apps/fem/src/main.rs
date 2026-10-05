//! fem — 有限元求解二维变系数扩散反应方程 (数值偏微分)
//!
//! 定解问题 (Dirichlet 边界, 单位方域 [0,1]^2):
//!     -div(a(x,y) grad u) + c(x,y) u = f,   u = 0 于边界
//! 扩散系数 a 为复值, 含复系数微扰项; 反应系数 c 为实值且处处为正:
//!     a(x,y) = 1 + 0.4 sin(2πx) sin(2πy) + i (0.2 + 0.1 cos(2πx) cos(2πy))
//!     c(x,y) = 0.5 + 0.3 cos(πx) cos(πy)
//! 复系数使刚度阵成为复对称而非 Hermite 矩阵, Cholesky 不再适用 (见下方求解说明)。
//!
//! 制造解取三个谐波叠加 (多谐波, 避免单一特征函数与常数系数下的超收敛):
//!     u(x,y) = Σ_{k=1..3} (wr_k + i wi_k) sin(kπx) sin(kπy)
//! 各谐波在边界上均为零, 故满足齐次边界条件。右端项由符号求导给出:
//!     f = -div(a grad u) + c u = -(grad a · grad u + a ∇²u) + c u
//!
//! 离散: 均匀 N×N 网格, 每个方格沿对角线拆成 2 个线性 (P1) 三角形单元。
//! 单元积分用三角形三点 (边中点) 求积, 权重各取单元面积的三分之一; 该规则对
//! 二次多项式精确, 使变系数与反应项的积分误差为四次量级, 低于二次的离散误差,
//! 不至于污染收敛阶 (单点形心求积仅对一次多项式精确, 与二次的形函数乘积不匹配)。
//! 组装与求解用 nalgebra 的 DMatrix / DVector (标量为 num_complex 的复数类型)。
//!
//! 自检: 线性元在 L2 范数下二阶收敛, 网格加密一倍误差降为 1/4。
//! 用 N=8 与 N=16 两套网格的离散 L2 误差之比验证 (预期 ratio ≈ 4),
//! 复误差与其实部、虚部分别检验, 三者都须落进容许区间。

use nalgebra::{Complex, DMatrix, DVector};
use std::f64::consts::PI;

/// 谐波个数
const NHARM: usize = 3;
/// 制造解的谐波权重实部, 下标 k-1 对应 sin(kπx) sin(kπy)
const U_RE: [f64; NHARM] = [1.0, 0.6, 0.35];
/// 制造解的谐波权重虚部
const U_IM: [f64; NHARM] = [0.25, -0.4, 0.15];

/// 扩散系数 a(x,y), 实部与虚部都随空间变化
fn coeff_a(x: f64, y: f64) -> Complex<f64> {
    let s = (2.0 * PI * x).sin() * (2.0 * PI * y).sin();
    let c = (2.0 * PI * x).cos() * (2.0 * PI * y).cos();
    Complex::new(1.0 + 0.4 * s, 0.2 + 0.1 * c)
}

/// 扩散系数的梯度, 返回 (∂a/∂x, ∂a/∂y)
fn grad_a(x: f64, y: f64) -> (Complex<f64>, Complex<f64>) {
    let s2x = (2.0 * PI * x).sin();
    let c2x = (2.0 * PI * x).cos();
    let s2y = (2.0 * PI * y).sin();
    let c2y = (2.0 * PI * y).cos();
    let k = 2.0 * PI;
    let ax = Complex::new(0.4 * k * c2x * s2y, -0.1 * k * s2x * c2y);
    let ay = Complex::new(0.4 * k * s2x * c2y, -0.1 * k * c2x * s2y);
    (ax, ay)
}

/// 反应系数 c(x,y), 取值区间 [0.2, 0.8], 恒为正
fn coeff_c(x: f64, y: f64) -> f64 {
    0.5 + 0.3 * (PI * x).cos() * (PI * y).cos()
}

/// 制造解 u(x,y)
fn exact(x: f64, y: f64) -> Complex<f64> {
    let mut acc = Complex::new(0.0, 0.0);
    for k in 1..=NHARM {
        let kf = k as f64;
        let phi = (kf * PI * x).sin() * (kf * PI * y).sin();
        acc += Complex::new(U_RE[k - 1], U_IM[k - 1]) * phi;
    }
    acc
}

/// 制造解的梯度, 返回 (∂u/∂x, ∂u/∂y)
fn grad_exact(x: f64, y: f64) -> (Complex<f64>, Complex<f64>) {
    let mut gx = Complex::new(0.0, 0.0);
    let mut gy = Complex::new(0.0, 0.0);
    for k in 1..=NHARM {
        let kf = k as f64;
        let w = Complex::new(U_RE[k - 1], U_IM[k - 1]) * (kf * PI);
        gx += w * (kf * PI * x).cos() * (kf * PI * y).sin();
        gy += w * (kf * PI * x).sin() * (kf * PI * y).cos();
    }
    (gx, gy)
}

/// 制造解的拉普拉斯算子值, 每个谐波满足 ∇²sin(kπx)sin(kπy) = -2(kπ)² sin(kπx)sin(kπy)
fn laplace_exact(x: f64, y: f64) -> Complex<f64> {
    let mut acc = Complex::new(0.0, 0.0);
    for k in 1..=NHARM {
        let kf = k as f64;
        let phi = (kf * PI * x).sin() * (kf * PI * y).sin();
        acc += Complex::new(U_RE[k - 1], U_IM[k - 1]) * (kf * kf) * phi;
    }
    acc * (-2.0 * PI * PI)
}

/// 右端项 f(x,y) = -div(a grad u) + c u
fn rhs(x: f64, y: f64) -> Complex<f64> {
    let (gax, gay) = grad_a(x, y);
    let (gux, guy) = grad_exact(x, y);
    let a = coeff_a(x, y);
    let lap = laplace_exact(x, y);
    -(gax * gux + gay * guy + a * lap) + exact(x, y) * coeff_c(x, y)
}

/// 网格: 节点编号规则与内部自由度映射
struct Mesh {
    /// 每边单元数
    n: usize,
    /// 每边节点数
    ntot: usize,
    /// 全局节点编号到内部自由度编号的映射, 边界节点不设自由度
    dof_of: Vec<Option<usize>>,
    /// 内部自由度总数
    ndof: usize,
}

impl Mesh {
    /// 建立均匀 N×N 网格
    fn new(n: usize) -> Self {
        let ntot = n + 1;
        let mut dof_of = vec![None; ntot * ntot];
        let mut ndof = 0usize;
        for i in 1..n {
            for j in 1..n {
                dof_of[i * ntot + j] = Some(ndof);
                ndof += 1;
            }
        }
        Self {
            n,
            ntot,
            dof_of,
            ndof,
        }
    }

    /// 节点坐标
    fn coord(&self, idx: usize) -> (f64, f64) {
        let i = idx / self.ntot;
        let j = idx % self.ntot;
        (i as f64 / self.n as f64, j as f64 / self.n as f64)
    }
}

/// 组装一个 P1 三角形单元对刚度阵与载荷向量的贡献。
///
/// 单元积分为 ∫ (a grad φ_m · grad φ_l + c φ_m φ_l) 与 ∫ f φ_m, 用三点
/// (边中点) 求积近似: 每个求积点取单元面积的三分之一, 该规则对二次多项式精确。
/// 边界自由度已置零消去, 故只累加两端都是内部节点的项。
fn assemble_triangle(
    mesh: &Mesh,
    tri: [usize; 3],
    k: &mut DMatrix<Complex<f64>>,
    fv: &mut DVector<Complex<f64>>,
) {
    let mut xs = [0.0f64; 3];
    let mut ys = [0.0f64; 3];
    for (m, &nd) in tri.iter().enumerate() {
        let (x, y) = mesh.coord(nd);
        xs[m] = x;
        ys[m] = y;
    }
    // 面积取行列式绝对值的一半
    let det = (xs[1] - xs[0]) * (ys[2] - ys[0]) - (xs[2] - xs[0]) * (ys[1] - ys[0]);
    let area = det.abs() / 2.0;
    // 形函数梯度: ∂φ_m/∂x 与 ∂φ_m/∂y 在单元内为常数
    let bx = [
        (ys[1] - ys[2]) / (2.0 * area),
        (ys[2] - ys[0]) / (2.0 * area),
        (ys[0] - ys[1]) / (2.0 * area),
    ];
    let by = [
        (xs[2] - xs[1]) / (2.0 * area),
        (xs[0] - xs[2]) / (2.0 * area),
        (xs[1] - xs[0]) / (2.0 * area),
    ];
    let weight = area / 3.0;

    for q in 0..3 {
        let i = q;
        let j = (q + 1) % 3;
        let xm = 0.5 * (xs[i] + xs[j]);
        let ym = 0.5 * (ys[i] + ys[j]);
        // 形函数在边中点的取值: 该边两个端点为二分之一, 第三个顶点为零
        let mut phi = [0.0f64; 3];
        phi[i] = 0.5;
        phi[j] = 0.5;

        let aq = coeff_a(xm, ym);
        let cq = coeff_c(xm, ym);
        let fq = rhs(xm, ym);

        for m in 0..3 {
            let dm = match mesh.dof_of[tri[m]] {
                Some(d) => d,
                None => continue,
            };
            fv[dm] += weight * fq * phi[m];
            for l in 0..3 {
                if let Some(dl) = mesh.dof_of[tri[l]] {
                    let diff = aq * (bx[m] * bx[l] + by[m] * by[l]);
                    let reac = Complex::new(cq * phi[m] * phi[l], 0.0);
                    k[(dm, dl)] += weight * (diff + reac);
                }
            }
        }
    }
}

/// 在 N×N 网格上求解, 返回内部节点上的数值解 ((N-1)² 长度)
fn solve(n: usize) -> DVector<Complex<f64>> {
    let mesh = Mesh::new(n);
    let mut k = DMatrix::<Complex<f64>>::zeros(mesh.ndof, mesh.ndof);
    let mut fv = DVector::<Complex<f64>>::zeros(mesh.ndof);

    // 遍历所有方格, 每个拆成两个三角形 (对角线: 左下到右上)
    for i in 0..n {
        for j in 0..n {
            let n00 = i * mesh.ntot + j;
            let n10 = (i + 1) * mesh.ntot + j;
            let n11 = (i + 1) * mesh.ntot + (j + 1);
            let n01 = i * mesh.ntot + (j + 1);
            assemble_triangle(&mesh, [n00, n10, n11], &mut k, &mut fv);
            assemble_triangle(&mesh, [n00, n11, n01], &mut k, &mut fv);
        }
    }

    // 复系数使刚度阵为复对称而非 Hermite 矩阵, Cholesky 不适用, 改用 LU 分解
    k.lu().solve(&fv).expect("linear system must be nonsingular")
}

/// 离散 L2 误差, 返回 (复模, 实部, 虚部) 三个量: sqrt(Σ |u_fem - u_exact|^2 / ndof)
fn l2_errors(n: usize, u: &DVector<Complex<f64>>) -> (f64, f64, f64) {
    let (mut e2, mut er2, mut ei2) = (0.0f64, 0.0f64, 0.0f64);
    let mut cnt = 0usize;
    for i in 1..n {
        for j in 1..n {
            let dof = (i - 1) * (n - 1) + (j - 1);
            let x = i as f64 / n as f64;
            let y = j as f64 / n as f64;
            let d = u[dof] - exact(x, y);
            e2 += d.norm_sqr();
            er2 += d.re * d.re;
            ei2 += d.im * d.im;
            cnt += 1;
        }
    }
    let cnt = cnt as f64;
    ((e2 / cnt).sqrt(), (er2 / cnt).sqrt(), (ei2 / cnt).sqrt())
}

/// 复数格式化为 "实部 ± 虚部 i", 各保留六位小数
fn fmt_c(z: Complex<f64>) -> String {
    format!("{:.6}{:+.6}i", z.re, z.im)
}

/// 误差比是否处于二阶收敛的容许区间内
fn ratio_ok(ratio: f64) -> bool {
    (3.0..=5.0).contains(&ratio)
}

fn main() {
    println!("=== FEM variable-coefficient diffusion-reaction (P1 triangles, complex) ===");
    println!("[fem] problem: -div(a*grad(u)) + c*u = f, u=0 on boundary, domain [0,1]^2");
    println!("[fem] a(x,y) = 1 + 0.4 sin(2pi x) sin(2pi y) + i (0.2 + 0.1 cos(2pi x) cos(2pi y))");
    println!("[fem] c(x,y) = 0.5 + 0.3 cos(pi x) cos(pi y)");
    println!("[fem] exact:   u = sum_k (wr_k + i wi_k) sin(k pi x) sin(k pi y), k = 1..3");

    // 两套网格, 验证二阶收敛
    let n1 = 8;
    let n2 = 16;
    let u1 = solve(n1);
    let u2 = solve(n2);
    println!("\n[fem] mesh N={n1}: dofs = {}", (n1 - 1) * (n1 - 1));
    println!("[fem] mesh N={n2}: dofs = {}", (n2 - 1) * (n2 - 1));

    let (c1, r1, i1) = l2_errors(n1, &u1);
    let (c2, r2, i2) = l2_errors(n2, &u2);
    println!("[fem] N={n1}: L2 error = {c1:.3e} (re {r1:.3e}, im {i1:.3e})");
    println!("[fem] N={n2}: L2 error = {c2:.3e} (re {r2:.3e}, im {i2:.3e})");

    let rc = c1 / c2;
    let rr = r1 / r2;
    let ri = i1 / i2;
    println!("[fem] error ratio complex/re/im = {rc:.3} / {rr:.3} / {ri:.3}");
    println!("[fem] expected ~4 for 2nd-order P1");
    let ok = ratio_ok(rc) && ratio_ok(rr) && ratio_ok(ri);
    println!(
        "[{}] second-order convergence (all three ratios in [3, 5])",
        if ok { "PASS" } else { "FAIL" }
    );

    // 中心点数值解 vs 精确解 (u(0.5,0.5) = Σ (-1)^((k-1)/2) 型权重之和)
    let center = (n2 / 2 - 1) * (n2 - 1) + (n2 / 2 - 1);
    println!(
        "\n[fem] u(0.5,0.5) = {} (exact {}) at N={n2}",
        fmt_c(u2[center]),
        fmt_c(exact(0.5, 0.5))
    );

    println!("\n=== fem: done ===");
}
