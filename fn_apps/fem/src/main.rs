//! fem_poisson — 有限元求解二维 Poisson 方程 (数值偏微分)
//!
//! 定解问题 (Dirichlet 边界, 单位方域 [0,1]^2):
//!     -∇²u = f,  u|_边界 = 0
//! 取精确解 u(x,y) = sin(πx) sin(πy), 则右端 f = 2π² sin(πx) sin(πy)。
//!
//! 离散: 均匀 N×N 网格, 每个方格沿对角线拆成 2 个线性 (P1) 三角形单元。
//! 单元刚度阵 K_e[i][j] = A (b_i b_j + c_i c_j), 一致质量阵 M_e[i][j] = A/12 (1+δ_ij)
//! (用于右端 Galerkin 投影 f_e = M_e f_node)。组装后对内部自由度做高斯消元求解。
//!
//! 自检: 线性元在 L2 范数下二阶收敛, 网格加密一倍误差降为 1/4。
//! 用 N=8 与 N=16 两套网格的离散 L2 误差之比验证 (预期 ratio ≈ 4)。

use std::f64::consts::PI;

/// 右端项 f(x,y) = 2π² sin(πx) sin(πy)
fn rhs(x: f64, y: f64) -> f64 {
    2.0 * PI * PI * (PI * x).sin() * (PI * y).sin()
}

/// 精确解 u(x,y) = sin(πx) sin(πy)
fn exact(x: f64, y: f64) -> f64 {
    (PI * x).sin() * (PI * y).sin()
}

/// 在 N×N 网格上求解, 返回内部节点上的数值解与对应坐标 (均为 (N-1)² 长度)
fn solve_poisson(n: usize) -> Vec<f64> {
    let ntot = n + 1; // 每边节点数
    let nn = ntot * ntot; // 总节点数
    // 内部节点 DOF 映射: dof_of[global] = Some(dof) (内部) / None (边界)
    let mut dof_of = vec![None; nn];
    let mut ndof = 0usize;
    for i in 1..n {
        for j in 1..n {
            dof_of[i * ntot + j] = Some(ndof);
            ndof += 1;
        }
    }

    let mut ka = vec![0.0f64; ndof * ndof]; // 刚度阵 (行主序)
    let mut fb = vec![0.0f64; ndof]; // 载荷向量

    // 节点坐标
    let coord = |idx: usize| -> (f64, f64) {
        let i = idx / ntot;
        let j = idx % ntot;
        (i as f64 / n as f64, j as f64 / n as f64)
    };

    // 组装单个三角形单元 (全局节点编号 a, b, c)
    let assemble = |a: usize, b: usize, c: usize, ka: &mut Vec<f64>, fb: &mut Vec<f64>| {
        let nodes = [a, b, c];
        let (mut xs, mut ys) = ([0.0f64; 3], [0.0f64; 3]);
        for (k, &nd) in nodes.iter().enumerate() {
            let (x, y) = coord(nd);
            xs[k] = x;
            ys[k] = y;
        }
        // 面积 (行列式的一半)
        let det = (xs[1] - xs[0]) * (ys[2] - ys[0]) - (xs[2] - xs[0]) * (ys[1] - ys[0]);
        let area = det.abs() / 2.0;
        // 形函数梯度系数
        let b = [
            (ys[1] - ys[2]) / (2.0 * area),
            (ys[2] - ys[0]) / (2.0 * area),
            (ys[0] - ys[1]) / (2.0 * area),
        ];
        let c = [
            (xs[2] - xs[1]) / (2.0 * area),
            (xs[0] - xs[2]) / (2.0 * area),
            (xs[1] - xs[0]) / (2.0 * area),
        ];
        // 右端节点值
        let fnode = [rhs(xs[0], ys[0]), rhs(xs[1], ys[1]), rhs(xs[2], ys[2])];
        for i in 0..3 {
            let di = match dof_of[nodes[i]] {
                Some(d) => d,
                None => continue,
            };
            // 载荷: f_e[di] += Σ_j M_e[i][j] f_node[j]
            let mut load = 0.0;
            for j in 0..3 {
                let mij = if i == j { area / 6.0 } else { area / 12.0 };
                load += mij * fnode[j];
            }
            fb[di] += load;
            // 刚度: 仅内部-内部自由度 (边界 u=0 贡献为零)
            for j in 0..3 {
                if let Some(dj) = dof_of[nodes[j]] {
                    ka[di * ndof + dj] += area * (b[i] * b[j] + c[i] * c[j]);
                }
            }
        }
    };

    // 遍历所有方格, 每个拆成两个三角形 (对角线: 左下 -> 右上)
    for i in 0..n {
        for j in 0..n {
            let n00 = i * ntot + j;
            let n10 = (i + 1) * ntot + j;
            let n11 = (i + 1) * ntot + (j + 1);
            let n01 = i * ntot + (j + 1);
            assemble(n00, n10, n11, &mut ka, &mut fb);
            assemble(n00, n11, n01, &mut ka, &mut fb);
        }
    }

    // 高斯消元 (部分主元) 求解 K u = f
    let mut a = ka;
    let mut b = fb;
    for col in 0..ndof {
        let mut pivot = col;
        for row in (col + 1)..ndof {
            if a[row * ndof + col].abs() > a[pivot * ndof + col].abs() {
                pivot = row;
            }
        }
        if pivot != col {
            for k in 0..ndof {
                a.swap(col * ndof + k, pivot * ndof + k);
            }
            b.swap(col, pivot);
        }
        for row in (col + 1)..ndof {
            let f = a[row * ndof + col] / a[col * ndof + col];
            for k in col..ndof {
                a[row * ndof + k] -= f * a[col * ndof + k];
            }
            b[row] -= f * b[col];
        }
    }
    let mut u = vec![0.0f64; ndof];
    for row in (0..ndof).rev() {
        let mut s = b[row];
        for k in (row + 1)..ndof {
            s -= a[row * ndof + k] * u[k];
        }
        u[row] = s / a[row * ndof + row];
    }
    u
}

/// 离散 L2 误差: sqrt( Σ (u_fem - u_exact)^2 / num_dof )
fn l2_error(n: usize, u: &[f64]) -> f64 {
    let mut err2 = 0.0;
    let mut cnt = 0usize;
    for i in 1..n {
        for j in 1..n {
            let dof = (i - 1) * (n - 1) + (j - 1);
            let x = i as f64 / n as f64;
            let y = j as f64 / n as f64;
            let d = u[dof] - exact(x, y);
            err2 += d * d;
            cnt += 1;
        }
    }
    (err2 / cnt as f64).sqrt()
}

fn main() {
    println!("=== FEM Poisson equation (P1 triangles, 2D) ===");
    println!("[fem] problem: -laplace(u) = 2*pi^2*sin(pi*x)*sin(pi*y), u=0 on boundary");
    println!("[fem] exact:   u = sin(pi*x)*sin(pi*y)");

    // 两套网格, 验证二阶收敛
    let n1 = 8;
    let n2 = 16;
    let u1 = solve_poisson(n1);
    let u2 = solve_poisson(n2);
    let e1 = l2_error(n1, &u1);
    let e2 = l2_error(n2, &u2);
    let ratio = e1 / e2;

    println!("\n[fem] mesh N={n1}: discrete L2 error = {e1:.3e}");
    println!("[fem] mesh N={n2}: discrete L2 error = {e2:.3e}");
    println!("[fem] error ratio (expected ~4 for 2nd-order P1): {ratio:.3}");
    println!(
        "[{}] second-order convergence (ratio in [3, 5])",
        if ratio >= 3.0 && ratio <= 5.0 { "PASS" } else { "FAIL" }
    );

    // 中心点数值解 vs 精确解 (u(0.5,0.5) = 1)
    let center = (n2 / 2 - 1) * (n2 - 1) + (n2 / 2 - 1);
    let exact_center = exact(0.5, 0.5);
    println!(
        "\n[fem] u(0.5,0.5) = {:.6} (exact {:.6}) at N={n2}",
        u2[center], exact_center
    );

    println!("\n=== fem_poisson: done ===");
}
