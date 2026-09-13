//! linalg — 稠密矩阵线性代数 (Cholesky 分解 + 回代求解)
//!
//! 用 nalgebra 对一个大稠密对称正定矩阵做 Cholesky 分解并回代求解,
//! 以残差校验数值正确性。矩阵阶数选得足够大, 使单个 f64 稠密矩阵的
//! 运行期堆占用 (n²×8 字节) 明显超过飞地内存池的 2 MiB 分区粒度,
//! 从而在 TEE 中触发多分区分配 / OOM 拒绝 / 释放路径。
//!
//! 数值设计:
//!   - 系数矩阵 A 为严格对角占优的对称正定矩阵 (对角 = n, 非对角按
//!     |i-j| 衰减), 保证 Cholesky 分解存在且条件数良好, 不会病态。
//!   - 已知精确解 x_true, 构造右端 b = A·x_true, 求解后比对残差。
//!   - 残差 = max|x - x_true|, 阈值 1e-9 内判 PASS。
//!
//! 飞地 ABI: main() 入口, 结果经 println! 输出, return 0 表示挂起。

use nalgebra::{DMatrix, DVector};

/// 矩阵维数
const N: usize = 2400;
/// 浮点判定阈值
const TOL: f64 = 1e-9;

/// 构造严格对角占优的对称正定矩阵: 对角为 n, 非对角按 |i-j| 衰减。
fn build_spd(n: usize) -> DMatrix<f64> {
    let mut a = DMatrix::<f64>::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            let v = if i == j {
                n as f64
            } else {
                1.0 / ((i as f64 - j as f64).abs() + 1.0)
            };
            a[(i, j)] = v;
        }
    }
    a
}

fn main() {
    let mib = (N * N * 8) as f64 / (1024.0 * 1024.0);

	println!("=== linalg: dense Cholesky solve (nalgebra) ===");
    println!("[linalg] n = {N}, single f64 dense matrix = {mib:.2} MiB (> 2 MiB)");
    let a = build_spd(N);
    // 已知精确解 x_true[i] = i+1, 构造右端 b = A·x_true。
    let x_true = DVector::from_fn(N, |i, _| (i + 1) as f64);
    let b = &a * &x_true;

    // Cholesky 分解原地消费 a, 随后回代求解 A·x = b。
    let chol = a.cholesky().expect("matrix must be symmetric positive definite");
    let x = chol.solve(&b);

    // 残差 = max|x - x_true|, 阈值内判 PASS。
    let resid = (&x - &x_true).amax();
    let verdict = if resid < TOL { "PASS" } else { "FAIL" };
    println!("[linalg] max|x - x_true| = {resid:.3e}  {verdict}");
    println!("=== linalg: done ===");
}
