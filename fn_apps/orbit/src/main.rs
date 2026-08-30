// 轨道动力学载荷 (orbit) — 两套演示, 均强调辛积分器的长期正确性
//
// 演示 1 (限制性三体, 惯性系): 地球-月球-测试粒子三体。惯性系中, 地球
// (m1=1-μ) 与月球 (m2=μ) 绕共同质心做圆轨道, 测试粒子 (m3=1e-5, 限制性
// 近似) 只受两主天体引力运动, 不扰动其圆轨道。归一化: 主天体间距 = 1,
// 总质量 = 1, 引力常数 G = 1, 角速度 = 1。质量参数 μ = M_moon/(M_earth+M_moon)
// = 0.012150668 (地月系统)。对比 RK4 (非辛) 的能量单调漂移 vs
// Verlet/Yoshida (辛) 的能量有界振荡。
//
// 为什么用惯性系而非 CR3BP 旋转系: 旋转系运动方程含科里奥利项 (速度相关力),
// 破坏哈密顿量可分离性 (H = T + V)。Störmer-Verlet/Yoshida 的辛性质仅对可分离
// 哈密顿量成立 (utsuroi 文档明确: 速度相关力使 Verlet 退化为普通二阶方法,
// 丧失能量有界保证)。惯性系中引力是纯位置函数, 哈密顿量可分离, 辛积分器
// 严格适用。
//
// 软化长度: 测试粒子掠经主天体时 r->0, 引力奇异。采用 Plummer 软化
// r_soft = sqrt(r^2+ε^2) — REBOUND / orbitr / CONCEPT 等开源 N-body 代码的
// 标准正则化方案, 物理含义是把点质量替换为尺度半径 ε 的 Plummer 球。
// ε = 0.01, 约为主天体间距的 1% (对应地月距离的 1%, 量级为月球半径的 2 倍).
// 椭圆轨道远拱点距月球仍有安全余量, 软化仅作为避免近距掠过奇点的数值保险。
//
// 守恒量: 惯性系牛顿总机械能 E = Σ ½ m v^2 - Σ G m_i m_j / r_soft
// (势能统计与动力学使用同一软化势能面, 故 E 才是演化系统的真守恒量).
//
// 演示 2 (J2 扁率拱线进动): 自转使行星呈扁球 (赤道隆起), 用 J2 系数描述。
// 赤道轨道卫星受内向 1/r⁴ 摄动力, 拱线 (近拱点经度) 顺行进动, 率
// dϖ/dN = 3π J2 (R/a)²/(1-e²)²。该摄动势是纯位置函数 (哈密顿量可分离),
// 辛积分器严格适用; 用 Yoshida4 长期积分, 逐周期定位近拱点, 与解析理论对照。
//
// 积分器: 开源 utsuroi crate (RK4 + Störmer-Verlet + Yoshida4);
// 运动方程与软化正则化均为教科书/开源标准形式 (自编).
//
// 飞地 ABI: main() 入口, 计算结果经 println! 输出, return 0 -> SUSPEND.
use nalgebra::SVector;
use utsuroi::{DynamicalSystem, Integrator, Rk4, State, StormerVerlet, Yoshida4};

/// 三体状态维度: 3 粒子 × 3 坐标 (位置); 速度为其一阶导
const DIM: usize = 9;
/// 质量参数 μ = M_moon/(M_earth+M_moon), 地球-月球系统标准值
const MU: f64 = 0.012150668;
/// 地球质量 (归一化总质量 1)
const M1: f64 = 1.0 - MU;
/// 月球质量
const M2: f64 = MU;
/// 测试粒子质量 (限制性近似: 远小于主天体, 不扰动其圆轨道)
const M3: f64 = 1e-5;
/// 软化长度 ε (Plummer 软化, 主天体有限尺度, 约主天体间距的 1%)
const EPS: f64 = 0.01;
/// 三粒子质量
const MASS: [f64; 3] = [M1, M2, M3];
/// 测试粒子绕地轨道半长轴 (归一化单位, 地月距的 50%)
const ORB_A: f64 = 0.5;
/// 测试粒子绕地轨道偏心率
const ORB_E: f64 = 0.4;

/// 演示 2 常量: 模型行星 (自转扁球) 与赤道轨道卫星
/// 卫星半长轴 3R 远在行星表面之外, 无需软化
const J2_GM: f64 = 1.0; // 行星 GM (归一化)
const J2_COEFF: f64 = 0.01; // 扁率系数 (类木行星量级, Jupiter J2≈0.0147)
const J2_R: f64 = 1.0; // 行星半径 (归一化)
const SAT_A: f64 = 3.0; // 卫星半长轴 a = 3R
const SAT_E: f64 = 0.2; // 卫星偏心率

/// 三体引力系统 (惯性系, Plummer 软化正则化)
struct ThreeBody;

impl DynamicalSystem for ThreeBody {
    type State = State<DIM, 2>;

    fn derivatives(&self, _t: f64, state: &Self::State) -> Self::State {
        let y = state.y();
        let v = state.dy();
        let mut a = SVector::<f64, DIM>::zeros();
        let eps2 = EPS * EPS;
        // 两两相互作用 O(3^2): 牛顿第三定律同时更新两端
        for i in 0..3 {
            let xi = 3 * i;
            for j in (i + 1)..3 {
                let xj = 3 * j;
                let dx = y[xi] - y[xj];
                let dy = y[xi + 1] - y[xj + 1];
                let dz = y[xi + 2] - y[xj + 2];
                let r2 = dx * dx + dy * dy + dz * dz;
                // Plummer 软化: 1/r_soft^3, r_soft = sqrt(r^2+ε^2)
                let r_soft2 = r2 + eps2;
                let inv_r3 = 1.0 / (r_soft2 * r_soft2.sqrt());
                let fi = MASS[j] * inv_r3; // 对 i 的加速度系数
                let fj = MASS[i] * inv_r3; // 对 j 的加速度系数
                a[xi] -= fi * dx;
                a[xi + 1] -= fi * dy;
                a[xi + 2] -= fi * dz;
                a[xj] += fj * dx;
                a[xj + 1] += fj * dy;
                a[xj + 2] += fj * dz;
            }
        }
        State::from_derivative(v.clone(), a)
    }
}

/// J2 扁率行星 + 卫星系统 (赤道轨道, 3D 势能面)
struct OblateSatellite;

impl DynamicalSystem for OblateSatellite {
    type State = State<3, 2>;

    fn derivatives(&self, _t: f64, state: &Self::State) -> Self::State {
        let y = state.y();
        let v = state.dy();
        let x = y[0];
        let yc = y[1];
        let z = y[2];
        let r2 = x * x + yc * yc + z * z;
        let r = r2.sqrt();
        let c = 1.5 * J2_GM * J2_COEFF * J2_R * J2_R;
        let r5 = r2 * r2 * r;
        let r7 = r2 * r2 * r2 * r;
        let mut a = SVector::<f64, 3>::zeros();
        // 中心引力: -GM/r³ · r
        let g = -J2_GM / (r2 * r);
        // J2 扁率摄动 (标准测地势 C20 展开的负梯度):
        //   a_J2 = (3/2)GM J2 R² · [x(5z²/r⁷ - 1/r⁵), y(5z²/r⁷ - 1/r⁵), z(5z²/r⁷ - 3/r⁵)]
        // 赤道平面 (z=0) 内为内向 1/r⁴ 力, 使拱线顺行进动
        a[0] = g * x + c * (5.0 * x * z * z / r7 - x / r5);
        a[1] = g * yc + c * (5.0 * yc * z * z / r7 - yc / r5);
        a[2] = g * z + c * (5.0 * z * z * z / r7 - 3.0 * z / r5);
        State::from_derivative(v.clone(), a)
    }
}

/// 三体系统总机械能 (动能 + Plummer 软化势能, 与动力学同一势能面)
fn total_energy(st: &State<DIM, 2>) -> f64 {
    let y = st.y();
    let v = st.dy();
    let mut kin = 0.0;
    for i in 0..3 {
        let vi = 3 * i;
        let v2 = v[vi].powi(2) + v[vi + 1].powi(2) + v[vi + 2].powi(2);
        kin += 0.5 * MASS[i] * v2;
    }
    let eps2 = EPS * EPS;
    let mut pot = 0.0;
    for i in 0..3 {
        let xi = 3 * i;
        for j in (i + 1)..3 {
            let xj = 3 * j;
            let dx = y[xi] - y[xj];
            let dy = y[xi + 1] - y[xj + 1];
            let dz = y[xi + 2] - y[xj + 2];
            pot += -MASS[i] * MASS[j] / (dx * dx + dy * dy + dz * dz + eps2).sqrt();
        }
    }
    kin + pot
}

/// 构建初始条件: 主天体圆轨道 + 测试粒子绕地椭圆轨道 (远拱点释放).
///
/// t=0 时旋转系与惯性系重合, 主天体位于 x 轴圆轨道位相 (角速度 1);
/// 测试粒子半长轴 a, 偏心率 e, 位于远拱点 (靠近月球侧).
fn init_elliptical(a: f64, e: f64) -> State<DIM, 2> {
    let r_apo = a * (1.0 + e); // 远拱点距地球距离
    let v_apo = (M1 * (2.0 / r_apo - 1.0 / a)).sqrt(); // 远拱点开普勒速度
    let mut y0 = SVector::<f64, DIM>::zeros();
    let mut v0 = SVector::<f64, DIM>::zeros();
    // 地球 m1: 位于 x=-μ, 圆轨道速度 (0, -μ)
    y0[0] = -MU;
    v0[1] = -MU;
    // 月球 m2: 位于 x=1-μ, 圆轨道速度 (0, 1-μ)
    y0[3] = 1.0 - MU;
    v0[4] = 1.0 - MU;
    // 测试粒子 m3: 绕地椭圆轨道远拱点 (地球右侧 x=-μ+r_apo), 速度沿 -y
    y0[6] = -MU + r_apo;
    v0[7] = -v_apo;
    State::new(y0, v0)
}

/// 用指定积分器推进, 采样总机械能, 返回 (末态能量, max|dE/E0|, 采样点数).
///
/// RK4 经 `Integrator` trait 调用; Verlet/Yoshida 为 inherent 方法,
/// 签名相同, 故用宏为三种积分器各生成一个专用函数.
macro_rules! bench {
    ($name:ident, $integrator:expr) => {
        fn $name(init: State<DIM, 2>, t_end: f64, dt: f64) -> (f64, f64, u64) {
            let e0 = total_energy(&init);
            let mut max_drift = 0.0f64;
            let mut count = 0u64;
            let final_state = $integrator.integrate(&ThreeBody, init, 0.0, t_end, dt, |_, st| {
                let e = total_energy(st);
                let drift = ((e - e0) / e0).abs();
                if drift > max_drift {
                    max_drift = drift;
                }
                count += 1;
            });
            (total_energy(&final_state), max_drift, count)
        }
    };
}

bench!(bench_rk4, Rk4);
bench!(bench_verlet, StormerVerlet);
bench!(bench_yoshida, Yoshida4);

/// J2 扁率拱线进动演示: 用辛积分器 (Yoshida4) 长期模拟, 逐轨道定位近拱点
/// (抛物线插值 r(t) 极小值求亚步长时刻), 对近拱点方位角序列做线性最小二乘
/// 拟合, 得平均进动率 (rad/周期), 与解析理论 dϖ/dN = 3πJ2(R/a)²/(1-e²)² 对照.
///
/// 直接几何定位比瞬时振动要素法鲁棒: 不依赖振动要素在摄动下的摆动 (在月球
/// 三体情形实测该摆动幅值可达 0.4 rad, 使测量不可靠; 此处改为中心势问题).
fn j2_precession(t_end: f64, dt: f64) {
    // 卫星初始条件: 赤道平面椭圆轨道, 远拱点释放, 逆时针运动 (进动角为正)
    let r_apo = SAT_A * (1.0 + SAT_E);
    let v_apo = (J2_GM * (2.0 / r_apo - 1.0 / SAT_A)).sqrt();
    let mut y0 = SVector::<f64, 3>::zeros();
    let mut v0 = SVector::<f64, 3>::zeros();
    y0[0] = r_apo;
    v0[1] = v_apo;
    let init = State::new(y0, v0);

    // 轨道周期与理论进动率 (赤道轨道近拱点经度, 每周期)
    let period = 2.0 * std::f64::consts::PI * SAT_A.powf(1.5) / J2_GM.sqrt();
    let theory =
        3.0 * std::f64::consts::PI * J2_COEFF * (J2_R / SAT_A).powi(2) / (1.0 - SAT_E * SAT_E).powi(2);

    // 滚动窗口保存最近 3 个积分步的 (t, r, 方位角), 供抛物线插值定位近拱点
    let mut w_t = [0.0f64; 3];
    let mut w_r = [0.0f64; 3];
    let mut w_a = [0.0f64; 3];
    let mut n_steps = 0usize;
    let mut unwrapped: Option<f64> = None;
    let mut ts: Vec<f64> = Vec::new();
    let mut thetas: Vec<f64> = Vec::new();
    let mut peri_min = f64::MAX;
    Yoshida4.integrate(&OblateSatellite, init, 0.0, t_end, dt, |t, st| {
        // 卫星位置与距行星距离
        let x = st.y()[0];
        let y = st.y()[1];
        let r = (x * x + y * y).sqrt();
        if r < peri_min {
            peri_min = r;
        }
        let ang = y.atan2(x);
        w_t[0] = w_t[1];
        w_t[1] = w_t[2];
        w_t[2] = t;
        w_r[0] = w_r[1];
        w_r[1] = w_r[2];
        w_r[2] = r;
        w_a[0] = w_a[1];
        w_a[1] = w_a[2];
        w_a[2] = ang;
        n_steps += 1;
        // 中部点是 r 局部最小值 -> 抛物线插值求精确近拱点时刻与方位角
        if n_steps >= 3 && w_r[1] < w_r[0] && w_r[1] < w_r[2] {
            // 三点抛物线 r = A(t-t1)² + B(t-t1) + C 求极小值时刻
            let a2 = w_r[0] - 2.0 * w_r[1] + w_r[2];
            let b2 = w_r[2] - w_r[0];
            let dt1 = w_t[1] - w_t[0];
            let t_min = if a2.abs() > 1e-30 {
                w_t[1] - 0.5 * b2 / a2 * dt1
            } else {
                w_t[1]
            };
            // 在 [t1, t2] 或 [t0, t1] 间线性插值方位角
            let ang_peri = if t_min >= w_t[1] {
                w_a[1] + (w_a[2] - w_a[1]) * (t_min - w_t[1]) / (w_t[2] - w_t[1])
            } else {
                w_a[0] + (w_a[1] - w_a[0]) * (t_min - w_t[0]) / (w_t[1] - w_t[0])
            };
            // 方位角解卷绕为连续序列
            let w = match unwrapped {
                Some(prev) => {
                    let mut d = ang_peri - prev;
                    while d > std::f64::consts::PI {
                        d -= 2.0 * std::f64::consts::PI;
                    }
                    while d < -std::f64::consts::PI {
                        d += 2.0 * std::f64::consts::PI;
                    }
                    prev + d
                }
                None => ang_peri,
            };
            unwrapped = Some(w);
            ts.push(t_min);
            thetas.push(w);
        }
    });

    println!();
    println!("[orbit] Part 2: apsidal precession from planetary oblateness (J2)");
    println!(
        "[orbit]   model planet: J2 = {:.3}, R = {:.1}, GM = {:.1}; satellite a = {:.1}R, e = {:.2}",
        J2_COEFF,
        J2_R,
        J2_GM,
        SAT_A / J2_R,
        SAT_E
    );
    println!(
        "[orbit]   min periapsis radius {peri_min:.4} (initial {:.4})",
        SAT_A * (1.0 - SAT_E)
    );
    let n = ts.len();
    if n >= 4 {
        // 线性最小二乘拟合近拱点角序列, 斜率即平均进动率 (rad/时间)
        let t0 = ts[0];
        let (mut sx, mut sy, mut sxx, mut sxy) = (0.0, 0.0, 0.0, 0.0);
        for i in 0..n {
            let x = ts[i] - t0;
            let y = thetas[i];
            sx += x;
            sy += y;
            sxx += x * x;
            sxy += x * y;
        }
        let denom = n as f64 * sxx - sx * sx;
        if denom.abs() > 1e-30 {
            let slope = (n as f64 * sxy - sx * sy) / denom;
            let rate = slope * period; // 每周期进动角
            let total = slope * t_end;
            println!(
                "[orbit]   {} periapsis passages, period {period:.2}, dt {dt}, steps {:.0}",
                n,
                t_end / dt
            );
            println!(
                "[orbit]   measured precession {rate:.6} rad/orbit, theory {theory:.6} rad/orbit"
            );
            println!(
                "[orbit]   total advance {total:.3} rad ≈ {:.1} deg (theory {:.3} rad)",
                total.to_degrees(),
                theory * n as f64
            );
            let ratio = if theory.abs() > 1e-30 {
                rate / theory
            } else {
                0.0
            };
            println!(
                "[orbit]   measured/theory = {ratio:.3}  (Yoshida4 long-term tracking)"
            );
        } else {
            println!("[orbit]   degenerate fit, cannot estimate precession");
        }
    } else {
        println!("[orbit]   not enough periapsis passages");
    }
}

fn main() {
    println!("=== orbit_payload: symplectic vs RK4, apsidal precession ===");
    println!("[orbit] Part 1: three-body energy conservation (inertial frame)");
    println!(
        "[orbit]   model: Earth-Moon-test particle, mu = {MU:.6}, m3 = {M3:.0e}, softening eps = {EPS}"
    );
    println!("[orbit]   test particle orbit: a = {ORB_A}, e = {ORB_E} (apoapsis release)");

    // 初始条件: 主天体圆轨道 + 测试粒子绕地椭圆轨道 (远拱点释放, 受月球摄动)
    let init = init_elliptical(ORB_A, ORB_E);
    let e0 = total_energy(&init);
    println!("[orbit]   initial total energy E0 = {e0:.6}");
    println!(
        "[orbit]   test particle at ({:.4}, {:.4}), v ({:.4}, {:.4})",
        init.y()[6],
        init.y()[7],
        init.dy()[6],
        init.dy()[7]
    );

    // 积分时长: 测试粒子轨道周期约 2π·a^{3/2} ≈ 2.2, dt=0.1 约 22 步/周期,
    // 覆盖 ~450 周期使 RK4 漂移超过辛积分器的有界振荡 (教科书长期行为)
    let t_end = 1000.0f64;
    let dt = 0.1f64;

    let (e_rk4, drift_rk4, n_rk4) = bench_rk4(init.clone(), t_end, dt);
    let (e_verlet, drift_verlet, n_verlet) = bench_verlet(init.clone(), t_end, dt);
    let (e_yosh, drift_yosh, n_yosh) = bench_yoshida(init.clone(), t_end, dt);

    println!();
    println!(
        "[orbit]   integration t = {t_end}, dt = {dt}, {} steps per integrator",
        t_end / dt
    );
    println!(
        "[orbit]   {:>10} {:>16} {:>18} {:>10}",
        "integrator", "final E", "max|dE/E0|", "samples"
    );
    println!(
        "[orbit]   {:>10} {:>16.6} {:>18.3e} {:>10}",
        "RK4", e_rk4, drift_rk4, n_rk4
    );
    println!(
        "[orbit]   {:>10} {:>16.6} {:>18.3e} {:>10}",
        "Verlet", e_verlet, drift_verlet, n_verlet
    );
    println!(
        "[orbit]   {:>10} {:>16.6} {:>18.3e} {:>10}",
        "Yoshida4", e_yosh, drift_yosh, n_yosh
    );

    // 结论判定: 辛积分器能量漂移应显著小于 RK4 (非辛方法)
    if drift_rk4 > 0.0 && drift_verlet > 0.0 {
        let ratio = drift_rk4 / drift_verlet;
        println!(
            "[orbit]   RK4/Verlet drift ratio = {ratio:.1}x (symplectic conserves energy over long time)"
        );
    }

    // J2 扁率拱线进动演示: 辛积分器 (Yoshida4) 长期模拟
    // 周期 32.65, ~400 个周期, dt=0.15 约 218 步/周期, 87000 步
    j2_precession(13060.0, 0.15);
    println!("=== orbit_payload: done ===");
}
