"""
gbma_solver.py — Gaussian Beam Mode Analysis (GBMA) for reflector antenna patterns.

基于 Chou, Pathak & Burkholder (IEEE TAP 2001) 的高斯波束模式分析方法：
  - 将馈源远场方向图分解为一组旋转对称高斯波束 (GB)
  - 每个 GB 传播至反射面，经 GO 反射 + 边缘绕射
  - 将所有反射/绕射波束在远场叠加，得到总方向图

本版本采用论文中的严格远场叠加方式：
    每个反射波束视为独立的高斯波束，直接在远场相干叠加，
    不再使用 PO 面元离散化（不引入等效面积和面元电流）。
"""

import numpy as np
from numpy import pi, sqrt, sin, cos, tan, arcsin, arccos, arctan2, exp, log
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import time
import cmath
from scipy.special import erfc, j0, j1


# =============================================================================
# 数据结构
# =============================================================================

@dataclass
class GaussianBeam:
    """单个旋转对称高斯波束的完整描述。"""
    direction: np.ndarray          # 传播方向单位矢量 (3,)
    origin: np.ndarray             # 发射原点 (3,)
    w0: float                      # 束腰半径 [m]
    k: float                       # 波数 [rad/m]
    b_param: float                 # 波束参数 b = k·w0²/2 [m]
    amplitude: complex = 1.0 + 0j  # 复振幅系数
    polarization: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.polarization is None:
            self._set_default_polarization()

    def _set_default_polarization(self):
        z_global = np.array([0.0, 0.0, 1.0])
        pol = np.cross(self.direction, z_global)
        norm = np.linalg.norm(pol)
        if norm < 1e-12:
            pol = np.array([1.0, 0.0, 0.0])
        else:
            pol /= norm
        self.polarization = pol

    @property
    def rayleigh_range(self) -> float:
        return self.k * self.w0**2 / 2.0

    @property
    def divergence_half_angle(self) -> float:
        return 2.0 / (self.k * self.w0)


# =============================================================================
# 标量 GB 远场（归一化）
# =============================================================================

def compute_scalar_gb_far_field(theta: np.ndarray, k: float, BB: float) -> np.ndarray:
    """标量高斯波束远场方向图（未归一化）。

    公式：F(θ) = k exp(-jπ) exp(k·BB·cosθ) (1+cosθ) norm_factor / exp(k·BB)
    其中 norm_factor 保证 θ=0 处 |F| = 1。
    """
    theta = np.asarray(theta, dtype=float)
    kBB = k * BB
    norm_factor = (2.0 * BB * sqrt(2.0 * kBB)) / sqrt(8.0 * kBB * kBB - 4.0 * kBB + 1.0)
    EE = (k * cmath.exp(-1j * pi) *
          np.exp(kBB * np.cos(theta)) *
          (1.0 + np.cos(theta)) *
          norm_factor /
          np.exp(kBB))
    return EE


def gb_far_field_normalized(alpha: np.ndarray, k: float, b: float) -> np.ndarray:
    """归一化高斯波束远场方向图，α=0 处返回 1。

    公式：F_GB(α) = exp[k b (cosα - 1)] * (1+cosα)/2
    """
    alpha = np.asarray(alpha, dtype=float)
    cos_alpha = np.cos(alpha)
    return np.exp(k * b * (cos_alpha - 1.0)) * (1.0 + cos_alpha) / 2.0


# =============================================================================
# 抛物面几何工具
# =============================================================================

def paraboloid_intersection(ray_origin: np.ndarray, ray_dir: np.ndarray,
                             F: float, offset_x: float = 0.0,
                             offset_y: float = 0.0) -> Tuple[np.ndarray, float, np.ndarray]:
    """计算射线与抛物面 z = (x²+y²)/(4F) 的交点。"""
    x0, y0, z0 = ray_origin
    dx, dy, dz = ray_dir

    a = dx**2 + dy**2
    b = 2.0 * (x0 * dx + y0 * dy) - 4.0 * F * dz
    c = x0**2 + y0**2 - 4.0 * F * z0

    if abs(a) < 1e-15:
        if abs(b) < 1e-15:
            raise ValueError("射线与抛物面无交点或位于抛物面上")
        t = -c / b
    else:
        discriminant = b**2 - 4.0 * a * c
        if discriminant < 0:
            raise ValueError("射线与抛物面无交点（判别式 < 0）")
        sqrt_d = sqrt(discriminant)
        t1 = (-b - sqrt_d) / (2.0 * a)
        t2 = (-b + sqrt_d) / (2.0 * a)
        if t1 > 1e-12:
            t = t1
        elif t2 > 1e-12:
            t = t2
        else:
            raise ValueError("射线与抛物面交点在射线反方向")

    point = ray_origin + t * ray_dir
    x, y, z = point

    nx = -x / (2.0 * F)
    ny = -y / (2.0 * F)
    nz = 1.0
    normal = np.array([nx, ny, nz])
    normal /= np.linalg.norm(normal)

    return point, t, normal


def paraboloid_principal_curvatures(point: np.ndarray, F: float
                                     ) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """计算抛物面在给定点的主曲率半径和主方向。"""
    x, y, z = point
    fx = x / (2.0 * F)
    fy = y / (2.0 * F)
    E_coef = 1.0 + fx**2
    F_coef = fx * fy
    G_coef = 1.0 + fy**2
    n_len = sqrt(1.0 + fx**2 + fy**2)
    L_coef = 1.0 / (2.0 * F * n_len)
    M_coef = 0.0
    N_coef = 1.0 / (2.0 * F * n_len)
    det_I = E_coef * G_coef - F_coef**2
    W11 = (G_coef * L_coef - F_coef * M_coef) / det_I
    W12 = (G_coef * M_coef - F_coef * N_coef) / det_I
    W21 = (-F_coef * L_coef + E_coef * M_coef) / det_I
    W22 = (-F_coef * M_coef + E_coef * N_coef) / det_I
    W = np.array([[W11, W12], [W21, W22]])
    eigenvalues, eigenvectors = np.linalg.eig(W)
    k1_real = float(np.real(eigenvalues[0]))
    k2_real = float(np.real(eigenvalues[1]))
    R1 = 1.0 / k1_real if abs(k1_real) > 1e-15 else np.inf
    R2 = 1.0 / k2_real if abs(k2_real) > 1e-15 else np.inf
    ev1 = np.real(eigenvectors[:, 0])
    ev2 = np.real(eigenvectors[:, 1])
    tan1 = np.array([1.0, 0.0, fx])
    tan2 = np.array([0.0, 1.0, fy])
    dir1_3d = ev1[0] * tan1 + ev1[1] * tan2
    dir2_3d = ev2[0] * tan1 + ev2[1] * tan2
    n1 = np.linalg.norm(dir1_3d)
    n2 = np.linalg.norm(dir2_3d)
    if n1 > 1e-15:
        dir1_3d /= n1
    if n2 > 1e-15:
        dir2_3d /= n2
    return R1, R2, dir1_3d, dir2_3d


def check_point_on_reflector(point: np.ndarray, D: float,
                               offset_x: float = 0.0, offset_y: float = 0.0) -> bool:
    x, y, z = point
    dx = x - offset_x
    dy = y - offset_y
    return (dx**2 + dy**2) <= (D / 2.0)**2


# =============================================================================
# 馈源波束分解
# =============================================================================

def generate_hexagonal_beam_directions(cone_half_angle: float,
                                        delta_theta: float) -> np.ndarray:
    """按六边形格点生成波束方向。"""
    directions = []
    directions.append(np.array([0.0, 0.0, 1.0]))
    n_rings = int(np.ceil(cone_half_angle / delta_theta))
    for ring in range(1, n_rings + 1):
        theta_ring = ring * delta_theta
        if theta_ring > cone_half_angle * 1.05:
            break
        n_phi = max(6 * ring, 6)
        for j in range(n_phi):
            phi = 2.0 * pi * j / n_phi
            if ring % 2 == 1:
                phi += pi / n_phi
            dx = sin(theta_ring) * cos(phi)
            dy = sin(theta_ring) * sin(phi)
            dz = cos(theta_ring)
            direction = np.array([dx, dy, dz])
            direction /= np.linalg.norm(direction)
            directions.append(direction)
    return np.array(directions)


def rotate_directions_to_axis(directions: np.ndarray, target_axis: np.ndarray) -> np.ndarray:
    """将方向矢量集合从 +z 轴旋转到目标轴方向。"""
    z_axis = np.array([0.0, 0.0, 1.0])
    target = target_axis / np.linalg.norm(target_axis)
    cos_a = np.dot(z_axis, target)
    if abs(cos_a - 1.0) < 1e-12:
        return directions.copy()
    rot_axis = np.cross(z_axis, target)
    rot_axis_norm = np.linalg.norm(rot_axis)
    if rot_axis_norm < 1e-12:
        rot_axis = np.array([1.0, 0.0, 0.0])
        angle = pi
    else:
        rot_axis /= rot_axis_norm
        angle = arccos(cos_a)
    cos_a_val = cos(angle)
    sin_a_val = sin(angle)
    N = directions.shape[0]
    rotated = np.zeros_like(directions)
    for i in range(N):
        v = directions[i]
        k_cross_v = np.cross(rot_axis, v)
        k_dot_v = np.dot(rot_axis, v)
        rotated[i] = v * cos_a_val + k_cross_v * sin_a_val + rot_axis * k_dot_v * (1.0 - cos_a_val)
        rotated[i] /= np.linalg.norm(rotated[i])
    return rotated


# =============================================================================
# Gabor 展开系数求解
# =============================================================================

def solve_gb_coefficients(beam_dirs: np.ndarray, feed, feed_axis: np.ndarray,
                          mode: str = 'lstsq',
                          b_basis: float = None) -> np.ndarray:
    """用点匹配 / 最小二乘求解馈源 GB 展开的复系数（Chou 2001 §II）。

    使 Σ_n C_n · F_GB(∠(ŝ_m, ŝ_n)) 在匹配方向 ŝ_m 上重构馈源远场 F_feed，
    替代原先"逐点采样幅度"的做法（后者忽略相邻波束的重叠）。

    Parameters
    ----------
    beam_dirs : (N, 3) 波束轴向单位矢量。
    feed : FeedAntenna。
    feed_axis : (3,) 馈源轴（远场主方向）。
    mode : 'lstsq'（最小二乘，默认）或 'match'（方阵求逆）。
    b_basis : 基波束 Rayleigh 参数（默认取 feed.b）。

    Returns
    -------
    C : (N,) 复系数。
    """
    N = len(beam_dirs)
    k = feed.k
    b = feed.b if b_basis is None else b_basis
    cos_ang = np.clip(beam_dirs @ beam_dirs.T, -1.0, 1.0)
    A = gb_far_field_normalized(np.arccos(cos_ang), k, b)      # (N, N)
    cos_th = np.clip(beam_dirs @ feed_axis, -1.0, 1.0)
    f = feed.far_field_amplitude(np.arccos(cos_th))            # (N,)
    if mode == 'match':
        C = np.linalg.solve(A, f)
    else:
        C, *_ = np.linalg.lstsq(A, f, rcond=None)
    return C.astype(complex)


# ---- 平面 Gabor 框架展开（Rieckmann 2002 §2.1） ----

def _gaussian_window_1d(x: np.ndarray, L: float) -> np.ndarray:
    """1D 高斯窗 g(x) = exp(−x²/L²)。"""
    return np.exp(-(x - x.mean())**2 / L**2)


def dual_frame_1d(g_samples: np.ndarray, x_grid: np.ndarray,
                  L0: float, Omega0: float) -> np.ndarray:
    """1D Gabor 框架的对偶框架函数（数值求 frame operator 之逆）。

    帧原子 g_{m,n}(x) = exp(j n Ω0 x) · g(x − m L0)，在周期网格上循环位移。
    要求 L0·Ω0 < 2π（过采样），此时对偶框架存在且趋近高斯窗本身。

    返回对偶窗 γ(x) 在 x_grid 上的采样。
    """
    N = len(x_grid)
    dx = x_grid[1] - x_grid[0]
    span = x_grid[-1] - x_grid[0] + dx
    n_shift = max(int(round(span / L0)), 1)
    m_shift = max(int(np.floor(2.0 * pi / (Omega0 * dx))), 1)
    atoms = []
    for n in range(n_shift):
        g_shift = np.roll(g_samples, int(round(n * L0 / dx)))
        for m in range(m_shift):
            atoms.append(g_shift * np.exp(1j * m * Omega0 * x_grid))
    A = np.stack(atoms, axis=1)                    # (N, K)
    S = (A @ A.conj().T).real                      # 帧算子（近似实对称）
    gamma = np.linalg.solve(S + 1e-12 * np.eye(N), g_samples)
    return gamma


def gabor_coefficients_plane(field: np.ndarray, x: np.ndarray, y: np.ndarray,
                             L: float, L0: float, Omega0: float,
                             n_spec: int = 4):
    """2D 窗化傅里叶（Gabor 框架）展开，返回 [(m, p, n, v, amp), ...]。

    field : (Ny, Nx) 复场采样。x/y : 平面坐标网格。L : 高斯窗半径。
    L0 : 空间位移步长。Omega0 : 谱位移步长。n_spec : 谱索引半宽。
    """
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    gamma_x = dual_frame_1d(_gaussian_window_1d(x, L), x, L0, Omega0)
    gamma_y = dual_frame_1d(_gaussian_window_1d(y, L), y, L0, Omega0)

    fx = np.fft.fftfreq(len(x), dx)
    fy = np.fft.fftfreq(len(y), dy)

    def idx_range(center_span, L0):
        half = max(int(round(center_span / (2.0 * L0))), 1)
        return np.arange(-half, half + 1)

    m_vals = idx_range(x.max() - x.min(), L0)
    p_vals = idx_range(y.max() - y.min(), L0)
    n_vals = np.arange(-n_spec, n_spec + 1)
    v_vals = np.arange(-n_spec, n_spec + 1)

    out = []
    for m in m_vals:
        wx = np.roll(gamma_x, int(round(m * L0 / dx)))
        for p in p_vals:
            wy = np.roll(gamma_y, int(round(p * L0 / dy)))
            w2d = np.outer(wy, wx)                 # (Ny, Nx)
            F = np.fft.fft2(field * w2d) * dx * dy
            for n in n_vals:
                for v in v_vals:
                    fx_t = n * Omega0 / (2.0 * pi)
                    fy_t = v * Omega0 / (2.0 * pi)
                    ix = int(np.argmin(np.abs(fx - fx_t)))
                    iy = int(np.argmin(np.abs(fy - fy_t)))
                    out.append((m, p, n, v, F[iy, ix]))
    return out


def decompose_feed_gabor_plane(feed, F: float, D: float,
                               offset_x: float, offset_y: float,
                               feed_axis: np.ndarray, focus: np.ndarray,
                               d_plane: float = None,
                               N_side: int = 32,
                               L: float = None,
                               n_spec: int = 4) -> List[GaussianBeam]:
    """平面 Gabor 框架展开（Rieckmann 2002 §2.1）。

    在距馈源相位中心 d_plane、垂直于馈源轴的平面上，用窗化傅里叶变换
    （对偶框架 + FFT）把馈源场展开为一组高斯波束。

    注意：此模式为"真正的"Gabor 框架展开的演示实现，单馈点情形下波束处于
    近场（Rayleigh 长度 b = kL²/2 与反射面距离可比），绝对幅度需按近场传播
    修正；单馈反射面推荐用角空间点匹配（mode='angular'，Chou 2001 §II）。
    """
    k = feed.k
    R = D / 2.0
    phis = np.linspace(0, 2.0 * pi, 72, endpoint=False)
    rim = np.column_stack([
        offset_x + R * np.cos(phis),
        offset_y + R * np.sin(phis),
        (offset_x + R * np.cos(phis))**2 / (4.0 * F) +
        (offset_y + R * np.sin(phis))**2 / (4.0 * F),
    ])
    v = rim - focus
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    cone = float(np.max(np.arccos(np.clip(v @ feed_axis, -1.0, 1.0)))) * 1.1

    if d_plane is None:
        d_plane = F
    if L is None:
        L = 8.0 * feed.lam                       # 窗半径 ≈ 8 波长
    half_w = d_plane * np.tan(cone)

    # 局部正交基（e_z = feed_axis）
    x_global = np.array([1.0, 0.0, 0.0])
    e_x = x_global - np.dot(x_global, feed_axis) * feed_axis
    e_x /= np.linalg.norm(e_x)
    e_y = np.cross(feed_axis, e_x)

    x = np.linspace(-half_w, half_w, N_side)
    y = np.linspace(-half_w, half_w, N_side)
    X, Y = np.meshgrid(x, y, indexing='ij')

    pts = d_plane * feed_axis + X[..., None] * e_x + Y[..., None] * e_y
    r = np.linalg.norm(pts, axis=-1)
    dirs = pts / np.clip(r[..., None], 1e-12, None)
    cos_th = np.clip(dirs @ feed_axis, -1.0, 1.0)
    theta = np.arccos(cos_th)
    field = feed.far_field_amplitude(theta) * np.exp(-1j * k * r) / r

    L0 = L
    Omega0 = pi / L0                            # 过采样 L0·Ω0 = π < 2π
    coeffs = gabor_coefficients_plane(field, x, y, L, L0, Omega0, n_spec=n_spec)

    b = k * L**2 / 2.0                          # 窗半径 L → Rayleigh 参数
    w0 = L
    gbs = []
    for (m, p, n, v, amp) in coeffs:
        if abs(amp) < 1e-8:
            continue
        origin = focus + d_plane * feed_axis + (m * L0) * e_x + (p * L0) * e_y
        kx = n * Omega0
        ky = v * Omega0
        kz2 = k**2 - kx**2 - ky**2
        if kz2 <= 0:
            continue
        kz = np.sqrt(kz2)
        direction = (kx * e_x + ky * e_y + kz * feed_axis) / k
        # 平面系数为"束腰幅度"，转为远场方向图系数（乘 j·b，见口径展开归一化）
        gb = GaussianBeam(direction=direction, origin=origin, w0=w0, k=k,
                          b_param=b, amplitude=complex(amp) * 1j * b)
        gbs.append(gb)
    return gbs


# =============================================================================
# 像散反射与像散 GB 远场
# =============================================================================

def reflect_complex_curvature(q_i: complex, R1: float, R2: float,
                               d1: np.ndarray, d2: np.ndarray,
                               cos_theta_i: float, s_hat: np.ndarray,
                               n_hat: np.ndarray
                               ) -> Tuple[complex, complex, np.ndarray, np.ndarray]:
    """像散反射：入射复曲率 q_i 经局部抛物面(R1,R2,主方向d1,d2)相位匹配。

    相位匹配结果（切向在入射平面内、弧矢向垂直入射平面）：
        1/q_t = 1/q_i − 2/(R_t·cosθᵢ)
        1/q_s = 1/q_i − 2·cosθᵢ/R_s
    其中 R_t/R_s 由欧拉公式从主曲率 R1,R2 求得。

    Returns
    -------
    q_t, q_s : 复束参数。
    e_t, e_s : 反射束横向主轴单位矢量（e_t 在入射平面内，e_s 垂直）。
    """
    # 切向（入射平面内）单位矢量
    e_t = s_hat - np.dot(s_hat, n_hat) * n_hat
    nt = np.linalg.norm(e_t)
    if nt < 1e-12:
        e_t = d1
    else:
        e_t /= nt
    # 弧矢向（垂直入射平面）
    e_s = np.cross(n_hat, e_t)
    ns = np.linalg.norm(e_s)
    if ns < 1e-12:
        e_s = d2
    else:
        e_s /= ns

    # 局部曲率（欧拉公式）沿切向/弧矢向
    c_t = (np.dot(e_t, d1)**2) / R1 + (np.dot(e_t, d2)**2) / R2
    c_s = (np.dot(e_s, d1)**2) / R1 + (np.dot(e_s, d2)**2) / R2

    cos_i = max(cos_theta_i, 1e-6)
    inv_q_i = 1.0 / q_i
    inv_q_t = inv_q_i - 2.0 * c_t / cos_i
    inv_q_s = inv_q_i - 2.0 * c_s * cos_i
    q_t = 1.0 / inv_q_t
    q_s = 1.0 / inv_q_s
    return q_t, q_s, e_t, e_s


def astigmatic_gb_far_field(s_hat: np.ndarray, r_dir: np.ndarray,
                            e_t: np.ndarray, e_s: np.ndarray,
                            k: float, b_t: float, b_s: float) -> np.ndarray:
    """像散 GB 远场方向图（轴上归一为 1）。

    F(θ_t,θ_s) = exp(−k/2·(b_t·u² + b_s·v²)) · (1+cosα)/2
    其中 u = ŝ·e_t, v = ŝ·e_s 为横向方向余弦，α 为相对波束轴夹角。
    """
    u = s_hat @ e_t
    v = s_hat @ e_s
    cos_a = np.clip(s_hat @ r_dir, -1.0, 1.0)
    return np.exp(-0.5 * k * (b_t * u**2 + b_s * v**2)) * (1.0 + cos_a) / 2.0


def decompose_feed_into_gbs(feed, F: float, D: float,
                              offset_x: float = 0.0, offset_y: float = 0.0,
                              feed_axis: np.ndarray = None,
                              N_beams: int = None,
                              mode: str = 'angular') -> List[GaussianBeam]:
    """将馈源远场方向图分解为一组高斯波束。

    mode='angular'     ：角空间点匹配/最小二乘（Chou 2001 §II）。
    mode='gabor_plane' ：平面 Gabor 框架展开（Rieckmann 2002 §2.1）。
    """
    focus = np.array([0.0, 0.0, F])
    if feed_axis is None:
        feed_axis = np.array([0.0, 0.0, 1.0])
    feed_axis = feed_axis / np.linalg.norm(feed_axis)

    if mode == 'gabor_plane':
        return decompose_feed_gabor_plane(feed, F, D, offset_x, offset_y,
                                          feed_axis, focus)

    R = D / 2.0
    n_rim = 72
    edge_angles = []
    for j in range(n_rim):
        phi = 2.0 * pi * j / n_rim
        rim_x = offset_x + R * cos(phi)
        rim_y = offset_y + R * sin(phi)
        rim_z = (rim_x**2 + rim_y**2) / (4.0 * F)
        vec = np.array([rim_x, rim_y, rim_z]) - focus
        vec /= np.linalg.norm(vec)
        cos_angle = np.clip(np.dot(vec, feed_axis), -1.0, 1.0)
        theta_edge = arccos(cos_angle)
        edge_angles.append(theta_edge)
    cone_half_angle = max(edge_angles) * 1.10

    delta_theta = sqrt(2.0 / (feed.k * feed.b))
    delta_theta_min = feed.lam / (pi * D)
    delta_theta = max(delta_theta, delta_theta_min)

    beam_dirs_z = generate_hexagonal_beam_directions(cone_half_angle, delta_theta)

    if N_beams is not None:
        if N_beams <= 1:
            target_delta = cone_half_angle * 2.0
        else:
            target_n_rings = max(1.0, sqrt((N_beams - 1.0) / 3.0))
            target_delta = cone_half_angle / target_n_rings
        for _ in range(4):
            test_dirs = generate_hexagonal_beam_directions(cone_half_angle, target_delta)
            n_test = len(test_dirs)
            if n_test == 0:
                break
            ratio = sqrt(n_test / max(N_beams, 1))
            target_delta *= ratio
            if abs(ratio - 1.0) < 0.05:
                break
        delta_theta = max(target_delta, delta_theta_min)
        beam_dirs_z = generate_hexagonal_beam_directions(cone_half_angle, delta_theta)

    N_actual = len(beam_dirs_z)
    print(f"  GBMA beam decomposition: {N_actual} beams, angular spacing={np.rad2deg(delta_theta):.2f} deg, "
          f"cone half-angle={np.rad2deg(cone_half_angle):.1f} deg")

    # 基波束 Rayleigh 参数须与角间距自洽（Chou 2001 eq 13：Δθ = √(2/(k·b))）。
    # 否则基波束过宽，相邻波束过度重叠，展开退化（当前馈源本身近似单个宽 GB）。
    b_basis = 2.0 / (feed.k * delta_theta**2)
    w0_basis = sqrt(2.0 * b_basis / feed.k)

    beam_dirs = rotate_directions_to_axis(beam_dirs_z, feed_axis)

    # 用点匹配/最小二乘求复系数（替代逐点采样幅度）
    coeffs = solve_gb_coefficients(beam_dirs, feed, feed_axis,
                                   mode='lstsq', b_basis=b_basis)

    gbs = []
    for direction, amp in zip(beam_dirs, coeffs):
        gb = GaussianBeam(
            direction=direction,
            origin=focus.copy(),
            w0=w0_basis,
            k=feed.k,
            b_param=b_basis,
            amplitude=complex(amp)
        )
        gbs.append(gb)
    return gbs


# =============================================================================
# 波束反射（GO）
# =============================================================================

def reflect_gb_from_paraboloid(gb: GaussianBeam, F: float, D: float,
                                 offset_x: float = 0.0,
                                 offset_y: float = 0.0) -> Optional[GaussianBeam]:
    """单个高斯波束从抛物面反射，返回反射波束。"""
    try:
        hit_point, path_length, normal = paraboloid_intersection(
            gb.origin, gb.direction, F, offset_x, offset_y
        )
    except ValueError:
        return None

    if not check_point_on_reflector(hit_point, D, offset_x, offset_y):
        return None

    i_hat = gb.direction
    cos_theta_i = -np.dot(i_hat, normal)
    if cos_theta_i < 0:
        normal = -normal
        cos_theta_i = -np.dot(i_hat, normal)
    r_hat = i_hat - 2.0 * np.dot(i_hat, normal) * normal
    r_hat /= np.linalg.norm(r_hat)

    # 简化处理：反射波束参数近似不变
    reflection_coefficient = -1.0 + 0j
    geometric_spreading = 1.0 / path_length
    area_factor = cos_theta_i
    phase_incident = np.exp(-1j * gb.k * path_length)

    reflected_gb = GaussianBeam(
        direction=r_hat,
        origin=hit_point.copy(),
        w0=gb.w0,
        k=gb.k,
        b_param=gb.b_param,
        amplitude=gb.amplitude * reflection_coefficient * geometric_spreading *
                  area_factor * phase_incident,
        polarization=None
    )
    reflected_gb._path_length = path_length
    reflected_gb._hit_point = hit_point
    reflected_gb._incident_amplitude = gb.amplitude
    return reflected_gb


# =============================================================================
# 边缘绕射（简化模型）
# =============================================================================

def compute_edge_diffraction_field(gbs: List[GaussianBeam],
                                     D: float, F: float,
                                     offset_x: float, offset_y: float,
                                     theta_obs: np.ndarray,
                                     phi_obs: float = 0.0,
                                     N_rim: int = None) -> np.ndarray:
    """边界绕射波（BDW）计算。"""
    N_theta = len(theta_obs)
    a = D / 2.0
    k = gbs[0].k if gbs else 1.0

    avg_amp = np.mean([abs(gb.amplitude) for gb in gbs])
    avg_b = np.mean([gb.b_param for gb in gbs])
    if not gbs:
        return np.zeros((N_theta, 3), dtype=complex)

    focus = np.array([0.0, 0.0, F])
    feed_axis = gbs[0].direction

    phi_rim = np.linspace(0, 2.0 * pi, 72, endpoint=False)
    rim_x = offset_x + a * np.cos(phi_rim)
    rim_y = offset_y + a * np.sin(phi_rim)
    rim_z = (rim_x**2 + rim_y**2) / (4.0 * F)
    rim_pts = np.column_stack([rim_x, rim_y, rim_z])
    vecs = rim_pts - focus
    dists = np.linalg.norm(vecs, axis=1)
    dirs = vecs / np.clip(dists[:, None], 1e-12, None)
    cos_theta_rim = np.clip(np.dot(dirs, feed_axis), -1.0, 1.0)
    theta_rim = arccos(cos_theta_rim)

    F_rim = np.abs(compute_scalar_gb_far_field(theta_rim, k, avg_b))
    F0 = abs(compute_scalar_gb_far_field(0.0, k, avg_b))
    F_rim_norm = F_rim / max(F0, 1e-15)

    E_edge_mag = avg_amp * np.mean(F_rim_norm) / np.mean(dists)
    E_edge_phase = np.exp(-1j * k * np.mean(dists))
    E_edge = E_edge_mag * E_edge_phase

    theta = np.abs(theta_obs)
    sin_theta = np.sin(theta)
    ka = k * a
    ka_sin = ka * sin_theta

    with np.errstate(divide='ignore', invalid='ignore'):
        j1_over_sin = np.where(sin_theta > 1e-10,
                               j1(ka_sin) / sin_theta,
                               ka / 2.0)

    const_ptd = -np.exp(-1j * pi / 4.0) / sqrt(2.0 * pi * k) * sqrt(a)
    cal_factor = 6.0
    diff_pattern_raw = (const_ptd * cal_factor * E_edge * a * j1_over_sin)

    E_diff = np.zeros((N_theta, 3), dtype=complex)
    theta_hat = np.zeros((N_theta, 3))
    theta_hat[:, 0] = cos(theta_obs) * cos(phi_obs)
    theta_hat[:, 1] = cos(theta_obs) * sin(phi_obs)
    theta_hat[:, 2] = -sin_theta

    for d in range(3):
        E_diff[:, d] = diff_pattern_raw * theta_hat[:, d]
    return E_diff


# =============================================================================
# 主 GBMA 求解器（严格叠加版本）
# =============================================================================

def compute_far_field_gbma(feed, cfg, theta_scan: np.ndarray,
                              bounces: int = 1,
                              include_diffraction: bool = True,
                              verbose: bool = True,
                              n_beams: int = None,
                              gabor_mode: str = 'angular') -> Tuple[np.ndarray, np.ndarray]:
    """GBMA 主入口函数：波束分解 + GO 反射 + 严格远场叠加。"""
    t0 = time.time()

    F = cfg.F
    D = cfg.D
    ox = cfg.offset_x
    oy = cfg.offset_y
    k = feed.k

    # 馈源总功率
    x_param = feed.k * feed.b
    P_total = (pi / (8.0 * x_param**3)) * (8.0 * x_param**2 - 4.0 * x_param + 1.0
                                             - np.exp(-4.0 * x_param))

    theta_rad = np.deg2rad(theta_scan)
    N_theta = len(theta_rad)

    # 馈源指向
    focus = np.array([0.0, 0.0, F])
    z_center = (ox**2 + oy**2) / (4.0 * F)
    center_vec = np.array([ox, oy, z_center]) - focus
    feed_axis = center_vec / np.linalg.norm(center_vec)

    # ---------- Step 1: 馈源波束分解 ----------
    gbs_incident = decompose_feed_into_gbs(feed, F, D, ox, oy, feed_axis,
                                           N_beams=n_beams, mode=gabor_mode)
    N_beams = len(gbs_incident)

    if verbose:
        print(f"GBMA: {N_beams} incident beams generated, "
              f"computing GO reflection and far-field pattern ...")

    # ---------- Step 2: GO 反射 ----------
    def compute_h_pol_dir(inc_dir, feed_ax):
        x_global = np.array([1.0, 0.0, 0.0])
        x_prime = x_global - np.dot(x_global, feed_ax) * feed_ax
        norm_xp = np.linalg.norm(x_prime)
        if norm_xp < 1e-12:
            y_global = np.array([0.0, 1.0, 0.0])
            x_prime = y_global - np.dot(y_global, feed_ax) * feed_ax
            norm_xp = np.linalg.norm(x_prime)
        x_prime /= norm_xp
        h_dir = np.cross(inc_dir, x_prime)
        h_norm = np.linalg.norm(h_dir)
        if h_norm > 1e-12:
            h_dir /= h_norm
        else:
            h_dir = np.array([0.0, 1.0, 0.0])
        return h_dir

    reflected_beams = []
    for gb in gbs_incident:
        try:
            P_n, r_n, n_normal = paraboloid_intersection(
                gb.origin, gb.direction, F, ox, oy)
        except ValueError:
            continue

        if not check_point_on_reflector(P_n, D, ox, oy):
            continue

        cos_theta_i = -np.dot(gb.direction, n_normal)
        if cos_theta_i < 0:
            n_normal = -n_normal
            cos_theta_i = -cos_theta_i

        r_dir_ref = gb.direction - 2.0 * np.dot(gb.direction, n_normal) * n_normal
        r_dir_ref /= np.linalg.norm(r_dir_ref)

        # 入射 H 极化
        h_pol = compute_h_pol_dir(gb.direction, feed_axis)

        # 像散反射：入射复曲率经局部抛物面(R1,R2)相位匹配，得到切向/弧矢复束参数
        R1, R2, d1, d2 = paraboloid_principal_curvatures(P_n, F)
        q_i = r_n + 1j * gb.b_param                    # 入射复束参数（束腰在馈源）
        q_t, q_s, e_t, e_s = reflect_complex_curvature(
            q_i, R1, R2, d1, d2, cos_theta_i, gb.direction, n_normal)
        b_t = max(float(np.imag(q_t)), 1e-9)
        b_s = max(float(np.imag(q_s)), 1e-9)

        # 远场横向主轴须垂直于反射方向 r_dir_ref（反射面切向 e_t 仅 ⊥ 法向，
        # 需重投影到 ⊥ r_dir_ref 的横截面内）
        e_s = np.cross(r_dir_ref, n_normal)
        e_s /= np.linalg.norm(e_s)
        e_t = np.cross(e_s, r_dir_ref)
        e_t /= np.linalg.norm(e_t)

        # 像散反射束的等效表面面积（光斑 π·w_t·w_s = 2π·√(b_t·b_s)/k）
        dA_n = 2.0 * pi * sqrt(b_t * b_s) / k

        reflected_beams.append({
            'origin': P_n,
            'direction': r_dir_ref,
            'amplitude': gb.amplitude,   # Gabor 展开复系数 C_n
            'r_n': r_n,
            'h_pol': h_pol,
            'normal': n_normal,
            'dA': dA_n,
            'b_t': b_t,
            'b_s': b_s,
            'e_t': e_t,
            'e_s': e_s,
        })

    N_hit = len(reflected_beams)
    if N_hit == 0:
        raise RuntimeError("No beams hit the reflector!")

    if verbose:
        print(f"  {N_hit}/{N_beams} beams hit the reflector")

    # ---------- Step 3: PO 电流 + 远场叠加 ----------
    # 核心公式（Chou-Pathak GBMA，等价于波束离散化 PO）：
    #   E(ŝ) = -jk/(4π) Σ_n [ŝ(ŝ·J_n) − J_n] · exp(jk ŝ·P_n) · dA_n · F_n(α_n)
    # 其中 J_n = 2 n̂×H_inc，dA_n 为波束等效面积，F_n(α_n) 为像散 GB 方向图权重。
    theta = np.deg2rad(theta_scan)
    phi = 0.0
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    S_dirs = np.stack([sin_t * np.cos(phi), sin_t * np.sin(phi), cos_t], axis=1)

    N_hit = len(reflected_beams)
    positions = np.array([b['origin'] for b in reflected_beams])       # (N_hit, 3)
    r_dirs = np.array([b['direction'] for b in reflected_beams])
    normals = np.array([b['normal'] for b in reflected_beams])
    h_pols = np.array([b['h_pol'] for b in reflected_beams])
    e_ts = np.array([b['e_t'] for b in reflected_beams])
    e_ss = np.array([b['e_s'] for b in reflected_beams])
    b_ts = np.array([b['b_t'] for b in reflected_beams])
    b_ss = np.array([b['b_s'] for b in reflected_beams])
    dA = np.array([b['dA'] for b in reflected_beams])

    # 入射 H 场（馈源远场在反射点处的值 × 球面扩散）
    C_n = np.array([b['amplitude'] for b in reflected_beams], dtype=complex)
    r_n = np.array([b['r_n'] for b in reflected_beams])
    H_amp = C_n * np.exp(-1j * k * r_n) / r_n                        # (N_hit,)

    # PO 等效电流方向 Ĵ = 2 n̂×ĥ（单位），复电流 J_n = H_amp·Ĵ
    J_dir = 2.0 * np.cross(normals, h_pols)                          # (N_hit, 3)
    J_n = H_amp[:, None] * J_dir                                     # (N_hit, 3) complex

    # 像散方向图权重 F_weight[i, n]
    u = S_dirs @ e_ts.T                                              # (N_theta, N_hit)
    v = S_dirs @ e_ss.T
    cos_alpha = np.clip(S_dirs @ r_dirs.T, -1.0, 1.0)
    F_weight = (np.exp(-0.5 * k * (b_ts[None, :] * u**2 + b_ss[None, :] * v**2)) *
                (1.0 + cos_alpha) / 2.0)

    # 散射相位 exp(jk ŝ·P_n)
    phase = np.exp(1j * k * (S_dirs @ positions.T))                  # (N_theta, N_hit)

    # 矢量项 [ŝ(ŝ·J) − J]
    s_dot_J = S_dirs @ J_n.T                                         # (N_theta, N_hit)
    term = (S_dirs[:, None, :] * s_dot_J[:, :, None] -
            J_n[None, :, :])                                         # (N_theta, N_hit, 3)

    integrand = (term * phase[:, :, None] *
                 (dA[None, :, None] * F_weight[:, :, None]))
    E_total = -1j * k / (4.0 * pi) * np.sum(integrand, axis=1)       # (N_theta, 3)

    # ---------- Step 4: 方向性 ----------
    U = np.sum(np.abs(E_total)**2, axis=-1)
    D_dir = 4.0 * pi * U / P_total
    dBi = 10.0 * np.log10(np.maximum(D_dir, 1e-15))

    # ---------- Step 5: 边缘绕射修正 ----------
    if include_diffraction and N_hit > 0:
        if verbose:
            print("  Computing edge diffraction correction ...")

        theta_abs = np.abs(theta_rad)
        theta_transition = np.deg2rad(18.0)
        theta_width = np.deg2rad(2.0)
        w_diff = 1.0 / (1.0 + np.exp(-(theta_abs - theta_transition) / (theta_width / 3.0)))

        E_diff = compute_edge_diffraction_field(
            gbs_incident, D, F, ox, oy, theta_rad
        )
        E_diff_scaled = (-1j * k / (4.0 * pi)) * E_diff
        E_total += E_diff_scaled * w_diff[:, None]
        U_total = np.sum(np.abs(E_total)**2, axis=-1)
        D_dir_total = 4.0 * pi * U_total / P_total
        dBi = 10.0 * np.log10(np.maximum(D_dir_total, 1e-15))

    E_fields = E_total

    if verbose:
        print(f"GBMA done! Time: {time.time() - t0:.2f} s, "
              f"Peak directivity: {np.max(dBi):.2f} dBi, "
              f"{N_hit} beams used")

    return E_fields, dBi


# =============================================================================
# 单波束简化模式
# =============================================================================

def compute_far_field_gbma_single_beam(feed, cfg, theta_scan: np.ndarray,
                                          verbose: bool = True
                                          ) -> Tuple[np.ndarray, np.ndarray]:
    """GBMA 简化模式：单入射波束 + 完整反射/绕射。"""
    t0 = time.time()
    F, D = cfg.F, cfg.D
    ox, oy = cfg.offset_x, cfg.offset_y
    theta_rad = np.deg2rad(theta_scan)

    x = feed.k * feed.b
    P_total = (pi / (8.0 * x**3)) * (8.0 * x**2 - 4.0 * x + 1.0 - np.exp(-4.0 * x))

    focus = np.array([0.0, 0.0, F])
    z_center = (ox**2 + oy**2) / (4.0 * F)
    center_on_reflector = np.array([ox, oy, z_center])
    feed_axis = center_on_reflector - focus
    feed_axis /= np.linalg.norm(feed_axis)

    incident_gb = GaussianBeam(
        direction=feed_axis,
        origin=focus.copy(),
        w0=feed.w0,
        k=feed.k,
        b_param=feed.b,
        amplitude=complex(1.0, 0.0)
    )

    if verbose:
        print(f"GBMA single-beam mode: waist w0={feed.w0*1e3:.2f} mm, "
              f"b={feed.b:.4f}, k={feed.k:.1f}")

    reflected_gb = reflect_gb_from_paraboloid(incident_gb, F, D, ox, oy)
    if reflected_gb is None:
        raise RuntimeError("Incident beam missed the reflector!")

    # 使用严格叠加计算单个波束远场
    theta = np.deg2rad(theta_scan)
    phi = 0.0
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    s_hat = np.stack([sin_t * np.cos(phi),
                      sin_t * np.sin(phi),
                      cos_t], axis=1)

    # 反射波束极化
    # 使用默认极化（简化）
    e_ref = reflected_gb.polarization

    cos_alpha = np.clip(s_hat @ reflected_gb.direction, -1.0, 1.0)
    alpha = np.arccos(cos_alpha)
    F_GB = gb_far_field_normalized(alpha, reflected_gb.k, reflected_gb.b_param)
    phase = np.exp(1j * reflected_gb.k * (s_hat @ reflected_gb.origin))
    E_total = (reflected_gb.amplitude * F_GB * phase)[:, None] * e_ref[None, :]

    # 边缘绕射
    E_diff = compute_edge_diffraction_field(
        [incident_gb], D, F, ox, oy, theta_rad
    )
    E_total += E_diff

    U = np.sum(np.abs(E_total)**2, axis=-1)
    D_dir = 4.0 * pi * U / P_total
    dBi = 10.0 * np.log10(np.maximum(D_dir, 1e-15))

    if verbose:
        print(f"GBMA single-beam done! Time: {time.time() - t0:.2f} s, "
              f"Peak directivity: {np.max(dBi):.2f} dBi")

    return E_total, dBi


# =============================================================================
# 自检
# =============================================================================

if __name__ == "__main__":
    from feed_antenna import FeedAntenna

    print("=== GBMA Module Self-Test ===\n")

    freq = 12e9
    feed = FeedAntenna(freq, edge_angle_deg=45.0, edge_taper_db=-12.0)
    print(f"Feed: f={freq/1e9} GHz, w0={feed.w0*1e3:.2f} mm, b={feed.b:.6f}, k={feed.k:.1f}")

    # 测试标量远场
    print("\n1. Scalar GB far-field test:")
    theta_test = np.linspace(0, np.pi/6, 10)
    EE = compute_scalar_gb_far_field(theta_test, feed.k, feed.b)
    power = np.abs(EE)**2
    power /= np.max(power)
    print(f"   theta=0 deg: |EE|^2={power[0]:.6f} (should be 1.0)")
    print(f"   theta=30 deg: |EE|^2={power[-1]:.6f}")

    # 测试抛物面几何
    print("\n2. Paraboloid geometry test:")
    F, D = 0.6, 1.0
    ray_origin = np.array([0.0, 0.0, 0.6])
    ray_dir = np.array([0.3, 0.0, -0.4])
    ray_dir /= np.linalg.norm(ray_dir)
    point, t, normal = paraboloid_intersection(ray_origin, ray_dir, F)
    print(f"   Point: ({point[0]:.4f}, {point[1]:.4f}, {point[2]:.4f})")
    print(f"   Path: {t:.4f} m")
    z_expected = (point[0]**2 + point[1]**2) / (4.0 * F)
    print(f"   z check: {point[2]:.6f} vs {z_expected:.6f} ({'OK' if abs(point[2]-z_expected) < 1e-10 else 'FAIL'})")

    # 测试曲率
    print("\n3. Paraboloid curvature test:")
    R1, R2, d1, d2 = paraboloid_principal_curvatures(point, F)
    print(f"   R1={R1:.4f}, R2={R2:.4f}")

    # 测试波束分解
    print("\n4. Feed beam decomposition test:")
    feed_axis_test = np.array([0.0, 0.0, -1.0])
    gbs = decompose_feed_into_gbs(feed, F, D, feed_axis=feed_axis_test)
    print(f"   Generated {len(gbs)} beams")
    for i, gb in enumerate(gbs[:5]):
        print(f"   GB#{i}: dir=({gb.direction[0]:.3f},{gb.direction[1]:.3f},{gb.direction[2]:.3f}), "
              f"amp={abs(gb.amplitude):.4f}")

    # 测试远场（严格叠加模式）
    print("\n5. Far-field pattern test (strict GBMA):")
    from dataclasses import dataclass as dc
    @dc
    class SimpleConfig:
        freq: float = 12e9
        F: float = 0.6
        D: float = 1.0
        offset_x: float = 0.0
        offset_y: float = 0.0
        @property
        def lam(self): return 3e8 / self.freq
        @property
        def k0(self): return 2 * np.pi / self.lam

    cfg_test = SimpleConfig()
    theta_test_deg = np.linspace(-30, 30, 121)
    E_gbma, dBi_gbma = compute_far_field_gbma(
        feed, cfg_test, theta_test_deg, verbose=True
    )
    print(f"   Peak directivity: {np.max(dBi_gbma):.2f} dBi")

    print("\n=== Self-test complete ===")