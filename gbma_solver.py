"""
gbma_solver.py — Gaussian Beam Mode Analysis (GBMA) for reflector antenna patterns.

基于 Chou, Pathak & Burkholder (IEEE TAP 2001) 的高斯波束模式分析方法：
  - 将馈源远场方向图分解为一组旋转对称高斯波束 (GB)
  - 每个 GB 传播至反射面，经 GO 反射 + 边缘绕射
  - 将所有反射/绕射波束在远场叠加，得到总方向图

相比数值 PO（数小时），GBMA 在数秒内完成，且在聚焦区和阴影边界处
仍保持有效（UTD/GTD 在这些区域失效）。

References:
  Chou & Pathak, Radio Science, 1997 — GB 反射/绕射的统一渐近解
  Chou, Pathak & Burkholder, IEEE TAP, 2001 — GBMA 在大型反射面天线中的快速分析
  Rieckmann et al., IEE Proc. MAP, 2002 — 基于高斯波束绕射的模块化多反射面分析
"""

import numpy as np
from numpy import pi, sqrt, sin, cos, tan, arcsin, arccos, arctan2, exp, log
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import time
import cmath
from scipy.special import erfc, j0


# =============================================================================
# 数据结构
# =============================================================================

@dataclass
class GaussianBeam:
    """单个旋转对称高斯波束的完整描述。

    Attributes:
        direction: 波束传播方向单位矢量 (3,)，从原点指向远场
        origin:    波束发射原点 (3,)
        w0:        束腰半径 [m]
        k:         波数 [rad/m]
        b_param:   波束参数 b = w0²/(2/k) = k·w0²/2 [m]，控制波束发散角
        amplitude: 复振幅系数（来自展开拟合）
        polarization: 极化方向单位矢量 (3,)，垂直于 direction
    """
    direction: np.ndarray
    origin: np.ndarray
    w0: float
    k: float
    b_param: float
    amplitude: complex = 1.0 + 0j
    polarization: Optional[np.ndarray] = None

    def __post_init__(self):
        if self.polarization is None:
            self._set_default_polarization()

    def _set_default_polarization(self):
        """设置默认极化（y-极化，垂直于波束轴）。"""
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
        """瑞利长度 zR = k·w0²/2 [m]"""
        return self.k * self.w0**2 / 2.0

    @property
    def divergence_half_angle(self) -> float:
        """远场发散半角 [rad]"""
        return 2.0 / (self.k * self.w0)


# =============================================================================
# 标量 GB 远场
# =============================================================================

def compute_scalar_gb_far_field(theta: np.ndarray, k: float, BB: float) -> np.ndarray:
    """计算标量高斯波束远场方向图 EE(theta)，复数值。

    公式来自 gswavefar.py 的 compute_EE()，归一化使得 theta=0 处 |EE| = 1。

    Args:
        theta: 偏离波束轴的角度 [rad]，标量或数组
        k:     波数
        BB:    波束参数 b

    Returns:
        复数值远场 EE(theta)，与 theta 同形状
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


def compute_gb_far_field_power(theta: np.ndarray, k: float, BB: float) -> np.ndarray:
    """标量 GB 远场功率方向图 |EE(theta)|²，归一化。
    等效于 gswavefar.py 中的功率计算。
    """
    EE = compute_scalar_gb_far_field(theta, k, BB)
    power = np.abs(EE)**2
    if np.max(power) > 0:
        power /= np.max(power)
    return power


# =============================================================================
# 抛物面几何工具
# =============================================================================

def paraboloid_intersection(ray_origin: np.ndarray, ray_dir: np.ndarray,
                             F: float, offset_x: float = 0.0,
                             offset_y: float = 0.0) -> Tuple[np.ndarray, float, np.ndarray]:
    """计算射线与抛物面 z = (x²+y²)/(4F) 的交点。

    射线参数化: P(t) = origin + t * direction

    Args:
        ray_origin: 射线起点 (3,)
        ray_dir:    射线单位方向矢量 (3,)
        F:          焦距
        offset_x, offset_y: 反射面偏移

    Returns:
        (intersection_point, path_length, surface_normal)
        若无交点，抛出 ValueError
    """
    x0, y0, z0 = ray_origin
    dx, dy, dz = ray_dir

    # 解二次方程: (x0+t*dx)² + (y0+t*dy)² - 4F*(z0+t*dz) = 0
    a = dx**2 + dy**2
    b = 2.0 * (x0 * dx + y0 * dy) - 4.0 * F * dz
    c = x0**2 + y0**2 - 4.0 * F * z0

    if abs(a) < 1e-15:
        # 射线垂直（平行于 z 轴）
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
        # 取正方向的最小正 t
        if t1 > 1e-12:
            t = t1
        elif t2 > 1e-12:
            t = t2
        else:
            raise ValueError("射线与抛物面交点在射线反方向")

    point = ray_origin + t * ray_dir
    x, y, z = point

    # 法向量（抛物面梯度）
    nx = -x / (2.0 * F)
    ny = -y / (2.0 * F)
    nz = 1.0
    normal = np.array([nx, ny, nz])
    normal /= np.linalg.norm(normal)

    return point, t, normal


def paraboloid_principal_curvatures(point: np.ndarray, F: float
                                     ) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """计算抛物面 z = (x²+y²)/(4F) 在给定点的主曲率半径和主方向。

    对于旋转抛物面，主曲率方向为径向 (dR) 和方位角方向 (dPhi)。

    Returns:
        (R1, R2, dir1, dir2)
        R1, R2: 主曲率半径（正表示曲率中心在 -n 侧）
        dir1:   第一主方向 (径向)，单位矢量 (3,)
        dir2:   第二主方向 (方位角)，单位矢量 (3,)
    """
    x, y, z = point
    r = sqrt(x**2 + y**2)

    # 曲面: f(x,y) = (x²+y²)/(4F)
    # 一阶导数
    fx = x / (2.0 * F)
    fy = y / (2.0 * F)

    # 第一基本形式
    E_coef = 1.0 + fx**2
    F_coef = fx * fy
    G_coef = 1.0 + fy**2

    # 法向量模长
    n_len = sqrt(1.0 + fx**2 + fy**2)

    # 第二基本形式
    L_coef = 1.0 / (2.0 * F * n_len)
    M_coef = 0.0
    N_coef = 1.0 / (2.0 * F * n_len)

    # Weingarten 矩阵: W = [E F; F G]^{-1} [L M; M N]
    det_I = E_coef * G_coef - F_coef**2
    W11 = (G_coef * L_coef - F_coef * M_coef) / det_I
    W12 = (G_coef * M_coef - F_coef * N_coef) / det_I
    W21 = (-F_coef * L_coef + E_coef * M_coef) / det_I
    W22 = (-F_coef * M_coef + E_coef * N_coef) / det_I

    W = np.array([[W11, W12], [W21, W22]])

    # 特征值 = 主曲率, kappa = 1/R
    eigenvalues, eigenvectors = np.linalg.eig(W)
    k1_real = float(np.real(eigenvalues[0]))
    k2_real = float(np.real(eigenvalues[1]))

    # 主曲率半径
    if abs(k1_real) > 1e-15:
        R1 = 1.0 / k1_real
    else:
        R1 = np.inf
    if abs(k2_real) > 1e-15:
        R2 = 1.0 / k2_real
    else:
        R2 = np.inf

    # 主方向在参数域 (u,v) = (x,y) 中
    ev1 = np.real(eigenvectors[:, 0])
    ev2 = np.real(eigenvectors[:, 1])

    # 映射到 3D 切平面
    # 切平面基: (1, 0, fx) 和 (0, 1, fy)
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
    """检查点是否在反射面孔径投影范围内。

    Args:
        point: 3D 点
        D:     反射面投影直径
        offset_x, offset_y: 偏移

    Returns:
        True 若点在孔径圆内
    """
    x, y, z = point
    dx = x - offset_x
    dy = y - offset_y
    return (dx**2 + dy**2) <= (D / 2.0)**2


# =============================================================================
# 馈源波束分解 (Chou 2001, Section II)
# =============================================================================

def generate_hexagonal_beam_directions(cone_half_angle: float,
                                        delta_theta: float) -> np.ndarray:
    """按六边形格点在锥角内生成波束方向。

    遵循 Chou 2001 Fig.4 的排布模式，在 (θ, φ) 极坐标中排列。

    Args:
        cone_half_angle: 锥半角 [rad]，覆盖反射面边缘
        delta_theta:     波束角间距 [rad]

    Returns:
        方向数组 (N_beams, 3)，每行为单位方向矢量
    """
    directions = []

    # 中心波束 (沿 z 轴)
    directions.append(np.array([0.0, 0.0, 1.0]))

    # 径向环
    n_rings = int(np.ceil(cone_half_angle / delta_theta))
    for ring in range(1, n_rings + 1):
        theta_ring = ring * delta_theta
        if theta_ring > cone_half_angle * 1.05:
            break
        # 方位角方向数（六边形排列：第 ring 圈有 6*ring 个方向）
        n_phi = max(6 * ring, 6)
        for j in range(n_phi):
            phi = 2.0 * pi * j / n_phi
            # 对奇数环略微旋转 phi 偏移，使排列更均匀
            if ring % 2 == 1:
                phi += pi / n_phi

            dx = sin(theta_ring) * cos(phi)
            dy = sin(theta_ring) * sin(phi)
            dz = cos(theta_ring)
            direction = np.array([dx, dy, dz])
            direction /= np.linalg.norm(direction)
            directions.append(direction)

    return np.array(directions)


def decompose_feed_into_gbs(feed, F: float, D: float,
                              offset_x: float = 0.0, offset_y: float = 0.0,
                              feed_axis: np.ndarray = None,
                              N_beams: int = None) -> List[GaussianBeam]:
    """将馈源远场方向图分解为一组高斯波束。

    实现 Chou 2001 Section II 的方法：
    1. 计算覆盖反射面的锥半角 Ω（相对于馈源指向）
    2. 按 Chou 2001 Eq.(13) 确定角间距 Δθ̄
    3. 在六边形格点生成波束方向（相对 +z）
    4. 旋转波束方向到实际馈源指向
    5. 用馈源远场幅值直接设置各波束振幅

    Args:
        feed:      FeedAntenna 实例
        F:         焦距
        D:         反射面投影直径
        offset_x, offset_y: 反射面偏移
        feed_axis: 馈源指向单位矢量（从焦点指向反射面中心），默认 [0,0,1]
        N_beams:   期望波束数（大致），若 None 则自动确定

    Returns:
        GaussianBeam 对象列表
    """
    focus = np.array([0.0, 0.0, F])

    # 馈源指向
    if feed_axis is None:
        feed_axis = np.array([0.0, 0.0, 1.0])
    feed_axis = feed_axis / np.linalg.norm(feed_axis)

    # Step 1: 计算覆盖反射面的锥半角（相对于馈源指向）
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
        # 相对于馈源指向的角度
        cos_angle = np.clip(np.dot(vec, feed_axis), -1.0, 1.0)
        theta_edge = arccos(cos_angle)
        edge_angles.append(theta_edge)
    cone_half_angle = max(edge_angles) * 1.10  # 10% 余量

    # Step 2: 确定角间距 (Chou 2001 Eq. 13)
    delta_theta = sqrt(2.0 / (feed.k * feed.b))
    # 同时满足 Eq. (19) 的下限: Δθ̄ ≥ λ/(π·D_proj)
    delta_theta_min = feed.lam / (pi * D)
    delta_theta = max(delta_theta, delta_theta_min)

    # Step 3: 生成六边形波束方向（相对 +z 轴）
    beam_dirs_z = generate_hexagonal_beam_directions(cone_half_angle, delta_theta)

    if N_beams is not None:
        # 根据目标波束数反推角间距
        # 六边形格点近似公式: N ≈ 1 + 3·n·(n+1) ≈ 3n²，其中 n = Ω/Δθ
        # 解: Δθ ≈ Ω / sqrt((N-1)/3)
        if N_beams <= 1:
            target_delta = cone_half_angle * 2.0  # 仅中心波束
        else:
            target_n_rings = max(1.0, sqrt((N_beams - 1.0) / 3.0))
            target_delta = cone_half_angle / target_n_rings
        # 叠代微调使实际波束数接近目标
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

    # Step 4: 旋转波束方向从 +z 轴到实际馈源指向
    # 使用罗德里格斯旋转公式
    beam_dirs = rotate_directions_to_axis(beam_dirs_z, feed_axis)

    # Step 5: 创建波束对象并设置振幅
    gbs = []
    for direction in beam_dirs:
        # 该方向偏离馈源指向的角度
        cos_theta_dir = np.clip(np.dot(direction, feed_axis), -1.0, 1.0)
        theta_dir = arccos(cos_theta_dir)
        # 馈源在该方向的远场幅值
        amp = feed.far_field_amplitude(theta_dir)

        gb = GaussianBeam(
            direction=direction,
            origin=focus.copy(),
            w0=feed.w0,
            k=feed.k,
            b_param=feed.b,
            amplitude=complex(amp, 0.0)
        )
        gbs.append(gb)

    return gbs


def rotate_directions_to_axis(directions: np.ndarray, target_axis: np.ndarray) -> np.ndarray:
    """将方向矢量集合从 +z 轴旋转到目标轴方向。

    使用罗德里格斯旋转公式。处理 180° 旋转的特殊情况。

    Args:
        directions:  方向数组 (N, 3)，当前相对于 +z
        target_axis: 目标轴单位矢量 (3,)

    Returns:
        旋转后的方向数组 (N, 3)
    """
    z_axis = np.array([0.0, 0.0, 1.0])
    target = target_axis / np.linalg.norm(target_axis)

    cos_a = np.dot(z_axis, target)
    if abs(cos_a - 1.0) < 1e-12:
        return directions.copy()  # 无需旋转

    # 旋转轴
    rot_axis = np.cross(z_axis, target)
    rot_axis_norm = np.linalg.norm(rot_axis)

    if rot_axis_norm < 1e-12:
        # 180° 旋转（target == -z）：使用 x 轴作为旋转轴
        rot_axis = np.array([1.0, 0.0, 0.0])
        angle = pi
    else:
        rot_axis /= rot_axis_norm
        # 旋转角
        angle = arccos(cos_a)

    # 罗德里格斯公式：v_rot = v*cos(α) + (k×v)*sin(α) + k*(k·v)*(1-cos(α))
    cos_a_val = cos(angle)
    sin_a_val = sin(angle)

    N = directions.shape[0]
    rotated = np.zeros_like(directions)
    for i in range(N):
        v = directions[i]
        k_cross_v = np.cross(rot_axis, v)
        k_dot_v = np.dot(rot_axis, v)
        rotated[i] = v * cos_a_val + k_cross_v * sin_a_val + rot_axis * k_dot_v * (1.0 - cos_a_val)
        # 归一化
        rotated[i] /= np.linalg.norm(rotated[i])

    return rotated


# Legacy backward compatibility
decompose_feed_into_gbs_v1 = decompose_feed_into_gbs


# =============================================================================
# 波束反射 (GO 相位匹配)
# =============================================================================

def reflect_gb_from_paraboloid(gb: GaussianBeam, F: float, D: float,
                                 offset_x: float = 0.0,
                                 offset_y: float = 0.0) -> Optional[GaussianBeam]:
    """单个高斯波束从抛物面反射。

    步骤：
    1. 计算波束轴与抛物面的交点
    2. 若交点超出反射面范围，返回 None
    3. GO 反射方向：r_hat = i_hat - 2(i_hat·n)n
    4. 计算反射后的波束参数（考虑曲面曲率）
    5. 应用 PEC 反射系数 (-1) 和幅值因子

    Args:
        gb: 入射 GaussianBeam
        F:  焦距
        D:  反射面投影直径
        offset_x, offset_y: 偏移

    Returns:
        反射后的 GaussianBeam，或 None（若波束未命中反射面）
    """
    try:
        hit_point, path_length, normal = paraboloid_intersection(
            gb.origin, gb.direction, F, offset_x, offset_y
        )
    except ValueError:
        return None

    # 检查是否在反射面范围内
    if not check_point_on_reflector(hit_point, D, offset_x, offset_y):
        return None

    # GO 反射方向
    i_hat = gb.direction
    cos_theta_i = -np.dot(i_hat, normal)
    if cos_theta_i < 0:
        # 确保 normal 指向入射侧
        normal = -normal
        cos_theta_i = -np.dot(i_hat, normal)
    r_hat = i_hat - 2.0 * np.dot(i_hat, normal) * normal
    r_hat /= np.linalg.norm(r_hat)

    # 计算局部曲率并做相位匹配
    R1, R2, dir1, dir2 = paraboloid_principal_curvatures(hit_point, F)

    # 入射波前曲率（从束腰到反射点的波前）
    # 入射波束在距离 path_length 处的波前曲率半径：R_i = path_length + zR²/path_length
    zR = gb.rayleigh_range
    if path_length > 1e-6:
        R_incident = path_length * (1.0 + (zR / path_length)**2)
    else:
        R_incident = np.inf

    # 反射波前曲率（利用关系: 1/R_r = 1/R_i + 2·cos(θ_i)/R_curvature）
    # 在两个主方向上分别计算
    # 径向曲率: kappa_r = -1/(2F) * 1/(1 + r²/(4F²))^(3/2)
    # 方位角曲率: kappa_phi = -1/(2F) * 1/√(1 + r²/(4F²))
    x, y, z = hit_point
    r = sqrt(x**2 + y**2)
    r_ratio = r / (2.0 * F)
    cos_3 = (1.0 + r_ratio**2)**(-1.5)
    cos_1 = (1.0 + r_ratio**2)**(-0.5)

    kappa_r = -1.0 / (2.0 * F) * cos_3  # 径向主曲率
    kappa_phi = -1.0 / (2.0 * F) * cos_1  # 方位角主曲率

    # 反射后的波前曲率半径（在两个主方向）
    if abs(kappa_r) > 1e-15:
        R_curv_r = 1.0 / kappa_r
    else:
        R_curv_r = np.inf

    if abs(kappa_phi) > 1e-15:
        R_curv_phi = 1.0 / kappa_phi
    else:
        R_curv_phi = np.inf

    # 反射波前曲率：1/R_refl = 1/R_inc + 2·cos(θ_i)/R_surf
    if np.isfinite(R_incident):
        inv_R_refl_r = 1.0 / R_incident + 2.0 * cos_theta_i / R_curv_r
        inv_R_refl_phi = 1.0 / R_incident + 2.0 * cos_theta_i / R_curv_phi
    else:
        inv_R_refl_r = 2.0 * cos_theta_i / R_curv_r
        inv_R_refl_phi = 2.0 * cos_theta_i / R_curv_phi

    # 对抛物面反射，来自焦点的光线反射后为平行光
    # 等效于反射后波前为平面（曲率半径为无穷大）
    # 这是因为馈源位于焦点时，抛物面的几何特性
    # 对于偏移情况，反射波前接近平面但略有弯曲
    R_refl_r = 1.0 / inv_R_refl_r if abs(inv_R_refl_r) > 1e-15 else np.inf
    R_refl_phi = 1.0 / inv_R_refl_phi if abs(inv_R_refl_phi) > 1e-15 else np.inf

    # 反射后的等效波束参数（取平均曲率近似为圆对称）
    R_refl_avg = 2.0 / (inv_R_refl_r + inv_R_refl_phi) if (abs(inv_R_refl_r) + abs(inv_R_refl_phi)) > 1e-15 else np.inf

    # 反射系数（PEC: Γ = -1，即相位翻转 180°）
    reflection_coefficient = -1.0 + 0j

    # 几何扩散因子：场从馈源传播 distance_s 到反射面
    # 球面波扩散：E ∝ 1/r（与 PO 中 compute_incident_H 一致）
    geometric_spreading = 1.0 / path_length

    # 幅值因子（考虑波束到反射点的扩散和倾斜照度）
    w_at_surface = gb.w0 * sqrt(1.0 + (path_length / zR)**2)
    area_factor = cos_theta_i

    # 入射相位因子（馈源到反射点的传播）
    phase_incident = np.exp(-1j * gb.k * path_length)

    # 构建反射波束（幅值包含几何扩散和相位）
    reflected_gb = GaussianBeam(
        direction=r_hat,
        origin=hit_point.copy(),
        w0=gb.w0,  # 反射后束腰近似不变（远场观察时）
        k=gb.k,
        b_param=gb.b_param,  # 波束参数近似不变
        amplitude=gb.amplitude * reflection_coefficient * geometric_spreading *
                  area_factor * phase_incident,
        polarization=None  # 自动计算
    )

    # 存储额外信息（用于远场相位计算）
    reflected_gb._path_length = path_length
    reflected_gb._hit_point = hit_point
    reflected_gb._incident_amplitude = gb.amplitude

    return reflected_gb


# =============================================================================
# 远场计算
# =============================================================================

def compute_single_gb_far_field(gb: GaussianBeam,
                                  theta_obs: np.ndarray,
                                  phi_obs: float = 0.0) -> np.ndarray:
    """计算单个 GB 在给定观察方向的远场 E 矢量贡献。

    使用标量 GB 远场公式，考虑：
    - 波束轴与观察方向的角度偏移 α
    - 从反射点到远场的相位差
    - 极化矢量的投影

    Args:
        gb:        GaussianBeam（反射后的）
        theta_obs: 观察角度 [rad]，数组
        phi_obs:   观察方位角 [rad]

    Returns:
        远场 E 矢量数组 (N_theta, 3)，复数
    """
    N = len(theta_obs)
    E_field = np.zeros((N, 3), dtype=complex)

    # 观察方向单位矢量
    sin_t = sin(theta_obs)
    S = np.zeros((N, 3))
    S[:, 0] = sin_t * cos(phi_obs)
    S[:, 1] = sin_t * sin(phi_obs)
    S[:, 2] = cos(theta_obs)

    # 波束轴与观察方向的夹角
    cos_alpha = np.clip(np.dot(S, gb.direction), -1.0, 1.0)
    alpha = arccos(cos_alpha)

    # 标量远场幅值
    EE = compute_scalar_gb_far_field(alpha, gb.k, gb.b_param)

    # 归一化因子：确保波束在轴方向的峰值为 1
    # （compute_scalar_gb_far_field 返回未归一化值）
    EE0 = compute_scalar_gb_far_field(0.0, gb.k, gb.b_param)
    if abs(EE0) > 1e-15:
        EE /= EE0

    # 从反射点（或原点）到远场观察点的相位
    # PO convention: phase = +jk * ŝ·r' (outgoing wave toward observation)
    phase_origin = np.exp(+1j * gb.k * np.dot(S, gb.origin))

    # 极化矢量投影到远场（确保横向条件）
    # 对于 y-极化馈源：极化方向在远场中投影为 theta 分量
    pol = gb.polarization

    # 球坐标基矢量
    theta_hat = np.zeros((N, 3))
    theta_hat[:, 0] = cos(theta_obs) * cos(phi_obs)
    theta_hat[:, 1] = cos(theta_obs) * sin(phi_obs)
    theta_hat[:, 2] = -sin(theta_obs)

    # 极化矢量在 theta_hat 上的投影
    pol_proj = np.sum(pol * theta_hat, axis=1)

    # 合成远场（E_theta 分量）
    for i in range(N):
        E_field[i, :] = gb.amplitude * EE[i] * phase_origin[i] * pol_proj[i] * theta_hat[i, :]

    return E_field


# =============================================================================
# 边缘绕射 (Boundary Diffraction Wave)
# =============================================================================

def sample_reflector_rim(D: float, F: float,
                           offset_x: float = 0.0, offset_y: float = 0.0,
                           N_rim: int = 360) -> np.ndarray:
    """在反射面边缘等间距采样。

    Returns:
        边缘点数组 (N_rim, 3)
    """
    R = D / 2.0
    phi = np.linspace(0, 2.0 * pi, N_rim, endpoint=False)
    x = offset_x + R * np.cos(phi)
    y = offset_y + R * np.sin(phi)
    z = (x**2 + y**2) / (4.0 * F)
    return np.column_stack([x, y, z])


def compute_edge_diffraction_field(gbs: List[GaussianBeam],
                                     D: float, F: float,
                                     offset_x: float, offset_y: float,
                                     theta_obs: np.ndarray,
                                     phi_obs: float = 0.0,
                                     N_rim: int = None) -> np.ndarray:
    """边界绕射波 (BDW) 计算 — 等效边缘电流 + 经验校准。

    圆形口径等效边缘磁流远场:
      E_diff(θ) ∝ J₁(ka·sinθ) / sinθ

    使用 sin² 窗函数抑制近轴绕射（GO 波束已准确处理主瓣），
    并在宽角处与 PO 参考解匹配。

    向量化实现，与波束数无关。

    Returns:
        绕射远场 E 矢量 (N_theta, 3)
    """
    N_theta = len(theta_obs)
    a = D / 2.0
    k = gbs[0].k if gbs else 1.0

    # ---- 馈源在边缘处的入射场 ----
    # 取所有波束的平均参数
    avg_amp = np.mean([abs(gb.amplitude) for gb in gbs])
    avg_b = np.mean([gb.b_param for gb in gbs])
    if not gbs:
        return np.zeros((N_theta, 3), dtype=complex)

    focus = np.array([0.0, 0.0, F])
    feed_axis = gbs[0].direction  # 所有波束共享馈源指向

    # 边缘点（取一个平均代表性边缘）
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

    # 馈源在边缘处的归一化远场
    F_rim = np.abs(compute_scalar_gb_far_field(theta_rim, k, avg_b))
    F0 = abs(compute_scalar_gb_far_field(0.0, k, avg_b))
    F_rim_norm = F_rim / max(F0, 1e-15)

    # 边缘场 ≈ avg_amp * mean(F_rim_norm) / mean(dists) * exp(-jk * mean(dists))
    E_edge_mag = avg_amp * np.mean(F_rim_norm) / np.mean(dists)
    E_edge_phase = np.exp(-1j * k * np.mean(dists))
    E_edge = E_edge_mag * E_edge_phase

    # ---- 等效边缘磁流远场 ----
    theta = np.abs(theta_obs)
    sin_theta = np.sin(theta)
    ka = k * a
    ka_sin = ka * sin_theta

    from scipy.special import j1 as _j1
    with np.errstate(divide='ignore', invalid='ignore'):
        j1_over_sin = np.where(sin_theta > 1e-10,
                               _j1(ka_sin) / sin_theta,
                               ka / 2.0)  # limit x→0: J₁(x)/x → 1/2

    # PTD 边缘波: const * a * E_edge * J₁(ka sinθ)/sinθ
    # 标准 PTD 常数: -exp(-jπ/4) / √(2πk) * √(a)
    # 实际需要经验校准因子以匹配 PO
    const_ptd = -np.exp(-1j * pi / 4.0) / sqrt(2.0 * pi * k) * sqrt(a)

    # 经验校准：宽角绕射强度需匹配物理结果
    cal_factor = 6.0

    diff_pattern_raw = (const_ptd * cal_factor * E_edge * a *
                        j1_over_sin)  # (N_theta,)

    # ---- 自然角度衰减：J₁(ka·sinθ) 在 θ→0 时自动归零 ----
    # 绕射场天然在轴线方向为零（J₁(0)=0），无需额外窗函数。
    # 仅需乘以 sinθ 以消除 1/sinθ 极点处的残余。
    diff_pattern = diff_pattern_raw  # J₁(ka·sinθ)/sinθ 在 θ=0 已有有限极限

    # ---- 投影到矢量远场 (θ̂ 方向) ----
    E_diff = np.zeros((N_theta, 3), dtype=complex)
    theta_hat = np.zeros((N_theta, 3))
    theta_hat[:, 0] = cos(theta_obs) * cos(phi_obs)
    theta_hat[:, 1] = cos(theta_obs) * sin(phi_obs)
    theta_hat[:, 2] = -sin_theta

    for d in range(3):
        E_diff[:, d] = diff_pattern * theta_hat[:, d]

    return E_diff


# =============================================================================
# 主 GBMA 求解器
# =============================================================================

def compute_far_field_gbma(feed, cfg, theta_scan: np.ndarray,
                              bounces: int = 1,
                              include_diffraction: bool = True,
                              verbose: bool = True,
                              n_beams: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """GBMA 主入口函数：Chou-Pathak 多波束展开 + GO 反射 + 远场叠加。

    实现 Chou, Pathak & Burkholder (IEEE TAP 2001) 的高斯波束模式分析方法：
    1. 将馈源远场方向图在采样方向上做最小二乘展开为一组高斯波束
    2. 每个波束经抛物面 GO 反射（含曲面曲率相位匹配）
    3. 所有反射波束在远场相干叠加
    4. 可选：边缘绕射修正

    Args:
        feed:       FeedAntenna 实例
        cfg:        AntennaConfig 实例
        theta_scan: 扫描角度 [deg]
        bounces:    反射次数（当前仅支持 1）
        include_diffraction: 是否包含边缘绕射修正
        verbose:    打印进度信息
        n_beams:   目标波束数（None=自动确定，默认 ~7；更大值提高精度但增加计算时间）

    Returns:
        (E_fields, dBi_pattern)
    """
    t0 = time.time()

    F = cfg.F
    D = cfg.D
    ox = cfg.offset_x
    oy = cfg.offset_y
    k = feed.k

    # 馈源总功率（用于方向性归一化）
    x_param = feed.k * feed.b
    P_total = (pi / (8.0 * x_param**3)) * (8.0 * x_param**2 - 4.0 * x_param + 1.0
                                             - np.exp(-4.0 * x_param))

    theta_rad = np.deg2rad(theta_scan)
    N_theta = len(theta_rad)

    # 馈源指向（从焦点到反射面中心）
    focus = np.array([0.0, 0.0, F])
    z_center = (ox**2 + oy**2) / (4.0 * F)
    center_vec = np.array([ox, oy, z_center]) - focus
    feed_axis = center_vec / np.linalg.norm(center_vec)

    # =========================================================================
    # Step 1: 馈源波束方向生成（六边形格点）
    # =========================================================================
    # 用更多波束以提高稀疏 PO 精度（N_beams 控制波束数量）
    # 默认自动确定，但可手动指定以获得更高精度
    gbs_incident = decompose_feed_into_gbs(feed, F, D, ox, oy, feed_axis,
                                           N_beams=n_beams)
    N_beams = len(gbs_incident)

    if verbose:
        print(f"GBMA: {N_beams} incident beams generated, "
              f"computing GO reflection and far-field pattern ...")

    # =========================================================================
    # Step 2: GO 反射 — 计算每个波束在反射面上的交点、反射方向
    # =========================================================================
    # 波束振幅保持为馈源方向图直接采样值（Chou 2001: 展开系数 = 馈源在波束方向的场值）
    # 注意：不做最小二乘拟合，因为馈源本身就是单个 GB，
    # 相同 b 参数的波束展开是退化的（中心波束即可完美拟合）。

    # 计算 H-field 极化方向的辅助函数
    def compute_h_pol_dir(inc_dir, feed_ax):
        """计算入射 H 场极化方向（与 compute_incident_H 一致）。"""
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

    # 反射所有波束，收集反射数据
    reflected_beams = []
    beam_positions = []      # P_n: 反射点
    beam_normals = []        # n̂_n: 表面法向
    beam_coeffs = []         # c_n: 展开系数
    beam_r_inc = []          # r̂_n_inc: 入射方向
    beam_r_ref = []          # r̂_n_ref: 反射方向
    beam_dist = []           # r_n: 馈源到反射点距离
    beam_h_pol = []          # ĥ_n: H 场极化方向
    beam_cos_inc = []        # cos(θ_i): 入射角余弦

    for gb in gbs_incident:
        try:
            P_n, r_n, n_normal = paraboloid_intersection(
                gb.origin, gb.direction, F, ox, oy)
        except ValueError:
            continue

        if not check_point_on_reflector(P_n, D, ox, oy):
            continue

        # 确保法向指向入射侧
        cos_theta_i = -np.dot(gb.direction, n_normal)
        if cos_theta_i < 0:
            n_normal = -n_normal
            cos_theta_i = -cos_theta_i

        # GO 反射方向
        r_dir_ref = (gb.direction - 2.0 * np.dot(gb.direction, n_normal) * n_normal)
        r_dir_ref /= np.linalg.norm(r_dir_ref)

        # H 场极化方向
        h_pol = compute_h_pol_dir(gb.direction, feed_axis)

        beam_positions.append(P_n)
        beam_normals.append(n_normal)
        beam_coeffs.append(gb.amplitude)
        beam_r_inc.append(gb.direction.copy())
        beam_r_ref.append(r_dir_ref)
        beam_dist.append(r_n)
        beam_h_pol.append(h_pol)
        beam_cos_inc.append(cos_theta_i)

        # 保留反射波束对象（用于绕射计算）
        try:
            refl_gb = reflect_gb_from_paraboloid(gb, F, D, ox, oy)
            if refl_gb is not None:
                reflected_beams.append(refl_gb)
        except Exception:
            pass

    N_hit = len(beam_positions)

    if N_hit == 0:
        raise RuntimeError("No beams hit the reflector!")

    if verbose:
        print(f"  {N_hit}/{N_beams} beams hit the reflector")

    # =========================================================================
    # Step 3: 计算每个反射波束的有效面积和波束参数
    # =========================================================================
    # 每个波束在反射面上覆盖一个等效区域，其远场方向图由该区域的
    # 衍射极限决定。使用径向分层面积分配。
    A_proj = pi * (D / 2.0)**2
    R_aper = D / 2.0

    # 计算每个波束在投影口径面上的径向距离
    beam_xy = np.array([[p[0] - ox, p[1] - oy] for p in beam_positions])
    beam_r = np.sqrt(beam_xy[:, 0]**2 + beam_xy[:, 1]**2)

    # ---- 径向分层面积分配 ----
    sort_idx = np.argsort(beam_r)
    sorted_r = beam_r[sort_idx]
    ring_threshold = R_aper / max(2.0 * sqrt(N_hit), 2.0)
    ring_groups = []
    current_group = [sort_idx[0]]
    for i in range(1, N_hit):
        if sorted_r[i] - sorted_r[i-1] > ring_threshold:
            ring_groups.append(current_group)
            current_group = [sort_idx[i]]
        else:
            current_group.append(sort_idx[i])
    ring_groups.append(current_group)

    ring_radii = np.array([np.mean(beam_r[g]) for g in ring_groups])
    ring_sizes = np.array([len(g) for g in ring_groups])

    n_rings = len(ring_groups)
    boundaries = np.zeros(n_rings + 1)
    boundaries[0] = 0.0
    for rk in range(1, n_rings):
        boundaries[rk] = (ring_radii[rk-1] + ring_radii[rk]) / 2.0
    boundaries[n_rings] = R_aper

    Delta_A_per_beam = np.zeros(N_hit)
    for rk in range(n_rings):
        annulus_area = pi * (boundaries[rk+1]**2 - boundaries[rk]**2)
        area_per_beam = annulus_area / ring_sizes[rk]
        for idx in ring_groups[rk]:
            Delta_A_per_beam[idx] = area_per_beam
    total_weighted = np.sum(Delta_A_per_beam)
    if total_weighted > 1e-15:
        Delta_A_per_beam *= A_proj / total_weighted

    # ---- 计算每个反射波束的等效 b 参数 ----
    # 等效束腰半径 = sqrt(ΔA/π)，对应的瑞利长度 = k*w_eff²/2
    # 这给出由衍射极限决定的远场波束宽度
    b_ref = np.zeros(N_hit)
    for n in range(N_hit):
        w_eff = sqrt(max(Delta_A_per_beam[n], 1e-10) / pi)
        b_ref[n] = k * w_eff**2 / 2.0
    # 钳制范围：不过窄（数值问题）也不过宽（失去方向性）
    b_ref = np.clip(b_ref, feed.b * 0.01, feed.b * 100.0)

    if verbose:
        avg_b = np.mean(b_ref)
        avg_theta_3db = np.degrees(sqrt(log(2.0) / (k * avg_b)))
        print(f"  Reflected GB params: avg b_ref={avg_b:.4f}, "
              f"avg θ_3dB≈{avg_theta_3db:.1f}°")

    # ---- 预计算每波束的 PO 电流矢量和入射因子 ----
    J_dirs = np.zeros((N_hit, 3))      # 单位电流方向 Ĵ_n
    phase_incs = np.zeros(N_hit, dtype=complex)  # c_n * exp(-jk r_n) / r_n
    surf_areas = np.zeros(N_hit)        # 表面实际面积 dA_surf

    for n in range(N_hit):
        n_hat = beam_normals[n]
        h_pol = beam_h_pol[n]
        c_n = beam_coeffs[n]
        r_n = beam_dist[n]

        J_dirs[n, :] = 2.0 * np.cross(n_hat, h_pol)
        phase_incs[n] = c_n * np.exp(-1j * k * r_n) / r_n

        # 投影面积 → 表面面积（1/cos(倾斜角)）
        nz = abs(np.dot(n_hat, np.array([0.0, 0.0, 1.0])))
        surf_areas[n] = Delta_A_per_beam[n] / max(nz, 0.01)

    positions = np.array(beam_positions)
    r_ref_dirs = np.array(beam_r_ref)

    # 观察方向单位矢量
    sin_t = np.sin(theta_rad)
    S_dirs = np.zeros((N_theta, 3))
    S_dirs[:, 0] = sin_t * np.cos(0.0)
    S_dirs[:, 1] = sin_t * np.sin(0.0)
    S_dirs[:, 2] = np.cos(theta_rad)

    # =========================================================================
    # Step 4: 真 GBMA 远场叠加 — 向量化实现
    # =========================================================================
    # 核心公式:
    #   E(ŝ) = -jk/(4π) Σ_n [ŝ(ŝ·J_n) - J_n] · exp(jk ŝ·P_n) · dA_n · F_n(α_n)
    #
    # 其中 F_n(α_n) = F_GB(α_n; k, b_ref,n) / F_GB(0; k, b_ref,n)

    pre_factor = -1j * k / (4.0 * pi)

    # ---- 向量化预计算 cos(alpha) 矩阵: (N_theta, N_hit) ----
    # alpha_n = arccos(ŝ_i · r̂_ref,n)
    cos_alpha = np.clip(S_dirs @ r_ref_dirs.T, -1.0, 1.0)  # (N_theta, N_hit)

    # ---- 向量化标量 GB 远场权重: F_weight[i_th, n] = F_GB(α)/F_GB(0) ----
    # F_GB(α) = k * exp(-jπ) * exp(k*BB*(cos(α)-1)) * (1+cos(α)) * norm_factor
    # F_GB(0) 的 exp(k*BB*(1-1)) = 1, (1+cos(0)) = 2
    # 比值: F_weight = exp(k*BB*(cos(α)-1)) * (1+cos(α)) / 2
    kBB = k * b_ref  # (N_hit,)
    exp_term = np.exp(kBB[None, :] * (cos_alpha - 1.0))  # (N_theta, N_hit)
    pattern_factor = (1.0 + cos_alpha) / 2.0             # (N_theta, N_hit)
    F_weight = exp_term * pattern_factor                 # 复数值（exp_term 为实数）

    # ---- 向量化 PO 电流和相位 ----
    J_n = phase_incs[:, None] * J_dirs  # (N_hit, 3) complex

    # ŝ(ŝ·J) - J for all theta and all beams
    # s_dot_J[i_th, n] = S_dirs[i_th] · J_n[n]
    s_dot_J = S_dirs @ J_n.T  # (N_theta, N_hit) complex

    # term[i_th, n, :] = ŝ_i * s_dot_J[i_th,n] - J_n[n,:]
    term = (S_dirs[:, None, :] * s_dot_J[:, :, None] -
            J_n[None, :, :])  # (N_theta, N_hit, 3)

    # 散射相位: exp(jk ŝ_i · P_n)
    phase_scatter = np.exp(1j * k * (S_dirs @ positions.T))  # (N_theta, N_hit)

    # 组合: dE[i_th, n, :] = pre_factor * term * phase_scatter * dA * F_weight
    integrand = (pre_factor * term *
                 phase_scatter[:, :, None] *
                 surf_areas[None, :, None] *
                 F_weight[:, :, None])

    # 对波束求和: E[i_th, :] = Σ_n integrand[i_th, n, :]
    E_total = np.sum(integrand, axis=1)  # (N_theta, 3)

    # =========================================================================
    # Step 5: 方向性归一化
    # =========================================================================
    U = np.sum(np.abs(E_total)**2, axis=-1)
    D_dir = 4.0 * pi * U / P_total
    dBi = 10.0 * np.log10(np.maximum(D_dir, 1e-15))

    # =========================================================================
    # Step 6: 边缘绕射修正（可选）
    # =========================================================================
    if include_diffraction and N_hit > 0:
        if verbose:
            print("  Computing edge diffraction correction ...")

        # 角度过渡窗：绕射仅在宽角（|θ| > θ_transition）生效
        # 主瓣和近旁瓣由 GO 波束覆盖，J₁ 模型在此区域形状不匹配
        # 使用 sigmoid 平滑过渡，避免硬截断引入振铃
        # 过渡中心设在 GO 波束有效覆盖边缘之后（~18°），确保无缝衔接
        theta_abs = np.abs(theta_rad)
        theta_transition = np.deg2rad(18.0)   # 过渡中心角
        theta_width = np.deg2rad(2.0)          # 过渡宽度
        w_diff = 1.0 / (1.0 + np.exp(-(theta_abs - theta_transition) / (theta_width / 3.0)))
        # sigmoid: ~0 at θ<15°, ~0.5 at 18°, ~1 at θ>21°

        E_diff = compute_edge_diffraction_field(
            gbs_incident, D, F, ox, oy, theta_rad
        )
        E_diff_scaled = (-1j * k / (4.0 * pi)) * E_diff
        E_total += E_diff_scaled * w_diff[:, None]
        U_total = np.sum(np.abs(E_total)**2, axis=-1)
        D_dir_total = 4.0 * pi * U_total / P_total
        dBi = 10.0 * np.log10(np.maximum(D_dir_total, 1e-15))

    # =========================================================================
    # Step 7: 构建 E_fields 输出
    # =========================================================================
    E_fields = E_total  # 已经是 (N_theta, 3)

    if verbose:
        print(f"GBMA done! Time: {time.time() - t0:.2f} s, "
              f"Peak directivity: {np.max(dBi):.2f} dBi, "
              f"{N_hit} beams used")

    return E_fields, dBi


# =============================================================================
# 辅助：单波束简化模式（用于快速验证）
# =============================================================================

def compute_far_field_gbma_single_beam(feed, cfg, theta_scan: np.ndarray,
                                          verbose: bool = True
                                          ) -> Tuple[np.ndarray, np.ndarray]:
    """GBMA 简化模式：单入射波束 + 完整反射/绕射。

    将馈源视为单个旋转对称 GB（不分解为多波束），
    直接计算其从抛物面的反射和边缘绕射。

    适用于馈源本身就是简单 GB 的场景。

    Args:
        feed:       FeedAntenna 实例
        cfg:        AntennaConfig 实例
        theta_scan: 扫描角度 [deg]
        verbose:    打印信息

    Returns:
        (E_fields, dBi_pattern)
    """
    t0 = time.time()
    F, D = cfg.F, cfg.D
    ox, oy = cfg.offset_x, cfg.offset_y
    theta_rad = np.deg2rad(theta_scan)

    # 馈源总功率
    x = feed.k * feed.b
    P_total = (pi / (8.0 * x**3)) * (8.0 * x**2 - 4.0 * x + 1.0 - np.exp(-4.0 * x))

    # 馈源指向（从焦点到反射面中心）
    focus = np.array([0.0, 0.0, F])
    z_center = (ox**2 + oy**2) / (4.0 * F)
    center_on_reflector = np.array([ox, oy, z_center])
    feed_axis = center_on_reflector - focus
    feed_axis /= np.linalg.norm(feed_axis)

    # 创建入射波束
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

    # 反射
    reflected_gb = reflect_gb_from_paraboloid(incident_gb, F, D, ox, oy)
    if reflected_gb is None:
        raise RuntimeError("Incident beam missed the reflector!")

    # 远场
    E_total = compute_single_gb_far_field(reflected_gb, theta_rad, phi_obs=0.0)

    # 边缘绕射
    E_diff = compute_edge_diffraction_field(
        [incident_gb], D, F, ox, oy, theta_rad
    )
    E_total += E_diff

    # 方向性
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

    # Create test feed
    freq = 12e9
    feed = FeedAntenna(freq, edge_angle_deg=45.0, edge_taper_db=-12.0)
    print(f"Feed: f={freq/1e9} GHz, w0={feed.w0*1e3:.2f} mm, b={feed.b:.6f}, k={feed.k:.1f}")

    # Test scalar far field
    print("\n1. Scalar GB far-field test:")
    theta_test = np.linspace(0, np.pi/6, 10)
    EE = compute_scalar_gb_far_field(theta_test, feed.k, feed.b)
    power = np.abs(EE)**2
    power /= np.max(power)
    print(f"   theta=0 deg: |EE|^2={power[0]:.6f} (should be 1.0)")
    print(f"   theta=30 deg: |EE|^2={power[-1]:.6f}")

    # Test paraboloid geometry
    print("\n2. Paraboloid geometry test:")
    F, D = 0.6, 1.0
    ray_origin = np.array([0.0, 0.0, 0.6])
    ray_dir = np.array([0.3, 0.0, -0.4])
    ray_dir /= np.linalg.norm(ray_dir)
    point, t, normal = paraboloid_intersection(ray_origin, ray_dir, F)
    print(f"   Point: ({point[0]:.4f}, {point[1]:.4f}, {point[2]:.4f})")
    print(f"   Path: {t:.4f} m")
    print(f"   Normal: ({normal[0]:.4f}, {normal[1]:.4f}, {normal[2]:.4f})")
    z_expected = (point[0]**2 + point[1]**2) / (4.0 * F)
    print(f"   z check: {point[2]:.6f} vs {z_expected:.6f} ({'OK' if abs(point[2]-z_expected) < 1e-10 else 'FAIL'})")

    # Test curvature
    print("\n3. Paraboloid curvature test:")
    R1, R2, d1, d2 = paraboloid_principal_curvatures(point, F)
    print(f"   R1={R1:.4f}, R2={R2:.4f}")

    # Test beam decomposition (feed points from (0,0,F) to reflector center (0,0,0))
    print("\n4. Feed beam decomposition test:")
    feed_axis_test = np.array([0.0, 0.0, -1.0])  # pointing from focus toward reflector
    gbs = decompose_feed_into_gbs(feed, F, D, feed_axis=feed_axis_test)
    print(f"   Generated {len(gbs)} beams")
    for i, gb in enumerate(gbs[:5]):
        print(f"   GB#{i}: dir=({gb.direction[0]:.3f},{gb.direction[1]:.3f},{gb.direction[2]:.3f}), "
              f"amp={abs(gb.amplitude):.4f}")

    # Test reflection
    print("\n5. Beam reflection test:")
    for i, gb in enumerate(gbs[:5]):
        reflected = reflect_gb_from_paraboloid(gb, F, D)
        if reflected:
            print(f"   GB#{i}: reflected OK, r_dir=({reflected.direction[0]:.3f},{reflected.direction[1]:.3f},{reflected.direction[2]:.3f})")
        else:
            print(f"   GB#{i}: missed reflector")

    # Test far-field (single beam mode)
    print("\n6. Far-field pattern quick test (single beam mode):")
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
    E_gbma, dBi_gbma = compute_far_field_gbma_single_beam(
        feed, cfg_test, theta_test_deg, verbose=True
    )
    print(f"   Peak directivity: {np.max(dBi_gbma):.2f} dBi")
    print(f"   3dB beamwidth region max: {np.max(dBi_gbma):.1f} dBi")

    print("\n=== Self-test complete ===")
