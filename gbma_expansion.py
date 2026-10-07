"""
gbma_expansion.py — 馈源远场的 GB 展开（GBMA 第 2 层，Chou 2001 §II）

包含：
  - generate_hexagonal_beam_directions: 六边形波束格点生成
  - rotate_directions_to_axis: 方向矢量从 +z 旋转到馈源指向
  - make_basis_grid: 基波束格点构造（重叠判据 Δθ̄ = √(2/(k·b_basis))），
      支持两种模式（mode='surface' / mode='farfield'）
  - surface_expansion: 密集窄波束模式的展开系数——在反射面上用近场
      基函数做最小二乘匹配（w(z), R(z), Gouy 相位, 波前相位, 倾斜因子）
  - far_field_expansion: 稀疏宽波束模式的展开系数——Chou 2001 §II
      原文献做法，在远场方向图上最小二乘匹配
  - decompose_feed_into_gbs: 文献模式兼容入口（返回 GaussianBeam 列表）

两种模式的关系：
  'surface'（主流程默认）使用密集窄基波束（b_basis = 100×b_feed →
  ~500 束），基波束在反射面处仍处于近场（zR 与馈源-反射面距离同量级），
  必须在反射面上匹配近场基；宽角边界波采样细、精度高。
  'farfield'（Chou 2001 原文献量级 ~45–60 束）使用稀疏宽基波束
  （b 较小、反射面位于准远场 r/zR≈4–6），在远场方向图上匹配即可；
  波束少、计算快，主瓣精度相当，宽角区略粗。
"""

import numpy as np
from numpy import pi, sqrt, sin, cos, tan, arccos, exp
from typing import List

from gbma_beam import GaussianBeam


# =============================================================================
# 六边形波束格点
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
    try:
        from gbma_fast import hex_directions_fast
        return hex_directions_fast(cone_half_angle, delta_theta)
    except ImportError:
        pass

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


def rotate_directions_to_axis(directions: np.ndarray, target_axis: np.ndarray) -> np.ndarray:
    """将方向矢量集合从 +z 轴旋转到目标轴方向。

    使用罗德里格斯旋转公式。处理 180° 旋转的特殊情况。

    Args:
        directions:  方向数组 (N, 3)，当前相对于 +z
        target_axis: 目标轴单位矢量 (3,)

    Returns:
        旋转后的方向数组 (N, 3)
    """
    try:
        from gbma_fast import rotate_directions_fast
        return rotate_directions_fast(directions, target_axis)
    except ImportError:
        pass

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


# =============================================================================
# 基波束格点构造（两种模式）
# =============================================================================

def make_basis_grid(feed, F: float, D: float, offset_x: float,
                    offset_y: float, feed_axis: np.ndarray,
                    n_beams: int = None, mode: str = 'surface',
                    spot_fill: float = 0.25):
    """基波束格点构造（Chou 2001 §II 重叠判据 Δθ̄ = √(2/(k·b_basis))）。

    mode 选择展开方案：
      'surface'  （默认）密集窄基波束：b_basis = 100×b_feed，基波束在
                 反射面处处于近场，展开系数必须在反射面上匹配近场基
                 （surface_expansion）；宽角边界波采样细、精度高。
      'farfield' Chou 2001 §II 原文献链条（式 (13)–(19)）：
                 · 由锥半角 Ω 经光斑判据 (14)(15) 解 b：波束在反射面
                   边缘距离 z̄ 处的光斑半径取 w(z̄)=z̄·tanΩ·spot_fill，
                   代入 w²=(2/k)(b+z̄²/b) 解二次方程、取小根
                   （更少波束、反射面处于准远场；spot_fill≈0.25 给出
                   文献量级 ~40–60 束）；
                 · (13) 定角间距 Δθ̄=√(2/(k·b))（只有最近邻波束显著
                   重叠）；
                 · (19) 下限 Δθ̄ ≥ λ/(π·D)；
                 · 波束数 N 是链条的产出而非输入；n_beams 请求更多
                   波束时，按论文规则固定 b、只减小 Δθ̄（波束物理结构
                   不变，主瓣峰值保持稳定）。

    Returns:
        (dirs, b_basis, delta_theta, cone_half_angle)
    """
    focus = np.array([0.0, 0.0, F])

    # ---- 覆盖反射面的锥半角（相对馈源指向，10% 余量）----
    R = D / 2.0
    edge_angles = []
    for j in range(72):
        phi = 2.0 * pi * j / 72
        rim_x = offset_x + R * cos(phi)
        rim_y = offset_y + R * sin(phi)
        rim_z = (rim_x**2 + rim_y**2) / (4.0 * F)
        vec = np.array([rim_x, rim_y, rim_z]) - focus
        vec /= np.linalg.norm(vec)
        edge_angles.append(arccos(np.clip(np.dot(vec, feed_axis), -1.0, 1.0)))
    cone_half_angle = max(edge_angles) * 1.10

    # ---- 基波束参数与角间距（重叠判据 Δθ̄ = √(2/(k·b))）----
    if mode == 'surface':
        b_basis = feed.b * 100.0
        delta_theta = sqrt(2.0 / (feed.k * b_basis))
        if n_beams is not None and n_beams > 1:
            # 按目标波束数反推角间距（六边形格点 N ≈ 1 + 3n(n+1)）
            target_n_rings = max(1.0, sqrt((n_beams - 1.0) / 3.0))
            target_delta = cone_half_angle / target_n_rings
            for _ in range(4):
                test_dirs = generate_hexagonal_beam_directions(cone_half_angle, target_delta)
                ratio = sqrt(len(test_dirs) / max(n_beams, 1))
                target_delta *= ratio
                if abs(ratio - 1.0) < 0.05:
                    break
            delta_theta = target_delta
            b_basis = 2.0 / (feed.k * delta_theta**2)
        # 至少约 6 圈，保证能分辨馈源方向图结构
        if delta_theta > cone_half_angle / 6.0:
            delta_theta = cone_half_angle / 6.0
            b_basis = 2.0 / (feed.k * delta_theta**2)
    elif mode == 'farfield':
        # ---- Chou 2001 §II 原文献链条 ----
        # (14)(15): 波束在反射面边缘距离 z̄ 处的光斑半径取
        #   w(z̄) = z̄·tanΩ·spot_fill，代入 w² = (2/k)(b + z̄²/b)
        #   解关于 b 的二次方程，取小根（更少波束、反射面处于准远场）。
        # (13): Δθ̄ = √(2/(k·b))（只有最近邻波束显著重叠）
        # (19): Δθ̄ ≥ λ/(π·D) 下限
        # 波束数 N 是以上链条的产出，不是输入；
        # 若需更多波束：固定 b、只减小 Δθ̄（论文 §II 末段的规则）。
        rim_dist = []
        for j in range(72):
            phi = 2.0 * pi * j / 72
            rim_x = offset_x + (D / 2.0) * cos(phi)
            rim_y = offset_y + (D / 2.0) * sin(phi)
            rim_z = (rim_x**2 + rim_y**2) / (4.0 * F)
            rim_dist.append(np.linalg.norm(np.array([rim_x, rim_y, rim_z]) - focus))
        z_bar = float(np.mean(rim_dist))

        w_target = z_bar * tan(cone_half_angle) * spot_fill
        half = feed.k * w_target**2 / 2.0
        disc = half**2 - 4.0 * z_bar**2
        if disc >= 0.0:
            b_basis = (half - sqrt(disc)) / 2.0   # 小根
        else:
            b_basis = z_bar                        # 判别式<0: 最小光斑 b=z̄
        b_basis = max(b_basis, 1e-9)

        delta_theta = sqrt(2.0 / (feed.k * b_basis))       # (13)
        delta_theta_min = feed.lam / (pi * D)              # (19) 下限
        if delta_theta < delta_theta_min:
            delta_theta = delta_theta_min
            b_basis = 2.0 / (feed.k * delta_theta**2)

        # 可选加密：固定 b、只减小 Δθ̄（波束物理结构不变，峰值稳定）
        if n_beams is not None and n_beams > 1:
            n0 = len(generate_hexagonal_beam_directions(cone_half_angle, delta_theta))
            if n_beams > n0:
                target_delta = delta_theta * sqrt(n0 / max(n_beams, 1))
                for _ in range(4):
                    test_dirs = generate_hexagonal_beam_directions(cone_half_angle, target_delta)
                    ratio = sqrt(len(test_dirs) / max(n_beams, 1))
                    target_delta *= ratio
                    if abs(ratio - 1.0) < 0.05:
                        break
                delta_theta = min(target_delta, delta_theta)
            # n_beams <= n0: 保持论文默认（不放大间距，避免重叠不足）
    else:
        raise ValueError(f"unknown mode '{mode}' (choose 'surface' or 'farfield')")

    # ---- 波束格点（旋转到实际馈源指向）----
    dirs_z = generate_hexagonal_beam_directions(cone_half_angle, delta_theta)
    dirs = rotate_directions_to_axis(dirs_z, feed_axis)
    return dirs, b_basis, delta_theta, cone_half_angle


# =============================================================================
# 展开系数求解（两种模式）
# =============================================================================

def surface_expansion(feed, P: np.ndarray, r_n: np.ndarray,
                      P_all: np.ndarray, r_all: np.ndarray,
                      dirs: np.ndarray, focus: np.ndarray,
                      feed_axis: np.ndarray, k: float,
                      w0_basis: float, zR_basis: float,
                      verbose: bool = True):
    """mode='surface' 的展开系数：反射面上近场基最小二乘展开。

    基波束 n 在命中点 P_m 处的近场：
      幅值 (w0/w_n)·(r_m/r_n)，横向高斯 exp(−ρ²/w_n²)，
      波前相位 exp(−jk·ρ²/(2R_n))，轴上相位 exp(−jk·(z_mn−r_m))，
      Gouy 相位 exp(jψ_n)，倾斜因子 (1+cosα_mn)/2。
    目标 = 馈源在命中点处剥离 e^{-jk r}/r 后的方向图值（与 PO 参考一致）。

    Args:
        feed:      FeedAntenna 实例
        P, r_n:    命中点坐标 (N_hit,3) 与光程 (N_hit,)
        P_all, r_all: 全部波束轴上交点 (N_beams,3) 与光程 (N_beams,)
        dirs:      全部波束方向 (N_beams,3)
        focus:     焦点坐标 (3,)
        feed_axis: 馈源指向单位矢量 (3,)
        k:         波数
        w0_basis, zR_basis: 基波束束腰半径与瑞利长度

    Returns:
        (C, rel_err): 展开系数 (N_beams,) 与重构相对误差
    """
    w_all = w0_basis * sqrt(1.0 + (r_all / zR_basis)**2)
    R_all = r_all * (1.0 + (zR_basis / r_all)**2)
    gouy_all = np.arctan(r_all / zR_basis)

    k_hat = (P - focus) / r_n[:, None]           # (N_hit,3) 命中点馈源方向
    Dm = P[:, None, :] - P_all[None, :, :]       # (N_hit, N_beams, 3)
    proj = np.einsum('mnd,nd->mn', Dm, dirs)     # (P_m−P_n)·ŝ_n
    rho2 = np.einsum('mnd,mnd->mn', Dm, Dm) - proj**2
    z_mn = r_all[None, :] + proj
    cosA = np.clip(k_hat @ dirs.T, -1.0, 1.0)

    A = ((w0_basis / w_all)[None, :] * (r_n[:, None] / np.maximum(r_all[None, :], 1e-12))
         * (1.0 + cosA) / 2.0
         * np.exp(-rho2 / w_all[None, :]**2)
         * np.exp(-1j * k * (rho2 / (2.0 * R_all[None, :]) + z_mn - r_n[:, None]))
         * np.exp(1j * gouy_all[None, :]))
    A[:, ~np.isfinite(r_all)] = 0.0
    y = feed.far_field_amplitude(arccos(np.clip(k_hat @ feed_axis, -1.0, 1.0)))
    try:
        from gbma_fast import solve_lsq_fast
        C = solve_lsq_fast(A, y)
    except ImportError:
        C, *_ = np.linalg.lstsq(A, y, rcond=1e-8)

    rel = np.linalg.norm(A @ C - y) / max(np.linalg.norm(y), 1e-30)
    if verbose:
        print(f"  GBMA surface expansion: {len(dirs)} beams, {len(P)} match pts, "
              f"LSQ rel.err={rel:.3e}")
    return C, rel


def far_field_expansion(feed, dirs: np.ndarray, b_basis: float,
                        delta_theta: float, cone_half_angle: float,
                        feed_axis: np.ndarray, verbose: bool = True):
    """mode='farfield' 的展开系数：Chou 2001 §II 远场方向图最小二乘。

    基函数 F_GB(α) = e^{kb(cosα−1)}·(1+cosα)/2 在加密一倍的远场测试
    格点上匹配馈源远场方向图，求复数展开系数 C：
        Σ_n C_n · F_GB(∠(ŝ_t, ŝ_n)) = F_feed(θ_t)

    Returns:
        (C, rel_err): 展开系数 (N_beams,) 与重构相对误差
    """
    test_dirs_z = generate_hexagonal_beam_directions(cone_half_angle, delta_theta / 2.0)
    test_dirs = rotate_directions_to_axis(test_dirs_z, feed_axis)
    cosA = np.clip(test_dirs @ dirs.T, -1.0, 1.0)
    A = np.exp(feed.k * b_basis * (cosA - 1.0)) * (1.0 + cosA) / 2.0
    y = feed.far_field_amplitude(arccos(np.clip(test_dirs @ feed_axis, -1.0, 1.0)))
    C, *_ = np.linalg.lstsq(A, y, rcond=1e-8)

    rel = np.linalg.norm(A @ C - y) / max(np.linalg.norm(y), 1e-30)
    if verbose:
        print(f"  GB feed decomposition (Chou 2001 far-field LSQ): {len(dirs)} beams, "
              f"dtheta={np.rad2deg(delta_theta):.2f} deg, b_basis={b_basis:.4f} m, "
              f"LSQ rel.err={rel:.3e}")
    return C, rel


# =============================================================================
# 文献模式兼容入口
# =============================================================================

def decompose_feed_into_gbs(feed, F: float, D: float,
                              offset_x: float = 0.0, offset_y: float = 0.0,
                              feed_axis: np.ndarray = None,
                              N_beams: int = None) -> List[GaussianBeam]:
    """（兼容入口）Chou 2001 §II 原文献形式：远场最小二乘 GB 分解。

    = make_basis_grid(mode='farfield') + far_field_expansion。
    返回携带复数展开系数 C_n 的 GaussianBeam 列表。

    Args:
        feed:      FeedAntenna 实例
        F:         焦距
        D:         反射面投影直径
        offset_x, offset_y: 反射面偏移
        feed_axis: 馈源指向单位矢量（从焦点指向反射面中心），默认 [0,0,1]
        N_beams:   目标波束数（None=文献量级 ~45）

    Returns:
        GaussianBeam 对象列表（amplitude 为复数展开系数 C_n）
    """
    if feed_axis is None:
        feed_axis = np.array([0.0, 0.0, 1.0])
    feed_axis = feed_axis / np.linalg.norm(feed_axis)

    dirs, b_basis, delta_theta, cone_half_angle = make_basis_grid(
        feed, F, D, offset_x, offset_y, feed_axis, n_beams=N_beams,
        mode='farfield')
    C, _ = far_field_expansion(feed, dirs, b_basis, delta_theta,
                               cone_half_angle, feed_axis, verbose=True)

    focus = np.array([0.0, 0.0, F])
    w0_basis = sqrt(2.0 * b_basis / feed.k)
    gbs = []
    for direction, c_n in zip(dirs, C):
        gbs.append(GaussianBeam(
            direction=direction,
            origin=focus.copy(),
            w0=w0_basis,
            k=feed.k,
            b_param=b_basis,
            amplitude=complex(c_n)
        ))
    return gbs


# Legacy backward compatibility
decompose_feed_into_gbs_v1 = decompose_feed_into_gbs
