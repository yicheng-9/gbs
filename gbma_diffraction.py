"""
gbma_diffraction.py — 经验边缘绕射修正（GBMA 第 3 层，旧版 J₁ 模型）

注意：主流程的边缘贡献（PO 积分自身的边界绕射波 P_d）已内置于
gbma_reflection.far_field_sum 的方向相关 erf 因子中；本模块是旧版
经验 J₁(ka·sinθ)/sinθ 修正，仅当 include_diffraction=True 时叠加，
用于快速对比，不代表论文的 PTD 修正。
"""

import numpy as np
from numpy import pi, sqrt, sin, cos, arccos
from typing import List

from gbma_beam import GaussianBeam, compute_scalar_gb_far_field


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
