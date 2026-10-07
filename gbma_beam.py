"""
gbma_beam.py — 高斯波束基础模块（GBMA 第 1 层）

包含：
  - GaussianBeam 数据结构（旋转对称高斯波束的完整描述）
  - 标量高斯波束远场方向图（GRASP 复源点模型的远场形式）
  - H 场极化方向辅助函数

References:
  GRASP Technical Description §4.5 — 复源点高斯馈源模型
  Chou, Pathak & Burkholder, IEEE TAP, 2001 — GBMA
"""

import numpy as np
from numpy import pi, sqrt
from dataclasses import dataclass
from typing import Optional
import cmath


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
# 极化辅助
# =============================================================================

def compute_h_pol_dir(inc_dir: np.ndarray, feed_ax: np.ndarray) -> np.ndarray:
    """入射 H 场极化方向（与 mesh_generator.compute_incident_H 保持一致）。"""
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


# 向后兼容别名
_compute_h_pol_dir = compute_h_pol_dir
