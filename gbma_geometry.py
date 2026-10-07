"""
gbma_geometry.py — 抛物面几何工具（GBMA 第 1 层）

包含：
  - 射线与抛物面 z = (x²+y²)/(4F) 的交点
  - 抛物面主曲率半径与主方向（Weingarten 矩阵特征分解）
  - 反射面孔径投影范围判断
"""

import numpy as np
from numpy import sqrt
from typing import Tuple


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
