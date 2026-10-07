"""
gbma_solver.py — GBMA 主求解器（整合层，GBMA 第 4 层）

分层结构：
  gbma_beam.py        第 1 层: 高斯波束数据结构与标量远场
  gbma_geometry.py    第 1 层: 抛物面几何工具
  gbma_expansion.py   第 2 层: 馈源远场 GB 展开 (Chou 2001 §II)
  gbma_reflection.py  第 3 层: 波束追踪/像散反射/截断+边界绕射波
                              (Chou & Pathak, Radio Science 1997)
  gbma_diffraction.py 第 3 层: 旧版经验 J₁ 边缘绕射修正（可选）
  gbma_solver.py      第 4 层: compute_far_field_gbma 流程整合（本文件）

主流程（compute_far_field_gbma，默认不含经验绕射修正）：
  Step 1    make_basis_grid        六边形格点 + 重叠判据 (gbma_expansion)
  Step 2    trace_beams            波束追踪 + GO 反射几何 (gbma_reflection)
  Step 2.5  surface_expansion      反射面上近场基最小二乘展开 (gbma_expansion)
  Step 3    incident_beam_params   入射近场参数 + PO 等效电流 (gbma_reflection)
  Step 4/4.5/5 far_field_sum       闭式光斑远场叠加，含截断与边界绕射波
                                   P_d（PO 积分自身的边缘贡献）(gbma_reflection)
  Step 6    方向性归一化
  Step 7    可选: 旧版经验 J₁ 绕射修正 (gbma_diffraction, include_diffraction=True)

References:
  Chou & Pathak, Radio Science, 1997 — GB 反射/绕射的统一渐近解
  Chou, Pathak & Burkholder, IEEE TAP, 2001 — GBMA 在大型反射面天线中的快速分析
  Rieckmann et al., IEE Proc. MAP, 2002 — 基于高斯波束绕射的模块化多反射面分析
"""

import time

import numpy as np
from numpy import pi, sqrt, sin, cos
from typing import Tuple

from gbma_beam import (GaussianBeam, compute_scalar_gb_far_field,
                       compute_gb_far_field_power, compute_h_pol_dir,
                       _compute_h_pol_dir)
from gbma_geometry import (paraboloid_intersection,
                           paraboloid_principal_curvatures,
                           check_point_on_reflector)
from gbma_expansion import (generate_hexagonal_beam_directions,
                            rotate_directions_to_axis, make_basis_grid,
                            surface_expansion, far_field_expansion,
                            decompose_feed_into_gbs,
                            decompose_feed_into_gbs_v1)
from gbma_reflection import (trace_beams, incident_beam_params,
                             far_field_sum, reflect_gb_from_paraboloid,
                             compute_single_gb_far_field)
from gbma_diffraction import (sample_reflector_rim,
                              compute_edge_diffraction_field)

__all__ = [
    'compute_far_field_gbma',
    'compute_far_field_gbma_single_beam',
    # 第 1 层
    'GaussianBeam', 'compute_scalar_gb_far_field', 'compute_gb_far_field_power',
    'compute_h_pol_dir', '_compute_h_pol_dir',
    'paraboloid_intersection', 'paraboloid_principal_curvatures',
    'check_point_on_reflector',
    # 第 2 层
    'generate_hexagonal_beam_directions', 'rotate_directions_to_axis',
    'make_basis_grid', 'surface_expansion',
    'decompose_feed_into_gbs', 'decompose_feed_into_gbs_v1',
    # 第 3 层
    'trace_beams', 'incident_beam_params', 'far_field_sum',
    'reflect_gb_from_paraboloid', 'compute_single_gb_far_field',
    'sample_reflector_rim', 'compute_edge_diffraction_field',
]


def _feed_total_power(feed) -> float:
    """馈源总辐射功率（解析式，用于方向性归一化）。"""
    x = feed.k * feed.b
    return (pi / (8.0 * x**3)) * (8.0 * x**2 - 4.0 * x + 1.0 - np.exp(-4.0 * x))


# =============================================================================
# 主 GBMA 求解器
# =============================================================================

def compute_far_field_gbma(feed, cfg, theta_scan: np.ndarray,
                              bounces: int = 1,
                              include_diffraction: bool = False,
                              verbose: bool = True,
                              n_beams: int = None,
                              mode: str = 'surface') -> Tuple[np.ndarray, np.ndarray]:
    """GBMA 主入口函数（v2）：Chou 2001 §II 馈源 GB 展开 + 1997 像散反射，
    边缘贡献（边界绕射波 P_d）已内置，默认无经验绕射修正。

    流程：
    1. 六边形格点生成旋转对称高斯波束基 (Chou 2001 §II)：基波束参数
       b_basis 与角间距 Δθ̄ 满足重叠判据 Δθ̄ = √(2/(k·b_basis))；
       展开系数用近场基函数在反射面上最小二乘求解（Step 2.5），
       保证馈源场在反射面上被准确重建
       （旧版以采样代替拟合、且在远场方向图上匹配，展开退化或近场失配）。
    2. 每个波束按 GB 近场规律 (w(z), R(z), Gouy 相位) 从馈源相位中心
       传播至反射面，在反射点做 GO 反射。
    3. 反射场按 PEC 边界条件取 −入射切向场，每个波束的远场 = 其入射
       光斑投影在倾斜表面上的高斯辐射积分闭式结果（含表面 sag 二次
       相位、线性载波与像散相位匹配 1/R_ref = 1/R_inc − 2c/cosθᵢ，
       等价于 Chou & Pathak 1997 反射像散 GB 远场的抛物近似）。
    4. 边缘光斑截断 + 边界绕射波 P_d（Chou-Pathak 1997 过渡函数 T(S̄)
       与边缘项 P_d 的局部直边形式）：方向相关因子
       T(Ŝ) = ½[1 + erf(√c_r·d_r − jk·ũ/(2√c_r))]，
       d_r 为光斑中心沿表面到边缘的带符号距离，ũ 为观察方向相对
       GO 反射方向的投影；ũ=0 为纯截断，erf 尾部即 PO 积分的边缘
       贡献（数值 PO 天然包含的边界绕射波）。
    5. include_diffraction=True 时追加经验 J₁ 边缘绕射修正（默认关闭）。

    Args:
        feed:       FeedAntenna 实例
        cfg:        AntennaConfig 实例
        theta_scan: 扫描角度 [deg]
        bounces:    反射次数（当前仅支持 1）
        include_diffraction: 是否叠加经验边缘绕射修正
        verbose:    打印进度信息
        n_beams:   目标波束数（None=按模式默认自动确定）
        mode:      展开模式：
                     'surface'（默认）密集窄束 + 反射面上近场匹配，
                       宽角精度高（~500 束，约 1-2 s）；
                     'farfield' Chou 2001 §II 原文献链条：由 Ω 经光斑
                       判据 (14)(15) 解 b、(13) 定 Δθ̄、(19) 下限，波束数
                       为产出（~40-60 束）；n_beams 加密时固定 b、只减
                       Δθ̄（论文规则，峰值保持稳定）

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
    P_total = _feed_total_power(feed)

    theta_rad = np.deg2rad(np.asarray(theta_scan, dtype=float))
    N_theta = len(theta_rad)

    # 馈源指向（从焦点到反射面中心）
    focus = np.array([0.0, 0.0, F])
    z_center = (ox**2 + oy**2) / (4.0 * F)
    feed_axis = np.array([ox, oy, z_center]) - focus
    feed_axis /= np.linalg.norm(feed_axis)

    # =========================================================================
    # Step 1: 基波束格点 (gbma_expansion.make_basis_grid)
    # =========================================================================
    dirs, b_basis, delta_theta, cone_half_angle = make_basis_grid(
        feed, F, D, ox, oy, feed_axis, n_beams=n_beams, mode=mode)
    N_beams = len(dirs)
    w0_basis = sqrt(2.0 * b_basis / k)   # 基波束束腰半径
    zR_basis = b_basis                   # 基波束瑞利长度

    # =========================================================================
    # Step 2: 波束追踪 + GO 反射几何 (gbma_reflection.trace_beams)
    # =========================================================================
    tr = trace_beams(dirs, focus, F, D, ox, oy, feed_axis,
                     w0_basis, zR_basis, verbose=verbose)
    P_all, r_all = tr['P_all'], tr['r_all']
    P, r_n = tr['P'], tr['r_n']
    n_hats, cosi = tr['n_hats'], tr['cosi']
    e_t, e_s = tr['e_t'], tr['e_s']
    c_t, c_s = tr['c_t'], tr['c_s']
    h_pol, hit_idx = tr['h_pol'], tr['hit_idx']
    N_hit = tr['N_hit']

    # =========================================================================
    # Step 2.5: 反射面上近场基 LSQ 展开 (gbma_expansion.surface_expansion)
    # =========================================================================
    if mode == 'surface':
        C, _ = surface_expansion(feed, P, r_n, P_all, r_all, dirs, focus,
                                 feed_axis, k, w0_basis, zR_basis, verbose=verbose)
    elif mode == 'farfield':
        C, _ = far_field_expansion(feed, dirs, b_basis, delta_theta,
                                   cone_half_angle, feed_axis, verbose=verbose)
    else:
        raise ValueError(f"unknown mode '{mode}' (choose 'surface' or 'farfield')")
    C_hit = C[hit_idx]

    # =========================================================================
    # Step 3: 入射近场参数与 PO 等效电流 (gbma_reflection.incident_beam_params)
    # =========================================================================
    _, J_m, w_inc, R_inc = incident_beam_params(
        C_hit, r_n, n_hats, h_pol, k, w0_basis, zR_basis,
        spherical=(mode == 'farfield'))

    # =========================================================================
    # Step 4/4.5/5: 闭式光斑远场叠加，含截断与边界绕射波 P_d
    # (gbma_reflection.far_field_sum)
    # =========================================================================
    S = np.zeros((N_theta, 3))
    S[:, 0] = sin(theta_rad)
    S[:, 2] = cos(theta_rad)
    E_total = far_field_sum(P, n_hats, cosi, e_t, e_s, c_t, c_s,
                            w_inc, R_inc, J_m, k, D, ox, oy, S,
                            verbose=verbose)

    # =========================================================================
    # Step 6: 方向性归一化
    # =========================================================================
    U = np.sum(np.abs(E_total)**2, axis=-1)
    D_dir = 4.0 * pi * U / P_total
    dBi = 10.0 * np.log10(np.maximum(D_dir, 1e-15))

    # =========================================================================
    # Step 7: 经验边缘绕射修正（可选，默认关闭，gbma_diffraction）
    # =========================================================================
    if include_diffraction and N_hit > 0:
        if verbose:
            print("  Computing edge diffraction correction (empirical J1 model) ...")
        legacy_gb = GaussianBeam(direction=feed_axis.copy(), origin=focus.copy(),
                                 w0=w0_basis, k=k, b_param=b_basis,
                                 amplitude=complex(np.mean(np.abs(C_hit)), 0.0))
        E_diff = compute_edge_diffraction_field([legacy_gb], D, F, ox, oy, theta_rad)
        theta_abs = np.abs(theta_rad)
        w_diff = 1.0 / (1.0 + np.exp(-(theta_abs - np.deg2rad(18.0))
                                     / (np.deg2rad(2.0) / 3.0)))
        E_total = E_total + (-1j * k / (4.0 * pi)) * E_diff * w_diff[:, None]
        U = np.sum(np.abs(E_total)**2, axis=-1)
        D_dir = 4.0 * pi * U / P_total
        dBi = 10.0 * np.log10(np.maximum(D_dir, 1e-15))

    if verbose:
        print(f"GBMA done! Time: {time.time() - t0:.2f} s, "
              f"Peak directivity: {np.max(dBi):.2f} dBi, "
              f"{N_hit} beams used")

    return E_total, dBi


# =============================================================================
# 辅助：单波束简化模式（用于快速验证，旧版路径）
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
    P_total = _feed_total_power(feed)

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

    # Test far-field (new pipeline, no diffraction)
    print("\n7. Far-field pattern quick test (new pipeline, no diffraction):")
    E2, dBi2 = compute_far_field_gbma(
        feed, cfg_test, theta_test_deg, include_diffraction=False, verbose=True
    )
    print(f"   Peak directivity: {np.max(dBi2):.2f} dBi")

    print("\n=== Self-test complete ===")
