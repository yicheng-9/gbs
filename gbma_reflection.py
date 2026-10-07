"""
gbma_reflection.py — 反射物理（GBMA 第 3 层，Chou & Pathak 1997）

包含：
  - trace_beams: 波束追踪 + GO 反射几何（主流程 Step 2）
  - incident_beam_params: 入射波束近场参数与 PO 等效电流（Step 3）
  - far_field_sum: 每波束闭式光斑远场叠加（Step 4/4.5/5）：
      · 像散相位匹配 γ_t/γ_s（含表面 sag 二次相位）
      · 线性载波 sinθᵢ（方向图峰值对准 GO 反射方向）
      · 边缘光斑截断 + 边界绕射波 P_d（方向相关 erf 过渡因子）
  - reflect_gb_from_paraboloid / compute_single_gb_far_field:
      旧版单波束路径（保留兼容）

物理要点：
  PEC 边界条件下反射场在表面上的切向分量 = −入射切向场，单次反射的
  远场 = 入射光斑在倾斜表面上的闭式辐射积分。总相位（入射波前
  exp(−jkρ²/(2R_inc)) + 表面 sag 二次相位）在表面切平面展开后得到
  像散相位匹配关系 1/R_ref = 1/R_inc − 2c/cosθᵢ 类系数；光斑跨越
  反射面边缘时乘以 1D 直边精确过渡因子
  T(Ŝ) = ½[1 + erf(√c_r·d_r − jk·ũ/(2√c_r))]，
  其中 erf 尾部即 PO 积分自身的边界绕射波 P_d（非 PTD 修正）。
"""

import numpy as np
from numpy import pi, sqrt, sin, cos, arccos, exp
from typing import List, Optional, Tuple
from scipy.special import erf

from gbma_beam import GaussianBeam, compute_h_pol_dir, compute_scalar_gb_far_field
from gbma_geometry import (paraboloid_intersection,
                           paraboloid_principal_curvatures,
                           check_point_on_reflector)


# =============================================================================
# 主流程 Step 2: 波束追踪 + GO 反射几何
# =============================================================================

def trace_beams(dirs: np.ndarray, focus: np.ndarray, F: float, D: float,
                ox: float, oy: float, feed_axis: np.ndarray,
                w0_basis: float, zR_basis: float,
                verbose: bool = True) -> dict:
    """波束追踪至反射面 + GO 反射几何。

    对每条基波束：求与抛物面的交点、GO 反射方向、切向/弧矢基底、
    表面曲率（欧拉公式）。轴落在盘外但光斑仍覆盖边缘的波束也保留
    （其 erf 尾部提供来自盘外的边界绕射波贡献，对应 Chou-Pathak 对
    轴落在反射面解析延拓上的波束的处理）。

    Returns:
        dict:
          P_all, r_all: 全部波束的轴上交点 (N_beams,3) 与光程 (N_beams,)
          P, r_n, n_hats, cosi, e_t, e_s, c_t, c_s, h_pol: 保留波束 (N_hit,)
          hit_idx: 保留波束在原格点中的索引 (N_hit,)
          N_hit:   保留波束数
    """
    try:
        from gbma_fast import trace_beams_fast
        return trace_beams_fast(dirs, focus, F, D, ox, oy, feed_axis,
                                w0_basis, zR_basis, verbose=verbose)
    except ImportError:
        pass

    P_all, r_all = [], []
    P_list, r_list, n_list, cosi_list = [], [], [], []
    et_list, es_list, ct_list, cs_list, h_list, hit_idx = [], [], [], [], [], []
    for idx, s_hat in enumerate(dirs):
        try:
            P_n, r_n, n_hat = paraboloid_intersection(focus, s_hat, F, ox, oy)
        except ValueError:
            P_all.append(np.zeros(3))
            r_all.append(np.inf)
            continue
        P_all.append(P_n)
        r_all.append(r_n)
        if not check_point_on_reflector(P_n, D, ox, oy):
            # 轴在盘外但光斑仍覆盖边缘的波束也保留：
            # 其 erf 尾部即来自盘外的边界绕射波贡献（Chou-Pathak 对
            # 轴落在反射面解析延拓上的波束同样处理）。
            rho_b = sqrt((P_n[0] - ox)**2 + (P_n[1] - oy)**2)
            w_b = w0_basis * sqrt(1.0 + (r_n / zR_basis)**2)
            e_rho_b = np.array([P_n[0] - ox, P_n[1] - oy, 0.0])
            e_rho_b /= max(rho_b, 1e-12)
            tilt_b = sqrt(1.0 - min(np.dot(e_rho_b, n_hat)**2, 1.0))
            d_b = (D / 2.0 - rho_b) / max(tilt_b, 1e-6)
            if d_b <= -2.0 * w_b:
                continue

        # 法向指向入射侧
        cos_theta_i = -np.dot(s_hat, n_hat)
        if cos_theta_i < 0.0:
            n_hat = -n_hat
            cos_theta_i = -cos_theta_i

        # GO 反射方向
        r_dir = s_hat - 2.0 * np.dot(s_hat, n_hat) * n_hat
        r_dir /= np.linalg.norm(r_dir)

        # 切向（入射面内）/ 弧矢向单位矢量（位于表面切平面内）
        e_t = r_dir - np.dot(r_dir, n_hat) * n_hat
        nt = np.linalg.norm(e_t)
        if nt < 1e-12:
            continue
        e_t /= nt
        e_s = np.cross(n_hat, e_t)

        # 表面曲率沿 e_t / e_s（欧拉公式，用于像散相位匹配）
        R1, R2, d1, d2 = paraboloid_principal_curvatures(P_n, F)
        c_t = (np.dot(e_t, d1)**2) / R1 + (np.dot(e_t, d2)**2) / R2
        c_s = (np.dot(e_s, d1)**2) / R1 + (np.dot(e_s, d2)**2) / R2

        P_list.append(P_n)
        r_list.append(r_n)
        n_list.append(n_hat)
        cosi_list.append(cos_theta_i)
        et_list.append(e_t)
        es_list.append(e_s)
        ct_list.append(c_t)
        cs_list.append(c_s)
        h_list.append(compute_h_pol_dir(s_hat, feed_axis))
        hit_idx.append(idx)

    N_hit = len(P_list)
    if N_hit == 0:
        raise RuntimeError("No beams hit the reflector!")
    if verbose:
        print(f"  GBMA: {N_hit}/{len(dirs)} beams hit the reflector")

    return {
        'P_all': np.array(P_all),
        'r_all': np.array(r_all),
        'P': np.array(P_list),
        'r_n': np.array(r_list),
        'n_hats': np.array(n_list),
        'cosi': np.array(cosi_list),
        'e_t': np.array(et_list),
        'e_s': np.array(es_list),
        'c_t': np.array(ct_list),
        'c_s': np.array(cs_list),
        'h_pol': np.array(h_list),
        'hit_idx': np.array(hit_idx, dtype=int),
        'N_hit': N_hit,
    }


# =============================================================================
# 主流程 Step 3: 入射近场参数与 PO 等效电流
# =============================================================================

def incident_beam_params(C_hit: np.ndarray, r_n: np.ndarray,
                         n_hats: np.ndarray, h_pol: np.ndarray,
                         k: float, w0_basis: float, zR_basis: float,
                         spherical: bool = False
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """入射 GB 在反射点的近场参数与 PO 等效电流。

    基波束在轴上距离 r 处：光斑半径 w(z)=w0√(1+(z/zR)²)，
    波前曲率半径 R(z)=z(1+(zR/z)²)，Gouy 相位 ψ=arctan(z/zR)，
    轴上复振幅 ∝ (w0/w)·e^{jψ}·e^{-jkz}。

    spherical=True 时使用点源波束模型（farfield 模式）：
    轴上幅值不含 (w0/w)·e^{jψG} 近场修正、波前半径取球面 r_n，
    与 far_field_expansion 的归一化远场基自洽（该模式下展开系数
    已按"轴上值=1"的远场基拟合，不能再次乘近场修正）。

    Returns:
        (A_m, J_m, w_inc, R_inc)
        A_m:   入射场轴上复振幅 (N_hit,)
        J_m:   PO 等效电流矢量 J = 2n̂×H_inc (N_hit, 3)，复数
        w_inc: 反射点处光斑半径 w(z) (N_hit,)
        R_inc: 反射点处波前曲率半径 (N_hit,)
    """
    w_inc = w0_basis * sqrt(1.0 + (r_n / zR_basis)**2)
    if spherical:
        R_inc = r_n
        A_m = C_hit * exp(-1j * k * r_n) / r_n
    else:
        R_inc = r_n * (1.0 + (zR_basis / r_n)**2)
        gouy = np.arctan(r_n / zR_basis)
        A_m = (C_hit * (w0_basis / w_inc) * exp(1j * gouy)
               * exp(-1j * k * r_n) / r_n)

    # PO 等效电流 J = 2 n̂×H_inc
    J_m = 2.0 * np.cross(n_hats, A_m[:, None] * h_pol)
    return A_m, J_m, w_inc, R_inc


# =============================================================================
# 主流程 Step 4/4.5/5: 每波束闭式光斑远场叠加
# =============================================================================

def far_field_sum(P: np.ndarray, n_hats: np.ndarray, cosi: np.ndarray,
                  e_t: np.ndarray, e_s: np.ndarray,
                  c_t: np.ndarray, c_s: np.ndarray,
                  w_inc: np.ndarray, R_inc: np.ndarray,
                  J_m: np.ndarray, k: float, D: float,
                  ox: float, oy: float, S: np.ndarray,
                  verbose: bool = True) -> np.ndarray:
    """每波束闭式光斑远场叠加（含截断与边界绕射波 P_d）。

    PEC 边界条件：反射场在表面上的切向分量 = −入射切向场，单次反射的
    远场 = 入射光斑在倾斜表面上的闭式辐射积分。把总相位（入射波前
    exp(−jkρ²/(2R_inc)) + 表面 sag 的二次相位）在表面切平面展开：
      c_ξ = cos²θᵢ/w² · γ_t，  γ_t = 1 + j·(kw²/2)·(1/R_inc − 2c_t/cosθᵢ)
      c_η = 1/w² · γ_s，        γ_s = 1 + j·(kw²/2)·(1/R_inc − 2c_s·cosθᵢ)
    （即像散相位匹配 1/R_ref = 1/R_inc − 2c/cosθᵢ 类关系，sag 的
      Ŝ·n̂ 项在 GO 方向冻结为 cosθᵢ——主波束方向为精确值）
    另有线性相位载波 e^{-jk·ξ·sinθᵢ}（ŝ·ê_t = r̂·ê_t = sinθᵢ），把
    每个反射波束的方向图峰值移到 GO 反射方向 r̂。
    闭式结果:
      I(ŝ) = (πw²/cosθᵢ)/√(γ_t·γ_s) ·
             exp(−(k²w²/4)·((u_t−sinθᵢ)²/(cos²θᵢ·γ_t) + u_s²/γ_s))

    边缘光斑截断 + 边界绕射波（直边精确解 + 曲边首阶修正）：
      T(Ŝ) = ½[1 + erf(√c_r·d_r − jk·ũ/(2√c_r))] − S(Ŝ)/I_full
      d_r 为光斑中心沿表面到边缘的带符号距离（向内为正），
      ũ = Ŝ·ê_r − sinθᵢ·cosφ_r 为观察方向相对 GO 反射方向在 ê_r 上
      的投影。erf 尾部即 PO 积分自身的边缘贡献 P_d（数值 PO 天然
      包含的边界绕射波，非 PTD 修正）；S/I_full 为真实曲边（切平面
      内抛物线 s_bound=d_r−κη²/2）与切线之间的月牙区积分闭式
      （Chou-Pathak 1997 附录 A/B 的首阶形式），恢复边缘曲率的
      首阶影响，直边模型是其 κ→0 极限。

    Args:
        S: 观察方向单位矢量 (N_theta, 3)

    Returns:
        E_total: 远场 E 矢量 (N_theta, 3)，复数
    """
    gam_t = 1.0 + 1j * (k * w_inc**2 / 2.0) * (1.0 / R_inc - 2.0 * c_t / np.maximum(cosi, 1e-6))
    gam_s = 1.0 + 1j * (k * w_inc**2 / 2.0) * (1.0 / R_inc - 2.0 * c_s * cosi)
    sin_i = sqrt(1.0 - cosi**2)                        # sinθᵢ per beam

    u_t = S @ e_t.T
    u_s = S @ e_s.T
    arg = (-(k**2 / 4.0) * w_inc[None, :]**2
           * ((u_t - sin_i[None, :])**2 / (np.maximum(cosi[None, :], 1e-6)**2 * gam_t[None, :])
              + u_s**2 / gam_s[None, :]))
    F_pat = np.exp(arg) / np.sqrt(gam_t[None, :] * gam_s[None, :])

    # ---- 边缘光斑截断几何（局部直边近似）----
    a_rim = D / 2.0
    rho_n = sqrt((P[:, 0] - ox)**2 + (P[:, 1] - oy)**2)
    e_rho = np.zeros_like(P)
    inside = rho_n > 1e-9
    e_rho[inside, 0] = (P[inside, 0] - ox) / rho_n[inside]
    e_rho[inside, 1] = (P[inside, 1] - oy) / rho_n[inside]
    e_rho_dot_n = np.sum(e_rho * n_hats, axis=1)
    v_r = e_rho - e_rho_dot_n[:, None] * n_hats
    tilt = np.linalg.norm(v_r, axis=1)
    e_r = np.where(tilt[:, None] > 1e-9,
                   v_r / np.maximum(tilt, 1e-9)[:, None], e_t)
    d_r = (a_rim - rho_n) / np.maximum(tilt, 1e-6)

    cos_phi_r = np.sum(e_r * e_t, axis=1)
    sin_phi_r = np.sum(e_r * e_s, axis=1)
    c_r = (cos_phi_r**2 * (cosi**2 / w_inc**2) * gam_t
           + sin_phi_r**2 * (1.0 / w_inc**2) * gam_s)
    sqrt_c_r = sqrt(c_r + 0j)

    # 轴上的纯截断因子（ũ=0），用于统计被截断的波束数
    with np.errstate(all='ignore'):
        T_r_axis = 0.5 * (1.0 + erf(sqrt_c_r * d_r))
    if verbose:
        n_trunc = int(np.sum(np.abs(T_r_axis) < 0.999))
        print(f"  Edge truncation: {n_trunc}/{len(P)} beams truncated")

    # ---- 方向相关的截断/边界绕射波因子 ----
    #   ∫_{-∞}^{d} e^{-c·s²}·e^{jk·ũ·s} ds / ∫_{-∞}^{∞} e^{-c·s²}·e^{jk·ũ·s} ds
    #     = ½[1 + erf(√c·d − jk·ũ/(2√c))]
    u_r = S @ e_r.T - (sin_i * cos_phi_r)[None, :]     # (N_theta, N_hit)
    with np.errstate(all='ignore'):
        T_r = 0.5 * (1.0 + erf(sqrt_c_r[None, :] * d_r[None, :]
                                - 1j * k * u_r / (2.0 * sqrt_c_r[None, :])))
    # 数值保护：大扫描角（如 ±90°）下 erf 复宗量虚部很大，erf 溢出为
    # inf/nan；此时高斯方向图因子 F_pat 已下溢为 0，乘积物理上为 0，
    # 把非有限的 T_r 置 0，避免 0×inf=NaN 污染整个方向图。
    T_r = np.where(np.isfinite(T_r), T_r, 0.0)

    # ---- 曲边截断修正（Chou-Pathak 1997 附录 A/B 的首阶形式）----
    # 真实边缘在切平面内是抛物线 s_bound(η) = d_r − κ·η²/2，
    # κ = 1/(a·tilt)（投影圆在倾斜切平面内的曲率）；切线模型多算了
    # 边界与真实圆之间的"月牙区"，其积分在薄月牙近似下闭式：
    #   S/I_full = (κ/2√π)·√c_r·e^{−c_r d_r² + jkũ_r d_r + (kũ_r)²/(4c_r)}
    #              ·[1/(2c_φ) − k²ũ_φ²/(4c_φ²)]
    # c_φ 为光斑沿边缘方向 ê_φ 的复二次系数，ũ_φ 为观察方向相对 GO
    # 反射方向沿 ê_φ 的投影。该修正恢复边缘曲率对边界波的首阶影响
    # （沿边缘相位变化的驻相效应首阶），直边模型是它的 κ→0 极限。
    e_phi = np.cross(n_hats, e_r)                       # 沿边缘方向（切平面内 ⊥ e_r）
    cos_phi_phi = np.sum(e_phi * e_t, axis=1)
    u_phi = S @ e_phi.T - (sin_i * cos_phi_phi)[None, :]
    c_phi = (cos_phi_phi**2 * (cosi**2 / w_inc**2) * gam_t
             + (1.0 - cos_phi_phi**2) * (1.0 / w_inc**2) * gam_s)
    kappa = 1.0 / (a_rim * np.maximum(tilt, 1e-6))
    mom2 = 1.0 / (2.0 * c_phi) - (k**2) * u_phi**2 / (4.0 * c_phi**2)
    with np.errstate(all='ignore'):
        E_slv = (-c_r[None, :] * d_r[None, :]**2
                 + 1j * k * u_r * d_r[None, :]
                 + (k * u_r)**2 / (4.0 * c_r[None, :]))
        S_ratio = ((kappa[None, :] / (2.0 * sqrt(pi))) * sqrt_c_r[None, :]
                   * np.exp(E_slv) * mom2)
        T_r = T_r - S_ratio
        T_r = np.where(np.isfinite(T_r), T_r, 0.0)

    # ---- 相干叠加 ----
    s_dot_J = S @ J_m.T
    term = S[:, None, :] * s_dot_J[:, :, None] - J_m[None, :, :]
    phase = np.exp(1j * k * (S @ P.T))
    dA = pi * w_inc**2 / np.maximum(cosi, 1e-6)
    E_total = (-1j * k / (4.0 * pi)) * np.sum(
        term * phase[:, :, None] * dA[None, :, None] * F_pat[:, :, None]
        * T_r[:, :, None],
        axis=1)
    return E_total


# =============================================================================
# 旧版：单波束 GO 反射（保留兼容）
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
# 旧版：单波束远场（保留兼容）
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
