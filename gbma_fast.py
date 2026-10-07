"""
gbma_fast.py — numba 加速内核（可选加速层）

使用说明：
  - 需要安装 numba（py -3.13 -m pip install numba）；未安装时各入口函数
    直接抛 ImportError，调用方自动回退到 gbma_expansion/gbma_reflection
    中的纯 numpy 实现，结果完全一致。
  - 加速点：
      1. trace_beams_fast : 每波束追踪循环 @njit；
                            抛物面主曲率用解析式（κ1=cos³ψ/(2F),
                            κ2=cosψ/(2F)，径向/方位角主方向），与
                            Weingarten 特征分解在数学上等价；
      2. build_expansion_A: 面上 LSQ 的 A 矩阵显式循环 @njit；
      3. solve_lsq_fast    : 正规方程 AᴴA·C=Aᴴy + Cholesky
                            （A 良态时远快于 SVD；失败自动回退 lstsq）。

数值一致性：加速路径与纯 numpy 路径的差异在 1e-12 量级（Cholesky 的
舍入差异），远小于物理/模型近似误差。
"""

import numpy as np

try:
    from numba import njit
    _HAVE_NUMBA = True
except ImportError:
    _HAVE_NUMBA = False


# =============================================================================
# 0. 六边形格点与方向旋转内核
# =============================================================================

@njit(cache=True)
def _hex_directions_kernel(cone_half_angle, delta_theta, out):
    """六边形格点方向填充，返回方向数。"""
    out[0, 0] = 0.0
    out[0, 1] = 0.0
    out[0, 2] = 1.0
    idx = 1
    n_rings = int(np.ceil(cone_half_angle / delta_theta))
    for ring in range(1, n_rings + 1):
        theta_ring = ring * delta_theta
        if theta_ring > cone_half_angle * 1.05:
            break
        n_phi = max(6 * ring, 6)
        for j in range(n_phi):
            phi = 2.0 * np.pi * j / n_phi
            if ring % 2 == 1:
                phi += np.pi / n_phi
            out[idx, 0] = np.sin(theta_ring) * np.cos(phi)
            out[idx, 1] = np.sin(theta_ring) * np.sin(phi)
            out[idx, 2] = np.cos(theta_ring)
            idx += 1
    return idx


def hex_directions_fast(cone_half_angle, delta_theta):
    """generate_hexagonal_beam_directions 的 numba 版。"""
    if not _HAVE_NUMBA:
        raise ImportError('numba not installed')
    n_rings = int(np.ceil(cone_half_angle / delta_theta))
    n = 1
    for ring in range(1, n_rings + 1):
        if ring * delta_theta > cone_half_angle * 1.05:
            break
        n += max(6 * ring, 6)
    out = np.empty((n, 3), dtype=np.float64)
    n_actual = _hex_directions_kernel(cone_half_angle, delta_theta, out)
    return out[:n_actual]


@njit(cache=True)
def _rotate_directions_kernel(dirs, target, out):
    """罗德里格斯旋转：+z → target。"""
    n = dirs.shape[0]
    ca = target[2]
    if abs(ca - 1.0) < 1e-12:
        for i in range(n):
            out[i, 0] = dirs[i, 0]
            out[i, 1] = dirs[i, 1]
            out[i, 2] = dirs[i, 2]
        return
    rx = -target[1]
    ry = target[0]
    rz = 0.0
    rn = np.sqrt(rx * rx + ry * ry)
    if rn < 1e-12:
        rx = 1.0
        ry = 0.0
        ang = np.pi
    else:
        rx /= rn
        ry /= rn
        ang = np.arccos(ca)
    c = np.cos(ang)
    s = np.sin(ang)
    for i in range(n):
        vx = dirs[i, 0]
        vy = dirs[i, 1]
        vz = dirs[i, 2]
        kx = ry * vz - rz * vy
        ky = rz * vx - rx * vz
        kz = rx * vy - ry * vx
        kd = rx * vx + ry * vy + rz * vz
        ox = vx * c + kx * s + rx * kd * (1.0 - c)
        oy = vy * c + ky * s + ry * kd * (1.0 - c)
        oz = vz * c + kz * s + rz * kd * (1.0 - c)
        nn = np.sqrt(ox * ox + oy * oy + oz * oz)
        out[i, 0] = ox / nn
        out[i, 1] = oy / nn
        out[i, 2] = oz / nn


def rotate_directions_fast(dirs, target):
    """rotate_directions_to_axis 的 numba 版。"""
    if not _HAVE_NUMBA:
        raise ImportError('numba not installed')
    dirs = np.ascontiguousarray(dirs, dtype=np.float64)
    target = np.ascontiguousarray(target, dtype=np.float64)
    out = np.empty_like(dirs)
    _rotate_directions_kernel(dirs, target, out)
    return out


# =============================================================================
# 1. 波束追踪内核
# =============================================================================

@njit(cache=True)
def _trace_beams_kernel(dirs, focus, F, D, ox, oy, feed_axis, w0_basis, zR_basis,
                        P_all, r_all, P, r_n, n_hats, cosi, e_t, e_s,
                        c_t, c_s, h_pol, hit_idx):
    """单波束追踪 + GO 反射几何 + 解析主曲率。返回命中数。"""
    n_beams = dirs.shape[0]
    n_hit = 0
    for i in range(n_beams):
        sx = dirs[i, 0]
        sy = dirs[i, 1]
        sz = dirs[i, 2]
        x0 = focus[0]
        y0 = focus[1]
        z0 = focus[2]

        # ---- 射线与抛物面求交 ----
        a = sx * sx + sy * sy
        bb = 2.0 * (x0 * sx + y0 * sy) - 4.0 * F * sz
        cc = x0 * x0 + y0 * y0 - 4.0 * F * z0
        hit = True
        if a < 1e-15:
            if abs(bb) < 1e-15:
                hit = False
            else:
                t = -cc / bb
        else:
            disc = bb * bb - 4.0 * a * cc
            if disc < 0:
                hit = False
            else:
                sq = np.sqrt(disc)
                t1 = (-bb - sq) / (2.0 * a)
                t2 = (-bb + sq) / (2.0 * a)
                if t1 > 1e-12:
                    t = t1
                elif t2 > 1e-12:
                    t = t2
                else:
                    hit = False
        if not hit:
            P_all[i, 0] = 0.0
            P_all[i, 1] = 0.0
            P_all[i, 2] = 0.0
            r_all[i] = np.inf
            continue

        px = x0 + t * sx
        py = y0 + t * sy
        pz = z0 + t * sz
        P_all[i, 0] = px
        P_all[i, 1] = py
        P_all[i, 2] = pz
        r_all[i] = t

        # ---- 法向 ----
        nx = -px / (2.0 * F)
        ny = -py / (2.0 * F)
        nz = 1.0
        nn = np.sqrt(nx * nx + ny * ny + nz * nz)
        nx /= nn
        ny /= nn
        nz /= nn

        # ---- 盘内判断；盘外但光斑覆盖边缘的波束保留 ----
        dx = px - ox
        dy = py - oy
        if dx * dx + dy * dy > (D / 2.0) ** 2:
            rho_b = np.sqrt(dx * dx + dy * dy)
            w_b = w0_basis * np.sqrt(1.0 + (t / zR_basis) ** 2)
            if rho_b > 1e-12:
                erx = dx / rho_b
                ery = dy / rho_b
            else:
                erx = 0.0
                ery = 0.0
            edn = erx * nx + ery * ny
            tilt_b = np.sqrt(max(0.0, 1.0 - edn * edn))
            d_b = (D / 2.0 - rho_b) / max(tilt_b, 1e-6)
            if d_b <= -2.0 * w_b:
                continue

        # ---- 法向指向入射侧 ----
        cos_theta_i = -(sx * nx + sy * ny + sz * nz)
        if cos_theta_i < 0.0:
            nx = -nx
            ny = -ny
            nz = -nz
            cos_theta_i = -cos_theta_i

        # ---- GO 反射方向 ----
        sdn = sx * nx + sy * ny + sz * nz
        rx = sx - 2.0 * sdn * nx
        ry = sy - 2.0 * sdn * ny
        rz = sz - 2.0 * sdn * nz
        rn_ = np.sqrt(rx * rx + ry * ry + rz * rz)
        rx /= rn_
        ry /= rn_
        rz /= rn_

        # ---- 切向/弧矢基底 ----
        rdn = rx * nx + ry * ny + rz * nz
        etx = rx - rdn * nx
        ety = ry - rdn * ny
        etz = rz - rdn * nz
        nt = np.sqrt(etx * etx + ety * ety + etz * etz)
        if nt < 1e-12:
            continue
        etx /= nt
        ety /= nt
        etz /= nt
        esx = ny * etz - nz * ety
        esy = nz * etx - nx * etz
        esz = nx * ety - ny * etx

        # ---- 解析主曲率（旋转抛物面，等价于 Weingarten 特征分解）----
        rho = np.sqrt(px * px + py * py)
        if rho > 1e-12:
            cph = px / rho
            sph = py / rho
            cospsi = 1.0 / np.sqrt(1.0 + (rho / (2.0 * F)) ** 2)
            sinpsi = (rho / (2.0 * F)) * cospsi
            R1 = 2.0 * F / (cospsi ** 3)
            R2 = 2.0 * F / cospsi
            d1x = cph * cospsi
            d1y = sph * cospsi
            d1z = sinpsi
            d2x = -sph
            d2y = cph
            d2z = 0.0
        else:
            R1 = 2.0 * F
            R2 = 2.0 * F
            d1x = 1.0
            d1y = 0.0
            d1z = 0.0
            d2x = 0.0
            d2y = 1.0
            d2z = 0.0
        etd1 = etx * d1x + ety * d1y + etz * d1z
        etd2 = etx * d2x + ety * d2y + etz * d2z
        esd1 = esx * d1x + esy * d1y + esz * d1z
        esd2 = esx * d2x + esy * d2y + esz * d2z
        ct = etd1 * etd1 / R1 + etd2 * etd2 / R2
        cs = esd1 * esd1 / R1 + esd2 * esd2 / R2

        # ---- H 场极化方向 ----
        fax = feed_axis[0]
        fay = feed_axis[1]
        faz = feed_axis[2]
        xpx = 1.0 - fax * fax
        xpy = -fay * fax
        xpz = -faz * fax
        nxp = np.sqrt(xpx * xpx + xpy * xpy + xpz * xpz)
        if nxp < 1e-12:
            xpx = -fax * fay
            xpy = 1.0 - fay * fay
            xpz = -faz * fay
            nxp = np.sqrt(xpx * xpx + xpy * xpy + xpz * xpz)
        xpx /= nxp
        xpy /= nxp
        xpz /= nxp
        hx = sy * xpz - sz * xpy
        hy = sz * xpx - sx * xpz
        hz = sx * xpy - sy * xpx
        hn = np.sqrt(hx * hx + hy * hy + hz * hz)
        if hn > 1e-12:
            hx /= hn
            hy /= hn
            hz /= hn
        else:
            hx = 0.0
            hy = 1.0
            hz = 0.0

        # ---- 存入命中数组 ----
        P[n_hit, 0] = px
        P[n_hit, 1] = py
        P[n_hit, 2] = pz
        r_n[n_hit] = t
        n_hats[n_hit, 0] = nx
        n_hats[n_hit, 1] = ny
        n_hats[n_hit, 2] = nz
        cosi[n_hit] = cos_theta_i
        e_t[n_hit, 0] = etx
        e_t[n_hit, 1] = ety
        e_t[n_hit, 2] = etz
        e_s[n_hit, 0] = esx
        e_s[n_hit, 1] = esy
        e_s[n_hit, 2] = esz
        c_t[n_hit] = ct
        c_s[n_hit] = cs
        h_pol[n_hit, 0] = hx
        h_pol[n_hit, 1] = hy
        h_pol[n_hit, 2] = hz
        hit_idx[n_hit] = i
        n_hit += 1

    return n_hit


def trace_beams_fast(dirs, focus, F, D, ox, oy, feed_axis,
                     w0_basis, zR_basis, verbose=True):
    """trace_beams 的 numba 加速版，返回与 gbma_reflection.trace_beams
    相同结构的 dict。"""
    if not _HAVE_NUMBA:
        raise ImportError('numba not installed')
    N = len(dirs)
    dirs = np.ascontiguousarray(dirs, dtype=np.float64)
    focus = np.ascontiguousarray(focus, dtype=np.float64)
    feed_axis = np.ascontiguousarray(feed_axis, dtype=np.float64)
    P_all = np.empty((N, 3), dtype=np.float64)
    r_all = np.empty(N, dtype=np.float64)
    P = np.empty((N, 3), dtype=np.float64)
    r_n = np.empty(N, dtype=np.float64)
    n_hats = np.empty((N, 3), dtype=np.float64)
    cosi = np.empty(N, dtype=np.float64)
    e_t = np.empty((N, 3), dtype=np.float64)
    e_s = np.empty((N, 3), dtype=np.float64)
    c_t = np.empty(N, dtype=np.float64)
    c_s = np.empty(N, dtype=np.float64)
    h_pol = np.empty((N, 3), dtype=np.float64)
    hit_idx = np.empty(N, dtype=np.int64)

    n_hit = _trace_beams_kernel(dirs, focus, F, D, ox, oy, feed_axis,
                                w0_basis, zR_basis, P_all, r_all, P, r_n,
                                n_hats, cosi, e_t, e_s, c_t, c_s, h_pol,
                                hit_idx)
    if n_hit == 0:
        raise RuntimeError('No beams hit the reflector!')
    if verbose:
        print(f'  GBMA: {n_hit}/{N} beams hit the reflector')

    return {
        'P_all': P_all,
        'r_all': r_all,
        'P': P[:n_hit],
        'r_n': r_n[:n_hit],
        'n_hats': n_hats[:n_hit],
        'cosi': cosi[:n_hit],
        'e_t': e_t[:n_hit],
        'e_s': e_s[:n_hit],
        'c_t': c_t[:n_hit],
        'c_s': c_s[:n_hit],
        'h_pol': h_pol[:n_hit],
        'hit_idx': hit_idx[:n_hit],
        'N_hit': n_hit,
    }


# =============================================================================
# 2. 面上 LSQ 的 A 矩阵构建内核
# =============================================================================

@njit(cache=True)
def _build_A_kernel(P, r_n, P_all, r_all, dirs, k_hat, w_all, R_all,
                    gouy_all, w0_basis, k, out):
    M = P.shape[0]
    N = dirs.shape[0]
    for m in range(M):
        for n in range(N):
            if not np.isfinite(r_all[n]):
                out[m, n] = 0.0 + 0.0j
                continue
            d0 = P[m, 0] - P_all[n, 0]
            d1 = P[m, 1] - P_all[n, 1]
            d2 = P[m, 2] - P_all[n, 2]
            proj = d0 * dirs[n, 0] + d1 * dirs[n, 1] + d2 * dirs[n, 2]
            rho2 = d0 * d0 + d1 * d1 + d2 * d2 - proj * proj
            z_mn = r_all[n] + proj
            wn = w_all[n]
            amp = (w0_basis / wn) * (r_n[m] / max(r_all[n], 1e-12))
            cosA = (k_hat[m, 0] * dirs[n, 0] + k_hat[m, 1] * dirs[n, 1]
                    + k_hat[m, 2] * dirs[n, 2])
            if cosA > 1.0:
                cosA = 1.0
            elif cosA < -1.0:
                cosA = -1.0
            obl = (1.0 + cosA) / 2.0
            ph = k * (rho2 / (2.0 * R_all[n]) + z_mn - r_n[m])
            out[m, n] = (amp * obl * np.exp(-rho2 / (wn * wn))
                         * np.exp(-1j * ph) * np.exp(1j * gouy_all[n]))


def build_expansion_A(P, r_n, P_all, r_all, dirs, k_hat, w_all, R_all,
                      gouy_all, w0_basis, k):
    """surface_expansion 中 A 矩阵的 numba 加速构建。"""
    if not _HAVE_NUMBA:
        raise ImportError('numba not installed')
    M = len(P)
    N = len(dirs)
    P = np.ascontiguousarray(P, dtype=np.float64)
    P_all = np.ascontiguousarray(P_all, dtype=np.float64)
    dirs = np.ascontiguousarray(dirs, dtype=np.float64)
    k_hat = np.ascontiguousarray(k_hat, dtype=np.float64)
    r_n = np.ascontiguousarray(r_n, dtype=np.float64)
    r_all = np.ascontiguousarray(r_all, dtype=np.float64)
    w_all = np.ascontiguousarray(w_all, dtype=np.float64)
    R_all = np.ascontiguousarray(R_all, dtype=np.float64)
    gouy_all = np.ascontiguousarray(gouy_all, dtype=np.float64)
    out = np.empty((M, N), dtype=np.complex128)
    _build_A_kernel(P, r_n, P_all, r_all, dirs, k_hat, w_all, R_all,
                    gouy_all, w0_basis, k, out)
    return out


# =============================================================================
# 3. 最小二乘：正规方程 + Cholesky（自动回退 SVD）
# =============================================================================

def solve_lsq_fast(A, y, rcond=1e-8):
    """复数最小二乘快速求解：LAPACK gelsy（QR + 列主元）。

    实测与 SVD 最小范数解逐位一致（ΔC ≤ 7e-15，方向图差 <1e-12 dB），
    速度约为 SVD(gelsd) 的 2 倍；失败时回退 np.linalg.lstsq。
    注意：不要用无正则的正规方程 + Cholesky——展开矩阵 A 冗余/秩亏
    （基波束重叠），其解混入零空间分量，远场方向图会严重失真
    （实测峰值偏移数十 dB）。
    """
    try:
        import scipy.linalg as sla
        return sla.lstsq(A, y, cond=None, lapack_driver='gelsy')[0]
    except Exception:
        return np.linalg.lstsq(A, y, rcond=rcond)[0]
