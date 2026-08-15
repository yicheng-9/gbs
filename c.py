import numpy as np
import time
from dataclasses import dataclass
from scipy.optimize import brentq
from feed_antenna import FeedAntenna
from mesh_generator import generate_parabolic_mesh, compute_incident_H
from po_integrator_constant import POIntegrator
from gbma_solver import compute_far_field_gbma

# ========== 辅助函数 ==========
def compute_feed_total_power(feed):
    """计算馈源总辐射功率（用于方向性归一化）"""
    x = feed.k * feed.b
    P_total = (np.pi / (8.0 * x**3)) * (8.0*x**2 - 4.0*x + 1.0 - np.exp(-4.0*x))
    return P_total

def compute_theta_b_from_feed(feed, tol_deg=0.01, max_deg=90):
    """根据馈源远场幅度方向图计算半功率波束宽度 θ_b（度）"""
    target = 1.0 / np.sqrt(2.0)
    theta_test = np.linspace(0, max_deg, 500)
    amp_test = feed.far_field_amplitude(np.deg2rad(theta_test))
    idx = np.where(amp_test <= target)[0]
    if len(idx) == 0:
        return max_deg, np.deg2rad(max_deg)
    idx0 = idx[0]
    if idx0 == 0:
        return compute_theta_b_from_feed(feed, tol_deg, max_deg*2)
    a_deg = theta_test[idx0-1]
    b_deg = theta_test[idx0]
    def func(deg):
        return feed.far_field_amplitude(np.deg2rad(deg)) - target
    try:
        theta_b_deg = brentq(func, a_deg, b_deg, xtol=tol_deg/100)
    except ValueError:
        theta_b_deg = (a_deg + b_deg) / 2.0
    return theta_b_deg, np.deg2rad(theta_b_deg)

def compute_theta_0(D, F, h):
    """根据论文公式(2)计算反射器边缘半张角 θ0（弧度）"""
    term1 = (D + h) / (F - (D + h)**2 / (4 * F))
    term2 = h / (F - h**2 / (4 * F))
    theta_0 = 0.5 * (np.arctan(term1) - np.arctan(term2))
    return theta_0

def save_cut_file(file_path, theta_deg, gain_dbi, phi_deg=0.0):
    """
    保存方向图数据为标准 GRASP .cut 格式（单一切面）
    :param file_path: 输出文件名
    :param theta_deg: 角度数组（度，等间隔）
    :param gain_dbi: 增益数组（dBi）
    :param phi_deg: 切面角度（度）
    """
    # 确保角度为升序
    if theta_deg[0] > theta_deg[-1]:
        theta_deg = theta_deg[::-1]
        gain_dbi = gain_dbi[::-1]

    start_angle = theta_deg[0]
    step = theta_deg[1] - theta_deg[0]
    num_points = len(theta_deg)

    # 将增益转换为场强幅度（线性值）
    linear_gain = 10 ** (gain_dbi / 10.0)
    # 假设 Ex 和 Ey 各占一半功率，实部为幅度，虚部为 0
    e_mag = np.sqrt(linear_gain / 2.0)

    with open(file_path, 'w') as f:
        f.write("Field data in cuts\n")
        f.write(f"{start_angle:.6f} {step:.6f} {num_points} {phi_deg}\n")
        for mag in e_mag:
            f.write(f"{mag:.8e} 0.0 {mag:.8e} 0.0\n")

# ========== 参数配置 ==========
@dataclass
class AntennaConfig:
    freq: float = 12e9
    F: float = 0.6
    D: float = 1.0
    offset_x: float = 0.0     # 偏馈：投影圆心位于 (0.5, 0, 0)
    offset_y: float = 0.0
    edge_angle_deg: float = 45.0
    edge_taper_db: float = -12.0
    theta0_max_deg: float = 20.0
    theta_scan_deg: tuple = (-10, 10, 361)

    @property
    def lam(self):
        return 3e8 / self.freq

    @property
    def k0(self):
        return 2 * np.pi / self.lam

# ========== 计算并保存 PO 和 GBMA 结果 ==========
def compute_and_save_po_gbma(cfg: AntennaConfig):
    """执行 PO 常数近似和 GBMA 计算，保存结果到 .cut 文件"""
    feed = FeedAntenna(cfg.freq, edge_angle_deg=cfg.edge_angle_deg,
                       edge_taper_db=cfg.edge_taper_db)
    P_feed_total = compute_feed_total_power(feed)

    # 反射体网格生成
    theta0_max_rad = np.deg2rad(cfg.theta0_max_deg)
    z = 1.09 * (np.pi * cfg.D / cfg.lam) * np.sin(theta0_max_rad) + 10.0
    po2_min = int(np.round(z))
    po1_min = int(np.round(z / 2.4))
    tri_verts, tri_norms, tri_areas = generate_parabolic_mesh(
        cfg.F, cfg.D, N_radial=po1_min, N_azimuth=po2_min,
        offset_x=cfg.offset_x, offset_y=cfg.offset_y
    )
    print(f"网格: N_radial={po1_min}, N_azimuth={po2_min}")

    # 馈源指向
    focus = np.array([0.0, 0.0, cfg.F])
    z_center = (cfg.offset_x**2 + cfg.offset_y**2) / (4 * cfg.F)
    center_on_reflector = np.array([cfg.offset_x, cfg.offset_y, z_center])
    feed_axis = center_on_reflector - focus
    feed_axis /= np.linalg.norm(feed_axis)

    # 方向图扫描角度
    theta_start, theta_stop, num_theta = cfg.theta_scan_deg
    theta_scan = np.linspace(theta_start, theta_stop, num_theta)

    # ---------- PO 常数近似 ----------
    print("计算 PO 常数近似 ...")
    t0 = time.time()
    positions_c = np.mean(tri_verts, axis=1)
    norms_c = np.mean(tri_norms, axis=1)
    norms_c /= np.linalg.norm(norms_c, axis=1, keepdims=True)
    H_inc_c = compute_incident_H(positions_c, feed, cfg.F, feed_axis)
    po_c = POIntegrator(positions_c, norms_c, tri_areas, feed.k)

    E_c  = po_c.far_field_pattern(H_inc_c, theta_scan, phi_deg=0.0)
    U_c  = np.sum(np.abs(E_c)**2, axis=-1)
    D_c  = 4 * np.pi * U_c / P_feed_total
    dBi_po = 10 * np.log10(D_c + 1e-15)
    print(f"{time.time()-t0:.2f}s")

    print(f"PO 峰值方向性: {np.max(dBi_po):.2f} dBi, 耗时 {time.time()-t0:.2f}s")

    # ---------- GBMA ----------
    print("计算 GBMA ...")
    t0 = time.time()
    E_gbma, dBi_gbma = compute_far_field_gbma(
        feed, cfg, theta_scan,
        bounces=1,
        include_diffraction=False,
        n_beams=180,
        gabor_mode='angular'
    )
    print(f"{time.time()-t0:.2f}s")
    print(f"GBMA 峰值方向性: {np.max(dBi_gbma):.2f} dBi, 耗时 {time.time()-t0:.2f}s")

    # 保存为 .cut 文件
    save_cut_file("PO.cut", theta_scan, dBi_po, phi_deg=0.0)
    save_cut_file("gbma.cut", theta_scan, dBi_gbma, phi_deg=0.0)
    print("结果已保存为 PO.cut 和 gbma.cut")

if __name__ == "__main__":
    config = AntennaConfig()
    # 可在此修改参数，例如：
    # config.freq = 14e9
    # config.D = 1.5
    # config.offset_x = 0.5
    # config.edge_taper_db = -15
    compute_and_save_po_gbma(config)
