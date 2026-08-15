import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from feed_antenna import FeedAntenna
from mesh_generator import generate_parabolic_mesh, compute_incident_H
from po_integrator_constant import POIntegrator
from gbma_solver import compute_far_field_gbma
import time
from dataclasses import dataclass

# ========== 开关：要使用哪种方法？(True/False) ==========
USE_CONSTANT_PO = True
USE_GB_FORMULA = False   # 高斯波束经验公式估算
USE_GBMA      = True    # GBMA 高斯波束模式分析法（真远场叠加 + 边缘绕射）
USE_CUT_FILE  = False   # 从 single_cut.cut 导入仿真数据做对比（本轮只与 PO 对比）

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

def gaussian_beam_relative_pattern(theta_deg, theta_3_deg, SLL_dB, theta_SLL_deg=None):
    """根据论文公式(8)生成高斯波束方向图（相对值，峰值=0 dB）"""
    theta = np.asarray(theta_deg)
    theta_3 = theta_3_deg
    if theta_SLL_deg is None:
        theta_SLL = 1.2 * theta_3
    else:
        theta_SLL = theta_SLL_deg
    g_rel = np.zeros_like(theta)
    mask1 = (theta >= 0) & (theta < theta_SLL)
    g_rel[mask1] = -3 * (2 * theta[mask1] / theta_3)**2
    mask2 = (theta >= theta_SLL) & (theta < 1.5 * theta_SLL)
    g_rel[mask2] = SLL_dB
    mask3 = theta >= 1.5 * theta_SLL
    g_rel[mask3] = SLL_dB - 25 * np.log10(theta[mask3] / (1.5 * theta_SLL))
    return g_rel

def load_cut_file(file_path='single_cut.cut'):
    """从 .cut 文件中读取远场方向图数据（φ=0°, 45°, 90° 切面）
    返回: list[dict], 每个 dict 包含 'theta', 'gain_dbi', 'phi'"""
    import math
    with open(file_path, 'r') as f:
        lines = f.readlines()

    cuts = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('Field data in cuts'):
            i += 1
            param_line = lines[i].split()
            start_angle = float(param_line[0])
            step = float(param_line[1])
            num_points = int(param_line[2])
            phi_angle = float(param_line[3])

            theta = np.linspace(start_angle, start_angle + (num_points - 1) * step, num_points)

            data = []
            for j in range(i + 1, i + 1 + num_points):
                values = lines[j].split()
                if len(values) >= 4:
                    ex_real = float(values[0])
                    ex_imag = float(values[1])
                    ey_real = float(values[2])
                    ey_imag = float(values[3])
                    magnitude_squared = ex_real**2 + ex_imag**2 + ey_real**2 + ey_imag**2
                    data.append(magnitude_squared)

            data = np.array(data)

            # 球坐标系平均功率计算（与 plotdirectory.py 保持一致）
            theta_rad = theta * np.pi / 180
            delta_theta = step * math.pi / 180
            delta_phi = 2 * math.pi
            solid_angle_elements = abs(np.sin(theta_rad)) * delta_theta * delta_phi
            total_powerall = np.sum(data * solid_angle_elements)
            avg_powerall = total_powerall / (4 * math.pi)

            # 与 plotdirectory.py 一致：avg_power = 1.0（不做总功率归一化）
            avg_power = 1.0
            gain_dbi = 10 * np.log10(data / avg_power)

            print(f"  φ={phi_angle}° total_power={total_powerall:.4f}, avg_power_all={avg_powerall:.4f}")

            cuts.append({
                'theta': theta,
                'gain_dbi': gain_dbi,
                'phi': phi_angle
            })
            i += num_points + 1
        else:
            i += 1

    return cuts

# ========== 参数配置 ==========
@dataclass
class AntennaConfig:
    freq: float = 12e9
    F: float = 0.6
    D: float = 1.0
    offset_x: float = 0.5      # 偏馈：投影圆心位于 (0.5, 0, 0)
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

# ========== 核心分析流程 ==========
def run_antenna_analysis(cfg: AntennaConfig):
    feed = FeedAntenna(cfg.freq, edge_angle_deg=cfg.edge_angle_deg,
                       edge_taper_db=cfg.edge_taper_db)
    P_feed_total = compute_feed_total_power(feed)

    # 反射体网格生成（常数近似使用）
    theta0_max_rad = np.deg2rad(cfg.theta0_max_deg)
    z = 1.09 * (np.pi * cfg.D / cfg.lam) * np.sin(theta0_max_rad) + 10.0
    po2_min = int(np.round(z))
    po1_min = int(np.round(z / 2.4))
    tri_verts, tri_norms, tri_areas = generate_parabolic_mesh(
        cfg.F, cfg.D, N_radial=po1_min, N_azimuth=po2_min,
        offset_x=cfg.offset_x, offset_y=cfg.offset_y
    )
    print(f"根据收敛性要求，设置网格: N_radial={po1_min}, N_azimuth={po2_min}")

    # 馈源指向计算
    focus = np.array([0.0, 0.0, cfg.F])
    z_center = (cfg.offset_x**2 + cfg.offset_y**2) / (4 * cfg.F)
    center_on_reflector = np.array([cfg.offset_x, cfg.offset_y, z_center])
    feed_axis = center_on_reflector - focus
    feed_axis /= np.linalg.norm(feed_axis)

    # 方向图扫描角度
    theta_start, theta_stop, num_theta = cfg.theta_scan_deg
    theta_scan = np.linspace(theta_start, theta_stop, num_theta)

    # 用于存储各方法的方向性结果（dBi）
    dBi_results = {}

    # ---------- 方法1：常数近似 PO ----------
    if USE_CONSTANT_PO:
        tpostart = time.time()
        print("计算常数近似 PO ...")
        positions_c = np.mean(tri_verts, axis=1)
        norms_c = np.mean(tri_norms, axis=1)
        norms_c /= np.linalg.norm(norms_c, axis=1, keepdims=True)
        H_inc_c = compute_incident_H(positions_c, feed, cfg.F, feed_axis)
        po_c = POIntegrator(positions_c, norms_c, tri_areas, feed.k)

        E_c  = po_c.far_field_pattern(H_inc_c, theta_scan, phi_deg=0.0)
        U_c  = np.sum(np.abs(E_c)**2, axis=-1)
        D_c  = 4 * np.pi * U_c / P_feed_total
        dBi_c = 10 * np.log10(D_c + 1e-15)
        dBi_results['PO '] = dBi_c
        print(f"PO 近似峰值方向性: {np.max(dBi_c):.2f} dBi")
        print(f"PO 近似耗时: {time.time() - tpostart:.2f} 秒")

    # ---------- 方法2：GB 经验公式 ----------
    if USE_GB_FORMULA:
        print("计算 GB 经验公式 ...")
        # 计算馈源半功率宽度
        theta_b_deg, theta_b_rad = compute_theta_b_from_feed(feed)
        print(f"馈源半功率波束宽度 θ_b = {theta_b_deg:.3f}°")

        # 计算反射器边缘半张角
        h = cfg.offset_x - cfg.D / 2.0   # offset clearance
        theta0_rad = compute_theta_0(cfg.D, cfg.F, h)
        theta0_deg = np.rad2deg(theta0_rad)
        print(f"反射器边缘半张角 θ0 = {theta0_deg:.3f}°")

        # 边缘照射锥削
        E_edge = feed.far_field_amplitude(theta0_rad)
        T_dB = -20 * np.log10(E_edge)
        print(f"边缘照射锥削 T = {T_dB:.2f} dB")

        # 估算波束宽度、副瓣、效率、峰值方向性
        theta3_deg = (0.762 * T_dB + 58.44) * (cfg.lam / cfg.D)
        print(f"GB 估算半功率波束宽度 = {theta3_deg:.3f}°")
        SLL_dB = -0.037 * T_dB**2 - 0.376 * T_dB - 17.6
        print(f"GB 估算第一副瓣电平 = {SLL_dB:.2f} dB")
        cos_half = np.cos(theta0_rad / 2.0)
        n = -0.05 * T_dB / np.log10(cos_half)
        print(f"参数 n = {n:.3f}")
        cot_half = 1.0 / np.tan(theta0_rad / 2.0)
        efficiency = 4 * cot_half**2 * (1 - cos_half**n)**2 * (n + 1) / (n**2)
        print(f"反射面效率 η = {efficiency*100:.2f}%")
        D_peak = 10 * np.log10((np.pi * cfg.D / cfg.lam)**2 * efficiency)
        print(f"GB 估算峰值方向性 = {D_peak:.2f} dBi")

        # 生成方向图（对称）
        theta_pos = np.abs(theta_scan)  # 利用对称性，只对非负角度计算相对方向图
        gb_rel = gaussian_beam_relative_pattern(theta_pos, theta3_deg, SLL_dB)
        gb_dBi = D_peak + gb_rel
        dBi_results['GB 经验公式'] = gb_dBi

    # ---------- 方法3：GBMA 高斯波束模式分析 ----------
    if USE_GBMA:
        tstart = time.time()
        print("计算 GBMA (高斯波束模式分析) ...")
        E_gbma, dBi_gbma = compute_far_field_gbma(
            feed, cfg, theta_scan,
            bounces=1,
            include_diffraction=False,
            n_beams=181,
            gabor_mode='angular'
        )
        dBi_results['GBMA 高斯波束模式'] = dBi_gbma
        print(f"GBMA 峰值方向性: {np.max(dBi_gbma):.2f} dBi")
        print(f"GBMA 耗时: {time.time() - tstart:.2f} 秒")

    # ---------- 方法4：导入 .cut 仿真数据 ----------
    cut_curves = None
    if USE_CUT_FILE:
        try:
            cut_curves = load_cut_file('single_cut.cut')
            print(f"从 single_cut.cut 加载了 {len(cut_curves)} 个切面数据 (φ = {[c['phi'] for c in cut_curves]})")
            for cut in cut_curves:
                print(f"  φ={cut['phi']}° 峰值增益: {cut['gain_dbi'].max():.2f} dBi")
        except FileNotFoundError:
            print("警告: 未找到 single_cut.cut，跳过仿真数据对比")

    # ---------- 绘图 ----------
    if not dBi_results:
        print("没有启用任何方法，无法绘图。")
        return

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(10, 6))

    # 用不同线型区分各方法
    linestyles = ['--', '-.', '--', '--']
    for idx, (method_name, dBi) in enumerate(dBi_results.items()):
        ls = linestyles[idx % len(linestyles)]
        plt.plot(theta_scan, dBi, ls, label=method_name)

    # 叠加 .cut 文件中的仿真数据（仅 φ=0° 切面）
    if cut_curves is not None:
        for cut in cut_curves:
            if abs(cut['phi'] - 0.0) < 1e-6:
                plt.plot(cut['theta'], cut['gain_dbi'], linestyle=':', color='red', linewidth=1.5, label='GRASP')
                break

    plt.xlabel('θ (度)')
    plt.ylabel('方向性 (dBi)')
    # plt.title(f'抛物面天线方向图对比 (偏移: x={cfg.offset_x}, y={cfg.offset_y})\n含 GRASP 仿真数据切面对比')
    plt.title(f'抛物面天线方向图对比 ( φ=0°)')

    plt.grid(True)
    plt.legend()
    # plt.xlim(-10, 10)
    plt.ylim(-40, 50)
    plt.tight_layout()

    print(f"总辐射功率: {P_feed_total:.3e} W")
    plt.savefig('pattern_comparison.png', dpi=150, bbox_inches='tight')
    print("方向图已保存至 pattern_comparison.png")
    plt.show()

if __name__ == "__main__":
    config = AntennaConfig()
    # 可在此处修改参数，例如：
    # config.freq = 14e9
    # config.D = 1.5
    # config.offset_x = 0.2
    # config.edge_taper_db = -15
    run_antenna_analysis(config)