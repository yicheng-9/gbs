import cmath
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')          # 非交互式后端，无需 GUI
import matplotlib.pyplot as plt

# ============================================================
#  常量
# ============================================================
pi = math.pi
c = 299792458.0                # 光速 (m/s)
epsilon0 = 8.854187817e-12     # 真空介电常数 (F/m)
mu0 = 4 * pi * 1e-7            # 真空磁导率 (H/m)


def compute_EE(theta, k, BB):
    """
    计算标量高斯波束远场方向图 EE(theta)
    对应原 C++ 代码中的复标量场
    """
    EE = (k * cmath.exp(-1j * pi) *
          cmath.exp(k * BB * math.cos(theta)) *
          (1 + math.cos(theta)) *
          2.0 * BB * math.sqrt(2 * k * BB) /
          cmath.sqrt(8 * k * BB * k * BB - 4 * k * BB + 1) /
          cmath.exp(k * BB))
    return EE


def gswave():
    # ============================================================
    #  参数设置（与 C++ 一致）
    # ============================================================
    freq = 12e9                  # 频率 12 GHz
    k = 2.0 * pi * freq / c      # 波数
    print(f"k = {k:.6f}")

    eta = cmath.sqrt(mu0 / epsilon0)   # 波阻抗（保留计算）
    taper = -12.0                      # 锥削电平 (dB)
    angle = pi / 4                     # 锥削角

    # ---- 计算 BB（波束参数） ----
    BB = (20.0 * math.log10((1 + math.cos(angle)) / 2.0) - taper) / \
         ((1 - math.cos(angle)) * 20.0 * k * math.log10(math.e))
    W0 = math.sqrt(2 * BB / k)

    print(f"angle = {angle:.4f} rad  ({math.degrees(angle):.1f} deg)")
    print(f"k*BB  = {k * BB:.6f}")
    print(f"BB    = {BB:.6e}")
    print(f"W0    = {W0:.6e}")
    print(f"taper = {taper} dB")

    # ============================================================
    #  极化分解说明
    #  ----------------------------------------------------------
    #  EE(θ) 本身即球坐标远场的 θ 分量（TM 波，标量高斯波束）：
    #
    #    E_theta(θ, φ) =  EE(θ)                  (θ 极化，主分量)
    #    E_phi  (θ, φ) =  0                      (φ 极化，交叉极化为零)
    #
    #  物理意义：
    #    - 波束旋转对称，E_theta 不随 φ 变化
    #    - 标量理论下无交叉极化 (E_phi ≡ 0)
    #    - 若需线极化 (x/y-pol)，需组合 TM+TE 模式
    #
    #  总功率方向图 ∝ |E_theta|² + |E_phi|² = |EE(θ)|²
    # ============================================================

    # ============================================================
    #  第一部分：输出 .cut 文件（标准 CST/HFSS 远场格式）
    # ============================================================
    phi_cuts = [0, 45, 90]                       # 三个 phi 切面
    theta_vals_deg = list(range(-180, 181, 10))  # θ: -180° ~ 180°, 步长 10°
    n_theta = len(theta_vals_deg)                # 37 个点

    with open("farfield.cut", "w") as f:
        for phi_deg in phi_cuts:
            phi = phi_deg * pi / 180.0

            # 文件头：每个切面一个 header
            f.write("Field data in cuts\n")

            # 格式化 phi 值（与参考 .cut 文件一致）
            if phi_deg == 0:
                phi_str = " 0.0000000000E+00"
            else:
                phi_str = f" 0.{phi_deg * 10**8:010d}E+02"

            f.write(f" -0.1800000000E+03  0.1000000000E+02  {n_theta:3d} {phi_str}    3    1    2\n")

            # 数据行：Re(E_theta)  Im(E_theta)  Re(E_phi)  Im(E_phi)
            for theta_deg in theta_vals_deg:
                theta = theta_deg * pi / 180.0
                EE = compute_EE(theta, k, BB)

                # ---- 极化分解：EE(θ) 本身即球坐标 θ 分量 ----
                E_theta = EE                       # θ 分量 = 标量远场
                E_phi   = 0.0j                     # φ 分量 = 0 (标量理论无交叉极化)

                f.write(f" {E_theta.real: .10E}{E_theta.imag: .10E}"
                        f"{E_phi.real: .10E}{E_phi.imag: .10E}\n")

    print("\n[OK] farfield.cut written (3 phi cuts, 37 theta points each)")

    # ============================================================
    #  第二部分：文本输出（与原代码兼容）
    # ============================================================
    with open("output1.txt", "w") as out_file:
        for theta_deg in range(-180, 181):
            theta = theta_deg * pi / 180.0
            EE = compute_EE(theta, k, BB)
            out_file.write(f"{EE.real:.10e}{EE.imag:+.10e}j\n")

    print("[OK] output1.txt written (scalar EE, for backward compatibility)")

    # ============================================================
    #  第三部分：matplotlib 可视化 —— 远场二维方向图
    #           取 φ = 0°, 45°, 90° 三个切面
    # ============================================================
    theta_plot_deg = np.linspace(-180, 180, 721)   # 0.5° 分辨率
    theta_plot_rad = np.deg2rad(theta_plot_deg)

    # 预计算 EE
    EE_plot = np.array([compute_EE(th, k, BB) for th in theta_plot_rad], dtype=complex)

    # 颜色和线型
    colors     = {0: '#2166AC', 45: '#B2182B', 90: '#4DAF4A'}
    linestyles = {0: '-',       45: '--',      90: '-.'}
    labels     = {0: r'$\phi = 0^\circ, 45^\circ, 90^\circ$ (identical, rotationally symmetric)',
                  45: r'$\phi = 45^\circ$',
                  90: r'$\phi = 90^\circ$'}

    # ---- 直角坐标图 ----
    fig1, ax1 = plt.subplots(figsize=(12, 7))

    # E_theta = EE(θ), E_phi = 0 → 所有 phi 切面图案完全相同
    power = np.abs(EE_plot)**2              # |E_θ|² + |E_φ|² = |EE|²

    # 归一化到 0 dB，截断至 -60 dB
    power_max = np.max(power)
    pattern_db = 10 * np.log10(np.maximum(power / power_max, 1e-12))
    pattern_db = np.clip(pattern_db, -60, 0)

    ax1.plot(theta_plot_deg, pattern_db,
             color=colors[0], linestyle=linestyles[0],
             linewidth=1.8, label=labels[0])

    ax1.set_xlabel(r'$\theta$ (deg)', fontsize=13)
    ax1.set_ylabel('Normalized Pattern (dB)', fontsize=13)
    ax1.set_title('Far-Field 2D Radiation Pattern\n'
                  r'(Gaussian Beam, $f = 12\ \mathrm{GHz}$, TM polarization, $E_\theta = EE,\ E_\phi = 0$)',
                  fontsize=14)
    ax1.legend(fontsize=11, loc='lower left')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(-180, 180)
    ax1.set_ylim(-60, 3)
    ax1.axhline(y=-3, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)
    ax1.text(175, -2.5, '-3 dB', fontsize=9, color='gray', ha='right')

    plt.tight_layout()
    plt.savefig("farfield_rectangular.png", dpi=150, bbox_inches='tight')
    print("[OK] farfield_rectangular.png saved")

    # ---- 极坐标图 ----
    fig2, ax2 = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(10, 10))

    # 归一化线性幅度
    mag_lin = np.sqrt(power / power_max)

    # 所有 phi 切面相同，绘制一条即可
    ax2.plot(theta_plot_rad, mag_lin,
             color=colors[0], linewidth=1.8,
             label=r'All $\phi$ (rotationally symmetric)')

    ax2.set_theta_zero_location('N')
    ax2.set_theta_direction(-1)
    ax2.set_thetamin(-180)
    ax2.set_thetamax(180)
    ax2.set_title('Far-Field 2D Radiation Pattern -- Polar\n'
                  r'(Normalized Linear Amplitude, $f = 12\ \mathrm{GHz}$, $E_\theta = EE,\ E_\phi = 0$)',
                  fontsize=14, pad=25)
    ax2.legend(fontsize=11, loc='upper right', bbox_to_anchor=(1.35, 1.05))
    ax2.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig("farfield_polar.png", dpi=150, bbox_inches='tight')
    print("[OK] farfield_polar.png saved")

    print("\nDone.")


if __name__ == "__main__":
    gswave()
