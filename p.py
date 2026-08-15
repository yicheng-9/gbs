
import numpy as np
import matplotlib.pyplot as plt
import math
import os

def load_cut_file(file_path='single_cut.cut'):
    """
    从 .cut 文件中读取远场方向图数据（标准 GRASP 格式）
    返回: list[dict], 每个 dict 包含 'theta', 'gain_dbi', 'phi'
    """
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

            # 与原始 plotdirectory.py 保持一致：avg_power = 1.0，不做总功率归一化
            avg_power = 1.0
            gain_dbi = 10 * np.log10(data / avg_power)

            if file_path=="gbma.cut":
                gain_dbi = 10 * np.log10(data / avg_power)#-30


            cuts.append({
                'theta': theta,
                'gain_dbi': gain_dbi,
                'phi': phi_angle
            })
            i += num_points + 1
        else:
            i += 1

    return cuts

def plot_cut_files(file_list, labels=None, title="方向图对比", linestyles=None,xlabel="θ (度)", ylabel="方向性 (dBi)",
                   xlim=None, ylim=None, save_path=None, show=True):
    """
    读取一个或多个 .cut 文件并绘制方向图曲线。
    :param file_list: 文件名列表（str 或 list of str）
    :param labels: 曲线标签列表（与 file_list 长度相同），若为 None 则使用文件名
    :param title: 图标题
    :param xlabel: x轴标签
    :param ylabel: y轴标签
    :param xlim: (xmin, xmax) 或 None
    :param ylim: (ymin, ymax) 或 None
    :param save_path: 保存图片路径（如 'figure.png'），None 则不保存
    :param show: 是否显示图形
    """
    if isinstance(file_list, str):
        file_list = [file_list]

    if labels is None:
        labels = [os.path.splitext(os.path.basename(f))[0] for f in file_list]
    elif len(labels) != len(file_list):
        raise ValueError("labels 长度必须与 file_list 相同")

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(10, 6))


    if linestyles is None:
        linestyles = ['-'] * len(file_list)  # 默认全部实线
    for idx, (fname, label) in enumerate(zip(file_list, labels)):
        cuts = load_cut_file(fname)
        if not cuts:
            print(f"警告: {fname} 中未找到有效数据")
            continue
        # 取第一个切面（若包含多个，通常第一个是 φ=0°）
        cut = cuts[0]
        plt.plot(cut['theta'], cut['gain_dbi'], label=label, linewidth=1.5, linestyle=linestyles[idx])
        print(f"已加载 {fname} (φ={cut['phi']}°) 峰值 {np.max(cut['gain_dbi']):.2f} dBi")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图片已保存至 {save_path}")

    if show:
        plt.show()

if __name__ == "__main__":
    font_size =17
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = font_size                 # 基础字体大小
    plt.rcParams['axes.titlesize'] = font_size + 2        # 标题稍大
    plt.rcParams['axes.labelsize'] = font_size            # 轴标签
    plt.rcParams['xtick.labelsize'] = font_size - 2       # x轴刻度
    plt.rcParams['ytick.labelsize'] = font_size - 2       # y轴刻度
    plt.rcParams['legend.fontsize'] = font_size - 2       # 图例
    # 示例：绘制 PO.cut、gbma.cut 以及可能的 GRASP 仿真数据 single_cut.cut
    plot_cut_files(
        file_list=[ 'gbma.cut','PO.cut', 'single_cut1.cut'],
        labels=[ 'GBMA','PO', 'GRASP'],
        linestyles=['-', '--', ':'],   # 分别对应实线、虚线、点线
        title='抛物面天线方向图对比 (φ=0°)',
        xlim=(-10, 10),
        ylim=(-40, 50),
        save_path='pattern_comparison.png',
        show=True
    )
