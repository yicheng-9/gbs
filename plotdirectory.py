import numpy as np
import matplotlib.pyplot as plt
import math
# from math import pi
# from math import cos
# # 文件路径
file_path = 'single_cut.cut'
# 读取文件
with open(file_path, 'r') as f:
    lines = f.readlines()

# 存储所有切片的数据
cuts = []

i = 0
while i < len(lines):
    line = lines[i].strip()
    
    if line.startswith('Field data in cuts'):
        # 下一行是参数行
        i += 1
        param_line = lines[i].split()
        start_angle = float(param_line[0])  # -60.0
        step = float(param_line[1])         # 0.1
        num_points = int(param_line[2])     # 1201
        phi_angle = float(param_line[3])    # 0.0, 45.0, 90.0, ...
        
        # 构造角度数组
        # theta = np.arange(start_angle, start_angle + num_points * step, step)
        # 替换原来的 theta 生成方式
        theta = np.linspace(start_angle, start_angle + (num_points - 1) * step, num_points)
        # 提取接下来的1201行电场数据
        data = []
        for j in range(i + 1, i + 1 + num_points):
            values = lines[j].split()
            if len(values) >= 4:
                ex_real = float(values[0])
                ex_imag = float(values[1])
                ey_real = float(values[2])
                ey_imag = float(values[3])
                # 计算电场幅度的平方（功率密度）
                magnitude_squared = ex_real**2 + ex_imag**2 + ey_real**2 + ey_imag**2
            # 如果功率密度为零，则停止
                # if magnitude_squared==0:
                #     magnitude_squared=1e-20
                data.append(magnitude_squared)

        data = np.array(data)
        
        cuts.append({
            'theta': theta,
            'power': data,  # 存储功率密度
            'phi': phi_angle
        })
        
        # 更新索引到下一段
        i += num_points + 1
    else:
        i += 1

# 设置绘图样式
plt.figure(figsize=(12, 8))

# 颜色映射
colors = ['black', 'blue', 'red']
labels = ['φ = 0°', 'φ = 45°', 'φ = 90°']
# angle=np.pi*25/180
angle=math.pi*45/180
taper=-6
taper=-12
energyk=math.cos(angle)
# print(energyk)
# 绘制每个指定 φ 角度的曲线
for idx, phi_target in enumerate([0.0, 45.0, 90.0]):
    for cut in cuts:
        if abs(cut['phi'] - phi_target) < 1e-6:  # 匹配 φ 角度
            zero_index = np.argmin(np.abs(cut['theta']))
            
            # 提取前一半数据（从开始到theta=0）
            theta_half = cut['theta'][:zero_index+1]
            power_half = cut['power'][:zero_index+1]
            theta_rad_half = theta_half * np.pi / 180
            theta_rad = cut['theta'] * np.pi / 180

            # 正确的球坐标系平均功率计算
            # theta_rad = cut['theta'] * math.pi / 180  # 转换为弧度
            delta_theta = step * math.pi / 180  # theta步长（弧度）
            delta_phi = math.pi*2 # phi步长（假设等间隔）

            # 计算每个点的立体角元素
            solid_angle_elements_half = abs(np.sin(theta_rad_half)) * delta_theta * delta_phi
            solid_angle_elements = abs(np.sin(theta_rad)) * delta_theta * delta_phi

            # 总辐射功率（数值积分）
            total_powerhalf = np.sum(power_half * solid_angle_elements_half)
            total_powerall= np.sum(cut['power'] * solid_angle_elements)
            print(total_powerhalf,total_powerall)
            # 平均功率（总功率/4π）
            avg_power = 1.0
            avg_powerall = total_powerall / (4 * math.pi)
            print(avg_power,avg_powerall/2.0)
            # 归一化到全向天线并转换为 dB
            # Gain [dBi] = 10 * np.log10(Power / Average_Power)
            gain_dbi = 10 * np.log10(cut['power'] / avg_power)
            # if cut['theta']==0:
            print(gain_dbi.max())
            print(cut['power'].shape,gain_dbi.shape)
            plt.plot(cut['theta'], gain_dbi, 
                        color=colors[idx], linewidth=2, label=labels[idx])
plt.xlabel('θ [deg]', fontsize=12)
plt.ylabel('Gain [dBi]', fontsize=12)
plt.title('Far-field Radiation Pattern (Normalized to Isotropic Antenna)', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.ylim(-20, None)  # 设置合理的y轴范围
plt.tight_layout()
plt.show()