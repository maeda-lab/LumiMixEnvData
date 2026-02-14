import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

# ========== 参数 ==========
frames_folder = r"D:\vectionProject\public\ExperimentData3-Images"  # 图片所在文件夹
prefix = "capture1_"
num_steps = 101  # w 的采样点数（0~1）
ws = np.linspace(0, 1, num_steps)

# ========== 工具函数 ==========
def read_gray_linear(path):
    """读取彩色图并转换到线性灰度空间"""
    img = cv2.imread(path).astype(np.float32) / 255.0
    img_lin = np.power(img, 2.2)  # 去Gamma
    gray = 0.2126 * img_lin[..., 2] + 0.7152 * img_lin[..., 1] + 0.0722 * img_lin[..., 0]  # BGR->Gray
    return gray

def rms(x):
    """RMS 计算"""
    m = np.mean(x)
    return np.sqrt(np.mean((x - m) ** 2))

# ========== 读取图像 ==========
files = sorted([f for f in os.listdir(frames_folder) if f.startswith(prefix) and f.endswith(".png")])
print(f"共找到 {len(files)} 张图像")

if len(files) < 2:
    raise ValueError("至少需要两张帧图！")

# ========== 计算每对帧的 RMS 差能量 ==========
energy_curves = []
lut_list = []

for i in tqdm(range(len(files) - 1)):
    A = read_gray_linear(os.path.join(frames_folder, files[i]))
    B = read_gray_linear(os.path.join(frames_folder, files[i + 1]))

    rms_curve = []
    for w in ws:
        I = (1 - w) * A + w * B
        rms_curve.append(rms(I))
    rms_curve = np.array(rms_curve)
    rms_curve /= rms_curve.max()  # 归一化
    energy_curves.append(rms_curve)

    # ========== 计算 LUT ==========
    cumulative = np.cumsum(rms_curve)
    cumulative /= cumulative[-1]  # 归一化到 [0,1]
    # 构造 LUT: F = inverse(cumulative)
    lut_w = np.interp(ws, cumulative, ws)
    lut_list.append(lut_w)

# ========== 平均曲线 ==========
avg_energy = np.mean(energy_curves, axis=0)
avg_cumulative = np.cumsum(avg_energy)
avg_cumulative /= avg_cumulative[-1]
avg_lut = np.interp(ws, avg_cumulative, ws)

# ========== 可视化 ==========
plt.figure(figsize=(6,4))
for e in energy_curves:
    plt.plot(ws, e, alpha=0.4)
plt.plot(ws, avg_energy, 'k-', linewidth=2, label='Average Energy Curve')
plt.xlabel('w (blend ratio)')
plt.ylabel('Normalized RMS energy')
plt.title('RMS Energy Curves (each frame pair)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
plt.plot(ws, avg_cumulative, label='Cumulative Energy (average)')
plt.xlabel('w')
plt.ylabel('Cumulative normalized energy')
plt.title('Average cumulative curve')
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
plt.plot(ws, avg_lut, 'r-', linewidth=2, label='Average LUT mapping')
plt.plot(ws, ws, 'k--', alpha=0.5, label='Linear reference')
plt.xlabel('original w')
plt.ylabel("equalized w'")
plt.title('Average Equal-Energy LUT')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
