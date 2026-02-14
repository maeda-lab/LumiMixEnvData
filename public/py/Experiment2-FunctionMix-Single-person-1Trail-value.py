import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === Step 1: 定义混合函数 ===
def cosine_blend(x):
    return 0.5 * (1 - np.cos(np.pi * x))

def linear_blend(x):
    return x

def acos_blend(x):
    return np.arccos(-2 * x + 1) / np.pi

def dynamic_blend(x, knob_value):
    if knob_value <= 0.1 or knob_value >= 1.9:
        return cosine_blend(x)
    
    k = knob_value - 0.1
    if k <= 0.6:
        t = k / 0.6
        return (1 - t) * cosine_blend(x) + t * linear_blend(x)
    elif k <= 1.2:
        t = (k - 0.6) / 0.6
        return (1 - t) * linear_blend(x) + t * acos_blend(x)
    else:
        t = (k - 1.2) / 0.6
        return (1 - t) * acos_blend(x) + t * cosine_blend(x)

# === Step 2: 加载 CSV 文件 & 获取最后一个 FunctionRatio 值 ===
csv_path = "D:/vectionProject/public/BrightnessData/20250709_183035_Fps1_CameraSpeed1_ExperimentPattern_FunctionMix_ParticipantName_L_TrialNumber_1.csv"  # 替换为你自己的文件路径
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()
function_ratio = df["FunctionRatio"].dropna().iloc[-1]

# === Step 3: 生成混合曲线数据 ===
x_vals = np.linspace(0, 1, 500)
cos_vals = cosine_blend(x_vals)
lin_vals = linear_blend(x_vals)
acos_vals = acos_blend(x_vals)
dyn_vals = dynamic_blend(x_vals, function_ratio)

# === Step 4: 绘图 ===
plt.figure(figsize=(8, 6))
plt.plot(x_vals, cos_vals, linestyle="--", label="Cosine")
plt.plot(x_vals, lin_vals, linestyle="--", label="Linear")
plt.plot(x_vals, acos_vals, linestyle="--", label="Acos")
plt.plot(x_vals, dyn_vals, linewidth=2, label=f"Dynamic (FunctionRatio={function_ratio:.3f})")

plt.title("Dynamic mixing curves for a single experiment")
plt.xlabel("Normalized Time (x)")
plt.ylabel("Blend Value")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
