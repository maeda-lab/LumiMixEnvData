import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ========== 1. 读取单个 CSV 文件 ==========
file_path = "D:/vectionProject/public/BrightnessLinearData/20250701_175243_Fps1_CameraSpeed1_ExperimentPattern_Fourier_ParticipantName_KK_TrialNumber_1.csv"
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

# ========== 2. 提取参数 V0, A1, A2, A3, A4 ==========
param_names = ["V0", "A1", "A2", "A3", "A4"]

# V0 从 Velocity 列中读取最后一个值
v0_series = df[df["StepNumber"] == 0]["Velocity"]
V0 = v0_series.iloc[-1] if not v0_series.empty else 0

# A1~A4 从 StepNumber 对应 Amplitude 提取
A_params = []
for i in range(1, 5):
    amp_series = df[df["StepNumber"] == i]["Amplitude"]
    amp = amp_series.iloc[-1] if not amp_series.empty else 0
    A_params.append(amp)

params = np.array([V0] + A_params)

# ========== 3. 定义 v(t) 函数 ==========
def v_curve(par, t):
    V0, A1, A2, A3, A4 = par
    ω = 2*np.pi
    # return V0 + A1*np.sin(ω*t + A2) + A3*np.sin(2*ω*t + A4)
    return V0 + A1*np.sin(ω*t) + A2*np.cos(ω*t) + \
        A3*np.sin(2*ω*t) + A4*np.cos(2*ω*t)

t = np.linspace(0, 10, 2000)

# ========== 4. 读取亮度数据并修正 ==========
time = df["Time"] / 1000
mask = (time <= 2) & (df["BackFrameNum"] % 2 != 0)
df.loc[mask, ["BackFrameNum", "FrondFrameNum"]] = df.loc[mask, ["FrondFrameNum", "BackFrameNum"]].to_numpy()
df.loc[mask, ["BackFrameLuminance", "FrondFrameLuminance"]] = df.loc[mask, ["FrondFrameLuminance", "BackFrameLuminance"]].to_numpy()

# ========== 5. 作图 ==========
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True, gridspec_kw={'hspace': 0.3})

# 子图 1：亮度变化
ax1.plot(time, df["FrondFrameLuminance"], label="Frond Frame", alpha=0.7)
ax1.plot(time, df["BackFrameLuminance"], label="Back Frame", alpha=0.7)
ax1.set_ylabel("Luminance (0-1)")
ax1.set_title("Luminance vs Time")
ax1.set_xlim(0, 10)
ax1.grid(True)
ax1.legend()

# 子图 2：v(t)
v_vals = v_curve(params, t)
ax2.plot(t, v_vals, color="tab:blue", label="v(t)")

# 参数文字显示
param_text = "\n".join([f"{name} = {val:.3f}" for name, val in zip(param_names, params)])
ax2.text(0.02, 0.95, param_text, transform=ax2.transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

ax2.set_xlabel("Time (s)")
ax2.set_ylabel("v(t)")
ax2.set_xlim(0, 5)
ax2.set_ylim(-2, 4)
# ax2.set_title(r"v(t) = V0 + A1·sin(ωt + A2) + A3·sin(2ωt + A4)")
ax2.set_title(r"v(t)=V0+A1·sin(ωt)+A2·cos(ωt)+A3·sin(2ωt)+A4·cos(2ωt)")
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()
