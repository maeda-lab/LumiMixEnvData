import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 加载新的文件
file_path = "D:/vectionProject/public/BrightnessLinearData/20250709_145729_Fps1_CameraSpeed1_ExperimentPattern_Phase_ParticipantName_KK_TrialNumber_1_BrightnessBlendMode_CosineOnly.csv"
df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()

# 参数提取
v0_series = df[df["StepNumber"] == 0]["Velocity"]
V0 = v0_series.iloc[-1] if not v0_series.empty else 0
A1 = df[df["StepNumber"] == 1]["Amplitude"].iloc[-1] if not df[df["StepNumber"] == 1].empty else 0
φ1 = df[df["StepNumber"] == 2]["Amplitude"].iloc[-1] if not df[df["StepNumber"] == 2].empty else 0
A2 = df[df["StepNumber"] == 3]["Amplitude"].iloc[-1] if not df[df["StepNumber"] == 3].empty else 0
φ2 = df[df["StepNumber"] == 4]["Amplitude"].iloc[-1] if not df[df["StepNumber"] == 4].empty else 0

params = np.array([V0, A1, φ1, A2, φ2])
param_names = ["V0", "A1", "φ1", "A2", "φ2"]

# v(t) 函数
def v_curve(par, t):
    V0, A1, φ1, A2, φ2 = par
    ω = 2*np.pi
    return V0 + A1 * np.sin(ω * t + φ1 + np.pi) + A2 * np.sin(2 * ω * t + φ2 + np.pi)

t = np.linspace(0, 10, 2000)

# 亮度数据修正
time = df["Time"] / 1000
mask = (time <= 10) & (df["BackFrameNum"] % 2 != 0)
df.loc[mask, ["BackFrameNum", "FrondFrameNum"]] = df.loc[mask, ["FrondFrameNum", "BackFrameNum"]].to_numpy()
df.loc[mask, ["BackFrameLuminance", "FrondFrameLuminance"]] = df.loc[mask, ["FrondFrameLuminance", "BackFrameLuminance"]].to_numpy()

# 绘图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True, gridspec_kw={'hspace': 0.3})

# 子图 1：亮度
ax1.plot(time, df["FrondFrameLuminance"], label="Frond Frame", alpha=.7)
ax1.plot(time, df["BackFrameLuminance"], label="Back Frame", alpha=.7)
ax1.set_ylabel("Luminance (0-1)")
ax1.set_title("Luminance vs Time")
ax1.set_xlim(0, 10)
ax1.grid(True)
ax1.legend()

# 子图 2：v(t)
v_vals = v_curve(params, t)
ax2.plot(t, v_vals, color="tab:blue", label="v(t)")

# 参数显示
param_text = "\n".join([f"{name} = {val:.3f}" for name, val in zip(param_names, params)])
ax2.text(0.02, 0.95, param_text, transform=ax2.transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

ax2.set_xlabel("Time (s)")
ax2.set_ylabel("v(t)")
ax2.set_xlim(0, 5)
ax2.set_ylim(-2, 4)
ax2.set_title(r"v(t) = V0 + A1·sin(ωt + φ1) + A2·sin(2ωt + φ2)")
ax2.grid(True)
ax2.legend()

plt.tight_layout();
plt.show()
