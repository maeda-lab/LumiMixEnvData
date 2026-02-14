import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# === 1. 文件列表：H 与 K 各 3 个试验 ===
files = {
    "H": [
        "D:/vectionProject/public/Experiment2Data/20250529_165716_fps1_ParticipantName_H_TrialNumber_1.csv",
        "D:/vectionProject/public/Experiment2Data/20250529_170506_fps1_ParticipantName_H_TrialNumber_2.csv",
        "D:/vectionProject/public/Experiment2Data/20250529_171007_fps1_ParticipantName_H_TrialNumber_3.csv",
    ],
    "K": [
        "D:/vectionProject/public/Experiment2Data/20250528_201715_fps1_ParticipantName_K_TrialNumber_1.csv",
        "D:/vectionProject/public/Experiment2Data/20250528_185421_fps1_ParticipantName_K_TrialNumber_2.csv",
        "D:/vectionProject/public/Experiment2Data/20250528_185926_fps1_ParticipantName_K_TrialNumber_3.csv",
    ],
        
    "N": [
        "D:/vectionProject/public/Experiment2Data/20250529_143935_fps1_ParticipantName_N_TrialNumber_1.csv",
        "D:/vectionProject/public/Experiment2Data/20250529_144201_fps1_ParticipantName_N_TrialNumber_2.csv",
        "D:/vectionProject/public/Experiment2Data/20250529_145128_fps1_ParticipantName_N_TrialNumber_3.csv",
    ],
    "O": [
        "D:/vectionProject/public/Experiment2Data/20250529_153427_fps1_ParticipantName_O_TrialNumber_1.csv",
        "D:/vectionProject/public/Experiment2Data/20250529_154149_fps1_ParticipantName_O_TrialNumber_2.csv",
        "D:/vectionProject/public/Experiment2Data/20250529_155355_fps1_ParticipantName_O_TrialNumber_3.csv",
    ],
}
param_names = ["V0", "A1", "A2", "A3", "A4"]

# === 2. 提取 5 个参数，计算每人均值 & 总体均值 ===
person_mean = {}
for person, paths in files.items():
    vals = []
    for p in paths:
        df = pd.read_csv(p)
        df.columns = df.columns.str.strip()
        vals.append([df[df["StepNumber"] == i]["Amplitude"].iloc[-1] for i in range(5)])
    person_mean[person] = np.mean(vals, axis=0)

overall_mean = np.mean(list(person_mean.values()), axis=0)
overall_std  = np.std (list(person_mean.values()), axis=0)
print("Overall Mean:", overall_mean)
# === 3. 生成 v(t) 曲线及误差带 ===
omega = 2 * np.pi          # 1 Hz
t = np.linspace(0, 10, 2000)
V0, A1, A2, A3, A4 = overall_mean
v  = V0 + A1*np.sin(omega*t) + A2*np.cos(omega*t) + \
          A3*np.sin(2*omega*t) + A4*np.cos(2*omega*t)

V0s, A1s, A2s, A3s, A4s = overall_std
v_up = (V0+V0s) + (A1+A1s)*np.sin(omega*t) + (A2+A2s)*np.cos(omega*t) + \
       (A3+A3s)*np.sin(2*omega*t) + (A4+A4s)*np.cos(2*omega*t)
v_lo = (V0-V0s) + (A1-A1s)*np.sin(omega*t) + (A2-A2s)*np.cos(omega*t) + \
       (A3-A3s)*np.sin(2*omega*t) + (A4-A4s)*np.cos(2*omega*t)

# === 4. 读取亮度（随便选 H 的第 1 个试验做示例） ===
df_lum = pd.read_csv(files["H"][0])
df_lum.columns = df_lum.columns.str.strip()
time = df_lum["Time"] / 1000
# 交换奇数 BackFrame 的前后数据（保持你之前的逻辑）
mask = (time <= 10) & (df_lum["BackFrameNum"] % 2 != 0)
tmp_num = df_lum.loc[mask, "BackFrameNum"].copy()
tmp_lum = df_lum.loc[mask, "BackFrameLuminance"].copy()
df_lum.loc[mask, "BackFrameNum"]       = df_lum.loc[mask, "FrondFrameNum"]
df_lum.loc[mask, "BackFrameLuminance"] = df_lum.loc[mask, "FrondFrameLuminance"]
df_lum.loc[mask, "FrondFrameNum"]      = tmp_num
df_lum.loc[mask, "FrondFrameLuminance"]= tmp_lum

# === 5. 组合子图 ===
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True, gridspec_kw={'hspace': 0.3})

# -- 子图 1：亮度 --
ax1.plot(time, df_lum["FrondFrameLuminance"], label="Frond Frame", alpha=0.7)
ax1.plot(time, df_lum["BackFrameLuminance"],  label="Back Frame",  alpha=0.7)
ax1.set_ylabel("Luminance (0-1)")
ax1.set_title("Luminance vs Time")
ax1.set_xlim(0, 2); ax1.legend(); ax1.grid(True)

# -- 子图 2：v(t) ±1 SD --
ax2.plot(t, v, label="Mean v(t)")
ax2.fill_between(t, v_lo, v_up, alpha=0.3, label="±1 SD")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("v(t)")
ax2.set_xlim(0, 2); ax2.set_ylim(-1, 2)
ax2.set_title(r"v(t)=V0+A1·sin(ωt)+A2·cos(ωt)+A3·sin(2ωt)+A4·cos(2ωt)")
ax2.grid(True); ax2.legend()

plt.tight_layout()
plt.show()
