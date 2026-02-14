import pandas as pd, numpy as np, matplotlib.pyplot as plt

# ========== 1. 把你想分析的 CSV 路径放进来 ==========
# 示例：仅 1 位参与者 H（3 次试验）
files = {
    # "K": [
    #     "D:/vectionProject/public/Experiment2Data/20250601_190847_fps0.5_cameraSpeed1_ParticipantName_K_TrialNumber_1.csv",
    #     "D:/vectionProject/public/Experiment2Data/20250601_191538_fps0.5_cameraSpeed1_ParticipantName_K_TrialNumber_1.csv",
    #     "D:/vectionProject/public/Experiment2Data/20250601_191757_fps0.5_cameraSpeed1_ParticipantName_K_TrialNumber_1.csv",
    # ],
    "K": [
        "D:/vectionProject/public/BrightnessLinearData/20250701_200219_Fps1_CameraSpeed1_ExperimentPattern_Phase_ParticipantName_KK_TrialNumber_1.csv",
        "D:/vectionProject/public/BrightnessLinearData/20250701_195604_Fps1_CameraSpeed1_ExperimentPattern_Phase_ParticipantName_KK_TrialNumber_1.csv",
        "D:/vectionProject/public/BrightnessLinearData/20250701_194947_Fps1_CameraSpeed1_ExperimentPattern_Phase_ParticipantName_KK_TrialNumber_1.csv",
    ],

}
# 如果之后有第二个人 K，再加一段即可

param_names = ["V0", "A1", "A2", "A3", "A4"]
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

# ========== 2. 提取每个人的均值 & SD ==========
person_mean, person_sd = {}, {}
for idx, (person, paths) in enumerate(files.items()):
    vals = []
    for p in paths:
        df = pd.read_csv(p); df.columns = df.columns.str.strip()

        # V0 从 StepNumber == 0 中的 Velocity 提取
        v0_series = df[df["StepNumber"] == 0]["Velocity"]
        V0 = v0_series.iloc[-1] if not v0_series.empty else 0

        # A1 ~ A4 从 StepNumber == 1~4 中提取 Amplitude
        Ai = [
            df[df["StepNumber"] == i]["Amplitude"].iloc[-1]
            if not df[df["StepNumber"] == i]["Amplitude"].empty else 0
            for i in range(1, 5)
        ]
        vals.append([V0] + Ai)
        # vals.append([df[df["StepNumber"] == i]["Amplitude"].iloc[-1] for i in range(5)])
    arr = np.array(vals)
    person_mean[person] = arr.mean(axis=0)
    person_sd[person]   = arr.std (axis=0)

# ========== 3. 曲线生成函数 ==========
def v_curve(par, t):
    V0, A1, A2, A3, A4 = par
    ω = 2*np.pi
    return V0 + A1*np.sin(ω*t  + A2)  + \
        A3*np.sin(2*ω*t  + A4) 

t = np.linspace(0, 10, 2000)

# ========== 4. 亮度（用第一条文件） ==========
first_file = next(iter(files.values()))[0]          # 取字典第一人第一试
df_lum = pd.read_csv(first_file); df_lum.columns = df_lum.columns.str.strip()
time = df_lum["Time"] / 1000
mask = (time<=10) & (df_lum["BackFrameNum"]%2!=0)
df_lum.loc[mask, ["BackFrameNum","FrondFrameNum"]] = df_lum.loc[mask, ["FrondFrameNum","BackFrameNum"]].to_numpy()
df_lum.loc[mask, ["BackFrameLuminance","FrondFrameLuminance"]] = \
    df_lum.loc[mask, ["FrondFrameLuminance","BackFrameLuminance"]].to_numpy()

# ========== 5. 画图 ==========
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9,6), sharex=True, gridspec_kw={'hspace':0.3})

# --- 子图 1：亮度 ---
ax1.plot(time, df_lum["FrondFrameLuminance"], label="Frond Frame", alpha=.7)
ax1.plot(time, df_lum["BackFrameLuminance"],  label="Back Frame",  alpha=.7)
ax1.set_ylabel("Luminance (0-1)")
ax1.set_title("Luminance vs Time"); ax1.set_xlim(0,10)
ax1.grid(True); ax1.legend()

# --- 子图 2：每个人的 v(t) ---
for idx, (person, mean_par) in enumerate(person_mean.items()):
    sd_par   = person_sd[person]
    col      = colors[idx % len(colors)]
    v_mean   = v_curve(mean_par, t)
    v_upper  = v_curve(mean_par + sd_par, t)
    v_lower  = v_curve(mean_par - sd_par, t)

    ax2.plot(t, v_mean, color=col, label=f"")
    # 参数文本框
    param_names = ["V0", "A1", "φ1", "A3", "φ2"]
    param_text = "\n".join([f"{name} = {val:.3f}" for name, val in zip(param_names, mean_par)])
    ax2.text(0.02, 0.95 - idx * 0.25, param_text, transform=ax2.transAxes,
            fontsize=9, verticalalignment='top',
    bbox=dict(facecolor='white', alpha=0.75, edgecolor='gray'))
    
    ax2.fill_between(t, v_lower, v_upper, color=col, alpha=0.3, label="")

ax2.set_xlabel("Time (s)"); ax2.set_ylabel("v(t)")
ax2.set_xlim(0,5); ax2.set_ylim(-2,4)
title = "Single participant" if len(files)==1 else "Each participant"
ax2.set_title(r"v(t)=V0+A1·sin(ωt + φ1)+A2·sin(2ωt + φ2)")
ax2.grid(True); ax2.legend()

plt.tight_layout(); 
plt.show()
