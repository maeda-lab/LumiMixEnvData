import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. 读取 CSV 文件（修改为你的文件路径）
file_paths = [
    "D:/vectionProject/public/Experiment2Data/20250529_143935_fps1_ParticipantName_N_TrialNumber_1.csv",
        "D:/vectionProject/public/Experiment2Data/20250529_144201_fps1_ParticipantName_N_TrialNumber_2.csv",
        "D:/vectionProject/public/Experiment2Data/20250529_145128_fps1_ParticipantName_N_TrialNumber_3.csv",
             ]

""" file_paths = ["D:/vectionProject/public/Experiment2Data/20250518_104530_fps1_ParticipantName_K_TrialNumber_1.csv",
             "D:/vectionProject/public/Experiment2Data/20250518_102850_fps1_ParticipantName_K_TrialNumber_2.csv",
             "D:/vectionProject/public/Experiment2Data/20250518_104013_fps1_ParticipantName_K_TrialNumber_3.csv"] """

df = pd.read_csv(file_paths[0])

# 2. 去除列名空格
df.columns = df.columns.str.strip()
time = df['Time'] / 1000

mask = (time <= 10) & (df['BackFrameNum'] % 2 != 0)
# 先拷贝原值
orig_num = df.loc[mask, 'BackFrameNum'].copy()
orig_lum = df.loc[mask, 'BackFrameLuminance'].copy()

df.loc[mask, 'BackFrameNum']       = df.loc[mask, 'FrondFrameNum']
df.loc[mask, 'BackFrameLuminance'] = df.loc[mask, 'FrondFrameLuminance']
df.loc[mask, 'FrondFrameNum']      = orig_num
df.loc[mask, 'FrondFrameLuminance']= orig_lum

frond_frame_num = df['FrondFrameNum']
back_frame_num = df['BackFrameNum']
frond_frame_luminance = df['FrondFrameLuminance']
back_frame_luminance = df['BackFrameLuminance']


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True, gridspec_kw={'hspace': 0.3})

# Plot Frond Frame Luminance and Back Frame Luminance on the first subplot
ax1.plot(time, frond_frame_luminance, linestyle='-', color='b', label='Frond Frame Luminance', alpha=0.5)
ax1.plot(time, back_frame_luminance, linestyle='-', color='g', label='Back Frame Luminance', alpha=0.5)

# Plot points on the line for Frond Frame Luminance and Back Frame Luminance
ax1.scatter(time, frond_frame_luminance, color='b', s=3, alpha=0.4)
ax1.scatter(time, back_frame_luminance, color='g', s=3, alpha=0.4)

# Set labels for luminance
ax1.set_ylabel('Luminance Value (0-1)')
ax1.set_title('Luminance Value vs Time')
ax1.legend(loc='upper right')
ax1.grid()
# Limit x-axis to 10 seconds
ax1.set_xlim([-5, 15])

# 5. 设置时间轴和频率
t = np.linspace(0, 10, 2000)  # 0到10秒，2000个点
omega = 2 * np.pi  # 1Hz
param_names = ["V0", "A1", "A2", "A3", "A4"]
colors = ['orange', 'magenta', 'cyan']
for idx, fp in enumerate(file_paths, start=1):
    # 1) 读取 Excel
    df = pd.read_csv(fp)
    df.columns = df.columns.str.strip()
    time = df['Time'] / 1000

    # 2) 参数提取
    params = {}
    for pattern, name in enumerate(param_names):
        sub = df[df['StepNumber'] == pattern]
        params[name] = sub.iloc[-1]['Amplitude'] if not sub.empty else 0

    V0, A1, A2, A3, A4 = (params[n] for n in param_names)
    print(f"V0 = {V0}, A1 = {A1}, A2 = {A2}, A3 = {A3}, A4 = {A4}")

    # 3) 生成 y 曲线
    y = (V0
         + A1 * np.sin(omega * t)
         + A2 * np.cos(omega * t)
         + A3 * np.sin(2 * omega * t)
         + A4 * np.cos(2 * omega * t)
    )

    # 4) 画到 ax2
    # ax2.plot(t, y, label=f"Trial {idx}")
    ax2.plot(
        t, y,
        label=f"Trial {idx}",
        color=colors[idx-1],   # 第一次用 red，第二次 green，第三次 blue
        linewidth=1.5
    )



# 7. 绘图
ax2.set_title("v(t) = V0 + A1·sin(ωt) + A2·cos(ωt) + A3·sin(2ωt) + A4·cos(2ωt)")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("v(t)")
ax2.set_ylim(-1, 5)
ax2.grid(True)
ax2.legend()

# 限制横轴范围
ax1.set_xlim(0, 10)
ax2.set_xlim(0, 10)

# 设置刻度（可选）
ticks = np.arange(0, 11, 1)
ax1.set_xticks(ticks)
ax2.set_xticks(ticks)

plt.show()
