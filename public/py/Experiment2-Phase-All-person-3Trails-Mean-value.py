import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from collections import defaultdict

# 根目录
root_dir = "D:/vectionProject/public/BrightnessFunctionMixAndPhaseData"

# 文件名匹配模式
pattern = re.compile(r"ExperimentPattern_Phase_ParticipantName_(\w+)_TrialNumber_.*?_BrightnessBlendMode_(\w+)\.csv")

# 文件收集
participant_files = defaultdict(lambda: defaultdict(list))
for fname in os.listdir(root_dir):
    if fname.endswith(".csv") and "Test" not in fname:
        match = pattern.search(fname)
        if match:
            participant, mode = match.groups()
            full_path = os.path.join(root_dir, fname)
            participant_files[participant][mode].append(full_path)

# v(t)函数
def v_curve(par, t):
    V0, A1, φ1, A2, φ2 = par
    ω = 2 * np.pi
    return V0 + A1 * np.sin(ω * t + φ1 + np.pi) + A2 * np.sin(2 * ω * t + φ2 + np.pi)

# 数据提取函数
def extract_params(df):
    V0 = df[df["StepNumber"] == 0]["Velocity"].iloc[-1] if not df[df["StepNumber"] == 0].empty else 0
    A1 = df[df["StepNumber"] == 1]["Amplitude"].iloc[-1] if not df[df["StepNumber"] == 1].empty else 0
    φ1 = df[df["StepNumber"] == 2]["Amplitude"].iloc[-1] if not df[df["StepNumber"] == 2].empty else 0
    A2 = df[df["StepNumber"] == 3]["Amplitude"].iloc[-1] if not df[df["StepNumber"] == 3].empty else 0
    φ2 = df[df["StepNumber"] == 4]["Amplitude"].iloc[-1] if not df[df["StepNumber"] == 4].empty else 0
    return np.array([V0, A1, φ1, A2, φ2])

# 存储所有数据用于总体分析
overall_data = defaultdict(list)

# 遍历每个参与者
t = np.linspace(0, 10, 2000)
for participant, mode_files in participant_files.items():
    fig, axs = plt.subplots(2, 3, figsize=(15, 8))
    plt.suptitle(f"Participant {participant}: Brightness & v(t) Average with SD", fontsize=16)

    for i, mode in enumerate(["CosineOnly", "LinearOnly", "AcosOnly"]):
        files = mode_files.get(mode, [])
        params_list = []
        luminance_data = []

        for path in files:
            df = pd.read_csv(path)
            df.columns = df.columns.str.strip()
            params = extract_params(df)
            params_list.append(params)
            overall_data[mode].append(params)  # 加入总体数据

            time = df["Time"] / 1000
            mask = (time <= 10) & (df["BackFrameNum"] % 2 != 0)
            df.loc[mask, ["BackFrameNum", "FrondFrameNum"]] = df.loc[mask, ["FrondFrameNum", "BackFrameNum"]].to_numpy()
            df.loc[mask, ["BackFrameLuminance", "FrondFrameLuminance"]] = df.loc[mask, ["FrondFrameLuminance", "BackFrameLuminance"]].to_numpy()
            luminance_data.append((time, df["FrondFrameLuminance"], df["BackFrameLuminance"]))

        # 亮度曲线
        ax1 = axs[0, i]
        for time, front, back in luminance_data:
            ax1.plot(time, front,color='tab:blue', alpha=0.7)
            ax1.plot(time, back, color='tab:orange',alpha=0.7)
        ax1.set_title(f"Luminance - {mode}")
        ax1.set_xlim(0, 3)
        ax1.grid(True)

        # v(t) ± SD
        ax2 = axs[1, i]
        if params_list:
            avg_params = np.mean(params_list, axis=0)
            std_params = np.std(params_list, axis=0)
            v_mean = v_curve(avg_params, t)
            v_upper = v_curve(avg_params + std_params, t)
            v_lower = v_curve(avg_params - std_params, t)

            ax2.plot(t, v_mean, color="tab:blue", label="Mean v(t)")
            ax2.fill_between(t, v_lower, v_upper, color='tab:blue', alpha=0.2, label="±1 SD")

            param_text = "\n".join([
                f"{n}={m:.2f}±{s:.2f}" for n, m, s in zip(["V0", "A1", "φ1", "A2", "φ2"], avg_params, std_params)
            ])
            ax2.text(0.02, 0.95, param_text, transform=ax2.transAxes, fontsize=10,
                     verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
            ax2.set_title(f"v(t) ± SD - {mode}")
            ax2.set_xlim(0, 3)
            ax2.set_ylim(-2, 4)
            ax2.set_xlabel("Time (s)")
            ax2.grid(True)
            ax2.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

# ======== 总体均值和SD图 ==========
fig, axs = plt.subplots(3, 3, figsize=(15, 12))
plt.suptitle("Overall Participants: Brightness & v(t) Average with SD", fontsize=16)

# 第一行：辉度混合图（每个mode只取一个文件的辉度值即可）
for i, mode in enumerate(["CosineOnly", "LinearOnly", "AcosOnly"]):
    # 只取第一个文件
    for participant, mode_files in participant_files.items():
        files = mode_files.get(mode, [])
        if files:
            path = files[0]
            df = pd.read_csv(path)
            df.columns = df.columns.str.strip()
            time = df["Time"] / 1000
            mask = (time <= 10) & (df["BackFrameNum"] % 2 != 0)
            df.loc[mask, ["BackFrameNum", "FrondFrameNum"]] = df.loc[mask, ["FrondFrameNum", "BackFrameNum"]].to_numpy()
            df.loc[mask, ["BackFrameLuminance", "FrondFrameLuminance"]] = df.loc[mask, ["FrondFrameLuminance", "BackFrameLuminance"]].to_numpy()
            time_plot = np.linspace(0, 3, 300)
            front_interp = np.interp(time_plot, time, df["FrondFrameLuminance"])
            back_interp = np.interp(time_plot, time, df["BackFrameLuminance"])
            ax0 = axs[0, i]
            ax0.plot(time_plot, front_interp, color='tab:blue')
            ax0.plot(time_plot, back_interp, color='tab:orange')
            ax0.set_xlim(0, 3)
            ax0.set_title(f"Luminance Blend - {mode}")
            ax0.set_ylabel("Luminance")
            ax0.grid(True)
            ax0.legend()
            break  # 只取一个文件

# 第二行：v(t) ± SD
for i, mode in enumerate(["CosineOnly", "LinearOnly", "AcosOnly"]):
    all_params = np.array(overall_data[mode])
    if all_params.size == 0:
        continue

    overall_mean = np.mean(all_params, axis=0)
    overall_sd = np.std(all_params, axis=0)

    # 均值±SD曲线
    v_mean = v_curve(overall_mean, t)
    v_upper = v_curve(overall_mean + overall_sd, t)
    v_lower = v_curve(overall_mean - overall_sd, t)

    ax1 = axs[1, i]
    ax1.plot(t, v_mean, color='black', label="Overall Mean v(t)")
    ax1.fill_between(t, v_lower, v_upper, color='gray', alpha=0.3, label="±1 SD")
    ax1.set_xlim(0, 3)
    ax1.set_ylim(-2, 4)
    ax1.set_title(f"Overall v(t) ± SD - {mode}")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Velocity")
    ax1.grid(True)
    ax1.legend()

    # 第三行：参数均值和SD条形图
    ax2 = axs[2, i]
    names = ["V0", "A1", "φ1", "A2", "φ2"]
    bars = ax2.bar(names, overall_mean, yerr=overall_sd, capsize=5, color="lightblue")
    ax2.set_title(f"Overall Params - {mode}")
    ax2.set_ylim(min(overall_mean - overall_sd) - 0.5, max(overall_mean + overall_sd) + 0.5)
    ax2.grid(True, axis='y')
    # 在每个柱子下方显示数值，避免与errorbar重叠
    for idx, (bar, value) in enumerate(zip(bars, overall_mean)):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() - overall_sd[idx] - 0.2,  # 柱子下方
            f"{value:.2f}",
            ha='center',
            va='top',
            fontsize=10
        )

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
