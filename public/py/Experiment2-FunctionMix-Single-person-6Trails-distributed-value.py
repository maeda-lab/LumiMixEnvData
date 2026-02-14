import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from collections import defaultdict
from pathlib import Path
from collections import defaultdict

# ==== 设置数据文件夹路径 ====
folder = Path("D:/vectionProject/public/BrightnessFunctionMixAndPhaseData")  # 修改为你的数据路径
pattern = "*_ExperimentPattern_FunctionMix_ParticipantName_*_TrialNumber_*.csv"
files = [f for f in folder.glob(pattern) if "Test" not in f.name]

# ==== 提取每个参与者的 FunctionRatio ====
participant_data = defaultdict(list)
for file in files:
    match = re.search(r"ParticipantName_(\w+)_TrialNumber_(\d+)", file.name)
    if match:
        participant = match.group(1)
        trial_num = int(match.group(2))
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip()
        if "FunctionRatio" in df.columns and not df["FunctionRatio"].dropna().empty:
            ratio = df["FunctionRatio"].dropna().iloc[-1]
            participant_data[participant].append((trial_num, ratio))

# ==== 数据整理 ====
participant_labels = []
ratios_all = []

for participant, trials in participant_data.items():
    trials = sorted(trials, key=lambda x: x[0])  # trial_num排序
    ratios = [r for _, r in trials]
    participant_labels.append(participant)
    ratios_all.append(ratios)

# ==== 绘图 ====
plt.figure(figsize=(10, 6))

x_ticks = []
x_tick_labels = []

for i, (participant, ratios) in enumerate(zip(participant_labels, ratios_all)):
    bins = defaultdict(int)
    for r in ratios:
        key = round(r, 2)
        count = bins[key]
        spread = 0.1
        x_offset = spread * (count - (bins[key] - 1) / 2)
        bins[key] += 1
        x_pos = i + x_offset
        plt.scatter(x_pos, r, color='blue', zorder=3)
        plt.text(x_pos, r + 0.015, f"{r:.2f}", ha='center', va='bottom', fontsize=9)
    x_ticks.append(i)
    x_tick_labels.append(participant)

# 区间背景色
plt.axhspan(0.9, 1.0, facecolor='mistyrose', alpha=0.3, label='Acos (0.9–1.0)')
plt.axhspan(0.5, 0.9, facecolor='lightgray', alpha=0.3, label='Linear->Acos')
plt.axhspan(0.5, 0.5, facecolor='gray', alpha=0.3, label='Linear (0.5)')
plt.axhspan(0.1, 0.5, facecolor='lightgreen', alpha=0.3, label='Cosine->Acos')
plt.axhspan(0.0, 0.1, facecolor='lightblue', alpha=0.3, label='Cosine (0.0–0.1)')

# 坐标轴设定
plt.xticks(ticks=x_ticks, labels=x_tick_labels)
plt.ylim(0, 1.05)
plt.yticks([0, 0.1, 0.5, 0.9, 1.0])
plt.ylabel("FunctionRatio")
plt.xlabel("Participant")
plt.title("FunctionRatio values across 6 trials per participant")
plt.grid(axis='y', zorder=0)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()