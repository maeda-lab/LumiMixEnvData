import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from glob import glob
from collections import defaultdict
from statistics import mean
from pathlib import Path
import matplotlib.cm as cm  # 新增：导入 colormap
import matplotlib.colors as mcolors

# === 定义混合函数 ===
def cosine_blend(x):
    return 0.5 * (1 - np.cos(np.pi * x))

def linear_blend(x):
    return x

def acos_blend(x):
    return np.arccos(-2 * x + 1) / np.pi

def dynamic_blend(x, knob_value):
    if knob_value <= 0.1:
        return cosine_blend(x)
    elif knob_value <= 0.5:
        t = (knob_value - 0.1) / 0.4
        return (1 - t) * cosine_blend(x) + t * linear_blend(x)
    elif knob_value <= 0.9:
        t = (knob_value - 0.5) / 0.4
        return (1 - t) * linear_blend(x) + t * acos_blend(x)
    else:
        return acos_blend(x)

# === 扫描文件夹中的文件 ===
folder = Path("D:/vectionProject/public/BrightnessData")
pattern = "*_ExperimentPattern_FunctionMix_ParticipantName_*_TrialNumber_*.csv"
files = [f for f in folder.glob(pattern) if "Test" not in f.name]

# === 分组并提取每个 trial 的 FunctionRatio ===
participant_data = defaultdict(list)
for file in files:
    match = re.search(r"ParticipantName_(\w+)_TrialNumber_(\d+)", file.name)
    if match:
        participant = match.group(1)
        df = pd.read_csv(file)
        df.columns = df.columns.str.strip()
        if "FunctionRatio" in df.columns and not df["FunctionRatio"].dropna().empty:
            ratio = df["FunctionRatio"].dropna().iloc[-1]
            participant_data[participant].append(ratio)

# === 准备绘图 ===
x_vals = np.linspace(0, 1, 500)
cos_vals = cosine_blend(x_vals)
lin_vals = linear_blend(x_vals)
acos_vals = acos_blend(x_vals)

# === 每个被试者均值图 + 所有人的总均值图 ===
participants = list(participant_data.keys())
num_participants = len(participants)
cmap = cm.get_cmap('tab10', num_participants)

plt.figure(figsize=(10, 7))
plt.plot(x_vals, cos_vals, '--', label="Cosine")
plt.plot(x_vals, lin_vals, '--', label="Linear")
plt.plot(x_vals, acos_vals, '--', label="Acos")

all_ratios = []
colors = plt.cm.tab10.colors  # 使用 Matplotlib 默认颜色表

for idx, (participant, ratios) in enumerate(participant_data.items()):
    ratios = participant_data[participant]
    if len(ratios) == 0:
        continue
    mean_ratio = mean(ratios)
    dyn_vals = dynamic_blend(x_vals, mean_ratio)
    all_ratios.append(mean_ratio)
    color = colors[idx % len(colors)]
    plt.plot(x_vals, dyn_vals, label=f"{participant} (mean={mean_ratio:.3f})", color=color)

# 所有人平均
if all_ratios:
    overall_mean = mean(all_ratios)
    overall_vals = dynamic_blend(x_vals, overall_mean)
    plt.plot(x_vals, overall_vals, color='black', linewidth=3, label=f"Overall Mean ({overall_mean:.3f})")

plt.title("Mean Dynamic Curves by Participant and Overall")
plt.xlabel("Normalized Time")
plt.ylabel("Blend Value")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()
