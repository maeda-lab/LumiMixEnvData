import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import re
from collections import defaultdict

# ========== 设置路径和读取文件 ==========
root_dir = "D:/vectionProject/public/BrightnessFunctionMixAndPhaseData"  # ChatGPT环境路径
pattern = re.compile(r"ExperimentPattern_FunctionMix_ParticipantName_(\w+)_TrialNumber_(\w+)\.csv")
print(f"Looking for files in: {root_dir}")

participant_files = defaultdict(lambda: defaultdict(list))
for fname in os.listdir(root_dir):
    if fname.endswith(".csv") and "Test" not in fname:
        match = pattern.search(fname)
        if match:
            participant, mode = match.groups()
            full_path = os.path.join(root_dir, fname)
            print(f"Found file: {full_path} for participant: {participant}, mode: {mode}")
            participant_files[participant][mode].append(full_path)

# ========== 设置参与者和模式 ==========
selected_participant = "O"
selected_mode = "FunctionMix"
selected_files = participant_files[selected_participant]
file_paths = [(path, f"Trial {i+1}") for i, path in enumerate(selected_files)]

# ========== 定义函数 ==========
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

# ========== 定义区间 ==========
intervals = [(0, 0.1), (0.1, 0.5), (0.5, 0.9), (0.9, 1.0)]
interval_labels = ["0", "1", "2", "3", "4"]
values_per_trial = {label: [] for label in interval_labels}

# ========== 读取实际 ratio 并归类 ==========
for path, label in file_paths:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    if "FunctionRatio" not in df.columns or df["FunctionRatio"].dropna().empty:
        print(f"Trial: {label}, FunctionRatio column missing or empty in {path}")
        continue
    ratio = df["FunctionRatio"].dropna().iloc[-1]
    print(f"Trial: {label}, Function Ratio: {ratio}")
    for i, (start, end) in enumerate(intervals):
        if start <= ratio < end:
            val = dynamic_blend(0.5, ratio)
            values_per_trial[interval_labels[i]].append(val)
            break

# ========== 画图 ==========
plt.figure(figsize=(12, 7))
data = [values_per_trial[label] for label in interval_labels]
plt.boxplot(data, tick_labels=interval_labels, showmeans=True)

for i, vals in enumerate(data):
    plt.scatter([i+1]*len(vals), vals, color='red', zorder=3, label='FDG' if i==0 else "")

plt.ylabel("Blend Value")
plt.title("Function Mix Blend Values by Interval")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
