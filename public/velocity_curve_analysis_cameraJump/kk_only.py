"""
速度曲线分析スクリプト
BrightnessFunctionMixAndPhaseDataのデータを分析し、速度曲線を可視化
実験1のLinearOnlyデータのみを分析し、5人の被験者の速度曲線を表示
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import sys

# Ensure UTF-8 output in Windows console
try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def get_A1_from_file(p: Path):
    """从单个 LinearOnly/类似文件中提取 A1（如果可用），不存在或失败返回 np.nan"""
    try:
        if not p.exists():
            return np.nan
        df = pd.read_csv(p)
        df.columns = df.columns.str.strip()
        if "StepNumber" in df.columns and "Amplitude" in df.columns:
            s = df[df["StepNumber"] == 1]["Amplitude"]
            return float(s.iloc[-1]) if not s.empty else np.nan
        # 兼容部分文件把 A1 存在其它列名的情况（尝试 Velocity/Amplitude fallback）
        if "Amplitude1" in df.columns:
            return float(df["Amplitude1"].iloc[-1])
    except Exception as e:
        print(f"Error reading A1 from {p}: {e}")
    return np.nan

def get_params_from_file(p: Path):
    """从文件提取 V0, A1, φ1, A2, φ2，尽量兼容列名差异，失败返回 dict of np.nan"""
    params = {'V0': np.nan, 'A1': np.nan, 'φ1': np.nan, 'A2': np.nan, 'φ2': np.nan}
    try:
        if not p.exists():
            return params
        df = pd.read_csv(p)
        df.columns = df.columns.str.strip()
        if "StepNumber" in df.columns:
            # V0 优先从 Velocity 列取，否则尝试 Amplitude（有些文件格式不同）
            if "Velocity" in df.columns:
                s0 = df[df["StepNumber"] == 0]["Velocity"]
                if not s0.empty:
                    params['V0'] = float(s0.iloc[-1])
            else:
                s0 = df[df["StepNumber"] == 0]["Amplitude"] if "Amplitude" in df.columns else None
                if s0 is not None and not s0.empty:
                    params['V0'] = float(s0.iloc[-1])

            # 后续步骤通常存在于 Amplitude 列（兼容性处理）
            def get_amp(step):
                try:
                    if "Amplitude" in df.columns:
                        s = df[df["StepNumber"] == step]["Amplitude"]
                        return float(s.iloc[-1]) if not s.empty else np.nan
                    # 备选列名
                    for alt in ("Amplitude1", "Amp", "A"):
                        if alt in df.columns:
                            s = df[df["StepNumber"] == step][alt]
                            return float(s.iloc[-1]) if not s.empty else np.nan
                except Exception:
                    pass
                return np.nan

            params['A1'] = get_amp(1)
            params['φ1'] = get_amp(2)
            params['A2'] = get_amp(3)
            params['φ2'] = get_amp(4)
        else:
            # 若无 StepNumber，尝试从列名直接找 Amplitude1 / Phase1 等
            for key, col in (('A1','Amplitude1'),('φ1','Phase1'),('A2','Amplitude2'),('φ2','Phase2'),('V0','Velocity')):
                if col in df.columns:
                    try:
                        params[key] = float(df[col].iloc[-1])
                    except Exception:
                        pass
    except Exception:
        pass
    return params

def analyze_and_plot_A1_comparison(group_paths_list, group_names=None, out_png=None):
    """比较多组文件的 A1 值，画条形图并打印均值
    group_paths_list: list of lists of Path
    """
    if group_names is None:
        group_names = [f"Group{i+1}" for i in range(len(group_paths_list))]
    means = []
    stds = []
    counts = []
    values_all = []
    for paths in group_paths_list:
        vals = []
        for p in paths:
            a1 = get_A1_from_file(p)
            if not np.isnan(a1):
                vals.append(a1)
        vals = np.array(vals, dtype=float)
        values_all.append(vals)
        counts.append(len(vals))
        means.append(float(np.nan) if vals.size == 0 else float(np.nanmean(vals)))
        stds.append(float(np.nan) if vals.size == 0 else float(np.nanstd(vals, ddof=0)))

    # 打印统计信息
    for name, cnt, m, s, vals in zip(group_names, counts, means, stds, values_all):
        if cnt == 0:
            print(f"{name}: no valid A1 values found.")
        else:
            print(f"{name}: n={cnt}, mean A1 = {m:.4f}, std = {s:.4f}, values = {np.round(vals,4).tolist()}")

    # 画条形图（均值 + std error）
    x = np.arange(len(group_names))
    errs = [ (stds[i] / np.sqrt(counts[i])) if counts[i] > 0 else 0.0 for i in range(len(group_names)) ]

    fig, ax = plt.subplots(figsize=(6,4))
    bars = ax.bar(x, means, yerr=errs, capsize=6, color=['#1f77b4','#2ca02c','#ff7f0e'][:len(group_names)])
    ax.set_xticks(x)
    ax.set_xticklabels(group_names)
    ax.set_ylabel('A1')
    ax.set_title('KK: A1 comparison')
    ax.grid(axis='y', alpha=0.3)

    # overlay individual points with jitter
    for i, vals in enumerate(values_all):
        if vals.size == 0:
            continue
        
        # 增加抖动强度，特别是对于相同值的点
        base_jitter = 0.1  # 基础抖动范围
        unique_vals, counts = np.unique(vals, return_counts=True)
        
        # 为每个值生成抖动位置
        x_positions = []
        for val in vals:
            count = counts[unique_vals == val][0]
            if count > 1:
                # 对于重复值，使用更大的抖动
                jitter_strength = base_jitter * (1 + count * 0.3)
            else:
                jitter_strength = base_jitter * 0.5
            
            jitter = np.random.normal(0, jitter_strength)
            x_positions.append(x[i] + jitter)
        
        ax.scatter(x_positions, vals, color='k', alpha=0.8, s=30)

    plt.tight_layout()
    if out_png is None:
        out_png = "KK_A1_comparison.png"
    out_path = Path(out_png).resolve()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved A1 comparison figure to: {out_path}  (exists={out_path.exists()})")

    # 不再吞掉 plt.show 的异常，强制显示（若在无显示环境会抛错，便于诊断）
    try:
        plt.show()
    except Exception as e:
        print("plt.show() failed:", e)
        print("You can open the saved image file manually.")
    plt.close(fig)

def analyze_groups_params_and_plot(group_paths_list, group_names=None, out_png=None):
    """对每组文件提取五个参数，打印每个文件参数与组均值；并绘制 A1 条形比较图。"""
    if group_names is None:
        group_names = [f"Group{i+1}" for i in range(len(group_paths_list))]

    # 收集每组每个参数值
    groups_params = []
    for grp_idx, paths in enumerate(group_paths_list):
        grp_vals = {'V0': [], 'A1': [], 'φ1': [], 'A2': [], 'φ2': []}
        print(f"\n--- Group: {group_names[grp_idx]} ---")
        for p in paths:
            params = get_params_from_file(p)
            print(f"{p.name} exists={p.exists()} -> V0={params['V0']}, A1={params['A1']}, φ1={params['φ1']}, A2={params['A2']}, φ2={params['φ2']}")
            for k in grp_vals:
                if not np.isnan(params[k]):
                    grp_vals[k].append(params[k])
        groups_params.append(grp_vals)

    # 计算并打印每组统计（均值、std、n）
    print("\n=== Group summary ===")
    for name, grp in zip(group_names, groups_params):
        print(f"\n{name}:")
        for k in ('V0','A1','φ1','A2','φ2'):
            arr = np.array(grp[k], dtype=float)
            if arr.size == 0:
                print(f"  {k}: n=0")
            else:
                print(f"  {k}: n={arr.size}, mean={np.nanmean(arr):.4f}, std={np.nanstd(arr):.4f}, values={np.round(arr,4).tolist()}")

    # 为所有参数绘制条形图
    params_to_plot = ['V0', 'A1', 'φ1', 'A2', 'φ2']
    
    # 创建子图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for param_idx, param in enumerate(params_to_plot):
        ax = axes[param_idx]
        
        # 准备当前参数的数据
        param_groups = [np.array(g[param], dtype=float) for g in groups_params]
        
        # 计算统计数据
        means = []
        stds = []
        counts = []
        for vals in param_groups:
            counts.append(len(vals))
            means.append(float(np.nan) if vals.size == 0 else float(np.nanmean(vals)))
            stds.append(float(np.nan) if vals.size == 0 else float(np.nanstd(vals, ddof=0)))

        # 画条形图（均值 + std error）
        x = np.arange(len(group_names))
        errs = [ (stds[i] / np.sqrt(counts[i])) if counts[i] > 0 else 0.0 for i in range(len(group_names)) ]

        bars = ax.bar(x, means, yerr=errs, capsize=6, color=['#1f77b4','#2ca02c','#ff7f0e'][:len(group_names)])
        ax.set_xticks(x)
        ax.set_xticklabels(group_names, rotation=45, ha='right')
        ax.set_ylabel(param)
        ax.set_title(f'KK: {param} comparison')
        ax.grid(axis='y', alpha=0.3)

        # overlay individual points with jitter
        for i, vals in enumerate(param_groups):
            if vals.size == 0:
                continue
            
            # 增加抖动强度，特别是对于相同值的点
            base_jitter = 0.1  # 基础抖动范围
            unique_vals, counts_unique = np.unique(vals, return_counts=True)
            
            # 为每个值生成抖动位置
            x_positions = []
            for val in vals:
                count = counts_unique[unique_vals == val][0]
                if count > 1:
                    # 对于重复值，使用更大的抖动
                    jitter_strength = base_jitter * (1 + count * 0.3)
                else:
                    jitter_strength = base_jitter * 0.5
                
                jitter = np.random.normal(0, jitter_strength)
                x_positions.append(x[i] + jitter)
            
            ax.scatter(x_positions, vals, color='k', alpha=0.8, s=20)

        # 打印当前参数的统计信息
        print(f"\n=== {param} Bar Chart Data ===")
        for name, cnt, m, s, vals in zip(group_names, counts, means, stds, param_groups):
            if cnt == 0:
                print(f"{name}: no valid {param} values found.")
            else:
                print(f"{name}: n={cnt}, mean {param} = {m:.4f}, std = {s:.4f}, values = {np.round(vals,4).tolist()}")

    # 删除多余的子图
    if len(params_to_plot) < len(axes):
        axes[-1].remove()

    plt.tight_layout()
    if out_png is None:
        out_png = "KK_all_parameters_comparison.png"
    out_path = Path(out_png).resolve()
    fig.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved all parameters comparison figure to: {out_path}  (exists={out_path.exists()})")

    # 显示图表
    try:
        plt.show()
    except Exception as e:
        print("plt.show() failed:", e)
        print("You can open the saved image file manually.")
    plt.close(fig)

def main():
    #之前的control数据
    extra_paths1 = [
        Path("../BrightnessData/20250709_152809_Fps1_CameraSpeed1_ExperimentPattern_Phase_ParticipantName_KK_TrialNumber_1_BrightnessBlendMode_LinearOnly.csv"),
        Path("../BrightnessData/20250709_151437_Fps1_CameraSpeed1_ExperimentPattern_Phase_ParticipantName_KK_TrialNumber_3_BrightnessBlendMode_LinearOnly.csv"),
        Path("../BrightnessData/20250709_154001_Fps1_CameraSpeed1_ExperimentPattern_Phase_ParticipantName_KK_TrialNumber_2_BrightnessBlendMode_LinearOnly.csv"),
    ]
    #现在的control数据
    extra_paths2 = [
        Path("../ExperimentData3/20251102_181429_Fps1_CameraSpeed1_ExperimentPattern_Phase_ParticipantName_KK_TrialNumber_1_LinearOnly.csv"),
        Path("../ExperimentData3/20251102_180357_Fps1_CameraSpeed1_ExperimentPattern_Phase_ParticipantName_KK_TrialNumber_2_LinearOnly.csv"),
        Path("../ExperimentData3/20251102_181106_Fps1_CameraSpeed1_ExperimentPattern_Phase_ParticipantName_KK_TrialNumber_3_LinearOnly.csv"),
    ]
    #现在的反方向数据
    extra_paths3 = [
        Path("../ExperimentData33/20251104_155123_Fps1_CameraSpeed1_ExperimentPattern_CameraJumpMove_ParticipantName_KK_TrialNumber_1.csv"),
        Path("../ExperimentData33/20251104_155424_Fps1_CameraSpeed1_ExperimentPattern_CameraJumpMove_ParticipantName_KK_TrialNumber_2.csv"),
        Path("../ExperimentData33/20251104_155959_Fps1_CameraSpeed1_ExperimentPattern_CameraJumpMove_ParticipantName_KK_TrialNumber_3.csv"),
    ]
    # Debug/analysis: 检查文件是否存在并打印列名样例，便于定位为什么没有打印/绘图
    for grp_name, paths in zip(["KK_prev_control", "KK_now_control", "KK_reverse"], [extra_paths1, extra_paths2, extra_paths3]):
        for p in paths:
            print(f"CHECK {grp_name}: {p} exists={p.exists()}")
            if p.exists():
                try:
                    df_sample = pd.read_csv(p, nrows=3)
                    print(f"  sample columns: {list(df_sample.columns)}")
                except Exception as _e:
                    print(f"  failed to read sample: {_e}")

    # 提取并打印 V0,A1,φ1,A2,φ2 并绘制所有参数比较图
    try:
        analyze_groups_params_and_plot(
            [extra_paths1, extra_paths2, extra_paths3],
            group_names=["KK_prev_control", "KK_now_control", "KK_reverse"],
            out_png="KK_all_parameters_comparison.png"
        )
    except Exception as e:
        print("Group analysis failed:", e)



if __name__ == "__main__":
    main()