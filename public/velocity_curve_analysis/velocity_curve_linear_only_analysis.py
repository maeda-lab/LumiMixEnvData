#!/usr/bin/env python3
"""
速度曲線分析スクリプト
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

def extract_parameters(df):
    """从数据中提取速度参数"""
    # 清理列名（去除空格）
    df.columns = df.columns.str.strip()
    
    # 参数提取 - 使用正确的StepNumber方法
    v0_series = df[df["StepNumber"] == 0]["Velocity"]
    V0 = v0_series.iloc[-1] if not v0_series.empty else 0
    
    A1 = df[df["StepNumber"] == 1]["Amplitude"].iloc[-1] if not df[df["StepNumber"] == 1].empty else 0
    φ1 = df[df["StepNumber"] == 2]["Amplitude"].iloc[-1] if not df[df["StepNumber"] == 2].empty else 0
    A2 = df[df["StepNumber"] == 3]["Amplitude"].iloc[-1] if not df[df["StepNumber"] == 3].empty else 0
    φ2 = df[df["StepNumber"] == 4]["Amplitude"].iloc[-1] if not df[df["StepNumber"] == 4].empty else 0
    
    return V0, A1, φ1, A2, φ2

def calculate_mean_parameters(participant_files):
    """计算参与者的平均参数"""
    all_params = []
    
    for file_path, trial_number in participant_files:
        try:
            df = pd.read_csv(file_path)
            V0, A1, φ1, A2, φ2 = extract_parameters(df)
            all_params.append({
                'V0': V0, 'A1': A1, 'φ1': φ1, 'A2': A2, 'φ2': φ2,
                'trial': trial_number, 'file': file_path.name
            })
            print(f"  试验 {trial_number}: V0={V0:.3f}, A1={A1:.3f}, φ1={φ1:.3f},A2={A2:.3f}, φ2={φ2:.3f}")
        except Exception as e:
            print(f"  试验 {trial_number} 处理出错: {e}")
    
    if not all_params:
        return None
    
    # 计算平均参数
    mean_params = {
        'V0': np.mean([p['V0'] for p in all_params]),
        'A1': np.mean([p['A1'] for p in all_params]),
        'φ1': np.mean([p['φ1'] for p in all_params]),
        'A2': np.mean([p['A2'] for p in all_params]),
        'φ2': np.mean([p['φ2'] for p in all_params])
    }
    
    return mean_params, all_params

def plot_velocity_curve(V0, A1, φ1, A2, φ2, ax, participant_letter, fill_color=None):
    """绘制速度曲线，时间轴为秒 [0, 2]（两秒、两周期）"""
    # 时间轴（秒）：两秒覆盖两周期 → 基本角频率为 2π rad/s
    t_sec = np.linspace(0.0, 2.0, 1000)
    
    # 速度函数 - 使用秒为单位，sin(2π t) 对应 1 Hz；第二谐波为 sin(4π t)
    velocity = (
        V0
        + A1 * np.sin(2 * np.pi * t_sec + φ1 + np.pi)
        + A2 * np.sin(4 * np.pi * t_sec + φ2 + np.pi)
    )
    
    # 绘制曲线
    ax.plot(t_sec, velocity, 'b-', linewidth=2)
    ax.set_title(f'Participant {participant_letter}', fontsize=12, fontweight='bold')
    ax.set_xlabel('t(s)')
    ax.set_ylabel('Velocity')
    ax.grid(True, alpha=0.3)
    
    # 填充第一个周期区域 [0, 1]
    if fill_color:
        ax.axvspan(0.0, 1.0, color=fill_color, alpha=0.15, linewidth=0)
    
    # x 轴为 0–2 秒，并标注 0, 0.5, 1, 1.5, 2
    ax.set_xlim(0.0, 2.0)
    ax.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])
    ax.set_xticklabels(['0', '0.5', '1', '1.5', '2'])
        # 在图中显示参数值
    param_text = (
        f"V0={V0:.2f}\n"
        f"A1={A1:.2f}, φ1={φ1:.2f}\n"
        f"A2={A2:.2f}, φ2={φ2:.2f}"
    )
    ax.text(
        0.02, 0.95, param_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment='top',
        bbox=dict(facecolor='white', alpha=0.6, edgecolor='none')
    )


def main():
    # 数据文件夹路径
    data_folder = Path("../BrightnessFunctionMixAndPhaseData")
    
    # 查找LinearOnly文件
    linear_only_files = []
    for file in data_folder.glob("*LinearOnly*.csv"):
        linear_only_files.append(file)
    
    print(f"找到 {len(linear_only_files)} 个LinearOnly文件")
    if len(linear_only_files) == 0:
        print("未找到 LinearOnly 数据文件，结束。")
        return
    
    # 按参与者分组
    participants = {}
    for file in linear_only_files:
        filename = file.name
        # 提取参与者名称和试验编号
        if "ParticipantName_" in filename:
            participant = filename.split("ParticipantName_")[1].split("_")[0]
            trial_match = filename.split("TrialNumber_")[1].split("_")[0]
            trial_number = int(trial_match)
            
            if participant not in participants:
                participants[participant] = []
            participants[participant].append((file, trial_number))
    
    # 选择前5个参与者，并重新排序
    all_participants = list(participants.keys())
    # 将HOU和LL放在第4和第5位置
    selected_participants = []
    
    # 先添加其他参与者（前3个位置）
    for participant in all_participants:
        if participant not in ['HOU', 'LL'] and len(selected_participants) < 3:
            selected_participants.append(participant)
    
    # 添加HOU和LL到第4和第5位置
    if 'HOU' in all_participants:
        selected_participants.append('HOU')
    if 'LL' in all_participants:
        selected_participants.append('LL')
    
    # 如果还不够5个，继续添加其他参与者
    for participant in all_participants:
        if participant not in selected_participants and len(selected_participants) < 5:
            selected_participants.append(participant)
    
    print(f"选择的参与者顺序: {selected_participants}")
    
    # 创建子图 - 改为一行5列
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    # 透明背景设置
    try:
        fig.patch.set_alpha(0.0)
        for ax in axes:
            ax.set_facecolor('none')
    except Exception:
        pass
    
    # 先计算所有参与者的参数，以确定y轴范围
    all_velocities = []
    participant_results = {}
    
    for i, participant in enumerate(selected_participants):
        if i >= 5:  # 只处理前5个参与者
            break
            
        print(f"\n处理参与者 {participant}:")
        
        # 计算平均参数
        result = calculate_mean_parameters(participants[participant])
        if result is None:
            print(f"参与者 {participant} 没有有效数据")
            continue
            
        mean_params, all_params = result
        participant_results[participant] = mean_params
        
        print(f"平均参数: V0={mean_params['V0']:.3f}, A1={mean_params['A1']:.3f}, φ1={mean_params['φ1']:.3f}, A2={mean_params['A2']:.3f}, φ2={mean_params['φ2']:.3f}")
        
        # 计算速度范围（以秒为横轴的两秒域）
        t_sec = np.linspace(0.0, 2.0, 1000)
        velocity = (
            mean_params['V0']
            + mean_params['A1'] * np.sin(2 * np.pi * t_sec + mean_params['φ1'] + np.pi)
            + mean_params['A2'] * np.sin(4 * np.pi * t_sec + mean_params['φ2'] + np.pi)
        )
        all_velocities.extend(velocity)
    
    # 若没有任何可用数据，退出
    if len(all_velocities) == 0 or len(participant_results) == 0:
        print("没有可用于绘图的参与者数据，结束。")
        return

    # 计算统一的y轴范围
    y_min = min(all_velocities) - 0.1
    y_max = max(all_velocities) + 0.1
    
    # 绘制所有参与者的速度曲线
    for i, participant in enumerate(selected_participants):
        if i >= 5 or participant not in participant_results:
            continue
            
        mean_params = participant_results[participant]
        
        # 颜色：前3个蓝色，后2个黄色
        fill_color = '#1f77b4' if i < 3 else '#ffeb3b'
        plot_velocity_curve(
            mean_params['V0'], mean_params['A1'], mean_params['φ1'], 
            mean_params['A2'], mean_params['φ2'], axes[i], chr(65 + i), fill_color
        )
        
        # 设置统一的y轴范围
        axes[i].set_ylim(y_min, y_max)
    
    plt.tight_layout()
    plt.savefig('velocity_curves_linear_only_mean_background_opaque.png', dpi=300, bbox_inches='tight', transparent=True)
    try:
        plt.show()
    except Exception:
        pass
    
    print("\n速度曲线图已保存为 velocity_curves_linear_only_mean_background_opaque.png")

if __name__ == "__main__":
    main() 