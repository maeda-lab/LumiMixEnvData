import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

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
            print(f"  试验 {trial_number}: V0={V0:.3f}, A1={A1:.3f}, A2={A2:.3f}")
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

def plot_velocity_curve(V0, A1, φ1, A2, φ2, ax, participant_letter):
    """绘制速度曲线"""
    # 时间轴 (2个周期)
    t = np.linspace(0, 4*np.pi, 1000)
    
    # 速度函数 - 添加π偏移以匹配实验中的公式
    # 实验中使用: v(t) = V0 + A1·sin(ωt + φ1 + π) + A2·sin(2ωt + φ2 + π)
    velocity = V0 + A1 * np.sin(t + φ1 + np.pi) + A2 * np.sin(2*t + φ2 + np.pi)
    
    # 绘制曲线
    ax.plot(t, velocity, 'r-', linewidth=2)
    ax.set_xlabel(f'Participant {participant_letter}', fontsize=12, fontweight='bold')
    ax.set_ylabel('Velocity')
    ax.grid(True, alpha=0.3)
    
    # 限制x轴到2个周期
    ax.set_xlim(0, 4*np.pi)
    ax.set_xticks([0, np.pi, 2*np.pi, 3*np.pi, 4*np.pi])
    ax.set_xticklabels(['0', 'π', '2π', '3π', '4π'])

def main():
    # 数据文件夹路径
    data_folder = Path("../public/BrightnessFunctionMixAndPhaseData")
    
    # 查找Dynamic文件
    dynamic_files = []
    for file in data_folder.glob("*Dynamic*.csv"):
        dynamic_files.append(file)
    
    print(f"找到 {len(dynamic_files)} 个Dynamic文件")
    
    # 按参与者分组
    participants = {}
    for file in dynamic_files:
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
        
        print(f"平均参数: V0={mean_params['V0']:.3f}, A1={mean_params['A1']:.3f}, A2={mean_params['A2']:.3f}")
        
        # 计算速度范围
        t = np.linspace(0, 4*np.pi, 1000)
        velocity = mean_params['V0'] + mean_params['A1'] * np.sin(t + mean_params['φ1'] + np.pi) + mean_params['A2'] * np.sin(2*t + mean_params['φ2'] + np.pi)
        all_velocities.extend(velocity)
    
    # 计算统一的y轴范围
    y_min = min(all_velocities) - 0.1
    y_max = max(all_velocities) + 0.1
    
    # 绘制所有参与者的速度曲线
    for i, participant in enumerate(selected_participants):
        if i >= 5 or participant not in participant_results:
            continue
            
        mean_params = participant_results[participant]
        
        # 绘制速度曲线
        plot_velocity_curve(
            mean_params['V0'], mean_params['A1'], mean_params['φ1'], 
            mean_params['A2'], mean_params['φ2'], axes[i], chr(65 + i) # 使用字母作为标签
        )
        
        # 设置统一的y轴范围
        axes[i].set_ylim(y_min, y_max)
    
    plt.tight_layout()
    plt.savefig('velocity_curves_dynamic_mean.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n速度曲线图已保存为 velocity_curves_dynamic_mean.png")

if __name__ == "__main__":
    main() 