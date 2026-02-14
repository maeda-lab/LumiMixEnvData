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

def plot_combined_velocity_curves(linear_params, dynamic_params, ax, participant_letter):
    """绘制两种条件的速度曲线"""
    # 时间轴 (2个周期)
    t = np.linspace(0, 4*np.pi, 1000)
    
    # 绘制LinearOnly曲线（蓝色）
    if linear_params:
        velocity_linear = linear_params['V0'] + linear_params['A1'] * np.sin(t + linear_params['φ1'] + np.pi) + linear_params['A2'] * np.sin(2*t + linear_params['φ2'] + np.pi)
        ax.plot(t, velocity_linear, 'b-', linewidth=2, label='LinearOnly')
    
    # 绘制Dynamic曲线（红色）
    if dynamic_params:
        velocity_dynamic = dynamic_params['V0'] + dynamic_params['A1'] * np.sin(t + dynamic_params['φ1'] + np.pi) + dynamic_params['A2'] * np.sin(2*t + dynamic_params['φ2'] + np.pi)
        ax.plot(t, velocity_dynamic, 'r-', linewidth=2, label='Dynamic')
    
    ax.set_xlabel(f'Participant {participant_letter}', fontsize=12, fontweight='bold')
    ax.set_ylabel('Velocity')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 限制x轴到2个周期
    ax.set_xlim(0, 4*np.pi)
    ax.set_xticks([0, np.pi, 2*np.pi, 3*np.pi, 4*np.pi])
    ax.set_xticklabels(['0', 'π', '2π', '3π', '4π'])

def load_condition_data(data_folder, condition_type):
    """加载指定条件的数据"""
    files = []
    for file in data_folder.glob(f"*{condition_type}*.csv"):
        files.append(file)
    
    print(f"找到 {len(files)} 个{condition_type}文件")
    
    # 按参与者分组
    participants = {}
    for file in files:
        filename = file.name
        # 提取参与者名称和试验编号
        if "ParticipantName_" in filename:
            participant = filename.split("ParticipantName_")[1].split("_")[0]
            trial_match = filename.split("TrialNumber_")[1].split("_")[0]
            trial_number = int(trial_match)
            
            if participant not in participants:
                participants[participant] = []
            participants[participant].append((file, trial_number))
    
    return participants

def main():
    # 数据文件夹路径
    data_folder = Path("../public/BrightnessFunctionMixAndPhaseData")
    
    # 加载两种条件的数据
    linear_participants = load_condition_data(data_folder, "LinearOnly")
    dynamic_participants = load_condition_data(data_folder, "Dynamic")
    
    # 选择前5个参与者，并重新排序
    all_participants = list(set(list(linear_participants.keys()) + list(dynamic_participants.keys())))
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
    
    # 创建子图 - 一行5列
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    # 先计算所有参与者的参数，以确定y轴范围
    all_velocities = []
    participant_results = {}
    
    for i, participant in enumerate(selected_participants):
        if i >= 5:  # 只处理前5个参与者
            break
            
        print(f"\n处理参与者 {participant}:")
        participant_results[participant] = {}
        
        # 计算LinearOnly平均参数
        if participant in linear_participants:
            print("  LinearOnly条件:")
            linear_result = calculate_mean_parameters(linear_participants[participant])
            if linear_result:
                linear_params, _ = linear_result
                participant_results[participant]['linear'] = linear_params
                print(f"    平均参数: V0={linear_params['V0']:.3f}, A1={linear_params['A1']:.3f}, A2={linear_params['A2']:.3f}")
                
                # 计算速度范围
                t = np.linspace(0, 4*np.pi, 1000)
                velocity = linear_params['V0'] + linear_params['A1'] * np.sin(t + linear_params['φ1'] + np.pi) + linear_params['A2'] * np.sin(2*t + linear_params['φ2'] + np.pi)
                all_velocities.extend(velocity)
        
        # 计算Dynamic平均参数
        if participant in dynamic_participants:
            print("  Dynamic条件:")
            dynamic_result = calculate_mean_parameters(dynamic_participants[participant])
            if dynamic_result:
                dynamic_params, _ = dynamic_result
                participant_results[participant]['dynamic'] = dynamic_params
                print(f"    平均参数: V0={dynamic_params['V0']:.3f}, A1={dynamic_params['A1']:.3f}, A2={dynamic_params['A2']:.3f}")
                
                # 计算速度范围
                t = np.linspace(0, 4*np.pi, 1000)
                velocity = dynamic_params['V0'] + dynamic_params['A1'] * np.sin(t + dynamic_params['φ1'] + np.pi) + dynamic_params['A2'] * np.sin(2*t + dynamic_params['φ2'] + np.pi)
                all_velocities.extend(velocity)
    
    # 计算统一的y轴范围
    y_min = min(all_velocities) - 0.1
    y_max = max(all_velocities) + 0.1
    
    # 绘制所有参与者的速度曲线
    for i, participant in enumerate(selected_participants):
        if i >= 5 or participant not in participant_results:
            continue
            
        linear_params = participant_results[participant].get('linear', None)
        dynamic_params = participant_results[participant].get('dynamic', None)
        
        # 绘制速度曲线
        plot_combined_velocity_curves(linear_params, dynamic_params, axes[i], chr(65 + i))
        
        # 设置统一的y轴范围
        axes[i].set_ylim(y_min, y_max)
    
    plt.tight_layout()
    plt.savefig('velocity_curves_combined.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n速度曲线图已保存为 velocity_curves_combined.png")

if __name__ == "__main__":
    main() 