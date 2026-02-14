import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_function_mix_data(data_dir):
    """加载FunctionMix数据"""
    pattern = os.path.join(data_dir, "*ExperimentPattern_FunctionMix_ParticipantName_*.csv")
    files = glob.glob(pattern)
    
    all_data = {}
    for file in files:
        # 从文件名提取参与者信息
        filename = os.path.basename(file)
        parts = filename.split('_')
        
        # 提取参与者名称
        participant_idx = parts.index('ParticipantName') + 1
        participant = parts[participant_idx]
        
        # 提取试验编号
        trial_idx = parts.index('TrialNumber') + 1
        trial_str = parts[trial_idx]
        if trial_str.endswith('.csv'):
            trial_str = trial_str.replace('.csv', '')
        trial = int(trial_str)
        
        try:
            df = pd.read_csv(file)
            # 清理列名
            df.columns = df.columns.str.strip()
            df['Participant'] = participant
            df['Trial'] = trial
            df['Filename'] = filename
            
            if participant not in all_data:
                all_data[participant] = {}
            all_data[participant][trial] = df
            
            print(f"加载成功: {filename} - 参与者: {participant}, 试验: {trial}")
        except Exception as e:
            print(f"加载错误: {filename} - {e}")
    
    return all_data

def plot_mixing_functions(ax, function_type, direction='forward'):
    """绘制混合函数"""
    x = np.linspace(0, 1, 100)
    
    if function_type == 'cos':
        if direction == 'forward':
            # cos: 0.5f * (1f - Mathf.Cos(Mathf.PI * x)) - 从0到1
            y = 0.5 * (1 - np.cos(np.pi * x))
        else:
            # 从1到0变化
            y = 0.5 * (1 + np.cos(np.pi * x))
        title = 'Cos'
        color = 'green'
    elif function_type == 'linear':
        if direction == 'forward':
            # linear: x - 从0到1
            y = x
        else:
            # 从1到0变化
            y = 1 - x
        title = 'Linear'
        color = 'blue'
    elif function_type == 'acos':
        if direction == 'forward':
            # acos: Mathf.Acos(-2f * x + 1f) / Mathf.PI - 从0到1
            y = np.arccos(-2 * x + 1) / np.pi
        else:
            # 从1到0变化
            y = 1 - np.arccos(-2 * x + 1) / np.pi
        title = 'Acos '
        color = 'darkred'
    
    ax.plot(x, y, color=color, linewidth=2)
    ax.set_xlabel('t')
    ax.set_ylabel('')
    ax.set_title('')  # 移除标题
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])  # 不显示任何横轴刻度
    ax.set_yticks([0, 1])

def extract_final_function_ratio(df):
    """提取最终的FunctionRatio值"""
    # 取最后20%的数据的平均值作为最终值
    final_data = df.tail(int(len(df) * 0.2))
    final_ratio = final_data['FunctionRatio'].iloc[-1]
    return final_ratio

def create_function_mix_plot(data_dir):
    """创建FunctionMix分析图"""
    print("加载FunctionMix数据...")
    function_mix_data = load_function_mix_data(data_dir)
    
    if not function_mix_data:
        print("没有找到FunctionMix数据")
        return
    
    # 参与者顺序（根据截图）
    participants_order = ['ONO', 'OMU', 'YAMA', 'HOU', 'LL']
    participant_labels = ['A', 'B', 'C', 'D', 'E']
    
    # 创建图形：使用GridSpec来调整子图大小比例
    fig = plt.figure(figsize=(15, 6))  # 高度降低一半
    gs = fig.add_gridspec(3, 2, width_ratios=[0.2, 0.8], height_ratios=[1, 1, 1], hspace=0.3)  # 左边占20%，右边占80%，子图间距均匀
    
    # 左侧：混合函数图 - 显示两种变化方向
    ax1 = fig.add_subplot(gs[0, 0])
    plot_mixing_functions(ax1, 'acos', 'forward')
    plot_mixing_functions(ax1, 'acos', 'reverse')
    
    ax2 = fig.add_subplot(gs[1, 0])
    plot_mixing_functions(ax2, 'linear', 'forward')
    plot_mixing_functions(ax2, 'linear', 'reverse')
    
    ax3 = fig.add_subplot(gs[2, 0])
    plot_mixing_functions(ax3, 'cos', 'forward')
    plot_mixing_functions(ax3, 'cos', 'reverse')
    
    # 右侧：参与者数据散点图（合并为一张图）
    ax4 = fig.add_subplot(gs[:, 1])  # 跨越右侧所有行
    
    # 收集所有数据用于计算y轴范围
    all_ratios = []
    
    # 为每个参与者绘制散点图
    # ABC使用一个颜色，DE使用另一个颜色
    colors = ['#1f77b4', '#1f77b4', '#1f77b4', '#ff7f0e', '#ff7f0e']  # ABC蓝色，DE橙色
    
    for i, (participant, label) in enumerate(zip(participants_order, participant_labels)):
        if participant in function_mix_data:
            trials = function_mix_data[participant]
            
            # 提取每个试验的最终FunctionRatio值
            ratios = []
            for trial_num in sorted(trials.keys()):
                df = trials[trial_num]
                final_ratio = extract_final_function_ratio(df)
                ratios.append(final_ratio)
                all_ratios.append(final_ratio)
            
            # 计算中位数和误差
            median_ratio = np.median(ratios)
            std_ratio = np.std(ratios)
            sem_ratio = std_ratio / np.sqrt(len(ratios))  # 标准误差
            
            # 绘制散点图，错开位置避免重叠
            x_positions = np.linspace(-0.15, 0.15, len(ratios))  # 在标签位置左右错开
            x_coords = [i + x_pos for x_pos in x_positions]
            
            ax4.scatter(x_coords, ratios, color=colors[i], alpha=0.7, s=60, edgecolors='white', linewidth=1)
            # 绘制中位数线和误差条
            ax4.plot([i], [median_ratio], 'o', color=colors[i], markersize=10, markeredgecolor='black', markeredgewidth=2)
            ax4.errorbar([i], [median_ratio], yerr=sem_ratio, color=colors[i], capsize=5, capthick=2, linewidth=2)
            
            # 添加标签
            ax4.text(i + 0.08, float(median_ratio), 'median', fontsize=8, ha='left', va='center')
            ax4.text(i, float(median_ratio + sem_ratio + 0.02), '95% CI', fontsize=8, ha='center', va='bottom')
    
    # 设置右侧子图的属性
    ax4.set_xlabel('Participant')
    ax4.set_ylabel('')  # 去掉纵轴标签
    ax4.set_title('Function Mixing Ratio by Participant')
    ax4.grid(False)  # 移除网格
    ax4.set_xticks(range(len(participant_labels)))
    ax4.set_xticklabels(participant_labels)
    ax4.set_yticks([0.0, 0.1, 0.5, 0.9, 1.0])  # 只显示指定的刻度值
    
    # 添加横线
    horizontal_lines = [0.0, 0.1, 0.5, 0.9, 1.0]
    for line_y in horizontal_lines:
        if line_y in [0.0, 1.0]:
            # 0.0和1.0的横线颜色为灰色
            ax4.axhline(y=line_y, color='gray', alpha=0.6, linewidth=1)
        else:
            # 0.1, 0.5, 0.9的横线颜色为黑色
            ax4.axhline(y=line_y, color='black', alpha=0.8, linewidth=1)
    
    # 在指定位置添加函数名称
    ax4.text(-0.5, 0.1, 'Cos', fontsize=10, ha='right', va='center', color='green')
    ax4.text(-0.5, 0.5, 'Linear', fontsize=10, ha='right', va='center', color='blue')
    ax4.text(-0.5, 0.9, 'Arccos', fontsize=10, ha='right', va='center', color='darkred')
    
    # 在右图右侧添加竖写标签
    fig.text(0.91, 0.5, 'Normalized Subjective response [arb. unit]', fontsize=12, 
             ha='center', va='center', rotation=90)
    
    # 设置y轴范围
    if all_ratios:
        y_min = min(all_ratios) - 0.05
        y_max = max(all_ratios) + 0.05
        ax4.set_ylim(max(0, y_min), min(1, y_max))
    
    plt.tight_layout()
    plt.savefig('function_mix_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 打印统计信息
    print("\n=== FunctionMix分析结果 ===")
    for i, (participant, label) in enumerate(zip(participants_order, participant_labels)):
        if participant in function_mix_data:
            trials = function_mix_data[participant]
            ratios = []
            for trial_num in sorted(trials.keys()):
                df = trials[trial_num]
                final_ratio = extract_final_function_ratio(df)
                ratios.append(final_ratio)
            
            median_ratio = np.median(ratios)
            mean_ratio = np.mean(ratios)
            std_ratio = np.std(ratios)
            
            print(f"参与者 {label} ({participant}):")
            print(f"  试验值: {[f'{r:.3f}' for r in ratios]}")
            print(f"  中位数: {median_ratio:.3f}")
            print(f"  平均值: {mean_ratio:.3f}")
            print(f"  标准差: {std_ratio:.3f}")
            print()
    
    print("FunctionMix分析图已保存为 function_mix_analysis.png")

def main():
    """主函数"""
    data_dir = "../BrightnessFunctionMixAndPhaseData"
    
    if not os.path.exists(data_dir):
        print(f"数据目录不存在: {data_dir}")
        return
    
    create_function_mix_plot(data_dir)

if __name__ == "__main__":
    main() 