import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os
import glob
from pathlib import Path
from matplotlib.legend_handler import HandlerTuple, HandlerErrorbar
from matplotlib.container import ErrorbarContainer


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
    
    # 创建图形：只保留右侧参与者数据散点图
    fig = plt.figure(figsize=(12, 6))  # 宽度调整，去掉左侧空间
    # 设置透明背景
    fig.patch.set_facecolor('none')
    fig.patch.set_alpha(0.0)
    ax4 = plt.subplot(111)  # 单个子图占满整个图形
    ax4.set_facecolor('none')
    ax4.patch.set_alpha(0.0)
    
    # 收集所有数据用于计算y轴范围
    all_ratios = []
    
    # 为每个参与者绘制散点图
    # ABC使用一个颜色，DE使用另一个颜色
    colors = ['#1f77b4', '#1f77b4', '#1f77b4', '#ff7f0e', '#ff7f0e']  # ABC蓝色，DE橙色
    
    error_legend_handle = None
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
            
            # 绘制散点图（只在第一次绘制时添加标签用于图例）
            if i == 0:
                scatter = ax4.scatter(x_coords, ratios, color=colors[i], alpha=0.7, s=60, edgecolors='white', linewidth=1, label='Individual trials')
            else:
                ax4.scatter(x_coords, ratios, color=colors[i], alpha=0.7, s=60, edgecolors='white', linewidth=1)
            
            # 绘制中位数线和误差条（只在第一次绘制时添加标签用于图例）
            if i == 0:
                median_line = ax4.plot([i], [median_ratio], 'o', color=colors[i], markersize=10, markeredgecolor='black', markeredgewidth=2, label='Median')[0]
                error_legend_handle = ax4.errorbar([i], [median_ratio], yerr=sem_ratio, color=colors[i], capsize=5, capthick=2, linewidth=2)
            else:
                ax4.plot([i], [median_ratio], 'o', color=colors[i], markersize=10, markeredgecolor='black', markeredgewidth=2)
                ax4.errorbar([i], [median_ratio], yerr=sem_ratio, color=colors[i], capsize=5, capthick=2, linewidth=2)
    
    # 设置子图的属性
    ax4.set_xlabel('Participant')
    ax4.set_ylabel('')  # 去掉纵轴标签
    ax4.set_title('Preferred mixing ratio by participant (median ± 95% CI)')
    ax4.grid(False)  # 移除网格
    ax4.set_xticks(range(len(participant_labels)))
    ax4.set_xticklabels(participant_labels)
    ax4.set_yticks([0.0, 0.1, 0.5, 0.9, 1.0])  # 只显示指定的刻度值
    
    # 合并图例（颜色分组的 trial + 实验元素说明）
    blue_color = '#1f77b4'
    orange_color = '#ff7f0e'
    # 颜色分组的 trial 代理
    h_trial_blue = mlines.Line2D([], [], color=blue_color, marker='o', linestyle='None', markersize=6, label='Trial (Acos preference group)')
    h_trial_orange = mlines.Line2D([], [], color=orange_color, marker='o', linestyle='None', markersize=6, label='Trial (Cos preference group)')
    # 实验元素代理
    h_median_blue = mlines.Line2D([], [], color=blue_color, marker='o', linestyle='None', markersize=8, markeredgecolor='black', markeredgewidth=2, label='Median (outlined dot)')
    h_median_orange = mlines.Line2D([], [], color=orange_color, marker='o', linestyle='None', markersize=8, markeredgecolor='black', markeredgewidth=2, label='')
    # 图例中的误差棒：垂直 yerr，带端帽；中心用小的空心圆点，提升辨识度（仅用于图例，空数据不绘制到图上）
    h_error = ax4.errorbar(
        [0], [0], yerr=[[0.05], [0.05]],
        fmt='o', mfc='white', mec='black', mew=1.2, ms=5,
        ecolor='black', elinewidth=2, capsize=5, capthick=2,
        label='Error bar (95% CI)'
    )
    # 2) 建立图例，让它把这条误差棒当作示例（会自动绘制竖线+上下短横线）
    leg = ax4.legend(handler_map={type(h_error): HandlerErrorbar()})

    # 3) 如果不想这条“代理误差棒”出现在图上，创建图例后把它从坐标轴移除
    h_error[0].remove()                 # 中心的小圆点
    for c in h_error[1]: c.remove()     # 上下两个端帽（短横线）
    for col in h_error[2]: col.remove() # 竖直误差线(LineCollection)


    median_pair = (h_median_blue, h_median_orange)
    handles = [h_trial_blue, h_trial_orange, median_pair, h_error]
    labels = ['Trial (Arccos preference group)', 'Trial (Cos preference group)', 'Median (outlined dot)', 'Error bar (95% CI)']
    ax4.legend(
        handles=handles,
        labels=labels,
        loc='upper right', bbox_to_anchor=(1.0, 1.0),
        ncol=1, columnspacing=0.8, handletextpad=0.6, borderaxespad=0.3,
        handler_map={tuple: HandlerTuple(ndivide=None, pad=0.3), ErrorbarContainer: HandlerErrorbar()}
    )



    # 分组强调：在 C 与 D 之间画虚线分隔（x=2.5），并在下方添加组标签
    ax4.axvline(x=2.5, color='gray', linestyle='--', linewidth=1, alpha=0.6)
    ax4.text(0.25, -0.12, 'A–C: Arccos preference', transform=ax4.transAxes, ha='center', va='top', fontsize=10)
    ax4.text(0.875, -0.12, 'D–E: Cos preference', transform=ax4.transAxes, ha='center', va='top', fontsize=10)
    
    # 添加横线
    horizontal_lines = [0.0, 0.1, 0.5, 0.9, 1.0]
    for line_y in horizontal_lines:
        if line_y in [0.0, 1.0]:
            # 0.0和1.0的横线颜色为灰色
            ax4.axhline(y=line_y, color='gray', alpha=0.6, linewidth=1)
        else:
            # 0.1, 0.5, 0.9的横线颜色为黑色
            ax4.axhline(y=line_y, color='black', alpha=0.8, linewidth=1)
    
    # 在指定位置添加区域标注（统一黑色，配合背景色带）
    ax4.text(-0.5, 0.1, 'Cos region (≤)', fontsize=10, ha='right', va='center', color='black')
    ax4.text(-0.5, 0.5, 'Linear region (=)', fontsize=10, ha='right', va='center', color='black')
    ax4.text(-0.5, 0.9, 'Arccos region (≥)', fontsize=10, ha='right', va='center', color='black')
    # 移除左上角的文字说明，改为合并图例

    regions = [(0.0, 0.1), (0.1, 0.5), (0.5, 0.9), (0.9, 1.0)]
    colors  = ['#d0f0d0', '#f0f0ff', '#fff0f0', '#ffe0b3']  # 可以手工挑选
    alphas  = [0.05, 0.2, 0.2, 0.05]  # 上下稍微深一点

    for (ymin, ymax), color, a in zip(regions, colors, alphas):
        ax4.axhspan(ymin, ymax, facecolor=color, alpha=a, zorder=0)

    # 在右侧添加竖写标签
    # 创建右侧的 twin 轴
    ax_right = ax4.twinx()

    # 设置右侧纵轴标签
    ax_right.set_ylabel('Normalized Subjective response [arb. unit]',
                        fontsize=12, rotation=270, labelpad=25)

    # 隐藏右轴的刻度和刻度线（只保留标签）
    ax_right.tick_params(right=False, labelright=False)

    # 设置y轴范围
    if all_ratios:
        y_min = min(all_ratios) - 0.05
        y_max = max(all_ratios) + 0.05
        ax4.set_ylim(max(0, y_min), min(1, y_max))
    
    plt.tight_layout()
    plt.savefig('function_mix_analysis.png', dpi=300, bbox_inches='tight', transparent=True)
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
    
    print("FunctionMix分析图已保存为 function_mix_analysis_no_subcurve.png")

def main():
    """主函数"""
    data_dir = "public/BrightnessFunctionMixAndPhaseData"
    
    if not os.path.exists(data_dir):
        print(f"数据目录不存在: {data_dir}")
        return
    
    create_function_mix_plot(data_dir)

if __name__ == "__main__":
    main() 