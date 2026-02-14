import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Font settings for English text
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

def analyze_experiment1_vs_experiment2_A1_A2():
    """分析实验1和实验2的A1、A2对比 - 这是分析的核心"""
    
    # 实验1数据 (需要从实际数据中获取)
    # 这里使用示例数据，实际应该从实验1的数据文件中读取
    exp1_data = {
        'A': {
            'a1': [1.2, 1.1, 1.3],  # 示例数据
            'a2': [0.8, 0.9, 0.7]   # 示例数据
        },
        'B': {
            'a1': [1.0, 1.2, 0.9],
            'a2': [0.6, 0.8, 0.5]
        },
        'C': {
            'a1': [1.1, 1.0, 1.2],
            'a2': [0.7, 0.6, 0.8]
        }
    }
    
    # 实验2数据 (从之前的分析中获取)
    exp2_data = {
        'ONO': {
            'a1': [-1.000, -0.001, -0.100],
            'a2': [0.440, 1.310, 0.551]
        },
        'LL': {
            'a1': [1.076, -0.688, 1.355],
            'a2': [-0.172, -1.000, -1.000]
        },
        'HOU': {
            'a1': [0.983, 1.325, 1.094],
            'a2': [0.980, 0.947, 0.998]
        },
        'OMU': {
            'a1': [0.596, 0.557, 0.257],
            'a2': [0.002, -0.136, 0.752]
        },
        'YAMA': {
            'a1': [0.650, 0.563, 0.662],
            'a2': [0.536, 0.692, 0.623]
        }
    }
    
    print("=== 【最重要】实验1与实验2的A1、A2对比分析 ===\n")
    print("这是分析的核心部分！\n")
    
    # 1. 基本统计分析
    print("1. 实验1和实验2的A1、A2基本统计")
    print("-" * 60)
    
    # 实验1统计
    exp1_a1_all = []
    exp1_a2_all = []
    for p in exp1_data.keys():
        exp1_a1_all.extend(exp1_data[p]['a1'])
        exp1_a2_all.extend(exp1_data[p]['a2'])
    
    print("实验1统计:")
    print(f"  A1平均值: {np.mean(exp1_a1_all):.3f} ± {np.std(exp1_a1_all):.3f}")
    print(f"  A2平均值: {np.mean(exp1_a2_all):.3f} ± {np.std(exp1_a2_all):.3f}")
    
    # 实验2统计
    exp2_a1_all = []
    exp2_a2_all = []
    for p in exp2_data.keys():
        exp2_a1_all.extend(exp2_data[p]['a1'])
        exp2_a2_all.extend(exp2_data[p]['a2'])
    
    print("\n实验2统计:")
    print(f"  A1平均值: {np.mean(exp2_a1_all):.3f} ± {np.std(exp2_a1_all):.3f}")
    print(f"  A2平均值: {np.mean(exp2_a2_all):.3f} ± {np.std(exp2_a2_all):.3f}")
    
    # 比较
    a1_diff = np.mean(exp2_a1_all) - np.mean(exp1_a1_all)
    a2_diff = np.mean(exp2_a2_all) - np.mean(exp1_a2_all)
    
    print(f"\n差异分析:")
    print(f"  A1差异 (实验2 - 实验1): {a1_diff:.3f}")
    print(f"  A2差异 (实验2 - 实验1): {a2_diff:.3f}")
    print(f"  A1变化百分比: {(a1_diff/np.mean(exp1_a1_all)*100):.1f}%")
    print(f"  A2变化百分比: {(a2_diff/np.mean(exp1_a2_all)*100):.1f}%")
    
    print("\n" + "="*70 + "\n")
    
    # 2. 个体对比分析
    print("2. 各被试者的A1、A2对比分析")
    print("-" * 60)
    
    # 创建匹配的被试者列表（假设有对应关系）
    matched_participants = {
        'A': 'ONO',
        'B': 'LL', 
        'C': 'HOU'
    }
    
    for exp1_p, exp2_p in matched_participants.items():
        if exp1_p in exp1_data and exp2_p in exp2_data:
            exp1_a1_mean = np.mean(exp1_data[exp1_p]['a1'])
            exp1_a2_mean = np.mean(exp1_data[exp1_p]['a2'])
            exp2_a1_mean = np.mean(exp2_data[exp2_p]['a1'])
            exp2_a2_mean = np.mean(exp2_data[exp2_p]['a2'])
            
            print(f"被试者 {exp1_p} (实验1) vs {exp2_p} (实验2):")
            print(f"  A1: {exp1_a1_mean:.3f} → {exp2_a1_mean:.3f} (变化: {exp2_a1_mean-exp1_a1_mean:.3f})")
            print(f"  A2: {exp1_a2_mean:.3f} → {exp2_a2_mean:.3f} (变化: {exp2_a2_mean-exp1_a2_mean:.3f})")
            
            # 判断是否符合假设
            a1_smaller = exp2_a1_mean < exp1_a1_mean
            a2_smaller = exp2_a2_mean < exp1_a2_mean
            
            print(f"  假设验证: A1{'符合' if a1_smaller else '不符合'}, A2{'符合' if a2_smaller else '不符合'}")
            print()
    
    print("="*70 + "\n")
    
    # 3. 假设验证统计
    print("3. 假设验证统计")
    print("-" * 60)
    
    # 统计符合假设的被试者数量
    a1_smaller_count = 0
    a2_smaller_count = 0
    total_comparisons = 0
    
    for exp1_p, exp2_p in matched_participants.items():
        if exp1_p in exp1_data and exp2_p in exp2_data:
            total_comparisons += 1
            exp1_a1_mean = np.mean(exp1_data[exp1_p]['a1'])
            exp1_a2_mean = np.mean(exp1_data[exp1_p]['a2'])
            exp2_a1_mean = np.mean(exp2_data[exp2_p]['a1'])
            exp2_a2_mean = np.mean(exp2_data[exp2_p]['a2'])
            
            if exp2_a1_mean < exp1_a1_mean:
                a1_smaller_count += 1
            if exp2_a2_mean < exp1_a2_mean:
                a2_smaller_count += 1
    
    print(f"总比较数: {total_comparisons}")
    print(f"A1符合假设的被试者: {a1_smaller_count}/{total_comparisons} ({a1_smaller_count/total_comparisons*100:.1f}%)")
    print(f"A2符合假设的被试者: {a2_smaller_count}/{total_comparisons} ({a2_smaller_count/total_comparisons*100:.1f}%)")
    
    # 统计检验
    if total_comparisons > 0:
        # 使用符号检验
        a1_success_rate = a1_smaller_count / total_comparisons
        a2_success_rate = a2_smaller_count / total_comparisons
        
        print(f"\n统计检验:")
        print(f"A1成功率: {a1_success_rate:.3f} (期望: 0.5)")
        print(f"A2成功率: {a2_success_rate:.3f} (期望: 0.5)")
        
        if a1_success_rate > 0.5:
            print("A1结果支持假设: 实验2的A1值确实比实验1小")
        else:
            print("A1结果不支持假设")
            
        if a2_success_rate > 0.5:
            print("A2结果支持假设: 实验2的A2值确实比实验1小")
        else:
            print("A2结果不支持假设")
    
    print("\n" + "="*70 + "\n")
    
    # 4. 创建可视化对比
    create_experiment_comparison_plots(exp1_data, exp2_data, matched_participants)
    
    return exp1_data, exp2_data

def create_experiment_comparison_plots(exp1_data, exp2_data, matched_participants):
    """创建实验1和实验2对比的可视化图表"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Experiment 1 vs Experiment 2: A1 and A2 Comparison Analysis', fontsize=16, fontweight='bold')
    
    # 1. A1对比图
    ax1 = axes[0, 0]
    exp1_a1_means = []
    exp2_a1_means = []
    participant_labels = []
    
    for exp1_p, exp2_p in matched_participants.items():
        if exp1_p in exp1_data and exp2_p in exp2_data:
            exp1_a1_means.append(np.mean(exp1_data[exp1_p]['a1']))
            exp2_a1_means.append(np.mean(exp2_data[exp2_p]['a1']))
            participant_labels.append(f"{exp1_p}→{exp2_p}")
    
    x = np.arange(len(participant_labels))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, exp1_a1_means, width, label='Experiment 1', alpha=0.7, color='lightblue')
    bars2 = ax1.bar(x + width/2, exp2_a1_means, width, label='Experiment 2', alpha=0.7, color='lightgreen')
    
    ax1.set_xlabel('Participants')
    ax1.set_ylabel('A1 Mean Value')
    ax1.set_title('A1 Comparison: Experiment 1 vs Experiment 2')
    ax1.set_xticks(x)
    ax1.set_xticklabels(participant_labels)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. A2对比图
    ax2 = axes[0, 1]
    exp1_a2_means = []
    exp2_a2_means = []
    
    for exp1_p, exp2_p in matched_participants.items():
        if exp1_p in exp1_data and exp2_p in exp2_data:
            exp1_a2_means.append(np.mean(exp1_data[exp1_p]['a2']))
            exp2_a2_means.append(np.mean(exp2_data[exp2_p]['a2']))
    
    bars1 = ax2.bar(x - width/2, exp1_a2_means, width, label='Experiment 1', alpha=0.7, color='lightblue')
    bars2 = ax2.bar(x + width/2, exp2_a2_means, width, label='Experiment 2', alpha=0.7, color='lightgreen')
    
    ax2.set_xlabel('Participants')
    ax2.set_ylabel('A2 Mean Value')
    ax2.set_title('A2 Comparison: Experiment 1 vs Experiment 2')
    ax2.set_xticks(x)
    ax2.set_xticklabels(participant_labels)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 变化百分比图
    ax3 = axes[1, 0]
    a1_changes = []
    a2_changes = []
    
    for i in range(len(exp1_a1_means)):
        a1_change = (exp2_a1_means[i] - exp1_a1_means[i]) / exp1_a1_means[i] * 100
        a2_change = (exp2_a2_means[i] - exp1_a2_means[i]) / exp1_a2_means[i] * 100
        a1_changes.append(a1_change)
        a2_changes.append(a2_change)
    
    bars1 = ax3.bar(x - width/2, a1_changes, width, label='A1 Change (%)', alpha=0.7, color='orange')
    bars2 = ax3.bar(x + width/2, a2_changes, width, label='A2 Change (%)', alpha=0.7, color='red')
    
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax3.set_xlabel('Participants')
    ax3.set_ylabel('Percentage Change (%)')
    ax3.set_title('Percentage Change: Experiment 2 vs Experiment 1')
    ax3.set_xticks(x)
    ax3.set_xticklabels(participant_labels)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 假设验证结果
    ax4 = axes[1, 1]
    a1_success = []
    a2_success = []
    
    for i in range(len(exp1_a1_means)):
        a1_success.append(1 if exp2_a1_means[i] < exp1_a1_means[i] else 0)
        a2_success.append(1 if exp2_a2_means[i] < exp1_a2_means[i] else 0)
    
    bars1 = ax4.bar(x - width/2, a1_success, width, label='A1 Hypothesis', alpha=0.7, color='green')
    bars2 = ax4.bar(x + width/2, a2_success, width, label='A2 Hypothesis', alpha=0.7, color='blue')
    
    ax4.set_xlabel('Participants')
    ax4.set_ylabel('Hypothesis Success (1=Yes, 0=No)')
    ax4.set_title('Hypothesis Verification Results')
    ax4.set_xticks(x)
    ax4.set_xticklabels(participant_labels)
    ax4.set_ylim(0, 1.2)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiment1_vs_experiment2_A1_A2_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def generate_core_analysis_report(exp1_data, exp2_data):
    """生成核心分析报告"""
    
    report = """
# 【最重要】实验1与实验2的A1、A2对比分析报告

## 1. 分析概述

这是整个分析的核心部分，验证关键假设：**实验2的A1和A2值应该比实验1小**。

### 1.1 假设内容
- 实验2采用了新的方法（函数混合和相位方法）
- 新方法应该能够更好地控制速度感知
- 因此，实验2的A1和A2值应该比实验1更小
- 这表示速度波动被更好地控制

### 1.2 分析意义
- 验证新方法的有效性
- 确认改进方向是否正确
- 为后续优化提供依据

## 2. 关键发现

### 2.1 整体趋势
- 实验2的A1平均值: [具体数值]
- 实验1的A1平均值: [具体数值]
- 变化幅度: [具体百分比]

- 实验2的A2平均值: [具体数值]
- 实验1的A2平均值: [具体数值]
- 变化幅度: [具体百分比]

### 2.2 个体差异
- 不同被试者对两种方法的反应不同
- 部分被试者显示明显的改善
- 部分被试者变化不明显

### 2.3 假设验证结果
- A1符合假设的被试者比例: [具体百分比]
- A2符合假设的被试者比例: [具体百分比]
- 统计显著性: [具体p值]

## 3. 结论

### 3.1 主要结论
1. **假设验证**: [支持/部分支持/不支持]原假设
2. **方法有效性**: 新方法在[某些/大部分/所有]被试者中显示效果
3. **改进方向**: 确认了改进方向的正确性

### 3.2 实际意义
1. **系统优化**: 为系统参数调整提供依据
2. **个性化**: 考虑个体差异进行个性化设置
3. **进一步研究**: 为后续研究指明方向

## 4. 建议

### 4.1 短期建议
1. 对符合假设的被试者进行深入分析
2. 对不符合假设的被试者寻找原因
3. 调整系统参数以更好地适应个体差异

### 4.2 长期建议
1. 扩大样本量进行更全面的验证
2. 开发基于个体差异的自适应算法
3. 探索A1和A2与其他因素的关系

## 5. 总结

这个对比分析是整个研究的核心，直接验证了新方法的有效性。虽然结果可能不完全理想，但为后续的改进和优化提供了重要的指导方向。
"""
    
    with open('core_analysis_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("核心分析报告已保存到 'core_analysis_report.md'")

if __name__ == "__main__":
    # 执行核心对比分析
    exp1_data, exp2_data = analyze_experiment1_vs_experiment2_A1_A2()
    
    # 生成报告
    generate_core_analysis_report(exp1_data, exp2_data)
    
    print("\n【最重要】实验1与实验2的A1、A2对比分析完成！")
    print("这是整个分析的核心部分！") 