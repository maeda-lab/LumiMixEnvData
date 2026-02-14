import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Font settings for English text
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

def analyze_A1_A2_comparison():
    """专门分析A1和A2参数的对比"""
    
    # 实验2数据 - 包含所有参数
    exp2_data = {
        'ONO': {
            'v0': [1.410, 1.288, 1.028],
            'a1': [-1.000, -0.001, -0.100],
            'phi1': [4.920, 6.283, 6.283],
            'a2': [0.440, 1.310, 0.551],
            'phi2': [1.294, 6.000, 6.283]
        },
        'LL': {
            'v0': [1.100, 1.116, 0.914],
            'a1': [1.076, -0.688, 1.355],
            'phi1': [1.759, 0.496, 4.543],
            'a2': [-0.172, -1.000, -1.000],
            'phi2': [4.606, 2.545, 0.000]
        },
        'HOU': {
            'v0': [1.076, 1.000, 1.006],
            'a1': [0.983, 1.325, 1.094],
            'phi1': [1.257, 2.859, 1.891],
            'a2': [0.980, 0.947, 0.998],
            'phi2': [0.000, 0.308, 0.000]
        },
        'OMU': {
            'v0': [0.992, 1.118, 1.024],
            'a1': [0.596, 0.557, 0.257],
            'phi1': [2.457, 3.198, 6.283],
            'a2': [0.002, -0.136, 0.752],
            'phi2': [3.506, 0.170, 2.149]
        },
        'YAMA': {
            'v0': [0.944, 1.026, 1.044],
            'a1': [0.650, 0.563, 0.662],
            'phi1': [0.000, 0.031, 0.000],
            'a2': [0.536, 0.692, 0.623],
            'phi2': [0.069, 0.924, 0.000]
        }
    }
    
    print("=== A1和A2参数对比分析 ===\n")
    
    # 1. 基本统计分析
    print("1. A1和A2的基本统计分析")
    print("-" * 50)
    
    # 收集所有A1和A2数据
    all_a1 = []
    all_a2 = []
    for p in exp2_data.keys():
        all_a1.extend(exp2_data[p]['a1'])
        all_a2.extend(exp2_data[p]['a2'])
    
    print(f"A1统计:")
    print(f"  平均值: {np.mean(all_a1):.3f}")
    print(f"  标准差: {np.std(all_a1):.3f}")
    print(f"  最小值: {np.min(all_a1):.3f}")
    print(f"  最大值: {np.max(all_a1):.3f}")
    print(f"  变异系数: {np.std(all_a1)/np.mean(np.abs(all_a1)):.3f}")
    
    print(f"\nA2统计:")
    print(f"  平均值: {np.mean(all_a2):.3f}")
    print(f"  标准差: {np.std(all_a2):.3f}")
    print(f"  最小值: {np.min(all_a2):.3f}")
    print(f"  最大值: {np.max(all_a2):.3f}")
    print(f"  变异系数: {np.std(all_a2)/np.mean(np.abs(all_a2)):.3f}")
    
    print("\n" + "="*60 + "\n")
    
    # 2. 个体差异分析
    print("2. A1和A2的个体差异分析")
    print("-" * 50)
    
    for p in exp2_data.keys():
        a1_mean = np.mean(exp2_data[p]['a1'])
        a1_std = np.std(exp2_data[p]['a1'])
        a2_mean = np.mean(exp2_data[p]['a2'])
        a2_std = np.std(exp2_data[p]['a2'])
        
        print(f"{p}:")
        print(f"  A1: {a1_mean:.3f} ± {a1_std:.3f}")
        print(f"  A2: {a2_mean:.3f} ± {a2_std:.3f}")
        print(f"  A1/A2比值: {a1_mean/a2_mean:.3f}" if a2_mean != 0 else "  A1/A2比值: 无穷大")
        print()
    
    print("="*60 + "\n")
    
    # 3. A1和A2的相关性分析
    print("3. A1和A2的相关性分析")
    print("-" * 50)
    
    # 计算相关性
    corr, p_value = stats.pearsonr(all_a1, all_a2)
    print(f"皮尔逊相关系数: {corr:.3f}")
    print(f"p值: {p_value:.3f}")
    print(f"相关性解释: {'强相关' if abs(corr) > 0.7 else '中等相关' if abs(corr) > 0.3 else '弱相关'}")
    
    # 按个体计算相关性
    print(f"\n各被试者的A1-A2相关性:")
    for p in exp2_data.keys():
        if len(exp2_data[p]['a1']) > 1:  # 需要至少2个数据点
            corr_ind, p_ind = stats.pearsonr(exp2_data[p]['a1'], exp2_data[p]['a2'])
            print(f"  {p}: r = {corr_ind:.3f}, p = {p_ind:.3f}")
    
    print("\n" + "="*60 + "\n")
    
    # 4. 理论意义分析
    print("4. A1和A2的理论意义分析")
    print("-" * 50)
    
    print("A1 (第一频率分量振幅):")
    print("- 物理意义: 控制主要速度波动分量的幅度")
    print("- 感知影响: 影响速度感知的周期性变化强度")
    print("- 理论预期: 通常为正值，表示正向速度调制")
    print("- 实际观察: 有正有负，显示个体差异")
    
    print("\nA2 (第二频率分量振幅):")
    print("- 物理意义: 控制次要速度波动分量的幅度")
    print("- 感知影响: 影响速度感知的精细调节")
    print("- 理论预期: 通常为正值，但可能较小")
    print("- 实际观察: 变化范围较大，个体差异明显")
    
    print("\nA1/A2比值意义:")
    print("- 反映两个频率分量的相对重要性")
    print("- 比值>1: 第一分量占主导")
    print("- 比值<1: 第二分量占主导")
    print("- 比值≈1: 两个分量平衡")
    
    print("\n" + "="*60 + "\n")
    
    # 5. 创建可视化分析
    create_A1_A2_visualization(exp2_data)
    
    return exp2_data

def create_A1_A2_visualization(exp2_data):
    """创建A1和A2的可视化分析图表"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('A1 and A2 Parameter Comparison Analysis', fontsize=16, fontweight='bold')
    
    # 1. A1和A2的箱线图比较
    ax1 = axes[0, 0]
    a1_data = []
    a2_data = []
    for p in exp2_data.keys():
        a1_data.extend(exp2_data[p]['a1'])
        a2_data.extend(exp2_data[p]['a2'])
    
    bp = ax1.boxplot([a1_data, a2_data], labels=['A1', 'A2'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightgreen')
    
    ax1.set_title('A1 vs A2 Distribution Comparison')
    ax1.set_ylabel('Amplitude Value')
    ax1.grid(True, alpha=0.3)
    
    # 2. A1和A2的散点图
    ax2 = axes[0, 1]
    ax2.scatter(a1_data, a2_data, alpha=0.6, color='purple')
    ax2.set_xlabel('A1 Value')
    ax2.set_ylabel('A2 Value')
    ax2.set_title('A1 vs A2 Scatter Plot')
    ax2.grid(True, alpha=0.3)
    
    # 添加趋势线
    z = np.polyfit(a1_data, a2_data, 1)
    p = np.poly1d(z)
    ax2.plot(a1_data, p(a1_data), "r--", alpha=0.8)
    
    # 3. 各被试者的A1和A2对比
    ax3 = axes[0, 2]
    participants = list(exp2_data.keys())
    a1_means = [np.mean(exp2_data[p]['a1']) for p in participants]
    a2_means = [np.mean(exp2_data[p]['a2']) for p in participants]
    
    x = np.arange(len(participants))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, a1_means, width, label='A1', alpha=0.7, color='lightblue')
    bars2 = ax3.bar(x + width/2, a2_means, width, label='A2', alpha=0.7, color='lightgreen')
    
    ax3.set_xlabel('Participants')
    ax3.set_ylabel('Mean Amplitude Value')
    ax3.set_title('A1 vs A2 by Participant')
    ax3.set_xticks(x)
    ax3.set_xticklabels(participants)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. A1/A2比值分析
    ax4 = axes[1, 0]
    ratios = []
    for p in participants:
        a1_mean = np.mean(exp2_data[p]['a1'])
        a2_mean = np.mean(exp2_data[p]['a2'])
        if a2_mean != 0:
            ratios.append(a1_mean / a2_mean)
        else:
            ratios.append(np.nan)
    
    bars = ax4.bar(participants, ratios, alpha=0.7, color='orange')
    ax4.axhline(y=1, color='red', linestyle='--', label='A1/A2 = 1')
    ax4.set_title('A1/A2 Ratio by Participant')
    ax4.set_ylabel('A1/A2 Ratio')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 个体一致性分析
    ax5 = axes[1, 1]
    a1_stds = [np.std(exp2_data[p]['a1']) for p in participants]
    a2_stds = [np.std(exp2_data[p]['a2']) for p in participants]
    
    bars1 = ax5.bar(x - width/2, a1_stds, width, label='A1 Std', alpha=0.7, color='lightblue')
    bars2 = ax5.bar(x + width/2, a2_stds, width, label='A2 Std', alpha=0.7, color='lightgreen')
    
    ax5.set_xlabel('Participants')
    ax5.set_ylabel('Standard Deviation')
    ax5.set_title('Individual Consistency (Standard Deviation)')
    ax5.set_xticks(x)
    ax5.set_xticklabels(participants)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 参数分布直方图
    ax6 = axes[1, 2]
    ax6.hist(a1_data, bins=10, alpha=0.7, label='A1', color='lightblue', density=True)
    ax6.hist(a2_data, bins=10, alpha=0.7, label='A2', color='lightgreen', density=True)
    ax6.set_xlabel('Amplitude Value')
    ax6.set_ylabel('Density')
    ax6.set_title('A1 and A2 Distribution')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('A1_A2_comparison_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def generate_A1_A2_analysis_report(exp2_data):
    """生成A1和A2对比分析报告"""
    
    report = """
# A1和A2参数对比分析报告

## 1. 参数定义和理论意义

### A1 (第一频率分量振幅)
- **定义**: 控制主要速度波动分量的幅度
- **物理意义**: 影响速度感知的周期性变化强度
- **理论预期**: 通常为正值，表示正向速度调制
- **感知影响**: 决定速度波动的整体幅度

### A2 (第二频率分量振幅)
- **定义**: 控制次要速度波动分量的幅度
- **物理意义**: 影响速度感知的精细调节
- **理论预期**: 通常为正值，但可能较小
- **感知影响**: 提供速度变化的细节调节

## 2. 为什么A1和A2对比很重要

### 2.1 理论重要性
1. **频率分量平衡**: A1/A2比值反映两个频率分量的相对重要性
2. **感知机制**: 不同个体可能对不同的频率分量更敏感
3. **系统优化**: 了解A1和A2的关系有助于优化系统参数

### 2.2 实际应用价值
1. **个性化调节**: 可以根据A1/A2比值进行个性化设置
2. **系统设计**: 帮助设计更有效的速度调制算法
3. **用户体验**: 提供更好的用户感知体验

## 3. A1和A2的统计特征

### 3.1 基本统计
- A1和A2都显示出较大的个体差异
- 两个参数都有正负值，表明个体差异显著
- A1/A2比值在不同个体间变化很大

### 3.2 相关性分析
- A1和A2之间可能存在相关性
- 不同个体的相关性模式不同
- 这种相关性可能反映感知机制

## 4. 与V0的关系

### 4.1 参数互补性
- V0提供基准速度信息
- A1和A2提供速度变化信息
- 三个参数共同描述完整的速度感知

### 4.2 综合分析价值
- V0适合作为主要指标（稳定性好）
- A1和A2适合作为辅助指标（提供细节信息）
- 三个参数结合提供全面的感知描述

## 5. 结论和建议

### 5.1 主要发现
1. A1和A2都是重要的感知参数
2. 个体间存在显著差异
3. 两个参数可能相互关联

### 5.2 应用建议
1. **主要指标**: 继续使用V0作为主要比较指标
2. **辅助指标**: 将A1和A2作为重要的辅助指标
3. **综合分析**: 结合三个参数进行综合分析
4. **个性化**: 考虑基于A1/A2比值进行个性化调节

### 5.3 未来研究方向
1. 深入研究A1和A2的感知机制
2. 探索A1/A2比值与个体特征的关系
3. 开发基于多参数的个性化算法
4. 验证A1和A2在实际应用中的效果

## 6. 总结

A1和A2的对比分析确实非常重要，它们提供了V0无法提供的详细信息。虽然V0作为主要指标有其优势，但A1和A2的分析对于理解感知机制和优化系统性能同样重要。建议将三个参数（V0、A1、A2）结合使用，以获得最全面的分析结果。
"""
    
    with open('A1_A2_analysis_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("A1和A2对比分析报告已保存到 'A1_A2_analysis_report.md'")

if __name__ == "__main__":
    # 执行A1和A2对比分析
    exp2_data = analyze_A1_A2_comparison()
    
    # 生成报告
    generate_A1_A2_analysis_report(exp2_data)
    
    print("\nA1和A2对比分析完成！") 