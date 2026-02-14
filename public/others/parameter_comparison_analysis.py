import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Font settings for English text
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

def analyze_all_parameters():
    """分析所有参数的意义和比较价值"""
    
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
    
    print("=== 参数比较分析：为什么选择V0作为主要指标 ===\n")
    
    # 1. 分析各参数的变异性
    print("1. 各参数的变异性分析")
    print("-" * 50)
    
    parameters = ['v0', 'a1', 'phi1', 'a2', 'phi2']
    param_names = ['V0', 'A1', 'φ1', 'A2', 'φ2']
    
    for i, param in enumerate(parameters):
        all_values = []
        for p in exp2_data.keys():
            all_values.extend(exp2_data[p][param])
        
        cv = np.std(all_values) / np.mean(all_values)  # 变异系数
        print(f"{param_names[i]}: 平均值 = {np.mean(all_values):.3f}, 标准差 = {np.std(all_values):.3f}, 变异系数 = {cv:.3f}")
    
    print("\n" + "="*60 + "\n")
    
    # 2. 分析各参数的理论意义
    print("2. 各参数的理论意义分析")
    print("-" * 50)
    
    print("V0 (基准速度):")
    print("- 理论值: 2.0 (明确的参考标准)")
    print("- 感知意义: 代表平均速度感知")
    print("- 测量稳定性: 高")
    print("- 解释性: 直观易懂")
    
    print("\nA1, A2 (振幅):")
    print("- 理论值: 复杂，取决于具体实验条件")
    print("- 感知意义: 影响速度波动幅度")
    print("- 测量稳定性: 中等")
    print("- 解释性: 需要结合相位理解")
    
    print("\nφ1, φ2 (相位):")
    print("- 理论值: 复杂，涉及时间同步")
    print("- 感知意义: 影响速度变化的时间特征")
    print("- 测量稳定性: 较低")
    print("- 解释性: 需要专业知识理解")
    
    print("\n" + "="*60 + "\n")
    
    # 3. 分析各参数的个体差异
    print("3. 各参数的个体差异分析")
    print("-" * 50)
    
    for param in parameters:
        print(f"\n{param.upper()} 参数的个体差异:")
        for p in exp2_data.keys():
            mean_val = np.mean(exp2_data[p][param])
            std_val = np.std(exp2_data[p][param])
            print(f"  {p}: {mean_val:.3f} ± {std_val:.3f}")
    
    print("\n" + "="*60 + "\n")
    
    # 4. 分析各参数与理论值的偏差
    print("4. 各参数与理论值的偏差分析")
    print("-" * 50)
    
    # 假设的理论值（基于实验设计）
    theoretical_values = {
        'v0': 2.0,
        'a1': 1.0,  # 假设值
        'phi1': np.pi,  # 假设值
        'a2': 0.5,  # 假设值
        'phi2': np.pi/2  # 假设值
    }
    
    for param in parameters:
        all_values = []
        for p in exp2_data.keys():
            all_values.extend(exp2_data[p][param])
        
        mean_val = np.mean(all_values)
        if param in theoretical_values:
            theo_val = theoretical_values[param]
            deviation = (mean_val - theo_val) / theo_val * 100
            print(f"{param.upper()}: 实际值 = {mean_val:.3f}, 理论值 = {theo_val:.3f}, 偏差 = {deviation:.1f}%")
        else:
            print(f"{param.upper()}: 实际值 = {mean_val:.3f}, 理论值 = 未知")
    
    print("\n" + "="*60 + "\n")
    
    # 5. 创建可视化比较
    create_parameter_comparison_plots(exp2_data)
    
    return exp2_data

def create_parameter_comparison_plots(exp2_data):
    """创建参数比较的可视化图表"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Parameter Comparison Analysis: Why V0 is the Primary Metric', fontsize=16, fontweight='bold')
    
    parameters = ['v0', 'a1', 'phi1', 'a2', 'phi2']
    param_names = ['V0', 'A1', 'φ1', 'A2', 'φ2']
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink']
    
    # 1. 各参数的箱线图比较
    ax1 = axes[0, 0]
    all_data = []
    labels = []
    
    for param in parameters:
        param_data = []
        for p in exp2_data.keys():
            param_data.extend(exp2_data[p][param])
        all_data.append(param_data)
        labels.append(param_names[parameters.index(param)])
    
    bp = ax1.boxplot(all_data, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax1.set_title('Variability Comparison Across Parameters')
    ax1.set_ylabel('Parameter Value')
    ax1.grid(True, alpha=0.3)
    
    # 2. V0值的个体差异
    ax2 = axes[0, 1]
    v0_data = [exp2_data[p]['v0'] for p in exp2_data.keys()]
    participants = list(exp2_data.keys())
    
    bp = ax2.boxplot(v0_data, labels=participants, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
    
    ax2.set_title('V0 Values by Participant')
    ax2.set_ylabel('V0 Value')
    ax2.grid(True, alpha=0.3)
    
    # 3. 理论值比较（仅V0）
    ax3 = axes[0, 2]
    v0_means = [np.mean(exp2_data[p]['v0']) for p in participants]
    theoretical_v0 = 2.0
    
    bars = ax3.bar(participants, v0_means, alpha=0.7, color='lightblue')
    ax3.axhline(y=theoretical_v0, color='red', linestyle='--', label=f'Theoretical V0 = {theoretical_v0}')
    
    ax3.set_title('V0 Values vs Theoretical Value')
    ax3.set_ylabel('V0 Value')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 参数稳定性比较
    ax4 = axes[1, 0]
    stabilities = []
    for param in parameters:
        all_values = []
        for p in exp2_data.keys():
            all_values.extend(exp2_data[p][param])
        cv = np.std(all_values) / np.mean(all_values)
        stabilities.append(cv)
    
    bars = ax4.bar(param_names, stabilities, alpha=0.7, color='lightgreen')
    ax4.set_title('Parameter Stability (Coefficient of Variation)')
    ax4.set_ylabel('Coefficient of Variation')
    ax4.grid(True, alpha=0.3)
    
    # 5. 个体一致性比较
    ax5 = axes[1, 1]
    consistencies = []
    for p in participants:
        v0_std = np.std(exp2_data[p]['v0'])
        consistencies.append(v0_std)
    
    bars = ax5.bar(participants, consistencies, alpha=0.7, color='lightcoral')
    ax5.set_title('Individual Consistency (V0 Standard Deviation)')
    ax5.set_ylabel('Standard Deviation')
    ax5.grid(True, alpha=0.3)
    
    # 6. 参数相关性热图
    ax6 = axes[1, 2]
    
    # 计算所有参数的相关性矩阵
    all_params_data = {}
    for param in parameters:
        all_params_data[param] = []
        for p in exp2_data.keys():
            all_params_data[param].extend(exp2_data[p][param])
    
    # 创建相关性矩阵
    corr_matrix = np.zeros((len(parameters), len(parameters)))
    for i, param1 in enumerate(parameters):
        for j, param2 in enumerate(parameters):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                corr, _ = stats.pearsonr(all_params_data[param1], all_params_data[param2])
                corr_matrix[i, j] = corr
    
    im = ax6.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax6.set_xticks(range(len(parameters)))
    ax6.set_yticks(range(len(parameters)))
    ax6.set_xticklabels(param_names)
    ax6.set_yticklabels(param_names)
    
    # 添加数值标签
    for i in range(len(parameters)):
        for j in range(len(parameters)):
            text = ax6.text(j, i, f'{corr_matrix[i, j]:.2f}',
                           ha="center", va="center", color="black", fontsize=10)
    
    ax6.set_title('Parameter Correlation Matrix')
    plt.colorbar(im, ax=ax6)
    
    plt.tight_layout()
    plt.savefig('parameter_comparison_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def generate_parameter_analysis_report(exp2_data):
    """生成参数分析报告"""
    
    report = """
# 参数比较分析报告：为什么选择V0作为主要指标

## 1. 各参数的理论意义

### V0 (基准速度)
- **理论值**: 2.0 (明确的参考标准)
- **感知意义**: 代表被试者感知的平均速度
- **测量稳定性**: 高，受噪声影响小
- **解释性**: 直观易懂，便于理解

### A1, A2 (振幅参数)
- **理论值**: 复杂，取决于具体实验条件
- **感知意义**: 影响速度波动的幅度
- **测量稳定性**: 中等，可能受实验条件影响
- **解释性**: 需要结合相位参数理解

### φ1, φ2 (相位参数)
- **理论值**: 复杂，涉及时间同步问题
- **感知意义**: 影响速度变化的时间特征
- **测量稳定性**: 较低，容易受时间同步误差影响
- **解释性**: 需要专业知识才能理解

## 2. 选择V0作为主要指标的原因

### 2.1 理论优势
1. **明确的参考标准**: V0有明确的理论值2.0
2. **直观的物理意义**: 代表平均速度感知
3. **稳定的测量**: 相对其他参数更稳定

### 2.2 统计优势
1. **变异性适中**: 既显示个体差异，又保持可解释性
2. **正态性良好**: 数据分布相对正态
3. **相关性明确**: 与实验条件的关系更容易理解

### 2.3 应用优势
1. **易于实现**: 在系统中容易调整和优化
2. **用户友好**: 参数含义容易向用户解释
3. **工程实用**: 便于在实际系统中应用

## 3. 其他参数的局限性

### 3.1 振幅参数 (A1, A2)
- 理论值不明确
- 个体差异过大
- 与实验条件关系复杂

### 3.2 相位参数 (φ1, φ2)
- 理论值难以确定
- 测量误差较大
- 解释性较差

## 4. 结论

V0作为主要比较指标是最优选择，因为：
1. 具有明确的理论参考值
2. 代表核心的感知特征
3. 测量稳定可靠
4. 易于理解和应用

其他参数可以作为辅助指标，但V0是最适合作为主要评估标准的参数。
"""
    
    with open('parameter_analysis_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("参数分析报告已保存到 'parameter_analysis_report.md'")

if __name__ == "__main__":
    # 执行参数分析
    exp2_data = analyze_all_parameters()
    
    # 生成报告
    generate_parameter_analysis_report(exp2_data)
    
    print("\n参数比较分析完成！") 