import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import glob

# Font settings for English text
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['axes.unicode_minus'] = False

def load_phase_data():
    """加载Phase实验的Dynamic和LinearOnly数据"""
    
    data_dir = "public/BrightnessFunctionMixAndPhaseData"
    
    # 存储数据
    dynamic_data = {}  # Dynamic模式数据
    linear_data = {}   # LinearOnly模式数据
    
    participants = ['ONO', 'LL', 'HOU', 'OMU', 'YAMA']
    
    print("=== 加载Phase实验数据：Dynamic vs LinearOnly ===\n")
    
    for participant in participants:
        print(f"处理被试者: {participant}")
        
        # 加载Dynamic数据
        dynamic_files = glob.glob(f"{data_dir}/*Phase*{participant}*Dynamic*.csv")
        dynamic_files = [f for f in dynamic_files if "Test" not in f]
        
        if dynamic_files:
            dynamic_data[participant] = []
            for file in dynamic_files:
                try:
                    df = pd.read_csv(file)
                    df.columns = df.columns.str.strip()
                    
                    params = extract_parameters_from_file(df)
                    if params:
                        dynamic_data[participant].append(params)
                        print(f"  Dynamic: {os.path.basename(file)} - V0:{params['v0']:.3f}, A1:{params['a1']:.3f}, A2:{params['a2']:.3f}")
                except Exception as e:
                    print(f"  错误读取文件 {file}: {e}")
        
        # 加载LinearOnly数据
        linear_files = glob.glob(f"{data_dir}/*Phase*{participant}*LinearOnly*.csv")
        linear_files = [f for f in linear_files if "Test" not in f]
        
        if linear_files:
            linear_data[participant] = []
            for file in linear_files:
                try:
                    df = pd.read_csv(file)
                    df.columns = df.columns.str.strip()
                    
                    params = extract_parameters_from_file(df)
                    if params:
                        linear_data[participant].append(params)
                        print(f"  LinearOnly: {os.path.basename(file)} - V0:{params['v0']:.3f}, A1:{params['a1']:.3f}, A2:{params['a2']:.3f}")
                except Exception as e:
                    print(f"  错误读取文件 {file}: {e}")
        
        print()
    
    return dynamic_data, linear_data

def extract_parameters_from_file(df):
    """从数据文件中提取V0, A1, A2参数"""
    try:
        # 检查是否有StepNumber列
        if 'StepNumber' not in df.columns:
            return None
        
        # 提取V0 (从StepNumber=0的Velocity列)
        v0_data = df[df['StepNumber'] == 0]['Velocity']
        if v0_data.empty:
            return None
        v0 = v0_data.iloc[-1]  # 取最后一个值
        
        # 提取A1 (从StepNumber=1的Amplitude列)
        a1_data = df[df['StepNumber'] == 1]['Amplitude']
        if a1_data.empty:
            return None
        a1 = a1_data.iloc[-1]
        
        # 提取A2 (从StepNumber=3的Amplitude列，因为StepNumber=2是φ1)
        a2_data = df[df['StepNumber'] == 3]['Amplitude']
        if a2_data.empty:
            return None
        a2 = a2_data.iloc[-1]
        
        return {
            'v0': v0,
            'a1': a1,
            'a2': a2
        }
    except Exception as e:
        print(f"提取参数时出错: {e}")
        return None

def analyze_dynamic_vs_linearonly(dynamic_data, linear_data):
    """分析Dynamic vs LinearOnly的A1、A2对比"""
    
    print("=== Dynamic vs LinearOnly: A1、A2稳定性对比分析 ===\n")
    
    # 每个被试者的functionRatio（中位数）
    function_ratios = {
        'ONO': 0.583,   # [0.517, 0.713, 0.581, 0.583, 0.684, 1.0]
        'LL': 0.218,    # [0.0, 0.492, 0.471, 0.231, 0.178, 0.205]
        'HOU': 0.316,   # [0.163, 0.206, 0.555, 0.336, 0.295, 0.712]
        'OMU': 0.734,   # [0.817, 0.651, 0.551, 0.84, 0.582, 0.841]
        'YAMA': 0.615   # [0.683, 0.616, 0.785, 0.583, 0.613, 0.581]
    }
    
    # 1. 个体对比分析
    print("1. 各被试者的Dynamic vs LinearOnly对比")
    print("-" * 70)
    
    participants = ['ONO', 'LL', 'HOU', 'OMU', 'YAMA']
    
    comparison_results = {}
    
    for participant in participants:
        print(f"被试者 {participant} (functionRatio = {function_ratios[participant]:.3f}):")
        
        # Dynamic条件
        if participant in dynamic_data and dynamic_data[participant]:
            dyn_a1_mean = np.mean([t['a1'] for t in dynamic_data[participant]])
            dyn_a1_std = np.std([t['a1'] for t in dynamic_data[participant]])
            dyn_a2_mean = np.mean([t['a2'] for t in dynamic_data[participant]])
            dyn_a2_std = np.std([t['a2'] for t in dynamic_data[participant]])
            dyn_v0_mean = np.mean([t['v0'] for t in dynamic_data[participant]])
            
            print(f"  Dynamic: A1={dyn_a1_mean:.3f}±{dyn_a1_std:.3f}, A2={dyn_a2_mean:.3f}±{dyn_a2_std:.3f}, V0={dyn_v0_mean:.3f}")
        else:
            dyn_a1_mean = dyn_a1_std = dyn_a2_mean = dyn_a2_std = dyn_v0_mean = np.nan
            print(f"  Dynamic: 无数据")
        
        # LinearOnly条件
        if participant in linear_data and linear_data[participant]:
            lin_a1_mean = np.mean([t['a1'] for t in linear_data[participant]])
            lin_a1_std = np.std([t['a1'] for t in linear_data[participant]])
            lin_a2_mean = np.mean([t['a2'] for t in linear_data[participant]])
            lin_a2_std = np.std([t['a2'] for t in linear_data[participant]])
            lin_v0_mean = np.mean([t['v0'] for t in linear_data[participant]])
            
            print(f"  LinearOnly: A1={lin_a1_mean:.3f}±{lin_a1_std:.3f}, A2={lin_a2_mean:.3f}±{lin_a2_std:.3f}, V0={lin_v0_mean:.3f}")
        else:
            lin_a1_mean = lin_a1_std = lin_a2_mean = lin_a2_std = lin_v0_mean = np.nan
            print(f"  LinearOnly: 无数据")
        
        # 比较结果
        if not (np.isnan(dyn_a1_mean) or np.isnan(lin_a1_mean)):
            a1_improvement = (lin_a1_mean - dyn_a1_mean) / lin_a1_mean * 100 if lin_a1_mean != 0 else 0
            a2_improvement = (lin_a2_mean - dyn_a2_mean) / lin_a2_mean * 100 if lin_a2_mean != 0 else 0
            
            a1_more_stable = dyn_a1_std < lin_a1_std
            a2_more_stable = dyn_a2_std < lin_a2_std
            
            print(f"  稳定性比较:")
            print(f"    A1: Dynamic{'更稳定' if a1_more_stable else '更不稳定'} (Dynamic:{dyn_a1_std:.3f} vs LinearOnly:{lin_a1_std:.3f})")
            print(f"    A2: Dynamic{'更稳定' if a2_more_stable else '更不稳定'} (Dynamic:{dyn_a2_std:.3f} vs LinearOnly:{lin_a2_std:.3f})")
            print(f"  改善幅度: A1={a1_improvement:.1f}%, A2={a2_improvement:.1f}%")
            
            comparison_results[participant] = {
                'dynamic': {'a1_mean': dyn_a1_mean, 'a1_std': dyn_a1_std, 'a2_mean': dyn_a2_mean, 'a2_std': dyn_a2_std},
                'linear': {'a1_mean': lin_a1_mean, 'a1_std': lin_a1_std, 'a2_mean': lin_a2_mean, 'a2_std': lin_a2_std},
                'a1_more_stable': a1_more_stable,
                'a2_more_stable': a2_more_stable,
                'function_ratio': function_ratios[participant]
            }
        else:
            print(f"  无法比较：数据不完整")
            comparison_results[participant] = None
        
        print()
    
    print("="*70 + "\n")
    
    # 2. 整体统计
    print("2. 整体统计分析")
    print("-" * 70)
    
    # 收集所有数据
    all_dyn_a1 = []
    all_dyn_a2 = []
    all_lin_a1 = []
    all_lin_a2 = []
    
    for participant in participants:
        if participant in dynamic_data:
            all_dyn_a1.extend([t['a1'] for t in dynamic_data[participant]])
            all_dyn_a2.extend([t['a2'] for t in dynamic_data[participant]])
        
        if participant in linear_data:
            all_lin_a1.extend([t['a1'] for t in linear_data[participant]])
            all_lin_a2.extend([t['a2'] for t in linear_data[participant]])
    
    if all_dyn_a1 and all_lin_a1:
        print("A1参数对比:")
        print(f"  Dynamic: {np.mean(all_dyn_a1):.3f} ± {np.std(all_dyn_a1):.3f}")
        print(f"  LinearOnly: {np.mean(all_lin_a1):.3f} ± {np.std(all_lin_a1):.3f}")
        
        # 统计检验
        t_stat, p_value = stats.ttest_ind(all_dyn_a1, all_lin_a1)
        print(f"  t检验: t={t_stat:.3f}, p={p_value:.3f}")
        print(f"  效应量: {abs(np.mean(all_dyn_a1) - np.mean(all_lin_a1)) / np.sqrt((np.var(all_dyn_a1) + np.var(all_lin_a1)) / 2):.3f}")
        print()
    
    if all_dyn_a2 and all_lin_a2:
        print("A2参数对比:")
        print(f"  Dynamic: {np.mean(all_dyn_a2):.3f} ± {np.std(all_dyn_a2):.3f}")
        print(f"  LinearOnly: {np.mean(all_lin_a2):.3f} ± {np.std(all_lin_a2):.3f}")
        
        # 统计检验
        t_stat, p_value = stats.ttest_ind(all_dyn_a2, all_lin_a2)
        print(f"  t检验: t={t_stat:.3f}, p={p_value:.3f}")
        print(f"  效应量: {abs(np.mean(all_dyn_a2) - np.mean(all_lin_a2)) / np.sqrt((np.var(all_dyn_a2) + np.var(all_lin_a2)) / 2):.3f}")
        print()
    
    print("="*70 + "\n")
    
    # 3. 稳定性分析
    print("3. 稳定性分析")
    print("-" * 70)
    
    a1_stable_count = sum(1 for result in comparison_results.values() if result and result['a1_more_stable'])
    a2_stable_count = sum(1 for result in comparison_results.values() if result and result['a2_more_stable'])
    total_comparisons = sum(1 for result in comparison_results.values() if result is not None)
    
    print(f"总比较数: {total_comparisons}")
    print(f"A1更稳定的被试者: {a1_stable_count}/{total_comparisons} ({a1_stable_count/total_comparisons*100:.1f}%)")
    print(f"A2更稳定的被试者: {a2_stable_count}/{total_comparisons} ({a2_stable_count/total_comparisons*100:.1f}%)")
    
    # 统计显著性检验（二项检验）
    if total_comparisons > 0:
        from scipy.stats import binomtest
        
        # A1稳定性检验
        a1_test = binomtest(a1_stable_count, total_comparisons, p=0.5)
        print(f"A1稳定性二项检验: p={a1_test.pvalue:.3f}")
        
        # A2稳定性检验
        a2_test = binomtest(a2_stable_count, total_comparisons, p=0.5)
        print(f"A2稳定性二项检验: p={a2_test.pvalue:.3f}")
    
    print("\n" + "="*70 + "\n")
    
    # 4. 创建可视化
    create_stability_visualizations(dynamic_data, linear_data, comparison_results)
    
    return dynamic_data, linear_data, comparison_results

def create_stability_visualizations(dynamic_data, linear_data, comparison_results):
    """创建稳定性对比的可视化图表"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Dynamic vs LinearOnly: A1 and A2 Stability Comparison', fontsize=16, fontweight='bold')
    
    participants = ['ONO', 'LL', 'HOU', 'OMU', 'YAMA']
    
    # 1. A1参数对比
    ax1 = axes[0, 0]
    dyn_a1_means = []
    lin_a1_means = []
    dyn_a1_stds = []
    lin_a1_stds = []
    valid_participants = []
    
    for p in participants:
        if p in comparison_results and comparison_results[p]:
            result = comparison_results[p]
            dyn_a1_means.append(result['dynamic']['a1_mean'])
            lin_a1_means.append(result['linear']['a1_mean'])
            dyn_a1_stds.append(result['dynamic']['a1_std'])
            lin_a1_stds.append(result['linear']['a1_std'])
            valid_participants.append(p)
    
    x = np.arange(len(valid_participants))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, dyn_a1_means, width, label='Dynamic', alpha=0.7, color='lightblue', yerr=dyn_a1_stds, capsize=5)
    bars2 = ax1.bar(x + width/2, lin_a1_means, width, label='LinearOnly', alpha=0.7, color='lightgreen', yerr=lin_a1_stds, capsize=5)
    
    ax1.set_xlabel('Participants')
    ax1.set_ylabel('A1 Mean Value')
    ax1.set_title('A1 Parameter Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(valid_participants)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. A2参数对比
    ax2 = axes[0, 1]
    dyn_a2_means = []
    lin_a2_means = []
    dyn_a2_stds = []
    lin_a2_stds = []
    
    for p in participants:
        if p in comparison_results and comparison_results[p]:
            result = comparison_results[p]
            dyn_a2_means.append(result['dynamic']['a2_mean'])
            lin_a2_means.append(result['linear']['a2_mean'])
            dyn_a2_stds.append(result['dynamic']['a2_std'])
            lin_a2_stds.append(result['linear']['a2_std'])
    
    bars1 = ax2.bar(x - width/2, dyn_a2_means, width, label='Dynamic', alpha=0.7, color='lightblue', yerr=dyn_a2_stds, capsize=5)
    bars2 = ax2.bar(x + width/2, lin_a2_means, width, label='LinearOnly', alpha=0.7, color='lightgreen', yerr=lin_a2_stds, capsize=5)
    
    ax2.set_xlabel('Participants')
    ax2.set_ylabel('A2 Mean Value')
    ax2.set_title('A2 Parameter Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(valid_participants)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 稳定性比较（标准差）
    ax3 = axes[0, 2]
    
    bars1 = ax3.bar(x - width/2, dyn_a1_stds, width, label='Dynamic A1 Std', alpha=0.7, color='lightblue')
    bars2 = ax3.bar(x + width/2, lin_a1_stds, width, label='LinearOnly A1 Std', alpha=0.7, color='lightgreen')
    
    ax3.set_xlabel('Participants')
    ax3.set_ylabel('Standard Deviation')
    ax3.set_title('A1 Stability Comparison (Lower is Better)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(valid_participants)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. A2稳定性比较
    ax4 = axes[1, 0]
    
    bars1 = ax4.bar(x - width/2, dyn_a2_stds, width, label='Dynamic A2 Std', alpha=0.7, color='lightblue')
    bars2 = ax4.bar(x + width/2, lin_a2_stds, width, label='LinearOnly A2 Std', alpha=0.7, color='lightgreen')
    
    ax4.set_xlabel('Participants')
    ax4.set_ylabel('Standard Deviation')
    ax4.set_title('A2 Stability Comparison (Lower is Better)')
    ax4.set_xticks(x)
    ax4.set_xticklabels(valid_participants)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 稳定性胜率
    ax5 = axes[1, 1]
    
    a1_wins = sum(1 for result in comparison_results.values() if result and result['a1_more_stable'])
    a2_wins = sum(1 for result in comparison_results.values() if result and result['a2_more_stable'])
    total = sum(1 for result in comparison_results.values() if result is not None)
    
    categories = ['A1 Stability', 'A2 Stability']
    wins = [a1_wins, a2_wins]
    total_trials = [total, total]
    
    bars = ax5.bar(categories, [w/t*100 for w, t in zip(wins, total_trials)], alpha=0.7, color=['lightblue', 'lightgreen'])
    ax5.axhline(y=50, color='red', linestyle='--', alpha=0.7, label='50% (Random)')
    
    # 添加数值标签
    for i, (bar, win, tot) in enumerate(zip(bars, wins, total_trials)):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{win}/{tot}\n({height:.1f}%)', ha='center', va='bottom')
    
    ax5.set_ylabel('Win Rate (%)')
    ax5.set_title('Dynamic vs LinearOnly Stability Win Rate')
    ax5.set_ylim(0, 100)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. FunctionRatio vs 稳定性关系
    ax6 = axes[1, 2]
    
    function_ratios = []
    a1_improvements = []
    a2_improvements = []
    
    for p in participants:
        if p in comparison_results and comparison_results[p]:
            result = comparison_results[p]
            function_ratios.append(result['function_ratio'])
            
            # 计算改善幅度
            a1_imp = (result['linear']['a1_std'] - result['dynamic']['a1_std']) / result['linear']['a1_std'] * 100
            a2_imp = (result['linear']['a2_std'] - result['dynamic']['a2_std']) / result['linear']['a2_std'] * 100
            
            a1_improvements.append(a1_imp)
            a2_improvements.append(a2_imp)
    
    if function_ratios:
        ax6.scatter(function_ratios, a1_improvements, label='A1 Improvement', alpha=0.7, color='blue', s=100)
        ax6.scatter(function_ratios, a2_improvements, label='A2 Improvement', alpha=0.7, color='green', s=100)
        
        # 添加趋势线
        if len(function_ratios) > 1:
            z1 = np.polyfit(function_ratios, a1_improvements, 1)
            p1 = np.poly1d(z1)
            ax6.plot(function_ratios, p1(function_ratios), "b--", alpha=0.8)
            
            z2 = np.polyfit(function_ratios, a2_improvements, 1)
            p2 = np.poly1d(z2)
            ax6.plot(function_ratios, p2(function_ratios), "g--", alpha=0.8)
        
        ax6.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax6.set_xlabel('Function Ratio')
        ax6.set_ylabel('Improvement (%)')
        ax6.set_title('Function Ratio vs Stability Improvement')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dynamic_vs_linearonly_stability_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def generate_stability_report(dynamic_data, linear_data, comparison_results):
    """生成稳定性对比分析报告"""
    
    report = """
# Dynamic vs LinearOnly: A1、A2稳定性对比分析报告

## 1. 分析目的

验证Dynamic条件（使用实验1数据的个性化设置）是否比LinearOnly条件（线性混合）在A1、A2参数上更稳定。

## 2. 实验设计

### 2.1 实验条件
- **Dynamic**: 基于每个被试者的functionRatio进行个性化设置
- **LinearOnly**: 使用线性混合函数
- **被试者**: ONO, LL, HOU, OMU, YAMA

### 2.2 FunctionRatio（中位数）
- ONO: 0.583 (Cosine Ease-In-Out区间)
- LL: 0.218 (Cosine Ease-In-Out区间)
- HOU: 0.316 (Cosine Ease-In-Out区间)
- OMU: 0.734 (Linear-Acos补间区间)
- YAMA: 0.615 (Linear-Acos补间区间)

## 3. 关键发现

### 3.1 个体稳定性对比
[具体每个被试者的A1、A2稳定性对比结果]

### 3.2 整体统计结果
- A1参数: Dynamic vs LinearOnly的统计检验结果
- A2参数: Dynamic vs LinearOnly的统计检验结果

### 3.3 稳定性胜率
- A1更稳定的被试者比例
- A2更稳定的被试者比例
- 统计显著性检验结果

## 4. 结论

### 4.1 主要结论
1. Dynamic条件在[某些/大部分/所有]被试者中显示更好的稳定性
2. A1和A2参数的稳定性改善程度不同
3. 个体差异在稳定性改善中起重要作用

### 4.2 实际意义
1. **个性化设置的有效性**: 验证了个性化functionRatio设置的价值
2. **方法选择**: 为选择最佳混合方法提供依据
3. **系统优化**: 指导实际系统的参数设置

## 5. 建议

### 5.1 应用建议
1. 在实际系统中优先使用Dynamic方法
2. 根据个体差异进行个性化设置
3. 进一步优化functionRatio的计算方法

### 5.2 研究建议
1. 扩大样本量验证结果的可靠性
2. 探索functionRatio与稳定性的关系
3. 研究其他可能影响稳定性的因素

## 6. 总结

Dynamic条件相比LinearOnly条件在A1、A2参数的稳定性方面显示出[显著/部分/不明显]的改善，验证了个性化设置的重要性。
"""
    
    with open('dynamic_vs_linearonly_stability_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("稳定性对比分析报告已保存到 'dynamic_vs_linearonly_stability_report.md'")

if __name__ == "__main__":
    # 加载Phase实验数据
    dynamic_data, linear_data = load_phase_data()
    
    # 执行稳定性对比分析
    analyze_dynamic_vs_linearonly(dynamic_data, linear_data)
    
    # 生成报告
    generate_stability_report(dynamic_data, linear_data, {})
    
    print("\nDynamic vs LinearOnly稳定性对比分析完成！") 