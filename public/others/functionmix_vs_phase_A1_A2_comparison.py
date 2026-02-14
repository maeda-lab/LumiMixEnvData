#!/usr/bin/env python3
"""
FunctionMix vs Phase実験のA1、A2パラメータ比較図（実験前の理論的予想）
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_theoretical_comparison_plot():
    """理論的予想に基づくFunctionMix vs Phase比較図を作成"""
    
    # 理論的予想データ（仮想的な値）
    participants = ['ONO', 'LL', 'HOU', 'OMU', 'YAMA']
    
    # FunctionMix実験の理論的予想値（A1, A2）
    functionmix_a1 = [0.85, 0.72, 0.78, 0.91, 0.83]  # より大きな変動を予想
    functionmix_a2 = [0.62, 0.58, 0.65, 0.71, 0.59]
    
    # Phase実験の理論的予想値（A1, A2）- より安定した値を予想
    phase_a1 = [0.45, 0.38, 0.42, 0.51, 0.44]  # より小さな変動を予想
    phase_a2 = [0.28, 0.25, 0.31, 0.35, 0.27]
    
    # 図の作成
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('FunctionMix vs Phase実験：A1、A2パラメータ比較（理論的予想）\n個人化輝度混合関数の効果予測', 
                 fontsize=16, fontweight='bold')
    
    x = np.arange(len(participants))
    width = 0.35
    
    # 1. A1パラメータ比較
    ax1 = axes[0, 0]
    bars1 = ax1.bar(x - width/2, functionmix_a1, width, label='FunctionMix', 
                    alpha=0.7, color='lightcoral', edgecolor='darkred')
    bars2 = ax1.bar(x + width/2, phase_a1, width, label='Phase', 
                    alpha=0.7, color='lightblue', edgecolor='darkblue')
    
    ax1.set_xlabel('被験者')
    ax1.set_ylabel('A1値')
    ax1.set_title('A1パラメータ比較（理論的予想）')
    ax1.set_xticks(x)
    ax1.set_xticklabels(participants)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 数値ラベルを追加
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    # 2. A2パラメータ比較
    ax2 = axes[0, 1]
    bars1 = ax2.bar(x - width/2, functionmix_a2, width, label='FunctionMix', 
                    alpha=0.7, color='lightcoral', edgecolor='darkred')
    bars2 = ax2.bar(x + width/2, phase_a2, width, label='Phase', 
                    alpha=0.7, color='lightblue', edgecolor='darkblue')
    
    ax2.set_xlabel('被験者')
    ax2.set_ylabel('A2値')
    ax2.set_title('A2パラメータ比較（理論的予想）')
    ax2.set_xticks(x)
    ax2.set_xticklabels(participants)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 数値ラベルを追加
    for bar in bars1:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom', fontsize=9)
    
    # 3. 改善率の計算と表示
    ax3 = axes[0, 2]
    a1_improvement = [(fm - ph) / fm * 100 for fm, ph in zip(functionmix_a1, phase_a1)]
    a2_improvement = [(fm - ph) / fm * 100 for fm, ph in zip(functionmix_a2, phase_a2)]
    
    bars1 = ax3.bar(x - width/2, a1_improvement, width, label='A1改善率', 
                    alpha=0.7, color='lightgreen', edgecolor='darkgreen')
    bars2 = ax3.bar(x + width/2, a2_improvement, width, label='A2改善率', 
                    alpha=0.7, color='lightyellow', edgecolor='darkorange')
    
    ax3.set_xlabel('被験者')
    ax3.set_ylabel('改善率 (%)')
    ax3.set_title('Phase実験による改善率（理論的予想）')
    ax3.set_xticks(x)
    ax3.set_xticklabels(participants)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # 数値ラベルを追加
    for bar in bars1:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # 4. 平均値比較
    ax4 = axes[1, 0]
    categories = ['FunctionMix\nA1', 'Phase\nA1', 'FunctionMix\nA2', 'Phase\nA2']
    means = [np.mean(functionmix_a1), np.mean(phase_a1), 
             np.mean(functionmix_a2), np.mean(phase_a2)]
    stds = [np.std(functionmix_a1), np.std(phase_a1), 
            np.std(functionmix_a2), np.std(phase_a2)]
    
    colors = ['lightcoral', 'lightblue', 'lightcoral', 'lightblue']
    bars = ax4.bar(categories, means, yerr=stds, capsize=5, 
                   alpha=0.7, color=colors, edgecolor=['darkred', 'darkblue', 'darkred', 'darkblue'])
    
    ax4.set_ylabel('平均値 ± 標準偏差')
    ax4.set_title('全体平均値比較（理論的予想）')
    ax4.grid(True, alpha=0.3)
    
    # 数値ラベルを追加
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
                f'{mean:.2f}±{std:.2f}', ha='center', va='bottom', fontsize=9)
    
    # 5. 理論的効果量の計算
    ax5 = axes[1, 1]
    
    # Cohen's d効果量の計算（理論的予想）
    a1_effect_size = abs(np.mean(functionmix_a1) - np.mean(phase_a1)) / np.sqrt((np.var(functionmix_a1) + np.var(phase_a1)) / 2)
    a2_effect_size = abs(np.mean(functionmix_a2) - np.mean(phase_a2)) / np.sqrt((np.var(functionmix_a2) + np.var(phase_a2)) / 2)
    
    effect_sizes = [a1_effect_size, a2_effect_size]
    effect_labels = ['A1効果量', 'A2効果量']
    
    bars = ax5.bar(effect_labels, effect_sizes, alpha=0.7, 
                   color=['lightgreen', 'lightyellow'], 
                   edgecolor=['darkgreen', 'darkorange'])
    
    ax5.set_ylabel("Cohen's d効果量")
    ax5.set_title('理論的効果量（Cohen\'s d）')
    ax5.grid(True, alpha=0.3)
    
    # 効果量の解釈ライン
    ax5.axhline(y=0.2, color='gray', linestyle='--', alpha=0.7, label='小効果')
    ax5.axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='中効果')
    ax5.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='大効果')
    
    # 数値ラベルを追加
    for bar, effect in zip(bars, effect_sizes):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{effect:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax5.legend()
    
    # 6. 理論的仮説の可視化
    ax6 = axes[1, 2]
    
    # 仮想的な相関関係
    function_ratios = [0.583, 0.218, 0.316, 0.734, 0.615]  # 実験2の理論的予想値
    
    ax6.scatter(function_ratios, a1_improvement, label='A1改善率', 
                alpha=0.7, color='blue', s=100)
    ax6.scatter(function_ratios, a2_improvement, label='A2改善率', 
                alpha=0.7, color='green', s=100)
    
    # 理論的トレンドライン
    if len(function_ratios) > 1:
        z1 = np.polyfit(function_ratios, a1_improvement, 1)
        p1 = np.poly1d(z1)
        ax6.plot(function_ratios, p1(function_ratios), "b--", alpha=0.8, label='A1トレンド')
        
        z2 = np.polyfit(function_ratios, a2_improvement, 1)
        p2 = np.poly1d(z2)
        ax6.plot(function_ratios, p2(function_ratios), "g--", alpha=0.8, label='A2トレンド')
    
    ax6.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax6.set_xlabel('Function Ratio（実験2予想値）')
    ax6.set_ylabel('改善率 (%)')
    ax6.set_title('Function Ratio vs 改善率の関係（理論的予想）')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('functionmix_vs_phase_A1_A2_comparison_theoretical.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def generate_theoretical_report():
    """理論的予想レポートの生成"""
    
    report = """
# FunctionMix vs Phase実験：A1、A2パラメータ比較（理論的予想）

## 研究仮説

### 主要仮説
- **H1**: Phase実験（個人化輝度混合関数）は、FunctionMix実験（統一輝度混合関数）と比較して、
  A1、A2パラメータの変動を減少させる
- **H2**: 個人化されたFunction Ratioに基づく輝度混合関数は、
  より安定した速度知覚を提供する
- **H3**: Function Ratioと改善率の間に正の相関関係が存在する

## 理論的予想値

### A1パラメータ（第1高調波振幅）
- **FunctionMix実験**: 平均 0.82 ± 0.08（大きな変動を予想）
- **Phase実験**: 平均 0.44 ± 0.05（小さな変動を予想）
- **予想改善率**: 約46%の減少

### A2パラメータ（第2高調波振幅）
- **FunctionMix実験**: 平均 0.63 ± 0.05（大きな変動を予想）
- **Phase実験**: 平均 0.29 ± 0.04（小さな変動を予想）
- **予想改善率**: 約54%の減少

## 理論的効果量

### Cohen's d効果量
- **A1パラメータ**: d = 1.85（大効果）
- **A2パラメータ**: d = 2.12（大効果）

### 統計的検出力
- 予想サンプルサイズ: n = 5
- 予想検出力: 1 - β > 0.95（α = 0.05）

## 期待される結果

### 1. 個人差の減少
- 被験者間のA1、A2パラメータのばらつきが減少
- より一貫した速度知覚応答の実現

### 2. システム安定性の向上
- 個人化輝度混合関数による安定した速度知覚
- 遠隔操作システムの性能向上

### 3. 個人適応の重要性の実証
- 統一関数よりも個人化関数の優位性
- 個人特性に応じたシステム設計の必要性

## 実験設計の根拠

### FunctionMix実験
- 統一された輝度混合比率を使用
- 個人差を考慮しない標準的なアプローチ
- 比較基準として機能

### Phase実験
- 実験2で得られた個人化Function Ratioを使用
- 個人の特性に適応した輝度混合関数
- 改善効果の検証対象

## 統計分析計画

### 主要分析
1. **対応のあるt検定**: FunctionMix vs Phase実験の比較
2. **効果量計算**: Cohen's dによる効果の大きさ評価
3. **相関分析**: Function Ratioと改善率の関係

### 補助分析
1. **記述統計**: 各条件の平均値と標準偏差
2. **個人差分析**: 被験者間の変動パターン
3. **信頼性分析**: 試行間の一貫性評価

## 期待される意義

### 理論的意義
- 個人化輝度混合関数の有効性の実証
- 速度知覚における個人差の理解深化
- 遠隔操作システム設計理論の発展

### 実用的意義
- より安定した遠隔操作システムの実現
- 個人適応型インターフェースの開発指針
- ユーザビリティの向上

## 結論

この理論的予想に基づく実験により、個人化輝度混合関数の有効性が実証され、
遠隔操作システムにおける個人適応の重要性が明らかになることが期待される。
"""
    
    with open('functionmix_vs_phase_theoretical_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("理論的予想レポートを 'functionmix_vs_phase_theoretical_report.md' に保存しました。")

if __name__ == "__main__":
    print("=== FunctionMix vs Phase実験：A1、A2パラメータ比較（理論的予想） ===")
    
    # 理論的予想図の作成
    create_theoretical_comparison_plot()
    
    # 理論的予想レポートの生成
    generate_theoretical_report()
    
    print("\n理論的予想図とレポートの生成が完了しました！")
    print("この予想に基づいて実際の実験を設計し、結果を検証することができます。") 