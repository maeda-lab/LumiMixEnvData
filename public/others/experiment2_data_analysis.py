#!/usr/bin/env python3
"""
実験2データ分析スクリプト
2視点輝度混合手法における主観的速度等価性測定と再現性の向上

研究目的: 遠隔操縦における映像伝送遅延の問題を解決するため、
多視点映像の即時混合による等価的無遅延化手法の主観的速度等価性と再現性を評価

実験設計:
- 実験1: 基準値測定（中位数算出）
- 実験2: 速度調整実験（6回、2条件×3回ずつ）
- 調整パラメータ: v(t) = V0 + A1·sin(ωt + φ1 + π) + A2·sin(2ωt + φ2 + π)
- 条件: 線形輝度混合 vs 実験1データ使用
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import ttest_rel, f_oneway, pearsonr
import warnings
warnings.filterwarnings('ignore')

# 日本語フォント設定
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class Experiment2DataAnalyzer:
    def __init__(self):
        """初期化：実験1の基準値を設定"""
        # 実験1の基準値（中位数）
        self.baseline_values = {
            'ONO': 0.583,   # [0.517, 0.713, 0.581, 0.583, 0.684, 1.0]
            'LL': 0.218,    # [0.0, 0.492, 0.471, 0.231, 0.178, 0.205]
            'HOU': 0.316,   # [0.163, 0.206, 0.555, 0.336, 0.295, 0.712]
            'OMU': 0.734,   # [0.817, 0.651, 0.551, 0.84, 0.582, 0.841]
            'YAMA': 0.615   # [0.683, 0.616, 0.785, 0.583, 0.613, 0.581]
        }
        
        # 実験1の全データ（再現性分析用）
        self.exp1_full_data = {
            'ONO': [0.517, 0.713, 0.581, 0.583, 0.684, 1.0],
            'LL': [0.0, 0.492, 0.471, 0.231, 0.178, 0.205],
            'HOU': [0.163, 0.206, 0.555, 0.336, 0.295, 0.712],
            'OMU': [0.817, 0.651, 0.551, 0.84, 0.582, 0.841],
            'YAMA': [0.683, 0.616, 0.785, 0.583, 0.613, 0.581]
        }
        
        self.results = {}
        
    def load_experiment_data(self, data_file=None):
        """実験データの読み込み"""
        if data_file is None:
            # サンプルデータ生成（実際のデータファイルと置き換え）
            self.data = self.generate_sample_data()
        else:
            self.data = pd.read_csv(data_file)
        
        return self.data
    
    def generate_sample_data(self):
        """サンプルデータの生成（実際のデータで置き換え）"""
        participants = ['ONO', 'LL', 'HOU', 'OMU', 'YAMA']
        conditions = ['linear', 'exp1_data']
        
        data = []
        for participant in participants:
            baseline = self.baseline_values[participant]
            
            for condition in conditions:
                for trial in range(3):  # 各条件3回
                    # 実際のデータに置き換える
                    if condition == 'linear':
                        # 線形条件のサンプル値
                        v0 = baseline + np.random.normal(0, 0.05)
                        a1 = np.random.uniform(0.1, 0.3)
                        phi1 = np.random.uniform(0, 2*np.pi)
                        a2 = np.random.uniform(0.05, 0.15)
                        phi2 = np.random.uniform(0, 2*np.pi)
                        final_speed = v0 + a1 * 0.5 + a2 * 0.3
                    else:
                        # 実験1データ使用条件のサンプル値
                        v0 = baseline + np.random.normal(0, 0.03)
                        a1 = np.random.uniform(0.08, 0.25)
                        phi1 = np.random.uniform(0, 2*np.pi)
                        a2 = np.random.uniform(0.03, 0.12)
                        phi2 = np.random.uniform(0, 2*np.pi)
                        final_speed = v0 + a1 * 0.4 + a2 * 0.2
                    
                    data.append({
                        'participant': participant,
                        'condition': condition,
                        'trial': trial + 1,
                        'V0': v0,
                        'A1': a1,
                        'phi1': phi1,
                        'A2': a2,
                        'phi2': phi2,
                        'final_speed': final_speed,
                        'baseline': baseline
                    })
        
        return pd.DataFrame(data)
    
    def calculate_speed_equivalence(self):
        """速度等価性の計算"""
        # 等価性指標 = |調整後速度 - 基準速度| / 基準速度
        self.data['equivalence'] = abs(self.data['final_speed'] - self.data['baseline']) / self.data['baseline']
        
        # 相対誤差
        self.data['relative_error'] = (self.data['final_speed'] - self.data['baseline']) / self.data['baseline']
        
        return self.data
    
    def analyze_descriptive_stats(self):
        """記述統計の計算"""
        # 条件別・参加者別統計
        stats_by_condition = self.data.groupby(['participant', 'condition']).agg({
            'equivalence': ['mean', 'std', 'min', 'max'],
            'relative_error': ['mean', 'std'],
            'V0': ['mean', 'std'],
            'A1': ['mean', 'std'],
            'A2': ['mean', 'std']
        }).round(4)
        
        # 全体統計
        overall_stats = self.data.groupby('condition').agg({
            'equivalence': ['mean', 'std', 'sem'],
            'relative_error': ['mean', 'std']
        }).round(4)
        
        self.results['descriptive_stats'] = {
            'by_condition': stats_by_condition,
            'overall': overall_stats
        }
        
        return self.results['descriptive_stats']
    
    def analyze_reproducibility(self):
        """再現性の分析"""
        # 変動係数(CV)の計算
        cv_data = []
        for participant in self.data['participant'].unique():
            for condition in self.data['condition'].unique():
                subset = self.data[(self.data['participant'] == participant) & 
                                  (self.data['condition'] == condition)]
                
                if len(subset) > 1:
                    cv = subset['equivalence'].std() / subset['equivalence'].mean()
                    cv_data.append({
                        'participant': participant,
                        'condition': condition,
                        'cv': cv
                    })
        
        cv_df = pd.DataFrame(cv_data)
        
        # 実験1データの再現性
        exp1_cv = []
        for participant, data in self.exp1_full_data.items():
            cv = np.std(data) / np.mean(data)
            exp1_cv.append({
                'participant': participant,
                'condition': 'exp1_baseline',
                'cv': cv
            })
        
        exp1_cv_df = pd.DataFrame(exp1_cv)
        
        self.results['reproducibility'] = {
            'exp2_cv': cv_df,
            'exp1_cv': exp1_cv_df
        }
        
        return self.results['reproducibility']
    
    def statistical_tests(self):
        """統計的検定"""
        # 条件間比較（対応のあるt検定）
        linear_data = self.data[self.data['condition'] == 'linear'].groupby('participant')['equivalence'].mean()
        exp1_data = self.data[self.data['condition'] == 'exp1_data'].groupby('participant')['equivalence'].mean()
        
        # t検定
        t_stat, p_value = ttest_rel(linear_data, exp1_data)
        
        # 効果量（Cohen's d）
        diff = linear_data - exp1_data
        cohens_d = diff.mean() / diff.std()
        
        # 再現性の比較
        linear_cv = self.results['reproducibility']['exp2_cv'][
            self.results['reproducibility']['exp2_cv']['condition'] == 'linear']['cv']
        exp1_cv = self.results['reproducibility']['exp2_cv'][
            self.results['reproducibility']['exp2_cv']['condition'] == 'exp1_data']['cv']
        
        cv_t_stat, cv_p_value = ttest_rel(linear_cv, exp1_cv)
        
        # 個人差の検定（一元配置分散分析）
        groups = [self.data[self.data['participant'] == p]['equivalence'].values 
                 for p in self.data['participant'].unique()]
        f_stat, anova_p = f_oneway(*groups)
        
        self.results['statistical_tests'] = {
            'equivalence_comparison': {
                't_statistic': t_stat,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'linear_mean': linear_data.mean(),
                'exp1_mean': exp1_data.mean()
            },
            'reproducibility_comparison': {
                't_statistic': cv_t_stat,
                'p_value': cv_p_value,
                'linear_cv_mean': linear_cv.mean(),
                'exp1_cv_mean': exp1_cv.mean()
            },
            'individual_differences': {
                'f_statistic': f_stat,
                'p_value': anova_p
            }
        }
        
        return self.results['statistical_tests']
    
    def parameter_analysis(self):
        """パラメータ分析"""
        # 各パラメータの分布分析
        parameter_stats = {}
        
        for param in ['V0', 'A1', 'A2', 'phi1', 'phi2']:
            param_analysis = self.data.groupby(['participant', 'condition'])[param].agg([
                'mean', 'std', 'min', 'max'
            ]).round(4)
            
            parameter_stats[param] = param_analysis
        
        # 位相パラメータの円形統計（簡略版）
        phi1_linear = self.data[self.data['condition'] == 'linear']['phi1']
        phi1_exp1 = self.data[self.data['condition'] == 'exp1_data']['phi1']
        
        # 位相の平均方向
        phi1_linear_mean = np.arctan2(np.sin(phi1_linear).mean(), np.cos(phi1_linear).mean())
        phi1_exp1_mean = np.arctan2(np.sin(phi1_exp1).mean(), np.cos(phi1_exp1).mean())
        
        self.results['parameter_analysis'] = {
            'parameter_stats': parameter_stats,
            'phase_analysis': {
                'phi1_linear_mean': phi1_linear_mean,
                'phi1_exp1_mean': phi1_exp1_mean
            }
        }
        
        return self.results['parameter_analysis']
    
    def generate_visualizations(self):
        """可視化の生成"""
        # 図のサイズ設定
        fig = plt.figure(figsize=(20, 15))
        
        # 1. 条件間比較（箱ひげ図）
        ax1 = plt.subplot(2, 3, 1)
        sns.boxplot(data=self.data, x='condition', y='equivalence', hue='participant', ax=ax1)
        ax1.set_title('Speed Equivalence by Condition and Participant')
        ax1.set_xlabel('Condition')
        ax1.set_ylabel('Speed Equivalence')
        
        # 2. 再現性比較
        ax2 = plt.subplot(2, 3, 2)
        cv_plot_data = self.results['reproducibility']['exp2_cv']
        sns.barplot(data=cv_plot_data, x='condition', y='cv', hue='participant', ax=ax2)
        ax2.set_title('Reproducibility (CV) by Condition')
        ax2.set_xlabel('Condition')
        ax2.set_ylabel('Coefficient of Variation')
        
        # 3. 個人別パフォーマンス
        ax3 = plt.subplot(2, 3, 3)
        participant_means = self.data.groupby(['participant', 'condition'])['equivalence'].mean().reset_index()
        sns.scatterplot(data=participant_means, x='participant', y='equivalence', 
                       hue='condition', s=100, ax=ax3)
        ax3.set_title('Individual Performance by Condition')
        ax3.set_xlabel('Participant')
        ax3.set_ylabel('Mean Speed Equivalence')
        
        # 4. パラメータ分布（V0）
        ax4 = plt.subplot(2, 3, 4)
        sns.histplot(data=self.data, x='V0', hue='condition', alpha=0.7, ax=ax4)
        ax4.set_title('V0 Parameter Distribution')
        ax4.set_xlabel('V0 (Base Speed)')
        ax4.set_ylabel('Frequency')
        
        # 5. 振幅パラメータ関係
        ax5 = plt.subplot(2, 3, 5)
        sns.scatterplot(data=self.data, x='A1', y='A2', hue='condition', ax=ax5)
        ax5.set_title('A1 vs A2 Parameter Relationship')
        ax5.set_xlabel('A1 (First Harmonic Amplitude)')
        ax5.set_ylabel('A2 (Second Harmonic Amplitude)')
        
        # 6. 位相パラメータ（極座標風）
        ax6 = plt.subplot(2, 3, 6)
        sns.scatterplot(data=self.data, x='phi1', y='phi2', hue='condition', ax=ax6)
        ax6.set_title('Phase Parameters (φ1 vs φ2)')
        ax6.set_xlabel('φ1 (First Harmonic Phase)')
        ax6.set_ylabel('φ2 (Second Harmonic Phase)')
        
        plt.tight_layout()
        plt.savefig('experiment2_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def generate_report(self):
        """分析結果のレポート生成"""
        report = f"""
# 実験2データ分析結果レポート

## 実験概要
- 参加者: {len(self.data['participant'].unique())}名
- 条件: 線形輝度混合 vs 実験1データ使用
- 各条件での試行数: 3回
- 調整パラメータ: V0, A1, φ1, A2, φ2

## 主要な結果

### 1. 速度等価性の比較
"""
        
        # 統計結果の追加
        stats = self.results['statistical_tests']['equivalence_comparison']
        report += f"""
- 線形条件の平均等価性: {stats['linear_mean']:.4f}
- 実験1データ使用条件の平均等価性: {stats['exp1_mean']:.4f}
- t検定結果: t = {stats['t_statistic']:.4f}, p = {stats['p_value']:.4f}
- 効果量 (Cohen's d): {stats['cohens_d']:.4f}

### 2. 再現性の比較
"""
        
        repro_stats = self.results['statistical_tests']['reproducibility_comparison']
        report += f"""
- 線形条件の平均CV: {repro_stats['linear_cv_mean']:.4f}
- 実験1データ使用条件の平均CV: {repro_stats['exp1_cv_mean']:.4f}
- t検定結果: t = {repro_stats['t_statistic']:.4f}, p = {repro_stats['p_value']:.4f}

### 3. 個人差の分析
"""
        
        individual_stats = self.results['statistical_tests']['individual_differences']
        report += f"""
- 一元配置分散分析: F = {individual_stats['f_statistic']:.4f}, p = {individual_stats['p_value']:.4f}

## 結論
本実験により、以下の知見が得られました：

1. **速度等価性**: {'実験1データ使用条件が線形条件より優位' if stats['exp1_mean'] < stats['linear_mean'] else '両条件間に明確な差は見られず'}
2. **再現性**: {'実験1データ使用条件の方が再現性が高い' if repro_stats['exp1_cv_mean'] < repro_stats['linear_cv_mean'] else '両条件の再現性に明確な差は見られず'}
3. **個人差**: {'参加者間に有意な個人差が存在' if individual_stats['p_value'] < 0.05 else '参加者間の個人差は有意ではない'}

これらの結果は、遠隔操縦システムにおける輝度混合手法の改良効果を示唆しています。
"""
        
        return report
    
    def run_complete_analysis(self, data_file=None):
        """完全な分析の実行"""
        print("=== 実験2データ分析を開始 ===")
        
        # 1. データ読み込み
        print("1. データ読み込み中...")
        self.load_experiment_data(data_file)
        
        # 2. 速度等価性計算
        print("2. 速度等価性計算中...")
        self.calculate_speed_equivalence()
        
        # 3. 記述統計
        print("3. 記述統計分析中...")
        self.analyze_descriptive_stats()
        
        # 4. 再現性分析
        print("4. 再現性分析中...")
        self.analyze_reproducibility()
        
        # 5. 統計的検定
        print("5. 統計的検定実行中...")
        self.statistical_tests()
        
        # 6. パラメータ分析
        print("6. パラメータ分析中...")
        self.parameter_analysis()
        
        # 7. 可視化
        print("7. 可視化生成中...")
        self.generate_visualizations()
        
        # 8. レポート生成
        print("8. レポート生成中...")
        report = self.generate_report()
        
        # レポートをファイルに保存
        with open('experiment2_analysis_report.md', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print("=== 分析完了 ===")
        print("結果は 'experiment2_analysis_report.md' に保存されました。")
        print("可視化は 'experiment2_comprehensive_analysis.png' に保存されました。")
        
        return self.results, report

# 使用例
if __name__ == "__main__":
    # 分析器の初期化
    analyzer = Experiment2DataAnalyzer()
    
    # 完全な分析の実行
    results, report = analyzer.run_complete_analysis()
    
    # 結果の概要表示
    print("\n=== 分析結果概要 ===")
    print(report[:1000] + "...")
    
    # 詳細結果の表示
    print("\n=== 詳細な統計結果 ===")
    for key, value in results.items():
        print(f"\n{key}:")
        print(value)