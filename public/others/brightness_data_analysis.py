#!/usr/bin/env python3
"""
Brightness Function Mix and Phase Data Analysis
For research on subjective speed equivalence in two-point brightness blending

Author: Data Analysis Script
Date: 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
import re
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
try:
    plt.style.use('seaborn-v0_8')
except:
    try:
        plt.style.use('seaborn')
    except:
        pass
        
sns.set_palette("husl")

class BrightnessDataAnalyzer:
    def __init__(self, data_path="public/BrightnessFunctionMixAndPhaseData"):
        self.data_path = Path(data_path)
        self.function_mix_data = {}
        self.phase_data = {}
        self.participants = []
        
    def load_data(self):
        """Load all CSV files and organize by experiment type"""
        print("Loading data files...")
        
        # Get all CSV files
        csv_files = list(self.data_path.glob("*.csv"))
        
        for file in csv_files:
            # Skip test files
            if 'Test' in file.name:
                continue
                
            # Parse filename to extract metadata
            metadata = self._parse_filename(file.name)
            if metadata:
                try:
                    df = pd.read_csv(file)
                    
                    # Store data by experiment type
                    if metadata['experiment_type'] == 'FunctionMix':
                        key = f"{metadata['participant']}_{metadata['trial']}"
                        self.function_mix_data[key] = df
                    elif metadata['experiment_type'] == 'Phase':
                        key = f"{metadata['participant']}_{metadata['trial']}_{metadata['blend_mode']}"
                        self.phase_data[key] = df
                        
                    # Track participants
                    if metadata['participant'] not in self.participants:
                        self.participants.append(metadata['participant'])
                        
                except Exception as e:
                    print(f"Error loading {file.name}: {e}")
                    
        print(f"Loaded {len(self.function_mix_data)} FunctionMix files")
        print(f"Loaded {len(self.phase_data)} Phase files")
        print(f"Participants: {self.participants}")
        
    def _parse_filename(self, filename):
        """Parse filename to extract metadata"""
        # Pattern for FunctionMix: 20250715_135327_Fps1_CameraSpeed1_ExperimentPattern_FunctionMix_ParticipantName_ONO_TrialNumber_1.csv
        # Pattern for Phase: 20250715_144516_Fps1_CameraSpeed1_ExperimentPattern_Phase_ParticipantName_ONO_TrialNumber_1_BrightnessBlendMode_LinearOnly.csv
        
        pattern_function = r'(\d{8}_\d{6})_Fps(\d+)_CameraSpeed(\d+)_ExperimentPattern_FunctionMix_ParticipantName_(\w+)_TrialNumber_(\d+)\.csv'
        pattern_phase = r'(\d{8}_\d{6})_Fps(\d+)_CameraSpeed(\d+)_ExperimentPattern_Phase_ParticipantName_(\w+)_TrialNumber_(\d+)_BrightnessBlendMode_(\w+)\.csv'
        
        match_function = re.match(pattern_function, filename)
        match_phase = re.match(pattern_phase, filename)
        
        if match_function:
            return {
                'timestamp': match_function.group(1),
                'fps': int(match_function.group(2)),
                'camera_speed': int(match_function.group(3)),
                'participant': match_function.group(4),
                'trial': int(match_function.group(5)),
                'experiment_type': 'FunctionMix',
                'blend_mode': None
            }
        elif match_phase:
            return {
                'timestamp': match_phase.group(1),
                'fps': int(match_phase.group(2)),
                'camera_speed': int(match_phase.group(3)),
                'participant': match_phase.group(4),
                'trial': int(match_phase.group(5)),
                'experiment_type': 'Phase',
                'blend_mode': match_phase.group(6)
            }
        else:
            print(f"Could not parse filename: {filename}")
            return None
    
    def analyze_function_mix_experiment(self):
        """Analyze Function Mix experiment data"""
        print("\n=== Function Mix Experiment Analysis ===")
        
        results = {}
        
        for participant in self.participants:
            participant_data = []
            
            # Collect all trials for this participant
            for trial in range(1, 7):  # 6 trials per participant
                key = f"{participant}_{trial}"
                if key in self.function_mix_data:
                    df = self.function_mix_data[key]
                    
                    # Calculate key metrics
                    metrics = self._calculate_trial_metrics(df, trial)
                    if metrics is not None:
                        metrics['participant'] = participant
                        participant_data.append(metrics)
            
            if participant_data:
                results[participant] = participant_data
                
        return results
    
    def analyze_phase_experiment(self):
        """Analyze Phase experiment data"""
        print("\n=== Phase Experiment Analysis ===")
        
        results = {}
        
        for participant in self.participants:
            participant_results = {'LinearOnly': [], 'Dynamic': []}
            
            # Analyze LinearOnly trials
            for trial in range(1, 4):  # Assuming 3 trials per mode
                key = f"{participant}_{trial}_LinearOnly"
                if key in self.phase_data:
                    df = self.phase_data[key]
                    metrics = self._calculate_trial_metrics(df, trial)
                    if metrics is not None:
                        metrics['participant'] = participant
                        metrics['blend_mode'] = 'LinearOnly'
                        participant_results['LinearOnly'].append(metrics)
            
            # Analyze Dynamic trials
            for trial in range(1, 4):
                key = f"{participant}_{trial}_Dynamic"
                if key in self.phase_data:
                    df = self.phase_data[key]
                    metrics = self._calculate_trial_metrics(df, trial)
                    if metrics is not None:
                        metrics['participant'] = participant
                        metrics['blend_mode'] = 'Dynamic'
                        participant_results['Dynamic'].append(metrics)
            
            if participant_results['LinearOnly'] or participant_results['Dynamic']:
                results[participant] = participant_results
                
        return results
    
    def _calculate_trial_metrics(self, df, trial):
        """Calculate key metrics for a single trial"""
        # Clean the data
        df_clean = df.dropna()
        
        # Check if we have enough data
        if len(df_clean) < 2:
            print(f"Warning: Trial {trial} has insufficient data ({len(df_clean)} points)")
            return None
        
        # Basic statistics
        knob_mean = df_clean[' Knob'].mean()
        knob_std = df_clean[' Knob'].std()
        knob_median = df_clean[' Knob'].median()
        
        # Response stability (coefficient of variation)
        response_stability = knob_std / knob_mean if knob_mean != 0 else np.inf
        
        # Velocity analysis
        velocity_mean = df_clean[' Velocity'].mean()
        velocity_std = df_clean[' Velocity'].std()
        
        # Function ratio analysis
        function_ratio_mean = df_clean[' FunctionRatio'].mean()
        function_ratio_std = df_clean[' FunctionRatio'].std()
        
        # Response time analysis
        try:
            response_time = df_clean[' Time'].iloc[-1] - df_clean[' Time'].iloc[0]
        except IndexError:
            response_time = 0
        
        # Count number of adjustments (significant changes in knob value)
        knob_diff = np.abs(df_clean[' Knob'].diff())
        adjustment_threshold = 0.01  # Threshold for considering a change significant
        num_adjustments = np.sum(knob_diff > adjustment_threshold)
        
        return {
            'trial': trial,
            'knob_mean': knob_mean,
            'knob_std': knob_std,
            'knob_median': knob_median,
            'response_stability': response_stability,
            'velocity_mean': velocity_mean,
            'velocity_std': velocity_std,
            'function_ratio_mean': function_ratio_mean,
            'function_ratio_std': function_ratio_std,
            'response_time': response_time,
            'num_adjustments': num_adjustments,
            'data_points': len(df_clean)
        }
    
    def calculate_speed_equivalence(self, function_mix_results, phase_results):
        """Calculate speed equivalence metrics"""
        print("\n=== Speed Equivalence Analysis ===")
        
        equivalence_results = {}
        
        for participant in self.participants:
            if participant not in function_mix_results or participant not in phase_results:
                continue
                
            # Get Function Mix baseline
            function_mix_data = function_mix_results[participant]
            function_mix_knob_values = [trial['knob_mean'] for trial in function_mix_data]
            baseline_knob = np.mean(function_mix_knob_values)
            baseline_std = np.std(function_mix_knob_values)
            
            # Analyze Phase experiment results
            linear_only_data = phase_results[participant]['LinearOnly']
            dynamic_data = phase_results[participant]['Dynamic']
            
            # Calculate equivalence for LinearOnly
            if linear_only_data:
                linear_knob_values = [trial['knob_mean'] for trial in linear_only_data]
                linear_mean = np.mean(linear_knob_values)
                linear_std = np.std(linear_knob_values)
                linear_equivalence = abs(linear_mean - baseline_knob) / baseline_knob * 100
                
            # Calculate equivalence for Dynamic
            if dynamic_data:
                dynamic_knob_values = [trial['knob_mean'] for trial in dynamic_data]
                dynamic_mean = np.mean(dynamic_knob_values)
                dynamic_std = np.std(dynamic_knob_values)
                dynamic_equivalence = abs(dynamic_mean - baseline_knob) / baseline_knob * 100
                
            equivalence_results[participant] = {
                'baseline_knob': baseline_knob,
                'baseline_std': baseline_std,
                'linear_mean': linear_mean if linear_only_data else None,
                'linear_std': linear_std if linear_only_data else None,
                'linear_equivalence': linear_equivalence if linear_only_data else None,
                'dynamic_mean': dynamic_mean if dynamic_data else None,
                'dynamic_std': dynamic_std if dynamic_data else None,
                'dynamic_equivalence': dynamic_equivalence if dynamic_data else None
            }
            
        return equivalence_results
    
    def generate_visualizations(self, function_mix_results, phase_results, equivalence_results):
        """Generate comprehensive visualizations"""
        print("\n=== Generating Visualizations ===")
        
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Function Mix Trial Consistency
        plt.subplot(2, 3, 1)
        self._plot_function_mix_consistency(function_mix_results)
        
        # 2. Phase Experiment Comparison
        plt.subplot(2, 3, 2)
        self._plot_phase_comparison(phase_results)
        
        # 3. Speed Equivalence Analysis
        plt.subplot(2, 3, 3)
        self._plot_speed_equivalence(equivalence_results)
        
        # 4. Response Stability Analysis
        plt.subplot(2, 3, 4)
        self._plot_response_stability(function_mix_results, phase_results)
        
        # 5. Individual Participant Performance
        plt.subplot(2, 3, 5)
        self._plot_individual_performance(function_mix_results, phase_results)
        
        # 6. Statistical Summary
        plt.subplot(2, 3, 6)
        self._plot_statistical_summary(equivalence_results)
        
        plt.tight_layout()
        plt.savefig('brightness_analysis_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def _plot_function_mix_consistency(self, results):
        """Plot Function Mix trial consistency"""
        data = []
        for participant, trials in results.items():
            for trial in trials:
                data.append({
                    'Participant': participant,
                    'Trial': trial['trial'],
                    'Knob Value': trial['knob_mean'],
                    'Stability': trial['response_stability']
                })
        
        df = pd.DataFrame(data)
        sns.boxplot(data=df, x='Participant', y='Knob Value')
        plt.title('Function Mix: Knob Value Consistency\nAcross Trials')
        plt.ylabel('Mean Knob Value')
        
    def _plot_phase_comparison(self, results):
        """Plot Phase experiment comparison"""
        data = []
        for participant, modes in results.items():
            for mode, trials in modes.items():
                for trial in trials:
                    data.append({
                        'Participant': participant,
                        'Mode': mode,
                        'Knob Value': trial['knob_mean']
                    })
        
        df = pd.DataFrame(data)
        sns.boxplot(data=df, x='Participant', y='Knob Value', hue='Mode')
        plt.title('Phase Experiment: LinearOnly vs Dynamic\nKnob Value Comparison')
        plt.ylabel('Mean Knob Value')
        
    def _plot_speed_equivalence(self, results):
        """Plot speed equivalence analysis"""
        participants = list(results.keys())
        linear_equiv = [results[p]['linear_equivalence'] for p in participants if results[p]['linear_equivalence'] is not None]
        dynamic_equiv = [results[p]['dynamic_equivalence'] for p in participants if results[p]['dynamic_equivalence'] is not None]
        
        x = np.arange(len(participants))
        width = 0.35
        
        plt.bar(x - width/2, linear_equiv, width, label='LinearOnly', alpha=0.7)
        plt.bar(x + width/2, dynamic_equiv, width, label='Dynamic', alpha=0.7)
        
        plt.xlabel('Participants')
        plt.ylabel('Speed Equivalence Error (%)')
        plt.title('Speed Equivalence: Deviation from Baseline\n(Lower is Better)')
        plt.xticks(x, participants)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
    def _plot_response_stability(self, function_mix_results, phase_results):
        """Plot response stability analysis"""
        data = []
        
        # Function Mix stability
        for participant, trials in function_mix_results.items():
            for trial in trials:
                data.append({
                    'Participant': participant,
                    'Experiment': 'FunctionMix',
                    'Stability': trial['response_stability']
                })
        
        # Phase stability
        for participant, modes in phase_results.items():
            for mode, trials in modes.items():
                for trial in trials:
                    data.append({
                        'Participant': participant,
                        'Experiment': f'Phase_{mode}',
                        'Stability': trial['response_stability']
                    })
        
        df = pd.DataFrame(data)
        sns.boxplot(data=df, x='Participant', y='Stability', hue='Experiment')
        plt.title('Response Stability Across Experiments\n(Lower is More Stable)')
        plt.ylabel('Response Stability (CV)')
        plt.yscale('log')
        
    def _plot_individual_performance(self, function_mix_results, phase_results):
        """Plot individual participant performance"""
        # This will show the learning/adaptation curve for each participant
        for i, participant in enumerate(self.participants):
            if participant in function_mix_results:
                trials = function_mix_results[participant]
                trial_nums = [t['trial'] for t in trials]
                knob_values = [t['knob_mean'] for t in trials]
                plt.plot(trial_nums, knob_values, 'o-', label=f'{participant}', alpha=0.7)
        
        plt.xlabel('Trial Number')
        plt.ylabel('Mean Knob Value')
        plt.title('Individual Learning Curves\n(Function Mix Experiment)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
    def _plot_statistical_summary(self, results):
        """Plot statistical summary"""
        # Create a summary table visualization
        summary_data = []
        for participant, data in results.items():
            summary_data.append([
                participant,
                f"{data['baseline_knob']:.3f}±{data['baseline_std']:.3f}",
                f"{data['linear_equivalence']:.1f}%" if data['linear_equivalence'] else 'N/A',
                f"{data['dynamic_equivalence']:.1f}%" if data['dynamic_equivalence'] else 'N/A'
            ])
        
        # Create table
        table = plt.table(cellText=summary_data,
                         colLabels=['Participant', 'Baseline\n(Mean±SD)', 'LinearOnly\nError (%)', 'Dynamic\nError (%)'],
                         cellLoc='center',
                         loc='center')
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2)
        
        plt.axis('off')
        plt.title('Statistical Summary Table\nSpeed Equivalence Results')
        
    def generate_report(self, function_mix_results, phase_results, equivalence_results):
        """Generate comprehensive analysis report"""
        print("\n=== Generating Analysis Report ===")
        
        report = []
        report.append("# 二視点輝度混合手法における主観的速度等価性測定と再現性の向上")
        report.append("## Data Analysis Report\n")
        
        # Basic statistics
        report.append("## 1. 実験概要")
        report.append(f"- 参加者数: {len(self.participants)} 名 ({', '.join(self.participants)})")
        report.append(f"- Function Mix実験: 各参加者6回実施")
        report.append(f"- Phase実験: LinearOnlyとDynamicの2条件")
        report.append("")
        
        # Function Mix Analysis
        report.append("## 2. Function Mix実験結果")
        report.append("### 2.1 試行間一貫性分析")
        
        for participant in self.participants:
            if participant in function_mix_results:
                trials = function_mix_results[participant]
                knob_values = [t['knob_mean'] for t in trials]
                mean_knob = np.mean(knob_values)
                std_knob = np.std(knob_values)
                cv = std_knob / mean_knob * 100
                
                report.append(f"- {participant}: 平均調整値 = {mean_knob:.3f}±{std_knob:.3f} (CV = {cv:.1f}%)")
        
        report.append("")
        
        # Phase Experiment Analysis
        report.append("## 3. Phase実験結果")
        report.append("### 3.1 輝度混合手法の比較")
        
        for participant in self.participants:
            if participant in phase_results:
                modes = phase_results[participant]
                
                linear_data = modes.get('LinearOnly', [])
                dynamic_data = modes.get('Dynamic', [])
                
                if linear_data:
                    linear_mean = np.mean([t['knob_mean'] for t in linear_data])
                    linear_std = np.std([t['knob_mean'] for t in linear_data])
                    report.append(f"- {participant} LinearOnly: {linear_mean:.3f}±{linear_std:.3f}")
                
                if dynamic_data:
                    dynamic_mean = np.mean([t['knob_mean'] for t in dynamic_data])
                    dynamic_std = np.std([t['knob_mean'] for t in dynamic_data])
                    report.append(f"- {participant} Dynamic: {dynamic_mean:.3f}±{dynamic_std:.3f}")
        
        report.append("")
        
        # Speed Equivalence Analysis
        report.append("## 4. 速度等価性分析")
        report.append("### 4.1 ベースラインからの偏差")
        
        for participant in self.participants:
            if participant in equivalence_results:
                data = equivalence_results[participant]
                report.append(f"- {participant}:")
                report.append(f"  - ベースライン: {data['baseline_knob']:.3f}±{data['baseline_std']:.3f}")
                
                if data['linear_equivalence']:
                    report.append(f"  - LinearOnly偏差: {data['linear_equivalence']:.1f}%")
                
                if data['dynamic_equivalence']:
                    report.append(f"  - Dynamic偏差: {data['dynamic_equivalence']:.1f}%")
        
        report.append("")
        
        # Statistical Analysis
        report.append("## 5. 統計的分析")
        
        # Calculate overall statistics
        all_linear_equiv = [data['linear_equivalence'] for data in equivalence_results.values() if data['linear_equivalence']]
        all_dynamic_equiv = [data['dynamic_equivalence'] for data in equivalence_results.values() if data['dynamic_equivalence']]
        
        if all_linear_equiv and all_dynamic_equiv:
            # Paired t-test
            t_stat, p_value = stats.ttest_rel(all_linear_equiv, all_dynamic_equiv)
            
            report.append(f"### 5.1 LinearOnly vs Dynamic (対応のあるt検定)")
            report.append(f"- LinearOnly平均偏差: {np.mean(all_linear_equiv):.1f}±{np.std(all_linear_equiv):.1f}%")
            report.append(f"- Dynamic平均偏差: {np.mean(all_dynamic_equiv):.1f}±{np.std(all_dynamic_equiv):.1f}%")
            report.append(f"- t統計量: {t_stat:.3f}")
            report.append(f"- p値: {p_value:.3f}")
            
            if p_value < 0.05:
                report.append("- **統計的に有意な差が認められた (p < 0.05)**")
            else:
                report.append("- 統計的に有意な差は認められなかった (p ≥ 0.05)")
        
        report.append("")
        
        # Conclusions
        report.append("## 6. 結論")
        report.append("### 6.1 主要な発見")
        
        # Determine which method is better
        if all_linear_equiv and all_dynamic_equiv:
            linear_mean = np.mean(all_linear_equiv)
            dynamic_mean = np.mean(all_dynamic_equiv)
            
            if dynamic_mean < linear_mean:
                report.append("- Dynamic混合手法の方が速度等価性が高い")
            else:
                report.append("- LinearOnly混合手法の方が速度等価性が高い")
        
        report.append("- 個人差が存在し、参加者によって最適な手法が異なる可能性がある")
        report.append("- 今後の研究では、個人適応型の混合手法の開発が重要")
        
        report.append("")
        report.append("### 6.2 研究の限界")
        report.append("- 参加者数が限られている")
        report.append("- 実験環境による影響を完全に排除できていない可能性")
        report.append("- 長期的な学習効果は評価されていない")
        
        # Save report
        with open('brightness_analysis_report.md', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print("Analysis report saved as 'brightness_analysis_report.md'")
        
        return '\n'.join(report)
    
    def run_complete_analysis(self):
        """Run complete analysis pipeline"""
        print("Starting comprehensive brightness data analysis...")
        
        # Load data
        self.load_data()
        
        # Analyze experiments
        function_mix_results = self.analyze_function_mix_experiment()
        phase_results = self.analyze_phase_experiment()
        
        # Calculate speed equivalence
        equivalence_results = self.calculate_speed_equivalence(function_mix_results, phase_results)
        
        # Generate visualizations
        self.generate_visualizations(function_mix_results, phase_results, equivalence_results)
        
        # Generate report
        report = self.generate_report(function_mix_results, phase_results, equivalence_results)
        
        print("\n=== Analysis Complete ===")
        print("Files generated:")
        print("- brightness_analysis_results.png")
        print("- brightness_analysis_report.md")
        
        return {
            'function_mix_results': function_mix_results,
            'phase_results': phase_results,
            'equivalence_results': equivalence_results,
            'report': report
        }

# Main execution
if __name__ == "__main__":
    analyzer = BrightnessDataAnalyzer()
    results = analyzer.run_complete_analysis()