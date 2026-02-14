import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re

# Function ratio data provided by user
participant_data = {
    'ONO': [0.517, 0.713, 0.581, 0.583, 0.684, 1.0],
    'LL': [0.0, 0.492, 0.471, 0.231, 0.178, 0.205],
    'HOU': [0.163, 0.206, 0.555, 0.336, 0.295, 0.712],
    'OMU': [0.817, 0.651, 0.551, 0.84, 0.582, 0.841],
    'YAMA': [0.683, 0.616, 0.785, 0.583, 0.613, 0.581]
}

def analyze_function_ratios():
    """Analyze the function ratio data from experiment 2"""
    
    print("=== 実験2 前半部分データ分析 ===\n")
    
    # Calculate statistics for each participant
    results = {}
    
    for participant, ratios in participant_data.items():
        ratios_array = np.array(ratios)
        
        # Calculate statistics
        mean_val = np.mean(ratios_array)
        median_val = np.median(ratios_array)
        std_val = np.std(ratios_array)
        min_val = np.min(ratios_array)
        max_val = np.max(ratios_array)
        
        results[participant] = {
            'trials': ratios,
            'mean': mean_val,
            'median': median_val,
            'std': std_val,
            'min': min_val,
            'max': max_val
        }
        
        print(f"被験者 {participant}:")
        print(f"  6回の試行値: {ratios}")
        print(f"  平均値: {mean_val:.3f}")
        print(f"  中央値: {median_val:.3f}")
        print(f"  標準偏差: {std_val:.3f}")
        print(f"  最小値: {min_val:.3f}")
        print(f"  最大値: {max_val:.3f}")
        print()
    
    # Create summary table
    print("=== 統計サマリー ===")
    summary_data = []
    for participant, stats in results.items():
        summary_data.append({
            'Participant': participant,
            'Mean': f"{stats['mean']:.3f}",
            'Median': f"{stats['median']:.3f}",
            'Std': f"{stats['std']:.3f}",
            'Min': f"{stats['min']:.3f}",
            'Max': f"{stats['max']:.3f}"
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    print()
    
    # Overall statistics
    all_ratios = []
    for ratios in participant_data.values():
        all_ratios.extend(ratios)
    
    print("=== 全体統計 ===")
    print(f"全被験者の総試行数: {len(all_ratios)}")
    print(f"全体平均: {np.mean(all_ratios):.3f}")
    print(f"全体中央値: {np.median(all_ratios):.3f}")
    print(f"全体標準偏差: {np.std(all_ratios):.3f}")
    print(f"全体最小値: {np.min(all_ratios):.3f}")
    print(f"全体最大値: {np.max(all_ratios):.3f}")
    print()
    
    # Create visualizations
    create_visualizations(results, all_ratios)
    
    return results

def create_visualizations(results, all_ratios):
    """Create visualizations for the function ratio data"""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Box plot of all participants
    participants = list(results.keys())
    ratios_list = [results[p]['trials'] for p in participants]
    
    ax1.boxplot(ratios_list, labels=participants)
    ax1.set_title('Function Ratio Distribution by Participant', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Function Ratio')
    ax1.set_xlabel('Participant')
    ax1.grid(True, alpha=0.3)
    
    # 2. Individual trial values
    for i, participant in enumerate(participants):
        trials = results[participant]['trials']
        ax2.scatter([i] * len(trials), trials, alpha=0.7, s=100, label=participant)
        ax2.axhline(y=results[participant]['median'], color='red', linestyle='--', alpha=0.7)
    
    ax2.set_title('Individual Trial Values with Median Lines', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Function Ratio')
    ax2.set_xlabel('Participant')
    ax2.set_xticks(range(len(participants)))
    ax2.set_xticklabels(participants)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Histogram of all values
    ax3.hist(all_ratios, bins=15, alpha=0.7, edgecolor='black')
    ax3.axvline(np.mean(all_ratios), color='red', linestyle='--', label=f'Mean: {np.mean(all_ratios):.3f}')
    ax3.axvline(np.median(all_ratios), color='green', linestyle='--', label=f'Median: {np.median(all_ratios):.3f}')
    ax3.set_title('Distribution of All Function Ratios', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Function Ratio')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Bar plot of medians
    medians = [results[p]['median'] for p in participants]
    bars = ax4.bar(participants, medians, alpha=0.7)
    ax4.set_title('Median Function Ratios by Participant', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Median Function Ratio')
    ax4.set_xlabel('Participant')
    
    # Add value labels on bars
    for bar, median in zip(bars, medians):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{median:.3f}', ha='center', va='bottom')
    
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('experiment2_function_ratio_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create detailed trial analysis
    create_trial_analysis(results)

def create_trial_analysis(results):
    """Create detailed analysis of individual trials"""
    
    print("=== 試行別詳細分析 ===")
    
    # Create trial-by-trial comparison
    trial_data = []
    for participant, stats in results.items():
        for i, ratio in enumerate(stats['trials'], 1):
            trial_data.append({
                'Participant': participant,
                'Trial': i,
                'Ratio': ratio,
                'Deviation_from_Median': ratio - stats['median']
            })
    
    trial_df = pd.DataFrame(trial_data)
    
    print("\n試行別データ:")
    print(trial_df.to_string(index=False))
    
    # Analyze consistency
    print("\n=== 一貫性分析 ===")
    for participant, stats in results.items():
        ratios = np.array(stats['trials'])
        cv = stats['std'] / stats['mean'] if stats['mean'] != 0 else float('inf')
        print(f"{participant}: 変動係数 (CV) = {cv:.3f}")
    
    # Save detailed results
    trial_df.to_csv('experiment2_trial_analysis.csv', index=False)
    print(f"\n詳細データを 'experiment2_trial_analysis.csv' に保存しました。")

def analyze_phase_data():
    """Analyze the phase data files for the second part of experiment 2"""
    
    print("\n=== 実験2 後半部分データ分析 ===")
    
    data_dir = Path("public/BrightnessFunctionMixAndPhaseData")
    
    # Find all phase data files
    phase_files = list(data_dir.glob("*Phase*Dynamic.csv"))
    
    print(f"発見されたPhaseデータファイル数: {len(phase_files)}")
    
    # Group files by participant
    participant_files = {}
    for file in phase_files:
        # Extract participant name from filename
        match = re.search(r'ParticipantName_(\w+)_', file.name)
        if match:
            participant = match.group(1)
            if participant not in participant_files:
                participant_files[participant] = []
            participant_files[participant].append(file)
    
    print(f"\n被験者別ファイル数:")
    for participant, files in participant_files.items():
        print(f"  {participant}: {len(files)} ファイル")
    
    # Analyze one file as example
    if phase_files:
        example_file = phase_files[0]
        print(f"\n例として {example_file.name} を分析:")
        
        try:
            df = pd.read_csv(example_file)
            print(f"  データ行数: {len(df)}")
            print(f"  カラム: {list(df.columns)}")
            
            # Show first few rows
            print("\n  最初の5行:")
            print(df.head().to_string())
            
        except Exception as e:
            print(f"  ファイル読み込みエラー: {e}")

if __name__ == "__main__":
    # Analyze function ratio data (first part of experiment 2)
    results = analyze_function_ratios()
    
    # Analyze phase data (second part of experiment 2)
    analyze_phase_data()