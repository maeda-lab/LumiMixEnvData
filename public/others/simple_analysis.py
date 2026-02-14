import math
import statistics

# Function ratio data provided by user
participant_data = {
    'ONO': [0.517, 0.713, 0.581, 0.583, 0.684, 1.0],
    'LL': [0.0, 0.492, 0.471, 0.231, 0.178, 0.205],
    'HOU': [0.163, 0.206, 0.555, 0.336, 0.295, 0.712],
    'OMU': [0.817, 0.651, 0.551, 0.84, 0.582, 0.841],
    'YAMA': [0.683, 0.616, 0.785, 0.583, 0.613, 0.581]
}

def calculate_statistics(data):
    """Calculate basic statistics for a list of numbers"""
    if not data:
        return None
    
    n = len(data)
    mean_val = sum(data) / n
    median_val = statistics.median(data)
    
    # Calculate standard deviation
    variance = sum((x - mean_val) ** 2 for x in data) / n
    std_val = math.sqrt(variance)
    
    min_val = min(data)
    max_val = max(data)
    
    return {
        'count': n,
        'mean': mean_val,
        'median': median_val,
        'std': std_val,
        'min': min_val,
        'max': max_val
    }

def analyze_function_ratios():
    """Analyze the function ratio data from experiment 2"""
    
    print("=== 実験2 前半部分データ分析 ===\n")
    
    # Calculate statistics for each participant
    results = {}
    
    for participant, ratios in participant_data.items():
        stats = calculate_statistics(ratios)
        results[participant] = stats
        
        print(f"被験者 {participant}:")
        print(f"  6回の試行値: {ratios}")
        print(f"  平均値: {stats['mean']:.3f}")
        print(f"  中央値: {stats['median']:.3f}")
        print(f"  標準偏差: {stats['std']:.3f}")
        print(f"  最小値: {stats['min']:.3f}")
        print(f"  最大値: {stats['max']:.3f}")
        print()
    
    # Create summary table
    print("=== 統計サマリー ===")
    print(f"{'被験者':<8} {'平均':<8} {'中央値':<8} {'標準偏差':<8} {'最小値':<8} {'最大値':<8}")
    print("-" * 60)
    for participant, stats in results.items():
        print(f"{participant:<8} {stats['mean']:<8.3f} {stats['median']:<8.3f} {stats['std']:<8.3f} {stats['min']:<8.3f} {stats['max']:<8.3f}")
    print()
    
    # Overall statistics
    all_ratios = []
    for ratios in participant_data.values():
        all_ratios.extend(ratios)
    
    overall_stats = calculate_statistics(all_ratios)
    
    print("=== 全体統計 ===")
    print(f"全被験者の総試行数: {overall_stats['count']}")
    print(f"全体平均: {overall_stats['mean']:.3f}")
    print(f"全体中央値: {overall_stats['median']:.3f}")
    print(f"全体標準偏差: {overall_stats['std']:.3f}")
    print(f"全体最小値: {overall_stats['min']:.3f}")
    print(f"全体最大値: {overall_stats['max']:.3f}")
    print()
    
    # Analyze consistency using coefficient of variation
    print("=== 一貫性分析 (変動係数) ===")
    for participant, stats in results.items():
        cv = stats['std'] / stats['mean'] if stats['mean'] != 0 else float('inf')
        print(f"{participant}: CV = {cv:.3f}")
    print()
    
    # Find the median values for each participant (these will be used in the second part)
    print("=== 後半実験で使用する中央値 ===")
    for participant, stats in results.items():
        print(f"{participant}: {stats['median']:.3f}")
    print()
    
    # Analyze the range of values
    print("=== 値の範囲分析 ===")
    for participant, stats in results.items():
        range_val = stats['max'] - stats['min']
        print(f"{participant}: 範囲 = {range_val:.3f} (最小: {stats['min']:.3f}, 最大: {stats['max']:.3f})")
    print()
    
    # Check for outliers (values more than 2 standard deviations from mean)
    print("=== 外れ値分析 (平均±2標準偏差) ===")
    for participant, stats in results.items():
        ratios = participant_data[participant]
        lower_bound = stats['mean'] - 2 * stats['std']
        upper_bound = stats['mean'] + 2 * stats['std']
        
        outliers = [r for r in ratios if r < lower_bound or r > upper_bound]
        if outliers:
            print(f"{participant}: 外れ値 = {outliers}")
        else:
            print(f"{participant}: 外れ値なし")
    print()
    
    return results

def analyze_phase_data_files():
    """Analyze the phase data files for the second part of experiment 2"""
    
    print("=== 実験2 後半部分データ分析 ===")
    
    import os
    import re
    
    data_dir = "public/BrightnessFunctionMixAndPhaseData"
    
    if not os.path.exists(data_dir):
        print(f"データディレクトリが見つかりません: {data_dir}")
        return
    
    # Find all phase data files
    phase_files = []
    for file in os.listdir(data_dir):
        if "Phase" in file and "Dynamic" in file and file.endswith(".csv"):
            phase_files.append(file)
    
    print(f"発見されたPhaseデータファイル数: {len(phase_files)}")
    
    # Group files by participant
    participant_files = {}
    for file in phase_files:
        # Extract participant name from filename
        match = re.search(r'ParticipantName_(\w+)_', file)
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
        example_file = os.path.join(data_dir, phase_files[0])
        print(f"\n例として {phase_files[0]} を分析:")
        
        try:
            with open(example_file, 'r') as f:
                lines = f.readlines()
                print(f"  データ行数: {len(lines)}")
                
                if lines:
                    # Parse header
                    header = lines[0].strip().split(',')
                    print(f"  カラム: {header}")
                    
                    # Show first few data rows
                    print("\n  最初の5行:")
                    for i in range(1, min(6, len(lines))):
                        print(f"    {lines[i].strip()}")
                        
        except Exception as e:
            print(f"  ファイル読み込みエラー: {e}")

if __name__ == "__main__":
    # Analyze function ratio data (first part of experiment 2)
    results = analyze_function_ratios()
    
    # Analyze phase data (second part of experiment 2)
    analyze_phase_data_files()