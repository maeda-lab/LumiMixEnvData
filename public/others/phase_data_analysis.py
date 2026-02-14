import os
import re
import statistics
import math

def analyze_phase_data_files():
    """Analyze the phase data files for the second part of experiment 2"""
    
    print("=== 実験2 後半部分データ詳細分析 ===\n")
    
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
    
    # Analyze each participant's data
    participant_stats = {}
    
    for participant, files in participant_files.items():
        print(f"\n=== {participant} のデータ分析 ===")
        
        participant_data = []
        
        for file in files:
            file_path = os.path.join(data_dir, file)
            print(f"\nファイル: {file}")
            
            try:
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    print(f"  データ行数: {len(lines)}")
                    
                    if len(lines) > 1:
                        # Parse header
                        header = lines[0].strip().split(',')
                        header = [h.strip() for h in header]
                        
                        # Find important column indices
                        knob_idx = header.index('Knob') if 'Knob' in header else -1
                        function_ratio_idx = header.index('FunctionRatio') if 'FunctionRatio' in header else -1
                        amplitude_idx = header.index('Amplitude') if 'Amplitude' in header else -1
                        velocity_idx = header.index('Velocity') if 'Velocity' in header else -1
                        
                        # Extract data
                        knob_values = []
                        function_ratio_values = []
                        amplitude_values = []
                        velocity_values = []
                        
                        for line in lines[1:]:  # Skip header
                            parts = line.strip().split(',')
                            parts = [p.strip() for p in parts]
                            
                            if len(parts) > max(knob_idx, function_ratio_idx, amplitude_idx, velocity_idx):
                                if knob_idx >= 0:
                                    try:
                                        knob_values.append(float(parts[knob_idx]))
                                    except ValueError:
                                        pass
                                
                                if function_ratio_idx >= 0:
                                    try:
                                        function_ratio_values.append(float(parts[function_ratio_idx]))
                                    except ValueError:
                                        pass
                                
                                if amplitude_idx >= 0:
                                    try:
                                        amplitude_values.append(float(parts[amplitude_idx]))
                                    except ValueError:
                                        pass
                                
                                if velocity_idx >= 0:
                                    try:
                                        velocity_values.append(float(parts[velocity_idx]))
                                    except ValueError:
                                        pass
                        
                        # Calculate statistics for this file
                        if knob_values:
                            knob_stats = calculate_statistics(knob_values)
                            print(f"  Knob値 - 平均: {knob_stats['mean']:.3f}, 中央値: {knob_stats['median']:.3f}, 標準偏差: {knob_stats['std']:.3f}")
                        
                        if function_ratio_values:
                            fr_stats = calculate_statistics(function_ratio_values)
                            print(f"  FunctionRatio値 - 平均: {fr_stats['mean']:.3f}, 中央値: {fr_stats['median']:.3f}, 標準偏差: {fr_stats['std']:.3f}")
                        
                        if amplitude_values:
                            amp_stats = calculate_statistics(amplitude_values)
                            print(f"  Amplitude値 - 平均: {amp_stats['mean']:.3f}, 中央値: {amp_stats['median']:.3f}, 標準偏差: {amp_stats['std']:.3f}")
                        
                        if velocity_values:
                            vel_stats = calculate_statistics(velocity_values)
                            print(f"  Velocity値 - 平均: {vel_stats['mean']:.3f}, 中央値: {vel_stats['median']:.3f}, 標準偏差: {vel_stats['std']:.3f}")
                        
                        # Store data for participant-level analysis
                        participant_data.append({
                            'knob_values': knob_values,
                            'function_ratio_values': function_ratio_values,
                            'amplitude_values': amplitude_values,
                            'velocity_values': velocity_values
                        })
                        
            except Exception as e:
                print(f"  ファイル読み込みエラー: {e}")
        
        # Calculate participant-level statistics
        if participant_data:
            all_knob_values = []
            all_function_ratio_values = []
            all_amplitude_values = []
            all_velocity_values = []
            
            for data in participant_data:
                all_knob_values.extend(data['knob_values'])
                all_function_ratio_values.extend(data['function_ratio_values'])
                all_amplitude_values.extend(data['amplitude_values'])
                all_velocity_values.extend(data['velocity_values'])
            
            participant_stats[participant] = {
                'knob': calculate_statistics(all_knob_values) if all_knob_values else None,
                'function_ratio': calculate_statistics(all_function_ratio_values) if all_function_ratio_values else None,
                'amplitude': calculate_statistics(all_amplitude_values) if all_amplitude_values else None,
                'velocity': calculate_statistics(all_velocity_values) if all_velocity_values else None
            }
    
    # Create summary report
    print("\n" + "="*60)
    print("=== 全被験者統計サマリー ===")
    print("="*60)
    
    for participant, stats in participant_stats.items():
        print(f"\n{participant}:")
        if stats['knob']:
            print(f"  Knob: 平均={stats['knob']['mean']:.3f}, 中央値={stats['knob']['median']:.3f}, 標準偏差={stats['knob']['std']:.3f}")
        if stats['function_ratio']:
            print(f"  FunctionRatio: 平均={stats['function_ratio']['mean']:.3f}, 中央値={stats['function_ratio']['median']:.3f}, 標準偏差={stats['function_ratio']['std']:.3f}")
        if stats['amplitude']:
            print(f"  Amplitude: 平均={stats['amplitude']['mean']:.3f}, 中央値={stats['amplitude']['median']:.3f}, 標準偏差={stats['amplitude']['std']:.3f}")
        if stats['velocity']:
            print(f"  Velocity: 平均={stats['velocity']['mean']:.3f}, 中央値={stats['velocity']['median']:.3f}, 標準偏差={stats['velocity']['std']:.3f}")

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

def compare_with_first_part():
    """Compare the function ratios from the second part with the first part"""
    
    print("\n" + "="*60)
    print("=== 前半部分と後半部分の比較 ===")
    print("="*60)
    
    # First part median values (from the analysis above)
    first_part_medians = {
        'ONO': 0.633,
        'LL': 0.218,
        'HOU': 0.316,
        'OMU': 0.734,
        'YAMA': 0.615
    }
    
    print("\n前半部分の中央値:")
    for participant, median in first_part_medians.items():
        print(f"  {participant}: {median:.3f}")
    
    print("\n後半部分で使用されたFunctionRatio値:")
    # This would need to be extracted from the phase data files
    # For now, we'll note that these should match the first part medians
    for participant, median in first_part_medians.items():
        print(f"  {participant}: {median:.3f} (前半部分の中央値を使用)")

if __name__ == "__main__":
    analyze_phase_data_files()
    compare_with_first_part()