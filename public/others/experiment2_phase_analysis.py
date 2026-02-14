import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set up matplotlib for better plots
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

class Experiment2PhaseAnalyzer:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.participants = ['ONO', 'LL', 'HOU', 'OMU', 'YAMA']
        self.function_ratios = {
            'ONO': 0.583,
            'LL': 0.218,
            'HOU': 0.316,
            'OMU': 0.734,
            'YAMA': 0.615
        }
        self.raw_data = {}
        self.processed_data = {}
        
    def load_data(self):
        """Load all Phase experiment data files"""
        print("Loading Phase experiment data...")
        
        for participant in self.participants:
            self.raw_data[participant] = {}
            
            # Find all Phase files for this participant
            pattern = f"*ExperimentPattern_Phase_ParticipantName_{participant}_*BrightnessBlendMode_Dynamic.csv"
            files = list(self.data_dir.glob(pattern))
            
            print(f"Found {len(files)} Phase files for {participant}")
            
            for file in files:
                # Extract trial number from filename
                trial_match = re.search(r'TrialNumber_(\d+)', file.name)
                if trial_match:
                    trial_num = int(trial_match.group(1))
                    
                    try:
                        df = pd.read_csv(file)
                        # Clean column names
                        df.columns = df.columns.str.strip()
                        self.raw_data[participant][trial_num] = df
                        print(f"  Loaded {participant} Trial {trial_num}: {len(df)} rows")
                    except Exception as e:
                        print(f"  Error loading {file}: {e}")
                        
    def extract_parameters(self, df):
        """Extract speed perception parameters from adjustment data"""
        # Focus on the final adjustments (last 20% of the data)
        final_data = df.tail(int(len(df) * 0.2))
        
        # Extract knob values (speed adjustments)
        knob_values = final_data['Knob'].values
        velocity_values = final_data['Velocity'].values
        
        # Calculate statistics
        params = {
            'mean_knob': np.mean(knob_values),
            'std_knob': np.std(knob_values),
            'final_knob': knob_values[-1] if len(knob_values) > 0 else np.nan,
            'mean_velocity': np.mean(velocity_values),
            'std_velocity': np.std(velocity_values),
            'final_velocity': velocity_values[-1] if len(velocity_values) > 0 else np.nan,
            'adjustment_range': np.max(knob_values) - np.min(knob_values),
            'total_duration': final_data['Time'].iloc[-1] - final_data['Time'].iloc[0] if len(final_data) > 1 else 0
        }
        
        return params
    
    def analyze_speed_perception(self):
        """Analyze speed perception characteristics for each participant"""
        print("\nAnalyzing speed perception characteristics...")
        
        results = {}
        
        for participant in self.participants:
            if participant not in self.raw_data or not self.raw_data[participant]:
                continue
                
            participant_results = {
                'function_ratio': self.function_ratios[participant],
                'trials': {},
                'summary': {}
            }
            
            trial_params = []
            
            for trial_num, df in self.raw_data[participant].items():
                params = self.extract_parameters(df)
                participant_results['trials'][trial_num] = params
                trial_params.append(params)
                
            if trial_params:
                # Calculate summary statistics across trials
                summary = {
                    'n_trials': len(trial_params),
                    'mean_final_knob': np.mean([p['final_knob'] for p in trial_params if not np.isnan(p['final_knob'])]),
                    'std_final_knob': np.std([p['final_knob'] for p in trial_params if not np.isnan(p['final_knob'])]),
                    'mean_final_velocity': np.mean([p['final_velocity'] for p in trial_params if not np.isnan(p['final_velocity'])]),
                    'std_final_velocity': np.std([p['final_velocity'] for p in trial_params if not np.isnan(p['final_velocity'])]),
                    'mean_adjustment_range': np.mean([p['adjustment_range'] for p in trial_params]),
                    'consistency': 1 / (1 + np.mean([p['std_knob'] for p in trial_params]))  # Higher = more consistent
                }
                
                participant_results['summary'] = summary
                
            results[participant] = participant_results
            
        self.processed_data = results
        return results
    
    def create_summary_table(self):
        """Create a summary table of results"""
        print("\nCreating summary table...")
        
        summary_data = []
        
        for participant, data in self.processed_data.items():
            if 'summary' in data and data['summary']:
                summary = data['summary']
                summary_data.append({
                    'Participant': participant,
                    'Function Ratio': data['function_ratio'],
                    'N Trials': summary['n_trials'],
                    'Mean Final Knob': summary['mean_final_knob'],
                    'Std Final Knob': summary['std_final_knob'],
                    'Mean Final Velocity': summary['mean_final_velocity'],
                    'Std Final Velocity': summary['std_final_velocity'],
                    'Mean Adjustment Range': summary['mean_adjustment_range'],
                    'Consistency Score': summary['consistency']
                })
        
        df_summary = pd.DataFrame(summary_data)
        return df_summary
    
    def plot_results(self):
        """Create comprehensive plots of the results"""
        print("\nCreating plots...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Experiment 2 Phase Analysis: Speed Perception Characteristics', fontsize=16, fontweight='bold')
        
        # Prepare data for plotting
        participants = []
        function_ratios = []
        final_knobs = []
        final_velocities = []
        adjustment_ranges = []
        consistency_scores = []
        
        for participant, data in self.processed_data.items():
            if 'summary' in data and data['summary']:
                summary = data['summary']
                participants.append(participant)
                function_ratios.append(data['function_ratio'])
                final_knobs.append(summary['mean_final_knob'])
                final_velocities.append(summary['mean_final_velocity'])
                adjustment_ranges.append(summary['mean_adjustment_range'])
                consistency_scores.append(summary['consistency'])
        
        # Plot 1: Function Ratio vs Final Knob Setting
        axes[0, 0].scatter(function_ratios, final_knobs, s=100, alpha=0.7, color='blue')
        for i, participant in enumerate(participants):
            axes[0, 0].annotate(participant, (function_ratios[i], final_knobs[i]), 
                               xytext=(5, 5), textcoords='offset points')
        axes[0, 0].set_xlabel('Function Ratio')
        axes[0, 0].set_ylabel('Mean Final Knob Setting')
        axes[0, 0].set_title('Function Ratio vs Final Knob Setting')
        
        # Add trend line
        if len(function_ratios) > 1:
            z = np.polyfit(function_ratios, final_knobs, 1)
            p = np.poly1d(z)
            axes[0, 0].plot(function_ratios, p(function_ratios), "r--", alpha=0.8)
            
            # Calculate correlation
            corr, p_val = stats.pearsonr(function_ratios, final_knobs)
            axes[0, 0].text(0.05, 0.95, f'r = {corr:.3f}\np = {p_val:.3f}', 
                           transform=axes[0, 0].transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot 2: Function Ratio vs Final Velocity
        axes[0, 1].scatter(function_ratios, final_velocities, s=100, alpha=0.7, color='green')
        for i, participant in enumerate(participants):
            axes[0, 1].annotate(participant, (function_ratios[i], final_velocities[i]), 
                               xytext=(5, 5), textcoords='offset points')
        axes[0, 1].set_xlabel('Function Ratio')
        axes[0, 1].set_ylabel('Mean Final Velocity')
        axes[0, 1].set_title('Function Ratio vs Final Velocity')
        
        # Add trend line
        if len(function_ratios) > 1:
            z = np.polyfit(function_ratios, final_velocities, 1)
            p = np.poly1d(z)
            axes[0, 1].plot(function_ratios, p(function_ratios), "r--", alpha=0.8)
            
            # Calculate correlation
            corr, p_val = stats.pearsonr(function_ratios, final_velocities)
            axes[0, 1].text(0.05, 0.95, f'r = {corr:.3f}\np = {p_val:.3f}', 
                           transform=axes[0, 1].transAxes, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot 3: Function Ratio vs Adjustment Range
        axes[0, 2].scatter(function_ratios, adjustment_ranges, s=100, alpha=0.7, color='orange')
        for i, participant in enumerate(participants):
            axes[0, 2].annotate(participant, (function_ratios[i], adjustment_ranges[i]), 
                               xytext=(5, 5), textcoords='offset points')
        axes[0, 2].set_xlabel('Function Ratio')
        axes[0, 2].set_ylabel('Mean Adjustment Range')
        axes[0, 2].set_title('Function Ratio vs Adjustment Range')
        
        # Plot 4: Function Ratio vs Consistency Score
        axes[1, 0].scatter(function_ratios, consistency_scores, s=100, alpha=0.7, color='purple')
        for i, participant in enumerate(participants):
            axes[1, 0].annotate(participant, (function_ratios[i], consistency_scores[i]), 
                               xytext=(5, 5), textcoords='offset points')
        axes[1, 0].set_xlabel('Function Ratio')
        axes[1, 0].set_ylabel('Consistency Score')
        axes[1, 0].set_title('Function Ratio vs Consistency Score')
        
        # Plot 5: Bar chart of final knob settings
        axes[1, 1].bar(participants, final_knobs, alpha=0.7, color='skyblue')
        axes[1, 1].set_ylabel('Mean Final Knob Setting')
        axes[1, 1].set_title('Final Knob Settings by Participant')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Plot 6: Bar chart of consistency scores
        axes[1, 2].bar(participants, consistency_scores, alpha=0.7, color='lightcoral')
        axes[1, 2].set_ylabel('Consistency Score')
        axes[1, 2].set_title('Consistency Scores by Participant')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig('experiment2_phase_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_individual_trials(self):
        """Analyze individual trial data for each participant"""
        print("\nAnalyzing individual trials...")
        
        fig, axes = plt.subplots(len(self.participants), 1, figsize=(15, 4*len(self.participants)))
        if len(self.participants) == 1:
            axes = [axes]
        
        for i, participant in enumerate(self.participants):
            if participant not in self.processed_data:
                continue
                
            data = self.processed_data[participant]
            
            # Plot knob adjustments for each trial
            for trial_num in data['trials'].keys():
                trial_data = data['trials'][trial_num]
                axes[i].scatter(trial_num, trial_data['final_knob'], 
                              s=100, alpha=0.7, label=f'Trial {trial_num}')
            
            axes[i].set_title(f'{participant} - Function Ratio: {data["function_ratio"]:.3f}')
            axes[i].set_xlabel('Trial Number')
            axes[i].set_ylabel('Final Knob Setting')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            
            # Add mean line
            if data['trials']:
                final_knobs = [data['trials'][t]['final_knob'] for t in data['trials'].keys() 
                              if not np.isnan(data['trials'][t]['final_knob'])]
                if final_knobs:
                    mean_knob = np.mean(final_knobs)
                    axes[i].axhline(y=mean_knob, color='red', linestyle='--', alpha=0.7, 
                                   label=f'Mean: {mean_knob:.3f}')
        
        plt.tight_layout()
        plt.savefig('experiment2_individual_trials.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_report(self):
        """Generate a comprehensive analysis report"""
        print("\nGenerating analysis report...")
        
        report = []
        report.append("# Experiment 2 Phase Analysis Report")
        report.append("## Speed Perception Characteristics Analysis")
        report.append("")
        
        # Summary table
        summary_df = self.create_summary_table()
        report.append("## Summary Statistics")
        report.append("")
        report.append(summary_df.to_string(index=False, float_format='%.3f'))
        report.append("")
        
        # Individual participant analysis
        report.append("## Individual Participant Analysis")
        report.append("")
        
        for participant in self.participants:
            if participant not in self.processed_data:
                continue
                
            data = self.processed_data[participant]
            if 'summary' not in data or not data['summary']:
                continue
                
            summary = data['summary']
            
            report.append(f"### {participant}")
            report.append(f"- Function Ratio: {data['function_ratio']:.3f}")
            report.append(f"- Number of Trials: {summary['n_trials']}")
            report.append(f"- Mean Final Knob Setting: {summary['mean_final_knob']:.3f} ± {summary['std_final_knob']:.3f}")
            report.append(f"- Mean Final Velocity: {summary['mean_final_velocity']:.3f} ± {summary['std_final_velocity']:.3f}")
            report.append(f"- Mean Adjustment Range: {summary['mean_adjustment_range']:.3f}")
            report.append(f"- Consistency Score: {summary['consistency']:.3f}")
            report.append("")
        
        # Correlation analysis
        report.append("## Correlation Analysis")
        report.append("")
        
        if len(self.processed_data) > 1:
            # Extract data for correlation analysis
            function_ratios = []
            final_knobs = []
            final_velocities = []
            adjustment_ranges = []
            consistency_scores = []
            
            for participant, data in self.processed_data.items():
                if 'summary' in data and data['summary']:
                    summary = data['summary']
                    function_ratios.append(data['function_ratio'])
                    final_knobs.append(summary['mean_final_knob'])
                    final_velocities.append(summary['mean_final_velocity'])
                    adjustment_ranges.append(summary['mean_adjustment_range'])
                    consistency_scores.append(summary['consistency'])
            
            if len(function_ratios) > 1:
                # Calculate correlations
                corr_knob, p_knob = stats.pearsonr(function_ratios, final_knobs)
                corr_vel, p_vel = stats.pearsonr(function_ratios, final_velocities)
                corr_range, p_range = stats.pearsonr(function_ratios, adjustment_ranges)
                corr_cons, p_cons = stats.pearsonr(function_ratios, consistency_scores)
                
                report.append(f"- Function Ratio vs Final Knob: r = {corr_knob:.3f}, p = {p_knob:.3f}")
                report.append(f"- Function Ratio vs Final Velocity: r = {corr_vel:.3f}, p = {p_vel:.3f}")
                report.append(f"- Function Ratio vs Adjustment Range: r = {corr_range:.3f}, p = {p_range:.3f}")
                report.append(f"- Function Ratio vs Consistency: r = {corr_cons:.3f}, p = {p_cons:.3f}")
                report.append("")
        
        # Key findings
        report.append("## Key Findings")
        report.append("")
        
        # Find participant with highest/lowest function ratio
        if self.processed_data:
            sorted_participants = sorted(self.processed_data.items(), 
                                       key=lambda x: x[1]['function_ratio'])
            
            lowest_participant = sorted_participants[0]
            highest_participant = sorted_participants[-1]
            
            report.append(f"- Participant with lowest function ratio: {lowest_participant[0]} ({lowest_participant[1]['function_ratio']:.3f})")
            report.append(f"- Participant with highest function ratio: {highest_participant[0]} ({highest_participant[1]['function_ratio']:.3f})")
            report.append("")
            
            # Compare their performance
            if 'summary' in lowest_participant[1] and 'summary' in highest_participant[1]:
                low_summary = lowest_participant[1]['summary']
                high_summary = highest_participant[1]['summary']
                
                report.append("- Performance comparison:")
                report.append(f"  - Final knob setting: {low_summary['mean_final_knob']:.3f} vs {high_summary['mean_final_knob']:.3f}")
                report.append(f"  - Consistency: {low_summary['consistency']:.3f} vs {high_summary['consistency']:.3f}")
                report.append("")
        
        # Save report
        with open('experiment2_phase_analysis_report.md', 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print("Report saved to: experiment2_phase_analysis_report.md")
        return '\n'.join(report)

def main():
    # Initialize analyzer
    analyzer = Experiment2PhaseAnalyzer('public/BrightnessFunctionMixAndPhaseData')
    
    # Run analysis
    analyzer.load_data()
    analyzer.analyze_speed_perception()
    
    # Create summary table
    summary_df = analyzer.create_summary_table()
    print("\nSummary Table:")
    print(summary_df.to_string(index=False, float_format='%.3f'))
    
    # Create plots
    analyzer.plot_results()
    analyzer.analyze_individual_trials()
    
    # Generate report
    report = analyzer.generate_report()
    
    return analyzer, summary_df, report

if __name__ == "__main__":
    analyzer, summary_df, report = main()