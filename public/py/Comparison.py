import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# File paths for different experimental conditions
file_paths = [
    '/Users/jasmine/Documents/GitHub/vectionProject/ExperimentData/20241105_180723_continuous_cameraSpeed4_fps60_I_trialNumber1.csv',
    '/Users/jasmine/Documents/GitHub/vectionProject/ExperimentData/20241105_181512_continuous_cameraSpeed4_fps60_I_trialNumber2.csv',
    '/Users/jasmine/Documents/GitHub/vectionProject/ExperimentData/20241105_182431_continuous_cameraSpeed4_fps60_I_trialNumber3.csv',

    '/Users/jasmine/Documents/GitHub/vectionProject/ExperimentData/20241106_151409_continuous_cameraSpeed4_fps60_O_trialNumber1.csv',
    '/Users/jasmine/Documents/GitHub/vectionProject/ExperimentData/20241106_152000_continuous_cameraSpeed4_fps60_O_trialNumber2.csv',
    '/Users/jasmine/Documents/GitHub/vectionProject/ExperimentData/20241106_152448_continuous_cameraSpeed4_fps60_O_trialNumber3.csv',

    '/Users/jasmine/Documents/GitHub/vectionProject/ExperimentData/20241113_150930_continuous_cameraSpeed4_fps60_G_trialNumber1.csv',
    '/Users/jasmine/Documents/GitHub/vectionProject/ExperimentData/20241113_151635_continuous_cameraSpeed4_fps60_G_trialNumber2.csv',
    '/Users/jasmine/Documents/GitHub/vectionProject/ExperimentData/20241113_152721_continuous_cameraSpeed4_fps60_G_trialNumber3.csv',

    '/Users/jasmine/Documents/GitHub/vectionProject/ExperimentData/20241118_161948_continuous_cameraSpeed4_fps60_K_trialNumber1.csv',
    '/Users/jasmine/Documents/GitHub/vectionProject/ExperimentData/20241118_162422_continuous_cameraSpeed4_fps60_K_trialNumber2.csv',
    '/Users/jasmine/Documents/GitHub/vectionProject/ExperimentData/20241118_162911_continuous_cameraSpeed4_fps60_K_trialNumber3.csv',

    '/Users/jasmine/Documents/GitHub/vectionProject/ExperimentData/20241118_143922_continuous_cameraSpeed4_fps60_Y_trialNumber1.csv',
    '/Users/jasmine/Documents/GitHub/vectionProject/ExperimentData/20241118_144959_continuous_cameraSpeed4_fps60_Y_trialNumber2.csv',
    '/Users/jasmine/Documents/GitHub/vectionProject/ExperimentData/20241118_145508_continuous_cameraSpeed4_fps60_Y_trialNumber3.csv',

    '/Users/jasmine/Documents/GitHub/vectionProject/ExperimentData/20241122_170146_continuous_cameraSpeed4_fps60_A_trialNumber1.csv',
    '/Users/jasmine/Documents/GitHub/vectionProject/ExperimentData/20241122_170704_continuous_cameraSpeed4_fps60_A_trialNumber2.csv',
    '/Users/jasmine/Documents/GitHub/vectionProject/ExperimentData/20241122_171154_continuous_cameraSpeed4_fps60_A_trialNumber3.csv',
]

luminance_mixture_paths = {
    '5 fps': [
        '/Users/jasmine/Documents/GitHub/vectionProject/ExperimentData/20241113_154414_luminanceMixture_cameraSpeed4_fps5_G_trialNumber1.csv',
        '/Users/jasmine/Documents/GitHub/vectionProject/ExperimentData/20241113_155123_luminanceMixture_cameraSpeed4_fps5_G_trialNumber2.csv',
        '/Users/jasmine/Documents/GitHub/vectionProject/ExperimentData/20241113_155817_luminanceMixture_cameraSpeed4_fps5_G_trialNumber3.csv',

        '/Users/jasmine/Documents/GitHub/vectionProject/ExperimentData/20241118_163429_luminanceMixture_cameraSpeed4_fps5_K_trialNumber1.csv',
        '/Users/jasmine/Documents/GitHub/vectionProject/ExperimentData/20241118_163949_luminanceMixture_cameraSpeed4_fps5_K_trialNumber2.csv',
        '/Users/jasmine/Documents/GitHub/vectionProject/ExperimentData/20241118_164408_luminanceMixture_cameraSpeed4_fps5_K_trialNumber3.csv',

        '/Users/jasmine/Documents/GitHub/vectionProject/ExperimentData/20241118_150020_luminanceMixture_cameraSpeed4_fps5_Y_trialNumber1.csv',
        '/Users/jasmine/Documents/GitHub/vectionProject/ExperimentData/20241118_150543_luminanceMixture_cameraSpeed4_fps5_Y_trialNumber2.csv',
        '/Users/jasmine/Documents/GitHub/vectionProject/ExperimentData/20241118_151108_luminanceMixture_cameraSpeed4_fps5_Y_trialNumber3.csv',

        '/Users/jasmine/Documents/GitHub/vectionProject/ExperimentData/20241122_171609_luminanceMixture_cameraSpeed4_fps5_A_trialNumber1.csv',
        '/Users/jasmine/Documents/GitHub/vectionProject/ExperimentData/20241122_172016_luminanceMixture_cameraSpeed4_fps5_A_trialNumber2.csv',
        '/Users/jasmine/Documents/GitHub/vectionProject/ExperimentData/20241122_172417_luminanceMixture_cameraSpeed4_fps5_A_trialNumber3.csv',
    ],
    '10 fps': [
        '/Users/jasmine/Documents/GitHub/vectionProject/ExperimentData/20241113_161351_luminanceMixture_cameraSpeed4_fps10_G_trialNumber1.csv',
        '/Users/jasmine/Documents/GitHub/vectionProject/ExperimentData/20241113_162151_luminanceMixture_cameraSpeed4_fps10_G_trialNumber2.csv',
        '/Users/jasmine/Documents/GitHub/vectionProject/ExperimentData/20241113_162914_luminanceMixture_cameraSpeed4_fps10_G_trialNumber3.csv',

        '/Users/jasmine/Documents/GitHub/vectionProject/ExperimentData/20241118_165443_luminanceMixture_cameraSpeed4_fps10_K_trialNumber1.csv',
        '/Users/jasmine/Documents/GitHub/vectionProject/ExperimentData/20241118_170330_luminanceMixture_cameraSpeed4_fps10_K_trialNumber2.csv',
        '/Users/jasmine/Documents/GitHub/vectionProject/ExperimentData/20241118_171626_luminanceMixture_cameraSpeed4_fps10_K_trialNumber3.csv',

        '/Users/jasmine/Documents/GitHub/vectionProject/ExperimentData/20241118_151809_luminanceMixture_cameraSpeed4_fps10_Y_trialNumber1.csv',
        '/Users/jasmine/Documents/GitHub/vectionProject/ExperimentData/20241118_152450_luminanceMixture_cameraSpeed4_fps10_Y_trialNumber2.csv',
        '/Users/jasmine/Documents/GitHub/vectionProject/ExperimentData/20241118_153346_luminanceMixture_cameraSpeed4_fps10_Y_trialNumber3.csv',

        '/Users/jasmine/Documents/GitHub/vectionProject/ExperimentData/20241122_173828_luminanceMixture_cameraSpeed4_fps10_A_trialNumber1.csv',
        '/Users/jasmine/Documents/GitHub/vectionProject/ExperimentData/20241122_174243_luminanceMixture_cameraSpeed4_fps10_A_trialNumber2.csv',
        '/Users/jasmine/Documents/GitHub/vectionProject/ExperimentData/20241122_174654_luminanceMixture_cameraSpeed4_fps10_A_trialNumber3.csv',
        
    ],
    '30 fps': [
        '/Users/jasmine/Documents/GitHub/vectionProject/ExperimentData/20241113_163839_luminanceMixture_cameraSpeed4_fps30_G_trialNumber1.csv',
        '/Users/jasmine/Documents/GitHub/vectionProject/ExperimentData/20241113_164516_luminanceMixture_cameraSpeed4_fps30_G_trialNumber2.csv',
        '/Users/jasmine/Documents/GitHub/vectionProject/ExperimentData/20241113_165322_luminanceMixture_cameraSpeed4_fps30_G_trialNumber3.csv',
        
        '/Users/jasmine/Documents/GitHub/vectionProject/ExperimentData/20241118_172549_luminanceMixture_cameraSpeed4_fps30_K_trialNumber1.csv',
        '/Users/jasmine/Documents/GitHub/vectionProject/ExperimentData/20241118_173417_luminanceMixture_cameraSpeed4_fps30_K_trialNumber2.csv',
        '/Users/jasmine/Documents/GitHub/vectionProject/ExperimentData/20241118_174225_luminanceMixture_cameraSpeed4_fps30_K_trialNumber3.csv',

        '/Users/jasmine/Documents/GitHub/vectionProject/ExperimentData/20241118_154342_luminanceMixture_cameraSpeed4_fps30_Y_trialNumber1.csv',
        '/Users/jasmine/Documents/GitHub/vectionProject/ExperimentData/20241118_155057_luminanceMixture_cameraSpeed4_fps30_Y_trialNumber2.csv',
        '/Users/jasmine/Documents/GitHub/vectionProject/ExperimentData/20241118_160156_luminanceMixture_cameraSpeed4_fps30_Y_trialNumber3.csv',

        '/Users/jasmine/Documents/GitHub/vectionProject/ExperimentData/20241122_175110_luminanceMixture_cameraSpeed4_fps30_A_trialNumber1.csv',
        '/Users/jasmine/Documents/GitHub/vectionProject/ExperimentData/20241122_180111_luminanceMixture_cameraSpeed4_fps30_A_trialNumber2.csv',
        '/Users/jasmine/Documents/GitHub/vectionProject/ExperimentData/20241122_180524_luminanceMixture_cameraSpeed4_fps30_A_trialNumber3.csv',
    ]
}

# Process the files to compute latent time and duration time for each participant
participant_latent_times = {}
participant_duration_times = {}

for file_path in file_paths:
    participant_name = file_path.split('_')[5]
    if participant_name not in participant_latent_times:
        participant_latent_times[participant_name] = []
        participant_duration_times[participant_name] = []
    
    df = pd.read_csv(file_path)
    time = df['Time'] / 1000  # Convert time to seconds
    vection_response = df['Vection Response']
    
    # Calculate latent time: time of first occurrence of 1
    if (vection_response == 1).any():
        first_occurrence_index = vection_response[vection_response == 1].index[0]
        latent_time = time[first_occurrence_index]
    else:
        latent_time = 'no'
        duration_time = 0
    
    # Calculate duration time: total time where response is 1
    if latent_time != 'no':
        time_diff = time.diff().fillna(0)
        duration_time = time_diff[vection_response == 1].sum()
    
    # Store the data
    participant_latent_times[participant_name].append(latent_time)
    participant_duration_times[participant_name].append(duration_time)

# Calculate average latent time for each participant
avg_latent_times = {
    participant: np.mean([t for t in participant_latent_times[participant] if t != 'no']) if len([t for t in participant_latent_times[participant] if t != 'no']) > 0 else 'no'
    for participant in participant_latent_times
}
avg_duration_times = {
    participant: np.mean(participant_duration_times[participant])
    for participant in participant_duration_times
}
combined_latent_time = np.mean([t for t in avg_latent_times.values() if t != 'no']) if len([t for t in avg_latent_times.values() if t != 'no']) > 0 else 'no'
combined_duration_time = np.mean(list(avg_duration_times.values()))
combined_latent_std = np.std([t for t in avg_latent_times.values() if t != 'no']) if len([t for t in avg_latent_times.values() if t != 'no']) > 1 else 0
combined_duration_std = np.std(list(avg_duration_times.values()))

# Process luminance mixture conditions
luminance_latent_times = {}
luminance_duration_times = {}
luminance_latent_times_std = {}
luminance_duration_times_std = {}

for condition, paths in luminance_mixture_paths.items():
    participant_latent_times = {}
    participant_duration_times = {}
    
    for file_path in paths:
        participant_name = file_path.split('_')[5]
        df = pd.read_csv(file_path)
        time = df['Time'] / 1000  # Convert time to seconds
        vection_response = df['Vection Response']
        
        # Calculate latent time: time of first occurrence of 1
        if (vection_response == 1).any():
            first_occurrence_index = vection_response[vection_response == 1].index[0]
            latent_time = time[first_occurrence_index]
        else:
            latent_time = 'no'
            duration_time = 0
        
        # Calculate duration time: total time where response is 1
        if latent_time != 'no':
            time_diff = time.diff().fillna(0)
            duration_time = time_diff[vection_response == 1].sum()
        
        if participant_name not in participant_latent_times:
            participant_latent_times[participant_name] = []
            participant_duration_times[participant_name] = []
        
        participant_latent_times[participant_name].append(latent_time)
        participant_duration_times[participant_name].append(duration_time)
    
    luminance_latent_times[condition] = [
        np.mean([t for t in participant_latent_times[participant] if t != 'no']) if len([t for t in participant_latent_times[participant] if t != 'no']) > 0 else 'no'
        for participant in participant_latent_times
    ]
    luminance_duration_times[condition] = [
        np.mean(participant_duration_times[participant])
        for participant in participant_duration_times
    ]
    # Calculate standard deviation
    luminance_latent_times_std[condition] = [
        np.std([t for t in participant_latent_times[participant] if t != 'no']) if len([t for t in participant_latent_times[participant] if t != 'no']) > 0 else 0
        for participant in participant_latent_times
    ]
    luminance_duration_times_std[condition] = [
        np.std(participant_duration_times[participant]) if len(participant_duration_times[participant]) > 0 else 0
        for participant in participant_duration_times
    ]

# Plot latent times and duration times as scatter plots with error bars
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Set colors for consistency with provided image
scatter_color_1 = '#ea7d8a'  # Light red for participants
scatter_color_2 = '#2a8190'  # Teal for conditions
errorbar_color = '#08bcc8'  # Dark blue for error bars

# Latent time scatter plot with error bars
fps_labels = ['No Luminance Mixture\n(60 Frames Control)', 'Luminance Mixture\n(5 Frames)', 'Luminance Mixture\n(10 Frames)', 'Luminance Mixture\n(30 Frames)']
x_positions = [10, 40, 70, 100]

# Scatter points for each participant's average latent times
ax1.scatter([x_positions[0]] * len(avg_latent_times), [t for t in avg_latent_times.values() if t != 'no'], color=scatter_color_1, alpha=0.8, label='', marker='o')
for condition, x_pos in zip(luminance_mixture_paths.keys(), x_positions[1:]):
    condition_latent_times = [t for t in luminance_latent_times[condition] if t != 'no']
    if len(condition_latent_times) > 0:
        x_offsets = np.linspace(-0.5, 0.5, len(condition_latent_times))
        for i, offset in enumerate(x_offsets):
            ax1.scatter(x_pos + offset, condition_latent_times[i], color=scatter_color_2, alpha=0.8, label='', marker='^', s=80)

# Error bars for average latent time per condition
all_latent_times_avg = [combined_latent_time] + [
    np.mean([t for t in luminance_latent_times[condition] if t != 'no']) if len([t for t in luminance_latent_times[condition] if t != 'no']) > 0 else 'no'
    for condition in luminance_mixture_paths.keys()
]
all_latent_std = [combined_latent_std] + [
    np.std([t for t in luminance_latent_times[condition] if t != 'no']) if len([t for t in luminance_latent_times[condition] if t != 'no']) > 0 else 0
    for condition in luminance_mixture_paths.keys()
]
ax1.errorbar(x_positions, all_latent_times_avg, yerr=all_latent_std, fmt='o-', color=errorbar_color, alpha=0.6, capsize=5, linewidth=2.5, label='')

ax1.set_ylabel('Vection Latency (s)')
 
ax1.set_xticks(x_positions)
ax1.set_xticklabels(fps_labels)
ax1.grid(axis='y')

# Duration time scatter plot with error bars
# Scatter points for each participant's average duration times
ax2.scatter([x_positions[0]] * len(avg_duration_times), list(avg_duration_times.values()), color=scatter_color_1, alpha=0.8, marker='o')
for condition, x_pos in zip(luminance_mixture_paths.keys(), x_positions[1:]):
    condition_duration_times = luminance_duration_times[condition]
    if len(condition_duration_times) > 0:
        x_offsets = np.linspace(-0.5, 0.5, len(condition_duration_times))
        for i, offset in enumerate(x_offsets):
            ax2.scatter(x_pos + offset, condition_duration_times[i], color=scatter_color_2, alpha=0.8, marker='^', s=80)

# Error bars for average duration time per condition
all_duration_times_avg = [combined_duration_time] + [
    np.mean(luminance_duration_times[condition]) for condition in luminance_mixture_paths.keys()
]
all_duration_std = [combined_duration_std] + [
    np.std(luminance_duration_times[condition]) if len(luminance_duration_times[condition]) > 0 else 0
    for condition in luminance_mixture_paths.keys()
]
ax2.errorbar(x_positions, all_duration_times_avg, yerr=all_duration_std, fmt='o-', color=errorbar_color, alpha=0.6, capsize=5, linewidth=2.5)

ax2.set_ylabel('Vection Duration (s)')
 
ax2.set_xticks(x_positions)
ax2.set_xticklabels(fps_labels)
ax2.grid(axis='y')

# Add thick horizontal lines at 0 and 180 seconds
ax1.axhline(0, color='black', linewidth=1)
ax1.axhline(180, color='black', linewidth=1)
ax1.set_yticks([0, 60, 120, 180])

ax2.axhline(0, color='black', linewidth=1)
ax2.axhline(180, color='black', linewidth=1)
ax2.set_yticks([0, 60, 120, 180])

# Show plot
plt.tight_layout()
plt.show()
