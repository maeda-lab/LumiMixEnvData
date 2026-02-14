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

# Initialize dictionaries for storing latent and duration times
participant_latent_times = {}
participant_duration_times = {}

# Process continuous vection files
for file_path in file_paths:
    participant_name = file_path.split('_')[5]  # Extract participant identifier
    if participant_name not in participant_latent_times:
        participant_latent_times[participant_name] = []
        participant_duration_times[participant_name] = []

    # Load the data
    df = pd.read_csv(file_path)
    time = df['Time'] / 1000  # Convert time to seconds
    vection_response = df['Vection Response']

    # Calculate latent time
    if (vection_response == 1).any():
        first_occurrence_index = vection_response[vection_response == 1].index[0]
        latent_time = time[first_occurrence_index]
    else:
        latent_time = np.nan  # Mark as NaN if no vection occurs

    # Calculate duration time
    time_diff = time.diff().fillna(0)
    duration_time = time_diff[vection_response == 1].sum() if latent_time is not np.nan else 0

    # Store the data
    participant_latent_times[participant_name].append(latent_time)
    participant_duration_times[participant_name].append(duration_time)

# Compute average and standard deviation
avg_latent_times = {p: np.nanmean(participant_latent_times[p]) for p in participant_latent_times}
avg_duration_times = {p: np.mean(participant_duration_times[p]) for p in participant_duration_times}
std_latent_times = {p: np.nanstd(participant_latent_times[p]) for p in participant_latent_times}
std_duration_times = {p: np.std(participant_duration_times[p]) for p in participant_duration_times}

# Process luminance mixture files
luminance_latent_times = {}
luminance_duration_times = {}

for condition, paths in luminance_mixture_paths.items():
    participant_latent_times = {}
    participant_duration_times = {}
    
    for file_path in paths:
        participant_name = file_path.split('_')[5]  # Extract participant identifier
        if participant_name not in participant_latent_times:
            participant_latent_times[participant_name] = []
            participant_duration_times[participant_name] = []

        # Load the data
        df = pd.read_csv(file_path)
        time = df['Time'] / 1000
        vection_response = df['Vection Response']

        # Calculate latent time
        if (vection_response == 1).any():
            first_occurrence_index = vection_response[vection_response == 1].index[0]
            latent_time = time[first_occurrence_index]
        else:
            latent_time = np.nan

        # Calculate duration time
        time_diff = time.diff().fillna(0)
        duration_time = time_diff[vection_response == 1].sum() if latent_time is not np.nan else 0

        participant_latent_times[participant_name].append(latent_time)
        participant_duration_times[participant_name].append(duration_time)
    
    # Store average values for each condition
    luminance_latent_times[condition] = [
        np.nanmean(participant_latent_times[p]) for p in participant_latent_times
    ]
    luminance_duration_times[condition] = [
        np.mean(participant_duration_times[p]) for p in participant_duration_times
    ]

# Plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

x_positions = [10, 40, 70, 100]
fps_labels = ['No Luminance Mixture\n(60 fps)', '5 fps', '10 fps', '30 fps']

# Scatter plot for latent times
for i, (condition, x_pos) in enumerate(zip(['No Luminance Mixture'] + list(luminance_mixture_paths.keys()), x_positions)):
    if i == 0:  # Control condition
        ax1.scatter([x_pos] * len(avg_latent_times), list(avg_latent_times.values()), color='blue', alpha=0.7, label='Participants')
    else:
        ax1.scatter([x_pos] * len(luminance_latent_times[condition]), luminance_latent_times[condition], color='orange', alpha=0.7)

ax1.errorbar(x_positions, 
             [np.nanmean(list(avg_latent_times.values()))] + [np.nanmean(luminance_latent_times[c]) for c in luminance_mixture_paths.keys()],
             yerr=[np.nanstd(list(avg_latent_times.values()))] + [np.nanstd(luminance_latent_times[c]) for c in luminance_mixture_paths.keys()],
             fmt='o-', color='red', capsize=5)

ax1.set_title('Vection Latent Times')
ax1.set_ylabel('Time (s)')
ax1.set_xticks(x_positions)
ax1.set_xticklabels(fps_labels)
ax1.grid()

# Scatter plot for duration times
for i, (condition, x_pos) in enumerate(zip(['No Luminance Mixture'] + list(luminance_mixture_paths.keys()), x_positions)):
    if i == 0:  # Control condition
        ax2.scatter([x_pos] * len(avg_duration_times), list(avg_duration_times.values()), color='blue', alpha=0.7, label='Participants')
    else:
        ax2.scatter([x_pos] * len(luminance_duration_times[condition]), luminance_duration_times[condition], color='orange', alpha=0.7)

ax2.errorbar(x_positions, 
             [np.mean(list(avg_duration_times.values()))] + [np.mean(luminance_duration_times[c]) for c in luminance_mixture_paths.keys()],
             yerr=[np.std(list(avg_duration_times.values()))] + [np.std(luminance_duration_times[c]) for c in luminance_mixture_paths.keys()],
             fmt='o-', color='red', capsize=5)

ax2.set_title('Vection Duration Times')
ax2.set_ylabel('Time (s)')
ax2.set_xticks(x_positions)
ax2.set_xticklabels(fps_labels)
ax2.grid()

plt.tight_layout()
plt.show()
