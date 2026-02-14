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

# Function to calculate reoccurrence times for each participant
def calculate_reoccurrence_times(file_paths):
    participant_reoccurrence_times = {}

    for file_path in file_paths:
        # Extract participant identifier from file name (e.g., "_G_")
        participant_name = file_path.split('_')[5]

        # Load the data
        df = pd.read_csv(file_path)
        time = df['Time'] / 1000  # Convert time to seconds
        vection_response = df['Vection Response']

        # Identify where Vection stops (1 → 0) and reoccurs (0 → 1)
        stop_indexes = vection_response[(vection_response.shift(1) == 1) & (vection_response == 0)].index
        reoccur_indexes = vection_response[(vection_response.shift(1) == 0) & (vection_response == 1)].index

        # Calculate the time it takes for Vection to stop and reoccur
        reoccurrence_times = []
        for stop_index in stop_indexes:
            reoccur_index = reoccur_indexes[reoccur_indexes > stop_index].min()  # Find the next occurrence after stop
            if not np.isnan(reoccur_index):
                reoccurrence_time = time[reoccur_index] - time[stop_index]
                reoccurrence_times.append(reoccurrence_time)

        # Store reoccurrence times by participant
        if participant_name not in participant_reoccurrence_times:
            participant_reoccurrence_times[participant_name] = []
        participant_reoccurrence_times[participant_name].extend(reoccurrence_times)

    # Calculate average reoccurrence time for each participant
    avg_reoccurrence_times = {
        participant: np.mean(times) if len(times) > 0 else np.nan
        for participant, times in participant_reoccurrence_times.items()
    }

    return avg_reoccurrence_times

# Calculate reoccurrence times for continuous and luminance mixture conditions
continuous_avg_reoccurrence_times = calculate_reoccurrence_times(file_paths)
luminance_avg_reoccurrence_times = {
    condition: calculate_reoccurrence_times(paths)
    for condition, paths in luminance_mixture_paths.items()
}

# Combine results into a dictionary for easier plotting
combined_avg_reoccurrence_times = {'Continuous': continuous_avg_reoccurrence_times}
combined_avg_reoccurrence_times.update(luminance_avg_reoccurrence_times)

# Prepare data for plotting
conditions = list(combined_avg_reoccurrence_times.keys())
participants = set()
for condition in combined_avg_reoccurrence_times.values():
    participants.update(condition.keys())
participants = sorted(participants)

# Create a data structure to store the average times for each participant across conditions
participant_data = {participant: [np.nan] * len(conditions) for participant in participants}

for i, condition in enumerate(conditions):
    for participant, avg_time in combined_avg_reoccurrence_times[condition].items():
        if participant in participant_data:
            participant_data[participant][i] = avg_time

# Plotting the results
fig, ax = plt.subplots(figsize=(12, 8))

x_positions = range(len(conditions))
for participant, avg_times in participant_data.items():
    # Filter out NaN values for plotting (Matplotlib does not handle NaNs in plot)
    valid_x = [x for x, y in zip(x_positions, avg_times) if not np.isnan(y)]
    valid_y = [y for y in avg_times if not np.isnan(y)]
    
    ax.plot(valid_x, valid_y, marker='o', label=f'Participant {participant}', alpha=0.7)

# Adding labels, title, and legend
ax.set_title('Average Time for Vection to Stop and Reoccur by Participant and Condition')
ax.set_xticks(x_positions)
ax.set_xticklabels(conditions)
ax.set_xlabel('Condition')
ax.set_ylabel('Average Reoccurrence Time (seconds)')
ax.legend(title="Participants", bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True)

plt.tight_layout()
plt.show()

# Save results to CSV
results_data = {
    'Participant': [],
    'Condition': [],
    'Average Reoccurrence Time (s)': []
}
for condition, participant_times in combined_avg_reoccurrence_times.items():
    for participant, avg_time in participant_times.items():
        results_data['Participant'].append(participant)
        results_data['Condition'].append(condition)
        results_data['Average Reoccurrence Time (s)'].append(avg_time)

results_df = pd.DataFrame(results_data)
results_df.to_csv('Average_Reoccurrence_Time_by_Participant_and_Condition.csv', index=False)
print("Results saved to 'Average_Reoccurrence_Time_by_Participant_and_Condition.csv'.")
