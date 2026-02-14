import pandas as pd
import matplotlib.pyplot as plt

# Load data from three CSV files (replace 'file_path_X.csv' with your actual file paths)
file_paths = ['/Users/jasmine/Documents/GitHub/vectionProject/ExperimentData/20241113_154414_luminanceMixture_cameraSpeed4_fps5_G_trialNumber1.csv',
              '/Users/jasmine/Documents/GitHub/vectionProject/ExperimentData/20241113_155123_luminanceMixture_cameraSpeed4_fps5_G_trialNumber2.csv',
              '/Users/jasmine/Documents/GitHub/vectionProject/ExperimentData/20241113_155817_luminanceMixture_cameraSpeed4_fps5_G_trialNumber3.csv']
first_occurrence_times = []
total_durations_1 = []

# Loop through each file and calculate first occurrence time and total duration when Vection Response is 1
for file_path in file_paths:
    df = pd.read_csv(file_path)

    # Extract 'Time' and 'Vection Response' columns
    time = df['Time']/1000  # Second column as Time
    vection_response = df['Vection Response']  # Third column as Vection Response

    # Calculate the first occurrence of Vection Response equal to 1
    first_occurrence_index = vection_response[vection_response == 1].index[0]
    first_occurrence_time = time[first_occurrence_index]
    first_occurrence_times.append(first_occurrence_time)

    # Calculate the total duration when Vection Response is 1
    time_diff = time.diff().fillna(0)
    total_duration_1 = time_diff[vection_response == 1].sum()
    total_durations_1.append(total_duration_1)

# Plotting the bar chart for first occurrence times and total durations
x_labels = ['Experiment 1', 'Experiment 2', 'Experiment 3']
x = range(len(x_labels))

# Create figure and axis
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plotting first occurrence times as bars
bar_width = 0.35
ax1.bar(x, first_occurrence_times, bar_width, label='First Occurrence Time (ms)', color='b', alpha=0.6)

# Create a second y-axis for total duration bars
ax2 = ax1.twinx()
ax2.bar([i + bar_width for i in x], total_durations_1, bar_width, label='Total Duration (Response=1) (ms)', color='g', alpha=0.6)

# Adding labels, title, and legend
ax1.set_xlabel('Experiments')
ax1.set_ylabel('First Occurrence Time (s)', color='b')
ax2.set_ylabel('Total Duration (Response=1) (s)', color='g')
plt.title('First Occurrence Time and Total Duration of Vection Response=1')

ax1.set_xticks([i + bar_width / 2 for i in x])
ax1.set_xticklabels(x_labels)

# Adding legends for both y-axes
fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.95))

# Display the plot
plt.grid()
plt.show()


