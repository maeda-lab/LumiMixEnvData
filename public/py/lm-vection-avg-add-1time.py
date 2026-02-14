import pandas as pd
import matplotlib.pyplot as plt

# File paths for the four individuals' data (three trials each)
file_paths = [
    # Data for person 1 (G)
    '/Users/jasmine/Documents/GitHub/vectionProject/public/ExperimentData/20241113_161351_luminanceMixture_cameraSpeed4_fps10_G_trialNumber1.csv',
    '/Users/jasmine/Documents/GitHub/vectionProject/public/ExperimentData/20241113_162151_luminanceMixture_cameraSpeed4_fps10_G_trialNumber2.csv',
    '/Users/jasmine/Documents/GitHub/vectionProject/public/ExperimentData/20241113_162914_luminanceMixture_cameraSpeed4_fps10_G_trialNumber3.csv',

    # Data for person 2 (K)
    '/Users/jasmine/Documents/GitHub/vectionProject/public/ExperimentData/20241118_165443_luminanceMixture_cameraSpeed4_fps10_K_trialNumber1.csv',
    '/Users/jasmine/Documents/GitHub/vectionProject/public/ExperimentData/20241118_170330_luminanceMixture_cameraSpeed4_fps10_K_trialNumber2.csv',
    '/Users/jasmine/Documents/GitHub/vectionProject/public/ExperimentData/20241118_171626_luminanceMixture_cameraSpeed4_fps10_K_trialNumber3.csv',

    # Data for person 3 (Y)
    '/Users/jasmine/Documents/GitHub/vectionProject/public/ExperimentData/20241118_151809_luminanceMixture_cameraSpeed4_fps10_Y_trialNumber1.csv',
    '/Users/jasmine/Documents/GitHub/vectionProject/public/ExperimentData/20241118_152450_luminanceMixture_cameraSpeed4_fps10_Y_trialNumber2.csv',
    '/Users/jasmine/Documents/GitHub/vectionProject/public/ExperimentData/20241118_153346_luminanceMixture_cameraSpeed4_fps10_Y_trialNumber3.csv',

    # Data for person 4 (A)
    '/Users/jasmine/Documents/GitHub/vectionProject/public/ExperimentData/20241122_173828_luminanceMixture_cameraSpeed4_fps10_A_trialNumber1.csv',
    '/Users/jasmine/Documents/GitHub/vectionProject/public/ExperimentData/20241122_174243_luminanceMixture_cameraSpeed4_fps10_A_trialNumber2.csv',
    '/Users/jasmine/Documents/GitHub/vectionProject/public/ExperimentData/20241122_174654_luminanceMixture_cameraSpeed4_fps10_A_trialNumber3.csv'
]

# Load the CSV files into DataFrames
dfs = [pd.read_csv(file_path) for file_path in file_paths]

# Extract data for each DataFrame
times = [df['Time'] / 1000 for df in dfs]  # Convert time from ms to s
vection_responses = [df['Vection Response'] for df in dfs]
frond_frame_luminances = [df['FrondFrameLuminance'] for df in dfs]
back_frame_luminances = [df['BackFrameLuminance'] for df in dfs]

# Time and vection response should be aligned for averaging
# Assuming all CSV files have the same time values
time = times[0]  # We can use the time from the first dataset

# Function to calculate average Vection Response for each person
def calculate_average_vection_response(start_index, end_index):
    avg_vection_response = sum(vection_responses[start_index:end_index]) / (end_index - start_index)
    return avg_vection_response

# Calculate average Vection Response for each person
avg_vection_response_G = calculate_average_vection_response(0, 3)
avg_vection_response_K = calculate_average_vection_response(3, 6)
avg_vection_response_Y = calculate_average_vection_response(6, 9)
avg_vection_response_A = calculate_average_vection_response(9, 12)

# Sum the average Vection Responses of all individuals
summed_vection_response = avg_vection_response_G + avg_vection_response_K + avg_vection_response_Y + avg_vection_response_A

# Calculate the total time when the summed 'Vection Response' is equal to 1
time_diff = time.diff().fillna(0)
time_intervals = time_diff[summed_vection_response == 1]
total_duration_1 = time_intervals.sum()

# Find the first occurrence of summed 'Vection Response' equal to 1
first_occurrence_index = summed_vection_response[summed_vection_response == 1].index[0]
first_occurrence_time = time[first_occurrence_index]

# Calculate the time interval from 0 to the first occurrence of 1
time_to_first_1 = first_occurrence_time

# Create the figure and axes for plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True, gridspec_kw={'hspace': 0.3})

# Plot Frond Frame Luminance and Back Frame Luminance for the first dataset on the first subplot
ax1.plot(time, frond_frame_luminances[0], linestyle='-', color='b', label='Frond Frame Luminance', alpha=0.5)
ax1.plot(time, back_frame_luminances[0], linestyle='-', color='g', label='Back Frame Luminance', alpha=0.5)

# Plot points on the line for Frond Frame Luminance and Back Frame Luminance
ax1.scatter(time, frond_frame_luminances[0], color='b', s=3, alpha=0.4)
ax1.scatter(time, back_frame_luminances[0], color='g', s=3, alpha=0.4)

# Set labels for luminance
ax1.set_ylabel('Luminance Value (0-1)')
ax1.set_title('Luminance Value vs Time')
ax1.legend(loc='upper right')
ax1.grid()
# Limit x-axis to 10 seconds
ax1.set_xlim([-5, 15])

# Plot the summed Vection Response on the second subplot
ax2.plot(time, summed_vection_response, linestyle='-', color='r', label='Summed Vection Response', alpha=0.7)
ax2.fill_between(time, summed_vection_response, alpha=0.3, color='r')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Summed Vection Response (0-12)')
ax2.set_title('Summed Vection Response vs Time')
ax2.set_yticks([0, 1, 2, 3, 4, ])
ax2.legend(loc='upper right')
ax2.grid()
# Limit x-axis to 10 seconds
ax2.set_xlim([-5, 15])

# Annotate important points
# Annotate the initial and final time points
ax2.text(time.min(), -0.12, f'{time.min():.2f}', color='black', fontsize=9, horizontalalignment='center')
ax2.text(time.max(), -0.12, f'{time.max():.2f}', color='black', fontsize=9, horizontalalignment='center')

# Annotate the time interval from 0 to the first occurrence of 1 on the second subplot
ax2.axvspan(0, first_occurrence_time, color='yellow', alpha=0.3, label=f'Time to First Response=1: {time_to_first_1:.2f} s')

# Annotate the first occurrence of Vection Response equal to 1
ax2.axvline(x=first_occurrence_time, color='g', linestyle='--', label=f'First Response=1 at {first_occurrence_time:.2f} s')
ax2.text(first_occurrence_time, -0.16, f'{first_occurrence_time:.2f} s\nVection Onset Time', color='g', fontsize=9, horizontalalignment='center')

# Annotate the total time when Vection Response is 1 on the second subplot
ax2.text(0.95, 0.8, f'Total Time (Response=1): {total_duration_1:.2f} s',
         horizontalalignment='right',
         verticalalignment='top',
         transform=plt.gca().transAxes,
         fontsize=10,
         bbox=dict(facecolor='white', alpha=0.5))

# Display the plot
plt.show()
