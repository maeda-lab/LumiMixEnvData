import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame
file_path = '/Users/jasmine/Documents/GitHub/vectionProject/ExperimentData/20241113_150930_continuous_cameraSpeed4_fps60_G_trialNumber1.csv'  # 请替换为你的实际文件路径
df = pd.read_csv(file_path)
print(df);
# Extract 'Time' and 'Vection Response' columns
time = df['Time']/1000  # 第二列作为横轴
vection_response = df['Vection Response']  # 第三列作为纵轴

# Calculate the total time when the 'Vection Response' is equal to 1
# Calculate the time differences between consecutive frames
time_diff = time.diff().fillna(0)
# Get the time differences only when Vection Response is 1
time_intervals = time_diff[vection_response == 1]

# Calculate the total duration
total_duration_1 = time_intervals.sum()

# Find the first occurrence of Vection Response equal to 1
first_occurrence_index = vection_response[vection_response == 1].index[0]
first_occurrence_time = time[first_occurrence_index]

# Calculate the time interval from 0 to the first occurrence of 1
time_to_first_1 = first_occurrence_time

# Create the figure and axes for plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw={'hspace': 0.3})

# Plot Time Comparison on the first subplot
ax1.plot(time, [1 if (t >= 0 and t <= 180) else 0 for t in time], linestyle='-', color='b', alpha=0.5, label='Time Comparison (0-180s)')
ax1.set_ylabel('Time Comparison (0 or 1)')
ax1.set_title('Time Comparison vs Time')
ax1.legend(loc='upper right')
ax1.grid()

# Plot Vection Response on the second subplot
ax2.plot(time, vection_response, linestyle='-', color='r', label='Vection Response', alpha=0.7)
ax2.fill_between(time, vection_response, alpha=0.3, color='r')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Vection Response (0-1)')
ax2.set_title('Vection Response vs Time')
ax2.set_yticks([0, 1])
ax2.legend(loc='upper right')
ax2.grid()

# Set x-axis limit to include the full range of time
ax1.set_xlim([time.min(), time.max()])

# Set x-axis ticks to evenly divide from 0 to 180 by 30
plt.xticks(ticks=[i for i in range(0, 181, 30)])

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

# Draw vertical dashed lines connecting the two subplots at specific points (e.g., 0s and 180s)
fig.lines.extend([plt.Line2D([ax1.transData.transform((0, 0))[0] / fig.dpi / fig.get_size_inches()[0], ax1.transData.transform((0, 0))[0] / fig.dpi / fig.get_size_inches()[0]], [0.1, 0.9], transform=fig.transFigure, color='g', linestyle='--'),
                  plt.Line2D([ax1.transData.transform((180, 0))[0] / fig.dpi / fig.get_size_inches()[0], ax1.transData.transform((180, 0))[0] / fig.dpi / fig.get_size_inches()[0]], [0.1, 0.9], transform=fig.transFigure, color='g', linestyle='--')])

# Display the plot
plt.show()
