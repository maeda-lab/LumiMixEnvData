import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file into a DataFrame 

file_path = 'D:/vectionProject/public/ExperimentData/20250117_161507_Dots_right_luminanceMixture_cameraSpeed4_fps10_C_trialNumber2.csv'  # 请替换为你的实际文件路径
# file_path = '../ExperimentData/20250117_155851_Natural_right_luminanceMixture_cameraSpeed4_fps10_C_trialNumber2.csv'  # 请替换为你的实际文件路径
# file_path = '../ExperimentData/20250117_153628_Dots_right_luminanceMixture_cameraSpeed4_fps10_C_trialNumber1.csv'  # 请替换为你的实际文件路径
df = pd.read_csv(file_path)

# Extract data from the DataFrame
time = df['Time'] / 1000  # 将时间列作为横轴 (秒)，除以1000将ms转换为s
vection_response = df['Vection Response']

# Calculate the total time when the 'Vection Response' is equal to 1
# Calculate the time differences between consecutive frames
time_diff = time.diff().fillna(0)
# Get the time differences only when Vection Response is 1
time_intervals = time_diff[vection_response == 1]

# Calculate the total duration
total_duration_1 = time_intervals.sum()

# Find the first occurrence of Vection Response equal to 1
if not vection_response.eq(1).any():
    first_occurrence_time = None
else:
    first_occurrence_index = vection_response[vection_response == 1].index[0]
    first_occurrence_time = time[first_occurrence_index]

# Calculate the time interval from 0 to the first occurrence of 1
time_to_first_1 = first_occurrence_time

# Extract luminance values based on FrondFrameNum and BackFrameNum
frond_frame_num = df['FrondFrameNum']
back_frame_num = df['BackFrameNum']
frond_frame_luminance = df['FrondFrameLuminance']
back_frame_luminance = df['BackFrameLuminance']

# Replace BackFrameNum odd values with corresponding BackFrameLuminance and FrondFrameNum even values
for i in range(len(frond_frame_num)):
    if back_frame_num[i] % 2 != 0:
        temp_back_frame_num = back_frame_num[i]
        temp_back_frame_luminance = back_frame_luminance[i]
        back_frame_num[i] = frond_frame_num[i]
        back_frame_luminance[i] = frond_frame_luminance[i]
        frond_frame_num[i] = temp_back_frame_num
        frond_frame_luminance[i] = temp_back_frame_luminance

# Create the figure and axes for plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5), sharex=True, gridspec_kw={'hspace': 0.3})

# Plot Frond Frame Luminance and Back Frame Luminance on the first subplot
ax1.plot(time, frond_frame_luminance, linestyle='-', color='b', label='Frond Frame Luminance', alpha=0.5)
ax1.plot(time, back_frame_luminance, linestyle='-', color='g', label='Back Frame Luminance', alpha=0.5)

# Plot points on the line for Frond Frame Luminance and Back Frame Luminance
ax1.scatter(time, frond_frame_luminance, color='b', s=3, alpha=0.4)
ax1.scatter(time, back_frame_luminance, color='g', s=3, alpha=0.4)

# Set labels for luminance
ax1.set_ylabel('Luminance Value (0-1)')
ax1.set_title('Luminance Value vs Time')
ax1.legend(loc='upper right')
ax1.grid()
# Limit x-axis to 10 seconds
# ax1.set_xlim([-5, 15])
 
# Plot Vection Response on the second subplot
ax2.plot(time, vection_response, linestyle='-', color='b', alpha=0.5)
ax2.fill_between(time, vection_response, alpha=0.3, color='b')
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Vection Response (0-1)')
ax2.set_title('Vection Response vs Time')
ax2.set_yticks([0, 1])
ax2.legend(loc='upper right')
ax2.grid()
# Limit x-axis to 10 seconds
# ax2.set_xlim([-5, 15])
# Set x-axis limit to include the full range of time
# ax1.set_xlim([time.min(), time.max()])

# Set x-axis ticks to evenly divide from 0 to 180 by 30
# plt.xticks(ticks=[i for i in range(0, 181, 30)])

# Annotate important points
# Annotate the initial and final time points
# ax2.text(time.min(), -0.12, f'{time.min():.2f}', color='black', fontsize=9, horizontalalignment='center')
# ax2.text(time.max(), -0.12, f'{time.max():.2f}', color='black', fontsize=9, horizontalalignment='center')

 
# Annotate the time interval from 0 to the first occurrence of 1 on the second subplot
# ax2.axvspan(0, first_occurrence_time, color='yellow', alpha=0.3, label=f'Time to First Response=1: {time_to_first_1:.2f} s')

# Annotate the first occurrence of Vection Response equal to 1
# ax2.axvline(x=first_occurrence_time, color='g', linestyle='--', label=f'First Response=1 at {first_occurrence_time:.2f} s')
# ax2.text(first_occurrence_time, -0.16, f'{first_occurrence_time:.2f} s\nVection Onset Time', color='g', fontsize=9, horizontalalignment='center')

# Annotate the total time when Vection Response is 1 on the second subplot
# ax2.text(0.95, 0.8, f'Total Time (Response=1): {total_duration_1:.2f} s',
#          horizontalalignment='right',
#          verticalalignment='top',
#          transform=plt.gca().transAxes,
#          fontsize=10,
#          bbox=dict(facecolor='white', alpha=0.5))
# Draw vertical dashed lines connecting the two subplots at specific points (e.g., 0s and 180s)
# fig.lines.extend([plt.Line2D([ax1.transData.transform((0, 0))[0] / fig.dpi / fig.get_size_inches()[0], ax1.transData.transform((0, 0))[0] / fig.dpi / fig.get_size_inches()[0]], [0.1, 0.9], transform=fig.transFigure, color='g', linestyle='--'),
                #   plt.Line2D([ax1.transData.transform((180, 0))[0] / fig.dpi / fig.get_size_inches()[0], ax1.transData.transform((180, 0))[0] / fig.dpi / fig.get_size_inches()[0]], [0.1, 0.9], transform=fig.transFigure, color='g', linestyle='--')])
# Display the plot
plt.show()

#print("total_changes: "+str(total_changes));
#print("luminance_counts: "+str(luminance_counts));