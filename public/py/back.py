import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# File paths for the three CSV files
file_paths = [
    '/Users/jasmine/Documents/GitHub/vectionProject/public/ExperimentData/20241113_161351_luminanceMixture_cameraSpeed4_fps10_G_trialNumber1.csv',
]

# Initialize a dictionary to store the total counts for each FrondFrameLuminance value
total_counts = {}
total_occurrences = {}

# Loop through each file path to process and calculate average occurrences
for file_path in file_paths:
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    
    # Extract data from the DataFrame
    vection_response = df['Vection Response']
    frond_frame_luminance = df['FrondFrameLuminance']
    
    # Round FrondFrameLuminance to 2 decimal places for comparison
    frond_frame_luminance_rounded = frond_frame_luminance.round(2)
    
    # Identify the points where Vection Response changes from 1 to 0
    changes_1_to_0 = (vection_response.diff() == -1)  # Find where the value goes from 1 to 0
    
    # Extract the corresponding FrondFrameLuminance values (rounded to 2 decimal places)
    frond_luminance_changes = frond_frame_luminance_rounded[changes_1_to_0]
    
    # Count occurrences of each FrondFrameLuminance value
    frond_luminance_counts = Counter(frond_luminance_changes)
    
    # Print total counts so far
    print(f"Total counts so far (before updating): {total_counts}")
    print(f"Current file's counts: {frond_luminance_counts}")
    
    # Update the total counts and occurrences for each FrondFrameLuminance value
    for frond, count in frond_luminance_counts.items():
        if frond in total_counts:
            total_counts[frond] += count
            total_occurrences[frond] += 1
        else:
            total_counts[frond] = count
            total_occurrences[frond] = 1

# Calculate the average occurrences for each FrondFrameLuminance value
average_counts = {frond: total_counts[frond] / total_occurrences[frond] for frond in total_counts}

# Prepare the data for plotting
frond_frame_values = list(average_counts.keys())
average_count_values = list(average_counts.values())

# Create the figure for plotting
fig, ax1 = plt.subplots(figsize=(8, 6))

# Plot the average occurrences of FrondFrameLuminance as a bar chart
ax1.bar(frond_frame_values, average_count_values, width=0.02, label='Frond Frame Luminance', color='b')

# Set labels and title
ax1.set_xlabel('Frond Frame Luminance (0-1)')
ax1.set_ylabel('Average Number of Occurrences')
ax1.set_title('Average Occurrences of Frond Frame Luminance Values')

# Set grid and limits
plt.grid()
plt.xlim(0, 1)  # Ensure the x-axis is between 0 and 1 for FrondFrameLuminance
plt.ylim(0, max(average_count_values) + 1)  # Ensure y-axis accommodates all counts

# Show the plot
plt.tight_layout()
plt.show()
