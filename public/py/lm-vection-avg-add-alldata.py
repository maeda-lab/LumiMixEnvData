import pandas as pd
import matplotlib.pyplot as plt

# File paths for each condition (5 fps, 10 fps, 30 fps)
luminance_mixture_paths = {
    '5 fps': [
        '/Users/jasmine/Documents/GitHub/vectionProject/public/ExperimentData/20241113_154414_luminanceMixture_cameraSpeed4_fps5_G_trialNumber1.csv',
        '/Users/jasmine/Documents/GitHub/vectionProject/public/ExperimentData/20241113_155123_luminanceMixture_cameraSpeed4_fps5_G_trialNumber2.csv',
        '/Users/jasmine/Documents/GitHub/vectionProject/public/ExperimentData/20241113_155817_luminanceMixture_cameraSpeed4_fps5_G_trialNumber3.csv',
        '/Users/jasmine/Documents/GitHub/vectionProject/public/ExperimentData/20241118_163429_luminanceMixture_cameraSpeed4_fps5_K_trialNumber1.csv',
        '/Users/jasmine/Documents/GitHub/vectionProject/public/ExperimentData/20241118_163949_luminanceMixture_cameraSpeed4_fps5_K_trialNumber2.csv',
        '/Users/jasmine/Documents/GitHub/vectionProject/public/ExperimentData/20241118_164408_luminanceMixture_cameraSpeed4_fps5_K_trialNumber3.csv',
        '/Users/jasmine/Documents/GitHub/vectionProject/public/ExperimentData/20241118_150020_luminanceMixture_cameraSpeed4_fps5_Y_trialNumber1.csv',
        '/Users/jasmine/Documents/GitHub/vectionProject/public/ExperimentData/20241118_150543_luminanceMixture_cameraSpeed4_fps5_Y_trialNumber2.csv',
        '/Users/jasmine/Documents/GitHub/vectionProject/public/ExperimentData/20241118_151108_luminanceMixture_cameraSpeed4_fps5_Y_trialNumber3.csv',
        '/Users/jasmine/Documents/GitHub/vectionProject/public/ExperimentData/20241122_171609_luminanceMixture_cameraSpeed4_fps5_A_trialNumber1.csv',
        '/Users/jasmine/Documents/GitHub/vectionProject/public/ExperimentData/20241122_172016_luminanceMixture_cameraSpeed4_fps5_A_trialNumber2.csv',
        '/Users/jasmine/Documents/GitHub/vectionProject/public/ExperimentData/20241122_172417_luminanceMixture_cameraSpeed4_fps5_A_trialNumber3.csv',

        '/Users/jasmine/Documents/GitHub/vectionProject/public/ExperimentData/20241210_164041_luminanceMixture_cameraSpeed4_fps5_b_trialNumber1.csv',
        '/Users/jasmine/Documents/GitHub/vectionProject/public/ExperimentData/20241210_171746_luminanceMixture_cameraSpeed4_fps5_b_trialNumber2.csv',
    ],
    '10 fps': [
        '/Users/jasmine/Documents/GitHub/vectionProject/public/ExperimentData/20241113_161351_luminanceMixture_cameraSpeed4_fps10_G_trialNumber1.csv',
        '/Users/jasmine/Documents/GitHub/vectionProject/public/ExperimentData/20241113_162151_luminanceMixture_cameraSpeed4_fps10_G_trialNumber2.csv',
        '/Users/jasmine/Documents/GitHub/vectionProject/public/ExperimentData/20241113_162914_luminanceMixture_cameraSpeed4_fps10_G_trialNumber3.csv',
        '/Users/jasmine/Documents/GitHub/vectionProject/public/ExperimentData/20241118_165443_luminanceMixture_cameraSpeed4_fps10_K_trialNumber1.csv',
        '/Users/jasmine/Documents/GitHub/vectionProject/public/ExperimentData/20241118_170330_luminanceMixture_cameraSpeed4_fps10_K_trialNumber2.csv',
        '/Users/jasmine/Documents/GitHub/vectionProject/public/ExperimentData/20241118_171626_luminanceMixture_cameraSpeed4_fps10_K_trialNumber3.csv',
        '/Users/jasmine/Documents/GitHub/vectionProject/public/ExperimentData/20241118_151809_luminanceMixture_cameraSpeed4_fps10_Y_trialNumber1.csv',
        '/Users/jasmine/Documents/GitHub/vectionProject/public/ExperimentData/20241118_152450_luminanceMixture_cameraSpeed4_fps10_Y_trialNumber2.csv',
        '/Users/jasmine/Documents/GitHub/vectionProject/public/ExperimentData/20241118_153346_luminanceMixture_cameraSpeed4_fps10_Y_trialNumber3.csv',
        '/Users/jasmine/Documents/GitHub/vectionProject/public/ExperimentData/20241122_173828_luminanceMixture_cameraSpeed4_fps10_A_trialNumber1.csv',
        '/Users/jasmine/Documents/GitHub/vectionProject/public/ExperimentData/20241122_174243_luminanceMixture_cameraSpeed4_fps10_A_trialNumber2.csv',
        '/Users/jasmine/Documents/GitHub/vectionProject/public/ExperimentData/20241122_174654_luminanceMixture_cameraSpeed4_fps10_A_trialNumber3.csv',

        '/Users/jasmine/Documents/GitHub/vectionProject/public/ExperimentData/20241210_170118_luminanceMixture_cameraSpeed4_fps10_b_trialNumber1.csv',
        '/Users/jasmine/Documents/GitHub/vectionProject/public/ExperimentData/20241210_174423_luminanceMixture_cameraSpeed4_fps10_b_trialNumber2.csv',
        '/Users/jasmine/Documents/GitHub/vectionProject/public/ExperimentData/20241210_180422_luminanceMixture_cameraSpeed4_fps10_b_trialNumber3.csv',

    ],
    '30 fps': [
        '/Users/jasmine/Documents/GitHub/vectionProject/public/ExperimentData/20241113_163839_luminanceMixture_cameraSpeed4_fps30_G_trialNumber1.csv',
        '/Users/jasmine/Documents/GitHub/vectionProject/public/ExperimentData/20241113_164516_luminanceMixture_cameraSpeed4_fps30_G_trialNumber2.csv',
        '/Users/jasmine/Documents/GitHub/vectionProject/public/ExperimentData/20241113_165322_luminanceMixture_cameraSpeed4_fps30_G_trialNumber3.csv',
        '/Users/jasmine/Documents/GitHub/vectionProject/public/ExperimentData/20241118_172549_luminanceMixture_cameraSpeed4_fps30_K_trialNumber1.csv',
        '/Users/jasmine/Documents/GitHub/vectionProject/public/ExperimentData/20241118_173417_luminanceMixture_cameraSpeed4_fps30_K_trialNumber2.csv',
        '/Users/jasmine/Documents/GitHub/vectionProject/public/ExperimentData/20241118_174225_luminanceMixture_cameraSpeed4_fps30_K_trialNumber3.csv',
        '/Users/jasmine/Documents/GitHub/vectionProject/public/ExperimentData/20241118_154342_luminanceMixture_cameraSpeed4_fps30_Y_trialNumber1.csv',
        '/Users/jasmine/Documents/GitHub/vectionProject/public/ExperimentData/20241118_155057_luminanceMixture_cameraSpeed4_fps30_Y_trialNumber2.csv',
        '/Users/jasmine/Documents/GitHub/vectionProject/public/ExperimentData/20241118_160156_luminanceMixture_cameraSpeed4_fps30_Y_trialNumber3.csv',
        '/Users/jasmine/Documents/GitHub/vectionProject/public/ExperimentData/20241122_175110_luminanceMixture_cameraSpeed4_fps30_A_trialNumber1.csv',
        '/Users/jasmine/Documents/GitHub/vectionProject/public/ExperimentData/20241122_180111_luminanceMixture_cameraSpeed4_fps30_A_trialNumber2.csv',
        '/Users/jasmine/Documents/GitHub/vectionProject/public/ExperimentData/20241122_180524_luminanceMixture_cameraSpeed4_fps30_A_trialNumber3.csv',

        '/Users/jasmine/Documents/GitHub/vectionProject/public/ExperimentData/20241210_165101_luminanceMixture_cameraSpeed4_fps30_b_trialNumber1.csv',
        '/Users/jasmine/Documents/GitHub/vectionProject/public/ExperimentData/20241210_180903_luminanceMixture_cameraSpeed4_fps30_b_trialNumber2.csv',
        '/Users/jasmine/Documents/GitHub/vectionProject/public/ExperimentData/20241210_182823_luminanceMixture_cameraSpeed4_fps30_b_trialNumber3.csv',

    ]
}

# Function to load CSV files and calculate the average Vection Response for each person
def calculate_avg_vection_response(file_paths):
    # Load the CSV files into DataFrames
    dfs = [pd.read_csv(file_path) for file_path in file_paths]
    
    # Extract data for each DataFrame
    times = [df['Time'] / 1000 for df in dfs]  # Convert time from ms to s
    vection_responses = [df['Vection Response'] for df in dfs]

    # Time and vection response should be aligned for averaging
    # Assuming all CSV files have the same time values
    time = times[0]  # We can use the time from the first dataset

    # Calculate the average Vection Response for each person's three trials
    avg_vection_response = sum(vection_responses) / len(vection_responses)

    return time, avg_vection_response

# Create subplots for each frequency condition
fig, axes = plt.subplots(3, 2, figsize=(12, 15), sharex=True, gridspec_kw={'hspace': 0.3})

# Plot for each frequency condition (5 fps, 10 fps, 30 fps)
for i, (fps, paths) in enumerate(luminance_mixture_paths.items()):
    # First column: Plot Frond and Back Frame Luminance for this condition
    ax1 = axes[i][0]
    # Load luminance data for the first trial (to use as an example)
    df = pd.read_csv(paths[0])
    time = df['Time'] / 1000
    frond_frame_luminance = df['FrondFrameLuminance']
    back_frame_luminance = df['BackFrameLuminance']
    
    ax1.plot(time, frond_frame_luminance, linestyle='-', color='b', label='Frond Frame Luminance', alpha=0.5)
    ax1.plot(time, back_frame_luminance, linestyle='-', color='g', label='Back Frame Luminance', alpha=0.5)
    ax1.set_ylabel('Luminance Value (0-1)')
    ax1.set_title(f'Luminance Value vs Time ({fps})')
    ax1.legend(loc='upper right')
    ax1.grid()
    ax1.set_xlim([-5, 15])
    
    # Second column: Plot Average Vection Response for this condition
    ax2 = axes[i][1]
    # Initialize an array to hold the summed vection response
    summed_vection_response = 0
    
    for j in range(0, len(paths), 3):  # Each person has 3 trials
        person_paths = paths[j:j+3]
        time, avg_vection_response = calculate_avg_vection_response(person_paths)
        summed_vection_response += avg_vection_response  # Add the average for this person
    
    # Plot the summed Vection Response
    ax2.plot(time, summed_vection_response, linestyle='-', color='r', label=f'Summed Vection Response at {fps}')
    ax2.fill_between(time, summed_vection_response, alpha=0.3, color='r')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Summed Vection Response (0-12)')
    ax2.set_title(f'Summed Vection Response vs Time ({fps})')
    ax2.set_yticks([0, 1, 2, 3, 4, 5,])
    ax2.legend(loc='upper right')
    ax2.grid()
    ax2.set_xlim([-5, 15])

# Display the plot
plt.show()
