import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import json


# 定义分类结构
# categories = {
#     "Dots_forward": {"fps5": [], "fps10": [], "fps20": [], "fps30": [], "fps60": []},
#     "Dots_right": {"fps5": [], "fps10": [], "fps20": [], "fps30": [], "fps60": []},
#     "Natural_right": {"fps5": [], "fps10": [], "fps20": [], "fps30": [], "fps60": []},
#     "Natural_forward": {"fps5": [], "fps10": [], "fps20": [], "fps30": [], "fps60": []},
# }

# 读取现有的A.json文件内容
with open('../data/A.json', 'r', encoding='utf-8') as file:
    simulated_data = json.load(file)

# 定义文件夹路径
folder_path = "../ExperimentData/"  # 替换为你的文件夹路径

# # 遍历文件夹中的文件
# for file_name in os.listdir(folder_path):
#     if file_name.endswith(".csv"):
#         # 根据文件名提取分类信息
#         parts = file_name.split("_")
#         if len(parts) >= 7:
#             category = parts[2] + "_" + parts[3]  # 提取 "Dots_forward", "Dots_right", 等类别
#             fps = parts[6]  # 提取 "30 fps", "60 fps", 等
            
#             # 将文件路径添加到对应类别中
#             if category in simulated_data and fps in simulated_data[category]:
#                 simulated_data[category][fps].append(folder_path+file_name)
            
# # 将分类结果保存到A.json文件中
# with open('../data/A.json', 'w', encoding='utf-8') as file:
#     json.dump(simulated_data, file, indent=4)


latent_times = {}
duration_times = {}
for condition, data in simulated_data.items():
    luminance_latent_times = {}
    luminance_duration_times = {}
    for xcondition, paths in data.items():
        participant_latent_times = {}
        participant_duration_times = {}
        for file_path in paths:
            participant_name = file_path.split('_')[-2]  # Extract participant identifier
            if participant_name not in participant_latent_times:
                participant_latent_times[participant_name] = []
                participant_duration_times[participant_name] = []

            # Load the data
            df = pd.read_csv(file_path)
            time = df['Time'] / 1000
            vection_response = df['Vection Response']

            mask = (time >= 0) & (time <= 60)

                # 应用掩码筛选数据
            filtered_vection = vection_response[mask]
            filtered_time = time[mask]
            # Calculate latent time
            if (filtered_vection == 1).any():
                first_occurrence_index = filtered_vection[filtered_vection == 1].index[0]
                latent_time = filtered_time[first_occurrence_index]
            else:
                latent_time = np.nan

            # Calculate duration time
            time_diff = filtered_time.diff().fillna(0)
            duration_time = time_diff[vection_response == 1].sum() if latent_time is not np.nan else 0

            participant_latent_times[participant_name].append(latent_time)
            participant_duration_times[participant_name].append(duration_time)
        
        # Store average values for each condition
        # luminance_latent_times[xcondition] = [
        #     np.nanmean(participant_latent_times[p]) for p in participant_latent_times
        # ]
        luminance_duration_times[xcondition] = [
            np.mean(participant_duration_times[p]) for p in participant_duration_times
        ]

    # latent_times[condition] = luminance_latent_times
    # print(luminance_duration_times)
    for xcondition in luminance_duration_times:
        luminance_duration_times[xcondition] = np.nanmean(luminance_duration_times[xcondition])
    # print(luminance_duration_times)
    duration_times[condition] = luminance_duration_times


print(duration_times)

# Prepare data for plotting
fps_values = ['fps5', 'fps10', 'fps30', 'fps60']

 
# Plot the data without error bars
plt.figure(figsize=(10, 6))
# plt.plot(fps_values, [duration_times['Dots_right'][fps] for fps in fps_values], marker='o', label='Dots Right')
# plt.plot(fps_values, [duration_times['Dots_forward'][fps] for fps in fps_values], marker='o', label='Dots Forward')
plt.plot(fps_values, [duration_times['Natural_right'][fps] for fps in fps_values], marker='o', label='Natural Right')
# plt.plot(fps_values, [duration_times['Natural_forward'][fps] for fps in fps_values], marker='o', label='Natural Forward')

plt.title('Vection Duration Time by FPS')
plt.ylabel('Vection Duration Time (s)')
plt.xlabel('')

custom_labels = ['LM(5 FPS)', 'LM(10 FPS)', 'LM(30 FPS)', 'No Luminance Mixture\n(60 FPS)']  # Custom labels for x-axis
plt.xticks(ticks=range(len(fps_values)), labels=custom_labels)  # Set custom x-axis tick labels
plt.legend()
plt.grid(True)
plt.show()
