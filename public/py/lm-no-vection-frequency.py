import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# 新的数据结构：luminance_mixture_paths 按帧率分组
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

# 初始化字典来存储每个 FrondFrameLuminance 值的总计数
total_counts = {}
total_occurrences = {}

# 创建子图
fig, axs = plt.subplots(3, 1, figsize=(8, 6))  # 创建3行1列的子图

# 设置每个子图的标题和轴标签
fps_titles = ['5 fps', '10 fps', '30 fps']
colors = ['g', 'b', 'r']

# 对于每个帧率组（'5 fps', '10 fps', '30 fps'）
for i, (fps_group, file_paths) in enumerate(luminance_mixture_paths.items()):
    # 初始化每个子图的总计数和发生次数
    total_counts = {}
    total_occurrences = {}

    # 遍历每个文件路径以处理和计算每个 FrondFrameLuminance 的平均出现次数
    for file_path in file_paths:
        df = pd.read_csv(file_path)
        vection_response = df['Vection Response']
        frond_frame_luminance = df['FrondFrameLuminance']

            # Round FrondFrameLuminance to 2 decimal places for comparison
        frond_frame_luminance_rounded = frond_frame_luminance.round(2)
        #changes_1_to_0 = (vection_response.diff() == -1)  # 找到从1变为0的地方
        changes_1_to_0 = (vection_response.diff() == 1)  # 找到从0变为1的地方
        frond_luminance_changes = frond_frame_luminance_rounded[changes_1_to_0]
        
        frond_luminance_counts = Counter(frond_luminance_changes)
        
        for frond, count in frond_luminance_counts.items():
            if frond in total_counts:
                total_counts[frond] += count
                total_occurrences[frond] += 1
            else:
                total_counts[frond] = count
                total_occurrences[frond] = 1

    average_counts = {frond: total_counts[frond] / total_occurrences[frond] for frond in total_counts}
    
    # 准备绘图数据
    frond_frame_values = list(average_counts.keys())
    average_count_values = list(average_counts.values())
    
    # 在对应的子图上绘制柱状图
    axs[i].bar(frond_frame_values, average_count_values, width=0.02, label=f'{fps_group} Frame', color=colors[i])
    axs[i].set_xlabel('Frond Frame Luminance (0-1)')
    axs[i].set_ylabel('Average Number of Occurrences', fontsize=8)
    axs[i].set_title(f'Average Occurrences When Self-Motion Changes from 1 to 0 ({fps_titles[i]})')
    axs[i].grid(True)
    axs[i].set_xlim(0-0.1, 1+0.1)  # 设置 x 轴范围
    axs[i].set_ylim(0, max(average_count_values) + 1)  # 设置 y 轴范围

plt.tight_layout()  # 调整子图间的布局
plt.show()
