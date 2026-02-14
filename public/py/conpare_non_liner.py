#
# Velocity Pulsation Experiment 
# 
# Coded by Masahiro Furukawa, Oct 25, 2024
# Concepted by Taro Maeda, Aug 18, 2024


from psychopy import visual, core, event, clock
import numpy as np
import csv
import matplotlib.pyplot as plt
from datetime import datetime

# 実行時刻を取得し、ファイル名用にフォーマット
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
data_filename = f"../experimental_results/experiment_data_{current_time}.csv"
key_pos_filename = f"../experimental_results/experiment_data_key_pos_{current_time}.csv"
key_neg_filename = f"../experimental_results/experiment_data_key_neg_{current_time}.csv"
result_image_filename = f"../experimental_results/experiment_results_{current_time}.png"

# ウィンドウの設定（全画面表示に変更）
win = visual.Window(color=[0.5, 0.5, 0.5], units='pix', fullscr=True)

# ウィンドウのサイズを取得
win_width, win_height = win.size

# 画像のロード（仮のパターン画像としてsin波を生成）
spatial_freq = 3  # 空間周波数の変更
x = np.linspace(-np.pi, np.pi, 256)
grating_texture = np.sin(x * spatial_freq)
texture_top     = np.tile(grating_texture, (256, 1))
texture_bottom1 = np.tile(np.sin((x - np.pi/4) * spatial_freq), (256, 1))
texture_bottom2 = np.tile(np.sin((x + np.pi/4) * spatial_freq), (256, 1))

# A light green text
focus_point_image = visual.Circle(win,radius=10, edges=5, pos=(0, 0), fillColor=(1, 0, 0), colorSpace='rgb', opacity=1.0, autoDraw=True)

# 画像刺激の設定
top_image = visual.GratingStim(win, tex=texture_top, size=[win_height // 2, win_height // 2], pos=(-win_width // 4, 0), mask='gauss', opacity=1.0)
bottom_image1 = visual.GratingStim(win, tex=texture_bottom1, size=[win_height // 2, win_height // 2], pos=(win_width // 4, 0), mask='gauss', opacity=0.5)
bottom_image2 = visual.GratingStim(win, tex=texture_bottom2, size=[win_height // 2, win_height // 2], pos=(win_width // 4, 0), mask='gauss', opacity=0.5)

# パラメータ設定
speed_top = 0.2  # 上の画像の一定速度の変更
speed_amp = 0.2  # 下の画像の三角波の速度振幅の変更
triangle_freq = 0.2  # 三角波の周波数（Hz）の変更
mix_amp = 0.25  # 三角波の振幅の変更
key_store = {}
key_store['neg'] = []
key_store['pos'] = []

# 記録用のリスト
time_points = []
tri_speed_values = []
blend_ratio_values = []

# キー操作の定義
def update_mix_amplitude(keys, t):
    global mix_amp
    global key_store
    if 'w' in keys:
        mix_amp = min(1, mix_amp + 0.05)
        key_store['pos'].append(t)
    if 's' in keys:
        mix_amp = max(0, mix_amp - 0.05)
        key_store['neg'].append(t)

# 実験用の時間管理
global_clock = core.Clock()
trial_clock = clock.CountdownTimer(20)  # 20秒間の試行

# 実験ループ
while trial_clock.getTime() > 0:
    t = global_clock.getTime()

    # 上の画像（甲）の移動設定
    top_image.phase += speed_top / 60.0

    # 下の画像（乙）の速度を三角波に基づいて設定
    tri_speed = speed_amp * (np.sin(2 * np.pi * triangle_freq * t)) + 0.5
    blend_ratio = mix_amp * (np.cos(2 * np.pi * triangle_freq * t) + 1) * 0.5

    # データの記録
    time_points.append(t)
    tri_speed_values.append(tri_speed)
    blend_ratio_values.append(blend_ratio)

    # 乙の合成と移動
    bottom_image1.phase += tri_speed / 60.0
    bottom_image2.phase += tri_speed / 60.0
    bottom_image1.opacity = 1 - blend_ratio
    bottom_image2.opacity = blend_ratio

    # 画面更新
    top_image.draw()
    bottom_image1.draw()
    bottom_image2.draw()
    focus_point_image.draw()
    win.flip()

    # キー入力処理
    keys = event.getKeys()
    if 'escape' in keys:
        break
    update_mix_amplitude(keys, t)

ddt_blend_ratio_values = np.gradient(blend_ratio_values) * 20.0
sum_speed_ddt = [speed + ddt for speed, ddt in zip(tri_speed_values, ddt_blend_ratio_values)]

# CSVファイルにデータを保存
with open(data_filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Time (s)', 'Triangular Speed', 'Blending Ratio', 'd/dt Blending Ratio'])
    for t, speed, ratio, ddt in zip(time_points, tri_speed_values, blend_ratio_values, ddt_blend_ratio_values):
        writer.writerow([t, speed, ratio, ddt])
print(f"データが '{data_filename}' に保存されました。")

with open(key_pos_filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Time (s)'])
    for t in key_store['pos']:
        writer.writerow([t])
print(f"データが '{key_pos_filename}' に保存されました。")

with open(key_neg_filename, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Time (s)'])
    for t in key_store['neg']:
        writer.writerow([t])
print(f"データが '{key_neg_filename}' に保存されました。")

# 終了処理
win.close()

# グラフの表示
plt.figure(figsize=(12, 8))

# 三角波速度のグラフ
plt.subplot(4, 1, 1)
plt.plot(time_points, tri_speed_values, label='Pulsation Speed', color='b')
plt.xlabel('Time (s)')
plt.ylabel('Speed')
plt.title('Pulsation Speed - Over Time')
plt.grid(True)
plt.ylim(0, 1)
plt.legend()

# d/dt 合成比率のグラフ
plt.subplot(4, 1, 2)
plt.plot(time_points, ddt_blend_ratio_values, label='d/dt Blending Ratio', color='r')
plt.xlabel('Time (s)')
plt.ylabel('Speed')
plt.title('d/dt Blending Ratio - Over Time')
plt.grid(True)
plt.legend()

# 合成比率のグラフ
plt.subplot(4, 1, 3)
plt.plot(time_points, blend_ratio_values, label='Blending Ratio', color='gray')
plt.plot(key_store['pos'], 1.1 * np.ones(len(key_store['pos'])), 'ro', label='key+')
plt.plot(key_store['neg'], -0.1 * np.ones(len(key_store['neg'])), 'bo', label='key-')
plt.xlabel('Time (s)')
plt.ylabel('Blending Ratio')
plt.title('Blending Ratio - Over Time')
plt.grid(True)
plt.ylim(-0.2, 1.2)
plt.legend(ncol=3)

# 速度 + d/dt 合成比率のグラフ
plt.subplot(4, 1, 4)
plt.plot(time_points, sum_speed_ddt, label='Sum of Pulsation Speed and d/dt Blending Ratio', color='g')
plt.xlabel('Time (s)')
plt.ylabel('Value')
plt.title('Sum of Pulsation Speed and d/dt Blending Ratio - Over Time')
plt.grid(True)
plt.legend()

# グラフの表示
plt.tight_layout()
plt.savefig(result_image_filename)
print(f"グラフが '{result_image_filename}' として保存されました。")

plt.show(block=True)  # ここでキー入力を待つ

core.quit()
