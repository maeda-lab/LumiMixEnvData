# Masahiro Furukawa with ChatGPT
# Dec 10, 2024

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# 動画パス
video_path = "LR_Continuous - Trim.mp4"  # 動画ファイルのパス
# video_path = "LR_LuminanceMixing - Trim.mp4"  # 動画ファイルのパス

# サンプリング縮小率
resize_scale = 1.0  # 画像を50%に縮小

# 動画の読み込み
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("動画が読み込めません。パスを確認してください。")
    exit()

# 最初のフレームを取得
CAP_PROP_INITIAL_FRAME_NUMBER = 90
cap.set(cv2.CAP_PROP_POS_FRAMES, CAP_PROP_INITIAL_FRAME_NUMBER)
ret, first_frame = cap.read()
if not ret:
    print("最初のフレームが取得できません。")
    cap.release()
    exit()

first_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
first_frame_gray = cv2.resize(first_frame_gray, (0, 0), fx=resize_scale, fy=resize_scale)

# 総フレーム数の取得
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

# フーリエ変換とスペクトル計算
def calculate_spectra(image):
    f_transform = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f_transform)
    power_spectrum = 20 * np.log(np.abs(f_shift) + 1)  # パワースペクトル（+1でlog(0)を回避）
    phase_spectrum = np.angle(f_shift)  # 位相スペクトル
    return power_spectrum, phase_spectrum

# 最初のフレームに関する計算を事前に実行
power1, phase1 = calculate_spectra(first_frame_gray)

# プロットの準備
fig, axes = plt.subplots(3, 3, figsize=(15, 9))
plt.subplots_adjust(bottom=0.2)

# 初期プロットを表示
def update_plot(n_frame):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, n_frame)
    ret, current_frame = cap.read()
    cap.release()

    if not ret:
        for ax in axes.flatten():
            ax.clear()
        axes[0, 0].text(0.5, 0.5, "フレームが取得できません。", fontsize=16, ha='center', va='center')
        plt.draw()
        return

    current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    current_frame_gray = cv2.resize(current_frame_gray, (0, 0), fx=resize_scale, fy=resize_scale)

    power2, phase2 = calculate_spectra(current_frame_gray)

    power_diff = np.abs(power1 - power2)
    phase_diff = phase1 - phase2

    wrapped_phase_diff = (phase_diff + np.pi) % (2 * np.pi) - np.pi
    mask = (wrapped_phase_diff < -np.pi/2) | (wrapped_phase_diff > np.pi/2)

    # プロット更新
    axes[0, 0].imshow(first_frame_gray, cmap="gray")
    axes[0, 0].set_title(f"Frame {CAP_PROP_INITIAL_FRAME_NUMBER} (Original)")
    axes[0, 0].axis("off")

    axes[1, 0].imshow(power1, cmap="gray")
    axes[1, 0].set_title(f"Frame {CAP_PROP_INITIAL_FRAME_NUMBER} (Power Spectrum)")
    axes[1, 0].axis("off")

    axes[2, 0].imshow(phase1, cmap="gray")
    axes[2, 0].set_title(f"Frame {CAP_PROP_INITIAL_FRAME_NUMBER} (Phase Spectrum)")
    axes[2, 0].axis("off")

    axes[0, 1].imshow(current_frame_gray, cmap="gray")
    axes[0, 1].set_title(f"Frame {n_frame} (Original)")
    axes[0, 1].axis("off")

    axes[1, 1].imshow(power2, cmap="gray")
    axes[1, 1].set_title(f"Frame {n_frame} (Power Spectrum)")
    axes[1, 1].axis("off")

    axes[2, 1].imshow(phase2, cmap="gray")
    axes[2, 1].set_title(f"Frame {n_frame} (Phase Spectrum)")
    axes[2, 1].axis("off")

    axes[0, 2].axis("off")
    axes[1, 2].imshow(power_diff, cmap="gray")
    axes[1, 2].set_title("Power Spectrum Difference")
    axes[1, 2].axis("off")

    axes[2, 2].imshow(phase_diff, cmap="gray")
    axes[2, 2].imshow(mask, cmap="Reds", alpha=0.5)
    axes[2, 2].set_title("Phase Spectrum Difference ($ \\Delta \\phi > |\\pi/2|$ Highlighted)")
    axes[2, 2].axis("off")

    # 動画ファイル名を右上に表示
    axes[0, 2].text(0.5, 0.5, f"Video: {video_path}", fontsize=12, ha='center', va='center')
    axes[0, 2].set_title("Video Info")
    axes[0, 2].axis("off")

    plt.draw()

# スライダーを作成
ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03], facecolor='lightgray')
frame_slider = Slider(ax_slider, "Frame", 0, total_frames - 1, valinit=CAP_PROP_INITIAL_FRAME_NUMBER+10, valstep=1)

# スライダーの値変更イベント
def slider_update(val):
    update_plot(int(frame_slider.val))

frame_slider.on_changed(slider_update)

# 初期プロット
update_plot(CAP_PROP_INITIAL_FRAME_NUMBER+10)

# インタラクティブ表示
plt.show()
