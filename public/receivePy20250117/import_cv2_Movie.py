# Masahiro Furukawa with ChatGPT
# Dec 10, 2024

# pip install opencv-python

import cv2
import numpy as np
import matplotlib.pyplot as plt

# # 画像の読み込み（グレースケールで）
# image_path1 = "fig1.png"  # 1つ目の画像パスを指定
# image_path2 = "fig2.png"  # 2つ目の画像パスを指定

# image1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
# image2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)




# 動画パス
video_path = "LR_Continuous - Trim.mp4"  # 動画ファイルのパス
n_frame = 3  # 抽出するフレーム間隔

# 動画の読み込み
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("動画が読み込めません。パスを確認してください。")
else:
    # 最初のフレームを取得
    ret, frame1 = cap.read()
    if not ret:
        print("最初のフレームが取得できません。")
        cap.release()
        exit()
    image1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    # nフレーム目を取得
    cap.set(cv2.CAP_PROP_POS_FRAMES, n_frame)
    ret, frame2 = cap.read()
    if not ret:
        print(f"{n_frame} フレーム目が取得できません。")
        cap.release()
        exit()
    image2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # 動画リソースを解放
    cap.release()



SHOW_AFFINE_MATRIX = False

if image1 is None or image2 is None:
    print("画像が読み込めません。パスを確認してください。")
else:

    # 同一画像を 0 ピクセルシフト
    if(False):
        rows, cols = image1.shape
        # M = np.float32([[1, 0,  0], [0, 1, 0]])  # 平行移動行列（0ピクセル）
        # M = np.float32([[1, 0,  1], [0, 1, 0]])  # 平行移動行列（右に1ピクセル）
        # M = np.float32([[1, 0, -1], [0, 1, 0]])  # 平行移動行列（左に1ピクセル）
        # M = np.float32([[1, 0,  0], [0, 1, 1]])  # 平行移動行列（下に1ピクセル）
        # M = np.float32([[1, 0,  0], [0, 1, 2]])  # 平行移動行列（下に2ピクセル）
        M = np.float32([[1, 0,  1], [0, 1, 1]])  # 平行移動行列（右に1ピクセル、下に1ピクセル）
        image2 = cv2.warpAffine(image1, M, (cols, rows))
        SHOW_AFFINE_MATRIX = True

    # 2画像の比較
    if(True):
        # リサイズして同じ形状に統一
        target_shape = (min(image1.shape[1], image2.shape[1]), min(image1.shape[0], image2.shape[0]))
        image1 = cv2.resize(image1, target_shape)
        image2 = cv2.resize(image2, target_shape)

    # フーリエ変換とスペクトル計算
    def calculate_spectra(image):
        f_transform = np.fft.fft2(image)
        f_shift = np.fft.fftshift(f_transform)
        power_spectrum = 20 * np.log(np.abs(f_shift) + 1)  # パワースペクトル（+1でlog(0)を回避）
        phase_spectrum = np.angle(f_shift)  # 位相スペクトル
        return power_spectrum, phase_spectrum

    power1, phase1 = calculate_spectra(image1)
    power2, phase2 = calculate_spectra(image2)

    # 差分計算
    power_diff = np.abs(power1 - power2)
    phase_diff = phase1 - phase2  # 位相差分

    # 赤く塗るマスク作成
    # 位相差分を -π ～ π にラップ
    wrapped_phase_diff = (phase_diff + np.pi) % (2 * np.pi) - np.pi

    # -π/2 ～ π/2 の範囲外をマスク
    mask = (wrapped_phase_diff < -np.pi/2) | (wrapped_phase_diff > np.pi/2)
    highlighted_phase_diff = np.copy(phase_diff)
    highlighted_phase_diff = np.ma.masked_where(~mask, highlighted_phase_diff)  # 範囲内を隠す

    # プロット
    fig, axes = plt.subplots(3, 3, figsize=(15, 9))

    # 1つ目の画像
    axes[0, 0].imshow(image1, cmap="gray")
    axes[0, 0].set_title("Image 1 (Original)")
    axes[0, 0].axis("off")

    axes[1, 0].imshow(power1, cmap="gray")
    axes[1, 0].set_title("Image 1 (Power Spectrum)")
    axes[1, 0].axis("off")

    axes[2, 0].imshow(phase1, cmap="gray")
    axes[2, 0].set_title("Image 1 (Phase Spectrum)")
    axes[2, 0].axis("off")

    # 2つ目の画像
    axes[0, 1].imshow(image2, cmap="gray")
    axes[0, 1].set_title("Image 2 (Original)")
    axes[0, 1].axis("off")

    axes[1, 1].imshow(power2, cmap="gray")
    axes[1, 1].set_title("Image 2 (Power Spectrum)")
    axes[1, 1].axis("off")

    axes[2, 1].imshow(phase2, cmap="gray")
    axes[2, 1].set_title("Image 2 (Phase Spectrum)")
    axes[2, 1].axis("off")

    # 差分
    axes[0, 2].axis("off")  # 差分の上段は空白
    axes[1, 2].imshow(power_diff, cmap="gray")
    axes[1, 2].set_title("Power Spectrum Difference")
    axes[1, 2].axis("off")

    # 位相差分を表示（赤い部分を範囲外としてマスク）
    axes[2, 2].imshow(phase_diff, cmap="gray")
    axes[2, 2].imshow(mask, cmap="Reds", alpha=0.5)  # 範囲外を赤く塗る
    axes[2, 2].set_title("Phase Spectrum Difference ($ \Delta \phi > |\pi/2|$ Highlighted)")
    axes[2, 2].axis("off")

    if(SHOW_AFFINE_MATRIX):
        print("M = {:.1f} {:.1f} ".format(M[0,2], M[1,2]))
        # M行列をLaTeXフォーマットで表示
        # M_latex = r"$\mathbf{M} = \begin{bmatrix} 1 & 0 & {:.1f} \\ 0 & 1 & {:.1f} \end{bmatrix}$".format(M[0,2], M[1,2])
        M_latex = "Affine Matrix M =  1  0  {:.0f} \n 0  1  {:.0f} ".format(M[0,2], M[1,2])
        axes[0, 2].text(0.8, 0.5, M_latex, fontsize=16, ha='right', va='center')
        axes[0, 2].set_title("Affine Matrix (M)")
        axes[0, 2].axis("off")
    
    # レイアウト調整
    plt.tight_layout()
    plt.show()
