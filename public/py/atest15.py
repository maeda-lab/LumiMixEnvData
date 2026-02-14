import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

MP4 = Path(r"D:\vectionProject\public\freq_test_videos\orig_lin.mp4")

FPS = 60
SECONDS_PER_STEP = 1.0
N = int(round(FPS * SECONDS_PER_STEP))

MAX_STEPS = 120

# 建议盯树就框树干附近，效果会明显很多
ROI = None  # (x0,y0,w,h) 例如 (1100, 220, 900, 420)

USE_DOG = True
DOG_SMALL = 2.0
DOG_LARGE = 8.0

def preprocess(gray_u8):
    g = gray_u8.astype(np.float32) / 255.0
    if ROI is not None:
        x0,y0,w,h = ROI
        g = g[y0:y0+h, x0:x0+w]
    g = g - float(g.mean())  # 去 DC，避免亮度呼吸影响
    if USE_DOG:
        lo1 = cv2.GaussianBlur(g, (0,0), DOG_SMALL, borderType=cv2.BORDER_REFLECT)
        lo2 = cv2.GaussianBlur(g, (0,0), DOG_LARGE, borderType=cv2.BORDER_REFLECT)
        g = lo1 - lo2
    # 归一化一下让光流更稳（可选但建议）
    s = float(g.std()) + 1e-6
    g = np.clip(g / (3*s), -1.0, 1.0)
    g = (g + 1.0) * 0.5  # 映射到 0..1
    return (g * 255.0 + 0.5).astype(np.uint8)

cap = cv2.VideoCapture(str(MP4))
if not cap.isOpened():
    raise RuntimeError("cannot open video")

frames = []
while True:
    ok, fr = cap.read()
    if not ok:
        break
    gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    frames.append(preprocess(gray))
cap.release()

T = len(frames)
steps = min(T // N, MAX_STEPS)
if steps <= 0:
    raise RuntimeError("not enough frames for one step")

# 计算每个 step 内每一帧到下一帧的光流幅值均值
all_curves = []
for s in range(steps):
    seg = frames[s*N:(s+1)*N]
    mags = []
    prev = seg[0]
    for k in range(1, N):
        cur = seg[k]
        flow = cv2.calcOpticalFlowFarneback(
            prev, cur, None,
            pyr_scale=0.5, levels=3, winsize=21,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        mag = np.sqrt(flow[...,0]**2 + flow[...,1]**2)
        mags.append(float(np.mean(mag)))
        prev = cur
    all_curves.append(mags)

M = np.array(all_curves, dtype=np.float32)  # (steps, N-1)
mean_curve = M.mean(axis=0)
std_curve  = M.std(axis=0)

x = np.arange(N-1)
plt.figure()
plt.plot(x, mean_curve, label="motion energy proxy: mean |flow|")
plt.fill_between(x, mean_curve-std_curve, mean_curve+std_curve, alpha=0.15)
plt.title("Per-step motion energy (optical flow magnitude)")
plt.xlabel("within-step frame index (0..N-2)")
plt.ylabel("mean |flow|")
plt.legend()
plt.show()

print("If your perception is 'move when ghosting', this curve should peak near the middle of the step.")
