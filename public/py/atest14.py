import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

MP4 = Path(r"D:\vectionProject\public\freq_test_videos\orig_lin.mp4")

FPS = 60
SECONDS_PER_STEP = 1.0
N = int(round(FPS * SECONDS_PER_STEP))

# 建议你盯树，就框树干附近：ROI=(x0,y0,w,h)
ROI = None  # 例如 ROI=(1200, 250, 700, 450)

MAX_STEPS = 80

def preprocess(gray_u8):
    g = gray_u8.astype(np.float32) / 255.0
    if ROI is not None:
        x0,y0,w,h = ROI
        g = g[y0:y0+h, x0:x0+w]
    g = g - float(g.mean())  # 去掉整体亮度漂移（很重要）
    return g

def estimate_p(F, A, B):
    # p_hat = <F-A, B-A> / ||B-A||^2
    d = (B - A).reshape(-1)
    y = (F - A).reshape(-1)
    denom = float(d @ d) + 1e-12
    p = float((y @ d) / denom)
    return float(np.clip(p, 0.0, 1.0))

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

frames = np.stack(frames, axis=0)  # (T,H,W)
T = frames.shape[0]
steps = min(T // N, MAX_STEPS)
if steps <= 0:
    raise RuntimeError("not enough frames")

p_all = []
g_all = []

for s in range(steps):
    seg = frames[s*N:(s+1)*N]  # (N,H,W)
    A = seg[0]
    B = seg[-1]
    p_list = []
    g_list = []
    for k in range(N):
        F = seg[k]
        p = estimate_p(F, A, B)
        g = min(p, 1.0 - p)
        p_list.append(p)
        g_list.append(g)
    p_all.append(p_list)
    g_all.append(g_list)

p_mean = np.mean(np.array(p_all), axis=0)
p_std  = np.std (np.array(p_all), axis=0)
g_mean = np.mean(np.array(g_all), axis=0)
g_std  = np.std (np.array(g_all), axis=0)

x = np.arange(N)
plt.figure()
plt.plot(x, g_mean, label="ghost visibility proxy g=min(p,1-p)")
plt.fill_between(x, g_mean-g_std, g_mean+g_std, alpha=0.15)
plt.title("Estimated ghosting visibility within step")
plt.xlabel("within-step frame index (0..N-1)")
plt.ylabel("g(k)")
plt.legend()
plt.show()

print("Hint: pick a threshold tau, e.g. tau=0.12")
print("Frames with g < tau are likely perceived as 'single image' (stop-like).")
