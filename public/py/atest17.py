import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ======================
# CONFIG
# ======================
IMG_DIR   = r"D:\vectionProject\public\camear1images"
CSV_PATH  = os.path.join(IMG_DIR, "cam1_tree_bbox.csv")
VIDEO_IN  = os.path.join(r"D:\vectionProject\public\freq_test_videos", "orig_lin.mp4")

FPS = 60
SECONDS_PER_STEP = 1.0
N_PER_STEP = int(round(FPS * SECONDS_PER_STEP))  # 60

WIN = 256  # 256 or 320

# Optional filters
SKIP_SEGMENT_FIRST_FRAME = True   # skip k==0 when step>0 (avoid duplicate endpoints)
SKIP_TOUCH_BORDER = False         # True if you want to discard frames where ROI is clamped

# ======================
# utils
# ======================
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def crop_center(im, cx, cy, win):
    H, W = im.shape[:2]
    half = win // 2
    x0 = int(round(cx - half))
    y0 = int(round(cy - half))
    x0 = clamp(x0, 0, W - win)
    y0 = clamp(y0, 0, H - win)
    crop = im[y0:y0+win, x0:x0+win]
    return crop, (x0, y0)

def to_gray_float(im_bgr):
    return cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

def metrics(prev_g, curr_g):
    diff_energy = float(np.mean(np.abs(curr_g - prev_g)))

    gx = cv2.Sobel(curr_g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(curr_g, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx*gx + gy*gy)
    grad_energy = float(np.mean(np.abs(grad)))

    rms_contrast = float(np.std(curr_g))
    return diff_energy, grad_energy, rms_contrast

# ======================
# read bbox list from CSV
# ======================
rows = []
with open(CSV_PATH, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for r in reader:
        if int(r.get("valid", "1")) != 1:
            continue
        rows.append(r)

rows.sort(key=lambda r: int(r["secIndex"]))

# boxes[i] corresponds to secIndex i (after sorting)
boxes = []
for r in rows:
    x = int(r["x_tl"])
    y = int(r["y_tl"])
    w = int(r["w_tl"])
    h = int(r["h_tl"])
    boxes.append((x, y, w, h))

M = len(boxes)
if M < 2:
    raise RuntimeError("Need at least 2 bbox rows in CSV")

expected_frames = (M - 1) * N_PER_STEP
print("Keyframes:", M, "Expected video frames:", expected_frames)

# ======================
# open video
# ======================
cap = cv2.VideoCapture(VIDEO_IN)
if not cap.isOpened():
    raise RuntimeError(f"Failed to open video: {VIDEO_IN}")

vid_fps = cap.get(cv2.CAP_PROP_FPS)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print("Video:", W, "x", H, "fps=", vid_fps, "frames=", total)

T = min(total, expected_frames)

diff_series = []
grad_series = []
contrast_series = []
t_series = []

prev_crop_g = None

for fidx in range(T):
    ok, frame = cap.read()
    if not ok:
        break

    step = fidx // N_PER_STEP  # 0..M-2
    k = fidx % N_PER_STEP      # 0..N-1
    if step >= M - 1:
        break

    # ===== Union bbox center for this whole 1-second segment (A->B) =====
    x0, y0, w0, h0 = boxes[step]
    x1, y1, w1, h1 = boxes[step + 1]

    ux0 = min(x0, x1)
    uy0 = min(y0, y1)
    ux1 = max(x0 + w0, x1 + w1)
    uy1 = max(y0 + h0, y1 + h1)

    cx = 0.5 * (ux0 + ux1)
    cy = 0.5 * (uy0 + uy1)

    crop, (rx0, ry0) = crop_center(frame, cx, cy, WIN)
    g = to_gray_float(crop)

    # optional: border-touch detection (ROI clamped)
    touching_border = (rx0 == 0) or (ry0 == 0) or ((rx0 + WIN) >= W) or ((ry0 + WIN) >= H)

    # skip first frame of each segment to avoid duplicate endpoint frame (B,B)
    if SKIP_SEGMENT_FIRST_FRAME and step > 0 and k == 0:
        prev_crop_g = g
        continue

    # optional: skip if touching border
    if SKIP_TOUCH_BORDER and touching_border:
        prev_crop_g = g
        continue

    if prev_crop_g is not None:
        de, ge, rc = metrics(prev_crop_g, g)
        diff_series.append(de)
        grad_series.append(ge)
        contrast_series.append(rc)
        t_series.append(fidx / FPS)

    prev_crop_g = g

cap.release()

# ======================
# plot
# ======================
plt.figure()
plt.plot(t_series, diff_series)
plt.xlabel("time (s)")
plt.ylabel("diff_energy")
plt.title("diff_energy on orig_lin.mp4 (ROI=union bbox center per segment)")
plt.tight_layout()

plt.figure()
plt.plot(t_series, grad_series)
plt.xlabel("time (s)")
plt.ylabel("grad_energy")
plt.title("grad_energy on orig_lin.mp4 (ROI=union bbox center per segment)")
plt.tight_layout()

plt.figure()
plt.plot(t_series, contrast_series)
plt.xlabel("time (s)")
plt.ylabel("rms_contrast")
plt.title("rms_contrast on orig_lin.mp4 (ROI=union bbox center per segment)")
plt.tight_layout()

plt.show()
