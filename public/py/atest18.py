import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ======================
# CONFIG (edit these)
# ======================
IMG_DIR   = r"D:\vectionProject\public\camear1images"
CSV_PATH  = os.path.join(IMG_DIR, "cam1_tree_bbox.csv")

VIDEO_IN  = os.path.join(r"D:\vectionProject\public\freq_test_videos", "orig_lin.mp4")

FPS = 60
SECONDS_PER_STEP = 1.0
N_PER_STEP = int(round(FPS * SECONDS_PER_STEP))  # 60

WIN = 256  # ROI window size (256 or 320 recommended)

# Preview video (ROI overlay)
MAKE_PREVIEW = True
PREVIEW_OUT  = os.path.join(os.path.dirname(VIDEO_IN), "roi_preview.mp4")

# Filters
SKIP_SEGMENT_FIRST_FRAME = True   # skip k==0 for step>0 to avoid duplicate endpoints
SKIP_IF_TOUCH_BORDER     = True   # skip frames where ROI window is clamped to image border

# ======================
# utils
# ======================
def clamp_int(v, lo, hi):
    return max(lo, min(hi, int(v)))

def crop_center(im, cx, cy, win):
    H, W = im.shape[:2]
    half = win // 2
    x0 = int(round(cx - half))
    y0 = int(round(cy - half))
    x0 = clamp_int(x0, 0, W - win)
    y0 = clamp_int(y0, 0, H - win)
    crop = im[y0:y0+win, x0:x0+win]
    return crop, (x0, y0)

def to_gray_float(im_bgr):
    g = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return g

def compute_metrics(prev_g, curr_g):
    # diff energy
    diff_energy = float(np.mean(np.abs(curr_g - prev_g)))

    # grad energy
    gx = cv2.Sobel(curr_g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(curr_g, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx*gx + gy*gy)
    grad_energy = float(np.mean(np.abs(grad)))

    # rms contrast
    rms_contrast = float(np.std(curr_g))

    return diff_energy, grad_energy, rms_contrast

def read_centers_from_csv(csv_path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # valid column may be missing; assume valid if absent
            valid = int(r.get("valid", "1"))
            if valid != 1:
                continue
            rows.append(r)

    if not rows:
        raise RuntimeError("CSV has no valid rows.")

    rows.sort(key=lambda r: int(r["secIndex"]))

    centers = []
    for r in rows:
        # use top-left coords (image-style)
        x = int(r["x_tl"]); y = int(r["y_tl"])
        w = int(r["w_tl"]); h = int(r["h_tl"])
        cx = x + 0.5 * w
        cy = y + 0.5 * h
        centers.append((cx, cy))

    return centers

# ======================
# main
# ======================
def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(CSV_PATH)
    if not os.path.exists(VIDEO_IN):
        raise FileNotFoundError(VIDEO_IN)

    centers = read_centers_from_csv(CSV_PATH)
    M = len(centers)
    if M < 2:
        raise RuntimeError("Need at least 2 bbox rows to interpolate centers.")

    expected_frames = (M - 1) * N_PER_STEP
    print("Keyframes (bbox rows):", M)
    print("Expected frames for full video:", expected_frames)

    cap = cv2.VideoCapture(VIDEO_IN)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {VIDEO_IN}")

    vid_fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {W}x{H} fps={vid_fps} frames={total}")
    if WIN > W or WIN > H:
        raise RuntimeError(f"WIN={WIN} too big for video size {W}x{H}")

    # Process min(total, expected_frames)
    T = min(total, expected_frames)
    print("Processing frames:", T)

    # preview writer
    preview_writer = None
    if MAKE_PREVIEW:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        preview_writer = cv2.VideoWriter(PREVIEW_OUT, fourcc, FPS, (W, H))
        if not preview_writer.isOpened():
            preview_writer = None
            print("Warning: failed to open preview writer, preview disabled.")

    diff_series = []
    grad_series = []
    contrast_series = []
    t_series = []

    prev_crop_g = None

    for fidx in range(T):
        ok, frame = cap.read()
        if not ok:
            break

        step = fidx // N_PER_STEP      # 0..M-2
        k = fidx % N_PER_STEP          # 0..N-1
        if step >= M - 1:
            break

        # skip segment first frame (k==0) for step>0 to avoid duplicate endpoints (B,B)
        if SKIP_SEGMENT_FIRST_FRAME and step > 0 and k == 0:
            # still write preview for inspection if you want
            # (we'll compute ROI anyway)
            pass

        p = k / (N_PER_STEP - 1) if N_PER_STEP > 1 else 1.0
        cx0, cy0 = centers[step]
        cx1, cy1 = centers[step + 1]
        cx = (1 - p) * cx0 + p * cx1
        cy = (1 - p) * cy0 + p * cy1

        crop, (x0, y0) = crop_center(frame, cx, cy, WIN)
        g = to_gray_float(crop)

        # border-touch filter (ROI clamped)
        touching_border = (x0 == 0) or (y0 == 0) or ((x0 + WIN) >= W) or ((y0 + WIN) >= H)

        # preview writing
        if preview_writer is not None:
            vis = frame.copy()
            cv2.rectangle(vis, (x0, y0), (x0 + WIN, y0 + WIN), (0, 255, 0), 2)

            # paste ROI thumb
            thumb_w = WIN // 2
            thumb_h = WIN // 2
            roi_thumb = cv2.resize(crop, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
            ox, oy = 10, 10
            vis[oy:oy + thumb_h, ox:ox + thumb_w] = roi_thumb

            txt = f"t={fidx/FPS:.2f}s step={step} k={k} p={p:.3f} border={1 if touching_border else 0}"
            cv2.putText(vis, txt, (10, H - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            preview_writer.write(vis)

        # Apply filters for metric computation
        if SKIP_SEGMENT_FIRST_FRAME and step > 0 and k == 0:
            prev_crop_g = g
            continue

        if SKIP_IF_TOUCH_BORDER and touching_border:
            prev_crop_g = g
            continue

        if prev_crop_g is not None:
            de, ge, rc = compute_metrics(prev_crop_g, g)
            diff_series.append(de)
            grad_series.append(ge)
            contrast_series.append(rc)
            t_series.append(fidx / FPS)

        prev_crop_g = g

    cap.release()
    if preview_writer is not None:
        preview_writer.release()
        print("Saved preview:", PREVIEW_OUT)

    # Plot
    plt.figure()
    plt.plot(t_series, diff_series)
    plt.xlabel("time (s)")
    plt.ylabel("diff_energy")
    plt.title("diff_energy on orig_lin.mp4 (ROI center interpolated)")
    plt.tight_layout()

    plt.figure()
    plt.plot(t_series, grad_series)
    plt.xlabel("time (s)")
    plt.ylabel("grad_energy")
    plt.title("grad_energy on orig_lin.mp4 (ROI center interpolated)")
    plt.tight_layout()

    plt.figure()
    plt.plot(t_series, contrast_series)
    plt.xlabel("time (s)")
    plt.ylabel("rms_contrast")
    plt.title("rms_contrast on orig_lin.mp4 (ROI center interpolated)")
    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()
