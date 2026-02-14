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
N_PER_STEP = int(round(FPS * SECONDS_PER_STEP))  # 60 (frames per step)
STEP_LEN_DIFF = N_PER_STEP - 1                   # 59 (diff samples per step)

WIN = 256  # ROI window size (256 or 320 recommended)

# Edge-only metrics
EDGE_TOP_PCT = 0.10        # top 10% strongest-gradient pixels
EDGE_DILATE  = 1           # 0=no, 1=3x3 dilate once (makes edge mask more stable)

# Preview video (ROI overlay)
MAKE_PREVIEW = True
PREVIEW_OUT  = os.path.join(os.path.dirname(VIDEO_IN), "roi_preview.mp4")

# Filters
SKIP_SEGMENT_FIRST_FRAME = True   # skip k==0 for step>0 to avoid duplicate endpoints
SKIP_IF_TOUCH_BORDER     = True   # skip frames where ROI window is clamped to image border

# Limit analysis (optional)
MAX_STEPS = None  # e.g. 13; None = use all available

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
    touching_border = (x0 == 0) or (y0 == 0) or ((x0 + win) >= W) or ((y0 + win) >= H)
    return crop, (x0, y0), touching_border

def to_gray_float(im_bgr):
    return cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

def grad_mag(gray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.sqrt(gx*gx + gy*gy)

def edge_mask_from_gray(gray, top_pct=0.10, dilate=1):
    g = grad_mag(gray)
    thr = np.quantile(g, 1.0 - float(top_pct))
    m = (g >= thr)
    if dilate and dilate > 0:
        k = np.ones((3,3), np.uint8)
        m = cv2.dilate(m.astype(np.uint8), k, iterations=int(dilate)).astype(bool)
    return m

def compute_metrics(prev_g, curr_g, mask=None):
    """
    diff_energy: mean |curr - prev|
    grad_energy: mean |∇curr|
    rms_contrast: std(curr)
    If mask provided, compute over masked pixels only.
    """
    if mask is None:
        diff_energy = float(np.mean(np.abs(curr_g - prev_g)))
        gmag = grad_mag(curr_g)
        grad_energy = float(np.mean(gmag))
        rms_contrast = float(np.std(curr_g))
        return diff_energy, grad_energy, rms_contrast

    m = mask.astype(bool)
    if m.sum() < 10:
        return np.nan, np.nan, np.nan

    diff_energy = float(np.mean(np.abs(curr_g[m] - prev_g[m])))
    gmag = grad_mag(curr_g)
    grad_energy = float(np.mean(gmag[m]))
    rms_contrast = float(np.std(curr_g[m]))
    return diff_energy, grad_energy, rms_contrast

def read_centers_from_csv(csv_path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            valid = int(r.get("valid", "1"))
            if valid != 1:
                continue
            rows.append(r)

    if not rows:
        raise RuntimeError("CSV has no valid rows.")

    rows.sort(key=lambda r: int(r["secIndex"]))

    centers = []
    for r in rows:
        x = int(r["x_tl"]); y = int(r["y_tl"])
        w = int(r["w_tl"]); h = int(r["h_tl"])
        cx = x + 0.5 * w
        cy = y + 0.5 * h
        centers.append((cx, cy))

    return centers

# ======================
# helper for per-step aligned stats
# ======================
def aligned_mean_std(M):
    """
    M: (steps, STEP_LEN_DIFF) with NaN allowed
    returns mean,std over steps for each within-step index
    """
    mean = np.nanmean(M, axis=0)
    std  = np.nanstd(M, axis=0)
    return mean, std

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

    # how many steps can we actually process?
    # step index ranges 0..(M-2); each step has N_PER_STEP frames in the mp4 construction
    max_steps_from_csv = M - 1
    max_steps_from_video = total // N_PER_STEP
    num_steps = min(max_steps_from_csv, max_steps_from_video)
    if MAX_STEPS is not None:
        num_steps = min(num_steps, int(MAX_STEPS))
    if num_steps <= 0:
        raise RuntimeError("No complete steps to process.")

    T = num_steps * N_PER_STEP
    print("Keyframes (bbox rows):", M)
    print("Using steps:", num_steps, "=> frames to read:", T)

    # preview writer
    preview_writer = None
    if MAKE_PREVIEW:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        preview_writer = cv2.VideoWriter(PREVIEW_OUT, fourcc, FPS, (W, H))
        if not preview_writer.isOpened():
            preview_writer = None
            print("Warning: failed to open preview writer, preview disabled.")

    # time series (continuous)
    t_series = []
    full_diff, full_grad, full_con = [], [], []
    edge_diff, edge_grad, edge_con = [], [], []

    # per-step aligned matrices: (steps, STEP_LEN_DIFF)
    full_Mdiff = np.full((num_steps, STEP_LEN_DIFF), np.nan, np.float32)
    full_Mgrad = np.full((num_steps, STEP_LEN_DIFF), np.nan, np.float32)
    full_Mcon  = np.full((num_steps, STEP_LEN_DIFF), np.nan, np.float32)

    edge_Mdiff = np.full((num_steps, STEP_LEN_DIFF), np.nan, np.float32)
    edge_Mgrad = np.full((num_steps, STEP_LEN_DIFF), np.nan, np.float32)
    edge_Mcon  = np.full((num_steps, STEP_LEN_DIFF), np.nan, np.float32)

    prev_crop_g = None
    prev_edge_mask = None

    for fidx in range(T):
        ok, frame = cap.read()
        if not ok:
            break

        step = fidx // N_PER_STEP      # 0..num_steps-1
        k = fidx % N_PER_STEP          # 0..N-1

        # interpolate ROI center between bbox keyframes (one per second)
        p = k / (N_PER_STEP - 1) if N_PER_STEP > 1 else 1.0
        cx0, cy0 = centers[step]
        cx1, cy1 = centers[step + 1]
        cx = (1 - p) * cx0 + p * cx1
        cy = (1 - p) * cy0 + p * cy1

        crop, (x0, y0), touching_border = crop_center(frame, cx, cy, WIN)
        g = to_gray_float(crop)

        # edge mask from current gray ROI
        m_edge = edge_mask_from_gray(g, top_pct=EDGE_TOP_PCT, dilate=EDGE_DILATE)

        # preview
        if preview_writer is not None:
            vis = frame.copy()
            cv2.rectangle(vis, (x0, y0), (x0 + WIN, y0 + WIN), (0, 255, 0), 2)

            thumb_w = WIN // 2
            thumb_h = WIN // 2
            roi_thumb = cv2.resize(crop, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
            ox, oy = 10, 10
            vis[oy:oy + thumb_h, ox:ox + thumb_w] = roi_thumb

            txt = f"t={fidx/FPS:.2f}s step={step} k={k} p={p:.3f} border={1 if touching_border else 0}"
            cv2.putText(vis, txt, (10, H - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
            preview_writer.write(vis)

        # skip duplicate endpoint frame at segment start (step>0,k==0)
        if SKIP_SEGMENT_FIRST_FRAME and step > 0 and k == 0:
            prev_crop_g = g
            prev_edge_mask = m_edge
            continue

        # optional border filter
        if SKIP_IF_TOUCH_BORDER and touching_border:
            prev_crop_g = g
            prev_edge_mask = m_edge
            continue

        if prev_crop_g is not None:
            # FULL metrics
            de, ge, rc = compute_metrics(prev_crop_g, g, mask=None)

            # EDGE-only metrics:
            # Use union mask of prev & curr to keep diff comparable
            if prev_edge_mask is None:
                union_mask = m_edge
            else:
                union_mask = (prev_edge_mask | m_edge)
            de2, ge2, rc2 = compute_metrics(prev_crop_g, g, mask=union_mask)

            # time series
            t = fidx / FPS
            t_series.append(t)
            full_diff.append(de); full_grad.append(ge); full_con.append(rc)
            edge_diff.append(de2); edge_grad.append(ge2); edge_con.append(rc2)

            # per-step aligned index (diff is between k-1 -> k)
            if k >= 1:
                j = k - 1  # 0..N-2
                full_Mdiff[step, j] = de
                full_Mgrad[step, j] = ge
                full_Mcon [step, j] = rc

                edge_Mdiff[step, j] = de2
                edge_Mgrad[step, j] = ge2
                edge_Mcon [step, j] = rc2

        prev_crop_g = g
        prev_edge_mask = m_edge

    cap.release()
    if preview_writer is not None:
        preview_writer.release()
        print("Saved preview:", PREVIEW_OUT)

    # ======================
    # PLOTS (time series)
    # ======================
    plt.figure()
    plt.plot(t_series, full_con, label="full ROI")
    plt.plot(t_series, edge_con, label=f"edge-only top{int(EDGE_TOP_PCT*100)}%")
    plt.xlabel("time (s)")
    plt.ylabel("rms_contrast (std)")
    plt.title("rms_contrast on orig_lin.mp4")
    plt.legend()
    plt.tight_layout()

    plt.figure()
    plt.plot(t_series, full_grad, label="full ROI")
    plt.plot(t_series, edge_grad, label=f"edge-only top{int(EDGE_TOP_PCT*100)}%")
    plt.xlabel("time (s)")
    plt.ylabel("grad_energy (mean |grad|)")
    plt.title("grad_energy on orig_lin.mp4")
    plt.legend()
    plt.tight_layout()

    plt.figure()
    plt.plot(t_series, full_diff, label="full ROI")
    plt.plot(t_series, edge_diff, label=f"edge-only top{int(EDGE_TOP_PCT*100)}%")
    plt.xlabel("time (s)")
    plt.ylabel("diff_energy (mean |diff|)")
    plt.title("diff_energy on orig_lin.mp4")
    plt.legend()
    plt.tight_layout()

    # ======================
    # PLOTS (per-step aligned mean±std)
    # x-axis in p (0..1) aligned to within-step index
    # ======================
    x = np.linspace(0.0, 1.0, STEP_LEN_DIFF)  # corresponds to k=1..N-1 roughly

    def plot_aligned(M_full, M_edge, title, ylabel):
        mf, sf = aligned_mean_std(M_full)
        me, se = aligned_mean_std(M_edge)
        plt.figure()
        plt.plot(x, mf, label="full ROI mean")
        plt.fill_between(x, mf - sf, mf + sf, alpha=0.15)
        plt.plot(x, me, label=f"edge-only top{int(EDGE_TOP_PCT*100)}% mean")
        plt.fill_between(x, me - se, me + se, alpha=0.15)
        plt.axvline(0.5, linestyle="--", alpha=0.6, label="p=0.5 (max ghost)")
        plt.xlabel("p within step (k/(N-1))")
        plt.ylabel(ylabel)
        plt.title(f"{title} (mean±std), steps={num_steps}")
        plt.legend()
        plt.tight_layout()

    plot_aligned(full_Mcon,  edge_Mcon,  "Per-step aligned rms_contrast", "std(gray)")
    plot_aligned(full_Mgrad, edge_Mgrad, "Per-step aligned grad_energy",  "mean |grad|")
    plot_aligned(full_Mdiff, edge_Mdiff, "Per-step aligned diff_energy",  "mean |diff|")

    plt.show()

if __name__ == "__main__":
    main()
