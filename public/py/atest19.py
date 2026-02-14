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
N_PER_STEP = int(round(FPS * SECONDS_PER_STEP))  # e.g., 60 frames/step
STEP_LEN = N_PER_STEP - 1                        # metrics length per step (diff is between frames)

WIN = 256  # ROI window size

# Preview video (ROI overlay)
MAKE_PREVIEW = True
PREVIEW_OUT  = os.path.join(os.path.dirname(VIDEO_IN), "roi_preview.mp4")

# Filters
SKIP_SEGMENT_FIRST_FRAME = True   # skip k==0 for step>0 to avoid duplicate endpoints (B,B)
SKIP_IF_TOUCH_BORDER     = True   # skip frames where ROI window is clamped to image border

# Per-step averaging options
MAX_STEPS_TO_USE = None   # e.g., 60, or None for all
SAVE_PLOTS = True
PLOTS_OUT_DIR = os.path.join(os.path.dirname(VIDEO_IN), "per_step_plots")
os.makedirs(PLOTS_OUT_DIR, exist_ok=True)

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
    # diff energy (frame-to-frame)
    diff_energy = float(np.mean(np.abs(curr_g - prev_g)))

    # grad energy (on current frame)
    gx = cv2.Sobel(curr_g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(curr_g, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx*gx + gy*gy)
    grad_energy = float(np.mean(np.abs(grad)))

    # rms contrast (on current frame)
    rms_contrast = float(np.std(curr_g))

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

def finalize_step_if_valid(step_idx, step_invalid, step_de, step_ge, step_rc,
                           steps_de, steps_ge, steps_rc, reason=""):
    """Append this step's metrics if it has full length and is not invalid."""
    if step_invalid:
        return
    if len(step_de) != STEP_LEN or len(step_ge) != STEP_LEN or len(step_rc) != STEP_LEN:
        return
    steps_de.append(np.array(step_de, dtype=np.float32))
    steps_ge.append(np.array(step_ge, dtype=np.float32))
    steps_rc.append(np.array(step_rc, dtype=np.float32))

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

    # Per-step containers
    steps_de, steps_ge, steps_rc = [], [], []

    # Current step buffers
    current_step = None
    step_invalid = False
    step_de, step_ge, step_rc = [], [], []

    prev_crop_g = None

    for fidx in range(T):
        ok, frame = cap.read()
        if not ok:
            break

        step = fidx // N_PER_STEP      # 0..M-2
        k = fidx % N_PER_STEP          # 0..N-1
        if step >= M - 1:
            break

        # Step transition: finalize previous step
        if current_step is None:
            current_step = step
        elif step != current_step:
            finalize_step_if_valid(current_step, step_invalid, step_de, step_ge, step_rc,
                                   steps_de, steps_ge, steps_rc)
            # reset step buffers
            current_step = step
            step_invalid = False
            step_de, step_ge, step_rc = [], [], []
            prev_crop_g = None  # important: avoid cross-step diff

        # p in [0,1]
        p = k / (N_PER_STEP - 1) if N_PER_STEP > 1 else 1.0
        cx0, cy0 = centers[step]
        cx1, cy1 = centers[step + 1]
        cx = (1 - p) * cx0 + p * cx1
        cy = (1 - p) * cy0 + p * cy1

        crop, (x0, y0) = crop_center(frame, cx, cy, WIN)
        g = to_gray_float(crop)

        touching_border = (x0 == 0) or (y0 == 0) or ((x0 + WIN) >= W) or ((y0 + WIN) >= H)

        # Preview writing
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

        # Border filter: invalidate this whole step (recommended, keeps clean step-aligned curves)
        if SKIP_IF_TOUCH_BORDER and touching_border:
            step_invalid = True
            prev_crop_g = g
            continue

        # Skip duplicated endpoint frame at step boundary: (step>0 and k==0)
        if SKIP_SEGMENT_FIRST_FRAME and step > 0 and k == 0:
            prev_crop_g = g
            continue

        # For k==0 of step0, just set prev
        if k == 0:
            prev_crop_g = g
            continue

        # Compute metrics (for this within-step transition index j=k-1)
        if prev_crop_g is not None:
            de, ge, rc = compute_metrics(prev_crop_g, g)
            step_de.append(de)
            step_ge.append(ge)
            step_rc.append(rc)

        prev_crop_g = g

        # Optional: stop after some steps (for speed)
        if MAX_STEPS_TO_USE is not None and len(steps_de) >= MAX_STEPS_TO_USE:
            break

    cap.release()
    if preview_writer is not None:
        preview_writer.release()
        print("Saved preview:", PREVIEW_OUT)

    # finalize last step
    if current_step is not None:
        finalize_step_if_valid(current_step, step_invalid, step_de, step_ge, step_rc,
                               steps_de, steps_ge, steps_rc)

    if len(steps_de) == 0:
        raise RuntimeError("No valid steps collected. Try disabling SKIP_IF_TOUCH_BORDER or increasing WIN margin.")

    # Stack: (S, STEP_LEN)
    A_de = np.stack(steps_de, axis=0)
    A_ge = np.stack(steps_ge, axis=0)
    A_rc = np.stack(steps_rc, axis=0)

    # Mean/std over steps
    m_de, s_de = A_de.mean(axis=0), A_de.std(axis=0)
    m_ge, s_ge = A_ge.mean(axis=0), A_ge.std(axis=0)
    m_rc, s_rc = A_rc.mean(axis=0), A_rc.std(axis=0)

    # x-axis: use p for each metric sample (diff is between k-1 -> k, so sample p=k/(N-1), k=1..N-1)
    x_p = np.linspace(1.0/(N_PER_STEP-1), 1.0, STEP_LEN)
    # also time within step
    x_t = x_p * SECONDS_PER_STEP

    print(f"Collected valid steps: {len(steps_de)} (each step_len={STEP_LEN})")

    def plot_mean_std(x, mean, std, title, ylabel, out_png_name):
        plt.figure(figsize=(7.2, 4.2))
        plt.plot(x, mean, label="mean")
        plt.fill_between(x, mean-std, mean+std, alpha=0.2, label="±1 std")
        # Mark p=0.5
        plt.axvline(0.5, linestyle="--", linewidth=1.5, label="p=0.5 (max ghost)")
        plt.title(title)
        plt.xlabel("p within step (k/(N-1))")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        if SAVE_PLOTS:
            out_png = os.path.join(PLOTS_OUT_DIR, out_png_name)
            plt.savefig(out_png, dpi=150)
            print("Saved plot:", out_png)

    plot_mean_std(
        x_p, m_de, s_de,
        f"Per-step aligned diff_energy (mean±std), steps={len(steps_de)}",
        "mean |diff|",
        "perstep_diff_energy.png"
    )

    plot_mean_std(
        x_p, m_ge, s_ge,
        f"Per-step aligned grad_energy (mean±std), steps={len(steps_ge)}",
        "mean |grad|",
        "perstep_grad_energy.png"
    )

    plot_mean_std(
        x_p, m_rc, s_rc,
        f"Per-step aligned rms_contrast (mean±std), steps={len(steps_rc)}",
        "std(gray)",
        "perstep_rms_contrast.png"
    )

    plt.show()

if __name__ == "__main__":
    main()
