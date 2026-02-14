import math
import numpy as np
import pandas as pd
import cv2
import imageio.v2 as imageio
import matplotlib.pyplot as plt

# =========================
# CONFIG (保持与你生成视频那份一致)
# =========================
FPS = 60
SECONDS_PER_STEP = 1.0

CSV_BBOX = r"D:\vectionProject\public\camera2images\cam2_tree_bbox.csv"
VID_LINEAR = r"D:\vectionProject\public\camera2images\cam2_linear.mp4"

# ----- Inject (Gauss global spikes) -----
SPIKE_PERIOD_SEC = 1.0
SPIKE_WIDTH_SEC  = 0.08
SPIKE_STRENGTH   = 1.0
GAIN_CLAMP_INJECT = (1.0, 2.2)

# ----- Suppress (Linear global suppress) -----
SUPPRESS_MODE = "rms"   # "rms" or "lap"
SMOOTH_SEC = 0.6
GAIN_CLAMP_SUPP = (0.35, 1.0)

EPS = 1e-8

# 输出图
OUT_PNG = r"D:\vectionProject\public\camera2images\global_gain_weights.png"


# =========================
# Helpers
# =========================
def load_bbox(csv_path: str):
    df = pd.read_csv(csv_path)
    if "valid" in df.columns:
        df = df[df["valid"] == 1].copy()
    df = df.sort_values("secIndex").reset_index(drop=True)
    return df

def bbox_at_time(df, t_idx: float):
    N = len(df)
    t_idx = float(np.clip(t_idx, 0.0, N - 1.0))
    if t_idx >= N - 1:
        r = df.iloc[N - 1]
        return float(r["x_tl"]), float(r["y_tl"]), float(r["w_tl"]), float(r["h_tl"])
    k = int(np.floor(t_idx))
    u = t_idx - k
    r0 = df.iloc[k]
    r1 = df.iloc[k + 1]
    x = (1-u)*float(r0["x_tl"]) + u*float(r1["x_tl"])
    y = (1-u)*float(r0["y_tl"]) + u*float(r1["y_tl"])
    w = (1-u)*float(r0["w_tl"]) + u*float(r1["w_tl"])
    h = (1-u)*float(r0["h_tl"]) + u*float(r1["h_tl"])
    return x, y, w, h

def clamp_box(x0, y0, x1, y1, W, H):
    x0=int(round(x0)); y0=int(round(y0)); x1=int(round(x1)); y1=int(round(y1))
    x0=max(0, min(x0, W-1)); y0=max(0, min(y0, H-1))
    x1=max(x0+1, min(x1, W)); y1=max(y0+1, min(y1, H))
    return x0, y0, x1, y1

def to_gray01(frame_rgb_u8):
    if frame_rgb_u8.ndim == 2:
        g = frame_rgb_u8.astype(np.float32)
    else:
        r = frame_rgb_u8[..., 0].astype(np.float32)
        gg = frame_rgb_u8[..., 1].astype(np.float32)
        b = frame_rgb_u8[..., 2].astype(np.float32)
        g = 0.2126 * r + 0.7152 * gg + 0.0722 * b
    return g / 255.0

def roi_metric(gray01, x0, y0, x1, y1, mode="rms"):
    patch = gray01[y0:y1, x0:x1]
    if mode == "rms":
        mu = float(patch.mean())
        sd = float(patch.std())
        return sd / (mu + EPS)
    elif mode == "lap":
        p_u8 = np.clip(patch * 255.0 + 0.5, 0, 255).astype(np.uint8)
        lap = cv2.Laplacian(p_u8, cv2.CV_32F, ksize=3)
        return float(lap.var())
    else:
        raise ValueError("mode must be 'rms' or 'lap'")

def moving_average(x, win):
    win = max(1, int(win))
    pad = win // 2
    xp = np.pad(x, (pad, pad), mode="reflect")
    kernel = np.ones(win, dtype=np.float32) / float(win)
    return np.convolve(xp, kernel, mode="valid")

def spike_gain(t_sec, period, width, strength):
    if period <= 0:
        return 1.0
    k_center = int(math.floor(t_sec / period))
    s = 0.0
    for k in range(k_center - 2, k_center + 3):
        tk = k * period
        s += math.exp(-0.5 * ((t_sec - tk) / width) ** 2)
    return 1.0 + strength * s


# =========================
# Main: compute two gains
# =========================
def main():
    df = load_bbox(CSV_BBOX)

    reader = imageio.get_reader(VID_LINEAR)
    n = reader.count_frames()
    first = reader.get_data(0)
    H, W = first.shape[0], first.shape[1]

    rtW = float(df.iloc[0]["rtW"]); rtH = float(df.iloc[0]["rtH"])
    sx = W / rtW if rtW > 1 else 1.0
    sy = H / rtH if rtH > 1 else 1.0

    t = np.arange(n, dtype=np.float32) / FPS

    # ---------- Inject gain(t): purely from formula ----------
    gain_inject = np.zeros(n, dtype=np.float32)
    for i in range(n):
        g = spike_gain(float(t[i]), SPIKE_PERIOD_SEC, SPIKE_WIDTH_SEC, SPIKE_STRENGTH)
        g = float(np.clip(g, GAIN_CLAMP_INJECT[0], GAIN_CLAMP_INJECT[1]))
        gain_inject[i] = g

    # ---------- Suppress gain(t): from ROI metric smoothing ----------
    metrics = np.zeros(n, dtype=np.float32)
    for i in range(n):
        frame = reader.get_data(i)
        g01 = to_gray01(frame)

        t_idx = float(t[i] / SECONDS_PER_STEP)
        x, y, w, h = bbox_at_time(df, t_idx)
        x0 = x * sx; y0 = y * sy; x1 = (x + w) * sx; y1 = (y + h) * sy
        x0, y0, x1, y1 = clamp_box(x0, y0, x1, y1, W, H)

        metrics[i] = roi_metric(g01, x0, y0, x1, y1, mode=SUPPRESS_MODE)

    win = int(round(SMOOTH_SEC * FPS))
    target = moving_average(metrics, win)

    gain_suppress = np.zeros(n, dtype=np.float32)
    for i in range(n):
        cur = float(metrics[i])
        tgt = float(target[i])
        if cur < 1e-6:
            g = 1.0
        else:
            g = tgt / cur
        g = min(g, 1.0)  # 只抑制不增强
        g = float(np.clip(g, GAIN_CLAMP_SUPP[0], GAIN_CLAMP_SUPP[1]))
        gain_suppress[i] = g

    reader.close()

    # =========================
    # Plot
    # =========================
    plt.figure()
    plt.plot(t, gain_inject, label="gain_inject (Gauss + global spikes)")
    plt.plot(t, gain_suppress, label="gain_suppress (Linear + global suppress)")
    plt.xlabel("time (sec)")
    plt.ylabel("gain g(t)")
    plt.title("Global contrast gain weights")
    plt.grid(True)
    plt.legend()
    plt.savefig(OUT_PNG, dpi=200)
    plt.close()
    print("Saved:", OUT_PNG)

    # 另外输出一张：metric vs target（方便你确认抑制逻辑）
    out_png2 = OUT_PNG.replace(".png", f"_{SUPPRESS_MODE}_metric_target.png")
    plt.figure()
    plt.plot(t, metrics, label=f"ROI metric ({SUPPRESS_MODE})")
    plt.plot(t, target, label=f"smoothed target ({SMOOTH_SEC}s)")
    plt.xlabel("time (sec)")
    plt.ylabel("metric value")
    plt.title(f"ROI metric and target (for computing gain_suppress)")
    plt.grid(True)
    plt.legend()
    plt.savefig(out_png2, dpi=200)
    plt.close()
    print("Saved:", out_png2)


if __name__ == "__main__":
    main()
