import math
import numpy as np
import pandas as pd
import cv2
import imageio.v2 as imageio

# =========================
# CONFIG
# =========================
FPS = 60
SECONDS_PER_STEP = 1.0

CSV_BBOX = r"D:\vectionProject\public\camera2images\cam2_tree_bbox.csv"

VID_GAUSS  = r"D:\vectionProject\public\camera2images\cam2_gauss_sigma0p6.mp4"
VID_LINEAR = r"D:\vectionProject\public\camera2images\cam2_linear.mp4"

OUT_GAUSS_EDGE_SPIKE = r"D:\vectionProject\public\camera2images\cam2_gauss_EDGE_injected_spikes.mp4"
OUT_LINEAR_EDGE_SUPP = r"D:\vectionProject\public\camera2images\cam2_linear_EDGE_suppressed_spikes_v3.mp4"

LOSSLESS = False

# =========================
# low/high split (bigger sigma -> less low-freq change)
# =========================
BLUR_SIGMA = 8.0
BLUR_KSIZE = 0

# =========================
# A) Inject spikes (gauss) - amplify high only
# =========================
SPIKE_PERIOD_SEC = 1.0
SPIKE_WIDTH_SEC  = 0.06
SPIKE_STRENGTH   = 0.9
GAIN_CLAMP_INJECT = (1.0, 2.0)

# =========================
# B) Suppress spikes (linear) - edge-masked soft limiting
# =========================
SUPPRESS_MODE = "lap"          # "lap" or "rms"
SMOOTH_SEC = 1.0

TARGET_MODE  = "quantile"      # "median" or "quantile"
TARGET_Q     = 0.25            # 0.2~0.35
TARGET_SCALE = 0.95            # 0.9~0.98

# keep global gain not too low to avoid blur
GAIN_CLAMP_SUPP = (0.55, 1.0)

# spike gate: easier to trigger & stronger
SPIKE_GATE_K = 1.05            # 1.05~1.20 (smaller -> more often suppress)
SPIKE_GATE_POWER = 2.5         # 1.0~4.0 (bigger -> stronger on spikes)

# soft limit strength (bigger -> suppress strong edges more)
BETA_MAX = 45.0                # 25 -> 45 for more visible effect

# edge mask (only strong edges get heavy suppression)
EDGE_TAU = 0.02                # 0.01~0.05
EDGE_POW = 2.0                 # 1.0~4.0

EPS = 1e-8


# =========================
# Helpers
# =========================
def write_video(frames_rgb_u8, out_path):
    kwargs = dict(fps=FPS, macro_block_size=None)
    if LOSSLESS:
        kwargs.update(codec="libx264", ffmpeg_params=["-crf", "0", "-preset", "veryslow"])
    else:
        kwargs.update(codec="libx264", quality=9)
    print("Writing:", out_path)
    imageio.mimwrite(out_path, frames_rgb_u8, **kwargs)

def to_gray01(frame_rgb_u8):
    if frame_rgb_u8.ndim == 2:
        g = frame_rgb_u8.astype(np.float32)
    else:
        r = frame_rgb_u8[..., 0].astype(np.float32)
        gg = frame_rgb_u8[..., 1].astype(np.float32)
        b = frame_rgb_u8[..., 2].astype(np.float32)
        g = 0.2126 * r + 0.7152 * gg + 0.0722 * b
    return g / 255.0

def gray01_to_rgb_u8(gray01):
    u8 = np.clip(gray01 * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return np.stack([u8, u8, u8], axis=-1)

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

def spike_gain(t_sec, period, width, strength):
    if period <= 0:
        return 1.0
    kc = int(math.floor(t_sec / period))
    s = 0.0
    for k in range(kc - 2, kc + 3):
        tk = k * period
        s += math.exp(-0.5 * ((t_sec - tk) / width) ** 2)
    return 1.0 + strength * s

def split_low_high(gray01, sigma):
    g = gray01.astype(np.float32)
    low = cv2.GaussianBlur(g, (BLUR_KSIZE, BLUR_KSIZE), sigmaX=sigma, sigmaY=sigma)
    high = g - low
    return low, high

def roi_metric_from_high(high01, x0, y0, x1, y1, mode="lap"):
    patch = high01[y0:y1, x0:x1]
    if mode == "rms":
        mu = float(patch.mean())
        sd = float(patch.std())
        return sd / (abs(mu) + EPS)
    elif mode == "lap":
        p2 = np.clip(patch + 0.5, 0.0, 1.0)
        p_u8 = (p2 * 255.0 + 0.5).astype(np.uint8)
        lap = cv2.Laplacian(p_u8, cv2.CV_32F, ksize=3)
        return float(lap.var())
    else:
        raise ValueError("mode must be lap or rms")

def rolling_target(metrics: np.ndarray, win: int, mode: str, q: float):
    s = pd.Series(metrics.astype(np.float32))
    r = s.rolling(win, center=True, min_periods=1)
    if mode == "median":
        return r.median().to_numpy(dtype=np.float32)
    elif mode == "quantile":
        return r.quantile(q).to_numpy(dtype=np.float32)
    else:
        raise ValueError("TARGET_MODE must be 'median' or 'quantile'")


# =========================
# MAIN
# =========================
def main():
    df = load_bbox(CSV_BBOX)

    rg = imageio.get_reader(VID_GAUSS)
    rl = imageio.get_reader(VID_LINEAR)

    n = min(rg.count_frames(), rl.count_frames())
    first = rg.get_data(0)
    H, W = first.shape[0], first.shape[1]

    rtW = float(df.iloc[0]["rtW"]); rtH = float(df.iloc[0]["rtH"])
    sx = W / rtW if rtW > 1 else 1.0
    sy = H / rtH if rtH > 1 else 1.0

    print("Frames:", n, "| size:", W, "x", H, "| scale:", sx, sy)

    # ---------------------------------------------------------
    # A) Gauss + EDGE injected spikes
    # ---------------------------------------------------------
    out_a = []
    for i in range(n):
        frame = rg.get_data(i)
        t_sec = i / FPS

        g01 = to_gray01(frame)
        low, high = split_low_high(g01, sigma=BLUR_SIGMA)

        g = spike_gain(t_sec, SPIKE_PERIOD_SEC, SPIKE_WIDTH_SEC, SPIKE_STRENGTH)
        g = float(np.clip(g, GAIN_CLAMP_INJECT[0], GAIN_CLAMP_INJECT[1]))

        out01 = np.clip(low + g * high, 0.0, 1.0)
        out_a.append(gray01_to_rgb_u8(out01))

        if i % 600 == 0 and i > 0:
            print("  [A inject] ", i, "/", n)

    write_video(out_a, OUT_GAUSS_EDGE_SPIKE)

    # ---------------------------------------------------------
    # B) Linear + EDGE suppressed spikes (edge-masked soft limit)
    # ---------------------------------------------------------
    metrics = np.zeros(n, dtype=np.float32)

    # pass1: ROI metric on HIGH only
    for i in range(n):
        frame = rl.get_data(i)
        t_sec = i / FPS
        t_idx = t_sec / SECONDS_PER_STEP

        x, y, w, h = bbox_at_time(df, t_idx)
        x0 = x*sx; y0 = y*sy; x1 = (x+w)*sx; y1 = (y+h)*sy
        x0, y0, x1, y1 = clamp_box(x0, y0, x1, y1, W, H)

        g01 = to_gray01(frame)
        _, high = split_low_high(g01, sigma=BLUR_SIGMA)

        metrics[i] = roi_metric_from_high(high, x0, y0, x1, y1, mode=SUPPRESS_MODE)

    win = int(round(SMOOTH_SEC * FPS))
    target = rolling_target(metrics, win, TARGET_MODE, TARGET_Q) * float(TARGET_SCALE)

    # build gate curve (1=no change, smaller=stronger suppression)
    gate = np.ones(n, dtype=np.float32)
    for i in range(n):
        cur = float(metrics[i])
        tgt = float(target[i])
        ratio = cur / (tgt + 1e-8)

        # mild gain baseline
        if cur < 1e-8:
            g0 = 1.0
        else:
            g0 = tgt / (cur + 1e-8)
        g0 = min(g0, 1.0)
        g0 = float(np.clip(g0, GAIN_CLAMP_SUPP[0], GAIN_CLAMP_SUPP[1]))

        if ratio <= SPIKE_GATE_K:
            gate[i] = 1.0
        else:
            s = min(1.0, ((ratio - SPIKE_GATE_K) / SPIKE_GATE_K) ** SPIKE_GATE_POWER)
            gate[i] = (1.0 - s) * 1.0 + s * g0

    print("metric range:", float(metrics.min()), float(metrics.max()))
    print("target range:", float(target.min()), float(target.max()))
    print("gate   range:", float(gate.min()), float(gate.max()))
    print("beta max used:", float(BETA_MAX * (1.0 - gate.min())))

    # reopen reader for output
    rl.close()
    rl = imageio.get_reader(VID_LINEAR)

    out_b = []
    for i in range(n):
        frame = rl.get_data(i)
        g01 = to_gray01(frame)
        low, high = split_low_high(g01, sigma=BLUR_SIGMA)

        g = float(gate[i])

        # edge mask from low (stable)
        gx = cv2.Sobel((low*255.0).astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel((low*255.0).astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx*gx + gy*gy) / 255.0

        mask = (mag / (mag + EDGE_TAU)) ** EDGE_POW  # 0..1

        beta = BETA_MAX * (1.0 - g)

        # edge-masked soft limit: strong edges suppressed, weak details preserved
        high2 = high / (1.0 + beta * mask * np.abs(high))

        out01 = np.clip(low + high2, 0.0, 1.0)
        out_b.append(gray01_to_rgb_u8(out01))

        if i % 600 == 0 and i > 0:
            print("  [B supp] ", i, "/", n)

    write_video(out_b, OUT_LINEAR_EDGE_SUPP)

    rg.close()
    rl.close()
    print("DONE.")
    print("A:", OUT_GAUSS_EDGE_SPIKE)
    print("B:", OUT_LINEAR_EDGE_SUPP)


if __name__ == "__main__":
    main()
