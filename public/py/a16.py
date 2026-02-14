import math
import numpy as np
import pandas as pd
import cv2
import imageio.v2 as imageio

# =========================
# CONFIG
# =========================
FPS = 60

CSV_BBOX = r"D:\vectionProject\public\camera2images\cam2_tree_bbox.csv"
VID_GAUSS = r"D:\vectionProject\public\camera2images\cam2_gauss_sigma0p6.mp4"

# 输出：二选一（你都可以生成）
OUT_C3_GLOBAL = r"D:\vectionProject\public\camera2images\C3_gauss_EDGE_injected_spikes_GLOBAL.mp4"
OUT_C3_ROI    = r"D:\vectionProject\public\camera2images\C3_gauss_EDGE_injected_spikes_ROI_feather.mp4"

LOSSLESS = False

# low/high split（保持和你旧视频一致）
BLUR_SIGMA = 8.0
BLUR_KSIZE = 0

# spike 门控（保持和你旧视频一致）
SPIKE_PERIOD_SEC = 1.0
SPIKE_WIDTH_SEC  = 0.06
SPIKE_STRENGTH   = 0.9
GAIN_CLAMP_INJECT = (1.0, 2.0)

# ROI feather（只在 ROI 版本用）
FEATHER_PX = 24   # 16~48 都行，越大边界越自然也越“扩散”

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
    r = frame_rgb_u8[..., 0].astype(np.float32)
    g = frame_rgb_u8[..., 1].astype(np.float32)
    b = frame_rgb_u8[..., 2].astype(np.float32)
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return y / 255.0

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
    kc = int(math.floor(t_sec / period))
    s = 0.0
    for k in range(kc - 2, kc + 3):
        tk = k * period
        s += math.exp(-0.5 * ((t_sec - tk) / width) ** 2)
    return 1.0 + strength * s

def split_low_high(gray01, sigma):
    low = cv2.GaussianBlur(gray01.astype(np.float32), (BLUR_KSIZE, BLUR_KSIZE), sigmaX=sigma, sigmaY=sigma)
    high = gray01.astype(np.float32) - low
    return low, high

def feather_mask(W, H, x0, y0, x1, y1, feather_px):
    """Return 0..1 mask with soft edges around ROI"""
    m = np.zeros((H, W), dtype=np.float32)
    m[y0:y1, x0:x1] = 1.0
    if feather_px <= 0:
        return m
    k = feather_px * 2 + 1
    m = cv2.GaussianBlur(m, (k, k), sigmaX=feather_px, sigmaY=feather_px)
    m = np.clip(m, 0.0, 1.0)
    return m

# =========================
# MAIN
# =========================
def main():
    df = load_bbox(CSV_BBOX)

    rg = imageio.get_reader(VID_GAUSS)
    n = rg.count_frames()
    first = rg.get_data(0)
    H, W = first.shape[0], first.shape[1]

    rtW = float(df.iloc[0]["rtW"]); rtH = float(df.iloc[0]["rtH"])
    sx = W / rtW if rtW > 1 else 1.0
    sy = H / rtH if rtH > 1 else 1.0
    print("Frames:", n, "| size:", W, "x", H, "| scale:", sx, sy)

    out_global = []
    out_roi = []

    # prebuild feather mask cache per second index (optional)
    for i in range(n):
        frame = rg.get_data(i)
        t_sec = i / FPS

        g01 = to_gray01(frame)
        low, high = split_low_high(g01, sigma=BLUR_SIGMA)

        g = spike_gain(t_sec, SPIKE_PERIOD_SEC, SPIKE_WIDTH_SEC, SPIKE_STRENGTH)
        g = float(np.clip(g, GAIN_CLAMP_INJECT[0], GAIN_CLAMP_INJECT[1]))

        # ---- C3a: GLOBAL ----
        out01_a = np.clip(low + g * high, 0.0, 1.0)
        out_global.append(gray01_to_rgb_u8(out01_a))

        # ---- C3b: ROI + feather ----
        t_idx = t_sec / 1.0  # secIndex is in seconds
        x, y, w, h = bbox_at_time(df, t_idx)
        x0 = x*sx; y0 = y*sy; x1 = (x+w)*sx; y1 = (y+h)*sy
        x0, y0, x1, y1 = clamp_box(x0, y0, x1, y1, W, H)

        m = feather_mask(W, H, x0, y0, x1, y1, FEATHER_PX)

        # apply only inside mask (blend original & injected)
        injected = np.clip(low + g * high, 0.0, 1.0)
        out01_b = (1.0 - m) * g01 + m * injected
        out_roi.append(gray01_to_rgb_u8(out01_b))

        if i % 600 == 0 and i > 0:
            print("  ", i, "/", n)

    rg.close()

    write_video(out_global, OUT_C3_GLOBAL)
    write_video(out_roi, OUT_C3_ROI)

    print("DONE.")
    print("C3 GLOBAL:", OUT_C3_GLOBAL)
    print("C3 ROI+feather:", OUT_C3_ROI)

if __name__ == "__main__":
    main()
