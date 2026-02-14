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
VID_GAUSS = r"D:\vectionProject\public\camera2images\cam2_gauss_sigma0p6.mp4"

# 输出目录（同一目录下）
OUT_PREFIX = r"D:\vectionProject\public\camera2images\cam2_gauss_EDGE_injected_ROI"

# 是否无损
LOSSLESS = False

# 高频分解：sigma 越大 => high 越接近“边缘/细节”
BLUR_SIGMA = 6.0   # 4~10

# 注入尖峰（1Hz）
SPIKE_PERIOD_SEC = 1.0
SPIKE_WIDTH_SEC  = 0.06

# ======= 关键：剂量反应（强度三档）=======
# 你要的 0.3 / 0.6 / 0.9
STRENGTH_LEVELS = [0.3, 0.6, 0.9]

# 增益夹紧（避免太炸）
GAIN_MIN = 1.0
GAIN_MAX = 2.0

# ROI feather：越大边缘过渡越柔和（像素）
FEATHER_PX = 18

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
    """每个 period 附近一个窄高斯尖峰（叠加近邻几个周期，避免边界）"""
    if period <= 0:
        return 1.0
    kc = int(math.floor(t_sec / period))
    s = 0.0
    for k in range(kc - 2, kc + 3):
        tk = k * period
        s += math.exp(-0.5 * ((t_sec - tk) / width) ** 2)
    return 1.0 + strength * s

def split_low_high(gray01, sigma=6.0):
    g = gray01.astype(np.float32)
    low = cv2.GaussianBlur(g, (0, 0), sigmaX=sigma, sigmaY=sigma)
    high = g - low
    return low, high

def make_feather_mask(H, W, x0, y0, x1, y1, feather_px=18):
    """ROI 二值 mask + 高斯羽化 -> 0..1"""
    m = np.zeros((H, W), dtype=np.float32)
    m[y0:y1, x0:x1] = 1.0
    if feather_px > 0:
        # feather_px 对应到 sigma：大概 feather/3
        sigma = max(1.0, feather_px / 3.0)
        m = cv2.GaussianBlur(m, (0, 0), sigmaX=sigma, sigmaY=sigma)
        m = np.clip(m, 0.0, 1.0)
    return m


# =========================
# MAIN: Gauss视频做 ROI-only 边缘尖峰注入（strength 三档）
# =========================
def main():
    df = load_bbox(CSV_BBOX)

    rg = imageio.get_reader(VID_GAUSS)
    n = rg.count_frames()
    first = rg.get_data(0)
    H, W = first.shape[0], first.shape[1]

    # bbox CSV 里坐标系统映射到视频分辨率
    rtW = float(df.iloc[0]["rtW"]); rtH = float(df.iloc[0]["rtH"])
    sx = W / rtW if rtW > 1 else 1.0
    sy = H / rtH if rtH > 1 else 1.0

    print("Frames:", n, "| size:", W, "x", H, "| scale:", sx, sy)

    # 预先算每帧 ROI + feather mask（省事也更一致）
    rois = []
    masks = []
    for i in range(n):
        t_sec = i / FPS
        t_idx = t_sec / SECONDS_PER_STEP
        x, y, w, h = bbox_at_time(df, t_idx)
        x0 = x*sx; y0 = y*sy; x1 = (x+w)*sx; y1 = (y+h)*sy
        x0, y0, x1, y1 = clamp_box(x0, y0, x1, y1, W, H)
        rois.append((x0, y0, x1, y1))
        masks.append(make_feather_mask(H, W, x0, y0, x1, y1, feather_px=FEATHER_PX))

    # 对每个强度输出一个视频
    for strength in STRENGTH_LEVELS:
        out_frames = []

        # 为了避免重复打开，直接用 rg.get_data(i) 取帧
        for i in range(n):
            frame = rg.get_data(i)
            t_sec = i / FPS

            g01 = to_gray01(frame)
            low, high = split_low_high(g01, sigma=BLUR_SIGMA)

            g = spike_gain(t_sec, SPIKE_PERIOD_SEC, SPIKE_WIDTH_SEC, strength)
            g = float(np.clip(g, GAIN_MIN, GAIN_MAX))

            # ROI-only：在 high 上乘一个空间增益场
            # gain_field = 1 + mask*(g-1)
            m = masks[i]
            gain_field = 1.0 + m * (g - 1.0)

            out01 = np.clip(low + gain_field * high, 0.0, 1.0)
            out_frames.append(gray01_to_rgb_u8(out01))

            if i % 600 == 0 and i > 0:
                print(f"  [inject ROI] strength={strength:.1f}  {i}/{n}")

        out_path = f"{OUT_PREFIX}_strength{strength:.1f}_soft.mp4"
        write_video(out_frames, out_path)

    rg.close()
    print("DONE.")

if __name__ == "__main__":
    main()
