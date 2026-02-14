import math
import numpy as np
import pandas as pd
import cv2
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# CONFIG
# =========================
FPS = 60
SECONDS_PER_STEP = 1.0

CSV_BBOX = r"D:\vectionProject\public\camera2images\cam2_tree_bbox.csv"

# 你生成的三个视频（按你之前 OUT_PREFIX 的命名）
VIDS = [
    (0.3, r"D:\vectionProject\public\camera2images\cam2_gauss_EDGE_injected_ROI_strength0.3_soft.mp4"),
    (0.6, r"D:\vectionProject\public\camera2images\cam2_gauss_EDGE_injected_ROI_strength0.6_soft.mp4"),
    (0.9, r"D:\vectionProject\public\camera2images\cam2_gauss_EDGE_injected_ROI_strength0.9_soft.mp4"),
]

OUT_DIR = Path(r"D:\vectionProject\public\camera2images\roi_metrics_injected_strengths")
OUT_CSV = OUT_DIR / "roi_metrics_injected_strengths.csv"
OUT_LAP_PNG = OUT_DIR / "lap_var_timeseries.png"
OUT_RMS_PNG = OUT_DIR / "rms_contrast_timeseries.png"

# 高频分解：sigma 越大 => high 越接近“边缘/细节”
BLUR_SIGMA = 6.0   # 4~10

# lap 计算方式：
# True  = 在 high-band 上算 lap_var（更像“边缘能量”）
# False = 直接在灰度 ROI 上算 lap_var（更像“清晰度/锐度”）
LAP_ON_HIGH = True

# rms 计算方式：
# True  = 用 high-band ROI 算 rms（更边缘）
# False = 用灰度 ROI 算 rms（整体局部对比度）
RMS_ON_HIGH = False

EPS = 1e-8


# =========================
# Helpers
# =========================
def to_gray01(frame_rgb_u8):
    if frame_rgb_u8.ndim == 2:
        g = frame_rgb_u8.astype(np.float32)
    else:
        r = frame_rgb_u8[..., 0].astype(np.float32)
        gg = frame_rgb_u8[..., 1].astype(np.float32)
        b = frame_rgb_u8[..., 2].astype(np.float32)
        g = 0.2126 * r + 0.7152 * gg + 0.0722 * b
    return g / 255.0

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

def split_low_high(gray01, sigma=6.0):
    g = gray01.astype(np.float32)
    low = cv2.GaussianBlur(g, (0, 0), sigmaX=sigma, sigmaY=sigma)
    high = g - low
    return low, high

def metric_lap_var(src01, x0, y0, x1, y1):
    """
    Laplacian variance on ROI.
    src01 can be gray01 (0..1) or high-band (-0.5..0.5 approx)
    """
    p = src01[y0:y1, x0:x1].astype(np.float32)

    # 如果是 high-band，先平移到 0..1 再做 Laplacian 更稳定
    if p.min() < 0.0:
        p = np.clip(p + 0.5, 0.0, 1.0)

    p_u8 = np.clip(p * 255.0 + 0.5, 0, 255).astype(np.uint8)
    lap = cv2.Laplacian(p_u8, cv2.CV_32F, ksize=3)
    return float(lap.var())

def metric_rms_contrast(src01, x0, y0, x1, y1):
    p = src01[y0:y1, x0:x1].astype(np.float32)
    mu = float(p.mean())
    sd = float(p.std())
    return sd / (abs(mu) + EPS)


def analyze_one_video(df_bbox: pd.DataFrame, video_path: str):
    r = imageio.get_reader(video_path)
    n = r.count_frames()
    first = r.get_data(0)
    H, W = first.shape[:2]

    # bbox坐标 -> 视频坐标缩放
    rtW = float(df_bbox.iloc[0]["rtW"]); rtH = float(df_bbox.iloc[0]["rtH"])
    sx = W / rtW if rtW > 1 else 1.0
    sy = H / rtH if rtH > 1 else 1.0

    lap_arr = np.zeros(n, dtype=np.float32)
    rms_arr = np.zeros(n, dtype=np.float32)
    t_arr   = np.zeros(n, dtype=np.float32)

    for i in range(n):
        frame = r.get_data(i)
        t_sec = i / FPS
        t_arr[i] = t_sec

        # ROI from bbox csv (per-second index)
        t_idx = t_sec / SECONDS_PER_STEP
        x, y, w, h = bbox_at_time(df_bbox, t_idx)
        x0 = x*sx; y0 = y*sy; x1 = (x+w)*sx; y1 = (y+h)*sy
        x0, y0, x1, y1 = clamp_box(x0, y0, x1, y1, W, H)

        gray01 = to_gray01(frame)
        _, high = split_low_high(gray01, sigma=BLUR_SIGMA)

        # lap
        src_lap = high if LAP_ON_HIGH else gray01
        lap_arr[i] = metric_lap_var(src_lap, x0, y0, x1, y1)

        # rms
        src_rms = high if RMS_ON_HIGH else gray01
        rms_arr[i] = metric_rms_contrast(src_rms, x0, y0, x1, y1)

        if i % 600 == 0 and i > 0:
            print(f"  {Path(video_path).name}: {i}/{n}")

    r.close()
    return t_arr, lap_arr, rms_arr


def main():
    df_bbox = load_bbox(CSV_BBOX)

    results = {}
    t_ref = None

    for strength, vp in VIDS:
        print("Analyzing:", vp)
        t, lap, rms = analyze_one_video(df_bbox, vp)

        # 对齐长度：以最短的为准（避免不同视频帧数不一致）
        if t_ref is None:
            t_ref = t
            min_len = len(t_ref)
        else:
            min_len = min(min_len, len(t_ref), len(t))

        results[strength] = (t, lap, rms)

    # 统一裁剪到 min_len
    t0 = t_ref[:min_len]
    out = pd.DataFrame({"time_sec": t0})

    for strength in sorted(results.keys()):
        t, lap, rms = results[strength]
        out[f"lap_var_s{strength:.1f}"] = lap[:min_len]
        out[f"rms_contrast_s{strength:.1f}"] = rms[:min_len]

    # 保存 CSV
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print("Saved CSV:", OUT_CSV)

    # 画 lap 图
    plt.figure(figsize=(9, 5))
    for strength in sorted(results.keys()):
        plt.plot(out["time_sec"], out[f"lap_var_s{strength:.1f}"], label=f"strength={strength:.1f}")
    title_lap = f"ROI metric: lap_var  (LAP_ON_HIGH={LAP_ON_HIGH}, blur_sigma={BLUR_SIGMA})"
    plt.title(title_lap)
    plt.xlabel("time (sec)")
    plt.ylabel("lap_var")
    plt.legend()
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig(OUT_LAP_PNG, dpi=200)
    plt.close()
    print("Saved plot:", OUT_LAP_PNG)

    # 画 rms 图
    plt.figure(figsize=(9, 5))
    for strength in sorted(results.keys()):
        plt.plot(out["time_sec"], out[f"rms_contrast_s{strength:.1f}"], label=f"strength={strength:.1f}")
    title_rms = f"ROI metric: rms_contrast  (RMS_ON_HIGH={RMS_ON_HIGH}, blur_sigma={BLUR_SIGMA})"
    plt.title(title_rms)
    plt.xlabel("time (sec)")
    plt.ylabel("rms_contrast")
    plt.legend()
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig(OUT_RMS_PNG, dpi=200)
    plt.close()
    print("Saved plot:", OUT_RMS_PNG)

    print("DONE.")


if __name__ == "__main__":
    main()
