import os
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import imageio.v2 as imageio


# ============================================================
# CONFIG: 你只需要改这里
# ============================================================
BASE_DIR = Path(r"D:\vectionProject\public\camera2images")  # <-- 改成你的目录

# still images (用于画权重时的“帧编号/时间轴”，以及可选验证)
IMG_GLOB = "cam2_*.png"  # 若你是 cam2_000.png 之类也能匹配

CSV_BBOX = BASE_DIR / "cam2_tree_bbox.csv"

# 5 个条件视频（按你实际文件名改）
VIDEO_PATHS = {
    "C1_linear": str(BASE_DIR / "cam2_linear.mp4"),
    "C2_gauss_sigma0p6": str(BASE_DIR / "cam2_gauss_sigma0p6.mp4"),
    "C3_gauss_injected": str(BASE_DIR / "C3_gauss_ROI_1Hz_injected.mp4"),
    "C4_linear_suppressed": str(BASE_DIR / "C4_linear_ROI_suppressed_softknee.mp4"),
    "C5_linear_energyMatched": str(BASE_DIR / "C5_linear_ROI_energyMatchedToGauss.mp4"),
}

OUT_DIR = BASE_DIR / "_analysis_out_5conds"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# common
FPS = 60
SEC_PER_STEP = 1.0

# weight plot
N_SHOW_FRAMES = 10         # 画前 10 张图的权重（类似你截图 0..9）
WEIGHT_DT = 0.002          # 权重曲线采样间隔（秒），越小越平滑

# gaussian weights
SIGMA_T = 0.6              # sigma0p6

# ROI output video
ROI_OUT_SIZE = 320         # ROI 裁剪后 resize 成固定大小，便于对比（0=不resize）
ROI_VIDEO_FPS = FPS

# Metrics
# Lap: mean(abs(laplacian))
# RMS: ROI 的 RMS contrast = std(roi)
LAPLACIAN_KSIZE = 3

# Spectrum
SPEC_FMAX = 5.0            # 频谱图只画到 5 Hz
SPEC_LO = 0.2              # 计算“总能量”时的下限（避免 DC）

# ============================================================
# Helpers
# ============================================================
def assert_exists(p: str):
    if not os.path.exists(p):
        raise FileNotFoundError(f"Not found: {p}")

def list_still_count(img_dir: Path, glob_pat: str) -> int:
    paths = sorted(img_dir.glob(glob_pat))
    if not paths:
        raise RuntimeError(f"No still images found: {img_dir}/{glob_pat}")
    return len(paths)

def soft_clip01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)

def rfft_power(x: np.ndarray, fs: float):
    x = x.astype(np.float32)
    x = x - x.mean()
    win = np.hanning(len(x)).astype(np.float32)
    X = np.fft.rfft(x * win)
    f = np.fft.rfftfreq(len(x), d=1.0/fs)
    p = (np.abs(X) ** 2).astype(np.float32)
    return f, p

def band_ratio_around(f, p, f0=1.0, bw=0.12, f_lo=0.2, f_hi=10.0):
    m1 = (f >= (f0-bw)) & (f <= (f0+bw))
    mt = (f >= f_lo) & (f <= f_hi)
    p1 = float(p[m1].sum())
    pt = float(p[mt].sum()) + 1e-12
    return p1 / pt

# ============================================================
# BBox track reader (your CSV)
# ============================================================
class BBoxTrack:
    """
    CSV columns required:
      secIndex, rtW, rtH, x_tl, y_tl, w_tl, h_tl, valid
    """
    def __init__(self, csv_path: Path):
        df = pd.read_csv(csv_path)
        df = df[df["valid"] == 1].sort_values("secIndex").copy()
        if df.empty:
            raise RuntimeError("No valid bbox rows in csv.")
        self.df = df

    def make_scaled_for_video(self, vid_w: int, vid_h: int):
        rtW = float(self.df["rtW"].iloc[0])
        rtH = float(self.df["rtH"].iloc[0])
        sx = vid_w / rtW
        sy = vid_h / rtH

        secs = self.df["secIndex"].to_numpy(dtype=np.int32)
        x = (self.df["x_tl"].to_numpy(dtype=np.float32) * sx)
        y = (self.df["y_tl"].to_numpy(dtype=np.float32) * sy)
        w = (self.df["w_tl"].to_numpy(dtype=np.float32) * sx)
        h = (self.df["h_tl"].to_numpy(dtype=np.float32) * sy)

        return _ScaledBBox(secs, x, y, w, h, vid_w, vid_h)

class _ScaledBBox:
    def __init__(self, secs, x, y, w, h, W, H):
        self.secs = secs
        self.x = x; self.y = y; self.w = w; self.h = h
        self.W = W; self.H = H

    def interp(self, t_sec: float):
        if t_sec <= float(self.secs[0]):
            return float(self.x[0]), float(self.y[0]), float(self.w[0]), float(self.h[0])
        if t_sec >= float(self.secs[-1]):
            return float(self.x[-1]), float(self.y[-1]), float(self.w[-1]), float(self.h[-1])

        k = int(math.floor(t_sec))
        i = k - int(self.secs[0])
        p = float(t_sec - k)

        x = self.x[i] * (1 - p) + self.x[i + 1] * p
        y = self.y[i] * (1 - p) + self.y[i + 1] * p
        w = self.w[i] * (1 - p) + self.w[i + 1] * p
        h = self.h[i] * (1 - p) + self.h[i + 1] * p
        return float(x), float(y), float(w), float(h)

    def roi_int(self, t_sec: float):
        x, y, w, h = self.interp(t_sec)
        x0 = int(round(x)); y0 = int(round(y))
        x1 = int(round(x + w)); y1 = int(round(y + h))

        x0 = max(0, min(self.W - 2, x0))
        y0 = max(0, min(self.H - 2, y0))
        x1 = max(x0 + 1, min(self.W - 1, x1))
        y1 = max(y0 + 1, min(self.H - 1, y1))
        return x0, y0, x1, y1

# ============================================================
# Weight functions
# ============================================================
def weights_linear(t_sec: float, n_frames: int):
    """
    normalized weights across all frames at time t
    only 2 nonzero (i and i+1)
    """
    u = t_sec / SEC_PER_STEP
    i = int(math.floor(u))
    p = u - i
    i0 = max(0, min(n_frames - 1, i))
    i1 = max(0, min(n_frames - 1, i + 1))
    w = np.zeros((n_frames,), dtype=np.float32)
    if i0 == i1:
        w[i0] = 1.0
    else:
        w[i0] = 1.0 - p
        w[i1] = p
    return w

def weights_gauss(t_sec: float, n_frames: int, sigma_steps: float):
    """
    normalized gaussian weights across all frames at time t
    (we compute full normalization over all frames for clean plot)
    """
    u = t_sec / SEC_PER_STEP
    ks = np.arange(n_frames, dtype=np.float32)
    d = ks - float(u)
    w = np.exp(-(d*d) / (2.0 * sigma_steps * sigma_steps)).astype(np.float32)
    w = w / (w.sum() + 1e-12)
    return w

def plot_weight_panels(n_frames_total: int):
    """
    5 subplots on one figure.
    Each subplot draws first N_SHOW_FRAMES weight curves.
    """
    t_max = (N_SHOW_FRAMES - 1) * SEC_PER_STEP
    ts = np.arange(0.0, t_max + WEIGHT_DT, WEIGHT_DT, dtype=np.float32)

    conds = [
        ("C1_linear", "linear"),
        ("C2_gauss_sigma0p6", "gauss"),
        ("C3_gauss_injected", "gauss"),
        ("C4_linear_suppressed", "linear"),
        ("C5_linear_energyMatched", "linear"),
    ]

    fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True)
    for ax, (title, mode) in zip(axes, conds):
        for k in range(N_SHOW_FRAMES):
            ys = []
            for t in ts:
                if mode == "linear":
                    w = weights_linear(float(t), n_frames_total)
                else:
                    w = weights_gauss(float(t), n_frames_total, SIGMA_T)
                ys.append(float(w[k]))
            ax.plot(ts, ys)
        ax.set_ylabel("w")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    out_path = OUT_DIR / "weights_5conds.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print("[OK] saved weight figure:", out_path)


# ============================================================
# ROI video + metrics
# ============================================================
def read_video_meta(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or FPS
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Cannot read first frame: {video_path}")
    H, W = frame.shape[:2]
    return float(fps), int(W), int(H)

def make_video_writer(out_path: Path, fps: float, w: int, h: int):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return imageio.get_writer(
        str(out_path),
        fps=fps,
        codec="libx264",
        ffmpeg_params=["-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium"],
    )

def process_one_condition(name: str, video_path: str, bbox_track: BBoxTrack):
    """
    - generates ROI video
    - computes lap & rms series in ROI
    - returns dataframe with metrics
    """
    fps, W, H = read_video_meta(video_path)
    sb = bbox_track.make_scaled_for_video(W, H)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")

    roi_out = OUT_DIR / f"ROI_{name}.mp4"
    metrics_rows = []

    # prepare ROI writer (lazy init after first ROI frame)
    writer = None
    fidx = 0
    prev_roi = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t = fidx / float(fps)
        x0, y0, x1, y1 = sb.roi_int(t)
        roi_bgr = frame[y0:y1, x0:x1]

        # convert to gray float 0..1
        roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

        # metrics
        lap = cv2.Laplacian(roi, cv2.CV_32F, ksize=LAPLACIAN_KSIZE)
        lap_meanabs = float(np.mean(np.abs(lap)))

        rms_contrast = float(np.std(roi))  # RMS contrast

        if prev_roi is None or prev_roi.shape != roi.shape:
            diff_rms = 0.0
        else:
            diff = roi - prev_roi
            diff_rms = float(np.sqrt(np.mean(diff * diff)))
        prev_roi = roi

        metrics_rows.append({
            "cond": name,
            "frame": fidx,
            "time_s": t,
            "lap_meanabs": lap_meanabs,
            "rms_contrast": rms_contrast,
            "diff_rms": diff_rms,
            "roi_w": int(x1-x0),
            "roi_h": int(y1-y0),
        })

        # write ROI video
        if ROI_OUT_SIZE and ROI_OUT_SIZE > 0:
            roi_show = cv2.resize(roi_bgr, (ROI_OUT_SIZE, ROI_OUT_SIZE), interpolation=cv2.INTER_AREA)
        else:
            roi_show = roi_bgr

        if writer is None:
            hh, ww = roi_show.shape[:2]
            writer = make_video_writer(roi_out, ROI_VIDEO_FPS, ww, hh)

        writer.append_data(cv2.cvtColor(roi_show, cv2.COLOR_BGR2RGB))

        fidx += 1

        if fidx % 600 == 0:
            print(f"  {name}: {fidx} frames...")

    cap.release()
    if writer is not None:
        writer.close()

    print("[OK] ROI video saved:", roi_out)
    return pd.DataFrame(metrics_rows)


def plot_metrics_all(df_all: pd.DataFrame):
    # ---- time series: Lap ----
    fig = plt.figure(figsize=(12, 5))
    for cond, g in df_all.groupby("cond"):
        plt.plot(g["time_s"], g["lap_meanabs"], label=cond)
    plt.xlabel("time (s)")
    plt.ylabel("ROI mean |Laplacian|")
    plt.title("ROI Laplacian energy over time (5 conditions)")
    plt.grid(True, alpha=0.25)
    plt.legend()
    out1 = OUT_DIR / "lap_timeseries_5conds.png"
    plt.tight_layout()
    plt.savefig(out1, dpi=200)
    plt.close(fig)
    print("[OK] saved:", out1)

    # ---- time series: RMS contrast ----
    fig = plt.figure(figsize=(12, 5))
    for cond, g in df_all.groupby("cond"):
        plt.plot(g["time_s"], g["rms_contrast"], label=cond)
    plt.xlabel("time (s)")
    plt.ylabel("ROI RMS contrast (std)")
    plt.title("ROI RMS contrast over time (5 conditions)")
    plt.grid(True, alpha=0.25)
    plt.legend()
    out2 = OUT_DIR / "rms_timeseries_5conds.png"
    plt.tight_layout()
    plt.savefig(out2, dpi=200)
    plt.close(fig)
    print("[OK] saved:", out2)

    # ---- time series: diff_rms (optional, motion energy proxy) ----
    fig = plt.figure(figsize=(12, 5))
    for cond, g in df_all.groupby("cond"):
        plt.plot(g["time_s"], g["diff_rms"], label=cond)
    plt.xlabel("time (s)")
    plt.ylabel("ROI diff RMS (frame-to-frame)")
    plt.title("ROI frame-difference RMS over time (5 conditions)")
    plt.grid(True, alpha=0.25)
    plt.legend()
    out3 = OUT_DIR / "diff_rms_timeseries_5conds.png"
    plt.tight_layout()
    plt.savefig(out3, dpi=200)
    plt.close(fig)
    print("[OK] saved:", out3)

def plot_spectrum_all(df_all: pd.DataFrame, col: str, title: str, out_name: str):
    fig = plt.figure(figsize=(12, 5))
    for cond, g in df_all.groupby("cond"):
        y = g[col].to_numpy(dtype=np.float32)
        # need uniform sampling -> it's per frame, ok
        # infer fps from time_s
        if len(g) < 10:
            continue
        dt = float(np.mean(np.diff(g["time_s"])))
        fs = 1.0 / dt if dt > 0 else FPS

        f, p = rfft_power(y, fs)
        m = (f >= 0.0) & (f <= SPEC_FMAX)
        plt.plot(f[m], p[m], label=cond)

    plt.xlabel("frequency (Hz)")
    plt.ylabel("power (a.u.)")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend()
    outp = OUT_DIR / out_name
    plt.tight_layout()
    plt.savefig(outp, dpi=200)
    plt.close(fig)
    print("[OK] saved:", outp)

def make_summary(df_all: pd.DataFrame):
    rows = []
    for cond, g in df_all.groupby("cond"):
        # infer fs
        if len(g) < 10:
            continue
        dt = float(np.mean(np.diff(g["time_s"])))
        fs = 1.0 / dt if dt > 0 else FPS

        # Lap
        y = g["lap_meanabs"].to_numpy(dtype=np.float32)
        f, p = rfft_power(y, fs)
        lap_1hz = band_ratio_around(f, p, f0=1.0, bw=0.12, f_lo=SPEC_LO, f_hi=10.0)

        # RMS
        y2 = g["rms_contrast"].to_numpy(dtype=np.float32)
        f2, p2 = rfft_power(y2, fs)
        rms_1hz = band_ratio_around(f2, p2, f0=1.0, bw=0.12, f_lo=SPEC_LO, f_hi=10.0)

        rows.append({
            "cond": cond,
            "frames": len(g),
            "duration_s": float(g["time_s"].iloc[-1]),
            "lap_mean": float(g["lap_meanabs"].mean()),
            "lap_std": float(g["lap_meanabs"].std()),
            "lap_cv": float(g["lap_meanabs"].std() / (g["lap_meanabs"].mean() + 1e-9)),
            "lap_1Hz_ratio": float(lap_1hz),
            "rms_mean": float(g["rms_contrast"].mean()),
            "rms_std": float(g["rms_contrast"].std()),
            "rms_cv": float(g["rms_contrast"].std() / (g["rms_contrast"].mean() + 1e-9)),
            "rms_1Hz_ratio": float(rms_1hz),
        })

    summ = pd.DataFrame(rows).sort_values("cond")
    out_csv = OUT_DIR / "summary_5conds.csv"
    summ.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("[OK] saved:", out_csv)
    print("\n==== SUMMARY (preview) ====\n", summ)
    return summ


# ============================================================
# Main
# ============================================================
def main():
    # sanity check
    assert_exists(str(CSV_BBOX))
    for k, v in VIDEO_PATHS.items():
        assert_exists(v)

    n_stills = list_still_count(BASE_DIR, IMG_GLOB)
    print("[OK] still images:", n_stills)

    # (A) 5-condition weight figure (5 subplots)
    plot_weight_panels(n_frames_total=n_stills)

    # (B) ROI videos + Lap/RMS metrics for 5 videos
    bbox_track = BBoxTrack(CSV_BBOX)

    all_dfs = []
    for name, vpath in VIDEO_PATHS.items():
        print(f"\n=== Processing {name} ===")
        df = process_one_condition(name, vpath, bbox_track)

        # per-condition metrics csv
        per_csv = OUT_DIR / f"metrics_{name}.csv"
        df.to_csv(per_csv, index=False, encoding="utf-8-sig")
        print("[OK] saved:", per_csv)

        all_dfs.append(df)

    df_all = pd.concat(all_dfs, ignore_index=True)

    # (C) Plots: Lap & RMS analysis
    plot_metrics_all(df_all)

    plot_spectrum_all(
        df_all,
        col="lap_meanabs",
        title="Power spectrum of ROI Laplacian energy (0–5 Hz)",
        out_name="lap_spectrum_5conds.png",
    )

    plot_spectrum_all(
        df_all,
        col="rms_contrast",
        title="Power spectrum of ROI RMS contrast (0–5 Hz)",
        out_name="rms_spectrum_5conds.png",
    )

    # summary table
    make_summary(df_all)

    # also save combined dataframe
    out_all = OUT_DIR / "metrics_ALL_5conds.csv"
    df_all.to_csv(out_all, index=False, encoding="utf-8-sig")
    print("[OK] saved:", out_all)

    print("\nALL DONE. Output folder:", OUT_DIR)


if __name__ == "__main__":
    main()
