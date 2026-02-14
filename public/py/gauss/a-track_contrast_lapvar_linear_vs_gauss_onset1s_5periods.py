import os
import numpy as np
import pandas as pd
import cv2
import imageio.v2 as imageio
import matplotlib.pyplot as plt

# ======================
# CONFIG
# ======================
CSV_PATH = r"D:\vectionProject\public\camera3images\rois.csv"  # ROI CSV（若只想算全画面，可不填，但这里保留ROI功能）
VIDEOS = {
    # 打开你要分析的条件
    "linear": r"D:\vectionProject\public\camera3images\cam3_linear.mp4",
    "gauss":  r"D:\vectionProject\public\camera3images\cam3_truncgauss3tap_sigma0p6.mp4",
}

OUT_DIR = r"D:\vectionProject\public\camera3images\track_contrast_out_roi_and_full"
os.makedirs(OUT_DIR, exist_ok=True)

FPS = 60
SECONDS_PER_STEP = 1.0    # 1 Hz: 一秒一个 ROI keyframe
INTERP_MODE = "lerp"      # "lerp" or "step"

# ===== 分析时间窗设置 =====
ONSET_SEC = 1.0           # 丢掉 0–1 s 的 onset
N_PERIODS = 5             # 使用 5 个周期
ANALYSIS_DURATION = N_PERIODS * SECONDS_PER_STEP   # 5 s
ANALYSIS_START = ONSET_SEC
ANALYSIS_END = ONSET_SEC + ANALYSIS_DURATION       # 1–6 s

# CSV columns
IDX_COL = "secIndex"
VALID_COL = "valid"
RTW_COL = "rtW"
RTH_COL = "rtH"
X_COL = "x_tl"
Y_COL = "y_tl"
W_COL = "w_tl"
H_COL = "h_tl"

EPS = 1e-8

# 输出带框验证视频（ROI 跟踪）
MAKE_DEBUG_VIDEO = True
BOX_COLOR_BGR = (0, 255, 0)
BOX_THICKNESS = 4

LABEL_FS = 25


# ======================
# Helpers
# ======================
def clamp_box(x0, y0, x1, y1, W, H):
    x0 = int(round(x0))
    y0 = int(round(y0))
    x1 = int(round(x1))
    y1 = int(round(y1))
    x0 = max(0, min(x0, W - 1))
    y0 = max(0, min(y0, H - 1))
    x1 = max(x0 + 1, min(x1, W))
    y1 = max(y0 + 1, min(y1, H))
    return x0, y0, x1, y1


def bbox_at_time(df, t_idx: float):
    """
    只在 x 方向做线性插值，y/w/h 固定使用前一秒 (r0) 的值。
    t_idx: 以 “秒” 为单位的连续索引（0,1,2,...）
    """
    N = len(df)
    t_idx = float(np.clip(t_idx, 0.0, N - 1.0))

    if INTERP_MODE == "step" or N == 1:
        k = int(round(t_idx))
        r = df.iloc[k]
        return float(r[X_COL]), float(r[Y_COL]), float(r[W_COL]), float(r[H_COL]), k, 0.0

    if t_idx >= N - 1:
        r = df.iloc[N - 1]
        return float(r[X_COL]), float(r[Y_COL]), float(r[W_COL]), float(r[H_COL]), N - 1, 0.0

    k = int(np.floor(t_idx))
    u = t_idx - k
    r0 = df.iloc[k]
    r1 = df.iloc[k + 1]

    x = (1.0 - u) * float(r0[X_COL]) + u * float(r1[X_COL])
    y = float(r0[Y_COL])
    w = float(r0[W_COL])
    h = float(r0[H_COL])

    return x, y, w, h, k, u


def to_gray01(frame):
    # frame from imageio is RGB uint8
    if frame.ndim == 2:
        g = frame.astype(np.float32)
    else:
        r = frame[..., 0].astype(np.float32)
        gg = frame[..., 1].astype(np.float32)
        b = frame[..., 2].astype(np.float32)
        g = 0.2126 * r + 0.7152 * gg + 0.0722 * b
    return g / 255.0


def metrics_patch(p01):
    """
    输入：float32 灰度 [0,1] patch
    输出：mean/std/RMS_contrast/michelson/lap_var
    """
    m = float(p01.mean())
    s = float(p01.std())
    mn = float(p01.min())
    mx = float(p01.max())

    rms_contrast = s / (m + EPS)
    michelson = (mx - mn) / (mx + mn + EPS)

    p_u8 = np.clip(p01 * 255.0 + 0.5, 0, 255).astype(np.uint8)
    lap = cv2.Laplacian(p_u8, cv2.CV_32F, ksize=3)
    lap_var = float(lap.var())

    return {
        "mean": m,
        "std": s,
        "RMS_contrast": rms_contrast,
        "michelson": michelson,
        "lap_var": lap_var,
    }


def add_prefixed_metrics(out_dict: dict, prefix: str, met: dict):
    """把 metrics 字典加上前缀写入行里，例如 ROI_mean / FULL_mean"""
    for k, v in met.items():
        out_dict[f"{prefix}_{k}"] = v


def analyze_one_video(name, video_path, df_bbox=None):
    reader = imageio.get_reader(video_path)
    first = reader.get_data(0)
    H, W = first.shape[0], first.shape[1]
    n_frames = reader.count_frames()

    # ROI scale（若有 ROI）
    sx = sy = 1.0
    has_roi = (df_bbox is not None) and (len(df_bbox) > 0)

    if has_roi:
        rtW = float(df_bbox.iloc[0][RTW_COL])
        rtH = float(df_bbox.iloc[0][RTH_COL])
        sx = W / rtW if rtW > 1 else 1.0
        sy = H / rtH if rtH > 1 else 1.0

    dbg_writer = None
    if MAKE_DEBUG_VIDEO and has_roi:
        dbg_path = os.path.join(OUT_DIR, f"{name}_ROIbox_track.mp4")
        dbg_writer = imageio.get_writer(dbg_path, fps=FPS, codec="libx264", quality=9)
        print("Debug video:", dbg_path)

    rows = []
    for i in range(n_frames):
        frame = reader.get_data(i)
        t_sec = i / FPS
        t_idx = t_sec / SECONDS_PER_STEP

        # --------- FULL metrics（全画面）---------
        g01 = to_gray01(frame)
        full_met = metrics_patch(g01)

        # --------- ROI box（仅用于ROI & debug）---------
        roi_met = None
        x0 = y0 = x1 = y1 = None
        k = 0
        u = 0.0

        if has_roi:
            x, y, w, h, k, u = bbox_at_time(df_bbox, t_idx)
            x0f = x * sx
            y0f = y * sy
            x1f = (x + w) * sx
            y1f = (y + h) * sy
            x0, y0, x1, y1 = clamp_box(x0f, y0f, x1f, y1f, W, H)

            patch = g01[y0:y1, x0:x1]
            roi_met = metrics_patch(patch)

            if dbg_writer is not None:
                rgb = frame.copy() if frame.ndim == 3 else np.stack([frame] * 3, axis=-1)
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.rectangle(bgr, (x0, y0), (x1, y1), BOX_COLOR_BGR, BOX_THICKNESS)
                txt = f"{name} t={t_sec:.2f}s idx={t_idx:.2f} k={k} u={u:.2f}"
                cv2.putText(bgr, txt, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3, cv2.LINE_AA)
                cv2.putText(bgr, txt, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 1, cv2.LINE_AA)
                dbg_writer.append_data(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        # --------- 分析窗外不记入统计（但 debug 依旧输出）---------
        if (t_sec < ANALYSIS_START) or (t_sec >= ANALYSIS_END):
            continue

        t_rel = t_sec - ANALYSIS_START

        row = {
            "video": name,
            "frame_idx": i,
            "time_sec": t_rel,        # 0–5 s（相对时间）
            "time_orig_sec": t_sec,   # 1–6 s（原始时间）
            "t_idx": t_idx,
            "video_W": W,
            "video_H": H,
        }

        # 写入 FULL 指标
        add_prefixed_metrics(row, "FULL", full_met)

        # 写入 ROI 指标（若有）
        if has_roi and roi_met is not None:
            add_prefixed_metrics(row, "ROI", roi_met)
            row.update({
                "k": k, "u": u,
                "roi_x0": x0, "roi_y0": y0, "roi_x1": x1, "roi_y1": y1,
                "scale_x": sx, "scale_y": sy,
            })
        else:
            # 没有ROI时也保留字段（方便统一处理）
            row.update({"k": np.nan, "u": np.nan,
                        "roi_x0": np.nan, "roi_y0": np.nan, "roi_x1": np.nan, "roi_y1": np.nan,
                        "scale_x": np.nan, "scale_y": np.nan})
            for key in ["mean", "std", "RMS_contrast", "michelson", "lap_var"]:
                row[f"ROI_{key}"] = np.nan

        rows.append(row)

    reader.close()
    if dbg_writer is not None:
        dbg_writer.close()

    return pd.DataFrame(rows)


def plot_metric(combo, metric_key):
    plt.figure(figsize=(9, 6), dpi=200)

    for name in combo["video"].unique():
        d = combo[combo["video"] == name]

        # FULL (保持默认颜色)
        plt.plot(d["time_sec"], d[f"FULL_{metric_key}"], label=f"{name}-FULL", linewidth=1.6)

        # ROI：linear -> blue, gauss -> green
        if d[f"ROI_{metric_key}"].notna().any():
            lname = name.lower()
            if "linear" in lname:
                roi_color = "blue"
            elif "gauss" in lname or "gaussian" in lname:
                roi_color = "green"
            else:
                roi_color = None

            if roi_color:
                plt.plot(d["time_sec"], d[f"ROI_{metric_key}"], label=f"{name}-ROI", linewidth=1.6, color=roi_color)
            else:
                plt.plot(d["time_sec"], d[f"ROI_{metric_key}"], label=f"{name}-ROI", linewidth=1.6)

    plt.xlabel("time t (s)", fontsize=LABEL_FS)
    plt.ylabel(metric_key, fontsize=LABEL_FS)
    plt.xlim(0, N_PERIODS)
    plt.legend()
    plt.grid(True)

    out_png = os.path.join(OUT_DIR, f"{metric_key}_ROI_vs_FULL.png")
    plt.savefig(out_png, dpi=200)
    plt.close()
    print("Saved plot:", out_png)


def plot_metric_one_period(combo, metric_key, T=1.0):
    plt.figure(figsize=(9, 6), dpi=200)
    sub = combo[(combo["time_sec"] >= 0.0) & (combo["time_sec"] <= T)].copy()

    for name in sub["video"].unique():
        d = sub[sub["video"] == name]
        plt.plot(d["time_sec"], d[f"FULL_{metric_key}"], label=f"{name}-FULL", linewidth=1.6)

        if d[f"ROI_{metric_key}"].notna().any():
            lname = name.lower()
            if "linear" in lname:
                roi_color = "blue"
            elif "gauss" in lname or "gaussian" in lname:
                roi_color = "green"
            else:
                roi_color = None

            if roi_color:
                plt.plot(d["time_sec"], d[f"ROI_{metric_key}"], label=f"{name}-ROI", linewidth=1.6, color=roi_color)
            else:
                plt.plot(d["time_sec"], d[f"ROI_{metric_key}"], label=f"{name}-ROI", linewidth=1.6)

    plt.xlabel("time t (s)", fontsize=LABEL_FS)
    plt.ylabel(metric_key, fontsize=LABEL_FS)
    plt.title(f"First period (0–{T:.0f}s): {metric_key}")
    plt.xlim(0, T)
    plt.legend()
    plt.grid(True)

    out_png = os.path.join(OUT_DIR, f"{metric_key}_1period_ROI_vs_FULL.png")
    plt.savefig(out_png, dpi=200)
    plt.close()
    print("Saved 1-period plot:", out_png)


def main():
    # 读取 ROI CSV（可选）
    df_bbox = None
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        if VALID_COL in df.columns:
            df = df[df[VALID_COL] == 1].copy()
        if IDX_COL in df.columns:
            df = df.sort_values(IDX_COL).reset_index(drop=True)
        df_bbox = df
        print("Loaded ROI CSV:", CSV_PATH, "rows=", len(df_bbox))
    else:
        print("ROI CSV not found, will compute FULL only:", CSV_PATH)

    all_df = []
    for name, vp in VIDEOS.items():
        print("Analyzing:", name, vp)
        out = analyze_one_video(name, vp, df_bbox=df_bbox)
        out_csv = os.path.join(OUT_DIR, f"track_metrics_{name}.csv")
        out.to_csv(out_csv, index=False)
        print("Saved:", out_csv)
        all_df.append(out)

    if len(all_df) == 0:
        raise RuntimeError("No videos in VIDEOS. Please add at least one video path.")

    combo = pd.concat(all_df, ignore_index=True)
    combo_csv = os.path.join(OUT_DIR, "track_metrics_ALL.csv")
    combo.to_csv(combo_csv, index=False)
    print("Saved:", combo_csv)

    for metric in ["lap_var", "RMS_contrast", "michelson", "mean"]:
        plot_metric(combo, metric)
        plot_metric_one_period(combo, metric, T=1.0)

    print("DONE. OUT_DIR:", OUT_DIR)


if __name__ == "__main__":
    main()
