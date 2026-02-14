import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

# ======================
# CONFIG
# ======================
CSV_PATH = r"D:\vectionProject\public\camear1images\cam1_tree_bbox.csv"

VIDEOS = {
    "linear":   r"D:\vectionProject\public\camear1images\cam1_linear.mp4",
    "tri_norm": r"D:\vectionProject\public\camear1images\cam1_tri_norm_r1p5.mp4",
    "tri_raw":  r"D:\vectionProject\public\camear1images\cam1_tri_raw_r1p5.mp4",
    "gauss":    r"D:\vectionProject\public\camear1images\cam1_gauss_sigma0p6.mp4",  # 不要就删掉
}

OUT_DIR = r"D:\vectionProject\public\camear1images\local_contrast_track_out"
os.makedirs(OUT_DIR, exist_ok=True)

FPS = 60
SECONDS_PER_STEP = 1.0   # ✅ 你生成视频时：1秒对应下一张原始图片 cam1_000->cam1_001

# bbox 时间插值：
#   "lerp" 推荐（框平滑移动）
#   "step" 会一秒一跳
INTERP_MODE = "lerp"

# 是否输出带框验证视频（建议 True，先确认框真的跟树走）
MAKE_DEBUG_VIDEO = True

# CSV字段（按你截图）
IDX_COL = "secIndex"
VALID_COL = "valid"
RTW_COL, RTH_COL = "rtW", "rtH"
X_COL, Y_COL, W_COL, H_COL = "x_tl", "y_tl", "w_tl", "h_tl"

EPS = 1e-8

# debug video box style
BOX_COLOR_BGR = (0, 255, 0)
BOX_THICKNESS = 4

# ======================
# Utils: grayscale + metrics
# ======================
def to_gray01(frame_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 -> gray float32 0..1"""
    b = frame_bgr[..., 0].astype(np.float32)
    g = frame_bgr[..., 1].astype(np.float32)
    r = frame_bgr[..., 2].astype(np.float32)
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return y / 255.0

def metrics_patch(p01: np.ndarray) -> dict:
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
        "rms_contrast": rms_contrast,
        "michelson": michelson,
        "lap_var": lap_var,
    }

def clamp_box(x0, y0, x1, y1, W, H):
    x0 = int(round(x0)); y0 = int(round(y0))
    x1 = int(round(x1)); y1 = int(round(y1))
    x0 = max(0, min(x0, W - 1))
    y0 = max(0, min(y0, H - 1))
    x1 = max(x0 + 1, min(x1, W))
    y1 = max(y0 + 1, min(y1, H))
    return x0, y0, x1, y1

# ======================
# bbox over time (from CSV)
# ======================
def bbox_at_time(df: pd.DataFrame, t_idx: float):
    """
    t_idx: 0..(N-1) on original-image index axis.
    Returns (x, y, w, h, k, u) in CSV coordinate space.
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

    x = (1 - u) * float(r0[X_COL]) + u * float(r1[X_COL])
    y = (1 - u) * float(r0[Y_COL]) + u * float(r1[Y_COL])
    w = (1 - u) * float(r0[W_COL]) + u * float(r1[W_COL])
    h = (1 - u) * float(r0[H_COL]) + u * float(r1[H_COL])

    return x, y, w, h, k, u

# ======================
# Analyze one video (tracking ROI)
# ======================
def analyze_video(name: str, video_path: str, df_bbox: pd.DataFrame):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_in = float(cap.get(cv2.CAP_PROP_FPS))  # 不一定准，但我们用 FPS 做时间轴

    # CSV coordinate base
    rtW = float(df_bbox.iloc[0][RTW_COL])
    rtH = float(df_bbox.iloc[0][RTH_COL])
    sx = W / rtW if rtW > 1 else 1.0
    sy = H / rtH if rtH > 1 else 1.0

    print(f"[{name}] size={W}x{H} frames={n_frames} fps_in={fps_in:.3f}  scale=({sx:.5f},{sy:.5f})")

    # optional debug video
    debug_writer = None
    if MAKE_DEBUG_VIDEO:
        out_dbg = os.path.join(OUT_DIR, f"{name}_ROIbox_track.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        debug_writer = cv2.VideoWriter(out_dbg, fourcc, FPS, (W, H), True)
        if not debug_writer.isOpened():
            raise RuntimeError("Failed to create debug video writer.")
        print(f"[{name}] debug video => {out_dbg}")

    rows = []
    for i in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break

        t_sec = i / FPS
        t_idx = t_sec / SECONDS_PER_STEP

        x, y, w, h, k, u = bbox_at_time(df_bbox, t_idx)

        # scale bbox to video pixels
        x0 = x * sx
        y0 = y * sy
        x1 = (x + w) * sx
        y1 = (y + h) * sy
        x0, y0, x1, y1 = clamp_box(x0, y0, x1, y1, W, H)

        gray01 = to_gray01(frame)
        patch = gray01[y0:y1, x0:x1]

        ms = metrics_patch(patch)
        ms.update({
            "video": name,
            "frame_idx": i,
            "time_sec": t_sec,
            "t_idx": t_idx,
            "k": k,
            "u": u,
            "roi_x0": x0, "roi_y0": y0, "roi_x1": x1, "roi_y1": y1,
            "video_W": W, "video_H": H,
            "scale_x": sx, "scale_y": sy
        })
        rows.append(ms)

        if debug_writer is not None:
            cv2.rectangle(frame, (x0, y0), (x1, y1), BOX_COLOR_BGR, BOX_THICKNESS)
            txt = f"{name} t={t_sec:.2f}s t_idx={t_idx:.2f} k={k} u={u:.2f}"
            cv2.putText(frame, txt, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 3, cv2.LINE_AA)
            cv2.putText(frame, txt, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 1, cv2.LINE_AA)
            debug_writer.write(frame)

        if i % 600 == 0 and i > 0:
            print(f"[{name}] processed {i}/{n_frames}")

    cap.release()
    if debug_writer is not None:
        debug_writer.release()

    return pd.DataFrame(rows)

# ======================
# Main
# ======================
def main():
    df = pd.read_csv(CSV_PATH)

    # 只用 valid=1，并按 secIndex 排序
    if VALID_COL in df.columns:
        df = df[df[VALID_COL] == 1].copy()
    df = df.sort_values(IDX_COL).reset_index(drop=True)

    if len(df) < 2:
        raise RuntimeError("bbox 行数太少（<2）。检查 CSV 或 valid 列。")

    all_out = []
    for name, vp in VIDEOS.items():
        out = analyze_video(name, vp, df)
        out_csv = os.path.join(OUT_DIR, f"track_metrics_{name}.csv")
        out.to_csv(out_csv, index=False)
        print("Saved:", out_csv)
        all_out.append(out)

    combo = pd.concat(all_out, ignore_index=True)
    combo_csv = os.path.join(OUT_DIR, "track_metrics_ALL.csv")
    combo.to_csv(combo_csv, index=False)
    print("Saved:", combo_csv)

    # plots
    for metric in ["rms_contrast", "michelson", "lap_var", "mean"]:
        plt.figure()
        for name in VIDEOS.keys():
            d = combo[combo["video"] == name]
            plt.plot(d["time_sec"], d[metric], label=name)
        plt.xlabel("time (sec)")
        plt.ylabel(metric)
        plt.title(f"Tracking ROI metric: {metric}  (interp={INTERP_MODE})")
        plt.legend()
        plt.grid(True)
        out_png = os.path.join(OUT_DIR, f"{metric}.png")
        plt.savefig(out_png, dpi=200)
        plt.close()
        print("Saved plot:", out_png)

    print("Done.")

if __name__ == "__main__":
    main()
