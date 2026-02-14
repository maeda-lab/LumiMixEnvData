import os
import numpy as np
import pandas as pd
import cv2
import imageio.v2 as imageio
import matplotlib.pyplot as plt

# ======================
# CONFIG
# ======================
CSV_PATH = r"D:\vectionProject\public\camera2images\cam2_tree_bbox.csv"

VIDEO_LINEAR = r"D:\vectionProject\public\camera2images\cam2_linear.mp4"
VIDEO_GAUSS  = r"D:\vectionProject\public\camera2images\cam2_gauss_sigma0p6.mp4"

OUT_DIR = r"D:\vectionProject\public\camera2images\track_contrast_out_linear_vs_gauss_1s_10s"
os.makedirs(OUT_DIR, exist_ok=True)

# 实验时间轴
FPS = 60
SECONDS_PER_STEP = 1.0
INTERP_MODE = "lerp"

# 输出的时间窗口（秒）
TIME_WINDOWS_SEC = [1.0, 10.0]

# 强制用 60fps（推荐：与你实验定义一致）
FORCE_FPS_60 = True

# CSV columns（保持和你原代码一致）
IDX_COL="secIndex"; VALID_COL="valid"
RTW_COL="rtW"; RTH_COL="rtH"
X_COL="x_tl"; Y_COL="y_tl"; W_COL="w_tl"; H_COL="h_tl"

EPS = 1e-8

# 输出带框验证视频
MAKE_DEBUG_VIDEO = True
BOX_COLOR_BGR = (0, 255, 0)
BOX_THICKNESS = 4

# ======================
def clamp_box(x0,y0,x1,y1,W,H):
    x0=int(round(x0)); y0=int(round(y0)); x1=int(round(x1)); y1=int(round(y1))
    x0=max(0,min(x0,W-1)); y0=max(0,min(y0,H-1))
    x1=max(x0+1,min(x1,W)); y1=max(y0+1,min(y1,H))
    return x0,y0,x1,y1

def bbox_at_time(df, t_idx: float):
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

    x = (1-u)*float(r0[X_COL]) + u*float(r1[X_COL])
    y = (1-u)*float(r0[Y_COL]) + u*float(r1[Y_COL])
    w = (1-u)*float(r0[W_COL]) + u*float(r1[W_COL])
    h = (1-u)*float(r0[H_COL]) + u*float(r1[H_COL])
    return x, y, w, h, k, u

def to_gray01(frame):
    # imageio -> RGB uint8
    if frame.ndim == 2:
        g = frame.astype(np.float32)
    else:
        r = frame[...,0].astype(np.float32)
        gg = frame[...,1].astype(np.float32)
        b = frame[...,2].astype(np.float32)
        g = 0.2126*r + 0.7152*gg + 0.0722*b
    return g / 255.0

def metrics_patch(p01):
    m = float(p01.mean())
    s = float(p01.std())
    rms_contrast = s / (m + EPS)

    p_u8 = np.clip(p01 * 255.0 + 0.5, 0, 255).astype(np.uint8)
    lap = cv2.Laplacian(p_u8, cv2.CV_32F, ksize=3)
    lap_var = float(lap.var())

    return {"mean": m, "std": s, "rms_contrast": rms_contrast, "lap_var": lap_var}

def get_reader_fps(reader, default_fps=60.0):
    if FORCE_FPS_60:
        return float(default_fps)
    try:
        meta = reader.get_meta_data()
        fps_real = float(meta.get("fps", default_fps))
        if not np.isfinite(fps_real) or fps_real <= 0:
            fps_real = float(default_fps)
        return fps_real
    except Exception:
        return float(default_fps)

def analyze_one_video(name, video_path, df_bbox, time_limit_sec):
    reader = imageio.get_reader(video_path)
    first = reader.get_data(0)
    H, W = first.shape[0], first.shape[1]

    fps_used = get_reader_fps(reader, default_fps=FPS)
    n_limit = int(round(fps_used * time_limit_sec))

    rtW = float(df_bbox.iloc[0][RTW_COL])
    rtH = float(df_bbox.iloc[0][RTH_COL])
    sx = W/rtW if rtW > 1 else 1.0
    sy = H/rtH if rtH > 1 else 1.0

    dbg_writer = None
    if MAKE_DEBUG_VIDEO:
        dbg_path = os.path.join(OUT_DIR, f"{name}_ROIbox_track_0_{int(time_limit_sec)}s.mp4")
        dbg_writer = imageio.get_writer(dbg_path, fps=fps_used, codec="libx264", quality=9)
        print("Debug video:", dbg_path)

    rows = []
    for i in range(n_limit):
        try:
            frame = reader.get_data(i)
        except Exception:
            break

        t_sec = i / fps_used
        t_idx = t_sec / SECONDS_PER_STEP

        x, y, w, h, k, u = bbox_at_time(df_bbox, t_idx)
        x0 = x*sx; y0 = y*sy; x1 = (x+w)*sx; y1 = (y+h)*sy
        x0,y0,x1,y1 = clamp_box(x0,y0,x1,y1,W,H)

        g01 = to_gray01(frame)
        patch = g01[y0:y1, x0:x1]
        ms = metrics_patch(patch)

        ms.update({
            "video": name,
            "frame_idx": i,
            "time_sec": t_sec,
            "t_idx": t_idx,
            "k": k,
            "u": u,
            "roi_x0": x0, "roi_y0": y0, "roi_x1": x1, "roi_y1": y1,
            "video_W": W, "video_H": H, "scale_x": sx, "scale_y": sy,
            "fps_used": fps_used,
            "time_window_sec": time_limit_sec
        })
        rows.append(ms)

        if dbg_writer is not None:
            rgb = frame.copy() if frame.ndim == 3 else np.stack([frame]*3, axis=-1)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.rectangle(bgr, (x0,y0), (x1,y1), BOX_COLOR_BGR, BOX_THICKNESS)
            txt = f"{name} t={t_sec:.3f}s idx={t_idx:.2f}"
            cv2.putText(bgr, txt, (15,35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 3, cv2.LINE_AA)
            cv2.putText(bgr, txt, (15,35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 1, cv2.LINE_AA)
            rgb2 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            dbg_writer.append_data(rgb2)

    reader.close()
    if dbg_writer is not None:
        dbg_writer.close()

    return pd.DataFrame(rows)

def plot_metric(combo, metric, time_limit_sec):
    plt.figure()
    for name in combo["video"].unique():
        d = combo[combo["video"] == name]
        plt.plot(d["time_sec"], d[metric], label=name)

    plt.xlabel("time (sec)")
    plt.ylabel(metric)
    plt.title(f"{metric} (0-{int(time_limit_sec)}s, interp={INTERP_MODE})")
    plt.legend()
    plt.grid(True)
    plt.xlim(0, time_limit_sec)

    out_png = os.path.join(OUT_DIR, f"{metric}_0_{int(time_limit_sec)}s_linear_vs_gauss.png")
    plt.savefig(out_png, dpi=200)
    plt.close()
    print("Saved plot:", out_png)

def main():
    df = pd.read_csv(CSV_PATH)
    if VALID_COL in df.columns:
        df = df[df[VALID_COL] == 1].copy()
    df = df.sort_values(IDX_COL).reset_index(drop=True)

    for T in TIME_WINDOWS_SEC:
        print("")
        print("===== TIME WINDOW: 0-" + str(int(T)) + "s =====")

        print("Analyzing: linear", VIDEO_LINEAR)
        df_lin = analyze_one_video("linear", VIDEO_LINEAR, df, T)
        df_lin.to_csv(os.path.join(OUT_DIR, f"track_metrics_linear_0_{int(T)}s.csv"), index=False)

        print("Analyzing: gauss", VIDEO_GAUSS)
        df_gau = analyze_one_video("gauss", VIDEO_GAUSS, df, T)
        df_gau.to_csv(os.path.join(OUT_DIR, f"track_metrics_gauss_0_{int(T)}s.csv"), index=False)

        combo = pd.concat([df_lin, df_gau], ignore_index=True)
        combo.to_csv(os.path.join(OUT_DIR, f"track_metrics_LINEAR_GAUSS_0_{int(T)}s.csv"), index=False)

        plot_metric(combo, "lap_var", T)
        plot_metric(combo, "rms_contrast", T)

    print("")
    print("DONE. OUT_DIR:", OUT_DIR)

if __name__ == "__main__":
    main()
