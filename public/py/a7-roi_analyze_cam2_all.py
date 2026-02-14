import os
import numpy as np
import pandas as pd
import cv2
import imageio.v2 as imageio
import matplotlib.pyplot as plt

# ======================
# CONFIG
# ======================
CSV_PATH = r"D:\vectionProject\public\camera2images\cam2_tree_bbox.csv"  #  你的这个正确CSV

VIDEOS = {
    "linear":   r"D:\vectionProject\public\camera2images\cam2_linear.mp4",
    "tri_norm": r"D:\vectionProject\public\camera2images\cam2_tri_norm_r1p5.mp4",
    "tri_raw":  r"D:\vectionProject\public\camera2images\cam2_tri_raw_r1p5.mp4",
    "gauss":    r"D:\vectionProject\public\camera2images\cam2_gauss_sigma0p6.mp4",
     # ✅ 新增：边缘域操控（避免大片颜色变化）
    "linear_EDGE_suppress":  r"D:\vectionProject\public\camera2images\cam2_linear_EDGE_suppressed_spikes_softknee.mp4",
    "gauss_EDGE_inject": r"D:\vectionProject\public\camera2images\cam2_gauss_EDGE_injected_spikes_soft.mp4",
}


OUT_DIR = r"D:\vectionProject\public\camera2images\track_contrast_out_add_injected_suppressed_lap_var"
os.makedirs(OUT_DIR, exist_ok=True)

FPS = 60
SECONDS_PER_STEP = 1.0    # 你依然是一秒一张关键帧
INTERP_MODE = "lerp"

# CSV columns
IDX_COL="secIndex"; VALID_COL="valid"
RTW_COL="rtW"; RTH_COL="rtH"
X_COL="x_tl"; Y_COL="y_tl"; W_COL="w_tl"; H_COL="h_tl"

EPS = 1e-8

# 输出带框验证视频（强烈建议 True）
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
    # frame from imageio is RGB uint8
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
    mn = float(p01.min())
    mx = float(p01.max())

    rms_contrast = s / (m + EPS)
    michelson = (mx - mn) / (mx + mn + EPS)

    p_u8 = np.clip(p01 * 255.0 + 0.5, 0, 255).astype(np.uint8)
    lap = cv2.Laplacian(p_u8, cv2.CV_32F, ksize=3)
    lap_var = float(lap.var())

    return {"mean": m, "std": s, "rms_contrast": rms_contrast, "michelson": michelson, "lap_var": lap_var}

def analyze_one_video(name, video_path, df_bbox):
    reader = imageio.get_reader(video_path)
    first = reader.get_data(0)
    H, W = first.shape[0], first.shape[1]
    n_frames = reader.count_frames()

    rtW = float(df_bbox.iloc[0][RTW_COL])
    rtH = float(df_bbox.iloc[0][RTH_COL])
    sx = W/rtW if rtW > 1 else 1.0
    sy = H/rtH if rtH > 1 else 1.0

    dbg_writer = None
    if MAKE_DEBUG_VIDEO:
        dbg_path = os.path.join(OUT_DIR, f"{name}_ROIbox_track.mp4")
        dbg_writer = imageio.get_writer(dbg_path, fps=FPS, codec="libx264", quality=9)
        print("Debug video:", dbg_path)

    rows = []
    for i in range(n_frames):
        frame = reader.get_data(i)
        t_sec = i / FPS
        t_idx = t_sec / SECONDS_PER_STEP

        x, y, w, h, k, u = bbox_at_time(df_bbox, t_idx)
        x0 = x*sx; y0 = y*sy; x1 = (x+w)*sx; y1 = (y+h)*sy
        x0,y0,x1,y1 = clamp_box(x0,y0,x1,y1,W,H)

        g01 = to_gray01(frame)
        patch = g01[y0:y1, x0:x1]
        ms = metrics_patch(patch)

        ms.update({
            "video": name, "frame_idx": i, "time_sec": t_sec, "t_idx": t_idx, "k": k, "u": u,
            "roi_x0": x0, "roi_y0": y0, "roi_x1": x1, "roi_y1": y1,
            "video_W": W, "video_H": H, "scale_x": sx, "scale_y": sy
        })
        rows.append(ms)

        if dbg_writer is not None:
            rgb = frame.copy() if frame.ndim == 3 else np.stack([frame]*3, axis=-1)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.rectangle(bgr, (x0,y0), (x1,y1), BOX_COLOR_BGR, BOX_THICKNESS)
            txt = f"{name} t={t_sec:.2f}s idx={t_idx:.2f} k={k} u={u:.2f}"
            cv2.putText(bgr, txt, (15,35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 3, cv2.LINE_AA)
            cv2.putText(bgr, txt, (15,35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 1, cv2.LINE_AA)
            rgb2 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            dbg_writer.append_data(rgb2)

    reader.close()
    if dbg_writer is not None:
        dbg_writer.close()

    return pd.DataFrame(rows)

def plot_metric(combo, metric):
    plt.figure()
    for name in combo["video"].unique():
        d = combo[combo["video"] == name]
        plt.plot(d["time_sec"], d[metric], label=name)
    plt.xlabel("time (sec)")
    plt.ylabel(metric)
    plt.title(f"Tracking ROI metric: {metric} (interp={INTERP_MODE})")
    plt.legend()
    plt.grid(True)
    out_png = os.path.join(OUT_DIR, f"{metric}.png")
    plt.savefig(out_png, dpi=200)
    plt.close()
    print("Saved plot:", out_png)

def main():
    df = pd.read_csv(CSV_PATH)
    if VALID_COL in df.columns:
        df = df[df[VALID_COL] == 1].copy()
    df = df.sort_values(IDX_COL).reset_index(drop=True)

    all_df = []
    for name, vp in VIDEOS.items():
        print("Analyzing:", name, vp)
        out = analyze_one_video(name, vp, df)
        out_csv = os.path.join(OUT_DIR, f"track_metrics_{name}.csv")
        out.to_csv(out_csv, index=False)
        print("Saved:", out_csv)
        all_df.append(out)

    combo = pd.concat(all_df, ignore_index=True)
    combo_csv = os.path.join(OUT_DIR, "track_metrics_ALL.csv")
    combo.to_csv(combo_csv, index=False)
    print("Saved:", combo_csv)

    for metric in ["lap_var", "rms_contrast", "michelson", "mean"]:
        plot_metric(combo, metric)

    print("DONE. OUT_DIR:", OUT_DIR)

if __name__ == "__main__":
    main()
