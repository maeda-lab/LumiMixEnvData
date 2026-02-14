import os
import math
import numpy as np
import pandas as pd
import cv2
import imageio.v2 as imageio
import matplotlib.pyplot as plt

# ======================
# CONFIG
# ======================
CSV_PATH = r"D:\vectionProject\public\camera2images\cam2_tree_bbox.csv"

VIDEOS = {
    # 这 5 个就是你要分析的条件（路径按你自己的文件名改一下）
    "cam2_linear": r"D:\vectionProject\public\camera2images\cam2_linear.mp4",
    "cam2_gauss_sigma0p6": r"D:\vectionProject\public\camera2images\cam2_gauss_sigma0p6.mp4",
    "C3_gauss_EDGE_injected_spikes_ROI_feather": r"D:\vectionProject\public\camera2images\C3_gauss_EDGE_injected_spikes_ROI_feather.mp4",
    "C4_linear_ROI_suppressed_softknee": r"D:\vectionProject\public\camera2images\C4_linear_ROI_suppressed_softknee.mp4",
    "C5_linear_ROI_energyMatchedToGauss": r"D:\vectionProject\public\camera2images\C5_linear_ROI_energyMatchedToGauss.mp4",
}

OUT_DIR = r"D:\vectionProject\public\camera2images\track_contrast_out_5conds_with_weights"
os.makedirs(OUT_DIR, exist_ok=True)

FPS = 60
SECONDS_PER_STEP = 1.0
INTERP_MODE = "lerp"  # bbox插值：lerp/step

# ---- 权重图相关（按你的生成逻辑设置）----
GAUSS_SIGMA_SEC = 0.6  # sigma0p6 -> 0.6秒（因为step=1秒）
MAX_WEIGHT_CURVES = 12  # 每张权重图最多画多少条曲线（太长会糊）

# 每个条件用哪种权重模型画“权重图”
WEIGHT_PROFILE = {
    "cam2_linear": "linear",
    "cam2_gauss_sigma0p6": "gauss",
    "C3_gauss_EDGE_injected_spikes_ROI_feather": "gauss",
    "C4_linear_ROI_suppressed_softknee": "linear",
    "C5_linear_ROI_energyMatchedToGauss": "linear",
}

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

# 输出带框验证视频（建议 True）
MAKE_DEBUG_VIDEO = True
BOX_COLOR_BGR = (0, 255, 0)
BOX_THICKNESS = 4


# ======================
# ROI utils
# ======================
def clamp_box(x0, y0, x1, y1, W, H):
    x0 = int(round(x0)); y0 = int(round(y0)); x1 = int(round(x1)); y1 = int(round(y1))
    x0 = max(0, min(x0, W - 1))
    y0 = max(0, min(y0, H - 1))
    x1 = max(x0 + 1, min(x1, W))
    y1 = max(y0 + 1, min(y1, H))
    return x0, y0, x1, y1


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

    x = (1 - u) * float(r0[X_COL]) + u * float(r1[X_COL])
    y = (1 - u) * float(r0[Y_COL]) + u * float(r1[Y_COL])
    w = (1 - u) * float(r0[W_COL]) + u * float(r1[W_COL])
    h = (1 - u) * float(r0[H_COL]) + u * float(r1[H_COL])
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
        "lap_var": lap_var
    }


# ======================
# Weight model (for plots)
# ======================
def infer_n_keyframes_from_video(n_frames: int):
    # 1秒一个关键帧：keyframe数大概等于 duration/step + 1
    duration_sec = n_frames / float(FPS)
    n_keys = int(math.floor(duration_sec / SECONDS_PER_STEP + 1e-6)) + 1
    return max(2, n_keys)


def compute_weights_linear(times_sec: np.ndarray, n_keys: int):
    """
    linear cross-dissolve:
      t in [k, k+1): w_k=1-u, w_{k+1}=u
    返回 W: [T, n_keys]
    """
    t_idx = times_sec / SECONDS_PER_STEP
    k = np.floor(t_idx).astype(int)
    u = t_idx - k

    k = np.clip(k, 0, n_keys - 2)
    u = np.clip(u, 0.0, 1.0)

    W = np.zeros((len(times_sec), n_keys), dtype=np.float32)
    W[np.arange(len(times_sec)), k] = (1.0 - u).astype(np.float32)
    W[np.arange(len(times_sec)), k + 1] = u.astype(np.float32)
    return W


def compute_weights_gauss(times_sec: np.ndarray, n_keys: int, sigma_sec: float):
    """
    gaussian temporal weights:
      w_j(t) = exp(-0.5 * ((t - j*step)/sigma)^2 ), normalized over j
    返回 W: [T, n_keys]
    """
    step = SECONDS_PER_STEP
    centers = (np.arange(n_keys, dtype=np.float32) * step)[None, :]  # [1, n_keys]
    t = times_sec[:, None].astype(np.float32)                        # [T, 1]
    z = (t - centers) / float(max(1e-6, sigma_sec))
    W = np.exp(-0.5 * (z ** 2)).astype(np.float32)
    s = W.sum(axis=1, keepdims=True) + 1e-12
    W = W / s
    return W


def pick_curve_indices(n_keys: int, max_curves: int):
    if n_keys <= max_curves:
        return list(range(n_keys))
    step = int(math.ceil(n_keys / float(max_curves)))
    idx = list(range(0, n_keys, step))
    if idx[-1] != n_keys - 1:
        idx.append(n_keys - 1)
    return idx[:max_curves]


def save_weight_csv_and_plot(cond_name: str, mode: str, n_frames: int):
    times = np.arange(n_frames, dtype=np.float32) / float(FPS)
    n_keys = infer_n_keyframes_from_video(n_frames)

    if mode == "linear":
        W = compute_weights_linear(times, n_keys)
    elif mode == "gauss":
        W = compute_weights_gauss(times, n_keys, GAUSS_SIGMA_SEC)
    else:
        raise ValueError(f"Unknown weight mode: {mode}")

    # --- Save CSV (只保存被选中曲线 + 总和检查) ---
    idx_sel = pick_curve_indices(n_keys, MAX_WEIGHT_CURVES)
    out_csv = os.path.join(OUT_DIR, f"weights_{cond_name}_{mode}.csv")

    df_out = pd.DataFrame({"time_sec": times})
    for j in idx_sel:
        df_out[f"w_{j:03d}"] = W[:, j]
    df_out["w_sum"] = W.sum(axis=1)
    df_out.to_csv(out_csv, index=False)
    print("Saved weight CSV:", out_csv)

    # --- Plot ---
    plt.figure(figsize=(10, 4))
    for j in idx_sel:
        plt.plot(times, W[:, j], label=f"k{j}")
    plt.xlabel("time (sec)")
    plt.ylabel("weight")
    plt.title(f"Weights ({cond_name}) mode={mode}  keys={n_keys}  step={SECONDS_PER_STEP}s")
    plt.grid(True)
    # legend太挤就不显示（你要就打开）
    # plt.legend(ncol=6, fontsize=8)
    out_png = os.path.join(OUT_DIR, f"weights_{cond_name}_{mode}.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print("Saved weight plot:", out_png)


# ======================
# Video analysis
# ======================
def analyze_one_video(name, video_path, df_bbox):
    reader = imageio.get_reader(video_path)
    first = reader.get_data(0)
    H, W = first.shape[0], first.shape[1]
    n_frames = reader.count_frames()

    rtW = float(df_bbox.iloc[0][RTW_COL])
    rtH = float(df_bbox.iloc[0][RTH_COL])
    sx = W / rtW if rtW > 1 else 1.0
    sy = H / rtH if rtH > 1 else 1.0

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
        x0 = x * sx; y0 = y * sy; x1 = (x + w) * sx; y1 = (y + h) * sy
        x0, y0, x1, y1 = clamp_box(x0, y0, x1, y1, W, H)

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
            "video_W": W, "video_H": H, "scale_x": sx, "scale_y": sy
        })
        rows.append(ms)

        if dbg_writer is not None:
            rgb = frame.copy() if frame.ndim == 3 else np.stack([frame] * 3, axis=-1)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.rectangle(bgr, (x0, y0), (x1, y1), BOX_COLOR_BGR, BOX_THICKNESS)
            txt = f"{name} t={t_sec:.2f}s idx={t_idx:.2f} k={k} u={u:.2f}"
            cv2.putText(bgr, txt, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(bgr, txt, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 1, cv2.LINE_AA)
            rgb2 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            dbg_writer.append_data(rgb2)

    reader.close()
    if dbg_writer is not None:
        dbg_writer.close()

    return pd.DataFrame(rows), n_frames


def plot_metric(combo, metric):
    plt.figure(figsize=(10,4))

    # ✅ 固定绘图顺序：先画别的，最后画 C4（红）避免被盖住
    order = [
        "cam2_linear",
        "cam2_gauss_sigma0p6",
        "C3_gauss_EDGE_injected_spikes_ROI_feather",
        "C5_linear_ROI_energyMatchedToGauss",
        "C4_linear_ROI_suppressed_softknee",  # <- 最后画它
    ]
    # 如果 combo 里名字不完全一致，就用实际存在的
    order = [n for n in order if n in set(combo["video"].unique())]
    rest = [n for n in combo["video"].unique() if n not in order]
    plot_names = order + rest

    for name in plot_names:
        d = combo[combo["video"] == name]

        # ✅ 让 C4 更“显眼”：更粗、虚线、在最上层
        if name == "C4_linear_ROI_suppressed_softknee":
            plt.plot(d["time_sec"], d[metric], label=name,
                     linewidth=3.0, linestyle="--", zorder=10)
        # ✅ 让 C5 稍微透明一点（不改颜色，只改透明度）
        elif name == "C5_linear_ROI_energyMatchedToGauss":
            plt.plot(d["time_sec"], d[metric], label=name,
                     alpha=0.75, zorder=5)
        else:
            plt.plot(d["time_sec"], d[metric], label=name, zorder=3)

    plt.xlabel("time (sec)")
    plt.ylabel(metric)
    plt.title(f"Tracking ROI metric: {metric} (interp={INTERP_MODE})")
    plt.legend()
    plt.grid(True)
    out_png = os.path.join(OUT_DIR, f"{metric}.png")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
    print("Saved plot:", out_png)



def main():
    # load bbox csv
    df = pd.read_csv(CSV_PATH)
    if VALID_COL in df.columns:
        df = df[df[VALID_COL] == 1].copy()
    df = df.sort_values(IDX_COL).reset_index(drop=True)

    all_df = []

    for name, vp in VIDEOS.items():
        print("\n======================")
        print("Analyzing:", name)
        print("Video:", vp)

        out, n_frames = analyze_one_video(name, vp, df)
        out_csv = os.path.join(OUT_DIR, f"track_metrics_{name}.csv")
        out.to_csv(out_csv, index=False)
        print("Saved:", out_csv)
        all_df.append(out)

        # --- weight figure + csv for this condition ---
        mode = WEIGHT_PROFILE.get(name, "linear")
        save_weight_csv_and_plot(name, mode, n_frames)

    combo = pd.concat(all_df, ignore_index=True)
    combo_csv = os.path.join(OUT_DIR, "track_metrics_ALL.csv")
    combo.to_csv(combo_csv, index=False)
    print("\nSaved:", combo_csv)

    for metric in ["lap_var", "rms_contrast", "michelson", "mean"]:
        plot_metric(combo, metric)

    print("\nDONE. OUT_DIR:", OUT_DIR)


if __name__ == "__main__":
    main()
