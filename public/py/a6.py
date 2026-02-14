import os
import pandas as pd
import cv2
import numpy as np

# ======================
# CONFIG
# ======================
CSV_PATH = r"D:\vectionProject\public\camear1images\cam1_tree_bbox.csv"

VIDEO_IN  = r"D:\vectionProject\public\camear1images\cam1_linear.mp4"
VIDEO_OUT = r"D:\vectionProject\public\camear1images\cam1_linear_ROIbox_TRACK.mp4"

FPS_OUT = 60
SECONDS_PER_STEP = 1.0   # 你生成视频时：1秒对应下一张原始图片（cam1_000 -> cam1_001）

# CSV字段（按你截图）
X_COL, Y_COL, W_COL, H_COL = "x_tl", "y_tl", "w_tl", "h_tl"
VALID_COL = "valid"
RTW_COL, RTH_COL = "rtW", "rtH"
IDX_COL = "secIndex"

# 框外观
BOX_THICKNESS = 4
BOX_COLOR_BGR = (0, 255, 0)
DRAW_LABEL = True

# bbox 时间插值方式：
#   "step" : 直接用最近的关键帧bbox（会一秒一跳）
#   "lerp" : 在 k 和 k+1 之间线性插值（推荐，更平滑）
INTERP_MODE = "lerp"

# ======================
def clamp_roi(x0, y0, x1, y1, W, H):
    x0 = int(round(x0)); y0 = int(round(y0))
    x1 = int(round(x1)); y1 = int(round(y1))
    x0 = max(0, min(x0, W - 1))
    y0 = max(0, min(y0, H - 1))
    x1 = max(x0 + 1, min(x1, W))
    y1 = max(y0 + 1, min(y1, H))
    return x0, y0, x1, y1

def bbox_at_time(df, t_idx: float):
    """
    t_idx: 0..(N-1) （以“原始图片序号”为时间轴）
    返回 bbox: (x, y, w, h) in CSV coordinate space
    """
    N = len(df)
    t_idx = float(np.clip(t_idx, 0.0, N - 1.0))

    if INTERP_MODE == "step" or N == 1:
        k = int(round(t_idx))
        row = df.iloc[k]
        return float(row[X_COL]), float(row[Y_COL]), float(row[W_COL]), float(row[H_COL]), k, 0.0

    # lerp
    if t_idx >= N - 1:
        row = df.iloc[N - 1]
        return float(row[X_COL]), float(row[Y_COL]), float(row[W_COL]), float(row[H_COL]), N - 1, 0.0

    k = int(np.floor(t_idx))
    u = t_idx - k
    r0 = df.iloc[k]
    r1 = df.iloc[k + 1]

    x = (1 - u) * float(r0[X_COL]) + u * float(r1[X_COL])
    y = (1 - u) * float(r0[Y_COL]) + u * float(r1[Y_COL])
    w = (1 - u) * float(r0[W_COL]) + u * float(r1[W_COL])
    h = (1 - u) * float(r0[H_COL]) + u * float(r1[H_COL])
    return x, y, w, h, k, u

def main():
    df = pd.read_csv(CSV_PATH)

    # 只取 valid=1 的行，并按 secIndex 排序
    if VALID_COL in df.columns:
        df = df[df[VALID_COL] == 1].copy()
    df = df.sort_values(IDX_COL).reset_index(drop=True)

    if df.empty:
        raise RuntimeError("CSV 中没有可用(valid=1)的 bbox 行。")

    # CSV坐标基准分辨率
    rtW = float(df.iloc[0][RTW_COL])
    rtH = float(df.iloc[0][RTH_COL])

    cap = cv2.VideoCapture(VIDEO_IN)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频: {VIDEO_IN}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = float(cap.get(cv2.CAP_PROP_FPS))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    sx = W / rtW if rtW > 1 else 1.0
    sy = H / rtH if rtH > 1 else 1.0

    print("Video:", VIDEO_IN)
    print(f"Video size: {W}x{H} | fps_in={fps_in:.3f} | frames={n_frames}")
    print(f"CSV rtW,rtH={rtW},{rtH}  => scale sx,sy={sx:.5f},{sy:.5f}")
    print("INTERP_MODE:", INTERP_MODE)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(VIDEO_OUT, fourcc, FPS_OUT, (W, H), True)
    if not out.isOpened():
        raise RuntimeError("无法创建输出视频（编码器问题）。")

    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 当前视频时间（秒）
        t_sec = i / FPS_OUT
        # 映射到“原始图片序号时间轴”
        t_idx = t_sec / SECONDS_PER_STEP

        x, y, w, h, k, u = bbox_at_time(df, t_idx)

        # 缩放到视频分辨率
        x0 = x * sx
        y0 = y * sy
        x1 = (x + w) * sx
        y1 = (y + h) * sy
        x0, y0, x1, y1 = clamp_roi(x0, y0, x1, y1, W, H)

        cv2.rectangle(frame, (x0, y0), (x1, y1), BOX_COLOR_BGR, BOX_THICKNESS)

        if DRAW_LABEL:
            txt = f"t={t_sec:.2f}s  t_idx={t_idx:.2f}  k={k} u={u:.2f}  box=({x0},{y0})-({x1},{y1})"
            cv2.putText(frame, txt, (15, 35), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(frame, txt, (15, 35), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 0, 0), 1, cv2.LINE_AA)

        out.write(frame)
        i += 1
        if i % 300 == 0:
            print(f"  processed {i} frames...")

    cap.release()
    out.release()
    print("Saved:", VIDEO_OUT)

if __name__ == "__main__":
    main()
