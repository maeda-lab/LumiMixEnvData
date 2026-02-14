import os
import math
from pathlib import Path

import cv2
import pandas as pd

# ======================
# CONFIG
# ======================
ROI_CSV = r"D:\vectionProject\public\camera3images\rois.csv"

VIDEOS = {
    "cam3_linear": r"D:\vectionProject\public\camera3images\cam3_linear.mp4",
    "gauss3tap_sigma0p6": r"D:\vectionProject\public\camera3images\cam3_truncgauss3tap_sigma0p6.mp4",
    # "ampnorm": r"D:\vectionProject\public\camera3images\\phase_comp_demo_\cam3_png_phase_ampnorm_single.mp4",
    "phase comp":   r"D:\vectionProject\public\camera3images\cam1_phase_linearized_d0p9pi.mp4",
}

OUT_DIR = r"D:\vectionProject\public\camera3images\roi_crops_cam3_nopad"
os.makedirs(OUT_DIR, exist_ok=True)

TIMES_SEC = [4.0, 4.5]   # 相对 ROI 记录起点的时间
STEP_SEC = 1.0           # secIndex 步长（通常 1Hz）
FPS_FALLBACK = 60.0

RECT_COLOR = (0, 255, 0)   # 绿色 (B, G, R)
RECT_THICKNESS = 4


# ======================
# Helpers
# ======================
def clamp_int(v, lo, hi):
    return int(max(lo, min(hi, v)))


def read_frame_at_time(video_path: str, t_sec: float):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-3:
        fps = float(FPS_FALLBACK)

    frame_idx = int(round(t_sec * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        raise RuntimeError(f"Failed to read frame at t={t_sec:.3f}s (frame={frame_idx}) from {video_path}")

    return frame, fps, frame_idx


def draw_rect_green(frame_bgr, x, y, w, h):
    out = frame_bgr.copy()
    cv2.rectangle(out, (x, y), (x + w, y + h), RECT_COLOR, RECT_THICKNESS)
    return out


def crop_exact_roi(frame_bgr, x, y, w, h):
    """No padding. Crop exactly ROI."""
    H, W = frame_bgr.shape[:2]
    x0 = clamp_int(x, 0, W - 1)
    y0 = clamp_int(y, 0, H - 1)
    x1 = clamp_int(x + w, 0, W)
    y1 = clamp_int(y + h, 0, H)
    crop = frame_bgr[y0:y1, x0:x1].copy()
    return crop, (x0, y0, x1, y1)


# ======================
# ROI CSV load + time interpolation
# ======================
def load_rois(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if "valid" in df.columns:
        df = df[df["valid"].astype(int) == 1].copy()

    if "secIndex" not in df.columns:
        raise ValueError("CSV missing column: secIndex")

    # Use top-left ROI
    required = ["x_tl", "y_tl", "w_tl", "h_tl", "rtW", "rtH"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing} (need x_tl,y_tl,w_tl,h_tl,rtW,rtH)")

    df = df[["secIndex", "x_tl", "y_tl", "w_tl", "h_tl", "rtW", "rtH"]].copy()
    df["secIndex"] = df["secIndex"].astype(int)

    for c in ["x_tl", "y_tl", "w_tl", "h_tl", "rtW", "rtH"]:
        df[c] = df[c].astype(float)

    df = df.sort_values("secIndex").reset_index(drop=True)
    return df


def roi_interp_by_sec(df: pd.DataFrame, sec_float: float):
    sec_min = int(df["secIndex"].min())
    sec_max = int(df["secIndex"].max())

    sec_float = max(sec_min, min(sec_max, sec_float))

    s0 = int(math.floor(sec_float))
    s1 = min(sec_max, s0 + 1)
    a = sec_float - s0

    r0 = df[df["secIndex"] == s0].iloc[0]
    r1 = df[df["secIndex"] == s1].iloc[0]

    x = (1 - a) * r0["x_tl"] + a * r1["x_tl"]
    y = (1 - a) * r0["y_tl"] + a * r1["y_tl"]
    w = (1 - a) * r0["w_tl"] + a * r1["w_tl"]
    h = (1 - a) * r0["h_tl"] + a * r1["h_tl"]

    rtW = r0["rtW"]
    rtH = r0["rtH"]
    return x, y, w, h, rtW, rtH, s0, s1, a


def scale_roi_to_video(x, y, w, h, rtW, rtH, Wv, Hv):
    sx = Wv / float(rtW)
    sy = Hv / float(rtH)
    return x * sx, y * sy, w * sx, h * sy


# ======================
# Main
# ======================
def main():
    rois = load_rois(ROI_CSV)
    base_sec = float(rois["secIndex"].min())  # e.g., 15

    for tag, vid in VIDEOS.items():
        for t in TIMES_SEC:
            frame, fps, frame_idx = read_frame_at_time(vid, t)
            Hv, Wv = frame.shape[:2]

            # Align: t is relative to ROI record start
            sec_float = base_sec + (t / STEP_SEC)

            x, y, w, h, rtW, rtH, s0, s1, a = roi_interp_by_sec(rois, sec_float)
            x, y, w, h = scale_roi_to_video(x, y, w, h, rtW, rtH, Wv, Hv)

            x = clamp_int(round(x), 0, Wv - 1)
            y = clamp_int(round(y), 0, Hv - 1)
            w = clamp_int(round(w), 1, Wv - x)
            h = clamp_int(round(h), 1, Hv - y)

            full = draw_rect_green(frame, x, y, w, h)
            crop, (x0, y0, x1, y1) = crop_exact_roi(full, x, y, w, h)

            t_str = f"{t:.3f}".replace(".", "p")
            base = f"{tag}_t{t_str}s_frame{frame_idx}_sec{sec_float:.2f}_({s0}-{s1}_a{a:.2f})"

            out_full = Path(OUT_DIR) / f"{base}_full_greenROI.png"
            out_crop = Path(OUT_DIR) / f"{base}_crop_exactROI.png"

            cv2.imwrite(str(out_full), full)
            cv2.imwrite(str(out_crop), crop)

            print(
                f"[OK] {tag} t={t:.3f}s -> secIndex~{sec_float:.2f} (interp {s0}-{s1}, a={a:.2f})\n"
                f"     video(W,H)=({Wv},{Hv}) rt(W,H)=({rtW:.0f},{rtH:.0f})\n"
                f"     ROI(x,y,w,h)=({x},{y},{w},{h}) crop=({x0},{y0})-({x1},{y1})\n"
                f"     -> {out_full.name}\n"
                f"     -> {out_crop.name}"
            )

    print(f"\nDone. Output dir:\n  {OUT_DIR}")


if __name__ == "__main__":
    main()
