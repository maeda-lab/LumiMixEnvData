"""
Generate experiment videos:
C3: Gauss (already have C2) + ROI 1 Hz high-frequency edge injection  -> OUT_C3_INJECT
C4: Linear + ROI high-frequency spike suppression (soft-knee)          -> OUT_C4_SUPPRESS
C5: Linear + ROI high-frequency ENERGY-MATCHED suppression to Gauss    -> OUT_C5_MATCH

You already have C2 video, but for C5 we still need the C2 (gauss) *signal*
to match E(t). We compute E(t) from the same still frames using Gauss weights
(sigma=0.6). No need to read your existing mp4.

Assumptions:
- Still frames are grayscale PNG: cam2_*.png (one image per 1s step)
- cam2_tree_bbox.csv exists (same format as you uploaded)

Run:
    python this_script.py
"""

import os
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import imageio.v2 as imageio

# =========================
# CONFIG (EDIT THESE)
# =========================
IMG_DIR = Path(r"D:\vectionProject\public\camera2images")   # <-- change
IMG_GLOB = "cam2_*.png"

CSV_BBOX = IMG_DIR / "cam2_tree_bbox.csv"

# Outputs
OUT_C3_INJECT = IMG_DIR / "C3_gauss_ROI_1Hz_injected.mp4"
OUT_C4_SUPPRESS = IMG_DIR / "C4_linear_ROI_suppressed_softknee.mp4"
OUT_C5_MATCH = IMG_DIR / "C5_linear_ROI_energyMatchedToGauss.mp4"

FPS = 60
SEC_PER_STEP = 1.0

# C2 Gauss sigma (used for computing base + target E(t))
SIGMA_T = 0.6

# Duration (None => (N-1)*SEC_PER_STEP)
DURATION_SEC = None

# ---- C3 Injection params ----
INJECT_FREQ_HZ = 1.0
INJECT_PHASE = 0.0
INJECT_STRENGTH = 1.2      # gain multiplier on HF when sin>0 (try 0.8~1.6)

HF_SPLIT_BLUR_SIGMA = 1.2  # for ROI low/high split

# ---- C4 Suppression (spike-only) ----
SUPPRESS_WINDOW_SEC = 0.5  # smoothing window of E(t) baseline
SUPPRESS_MARGIN = 1.05     # only suppress above baseline*margin
SUPPRESS_GAMMA = 0.9       # aggressiveness
SUPPRESS_GAIN_FLOOR = 0.35 # do not reduce HF below this
SUPPRESS_GAIN_CEIL = 1.0

# ---- C5 Energy-matched to Gauss ----
# Aim: make Linear ROI edge-energy E_lin(t) match E_gauss(t) as much as possible
MATCH_SMOOTH_SEC = 0.2     # smooth both E curves before computing gain
MATCH_GAIN_FLOOR = 0.20    # allow stronger suppression than C4
MATCH_GAIN_CEIL = 1.0
MATCH_GAMMA = 0.8          # 1.0 => strict sqrt-ratio; <1.0 => softer

# ---- Clipping ----
SOFT_CLIP = True
SOFT_KNEE = 0.04           # 0.02~0.08

# ---- Encoding ----
LOSSLESS = False
CRF = 18


# =========================
# Basic utils
# =========================
def load_still_frames(img_dir: Path, glob_pat: str):
    paths = sorted(img_dir.glob(glob_pat))
    if not paths:
        raise RuntimeError(f"No images found: {img_dir}/{glob_pat}")
    imgs = []
    for p in paths:
        im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if im is None:
            raise RuntimeError(f"Failed to read {p}")
        imgs.append(im.astype(np.float32) / 255.0)
    return paths, imgs

def make_writer(out_path: Path, fps: int):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if LOSSLESS:
        return imageio.get_writer(
            str(out_path),
            fps=fps,
            codec="libx264",
            ffmpeg_params=["-pix_fmt", "yuv420p", "-crf", "0", "-preset", "veryslow"],
        )
    return imageio.get_writer(
        str(out_path),
        fps=fps,
        codec="libx264",
        ffmpeg_params=["-pix_fmt", "yuv420p", "-crf", str(CRF), "-preset", "medium"],
    )

def to_u8(x01: np.ndarray) -> np.ndarray:
    x01 = np.clip(x01, 0.0, 1.0)
    return (x01 * 255.0 + 0.5).astype(np.uint8)

def soft_clip01(x: np.ndarray, knee: float = 0.04) -> np.ndarray:
    x = x.astype(np.float32)
    if knee <= 0:
        return np.clip(x, 0.0, 1.0)
    y = x.copy()
    y = np.clip(y, 0.0, 1.0)
    m0 = (y >= 0.0) & (y < knee)
    if np.any(m0):
        t = y[m0] / knee
        y[m0] = knee * np.sqrt(t)
    m1 = (y <= 1.0) & (y > 1.0 - knee)
    if np.any(m1):
        t = (1.0 - y[m1]) / knee
        y[m1] = 1.0 - knee * np.sqrt(t)
    return np.clip(y, 0.0, 1.0)

def moving_average(x: np.ndarray, win: int):
    win = int(max(1, win))
    if win % 2 == 0:
        win += 1
    k = np.ones((win,), dtype=np.float32) / float(win)
    return np.convolve(x.astype(np.float32), k, mode="same")


# =========================
# BBox track
# =========================
class BBoxTrack:
    def __init__(self, csv_path: Path, img_w: int, img_h: int):
        df = pd.read_csv(csv_path)
        df = df[df["valid"] == 1].sort_values("secIndex").copy()
        if df.empty:
            raise RuntimeError("No valid rows in bbox csv.")

        self.secs = df["secIndex"].to_numpy(dtype=np.int32)
        rtW = float(df["rtW"].iloc[0])
        rtH = float(df["rtH"].iloc[0])
        sx = img_w / rtW
        sy = img_h / rtH

        self.x = (df["x_tl"].to_numpy(dtype=np.float32) * sx)
        self.y = (df["y_tl"].to_numpy(dtype=np.float32) * sy)
        self.w = (df["w_tl"].to_numpy(dtype=np.float32) * sx)
        self.h = (df["h_tl"].to_numpy(dtype=np.float32) * sy)

        self.img_w = img_w
        self.img_h = img_h

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
        x0 = int(round(x))
        y0 = int(round(y))
        x1 = int(round(x + w))
        y1 = int(round(y + h))

        x0 = max(0, min(self.img_w - 2, x0))
        y0 = max(0, min(self.img_h - 2, y0))
        x1 = max(x0 + 1, min(self.img_w - 1, x1))
        y1 = max(y0 + 1, min(self.img_h - 1, y1))
        return x0, y0, x1, y1


# =========================
# Temporal mixing
# =========================
def blend_linear(frames, t_sec, sec_per_step):
    center = t_sec / sec_per_step
    i = int(math.floor(center))
    p = center - i
    i0 = max(0, min(len(frames) - 1, i))
    i1 = max(0, min(len(frames) - 1, i + 1))
    if i0 == i1:
        return frames[i0]
    return (1.0 - p) * frames[i0] + p * frames[i1]

def blend_gauss(frames, t_sec, sec_per_step, sigma_steps):
    center = t_sec / sec_per_step
    R = max(1, int(math.ceil(3.0 * sigma_steps)))
    k0 = int(math.floor(center)) - R
    k1 = int(math.floor(center)) + R + 1

    idxs = []
    ws = []
    for k in range(k0, k1):
        kk = max(0, min(len(frames) - 1, k))
        d = (k - center)
        w = math.exp(-(d * d) / (2.0 * sigma_steps * sigma_steps))
        idxs.append(kk)
        ws.append(w)

    ws = np.asarray(ws, dtype=np.float32)
    ws = ws / (ws.sum() + 1e-12)

    out = np.zeros_like(frames[0], dtype=np.float32)
    for kk, w in zip(idxs, ws):
        out += w * frames[kk]
    return out


# =========================
# ROI HF split + ops
# =========================
def hf_split(roi: np.ndarray, blur_sigma: float):
    k = int(round(blur_sigma * 6 + 1))
    k = max(3, k | 1)
    low = cv2.GaussianBlur(roi, (k, k), blur_sigma)
    high = roi - low
    return low, high

def roi_edge_energy_lap(roi: np.ndarray) -> float:
    L = cv2.Laplacian(roi, cv2.CV_32F, ksize=3)
    return float(np.mean(np.abs(L)))

def apply_roi_gain(frame01: np.ndarray, x0,y0,x1,y1, gain: float, blur_sigma: float):
    out = frame01.copy()
    roi = out[y0:y1, x0:x1]
    low, high = hf_split(roi, blur_sigma)
    roi2 = low + float(gain) * high
    roi2 = soft_clip01(roi2, SOFT_KNEE) if SOFT_CLIP else np.clip(roi2, 0, 1)
    out[y0:y1, x0:x1] = roi2
    return out


# =========================
# Build C3/C4/C5
# =========================
def main():
    # Load still frames
    img_paths, frames = load_still_frames(IMG_DIR, IMG_GLOB)
    n_steps = len(frames)
    H, W = frames[0].shape[:2]
    print(f"[OK] loaded {n_steps} still frames, size={W}x{H}")

    # Duration
    duration_sec = (n_steps - 1) * SEC_PER_STEP if DURATION_SEC is None else float(DURATION_SEC)
    n_out = int(round(duration_sec * FPS))
    print(f"[OK] output duration={duration_sec:.3f}s, frames={n_out}")

    # BBox
    bbox = BBoxTrack(CSV_BBOX, W, H)
    print("[OK] bbox loaded")

    # -------------------------
    # C3: Gauss + ROI injected
    # -------------------------
    print("[C3] generating gauss + ROI 1Hz injection...")
    with make_writer(OUT_C3_INJECT, FPS) as wr:
        for fidx in range(n_out):
            t = fidx / FPS
            base = blend_gauss(frames, t, SEC_PER_STEP, SIGMA_T)

            x0,y0,x1,y1 = bbox.roi_int(t)
            # periodic gain (boost only when sin>0)
            m = math.sin(2.0 * math.pi * INJECT_FREQ_HZ * t + INJECT_PHASE)
            g = 1.0 + INJECT_STRENGTH * max(0.0, m)

            out = apply_roi_gain(base, x0,y0,x1,y1, g, HF_SPLIT_BLUR_SIGMA)
            wr.append_data(to_u8(out))

            if fidx % 300 == 0:
                print(f"  C3 {fidx}/{n_out}")
    print("[C3] done:", OUT_C3_INJECT)

    # --------------------------------------------------------
    # C4/C5 need E(t) sequences:
    #   E_lin(t) from Linear base
    #   E_gau(t) from Gauss base (for C5 target)
    # --------------------------------------------------------
    print("[C4/C5] pass1: computing E_lin(t) and E_gau(t)...")
    E_lin = np.zeros((n_out,), dtype=np.float32)
    E_gau = np.zeros((n_out,), dtype=np.float32)

    for fidx in range(n_out):
        t = fidx / FPS
        x0,y0,x1,y1 = bbox.roi_int(t)

        lin = blend_linear(frames, t, SEC_PER_STEP)
        gau = blend_gauss(frames, t, SEC_PER_STEP, SIGMA_T)

        E_lin[fidx] = roi_edge_energy_lap(lin[y0:y1, x0:x1])
        E_gau[fidx] = roi_edge_energy_lap(gau[y0:y1, x0:x1])

        if fidx % 300 == 0:
            print(f"  pass1 {fidx}/{n_out}")

    # -------------------------
    # C4: spike-only suppression
    # -------------------------
    print("[C4] building gain(t) for spike-only suppression...")
    win4 = int(round(SUPPRESS_WINDOW_SEC * FPS))
    E_base = moving_average(E_lin, win4)
    target = E_base * float(SUPPRESS_MARGIN)

    gain4 = np.ones_like(E_lin, dtype=np.float32)
    over = E_lin > target
    gain4[over] = np.power((target[over] + 1e-6) / (E_lin[over] + 1e-6), float(SUPPRESS_GAMMA))
    gain4 = np.clip(gain4, SUPPRESS_GAIN_FLOOR, SUPPRESS_GAIN_CEIL)

    print("[C4] generating linear + ROI suppressed (spikes only)...")
    with make_writer(OUT_C4_SUPPRESS, FPS) as wr:
        for fidx in range(n_out):
            t = fidx / FPS
            base = blend_linear(frames, t, SEC_PER_STEP)
            x0,y0,x1,y1 = bbox.roi_int(t)
            out = apply_roi_gain(base, x0,y0,x1,y1, float(gain4[fidx]), HF_SPLIT_BLUR_SIGMA)
            wr.append_data(to_u8(out))
            if fidx % 300 == 0:
                print(f"  C4 {fidx}/{n_out}")
    print("[C4] done:", OUT_C4_SUPPRESS)

    # --------------------------------------------
    # C5: energy-matched suppression to Gauss E(t)
    # --------------------------------------------
    print("[C5] building gain(t) to match E_lin(t) -> E_gau(t)...")
    win5 = int(round(MATCH_SMOOTH_SEC * FPS))
    E_lin_s = moving_average(E_lin, win5)
    E_gau_s = moving_average(E_gau, win5)

    # Gain based on sqrt ratio (energy ~ squared amplitude), softened by MATCH_GAMMA
    # g = (E_target/E_current)^(0.5 * gamma)
    ratio = (E_gau_s + 1e-6) / (E_lin_s + 1e-6)
    gain5 = np.power(ratio, 0.5 * float(MATCH_GAMMA)).astype(np.float32)

    # Only suppress (do not boost above 1.0), since we want "linear -> gauss-like"
    gain5 = np.minimum(gain5, 1.0)
    gain5 = np.clip(gain5, MATCH_GAIN_FLOOR, MATCH_GAIN_CEIL)

    print("[C5] generating linear + ROI energy-matched-to-gauss...")
    with make_writer(OUT_C5_MATCH, FPS) as wr:
        for fidx in range(n_out):
            t = fidx / FPS
            base = blend_linear(frames, t, SEC_PER_STEP)
            x0,y0,x1,y1 = bbox.roi_int(t)
            out = apply_roi_gain(base, x0,y0,x1,y1, float(gain5[fidx]), HF_SPLIT_BLUR_SIGMA)
            wr.append_data(to_u8(out))
            if fidx % 300 == 0:
                print(f"  C5 {fidx}/{n_out}")
    print("[C5] done:", OUT_C5_MATCH)

    print("\nDONE:")
    print(" C3:", OUT_C3_INJECT)
    print(" C4:", OUT_C4_SUPPRESS)
    print(" C5:", OUT_C5_MATCH)


if __name__ == "__main__":
    main()
