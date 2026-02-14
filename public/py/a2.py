import math
from pathlib import Path

import numpy as np
from PIL import Image
import imageio.v2 as imageio

# ======================
# CONFIG
# ======================
IMG_DIR  = Path(r"D:\vectionProject\public\camear1images")
PATTERN  = "cam1_*.png"
OUT_MP4  = IMG_DIR / "cam1_gaussCDF_vs_linear.mp4"

FPS = 60
SECONDS_PER_STEP = 1.0
N_PER_STEP = int(round(FPS * SECONDS_PER_STEP))

# ---- Gaussian-CDF weight shape ----
# sigma is defined in "p units" where p in [0,1] within one step.
# Smaller sigma -> more "snap" around center (faster through p~0.5).
# Larger sigma -> smoother but spends longer near p~0.5 (may look slower/blurrier).
SIGMA_P = 0.18  # try 0.12 ~ 0.30

# ---- color space handling ----
# If your PNGs are normal screenshots/recorded frames -> True (sRGB images)
INPUT_FRAMES_ARE_SRGB = True

# If you want to visually amplify output (like previous scripts), keep 1.0 for now.
DISP_SCALE = 1.0

# ======================
# sRGB <-> Linear
# ======================
def srgb_to_linear(x):
    x = np.clip(x, 0.0, 1.0)
    a = 0.055
    return np.where(x <= 0.04045, x / 12.92, ((x + a) / (1.0 + a)) ** 2.4)

def linear_to_srgb(x):
    x = np.clip(x, 0.0, 1.0)
    a = 0.055
    return np.where(x <= 0.0031308, x * 12.92, (1.0 + a) * (x ** (1.0 / 2.4)) - a)

# ======================
# Gaussian CDF weight (monotonic 0->1)
# ======================
def normal_cdf(z):
    # Standard normal CDF using erf
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

def gauss_cdf_weight(p: float, sigma_p: float) -> float:
    """
    p in [0,1]. sigma_p controls steepness.
    Returns w in [0,1], with endpoints exactly mapped to 0 and 1.
    """
    p = float(np.clip(p, 0.0, 1.0))
    sigma_p = max(1e-6, float(sigma_p))

    z  = (p - 0.5) / sigma_p
    z0 = (0.0 - 0.5) / sigma_p
    z1 = (1.0 - 0.5) / sigma_p

    c  = normal_cdf(z)
    c0 = normal_cdf(z0)
    c1 = normal_cdf(z1)

    # normalize to [0,1]
    w = (c - c0) / max(1e-12, (c1 - c0))
    return float(np.clip(w, 0.0, 1.0))

# ======================
# IO
# ======================
def load_gray(path: Path) -> np.ndarray:
    im = Image.open(path).convert("L")
    arr = np.asarray(im, dtype=np.float32) / 255.0
    return arr

def to_u8(x01: np.ndarray) -> np.ndarray:
    x01 = np.clip(x01, 0.0, 1.0)
    return (x01 * 255.0 + 0.5).astype(np.uint8)

# ======================
# MAIN
# ======================
def main():
    paths = sorted(IMG_DIR.glob(PATTERN))
    if len(paths) < 2:
        raise RuntimeError(f"Need at least 2 frames matching {PATTERN} in {IMG_DIR}")

    # load all frames (grayscale 0..1)
    frames = [load_gray(p) for p in paths]
    H, W = frames[0].shape
    for i, f in enumerate(frames):
        if f.shape != (H, W):
            raise RuntimeError(f"Frame size mismatch at {paths[i]}: {f.shape} vs {(H,W)}")

    writer = imageio.get_writer(str(OUT_MP4), fps=FPS, codec="libx264", quality=8)

    try:
        for i in range(len(frames) - 1):
            A = frames[i]
            B = frames[i + 1]

            if INPUT_FRAMES_ARE_SRGB:
                A_lin = srgb_to_linear(A)
                B_lin = srgb_to_linear(B)
            else:
                A_lin = A
                B_lin = B

            for k in range(N_PER_STEP):
                p = 0.0 if N_PER_STEP == 1 else k / (N_PER_STEP - 1)

                # bottom: linear
                w_lin = p

                # top: gaussian-cdf shaped (still only A & B)
                w_g   = gauss_cdf_weight(p, SIGMA_P)

                # mix in linear light
                I_lin = (1.0 - w_lin) * A_lin + w_lin * B_lin
                I_g   = (1.0 - w_g)   * A_lin + w_g   * B_lin

                # optional display scaling
                I_lin = np.clip(I_lin * DISP_SCALE, 0.0, 1.0)
                I_g   = np.clip(I_g   * DISP_SCALE, 0.0, 1.0)

                # back to sRGB for saving
                if INPUT_FRAMES_ARE_SRGB:
                    I_lin = linear_to_srgb(I_lin)
                    I_g   = linear_to_srgb(I_g)

                # stack: top=gauss, bottom=linear (for direct comparison)
                out = np.vstack([to_u8(I_g), to_u8(I_lin)])
                out_rgb = np.stack([out, out, out], axis=-1)  # grayscale -> RGB

                writer.append_data(out_rgb)

        print("Saved:", OUT_MP4)

    finally:
        writer.close()

if __name__ == "__main__":
    main()
