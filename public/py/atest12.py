import math
from pathlib import Path
import numpy as np
import cv2
import imageio.v2 as imageio

# =========================
# CONFIG
# =========================
IN_DIR  = Path(r"D:\vectionProject\public\camear1images")
OUT_DIR = Path(r"D:\vectionProject\public\freq_test_videos")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FPS = 60
SECONDS_PER_STEP = 1.0
N = int(round(FPS * SECONDS_PER_STEP))

DOG_SIGMA_SMALL = 1.2
DOG_SIGMA_LARGE = 4.0
BAND_VIS_GAIN   = 10.0
BAND_VIS_BIAS   = 0.5

# KEY: make ghosting never vanish too much
P_EPS = 0.15      # try 0.08 then 0.15

# soft clip (recommended)
USE_SOFTCLIP = True

# =========================
# sRGB <-> Linear + luminance
# =========================
def srgb_to_linear(x):
    x = np.clip(x, 0.0, 1.0)
    a = 0.055
    return np.where(x <= 0.04045, x / 12.92, ((x + a) / (1 + a)) ** 2.4)

def linear_to_srgb(x):
    x = np.clip(x, 0.0, 1.0)
    a = 0.055
    return np.where(x <= 0.0031308, x * 12.92, (1 + a) * (x ** (1 / 2.4)) - a)

def clamp01(x):
    return np.clip(x, 0.0, 1.0)

def rgb_lin_to_luma_lin(img_lin: np.ndarray) -> np.ndarray:
    return (0.2126 * img_lin[..., 0] +
            0.7152 * img_lin[..., 1] +
            0.0722 * img_lin[..., 2]).astype(np.float32)

def luma3_from_rgb_lin(img_lin: np.ndarray) -> np.ndarray:
    g = rgb_lin_to_luma_lin(img_lin)
    return np.repeat(g[..., None], 3, axis=2)

# =========================
# IO
# =========================
def read_rgb_linear(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return srgb_to_linear(rgb).astype(np.float32)

def write_frame(writer, img_lin: np.ndarray):
    img_srgb = linear_to_srgb(clamp01(img_lin))
    u8 = (img_srgb * 255.0 + 0.5).astype(np.uint8)
    writer.append_data(u8)

# =========================
# Filters
# =========================
def gaussian_blur_lin(img_lin: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(
        img_lin, (0, 0),
        sigmaX=float(sigma), sigmaY=float(sigma),
        borderType=cv2.BORDER_REFLECT
    )

def dog_band_lin(img_lin: np.ndarray, sigma_small: float, sigma_large: float) -> np.ndarray:
    lo1 = gaussian_blur_lin(img_lin, sigma_small)
    lo2 = gaussian_blur_lin(img_lin, sigma_large)
    return lo1 - lo2

def band_to_vis_lin(band_lin: np.ndarray, gain: float, bias: float) -> np.ndarray:
    x = gain * band_lin
    if USE_SOFTCLIP:
        vis = bias + 0.5 * np.tanh(x)
        return clamp01(vis)
    else:
        return clamp01(bias + x)

# =========================
# p(t)
# =========================
def p_linear(t: float) -> float:
    return float(np.clip(t, 0.0, 1.0))

def apply_p_eps(p_raw: float, eps_p: float) -> float:
    eps_p = float(np.clip(eps_p, 0.0, 0.45))
    return float(eps_p + (1.0 - 2.0 * eps_p) * np.clip(p_raw, 0.0, 1.0))

# =========================
# Video
# =========================
def make_highpass_video(frames: list[Path], out_mp4: Path):
    writer = imageio.get_writer(str(out_mp4), fps=FPS, codec="libx264", quality=8, macro_block_size=1)
    try:
        for i in range(len(frames) - 1):
            A_rgb = read_rgb_linear(frames[i])
            B_rgb = read_rgb_linear(frames[i + 1])

            A = luma3_from_rgb_lin(A_rgb)
            B = luma3_from_rgb_lin(B_rgb)

            A_band = dog_band_lin(A, DOG_SIGMA_SMALL, DOG_SIGMA_LARGE)
            B_band = dog_band_lin(B, DOG_SIGMA_SMALL, DOG_SIGMA_LARGE)

            A_vis = band_to_vis_lin(A_band, BAND_VIS_GAIN, BAND_VIS_BIAS)
            B_vis = band_to_vis_lin(B_band, BAND_VIS_GAIN, BAND_VIS_BIAS)

            for k in range(N):
                t = k / (N - 1) if N > 1 else 1.0
                p = apply_p_eps(p_linear(t), P_EPS)
                out = (1.0 - p) * A_vis + p * B_vis
                write_frame(writer, out)

        print("saved:", out_mp4)
    finally:
        writer.close()

def main():
    exts = (".png", ".jpg", ".jpeg")
    frames = sorted([p for p in IN_DIR.iterdir() if p.suffix.lower() in exts])
    if len(frames) < 2:
        raise RuntimeError("Not enough frames")

    out = OUT_DIR / f"highpass_linear_eps{P_EPS:.2f}" + ("_soft.mp4" if USE_SOFTCLIP else "_hard.mp4")
    make_highpass_video(frames, out)

if __name__ == "__main__":
    main()
