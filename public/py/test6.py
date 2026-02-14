import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import imageio.v2 as imageio


# =========================
# CONFIG (edit if needed)
# =========================
IMG_DIR = Path(r"D:\vectionProject\public\camear1images")
GLOB_PATTERN = "cam1_*.png"

OUT_MP4 = IMG_DIR / "cam1_two_rows_contrastnorm.mp4"

FPS = 30
SEC_PER_TRANSITION = 1.0

# Your phase-linearized weight params (same as your earlier demo)
D_STEP = 0.9 * math.pi
ALPHA = 1.0

# Contrast-domain normalization params (for natural images)
BLUR_SIGMA = 6.0        # Gaussian blur radius (bigger -> smoother "mean")
TARGET_RMS = 0.08       # target contrast energy (tune 0.04~0.12)
GAIN_CAP = 3.0          # max gain to avoid over-boost

# Optional: resize for faster video generation (set to None to keep original)
RESIZE_TO = None        # e.g. (960, 200) or None

# Labels
_FONT = ImageFont.load_default()


# =========================
# Math: w_ph (same as your demo)
# =========================
def phase_linearized_weight_alpha(p: float, d: float, alpha: float = 1.0) -> float:
    p = float(np.clip(p, 0.0, 1.0))
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0

    alpha = float(np.clip(alpha, 0.2, 5.0))
    k = d * p

    cosK = math.cos(k)
    sinK = math.sin(k)
    if abs(cosK) < 1e-5:
        cosK = 1e-5 * (1.0 if cosK >= 0 else -1.0)

    s = sinK / cosK  # tan(k)
    sinD = math.sin(d)
    cosD = math.cos(d)

    denom = sinD + s * alpha - s * cosD
    if abs(denom) < 1e-6:
        return p

    w = (s * alpha) / denom
    return float(np.clip(w, 0.0, 1.0))


# =========================
# Image helpers
# =========================
def load_rgb01(path: Path, resize_to=None) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    if resize_to is not None:
        im = im.resize(resize_to, Image.BILINEAR)
    arr = np.asarray(im).astype(np.float32) / 255.0
    return arr

def rgb_to_gray01(rgb01: np.ndarray) -> np.ndarray:
    # luminance-like gray in [0,1]
    return (0.2126 * rgb01[..., 0] + 0.7152 * rgb01[..., 1] + 0.0722 * rgb01[..., 2]).astype(np.float32)

def gray01_to_rgb_u8(gray01: np.ndarray) -> np.ndarray:
    g8 = np.clip(gray01 * 255.0, 0, 255).astype(np.uint8)
    return np.stack([g8, g8, g8], axis=-1)

def gaussian_blur_gray(gray01: np.ndarray, sigma: float) -> np.ndarray:
    # PIL blur works on uint8; good enough for this sanity check
    im = Image.fromarray(np.clip(gray01 * 255.0, 0, 255).astype(np.uint8))
    im = im.filter(ImageFilter.GaussianBlur(radius=float(sigma)))
    return (np.asarray(im).astype(np.float32) / 255.0)

def rms(x: np.ndarray, eps: float = 1e-6) -> float:
    return float(np.sqrt(np.mean(x * x) + eps))

def add_label(frame_rgb_u8: np.ndarray, text: str, xy=(10, 10)) -> np.ndarray:
    im = Image.fromarray(frame_rgb_u8)
    draw = ImageDraw.Draw(im)
    x, y = xy
    draw.text((x + 1, y + 1), text, font=_FONT, fill=(0, 0, 0))
    draw.text((x, y), text, font=_FONT, fill=(255, 255, 255))
    return np.asarray(im)


# =========================
# Core: build frames and write video
# =========================
def make_video_two_rows(paths):
    paths = list(paths)
    if len(paths) < 2:
        raise ValueError("Need at least 2 images.")

    # Pre-load grayscale frames (fast enough for 60)
    rgbs = [load_rgb01(p, resize_to=RESIZE_TO) for p in paths]
    grays = [rgb_to_gray01(rgb) for rgb in rgbs]
    H, W = grays[0].shape

    total_transitions = len(grays) - 1
    frames_per = int(round(SEC_PER_TRANSITION * FPS))

    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    writer = imageio.get_writer(
        str(OUT_MP4),
        fps=FPS,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )

    for seg in range(total_transitions):
        Aname = letters[seg % len(letters)]
        Bname = letters[(seg + 1) % len(letters)]

        L0 = grays[seg]
        L1 = grays[seg + 1]

        # Precompute local mean (μ) and contrast (c) for both endpoints
        mu0 = gaussian_blur_gray(L0, BLUR_SIGMA)
        mu1 = gaussian_blur_gray(L1, BLUR_SIGMA)
        c0 = L0 - mu0
        c1 = L1 - mu1

        for f in range(frames_per):
            p = f / max(1, frames_per - 1)

            # TOP: linear dissolve on luminance
            top = (1.0 - p) * L0 + p * L1
            top = np.clip(top, 0.0, 1.0)

            # BOT: phase-linearized dissolve on contrast channel + RMS norm
            w_ph = phase_linearized_weight_alpha(p, D_STEP, alpha=ALPHA)

            # Mix contrast using w_ph
            c_mix = (1.0 - w_ph) * c0 + w_ph * c1

            # Normalize energy using actual RMS of c_mix (works for natural images)
            cur = rms(c_mix)
            gain = TARGET_RMS / max(1e-6, cur)
            gain = min(gain, GAIN_CAP)
            c_out = c_mix * gain

            # Mix mean using linear p (keeps overall brightness stable)
            mu_mix = (1.0 - p) * mu0 + p * mu1

            bot = np.clip(mu_mix + c_out, 0.0, 1.0)

            frame = np.concatenate([gray01_to_rgb_u8(top), gray01_to_rgb_u8(bot)], axis=0)

            # labels
            frame = add_label(frame, f"TOP Linear {Aname}->{Bname}  p={p:0.2f}", (10, 10))
            frame = add_label(
                frame,
                f"BOT contrast-norm  w_ph={w_ph:0.2f}  rms={cur:0.3f}  gain={gain:0.2f}  blur={BLUR_SIGMA}",
                (10, H + 10),
            )
            frame = add_label(
                frame,
                f"d_step={D_STEP/math.pi:0.2f}π  alpha={ALPHA:0.2f}  target_rms={TARGET_RMS}  cap={GAIN_CAP}",
                (10, 2 * H - 18),
            )

            writer.append_data(frame)

    writer.close()
    return OUT_MP4


def main():
    paths = sorted(IMG_DIR.glob(GLOB_PATTERN))
    print(f"Found {len(paths)} images in {IMG_DIR}")
    if len(paths) < 2:
        print("Not enough images to build video.")
        return

    out = make_video_two_rows(paths)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
