import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio


# ======================
# PATHS (Windows)
# ======================
IMG_DIR = Path(r"D:\vectionProject\public\camera3images")

# Output folder name: NO timestamp (as requested)
OUT_DIR = IMG_DIR / "phase_comp_demo_"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Output video (beta version removed as requested)
OUT_MP4 = OUT_DIR / "cam3_png_phase_ampnorm_vertical.mp4"


# ======================
# INPUT IMAGE SELECTION
# ======================
# If you want to restrict to a prefix (recommended), set PREFIX = "cam3_"
# If you want ALL png in the folder, set PREFIX = None
PREFIX = "cam1_"  # change to None if needed

# Natural sort helper (e.g., cam3_2.png < cam3_10.png)
def natural_key(s: str):
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


# ======================
# VIDEO CONFIG
# ======================
FPS = 60
SEC_PER_TRANSITION = 1.0  # 1 second per A->B
# Total duration will be (N-1)*SEC_PER_TRANSITION where N = number of images

# Rendering mode:
#   TOP: Linear cross-dissolve (baseline)
#   BOT: Phase-linearized + AmpNorm (requested)
DRAW_LABELS = True

# Convert to grayscale for stable comparison (recommended)
GRAYSCALE = True  # if False, keep RGB


# ======================
# Phase-linearized + AmpNorm parameters
# ======================
D_STEP = 0.9 * math.pi  # d in your theory/demo
ALPHA = 1.0             # alpha in u(t) = alpha * tan(d*p)

AMP_EPS = 0.08          # avoid huge gain when A(w) is small
GAIN_CAP = 2.5          # cap the gain


# ======================
# Math / models
# ======================
def phase_linearized_weight_screenshot(p: float, d: float, alpha: float = 1.0) -> float:
    """
    Screenshot formula:
        w_ph(t) = u(t) / ( sin(d) + u(t)*(1 - cos(d)) )
    with u(t) = alpha * tan(d*p).
    """
    p = float(np.clip(p, 0.0, 1.0))
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0

    alpha = float(np.clip(alpha, 0.2, 5.0))
    dp = d * p
    u = alpha * math.tan(dp)

    sinD = math.sin(d)
    one_minus_cosD = 1.0 - math.cos(d)

    denom = sinD + u * one_minus_cosD
    if abs(denom) < 1e-6:
        return p

    w = u / denom
    return float(np.clip(w, 0.0, 1.0))


def amplitude_of_mix(w: float, d: float) -> float:
    """
    A(w) = sqrt((1-w)^2 + w^2 + 2w(1-w)cos(d))
    """
    c = math.cos(d)
    A2 = (1.0 - w) ** 2 + w ** 2 + 2.0 * w * (1.0 - w) * c
    return math.sqrt(max(0.0, A2))


# ======================
# Image helpers
# ======================
_FONT = ImageFont.load_default()

def add_label(frame_rgb: np.ndarray, text: str, xy=(10, 10)) -> np.ndarray:
    im = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(im)
    x, y = xy
    draw.text((x + 1, y + 1), text, font=_FONT, fill=(0, 0, 0))
    draw.text((x, y), text, font=_FONT, fill=(255, 255, 255))
    return np.array(im)


def load_images(img_dir: Path, prefix: str | None):
    paths = sorted(img_dir.glob("*.png"), key=lambda p: natural_key(p.name))
    if prefix:
        paths = [p for p in paths if p.name.lower().startswith(prefix.lower())]
    if len(paths) < 2:
        raise RuntimeError(f"Need at least 2 png images in {img_dir} (prefix={prefix}). Found {len(paths)}")

    imgs = []
    for p in paths:
        im = Image.open(p)
        if GRAYSCALE:
            im = im.convert("L")  # single channel
            arr = np.asarray(im, dtype=np.float32) / 255.0
            # keep as (H,W,1) for uniform processing
            arr = arr[..., None]
        else:
            im = im.convert("RGB")
            arr = np.asarray(im, dtype=np.float32) / 255.0  # (H,W,3)
        imgs.append(arr)

    # size check
    h0, w0 = imgs[0].shape[:2]
    for i, a in enumerate(imgs):
        if a.shape[0] != h0 or a.shape[1] != w0:
            raise RuntimeError(f"Image size mismatch at index {i}: expected {(h0,w0)}, got {a.shape[:2]}")

    return paths, imgs


def to_u8_rgb(img01: np.ndarray) -> np.ndarray:
    """
    img01: float32 in [0,1], shape (H,W,1) or (H,W,3)
    return: uint8 RGB (H,W,3)
    """
    img01 = np.clip(img01, 0.0, 1.0)
    if img01.ndim == 3 and img01.shape[2] == 1:
        g = (img01[..., 0] * 255.0 + 0.5).astype(np.uint8)
        return np.stack([g, g, g], axis=-1)
    else:
        return (img01 * 255.0 + 0.5).astype(np.uint8)


# ======================
# Main rendering
# ======================
def main():
    paths, imgs = load_images(IMG_DIR, PREFIX)
    n_imgs = len(imgs)
    frames_per_transition = int(round(FPS * SEC_PER_TRANSITION))
    total_transitions = n_imgs - 1
    total_frames = total_transitions * frames_per_transition

    H, W = imgs[0].shape[:2]
    out_h = H * 2  # top+bottom
    out_w = W

    print(f"[INPUT]  {n_imgs} images from: {IMG_DIR}")
    print(f"[OUTPUT] {OUT_MP4}")
    print(f"[VIDEO]  FPS={FPS}  {SEC_PER_TRANSITION}s/transition  total={total_transitions*SEC_PER_TRANSITION:.1f}s")

    writer = imageio.get_writer(
        str(OUT_MP4),
        fps=FPS,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )

    try:
        frame_idx = 0
        for seg in range(total_transitions):
            I0 = imgs[seg]
            I1 = imgs[seg + 1]

            name0 = paths[seg].stem
            name1 = paths[seg + 1].stem

            for k in range(frames_per_transition):
                # progress within this transition
                p = k / (frames_per_transition - 1) if frames_per_transition > 1 else 1.0
                p = float(np.clip(p, 0.0, 1.0))

                # TOP: linear mix
                w_lin = p
                top = (1.0 - w_lin) * I0 + w_lin * I1

                # BOT: phase-linearized + AmpNorm
                w_ph = phase_linearized_weight_screenshot(p, D_STEP, alpha=ALPHA)
                bot_mix = (1.0 - w_ph) * I0 + w_ph * I1

                A = amplitude_of_mix(w_ph, D_STEP)
                gain = 1.0 / max(AMP_EPS, A)
                gain = min(gain, GAIN_CAP)

                bot = bot_mix * gain

                # compose frame: [top; bot]
                top_u8 = to_u8_rgb(top)
                bot_u8 = to_u8_rgb(bot)
                frame = np.concatenate([top_u8, bot_u8], axis=0)  # (2H,W,3)

                if DRAW_LABELS:
                    frame = add_label(frame, f"TOP: Linear  {name0} -> {name1}   p={p:0.2f}", (10, 10))
                    frame = add_label(frame, f"BOT: phase+ampnorm   w={w_ph:0.2f}   A={A:0.2f}   gain={gain:0.2f}", (10, H + 10))
                    frame = add_label(frame, f"d={D_STEP/math.pi:0.2f}π  alpha={ALPHA:0.2f}  eps={AMP_EPS}  cap={GAIN_CAP}", (10, out_h - 18))

                writer.append_data(frame)

                frame_idx += 1
                if frame_idx % (FPS * 1) == 0:
                    print(f"  wrote {frame_idx}/{total_frames} frames")

    finally:
        writer.close()

    print("Done.")


if __name__ == "__main__":
    main()
