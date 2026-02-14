import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio


# ======================
# PATHS (Windows)
# ======================
IMG_DIR = Path(r"D:\vectionProject\public\camera3images")

# Output folder name: no timestamp
OUT_DIR = IMG_DIR / "phase_comp_demo_"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_MP4 = OUT_DIR / "cam3_png_phase_ampnorm_single.mp4"


# ======================
# INPUT SELECTION
# ======================
# If your PNG filenames have a prefix (recommended), set it here, e.g. "cam3_".
# If you want ALL png in IMG_DIR, set PREFIX = None
PREFIX = None

# If your PNGs are inside subfolders, set RECURSIVE = True
RECURSIVE = False


# ======================
# VIDEO CONFIG
# ======================
FPS = 60
SEC_PER_TRANSITION = 1.0  # 1 second per A->B
DRAW_LABELS = True        # draw text on video
GRAYSCALE = True          # recommended for stable amplitude behavior


# ======================
# Phase-linearized + AmpNorm parameters
# ======================
D_STEP = 0.9 * math.pi  # d
ALPHA = 1.0             # u(t) = alpha * tan(d*p)
AMP_EPS = 0.08          # avoid huge gain
GAIN_CAP = 0.25          # cap gain


# ======================
# Sort helper
# ======================
def natural_key(s: str):
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


# ======================
# Phase-linearized (screenshot-style)
# ======================
def phase_linearized_weight_screenshot(p: float, d: float, alpha: float = 1.0) -> float:
    """
    w_ph(t) = u(t) / ( sin(d) + u(t)*(1 - cos(d)) )
    u(t) = alpha * tan(d*p)
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


def load_images(img_dir: Path, prefix: str | None, recursive: bool):
    if recursive:
        all_paths = list(img_dir.rglob("*.png"))
        # Use relative path string for sorting to keep subfolder order stable
        all_paths = sorted(all_paths, key=lambda p: natural_key(str(p.relative_to(img_dir))))
    else:
        all_paths = sorted(img_dir.glob("*.png"), key=lambda p: natural_key(p.name))

    if prefix:
        all_paths = [p for p in all_paths if p.name.lower().startswith(prefix.lower())]

    if len(all_paths) < 2:
        hint = "rglob" if recursive else "glob"
        raise RuntimeError(
            f"Need at least 2 png images in {img_dir} (prefix={prefix}, {hint}). Found {len(all_paths)}"
        )

    imgs = []
    for p in all_paths:
        im = Image.open(p)
        if GRAYSCALE:
            im = im.convert("L")
            arr = np.asarray(im, dtype=np.float32) / 255.0
            arr = arr[..., None]  # (H,W,1)
        else:
            im = im.convert("RGB")
            arr = np.asarray(im, dtype=np.float32) / 255.0  # (H,W,3)
        imgs.append(arr)

    # size check
    h0, w0 = imgs[0].shape[:2]
    for i, a in enumerate(imgs):
        if a.shape[0] != h0 or a.shape[1] != w0:
            raise RuntimeError(f"Image size mismatch at index {i}: expected {(h0,w0)}, got {a.shape[:2]}")

    return all_paths, imgs


def to_u8_rgb(img01: np.ndarray) -> np.ndarray:
    img01 = np.clip(img01, 0.0, 1.0)
    if img01.ndim == 3 and img01.shape[2] == 1:
        g = (img01[..., 0] * 255.0 + 0.5).astype(np.uint8)
        return np.stack([g, g, g], axis=-1)
    return (img01 * 255.0 + 0.5).astype(np.uint8)


# ======================
# Main
# ======================
def main():
    paths, imgs = load_images(IMG_DIR, PREFIX, RECURSIVE)
    n_imgs = len(imgs)

    frames_per_transition = int(round(FPS * SEC_PER_TRANSITION))
    total_transitions = n_imgs - 1
    total_frames = total_transitions * frames_per_transition
    total_sec = total_transitions * SEC_PER_TRANSITION

    H, W = imgs[0].shape[:2]

    print(f"[INPUT]  {n_imgs} PNGs from: {IMG_DIR}")
    print(f"         prefix={PREFIX}  recursive={RECURSIVE}")
    print(f"[OUTPUT] {OUT_MP4}")
    print(f"[VIDEO]  FPS={FPS}  {SEC_PER_TRANSITION}s/transition  total={total_sec:.1f}s  frames={total_frames}")

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
                # progress p in [0,1]
                p = k / (frames_per_transition - 1) if frames_per_transition > 1 else 1.0
                p = float(np.clip(p, 0.0, 1.0))

                # Phase-linearized + AmpNorm
                w = phase_linearized_weight_screenshot(p, D_STEP, alpha=ALPHA)
                mix = (1.0 - w) * I0 + w * I1

                A = amplitude_of_mix(w, D_STEP)
                gain_raw = min(1.0 / max(AMP_EPS, A), GAIN_CAP)
                GAIN_GAMMA = 0.6  # 0.4~0.8 之间试
                gain = gain_raw ** GAIN_GAMMA


                # out = mix * gain
                LAMBDA = 0.25  # 0=不补偿(不闪), 1=原始AmpNorm(最闪). 建议 0.15~0.35

                m = mix.mean(axis=(0,1), keepdims=True)
                gain_eff = 1.0 + LAMBDA * (gain - 1.0)
                out = m + gain_eff * (mix - m)



                frame = to_u8_rgb(out)

                if DRAW_LABELS:
                    frame = add_label(frame, f"{name0} -> {name1}   p={p:0.2f}   w={w:0.2f}", (10, 10))
                    frame = add_label(frame, f"A={A:0.2f}   gain={gain:0.2f}", (10, 26))
                    frame = add_label(frame, f"d={D_STEP/math.pi:0.2f}π  alpha={ALPHA:0.2f}  eps={AMP_EPS}  cap={GAIN_CAP}", (10, H - 18))

                writer.append_data(frame)
                frame_idx += 1

                if frame_idx % FPS == 0:
                    print(f"  wrote {frame_idx}/{total_frames} frames")

    finally:
        writer.close()

    print("Done.")


if __name__ == "__main__":
    main()
