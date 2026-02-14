import re
from pathlib import Path

import numpy as np
import imageio.v2 as imageio
from PIL import Image


# ======================
# CONFIG
# ======================
IMG_DIR = Path(r"D:\vectionProject\public\camera3images")
PATTERN = "cam1_*.png"          # 改成 "cam1_*.jpg" 也可以
OUT_MP4_NAME = "cam1_phase_linearized_d0p9pi.mp4"

FPS = 60
STEP_SEC = 1.0                  # 1s per transition
D = 0.9 * np.pi                 # phase shift d
ALPHA = 1.0                     # amplitude ratio alpha (通常取1)

AS_GRAYSCALE = False            # True: 转灰度混合；False: RGB 混合
USE_HALF_FRAME_OFFSET = True    # True: u=(f+0.5)/FPS 更接近论文/你之前实现（更平滑）


# ======================
# UTIL: robust natural sort (all numeric parts)
# ======================
_num_re = re.compile(r"\d+")

def natural_key(p: Path):
    parts = _num_re.split(p.name)
    nums = _num_re.findall(p.name)
    key = []
    for i, s in enumerate(parts):
        if s:
            key.append(("s", s))
        if i < len(nums):
            key.append(("n", int(nums[i])))
    return key


def load_image(path: Path, as_gray: bool) -> np.ndarray:
    if as_gray:
        im = Image.open(path).convert("L")
        arr = np.asarray(im, dtype=np.float32) / 255.0  # (H,W)
    else:
        im = Image.open(path).convert("RGB")
        arr = np.asarray(im, dtype=np.float32) / 255.0  # (H,W,3)
    return arr


def phase_linearized_weight(u: np.ndarray, d: float, alpha: float) -> np.ndarray:
    """
    w(u) = alpha*sin(k) / (alpha*sin(k) + sin(d-k)),  k = d*u
    u in [0,1]
    """
    k = d * u
    num = alpha * np.sin(k)
    den = alpha * np.sin(k) + np.sin(d - k)

    eps = 1e-12
    w = np.where(np.abs(den) < eps, np.where(u < 0.5, 0.0, 1.0), num / den)
    return np.clip(w, 0.0, 1.0)


def to_uint8(img01: np.ndarray) -> np.ndarray:
    img01 = np.clip(img01, 0.0, 1.0)
    return (img01 * 255.0 + 0.5).astype(np.uint8)


def main():
    if not IMG_DIR.exists():
        raise FileNotFoundError(f"IMG_DIR not found: {IMG_DIR}")

    keyframes = sorted(IMG_DIR.glob(PATTERN), key=natural_key)
    if len(keyframes) < 2:
        raise RuntimeError(f"Need at least 2 images, found {len(keyframes)}")

    print(f"[INFO] Found {len(keyframes)} keyframes.")
    print("[INFO] First 10 files:")
    for p in keyframes[:10]:
        print("  ", p.name)

    out_mp4 = IMG_DIR / OUT_MP4_NAME

    # Preload all keyframes (avoid IO in loop)
    frames = [load_image(p, AS_GRAYSCALE) for p in keyframes]

    # Size check
    H, W = frames[0].shape[:2]
    for i, a in enumerate(frames):
        if a.shape[:2] != (H, W):
            raise ValueError(f"Size mismatch at {keyframes[i].name}: {a.shape[:2]} vs {(H, W)}")

    # Build u grid for one 1s transition (FPS frames)
    if USE_HALF_FRAME_OFFSET:
        u = (np.arange(FPS, dtype=np.float32) + 0.5) / FPS  # (0,1)
    else:
        u = np.arange(FPS, dtype=np.float32) / (FPS - 1)    # includes 0 and 1

    w = phase_linearized_weight(u, D, ALPHA)  # shape (FPS,)

    # Writer
    pix_fmt = "gray" if AS_GRAYSCALE else "yuv420p"
    with imageio.get_writer(
        out_mp4,
        fps=FPS,
        codec="libx264",
        pixelformat=pix_fmt,
        macro_block_size=None,
        quality=8,
    ) as writer:

        # For each adjacent pair (In -> In+1), generate FPS blended frames
        for seg in range(len(frames) - 1):
            A = frames[seg]
            B = frames[seg + 1]

            for f in range(FPS):
                ww = float(w[f])
                out01 = (1.0 - ww) * A + ww * B
                writer.append_data(to_uint8(out01))

            if (seg + 1) % 5 == 0 or seg == len(frames) - 2:
                print(f"[INFO] Done segment {seg+1}/{len(frames)-1}")

        # Optional: append final keyframe as an exact endpoint
        writer.append_data(to_uint8(frames[-1]))

    print("[DONE] Saved:", out_mp4)


if __name__ == "__main__":
    main()
