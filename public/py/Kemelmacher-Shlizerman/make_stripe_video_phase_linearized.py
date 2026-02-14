import numpy as np
import imageio.v2 as imageio
from pathlib import Path


# ======================
# CONFIG
# ======================
# Output
OUT_MP4_NAME = "stripe_phase_linearized_d0p9pi_0_5s.mp4"
SAVE_KEYFRAMES_PNG = False
KEYFRAME_DIR_NAME = "stripe_keyframes"  # only used if SAVE_KEYFRAMES_PNG=True

# Timing
FPS = 60
STEP_SEC = 1.0        # 1s per transition (one cycle)
N_STEPS = 5           # 0..5s => 5 transitions, 6 keyframes

# Phase / compensation parameters
d = 0.9 * np.pi       # phase shift per step (between adjacent keyframes)
alpha = 1.0           # amplitude ratio in the inverse formula (usually 1)

USE_HALF_FRAME_OFFSET = True  # u=(f+0.5)/FPS (recommended)

# Stripe stimulus (grating)
W = 960
H = 540
MEAN = 0.5            # mean luminance in [0,1]
CONTRAST = 0.9        # Michelson-like amplitude around mean (keep <=1)
CYCLES_ACROSS_WIDTH = 8  # how many stripe cycles across the width
ORIENTATION_DEG = 0.0    # 0 = vertical stripes (vary in x); 90 = horizontal stripes


# ======================
# CORE: phase-linearized weight (inverse of paper Eq.(5))
# ======================
def phase_linearized_weight(u: np.ndarray, d: float, alpha: float) -> np.ndarray:
    """
    k_des(u) = d * u
    w(u) = alpha*sin(k) / (alpha*sin(k) + sin(d-k))
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


def make_grating(H: int, W: int, cycles: float, orientation_deg: float, phase: float,
                 mean: float, contrast: float) -> np.ndarray:
    """
    Return grayscale image in [0,1], shape (H,W).
    A sinusoidal grating: mean + (contrast/2)*sin(2π*f*coord + phase)
    """
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

    theta = np.deg2rad(orientation_deg)
    # rotated coordinate (in pixels)
    coord = xx * np.cos(theta) + yy * np.sin(theta)

    # spatial frequency in rad/pixel so that there are `cycles` periods across width
    # using W as reference scale
    omega = 2.0 * np.pi * (cycles / W)

    amp = 0.5 * contrast
    img = mean + amp * np.sin(omega * coord + phase)
    return np.clip(img, 0.0, 1.0)


def main():
    script_dir = Path(__file__).resolve().parent
    out_mp4 = script_dir / OUT_MP4_NAME

    # keyframe phases: phi_n = n*d (n=0..N_STEPS)
    phases = [n * d for n in range(N_STEPS + 1)]
    keyframes = [
        make_grating(H, W, CYCLES_ACROSS_WIDTH, ORIENTATION_DEG, ph, MEAN, CONTRAST)
        for ph in phases
    ]

    # Optional: save keyframes
    if SAVE_KEYFRAMES_PNG:
        kdir = script_dir / KEYFRAME_DIR_NAME
        kdir.mkdir(parents=True, exist_ok=True)
        for i, img in enumerate(keyframes):
            imageio.imwrite(kdir / f"stripe_{i:03d}.png", to_uint8(img))
        print("[INFO] Saved keyframes to:", kdir)

    # u grid for one transition (FPS frames)
    if USE_HALF_FRAME_OFFSET:
        u = (np.arange(FPS, dtype=np.float32) + 0.5) / FPS  # (0,1)
    else:
        u = np.arange(FPS, dtype=np.float32) / (FPS - 1)    # includes 0 and 1

    w = phase_linearized_weight(u, d, alpha)

# ===== u grid：包含 0 和 1，保证段首段尾严格对齐关键帧 =====
u_all = np.linspace(0.0, 1.0, FPS, endpoint=True).astype(np.float32)
w_all = phase_linearized_weight(u_all, d, alpha)

with imageio.get_writer(
    out_mp4,
    fps=FPS,
    codec="libx264",
    pixelformat="gray",
    macro_block_size=None,
    quality=8,
) as writer:

    for seg in range(N_STEPS):
        A = keyframes[seg]
        B = keyframes[seg + 1]

        # 从第二段开始跳过第一帧（u=0），避免与上一段末帧（u=1）重复
        start_f = 0 if seg == 0 else 1

        for f in range(start_f, FPS):
            ww = float(w_all[f])
            out01 = (1.0 - ww) * A + ww * B
            writer.append_data(to_uint8(out01))

    print("[DONE] Saved:", out_mp4)


if __name__ == "__main__":
    main()
