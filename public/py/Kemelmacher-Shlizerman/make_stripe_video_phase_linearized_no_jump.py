import numpy as np
import imageio.v2 as imageio
from pathlib import Path


# ======================
# CONFIG
# ======================
OUT_MP4_NAME = "stripe_phase_linearized_d0p9pi_0_5s_boundary_hold.mp4"

FPS = 60
N_STEPS = 5            # 0..5s => 5 transitions, 6 keyframes

d = 0.9 * np.pi        # phase shift per step
alpha = 1.0            # amplitude ratio

# Stripe stimulus (grating)
W = 960
H = 540
MEAN = 0.5
CONTRAST = 0.9
CYCLES_ACROSS_WIDTH = 8
ORIENTATION_DEG = 0.0  # 0=vertical stripes


# ======================
# CORE: phase-linearized weight
# ======================
def phase_linearized_weight(u: np.ndarray, d: float, alpha: float) -> np.ndarray:
    """
    k_des(u) = d*u
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
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    theta = np.deg2rad(orientation_deg)
    coord = xx * np.cos(theta) + yy * np.sin(theta)

    omega = 2.0 * np.pi * (cycles / W)   # rad/pixel
    amp = 0.5 * contrast
    img = mean + amp * np.sin(omega * coord + phase)
    return np.clip(img, 0.0, 1.0)


def main():
    script_dir = Path(__file__).resolve().parent
    out_mp4 = script_dir / OUT_MP4_NAME

    # Keyframes: phi_n = n*d, n=0..N_STEPS
    phases = [n * d for n in range(N_STEPS + 1)]
    keyframes = [
        make_grating(H, W, CYCLES_ACROSS_WIDTH, ORIENTATION_DEG, ph, MEAN, CONTRAST)
        for ph in phases
    ]

    # IMPORTANT: include 0 and 1 (endpoint=True)
    # This makes each segment start exactly at A and end exactly at B.
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

            # DO NOT skip the first frame (u=0).
            # This guarantees no sudden change at boundaries (may repeat 1 frame at each boundary).
            for f in range(FPS):
                ww = float(w_all[f])
                out01 = (1.0 - ww) * A + ww * B
                writer.append_data(to_uint8(out01))

        # 不额外追加最后关键帧：因为最后一段末帧（u=1）已经是它了

    print("[DONE] Saved:", out_mp4)


if __name__ == "__main__":
    main()
