import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio


# ======================
# OUTPUT (Windows)
# ======================
OUT_DIR = Path(r"D:\vectionProject\public\camera3images") / "phase_comp_demo_"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_MP4 = OUT_DIR / "grating_phase_ampnorm_single_no_flicker.mp4"


# ======================
# VIDEO / GRATING CONFIG
# ======================
FPS = 60
SEC_PER_TRANSITION = 1.0   # 1s per A->B
NUM_IMAGES = 11            # A..K => 10 transitions => 10s
W = 1920                   # image width
H = 360                    # image height
CYCLES = 10.0              # spatial cycles across width
DRAW_LABELS = True

# Phase step between keyframes (matches your demo)
D_STEP = 0.9 * math.pi
ALPHA = 1.0

# ======================
# "No-flicker" AmpNorm parameters (use your current ones)
# ======================
AMP_EPS = 0.08
GAIN_CAP = 0.25       # NOTE: < 1 => no boosting, only attenuation
GAIN_GAMMA = 0.6
LAMBDA = 0.25         # partial gain application


# ======================
# Text overlay
# ======================
_FONT = ImageFont.load_default()

def add_label(frame_rgb: np.ndarray, text: str, xy=(10, 10)) -> np.ndarray:
    im = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(im)
    x, y = xy
    draw.text((x + 1, y + 1), text, font=_FONT, fill=(0, 0, 0))
    draw.text((x, y), text, font=_FONT, fill=(255, 255, 255))
    return np.array(im)


# ======================
# Phase-linearized weight (your screenshot form)
# ======================
def phase_linearized_weight_screenshot(p: float, d: float, alpha: float = 1.0) -> float:
    p = float(np.clip(p, 0.0, 1.0))
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0

    alpha = float(np.clip(alpha, 0.2, 5.0))
    u = alpha * math.tan(d * p)

    sinD = math.sin(d)
    one_minus_cosD = 1.0 - math.cos(d)

    denom = sinD + u * one_minus_cosD
    if abs(denom) < 1e-6:
        return p

    w = u / denom
    return float(np.clip(w, 0.0, 1.0))


def amplitude_of_mix(w: float, d: float) -> float:
    c = math.cos(d)
    A2 = (1.0 - w) ** 2 + w ** 2 + 2.0 * w * (1.0 - w) * c
    return math.sqrt(max(0.0, A2))


# ======================
# Grating helpers
# ======================
def make_grating_1d(W: int, cycles: float, phase: float) -> np.ndarray:
    """Return 1D sine in [-1,1]"""
    x = np.linspace(0, 2 * math.pi * cycles, W, endpoint=False)
    return np.sin(x + phase).astype(np.float32)

def to_u8_gray_from_signal(sig_1d: np.ndarray, H: int) -> np.ndarray:
    """
    sig_1d in [-1,1] -> image in uint8 RGB
    """
    sig = np.clip(sig_1d, -1.0, 1.0)
    g = ((sig * 0.5) + 0.5) * 255.0
    row = g.astype(np.uint8)
    img = np.repeat(row[None, :], H, axis=0)
    rgb = np.stack([img, img, img], axis=-1)
    return rgb


# ======================
# Main render
# ======================
def main():
    phases = [i * D_STEP for i in range(NUM_IMAGES)]
    frames_per_transition = int(round(FPS * SEC_PER_TRANSITION))
    total_transitions = NUM_IMAGES - 1
    total_frames = total_transitions * frames_per_transition

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
            ph0 = phases[seg]
            ph1 = phases[seg + 1]

            I0 = make_grating_1d(W, CYCLES, ph0)  # [-1,1]
            I1 = make_grating_1d(W, CYCLES, ph1)  # [-1,1]

            for k in range(frames_per_transition):
                p = k / (frames_per_transition - 1) if frames_per_transition > 1 else 1.0
                p = float(np.clip(p, 0.0, 1.0))

                # Phase-linearized cross-dissolve on SIGNAL domain
                w = phase_linearized_weight_screenshot(p, D_STEP, alpha=ALPHA)
                mix = (1.0 - w) * I0 + w * I1  # still ~[-1,1]

                # Theoretical amplitude + "no-flicker" gain shaping
                A = amplitude_of_mix(w, D_STEP)
                gain_raw = min(1.0 / max(AMP_EPS, A), GAIN_CAP)
                gain = gain_raw ** GAIN_GAMMA

                # Partial gain around mean (mean of sine is ~0)
                m = mix.mean(dtype=np.float32)
                gain_eff = 1.0 + LAMBDA * (gain - 1.0)
                out = m + gain_eff * (mix - m)

                frame = to_u8_gray_from_signal(out, H)

                if DRAW_LABELS:
                    frame = add_label(frame, f"seg={seg}  p={p:0.2f}  w={w:0.2f}", (10, 10))
                    frame = add_label(frame, f"A={A:0.3f}  gain_raw={gain_raw:0.3f}  gain={gain:0.3f}  gain_eff={gain_eff:0.3f}", (10, 28))
                    frame = add_label(
                        frame,
                        f"d={D_STEP/math.pi:0.2f}π  alpha={ALPHA:0.2f}  eps={AMP_EPS}  cap={GAIN_CAP}  gamma={GAIN_GAMMA}  lambda={LAMBDA}",
                        (10, H - 18),
                    )

                writer.append_data(frame)
                frame_idx += 1

                if frame_idx % FPS == 0:
                    print(f"  wrote {frame_idx}/{total_frames} frames")

    finally:
        writer.close()

    print("Saved:", OUT_MP4)


if __name__ == "__main__":
    main()
