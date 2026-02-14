import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio


# ======================
# OUTPUT CONFIG (Windows)
# ======================
BASE_DIR = Path(r"D:\vectionProject\public\camera3images")
OUT_DIR = BASE_DIR / "phase_comp_demo_"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_MP4 = OUT_DIR / "cross_dissolve_1d_sequence_ABCD_ampnorm_vertical.mp4"


# ======================
# Phase-linearized weight (screenshot-style)
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
    u = alpha * math.tan(d * p)

    sinD = math.sin(d)
    one_minus_cosD = 1.0 - math.cos(d)

    denom = sinD + u * one_minus_cosD
    if abs(denom) < 1e-6:
        return p

    w = u / denom
    return float(np.clip(w, 0.0, 1.0))


# ======================
# AmpNorm amplitude
# ======================
def amplitude_of_mix(w: float, d: float) -> float:
    """
    A(w) = sqrt((1-w)^2 + w^2 + 2w(1-w)cos(d))
    """
    c = math.cos(d)
    A2 = (1.0 - w) ** 2 + w ** 2 + 2.0 * w * (1.0 - w) * c
    return math.sqrt(max(0.0, A2))


# ======================
# Grating helpers
# ======================
def make_sin_signal(W: int, cycles: float, phase: float, amp: float = 1.0) -> np.ndarray:
    x = np.linspace(0, 2 * math.pi * cycles, W, endpoint=False)
    return amp * np.sin(x + phase)


def signal_to_gray_rgb(signal_1d: np.ndarray, H: int, scale: float = 1.0) -> np.ndarray:
    """
    signal_1d: float in [-1,1] (approximately)
    -> uint8 RGB image (H,W,3)
    """
    sig = np.clip(signal_1d / (scale if scale > 1e-8 else 1.0), -1.0, 1.0)
    g = ((sig * 0.5) + 0.5) * 255.0
    row = g.astype(np.uint8)
    img = np.repeat(row[None, :], H, axis=0)
    return np.stack([img, img, img], axis=-1)


# Labels
_FONT = ImageFont.load_default()

def add_label(frame_rgb: np.ndarray, text: str, xy=(10, 10)) -> np.ndarray:
    im = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(im)
    x, y = xy
    draw.text((x + 1, y + 1), text, font=_FONT, fill=(0, 0, 0))
    draw.text((x, y), text, font=_FONT, fill=(255, 255, 255))
    return np.array(im)


# ======================
# Build SINGLE-video (ampnorm only, no top/bottom compare)
# ======================
def build_ampnorm_only_video(
    *,
    W=800,
    H=240,
    fps=30,
    cycles=10.0,
    d_step=0.9 * math.pi,
    num_images=11,          # 10 transitions
    sec_per_transition=1.0,
    alpha=1.0,
    amp_eps=0.08,
    gain_cap=2.5,
    draw_labels=True,
):
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    total_transitions = num_images - 1
    total_seconds = total_transitions * sec_per_transition
    n_frames = int(total_seconds * fps)

    phases = [i * d_step for i in range(num_images)]

    # Just for visualization (avoid saturation). You can set to 1.0 if you want stronger contrast.
    scale = 2.5

    frames = []
    for frame_idx in range(n_frames):
        t_global = frame_idx / fps
        seg = min(int(t_global / sec_per_transition), total_transitions - 1)
        t_local = t_global - seg * sec_per_transition
        p = float(np.clip(t_local / sec_per_transition, 0.0, 1.0))

        ph0 = phases[seg]
        ph1 = phases[seg + 1]
        I0 = make_sin_signal(W=W, cycles=cycles, phase=ph0, amp=1.0)
        I1 = make_sin_signal(W=W, cycles=cycles, phase=ph1, amp=1.0)

        # phase-linearized cross-dissolve
        w = phase_linearized_weight_screenshot(p, d_step, alpha=alpha)
        sig = (1.0 - w) * I0 + w * I1

        # AmpNorm
        A = amplitude_of_mix(w, d_step)
        gain = 1.0 / max(amp_eps, A)
        gain = min(gain, gain_cap)
        sig = sig * gain

        frame = signal_to_gray_rgb(sig, H=H, scale=scale)

        if draw_labels:
            Aname = letters[seg]
            Bname = letters[seg + 1]
            frame = add_label(frame, f"AmpNorm only  {Aname}->{Bname}  p={p:0.2f}  w={w:0.2f}", (10, 10))
            frame = add_label(frame, f"A={A:0.2f}  gain={gain:0.2f}  d={d_step/math.pi:0.2f}π  alpha={alpha:0.2f}", (10, 28))

        frames.append(frame)

    return frames, fps


def write_mp4(path: Path, frames, fps: int):
    writer = imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8, macro_block_size=1)
    for fr in frames:
        writer.append_data(fr)
    writer.close()


if __name__ == "__main__":
    frames, fps = build_ampnorm_only_video(
        W=800,
        H=240,
        fps=30,
        cycles=10.0,
        d_step=0.9 * math.pi,
        num_images=11,
        sec_per_transition=1.0,
        alpha=1.0,
        amp_eps=0.08,
        gain_cap=2.5,
        draw_labels=True,
    )
    write_mp4(OUT_MP4, frames, fps)
    print("Saved:", OUT_MP4)
