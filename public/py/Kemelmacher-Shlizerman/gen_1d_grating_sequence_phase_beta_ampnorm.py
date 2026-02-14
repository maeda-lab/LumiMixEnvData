import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio

# ======================
# OUTPUT CONFIG (Windows)
# ======================
from pathlib import Path
import datetime

BASE_DIR = Path(r"D:\vectionProject\public\camera3images")

# 新建子文件夹（自动时间戳，避免覆盖）
out_dir = BASE_DIR / f"phase_comp_demo_"
out_dir.mkdir(parents=True, exist_ok=True)

vid_ampn = out_dir / "cross_dissolve_1d_sequence_ABCD_ampnorm_vertical.mp4"
vid_phon = out_dir / "cross_dissolve_1d_sequence_ABCD_phase_only_vertical.mp4"


# ======================
# WEIGHTS (Phase-linearized)
# ======================
def phase_linearized_weight_screenshot(p: float, d: float, alpha: float = 1.0) -> float:
    """
    Screenshot-style formula:
        w_ph(t) = u(t) / ( sin(d) + u(t)*(1 - cos(d)) )
    where u(t) is typically proportional to tan(d*p).
    Here: u = alpha * tan(d*p).
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

# ======================
# AMPLITUDE (AmpNorm)
# ======================
def amplitude_of_mix(w: float, d: float) -> float:
    c = math.cos(d)
    A2 = (1 - w) ** 2 + w ** 2 + 2 * w * (1 - w) * c
    return math.sqrt(max(0.0, A2))

# ======================
# SIGNAL / IMAGE HELPERS
# ======================
def make_sin_signal(W: int, cycles: float, phase: float, amp: float = 1.0) -> np.ndarray:
    x = np.linspace(0, 2 * math.pi * cycles, W, endpoint=False)
    return amp * np.sin(x + phase)

def to_img_gray(signal_1d: np.ndarray, H: int, scale: float) -> np.ndarray:
    sig = np.clip(signal_1d / (scale if scale > 1e-8 else 1.0), -1.0, 1.0)
    g = ((sig * 0.5) + 0.5) * 255.0
    row = g.astype(np.uint8)
    img = np.repeat(row[None, :], H, axis=0)
    rgb = np.stack([img] * 3, axis=-1)
    return rgb

# Use default font only (fast, no filesystem scan)
_FONT = ImageFont.load_default()

def add_label(frame_rgb: np.ndarray, text: str, xy=(10, 10)) -> np.ndarray:
    im = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(im)
    x, y = xy
    draw.text((x + 1, y + 1), text, font=_FONT, fill=(0, 0, 0))
    draw.text((x, y), text, font=_FONT, fill=(255, 255, 255))
    return np.array(im)

# ======================
# VIDEO GENERATION
# ======================
def build_sequence_video(
    *,
    W=800,               # reduced for speed
    H=140,
    fps=30,
    cycles=10.0,
    d_step=0.9 * math.pi,
    num_images=11,       # 10 transitions => 10 seconds
    sec_per_transition=1.0,
    alpha=1.0,
    mode="beta",         # "beta" | "ampnorm" | "phase_only"
    beta=0.5,
    amp_eps=0.08,
    gain_cap=2.5,
):
    """
    Creates frames for a 1D grating sequence (A->B->C->... by phase stepping),
    with TOP = linear cross-dissolve, BOT = chosen method.
    """
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    total_transitions = num_images - 1
    total_seconds = total_transitions * sec_per_transition
    n_frames = int(total_seconds * fps)

    phases = [i * d_step for i in range(num_images)]

    # purely for display scaling (not normalization!)
    scale = 2.5 if mode == "ampnorm" else 1.2

    frames = []
    for frame_idx in range(n_frames):
        t_global = frame_idx / fps
        seg = min(int(t_global / sec_per_transition), total_transitions - 1)
        t_local = t_global - seg * sec_per_transition
        p = float(np.clip(t_local / sec_per_transition, 0.0, 1.0))

        phase0 = phases[seg]
        phase1 = phases[seg + 1]
        I0 = make_sin_signal(W=W, cycles=cycles, phase=phase0, amp=1.0)
        I1 = make_sin_signal(W=W, cycles=cycles, phase=phase1, amp=1.0)

        # TOP: Linear
        w_lin = p
        sig_top = (1 - w_lin) * I0 + w_lin * I1

        # BOT: Phase-linearized (screenshot formula)
        w_ph = phase_linearized_weight_screenshot(p, d_step, alpha=alpha)

        if mode == "beta":
            w_use = (1 - beta) * w_lin + beta * w_ph
            w_use = float(np.clip(w_use, 0.0, 1.0))
            sig_bot = (1 - w_use) * I0 + w_use * I1

        elif mode == "ampnorm":
            w_use = w_ph
            sig_bot = (1 - w_use) * I0 + w_use * I1
            A = amplitude_of_mix(w_use, d_step)
            gain = 1.0 / max(amp_eps, A)
            gain = min(gain, gain_cap)
            sig_bot = sig_bot * gain

        elif mode == "phase_only":
            # Phase-linearized only, NO amplitude normalization
            w_use = w_ph
            sig_bot = (1 - w_use) * I0 + w_use * I1

        else:
            raise ValueError(f"Unknown mode: {mode}")

        img_top = to_img_gray(sig_top, H=H, scale=scale)
        img_bot = to_img_gray(sig_bot, H=H, scale=scale)
        frame = np.concatenate([img_top, img_bot], axis=0)

        Aname = letters[seg]
        Bname = letters[seg + 1]

        frame = add_label(frame, f"TOP: Linear {Aname}->{Bname}  t={p:0.2f}", (10, 10))

        if mode == "beta":
            frame = add_label(frame, f"BOT: beta-mix(beta={beta:0.2f})  w={w_use:0.2f}", (10, H + 10))
        elif mode == "ampnorm":
            Aamp = amplitude_of_mix(w_use, d_step)
            gain = min(1.0 / max(amp_eps, Aamp), gain_cap)
            frame = add_label(
                frame,
                f"BOT: phase+ampnorm  w={w_use:0.2f}  A={Aamp:0.2f}  gain={gain:0.2f}",
                (10, H + 10),
            )
        else:  # phase_only
            frame = add_label(frame, f"BOT: phase-only (no norm)  w={w_use:0.2f}", (10, H + 10))

        frame = add_label(
            frame,
            f"cycles={cycles}  d={d_step/math.pi:0.2f}π  {sec_per_transition:.1f}s/transition  total={total_seconds:.0f}s",
            (10, 2 * H - 18),
        )

        frames.append(frame)

    return frames, fps

def write_mp4(path: Path, frames, fps: int):
    # macro_block_size=1 avoids auto-resize to multiples of 16
    writer = imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8, macro_block_size=1)
    for fr in frames:
        writer.append_data(fr)
    writer.close()

# ======================
# MAIN: generate 3 videos
# ======================
if __name__ == "__main__":
    # 1) beta mix (beta=0.5)
    frames_beta, fps = build_sequence_video(mode="beta", beta=0.5, alpha=1.0)

    # 2) phase + amplitude normalization
    frames_ampn, fps = build_sequence_video(mode="ampnorm", amp_eps=0.08, gain_cap=2.5, alpha=1.0)
    write_mp4(vid_ampn, frames_ampn, fps)

    # 3) phase-only (NO normalization)  <-- your requested new one
    frames_phon, fps = build_sequence_video(mode="phase_only", alpha=1.0)
    write_mp4(vid_phon, frames_phon, fps)

    print("Saved videos to:")
    print(" ", vid_ampn)
    print(" ", vid_phon)
