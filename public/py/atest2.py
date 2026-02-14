import math
import re
from pathlib import Path

import numpy as np
from PIL import Image
import imageio.v2 as imageio


# =========================
# Phase-linearized weight
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
# sRGB <-> Linear (IMPORTANT)
# =========================
def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    a = 0.055
    return np.where(x <= 0.04045, x / 12.92, ((x + a) / (1 + a)) ** 2.4)

def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    a = 0.055
    return np.where(x <= 0.0031308, x * 12.92, (1 + a) * (x ** (1 / 2.4)) - a)

def luminance_linear(rgb_lin: np.ndarray) -> np.ndarray:
    return 0.2126 * rgb_lin[..., 0] + 0.7152 * rgb_lin[..., 1] + 0.0722 * rgb_lin[..., 2]

def contrast_std(rgb_lin: np.ndarray) -> float:
    lum = luminance_linear(rgb_lin)
    return float(lum.std())


# =========================
# Image IO
# =========================
def load_rgb_linear(path: Path, size=None) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    if size is not None and im.size != size:
        im = im.resize(size, resample=Image.BICUBIC)
    arr = np.asarray(im).astype(np.float32) / 255.0
    return srgb_to_linear(arr)

_num_pat = re.compile(r"(\d+)")
def numeric_sort_key(p: Path):
    # sort by last number in filename (cam1_000.png -> 0)
    nums = _num_pat.findall(p.stem)
    return int(nums[-1]) if nums else 10**18

def collect_images(folder: Path, prefix: str = "cam1_", ext: str = ".png"):
    # robust: find all png then filter case-insensitively
    if not folder.exists():
        return []
    all_png = list(folder.glob("*.png"))
    prefix_l = prefix.lower()
    ext_l = ext.lower()
    files = [p for p in all_png if p.name.lower().startswith(prefix_l) and p.suffix.lower() == ext_l]
    files.sort(key=numeric_sort_key)
    return files


# =========================
# Build video (mode="ampnorm")
# =========================
def build_ampnorm_video_from_images(
    img_dir: str,
    out_mp4: str = "cam1_ampnorm.mp4",
    fps: int = 30,
    sec_per_transition: float = 1.0,
    d_step: float = 0.9 * math.pi,
    alpha: float = 1.0,
    amp_eps: float = 0.02,
    gain_cap: float = 2.5,
):
    # Try both spellings automatically
    candidates = [
        Path(img_dir),
        Path(r"D:\vectionProject\public\camear1images"),
        Path(r"D:\vectionProject\public\camera1images"),
    ]

    folder = None
    files = []
    for c in candidates:
        fs = collect_images(c, prefix="cam1_", ext=".png")
        if len(fs) >= 2:
            folder = c
            files = fs
            break

    print("=== DEBUG ===")
    for c in candidates:
        print("candidate:", str(c), "exists:", c.exists(), "cam1_png_count:", len(collect_images(c)))
    print("=============")

    if folder is None or len(files) < 2:
        raise RuntimeError(
            "Not enough images. Make sure you have at least 2 files like cam1_000.png, cam1_001.png in the folder.\n"
            "Checked candidates:\n"
            + "\n".join([f"- {str(c)}" for c in candidates])
        )

    # Output path (save in the same folder by default)
    out_path = Path(out_mp4)
    if not out_path.is_absolute():
        out_path = folder / out_path

    # Size based on first image
    first = Image.open(files[0]).convert("RGB")
    size = first.size  # (W,H)

    # Preload first frame stats
    prev_lin = load_rgb_linear(files[0], size=size)
    prev_std = contrast_std(prev_lin)

    frames_per_transition = int(round(sec_per_transition * fps))
    total_transitions = len(files) - 1
    total_frames = total_transitions * frames_per_transition

    writer = imageio.get_writer(
        str(out_path),
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1
    )

    try:
        for i in range(total_transitions):
            next_lin = load_rgb_linear(files[i + 1], size=size)
            next_std = contrast_std(next_lin)

            for f in range(frames_per_transition):
                # 0..1 within this 1s transition
                p = f / max(1, (frames_per_transition - 1))
                w = phase_linearized_weight_alpha(p, d_step, alpha=alpha)

                # 1) linear blend
                mix_lin = (1.0 - w) * prev_lin + w * next_lin

                # 2) target contrast: interpolate endpoint variances (stable)
                ref_var = (1.0 - w) * (prev_std ** 2) + w * (next_std ** 2)
                ref_std = math.sqrt(max(0.0, ref_var))

                # 3) current contrast
                cur_std = contrast_std(mix_lin)

                # 4) gain around mean (prevents "whole image gets white")
                gain = ref_std / max(amp_eps, cur_std)
                gain = min(gain, gain_cap)

                mean_rgb = mix_lin.mean(axis=(0, 1), keepdims=True)
                ampn_lin = mean_rgb + (mix_lin - mean_rgb) * gain

                # 5) clamp + linear->sRGB
                ampn_lin = np.clip(ampn_lin, 0.0, 1.0)
                ampn_srgb = np.clip(linear_to_srgb(ampn_lin), 0.0, 1.0)
                out_u8 = (ampn_srgb * 255.0 + 0.5).astype(np.uint8)

                writer.append_data(out_u8)

            prev_lin = next_lin
            prev_std = next_std

    finally:
        writer.close()

    print(f"Saved: {out_path}")
    print(f"Images used: {len(files)}  transitions: {total_transitions}  frames: {total_frames}  fps: {fps}")


if __name__ == "__main__":
    # ✅ Put your real path here (your screenshot shows "camear1images")
    img_dir = r"D:\vectionProject\public\camear1images"

    build_ampnorm_video_from_images(
        img_dir=img_dir,
        out_mp4="cam1_ampnorm.mp4",
        fps=30,
        sec_per_transition=1.0,
        d_step=0.9 * math.pi,
        alpha=1.0,
        amp_eps=0.02,
        gain_cap=2.5,
    )
