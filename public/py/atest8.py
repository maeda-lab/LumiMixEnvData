import math
from pathlib import Path
import numpy as np
import cv2
import imageio.v2 as imageio

# =========================
# CONFIG (edit these)
# =========================
IN_DIR  = Path(r"D:\vectionProject\public\camear1images")  # Unity exported frames (1Hz)
OUT_DIR = Path(r"D:\vectionProject\public\freq_test_videos_v2")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FPS = 60
SECONDS_PER_STEP = 1.0   # 1Hz: 1 second per transition

# Low-pass variants
LP_SIGMAS = [16.0, 64.0]     # try [16, 64] first
ULP_FACTORS = [32]           # downsample factor, try 32 (or 48)

# High-pass/band-pass variants (DoG)
DOG_PRESETS = [
    ("hi",  1.2,  4.0, 10.0),  # (name, sigma_small, sigma_large, vis_gain)
    ("mid", 2.0,  8.0,  6.0),
    ("low", 4.0, 16.0,  4.0),
]
BAND_VIS_BIAS = 0.5

# Macroblock-safe padding for H.264 (avoid imageio auto-resize)
PAD_TO_16 = True

# =========================
# sRGB <-> Linear
# =========================
def srgb_to_linear(x):
    x = np.clip(x, 0.0, 1.0)
    a = 0.055
    return np.where(x <= 0.04045, x / 12.92, ((x + a) / (1 + a)) ** 2.4)

def linear_to_srgb(x):
    x = np.clip(x, 0.0, 1.0)
    a = 0.055
    return np.where(x <= 0.0031308, x * 12.92, (1 + a) * (x ** (1 / 2.4)) - a)

def clamp01(x):
    return np.clip(x, 0.0, 1.0)

# =========================
# Padding to macroblock (16)
# =========================
def pad_to_16(img_lin: np.ndarray) -> np.ndarray:
    if not PAD_TO_16:
        return img_lin
    h, w, c = img_lin.shape
    new_h = ((h + 15) // 16) * 16
    new_w = ((w + 15) // 16) * 16
    pad_h = new_h - h
    pad_w = new_w - w
    if pad_h == 0 and pad_w == 0:
        return img_lin
    return np.pad(img_lin, ((0, pad_h), (0, pad_w), (0, 0)),
                  mode="constant", constant_values=0.0)

# =========================
# IO
# =========================
def read_rgb_linear(path: Path) -> np.ndarray:
    """Read image as linear RGB float32 in [0,1]."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return srgb_to_linear(rgb).astype(np.float32)

def write_frame(writer, img_lin: np.ndarray):
    """Write linear RGB [0,1] to video as sRGB uint8."""
    img_lin = pad_to_16(img_lin)
    img_srgb = linear_to_srgb(clamp01(img_lin))
    u8 = (img_srgb * 255.0 + 0.5).astype(np.uint8)
    writer.append_data(u8)

# =========================
# Filters in linear space
# =========================
def gaussian_blur_lin(img_lin: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(
        img_lin, (0, 0),
        sigmaX=float(sigma), sigmaY=float(sigma),
        borderType=cv2.BORDER_REFLECT
    )

def ultra_lowpass_lin(img_lin: np.ndarray, factor: int = 32) -> np.ndarray:
    """Very strong low-pass: downsample then upsample."""
    h, w, c = img_lin.shape
    fw = max(2, w // factor)
    fh = max(2, h // factor)
    small = cv2.resize(img_lin, (fw, fh), interpolation=cv2.INTER_AREA)
    up = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    return up

def dog_band_lin(img_lin: np.ndarray, sigma_small: float, sigma_large: float) -> np.ndarray:
    lo1 = gaussian_blur_lin(img_lin, sigma_small)
    lo2 = gaussian_blur_lin(img_lin, sigma_large)
    return lo1 - lo2  # signed

def band_to_vis_lin(band_lin: np.ndarray, gain: float, bias: float = 0.5) -> np.ndarray:
    return clamp01(bias + gain * band_lin)

# =========================
# Cross-dissolve video maker
# =========================
def make_dissolve_video(frames: list[Path], out_mp4: Path, kind: str, **kwargs):
    """
    kind:
      - "orig"
      - "lowpass_sigma"  (needs sigma)
      - "ultralowpass"   (needs factor)
      - "dog_vis"        (needs sigma_small, sigma_large, vis_gain)
    """
    writer = imageio.get_writer(
        str(out_mp4),
        fps=FPS,
        codec="libx264",
        quality=8
    )
    N = int(round(FPS * SECONDS_PER_STEP))

    try:
        for i in range(len(frames) - 1):
            A = read_rgb_linear(frames[i])
            B = read_rgb_linear(frames[i + 1])

            if kind == "orig":
                pass

            elif kind == "lowpass_sigma":
                sigma = float(kwargs["sigma"])
                A = gaussian_blur_lin(A, sigma)
                B = gaussian_blur_lin(B, sigma)

            elif kind == "ultralowpass":
                factor = int(kwargs["factor"])
                A = ultra_lowpass_lin(A, factor=factor)
                B = ultra_lowpass_lin(B, factor=factor)

            elif kind == "dog_vis":
                ss = float(kwargs["sigma_small"])
                sl = float(kwargs["sigma_large"])
                vg = float(kwargs["vis_gain"])
                A = band_to_vis_lin(dog_band_lin(A, ss, sl), vg, BAND_VIS_BIAS)
                B = band_to_vis_lin(dog_band_lin(B, ss, sl), vg, BAND_VIS_BIAS)

            else:
                raise ValueError(f"Unknown kind: {kind}")

            for k in range(N):
                p = k / (N - 1) if N > 1 else 1.0
                out = (1.0 - p) * A + p * B  # dissolve in LINEAR
                write_frame(writer, out)

        print("saved:", out_mp4)
    finally:
        writer.close()

def main():
    exts = (".png", ".jpg", ".jpeg")
    frames = sorted([p for p in IN_DIR.iterdir() if p.suffix.lower() in exts])
    if len(frames) < 2:
        raise RuntimeError("Not enough frames in input dir")

    # 0) Original
    make_dissolve_video(frames, OUT_DIR / "orig_lin.mp4", kind="orig")

    # 1) Low-pass (Gaussian)
    for sigma in LP_SIGMAS:
        make_dissolve_video(
            frames,
            OUT_DIR / f"lowpass_sigma{sigma:g}.mp4",
            kind="lowpass_sigma",
            sigma=sigma
        )

    # 2) Ultra-lowpass (downsample/upsample)
    for factor in ULP_FACTORS:
        make_dissolve_video(
            frames,
            OUT_DIR / f"ultralowpass_down{factor}.mp4",
            kind="ultralowpass",
            factor=factor
        )

    # 3) DoG band-pass (visualized)
    for name, ss, sl, vg in DOG_PRESETS:
        make_dissolve_video(
            frames,
            OUT_DIR / f"dog_{name}_ss{ss:g}_sl{sl:g}_gain{vg:g}.mp4",
            kind="dog_vis",
            sigma_small=ss,
            sigma_large=sl,
            vis_gain=vg
        )

    print("done:", OUT_DIR)

if __name__ == "__main__":
    main()
