import math
from pathlib import Path
import numpy as np
import cv2
import imageio.v2 as imageio

# =========================
# CONFIG
# =========================
IN_DIR  = Path(r"D:\vectionProject\public\camear1images")
OUT_DIR = Path(r"D:\vectionProject\public\freq_test_videos")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FPS = 60
SECONDS_PER_STEP = 1.0

LP_SIGMA = 16.0  # 你验证过：sigma=64 快慢节奏消失，但仍能看见左移

# phase parameters (apply ONLY to high-frequency residual)
D_RAD = 0.40 * math.pi
ALPHA = 1.0

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
def rgb_lin_to_luma_lin(img_lin: np.ndarray) -> np.ndarray:
    # Rec.709 luma in LINEAR space
    return (0.2126 * img_lin[..., 0] +
            0.7152 * img_lin[..., 1] +
            0.0722 * img_lin[..., 2]).astype(np.float32)

def luma3(img_lin: np.ndarray) -> np.ndarray:
    g = rgb_lin_to_luma_lin(img_lin)              # (H,W)
    return np.repeat(g[..., None], 3, axis=2)     # (H,W,3)

# =========================
# IO
# =========================
def read_rgb_linear(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return srgb_to_linear(rgb).astype(np.float32)

def write_frame(writer, img_lin: np.ndarray):
    img_srgb = linear_to_srgb(clamp01(img_lin))
    u8 = (img_srgb * 255.0 + 0.5).astype(np.uint8)
    writer.append_data(u8)

# =========================
# filters
# =========================
def gaussian_blur_lin(img_lin: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(
        img_lin, (0, 0),
        sigmaX=float(sigma), sigmaY=float(sigma),
        borderType=cv2.BORDER_REFLECT
    )

def split_low_high(img_lin: np.ndarray, sigma_low: float):
    low = gaussian_blur_lin(img_lin, sigma_low)
    high = img_lin - low   # signed residual, contains edges/texture
    return low, high

# =========================
# phase weight
# =========================
def phase_linearized_weight_alpha(p: float, d: float, alpha: float = 1.0) -> float:
    p = float(np.clip(p, 0.0, 1.0))
    if p <= 0.0: return 0.0
    if p >= 1.0: return 1.0

    alpha = float(np.clip(alpha, 0.2, 5.0))
    k = d * p
    cosK = math.cos(k)
    sinK = math.sin(k)
    if abs(cosK) < 1e-6:
        cosK = 1e-6 * (1.0 if cosK >= 0 else -1.0)

    tanK = sinK / cosK
    sinD = math.sin(d)
    cosD = math.cos(d)

    denom = (sinD + tanK * alpha - tanK * cosD)
    if abs(denom) < 1e-8:
        return 1.0 if denom >= 0 else 0.0

    w = (tanK * alpha) / denom
    return float(np.clip(w, 0.0, 1.0))

# =========================
# video makers
# =========================
def make_lowpass_video(frames: list[Path], out_mp4: Path):
    writer = imageio.get_writer(str(out_mp4), fps=FPS, codec="libx264", quality=8, macro_block_size=1)

    N = int(round(FPS * SECONDS_PER_STEP))
    try:
        for i in range(len(frames) - 1):
            A = luma3(read_rgb_linear(frames[i]))
            B = luma3(read_rgb_linear(frames[i + 1]))
            A = gaussian_blur_lin(A, LP_SIGMA)
            B = gaussian_blur_lin(B, LP_SIGMA)

            for k in range(N):
                p = k / (N - 1) if N > 1 else 1.0
                out = (1 - p) * A + p * B
                write_frame(writer, out)
        print("saved:", out_mp4)
    finally:
        writer.close()

def make_split_video(frames: list[Path], out_mp4: Path, high_mode: str):
    """
    high_mode:
      - "lin"   : high-frequency mixed linearly (w=p)
      - "phase" : high-frequency mixed by phase weight (w=phase_linearized(p))
    low-frequency is ALWAYS linear (p).
    """
    writer = imageio.get_writer(str(out_mp4), fps=FPS, codec="libx264", quality=8, macro_block_size=1)
    N = int(round(FPS * SECONDS_PER_STEP))
    try:
        for i in range(len(frames) - 1):
            A = luma3(read_rgb_linear(frames[i]))
            B = luma3(read_rgb_linear(frames[i + 1]))


            lowA, highA = split_low_high(A, LP_SIGMA)
            lowB, highB = split_low_high(B, LP_SIGMA)

            for k in range(N):
                p = k / (N - 1) if N > 1 else 1.0

                # low always linear
                low_out = (1 - p) * lowA + p * lowB

                # high: choose weight
                if high_mode == "lin":
                    w = p
                elif high_mode == "phase":
                    w = phase_linearized_weight_alpha(p, d=D_RAD, alpha=ALPHA)
                else:
                    raise ValueError("high_mode must be 'lin' or 'phase'")

                high_out = (1 - w) * highA + w * highB

                out = low_out + high_out
                write_frame(writer, out)

        print("saved:", out_mp4)
    finally:
        writer.close()

def main():
    exts = (".png", ".jpg", ".jpeg")
    frames = sorted([p for p in IN_DIR.iterdir() if p.suffix.lower() in exts])
    if len(frames) < 2:
        raise RuntimeError("Not enough frames")

    # 1) lowpass only (your observation baseline)
    make_lowpass_video(frames, OUT_DIR / f"lowpass_sigma{LP_SIGMA:g}.mp4")

    # 2) split: high mixed linearly (should feel like original lin, but safer separated)
    make_split_video(frames, OUT_DIR / f"split_high_lin_low{LP_SIGMA:g}.mp4", high_mode="lin")

    # 3) split: high mixed with phase (THIS is the key test)
    make_split_video(
        frames,
        OUT_DIR / f"split_high_phase_low{LP_SIGMA:g}_d{D_RAD/math.pi:.2f}pi.mp4",
        high_mode="phase"
    )

if __name__ == "__main__":
    main()
