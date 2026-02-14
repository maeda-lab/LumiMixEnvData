import math
from pathlib import Path
import numpy as np
import cv2
import imageio.v2 as imageio
import matplotlib.pyplot as plt

# =========================
# CONFIG (edit these)
# =========================
IN_DIR  = Path(r"D:\vectionProject\public\camear1images")
OUT_DIR = Path(r"D:\vectionProject\public\freq_test_videos")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FPS = 60
SECONDS_PER_STEP = 1.0
N = int(round(FPS * SECONDS_PER_STEP))  # frames per 1-second segment

# High-pass / band-pass (DoG) + visualization
DOG_SIGMA_SMALL = 1.2
DOG_SIGMA_LARGE = 4.0
BAND_VIS_GAIN   = 10.0
BAND_VIS_BIAS   = 0.5

# Change-energy analysis
MAX_STEPS_TO_ANALYZE = 60  # increase if needed

# p(t) setting
GAMMA = 0.5   # <-- you are using this now (gamma<1 gives U-shape)

# =========================
# sRGB <-> Linear + luminance
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
    return (0.2126 * img_lin[..., 0] +
            0.7152 * img_lin[..., 1] +
            0.0722 * img_lin[..., 2]).astype(np.float32)

def luma3_from_rgb_lin(img_lin: np.ndarray) -> np.ndarray:
    g = rgb_lin_to_luma_lin(img_lin)
    return np.repeat(g[..., None], 3, axis=2)

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
# DoG band-pass in linear luminance
# =========================
def gaussian_blur_lin(img_lin: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(
        img_lin, (0, 0),
        sigmaX=float(sigma), sigmaY=float(sigma),
        borderType=cv2.BORDER_REFLECT
    )

def dog_band_lin(img_lin: np.ndarray, sigma_small: float, sigma_large: float) -> np.ndarray:
    lo1 = gaussian_blur_lin(img_lin, sigma_small)
    lo2 = gaussian_blur_lin(img_lin, sigma_large)
    return lo1 - lo2  # signed

def band_to_vis_lin(band_lin: np.ndarray, gain: float, bias: float) -> np.ndarray:
    return clamp01(bias + gain * band_lin)

# =========================
# p(t) mappings
# =========================
def p_linear(t: float) -> float:
    return float(np.clip(t, 0.0, 1.0))

def p_fast_through_middle(t: float, gamma: float = 2.5) -> float:
    """
    gamma>1: dp peaks in the middle (arch shape)
    gamma<1: dp peaks at the ends (U shape)
    """
    t = float(np.clip(t, 0.0, 1.0))
    s = 0.5 - 0.5 * math.cos(math.pi * t)  # smooth 0->1
    a = s**gamma
    b = (1.0 - s)**gamma
    return float(a / (a + b + 1e-12))

def debug_p_profile(gamma: float, N: int):
    ps = np.array([p_fast_through_middle(k/(N-1), gamma=gamma) for k in range(N)], dtype=float)
    dps = np.diff(ps)
    print("==== p(t) debug ====")
    print(f"gamma = {gamma}")
    print("p[0,10,20,30,40,59] =", [round(ps[i], 6) for i in [0,10,20,30,40,59]])
    print("dp min/max =", float(dps.min()), float(dps.max()), "argmax(dp) =", int(np.argmax(dps)))
    print("Interpretation:",
          "middle-peaked (arch)" if 25 <= int(np.argmax(dps)) <= 34 else "end-peaked (U-ish)")
    print("====================")

# =========================
# Video synthesis (highpass only)
# =========================
def make_highpass_video(frames: list[Path], out_mp4: Path, p_func):
    writer = imageio.get_writer(
        str(out_mp4),
        fps=FPS,
        codec="libx264",
        quality=8,
        macro_block_size=1
    )
    try:
        for i in range(len(frames) - 1):
            A_rgb = read_rgb_linear(frames[i])
            B_rgb = read_rgb_linear(frames[i + 1])

            # unify to grayscale luminance (3ch)
            A = luma3_from_rgb_lin(A_rgb)
            B = luma3_from_rgb_lin(B_rgb)

            # DoG
            A_band = dog_band_lin(A, DOG_SIGMA_SMALL, DOG_SIGMA_LARGE)
            B_band = dog_band_lin(B, DOG_SIGMA_SMALL, DOG_SIGMA_LARGE)

            # visualize
            A_vis = band_to_vis_lin(A_band, BAND_VIS_GAIN, BAND_VIS_BIAS)
            B_vis = band_to_vis_lin(B_band, BAND_VIS_GAIN, BAND_VIS_BIAS)

            for k in range(N):
                t = k / (N - 1) if N > 1 else 1.0
                p = p_func(t)
                out = (1.0 - p) * A_vis + p * B_vis
                write_frame(writer, out)

        print("saved:", out_mp4)
    finally:
        writer.close()

# =========================
# Analysis: per-step internal change energy
# =========================
def per_step_internal_curve(mp4_path: Path):
    cap = cv2.VideoCapture(str(mp4_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = FPS

    diffs = []
    prev = None
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        if prev is not None:
            diffs.append(float(np.mean(np.abs(gray - prev))))
        prev = gray
    cap.release()

    diffs = np.array(diffs, dtype=np.float32)
    step_len = N - 1

    num_steps = len(diffs) // step_len
    num_steps = min(num_steps, MAX_STEPS_TO_ANALYZE)
    if num_steps <= 0:
        raise RuntimeError("Not enough frames to segment steps")

    M = diffs[:num_steps * step_len].reshape(num_steps, step_len)
    mean_curve = M.mean(axis=0)
    std_curve  = M.std(axis=0)
    return mean_curve, std_curve, fps

def main():
    exts = (".png", ".jpg", ".jpeg")
    frames = sorted([p for p in IN_DIR.iterdir() if p.suffix.lower() in exts])
    if len(frames) < 2:
        raise RuntimeError("Not enough frames")

    # Debug p(t) once
    debug_p_profile(GAMMA, N)

    out_lin  = OUT_DIR / "highpass_linear_p.mp4"
    out_gam  = OUT_DIR / f"highpass_gamma{GAMMA:.2f}.mp4"

    # 1) baseline (linear p)
    make_highpass_video(frames, out_lin, p_linear)

    # 2) gamma p(t)
    make_highpass_video(frames, out_gam, lambda t: p_fast_through_middle(t, gamma=GAMMA))

    # analyze curves
    m1, s1, _ = per_step_internal_curve(out_lin)
    m2, s2, _ = per_step_internal_curve(out_gam)

    x = np.arange(len(m1))
    plt.figure()
    plt.plot(x, m1, label="linear p(t)")
    plt.plot(x, m2, label=f"p_fast_through_middle, gamma={GAMMA:.2f}")
    plt.fill_between(x, m1 - s1, m1 + s1, alpha=0.15)
    plt.fill_between(x, m2 - s2, m2 + s2, alpha=0.15)
    plt.title("Per-step internal change energy (mean±std over steps)")
    plt.xlabel("within-step frame index (0..N-2)")
    plt.ylabel("mean |diff|")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
