import numpy as np
from pathlib import Path
from PIL import Image
import imageio.v2 as imageio
import cv2

# ======================
# CONFIG
# ======================
IMG_DIR  = Path(r"D:\vectionProject\public\camera2images")

FPS = 60
DT  = 1.0 / FPS

N = 10                 # 使用前 N 张 cam1_*.png
SIGMA_TIME = 0.60      # 固定时间高斯 sigma（你的“sweet spot”）

# 三档 lap 操控（建议先用这组“温和参数”，不容易把画面搞得很假）
LAP_CONDS = [
    ("lap_low_blur",     {"mode": "blur",    "sigma_sp": 2.0}),
    ("lap_base_none",    {"mode": "none"}),
    ("lap_high_sharpen", {"mode": "sharpen", "sigma_sp": 1.0, "amount": 2.0}),
]


# 平均亮度匹配增益限制（保留你原来的 fix）
GAIN_MIN, GAIN_MAX = 0.5, 1.5


# ======================
# Utils
# ======================
def load_first_n(img_dir: Path, n: int):
    paths = sorted(img_dir.glob("cam2_*.png"))
    if len(paths) < n:
        raise RuntimeError(f"需要至少 {n} 张 cam2_*.png 图片, 目前只有 {len(paths)} 张")
    paths = paths[:n]

    imgs = []
    for p in paths:
        im = Image.open(p).convert("L")
        arr = np.asarray(im, dtype=np.float32) / 255.0  # 0..1
        imgs.append(arr)
    return paths, imgs

def gaussian_weights(t: float, num_frames: int, sigma: float) -> np.ndarray:
    idx = np.arange(num_frames, dtype=np.float32)
    w = np.exp(-0.5 * ((idx - t) / sigma) ** 2)
    s = float(w.sum())
    if s > 1e-6:
        w /= s
    return w.astype(np.float32)

def apply_lap_control(img01: np.ndarray, cfg: dict) -> np.ndarray:
    """
    img01: float32 0..1
    """
    mode = cfg.get("mode", "none")
    if mode == "none":
        return img01

    im = img01.astype(np.float32)

    if mode == "blur":
        sigma_sp = float(cfg.get("sigma_sp", 0.9))
        out = cv2.GaussianBlur(im, (0, 0), sigmaX=sigma_sp, sigmaY=sigma_sp)
        return np.clip(out, 0.0, 1.0)

    if mode == "sharpen":
        sigma_sp = float(cfg.get("sigma_sp", 1.0))
        amount  = float(cfg.get("amount", 0.8))
        blur = cv2.GaussianBlur(im, (0, 0), sigmaX=sigma_sp, sigmaY=sigma_sp)
        out = im + amount * (im - blur)  # unsharp mask
        return np.clip(out, 0.0, 1.0)

    raise ValueError(f"Unknown mode: {mode}")

def mix_linear(imgs, t: float) -> np.ndarray:
    """top段：线性 A->B->C..."""
    N = len(imgs)
    if t >= N - 1:
        return imgs[-1]
    k = int(np.floor(t))
    u = t - k
    return (1.0 - u) * imgs[k] + u * imgs[k + 1]

def mix_time_gauss(imgs, t: float, sigma_time: float) -> np.ndarray:
    """bottom段：时间高斯混合（所有帧加权）"""
    N = len(imgs)
    ws = gaussian_weights(t, N, sigma=sigma_time)
    out = np.zeros_like(imgs[0], dtype=np.float32)
    for w, im in zip(ws, imgs):
        out += w * im
    return out

def match_mean_luminance(src01: np.ndarray, target_mean: float) -> np.ndarray:
    """把 src 的平均值拉到 target_mean（限制增益）"""
    m = float(src01.mean())
    if m <= 1e-6:
        return src01
    gain = target_mean / m
    gain = max(GAIN_MIN, min(gain, GAIN_MAX))
    return np.clip(src01 * gain, 0.0, 1.0)


# ======================
# Main
# ======================
def main():
    paths, imgs = load_first_n(IMG_DIR, N)
    print("Loaded:", [p.name for p in paths])

    T_start = 0.0
    T_end   = float(N - 1)
    num_steps = int(round((T_end - T_start) * FPS))

    for tag, cfg in LAP_CONDS:
        out_mp4 = IMG_DIR / f"cam1_timegauss_sig{SIGMA_TIME:.2f}_{tag}_vs_linear_fixbright.mp4"
        frames_out = []

        for step in range(num_steps):
            t = T_start + (step + 0.5) * DT

            # top: linear
            top_frame = mix_linear(imgs, t)

            # bottom: time-gauss (sigma fixed)
            bottom_frame = mix_time_gauss(imgs, t, SIGMA_TIME)

            # match mean brightness to top
            bottom_frame = match_mean_luminance(bottom_frame, float(top_frame.mean()))

            # control lap (only on bottom)
            bottom_frame = apply_lap_control(bottom_frame, cfg)

            # to uint8 + stack
            top_u8 = np.clip(top_frame * 255.0 + 0.5, 0, 255).astype(np.uint8)
            bot_u8 = np.clip(bottom_frame * 255.0 + 0.5, 0, 255).astype(np.uint8)
            frames_out.append(np.vstack([top_u8, bot_u8]))

        print("Writing:", out_mp4)
        imageio.mimwrite(out_mp4, frames_out, fps=FPS, codec="libx264", quality=8, macro_block_size=None)
        print("Done.")

if __name__ == "__main__":
    main()
