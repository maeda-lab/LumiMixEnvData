import math
import numpy as np
from pathlib import Path
from PIL import Image
import imageio.v2 as imageio

# ======================
# CONFIG
# ======================
IMG_DIR = Path(r"D:\vectionProject\public\camera2images")

OUT_LINEAR   = IMG_DIR / "cam2_linear.mp4"
OUT_TRI_NORM = IMG_DIR / "cam2_tri_norm_r1p5.mp4"
OUT_TRI_RAW  = IMG_DIR / "cam2_tri_raw_r1p5.mp4"
OUT_GAUSS    = IMG_DIR / "cam2_gauss_sigma0p6.mp4"

PATTERN = "cam2_*.png"   # 如果你的命名不是 cam2_000.png 这种，改这里

FPS = 60
DT = 1.0 / FPS
SECONDS_PER_STEP = 1.0   # ✅ 关键：每 1 秒推进到下一张原始图片索引

TRI_RADIUS = 1.5
GAUSS_SIGMA = 0.6

# 对比度分析更建议 True：在“线性光”中混合，再转回 sRGB 保存
INPUT_FRAMES_ARE_SRGB = True
BLEND_IN_LINEAR_LIGHT = True

# tri_raw 是否做亮度补偿（保持 raw 的“权重和不为1”的特性，就设 False）
TRI_RAW_MATCH_BRIGHTNESS_TO_LINEAR = False
GAIN_CLAMP = (0.5, 1.5)

# 编码设置
CODEC = "libx264"
QUALITY = 9  # 0~10 越大越好（imageio 的 quality）

# ======================
# sRGB <-> Linear
# ======================
def srgb_to_linear(x):
    x = np.clip(x, 0.0, 1.0)
    a = 0.055
    return np.where(x <= 0.04045, x / 12.92, ((x + a) / (1 + a)) ** 2.4)

def linear_to_srgb(x):
    x = np.clip(x, 0.0, 1.0)
    a = 0.055
    return np.where(x <= 0.0031308, 12.92 * x, (1 + a) * (x ** (1 / 2.4)) - a)

def to_u8(frame_lin_or_srgb):
    x = frame_lin_or_srgb
    if INPUT_FRAMES_ARE_SRGB and BLEND_IN_LINEAR_LIGHT:
        x = linear_to_srgb(x)
    x = np.clip(x, 0.0, 1.0)
    return (x * 255.0 + 0.5).astype(np.uint8)

# ======================
# Load images
# ======================
def load_images(img_dir: Path, pattern: str):
    paths = sorted(img_dir.glob(pattern))
    if len(paths) < 2:
        raise RuntimeError(f"Need >=2 images: {img_dir / pattern}, got {len(paths)}")
    imgs = []
    for p in paths:
        arr = np.asarray(Image.open(p).convert("L"), dtype=np.float32) / 255.0
        if INPUT_FRAMES_ARE_SRGB and BLEND_IN_LINEAR_LIGHT:
            arr = srgb_to_linear(arr)
        imgs.append(arr)
    stack = np.stack(imgs, axis=0).astype(np.float32)  # [N,H,W]
    return paths, stack

# ======================
# Weight functions
# ======================
def w_linear_two(t_idx: float, N: int) -> np.ndarray:
    if t_idx <= 0:
        w = np.zeros(N, np.float32); w[0] = 1.0; return w
    if t_idx >= N - 1:
        w = np.zeros(N, np.float32); w[-1] = 1.0; return w
    k = int(np.floor(t_idx))
    u = float(t_idx - k)
    w = np.zeros(N, np.float32)
    w[k] = 1.0 - u
    w[k+1] = u
    return w

def w_tri(t_idx: float, N: int, radius: float, normalize: bool) -> np.ndarray:
    idx = np.arange(N, dtype=np.float32)
    phi = np.maximum(0.0, 1.0 - np.abs(idx - t_idx) / float(radius)).astype(np.float32)
    if normalize:
        s = float(phi.sum())
        if s > 1e-8:
            phi /= s
    return phi

def w_gauss(t_idx: float, N: int, sigma: float, normalize: bool=True) -> np.ndarray:
    idx = np.arange(N, dtype=np.float32)
    w = np.exp(-0.5 * ((idx - t_idx) / float(sigma)) ** 2).astype(np.float32)
    if normalize:
        s = float(w.sum())
        if s > 1e-8:
            w /= s
    return w

def synth(stack: np.ndarray, w: np.ndarray) -> np.ndarray:
    # stack: [N,H,W], w: [N]
    return np.tensordot(w.astype(np.float32), stack, axes=(0, 0)).astype(np.float32)

# ======================
# Main
# ======================
def main():
    paths, stack = load_images(IMG_DIR, PATTERN)
    N, H, W = stack.shape
    print("Loaded:", N, "images, size:", W, "x", H)
    print("First/last:", paths[0].name, "->", paths[-1].name)

    total_sec = float(N - 1) * SECONDS_PER_STEP
    num_frames = int(round(total_sec * FPS))
    print("Output duration:", total_sec, "sec  | frames:", num_frames)

    out_lin = []
    out_tn  = []
    out_tr  = []
    out_g   = []

    for i in range(num_frames):
        t_sec = (i + 0.5) * DT
        t_idx = float(np.clip(t_sec / SECONDS_PER_STEP, 0.0, N - 1.0))

        # weights
        wl = w_linear_two(t_idx, N)
        wtn = w_tri(t_idx, N, TRI_RADIUS, normalize=True)
        wtr = w_tri(t_idx, N, TRI_RADIUS, normalize=False)
        wg = w_gauss(t_idx, N, GAUSS_SIGMA, normalize=True)

        # synth
        f_lin = synth(stack, wl)
        f_tn  = synth(stack, wtn)
        f_tr  = synth(stack, wtr)
        f_g   = synth(stack, wg)

        # 可选：让 tri_raw 的平均亮度匹配 linear（不想破坏 raw 特性就关掉）
        if TRI_RAW_MATCH_BRIGHTNESS_TO_LINEAR:
            m_lin = float(f_lin.mean())
            m_tr  = float(f_tr.mean())
            if m_tr > 1e-6:
                gain = m_lin / m_tr
                gain = max(GAIN_CLAMP[0], min(GAIN_CLAMP[1], gain))
                f_tr = np.clip(f_tr * gain, 0.0, 1.0)

        out_lin.append(to_u8(f_lin))
        out_tn.append(to_u8(f_tn))
        out_tr.append(to_u8(f_tr))
        out_g.append(to_u8(f_g))

        if (i % 600) == 0 and i > 0:
            print("  generated", i, "/", num_frames)

    def write_mp4(out_path: Path, frames):
        print("Writing:", out_path)
        imageio.mimwrite(
            out_path,
            frames,
            fps=FPS,
            codec=CODEC,
            quality=QUALITY,
            macro_block_size=None,
        )

    write_mp4(OUT_LINEAR, out_lin)
    write_mp4(OUT_TRI_NORM, out_tn)
    write_mp4(OUT_TRI_RAW, out_tr)
    write_mp4(OUT_GAUSS, out_g)

    print("DONE.")

if __name__ == "__main__":
    main()
