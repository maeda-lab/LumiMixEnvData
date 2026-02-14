import math
from pathlib import Path

import numpy as np
from PIL import Image
import imageio.v2 as imageio
import matplotlib.pyplot as plt

# ======================
# CONFIG (edit these)
# ======================
IMG_DIR  = Path(r"D:\vectionProject\public\camear1images")
pattern  = "cam1_*.png"                   # 改成你的文件名模式
OUT_MP4  = IMG_DIR / "gaussian_blend.mp4"
OUT_PNG  = IMG_DIR / "gaussian_weights.png"

FPS = 60
SEC_PER_TRANSITION = 1.0                  # 每两张图过渡时长（秒）

GAUSSIAN_SIGMA = 0.22                     # 关键参数：0.15~0.35 常用
MIN_WEIGHT = 0.0                          # 可选：例如 0.02 表示任何一张至少2%

# 如果你的输入是正常截图/录屏帧，一般 True（做 sRGB->Linear->sRGB）
INPUT_FRAMES_ARE_SRGB = True


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
    return np.where(x <= 0.0031308, x * 12.92, (1 + a) * (x ** (1/2.4)) - a)


# ======================
# Two-image Gaussian normalized weights
# w_top = g1/(g0+g1), g0 centered at p=0, g1 centered at p=1
# ======================
def gaussian2_weights(p, sigma=0.22, min_weight=0.0):
    p = float(np.clip(p, 0.0, 1.0))
    sigma = max(float(sigma), 1e-6)
    s2 = 2.0 * sigma * sigma

    g0 = math.exp(-(p * p) / s2)          # bottom(prev)
    d = 1.0 - p
    g1 = math.exp(-(d * d) / s2)          # top(next)

    denom = g0 + g1
    if denom < 1e-12:
        w_top = 0.5
    else:
        w_top = g1 / denom

    if min_weight > 0.0:
        w_top = float(np.clip(w_top, min_weight, 1.0 - min_weight))

    w_bottom = 1.0 - w_top
    return w_bottom, w_top


# ======================
# IO
# ======================
def load_gray_float01(path: Path):
    im = Image.open(path).convert("L")
    arr = np.asarray(im, dtype=np.float32) / 255.0
    return arr

def to_uint8(img01):
    return (np.clip(img01, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)

def stack_gray_to_rgb_u8(gray_u8):
    # imageio 写 mp4 通常更稳用 RGB 3通道
    return np.stack([gray_u8, gray_u8, gray_u8], axis=-1)


# ======================
# Main
# ======================
def main():
    paths = sorted(IMG_DIR.glob(pattern))
    if len(paths) < 2:
        raise RuntimeError(f"Need at least 2 images. Found {len(paths)} in {IMG_DIR} with {pattern}")

    # 画权重曲线
    p = np.linspace(0, 1, 1001)
    wb = np.zeros_like(p)
    wt = np.zeros_like(p)
    for i, pi in enumerate(p):
        wb[i], wt[i] = gaussian2_weights(pi, GAUSSIAN_SIGMA, MIN_WEIGHT)

    plt.figure(figsize=(9, 5))
    plt.plot(p, wb, label="Bottom (previous)")
    plt.plot(p, wt, label="Top (next)")
    plt.xlabel("p (linear progress 0→1)")
    plt.ylabel("weight")
    plt.title(f"Gaussian normalized weights (sigma={GAUSSIAN_SIGMA}, min_weight={MIN_WEIGHT})")
    plt.ylim(-0.02, 1.02)
    plt.xlim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=200)
    plt.close()

    # 写视频
    n_per = int(round(FPS * SEC_PER_TRANSITION))
    writer = imageio.get_writer(str(OUT_MP4), fps=FPS, codec="libx264", quality=8)

    try:
        for k in range(len(paths) - 1):
            a = load_gray_float01(paths[k])
            b = load_gray_float01(paths[k + 1])

            # 可选：转到 linear 空间混合，避免 gamma 偏差
            if INPUT_FRAMES_ARE_SRGB:
                a_lin = srgb_to_linear(a)
                b_lin = srgb_to_linear(b)
            else:
                a_lin = a
                b_lin = b

            for f in range(n_per):
                p01 = f / (n_per - 1) if n_per > 1 else 1.0
                w_bottom, w_top = gaussian2_weights(p01, GAUSSIAN_SIGMA, MIN_WEIGHT)

                out_lin = a_lin * w_bottom + b_lin * w_top

                if INPUT_FRAMES_ARE_SRGB:
                    out = linear_to_srgb(out_lin)
                else:
                    out = out_lin

                frame_u8 = to_uint8(out)
                writer.append_data(stack_gray_to_rgb_u8(frame_u8))

        print("Saved:", OUT_MP4)
        print("Saved:", OUT_PNG)

    finally:
        writer.close()


if __name__ == "__main__":
    main()
