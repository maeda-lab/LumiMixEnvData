import math
from pathlib import Path

import numpy as np
from PIL import Image
import imageio.v2 as imageio


# ======================
# CONFIG
# ======================
IMG_DIR   = Path(r"D:\vectionProject\public\camear1images")
IMG_GLOB  = "cam1_*.png"
OUT_MP4   = IMG_DIR / "cam1_compare_0_1_vs_0.1_0.9_gray.mp4"

FPS = 60
SECONDS_PER_STEP = 1.0
N_PER_STEP = int(round(FPS * SECONDS_PER_STEP))

# 截断范围：0.1~0.9
CLIP_MIN = 0.1
CLIP_MAX = 0.9  # 其实没用到，写着方便以后改


# ======================
# sRGB <-> Linear (灰度)
# ======================
def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    a = 0.055
    x = np.clip(x, 0, 1)
    return np.where(
        x <= 0.04045,
        x / 12.92,
        ((x + a) / (1 + a)) ** 2.4
    )


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    a = 0.055
    x = np.clip(x, 0, 1)
    return np.where(
        x <= 0.0031308,
        12.92 * x,
        (1 + a) * (x ** (1 / 2.4)) - a
    )


# ======================
# 读灰度图像
# ======================
def load_images_gray_sorted(img_dir: Path, pattern: str):
    paths = sorted(img_dir.glob(pattern))
    if not paths:
        raise RuntimeError(f"No images found in {img_dir} with pattern {pattern}")
    imgs = []
    for p in paths:
        im = Image.open(p).convert("L")  # 灰度
        arr = np.asarray(im, dtype=np.float32) / 255.0  # 0~1 sRGB
        imgs.append(arr)
    return paths, imgs


# ======================
# 主流程
# ======================
def main():
    paths, imgs_srgb = load_images_gray_sorted(IMG_DIR, IMG_GLOB)
    print(f"Loaded {len(imgs_srgb)} frames")

    # 转线性空间
    imgs_lin = [srgb_to_linear(im) for im in imgs_srgb]
    H, W = imgs_lin[0].shape
    print(f"Resolution: {W}x{H}")

    frames_out = []

    for i in range(len(imgs_lin) - 1):
        img1 = imgs_lin[i]
        img2 = imgs_lin[i + 1]
        print(f"Blending {paths[i].name} -> {paths[i+1].name}")

        for n in range(N_PER_STEP):
            # 归一化时间 p: 0..1
            p = (n + 0.5) / N_PER_STEP

            # ===== 上段：标准 0→1 线性混合 =====
            w_full = p
            top_lin = (1.0 - w_full) * img1 + w_full * img2

            # ===== 下段：截断 0.1→0.9 混合 =====
            w_clip = CLIP_MIN + (1.0 - 2 * CLIP_MIN) * p  # 0.1 + 0.8*p
            w_clip = float(np.clip(w_clip, 0.0, 1.0))
            bottom_lin = (1.0 - w_clip) * img1 + w_clip * img2

            # 回到 sRGB + 量化
            top_srgb = linear_to_srgb(top_lin)
            bottom_srgb = linear_to_srgb(bottom_lin)

            top_u8 = np.clip(top_srgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
            bottom_u8 = np.clip(bottom_srgb * 255.0 + 0.5, 0, 255).astype(np.uint8)

            # 垂直拼接成一张图：上段 = 0→1，下段 = 0.1→0.9
            stacked = np.vstack([top_u8, bottom_u8])

            frames_out.append(stacked)

    print("Writing video:", OUT_MP4)
    imageio.mimwrite(
        OUT_MP4,
        frames_out,
        fps=FPS,
        codec="libx264",
        quality=8,
        macro_block_size=None,  # 分辨率不是16倍数时避免报错
    )
    print("Done.")


if __name__ == "__main__":
    main()
