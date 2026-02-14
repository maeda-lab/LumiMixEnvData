import re
import math
from pathlib import Path

import numpy as np
from PIL import Image
import imageio.v2 as imageio


# ======================
# CONFIG
# ======================
IMG_DIR = Path(r"D:\vectionProject\public\camera2images")
PATTERN = "cam2_*.png"          # 你的 cam2 图片序列（也可以 jpg）
OUT_MP4 = IMG_DIR / "cam2_gauss_sigma0p6.mp4"

FPS = 60
SECONDS_PER_STEP = 1.0         # 1Hz：每 1 秒切到下一张关键帧
SIGMA_SEC = 0.6 * SECONDS_PER_STEP   # ✅ sigma0p6 的来源

# 混合窗口：只用邻近 3 张（i-1, i, i+1）
USE_THREE_FRAMES = True

# 编码参数
CRF = 18                       # 越小越清晰（文件更大）
PRESET = "medium"


# ======================
# utils
# ======================
def natural_key(p: Path):
    # 让 cam2_1.png cam2_2.png ... cam2_10.png 按数字顺序排序
    m = re.search(r"(\d+)", p.stem)
    return int(m.group(1)) if m else p.stem

def load_images(paths):
    imgs = []
    base_size = None
    for p in paths:
        im = Image.open(p).convert("RGB")
        if base_size is None:
            base_size = im.size
        elif im.size != base_size:
            im = im.resize(base_size, Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0
        imgs.append(arr)
    return np.stack(imgs, axis=0)  # [N,H,W,3] float32 0..1

def gaussian(x, sigma):
    # x: seconds
    return math.exp(-0.5 * (x / sigma) ** 2)


# ======================
# main
# ======================
def main():
    paths = sorted(IMG_DIR.glob(PATTERN), key=natural_key)
    if len(paths) < 2:
        raise RuntimeError(f"need >=2 images, got {len(paths)} in {IMG_DIR}")

    imgs = load_images(paths)
    n = imgs.shape[0]

    # 关键帧时间：第 i 张在 i*step 秒
    key_times = np.arange(n, dtype=np.float32) * SECONDS_PER_STEP

    # 输出时长：从第0张过渡到第(n-1)张
    duration = (n - 1) * SECONDS_PER_STEP
    n_out = int(round(duration * FPS)) + 1

    writer = imageio.get_writer(
        str(OUT_MP4),
        fps=FPS,
        codec="libx264",
        ffmpeg_params=["-crf", str(CRF), "-preset", PRESET, "-pix_fmt", "yuv420p"],
    )

    try:
        for k in range(n_out):
            t = k / FPS

            # 找最近的关键帧索引（中心）
            i_center = int(round(t / SECONDS_PER_STEP))
            i_center = max(0, min(n - 1, i_center))

            if USE_THREE_FRAMES:
                idxs = [i_center - 1, i_center, i_center + 1]
            else:
                idxs = [i_center, i_center + 1]

            idxs = [i for i in idxs if 0 <= i < n]

            # 计算高斯权重（按与各关键帧时间差）
            ws = []
            for i in idxs:
                dt = t - float(key_times[i])
                ws.append(gaussian(dt, SIGMA_SEC))
            ws = np.array(ws, dtype=np.float32)

            # 归一化
            s = float(ws.sum())
            if s <= 1e-12:
                ws[:] = 0.0
                ws[idxs.index(i_center)] = 1.0
            else:
                ws /= s

            # 加权混合
            out = np.zeros_like(imgs[0], dtype=np.float32)
            for w, i in zip(ws, idxs):
                out += w * imgs[i]

            out_u8 = (np.clip(out, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
            writer.append_data(out_u8)

        print("saved:", OUT_MP4)

    finally:
        writer.close()


if __name__ == "__main__":
    main()
