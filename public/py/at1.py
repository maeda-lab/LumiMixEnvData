import numpy as np
from pathlib import Path
from PIL import Image
import imageio.v2 as imageio

# ======================
# CONFIG
# ======================
IMG_DIR  = Path(r"D:\vectionProject\public\camear1images")
OUT_MP4  = IMG_DIR / "cam1_gauss_trim_overlap_vs_linear_fixbright.mp4"

FPS = 60
DT = 1.0 / FPS


# ======================
# 读前 N 张灰度图
# ======================
def load_first_n(img_dir: Path, n: int):
    paths = sorted(img_dir.glob("cam1_*.png"))
    if len(paths) < n:
        raise RuntimeError(f"需要至少 {n} 张 cam1_*.png 图片, 目前只有 {len(paths)} 张")
    paths = paths[:n]
    imgs = []
    for p in paths:
        im = Image.open(p).convert("L")
        arr = np.asarray(im, dtype=np.float32) / 255.0  # 0~1（sRGB 灰度）
        imgs.append(arr)
    return paths, imgs


# ======================
# 全局高斯 + 裁尾巴（下段用）
# ======================
def gaussian_weights_trimmed(t: float,
                             num_frames: int,
                             sigma: float = 0.6,
                             rel_thresh: float = 0.15) -> np.ndarray:
    """
    t: 当前时间（秒，对应帧 index）
    num_frames: 总帧数 N
    sigma: 高斯核标准差（越大越“记忆”长，越小越接近2帧）
    rel_thresh: 相对阈值，比如 0.15 表示只保留 >= 15%*最大值 的部分
                用来裁掉尾巴，减少远处帧带来的多重影
    """
    idx = np.arange(num_frames, dtype=np.float32)  # 0..N-1
    w = np.exp(-0.5 * ((idx - t) / sigma) ** 2)    # 高斯

    # 裁掉尾部：低于最大值一定比例的权重直接置 0
    w_max = float(w.max())
    if w_max > 0.0 and rel_thresh > 0.0:
        w[w < w_max * rel_thresh] = 0.0

    s = float(w.sum())
    if s > 1e-6:
        w /= s
    else:
        # 极端情况：退化为最近整数帧
        k = int(round(np.clip(t, 0, num_frames - 1)))
        w[:] = 0.0
        w[k] = 1.0

    return w.astype(np.float32)


# ======================
# 主流程
# ======================
def main():
    N = 10  # 使用 10 张照片
    paths, imgs = load_first_n(IMG_DIR, N)
    print("Loaded images:", [p.name for p in paths])

    # 时间范围：0 ~ N-1 秒（和上段一致，一秒一个“真实位置”）
    T_start = 0.0
    T_end   = float(N - 1)
    print("Time range:", T_start, "->", T_end, "sec")

    H, W_img = imgs[0].shape
    frames_out = []

    num_steps = int(round((T_end - T_start) * FPS))
    for step in range(num_steps):
        t = T_start + (step + 0.5) * DT  # 当前时刻（秒）

        # ===== 上段：正常线性 A->B, B->C, ... =====
        if t >= N - 1:
            top_frame = imgs[-1]
        else:
            k = int(np.floor(t))
            u = t - k  # 0..1
            w_curr = 1.0 - u
            w_next = u
            top_frame = w_curr * imgs[k] + w_next * imgs[k + 1]

        # ===== 下段：裁尾巴的全局高斯权重 =====
        ws = gaussian_weights_trimmed(t, N, sigma=0.6, rel_thresh=0.15)

        bottom_frame = np.zeros_like(imgs[0], dtype=np.float32)
        for w, im in zip(ws, imgs):
            if w != 0.0:
                bottom_frame += w * im

        # ✅ 匹配上下两段的平均亮度，避免下段整体偏暗/偏灰
        mean_top = float(top_frame.mean())
        mean_bottom = float(bottom_frame.mean())
        if mean_bottom > 1e-6:
            gain = mean_top / mean_bottom
            # 防止极端放大/缩小
            gain = max(0.5, min(gain, 1.5))
            bottom_frame = np.clip(bottom_frame * gain, 0.0, 1.0)

        # 量化为 uint8
        top_u8 = np.clip(top_frame * 255.0 + 0.5, 0, 255).astype(np.uint8)
        bottom_u8 = np.clip(bottom_frame * 255.0 + 0.5, 0, 255).astype(np.uint8)

        # 上下拼接成一帧
        stacked = np.vstack([top_u8, bottom_u8])
        frames_out.append(stacked)

    print("Writing:", OUT_MP4)
    imageio.mimwrite(
        OUT_MP4,
        frames_out,
        fps=FPS,
        codec="libx264",
        quality=8,
        macro_block_size=None,
    )
    print("Done.")


if __name__ == "__main__":
    main()
