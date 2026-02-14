import numpy as np
from pathlib import Path
from PIL import Image
import imageio.v2 as imageio

# ======================
# CONFIG
# ======================
IMG_DIR  = Path(r"D:\\vectionProject\\public\\camear1images")
OUT_MP4  = IMG_DIR / "cam1_tri_top2_overlap_vs_linear_fixbright.mp4"

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
# 完美三角形核（原始值，不归一化）
# phi_i(t) = max(1 - |t - i| / radius, 0)
# ======================
def tri_raw_weights(t: float, num_frames: int, radius: float = 1.5) -> np.ndarray:
    idx = np.arange(num_frames, dtype=np.float32)  # 0..N-1
    dist = np.abs(idx - t)
    w = 1.0 - dist / radius
    w[w < 0.0] = 0.0
    return w.astype(np.float32)


# ======================
# “三角形 + 只保留 top-2” 权重
# 对应你画的：三角形可以重叠，但实际只让两帧参与，和=1
# ======================
def tri_top2_weights(t: float, num_frames: int, radius: float = 1.5) -> np.ndarray:
    # 先算所有帧的三角形值
    phi = tri_raw_weights(t, num_frames, radius=radius)  # 0..1，可能多个非零

    # 找出最大的两帧
    # argsort 从小到大，取最后两个
    nonzero_idx = np.nonzero(phi)[0]
    w = np.zeros_like(phi)
    if len(nonzero_idx) == 0:
        # 安全兜底：退化到最近整数帧
        k = int(round(np.clip(t, 0, num_frames - 1)))
        w[k] = 1.0
        return w

    if len(nonzero_idx) == 1:
        # 只有一帧有非零三角形 → 直接用这一帧
        w[nonzero_idx[0]] = 1.0
        return w

    # 至少两帧非零：选出 phi 最大的两帧
    top2 = np.argsort(phi)[-2:]  # indices of two largest
    # 把这两帧的 phi 作为权重
    tmp = phi[top2]
    s = float(tmp.sum())
    if s > 1e-6:
        tmp /= s   # 归一化到和=1
    # 填回到完整数组
    w[top2[0]] = tmp[0]
    w[top2[1]] = tmp[1]
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

        # ===== 下段：三角核 + 只保留 top-2 帧 =====
        # radius 控制“观测时间宽度”，1.3~1.7 之间可以自己试
        ws = tri_top2_weights(t, N, radius=1.5)

        bottom_frame = np.zeros_like(imgs[0], dtype=np.float32)
        for w_i, im in zip(ws, imgs):
            if w_i != 0.0:
                bottom_frame += w_i * im

        # ✅ 匹配上下两段的平均亮度，避免下段整体偏暗/偏亮
        mean_top = float(top_frame.mean())
        mean_bottom = float(bottom_frame.mean())
        if mean_bottom > 1e-6:
            gain = mean_top / mean_bottom
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
