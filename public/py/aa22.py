import numpy as np
from pathlib import Path
from PIL import Image
import imageio.v2 as imageio
import matplotlib.pyplot as plt

# ======================
# CONFIG
# ======================
IMG_DIR = Path(r"D:\vectionProject\public\A-isi-images")
OUT_MP4 = IMG_DIR / "cam1_truncgauss2tap_sigma0p6.mp4"
OUT_WPNG = IMG_DIR / "weights_truncgauss2tap_sigma0p6.png"

FPS = 60
DT = 1.0 / FPS
STEP_SEC = 1.0
SIGMA = 0.6
N = 10  # cam1_000..cam1_009


# ======================
# IO
# ======================
def load_first_n_cam1(img_dir: Path, n: int):
    paths = sorted(img_dir.glob("cam1_*.png"))
    if len(paths) < n:
        raise RuntimeError(f"Need >= {n} images, found {len(paths)}")
    paths = paths[:n]

    imgs = []
    for p in paths:
        im = Image.open(p).convert("L")
        imgs.append(np.asarray(im, dtype=np.float32) / 255.0)

    H, W = imgs[0].shape
    for p, a in zip(paths, imgs):
        if a.shape != (H, W):
            raise RuntimeError(f"Size mismatch: {p.name} {a.shape} != {(H, W)}")
    return paths, imgs


# ======================
# 2-tap truncated Gaussian weights
# ======================
def weights_2tap(u: float, n: int, sigma: float):
    """
    任意时刻只混合两张：i=floor(u) 与 i+1
    权重 = exp(-0.5 * ((idx-u)/sigma)^2)，再在这两张上归一化（截断高斯）
    返回 shape=(n,) 的稀疏权重向量
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if sigma <= 1e-12:
        sigma = 1e-12

    u = float(np.clip(u, 0.0, max(0.0, n - 1)))

    i0 = int(np.floor(u))
    i1 = i0 + 1

    i0 = int(np.clip(i0, 0, n - 1))
    i1 = int(np.clip(i1, 0, n - 1))

    out = np.zeros(n, dtype=np.float32)

    # 末尾：只剩最后一张
    if i0 == i1:
        out[i0] = 1.0
        return out

    idxs = np.array([i0, i1], dtype=np.float32)
    d = (idxs - u) / float(sigma)
    w = np.exp(-0.5 * d * d).astype(np.float32)
    w /= max(float(w.sum()), 1e-12)

    out[i0] = w[0]
    out[i1] = w[1]
    return out


# ======================
# MAIN
# ======================
def main():
    paths, imgs = load_first_n_cam1(IMG_DIR, N)
    print("Loaded:", [p.name for p in paths])

    # 输出时长：0 .. (N-1) 秒，对应 u: 0 .. (N-1)
    T_start = 0.0
    T_end = float(N - 1)
    num_out = int(round((T_end - T_start) * FPS))

    frames_out = []
    Wmat = np.zeros((num_out, N), dtype=np.float32)
    t_list = np.zeros(num_out, dtype=np.float32)

    for f in range(num_out):
        t = T_start + (f + 0.5) * DT  # half-frame offset
        u = t / STEP_SEC

        w = weights_2tap(u, N, SIGMA)
        Wmat[f, :] = w
        t_list[f] = t

        out = np.zeros_like(imgs[0], dtype=np.float32)
        for wi, im in zip(w, imgs):
            if wi != 0.0:
                out += wi * im

        frames_out.append(np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8))

    # ---- write video ----
    print("Writing:", OUT_MP4)
    imageio.mimwrite(
        OUT_MP4,
        frames_out,
        fps=FPS,
        codec="libx264",
        quality=8,
        macro_block_size=None
    )
    print("Done video:", OUT_MP4)

    # ---- plot weights ----
    plt.figure(figsize=(12, 4.5))
    for i in range(N):
        plt.plot(t_list, Wmat[:, i], label=f"i={i}")

    plt.title(f"Truncated Gaussian 2-tap weights (sigma={SIGMA}, step={STEP_SEC}s, fps={FPS})")
    plt.xlabel("time (s)")
    plt.ylabel("weight")
    plt.ylim(-0.02, 1.02)
    plt.xlim(t_list[0], t_list[-1])
    plt.grid(True, alpha=0.3)

    # 如果 N 很大，legend 会很挤；N=10 还好
    plt.legend(ncol=5, fontsize=8, frameon=False)
    plt.tight_layout()
    plt.savefig(OUT_WPNG, dpi=200)
    plt.close()

    print("Done weights plot:", OUT_WPNG)


if __name__ == "__main__":
    main()
