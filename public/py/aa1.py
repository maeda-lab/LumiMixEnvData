import numpy as np
from pathlib import Path
from PIL import Image
import imageio.v2 as imageio

IMG_DIR = Path(r"D:\vectionProject\public\A-isi-images")
OUT_MP4 = IMG_DIR / "cam1_truncgauss3tap_sigma0p6.mp4"

FPS = 60
DT = 1.0 / FPS
STEP_SEC = 1.0
SIGMA = 0.6
N = 10  # cam2_000..cam2_009

def load_first_n_cam2(img_dir: Path, n: int):
    paths = sorted(img_dir.glob("cam1_*.png"))
    if len(paths) < n:
        raise RuntimeError(f"Need >= {n} images, found {len(paths)}")
    paths = paths[:n]
    imgs = []
    for p in paths:
        im = Image.open(p).convert("L")
        imgs.append(np.asarray(im, dtype=np.float32) / 255.0)
    # size check
    H, W = imgs[0].shape
    for p, a in zip(paths, imgs):
        if a.shape != (H, W):
            raise RuntimeError(f"Size mismatch: {p.name} {a.shape} != {(H,W)}")
    return paths, imgs

def weights_3tap(u: float, n: int, sigma: float):
    c = int(np.round(u))
    idxs = np.clip([c-1, c, c+1], 0, n-1)

    # unnormalized gaussian (center at u, python-style)
    d = (np.array(idxs, dtype=np.float32) - u) / sigma
    w = np.exp(-0.5 * d * d)
    w /= max(w.sum(), 1e-12)

    out = np.zeros(n, dtype=np.float32)
    for ii, wi in zip(idxs, w):
        out[ii] += wi
    return out

def main():
    paths, imgs = load_first_n_cam2(IMG_DIR, N)
    print("Loaded:", [p.name for p in paths])

    T_start = 0.0
    T_end = float(N - 1)
    num_out = int(round((T_end - T_start) * FPS))
    frames_out = []

    for f in range(num_out):
        t = T_start + (f + 0.5) * DT
        u = t / STEP_SEC
        w = weights_3tap(u, N, SIGMA)

        out = np.zeros_like(imgs[0], dtype=np.float32)
        for wi, im in zip(w, imgs):
            if wi != 0.0:
                out += wi * im

        frames_out.append(np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8))

    print("Writing:", OUT_MP4)
    imageio.mimwrite(OUT_MP4, frames_out, fps=FPS, codec="libx264", quality=8, macro_block_size=None)
    print("Done:", OUT_MP4)

if __name__ == "__main__":
    main()
