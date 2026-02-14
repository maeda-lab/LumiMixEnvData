import numpy as np
from pathlib import Path
from PIL import Image
import imageio.v2 as imageio

# ======================
# CONFIG
# ======================
IMG_DIR  = Path(r"D:\vectionProject\public\camear1images")

# 输出文件名
OUT_LINEAR   = IMG_DIR / "cam1_linear.mp4"
OUT_TRI_NORM = IMG_DIR / "cam1_tri_norm_r1p5.mp4"
OUT_TRI_RAW  = IMG_DIR / "cam1_tri_raw_r1p5.mp4"
OUT_GAUSS    = IMG_DIR / "cam1_gauss_sigma0p6.mp4"

FPS = 60
DT = 1.0 / FPS

N_IMAGES = 10          # 用前 N 张；想用全部就设 None
SECONDS_PER_STEP = 1.0 # 关键帧间隔（你现在是 1 秒一张真实位置）
DURATION_SEC = None    # None=自动 (N-1)*SECONDS_PER_STEP；也可手动设一个时长

# ====== 对比度分析相关建议 ======
INPUT_FRAMES_ARE_SRGB = True   # 大多数 PNG/截图都是 sRGB
BLEND_IN_LINEAR_LIGHT = True   # ✅ 做对比度分析通常建议 True（线性光域混合）
APPLY_MEAN_GAIN_MATCH = False  # ✅ 对比度分析建议 False（别做均值匹配）

# 视频压缩：你要做对比度分析，尽量减少压缩影响
USE_LOSSLESS_X264 = False      # True 会很大；但更适合分析（CRF=0）

# ======================
# sRGB <-> Linear (grayscale)
# ======================
def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    a = 0.055
    return np.where(x <= 0.04045, x / 12.92, ((x + a) / (1 + a)) ** 2.4)

def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    a = 0.055
    return np.where(x <= 0.0031308, 12.92 * x, (1 + a) * (x ** (1 / 2.4)) - a)

# ======================
# 读前 N 张灰度图
# ======================
def load_images(img_dir: Path, n: int | None):
    paths = sorted(img_dir.glob("cam1_*.png"))
    if n is not None:
        if len(paths) < n:
            raise RuntimeError(f"需要至少 {n} 张 cam1_*.png 图片, 目前只有 {len(paths)} 张")
        paths = paths[:n]
    if len(paths) < 2:
        raise RuntimeError("至少需要 2 张图片")

    imgs = []
    for p in paths:
        im = Image.open(p).convert("L")
        arr = np.asarray(im, dtype=np.float32) / 255.0  # 0~1 (sRGB灰度)
        if INPUT_FRAMES_ARE_SRGB and BLEND_IN_LINEAR_LIGHT:
            arr = srgb_to_linear(arr)
        imgs.append(arr)
    return paths, imgs

# ======================
# 权重函数
# ======================
def weights_linear_two_frame(t_idx: float, num_frames: int) -> np.ndarray:
    """
    标准两帧线性插值：只在 floor(t) 和 floor(t)+1 上有权重
    t_idx: 时间(以“帧序号”为单位：0..N-1)
    """
    if t_idx <= 0:
        w = np.zeros(num_frames, np.float32); w[0] = 1.0; return w
    if t_idx >= num_frames - 1:
        w = np.zeros(num_frames, np.float32); w[-1] = 1.0; return w
    k = int(np.floor(t_idx))
    u = float(t_idx - k)
    w = np.zeros(num_frames, np.float32)
    w[k] = 1.0 - u
    w[k + 1] = u
    return w

def weights_triangular(t_idx: float, num_frames: int, radius: float, normalize: bool) -> np.ndarray:
    """
    三角核：phi_i(t)=max(0, 1-|t-i|/radius)
    normalize=True: 再除以总和（权重和=1）
    normalize=False: raw，不归一化（权重和随时间变）
    """
    idx = np.arange(num_frames, dtype=np.float32)
    phi = np.maximum(0.0, 1.0 - np.abs(idx - t_idx) / float(radius)).astype(np.float32)
    if normalize:
        s = float(phi.sum())
        if s > 1e-8:
            phi /= s
    return phi

def weights_gaussian(t_idx: float, num_frames: int, sigma: float, normalize: bool = True) -> np.ndarray:
    idx = np.arange(num_frames, dtype=np.float32)
    w = np.exp(-0.5 * ((idx - t_idx) / float(sigma)) ** 2).astype(np.float32)
    if normalize:
        s = float(w.sum())
        if s > 1e-8:
            w /= s
    return w

# ======================
# 合成一帧（按权重加权求和）
# ======================
def synth_frame(imgs: list[np.ndarray], w: np.ndarray) -> np.ndarray:
    out = np.zeros_like(imgs[0], dtype=np.float32)
    for wi, im in zip(w, imgs):
        if wi != 0.0:
            out += wi * im
    return out

def to_u8(frame_linear_or_srgb: np.ndarray) -> np.ndarray:
    x = frame_linear_or_srgb
    if INPUT_FRAMES_ARE_SRGB and BLEND_IN_LINEAR_LIGHT:
        x = linear_to_srgb(x)
    x = np.clip(x, 0.0, 1.0)
    return (x * 255.0 + 0.5).astype(np.uint8)

# ======================
# 生成视频
# ======================
def write_video(out_path: Path, frames_u8: list[np.ndarray], fps: int):
    kwargs = dict(fps=fps, macro_block_size=None)
    if USE_LOSSLESS_X264:
        # 更适合做分析：几乎不引入额外压缩伪影，但文件很大
        kwargs.update(codec="libx264", ffmpeg_params=["-crf", "0", "-preset", "veryslow"])
    else:
        kwargs.update(codec="libx264", quality=9)

    print("Writing:", out_path)
    imageio.mimwrite(out_path, frames_u8, **kwargs)

def main():
    paths, imgs = load_images(IMG_DIR, N_IMAGES)
    N = len(imgs)
    print("Loaded:", N, "images")
    print("First/Last:", paths[0].name, "->", paths[-1].name)

    # 时间范围（以“关键帧序号”为单位）：0..N-1
    t_start = 0.0
    t_end = float(N - 1)

    # 自动时长： (N-1)*SECONDS_PER_STEP
    if DURATION_SEC is None:
        total_sec = (N - 1) * SECONDS_PER_STEP
    else:
        total_sec = float(DURATION_SEC)

    num_steps = int(round(total_sec * FPS))
    print("Duration:", total_sec, "sec  | frames:", num_steps)

    # 预先生成每种视频的帧列表
    frames_linear   = []
    frames_tri_norm = []
    frames_tri_raw  = []
    frames_gauss    = []

    radius = 1.5
    sigma = 0.6

    for step in range(num_steps):
        t_sec = (step + 0.5) * DT
        # 把真实时间映射到“关键帧 index”
        t_idx = t_start + (t_sec / SECONDS_PER_STEP)
        t_idx = float(np.clip(t_idx, t_start, t_end))

        # 1) Linear (两帧)
        w_lin = weights_linear_two_frame(t_idx, N)
        f_lin = synth_frame(imgs, w_lin)

        # 2) Triangular normalized
        w_tn = weights_triangular(t_idx, N, radius=radius, normalize=True)
        f_tn = synth_frame(imgs, w_tn)

        # 3) Triangular raw (no normalization)
        w_tr = weights_triangular(t_idx, N, radius=radius, normalize=False)
        f_tr = synth_frame(imgs, w_tr)

        # 4) Gaussian normalized
        w_g = weights_gaussian(t_idx, N, sigma=sigma, normalize=True)
        f_g = synth_frame(imgs, w_g)

        # （可选）均值匹配：⚠️ 对比度分析一般不建议开
        if APPLY_MEAN_GAIN_MATCH:
            def match_mean(src, ref):
                ms = float(src.mean()); mr = float(ref.mean())
                if ms > 1e-8:
                    gain = mr / ms
                    gain = max(0.5, min(gain, 1.5))
                    return np.clip(src * gain, 0.0, 1.0)
                return src

            f_tn = match_mean(f_tn, f_lin)
            f_tr = match_mean(f_tr, f_lin)
            f_g  = match_mean(f_g,  f_lin)

        frames_linear.append(to_u8(f_lin))
        frames_tri_norm.append(to_u8(f_tn))
        frames_tri_raw.append(to_u8(f_tr))
        frames_gauss.append(to_u8(f_g))

    write_video(OUT_LINEAR, frames_linear, FPS)
    write_video(OUT_TRI_NORM, frames_tri_norm, FPS)
    write_video(OUT_TRI_RAW, frames_tri_raw, FPS)
    write_video(OUT_GAUSS, frames_gauss, FPS)

    print("Done.")

if __name__ == "__main__":
    main()
