import numpy as np
from pathlib import Path
from PIL import Image
import imageio.v2 as imageio
import sys

try:
    # 重新配置标准输出为 UTF-8，允许 print 中文字符
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    # 在某些 Python/环境中可能不可用，忽略失败
    pass

# ======================
# CONFIG
# ======================
IMG_DIR = Path(r"D:\vectionProject\public\camera3images")
PATTERN = "cam1_*.png"      # 如果是 jpg 等，请改成对应后缀

OUT_LINEAR      = IMG_DIR / "cam3_linear.mp4"
OUT_GAUSS_FULL  = IMG_DIR / "cam3_gauss_full_sigma0p6.mp4"
OUT_GAUSS_TRUNC = IMG_DIR / "cam3_truncgauss3tap_sigma0p6.mp4"
OUT_GAUSS_TRUNC_0P5 = IMG_DIR / "cam3_truncgauss3tap_sigma0p5.mp4"
OUT_GAUSS_TRUNC_0P7 = IMG_DIR / "cam3_truncgauss3tap_sigma0p7.mp4"


FPS        = 60
DT         = 1.0 / FPS
STEP_SEC   = 1.0        # 1 Hz
SIGMA_STEP = 0.6        # sigma = 0.6 (step units)
N_FRAMES_SRC = 9    # 用前 10 张源图，如需改动就改这个


# ======================
# 先做灰度预处理
# ======================
def save_gray_images(img_dir: Path, pattern: str):
    paths = sorted(img_dir.glob(pattern))
    if not paths:
        raise RuntimeError(f"在 {img_dir} 下没有找到匹配 {pattern} 的图片")

    for p in paths:
        im = Image.open(p).convert("L")  # 转灰度
        im.save(p)                       # 覆盖保存（保留原文件名和格式）
    print(f"灰度处理完成，共处理 {len(paths)} 张图片")


# ======================
# 读取前 N 张图片（已是灰度）
# ======================
def load_first_n_images(img_dir: Path, pattern: str, n: int):
    paths = sorted(img_dir.glob(pattern))
    if len(paths) < n:
        raise RuntimeError(f"需要至少 {n} 张图片, 目前只有 {len(paths)} 张")

    paths = paths[:n]
    imgs = []

    for p in paths:
        im = Image.open(p).convert("L")  # 再保险转一次灰度
        arr = np.asarray(im, dtype=np.float32) / 255.0  # 归一化到 0..1
        imgs.append(arr)

    imgs = np.stack(imgs, axis=0)  # (N, H, W)
    return imgs


# ======================
# 写视频
# ======================
def write_video(frames, out_path: Path, fps: int):
    out_path = str(out_path)
    with imageio.get_writer(out_path, fps=fps, codec="libx264", bitrate="16M") as w:
        for frame in frames:
            f = np.clip(frame, 0.0, 1.0)
            f8 = (f * 255.0).astype(np.uint8)
            # 转成 3 通道，兼容性更好
            if f8.ndim == 2:
                f8 = np.stack([f8] * 3, axis=-1)
            w.append_data(f8)


# ======================
# 权重函数
# ======================
def weights_linear(u, N, sigma_step):
    """
    最初 linear cross-dissolve（最多 2 张参与）
    u: 连续时间（单位 = step），0..(N-1)
    sigma_step 参数在这里不用，只是为了统一接口。
    """
    w = np.zeros(N, dtype=np.float32)

    if u <= 0:
        w[0] = 1.0
        return w
    if u >= N - 1:
        w[N - 1] = 1.0
        return w

    k = int(np.floor(u))  # 左端 index
    frac = u - k          # 0..1
    w[k] = 1.0 - frac
    w[k + 1] = frac
    return w


def weights_gauss_full(u, N, sigma_step):
    """
    第一张图：完整时间高斯（所有 N 张都参与，只是远处权重很小）
    """
    ks = np.arange(N, dtype=np.float32)
    g = np.exp(-0.5 * ((u - ks) / sigma_step) ** 2)
    s = np.sum(g)
    if s <= 0:
        g[0] = 1.0
        s = 1.0
    return (g / s).astype(np.float32)


def weights_gauss_trunc3(u, N, sigma_step):
    """
    第二张图：3-tap 截断高斯
    最多 3 张参与（c-1, c, c+1）
    """
    w = np.zeros(N, dtype=np.float32)

    c = int(round(u))
    i0 = max(c - 1, 0)
    i1 = min(c + 1, N - 1)

    idxs = np.arange(i0, i1 + 1, dtype=np.int32)
    g = np.exp(-0.5 * ((u - idxs) / sigma_step) ** 2)
    s = np.sum(g)
    if s <= 0:
        w[np.clip(c, 0, N - 1)] = 1.0
        return w

    g = g / s
    for ii, gi in zip(idxs, g):
        w[ii] = gi
    return w


# ======================
# 通用合成函数
# ======================
def synth_video(imgs, weight_func, out_path, fps=FPS,
                step_sec=STEP_SEC, sigma_step=SIGMA_STEP):
    """
    imgs: (N, H, W)
    weight_func: 函数(u, N, sigma_step) -> (N,) 权重
    """
    N, H, W = imgs.shape

    # 时间范围：0 .. (N-1)*step_sec
    T_sec = (N - 1) * step_sec
    total_frames = int(round(T_sec * fps))

    frames = []

    for f in range(total_frames):
        t = f * (1.0 / fps)     # 真实时间 (s)
        u = t / step_sec        # 单位 = step（0..N-1）
        w = weight_func(u, N, sigma_step)

        # (N,) dot (N, H, W) -> (H, W)
        frame = np.tensordot(w, imgs, axes=(0, 0))
        frames.append(frame)

    write_video(frames, out_path, fps)


# ======================
# 主流程
# ======================
def main():
    # ① 先把所有 cam1_*.png 转成灰度
    save_gray_images(IMG_DIR, PATTERN)

    # ② 读取前 N_FRAMES_SRC 张图
    imgs = load_first_n_images(IMG_DIR, PATTERN, N_FRAMES_SRC)
    print("Loaded images:", imgs.shape)  # (N, H, W)

    # # 1. 最初的 linear
    # synth_video(
    #     imgs,
    #     weight_func=weights_linear,
    #     out_path=OUT_LINEAR,
    #     fps=FPS,
    #     step_sec=STEP_SEC,
    #     sigma_step=SIGMA_STEP,
    # )
    # print("Saved:", OUT_LINEAR)

    # # 2. 第一张图：完整时间高斯
    # synth_video(
    #     imgs,
    #     weight_func=weights_gauss_full,
    #     out_path=OUT_GAUSS_FULL,
    #     fps=FPS,
    #     step_sec=STEP_SEC,
    #     sigma_step=SIGMA_STEP,
    # )
    # print("Saved:", OUT_GAUSS_FULL)

    # # 3. 第二张图：3-tap 截断高斯
    # synth_video(
    #     imgs,
    #     weight_func=weights_gauss_trunc3,
    #     out_path=OUT_GAUSS_TRUNC,
    #     fps=FPS,
    #     step_sec=STEP_SEC,
    #     sigma_step=SIGMA_STEP,
    # )
    # print("Saved:", OUT_GAUSS_TRUNC)
    
     # 3b. 3-tap 截断高斯：sigma=0.5
    synth_video(
        imgs,
        weight_func=weights_gauss_trunc3,
        out_path=OUT_GAUSS_TRUNC_0P5,
        fps=FPS,
        step_sec=STEP_SEC,
        sigma_step=0.5,
    )
    print("Saved:", OUT_GAUSS_TRUNC_0P5)

    # 3c. 3-tap 截断高斯：sigma=0.7
    synth_video(
        imgs,
        weight_func=weights_gauss_trunc3,
        out_path=OUT_GAUSS_TRUNC_0P7,
        fps=FPS,
        step_sec=STEP_SEC,
        sigma_step=0.7,
    )
    print("Saved:", OUT_GAUSS_TRUNC_0P7)


if __name__ == "__main__":
    main()
