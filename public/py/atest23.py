import math
from pathlib import Path
import numpy as np
import cv2
import imageio.v2 as imageio

# =========================
# CONFIG
# =========================
IN_DIR  = Path(r"D:\vectionProject\public\camear1images")
OUT_DIR = Path(r"D:\vectionProject\public\freq_test_videos")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FPS = 60
SECONDS_PER_STEP = 1.0
N = int(round(FPS * SECONDS_PER_STEP))  # frames per step

# 补偿强度：0 = 完全线性，0.5 左右就已经很明显但仍然平滑、单调
BETA = 0.5

OUT_NAME = f"two_rows_luma_linear_vs_Sbeta{BETA:.2f}_NO_WARP.mp4"


# =========================
# sRGB <-> Linear
# =========================
def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    a = 0.055
    return np.where(
        x <= 0.04045,
        x / 12.92,
        ((x + a) / (1 + a)) ** 2.4
    )


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    a = 0.055
    return np.where(
        x <= 0.0031308,
        x * 12.92,
        (1 + a) * (x ** (1 / 2.4)) - a
    )


def clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


# =========================
# Linear RGB -> Linear Luma (grayscale)
# =========================
def rgb_lin_to_luma_lin(rgb_lin: np.ndarray) -> np.ndarray:
    # Rec.709 luma in LINEAR space
    return (
        0.2126 * rgb_lin[..., 0] +
        0.7152 * rgb_lin[..., 1] +
        0.0722 * rgb_lin[..., 2]
    ).astype(np.float32)


def luma3_from_rgb_lin(rgb_lin: np.ndarray) -> np.ndarray:
    y = rgb_lin_to_luma_lin(rgb_lin)          # (H, W)
    return np.repeat(y[..., None], 3, axis=2) # (H, W, 3)


# =========================
# IO
# =========================
def read_rgb_linear(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return srgb_to_linear(rgb).astype(np.float32)


def write_frame(writer, img_lin: np.ndarray) -> None:
    img_srgb = linear_to_srgb(clamp01(img_lin))
    u8 = (img_srgb * 255.0 + 0.5).astype(np.uint8)
    writer.append_data(u8)


# =========================
# p(t) functions（只改时间，不搞空间）
# =========================
def p_linear(t: float) -> float:
    """Baseline: p(t) = t"""
    t = float(np.clip(t, 0.0, 1.0))
    return t


def p_s_curve(t: float, beta: float) -> float:
    """
    S 型补偿：
      s(t) = 0.5 - 0.5 cos(pi t)
    然后 p = (1-beta)*t + beta*s(t)
    - 保证 p(0) = 0, p(1) = 1
    - 单调、平滑，不会猛窜、不诡异
    - beta 越大，中间越“快”、端点越“慢”
    """
    t = float(np.clip(t, 0.0, 1.0))
    s = 0.5 - 0.5 * math.cos(math.pi * t)  # 0..1, C^1 平滑
    p = (1.0 - beta) * t + beta * s
    return float(np.clip(p, 0.0, 1.0))


def p_compensated(t: float) -> float:
    return p_s_curve(t, BETA)


# =========================
# Two-row video synthesis（无任何 warp）
# =========================
def make_two_row_video(frames: list[Path], out_mp4: Path) -> None:
    """
    上段：orig luma cross-dissolve（p(t)=t）
    下段：orig luma cross-dissolve（p(t)=S 曲线）
    都不做空间变形，只做亮度混合。
    """
    writer = imageio.get_writer(
        str(out_mp4),
        fps=FPS,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )

    try:
        for i in range(len(frames) - 1):
            A_rgb = read_rgb_linear(frames[i])
            B_rgb = read_rgb_linear(frames[i + 1])

            A = luma3_from_rgb_lin(A_rgb)
            B = luma3_from_rgb_lin(B_rgb)

            for k in range(N):
                if N > 1:
                    t = k / (N - 1)
                else:
                    t = 1.0

                # 上段：线性
                p_top = p_linear(t)
                top = (1.0 - p_top) * A + p_top * B

                # 下段：S 型补偿
                p_bottom = p_compensated(t)
                bottom = (1.0 - p_bottom) * A + p_bottom * B

                frame_out = np.concatenate([top, bottom], axis=0)
                write_frame(writer, frame_out)

        print("saved:", out_mp4)
    finally:
        writer.close()


def main() -> None:
    exts = (".png", ".jpg", ".jpeg")
    frames = sorted([p for p in IN_DIR.iterdir()
                     if p.suffix.lower() in exts])
    if len(frames) < 2:
        raise RuntimeError("Not enough frames")

    make_two_row_video(frames, OUT_DIR / OUT_NAME)


if __name__ == "__main__":
    main()
