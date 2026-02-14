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
SECONDS_PER_STEP = 1.0   # 相邻两张原始图片之间的“物理间隔”：1 秒
N_PER_STEP = int(round(FPS * SECONDS_PER_STEP))

# 更强的时间平滑：5 帧高斯核 + 中心权重上限
OUT_NAME = "orig_luma_multiframe_sigma1.0_center0.7_shift05.mp4"

SIGMA_INDEX = 1.0   # 在“张号”维度的高斯 sigma
K_RADIUS    = 2     # 使用 j-2, j-1, j, j+1, j+2 共 5 张
CENTER_MAX  = 0.7   # 中心权重最多 0.7，防止“几乎单帧”


# =========================
# sRGB <-> Linear
# =========================
def srgb_to_linear(x):
    x = np.clip(x, 0.0, 1.0)
    a = 0.055
    return np.where(
        x <= 0.04045,
        x / 12.92,
        ((x + a) / (1 + a)) ** 2.4
    )

def linear_to_srgb(x):
    x = np.clip(x, 0.0, 1.0)
    a = 0.055
    return np.where(
        x <= 0.0031308,
        x * 12.92,
        (1 + a) * (x ** (1 / 2.4)) - a
    )

def clamp01(x):
    return np.clip(x, 0.0, 1.0)


# =========================
# Linear RGB -> Linear Luma (grayscale)
# =========================
def rgb_lin_to_luma_lin(rgb_lin: np.ndarray) -> np.ndarray:
    # Rec.709 luma in LINEAR space
    return (0.2126 * rgb_lin[..., 0] +
            0.7152 * rgb_lin[..., 1] +
            0.0722 * rgb_lin[..., 2]).astype(np.float32)

def luma3_from_rgb_lin(rgb_lin: np.ndarray) -> np.ndarray:
    y = rgb_lin_to_luma_lin(rgb_lin)              # (H,W)
    return np.repeat(y[..., None], 3, axis=2)     # (H,W,3)


# =========================
# IO
# =========================
def read_rgb_linear(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return srgb_to_linear(rgb).astype(np.float32)

def write_frame(writer, img_lin: np.ndarray):
    img_srgb = linear_to_srgb(clamp01(img_lin))
    u8 = (img_srgb * 255.0 + 0.5).astype(np.uint8)
    writer.append_data(u8)


# =========================
# Load all frames (luma)
# =========================
def load_luma_stack(frame_paths: list[Path]) -> list[np.ndarray]:
    stack = []
    for p in frame_paths:
        rgb_lin = read_rgb_linear(p)
        y3 = luma3_from_rgb_lin(rgb_lin)
        stack.append(y3)
        print("loaded:", p)
    return stack


# =========================
# Multi-frame temporal smoothing
# =========================
def gauss_weight(dist_idx: float, sigma: float) -> float:
    return math.exp(-0.5 * (dist_idx / sigma) ** 2)

def make_orig_luma_video_multiframe(frame_paths: list[Path], out_mp4: Path):
    frames_luma = load_luma_stack(frame_paths)
    M = len(frames_luma)
    if M < 2:
        raise RuntimeError("Not enough frames")

    total_out_frames = (M - 1) * N_PER_STEP

    writer = imageio.get_writer(
        str(out_mp4),
        fps=FPS,
        codec="libx264",
        quality=8,
        macro_block_size=1,
    )

    try:
        H, W, C = frames_luma[0].shape

        for out_idx in range(total_out_frames):
            # ★ 关键 1：时间轴加 0.5 偏移，永远不落在整数张号上
            # 原来：s = out_idx / N_PER_STEP
            # 现在：s = (out_idx + 0.5) / N_PER_STEP
            s = (out_idx + 0.5) / N_PER_STEP  # 0.5 frame shift

            # 最近的张号
            j_center_f = round(s)
            j_center = int(j_center_f)
            j_center = max(0, min(M - 1, j_center))

            acc = np.zeros((H, W, C), dtype=np.float32)
            weights = []
            indices = []

            # 先算高斯权重
            for dj in range(-K_RADIUS, K_RADIUS + 1):
                j = j_center + dj
                if j < 0 or j >= M:
                    continue
                dist = s - j   # 距离第 j 张有多远
                w = gauss_weight(dist, SIGMA_INDEX)
                weights.append(w)
                indices.append(j)

            if not weights:
                # 极端情况下兜底
                out = frames_luma[j_center]
            else:
                # ★ 关键 2：限制中心权重（防止“几乎单帧”）
                # 找出中心在 indices 里的位置
                if j_center in indices:
                    idx_c = indices.index(j_center)
                    w_c = weights[idx_c]
                    w_sum = sum(weights)
                    if w_sum > 0:
                        # 当前中心占比
                        center_ratio = w_c / w_sum
                        if center_ratio > CENTER_MAX:
                            # 把中心压到 CENTER_MAX，其余按比例放大
                            other_sum = w_sum - w_c
                            if other_sum > 0:
                                scale_other = (1.0 - CENTER_MAX) / other_sum
                                scale_center = CENTER_MAX / center_ratio
                                # 先缩再放（等效：中心减，其他加）
                                for k in range(len(weights)):
                                    if k == idx_c:
                                        weights[k] *= scale_center
                                    else:
                                        weights[k] *= scale_other
                            else:
                                # 万一只有中心一个点，就直接用中心（极罕见）
                                pass

                # 最后再归一化一次
                w_sum2 = sum(weights)
                if w_sum2 <= 0:
                    out = frames_luma[j_center]
                else:
                    acc[:] = 0
                    for w, j in zip(weights, indices):
                        acc += (w / w_sum2) * frames_luma[j]
                    out = acc

            write_frame(writer, out)

            if (out_idx % 200) == 0:
                print(f"frame {out_idx}/{total_out_frames}")

        print("saved:", out_mp4)
    finally:
        writer.close()


def main():
    exts = (".png", ".jpg", ".jpeg")
    frame_paths = sorted([p for p in IN_DIR.iterdir()
                          if p.suffix.lower() in exts])
    if len(frame_paths) < 2:
        raise RuntimeError("Not enough frames")

    make_orig_luma_video_multiframe(frame_paths, OUT_DIR / OUT_NAME)


if __name__ == "__main__":
    main()
