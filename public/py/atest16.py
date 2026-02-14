import math
from pathlib import Path
import numpy as np
import cv2
import imageio.v2 as imageio

# =========================
# CONFIG (edit these)
# =========================
IN_DIR  = Path(r"D:\vectionProject\public\camear1images")   # Unity exported frames
OUT_DIR = Path(r"D:\vectionProject\public\edge_ablation_videos")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FPS = 60
SECONDS_PER_STEP = 1.0
N = int(round(FPS * SECONDS_PER_STEP))  # frames per step

# 你盯着树的话，就把 ROI 框在树附近（建议先用这个，再自己微调）
# ROI = (x0, y0, w, h)
ROI = None

# “变化区域”的阈值分位数：越大越只取最强变化（边缘更干净）
CHANGE_PERCENTILE = 75  # 80~92 都可以试

# 边缘带宽度（像素）：8~20
EDGE_BAND_PX = 20

# 输出里非保留区域填充的中性灰（sRGB 0.5）
NEUTRAL_GRAY_SRGB = 0.5

# 是否输出灰度视频（建议 True，避免颜色干扰）
FORCE_GRAYSCALE = True

# =========================
# sRGB <-> Linear
# =========================
def srgb_to_linear(x):
    x = np.clip(x, 0.0, 1.0)
    a = 0.055
    return np.where(x <= 0.04045, x / 12.92, ((x + a) / (1 + a)) ** 2.4)

def linear_to_srgb(x):
    x = np.clip(x, 0.0, 1.0)
    a = 0.055
    return np.where(x <= 0.0031308, x * 12.92, (1 + a) * (x ** (1 / 2.4)) - a)

def clamp01(x):
    return np.clip(x, 0.0, 1.0)

def rgb_lin_to_luma_lin(img_lin):
    # Rec.709 luma in linear space
    return (0.2126 * img_lin[..., 0] +
            0.7152 * img_lin[..., 1] +
            0.0722 * img_lin[..., 2]).astype(np.float32)

def luma3_from_rgb_lin(img_lin):
    g = rgb_lin_to_luma_lin(img_lin)
    return np.repeat(g[..., None], 3, axis=2)

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
# Mask builder (per step)
# =========================
def build_masks_from_AB(A_lin: np.ndarray, B_lin: np.ndarray, roi, change_percentile=85, edge_band_px=12):
    H, W, _ = A_lin.shape
    if roi is None:
        x0, y0, w, h = 0, 0, W, H
    else:
        x0, y0, w, h = roi

    # luma diff inside ROI
    A_roi = A_lin[y0:y0+h, x0:x0+w]
    B_roi = B_lin[y0:y0+h, x0:x0+w]
    d = np.abs(rgb_lin_to_luma_lin(B_roi) - rgb_lin_to_luma_lin(A_roi))  # (h,w)

    # threshold by percentile
    thr = np.percentile(d, change_percentile)
    change = (d >= thr).astype(np.uint8) * 255

    # morph clean
    k = np.ones((3, 3), np.uint8)
    change = cv2.morphologyEx(change, cv2.MORPH_OPEN, k, iterations=1)
    change = cv2.morphologyEx(change, cv2.MORPH_CLOSE, k, iterations=2)

    # edge band = dilate(change) - erode(change)
    dil = cv2.dilate(change, k, iterations=edge_band_px)
    ero = cv2.erode(change, k, iterations=edge_band_px)
    edge_band = cv2.subtract(dil, ero)  # 0/255

    # interior = change & ~edge
    interior = cv2.bitwise_and(change, cv2.bitwise_not(edge_band))

    # place back to full size masks
    edge_full = np.zeros((H, W), dtype=np.uint8)
    int_full  = np.zeros((H, W), dtype=np.uint8)
    edge_full[y0:y0+h, x0:x0+w] = edge_band
    int_full [y0:y0+h, x0:x0+w] = interior

    return edge_full, int_full

def keep_mask(img_lin: np.ndarray, mask_u8: np.ndarray, base_lin: float):
    out = np.full_like(img_lin, base_lin, dtype=np.float32)
    m = (mask_u8 > 0)
    out[m] = img_lin[m]
    return out

def remove_mask(img_lin: np.ndarray, mask_u8: np.ndarray, base_lin: float):
    out = img_lin.copy()
    m = (mask_u8 > 0)
    out[m] = base_lin
    return out

# =========================
# Main: generate 3 videos
# =========================
def main():
    exts = (".png", ".jpg", ".jpeg")
    frames = sorted([p for p in IN_DIR.iterdir() if p.suffix.lower() in exts])
    if len(frames) < 2:
        raise RuntimeError("Not enough frames")

    base_lin = float(srgb_to_linear(np.array([NEUTRAL_GRAY_SRGB], np.float32))[0])

    out_edge_only    = OUT_DIR / "edge_only.mp4"
    out_interior_only= OUT_DIR / "interior_only.mp4"
    out_edge_removed = OUT_DIR / "edge_removed.mp4"

    writers = {
        "edge_only": imageio.get_writer(str(out_edge_only), fps=FPS, codec="libx264", quality=8, macro_block_size=1),
        "int_only":  imageio.get_writer(str(out_interior_only), fps=FPS, codec="libx264", quality=8, macro_block_size=1),
        "edge_rm":   imageio.get_writer(str(out_edge_removed), fps=FPS, codec="libx264", quality=8, macro_block_size=1),
    }

    try:
        for i in range(len(frames) - 1):
            A = read_rgb_linear(frames[i])
            B = read_rgb_linear(frames[i + 1])

            if FORCE_GRAYSCALE:
                A = luma3_from_rgb_lin(A)
                B = luma3_from_rgb_lin(B)

            edge_mask, int_mask = build_masks_from_AB(
                A, B, ROI,
                change_percentile=CHANGE_PERCENTILE,
                edge_band_px=EDGE_BAND_PX
            )

            # pre-process A,B for each condition
            A_edge = keep_mask(A, edge_mask, base_lin)
            B_edge = keep_mask(B, edge_mask, base_lin)

            A_int  = keep_mask(A, int_mask, base_lin)
            B_int  = keep_mask(B, int_mask, base_lin)

            A_rm   = remove_mask(A, edge_mask, base_lin)
            B_rm   = remove_mask(B, edge_mask, base_lin)

            # cross-dissolve within step
            for k in range(N):
                p = k / (N - 1) if N > 1 else 1.0

                out1 = (1 - p) * A_edge + p * B_edge
                out2 = (1 - p) * A_int  + p * B_int
                out3 = (1 - p) * A_rm   + p * B_rm

                write_frame(writers["edge_only"], out1)
                write_frame(writers["int_only"],  out2)
                write_frame(writers["edge_rm"],   out3)

            if i % 10 == 0:
                print(f"step {i}/{len(frames)-1} done")

    finally:
        for w in writers.values():
            w.close()

    print("saved:")
    print(" -", out_edge_only)
    print(" -", out_interior_only)
    print(" -", out_edge_removed)

if __name__ == "__main__":
    main()
