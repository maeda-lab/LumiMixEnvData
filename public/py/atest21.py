import os
import math
import csv
from pathlib import Path

import numpy as np
from PIL import Image
import imageio.v2 as imageio


# ======================
# CONFIG（改成你自己的路径和文件名）
# ======================
IMG_DIR   = Path(r"D:\vectionProject\public\camear1images")
IMG_GLOB  = "cam1_*.png"   # 帧文件名模式
CSV_PATH  = Path(r"D:\vectionProject\public\camear1images\cam1_tree_bbox.csv")
OUT_MP4   = Path(r"D:\vectionProject\public\camear1images\cam1_local_compensated_gray.mp4")

FPS = 60
SECONDS_PER_STEP = 1.0
N_PER_STEP = int(round(FPS * SECONDS_PER_STEP))

# 输入帧是否是普通 sRGB 灰度图片
INPUT_FRAMES_ARE_SRGB = True

# ROI 边缘柔化比例（0~1）
ROI_EDGE_SOFT = 0.3

# ROI 的相位补偿参数（针对树）
D_ROI     = 0.9 * math.pi
ALPHA_ROI = 1.0


# ======================
# sRGB <-> Linear  (标量/灰度)
# ======================
def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    """
    x: [0,1] sRGB
    return: [0,1] linear
    """
    x = np.clip(x, 0.0, 1.0)
    a = 0.055
    return np.where(
        x <= 0.04045,
        x / 12.92,
        ((x + a) / (1.0 + a)) ** 2.4
    )


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    """
    x: [0,1] linear
    return: [0,1] sRGB
    """
    x = np.clip(x, 0.0, 1.0)
    a = 0.055
    return np.where(
        x <= 0.0031308,
        12.92 * x,
        (1.0 + a) * np.power(x, 1.0 / 2.4) - a
    )


# ======================
# 相位线性化权重 w_roi(p; d, alpha)
# ======================
def phase_linearized_weight_alpha(p: float, d: float, alpha: float = 1.0) -> float:
    """
    p: 0..1 (当前 1s 内的归一化时间)
    d: 有效相位范围（rad），比如 0.9*pi
    alpha: 调整斜率
    """
    p = float(np.clip(p, 0.0, 1.0))
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0

    alpha = float(np.clip(alpha, 0.2, 5.0))

    dp = d * p
    cos_dp = math.cos(dp)
    sin_dp = math.sin(dp)

    # 避免 tan 爆炸
    if abs(cos_dp) < 1e-6:
        cos_dp = 1e-6 * (1.0 if cos_dp >= 0 else -1.0)

    tan_dp = sin_dp / cos_dp

    sin_d = math.sin(d)
    cos_d = math.cos(d)

    denom = sin_d + tan_dp * alpha - tan_dp * cos_d
    if abs(denom) < 1e-6:
        denom = 1e-6 * (1.0 if denom >= 0 else -1.0)

    w = (tan_dp * alpha) / denom
    w = float(np.clip(w, 0.0, 1.0))
    return w


# ======================
# ROI mask：椭圆 + 边缘 cos 柔化
# ======================
def make_elliptical_mask(H: int, W: int,
                         cx: float, cy: float,
                         w_box: float, h_box: float,
                         edge_soft: float = 0.3) -> np.ndarray:
    """
    生成形状为 (H,W) 的 mask（float32），
    中心处接近 1，外部 0，边缘区域用 cos 软化。

    cx, cy: bbox 中心
    w_box, h_box: bbox 宽高
    edge_soft: 0~1，表示椭圆半径外圈多少比例用于从 1 -> 0 过渡
    """
    mask = np.zeros((H, W), dtype=np.float32)

    rx = max(w_box / 2.0, 1.0)
    ry = max(h_box / 2.0, 1.0)

    yy, xx = np.ogrid[0:H, 0:W]
    dx = (xx - cx) / rx
    dy = (yy - cy) / ry
    r = np.sqrt(dx * dx + dy * dy)  # 椭圆半径

    edge_soft = float(np.clip(edge_soft, 0.0, 0.99))
    r0 = 1.0 - edge_soft

    inner = (r <= r0)
    trans = (r > r0) & (r <= 1.0)

    mask[inner] = 1.0

    if edge_soft > 0.0:
        # t: 0->1 对应 r: r0->1
        t = (r[trans] - r0) / edge_soft
        mask[trans] = 0.5 * (1.0 + np.cos(math.pi * t))

    # 其他保持 0
    return mask


# ======================
# 读 CSV：filename -> (cx,cy,w,h)
# ======================
def load_bbox_csv(csv_path: Path):
    """
    读取 CSV，返回 map: filename -> (cx,cy,w,h)
    支持常见的列名变体；如果没有表头则按前 5 列解析：
      filename, cx, cy, w, h
    """
    bbox_map = {}
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        # first try DictReader and detect common column names
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if fieldnames:
            fn_lower = [fn.strip().lower() for fn in fieldnames]
            def find_key(cands):
                for c in cands:
                    if c in fn_lower:
                        return fieldnames[fn_lower.index(c)]
                return None
            filename_key = find_key(["filename", "file", "name", "image", "img", "frame"])
            cx_key = find_key(["cx", "center_x", "x", "centerx"])
            cy_key = find_key(["cy", "center_y", "y", "centery"])
            w_key  = find_key(["w", "width"])
            h_key  = find_key(["h", "height"])
            if filename_key and cx_key and cy_key and w_key and h_key:
                for row in reader:
                    fname = row.get(filename_key, "").strip()
                    if not fname:
                        continue
                    try:
                        cx = float(row[cx_key])
                        cy = float(row[cy_key])
                        w  = float(row[w_key])
                        h  = float(row[h_key])
                    except Exception:
                        continue
                    bbox_map[fname] = (cx, cy, w, h)
                return bbox_map
            else:
                # report detected headers for debug, then fall back
                print(f"CSV headers found but required keys not all detected. headers={fieldnames}")
                # fall through to fallback parsing
        # fallback: try plain csv rows (no header) -> expect at least 5 columns
        f.seek(0)
        rdr = csv.reader(f)
        for row in rdr:
            if not row:
                continue
            # skip rows that look like a header (non-numeric in numeric cols)
            if len(row) >= 5:
                fname = row[0].strip()
                try:
                    cx = float(row[1]); cy = float(row[2]); w = float(row[3]); h = float(row[4])
                except Exception:
                    # skip header-like or malformed row
                    continue
                bbox_map[fname] = (cx, cy, w, h)
        return bbox_map


# ======================
# 读图（灰度）
# ======================
def load_images_gray_sorted(img_dir: Path, pattern: str):
    paths = sorted(img_dir.glob(pattern))
    if not paths:
        raise RuntimeError(f"No images found in {img_dir} with pattern {pattern}")

    imgs = []
    names = []
    for p in paths:
        im = Image.open(p).convert("L")   # 灰度
        arr = np.asarray(im, dtype=np.float32) / 255.0  # [0,1] sRGB 灰度
        imgs.append(arr)
        names.append(p.name)
    return paths, names, imgs


# ======================
# 主流程
# ======================
def main():
    # 1) 读入 CSV 的 bbox
    if not CSV_PATH.exists():
        raise RuntimeError(f"CSV not found: {CSV_PATH}")
    bbox_map = load_bbox_csv(CSV_PATH)
    print(f"Loaded {len(bbox_map)} bbox entries from CSV.")

    # 2) 读入灰度图像
    img_paths, img_names, imgs_srgb = load_images_gray_sorted(IMG_DIR, IMG_GLOB)
    print(f"Loaded {len(imgs_srgb)} images.")

    if INPUT_FRAMES_ARE_SRGB:
        imgs_lin = [srgb_to_linear(im) for im in imgs_srgb]
    else:
        imgs_lin = imgs_srgb

    H, W = imgs_lin[0].shape
    print(f"Image size: {W}x{H}")

    # 为了效率，可以缓存每一帧的 mask
    roi_mask_cache = {}

    frames_out = []

    # 3) 对每一对相邻帧做 cross-dissolve + 局部补偿
    for idx in range(len(imgs_lin) - 1):
        img1 = imgs_lin[idx]
        img2 = imgs_lin[idx + 1]
        name1 = img_names[idx]

        print(f"Blending {name1} -> {img_names[idx+1]}")

        # 为当前帧生成/获取 ROI mask（如果 CSV 里没有，就全 0）
        if name1 in roi_mask_cache:
            roi_mask = roi_mask_cache[name1]
        else:
            if name1 in bbox_map:
                cx, cy, w_box, h_box = bbox_map[name1]
                roi_mask = make_elliptical_mask(
                    H, W,
                    cx=cx,
                    cy=cy,
                    w_box=w_box,
                    h_box=h_box,
                    edge_soft=ROI_EDGE_SOFT
                )
            else:
                # 没有 bbox，就不做局部补偿
                roi_mask = np.zeros((H, W), dtype=np.float32)
            roi_mask_cache[name1] = roi_mask

        for n in range(N_PER_STEP):
            # 归一化时间 p ∈ (0,1)
            p = (n + 0.5) / N_PER_STEP

            # 全局线性权重
            w_global = p

            # ROI 相位补偿权重
            w_roi = phase_linearized_weight_alpha(p, d=D_ROI, alpha=ALPHA_ROI)

            # 组合成空间可变权重 w(x,y)
            # roi_mask=1 → w_roi, roi_mask=0 → w_global
            w_map = (1.0 - roi_mask) * w_global + roi_mask * w_roi  # shape=(H,W)

            # 在线性空间中混合
            frame_lin = (1.0 - w_map) * img1 + w_map * img2

            # 转回 sRGB，并量化为 uint8 灰度
            if INPUT_FRAMES_ARE_SRGB:
                frame_srgb = linear_to_srgb(frame_lin)
            else:
                frame_srgb = np.clip(frame_lin, 0.0, 1.0)

            frame_u8 = np.clip(frame_srgb * 255.0 + 0.5, 0, 255).astype(np.uint8)

            # imageio 需要 HxW 或 HxWx1，这里直接给灰度即可
            frames_out.append(frame_u8)

    # 4) 写出 mp4
    print(f"Writing video to: {OUT_MP4}")
    imageio.mimwrite(
        OUT_MP4,
        frames_out,
        fps=FPS,
        codec="libx264",
        quality=8,
        macro_block_size=None,  # 防止分辨率不是16倍数时报错
    )
    print("Done.")


if __name__ == "__main__":
    main()
