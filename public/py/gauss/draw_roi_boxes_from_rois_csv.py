import pandas as pd
from pathlib import Path
import cv2
import numpy as np
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# ========= 配置 =========
IMG_DIR = Path(r"D:\vectionProject\public\camera3images")
CSV_ROI = IMG_DIR / "rois.csv"
OUT_DIR = IMG_DIR / "roiimages"

IMG_PATTERN = "*.png"

# 与你的视频 debug 框保持一致（OpenCV: BGR）
BOX_COLOR_BGR = (0, 255, 0)
BOX_THICKNESS = 4


def clamp_box(x0, y0, x1, y1, W, H):
    x0 = int(round(x0))
    y0 = int(round(y0))
    x1 = int(round(x1))
    y1 = int(round(y1))
    x0 = max(0, min(x0, W - 1))
    y0 = max(0, min(y0, H - 1))
    x1 = max(x0 + 1, min(x1, W - 1))
    y1 = max(y0 + 1, min(y1, H - 1))
    return x0, y0, x1, y1


def main():
    if not CSV_ROI.exists():
        print(f"[roi] 找不到 rois.csv: {CSV_ROI}")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV_ROI)
    cols = set(df.columns)
    print("[roi] CSV 列名:", list(df.columns))

    if "frameName" not in cols:
        print("[roi] rois.csv 中没有 frameName 列，当前列有:", list(df.columns))
        return

    if not {"x_tl", "y_tl", "w_tl", "h_tl"}.issubset(cols):
        print("[roi] 没有找到 x_tl, y_tl, w_tl, h_tl 这四列，不能画 ROI")
        return

    # 预先列出所有图片
    all_imgs = {p.name: p for p in IMG_DIR.glob(IMG_PATTERN)}
    print(f"[roi] 在 {IMG_DIR} 中找到 {len(all_imgs)} 张图片 (pattern={IMG_PATTERN})")

    count_ok = 0
    count_missing = 0

    for _, row in df.iterrows():
        frame_name = str(row["frameName"])  # 例如 'cam1_015.png'
        img_path = all_imgs.get(frame_name)

        if img_path is None:
            print(f"[roi] 警告: 找不到与 frameName='{frame_name}' 对应的图片")
            count_missing += 1
            continue

        # OpenCV 读图：BGR
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[roi] 警告: 图片读取失败: {img_path}")
            count_missing += 1
            continue

        H, W = img.shape[:2]

        x = float(row["x_tl"])
        y = float(row["y_tl"])
        w = float(row["w_tl"])
        h = float(row["h_tl"])

        x0, y0 = x, y
        x1, y1 = x + w, y + h

        # 确保顺序正确
        x0, x1 = sorted([x0, x1])
        y0, y1 = sorted([y0, y1])

        # clamp
        x0, y0, x1, y1 = clamp_box(x0, y0, x1, y1, W, H)

        # 画框（颜色与视频一致）
        cv2.rectangle(img, (x0, y0), (x1, y1), BOX_COLOR_BGR, BOX_THICKNESS)

        out_path = OUT_DIR / img_path.name
        cv2.imwrite(str(out_path), img)

        count_ok += 1

    print(f"[roi] 完成: 生成 {count_ok} 张 ROI 图，找不到/读取失败 {count_missing} 条。")
    print(f"[roi] 输出目录: {OUT_DIR}")


if __name__ == "__main__":
    main()
