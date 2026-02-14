import re
from pathlib import Path

import numpy as np
from PIL import Image
import imageio.v2 as imageio
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


# ======================
# CONFIG
# ======================
SRC_DIR = Path(r"D:\vectionProject\public\A-continuous-images")
SRC_PATTERN = "cam0_*.png"

START_IDX = 960
END_IDX = 1319

OUT_GRAY_DIR = Path(r"D:\vectionProject\public\camera3fullimages")
OUT_VIDEO = Path(r"D:\vectionProject\public\camera3images") / "continuous.mp4"

FPS = 60
BITRATE = "16M"


# ======================
# Helpers
# ======================
def extract_index(p: Path) -> int | None:
    """
    cam0_000900.png -> 900
    """
    m = re.search(r"cam0_(\d+)\.png$", p.name, flags=re.IGNORECASE)
    if not m:
        return None
    return int(m.group(1))


def ensure_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True)


def to_gray_and_save(src_path: Path, dst_path: Path) -> np.ndarray:
    """
    Read src image, convert to grayscale (L), save to dst_path (PNG),
    return float frame in [0,1] (H,W).
    """
    im = Image.open(src_path).convert("L")
    ensure_dir(dst_path.parent)
    im.save(dst_path)  # save grayscale image (do NOT overwrite src)
    arr = np.asarray(im, dtype=np.float32) / 255.0
    return arr


def write_video(frames_float01, out_path: Path, fps: int):
    out_path = str(out_path)
    with imageio.get_writer(out_path, fps=fps, codec="libx264", bitrate=BITRATE) as w:
        for frame in frames_float01:
            f = np.clip(frame, 0.0, 1.0)
            f8 = (f * 255.0).astype(np.uint8)
            # make 3 channels for compatibility
            if f8.ndim == 2:
                f8 = np.stack([f8] * 3, axis=-1)
            w.append_data(f8)


# ======================
# Main
# ======================
def main():
    if not SRC_DIR.exists():
        raise RuntimeError(f"源目录不存在: {SRC_DIR}")

    ensure_dir(OUT_GRAY_DIR)
    ensure_dir(OUT_VIDEO.parent)

    # 1) collect and filter frames by index range
    candidates = sorted(SRC_DIR.glob(SRC_PATTERN))
    selected = []
    for p in candidates:
        idx = extract_index(p)
        if idx is None:
            continue
        if START_IDX <= idx <= END_IDX:
            selected.append((idx, p))

    if not selected:
        raise RuntimeError(
            f"没有找到范围内图片：{SRC_DIR}\\{SRC_PATTERN}，"
            f"index {START_IDX}..{END_IDX}"
        )

    # ensure ordered by index
    selected.sort(key=lambda x: x[0])

    # sanity check: confirm start/end exist
    found_start = selected[0][0]
    found_end = selected[-1][0]
    if found_start != START_IDX or found_end != END_IDX:
        print(f"[WARN] 实际找到的范围是 {found_start}..{found_end}，"
              f"与你要求的 {START_IDX}..{END_IDX} 不完全一致。")

    # 2) convert to gray + save + build video frames
    frames = []
    first_shape = None

    for idx, src_path in selected:
        dst_path = OUT_GRAY_DIR / src_path.name  # keep same filename
        frame = to_gray_and_save(src_path, dst_path)

        if first_shape is None:
            first_shape = frame.shape
        else:
            if frame.shape != first_shape:
                raise RuntimeError(
                    f"图片尺寸不一致：{src_path.name}={frame.shape}，"
                    f"期望={first_shape}"
                )

        frames.append(frame)

    print(f"已保存灰度PNG: {OUT_GRAY_DIR}  (共 {len(frames)} 张)")

    # 3) write mp4
    write_video(frames, OUT_VIDEO, FPS)
    duration = len(frames) / FPS
    print(f"视频已生成: {OUT_VIDEO}")
    print(f"FPS={FPS}, Frames={len(frames)}, Duration≈{duration:.3f}s")


if __name__ == "__main__":
    main()
