import cv2
import numpy as np
from pathlib import Path

# =========================
# 你只需要改这里
# =========================
IMG_DIR = Path(r"D:\vectionProject\public\camear1images")
IMG_GLOB = "cam1_*.png"

# 你那张“单独截出来的树”（第三张图）请保存为这个文件名，并放到同一目录
TEMPLATE_PATH = IMG_DIR / "tree_template.png"

# 输出 debug 视频（会画框）
OUT_DEBUG_MP4 = IMG_DIR / "tree_track_debug.mp4"
FPS = 1.0   # 你的图片是一秒一张，所以 fps=1
# =========================

# 多尺度匹配（树大小稍微变动时用）
SCALES = [0.90, 0.95, 1.00, 1.05, 1.10]

# 搜索窗口半径：上一帧位置附近找（更稳、更快）
SEARCH_RADIUS = 220  # px，可调大/小

def read_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return img

def best_match(gray: np.ndarray, tmpl0: np.ndarray, last_xy=None):
    """
    gray: HxW uint8
    tmpl0: ht x wt uint8
    last_xy: (x,y,w,h) from previous frame; if provided, search near it
    return: (score, x, y, w, h)
    """
    H, W = gray.shape[:2]

    # restrict search region
    if last_xy is not None:
        lx, ly, lw, lh = last_xy
        cx = lx + lw // 2
        cy = ly + lh // 2
        x0 = max(0, cx - SEARCH_RADIUS)
        y0 = max(0, cy - SEARCH_RADIUS)
        x1 = min(W, cx + SEARCH_RADIUS)
        y1 = min(H, cy + SEARCH_RADIUS)
        roi = gray[y0:y1, x0:x1]
        ox, oy = x0, y0
    else:
        roi = gray
        ox, oy = 0, 0

    best = None
    for s in SCALES:
        tmpl = cv2.resize(tmpl0, (0, 0), fx=s, fy=s, interpolation=cv2.INTER_AREA)
        th, tw = tmpl.shape[:2]
        if th >= roi.shape[0] or tw >= roi.shape[1]:
            continue

        res = cv2.matchTemplate(roi, tmpl, cv2.TM_CCOEFF_NORMED)
        _, maxv, _, maxloc = cv2.minMaxLoc(res)
        x = int(maxloc[0] + ox)
        y = int(maxloc[1] + oy)
        cand = (float(maxv), x, y, int(tw), int(th))

        if best is None or cand[0] > best[0]:
            best = cand

    if best is None:
        # fallback
        th, tw = tmpl0.shape[:2]
        return (0.0, 0, 0, tw, th)
    return best

def main():
    paths = sorted(IMG_DIR.glob(IMG_GLOB))
    if len(paths) < 2:
        raise RuntimeError(f"Need >=2 images, got {len(paths)} in {IMG_DIR}")

    tmpl0 = read_gray(TEMPLATE_PATH)

    # init writer
    first = cv2.imread(str(paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Failed to read: {paths[0]}")
    H, W = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUT_DEBUG_MP4), fourcc, FPS, (W, H))

    xs, ys, scores = [], [], []
    last = None

    for idx, p in enumerate(paths):
        frame = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Failed to read: {p}")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        score, x, y, tw, th = best_match(gray, tmpl0, last)
        last = (x, y, tw, th)

        cx = x + tw * 0.5
        cy = y + th * 0.5
        xs.append(cx)
        ys.append(cy)
        scores.append(score)

        # draw debug box
        cv2.rectangle(frame, (x, y), (x + tw, y + th), (0, 255, 0), 2)
        cv2.putText(frame, f"{idx:03d} score={score:.3f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

        writer.write(frame)

    writer.release()

    xs = np.array(xs, dtype=np.float32)
    ys = np.array(ys, dtype=np.float32)

    # velocity in px/sec since FPS=1
    vx = np.diff(xs) * FPS
    vy = np.diff(ys) * FPS

    print("Images:", len(paths))
    print("Template:", TEMPLATE_PATH)
    print("Debug video:", OUT_DEBUG_MP4)
    print("Match score mean/std:", float(np.mean(scores)), float(np.std(scores)))
    print("vx mean/std (px/s):", float(vx.mean()), float(vx.std()))
    print("vy mean/std (px/s):", float(vy.mean()), float(vy.std()))

if __name__ == "__main__":
    main()
