import os
import cv2
import numpy as np

# ======================
# CONFIG
# ======================
VID_TOP = r"D:\vectionProject\public\camera3images\continuous.mp4"                 # 上：linear
VID_BOTTOM = r"D:\vectionProject\public\camera3images\cam3_linear.mp4"  # 下：gauss3tap
# VID_BOTTOM = r"D:\vectionProject\public\camera3images\cam3_truncgauss3tap_sigma0p6.mp4"  # 下：gauss3tap

OUT_MP4 = r"D:\vectionProject\public\camera3images\stack_continuous_top_linear_bottom.mp4"

# 输出帧率：None 表示沿用 top 视频的 fps
OUT_FPS = None

# 尺寸策略：
# - "match_width": 两个视频都缩放到相同宽度（取较小宽度），高度等比
# - "match_height": 两个视频都缩放到相同高度（取较小高度），宽度等比
# - "no_resize": 不缩放，直接按最大宽度做 padding（需要两者宽高差别不太离谱）
RESIZE_MODE = "match_width"

# padding 背景色（灰度 0~255），0=黑
PAD_VALUE = 0

# 编码器（Windows上 mp4v 通常可用；如果不行可改成 'avc1' 试试）
FOURCC = "mp4v"


# ======================
# Helpers
# ======================
def get_video_info(cap: cv2.VideoCapture):
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return w, h, fps, n


def resize_keep_aspect(frame: np.ndarray, target_w=None, target_h=None):
    h, w = frame.shape[:2]
    if target_w is None and target_h is None:
        return frame
    if target_w is not None:
        scale = target_w / float(w)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
    else:
        scale = target_h / float(h)
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


def pad_to_width(frame: np.ndarray, target_w: int, pad_value: int):
    h, w = frame.shape[:2]
    if w == target_w:
        return frame
    if w > target_w:
        # 如果意外超了，就中心裁剪
        x0 = (w - target_w) // 2
        return frame[:, x0:x0 + target_w]
    # pad 左右
    pad_left = (target_w - w) // 2
    pad_right = target_w - w - pad_left
    return cv2.copyMakeBorder(
        frame, 0, 0, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT, value=(pad_value, pad_value, pad_value)
    )


def ensure_bgr(frame: np.ndarray):
    if frame is None:
        return None
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


# ======================
# Main
# ======================
def main():
    if not os.path.exists(VID_TOP):
        raise FileNotFoundError(VID_TOP)
    if not os.path.exists(VID_BOTTOM):
        raise FileNotFoundError(VID_BOTTOM)

    cap_top = cv2.VideoCapture(VID_TOP)
    cap_bot = cv2.VideoCapture(VID_BOTTOM)
    if not cap_top.isOpened():
        raise RuntimeError(f"Cannot open: {VID_TOP}")
    if not cap_bot.isOpened():
        raise RuntimeError(f"Cannot open: {VID_BOTTOM}")

    w1, h1, fps1, n1 = get_video_info(cap_top)
    w2, h2, fps2, n2 = get_video_info(cap_bot)

    # 输出 fps
    out_fps = fps1 if (OUT_FPS is None) else float(OUT_FPS)

    # 目标尺寸策略
    if RESIZE_MODE == "match_width":
        target_w = min(w1, w2)
        # 高度由缩放后决定（逐帧算也行，但这里先估算）
        # 先按原始比例计算目标高度
        top_h_est = int(round(h1 * (target_w / float(w1))))
        bot_h_est = int(round(h2 * (target_w / float(w2))))
        out_w = target_w
        out_h = top_h_est + bot_h_est

    elif RESIZE_MODE == "match_height":
        target_h = min(h1, h2)
        top_w_est = int(round(w1 * (target_h / float(h1))))
        bot_w_est = int(round(w2 * (target_h / float(h2))))
        out_w = max(top_w_est, bot_w_est)  # 可能需要 padding
        out_h = target_h + target_h

    elif RESIZE_MODE == "no_resize":
        out_w = max(w1, w2)
        out_h = h1 + h2

    else:
        raise ValueError("RESIZE_MODE must be one of: match_width, match_height, no_resize")

    os.makedirs(os.path.dirname(OUT_MP4), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*FOURCC)
    writer = cv2.VideoWriter(OUT_MP4, fourcc, out_fps, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open VideoWriter: {OUT_MP4}")

    # 以较短视频为准（任意一个结束就停止）
    frame_idx = 0
    while True:
        ok1, f1 = cap_top.read()
        ok2, f2 = cap_bot.read()
        if not ok1 or not ok2:
            break

        f1 = ensure_bgr(f1)
        f2 = ensure_bgr(f2)

        if RESIZE_MODE == "match_width":
            f1r = resize_keep_aspect(f1, target_w=out_w)
            f2r = resize_keep_aspect(f2, target_w=out_w)
            # 由于取整，宽度可能差 1px，做一次 pad/crop 保证一致
            f1r = pad_to_width(f1r, out_w, PAD_VALUE)
            f2r = pad_to_width(f2r, out_w, PAD_VALUE)
            stacked = np.vstack([f1r, f2r])

        elif RESIZE_MODE == "match_height":
            target_h = out_h // 2
            f1r = resize_keep_aspect(f1, target_h=target_h)
            f2r = resize_keep_aspect(f2, target_h=target_h)
            f1r = pad_to_width(f1r, out_w, PAD_VALUE)
            f2r = pad_to_width(f2r, out_w, PAD_VALUE)
            stacked = np.vstack([f1r, f2r])

        else:  # no_resize
            f1r = pad_to_width(f1, out_w, PAD_VALUE)
            f2r = pad_to_width(f2, out_w, PAD_VALUE)
            stacked = np.vstack([f1r, f2r])

        # 最终安全检查
        if stacked.shape[1] != out_w or stacked.shape[0] != out_h:
            # 如果因为取整造成高度不一致，做一次强制 resize（尽量避免，但确保不崩）
            stacked = cv2.resize(stacked, (out_w, out_h), interpolation=cv2.INTER_AREA)

        writer.write(stacked)
        frame_idx += 1

    cap_top.release()
    cap_bot.release()
    writer.release()
    print(f"Saved: {OUT_MP4}")
    print(f"Frames written: {frame_idx}, out_size=({out_w}x{out_h}), fps={out_fps:.3f}")


if __name__ == "__main__":
    main()
