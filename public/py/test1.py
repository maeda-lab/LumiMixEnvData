from PIL import Image
from pathlib import Path
import numpy as np
import imageio
from typing import List, Tuple

def load_images(image_paths: List[str]) -> Tuple[List[np.ndarray], Tuple[int,int]]:
    imgs = []
    if not image_paths:
        return [], (0,0)
    first = Image.open(image_paths[0]).convert("RGB")
    target_size = first.size  # (w,h)
    imgs.append(np.asarray(first, dtype=np.float32) / 255.0)
    for p in image_paths[1:]:
        im = Image.open(p).convert("RGB")
        if im.size != target_size:
            im = im.resize(target_size, Image.LANCZOS)
        imgs.append(np.asarray(im, dtype=np.float32) / 255.0)
    w, h = target_size
    return imgs, (w, h)

def lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return a * (1.0 - t) + b * t

def make_crossfade_video(image_paths: List[str], out_mp4="crossfade.mp4", fps=30, transition_sec=1.0):
    imgs, (w, h) = load_images(image_paths)
    if not imgs:
        raise ValueError("no images")
    n = len(imgs)
    frames_per = max(1, int(round(fps * transition_sec)))
    writer = imageio.get_writer(out_mp4, fps=fps, codec="libx264", ffmpeg_params=["-pix_fmt", "yuv420p"])
    try:
        # For each adjacent pair do a crossfade of duration transition_sec (no overlap with other pairs)
        for i in range(n - 1):
            a = imgs[i]
            b = imgs[i + 1]
            for k in range(frames_per):
                t = (k + 1) / frames_per  # 0..1
                frame = lerp(a, b, t)
                writer.append_data((np.clip(frame, 0, 1) * 255).astype(np.uint8))
        # optional: keep last frame visible for a short moment (0.5s)
        hold_last = max(0, int(round(fps * 0.5)))
        last = imgs[-1]
        for _ in range(hold_last):
            writer.append_data((np.clip(last, 0, 1) * 255).astype(np.uint8))
    finally:
        writer.close()

if __name__ == "__main__":
    here = Path(__file__).parent
    names = ["1", "2", "3", "4", "5"]
    exts = [".png", ".jpg", ".jpeg"]
    image_paths = []
    for n in names:
        found = None
        for e in exts:
            p = here / f"{n}{e}"
            if p.exists():
                found = str(p)
                break
        if found:
            image_paths.append(found)
    if len(image_paths) != 5:
        missing = [n for n in names if not any((here / f"{n}{e}").exists() for e in exts)]
        raise FileNotFoundError(f"Missing image files for: {', '.join(missing)} (expect 1..5 with .png/.jpg/.jpeg)")

    out_mp4 = here / "crossfade_1s_per_transition.mp4"
    make_crossfade_video(image_paths, out_mp4=str(out_mp4), fps=30, transition_sec=1.0)
    print("Saved:", out_mp4)
