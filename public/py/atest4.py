import os
from pathlib import Path
import numpy as np
import cv2

def srgb_to_linear(x):
    x = np.clip(x, 0.0, 1.0)
    a = 0.055
    return np.where(x <= 0.04045, x / 12.92, ((x + a) / (1 + a)) ** 2.4)

def linear_to_srgb(x):
    x = np.clip(x, 0.0, 1.0)
    a = 0.055
    return np.where(x <= 0.0031308, x * 12.92, (1 + a) * (x ** (1 / 2.4)) - a)

def dog_bandpass_linear(img_lin, sigma_small=2.0, sigma_large=8.0):
    lo1 = cv2.GaussianBlur(img_lin, (0, 0), sigmaX=sigma_small, sigmaY=sigma_small, borderType=cv2.BORDER_REFLECT)
    lo2 = cv2.GaussianBlur(img_lin, (0, 0), sigmaX=sigma_large, sigmaY=sigma_large, borderType=cv2.BORDER_REFLECT)
    return lo1 - lo2

def main():
    in_dir  = Path(r"D:\vectionProject\public\camear1images")
    out_dir = Path(r"D:\vectionProject\public\camear1images_bandpass_hi")  # <<< 改这里
    out_dir.mkdir(parents=True, exist_ok=True)

    sigma_small = 1.2   # <<< 改这里（高频）
    sigma_large = 4.0   # <<< 改这里（高频）

    gain = 10.0         # <<< 改这里（建议 10~16）
    bias = 0.5

    exts = (".png", ".jpg", ".jpeg")
    frames = sorted([p for p in in_dir.iterdir() if p.suffix.lower() in exts])

    for i, p in enumerate(frames):
        bgr = cv2.imread(str(p), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        img_lin = srgb_to_linear(rgb)
        band = dog_bandpass_linear(img_lin, sigma_small, sigma_large)

        vis_lin = np.clip(bias + gain * band, 0.0, 1.0)
        vis_rgb = linear_to_srgb(vis_lin)
        out_u8 = (vis_rgb * 255.0 + 0.5).astype(np.uint8)

        out_bgr = cv2.cvtColor(out_u8, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_dir / p.name), out_bgr)

        if i % 50 == 0:
            print("processed", i, "/", len(frames), p.name)

    print("done:", out_dir)

if __name__ == "__main__":
    main()
