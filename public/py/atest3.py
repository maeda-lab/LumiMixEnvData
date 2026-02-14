import numpy as np
from pathlib import Path
from PIL import Image
import imageio.v2 as imageio

# ---------- utils: sRGB <-> Linear ----------
def srgb_to_linear(x):
    x = np.clip(x, 0.0, 1.0)
    a = 0.055
    return np.where(x <= 0.04045, x / 12.92, ((x + a) / (1 + a)) ** 2.4)

def linear_to_srgb(x):
    x = np.clip(x, 0.0, 1.0)
    a = 0.055
    return np.where(x <= 0.0031308, x * 12.92, (1 + a) * (x ** (1 / 2.4)) - a)

# ---------- gaussian blur (separable) ----------
def gaussian_kernel1d(sigma, radius=None):
    if sigma <= 0:
        return np.array([1.0], dtype=np.float32)
    if radius is None:
        radius = int(np.ceil(3 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(x * x) / (2 * sigma * sigma))
    k /= np.sum(k)
    return k

def convolve1d(img, k, axis):
    # img: (H,W,C) float32
    pad = len(k) // 2
    if axis == 0:
        padded = np.pad(img, ((pad, pad), (0, 0), (0, 0)), mode="reflect")
        out = np.zeros_like(img)
        for i in range(img.shape[0]):
            out[i] = np.tensordot(padded[i:i+len(k)], k, axes=([0], [0]))
        return out
    else:
        padded = np.pad(img, ((0, 0), (pad, pad), (0, 0)), mode="reflect")
        out = np.zeros_like(img)
        for j in range(img.shape[1]):
            out[:, j] = np.tensordot(padded[:, j:j+len(k)], k, axes=([1], [0]))
        return out

def gaussian_blur(img, sigma):
    k = gaussian_kernel1d(sigma)
    tmp = convolve1d(img, k, axis=1)
    out = convolve1d(tmp, k, axis=0)
    return out

# ---------- DoG bandpass ----------
def dog_bandpass_linear(img_lin, sigma_small, sigma_large):
    # img_lin: linear RGB in [0,1], float32
    lo1 = gaussian_blur(img_lin, sigma_small)
    lo2 = gaussian_blur(img_lin, sigma_large)
    band = lo1 - lo2
    return band

def read_image_as_linear(path: Path):
    im = Image.open(path).convert("RGB")
    arr = np.asarray(im).astype(np.float32) / 255.0
    arr_lin = srgb_to_linear(arr)
    return arr_lin

def save_linear_as_png(path: Path, img_lin, gain=1.0, bias=0.5):
    """
    带通结果有正有负，必须映射到[0,1]才能保存PNG。
    常用：bias=0.5 把0映射到中灰；gain 控制对比度。
    """
    y = bias + gain * img_lin
    y = np.clip(y, 0.0, 1.0)
    y_srgb = linear_to_srgb(y)
    out = (y_srgb * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(out).save(path)

def main():
    in_dir  = Path(r"D:\vectionProject\public\camear1images")  # 改成你的帧目录
    out_dir = Path(r"D:\vectionProject\public\camear1images_bandpass")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ====== 关键参数：sigma（像素） ======
    # 建议先试一组“中频”：小模糊2px，大模糊8px
    sigma_small = 2.0
    sigma_large = 8.0

    # 带通可视化映射参数（只影响保存PNG观看，不影响你后续若用float处理）
    vis_gain = 4.0   # 对比度，越大越“边缘化”
    vis_bias = 0.5   # 让0变成中灰

    paths = sorted([p for p in in_dir.glob("*.png")])
    if not paths:
        paths = sorted([p for p in in_dir.glob("*.jpg")])

    print("frames:", len(paths))
    for i, p in enumerate(paths):
        img_lin = read_image_as_linear(p)
        band = dog_bandpass_linear(img_lin, sigma_small, sigma_large)

        # 仅为了保存可见图：band 是正负值，用 bias+gain 映射
        out_path = out_dir / p.name
        save_linear_as_png(out_path, band, gain=vis_gain, bias=vis_bias)

        if i % 50 == 0:
            print("processed", i, p.name)

    print("done:", out_dir)

if __name__ == "__main__":
    main()
