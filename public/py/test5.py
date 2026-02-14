import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio.v2 as imageio

# ======================
# CONFIG
# ======================
img_dir = Path(r"D:\vectionProject\public\camear1images")
pattern = "cam1_*.png"
out_mp4 = img_dir / "cam1_ampnorm_two_rows.mp4"

fps = 30
sec_per_transition = 1.0

# ----------------------
# Key params
# ----------------------
# ✅ 推荐：0.40~0.50π（不会碰到 tan 奇点，最平滑）
# d_step = 0.45 * math.pi

# 如果你坚持用 0.9π（更激进），这份代码会用 safe_tan + 平滑来抑制抽动
d_step = 0.90 * math.pi

alpha = 1.0
amp_eps = 0.08
gain_cap = 2.5
disp_scale = 2.5

# ✅ 关键开关：
# cam1_*.png 若是普通截图/录屏帧：通常是 sRGB -> True
# 若你确认导出时把线性值直接写进 PNG：改 False
INPUT_FRAMES_ARE_SRGB = True

# ----------------------
# Smoothing (reduce "jerk")
# ----------------------
# w 平滑：越小越顺，但会更“拖”
W_SMOOTH = 0.20      # 建议 0.10~0.30
# gain 平滑：减少 gain 抖动
G_SMOOTH = 0.20      # 建议 0.10~0.30

# safe_tan 强度：越大越不容易炸，但越偏离原公式
SAFE_TAN_CMIN = 0.06  # 建议 0.03~0.10

_FONT = ImageFont.load_default()

# ======================
# Color: sRGB <-> Linear
# ======================
def srgb_to_linear(x):
    x = np.clip(x, 0.0, 1.0).astype(np.float32)
    a = 0.055
    return np.where(x <= 0.04045, x / 12.92, ((x + a) / (1 + a)) ** 2.4).astype(np.float32)

def linear_to_srgb(x):
    x = np.clip(x, 0.0, 1.0).astype(np.float32)
    a = 0.055
    return np.where(x <= 0.0031308, x * 12.92, (1 + a) * (x ** (1 / 2.4)) - a).astype(np.float32)

# ======================
# Math
# ======================
def safe_tan(k: float, cmin: float = 0.06) -> float:
    """
    Continuous tan approximation:
    tan(k) = sin(k)/cos(k), but avoid blow-up near cos(k)=0.
    This keeps continuity and removes the "jerk" from the singularity.
    """
    s = math.sin(k)
    c = math.cos(k)
    # continuous "softened" denominator
    c_eff = math.copysign(math.sqrt(c * c + cmin * cmin), c)
    return s / c_eff

def phase_linearized_weight_alpha(p: float, d: float, alpha: float = 1.0, cmin: float = 0.06) -> float:
    p = float(np.clip(p, 0.0, 1.0))
    if p <= 0.0: return 0.0
    if p >= 1.0: return 1.0

    alpha = float(np.clip(alpha, 0.2, 5.0))
    k = d * p

    # ✅关键：用 safe_tan，别用 sin/cos + hard clamp
    s = safe_tan(k, cmin=cmin)

    sinD = math.sin(d)
    cosD = math.cos(d)

    denom = sinD + s * alpha - s * cosD
    if abs(denom) < 1e-6:
        return p

    w = (s * alpha) / denom
    return float(np.clip(w, 0.0, 1.0))

def amplitude_of_mix(w: float, d: float) -> float:
    c = math.cos(d)
    A2 = (1 - w) ** 2 + w ** 2 + 2 * w * (1 - w) * c
    return math.sqrt(max(0.0, A2))

def add_label(frame_rgb: np.ndarray, text: str, xy=(10, 10)) -> np.ndarray:
    im = Image.fromarray(frame_rgb)
    draw = ImageDraw.Draw(im)
    x, y = xy
    draw.text((x + 1, y + 1), text, font=_FONT, fill=(0, 0, 0))
    draw.text((x, y), text, font=_FONT, fill=(255, 255, 255))
    return np.array(im)

def ema(prev, cur, a):
    # a in (0,1], smaller => smoother
    if prev is None:
        return cur
    return prev + (cur - prev) * a

# ======================
# Image helpers (linear pipeline)
# ======================
def load_gray_lin01(path: Path) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    arr = (np.asarray(im).astype(np.float32) / 255.0)

    if INPUT_FRAMES_ARE_SRGB:
        rgb_lin = srgb_to_linear(arr)
    else:
        rgb_lin = np.clip(arr, 0.0, 1.0).astype(np.float32)

    gray_lin = rgb_lin[..., 0] * 0.2126 + rgb_lin[..., 1] * 0.7152 + rgb_lin[..., 2] * 0.0722
    return gray_lin.astype(np.float32)

def gray_lin01_to_rgb8(gray_lin: np.ndarray) -> np.ndarray:
    g_srgb = linear_to_srgb(gray_lin)
    g8 = np.clip(g_srgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return np.stack([g8, g8, g8], axis=-1)

# ======================
# Build video
# ======================
def build_two_row_ampnorm_video(image_paths):
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    num_images = len(image_paths)
    if num_images < 2:
        raise ValueError("Need at least 2 images.")

    total_transitions = num_images - 1
    frames_per_transition = int(round(sec_per_transition * fps))

    grays = [load_gray_lin01(p) for p in image_paths]
    H, W = grays[0].shape

    frames = []
    for seg in range(total_transitions):
        Aname = letters[seg % len(letters)]
        Bname = letters[(seg + 1) % len(letters)]

        I0 = grays[seg]
        I1 = grays[seg + 1]

        # state for smoothing (reset each transition)
        w_s = None
        g_s = None

        # base (per-pixel) — aligns with your Unity baseMode=avg idea
        base = 0.5 * (I0 + I1)

        for f in range(frames_per_transition):
            p = f / max(1, frames_per_transition - 1)  # 0..1

            # TOP: plain linear dissolve (in linear space)
            w_lin = p
            mix_top = (1.0 - w_lin) * I0 + w_lin * I1

            # BOT: phase-linearized + ampnorm
            w_raw = phase_linearized_weight_alpha(p, d_step, alpha=alpha, cmin=SAFE_TAN_CMIN)
            w_s = ema(w_s, w_raw, W_SMOOTH)  # ✅ 平滑 w，减少“抽动”

            mix_bot = (1.0 - w_s) * I0 + w_s * I1

            Aamp = amplitude_of_mix(w_s, d_step)
            gain_raw = 1.0 / max(amp_eps, Aamp)
            gain_raw = min(gain_raw, gain_cap)
            g_s = ema(g_s, gain_raw, G_SMOOTH)  # ✅ 平滑 gain

            sig = mix_bot - base
            out_bot = base + sig * (g_s / max(1e-6, disp_scale))
            out_bot = np.clip(out_bot, 0.0, 1.0)

            img_top = gray_lin01_to_rgb8(mix_top)
            img_bot = gray_lin01_to_rgb8(out_bot)
            frame = np.concatenate([img_top, img_bot], axis=0)

            frame = add_label(frame, f"TOP Linear {Aname}->{Bname}  p={p:0.2f}", (10, 10))
            frame = add_label(
                frame,
                f"BOT phase+ampnorm  w_raw={w_raw:0.2f}  w_s={w_s:0.2f}  A={Aamp:0.2f}  g={g_s:0.2f}  scale={disp_scale}",
                (10, H + 10),
            )
            frame = add_label(
                frame,
                f"d={d_step/math.pi:0.2f}π  fps={fps}  Wsmooth={W_SMOOTH}  Gsmooth={G_SMOOTH}  cmin={SAFE_TAN_CMIN}",
                (10, 2 * H - 18),
            )

            frames.append(frame)

    return frames

def write_mp4(path: Path, frames, fps: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(path), fps=fps, codec="libx264", quality=8, macro_block_size=1)
    for fr in frames:
        writer.append_data(fr)
    writer.close()

def main():
    image_paths = sorted(img_dir.glob(pattern))
    print(f"Found {len(image_paths)} images in: {img_dir}")
    if len(image_paths) < 2:
        print("Not enough images.")
        return

    frames = build_two_row_ampnorm_video(image_paths)
    write_mp4(out_mp4, frames, fps)
    print(f"Saved video: {out_mp4}")

if __name__ == "__main__":
    main()
