import math
from pathlib import Path
import numpy as np
import cv2
import imageio.v2 as imageio

# ---------- helpers ----------
def clamp01(x):
    return np.clip(x, 0.0, 1.0)

def phase_linearized_weight_alpha(p: float, d: float, alpha: float = 1.0) -> float:
    """
    w_ph = clip( (tan(d p)*alpha) / ( sin d + tan(d p)*alpha - tan(d p)*cos d ), 0, 1 )
    p in [0,1], d in radians
    """
    p = float(np.clip(p, 0.0, 1.0))
    if p <= 0.0: return 0.0
    if p >= 1.0: return 1.0

    alpha = float(np.clip(alpha, 0.2, 5.0))
    k = d * p
    cosK = math.cos(k)
    sinK = math.sin(k)
    if abs(cosK) < 1e-6:
        cosK = 1e-6 * (1.0 if cosK >= 0 else -1.0)

    tanK = sinK / cosK
    sinD = math.sin(d)
    cosD = math.cos(d)

    denom = (sinD + tanK * alpha - tanK * cosD)
    if abs(denom) < 1e-8:
        return 1.0 if denom >= 0 else 0.0

    w = (tanK * alpha) / denom
    return float(np.clip(w, 0.0, 1.0))

def ampnorm_gain(w: float, d: float, eps: float = 1e-6, gain_cap: float = 4.0) -> float:
    """
    A(w) = sqrt((1-w)^2 + w^2 + 2w(1-w)cos(d))
    gain = min( 1/max(eps, A(w)), gain_cap )
    """
    w = float(np.clip(w, 0.0, 1.0))
    A = math.sqrt((1-w)**2 + w**2 + 2*w*(1-w)*math.cos(d))
    g = 1.0 / max(eps, A)
    return float(min(g, gain_cap))

# ---------- IO ----------
def read_rgb_u8(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb

def to_float01(rgb_u8: np.ndarray) -> np.ndarray:
    return rgb_u8.astype(np.float32) / 255.0

def to_u8(rgb_f: np.ndarray) -> np.ndarray:
    return (clamp01(rgb_f) * 255.0 + 0.5).astype(np.uint8)

# ---------- synth ----------
def synth_frame(A: np.ndarray, B: np.ndarray, p: float, mode: str,
                d: float, alpha: float, beta: float,
                gain_cap: float) -> np.ndarray:
    """
    A,B: float RGB in [0,1] (注意：你这里输入的是 bandpass PNG 的可视化结果)
    """
    if mode == "lin":
        w = p
        out = (1 - w) * A + w * B
        return out

    if mode == "phase":
        w = phase_linearized_weight_alpha(p, d=d, alpha=alpha)
        out = (1 - w) * A + w * B
        return out

    if mode == "beta":
        w_lin = p
        w_ph  = phase_linearized_weight_alpha(p, d=d, alpha=alpha)
        w_use = (1 - beta) * w_lin + beta * w_ph
        out = (1 - w_use) * A + w_use * B
        return out

    if mode == "ampnorm":
        w = phase_linearized_weight_alpha(p, d=d, alpha=alpha)
        mix = (1 - w) * A + w * B
        g = ampnorm_gain(w, d=d, gain_cap=gain_cap)

        base = 0.5
        out = base + (mix - base) * g

        # ---- DC lock: 强制每帧平均亮度回到 base，消除“呼吸闪烁”----
        dc = out.mean(axis=(0, 1), keepdims=True)     # shape (1,1,3)
        out = out + (base - dc)

        out = clamp01(out)
        return out

    raise ValueError(f"unknown mode: {mode}")

def make_video(frames: list[Path], out_mp4: Path, mode: str,
               fps: int = 60, seconds_per_step: float = 1.0,
               d: float = 0.4 * math.pi, alpha: float = 1.0, beta: float = 0.5,
               gain_cap: float = 4.0):
    writer = imageio.get_writer(str(out_mp4), fps=fps, codec="libx264", quality=8)
    N = int(round(fps * seconds_per_step))
    try:
        for i in range(len(frames) - 1):
            A_u8 = read_rgb_u8(frames[i])
            B_u8 = read_rgb_u8(frames[i+1])
            A = to_float01(A_u8)
            B = to_float01(B_u8)

            for k in range(N):
                p = k / (N - 1) if N > 1 else 1.0
                out = synth_frame(A, B, p, mode=mode, d=d, alpha=alpha, beta=beta, gain_cap=gain_cap)
                writer.append_data(to_u8(out))
        print("saved:", out_mp4)
    finally:
        writer.close()

def main():
    in_dir = Path(r"D:\vectionProject\public\camear1images_bandpass_hi")  # 你已生成的bandpass帧
    out_dir = in_dir.parent / "bandpass_videos"
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = sorted([p for p in in_dir.glob("*.png")])
    if len(frames) < 2:
        raise RuntimeError("Not enough frames in camear1images_bandpass")

    # ===== 你要调的参数 =====
    fps = 60
    seconds_per_step = 1.0   # 你说 1Hz，一秒一张
    d = 0.40 * math.pi       # 先用你之前试过的
    alpha = 1.0
    beta = 0.5
    gain_cap = 4.0

    # make_video(frames, out_dir / "band_hi_lin.mp4",   mode="lin",    fps=fps, seconds_per_step=seconds_per_step, d=d, alpha=alpha, beta=beta, gain_cap=gain_cap)
    # make_video(frames, out_dir / "band_hi_phase.mp4", mode="phase",  fps=fps, seconds_per_step=seconds_per_step, d=d, alpha=alpha, beta=beta, gain_cap=gain_cap)
    make_video(frames, out_dir / "band_hi_ampnorm.mp4", mode="ampnorm", fps=fps, seconds_per_step=seconds_per_step, d=d, alpha=alpha, beta=beta, gain_cap=gain_cap)
    # 可选：beta-mix
    make_video(frames, out_dir / "band_hi_beta.mp4",  mode="beta",   fps=fps, seconds_per_step=seconds_per_step, d=d, alpha=alpha, beta=beta, gain_cap=gain_cap)

if __name__ == "__main__":
    main()
