import math
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
MP4_PATH = Path(r"D:\vectionProject\public\freq_test_videos\orig_lin.mp4")

FPS = 60
SECONDS_PER_STEP = 1.0
N = int(round(FPS * SECONDS_PER_STEP))  # frames per step

# p(t) must match your synthesis
MODE = "linear"     # "linear" or "gamma"
GAMMA = 2.5         # only if MODE="gamma"
P_EPS = 0.00        # set to your synthesis eps (e.g., 0.15 if used)

MAX_STEPS_TO_ANALYZE = 120

# preprocessing
REMOVE_DC = True            # remove per-frame mean luminance
USE_DOG = True              # recommend True for orig_lin to focus on structure
DOG_SIGMA_SMALL = 2.0
DOG_SIGMA_LARGE = 8.0

# optional ROI (set to None for full frame)
# ROI = (x0, y0, w, h)  # e.g. (1200, 250, 800, 400)
ROI = None

# =========================
# p(t)
# =========================
def p_linear(t: float) -> float:
    return float(np.clip(t, 0.0, 1.0))

def p_gamma(t: float, gamma: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    s = 0.5 - 0.5 * math.cos(math.pi * t)  # smooth 0->1
    a = s**gamma
    b = (1.0 - s)**gamma
    return float(a / (a + b + 1e-12))

def apply_p_eps(p_raw: float, eps_p: float) -> float:
    eps_p = float(np.clip(eps_p, 0.0, 0.45))
    return float(eps_p + (1.0 - 2.0 * eps_p) * np.clip(p_raw, 0.0, 1.0))

def p_of_t(t: float) -> float:
    if MODE == "linear":
        p_raw = p_linear(t)
    elif MODE == "gamma":
        p_raw = p_gamma(t, GAMMA)
    else:
        raise ValueError("MODE must be 'linear' or 'gamma'")
    return apply_p_eps(p_raw, P_EPS)

# =========================
# preprocessing
# =========================
def dog_band(gray: np.ndarray, s_small: float, s_large: float) -> np.ndarray:
    lo1 = cv2.GaussianBlur(gray, (0, 0), sigmaX=s_small, sigmaY=s_small, borderType=cv2.BORDER_REFLECT)
    lo2 = cv2.GaussianBlur(gray, (0, 0), sigmaX=s_large, sigmaY=s_large, borderType=cv2.BORDER_REFLECT)
    return lo1 - lo2

def preprocess(gray_u8: np.ndarray) -> np.ndarray:
    g = gray_u8.astype(np.float32) / 255.0

    if ROI is not None:
        x0, y0, w, h = ROI
        g = g[y0:y0+h, x0:x0+w]

    if REMOVE_DC:
        g = g - float(g.mean())

    if USE_DOG:
        g = dog_band(g, DOG_SIGMA_SMALL, DOG_SIGMA_LARGE)

    return g

# =========================
# edge energy per frame
# =========================
def edge_energy(frame_f: np.ndarray) -> float:
    # Sobel gradient magnitude mean
    gx = cv2.Sobel(frame_f, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(frame_f, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return float(np.mean(np.abs(mag)))

# =========================
# compute per-step aligned curves
# =========================
def per_step_curve_from_video(mp4_path: Path):
    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {mp4_path}")

    energies = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        gray_u8 = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        g = preprocess(gray_u8)
        energies.append(edge_energy(g))
    cap.release()

    energies = np.array(energies, dtype=np.float32)  # length ~ total_frames

    step_len = N  # energies are per frame (not diff), so step has N frames
    num_steps = len(energies) // step_len
    num_steps = min(num_steps, MAX_STEPS_TO_ANALYZE)
    if num_steps <= 0:
        raise RuntimeError("Not enough frames to segment steps. Check FPS/SECONDS_PER_STEP/N.")

    M = energies[:num_steps * step_len].reshape(num_steps, step_len)  # (steps, within-step frames)
    mean_curve = M.mean(axis=0)
    std_curve  = M.std(axis=0)
    return mean_curve, std_curve, num_steps

def ghost_proxy_curve():
    step_len = N
    t_list = [(k / (step_len - 1)) for k in range(step_len)]  # 0..1
    p_list = np.array([p_of_t(t) for t in t_list], dtype=np.float32)
    g = p_list * (1.0 - p_list)
    return g, p_list

def fit_scale(g, y):
    return float((g @ y) / (g @ g + 1e-12))

def main():
    mean_curve, std_curve, steps = per_step_curve_from_video(MP4_PATH)
    g, p_list = ghost_proxy_curve()

    a = fit_scale(g, mean_curve)
    g_scaled = a * g
    corr = float(np.corrcoef(mean_curve, g)[0, 1])

    print("Video:", MP4_PATH)
    print("steps used:", steps, "N:", N)
    print("MODE:", MODE, "GAMMA:", GAMMA, "P_EPS:", P_EPS)
    print("REMOVE_DC:", REMOVE_DC, "USE_DOG:", USE_DOG, "ROI:", ROI)
    print("corr(edge_energy, p(1-p)) =", corr)

    x = np.arange(len(mean_curve))
    plt.figure()
    plt.plot(x, mean_curve, label="measured edge energy (per-step avg)")
    plt.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, alpha=0.15)
    plt.plot(x, g_scaled, label="scaled p(t)(1-p(t)) proxy")
    plt.title("Edge-energy vs ghosting proxy p(1-p)")
    plt.xlabel("within-step frame index (0..N-1)")
    plt.ylabel("value")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
