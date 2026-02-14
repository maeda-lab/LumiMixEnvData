import os
import numpy as np
import matplotlib.pyplot as plt

# ======================
# CONFIG
# ======================
OUT_PNG = None  # e.g. r"D:\vectionProject\public\camera3images\weight_peak0p9_hold0p15_0_5s.png"

FPS = 60
T_START = 0.0
T_END = 5.0

STEP_SEC = 1.0     # centers are 1s apart (A at 0, B at 1, ...)
PEAK = 0.9
RISE = 1.0         # seconds from 0 -> PEAK
HOLD = 0.1        # seconds holding at PEAK
FALL = 1.0         # seconds from PEAK -> 0

# Number of frames to plot in 0..5s (A..E = 5)
N_FRAMES = 5
LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def w_single(t: np.ndarray, center: float) -> np.ndarray:
    """
    One frame's weight as a trapezoid: rise -> hold -> fall.
    Support: [center-RISE, center+HOLD+FALL]
    Peak plateau: [center, center+HOLD]
    """
    w = np.zeros_like(t, dtype=np.float64)

    # rise: (center-RISE) .. center
    m_rise = (t >= (center - RISE)) & (t < center)
    if RISE > 0:
        w[m_rise] = PEAK * (t[m_rise] - (center - RISE)) / RISE

    # hold: center .. center+HOLD
    m_hold = (t >= center) & (t <= (center + HOLD))
    w[m_hold] = PEAK

    # fall: (center+HOLD) .. (center+HOLD+FALL)
    m_fall = (t > (center + HOLD)) & (t <= (center + HOLD + FALL))
    if FALL > 0:
        w[m_fall] = PEAK * (1.0 - (t[m_fall] - (center + HOLD)) / FALL)

    return w


def main():
    t = np.linspace(T_START, T_END, int((T_END - T_START) * FPS) + 1)

    centers = np.arange(N_FRAMES, dtype=np.float64) * STEP_SEC  # 0,1,2,3,4 for A..E

    W = np.zeros((N_FRAMES, len(t)), dtype=np.float64)
    for i, c in enumerate(centers):
        W[i] = w_single(t, c)

    plt.figure(figsize=(10, 4))
    for i in range(N_FRAMES):
        label = LABELS[i] if i < len(LABELS) else f"F{i}"
        plt.plot(t, W[i], label=label)

    plt.xlim(T_START, T_END)
    plt.ylim(0.0, 1.0)
    plt.xlabel("time t (s)")
    plt.ylabel("weight")
    plt.title(f"Weights (peak={PEAK}, rise={RISE}s, hold={HOLD}s, fall={FALL}s)")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=5, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, 1.18))

    if OUT_PNG is None:
        out_dir = os.path.dirname(os.path.abspath(__file__))
        out_name = (
            f"weight_peak{str(PEAK).replace('.','p')}"
            f"_rise{str(RISE).replace('.','p')}"
            f"_hold{str(HOLD).replace('.','p')}"
            f"_fall{str(FALL).replace('.','p')}"
            f"_{int(T_START)}_{int(T_END)}s.png"
        )
        out_path = os.path.join(out_dir, out_name)
    else:
        out_path = OUT_PNG

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[Saved] {out_path}")


if __name__ == "__main__":
    main()
