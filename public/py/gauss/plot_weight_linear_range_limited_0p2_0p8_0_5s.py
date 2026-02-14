import os
import numpy as np
import matplotlib.pyplot as plt

# ======================
# CONFIG
# ======================
OUT_PNG = None  # e.g. r"D:\vectionProject\public\camera3images\weight_lin_range_0p2_0p8_0_5s.png"

FPS = 60
T_START = 0.0
T_END = 5.0

STEP_SEC = 1.0
W_MIN = 0.2
W_MAX = 0.8

LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

EPS = 1e-12


def wB_range(u: np.ndarray) -> np.ndarray:
    """Range-limited B weight: u in [0,1] -> [W_MIN, W_MAX]."""
    return W_MIN + (W_MAX - W_MIN) * u


def main():
    # 0..5s with 1s per segment => 5 segments => 6 frames (A..F)
    n_segments = int((T_END - T_START) / STEP_SEC)  # 5
    n_frames = n_segments + 1  # 6

    dt = 1.0 / FPS
    t = np.arange(T_START, T_END + 0.5 * dt, dt)  # include T_END exactly

    # Pre-fill with NaN so non-participating spans are not plotted (no vertical lines)
    W_plot = np.full((n_frames, len(t)), np.nan, dtype=np.float64)

    # Build per-segment weights, then write into the two participating frames only
    for k in range(n_segments):
        seg_start = k * STEP_SEC
        seg_end = (k + 1) * STEP_SEC

        # inclusive on both ends so boundary points (t=1,2,...) are kept
        m = (t >= seg_start - EPS) & (t <= seg_end + EPS)

        u = (t[m] - seg_start) / STEP_SEC
        u = np.clip(u, 0.0, 1.0)

        wB = wB_range(u)
        wA = 1.0 - wB

        # frame k is A, frame k+1 is B in this segment
        W_plot[k, m] = wA
        W_plot[k + 1, m] = wB

    # Plot
    plt.figure(figsize=(10, 4))
    for i in range(n_frames):
        label = LABELS[i] if i < len(LABELS) else f"F{i}"
        plt.plot(t, W_plot[i], label=label)

    plt.xlim(T_START, T_END)
    plt.ylim(0.0, 1.0)
    plt.xlabel("time t (s)")
    plt.ylabel("weight")
    plt.title(f"Range-limited linear cross-dissolve weights ({W_MIN:.1f}–{W_MAX:.1f})")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=6, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, 1.18))

    if OUT_PNG is None:
        out_dir = os.path.dirname(os.path.abspath(__file__))
        out_name = (
            f"weight_lin_range_{str(W_MIN).replace('.','p')}_{str(W_MAX).replace('.','p')}_"
            f"{int(T_START)}_{int(T_END)}s.png"
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
