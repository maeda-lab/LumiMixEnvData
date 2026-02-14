import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    # ======================
    # CONFIG
    # ======================
    FPS = 60
    T_START = 0.0
    T_END = 3.0          # 0..3s (3 transitions)
    STEP_SEC = 1.0       # each transition lasts 1s

    d = 0.9 * np.pi      # phase shift (e.g., 0.9π)
    alpha = 1.0          # amplitude ratio (common case = 1)

    # ---- file naming ----
    SCRIPT_NAME = "plot_phase_linearized_keyframe_weights_show0.py"
    OUT_NAME = "weights_phase_linearized_keyframes_show0_d0p9pi_0_5s.png"

    # Optional title (comment out if your matplotlib lacks Chinese fonts)
    TITLE = "Phase Comp weights"

    # Frame labels: for 5 transitions we need 6 keyframes: A..F
    LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    n_frames = int(round(T_END / STEP_SEC)) + 1  # 6

    # Save next to this script (same folder)
    out_dir = Path(__file__).resolve().parent
    out_png = out_dir / OUT_NAME

    # ======================
    # TIME GRID (INCLUDE ENDPOINT t = 5.0)
    # ======================
    n = int(round((T_END - T_START) * FPS)) + 1
    t = (np.arange(n) / FPS) + T_START

    # Which transition segment (0..4). Clamp endpoint safely.
    seg = np.floor((t - T_START) / STEP_SEC).astype(int)
    seg = np.clip(seg, 0, n_frames - 2)  # 0..4

    # normalized time within each 1s transition
    u = ((t - T_START) % STEP_SEC) / STEP_SEC  # endpoint becomes 0; we'll override endpoint later

    # ======================
    # Phase-linearized weight within transition
    # k_des(u) = d * u
    # w(u) = alpha*sin(k) / (alpha*sin(k) + sin(d-k))
    # ======================
    k = d * u
    num = alpha * np.sin(k)
    den = alpha * np.sin(k) + np.sin(d - k)

    eps = 1e-12
    w_next = np.where(np.abs(den) < eps, np.where(u < 0.5, 0.0, 1.0), num / den)  # for frame seg+1
    w_next = np.clip(w_next, 0.0, 1.0)
    w_curr = 1.0 - w_next  # for frame seg

    # ======================
    # Build per-keyframe weights over full window
    # ======================
    W = np.zeros((n_frames, n), dtype=float)
    for i in range(n_frames):
        # this frame is "current" in segment i: weight = (1 - w)
        mask_curr = (seg == i)
        W[i, mask_curr] = w_curr[mask_curr]

        # this frame is "next" in segment i-1: weight = w
        mask_next = (seg == i - 1)
        W[i, mask_next] = w_next[mask_next]

    # Force endpoint t = T_END: last frame weight = 1, others = 0
    W[:, -1] = 0.0
    W[n_frames - 1, -1] = 1.0  # Frame F

    # ======================
    # PLOT (show zeros)
    # ======================
    plt.figure(figsize=(5, 4), dpi=180)
    for i in range(n_frames):
        plt.plot(t, W[i], linewidth=2.2, label=f"{LABELS[i]}")

    plt.ylim(-0.02, 1.02)
    plt.xlim(T_START, T_END)
    plt.xlabel("time t (s)",fontsize=30)
    plt.ylabel("weight",fontsize=30)
    plt.title(TITLE, fontsize=30,pad=10)
    plt.grid(True, alpha=0.35)
    plt.legend(
        ncol=2,
        fontsize=9,
        frameon=True,
        loc="upper right",
    )
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    plt.close()

    print("Script name suggestion:", SCRIPT_NAME)
    print("Saved:", out_png)

if __name__ == "__main__":
    main()
