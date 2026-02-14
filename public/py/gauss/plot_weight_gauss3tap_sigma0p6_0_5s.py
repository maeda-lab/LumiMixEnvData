import os
import numpy as np
import matplotlib.pyplot as plt

# ======================
# CONFIG
# ======================
OUT_PNG  = None
FPS      = 60
DT       = 1.0 / FPS

STEP_SEC = 1.0
SIGMA_T  = 0.6
N_FRAMES = 10

SHIFT_SEC  = 1.0
WINDOW_SEC = 3   # show 0..3


def gauss3tap_weights_no_tie(t_eval: np.ndarray, frame_times: np.ndarray) -> np.ndarray:
    """
    Strict 3-tap:
      - center frame selected stably by round-half-up on u=t/STEP
      - neigh = {c-1,c,c+1}
    Key: t_eval uses half-frame offset so it never lands on midpoint ties.
    """
    n_frames = len(frame_times)
    W = np.zeros((n_frames, len(t_eval)), dtype=np.float64)

    for k, t in enumerate(t_eval):
        # stable center (round-half-up)
        u = t / STEP_SEC
        c = int(np.floor(u + 0.5))
        c = max(0, min(n_frames - 1, c))

        neigh = [c - 1, c, c + 1]
        neigh = [i for i in neigh if 0 <= i < n_frames]

        tw = np.array(
            [np.exp(-((t - frame_times[i]) ** 2) / (2.0 * SIGMA_T ** 2)) for i in neigh],
            dtype=np.float64
        )
        s = float(np.sum(tw))
        if s > 0:
            tw /= s

        for w, i in zip(tw, neigh):
            W[i, k] = w

    return W


def main():
    # segment to show (original times)
    t_min, t_max = 0.5, 3.5

    # ensure t_show/t_eval cover up to t_max (so mask includes full segment)
    n = int(max(WINDOW_SEC, t_max) * FPS) + 1
    t_show = np.arange(n) * DT  # 0 .. at least t_max
    # evaluation times (half-frame centers) must have same length
    t_eval = (np.arange(n) + 0.5) * DT + SHIFT_SEC

    frame_times = np.arange(N_FRAMES, dtype=np.float64) * STEP_SEC
    W = gauss3tap_weights_no_tie(t_eval, frame_times)

    # now select the segment and left-shift it
    mask = (t_show >= t_min) & (t_show <= t_max)
    t_seg = t_show[mask]
    t_shift = t_seg - t_min  # left shift

    fig, ax = plt.subplots(figsize=(5, 4))
    for i in range(W.shape[0]):
        w_seg = W[i, mask]
        if np.any(w_seg > 0):
            ax.plot(t_shift, w_seg, label=f"i={i}", solid_capstyle="butt", clip_on=True)
            # ax.plot(t_shift, w_seg, label=f"i={i}", solid_capstyle="butt", clip_on=True)

    # keep axis spanning 0..WINDOW_SEC (data will occupy 0..(t_max-t_min))
    ax.set_xlim(0.0, WINDOW_SEC)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("time (s)",fontsize=30)
    ax.set_ylabel("weight",fontsize=30)
    ax.set_title(f"Gaussian weights", fontsize=30, pad=10)
    # ax.set_title(f"Gaussian time weights (sigma={SIGMA_T})", fontsize=30)
    ax.grid(True, alpha=0.3)
    handles, _ = ax.get_legend_handles_labels()
    custom_labels = ['A','B','C','D','E']
    ax.legend(handles[:len(custom_labels)], custom_labels, ncol=2, fontsize=9, loc="upper right")

    if OUT_PNG is None:
        out_dir = os.path.dirname(os.path.abspath(__file__))
        out_name = (
            f"weight_gauss3tap_sigma{str(SIGMA_T).replace('.','p')}"
            f"_shift{str(SHIFT_SEC).replace('.','p')}_0_{int(WINDOW_SEC)}s.png"
        )
        out_path = os.path.join(out_dir, out_name)
    else:
        out_path = OUT_PNG

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[Saved] {out_path}")


if __name__ == "__main__":
    main()
