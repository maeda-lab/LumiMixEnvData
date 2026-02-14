import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ======================
# Participant D (HOU) parameters
# Model: v(t) = V0 + A1*sin(ωt + φ1) + A2*sin(2ωt + φ2)
# ======================
OMEGA = 2.0 * np.pi   # 1 Hz fundamental
T_END = 1.0
N_SAMPLES = 1000

trials = [
    # (V0,   A1,     phi1,   A2,     phi2)
    # (1.034,  0.662,  1.910,  1.061,  6.283),  # Trial 1
    (0.906,  0.836,  2.205,  0.833,  6.283),  # Trial 2
    # (0.912, -0.673,  4.976,  0.866,  5.378),  # Trial 3
]

# Horizontal reference line: participant's mean V0 from your printout
V0_MEAN = 0.951

# Output file name (saved next to this script)
OUT_PNG_NAME = "subjective_speed_profile_D_with_v0.png"

# ======================
# Helpers
# ======================
def script_dir() -> Path:
    """Folder containing this .py file. Falls back to current working directory if run interactively."""
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()

def v_model(t, V0, A1, phi1, A2, phi2):
    return V0 + A1 * np.sin(OMEGA * t + phi1) + A2 * np.sin(2.0 * OMEGA * t + phi2)

# ======================
# Compute curves
# ======================
t = np.linspace(0.0, T_END, N_SAMPLES, endpoint=True)

V = np.stack([v_model(t, *p) for p in trials], axis=0)  # (n_trials, n_time)
v_mean = V.mean(axis=0)
v_std  = V.std(axis=0, ddof=1)  # sample SD (n=3)

# ======================
# Plot (single panel: D only)
# ======================
fig, ax = plt.subplots(figsize=(5.6, 3.4), dpi=150)

ax.plot(t, v_mean, linewidth=2.0)
# ax.fill_between(t, v_mean - v_std, v_mean + v_std, alpha=0.25, label="± 1 SD")

# horizontal dashed line: 1s-period mean of the plotted profile
mean_speed = float(np.mean(v_mean))
ax.axhline(mean_speed, linewidth=1.5, linestyle="--", color="k")

# annotate numeric value at right edge
# ax.text(
#     0.99, mean_speed,
#     f"{mean_speed:.3f}",
#     ha="right", va="center",
#     transform=ax.get_yaxis_transform(),
#     fontsize=10,
#     bbox=dict(facecolor="white", edgecolor="none", pad=1)
# )

# Fonts: make labels/title bigger
LABEL_FS = 30
TITLE_FS = 20
TICK_FS = 12
LEGEND_FS = 12

# ax.set_title("推定された主観等価速度プロファイル\n𝑣(𝑡) v(t)（例：Participant D）", fontsize=TITLE_FS)
ax.set_xlabel("time t [s]", fontsize=LABEL_FS)
ax.set_ylabel("v(t)", fontsize=LABEL_FS)
ax.set_xlim(0.0, T_END)
ax.set_ylim(-1.0, 3.0)  # adjust if desired

# only show legend when there are labeled artists
handles, labels = ax.get_legend_handles_labels()
if labels:
    ax.legend(loc="upper right", frameon=True, fontsize=LEGEND_FS)
ax.grid(False)
ax.tick_params(axis="both", which="major", labelsize=TICK_FS)

plt.tight_layout()

# Save to the same folder as this script
out_path = script_dir() / OUT_PNG_NAME
fig.savefig(out_path, bbox_inches="tight")
print(f"[Saved] {out_path}")

plt.show()
