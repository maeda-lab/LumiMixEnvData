# bar_rmse_from_params_A_to_E.py
import os
import re
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# 1) Paste your parameter log here (exactly what you sent)
# =========================================================
RAW_TEXT = r"""
处理参与者 YAMA:  -A
  试验 3: V0=0.970, A1=0.332, φ1=1.803,A2=-0.547, φ2=2.325
  试验 2: V0=0.956, A1=0.455, φ1=1.778,A2=-0.574, φ2=2.061
  试验 1: V0=1.050, A1=0.833, φ1=1.967,A2=-0.463, φ2=0.000
平均参数: V0=0.992, A1=0.540, φ1=1.849, A2=-0.528, φ2=1.462

处理参与者 OMU:-B
  试验 3: V0=1.106, A1=0.575, φ1=2.488,A2=-1.000, φ2=2.953
  试验 1: V0=1.052, A1=0.176, φ1=2.846,A2=-0.214, φ2=2.168
  试验 2: V0=1.234, A1=0.815, φ1=2.249,A2=0.545, φ2=5.454
平均参数: V0=1.131, A1=0.522, φ1=2.528, A2=-0.223, φ2=3.525

处理参与者 ONO: -C
  试验 3: V0=1.050, A1=0.485, φ1=3.475,A2=0.407, φ2=5.743
  试验 2: V0=1.030, A1=0.431, φ1=3.563,A2=0.368, φ2=5.422
  试验 1: V0=1.120, A1=0.980, φ1=3.952,A2=0.608, φ2=4.203
平均参数: V0=1.067, A1=0.632, φ1=3.663, A2=0.461, φ2=5.123

处理参与者 HOU:- D
  试验 1: V0=1.034, A1=0.662, φ1=1.910,A2=1.061, φ2=6.283
  试验 2: V0=0.906, A1=0.836, φ1=2.205,A2=0.833, φ2=6.283
  试验 3: V0=0.912, A1=-0.673, φ1=4.976,A2=0.866, φ2=5.378
平均参数: V0=0.951, A1=0.275, φ1=3.031, A2=0.920, φ2=5.982

处理参与者 LL:- E
  试验 1: V0=0.922, A1=-0.682, φ1=1.451,A2=-0.721, φ2=3.236
  试验 2: V0=0.996, A1=0.848, φ1=3.475,A2=-0.460, φ2=1.665
  试验 3: V0=1.164, A1=-1.000, φ1=0.622,A2=0.305, φ2=6.283
平均参数: V0=1.027, A1=-0.278, φ1=1.849, A2=-0.292, φ2=3.728
"""

# =========================================================
# 2) Parse trials
# =========================================================
# Matches: 试验 k: V0=..., A1=..., φ1=...,A2=..., φ2=...
trial_re = re.compile(
    # allow separators like "-", " -", ":-", ":- ", or ":" and optional spaces before label
    r"处理参与者\s*(?P<name>[A-Za-z]+).*?[\-:：]\s*(?P<label>[A-E])(?P<body>.*?)(?=(处理参与者|$))",
    re.S
)
param_re = re.compile(
    r"试验\s*(?P<trial>\d+)\s*:\s*V0=(?P<V0>[-\d.]+)\s*,\s*A1=(?P<A1>[-\d.]+)\s*,\s*φ1=(?P<phi1>[-\d.]+)\s*,\s*A2=(?P<A2>[-\d.]+)\s*,\s*φ2=(?P<phi2>[-\d.]+)",
    re.S
)

def parse_params(raw: str):
    out = {}  # label -> list of (trial_id, V0,A1,phi1,A2,phi2)
    for m in trial_re.finditer(raw):
        label = m.group("label").strip()
        body = m.group("body")
        trials = []
        for pm in param_re.finditer(body):
            trials.append((
                int(pm.group("trial")),
                float(pm.group("V0")),
                float(pm.group("A1")),
                float(pm.group("phi1")),
                float(pm.group("A2")),
                float(pm.group("phi2")),
            ))
        # sort by trial id (optional)
        trials.sort(key=lambda x: x[0])
        out[label] = trials
    return out

# =========================================================
# 3) Compute RMSE per trial using eq (3.6)(3.7)
# =========================================================
FPS = 60
N = 60
t = np.arange(1, N + 1, dtype=float) / FPS  # t_n = n/60
omega = 2.0 * np.pi  # 1 Hz

def rmse_constant_model_from_params(V0, A1, phi1, A2, phi2):
    """
    v(t) model ASSUMED:
      v(t) = V0 + A1*sin(omega*t + phi1) + A2*sin(2*omega*t + phi2)

    If your Unity model differs, edit ONLY the v= line below.
    """
    # ---- EDIT HERE if needed ----
    v = V0 + A1 * np.sin(omega * t + phi1) + A2 * np.sin(2.0 * omega * t + phi2)
    # -----------------------------
    v_bar = v.mean()
    rmse = np.sqrt(np.mean((v - v_bar) ** 2))
    return float(rmse)

# =========================================================
# 4) Aggregate mean±SD by participant and plot
# =========================================================
data = parse_params(RAW_TEXT)

participants = ["A", "B", "C", "D", "E"]
rmse_trials = {p: [] for p in participants}

for p in participants:
    trials = data.get(p, [])
    if len(trials) == 0:
        continue
    for (trial_id, V0, A1, phi1, A2, phi2) in trials:
        rmse = rmse_constant_model_from_params(V0, A1, phi1, A2, phi2)
        rmse_trials[p].append(rmse)

rmse_mean = np.array([np.mean(rmse_trials[p]) if rmse_trials[p] else np.nan for p in participants], dtype=float)
rmse_sd   = np.array([np.std (rmse_trials[p], ddof=0) if rmse_trials[p] else np.nan for p in participants], dtype=float)

# Print a quick table to console
print("Participant | RMSE trials | mean | sd")
for p in participants:
    vals = rmse_trials[p]
    print(f"{p:>10} | {vals} | {np.mean(vals) if vals else np.nan:.4f} | {np.std(vals, ddof=0) if vals else np.nan:.4f}")

# Plot bar chart with error bars
def plot_improved_bar(values, errors, labels, out_path=None):
    import matplotlib.pyplot as plt
    import numpy as np

    vals = np.asarray(values, dtype=float)
    errs = np.asarray(errors, dtype=float)
    x = np.arange(len(vals))

    # figure size / resolution
    fig, ax = plt.subplots(figsize=(8, 3), dpi=180)

    # color mapping based on value magnitude (small->light, large->dark)
    vmin = float(np.nanmin(vals)) if not np.all(np.isnan(vals)) else 0.0
    vmax = float(np.nanmax(vals)) if not np.all(np.isnan(vals)) else 1.0
    if np.isclose(vmax, vmin):
        norm_vals = np.full_like(vals, 0.5)
    else:
        norm_vals = (vals - vmin) / (vmax - vmin)
    cmap = plt.cm.Blues
    colors = cmap(0.35 + 0.6 * norm_vals)

    bars = ax.bar(
        x, vals,
        color=colors,
        edgecolor="black",
        linewidth=0.8,
        width=0.7,
        zorder=2
    )

    # errorbars with larger caps and thicker line — draw on top and don't clip to axes
    ax.errorbar(
        x, vals, yerr=errs,
        fmt="none",
        ecolor="black",
        elinewidth=1.6,
        capsize=8,
        capthick=1.6,
        zorder=4,
        clip_on=False
    )
    # y limit with margin
    ymax = float(np.nanmax(vals + np.nan_to_num(errs)))
    # annotate bar tops placed above the error cap so they don't cover the cap
    for xi, v, e in zip(x, vals, errs):
        e_val = 0.0 if np.isnan(e) else float(e)
        y_text = v + e_val + ymax * 0.03
        ax.text(
            xi, y_text,
            f"{v:.3f}",
            ha="center", va="bottom",
            fontsize=11, fontweight="bold",
            zorder=5,
            bbox=dict(facecolor="white", edgecolor="none", pad=0.6)
        )

    # axis labels / title sizes
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=13, fontweight="medium")
    ax.set_ylabel("RMSE", fontsize=25)
    ax.set_xlabel("Participants", fontsize=25)
    ax.tick_params(axis="y", labelsize=12)


    ax.set_ylim(0, ymax * 1.20)

    # subtle grid for readability
    ax.yaxis.grid(True, linestyle="--", alpha=0.35, zorder=1)
    ax.set_axisbelow(True)
    plt.tight_layout()

    if out_path:
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
    return fig, ax

if __name__ == "__main__":
    display_labels = [f"S{i:02d}" for i in range(1, len(participants) + 1)]
    fig, ax = plot_improved_bar(rmse_mean, rmse_sd, display_labels)
    out_png = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "bar_rmse_by_participant_from_params.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_png)
