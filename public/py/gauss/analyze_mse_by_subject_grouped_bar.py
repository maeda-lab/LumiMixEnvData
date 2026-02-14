import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
ROOT_DIR = Path(r"D:\vectionProject\public\AAAGaussDatav0")

# Unity: public float omega = 2 * Mathf.PI;
OMEGA = 2.0 * np.pi
V0 = 1.0

ANALYSIS_FPS = 600
DEFAULT_DURATION_SEC = 10.0

OUT_PNG_NAME = "mse_by_subject_grouped_bar_paper.png"


# =========================
# Helpers
# =========================
def script_dir() -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()


def is_test_file(path: Path) -> bool:
    return "test" in path.stem.lower()


def parse_filename(path: Path):
    """
    condition: linear/gauss
    participant: ParticipantName_XXX
    trial: TrialNumber_N
    """
    name = path.name
    lower = name.lower()

    if "linear" in lower:
        condition = "linear"
    elif "gauss" in lower or "gaussian" in lower:
        condition = "gauss"
    else:
        condition = "unknown"

    m_p = re.search(r"ParticipantName_([^_\.]+)", name)
    participant = m_p.group(1) if m_p else "UNKNOWN"

    m_t = re.search(r"TrialNumber_(\d+)", name)
    trial = int(m_t.group(1)) if m_t else None

    return condition, participant, trial


def infer_duration_sec(df: pd.DataFrame) -> float:
    if "TimeMs" in df.columns:
        t0 = float(df["TimeMs"].min())
        t1 = float(df["TimeMs"].max())
        dur = max(0.0, (t1 - t0) / 1000.0)
        if dur >= 0.2:
            return dur
    return DEFAULT_DURATION_SEC


def extract_params(df: pd.DataFrame):
    """
    StepNumber=1..4 -> A1, phi1, A2, phi2
    each = last row's Amplitude within that Step
    """
    if "StepNumber" not in df.columns or "Amplitude" not in df.columns:
        raise ValueError("CSV must contain StepNumber and Amplitude columns")

    if "TimeMs" in df.columns:
        df = df.sort_values("TimeMs").reset_index(drop=True)

    step_to_key = {1: "A1", 2: "phi1", 3: "A2", 4: "phi2"}
    out = {"A1": np.nan, "phi1": np.nan, "A2": np.nan, "phi2": np.nan}

    for step, key in step_to_key.items():
        sub = df[df["StepNumber"] == float(step)]
        if len(sub) == 0:
            continue
        out[key] = float(sub.iloc[-1]["Amplitude"])

    return out


def build_v(t: np.ndarray, A1: float, phi1: float, A2: float, phi2: float) -> np.ndarray:
    # Unity step>=4:
    # v = V0 + A1*sin(omega*t + phi1 + pi) + A2*sin(2*omega*t + phi2 + pi)
    return (V0
            + A1 * np.sin(OMEGA * t + phi1 + np.pi)
            + A2 * np.sin(2.0 * OMEGA * t + phi2 + np.pi))


def mse(v: np.ndarray) -> float:
    return float(np.mean((v - V0) ** 2))


def process_one_csv(path: Path):
    df = pd.read_csv(path)
    condition, participant, trial = parse_filename(path)
    if condition not in ("linear", "gauss"):
        return None

    params = extract_params(df)
    A1, phi1, A2, phi2 = params["A1"], params["phi1"], params["A2"], params["phi2"]
    if any(np.isnan([A1, phi1, A2, phi2])):
        return None

    dur = infer_duration_sec(df)
    t = np.arange(0.0, dur, 1.0 / ANALYSIS_FPS)
    v = build_v(t, A1, phi1, A2, phi2)
    return {
        "participant": participant,
        "condition": condition,
        "trial": trial,
        "mse": mse(v),
    }


def main():
    # --------- load all trials ----------
    rows = []
    for p in ROOT_DIR.rglob("*.csv"):
        if is_test_file(p):
            continue
        item = process_one_csv(p)
        if item is not None:
            rows.append(item)

    if not rows:
        raise RuntimeError("No valid trials found. Check filenames and columns.")

    df = pd.DataFrame(rows)

    # --------- subject-level mean (average over trials) ----------
    df_subj = (
        df.groupby(["participant", "condition"], as_index=False)
          .agg(mse_mean=("mse", "mean"),
               mse_sd=("mse", "std"),
               n_trials=("mse", "count"))
    )
    # replace NaN std (single trial) with 0.0
    df_subj["mse_sd"] = df_subj["mse_sd"].fillna(0.0)

    # pivot to wide: index=participant, columns=condition
    mean_wide = df_subj.pivot(index="participant", columns="condition", values="mse_mean")
    sd_wide   = df_subj.pivot(index="participant", columns="condition", values="mse_sd")

    # keep only participants having both conditions
    if not {"linear", "gauss"}.issubset(set(mean_wide.columns)):
        raise RuntimeError(f"Need both linear & gauss. Current columns: {list(mean_wide.columns)}")

    # drop participants missing either condition
    mean_wide = mean_wide.dropna(subset=["linear", "gauss"], how="any")
    sd_wide   = sd_wide.reindex(mean_wide.index)  # align index
    if len(mean_wide) == 0:
        raise RuntimeError("No participant has both linear and gauss conditions.")

    # order participants by linear mse (ascending)
    mean_wide = mean_wide.sort_values("linear", ascending=True)
    sd_wide   = sd_wide.loc[mean_wide.index]

    participants = mean_wide.index.tolist()
    n = len(participants)

    # reduce horizontal spacing between participants by scaling the x positions
    spacing = 0.75  # try values between ~0.6..1.0 (smaller -> 更紧)
    x = np.arange(n) * spacing

    y_linear = mean_wide["linear"].values
    y_gauss  = mean_wide["gauss"].values
    e_linear = sd_wide["linear"].values
    e_gauss  = sd_wide["gauss"].values

    # anonymize labels to S01, S02, ...
    subj_labels = [f"S{i:02d}" for i in range(1, n + 1)]

    # --------- plot: grouped bar ----------
    # make figure width scale with number of subjects and spacing
    fig_w = max(8, 0.7 * n * spacing + 4)
    fig, ax = plt.subplots(figsize=(10, 4))

    bar_w = 0.22  # narrower bars; 调整为 0.18..0.28 之间试试不同效果
    # x positions for each group's bars
    x_lin = x - bar_w / 2
    x_gau = x + bar_w / 2

    # draw bars (store containers) — bars drawn below error bars (lower zorder)
    bars_lin = ax.bar(x_lin, y_linear, width=bar_w, label="linear", zorder=2)
    bars_gauss = ax.bar(x_gau, y_gauss, width=bar_w, label="gauss", zorder=2)

    # compute ylim so annotations and caps aren't cut off
    ymax = float(np.nanmax(np.concatenate([y_linear + e_linear, y_gauss + e_gauss])))
    ax.set_ylim(0.0, ymax * 1.18)

    # draw errorbars on top of bars (but below text)
    ax.errorbar(x_lin, y_linear, yerr=e_linear,
                fmt="none", ecolor="black", elinewidth=1.6, capsize=6, capthick=1.6,
                zorder=5, clip_on=False)
    ax.errorbar(x_gau, y_gauss, yerr=e_gauss,
                fmt="none", ecolor="black", elinewidth=1.6, capsize=6, capthick=1.6,
                zorder=5, clip_on=False)

    # annotate only mean above the error cap (uniformly placed)
    pad = ymax * 0.02
    for xi, h, err in zip(x_lin, y_linear, e_linear):
        y_text = h + (0.0 if np.isnan(err) else float(err)) + pad
        ax.text(xi, y_text, f"{h:.3f}",
                ha="center", va="bottom",
                fontsize=9, zorder=6, clip_on=False,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.4))

    for xi, h, err in zip(x_gau, y_gauss, e_gauss):
        y_text = h + (0.0 if np.isnan(err) else float(err)) + pad
        ax.text(xi, y_text, f"{h:.3f}",
                ha="center", va="bottom",
                fontsize=9, zorder=6, clip_on=False,
                bbox=dict(facecolor="white", edgecolor="none", pad=0.4))

    ax.set_xticks(x)
    ax.set_xticklabels(subj_labels, rotation=0, ha="center")  # labels horizontal
    # tighten x-limits to the used range (give a little margin)
    x_min = x[0] - spacing * 0.6
    x_max = x[-1] + spacing * 0.6
    ax.set_xlim(x_min, x_max)
    LITTLESIZE = 12
    ax.set_xlabel("Participants", fontsize=LITTLESIZE, labelpad=8)
    ax.set_ylabel("RMSE", fontsize=LITTLESIZE)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    out_path = script_dir() / OUT_PNG_NAME
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"[OK] saved: {out_path}")


if __name__ == "__main__":
    main()
