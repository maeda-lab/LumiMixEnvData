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

OUT_PNG_NAME = "mse_by_subject_lines_conditions.png"


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
               n_trials=("mse", "count"))
    )

    # pivot to wide: index=participant, columns=condition
    wide = df_subj.pivot(index="participant", columns="condition", values="mse_mean")

    # keep only participants having both conditions
    if not {"linear", "gauss"}.issubset(set(wide.columns)):
        raise RuntimeError(f"Need both linear & gauss. Current columns: {list(wide.columns)}")

    wide = wide.dropna(subset=["linear", "gauss"], how="any")
    if len(wide) == 0:
        raise RuntimeError("No participant has both linear and gauss conditions.")

    # order participants (optional): by linear mse
    wide = wide.sort_values("linear", ascending=True)

    participants = wide.index.tolist()
    x = np.arange(len(participants))

    y_linear = wide["linear"].values
    y_gauss = wide["gauss"].values

    # anonymize / shorten participant labels to S01, S02, ...
    subj_labels = [f"S{i:02d}" for i in range(1, len(participants) + 1)]

    # --------- plot: x=participant, lines=conditions ----------
    fig = plt.figure(figsize=(max(8, 0.7 * len(participants) + 4), 6))
    ax = fig.add_subplot(111)

    ax.plot(x, y_linear, marker="o", linewidth=2, label="linear")
    ax.plot(x, y_gauss, marker="o", linewidth=2, label="gauss")

    ax.set_xticks(x)
    ax.set_xticklabels(subj_labels, rotation=45, ha="right")
    ax.set_ylabel("MSE (mean over trials)")
    ax.set_title("MSE by subject (lines = conditions)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax.text(0.01, 0.98, f"n_subjects={len(participants)}",
            transform=ax.transAxes, va="top")

    out_path = script_dir() / OUT_PNG_NAME
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"[OK] saved: {out_path}")


if __name__ == "__main__":
    main()
