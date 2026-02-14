from __future__ import annotations

import math
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# CONFIG
# =========================
ROOT_DIR = Path(r"D:\vectionProject\public\AAAGaussDatav0")
PATTERN = "*_2AFC_results.csv"   # e.g., ParticipantName_OMU_2AFC_results.csv

SAVE_PLOT = True
OUT_PNG_NAME = "2afc_gauss_preference_by_participant.png"


# =========================
# Stats helpers (no scipy)
# =========================
def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    phat = k / n
    denom = 1 + (z**2) / n
    center = (phat + (z**2) / (2 * n)) / denom
    half = (z / denom) * math.sqrt((phat * (1 - phat) / n) + (z**2) / (4 * n**2))
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return lo, hi


# =========================
# Core logic
# =========================
def script_dir() -> Path:
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()


def parse_participant_name(file_path: Path) -> str:
    # Expected: ParticipantName_OMU_2AFC_results.csv
    stem = file_path.stem
    parts = stem.split("_")
    if len(parts) >= 2:
        return parts[1]
    return stem


def infer_selected_condition(setting: str, choice: str) -> str:
    """
    Map (Setting, Choice) -> selected condition {Gauss, Linear, Unknown}

    Setting examples:
      - TopLinear_BottomGauss
      - TopGauss_BottomLinear

    Choice examples:
      - Upper
      - Lower
    """
    s = (setting or "").strip()
    c = (choice or "").strip().lower()

    if s == "TopLinear_BottomGauss":
        if c == "upper":
            return "Linear"
        if c == "lower":
            return "Gauss"
        return "Unknown"

    if s == "TopGauss_BottomLinear":
        if c == "upper":
            return "Gauss"
        if c == "lower":
            return "Linear"
        return "Unknown"

    return "Unknown"


def load_one_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"Trial", "Setting", "Choice"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[{path.name}] Missing columns: {sorted(missing)}")

    df = df.copy()
    df["Participant"] = parse_participant_name(path)
    df["Selected"] = [
        infer_selected_condition(s, c)
        for s, c in zip(df["Setting"].astype(str), df["Choice"].astype(str))
    ]
    return df


def summarize(df_all: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for p, g in df_all.groupby("Participant", sort=True):
        n = int(g.shape[0])
        k_gauss = int((g["Selected"] == "Gauss").sum())
        k_linear = int((g["Selected"] == "Linear").sum())
        k_unk = int((g["Selected"] == "Unknown").sum())
        lo, hi = wilson_ci(k_gauss, n)

        rows.append({
            "Participant": p,
            "N_trials": n,
            "Gauss_count": k_gauss,
            "Linear_count": k_linear,
            "Unknown_count": k_unk,
            "Gauss_rate": (k_gauss / n) if n else float("nan"),
            "Gauss_rate_CI95_low": lo,
            "Gauss_rate_CI95_high": hi,
        })

    out = pd.DataFrame(rows)

    # Overall row
    n_all = int(df_all.shape[0])
    k_all = int((df_all["Selected"] == "Gauss").sum())
    lo, hi = wilson_ci(k_all, n_all)
    overall = pd.DataFrame([{
        "Participant": "ALL",
        "N_trials": n_all,
        "Gauss_count": k_all,
        "Linear_count": int((df_all["Selected"] == "Linear").sum()),
        "Unknown_count": int((df_all["Selected"] == "Unknown").sum()),
        "Gauss_rate": (k_all / n_all) if n_all else float("nan"),
        "Gauss_rate_CI95_low": lo,
        "Gauss_rate_CI95_high": hi,
    }])

    return pd.concat([out, overall], ignore_index=True)


def plot_rates(summary: pd.DataFrame, out_png: Path):
    df = summary[summary["Participant"] != "ALL"].copy()
    df = df.sort_values("Participant")

    x = df["Participant"].tolist()
    y = df["Gauss_rate"].tolist()

    # Wilson CI as error bars
    yerr_low = [rate - lo for rate, lo in zip(df["Gauss_rate"], df["Gauss_rate_CI95_low"])]
    yerr_high = [hi - rate for rate, hi in zip(df["Gauss_rate"], df["Gauss_rate_CI95_high"])]
    yerr = [yerr_low, yerr_high]

    plt.figure(figsize=(8, 4))

    # 柱子边框也变细（如果你不想要边框，可删 edgecolor/linewidth）
    plt.bar(x, y, width=0.5, edgecolor="black", linewidth=0.6)


    # 误差线变细：elinewidth
    # cap 也变细：capthick
    plt.errorbar(
        x, y, yerr=yerr,
        fmt="none",
        capsize=4,
        elinewidth=0.8,
        capthick=0.8
    )

    # 0.5 虚线也变细：linewidth
    plt.axhline(0.5, linestyle="--", linewidth=0.8)

    plt.ylim(0.0, 1.0)
    plt.ylabel("P(choose Gauss)")
    plt.title("2AFC preference for Gauss (per participant)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()



def main():
    paths = sorted(ROOT_DIR.glob(PATTERN))
    if not paths:
        raise FileNotFoundError(f"No files matched: {ROOT_DIR / PATTERN}")

    df_all = pd.concat([load_one_file(p) for p in paths], ignore_index=True)
    summary = summarize(df_all)

    print("\n===== 2AFC SUMMARY (Gauss vs Linear) =====")
    with pd.option_context("display.max_rows", 200, "display.max_columns", 200):
        print(summary)

    if SAVE_PLOT:
        out_dir = script_dir()  # 图保存到代码同一个文件夹
        out_png = out_dir / OUT_PNG_NAME
        plot_rates(summary, out_png)
        print(f"\nSaved plot PNG: {out_png}")


if __name__ == "__main__":
    main()
