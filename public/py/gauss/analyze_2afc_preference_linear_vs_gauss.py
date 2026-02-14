import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.stats import binomtest
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


ROOT_DIR = Path(r"D:\vectionProject\public\AAAGaussDatav0")  # 改成你的目录
PATTERN = "*2AFC*results*.csv"  # 覆盖更多类似文件
OUT_SUMMARY = "2afc_summary_by_participant.csv"
OUT_PLOT = "2afc_pGauss_by_participant.png"


def parse_participant_from_name(path: Path) -> str:
    m = re.search(r"ParticipantName_([^_]+)", path.name)
    return m.group(1) if m else path.stem


def chosen_condition(setting: str, choice: str) -> str:
    """
    setting: TopLinear / TopGauss
    choice: Upper / Lower
    return: 'linear' or 'gauss' or 'unknown'
    """
    s = str(setting).strip().lower()
    c = str(choice).strip().lower()

    if s == "toplinear":
        # upper=linear, lower=gauss
        if c == "upper":
            return "linear"
        if c == "lower":
            return "gauss"

    if s == "topgauss":
        # upper=gauss, lower=linear
        if c == "upper":
            return "gauss"
        if c == "lower":
            return "linear"

    return "unknown"


def binom_pvalue(k, n, p=0.5):
    if n <= 0:
        return np.nan
    if SCIPY_OK:
        return float(binomtest(k, n, p=p, alternative="two-sided").pvalue)
    # 简易兜底：无 scipy 时只给 NaN（你也可以自己实现 exact binom）
    return np.nan


def wilson_ci(k, n, z=1.96):
    """Wilson score interval for proportion k/n."""
    if n <= 0:
        return (np.nan, np.nan)
    phat = k / n
    denom = 1 + z**2 / n
    center = (phat + z**2 / (2*n)) / denom
    half = z * np.sqrt((phat*(1-phat) + z**2/(4*n)) / n) / denom
    return (center - half, center + half)


def main():
    files = sorted(ROOT_DIR.rglob(PATTERN))
    if not files:
        raise RuntimeError(f"No files matched: {PATTERN} under {ROOT_DIR}")

    rows = []

    for f in files:
        df = pd.read_csv(f)

        # 基本列检查
        need = {"Setting", "Choice"}
        if not need.issubset(df.columns):
            print(f"[skip] missing columns in {f.name}: {df.columns.tolist()}")
            continue

        # 还原选中条件
        df["chosen"] = [chosen_condition(a, b) for a, b in zip(df["Setting"], df["Choice"])]
        df = df[df["chosen"].isin(["linear", "gauss"])].copy()
        if len(df) == 0:
            print(f"[skip] no valid trials after mapping: {f.name}")
            continue

        participant = parse_participant_from_name(f)

        n = len(df)
        k_gauss = int((df["chosen"] == "gauss").sum())
        p_gauss = k_gauss / n

        # side bias（Upper 比例）
        upper_rate = float((df["Choice"].astype(str).str.lower() == "upper").mean())

        # randomization balance（TopLinear vs TopGauss）
        top_linear_n = int((df["Setting"].astype(str).str.lower() == "toplinear").sum())
        top_gauss_n = int((df["Setting"].astype(str).str.lower() == "topgauss").sum())

        pval = binom_pvalue(k_gauss, n, p=0.5)
        ci_lo, ci_hi = wilson_ci(k_gauss, n)

        rows.append({
            "participant": participant,
            "n_trials": n,
            "k_gauss": k_gauss,
            "p_gauss": p_gauss,
            "ci95_lo": ci_lo,
            "ci95_hi": ci_hi,
            "p_binom_vs_0.5": pval,
            "upper_rate": upper_rate,
            "n_TopLinear": top_linear_n,
            "n_TopGauss": top_gauss_n,
            "file": f.name,
        })

    if not rows:
        raise RuntimeError("No participant summaries produced.")

    out = pd.DataFrame(rows).sort_values("p_gauss", ascending=False)
    out_path = ROOT_DIR / OUT_SUMMARY
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"[OK] saved summary: {out_path}")

    # plot p_gauss
    fig = plt.figure(figsize=(max(8, 0.55 * len(out) + 4), 5))
    ax = fig.add_subplot(111)
    x = np.arange(len(out))
    ax.plot(x, out["p_gauss"].values, marker="o", linewidth=2)
    ax.axhline(0.5, linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(out["participant"].tolist(), rotation=45, ha="right")
    ax.set_ylabel("p(choose gauss)")
    ax.set_title("2AFC preference by participant")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = ROOT_DIR / OUT_PLOT
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    print(f"[OK] saved plot: {plot_path}")

    # group-level quick readout
    total_n = int(out["n_trials"].sum())
    total_k = int(out["k_gauss"].sum())
    group_p = total_k / total_n
    print(f"[GROUP] total k_gauss/n = {total_k}/{total_n} => p={group_p:.3f}")


if __name__ == "__main__":
    main()
