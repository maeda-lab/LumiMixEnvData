from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import textwrap

# --- utilities: extract params from file (StepNumber-based or direct columns) ---
def get_params_from_file(p: Path):
    params = {'V0': np.nan, 'A1': np.nan, 'φ1': np.nan, 'A2': np.nan, 'φ2': np.nan}
    try:
        if not p.exists():
            return params
        df = pd.read_csv(p)
        df.columns = df.columns.str.strip()
        if "StepNumber" in df.columns:
            try:
                v0_series = df[df["StepNumber"] == 0]["Velocity"]
                params['V0'] = float(v0_series.iloc[-1]) if not v0_series.empty else 0.0
            except Exception:
                params['V0'] = np.nan
            try:
                a1s = df[df["StepNumber"] == 1]["Amplitude"]
                params['A1'] = float(a1s.iloc[-1]) if not a1s.empty else 0.0
            except Exception:
                params['A1'] = np.nan
            try:
                phi1s = df[df["StepNumber"] == 2]["Amplitude"]
                params['φ1'] = float(phi1s.iloc[-1]) if not phi1s.empty else 0.0
            except Exception:
                params['φ1'] = np.nan
            try:
                a2s = df[df["StepNumber"] == 3]["Amplitude"]
                params['A2'] = float(a2s.iloc[-1]) if not a2s.empty else 0.0
            except Exception:
                params['A2'] = np.nan
            try:
                phi2s = df[df["StepNumber"] == 4]["Amplitude"]
                params['φ2'] = float(phi2s.iloc[-1]) if not phi2s.empty else 0.0
            except Exception:
                params['φ2'] = np.nan
        else:
            # fallback direct columns
            if "Velocity" in df.columns:
                try:
                    params['V0'] = float(df["Velocity"].iloc[-1])
                except Exception:
                    params['V0'] = np.nan
            for key, col in (('A1','Amplitude1'), ('A2','Amplitude2'), ('φ1','Phase1'), ('φ2','Phase2')):
                if col in df.columns:
                    try:
                        params[key] = float(df[col].iloc[-1])
                    except Exception:
                        params[key] = np.nan
    except Exception:
        pass
    return params

# --- reconstruct v(t) from params ---
def reconstruct_v_series_from_params(V0, A1, phi1, A2, phi2, n_samples=400, omega=1.0):
    # ensure numeric
    V0 = 0.0 if np.isnan(V0) else float(V0)
    A1 = 0.0 if np.isnan(A1) else float(A1)
    A2 = 0.0 if np.isnan(A2) else float(A2)
    phi1 = 0.0 if np.isnan(phi1) else float(phi1)
    phi2 = 0.0 if np.isnan(phi2) else float(phi2)
    # convert degrees to radians if needed
    if abs(phi1) > 2*np.pi:
        phi1 = phi1 * np.pi / 180.0
    if abs(phi2) > 2*np.pi:
        phi2 = phi2 * np.pi / 180.0
    t = np.linspace(0, 2*np.pi, n_samples)
    v = V0 + A1 * np.sin(omega * t + phi1 + np.pi) + A2 * np.sin(2 * omega * t + phi2 + np.pi)
    return t, v

# --- compute R and MSE (v_ideal=1) ---
def compute_R_from_params(V0, A1, A2):
    try:
        V0_f = float(V0)
        a1 = 0.0 if np.isnan(A1) else float(A1)
        a2 = 0.0 if np.isnan(A2) else float(A2)
        if V0_f == 0:
            return np.nan
        return np.sqrt(a1**2 + a2**2) / V0_f
    except Exception:
        return np.nan

def compute_MSE_from_params(V0, A1, phi1, A2, phi2, v_ideal=1.0, n_samples=400, omega=1.0):
    _, v = reconstruct_v_series_from_params(V0,A1,phi1,A2,phi2,n_samples=n_samples, omega=omega)
    return float(np.nanmean((v - v_ideal)**2))

# --- main analysis ---
def analyze_experiment55(data_dir=r"D:\vectionProject\public\ExperimentData55", out_prefix="Compensation"):
    data_path = Path(data_dir)
    if not data_path.exists():
        print("data dir not found:", data_path)
        return

    # conditions to include (order used for plotting).
    # 包含历史参数作为 LuminanceLinearMix_Old，脚本中读取到的文件中 LuminanceLinearMix 视为 New
    conds = [
        "LuminanceLinearMix_Old",
        "LuminanceLinearMix_New",
        "CameraJumpMoveMinusCompensate",
        "CameraJumpMovePlusCompensate",
        "LuminanceMinusCompensate",
        "LuminancePlusCompensate"
    ]

    # known subject codes (from provided ModParams)
    subject_codes = ["YAMA_A","OMU_B","ONO_C","HOU_D","LL_E","KK_F"]

    files = sorted([p for p in data_path.glob("*.csv") if "_Test" not in p.name and not p.name.endswith("Test.csv")])
    rows = []
    for p in files:
        name = p.name
        # map filename substring to base condition
        cond = None
        if "LuminanceLinearMix" in name:
            # 把读取到的 LuminanceLinearMix CSV 标为 New；历史参数已放到 LuminanceLinearMix_Old
            cond = "LuminanceLinearMix_New"
        else:
            for base in ("CameraJumpMoveMinusCompensate","CameraJumpMovePlusCompensate","LuminanceMinusCompensate","LuminancePlusCompensate"):
                if base in name:
                    cond = base
                    break
        if cond is None:
            continue
        # find subject code
        subj = "UNKNOWN"
        for s in subject_codes:
            if s in name:
                subj = s
                break
        # extract params
        params = get_params_from_file(p)
        V0 = params.get('V0', np.nan)
        A1 = params.get('A1', np.nan)
        phi1 = params.get('φ1', np.nan)
        A2 = params.get('A2', np.nan)
        phi2 = params.get('φ2', np.nan)
        R = compute_R_from_params(V0, A1, A2)
        MSE = compute_MSE_from_params(V0,A1,phi1,A2,phi2,v_ideal=1.0)
        rows.append({
            "file": str(p),
            "subject": subj,
            "condition": cond,
            "V0": V0, "A1": A1, "A2": A2, "φ1": phi1, "φ2": phi2,
            "R": R, "MSE": MSE
        })

    # --- add previous LuminanceLinearMix ModParams as an extra condition 'LuminanceLinearMix_Old' ---
    prev_modparams = {
        "YAMA_A": (0.992, 0.540, 1.849, -0.528, 1.462),
        "OMU_B":  (1.131, 0.522, 2.528, -0.223, 3.525),
        "ONO_C":  (1.067, 0.632, 3.663, 0.461, 5.123),
        "HOU_D":  (0.951, 0.275, 3.031, 0.920, 5.982),
        "LL_E":   (1.027, -0.278, 1.849, -0.292, 3.728),
        # 如果你也要包含 KK_F 的历史值，可取消下一行注释
        # "KK_F":   (1.129, 0.815, 3.462, 0.860, 5.854)
    }
    for subj, tup in prev_modparams.items():
        V0,A1,phi1,A2,phi2 = tup
        R = compute_R_from_params(V0,A1,A2)
        MSE = compute_MSE_from_params(V0,A1,phi1,A2,phi2,v_ideal=1.0)
        rows.append({
            "file": "PREV_MODPARAMS",
            "subject": subj,
            "condition": "LuminanceLinearMix_Old",
            "V0": V0, "A1": A1, "A2": A2, "φ1": phi1, "φ2": phi2,
            "R": R, "MSE": MSE
        })

    df = pd.DataFrame(rows)
    if df.empty:
        print("no matching data found")
        return

    # save raw table
    out_csv = Path.cwd() / f"{out_prefix}_raw_summary.csv"
    df.to_csv(out_csv, index=False)
    print("Saved raw table to", out_csv)

    def _wrap2(label, width=18):
        parts = textwrap.wrap(str(label), width=width)
        if len(parts) <= 2:
            return "\n".join(parts)
        return parts[0] + "\n" + parts[1] + "…"

    # --- per-subject per-condition mean V0,A1,A2 (plot) ---
    metrics = ["V0","A1","A2"]
    grouped = df.groupby(["subject","condition"])[metrics].mean().reset_index()

    # pivot for plotting: conditions as rows (x-axis), subjects as columns (different bars/colors)
    for m in metrics:
        pv = grouped.pivot(index="condition", columns="subject", values=m).reindex(index=conds).fillna(np.nan)
        fig, ax = plt.subplots(figsize=(max(8, pv.shape[0]*1.2), 5))
        pv.plot(kind='bar', ax=ax)
        ax.set_title(f"Mean {m} per condition (subjects as series)")
        ax.set_xlabel("condition")
        ax.set_ylabel(m)
        ax.legend(title="subject", bbox_to_anchor=(1.02,1), loc="upper left")
        # 更小的字体，最多两行，不旋转
        labels2 = [_wrap2(x) for x in pv.index]
        ax.set_xticklabels(labels2, rotation=0, ha='center', fontsize=6)
        ax.tick_params(axis='x', which='major', pad=6)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25)  # 给多行标签留空间
        outp = Path.cwd() / f"{out_prefix}_mean_{m}_by_condition.png"
        fig.savefig(outp, dpi=200, bbox_inches='tight')
        print("Saved", outp)
        plt.close(fig)

    # --- compute R and MSE summary per subject-condition (mean ± sem) and save ---
    stats = df.groupby(["subject","condition"]).agg(
        n = ("file","count"),
        mean_V0 = ("V0","mean"),
        mean_A1 = ("A1","mean"),
        mean_A2 = ("A2","mean"),
        mean_R = ("R","mean"),
        sem_R = ("R", lambda x: np.nanstd(x, ddof=0)/np.sqrt(x.size) if x.size>0 else np.nan),
        mean_MSE = ("MSE","mean"),
        sem_MSE = ("MSE", lambda x: np.nanstd(x, ddof=0)/np.sqrt(x.size) if x.size>0 else np.nan)
    ).reset_index()
    out_stats = Path.cwd() / f"{out_prefix}_stats_per_subject_condition.csv"
    stats.to_csv(out_stats, index=False)
    print("Saved stats table to", out_stats)

    # plot mean_R and mean_MSE grouped by condition (subjects as separate bars)
    for metric_key, label in (("mean_R","R"), ("mean_MSE","MSE")):
        pv = stats.pivot(index="condition", columns="subject", values=metric_key).reindex(index=conds).fillna(np.nan)
        fig, ax = plt.subplots(figsize=(max(8, pv.shape[0]*1.2), 5))
        pv.plot(kind='bar', ax=ax)
        ax.set_title(f"{label} per condition (subjects as series)")
        ax.set_xlabel("condition")
        ax.set_ylabel(label)
        ax.legend(title="subject", bbox_to_anchor=(1.02,1), loc="upper left")
        labels2 = [_wrap2(x) for x in pv.index]
        ax.set_xticklabels(labels2, rotation=0, ha='center', fontsize=6)
        ax.tick_params(axis='x', which='major', pad=6)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25)
        outp = Path.cwd() / f"{out_prefix}_{label}_by_condition.png"
        fig.savefig(outp, dpi=200, bbox_inches='tight')
        print("Saved", outp)
        plt.close(fig)

    # 折线图：每个 subject 为一条线，x 为 condition，y 为 mean_R
    pv_R = stats.pivot(index="condition", columns="subject", values="mean_R").reindex(index=conds)
    if pv_R.notna().any().any():
        fig, ax = plt.subplots(figsize=(max(8, pv_R.shape[0]*1.2), 5))
        pv_R.plot(ax=ax, marker='o', linewidth=1.5)
        ax.set_title("R across conditions (subjects as lines)")
        ax.set_xlabel("condition")
        ax.set_ylabel("R")
        ax.set_xticks(range(len(pv_R.index)))
        ax.set_xticklabels([_wrap2(x) for x in pv_R.index], rotation=0, fontsize=6)
        ax.legend(title="subject", bbox_to_anchor=(1.02,1), loc="upper left", fontsize=6)
        ax.grid(alpha=0.2)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25)
        outp = Path.cwd() / f"{out_prefix}_R_lines_by_condition.png"
        fig.savefig(outp, dpi=200, bbox_inches='tight')
        print("Saved", outp)
        plt.close(fig)

    # 折线图：每个 subject 为一条线，x 为 condition，y 为 mean_MSE
    pv_MSE = stats.pivot(index="condition", columns="subject", values="mean_MSE").reindex(index=conds)
    if pv_MSE.notna().any().any():
        fig, ax = plt.subplots(figsize=(max(8, pv_MSE.shape[0]*1.2), 5))
        pv_MSE.plot(ax=ax, marker='o', linewidth=1.5)
        ax.set_title("MSE (v_ideal=1) across conditions (subjects as lines)")
        ax.set_xlabel("condition")
        ax.set_ylabel("MSE")
        ax.set_xticks(range(len(pv_MSE.index)))
        ax.set_xticklabels([_wrap2(x) for x in pv_MSE.index], rotation=0, fontsize=6)
        ax.set_yscale('linear')
        ax.legend(title="subject", bbox_to_anchor=(1.02,1), loc="upper left", fontsize=6)
        ax.grid(alpha=0.2)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.25)
        outp = Path.cwd() / f"{out_prefix}_MSE_lines_by_condition.png"
        fig.savefig(outp, dpi=200, bbox_inches='tight')
        print("Saved", outp)
        plt.close(fig)

    # 折线图：A1, A2, V0（每个 subject 一条线，x 为 condition）
    for metric_key, ylabel in (("mean_V0","V0"), ("mean_A1","A1"), ("mean_A2","A2")):
        pv_metric = stats.pivot(index="condition", columns="subject", values=metric_key).reindex(index=conds)
        if pv_metric.notna().any().any():
            fig, ax = plt.subplots(figsize=(max(8, pv_metric.shape[0]*1.2), 5))
            pv_metric.plot(ax=ax, marker='o', linewidth=1.2)
            ax.set_title(f"{ylabel} across conditions (subjects as lines)")
            ax.set_xlabel("condition")
            ax.set_ylabel(ylabel)
            ax.set_xticks(range(len(pv_metric.index)))
            ax.set_xticklabels([_wrap2(x) for x in pv_metric.index], rotation=0, fontsize=6)
            ax.grid(alpha=0.2)
            ax.legend(title="subject", bbox_to_anchor=(1.02,1), loc="upper left", fontsize=6)
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.25)
            outp = Path.cwd() / f"{out_prefix}_{ylabel}_lines_by_condition.png"
            fig.savefig(outp, dpi=200, bbox_inches='tight')
            print("Saved", outp)
            try:
                plt.show()
            except Exception:
                pass
            plt.close(fig)

    print("Done.")

if __name__ == "__main__":
    analyze_experiment55()