import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import textwrap

def get_params_from_file(p: Path):
    """从文件提取 V0, A1, φ1, A2, φ2（兼容 StepNumber/Amplitude/Velocity/Phase 列），返回 dict"""
    params = {'V0': np.nan, 'A1': np.nan, 'φ1': np.nan, 'A2': np.nan, 'φ2': np.nan}
    try:
        if not p.exists():
            return params
        df = pd.read_csv(p)
        df.columns = df.columns.str.strip()
        if "StepNumber" in df.columns:
            # V0: StepNumber == 0, prefer Velocity 列
            if "Velocity" in df.columns:
                s0 = df[df["StepNumber"] == 0]["Velocity"]
                if not s0.empty:
                    params['V0'] = float(s0.iloc[-1])
            elif "Amplitude" in df.columns:
                s0 = df[df["StepNumber"] == 0]["Amplitude"]
                if not s0.empty:
                    params['V0'] = float(s0.iloc[-1])

            # Amplitudes and phases according to step mapping
            if "Amplitude" in df.columns:
                s1 = df[df["StepNumber"] == 1]["Amplitude"]
                if not s1.empty:
                    params['A1'] = float(s1.iloc[-1])
                s2 = df[df["StepNumber"] == 2]["Amplitude"]
                if not s2.empty:
                    params['φ1'] = float(s2.iloc[-1])
                s3 = df[df["StepNumber"] == 3]["Amplitude"]
                if not s3.empty:
                    params['A2'] = float(s3.iloc[-1])
                s4 = df[df["StepNumber"] == 4]["Amplitude"]
                if not s4.empty:
                    params['φ2'] = float(s4.iloc[-1])
            else:
                # fallback column names for amplitudes/phases
                for col in ("Amplitude1","A1","Amp","Amplitude"):
                    if col in df.columns and np.isnan(params['A1']):
                        try:
                            params['A1'] = float(df[col].iloc[-1])
                        except Exception:
                            pass
                for col in ("Phase1","φ1","Phase","Phase_1"):
                    if col in df.columns and np.isnan(params['φ1']):
                        try:
                            params['φ1'] = float(df[col].iloc[-1])
                        except Exception:
                            pass
                for col in ("Amplitude2","A2"):
                    if col in df.columns and np.isnan(params['A2']):
                        try:
                            params['A2'] = float(df[col].iloc[-1])
                        except Exception:
                            pass
                for col in ("Phase2","φ2","Phase_2"):
                    if col in df.columns and np.isnan(params['φ2']):
                        try:
                            params['φ2'] = float(df[col].iloc[-1])
                        except Exception:
                            pass
                if "Velocity" in df.columns and np.isnan(params['V0']):
                    try:
                        params['V0'] = float(df["Velocity"].iloc[-1])
                    except Exception:
                        pass
        else:
            # no StepNumber: try direct column names
            for key, col in (('V0','Velocity'), ('A1','Amplitude1'), ('φ1','Phase1'), ('A2','Amplitude2'), ('φ2','Phase2')):
                if col in df.columns:
                    try:
                        params[key] = float(df[col].iloc[-1])
                    except Exception:
                        pass
    except Exception:
        pass
    return params

def analyze_kk_experiment33(data_dir=r"D:\vectionProject\public\ExperimentData44"):
    """查找 KK 的条件文件，提取 V0/A1/φ1/A2/φ2 并画条形图比较（保存并打印均值）
    同时包含指定的 LinearOnly 额外文件作为独立组。
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        print("data_dir not found:", data_path)
        return

    csv_files = list(data_path.glob("*.csv"))
    # 只匹配包含 KK 且属于 NoLuminanceBlendSingleCameraMove（或同时包含 NoLuminanceBlend 与 SingleCameraMove）的文件名
    kk_files = []
    for p in csv_files:
        name = p.name
        has_kk = ("ParticipantName_KK" in name) or ("_KK_" in name)
        has_move = ("NoLuminanceBlendSingleCameraMove" in name) or ("NoLuminanceBlend" in name and "SingleCameraMove" in name)
        if has_kk and has_move:
            kk_files.append(p)
    print(f"Found {len(kk_files)} KK files matching NoLuminanceBlendSingleCameraMove:")
    for p in kk_files:
        print("  ->", p.name)

    # 额外的 three LinearOnly files（按用户提供路径）
    extra_paths1 = [
        # 旧 LinearOnly（BrightnessData）
        Path(r"D:\vectionProject\public\BrightnessData\20250709_152809_Fps1_CameraSpeed1_ExperimentPattern_Phase_ParticipantName_KK_TrialNumber_1_BrightnessBlendMode_LinearOnly.csv"),
        Path(r"D:\vectionProject\public\BrightnessData\20250709_151437_Fps1_CameraSpeed1_ExperimentPattern_Phase_ParticipantName_KK_TrialNumber_3_BrightnessBlendMode_LinearOnly.csv"),
        Path(r"D:\vectionProject\public\BrightnessData\20250709_154001_Fps1_CameraSpeed1_ExperimentPattern_Phase_ParticipantName_KK_TrialNumber_2_BrightnessBlendMode_LinearOnly.csv"),
        # 新增 2025-11-20 在 ExperimentData33 的 LuminanceLinearMix（作为新的 LinearOnly 数据）
        Path(r"D:\vectionProject\public\ExperimentData33\20251120_172632_ExperimentPattern_LuminanceLinearMix_ParticipantName_KK_Subject_Name_KK_F_V0_A1A2_TrialNumber_1.csv"),
        Path(r"D:\vectionProject\public\ExperimentData33\20251120_172825_ExperimentPattern_LuminanceLinearMix_ParticipantName_KK_Subject_Name_KK_F_V0_A1A2_TrialNumber_2.csv"),
        Path(r"D:\vectionProject\public\ExperimentData33\20251120_173043_ExperimentPattern_LuminanceLinearMix_ParticipantName_KK_Subject_Name_KK_F_V0_A1A2_TrialNumber_3.csv"),
    ]
    # 将 extra_paths1 中存在的文件加入文件列表（避免重复）
    for p in extra_paths1:
        if p.exists() and p not in kk_files:
            kk_files.append(p)
            print("  appended extra LinearOnly:", p.name)

    if not kk_files:
        print("No KK files found in", data_path, "or extra_paths1.")
        return

    # 分类四种条件 + 额外 LinearOnly 组
    groups = {
        "V0_only": [],
        "V0_A1": [],
        "V0_A2": [],
        "V0_A1A2": [],
        "KK_LinearOnly_Old": [],   # 旧 BrightnessData 的 LinearOnly
        "KK_LinearOnly_New": []    # 新 ExperimentData33 的 LuminanceLinearMix
    }
    for p in kk_files:
        name = p.name
        if "V0_A1A2" in name:
            groups["V0_A1A2"].append(p)
        elif "V0_A1" in name and "V0_A1A2" not in name:
            groups["V0_A1"].append(p)
        elif "V0_A2" in name and "V0_A1A2" not in name:
            groups["V0_A2"].append(p)
        elif "V0_" in name and all(k not in name for k in ("V0_A1","V0_A2","V0_A1A2")):
            groups["V0_only"].append(p)
        # 如果文件位于 extra_paths1，则根据其来源分配到 Old / New 子组
        for ep in extra_paths1:
            try:
                if p.resolve() == ep.resolve():
                    sp = str(ep)
                    # 简单判定：来自 ExperimentData33 或文件名包含 LuminanceLinearMix 视作新数据
                    if "ExperimentData33" in sp or "LuminanceLinearMix" in name:
                        groups["KK_LinearOnly_New"].append(p)
                    else:
                        groups["KK_LinearOnly_Old"].append(p)
                    break
            except Exception:
                continue

    # Report counts
    print("KK groups counts:")
    for k, lst in groups.items():
        print(f"  {k}: {len(lst)} files")

    # 保留每组前5次（若有）
    for k in groups:
        groups[k] = sorted(groups[k])[:5]

    summary = {}
    for k, files in groups.items():
        vals_V0, vals_A1, vals_phi1, vals_A2, vals_phi2 = [], [], [], [], []
        print(f"\n--- Group {k} ({len(files)} files) ---")
        for p in files:
            params = get_params_from_file(p)
            print(f"{p.name} -> V0={params['V0']}, A1={params['A1']}, φ1={params['φ1']}, A2={params['A2']}, φ2={params['φ2']}")
            if not np.isnan(params['V0']):
                vals_V0.append(params['V0'])
            if not np.isnan(params['A1']):
                vals_A1.append(params['A1'])
            if not np.isnan(params['φ1']):
                vals_phi1.append(params['φ1'])
            if not np.isnan(params['A2']):
                vals_A2.append(params['A2'])
            if not np.isnan(params['φ2']):
                vals_phi2.append(params['φ2'])
        summary[k] = {
            'V0': np.array(vals_V0, dtype=float),
            'A1': np.array(vals_A1, dtype=float),
            'φ1': np.array(vals_phi1, dtype=float),
            'A2': np.array(vals_A2, dtype=float),
            'φ2': np.array(vals_phi2, dtype=float)
        }

    metrics = ['V0','A1','A2','φ1','φ2']
    groups_order = ["V0_only","V0_A1","V0_A2","V0_A1A2","KK_LinearOnly_Old","KK_LinearOnly_New"]
    means = {m: [] for m in metrics}
    stds = {m: [] for m in metrics}
    # compute R per file-group for composite strength
    R_groups = {}
    for k in groups_order:
        # compute R per-group from available A1,A2,V0 arrays (pairwise by index not necessary; compute from means)
        arrA1 = summary[k]['A1']
        arrA2 = summary[k]['A2']
        arrV0 = summary[k]['V0']
        # compute R values per-trial when possible: if equal lengths not guaranteed, compute per-file when all three present individually above is hard.
        # approximate group R from means: R = sqrt(mean(A1^2)+mean(A2^2))/mean(V0) if mean V0 != 0
        if arrV0.size > 0:
            denom = np.nanmean(arrV0)
            num = 0.0
            if arrA1.size > 0:
                num += np.nanmean(arrA1**2)
            if arrA2.size > 0:
                num += np.nanmean(arrA2**2)
            R_groups[k] = np.sqrt(num) / denom if denom != 0 else np.array([], dtype=float)
        else:
            R_groups[k] = np.array([], dtype=float)

    for k in groups_order:
        for m in metrics:
            arr = summary[k][m]
            if arr.size == 0:
                means[m].append(np.nan)
                stds[m].append(np.nan)
            else:
                means[m].append(float(np.nanmean(arr)))
                stds[m].append(float(np.nanstd(arr, ddof=0)))

    print("\n=== Summary means (V0, A1, A2, φ1, φ2) per group ===")
    for k in groups_order:
        print(f"{k}:")
        for m in metrics:
            idx = groups_order.index(k)
            val = means[m][idx]
            sd = stds[m][idx]
            if np.isnan(val):
                print(f"  {m}: n=0")
            else:
                print(f"  {m}: mean={val:.4f}, std={sd:.4f}")

    # 画图：V0/A1/A2
    labels = groups_order
    wrapped_labels = [textwrap.fill(l, width=18) for l in labels]
    x = np.arange(len(labels))
    width = 0.18

    fig, ax = plt.subplots(figsize=(12,6))
    for i, m in enumerate(['V0','A1','A2']):
        vals = means[m]
        errs = []
        for j in range(len(labels)):
            arr = summary[labels[j]][m]
            errs.append((np.nanstd(arr, ddof=0)/np.sqrt(arr.size)) if arr.size>0 else 0.0)
        ax.bar(x + (i-1)*width, vals, width, yerr=errs, capsize=4, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(wrapped_labels, rotation=0, fontsize=9)
    ax.set_ylabel("value")
    ax.set_title("KK: V0 / A1 / A2 across conditions (including LinearOnly)")
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.28)
    out_png = Path.cwd() / "KK_conditions_V0_A1_A2_with_LinearOnly.png"
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    print("\nSaved figure to", out_png.resolve(), " (exists=", out_png.exists(), ")")
    try:
        plt.show()
    except Exception as e:
        print("plt.show() failed:", e)
    plt.close(fig)

    # 画图：φ1 and φ2
    fig3, ax3 = plt.subplots(figsize=(10,5))
    phi1_vals = means['φ1']
    phi2_vals = means['φ2']
    phi1_errs = []
    phi2_errs = []
    for j in range(len(labels)):
        arr1 = summary[labels[j]]['φ1']
        arr2 = summary[labels[j]]['φ2']
        phi1_errs.append((np.nanstd(arr1, ddof=0)/np.sqrt(arr1.size)) if arr1.size>0 else 0.0)
        phi2_errs.append((np.nanstd(arr2, ddof=0)/np.sqrt(arr2.size)) if arr2.size>0 else 0.0)
    ax3.bar(x - 0.12, phi1_vals, 0.22, yerr=phi1_errs, capsize=4, label='φ1')
    ax3.bar(x + 0.12, phi2_vals, 0.22, yerr=phi2_errs, capsize=4, label='φ2')
    ax3.set_xticks(x)
    ax3.set_xticklabels(wrapped_labels, rotation=0, fontsize=9)
    ax3.set_ylabel("phase (rad or file units)")
    ax3.set_title("KK: φ1 and φ2 across conditions")
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.28)
    out3 = Path.cwd() / "KK_conditions_phases_with_LinearOnly.png"
    fig3.savefig(out3, dpi=300, bbox_inches='tight')
    print("Saved phases figure to", out3.resolve(), " (exists=", out3.exists(), ")")
    try:
        plt.show()
    except Exception as e:
        print("plt.show() failed:", e)
    plt.close(fig3)

    # 画图：R（合成强度，使用近似组均值计算）
    fig2, ax2 = plt.subplots(figsize=(10,5))
    R_vals = [R_groups[k] if isinstance(R_groups[k], float) or (isinstance(R_groups[k], np.ndarray) and R_groups[k].size>0) else np.nan for k in groups_order]
    # normalize R_vals to scalars
    R_plot = []
    R_errs = []
    for k in groups_order:
        arrA1 = summary[k]['A1']
        arrA2 = summary[k]['A2']
        arrV0 = summary[k]['V0']
        if arrV0.size>0:
            denom = np.nanmean(arrV0)
            num = 0.0
            if arrA1.size>0:
                num += np.nanmean(arrA1**2)
            if arrA2.size>0:
                num += np.nanmean(arrA2**2)
            R_plot.append(np.sqrt(num)/denom if denom!=0 else np.nan)
            # estimate err via propagation (rough): std(R) ~ std(sqrt(A1^2+A2^2))/sqrt(n) approximated by std of sqrt per-trial if available
            R_errs.append(0.0)
        else:
            R_plot.append(np.nan)
            R_errs.append(0.0)
    ax2.bar(x, R_plot, width*1.5, yerr=R_errs, capsize=4, color='#6b5b95')
    ax2.set_xticks(x)
    ax2.set_xticklabels(wrapped_labels, rotation=0, fontsize=9)
    ax2.set_ylabel("R = sqrt(mean(A1^2)+mean(A2^2)) / mean(V0)")
    ax2.set_title("KK: composite motion strength R across conditions")
    ax2.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.28)
    out2 = Path.cwd() / "KK_conditions_R_with_LinearOnly.png"
    fig2.savefig(out2, dpi=300, bbox_inches='tight')
    print("Saved R figure to", out2.resolve(), " (exists=", out2.exists(), ")")
    try:
        plt.show()
    except Exception as e:
        print("plt.show() failed:", e)
    plt.close(fig2)

if __name__ == "__main__":
    analyze_kk_experiment33()
