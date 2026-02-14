import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import textwrap

def get_params_from_file(p: Path):
    """提取 V0, A1, A2（兼容 StepNumber/Amplitude/Velocity 列），返回 dict"""
    params = {'V0': np.nan, 'A1': np.nan, 'A2': np.nan}
    try:
        if not p.exists():
            return params
        df = pd.read_csv(p)
        df.columns = df.columns.str.strip()
        if "StepNumber" in df.columns:
            if "Velocity" in df.columns:
                s0 = df[df["StepNumber"] == 0]["Velocity"]
                if not s0.empty:
                    params['V0'] = float(s0.iloc[-1])
            elif "Amplitude" in df.columns:
                s0 = df[df["StepNumber"] == 0]["Amplitude"]
                if not s0.empty:
                    params['V0'] = float(s0.iloc[-1])
            if "Amplitude" in df.columns:
                s1 = df[df["StepNumber"] == 1]["Amplitude"]
                if not s1.empty:
                    params['A1'] = float(s1.iloc[-1])
                s3 = df[df["StepNumber"] == 3]["Amplitude"]
                if not s3.empty:
                    params['A2'] = float(s3.iloc[-1])
        else:
            for key, col in (('V0','Velocity'), ('A1','Amplitude1'), ('A2','Amplitude2')):
                if col in df.columns:
                    try:
                        params[key] = float(df[col].iloc[-1])
                    except Exception:
                        pass
    except Exception:
        pass
    return params

def compute_videal_sse(p: Path, v_ideal: float = 1.0, candidates=("Velocity","v_represent","RepresentVelocity")) -> float:
    """计算文件中 v_represent(t) 相对于常数 v_ideal 的 SSE（sum of squared errors）。返回 np.nan 表示失败/无数据。"""
    try:
        if not p.exists():
            return np.nan
        df = pd.read_csv(p)
        df.columns = df.columns.str.strip()
        for col in candidates:
            if col in df.columns:
                arr = pd.to_numeric(df[col], errors='coerce').to_numpy(dtype=float)
                if arr.size == 0:
                    return np.nan
                return float(np.nansum((arr - v_ideal) ** 2))
    except Exception:
        pass
    return np.nan

def collect_kk_files(dirs, kk_token="KK", must_contain=None):
    """收集指定文件夹中包含 kk_token 且包含 must_contain 中某一关键词的 csv 文件"""
    files = []
    for d in dirs:
        p = Path(d)
        if not p.exists():
            continue
        for f in p.glob("*.csv"):
            name = f.name
            if kk_token in name and (must_contain is None or any(k in name for k in must_contain)):
                files.append(f)
    return sorted(files)

def analyze_across_conditions(data_dirs, condition_keywords, extra_linearonly_paths=None, out_prefix="KK_analysis"):
    """对每个 condition_keywords 中的关键词作为一组，提取 V0/A1/A2，计算 R，打印并画图
    若提供 extra_linearonly_paths，则把它们作为单独的 'LinearOnly_Old' / 'LinearOnly_New' 组加入第一张图中。
    """
    # collect files
    all_files = []
    for d in data_dirs:
        p = Path(d)
        if p.exists():
            all_files.extend(list(p.glob("*.csv")))

    # map group -> list of files for condition keywords
    groups = {k: [] for k in condition_keywords}
    for f in sorted(all_files):
        name = f.name
        for key in condition_keywords:
            if key in name:
                groups[key].append(f)
                break

    # handle extra linearonly as separate old/new groups (do not mix into other groups)
    if extra_linearonly_paths:
        linear_old, linear_new = [], []
        for ep in extra_linearonly_paths:
            ep = Path(ep)
            if not ep.exists():
                continue
            sp = str(ep)
            # 判断依据：旧数据位于 BrightnessData，新的位于 ExperimentData33（用户提供的路径）
            if "BrightnessData" in sp:
                linear_old.append(ep)
            elif "ExperimentData33" in sp:
                linear_new.append(ep)
            else:
                # 未知来源默认归到 old
                linear_old.append(ep)
        if linear_old:
            groups['LinearOnly_Old'] = sorted(linear_old)
        if linear_new:
            groups['LinearOnly_New'] = sorted(linear_new)

    # collect params per group
    summary = {}
    for key, files in groups.items():
        V0s, A1s, A2s, Rs, SSEs = [], [], [], [], []
        print(f"\n=== Group: {key} ({len(files)} files) ===")
        for f in files:
            params = get_params_from_file(f)
            V0, A1, A2 = params['V0'], params['A1'], params['A2']
            sse = compute_videal_sse(f, v_ideal=1.0)
            print(f"{f.name} -> V0={V0}, A1={A1}, A2={A2}, SSE_videal={sse}")
            if not np.isnan(V0):
                V0s.append(V0)
            if not np.isnan(A1):
                A1s.append(A1)
            if not np.isnan(A2):
                A2s.append(A2)
            if (not np.isnan(A1)) and (not np.isnan(A2)) and (not np.isnan(V0)) and V0 != 0:
                R = np.sqrt(A1**2 + A2**2) / V0
                Rs.append(R)
            if not np.isnan(sse):
                SSEs.append(sse)
        summary[key] = {
            'V0': np.array(V0s, dtype=float),
            'A1': np.array(A1s, dtype=float),
            'A2': np.array(A2s, dtype=float),
            'R':  np.array(Rs, dtype=float),
            'SSE': np.array(SSEs, dtype=float)
        }

    # print group statistics
    print("\n--- Group statistics (mean ± std, n) ---")
    metrics = ['V0','A1','A2','R']
    for key in groups.keys():
        print(f"\nGroup {key}:")
        for m in metrics:
            arr = summary[key][m]
            if arr.size == 0:
                print(f"  {m}: n=0")
            else:
                print(f"  {m}: n={arr.size}, mean={np.nanmean(arr):.4f}, std={np.nanstd(arr):.4f}")

    # prepare labels (include LinearOnly if created)
    labels = list(groups.keys())
    x = np.arange(len(labels))
    width = 0.18

    # prepare means and sems
    means = {m: [] for m in metrics}
    sems = {m: [] for m in metrics}
    for key in labels:
        for m in metrics:
            arr = summary[key][m]
            if arr.size == 0:
                means[m].append(np.nan)
                sems[m].append(0.0)
            else:
                means[m].append(np.nanmean(arr))
                sems[m].append(np.nanstd(arr)/np.sqrt(arr.size))

    # three-panel figure: V0/A1/A2 (include LinearOnly)
    wrapped_labels = [textwrap.fill(l, width=18) for l in labels]

    fig, ax = plt.subplots(figsize=(12,6))
    for i, m in enumerate(['V0','A1','A2']):
        ax.bar(x + (i-1)*width, means[m], width, yerr=sems[m], capsize=4, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(wrapped_labels, rotation=0, fontsize=9)
    ax.set_ylabel("value")
    ax.set_title("KK: V0 / A1 / A2 across conditions")
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.28)
    out1 = Path.cwd() / f"{out_prefix}_V0_A1_A2.png"
    fig.savefig(out1, dpi=300, bbox_inches='tight')
    print("Saved", out1.resolve(), " (exists=", out1.exists(), ")")
    try:
        plt.show()
    except Exception as e:
        print("plt.show() failed:", e)
    plt.close(fig)

    # R plot (keep same label order) - wrap labels and increase bottom margin so labels are readable
    fig2, ax2 = plt.subplots(figsize=(10,5))
    ax2.bar(x, means['R'], width*1.5, yerr=sems['R'], capsize=4, color='#6b5b95')
    ax2.set_xticks(x)
    ax2.set_xticklabels(wrapped_labels, rotation=0, fontsize=9)
    ax2.set_ylabel("R = sqrt(A1^2 + A2^2) / V0")
    ax2.set_title("KK: composite motion strength R across conditions")
    ax2.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.28)
    out2 = Path.cwd() / f"{out_prefix}_R.png"
    fig2.savefig(out2, dpi=300, bbox_inches='tight')
    print("Saved", out2.resolve(), " (exists=", out2.exists(), ")")
    try:
        plt.show()
    except Exception as e:
        print("plt.show() failed:", e)
    plt.close(fig2)

    # 新增：每组的 v_ideal=1.0 的 SSE 条形图
    fig_sse, ax_sse = plt.subplots(figsize=(10,5))
    sse_means = []
    sse_sems = []
    for key in labels:
        arr = summary[key].get('SSE', np.array([], dtype=float))
        if arr.size == 0:
            sse_means.append(np.nan)
            sse_sems.append(0.0)
        else:
            sse_means.append(np.nanmean(arr))
            sse_sems.append(np.nanstd(arr)/np.sqrt(arr.size))
    ax_sse.bar(x, sse_means, width*1.5, yerr=sse_sems, capsize=4, color='#ff7f0e')
    ax_sse.set_xticks(x)
    wrapped_labels = [textwrap.fill(l, width=18) for l in labels]
    ax_sse.set_xticklabels(wrapped_labels, rotation=0, fontsize=9)
    ax_sse.set_ylabel("SSE (v_represent - 1)^2 summed")
    ax_sse.set_title("SSE to v_ideal=1.0 across conditions")
    ax_sse.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.28)
    out_sse = Path.cwd() / f"{out_prefix}_SSE_videal1.png"
    fig_sse.savefig(out_sse, dpi=300, bbox_inches='tight')
    print("Saved SSE figure to", out_sse.resolve(), " (exists=", out_sse.exists(), ")")
    try:
        plt.show()
    except Exception as e:
        print("plt.show() failed:", e)
    plt.close(fig_sse)

    # plot reconstructed v(t) per group (mean across files in group)
    n_groups = len(labels)
    if n_groups > 0:
        fig_v, axes = plt.subplots(1, n_groups, figsize=(4 * max(1, n_groups), 4), squeeze=False)
        for gi, key in enumerate(labels):
            axv = axes[0, gi]
            files = groups.get(key, [])
            stack = []
            for f in files:
                t, v = reconstruct_v_series(f, n_samples=400)
                if t is None:
                    continue
                stack.append(v)
            if len(stack) == 0:
                axv.text(0.5, 0.5, "no data", ha='center', va='center')
                axv.set_title(textwrap.fill(key, width=18))
                axv.set_xticks([])
                axv.set_yticks([])
                continue
            arr = np.vstack(stack)  # shape (n_files, n_samples)
            mean_v = np.nanmean(arr, axis=0)
            sem_v = np.nanstd(arr, axis=0) / np.sqrt(arr.shape[0])
            # common t
            t = np.linspace(0, 2*np.pi, mean_v.size)
            axv.plot(t, mean_v, color='C0', label='mean v(t)')
            axv.fill_between(t, mean_v - sem_v, mean_v + sem_v, color='C0', alpha=0.2)
            axv.axhline(1.0, color='k', linestyle='--', label='v_ideal=1.0')
            # annotate SSE mean for group if available
            s_arr = summary[key].get('SSE', np.array([], dtype=float))
            s_text = f"mean SSE={np.nanmean(s_arr):.3g}" if s_arr.size>0 else "SSE n=0"
            axv.text(0.02, 0.95, s_text, transform=axv.transAxes, va='top', fontsize=8)
            axv.set_title(textwrap.fill(key, width=18), fontsize=9)
            axv.set_xlabel("t (rad)")
            axv.set_ylabel("v(t)")
            axv.grid(alpha=0.2)
            axv.legend(fontsize=8)
        plt.tight_layout()
        out_v = Path.cwd() / f"{out_prefix}_v_represent_curves.png"
        fig_v.savefig(out_v, dpi=300, bbox_inches='tight')
        print("Saved v(t) curves to", out_v.resolve(), " (exists=", out_v.exists(), ")")
        try:
            plt.show()
        except Exception:
            pass
        plt.close(fig_v)

def reconstruct_v_series(p: Path, n_samples=200):
    """从文件重建 v(t) 序列（使用 get_params_from_file），返回 (t, v)；
    t 在 [0, 2π] 上等间隔，长度为 n_samples。若失败返回 (None, None)。"""
    try:
        if not p.exists():
            return None, None
        params = get_params_from_file(p)
        V0 = 0.0 if np.isnan(params.get('V0', np.nan)) else float(params.get('V0', 0.0))
        A1 = 0.0 if np.isnan(params.get('A1', np.nan)) else float(params.get('A1', 0.0))
        A2 = 0.0 if np.isnan(params.get('A2', np.nan)) else float(params.get('A2', 0.0))
        phi1 = 0.0 if np.isnan(params.get('φ1', np.nan)) else float(params.get('φ1', 0.0))
        phi2 = 0.0 if np.isnan(params.get('φ2', np.nan)) else float(params.get('φ2', 0.0))

        # ensure radians
        for ph in (phi1, phi2):
            if abs(ph) > 2*np.pi:
                phi1 = phi1 * np.pi / 180.0
                phi2 = phi2 * np.pi / 180.0
                break

        # time vector: use file length if available, otherwise fixed [0,2π]
        df = pd.read_csv(p)
        df.columns = df.columns.str.strip()
        if "Time" in df.columns or "time" in df.columns:
            col = "Time" if "Time" in df.columns else "time"
            t_raw = pd.to_numeric(df[col], errors='coerce').dropna().to_numpy(dtype=float)
            if t_raw.size >= 2:
                t = np.linspace(t_raw.min(), t_raw.max(), n_samples)
            else:
                t = np.linspace(0, 2*np.pi, n_samples)
        else:
            t = np.linspace(0, 2*np.pi, n_samples)

        # try to get omega from file (Frequency or Omega), fallback omega=1
        omega = None
        for col in ("Frequency","frequency","f","Omega","omega"):
            if col in df.columns:
                try:
                    val = float(pd.to_numeric(df[col], errors='coerce').dropna().iloc[0])
                    if col.lower() in ("frequency","f"):
                        omega = 2*np.pi*val
                    else:
                        omega = val
                    break
                except Exception:
                    pass
        if omega is None:
            omega = 1.0

        # v(t) = V0 + A1*sin(ωt + φ1 + π) + A2*sin(2ωt + φ2 + π)
        v = V0 + A1 * np.sin(omega * t + phi1 + np.pi) + A2 * np.sin(2 * omega * t + phi2 + np.pi)
        return t, v
    except Exception:
        return None, None

if __name__ == "__main__":
    # 数据来源目录（根据截图）
    dirs = [
        r"D:\vectionProject\public\ExperimentData44",
        r"D:\vectionProject\public\ExperimentData33"
    ]
    # 四个条件关键词（依据截图3 的文件名），如需调整可修改列表顺序/字符串
    condition_keywords = [
        "LuminancePlusCompensate",
        "LuminanceMinusCompensate",
        "CameraJumpMovePlusCompensate",
        "CameraJumpMoveMinusCompensate"
    ]
    # 额外 LinearOnly three files（如果需要将其作为一组加入，可传入 extra_linearonly_paths 参数）
    extra_linearonly = [
        # 旧的 LinearOnly 在 BrightnessData
        Path(r"D:\vectionProject\public\BrightnessData\20250709_152809_Fps1_CameraSpeed1_ExperimentPattern_Phase_ParticipantName_KK_TrialNumber_1_BrightnessBlendMode_LinearOnly.csv"),
        Path(r"D:\vectionProject\public\BrightnessData\20250709_151437_Fps1_CameraSpeed1_ExperimentPattern_Phase_ParticipantName_KK_TrialNumber_3_BrightnessBlendMode_LinearOnly.csv"),
        Path(r"D:\vectionProject\public\BrightnessData\20250709_154001_Fps1_CameraSpeed1_ExperimentPattern_Phase_ParticipantName_KK_TrialNumber_2_BrightnessBlendMode_LinearOnly.csv"),
        # 新的 LinearOnly（LuminanceLinearMix）位于 ExperimentData33，作为 "new" 组
        Path(r"D:\vectionProject\public\ExperimentData33\20251120_172632_ExperimentPattern_LuminanceLinearMix_ParticipantName_KK_Subject_Name_KK_F_V0_A1A2_TrialNumber_1.csv"),
        Path(r"D:\vectionProject\public\ExperimentData33\20251120_172825_ExperimentPattern_LuminanceLinearMix_ParticipantName_KK_Subject_Name_KK_F_V0_A1A2_TrialNumber_2.csv"),
        Path(r"D:\vectionProject\public\ExperimentData33\20251120_173043_ExperimentPattern_LuminanceLinearMix_ParticipantName_KK_Subject_Name_KK_F_V0_A1A2_TrialNumber_3.csv"),
    ]
    analyze_across_conditions(dirs, condition_keywords, extra_linearonly_paths=extra_linearonly, out_prefix="KK_44_33")