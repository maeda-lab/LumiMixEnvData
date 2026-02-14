import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import os
import glob
from pathlib import Path
from matplotlib.legend_handler import HandlerTuple, HandlerErrorbar
from matplotlib.container import ErrorbarContainer

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_function_mix_data(data_dir):
    """加载FunctionMix数据"""
    pattern = os.path.join(data_dir, "*ExperimentPattern_FunctionMix_ParticipantName_*.csv")
    files = glob.glob(pattern)
    
    all_data = {}
    for file in files:
        # 从文件名提取参与者信息
        filename = os.path.basename(file)
        parts = filename.split('_')
        
        # 提取参与者名称
        try:
            participant_idx = parts.index('ParticipantName') + 1
            participant = parts[participant_idx]
        except ValueError:
            continue
        
        # 提取试验编号
        try:
            trial_idx = parts.index('TrialNumber') + 1
            trial_str = parts[trial_idx]
            if trial_str.endswith('.csv'):
                trial_str = trial_str.replace('.csv', '')
            trial = int(trial_str)
        except Exception:
            trial = 0
        
        try:
            df = pd.read_csv(file)
            # 清理列名
            df.columns = df.columns.str.strip()
            df['Participant'] = participant
            df['Trial'] = trial
            df['Filename'] = filename
            
            if participant not in all_data:
                all_data[participant] = {}
            all_data[participant][trial] = df
            
            print(f"加载成功: {filename} - 参与者: {participant}, 试验: {trial}")
        except Exception as e:
            print(f"加载错误: {filename} - {e}")
    
    return all_data

def plot_mixing_functions(ax, function_type, direction='forward'):
    """绘制混合函数"""
    x = np.linspace(0, 1, 100)
    
    if function_type == 'cos':
        if direction == 'forward':
            y = 0.5 * (1 - np.cos(np.pi * x))
        else:
            y = 0.5 * (1 + np.cos(np.pi * x))
        title = 'Cos'
        color = 'green'
    elif function_type == 'linear':
        if direction == 'forward':
            y = x
        else:
            y = 1 - x
        title = 'Linear'
        color = 'blue'
    elif function_type == 'acos':
        if direction == 'forward':
            y = np.arccos(-2 * x + 1) / np.pi
        else:
            y = 1 - np.arccos(-2 * x + 1) / np.pi
        title = 'Acos '
        color = 'darkred'
    
    ax.plot(x, y, color=color, linewidth=2)
    ax.set_xlabel('t')
    ax.set_ylabel('')
    ax.set_title('')  # 移除标题
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])  # 不显示任何横轴刻度
    ax.set_yticks([0, 1])

def extract_final_function_ratio(df):
    """提取最终的FunctionRatio值"""
    final_data = df.tail(int(len(df) * 0.2)) if len(df) > 0 else df
    final_ratio = final_data['FunctionRatio'].iloc[-1] if ('FunctionRatio' in final_data.columns and len(final_data) > 0) else np.nan
    return final_ratio

def get_dynamic_blend(x, knob):
    """使混合基函数在 x=0 与 x=1 端点一致，从而单周期为 1s（备选，不用于 2s 周期方案）。"""
    x = np.asarray(x)
    x = np.mod(x, 1.0)
    cos_shape = 0.5 * (1.0 - np.cos(2.0 * np.pi * x))
    tri_shape = 1.0 - 2.0 * np.abs(x - 0.5)
    sin_shape = np.sin(np.pi * x)
    tri_shape = np.clip(tri_shape, 0.0, 1.0)
    if knob <= 0.1:
        return cos_shape
    elif knob <= 0.5:
        t = (knob - 0.1) / 0.4
        return (1.0 - t) * cos_shape + t * tri_shape
    elif knob <= 0.9:
        t = (knob - 0.5) / 0.4
        return (1.0 - t) * tri_shape + t * sin_shape
    else:
        return sin_shape

def _forward_blend(x, knob):
    """原始基函数（forward, x in [0,1]），与 C# 逻辑一致（从0->1）。"""
    x = np.asarray(x)
    cos_curve = 0.5 * (1 - np.cos(np.pi * x))
    linear = x
    acos_curve = np.arccos(-2 * x + 1) / np.pi
    if knob <= 0.1:
        return cos_curve
    elif knob <= 0.5:
        t = (knob - 0.1) / 0.4
        return (1.0 - t) * cos_curve + t * linear
    elif knob <= 0.9:
        t = (knob - 0.5) / 0.4
        return (1.0 - t) * linear + t * acos_curve
    else:
        return acos_curve

def get_dynamic_blend_2s(t, knob):
    """对时间 t 返回亮度：0..1s forward（0->1），1..2s backward（1->0），周期 2s 重复。"""
    t = np.asarray(t)
    phase = np.mod(t, 2.0)
    y = np.empty_like(phase, dtype=float)
    mask_fwd = phase < 1.0
    if np.any(mask_fwd):
        x_f = phase[mask_fwd]
        y[mask_fwd] = _forward_blend(x_f, knob)
    mask_bwd = ~mask_fwd
    if np.any(mask_bwd):
        x_b = phase[mask_bwd] - 1.0
        y[mask_bwd] = 1.0 - _forward_blend(x_b, knob)
    return y

# ---- velocity_linear_only code inlined ----
def extract_parameters(df):
    """从 LinearOnly 文件中提取速度参数"""
    df = df.copy()
    df.columns = df.columns.str.strip()
    if "StepNumber" in df.columns and "Velocity" in df.columns:
        v0_series = df[df["StepNumber"] == 0]["Velocity"]
    else:
        v0_series = pd.Series([], dtype=float)
    V0 = v0_series.iloc[-1] if not v0_series.empty else 0.0

    A1 = df[df["StepNumber"] == 1]["Amplitude"].iloc[-1] if ("StepNumber" in df.columns and "Amplitude" in df.columns and not df[df["StepNumber"] == 1].empty) else 0.0
    φ1 = df[df["StepNumber"] == 2]["Amplitude"].iloc[-1] if ("StepNumber" in df.columns and "Amplitude" in df.columns and not df[df["StepNumber"] == 2].empty) else 0.0
    A2 = df[df["StepNumber"] == 3]["Amplitude"].iloc[-1] if ("StepNumber" in df.columns and "Amplitude" in df.columns and not df[df["StepNumber"] == 3].empty) else 0.0
    φ2 = df[df["StepNumber"] == 4]["Amplitude"].iloc[-1] if ("StepNumber" in df.columns and "Amplitude" in df.columns and not df[df["StepNumber"] == 4].empty) else 0.0
    return V0, A1, φ1, A2, φ2

def calculate_mean_parameters_for_linear_only(participant_files):
    """计算参与者的平均速度参数（来自 LinearOnly 文件）"""
    all_params = []
    for file_path, trial_number in participant_files:
        try:
            df = pd.read_csv(file_path)
            V0, A1, φ1, A2, φ2 = extract_parameters(df)
            all_params.append({'V0': V0, 'A1': A1, 'φ1': φ1, 'A2': A2, 'φ2': φ2, 'trial': trial_number, 'file': Path(file_path).name})
        except Exception:
            continue
    if not all_params:
        return None
    mean_params = {
        'V0': float(np.mean([p['V0'] for p in all_params])),
        'A1': float(np.mean([p['A1'] for p in all_params])),
        'φ1': float(np.mean([p['φ1'] for p in all_params])),
        'A2': float(np.mean([p['A2'] for p in all_params])),
        'φ2': float(np.mean([p['φ2'] for p in all_params])),
    }
    return mean_params, all_params

def plot_velocity_curve_inline(V0, A1, φ1, A2, φ2, t_sec):
    """根据 mean params 生成 velocity(t)（返回数组）"""
    velocity = (
        V0
        + A1 * np.sin(2 * np.pi * t_sec + φ1 + np.pi)
        + A2 * np.sin(4 * np.pi * t_sec + φ2 + np.pi)
    )
    return velocity

def load_velocity_results_from_linear_only(data_dir, participants_order, total_seconds=4):
    """在当前文件中实现 velocity_curve_linear_only_analysis 的数据提取逻辑
    返回 participant -> list of (t_array, v_array)
    现在根据 total_seconds 生成速度时间轴（可绘制 4s 及以上）。
    """
    data_folder = Path(data_dir)
    if not data_folder.exists():
        data_folder = Path("../BrightnessFunctionMixAndPhaseData")
    linear_only_files = list(data_folder.glob("*LinearOnly*.csv"))
    if not linear_only_files:
        return {p: None for p in participants_order}

    participants = {}
    for file in linear_only_files:
        filename = file.name
        if "ParticipantName_" in filename and "TrialNumber_" in filename:
            participant = filename.split("ParticipantName_")[1].split("_")[0]
            trial_match = filename.split("TrialNumber_")[1].split("_")[0]
            try:
                trial_number = int(trial_match)
            except Exception:
                trial_number = 0
            participants.setdefault(participant, []).append((file, trial_number))

    all_participants = list(participants.keys())
    selected_participants = []
    for participant in all_participants:
        if participant not in ['HOU', 'LL'] and len(selected_participants) < 3:
            selected_participants.append(participant)
    if 'HOU' in all_participants:
        selected_participants.append('HOU')
    if 'LL' in all_participants:
        selected_participants.append('LL')
    for participant in all_participants:
        if participant not in selected_participants and len(selected_participants) < 5:
            selected_participants.append(participant)

    velocity_results = {p: None for p in participants_order}
    # 每秒采样率：保持原脚本等效密度（原来 2s -> 1000 点 => 500 Hz）
    samples_per_sec = 500
    N = max(2, int(float(total_seconds) * samples_per_sec))
    t_sec = np.linspace(0.0, float(total_seconds), N)
    for p in selected_participants:
        files = participants.get(p, [])
        if not files:
            continue
        res = calculate_mean_parameters_for_linear_only(files)
        if res is None:
            continue
        mean_params, all_params = res
        velocity = plot_velocity_curve_inline(
            mean_params['V0'], mean_params['A1'], mean_params['φ1'],
            mean_params['A2'], mean_params['φ2'], t_sec
        )
        if p in participants_order:
            velocity_results[p] = [(t_sec, velocity)]
    return velocity_results
 

# ---- main plotting function ----
def create_function_mix_plot(data_dir, total_seconds=4, samples_per_sec=200):
    """按参与者绘制中位混合函数（2s 周期）、导数及 LinearOnly 速度（第三排）"""
    print("加载FunctionMix数据...")
    function_mix_data = load_function_mix_data(data_dir)
    if not function_mix_data:
        print("没有找到FunctionMix数据")
        return

    participants_order = ['ONO', 'OMU', 'YAMA', 'HOU', 'LL']
    participant_labels = ['A', 'B', 'C', 'D', 'E']

    total_seconds = int(total_seconds)
    N = int(total_seconds * samples_per_sec)
    x = np.linspace(0.0, float(total_seconds), N, endpoint=False)

    # 收集每个参与者的最终 knob 值
    per_participant_knobs = {}
    for participant in participants_order:
        knobs = []
        if participant in function_mix_data:
            trials = function_mix_data[participant]
            for trial_num in sorted(trials.keys()):
                df = trials[trial_num]
                if 'FunctionRatio' not in df.columns:
                    continue
                try:
                    k = float(extract_final_function_ratio(df))
                    if not np.isnan(k):
                        knobs.append(k)
                except Exception:
                    continue
        per_participant_knobs[participant] = np.array(knobs, dtype=float)

    # 计算 y 与 dy（按 2s 周期重复）
    ys_by_participant = {}
    dys_by_participant = {}
    all_dy_maxabs = []
    for participant in participants_order:
        knobs = per_participant_knobs.get(participant, np.array([]))
        if knobs.size:
            median_knob = float(np.median(knobs))
            y = get_dynamic_blend_2s(x, median_knob)
            dy = np.gradient(y, x)
            phase2 = np.mod(x, 2.0)
            wrap_idx = np.where(np.diff(phase2) < 0)[0]
            for idx in wrap_idx:
                dy[idx] = np.nan
                if idx + 1 < dy.size:
                    dy[idx + 1] = np.nan
            ys_by_participant[participant] = (median_knob, y)
            dys_by_participant[participant] = dy
            all_dy_maxabs.append(np.nanmax(np.abs(dy)))
        else:
            ys_by_participant[participant] = None
            dys_by_participant[participant] = None

    # 统一第二排纵轴范围
    if all_dy_maxabs:
        global_maxabs = float(np.nanmax(all_dy_maxabs))
        if np.isnan(global_maxabs) or global_maxabs == 0:
            global_maxabs = 0.01
    else:
        global_maxabs = 1.0
    y_margin = global_maxabs * 0.05
    global_dy_ylim = (-global_maxabs - y_margin, global_maxabs + y_margin)

    # 取得 linear-only velocity 结果（内嵌实现）
    velocity_results = load_velocity_results_from_linear_only(data_dir, participants_order, total_seconds)


    # 将 velocity 结果插值到 x 轴（若可用）
    velocity_interp = {}
    for p in participants_order:
        val = velocity_results.get(p)
        if val is None:
            velocity_interp[p] = None
            continue
        if isinstance(val, list):
            interp_trials = []
            for (t_arr, v_arr) in val:
                t_arr = np.asarray(t_arr, dtype=float)
                v_arr = np.asarray(v_arr, dtype=float)
                if len(t_arr) < 2 or len(v_arr) < 2:
                    continue
                if t_arr.max() <= 1.0 and total_seconds > 1:
                    t_arr = t_arr * float(total_seconds)
                v_interp = np.interp(x, t_arr, v_arr, left=v_arr[0], right=v_arr[-1])
                interp_trials.append(v_interp)
            if interp_trials:
                arr = np.vstack(interp_trials)
                v_median = np.median(arr, axis=0)
                t0, v0 = val[0]
                velocity_interp[p] = (np.asarray(t0), np.asarray(v0), v_median)
            else:
                velocity_interp[p] = None
        elif isinstance(val, tuple) and len(val) == 2:
            t_arr, v_arr = np.asarray(val[0], dtype=float), np.asarray(val[1], dtype=float)
            if len(t_arr) < 2 or len(v_arr) < 2:
                velocity_interp[p] = None
            else:
                if t_arr.max() <= 1.0 and total_seconds > 1:
                    t_arr = t_arr * float(total_seconds)
                v_interp = np.interp(x, t_arr, v_arr, left=v_arr[0], right=v_arr[-1])
                velocity_interp[p] = (t_arr, v_arr, v_interp)
        else:
            v_arr = np.asarray(val, dtype=float)
            if len(v_arr) < 2:
                velocity_interp[p] = None
            else:
                t_arr = np.linspace(0.0, float(total_seconds), len(v_arr), endpoint=False)
                v_interp = np.interp(x, t_arr, v_arr, left=v_arr[0], right=v_arr[-1])
                velocity_interp[p] = (t_arr, v_arr, v_interp)

    # 绘图：三行 n 列
    n = len(participants_order)
    fig, axs = plt.subplots(3, n, figsize=(3 * n, 8), sharex='col')
    if n == 1:
        axs = np.array([[axs[0]], [axs[1]], [axs[2]]])

    for i, (participant, label) in enumerate(zip(participants_order, participant_labels)):
        ax_top = axs[0, i]
        ax_bot = axs[1, i]
        ax_vel = axs[2, i]

        entry = ys_by_participant.get(participant)
        if entry is not None:
            median_knob, y = entry
            ax_top.plot(x, y, color='black', linewidth=2, label=f'Median knob={median_knob:.3f}')
            # 画补函数（与原函数相加为 1），使用实线和不同颜色
            y_comp = 1.0 - y
            ax_top.plot(x, y_comp, color='tab:orange', linestyle='-', linewidth=1.5, label='Complement (1 - L)')
            for t in range(1, total_seconds):
                ax_top.axvline(t, color='gray', linestyle='--', linewidth=0.6, alpha=0.5)
            for t in range(2, total_seconds + 1, 2):
                ax_top.axvline(t, color='black', linestyle='-', linewidth=0.6, alpha=0.6)
            ax_top.set_title(f"{label} ({participant})  median={median_knob:.3f}", fontsize=10)
            ax_top.set_ylim(0, 1)
            ax_top.grid(alpha=0.25)

            dy = dys_by_participant.get(participant)
            if dy is not None:
                ax_bot.plot(x, dy, color='tab:red', linewidth=1.5, label='dL/dt')
                for t in range(1, total_seconds):
                    ax_bot.axvline(t, color='gray', linestyle='--', linewidth=0.6, alpha=0.5)
                for t in range(2, total_seconds + 1, 2):
                    ax_bot.axvline(t, color='black', linestyle='-', linewidth=0.6, alpha=0.6)
                ax_bot.set_ylim(global_dy_ylim)
                yticks = np.linspace(global_dy_ylim[0], global_dy_ylim[1], 5)
                ax_bot.set_yticks(yticks)
                ax_bot.grid(alpha=0.25)
        else:
            ax_top.text(0.5, 0.5, "No data", ha='center', va='center', fontsize=10, color='gray')
            ax_bot.text(0.5, 0.5, "No data", ha='center', va='center', fontsize=10, color='gray')
            ax_top.set_title(f"{label} ({participant})", fontsize=10)

        # 第三排：velocity 曲线（来自 LinearOnly 内嵌实现）
        v_entry = velocity_interp.get(participant)
        if v_entry is not None:
            t_arr, v_arr, v_interp = v_entry
            ax_vel.plot(np.linspace(0, total_seconds, len(v_interp)), v_interp, color='tab:purple', linewidth=2, label='Velocity (linear-only median)')
            # 绘制一些原始点（若有）
            try:
                sample_n = min(len(t_arr), 50)
                sample_t = np.linspace(t_arr.min(), t_arr.max(), sample_n)
                sample_v = np.interp(sample_t, t_arr, v_arr)
                ax_vel.scatter(sample_t, sample_v, s=6, color='tab:purple', alpha=0.6)
            except Exception:
                pass
            ax_vel.grid(alpha=0.25)
            if i == 0:
                ax_vel.set_ylabel('Velocity')
        else:
            ax_vel.text(0.5, 0.5, "No velocity data", ha='center', va='center', fontsize=10, color='gray')
 
        ax_vel.set_ylim(-1, 2.5)   # ← 第三排统一范围

        ax_top.set_xlim(0, total_seconds)
        ax_bot.set_xlim(0, total_seconds)
        ax_vel.set_xlim(0, total_seconds)

        ax_top.set_xticks([])  # 顶部不显示 x 刻度
        ax_bot.set_xticks([])  # 中间不显示 x 刻度
        ax_vel.set_xticks(list(range(0, total_seconds + 1)))

        if i == 0:
            ax_top.set_ylabel('Luminance (0–1)')
            ax_bot.set_ylabel('dL/dt (per second)')

    # 合并图例（只取第一列）
    handles = []
    labels = []
    for a in (axs[0,0], axs[1,0], axs[2,0]):
        h, l = a.get_legend_handles_labels()
        handles += h
        labels += l
    if handles:
        fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.95))

    plt.tight_layout()
    plt.savefig('function_mix_median_and_derivative_and_velocity_by_participant.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 打印每个参与者的中位 knob、亮度与导数参考值
    print("\n=== Participant median knob values (2s periodic) ===")
    for participant, label in zip(participants_order, participant_labels):
        knobs = per_participant_knobs.get(participant, np.array([]))
        if knobs.size:
            median_knob = np.median(knobs)
            y = ys_by_participant[participant][1]
            dy = dys_by_participant[participant]
            val_at_1 = float(np.interp(1.0, x, y))
            deriv_at_1 = float(np.interp(1.0, x, dy))
            print(f"{label} ({participant}): median, L(1.0)={val_at_1:.3f}, dL/dt(1.0)={deriv_at_1:.3f}")
        else:
            print(f"{label} ({participant}): No data")

def main():
    """主函数"""
    data_dir = "../BrightnessFunctionMixAndPhaseData"
    if not os.path.exists(data_dir):
        print(f"数据目录不存在: {data_dir}")
        return
    create_function_mix_plot(data_dir)

if __name__ == "__main__":
    main()