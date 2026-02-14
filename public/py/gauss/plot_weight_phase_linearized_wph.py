import os
import numpy as np
import matplotlib.pyplot as plt

# ======================
# CONFIG
# ======================
T = 1.0                 # 1个周期(过渡段)时长 (sec)
N_PERIODS = 5           # 5个周期
FPS = 600               # 采样密度（每秒点数）
d = np.deg2rad(60.0)    # <-- 改这里：d（弧度）

FIGSIZE = (10, 3.2)
DPI = 200

OUT_NAME = f"weight_phase_linearized_wph_{N_PERIODS}periods_d{int(round(np.rad2deg(d)))}deg.png"

# ======================
def w_phase_linearized(t, T, d, eps=1e-6):
    """
    w_ph(t) = u(t) / (sin d + u(t) (1 - cos d)),
    u(t)=tan(k*(t)), k*(t)=t/T * d
    """
    t = np.asarray(t, dtype=np.float64)
    k_star = (t / T) * d
    k_star = np.clip(k_star, -0.5*np.pi + eps, 0.5*np.pi - eps)  # 避免tan发散

    u = np.tan(k_star)
    denom = np.sin(d) + u * (1.0 - np.cos(d))
    w = u / denom
    return w

def main():
    # 전체时间轴：0..N_PERIODS*T
    total_T = N_PERIODS * T
    n_total = int(round(total_T * FPS)) + 1
    t_global = np.linspace(0.0, total_T, n_total)

    # 每个周期内的局部时间：t_local in [0, T)
    t_local = np.mod(t_global, T)

    # 线性权重（每周期重置）
    w_lin = t_local / T

    # 位相线形化权重（每周期重置）
    w_ph = w_phase_linearized(t_local, T, d)
    w_ph = np.clip(w_ph, 0.0, 1.0)

    plt.figure(figsize=FIGSIZE)
    plt.plot(t_global, w_lin, label="w_lin (repeated)")
    plt.plot(t_global, w_ph, label=r"$w_{ph}$ (phase-linearized, repeated)")
    plt.xlabel("time t (s)")
    plt.ylabel("weight")
    plt.ylim(0.0, 1.0)
    plt.xlim(0.0, total_T)
    plt.grid(True)
    plt.legend()

    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_png = os.path.join(out_dir, OUT_NAME)
    plt.savefig(out_png, dpi=DPI, bbox_inches="tight")
    plt.close()
    print("Saved:", out_png)

if __name__ == "__main__":
    main()
