import numpy as np
import matplotlib.pyplot as plt

# === 三组均值参数 ===
params_mean = dict(V0=1.0033, A1=1.8330, phi1=1.2022, A2=1.1990, phi2=2.7625)
params_prev = dict(V0=1.1293, A1=0.8150, phi1=3.4620, A2=0.8600, phi2=5.8538)
params_now  = dict(V0=0.9420, A1=0.7830, phi1=3.0348, A2=-0.1230, phi2=4.3542)

# === 时间轴 ===
t_sec = np.linspace(0, 4, 2000)

def velocity_curve(V0, A1, phi1, A2, phi2, t):
    """计算 v(t) = V0 + A1*sin(2πt+φ1+π) + A2*sin(4πt+φ2+π)"""
    return (
        V0
        + A1 * np.sin(2 * np.pi * t + phi1 + np.pi)
        + A2 * np.sin(4 * np.pi * t + phi2 + np.pi)
    )

# === 计算每组曲线 ===
v_mean = velocity_curve(**params_mean, t=t_sec)
v_prev = velocity_curve(**params_prev, t=t_sec)
v_now  = velocity_curve(**params_now,  t=t_sec)

# === 绘图 ===
plt.figure(figsize=(9, 4))
plt.plot(t_sec, v_mean, color='purple', linewidth=2.5, label="Mean parameters (global)")
plt.plot(t_sec, v_prev, color='red',   linewidth=2,   label="KK_prev_control (mean)")
plt.plot(t_sec, v_now,  color='blue',  linewidth=2,   label="KK_now_control (mean)")

plt.title("Velocity curves comparison (mean vs prev vs now)", fontsize=12)
plt.xlabel("Time (s)")
plt.ylabel("v(t)")
plt.grid(alpha=0.3)
plt.legend(fontsize=9, loc='upper right')
plt.tight_layout()
plt.savefig('test.png', dpi=300, bbox_inches='tight', transparent=True)
plt.show()
